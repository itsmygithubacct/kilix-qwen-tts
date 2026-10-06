"""CUDA offer, CPU fallback and device-profile controls; no GPU or torch needed.

The namespace tests stand in ordinary character devices for the NVIDIA nodes,
so they prove what the sandbox exposes, not that CUDA runs. Real CUDA runs are
recorded separately on GPU hardware.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts import bootstrap, device, sandbox
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import (ENGINE_COMMIT, RUNTIME_SCHEMA, InstalledRuntime, digest_file,
                                    run_job, tree_digest)
import test_runtime as fixtures

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
STAND_INS = (('/dev/null', '/dev/nvidiactl'), ('/dev/zero', '/dev/nvidia-uvm'),
             ('/dev/full', '/dev/nvidia0'))


class OutOfMemory(RuntimeError):
    pass


def fake_torch(*, available=True, count=1, bf16=True, probe_error=None):
    def is_available():
        if probe_error is not None:
            raise probe_error
        return available
    cuda = SimpleNamespace(is_available=is_available, device_count=lambda: count,
                           is_bf16_supported=lambda: bf16, OutOfMemoryError=OutOfMemory,
                           empty_cache=lambda: None)
    return SimpleNamespace(cuda=cuda, float32='float32', bfloat16='bfloat16', float16='float16')


class SelectionTests(unittest.TestCase):
    def test_cuda_is_used_only_when_offered_and_usable(self):
        self.assertEqual(device.select('cpu', fake_torch()), 'cpu')
        self.assertEqual(device.select('cuda', fake_torch()), 'cuda')
        self.assertEqual(device.select('cuda', fake_torch(available=False)), 'cpu')
        self.assertEqual(device.select('cuda', fake_torch(count=0)), 'cpu')
        self.assertEqual(device.select('cuda', fake_torch(probe_error=RuntimeError('driver too old'))), 'cpu')
        for invalid in ('gpu', 'cuda:1', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                device.select(invalid, fake_torch())

    def test_only_a_cuda_profile_is_offered_cuda(self):
        for pair in (('cpu', 'cpu'), ('cuda', 'cpu'), ('cuda', 'cuda')):
            self.assertEqual(device.offer(*pair), pair)
        for pair in (('cpu', 'cuda'), ('rocm', 'cpu'), ('cuda', 'gpu'), (None, 'cpu'), ('cuda', ['cuda'])):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                device.offer(*pair)

    def test_precision_follows_device(self):
        self.assertEqual(device.dtype('cpu', fake_torch()), 'float32')
        # Emulated bfloat16 on pre-Ampere GPUs still beats float16, which
        # overflows in this engine.
        for native in (True, False):
            self.assertEqual(device.dtype('cuda', fake_torch(bf16=native)), 'bfloat16')

    def record(self, failures=()):
        calls = []
        failures = list(failures)

        def load(where, dtype):
            calls.append(('load', where, dtype))
            if failures and failures[0][0] == ('load', where):
                raise failures.pop(0)[1]
            return 'model-' + where

        def generate(model, where):
            calls.append(('generate', model, where))
            if failures and failures[0][0] == ('generate', where):
                raise failures.pop(0)[1]
            return 'audio-' + where
        return calls, load, generate

    def test_cuda_success_never_touches_the_cpu(self):
        calls, load, generate = self.record()
        self.assertEqual(device.run('cuda', fake_torch(), load, generate), ('cuda', 'audio-cuda'))
        self.assertEqual(calls, [('load', 'cuda', 'bfloat16'), ('generate', 'model-cuda', 'cuda')])

    def test_cuda_failures_fall_back_to_a_float32_cpu_run(self):
        for failure in ((('load', 'cuda'), OutOfMemory('CUDA out of memory')),
                        (('load', 'cuda'), RuntimeError('CUDA error: no kernel image')),
                        (('generate', 'cuda'), OutOfMemory('CUDA out of memory'))):
            with self.subTest(failure=failure):
                calls, load, generate = self.record([failure])
                self.assertEqual(device.run('cuda', fake_torch(), load, generate), ('cpu', 'audio-cpu'))
                self.assertEqual(calls[-2:], [('load', 'cpu', 'float32'), ('generate', 'model-cpu', 'cpu')])

    def test_unavailable_cuda_runs_on_the_cpu_without_a_cuda_load(self):
        calls, load, generate = self.record()
        self.assertEqual(device.run('cuda', fake_torch(available=False), load, generate), ('cpu', 'audio-cpu'))
        self.assertEqual(calls, [('load', 'cpu', 'float32'), ('generate', 'model-cpu', 'cpu')])

    def test_no_retry_after_output_or_for_non_cuda_errors(self):
        calls, load, generate = self.record([(('generate', 'cuda'), OutOfMemory('CUDA out of memory'))])
        with self.assertRaises(OutOfMemory):
            device.run('cuda', fake_torch(), load, generate, retry_allowed=lambda: False)
        self.assertNotIn(('load', 'cpu', 'float32'), calls)
        calls, load, generate = self.record([(('generate', 'cuda'), ValueError('invalid engine audio'))])
        with self.assertRaises(ValueError):
            device.run('cuda', fake_torch(), load, generate)
        self.assertNotIn(('load', 'cpu', 'float32'), calls)


class NodeTests(unittest.TestCase):
    def test_all_nodes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = [Path(tmp) / 'nvidia', Path(tmp) / 'nvidia_uvm']
            for state in states:
                state.write_text('live\n')
            with patch.object(sandbox, 'GPU_NODES', ('/dev/null', '/dev/zero', '/dev/full')), \
                    patch.object(sandbox, 'GPU_SYSFS', tuple(map(str, states))), \
                    patch.object(sandbox, 'driver_binds', return_value=()):
                self.assertEqual(len(sandbox.accelerator_nodes()), 3)
                with patch.object(sandbox, 'driver_binds', return_value=None):
                    self.assertEqual(sandbox.accelerator_nodes(), ())
                states[1].unlink()
                self.assertEqual(sandbox.accelerator_nodes(), ())
        with patch.object(sandbox, 'GPU_NODES', ('/dev/null', '/dev/zero', '/dev/full')), \
                patch.object(sandbox, 'GPU_SYSFS', ()), patch.object(sandbox, 'driver_binds', return_value=()):
            self.assertEqual(sandbox.accelerator_nodes(),
                             (('/dev/null', '/dev/null'), ('/dev/zero', '/dev/zero'), ('/dev/full', '/dev/full')))
        with tempfile.TemporaryDirectory() as tmp:
            regular = Path(tmp) / 'nvidia0'
            regular.write_bytes(b'')
            for nodes in (('/dev/null', '/dev/zero', str(Path(tmp) / 'absent')),
                          ('/dev/null', '/dev/zero', str(regular))):
                with self.subTest(nodes=nodes), patch.object(sandbox, 'GPU_NODES', nodes):
                    self.assertEqual(sandbox.accelerator_nodes(), ())

    def test_launch_refuses_unsafe_or_misplaced_nodes(self):
        cuda = SimpleNamespace(device='cuda')
        with tempfile.TemporaryDirectory() as tmp:
            regular = Path(tmp) / 'file'
            regular.write_bytes(b'')
            cases = ((SimpleNamespace(device='cpu'), STAND_INS),
                     (SimpleNamespace(device='rocm'), ()),
                     (cuda, STAND_INS[:2]),
                     (cuda, STAND_INS[:2] + (('/dev/full', '/dev/sda'),)),
                     (cuda, STAND_INS[:2] + ((str(regular), '/dev/nvidia0'),)),
                     (cuda, list(STAND_INS)))
            for runtime, nodes in cases:
                with self.subTest(runtime=runtime, nodes=nodes), self.assertRaises(ProtocolError):
                    with sandbox.launch(runtime, tmp, None, lambda: None, gpu_nodes=nodes):
                        self.fail('unsafe accelerator nodes were accepted')


class DriverLibraryTests(unittest.TestCase):
    """Debian routes libcuda through /etc/alternatives; other layouts do not."""
    def layout(self, root):
        usr, etc = root / 'usr', root / 'etc'
        (usr / 'lib/nvidia').mkdir(parents=True)
        (etc / 'alternatives').mkdir(parents=True)
        real = usr / 'lib/nvidia/libcuda.so.550'
        real.write_bytes(b'driver')
        real.chmod(0o644)
        return usr, etc, real

    def patched(self, root, libraries):
        return (patch.object(sandbox, 'DRIVER_ROOT', str(root / 'usr') + '/'),
                patch.object(sandbox, 'HOP_ROOT', str(root / 'etc') + '/'),
                patch.object(sandbox, 'TRUSTED_UID', os.geteuid()),
                patch.object(sandbox, 'DRIVER_LIBRARIES', tuple(map(str, libraries))))

    def binds(self, root, libraries):
        a, b, c, d = self.patched(root, libraries)
        with a, b, c, d:
            return sandbox.driver_binds()

    def test_alternatives_hops_are_bound_and_direct_links_need_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usr, etc, real = self.layout(root)
            alternative = etc / 'alternatives/nvidia--libcuda.so.1'
            alternative.symlink_to(real)
            debian = usr / 'lib/libcuda.so.1'
            debian.symlink_to(alternative)
            direct = usr / 'lib/libcuda-direct.so.1'
            direct.symlink_to('nvidia/libcuda.so.550')
            self.assertEqual(self.binds(root, [debian]), (str(alternative),))
            self.assertEqual(self.binds(root, [direct]), ())
            # The optional JIT library adds its hops when present, and is
            # skipped when absent; libcuda itself is required.
            self.assertEqual(self.binds(root, [direct, usr / 'lib/absent.so.1']), ())
            self.assertEqual(self.binds(root, [direct, debian]), (str(alternative),))
            self.assertIsNone(self.binds(root, [usr / 'lib/absent.so.1', direct]))

    def test_unsafe_driver_chains_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usr, etc, real = self.layout(root)
            outside = root / 'home-library.so'
            outside.write_bytes(b'user')
            escape = usr / 'lib/escape.so.1'
            escape.symlink_to(outside)
            writable = usr / 'lib/nvidia/writable.so'
            writable.write_bytes(b'driver')
            writable.chmod(0o666)
            loop = usr / 'lib/loop.so.1'
            loop.symlink_to(loop)
            for library in (escape, writable, loop, usr / 'lib/nvidia'):
                with self.subTest(library=library.name):
                    self.assertIsNone(self.binds(root, [library]))
            a, b, c, d = self.patched(root, [real])
            with a, b, d:  # a regular user-owned file is not a trusted driver
                self.assertIsNone(sandbox.driver_binds())


class CommandTests(unittest.TestCase):
    def bwrap_command(self, runtime, nodes):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('python', 'packages', 'installation'):
                (root / name).mkdir(mode=0o700)
            runtime.python_root, runtime.site_packages, runtime.root = (
                root / 'python', root / 'packages', root / 'installation')
            runtime.manifest = {'files': {}, 'environment': {
                'python_root_sha256': tree_digest(root / 'python'),
                'site_packages_sha256': tree_digest(root / 'packages'), 'python_sha256': ''}}
            options = {'gpu_nodes': nodes} if nodes else {}
            with sandbox.launch(runtime, tmp, None, lambda: None, **options) as (_command, descriptors):
                configuration = descriptors[2]
                return json.loads(os.pread(configuration, 65536, 0))['command']

    def test_sysfs_module_states_are_bound_only_for_a_gpu_job(self):
        hop = '/etc/alternatives/nvidia--libcuda.so.1-x86_64-linux-gnu'
        with patch.object(sandbox, 'driver_binds', return_value=(hop,)):
            gpu = self.bwrap_command(SimpleNamespace(device='cuda'), STAND_INS)
            cpu = self.bwrap_command(SimpleNamespace(device='cpu'), ())
        binds = [gpu[i + 1:i + 3] for i, word in enumerate(gpu) if word == '--ro-bind-try']
        self.assertEqual(binds, [[path, path] for path in sandbox.GPU_SYSFS] + [[hop, hop]])
        self.assertNotIn(hop, cpu)
        self.assertEqual([gpu[i + 1:i + 3] for i, word in enumerate(gpu) if word == '--dev-bind'],
                         [list(pair) for pair in STAND_INS])
        self.assertFalse(any(word.startswith(('/sys', '/etc')) and word not in (*sandbox.GPU_SYSFS, hop)
                             for word in gpu))
        self.assertEqual(gpu[-1], 'cuda')
        for runtime in (SimpleNamespace(device='cuda'), SimpleNamespace(device='cpu')):
            command = self.bwrap_command(runtime, ())
            self.assertFalse(any(word.startswith(('/sys', '/dev/nvidia')) for word in command))
            self.assertNotIn('--dev-bind', command)
            self.assertEqual(command[-1], runtime.device)


class ProfileLimitTests(unittest.TestCase):
    def test_bootstrap_bounds_data_for_cuda_and_address_space_for_cpu(self):
        cpu, cuda = bootstrap.limits('cpu'), bootstrap.limits('cuda')
        self.assertEqual(cpu[resource.RLIMIT_AS], 20 * 1024**3)
        self.assertNotIn(resource.RLIMIT_DATA, cpu)
        self.assertEqual(cuda[resource.RLIMIT_DATA], 24 * 1024**3)
        self.assertNotIn(resource.RLIMIT_AS, cuda)
        for limits in (cpu, cuda):
            self.assertEqual({k: limits[k] for k in (resource.RLIMIT_CORE, resource.RLIMIT_CPU,
                                                    resource.RLIMIT_FSIZE, resource.RLIMIT_NOFILE)},
                             {resource.RLIMIT_CORE: 0, resource.RLIMIT_CPU: 3600,
                              resource.RLIMIT_FSIZE: 64 * 1024**2, resource.RLIMIT_NOFILE: 128})

    def test_bootstrap_refuses_an_unknown_profile_before_any_mount(self):
        for argv in (['bootstrap', '5', 'rocm'], ['bootstrap', '5', 'cpu', 'extra']):
            with self.subTest(argv=argv), patch.object(sys, 'argv', argv), \
                    patch.object(bootstrap, 'prepare_mount', side_effect=AssertionError('mounted')):
                self.assertEqual(bootstrap.main(), 125)

    def test_profile_bounds_match_between_host_and_bootstrap(self):
        for profile in ('cpu', 'cuda'):
            self.assertEqual(sandbox.PROFILES[profile], bootstrap.PROFILES[profile][:2])
        self.assertGreater(sandbox.PROFILES['cuda'][0], sandbox.PROFILES['cpu'][0])
        self.assertGreater(sandbox.PROFILES['cuda'][1], sandbox.PROFILES['cpu'][1])


@unittest.skipUnless(Path('/usr/bin/bwrap').exists(), 'Linux namespace launcher is required')
class DeviceJobTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RuntimeTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime = self.fixture.runtime

    def run_with(self, profile, nodes, text='speech'):
        self.runtime.device = profile
        self.runtime.last_device = None
        with patch('kilix_qwen_tts.sandbox.accelerator_nodes', return_value=nodes), \
                self.fixture.audio.open('rb') as source:
            return run_job(self.runtime, source.fileno(), self.fixture.arguments(text),
                           deadline=time.monotonic() + 10, cancel=threading.Event())

    def test_cuda_runtime_is_offered_only_the_checked_nodes(self):
        result, _ = self.run_with('cuda', STAND_INS)
        self.assertEqual(self.runtime.last_device, 'cuda')
        self.assertNotIn('device', result)

    def test_cuda_runtime_without_nodes_runs_on_the_cpu(self):
        result, _ = self.run_with('cuda', ())
        self.assertEqual(self.runtime.last_device, 'cpu')
        self.assertEqual(result['model_id'], 'qwen3-tts-0.6b-base')

    def test_cpu_runtime_is_never_offered_the_gpu(self):
        self.run_with('cpu', STAND_INS)
        self.assertEqual(self.runtime.last_device, 'cpu')

    def test_worker_fallback_is_reported_and_unoffered_gpu_is_refused(self):
        self.run_with('cuda', STAND_INS, 'no-gpu')
        self.assertEqual(self.runtime.last_device, 'cpu')
        for profile, nodes in (('cuda', ()), ('cpu', STAND_INS)):
            with self.subTest(profile=profile), self.assertRaises(ProtocolError) as caught:
                self.run_with(profile, nodes, 'claim-gpu')
            self.assertEqual(caught.exception.code, 'ENGINE_FAILED')

    def test_cuda_profile_admits_a_population_the_cpu_profile_refuses(self):
        for index in range(30):
            (self.runtime.site_packages / f'extra-{index}').write_bytes(b'')
        self.fixture.manifest['environment']['site_packages_sha256'] = tree_digest(self.runtime.site_packages)
        bounds = {'cpu': (sandbox.MAX_BUNDLE_BYTES, 20), 'cuda': (sandbox.MAX_BUNDLE_BYTES, 200)}
        with patch.object(sandbox, 'PROFILES', bounds):
            with self.assertRaises(ProtocolError):
                self.run_with('cpu', ())
            self.run_with('cuda', ())
        self.assertEqual(self.runtime.last_device, 'cpu')


class ManifestTests(unittest.TestCase):
    def manifest_runtime(self, root, device_value):
        python_root = root / 'python'
        packages = root / 'packages'
        installation = root / 'installation'
        (python_root / 'bin').mkdir(parents=True, mode=0o700)
        packages.mkdir(mode=0o700)
        python = python_root / 'bin/python3.12'
        python.write_text('#!/bin/sh\n')
        python.chmod(0o700)
        names = ('config.json', 'generation_config.json', 'model.safetensors', 'preprocessor_config.json',
                 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'speech_tokenizer/config.json',
                 'speech_tokenizer/model.safetensors', 'speech_tokenizer/preprocessor_config.json')
        files = {}
        for name in names:
            path = installation / 'model' / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(name.encode())
            files['model/' + name] = hashlib.sha256(name.encode()).hexdigest()
        installation.chmod(0o700)
        manifest = {'schema': RUNTIME_SCHEMA, 'engine_revision': ENGINE_COMMIT, 'device': device_value,
                    'model': {'id': 'qwen3-tts-1.7b-voicedesign',
                              'revision': '5ecdb67327fd37bb2e042aab12ff7391903235d3'},
                    'files': files,
                    'environment': {'python': str(python), 'python_sha256': digest_file(python, follow=True),
                                    'site_packages': str(packages), 'site_packages_sha256': tree_digest(packages),
                                    'python_root': str(python_root),
                                    'python_root_sha256': tree_digest(python_root, allow_file_links=True)}}
        (installation / 'runtime.json').write_text(json.dumps(manifest))
        return InstalledRuntime(installation)

    def test_manifest_device_is_cpu_or_cuda_and_reported(self):
        for value in ('cpu', 'cuda'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                runtime = self.manifest_runtime(Path(tmp), value)
                self.assertEqual(runtime.device, value)
                self.assertEqual(runtime.model_record()['device'], value)
                self.assertEqual(runtime.mode, 'voice_design')
        for value in ('gpu', 'rocm', 'cuda:0', None, ['cuda']):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp, \
                    self.assertRaises(ProtocolError):
                self.manifest_runtime(Path(tmp), value)


def load_tool(name):
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(name + '_under_test', TOOLS / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


class StagingDeviceTests(unittest.TestCase):
    def test_environment_torch_build_must_match_the_runtime_device(self):
        stage_runtime = load_tool('stage_runtime')
        private = Path(self.enterContext(tempfile.TemporaryDirectory()))
        def fake(build):
            def checked_output(command, check=lambda: None, *, deadline=None):
                text = ' '.join(command)
                if command[:2] == ['git', '-C'] and 'rev-parse' in command:
                    return ENGINE_COMMIT.encode() + b'\n'
                if 'm.version' in text:
                    return build.encode() + b'\n'
                if 'purelib' in text or 'base_prefix' in text:
                    return str(private).encode()
                if 'version_info' in text:
                    return b'3.12'
                raise LookupError('reached the engine source parity probe')
            return checked_output
        for device_value, build in (('cpu', '2.6.0+cpu'), ('cuda', '2.6.0+cu124')):
            with self.subTest(device=device_value), \
                    patch.object(stage_runtime, 'checked_output', fake(build)), \
                    self.assertRaises(LookupError):
                stage_runtime.environment_record(Path('/env'), Path('/src'), device=device_value)
        for device_value, build in (('cpu', '2.6.0+cu124'), ('cuda', '2.6.0+cpu'), ('cuda', '2.5.1+cu124')):
            with self.subTest(device=device_value, build=build), \
                    patch.object(stage_runtime, 'checked_output', fake(build)), \
                    self.assertRaises(ValueError):
                stage_runtime.environment_record(Path('/env'), Path('/src'), device=device_value)
        with self.assertRaises(ValueError):
            stage_runtime.environment_record(Path('/env'), Path('/src'), device='rocm')


BUILD_PYTHON = '''#!/usr/bin/python3
import json,os
print(json.dumps({'torch':os.environ['FAKE_TORCH'],'torchaudio':os.environ['FAKE_TORCH']}))
'''
BUILD_UV = '''#!/usr/bin/python3
import os,pathlib,sys
destination=pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT'])
(destination/'bin').mkdir(exist_ok=True)
python=destination/'bin/python'
python.write_text(PYTHON_SOURCE)
python.chmod(0o700)
with open(os.environ['FAKE_LOG'],'a') as log:log.write(' '.join(sys.argv[1:])+chr(10))
'''


class ClosureTests(unittest.TestCase):
    """The interpreter-closure seam: system or wrong-version interpreters refuse early."""
    def probe_outputs(self, root, version, *, hashed):
        stage_runtime = load_tool('stage_runtime')
        site = Path(self.enterContext(tempfile.TemporaryDirectory()))
        def checked_output(command, check=lambda: None, *, deadline=None):
            text = ' '.join(command)
            if 'rev-parse' in command:
                return ENGINE_COMMIT.encode()
            if 'm.version' in text:
                return b'2.6.0+cpu'
            if 'purelib' in text:
                return str(site).encode()
            if 'base_prefix' in text:
                return str(root).encode()
            if 'version_info' in text:
                return version.encode()
            raise LookupError('reached the engine source parity probe')
        with patch.object(stage_runtime, 'checked_output', checked_output), \
                patch.object(stage_runtime, 'tree_digest', side_effect=hashed):
            return stage_runtime.environment_record(Path('/env'), Path('/src'))

    def test_system_or_shared_prefix_refuses_before_any_hashing(self):
        hashed = AssertionError('hashed a refused closure')
        for root, version in ((Path('/usr'), '3.12'), (Path('/usr'), '3.13')):
            with self.subTest(root=root, version=version), self.assertRaises(ValueError):
                self.probe_outputs(root, version, hashed=hashed)
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / 'shared'
            shared.mkdir(mode=0o700)
            shared.chmod(0o775)
            with self.assertRaises(ValueError):
                self.probe_outputs(shared, '3.12', hashed=hashed)

    def test_wrong_interpreter_version_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / 'python'
            private.mkdir(mode=0o700)
            for version in ('3.13', '3.11'):
                with self.subTest(version=version), self.assertRaises(ValueError) as caught:
                    self.probe_outputs(private, version, hashed=AssertionError('hashed'))
                self.assertIn('3.12', str(caught.exception))

    def test_private_312_closure_proceeds_to_source_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / 'python'
            private.mkdir(mode=0o700)
            with self.assertRaises(LookupError):
                self.probe_outputs(private, '3.12', hashed=AssertionError('hashed'))


class BuildDeviceTests(unittest.TestCase):
    def build(self, root, device_value, torch_build):
        uv = root / 'uv'
        uv.write_text(BUILD_UV.replace('PYTHON_SOURCE', repr(BUILD_PYTHON)))
        uv.chmod(0o700)
        log = root / 'uv.log'
        destination = root / ('environment-' + (device_value or 'default'))
        command = [sys.executable, str(TOOLS / 'build_environment.py'), '--uv', str(uv),
                   '--destination', str(destination), '--offline', '--timeout', '20']
        if device_value is not None:
            command += ['--device', device_value]
        environment = {'PATH': '/usr/bin:/bin', 'FAKE_TORCH': torch_build, 'FAKE_LOG': str(log),
                       'HOME': str(root), 'TMPDIR': str(root)}
        process = subprocess.run(command, env=environment, capture_output=True, timeout=30)
        return process, destination, log.read_text() if log.exists() else ''

    def test_each_device_syncs_its_group_and_records_it(self):
        for device_value, group, build in ((None, 'cpu', '2.6.0+cpu'), ('cuda', 'cuda', '2.6.0+cu124')):
            with self.subTest(device=device_value), tempfile.TemporaryDirectory(prefix='kq-dev-') as tmp:
                root = Path(tmp)
                root.chmod(0o700)
                process, destination, log = self.build(root, device_value, build)
                self.assertEqual(process.returncode, 0, process.stderr.decode())
                second = log.splitlines()[1]
                self.assertIn(f'--group {group} --group build', second)
                self.assertNotIn('--group ' + ({'cpu', 'cuda'} - {group}).pop(), log)
                receipt = json.loads((destination / 'kilix-environment-build.json').read_text())
                self.assertEqual(receipt['device'], device_value or 'cpu')
                self.assertEqual(receipt['packages']['torch'], build)

    def test_a_mismatched_torch_build_refuses_and_removes_the_destination(self):
        for device_value, build in (('cuda', '2.6.0+cpu'), ('cpu', '2.6.0+cu124')):
            with self.subTest(device=device_value), tempfile.TemporaryDirectory(prefix='kq-dev-') as tmp:
                root = Path(tmp)
                root.chmod(0o700)
                process, destination, _log = self.build(root, device_value, build)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
