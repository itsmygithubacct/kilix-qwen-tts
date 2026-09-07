"""Real namespace/IPC controls with an explicit fake engine, not qualification."""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.protocol import ProtocolError, receive_packet, send_packet
from kilix_qwen_tts.runtime import ENGINE_COMMIT, run_job, tree_digest
from kilix_qwen_tts.sandbox import BUNDLE_MAGIC, RECORD, launch, memory_file, seal, snapshot
from kilix_qwen_tts.service import Service, client_request, request_value

ENGINE = '''#!/usr/bin/python3
import hashlib,json,os,pathlib,resource,signal,subprocess,sys,time,wave
assert resource.getrlimit(resource.RLIMIT_AS)==(20*1024**3,20*1024**3)
assert resource.getrlimit(resource.RLIMIT_CPU)==(3600,3600)
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(64*1024**2,64*1024**2)
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(128,128)
assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0)
request=json.load(sys.stdin);args=request['args']
assert pathlib.Path('/opt/runtime/model/data').read_bytes()==b'model bytes'
assert pathlib.Path('/opt/python/lib/python3.12/site-packages/data').read_bytes()==b'dependency bytes'
assert not pathlib.Path(HOST_PATH).exists()
status=dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert all(int(status[key].strip(),16)==0 for key in ('CapInh','CapPrm','CapEff','CapAmb'))
assert status['NoNewPrivs'].strip()=='1'
try:pathlib.Path('/opt/runtime/model/data').write_bytes(b'mutation');raise AssertionError('runtime is writable')
except OSError as error:assert error.errno==30
if args['text']=='escape':
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(60)'],start_new_session=True)
    pathlib.Path('/job/ready').write_text('ready')
    time.sleep(60)
if args['text']=='bad':
    print('{"result":NaN}');raise SystemExit(0)
destination=pathlib.Path('/job/output.wav')
with wave.open(str(destination),'wb') as output:
    output.setnchannels(1);output.setsampwidth(2);output.setframerate(24000)
    output.writeframes(b'\\x01\\x00'*2400)
payload=destination.read_bytes();manifest=request['manifest']
result={'engine_id':'qwen3-tts','engine_revision':ENGINE_REVISION,
        'model_id':manifest['model']['id'],'model_revision':manifest['model']['revision'],
        'duration_ms':100,'seed':args['seed'],'audio':{'byte_length':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}}
if args['mode']=='prompt_clone':
    pcm=pathlib.Path('/opt/prompt.pcm').read_bytes()
    assert hashlib.sha256(pcm).hexdigest()==args['prompt_audio']['sha256']
    consent=json.dumps(args['consent'],separators=(',',':'),sort_keys=True).encode()
    result['conditioning']={'prompt_sha256':hashlib.sha256(pcm).hexdigest(),'consent_sha256':hashlib.sha256(consent).hexdigest()}
print(json.dumps(result))
'''


@unittest.skipUnless(Path('/usr/bin/bwrap').exists(), 'Linux namespace launcher is required')
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='kq-runtime-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.installation = self.root / 'installation'
        python_root = self.root / 'python'
        packages = self.root / 'packages'
        self.ipc = self.root / 'ipc'
        self.jobs = self.root / 'jobs'
        for directory in (self.installation / 'model', python_root / 'bin', packages, self.ipc, self.jobs):
            directory.mkdir(mode=0o700, parents=True)
        (self.installation / 'model/data').write_bytes(b'model bytes')
        (packages / 'data').write_bytes(b'dependency bytes')
        self.host_only = self.root / 'host-only'
        self.host_only.write_text('synthetic host-only fixture')
        engine = python_root / 'bin/python3.12'
        engine.write_text(ENGINE.replace('HOST_PATH', repr(str(self.host_only))).replace('ENGINE_REVISION', repr(ENGINE_COMMIT)))
        engine.chmod(0o700)
        manifest = {'model': {'id': 'qwen3-tts-0.6b-base', 'revision': '5d83992436eae1d760afd27aff78a71d676296fc'},
                    'files': {'model/data': hashlib.sha256(b'model bytes').hexdigest()},
                    'environment': {'python_root_sha256': tree_digest(python_root),
                                    'python_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
                                    'site_packages_sha256': tree_digest(packages)}}
        self.runtime = SimpleNamespace(root=self.installation, python_root=python_root,
            site_packages=packages, manifest=manifest, mode='prompt_clone', model_id='qwen3-tts-0.6b-base',
            model_revision='5d83992436eae1d760afd27aff78a71d676296fc', model_record=lambda: {'id': 'qwen3-tts-0.6b-base', 'installed': True})
        self.manifest = manifest
        self.pcm = b'\x01\x00' * 2400
        self.audio = self.root / 'prompt.pcm'
        self.audio.write_bytes(self.pcm)
        self.service = None
        self.thread = None
        self.addCleanup(self.stop)
        original = tempfile.TemporaryDirectory
        override = patch('kilix_qwen_tts.runtime.tempfile.TemporaryDirectory',
                         side_effect=lambda *args, **kwargs: original(*args, dir=self.jobs, **kwargs))
        override.start()
        self.addCleanup(override.stop)

    def arguments(self, text='speech'):
        digest = hashlib.sha256(self.pcm).hexdigest()
        return {'task': 'synthesize', 'mode': 'prompt_clone', 'model_id': 'auto', 'language': 'en',
                'text': text, 'seed': 7, 'max_duration_ms': 1000,
                'output': {'sample_format': 's16le', 'sample_rate_hz': 24000, 'channels': 1},
                'prompt_fd': 0, 'prompt_audio': {'sample_format': 's16le', 'sample_rate_hz': 24000,
                    'channels': 1, 'frame_count': 2400, 'byte_length': len(self.pcm), 'duration_ms': 100, 'sha256': digest},
                'consent': {'schema': 'kilix.voice.consent/candidate-v1', 'source_sha256': digest,
                    'asserted_by_peer': True, 'allowed_use': 'this-project', 'purpose': None,
                    'recorded_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}}

    def start(self):
        self.service = Service(self.runtime, self.ipc)
        self.thread = threading.Thread(target=self.service.serve)
        self.thread.start()
        self.assertTrue(self.service.ready.wait(2))

    def stop(self):
        if self.service:
            self.service.stop()
        if self.thread:
            self.thread.join(5)
            self.assertFalse(self.thread.is_alive())

    def test_service_binds_prompt_consent_and_wav_result(self):
        self.start()
        value = request_value('submit', job_id='fake-job', args=self.arguments(), timeout=5)
        with self.audio.open('rb') as source:
            result, audio = client_request(self.ipc, value, source.fileno())
        self.assertEqual(result['duration_ms'], 100)
        self.assertEqual(result['audio']['sha256'], hashlib.sha256(audio).hexdigest())
        self.assertEqual(result['conditioning']['prompt_sha256'], hashlib.sha256(self.pcm).hexdigest())
        self.assertEqual(list(self.jobs.iterdir()), [])

    def test_bound_namespace_ignores_later_host_byte_replacements(self):
        original = launch
        @contextmanager
        def replaced(runtime, workspace, audio_fd, check):
            with original(runtime, workspace, audio_fd, check) as prepared:
                (runtime.root / 'model/data').write_text('replaced model')
                (runtime.site_packages / 'data').write_text('replaced dependency')
                (runtime.python_root / 'bin/python3.12').write_text('unverified executable')
                yield prepared
        with patch('kilix_qwen_tts.sandbox.launch', replaced), self.audio.open('rb') as source:
            result, _ = run_job(self.runtime, source.fileno(), self.arguments(),
                                deadline=time.monotonic() + 5, cancel=threading.Event())
        self.assertEqual(result['model_id'], 'qwen3-tts-0.6b-base')

    def test_large_environment_uses_bounded_bundle(self):
        for index in range(1700):
            (self.runtime.site_packages / f'fixture-{index}').write_bytes(b'')
        self.manifest['environment']['site_packages_sha256'] = tree_digest(self.runtime.site_packages)
        with self.audio.open('rb') as source:
            result, _ = run_job(self.runtime, source.fileno(), self.arguments(),
                                deadline=time.monotonic() + 10, cancel=threading.Event())
        self.assertEqual(result['duration_ms'], 100)

    def test_namespace_cancel_reaps_escaped_child_and_removes_workspace(self):
        unrelated = subprocess.Popen(['/usr/bin/python3', '-c', 'import time;time.sleep(60)'])
        def cleanup_unrelated():
            unrelated.terminate()
            unrelated.wait()
        self.addCleanup(cleanup_unrelated)
        self.start()
        value = request_value('submit', job_id='fake-job', args=self.arguments('escape'), timeout=5)
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(channel.close)
        channel.connect(str(self.ipc / 'kilix-qwen-tts.sock'))
        with self.audio.open('rb') as source:
            send_packet(channel, value, source.fileno())
        self.assertEqual(receive_packet(channel)[0]['type'], 'accepted')
        deadline = time.monotonic() + 3
        while not list(self.jobs.glob('*/ready')) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(list(self.jobs.glob('*/ready')))
        owned = []
        pending = [os.getpid()]
        while pending:
            parent = pending.pop()
            children = set()
            for task in Path(f'/proc/{parent}/task').iterdir():
                children.update((task / 'children').read_text().split())
            for child in children:
                if int(child) != unrelated.pid:
                    owned.append(int(child))
                    pending.append(int(child))
        self.assertGreaterEqual(len(owned), 3)
        client_request(self.ipc, request_value('cancel', job_id='fake-job'))
        event, descriptors = receive_packet(channel)
        self.assertFalse(descriptors)
        self.assertEqual(event['error']['code'], 'CANCELED')
        self.assertEqual(list(self.jobs.iterdir()), [])
        self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in owned))
        self.assertIsNone(unrelated.poll())

    def test_deadline_cleans_namespace_and_recovers_service(self):
        self.start()
        with self.audio.open('rb') as source, self.assertRaises(ProtocolError) as error:
            client_request(self.ipc, request_value('submit', job_id='deadline-job',
                args=self.arguments('escape'), timeout=0.3), source.fileno())
        self.assertEqual(error.exception.code, 'DEADLINE_EXCEEDED')
        self.assertEqual(list(self.jobs.iterdir()), [])
        result = client_request(self.ipc, request_value('status'))
        self.assertEqual(result['provider_state'], 'ready')

    def test_snapshot_creation_failure_closes_bundle(self):
        import kilix_qwen_tts.sandbox as sandbox
        original = sandbox.memory_file
        descriptors = []
        def fail_second():
            if descriptors:
                raise OSError('synthetic allocation failure')
            descriptors.append(original())
            return descriptors[-1]
        with patch.object(sandbox, 'memory_file', side_effect=fail_second), self.assertRaises(OSError):
            with launch(self.runtime, str(self.jobs), None, lambda: None):
                self.fail('allocation failure was ignored')
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    def test_bootstrap_rejects_unsealed_and_unsafe_bundles(self):
        from kilix_qwen_tts.bootstrap import unpack
        for path in ('../outside', '/outside', 'python/../outside', 'python//outside', 'python/\0outside'):
            with self.subTest(path=path):
                descriptor = memory_file()
                encoded = path.encode()
                os.write(descriptor, BUNDLE_MAGIC + RECORD.pack(len(encoded), 0, 0) + encoded + RECORD.pack(0, 0, 0))
                seal(descriptor)
                with self.assertRaises(ValueError):
                    unpack(descriptor)
        descriptor = memory_file()
        try:
            with self.assertRaises(ValueError):
                unpack(descriptor)
        finally:
            os.close(descriptor)

    def test_nonfinite_engine_result_refuses(self):
        with self.audio.open('rb') as source, self.assertRaises(ProtocolError):
            run_job(self.runtime, source.fileno(), self.arguments('bad'),
                    deadline=time.monotonic() + 5, cancel=threading.Event())
        self.assertEqual(list(self.jobs.iterdir()), [])

    def test_client_result_refuses_unbound_metadata_and_invalid_audio(self):
        from kilix_qwen_tts.results import validate_result
        arguments = self.arguments()
        with self.audio.open('rb') as source:
            result, audio = run_job(self.runtime, source.fileno(), arguments,
                                   deadline=time.monotonic() + 5, cancel=threading.Event())
        validate_result(result, arguments, audio)
        for key, value in (('model_id', []), ('model_revision', 'different'), ('seed', True),
                           ('seed', 8), ('duration_ms', 101), ('conditioning', {}), ('engine_id', 'other')):
            candidate = copy.deepcopy(result)
            candidate[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ProtocolError):
                validate_result(candidate, arguments, audio)
        invalid = b'x' * len(audio)
        result['audio']['sha256'] = hashlib.sha256(invalid).hexdigest()
        with self.assertRaises(ProtocolError):
            validate_result(result, arguments, invalid)

    def test_fifo_snapshot_refuses_and_snapshot_is_sealed(self):
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(ProtocolError):
            snapshot(fifo, lambda: None)
        fd, digest, _ = snapshot(self.audio, lambda: None)
        try:
            self.assertEqual(digest, hashlib.sha256(self.pcm).hexdigest())
            with self.assertRaises(OSError):
                os.write(fd, b'mutate')
        finally:
            os.close(fd)
