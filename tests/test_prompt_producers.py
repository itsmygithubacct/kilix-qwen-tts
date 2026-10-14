"""Actual service -> owned namespace -> worker -> cache, with private doubles.

CPU/.25 and simulated CUDA/.125 are sentinels, not measured model numerics.
Old-code controls use this same transport fixture without importing new APIs.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from kilix_qwen_tts import sandbox
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import tree_digest
from kilix_qwen_tts.service import Service, client_request, request_value
from test_backpressure import descendants
import test_runtime as fixtures

DOUBLE = Path(__file__).with_name('prompt_worker_double.py')
STAND_INS = (('/dev/null', '/dev/nvidiactl'), ('/dev/zero', '/dev/nvidia-uvm'),
             ('/dev/full', '/dev/nvidia0'))


def handlers_done(fixture):
    for thread in fixture.service._threads:
        thread.join(3)
        assert not thread.is_alive()


@contextmanager
def prepared(profile='cuda', *, other_model=False):
    fds, children = len(os.listdir('/proc/self/fd')), descendants(os.getpid())
    fixture = fixtures.RuntimeTests()
    fixture.setUp()
    fixture.traces = []
    fixture.embeddings = []
    fixture.errors = []
    original = sandbox.launch

    @contextmanager
    def launch(runtime, workspace, *args, **kwargs):
        from kilix_qwen_tts.owned import OwnedExecution
        original_spawn = OwnedExecution.spawn
        def spawn(owner, *args, **kwargs):
            kwargs['stderr'] = errors
            return original_spawn(owner, *args, **kwargs)
        with original(runtime, workspace, *args, **kwargs) as command:
            with (Path(workspace) / 'worker.stderr').open('wb') as errors, \
                    patch.object(OwnedExecution, 'spawn', new=spawn):
                try:
                    yield command
                finally:
                    errors.flush()
                    fixture.errors.append((Path(workspace) / 'worker.stderr').read_text())
                    trace = Path(workspace) / 'worker.trace'
                    fixture.traces.append([json.loads(row) for row in trace.read_text().splitlines()] if trace.exists() else [])
                    embedding = Path(workspace) / 'prompt.embedding'
                    fixture.embeddings.append(embedding.read_bytes() if embedding.exists() else None)

    try:
        with ExitStack() as stack:
            engine = fixture.runtime.python_root / 'bin/python3.12'
            engine.write_text('#!/usr/bin/python3\n' + DOUBLE.read_text())
            fixture.runtime.device = profile
            fixture.runtime.last_device = None
            fixture.runtime.cuda_fallbacks = 0
            fixture.manifest['device'] = profile
            environment = fixture.manifest['environment']
            environment['python_sha256'] = hashlib.sha256(engine.read_bytes()).hexdigest()
            environment['python_root_sha256'] = tree_digest(fixture.runtime.python_root)
            additional = []
            if other_model:
                other = copy.copy(fixture.runtime)
                other.manifest = copy.deepcopy(fixture.manifest)
                other.model_id = 'qwen3-tts-1.7b-base'
                other.model_revision = 'fd4b254389122332181a7c3db7f27e918eec64e3'
                other.device = other.manifest['device'] = 'cpu'
                other.manifest['model'] = {'id': other.model_id, 'revision': other.model_revision}
                additional.append(other)
            stack.enter_context(patch.object(sandbox, 'accelerator_nodes', return_value=STAND_INS))
            stack.enter_context(patch.object(sandbox, 'driver_binds', return_value=()))
            stack.enter_context(patch.object(sandbox, 'GPU_SYSFS', ()))
            stack.enter_context(patch.object(sandbox, 'launch', launch))
            stack.enter_context(patch('sys.stderr', io.StringIO()))
            fixture.service = Service(fixture.runtime, fixture.ipc, additional_runtimes=additional, prompt_cache=True)
            fixture.thread = threading.Thread(target=fixture.service.serve)
            fixture.thread.start()
            assert fixture.service.ready.wait(3)
            yield fixture
    finally:
        fixture.doCleanups()
        assert len(os.listdir('/proc/self/fd')) == fds
        assert descendants(os.getpid()) == children
        print(json.dumps({'private_worker_traces': fixture.traces, 'profile': profile,
            'embedding_sha256': [hashlib.sha256(v).hexdigest() if v is not None else None for v in fixture.embeddings],
            'worker_errors': fixture.errors, 'residual_children': False}))


def submit(fixture, text='speech', *, streaming=False, args=None, job='producer-job', on_chunk=None, timeout=8):
    arguments = args or fixture.arguments(text)
    def delivered(*row):
        if on_chunk is not None:
            on_chunk(*row)
        if text.startswith('fail-after-pcm-'):
            workspaces = list(fixture.jobs.iterdir())
            assert len(workspaces) == 1
            (workspaces[0] / 'pcm-accepted').write_text('accepted')
    with fixture.audio.open('rb') as audio:
        try:
            result = client_request(fixture.ipc, request_value('submit', job_id=job, args=arguments,
                timeout=timeout, stream=streaming), audio.fileno(), on_chunk=delivered if streaming else None)
        except ProtocolError as error:
            handlers_done(fixture)
            raise ProtocolError(error.code, fixture.errors[-1] if fixture.errors else str(error)) from error
    handlers_done(fixture)
    assert list(fixture.jobs.iterdir()) == []
    assert not fixture.service._jobs
    return result


def attempts(trace, kind):
    return [row[1:] for row in trace if row[0] == kind]


class ProducerTransportTests(unittest.TestCase):
    def test_warm_cuda_hit_then_unavailable_cpu_uses_distinct_producer_and_can_reuse(self):
        with prepared() as fixture:
            cuda = submit(fixture)
            warm_cuda = submit(fixture)
            cpu = submit(fixture, 'unavailable')
            warm_cpu = submit(fixture, 'unavailable-again')
            cuda_again = submit(fixture)
            self.assertEqual(cuda[1], warm_cuda[1])
            self.assertEqual(cpu[1], warm_cpu[1])
            self.assertNotEqual(cuda[1], cpu[1])  # Synthetic sentinels only.
            self.assertEqual(cuda_again[1], cuda[1])
            self.assertEqual([attempts(t, 'prompt') for t in fixture.traces],
                             [[['cuda', 'bfloat16']], [], [['cpu', 'float32']], [], []])
            self.assertEqual([attempts(t, 'generate') for t in fixture.traces],
                             [[['cuda', .125, 7]], [['cuda', .125, 7]], [['cpu', .25, 7]],
                              [['cpu', .25, 7]], [['cuda', .125, 7]]])
            self.assertEqual(len(fixture.service._prompt_cache._entries), 2)
            self.assertEqual(fixture.runtime.last_device, 'cuda')
            self.assertEqual(fixture.runtime.cuda_fallbacks, 2)
            self.assertNotIn('prompt_producer', cpu[0])

    def test_fallback_before_prompt_creation_never_poison_cuda_key(self):
        for failure in ('fail-load-oom', 'fail-prompt-cudnn'):
            with self.subTest(failure=failure), prepared() as fixture:
                cpu = submit(fixture, failure)
                cuda = submit(fixture)
                warm_cpu = submit(fixture, failure if failure == 'fail-load-oom' else 'unavailable')
                self.assertNotEqual(cpu[1], cuda[1])
                self.assertEqual(cpu[1], warm_cpu[1])
                self.assertEqual(attempts(fixture.traces[1], 'prompt'), [['cuda', 'bfloat16']])
                self.assertEqual(attempts(fixture.traces[2], 'generate'), [['cpu', .25, 7]])
                self.assertEqual(attempts(fixture.traces[2], 'seed'), [[7]])
                self.assertEqual(len(fixture.service._prompt_cache._entries), 2)

    def test_failure_after_cuda_cache_hit_reselects_cpu_entry_on_retry(self):
        for warm_cpu in (False, True):
            with self.subTest(warm_cpu=warm_cpu), prepared() as fixture:
                submit(fixture)
                if warm_cpu:
                    submit(fixture, 'unavailable')
                result = submit(fixture, 'fail-generate-oom')
                trace = fixture.traces[-1]
                self.assertEqual(attempts(trace, 'load'), [['cuda', 'bfloat16'], ['cpu', 'float32']])
                self.assertEqual(attempts(trace, 'generate'), [['cuda', .125, 7], ['cpu', .25, 7]])
                self.assertEqual(attempts(trace, 'prompt'), [] if warm_cpu else [['cpu', 'float32']])
                self.assertEqual(attempts(trace, 'seed'), [[7], [7]])
                self.assertEqual(fixture.embeddings[-1], fixture.embeddings[1 if warm_cpu else -1])
                self.assertEqual(fixture.runtime.last_device, 'cpu')
                self.assertNotIn('prompt_producer', result[0])
                self.assertEqual(len(fixture.service._prompt_cache._entries), 2)

    def test_cpu_profile_never_constructs_cuda_and_warm_seed_and_consent_are_fresh(self):
        with prepared('cpu') as fixture:
            first = submit(fixture)
            args = fixture.arguments('another text')
            args['seed'] = 19
            args['consent']['recorded_at'] = '2026-10-08T00:00:00Z'
            second = submit(fixture, args=args)
            self.assertEqual(first[1], second[1])
            self.assertEqual(second[0]['seed'], 19)
            self.assertNotEqual(first[0]['conditioning']['consent_sha256'], second[0]['conditioning']['consent_sha256'])
            self.assertEqual(attempts(fixture.traces[-1], 'load'), [['cpu', 'float32']])
            self.assertEqual(attempts(fixture.traces[-1], 'generate'), [['cpu', .25, 19]])
            self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [])
            self.assertEqual(len(fixture.service._prompt_cache._entries), 1)

    def test_concurrent_peer_and_profile_requests_cannot_exchange_embeddings(self):
        with prepared(other_model=True) as fixture:
            submit(fixture)  # Warm the parent peer / CUDA-profile Base0.6B.
            other_args = fixture.arguments()
            other_args['model_id'] = 'qwen3-tts-1.7b-base'
            code = '''
import json, sys
from pathlib import Path
from kilix_qwen_tts.service import client_request
from kilix_qwen_tts.protocol import ProtocolError
value=json.load(sys.stdin)
try:
    with Path(sys.argv[2]).open('rb') as audio:
        result=client_request(Path(sys.argv[1]),value,audio.fileno())
    print(json.dumps({'model':result[0]['model_id']}))
except ProtocolError as error:
    print(json.dumps({'error':error.code}))
'''
            # Hold the first private worker at spawn so overlapping requests
            # deterministically hit BUSY, without timing or a GPU requirement.
            from kilix_qwen_tts.owned import OwnedExecution
            original = OwnedExecution.spawn
            reached, release = threading.Event(), threading.Event()
            def pause(*args, **kwargs):
                reached.set()
                assert release.wait(4)
                return original(*args, **kwargs)
            with ThreadPoolExecutor(max_workers=1) as pool, patch.object(OwnedExecution, 'spawn', new=pause):
                future = pool.submit(submit, fixture, job='parent-warm')
                try:
                    self.assertTrue(reached.wait(3))
                    done = subprocess.run([sys.executable, '-B', '-c', code, str(fixture.ipc), str(fixture.audio)],
                        input=json.dumps(request_value('submit', job_id='other-peer', args=other_args, timeout=5)),
                        text=True, capture_output=True, timeout=8)
                    self.assertEqual(done.returncode, 0, done.stderr)
                    self.assertEqual(json.loads(done.stdout), {'error': 'BUSY'})
                finally:
                    release.set()
                future.result(timeout=8)
            # The other real process now makes the same prompt on its own
            # selected profile; a later parent request still cannot hit it.
            done = subprocess.run([sys.executable, '-B', '-c', code, str(fixture.ipc), str(fixture.audio)],
                input=json.dumps(request_value('submit', job_id='other-peer', args=other_args, timeout=5)),
                text=True, capture_output=True, timeout=8)
            self.assertEqual(done.returncode, 0, done.stderr)
            handlers_done(fixture)
            self.assertEqual(json.loads(done.stdout), {'model': 'qwen3-tts-1.7b-base'})
            self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [['cpu', 'float32']])
            submit(fixture, args=other_args)
            self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [['cpu', 'float32']])
            submit(fixture)
            self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [])
            status = client_request(fixture.ipc, request_value('status'))
            self.assertEqual(status['devices'], {
                'qwen3-tts-0.6b-base': {'profile': 'cuda', 'last_device': 'cuda', 'cuda_fallbacks': 0},
                'qwen3-tts-1.7b-base': {'profile': 'cpu', 'last_device': 'cpu', 'cuda_fallbacks': 0}})
            self.assertEqual(len(fixture.service._prompt_cache._entries), 3)

    def test_bad_producer_or_embedding_refuses_without_cache_commit_and_cleans(self):
        for failure in ('meta-missing', 'meta-wrong', 'meta-unknown', 'meta-type', 'bad-embedding', 'wrong-dtype', 'wrong-device'):
            with self.subTest(failure=failure), prepared() as fixture:
                with self.assertRaises(ProtocolError):
                    submit(fixture, failure)
                handlers_done(fixture)
                self.assertFalse(fixture.service._prompt_cache._entries)
                self.assertEqual(list(fixture.jobs.iterdir()), [])
                self.assertFalse(fixture.service._jobs)
                self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
                submit(fixture)
                self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [['cuda', 'bfloat16']])

    def test_warm_cache_still_requires_current_consent_and_exact_pcm(self):
        with prepared() as fixture:
            submit(fixture)
            before = dict(fixture.service._prompt_cache._entries)
            args = fixture.arguments()
            args['consent']['asserted_by_peer'] = False
            with self.assertRaises(ProtocolError) as error:
                submit(fixture, args=args)
            self.assertEqual(error.exception.code, 'CONSENT_REQUIRED')
            fixture.audio.write_bytes(b'\0' * len(fixture.pcm))
            with self.assertRaises(ProtocolError):
                submit(fixture)
            self.assertEqual(len(fixture.traces), 1)
            self.assertEqual(dict(fixture.service._prompt_cache._entries), before)

    def test_streaming_warm_hits_keep_the_same_producer_and_audio_contract(self):
        with prepared() as fixture:
            for text, where in (('speech', 'cuda'), ('speech', 'cuda'),
                                ('unavailable', 'cpu'), ('unavailable', 'cpu')):
                chunks = []
                result = submit(fixture, text, streaming=True, on_chunk=lambda *row: chunks.append(row))
                self.assertEqual(result[0]['duration_ms'], 100)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(fixture.runtime.last_device, where)
            self.assertEqual([attempts(trace, 'prompt') for trace in fixture.traces],
                             [[['cuda', 'bfloat16']], [], [['cpu', 'float32']], []])


class BackendWorkerTransportTests(unittest.TestCase):
    def test_bare_backend_failures_retry_before_pcm_with_matching_prompt_and_seed(self):
        for family in ('cudnn', 'cublas', 'nvml'):
            for streaming in (False, True):
                with self.subTest(family=family, streaming=streaming), prepared() as fixture:
                    submit(fixture)
                    chunks = []
                    submit(fixture, 'fail-generate-' + family, streaming=streaming,
                           on_chunk=(lambda *row: chunks.append(row)) if streaming else None)
                    trace = fixture.traces[-1]
                    self.assertEqual(attempts(trace, 'load'), [['cuda', 'bfloat16'], ['cpu', 'float32']])
                    self.assertEqual(attempts(trace, 'generate'), [['cuda', .125, 7], ['cpu', .25, 7]])
                    self.assertEqual(attempts(trace, 'seed'), [[7], [7]])
                    self.assertEqual(len(chunks), 1 if streaming else 0)
                    self.assertEqual(fixture.runtime.last_device, 'cpu')
                    self.assertEqual(fixture.runtime.cuda_fallbacks, 1)

    def test_backend_failure_after_first_pcm_never_retries_or_commits(self):
        for family in ('oom', 'cudnn', 'cublas', 'nvml'):
            with self.subTest(family=family), prepared() as fixture:
                submit(fixture)
                before = dict(fixture.service._prompt_cache._entries)
                chunks = []
                with self.assertRaises(ProtocolError) as error:
                    submit(fixture, 'fail-after-pcm-' + family, streaming=True,
                           on_chunk=lambda *row: chunks.append(row))
                self.assertEqual(error.exception.code, 'ENGINE_FAILED')
                handlers_done(fixture)
                self.assertEqual(attempts(fixture.traces[-1], 'load'), [['cuda', 'bfloat16']])
                self.assertEqual(attempts(fixture.traces[-1], 'generate'), [['cuda', .125, 7]])
                self.assertEqual(len(chunks), 1)
                self.assertEqual({k: v[1] for k, v in fixture.service._prompt_cache._entries.items()},
                                 {k: v[1] for k, v in before.items()})
                self.assertEqual(fixture.runtime.last_device, 'cuda')
                self.assertEqual(getattr(fixture.runtime, 'cuda_fallbacks', 0), 0)
                self.assertEqual(list(fixture.jobs.iterdir()), [])
                submit(fixture, 'unavailable')
                self.assertEqual(attempts(fixture.traces[-1], 'prompt'), [['cpu', 'float32']])

    def test_unrelated_user_and_configuration_errors_never_retry(self):
        for family in ('unrelated', 'user-cuda', 'bad-param', 'value'):
            with self.subTest(family=family), prepared() as fixture:
                with self.assertRaises(ProtocolError) as error:
                    submit(fixture, 'fail-generate-' + family)
                self.assertEqual(error.exception.code, 'ENGINE_FAILED')
                handlers_done(fixture)
                self.assertEqual(attempts(fixture.traces[-1], 'load'), [['cuda', 'bfloat16']])
                self.assertFalse(fixture.service._prompt_cache._entries)
                self.assertEqual(list(fixture.jobs.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
