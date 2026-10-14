"""Prompt scope isolation, bounded retention and real cached worker transport."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from kilix_qwen_tts.prompt_cache import (PromptCache, MAGIC, ENCODING, PAYLOAD_BYTES,
                                        peer_scope, prompt_key, read_embedding, validate_embedding)
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import tree_digest
from kilix_qwen_tts.service import Service, client_request, request_value
import test_runtime as fixtures

EMBEDDING = MAGIC + ENCODING.pack(*([0.125] * 1024))


class CacheTests(unittest.TestCase):
    def test_lru_population_idle_expiry_and_explicit_clear(self):
        now = [0.0]
        cache = PromptCache(clock=lambda: now[0])
        keys = [hashlib.sha256(str(i).encode()).digest() for i in range(9)]
        for key in keys[:8]:
            cache.put(key, EMBEDDING)
        self.assertEqual(cache.get(keys[0]), EMBEDDING)
        cache.put(keys[8], EMBEDDING)
        self.assertEqual(len(cache._entries), 8)
        self.assertIsNone(cache.get(keys[1]))
        self.assertEqual(cache.get(keys[0]), EMBEDDING)
        now[0] = 301
        cache.expire()
        self.assertFalse(cache._entries)
        cache.put(keys[0], EMBEDDING)
        cache.clear()
        self.assertIsNone(cache.get(keys[0]))
        self.assertLessEqual(sum(len(v[1]) for v in cache._entries.values()), 8 * PAYLOAD_BYTES)

    def test_all_peer_model_audio_and_authorized_scope_inputs_bind(self):
        runtime = fixtures.RuntimeTests()
        runtime.setUp()
        try:
            args = runtime.arguments()
            scope = (123, os.geteuid(), os.getegid(), '987654')
            key = prompt_key(scope, runtime.manifest, args, 'cpu-float32')
            for changed_scope in ((124, scope[1], scope[2], scope[3]), (*scope[:3], '987655')):
                self.assertNotEqual(prompt_key(changed_scope, runtime.manifest, args, 'cpu-float32'), key)
            for group, field in (('model', 'revision'), ('files', 'model/data'), ('environment', 'site_packages_sha256')):
                changed = copy.deepcopy(runtime.manifest)
                changed[group][field] += '-changed'
                self.assertNotEqual(prompt_key(scope, changed, args, 'cpu-float32'), key)
            for field, value in (('sha256', 'f'*64), ('sample_rate_hz', 48000), ('sample_format', 'f32le')):
                changed = copy.deepcopy(args)
                changed['prompt_audio'][field] = value
                self.assertNotEqual(prompt_key(scope, runtime.manifest, changed, 'cpu-float32'), key)
            changed = copy.deepcopy(args)
            changed['consent'].update(allowed_use='named-purpose', purpose='different project')
            self.assertNotEqual(prompt_key(scope, runtime.manifest, changed, 'cpu-float32'), key)
            changed = copy.deepcopy(args)
            changed['consent']['recorded_at'] = '2026-09-08T00:00:00Z'
            changed.update(text='new text', seed=8)
            self.assertEqual(prompt_key(scope, runtime.manifest, changed, 'cpu-float32'), key)
            self.assertIsNone(prompt_key(None, runtime.manifest, args, 'cpu-float32'))
            self.assertNotEqual(prompt_key(scope, runtime.manifest, args, 'cuda0-bfloat16'), key)
        finally:
            runtime.doCleanups()

    def test_peer_process_identity_is_observed_from_real_socket(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with left, right:
            observed = peer_scope(left)
            self.assertEqual(observed[:3], (os.getpid(), os.geteuid(), os.getegid()))
            self.assertTrue(observed[3].isdigit())

    def test_embedding_parser_and_input_file_refuse_bad_bytes_or_unsafe_nodes(self):
        for payload in (b'', EMBEDDING[:-1], EMBEDDING+b'x', b'X'+EMBEDDING[1:],
                        MAGIC+ENCODING.pack(*([float('nan')]*1024)),
                        MAGIC+ENCODING.pack(*([float('inf')]*1024)),
                        MAGIC+ENCODING.pack(*([10_001.0]*1024))):
            with self.subTest(length=len(payload)), self.assertRaises(ProtocolError):
                validate_embedding(payload)
        self.assertEqual(validate_embedding(EMBEDDING), EMBEDDING)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'embedding'
            path.write_bytes(EMBEDDING)
            path.chmod(0o600)
            self.assertEqual(read_embedding(path), EMBEDDING)
            path.chmod(0o444)
            self.assertEqual(read_embedding(path, readonly_mount=True), EMBEDDING)
            with self.assertRaises(ProtocolError):
                read_embedding(path)
            link = Path(d)/'link'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                read_embedding(link)
            fifo = Path(d)/'fifo'
            os.mkfifo(fifo, 0o600)
            started = time.monotonic()
            with self.assertRaises(ProtocolError):
                read_embedding(fifo)
            self.assertLess(time.monotonic()-started, .2)


CACHE_TAIL = '''
if request.get('prompt_cache'):
    import struct
    embedding=b'KQPE\\x01\\x00\\x00\\x00'+struct.pack('<1024f',*([.125]*1024))
    producer='cuda0-bfloat16' if used=='cuda' else 'cpu-float32'
    result['prompt_producer']=producer
    if producer in request.get('prompt_cache_inputs',[]):
        cached=pathlib.Path('/opt/prompt.'+producer+'.embedding')
        assert cached.read_bytes()==embedding
        try:cached.write_bytes(b'wrong');raise AssertionError('cache is writable')
        except OSError as error:assert error.errno==30
    if args['text']=='malformed-embedding':embedding=b'bad'
    os.umask(0o077)
    pathlib.Path('/job/prompt.embedding').write_bytes(embedding)
print(json.dumps(result))
'''


@contextmanager
def cached_fixture():
    fixture = fixtures.RuntimeTests()
    fixture.setUp()
    try:
        engine = fixture.runtime.python_root/'bin/python3.12'
        source = engine.read_text()
        assert source.count('print(json.dumps(result))') == 1
        engine.write_text(source.replace('print(json.dumps(result))', CACHE_TAIL))
        fixture.runtime.manifest['environment']['python_sha256'] = hashlib.sha256(engine.read_bytes()).hexdigest()
        fixture.runtime.manifest['environment']['python_root_sha256'] = tree_digest(fixture.runtime.python_root)
        fixture.service = Service(fixture.runtime, fixture.ipc, prompt_cache=True)
        fixture.thread = threading.Thread(target=fixture.service.serve)
        fixture.thread.start()
        assert fixture.service.ready.wait(2)
        yield fixture
    finally:
        fixture.doCleanups()


class ProcessTests(unittest.TestCase):
    def send(self, fixture, text='speech', args=None):
        value = request_value('submit', job_id='cache-job', args=args or fixture.arguments(text), timeout=5)
        with fixture.audio.open('rb') as source:
            return client_request(fixture.ipc, value, source.fileno())

    def test_actual_worker_hit_fresh_consent_and_unload_clear(self):
        with cached_fixture() as fixture:
            first = self.send(fixture)
            cache = fixture.service._prompt_cache
            self.assertEqual((cache.hits, cache.misses), (0, 1))
            args = fixture.arguments('different text')
            args['consent']['recorded_at'] = '2026-09-08T00:00:00Z'
            second = self.send(fixture, args=args)
            self.assertEqual((cache.hits, cache.misses), (1, 1))
            self.assertEqual(first[1], second[1])
            self.assertEqual(second[0]['conditioning']['consent_sha256'], hashlib.sha256(
                json.dumps(args['consent'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
            self.assertNotEqual(first[0]['conditioning']['consent_sha256'], second[0]['conditioning']['consent_sha256'])
            self.assertEqual(list(fixture.jobs.iterdir()), [])
            client_request(fixture.ipc, request_value('unload', timeout=5))
            self.assertFalse(cache._entries)
            self.send(fixture)
            self.assertEqual((cache.hits, cache.misses), (1, 2))
            fixture.stop()
            self.assertFalse(cache._entries)

    def test_other_process_cannot_hit_same_prompt_cache_entry(self):
        with cached_fixture() as fixture:
            self.send(fixture)
            script = '''
import json,sys
from pathlib import Path
from kilix_qwen_tts.service import client_request
value=json.load(sys.stdin)
with Path(sys.argv[2]).open('rb') as audio:client_request(Path(sys.argv[1]),value,audio.fileno())
'''
            value = request_value('submit', job_id='other-peer', args=fixture.arguments(), timeout=5)
            completed = subprocess.run([sys.executable, '-B', '-c', script, str(fixture.ipc), str(fixture.audio)],
                                       input=json.dumps(value), text=True, capture_output=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((fixture.service._prompt_cache.hits, fixture.service._prompt_cache.misses), (0, 2))
            self.send(fixture)
            self.assertEqual(fixture.service._prompt_cache.hits, 1)

    def test_failed_embedding_and_changed_prompt_never_populate(self):
        with cached_fixture() as fixture:
            with self.assertRaises(ProtocolError):
                self.send(fixture, 'malformed-embedding')
            self.assertFalse(fixture.service._prompt_cache._entries)
            args = fixture.arguments()
            fixture.audio.write_bytes(b'\0' * len(fixture.pcm))
            with self.assertRaises(ProtocolError):
                self.send(fixture, args=args)
            self.assertFalse(fixture.service._prompt_cache._entries)
            self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
            self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_failed_or_missing_consent_cannot_reuse_valid_embedding(self):
        with cached_fixture() as fixture:
            self.send(fixture)
            args = fixture.arguments()
            args['consent']['asserted_by_peer'] = False
            with self.assertRaises(ProtocolError) as error:
                self.send(fixture, args=args)
            self.assertEqual(error.exception.code, 'CONSENT_REQUIRED')
            self.assertEqual(fixture.service._prompt_cache.hits, 0)

    def test_cancel_during_cached_worker_keeps_new_key_uncommitted(self):
        from concurrent.futures import ThreadPoolExecutor
        with cached_fixture() as fixture, ThreadPoolExecutor(max_workers=1) as pool:
            self.send(fixture)
            args = fixture.arguments("escape")
            args["consent"].update(allowed_use="named-purpose", purpose="separate canceled project")
            future = pool.submit(self.send, fixture, args=args)
            until = time.monotonic()+3
            while not list(fixture.jobs.glob("*/ready")):
                if time.monotonic() >= until:
                    self.fail("owned escape worker never became ready")
                time.sleep(.005)
            client_request(fixture.ipc, request_value("cancel", job_id="cache-job"))
            with self.assertRaises(ProtocolError) as error:
                future.result(timeout=5)
            self.assertEqual(error.exception.code, "CANCELED")
            self.assertEqual(len(fixture.service._prompt_cache._entries), 1)
            self.assertEqual(list(fixture.jobs.iterdir()), [])
            self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
