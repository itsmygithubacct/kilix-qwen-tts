"""Cache publication follows successful transport and cannot undo a clear."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import threading
import time
import unittest
from unittest.mock import patch

from kilix_qwen_tts import service as module
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.service import client_request, request_value
from test_backpressure import descendants, wait_until
from test_prompt_cache import cached_fixture


@contextmanager
def prepared():
    fds = len(os.listdir('/proc/self/fd'))
    children = descendants(os.getpid())
    try:
        with cached_fixture() as fixture:
            yield fixture
    finally:
        assert len(os.listdir('/proc/self/fd')) == fds
        assert descendants(os.getpid()) == children


def submit(fixture, *, timeout=5):
    with fixture.audio.open('rb') as audio:
        return client_request(fixture.ipc, request_value('submit', job_id='cache-policy',
                              args=fixture.arguments(), timeout=timeout), audio.fileno())


def handlers_done(fixture):
    for thread in fixture.service._threads:
        thread.join(2)
        assert not thread.is_alive()


class CacheCommitTests(unittest.TestCase):
    def test_late_cancel_and_deadline_never_populate_or_supply_next_job(self):
        for mode in ('cancel', 'deadline'):
            with self.subTest(mode=mode), prepared() as fixture, ThreadPoolExecutor(max_workers=1) as pool:
                returned = threading.Event()
                release = threading.Event()
                original = module.run_job
                def pause(*args, **kwargs):
                    result = original(*args, **kwargs)
                    returned.set()
                    assert release.wait(3)
                    return result
                started = time.monotonic()
                with patch.object(module, 'run_job', side_effect=pause):
                    future = pool.submit(submit, fixture, timeout=1 if mode == 'deadline' else 5)
                    try:
                        self.assertTrue(returned.wait(2))
                        self.assertFalse(descendants(os.getpid()))
                        self.assertEqual(list(fixture.jobs.iterdir()), [])
                        self.assertFalse(fixture.service._prompt_cache._entries)
                        self.assertTrue(fixture.service._jobs)
                        if mode == 'cancel':
                            ack = client_request(fixture.ipc, request_value('cancel', job_id='cache-policy'))
                            self.assertTrue(ack['cancel_requested'])
                        else:
                            time.sleep(max(0, started + 1.05 - time.monotonic()))
                    finally:
                        release.set()
                    with self.assertRaises(ProtocolError) as error:
                        future.result(timeout=3)
                    self.assertEqual(error.exception.code, 'CANCELED' if mode == 'cancel' else 'DEADLINE_EXCEEDED')
                handlers_done(fixture)
                cache = fixture.service._prompt_cache
                self.assertFalse(cache._entries)
                submit(fixture)
                handlers_done(fixture)
                self.assertEqual((cache.hits, cache.misses, len(cache._entries)), (0, 2, 1))

    def test_terminal_send_refusal_keeps_embedding_uncommitted(self):
        with prepared() as fixture:
            original = module.send_packet
            def refuse(channel, value, descriptor=None):
                if value['type'] == 'result':
                    raise ProtocolError('TRANSPORT_ERROR', 'synthetic terminal send refusal')
                return original(channel, value, descriptor)
            with patch.object(module, 'send_packet', side_effect=refuse), self.assertRaises(ProtocolError) as error:
                submit(fixture)
            self.assertEqual(error.exception.code, 'TRANSPORT_ERROR')
            handlers_done(fixture)
            self.assertFalse(fixture.service._prompt_cache._entries)
            self.assertFalse(fixture.service._jobs)
            self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_unload_or_shutdown_after_terminal_send_prevents_old_commit(self):
        for mode in ('unload', 'shutdown'):
            with self.subTest(mode=mode), prepared() as fixture, ThreadPoolExecutor(max_workers=1) as pool:
                sent = threading.Event()
                release = threading.Event()
                original = module.send_packet
                def pause(channel, value, descriptor=None):
                    original(channel, value, descriptor)
                    if value['type'] == 'result':
                        sent.set()
                        assert release.wait(3)
                with patch.object(module, 'send_packet', side_effect=pause):
                    future = pool.submit(submit, fixture)
                    try:
                        self.assertTrue(sent.wait(2))
                        future.result(timeout=2)
                        self.assertFalse(fixture.service._jobs)
                        self.assertFalse(descendants(os.getpid()))
                        if mode == 'unload':
                            self.assertFalse(client_request(fixture.ipc, request_value('unload'))['loaded'])
                        else:
                            fixture.service.stop()
                        self.assertFalse(fixture.service._prompt_cache._entries)
                    finally:
                        release.set()
                    handlers_done(fixture)
                self.assertFalse(fixture.service._prompt_cache._entries)

    def test_clear_started_during_cache_commit_wins_after_put_returns(self):
        for mode in ('unload', 'shutdown'):
            with self.subTest(mode=mode), prepared() as fixture, ThreadPoolExecutor(max_workers=2) as pool:
                committing = threading.Event()
                release = threading.Event()
                clear_received = threading.Event()
                cache = fixture.service._prompt_cache
                original_put = cache.put
                original_receive = module.receive_packet
                def pause(*args):
                    committing.set()
                    assert release.wait(3)
                    return original_put(*args)
                def receive(channel):
                    value, descriptors = original_receive(channel)
                    if value.get('type') == 'request' and value.get('op') == 'unload':
                        clear_received.set()
                    return value, descriptors
                with patch.object(cache, 'put', side_effect=pause), \
                     patch.object(module, 'receive_packet', side_effect=receive):
                    success = pool.submit(submit, fixture)
                    try:
                        self.assertTrue(committing.wait(2))
                        success.result(timeout=2)
                        if mode == 'unload':
                            clear = pool.submit(client_request, fixture.ipc, request_value('unload'))
                            self.assertTrue(clear_received.wait(2))
                        else:
                            clear = pool.submit(fixture.service.stop)
                            wait_until(fixture.service.stopping.is_set, 2)
                        self.assertFalse(clear.done())
                    finally:
                        release.set()
                    clear.result(timeout=2)
                    handlers_done(fixture)
                self.assertFalse(cache._entries)


if __name__ == '__main__':
    unittest.main()
