"""Real worker cleanup/cache gates with a private deterministic grant double.

These verify the existing OwnedExecution ordering. They do not claim that an
installed broker or optional voicelib.device_leases adapter was exercised.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
import unittest

from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.service import client_request, request_value
from test_backpressure import descendants, wait_until
import test_prompt_producers as fixtures


class LeaseError(Exception):
    pass


class PrivateGrant:
    def __init__(self, fixture):
        self.fixture = fixture
        self.guard_fd, self.writer = os.pipe()
        self.released = False
        self.checks = 0

    def check(self):
        self.checks += 1

    def release(self, *, cleanup_complete):
        try:
            assert cleanup_complete is True
            assert not descendants(os.getpid())
            assert list(self.fixture.jobs.iterdir()) == []
            self.released = True
        finally:
            os.close(self.guard_fd)
            os.close(self.writer)


class PrivatePolicy:
    module = SimpleNamespace(LeaseError=LeaseError)

    def __init__(self, fixture, refusal=None):
        self.fixture, self.refusal = fixture, refusal
        self.grants = []

    def acquire(self, **kwargs):
        if self.refusal:
            raise ProtocolError(self.refusal, 'cuDNN error: CUDNN_STATUS_INTERNAL_ERROR')
        grant = PrivateGrant(self.fixture)
        self.grants.append(grant)
        return grant


class PromptGrantTests(unittest.TestCase):
    def test_fallback_and_engine_failure_release_after_proved_teardown(self):
        for text in ('fail-generate-cudnn', 'fail-generate-unrelated'):
            with self.subTest(text=text), fixtures.prepared() as fixture:
                policy = PrivatePolicy(fixture)
                fixture.service.execution_policy = policy
                if text.endswith('unrelated'):
                    with self.assertRaises(ProtocolError):
                        fixtures.submit(fixture, text)
                    fixtures.handlers_done(fixture)
                    self.assertFalse(fixture.service._prompt_cache._entries)
                else:
                    fixtures.submit(fixture, text)
                    self.assertEqual(len(fixture.service._prompt_cache._entries), 1)
                self.assertEqual(len(policy.grants), 1)
                self.assertTrue(policy.grants[0].released)
                self.assertGreater(policy.grants[0].checks, 0)

    def test_admission_refusal_never_starts_or_retries_worker(self):
        for code in ('BUSY', 'CANCELED', 'DEADLINE_EXCEEDED', 'INVALID_RUNTIME', 'CONSENT_REQUIRED'):
            with self.subTest(code=code), fixtures.prepared() as fixture:
                policy = PrivatePolicy(fixture, refusal=code)
                fixture.service.execution_policy = policy
                with self.assertRaises(ProtocolError) as error:
                    fixtures.submit(fixture)
                self.assertEqual(error.exception.code, code)
                self.assertFalse(policy.grants)
                self.assertFalse(fixture.traces)
                self.assertFalse(fixture.service._prompt_cache._entries)
                self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_cancel_and_deadline_reap_worker_before_release_and_never_commit(self):
        for ending in ('cancel', 'deadline'):
            with self.subTest(ending=ending), fixtures.prepared() as fixture, ThreadPoolExecutor(max_workers=1) as pool:
                policy = PrivatePolicy(fixture)
                fixture.service.execution_policy = policy
                future = pool.submit(fixtures.submit, fixture, 'wait-generate',
                                     timeout=1 if ending == 'deadline' else 8)
                wait_until(lambda: bool(list(fixture.jobs.glob('*/ready'))), 3)
                self.assertFalse(policy.grants[0].released)
                if ending == 'cancel':
                    result = client_request(fixture.ipc, request_value('cancel', job_id='producer-job'))
                    self.assertTrue(result['cancel_requested'])
                with self.assertRaises(ProtocolError) as error:
                    future.result(timeout=5)
                self.assertEqual(error.exception.code, 'CANCELED' if ending == 'cancel' else 'DEADLINE_EXCEEDED')
                self.assertTrue(policy.grants[0].released)
                self.assertEqual(fixtures.attempts(fixture.traces[-1], 'load'), [['cuda', 'bfloat16']])
                self.assertFalse(fixture.service._prompt_cache._entries)
                self.assertEqual(list(fixture.jobs.iterdir()), [])
                fixtures.submit(fixture)
                self.assertTrue(policy.grants[-1].released)
