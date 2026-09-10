"""Real private services and engine trees under the shared execution API."""
from __future__ import annotations

from contextlib import contextmanager
import json
import subprocess
import sys
from pathlib import Path
import socket
import time
import unittest

from kilix_qwen_tts.owned import ExecutionPolicy
from kilix_qwen_tts.protocol import ProtocolError, receive_packet, send_packet
from kilix_qwen_tts.service import client_request, request_value
import test_runtime as runtime_fixture

try:
    from voicelib import device_leases
except ImportError:
    device_leases = None


@contextmanager
def prepared(mode='success'):
    fixture = runtime_fixture.RuntimeTests('runTest')
    fixture.setUp()
    try:
        fixture.start()
        fixture.service.execution_policy = ExecutionPolicy('cpu-development',
            namespace=str(fixture.root / 'shared-lease'))
        yield fixture
    finally:
        fixture.doCleanups()


def value(fixture, mode='success', timeout=5):
    return request_value('submit', job_id='leased-job', timeout=timeout, args=fixture.arguments({'success': 'speech', 'fail': 'bad'}.get(mode, mode)))


def invoke(fixture, mode='success', timeout=5):
    with fixture.audio.open('rb') as audio:
        return client_request(fixture.ipc, value(fixture, mode, timeout), audio.fileno())


def raw(fixture, mode='success', timeout=5):
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel.settimeout(5)
    channel.connect(str(fixture.service.path))
    try:
        with fixture.audio.open('rb') as audio:
            send_packet(channel, value(fixture, mode, timeout), audio.fileno())
        event, descriptors = receive_packet(channel)
        assert event['type'] == 'accepted' and not descriptors
        return channel
    except BaseException:
        channel.close()
        raise


def waiting(fixture):
    return bool(list(fixture.jobs.glob('*/ready')))


def until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError('owned fixture did not reach expected state')
        time.sleep(.005)


def assert_reusable(fixture):
    until(lambda: client_request(fixture.ipc, request_value('status'))['provider_state'] == 'ready')
    lease = device_leases.acquire(job_id='next-owner', workload='llm-turn', device='other-label',
                                 deadline=time.monotonic()+1,
                                 namespace=str(fixture.root/'shared-lease'))
    lease.release(cleanup_complete=True)


@unittest.skipIf(device_leases is None, 'optional shared lease implementation is absent')
class LeasedRuntimeTests(unittest.TestCase):
    def test_actual_success_and_engine_failure_release_only_after_cleanup(self):
        for mode in ('success', 'fail'):
            with self.subTest(mode=mode), prepared(mode) as fixture:
                if mode == 'success':
                    invoke(fixture)
                else:
                    with self.assertRaises(ProtocolError):
                        invoke(fixture, mode)
                assert_reusable(fixture)

    def test_actual_cancel_deadline_and_disconnect_allow_safe_handoff(self):
        for ending in ('cancel', 'deadline', 'disconnect'):
            with self.subTest(ending=ending), prepared('escape') as fixture:
                with raw(fixture, 'escape', .6 if ending == 'deadline' else 5) as channel:
                    until(lambda: waiting(fixture))
                    if ending == 'cancel':
                        client_request(fixture.ipc, request_value('cancel', job_id='leased-job'))
                    if ending == 'disconnect':
                        channel.close()
                    else:
                        event, descriptors = receive_packet(channel)
                        self.assertFalse(descriptors)
                        self.assertEqual(event['type'], 'error')
                        self.assertEqual(event['error']['code'],
                            'CANCELED' if ending == 'cancel' else 'DEADLINE_EXCEEDED')
                assert_reusable(fixture)

    def test_queued_cancel_and_deadline_do_not_start_an_engine_or_release_other_owner(self):
        for ending in ('cancel', 'deadline'):
            with self.subTest(ending=ending), prepared() as fixture:
                policy = fixture.service.execution_policy
                holder = device_leases.acquire(job_id='external-holder', workload='llm-turn',
                    device='another-device', deadline=time.monotonic()+5, namespace=policy.namespace)
                try:
                    with raw(fixture, timeout=.15 if ending == 'deadline' else 5) as channel:
                        event, descriptors = receive_packet(channel)
                        self.assertEqual(event['type'], 'queued')
                        self.assertFalse(descriptors)
                        self.assertEqual(event['result']['lease_version'], device_leases.VERSION)
                        self.assertFalse(waiting(fixture))
                        if ending == 'cancel':
                            client_request(fixture.ipc, request_value('cancel', job_id='leased-job'))
                        event, descriptors = receive_packet(channel)
                        self.assertFalse(descriptors)
                        self.assertEqual(event['type'], 'error')
                        self.assertEqual(event['error']['code'],
                            'CANCELED' if ending == 'cancel' else 'DEADLINE_EXCEEDED')
                    holder.check()
                finally:
                    holder.release(cleanup_complete=True)
                assert_reusable(fixture)
                invoke(fixture)

    def test_supervisor_sigkill_requires_persistent_quarantine_after_all_fds_close(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('owned_loss_probe.py'))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        print('OWNED_LOSS_EVIDENCE ' + result.stdout.strip())
        self.assertEqual(evidence['terminal'], 'SUPERVISOR_FAILED')
        self.assertEqual(evidence['successor_after_complete_observer_cleanup'], 'unavailable')
        self.assertTrue(evidence['unrelated_preserved'])
        self.assertGreaterEqual(evidence['guard_holders'], 2)
        self.assertEqual(evidence['children_after_probe_cleanup'], [])
