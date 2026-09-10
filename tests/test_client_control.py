"""Controlled clients cancel only their job and preserve result/FD boundaries."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch

from kilix_qwen_tts.protocol import ProtocolError, receive_packet, send_packet
from kilix_qwen_tts.service import SOCKET_NAME, client_request, request_value
import kilix_qwen_tts.results as result_module
import kilix_qwen_tts.runtime as runtime_module
import kilix_qwen_tts.service as service_module
from test_backpressure import descendants, wait_until
from test_streaming import prepared
import test_runtime as fixtures


@contextmanager
def runtime():
    before = len(os.listdir('/proc/self/fd'))
    children = descendants(os.getpid())
    fixture = fixtures.RuntimeTests('runTest')
    fixture.setUp()
    try:
        fixture.start()
        yield fixture
    finally:
        fixture.doCleanups()
        assert len(os.listdir('/proc/self/fd')) == before
        assert descendants(os.getpid()) == children


class ClientControlTests(unittest.TestCase):
    def test_early_cancel_and_invalid_callbacks_refuse_before_connect(self):
        with runtime() as fixture:
            submit = request_value('submit', job_id='controlled', args=fixture.arguments())
            for callback, code in ((lambda: True, 'CANCELED'), (lambda: 1, 'INVALID_REQUEST'),
                                   (lambda: [], 'INVALID_REQUEST'), (1, 'INVALID_REQUEST')):
                with self.subTest(code=code), self.assertRaises(ProtocolError) as error:
                    client_request(fixture.ipc / 'absent', submit, cancelled=callback)
                self.assertEqual(error.exception.code, code)
            with self.assertRaises(ProtocolError) as error:
                client_request(fixture.ipc / 'absent', request_value('status'), cancelled=lambda: False)
            self.assertEqual(error.exception.code, 'INVALID_REQUEST')
            self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_local_cancel_does_not_claim_gated_owned_teardown(self):
        with runtime() as fixture, fixture.audio.open('rb') as audio:
            observed = set()
            before = descendants(os.getpid())
            def cancelled():
                if list(fixture.jobs.glob('*/ready')):
                    observed.update(descendants(os.getpid()) - before)
                    return True
                return False
            request = request_value('submit', job_id='controlled', args=fixture.arguments('escape'), timeout=5)
            stopping = threading.Event()
            release = threading.Event()
            original_stop = runtime_module.stop_process
            def held_stop(process):
                stopping.set()
                assert release.wait(3)
                original_stop(process)
            with patch.object(runtime_module, 'stop_process', side_effect=held_stop):
                try:
                    with self.assertRaises(ProtocolError) as error:
                        client_request(fixture.ipc, request, audio.fileno(), cancelled=cancelled)
                    self.assertEqual(error.exception.code, 'CANCELED')
                    self.assertGreaterEqual(len(observed), 4)
                    self.assertTrue(stopping.wait(2))
                    self.assertTrue(all(Path(f'/proc/{pid}').exists() for pid in observed))
                    self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'busy')
                finally:
                    release.set()
                wait_until(lambda: not fixture.service._jobs, 2)
            self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in observed))
            self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
            self.assertEqual(list(fixture.jobs.iterdir()), [])
            self.assertEqual(os.fstat(audio.fileno()).st_size, len(fixture.pcm))

    def test_stream_callback_cancel_stops_delivery_and_provider_reaps(self):
        before = len(os.listdir('/proc/self/fd'))
        children = descendants(os.getpid())
        with prepared() as fixture, fixture.audio.open('rb') as audio:
            cancel = threading.Event()
            seen = []
            def consume(*row):
                seen.append(row)
                cancel.set()
            request = request_value('submit', job_id='stream-control', args=fixture.arguments('stream-hold'), timeout=5, stream=True)
            with self.assertRaises(ProtocolError) as error:
                client_request(fixture.ipc, request, audio.fileno(), on_chunk=consume, cancelled=cancel.is_set)
            self.assertEqual(error.exception.code, 'CANCELED')
            self.assertEqual(len(seen), 1)
            wait_until(lambda: not fixture.service._jobs, 2)
            self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
            self.assertEqual(list(fixture.jobs.iterdir()), [])
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)
        self.assertEqual(descendants(os.getpid()), children)

    def test_invalid_live_callback_closes_channel_then_provider_reaps(self):
        with runtime() as fixture, fixture.audio.open('rb') as audio:
            def invalid():
                return [] if list(fixture.jobs.glob('*/ready')) else False
            request = request_value('submit', job_id='invalid-control', args=fixture.arguments('escape'), timeout=5)
            with self.assertRaises(ProtocolError) as error:
                client_request(fixture.ipc, request, audio.fileno(), cancelled=invalid)
            self.assertEqual(error.exception.code, 'INVALID_REQUEST')
            wait_until(lambda: not fixture.service._jobs, 2)
            self.assertEqual(list(fixture.jobs.iterdir()), [])

    def test_cancel_after_final_validation_never_delivers_audio(self):
        with runtime() as fixture, fixture.audio.open('rb') as audio:
            cancel = threading.Event()
            original = result_module.validate_result
            def validated(*args):
                original(*args)
                cancel.set()
            request = request_value('submit', job_id='final-control', args=fixture.arguments(), timeout=5)
            with patch.object(result_module, 'validate_result', side_effect=validated), self.assertRaises(ProtocolError) as error:
                client_request(fixture.ipc, request, audio.fileno(), cancelled=cancel.is_set)
            self.assertEqual(error.exception.code, 'CANCELED')
            self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')

    def test_cancel_ack_returns_local_cancellation_without_waiting_for_terminal(self):
        self.silent_peer(acknowledge=True)

    def test_unresponsive_cancel_peer_returns_bounded_local_cancellation(self):
        self.silent_peer(acknowledge=False)

    def test_silent_peer_cannot_extend_controlled_request_deadline(self):
        self.silent_peer(acknowledge=None)

    def test_cancel_or_expired_deadline_after_connect_sends_no_job(self):
        import array
        for mode in ('cancel', 'deadline'):
            with self.subTest(mode=mode):
                before = len(os.listdir('/proc/self/fd'))
                fixture = fixtures.RuntimeTests('runTest')
                fixture.setUp()
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                listener.bind(str(fixture.ipc / SOCKET_NAME))
                (fixture.ipc / SOCKET_NAME).chmod(0o600)
                listener.listen(1)
                listener.settimeout(2)
                packets, failures = [], []
                def server():
                    try:
                        with listener.accept()[0] as channel:
                            channel.settimeout(2)
                            packet, control, _flags, _address = channel.recvmsg(65540, socket.CMSG_SPACE(16), socket.MSG_CMSG_CLOEXEC)
                            packets.append(packet)
                            for level, kind, data in control:
                                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                                    fds = array.array('i'); fds.frombytes(data[:len(data)//fds.itemsize*fds.itemsize])
                                    for fd in fds: os.close(fd)
                    except BaseException as error:
                        failures.append(error)
                thread = threading.Thread(target=server)
                thread.start()
                cancel = threading.Event()
                original_peer = service_module.require_same_uid_peer
                def checked(channel):
                    original_peer(channel)
                    if mode == 'cancel':
                        cancel.set()
                    else:
                        time.sleep(.06)
                try:
                    args = fixture.arguments() if hasattr(fixture, 'arguments') else fixture.value()['args']
                    request = request_value('submit', job_id='not-submitted', args=args, timeout=.03 if mode == 'deadline' else 5)
                    with patch.object(service_module, 'require_same_uid_peer', side_effect=checked), \
                         fixture.audio.open('rb') as audio, self.assertRaises(ProtocolError) as error:
                        client_request(fixture.ipc, request, audio.fileno(), cancelled=cancel.is_set)
                    self.assertEqual(error.exception.code, 'CANCELED' if mode == 'cancel' else 'DEADLINE_EXCEEDED')
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(packets, [b''])
                    self.assertEqual(failures, [])
                finally:
                    listener.close()
                    thread.join(3)
                    fixture.doCleanups()
                    self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def silent_peer(self, *, acknowledge):
        before = len(os.listdir('/proc/self/fd'))
        fixture = fixtures.RuntimeTests('runTest')
        fixture.setUp()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.bind(str(fixture.ipc / SOCKET_NAME))
        (fixture.ipc / SOCKET_NAME).chmod(0o600)
        listener.listen(2)
        listener.settimeout(2)
        submitted = threading.Event()
        complete = threading.Event()
        failures = []
        seen = []
        def server():
            try:
                with listener.accept()[0] as original:
                    original.settimeout(2)
                    request, fds = receive_packet(original)
                    for fd in fds:
                        os.close(fd)
                    seen.append(request)
                    base = {'schema': request['schema'], 'request_id': request['request_id'],
                            'job_id': request['job_id']}
                    send_packet(original, {**base, 'type': 'accepted', 'result': {}})
                    submitted.set()
                    if acknowledge is not None:
                        with listener.accept()[0] as control:
                            control.settimeout(2)
                            cancel, fds = receive_packet(control)
                            for fd in fds:
                                os.close(fd)
                            seen.append(cancel)
                            if acknowledge:
                                send_packet(control, {'schema': cancel['schema'], 'request_id': cancel['request_id'],
                                            'job_id': cancel['job_id'], 'type': 'canceled', 'result': {'cancel_requested': True}})
                            else:
                                assert control.recv(1) == b''
                    try:
                        assert original.recv(1) == b''
                    except ConnectionResetError:
                        # Cancellation may close before consuming accepted.
                        # A closed seqpacket peer can then report reset, not EOF.
                        pass
            except BaseException as error:
                failures.append(error)
            finally:
                complete.set()
        thread = threading.Thread(target=server)
        thread.start()
        try:
            timeout = .45 if acknowledge is None else 5
            request = request_value('submit', job_id='only-this-job', args=fixture.arguments(), timeout=timeout)
            started = time.monotonic()
            with fixture.audio.open('rb') as audio, self.assertRaises(ProtocolError) as error:
                client_request(fixture.ipc, request, audio.fileno(),
                               cancelled=(lambda: False) if acknowledge is None else submitted.is_set)
            elapsed = time.monotonic() - started
            self.assertEqual(error.exception.code, 'DEADLINE_EXCEEDED' if acknowledge is None else 'CANCELED')
            self.assertLess(elapsed, .8)
            self.assertTrue(complete.wait(2))
            expected = [('submit', 'only-this-job')]
            if acknowledge is not None:
                expected.append(('cancel', 'only-this-job'))
                self.assertLessEqual(seen[1]['deadline_ms'], 200)
            self.assertEqual([(r['op'], r['job_id']) for r in seen], expected)
            self.assertEqual(failures, [])
        finally:
            listener.close()
            thread.join(3)
            fixture.doCleanups()
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)


if __name__ == '__main__':
    unittest.main()
