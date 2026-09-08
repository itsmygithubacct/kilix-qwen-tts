"""Actual client validation may finish after its original delivery budget."""
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kilix_qwen_tts import protocol, results, service, streaming
from test_client_control import descendants, runtime
import test_runtime as fixtures


class DeliveryDeadlineTests(unittest.TestCase):
    def test_final_delivery_checks_time_after_real_validation(self):
        for mode in ('normal', 'deadline', 'cancel', 'legacy'):
            with self.subTest(mode=mode), runtime() as fixture, fixture.audio.open('rb') as source:
                cancel = threading.Event()
                original = results.validate_result
                budget = 1.0
                began = time.monotonic()
                def validated(*args):
                    value = original(*args)
                    self.assertLess(time.monotonic() - began, budget)
                    if mode in ('deadline', 'legacy'):
                        time.sleep(max(0, began + budget + .08 - time.monotonic()))
                    if mode == 'cancel':
                        cancel.set()
                    return value
                request = service.request_value('submit', job_id='delivery-budget',
                                                args=fixture.arguments(), timeout=budget)
                with patch.object(results, 'validate_result', validated):
                    if mode in ('normal', 'legacy'):
                        result, payload = service.client_request(
                            fixture.ipc, request, source.fileno(),
                            cancelled=None if mode == 'legacy' else cancel.is_set)
                        self.assertTrue(result['audio'])
                        self.assertGreater(len(payload), 44)
                    else:
                        with self.assertRaises(protocol.ProtocolError) as error:
                            service.client_request(fixture.ipc, request, source.fileno(), cancelled=cancel.is_set)
                        self.assertEqual(error.exception.code, 'CANCELED' if mode == 'cancel' else 'DEADLINE_EXCEEDED')
                os.fstat(source.fileno())
                self.assertEqual(service.client_request(fixture.ipc, service.request_value('status'))['provider_state'], 'ready')

    def test_short_control_delivery_retains_its_original_deadline(self):
        with runtime() as fixture:
            original = service.receive_packet
            def received(*args):
                value = original(*args)
                if value[0].get('type') != 'request':
                    time.sleep(.16)
                return value
            with patch.object(service, 'receive_packet', received), self.assertRaises(protocol.ProtocolError) as error:
                service.client_request(fixture.ipc, service.request_value('status', timeout=.1))
            self.assertEqual(error.exception.code, 'DEADLINE_EXCEEDED')
            self.assertEqual(service.client_request(fixture.ipc, service.request_value('status'))['provider_state'], 'ready')

    def test_external_pcm_callback_refuses_expired_delivery(self):
        for mode in ('normal', 'deadline', 'cancel'):
            with self.subTest(mode=mode):
                before = len(os.listdir('/proc/self/fd'))
                children = descendants(os.getpid())
                fixture = fixtures.RuntimeTests('runTest')
                fixture.setUp()
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                listener.bind(str(fixture.ipc / service.SOCKET_NAME))
                (fixture.ipc / service.SOCKET_NAME).chmod(0o600)
                listener.listen(1)
                listener.settimeout(2)
                cancelled = threading.Event()
                delivered, failures = [], []
                pcm = b'\x01\x00' * 24
                def peer():
                    try:
                        with listener.accept()[0] as channel:
                            channel.settimeout(2)
                            request, fds = protocol.receive_packet(channel)
                            for fd in fds:
                                os.close(fd)
                            with tempfile.TemporaryFile() as source:
                                source.write(pcm)
                                source.flush()
                                fd = os.open(f'/proc/self/fd/{source.fileno()}', os.O_RDONLY | os.O_CLOEXEC)
                                try:
                                    protocol.send_packet(channel, {
                                        'schema': request['schema'], 'request_id': request['request_id'],
                                        'job_id': request['job_id'], 'type': 'chunk',
                                        'result': streaming.chunk_metadata(0, 0, pcm)}, fd)
                                finally:
                                    os.close(fd)
                            self.assertEqual(channel.recv(1), b'')
                    except BaseException as error:
                        failures.append(error)
                original = streaming.ClientStream.append
                budget = .5
                began = time.monotonic()
                def paused(reader, *args):
                    self.assertLess(time.monotonic() - began, budget)
                    if mode != 'normal':
                        time.sleep(max(0, began + budget + .08 - time.monotonic()))
                    if mode == 'cancel':
                        cancelled.set()
                    return original(reader, *args)
                thread = threading.Thread(target=peer)
                thread.start()
                try:
                    with fixture.audio.open('rb') as source, patch.object(streaming.ClientStream, 'append', paused):
                        with self.assertRaises(protocol.ProtocolError) as error:
                            service.client_request(fixture.ipc, service.request_value(
                                'submit', job_id='chunk-budget', args=fixture.arguments(), timeout=budget, stream=True),
                                source.fileno(), cancelled=cancelled.is_set,
                                on_chunk=lambda *row: delivered.append(row))
                        self.assertEqual(error.exception.code, 'DEADLINE_EXCEEDED')
                        os.fstat(source.fileno())
                    self.assertEqual(delivered, [(0, 0, pcm)] if mode == 'normal' else [])
                finally:
                    listener.close()
                    thread.join(3)
                    fixture.doCleanups()
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])
                self.assertEqual(len(os.listdir('/proc/self/fd')), before)
                self.assertEqual(descendants(os.getpid()), children)


if __name__ == '__main__':
    unittest.main()
