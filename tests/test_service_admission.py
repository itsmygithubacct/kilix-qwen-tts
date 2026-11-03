"""Transient handler allocation failures must not leak admission resources."""
import os
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.runtime import MODEL_CANDIDATES
from kilix_qwen_tts.service import Service, client_request, request_value


class ServiceAdmissionTests(unittest.TestCase):
    def test_failed_handler_start_closes_peer_and_restores_capacity(self):
        before = len(os.listdir('/proc/self/fd'))
        with tempfile.TemporaryDirectory(prefix='qwen-admission-') as folder:
            model_id = 'qwen3-tts-0.6b-customvoice'
            revision, mode = MODEL_CANDIDATES[model_id]
            service = Service(SimpleNamespace(model_id=model_id, model_revision=revision,
                                             mode=mode), Path(folder))
            failures = []
            def serve():
                try:
                    service.serve()
                except BaseException as error:
                    failures.append(error)
            server = threading.Thread(target=serve)
            server.start()
            try:
                self.assertTrue(service.ready.wait(2), failures)
                start = threading.Thread.start
                attempts = []
                def unavailable(thread):
                    if thread._target == service._handle:
                        attempts.append(thread)
                        raise RuntimeError('synthetic thread resource exhaustion')
                    return start(thread)
                with patch.object(threading.Thread, 'start', unavailable):
                    # More failures than the eight-handler capacity proves
                    # every admission permit is returned, not just the socket.
                    for _ in range(12):
                        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
                            client.settimeout(2)
                            client.connect(str(service.path))
                            self.assertEqual(client.recv(1), b'')
                        self.assertFalse(failures, failures)
                        self.assertTrue(server.is_alive())
                self.assertEqual(len(attempts), 12)
                self.assertEqual(client_request(Path(folder), request_value('status'))['provider_state'], 'ready')
            finally:
                service.stop()
                server.join(3)
                self.assertFalse(server.is_alive())
            self.assertFalse(failures, failures)
            self.assertFalse(service.path.exists())
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)


if __name__ == '__main__':
    unittest.main()
