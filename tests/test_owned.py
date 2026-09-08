"""Real kernel cleanup credentials and optional shared lease controls."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.owned import ExecutionPolicy, OwnedExecution, from_options
from kilix_qwen_tts.protocol import ProtocolError

try:
    from voicelib import device_leases
except ImportError:
    device_leases = None

SENDER = r"""
import json, os, socket, sys
fd = int(sys.argv[-1])
sock = socket.socket(fileno=fd)
mode = sys.argv[1]
if mode == 'forged':
    child = os.fork()
    if child:
        sock.close()
        os.waitpid(child, 0)
        raise SystemExit(0)
if mode != 'missing':
    sock.send(b'KILIX_REAPED_V1' if mode != 'wrong' else b'NOT_REAPED')
if mode == 'extra':
    sock.send(b'x')
sock.close()
if mode == 'forged':
    os._exit(0)
"""


class OwnedCleanupTests(unittest.TestCase):
    @staticmethod
    def owner(policy=None):
        return OwnedExecution(policy, job_id='owned-probe', workload='tts-utterance',
                              deadline=time.monotonic() + 10,
                              cancelled=lambda: False, disconnected=lambda: False)

    @staticmethod
    def spawn(owner, mode):
        return owner.spawn([sys.executable, '-I', '-B', '-c', SENDER, mode],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    @staticmethod
    def finish(owner, process):
        try:
            owner.finish(process, lambda child: child.wait(timeout=5))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_only_exact_supervisor_credentials_and_one_marker_prove_cleanup(self):
        for mode in ('success', 'missing', 'wrong', 'extra', 'forged'):
            with self.subTest(mode=mode):
                before = len(os.listdir('/proc/self/fd'))
                with self.owner() as owner:
                    process = self.spawn(owner, mode)
                    if mode == 'success':
                        self.finish(owner, process)
                        self.assertTrue(owner.cleanup_complete)
                    else:
                        with self.assertRaises(ProtocolError) as caught:
                            self.finish(owner, process)
                        self.assertEqual(caught.exception.code, 'SUPERVISOR_FAILED')
                        self.assertFalse(owner.cleanup_complete)
                self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def test_missing_api_and_partial_selection_never_fall_back(self):
        self.assertIsNone(from_options(SimpleNamespace()))
        with self.assertRaises(ProtocolError):
            from_options(SimpleNamespace(lease_namespace='/tmp/lease'))
        with patch.dict(sys.modules, {'voicelib': None}):
            with self.assertRaises(ProtocolError) as caught:
                ExecutionPolicy('cpu-development')
        self.assertEqual(caught.exception.code, 'PROVIDER_UNAVAILABLE')

    @unittest.skipIf(device_leases is None, 'optional shared lease implementation is absent')
    def test_proved_cleanup_and_failed_spawn_release_real_grants_without_fd_leaks(self):
        with tempfile.TemporaryDirectory(prefix='owned-grant-') as folder:
            policy = ExecutionPolicy('cpu-development', namespace=folder + '/coordination')
            before = len(os.listdir('/proc/self/fd'))
            for ending in ('spawn-failure', 'success', 'success'):
                with self.subTest(ending=ending), self.owner(policy) as owner:
                    if ending == 'spawn-failure':
                        with patch('kilix_qwen_tts.owned.subprocess.Popen', side_effect=OSError('fixture')):
                            with self.assertRaises(OSError):
                                self.spawn(owner, 'success')
                        self.assertTrue(owner.cleanup_complete)
                    else:
                        process = self.spawn(owner, 'success')
                        self.assertIn(owner.lease.guard_fd, [int(name) for name in os.listdir('/proc/self/fd')])
                        self.finish(owner, process)
                self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    @unittest.skipIf(device_leases is None, 'optional shared lease implementation is absent')
    def test_missing_proof_quarantines_after_every_process_and_fd_is_gone(self):
        with tempfile.TemporaryDirectory(prefix='owned-quarantine-') as folder:
            policy = ExecutionPolicy('cpu-development', namespace=folder + '/coordination')
            before = len(os.listdir('/proc/self/fd'))
            with self.assertRaises(ProtocolError) as caught:
                with self.owner(policy) as owner:
                    process = self.spawn(owner, 'missing')
                    self.finish(owner, process)
            self.assertEqual(caught.exception.code, 'SUPERVISOR_FAILED')
            self.assertIsNotNone(process.poll())
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)
            with self.assertRaises(ProtocolError) as caught:
                with self.owner(policy):
                    self.fail('unproved grant handed off')
            self.assertEqual(caught.exception.code, 'PROVIDER_UNAVAILABLE')
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)
