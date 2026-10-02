"""The stager binds a uv environment's symlinked interpreter as the runtime verifies it."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from kilix_qwen_tts.runtime import digest_file

_SPEC = importlib.util.spec_from_file_location(
    "stage_runtime", Path(__file__).resolve().parents[1] / "tools/stage_runtime.py")
stage_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_runtime)


class StageRuntimeTests(unittest.TestCase):
    def test_escaped_probe_descendants_reaped_on_success_cancel_and_deadline(self):
        for mode in ('success', 'cancel', 'deadline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                pid_path = Path(tmp)/'child.pid'
                program = ('import subprocess,sys,time; '
                    'p=subprocess.Popen([sys.executable,"-I","-c",'
                    '"import time; time.sleep(30)"],start_new_session=True); '
                    f'open({str(pid_path)!r},"w").write(str(p.pid)); '
                    + ('print("complete")' if mode == 'success' else 'time.sleep(30)'))
                def check():
                    if mode == 'cancel' and pid_path.exists():
                        raise InterruptedError('cancelled with escaped child')
                deadline = time.monotonic() + 1 if mode == 'deadline' else None
                try:
                    if mode == 'success':
                        self.assertEqual(stage_runtime.checked_output(
                            [sys.executable, '-I', '-c', program], check), b'complete\n')
                    else:
                        with self.assertRaises(InterruptedError if mode == 'cancel' else TimeoutError):
                            stage_runtime.checked_output([sys.executable, '-I', '-c', program],
                                                         check, deadline=deadline)
                    self.assertTrue(pid_path.exists(), 'escaped child was actually launched')
                    pid = int(pid_path.read_text())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                finally:
                    # Failure of the ownership regression must not leak its fixture.
                    if pid_path.exists():
                        try:
                            os.kill(int(pid_path.read_text()), 9)
                        except ProcessLookupError:
                            pass

    def test_short_global_deadline_stops_probe_and_reaps_child(self):
        real_popen = stage_runtime.subprocess.Popen
        processes = []
        def capture(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        start = time.monotonic()
        with patch.object(stage_runtime.subprocess, 'Popen', capture):
            with self.assertRaises(TimeoutError):
                stage_runtime.checked_output([sys.executable, '-I', '-c',
                    'import time; time.sleep(30)'], deadline=start + .08)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    def test_cancellation_after_first_probe_prevents_next_probe(self):
        real_popen = stage_runtime.subprocess.Popen
        processes = []
        def capture(*args, **kwargs):
            process = real_popen([sys.executable, '-I', '-c',
                'print('+repr(stage_runtime.ENGINE_COMMIT)+')'], **kwargs)
            processes.append(process)
            return process
        def check():
            if processes and processes[0].poll() is not None:
                raise InterruptedError('cancelled after probe')
        with patch.object(stage_runtime.subprocess, 'Popen', capture):
            with self.assertRaises(InterruptedError):
                stage_runtime.environment_record(Path('/unused'), Path('/unused'), check)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    def test_interpreter_hash_observes_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'python'
            path.write_bytes(b'interpreter')
            def check():
                raise InterruptedError('cancelled while hashing interpreter')
            with self.assertRaises(InterruptedError):
                stage_runtime.interpreter_sha256(path, check)

    def test_symlinked_environment_interpreter_binds_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "python-root/bin/python3.12"
            base.parent.mkdir(parents=True)
            base.write_bytes(b"interpreter")
            python = root / "environment/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(base)
            self.assertEqual(stage_runtime.interpreter_sha256(python),
                             digest_file(python, follow=True))
            self.assertEqual(stage_runtime.interpreter_sha256(python), digest_file(base))


if __name__ == "__main__":
    unittest.main()
