"""A stalled worker input pipe must not prevent authenticated owned cleanup."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from kilix_qwen_tts.owned import OwnedExecution
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import stop_process
import kilix_qwen_tts.supervisor as supervisor_module

# Limit only the fixture worker's real kernel pipe so saturation is repeatable
# on hosts with different default pipe capacities. No production I/O is mocked.
BOOTSTRAP = '''
import runpy, subprocess, sys
original = subprocess.Popen
def small_pipe(*args, **kwargs):
    kwargs['pipesize'] = 4096
    return original(*args, **kwargs)
subprocess.Popen = small_pipe
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
WORKER = '''
import os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(__file__).parent
mode = (root / 'mode').read_text()
if mode == 'read':
    data = sys.stdin.buffer.read()
    (root / 'received').write_bytes(data)
    raise SystemExit(0)
if mode == 'exit':
    raise SystemExit(9)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
(root / 'ready').touch()
time.sleep(60)
'''


def children():
    return {int(pid) for pid in Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()}


def reap_fixture(exclude):
    end = time.monotonic() + 3
    while children() - exclude:
        if time.monotonic() >= end:
            raise AssertionError('owned probe children survived cleanup')
        for pid in children() - exclude:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, os.WNOHANG)
            except (ProcessLookupError, ChildProcessError):
                pass
        time.sleep(.005)


def probe(mode):
    # If the baseline kills its supervisor without reaping descendants, this
    # dedicated observer adopts and cleans only its own fixture descendants.
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    result = {}
    before_fds = len(os.listdir('/proc/self/fd'))
    try:
        with tempfile.TemporaryDirectory(prefix='qwen-input-') as folder:
            root = Path(folder)
            script = root / 'supervisor.py'
            script.write_bytes(Path(supervisor_module.__file__).read_bytes())
            (root / 'worker.py').write_text(WORKER)
            (root / 'mode').write_text(mode)
            payload = json.dumps({'text': 'x' * 32000}).encode()
            with OwnedExecution(None, job_id='input-probe', workload='tts-utterance',
                                deadline=time.monotonic()+10, cancelled=lambda: False,
                                disconnected=lambda: False) as owner:
                process = owner.spawn([sys.executable, '-I', '-B', '-c', BOOTSTRAP, str(script)],
                                      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, start_new_session=True)
                try:
                    process.stdin.write(payload)
                    process.stdin.close()
                    end = time.monotonic() + 3
                    while not (root / ('ready' if mode == 'stall' else 'received')).exists():
                        if mode == 'exit' and process.poll() is not None:
                            break
                        assert time.monotonic() < end, 'worker did not reach fixture state'
                        time.sleep(.005)
                    if mode == 'stall':
                        # Allow the supervisor to enter its saturated write.
                        time.sleep(.1)
                    else:
                        process.wait(timeout=3)
                    started = time.monotonic()
                    try:
                        owner.finish(process, stop_process)
                    except ProtocolError as error:
                        result['error'] = error.code
                    result.update(cleanup_complete=owner.cleanup_complete,
                                  cleanup_s=time.monotonic()-started,
                                  worker_exit=process.returncode,
                                  survivors=len(children()-{unrelated.pid}),
                                  unrelated_preserved=unrelated.poll() is None)
                    if mode == 'read':
                        result['exact_input'] = (root / 'received').read_bytes() == payload
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                    process.stdin.close()
                    reap_fixture({unrelated.pid})
        result['fd_delta'] = len(os.listdir('/proc/self/fd')) - before_fds
    finally:
        unrelated.kill()
        unrelated.wait(timeout=3)
        reap_fixture(set())
    result['children_after_probe'] = sorted(children())
    return result


class SupervisorInputTests(unittest.TestCase):
    def run_probe(self, mode):
        run = subprocess.run([sys.executable, __file__, '--probe', mode],
                             text=True, capture_output=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result['children_after_probe'], [])
        self.assertTrue(result['unrelated_preserved'])
        self.assertEqual(result['fd_delta'], 0)
        return result

    def test_cancel_during_saturated_worker_input_proves_cleanup(self):
        result = self.run_probe('stall')
        self.assertTrue(result['cleanup_complete'], result)
        self.assertLess(result['cleanup_s'], 2, result)
        self.assertEqual(result['survivors'], 0, result)
        self.assertEqual(result['worker_exit'], 130, result)

    def test_large_input_is_delivered_exactly(self):
        result = self.run_probe('read')
        self.assertTrue(result['exact_input'], result)
        self.assertTrue(result['cleanup_complete'], result)
        self.assertEqual(result['worker_exit'], 0, result)

    def test_closed_worker_input_still_proves_cleanup(self):
        result = self.run_probe('exit')
        self.assertTrue(result['cleanup_complete'], result)
        self.assertNotEqual(result['worker_exit'], 0, result)
        self.assertEqual(result['survivors'], 0, result)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--probe':
        print(json.dumps(probe(sys.argv[2]), sort_keys=True))
    else:
        unittest.main()
