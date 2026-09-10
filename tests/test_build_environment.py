"""Build ownership and destination controls using explicit fake installers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


COMMON = '''import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(os.environ['BUILD_TEST_ROOT'])
def live(pid):
 try:
  raw=pathlib.Path(f'/proc/{pid}/stat').read_text()
  return raw[raw.rfind(')')+2:].split()[0]!='Z'
 except FileNotFoundError:return False
def phase(name):
 record=root/'pids.json'
 if os.environ['BUILD_TEST_PHASE']==name:
  child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True)
  record.write_text(json.dumps([os.getpid(),child.pid]))
  if os.environ['BUILD_TEST_ACTION']!='success':
   signal.signal(signal.SIGTERM,signal.SIG_IGN)
   time.sleep(60)
 elif record.exists():
  assert all(not live(pid) for pid in json.loads(record.read_text()))
'''
PYTHON = '#!/usr/bin/python3\n' + COMMON + '''phase('probe')
print(json.dumps({'torch':'2.6.0+cpu','torchaudio':'2.6.0+cpu'}))
'''
INSTALLER = '#!/usr/bin/python3\n' + COMMON + '''destination=pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT'])
(destination/'bin').mkdir(exist_ok=True)
python=destination/'bin/python'
python.write_text(PYTHON_SOURCE)
python.chmod(0o700)
(root/'invoked').touch()
phase('first' if '--only-group' in sys.argv else 'second')
'''


def live(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return raw[raw.rfind(')') + 2:].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class BuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='kq-build-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tool = Path(__file__).resolve().parents[1] / 'tools/build_environment.py'
        self.installer = self.root / 'fake-uv'
        self.installer.write_text(INSTALLER.replace('PYTHON_SOURCE', repr(PYTHON)))
        self.installer.chmod(0o700)

    def start(self, path, *, phase='first', action='success', timeout=5):
        environment = dict(os.environ, BUILD_TEST_ROOT=str(self.root),
                           BUILD_TEST_PHASE=phase, BUILD_TEST_ACTION=action)
        log = (self.root / 'builder.log').open('ab')
        self.addCleanup(log.close)
        process = subprocess.Popen([sys.executable, str(self.tool), '--uv', str(self.installer),
                                    '--destination', str(path), '--offline', '--timeout', str(timeout)],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=8)
        self.addCleanup(cleanup)
        return process

    def pids(self, process):
        path = self.root / 'pids.json'
        end = time.monotonic() + 5
        while not path.exists() and process.poll() is None and time.monotonic() < end:
            time.sleep(.01)
        self.assertTrue(path.exists(), (self.root / 'builder.log').read_text())
        return json.loads(path.read_text())

    def test_all_three_commands_reap_escaped_children_before_success_or_failure(self):
        unrelated = subprocess.Popen(['/usr/bin/python3', '-c', 'import time;time.sleep(60)'])
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()))
        for phase in ('first', 'second', 'probe'):
            for action in ('success', 'interrupt', 'timeout'):
                with self.subTest(phase=phase, action=action):
                    (self.root / 'pids.json').unlink(missing_ok=True)
                    destination = self.root / (phase + '-' + action)
                    process = self.start(destination, phase=phase, action=action,
                                         timeout=.5 if action == 'timeout' else 5)
                    pids = self.pids(process)
                    if action == 'interrupt':
                        process.send_signal(signal.SIGINT)
                    code = process.wait(timeout=8)
                    self.assertEqual(code == 0, action == 'success',
                                     (self.root / 'builder.log').read_text())
                    self.assertTrue(all(not live(pid) for pid in pids))
                    self.assertEqual(destination.exists(), action == 'success')
                    self.assertIsNone(unrelated.poll())

    def test_unsafe_ancestors_refuse_before_installer(self):
        for case in ('shared', 'symlink'):
            with self.subTest(case=case):
                (self.root / 'invoked').unlink(missing_ok=True)
                parent = self.root / case
                if case == 'shared':
                    parent.mkdir(mode=0o777)
                    parent.chmod(0o777)
                else:
                    parent.symlink_to(self.root, target_is_directory=True)
                process = self.start(parent / 'environment')
                self.assertNotEqual(process.wait(timeout=8), 0)
                self.assertFalse((self.root / 'invoked').exists())
                if case == 'shared':
                    parent.chmod(0o700)

    def test_replaced_ancestor_cancels_and_cleans_only_pinned_tree(self):
        parent = self.root / 'parent'
        parent.mkdir(mode=0o700)
        process = self.start(parent / 'environment', action='interrupt')
        pids = self.pids(process)
        moved = self.root / 'moved'
        parent.rename(moved)
        parent.mkdir(mode=0o700)
        substitute = parent / 'environment'
        substitute.mkdir(mode=0o700)
        (substitute / 'keep').write_text('preserve')
        self.assertNotEqual(process.wait(timeout=8), 0)
        self.assertTrue(all(not live(pid) for pid in pids))
        self.assertFalse((moved / 'environment').exists())
        self.assertEqual((substitute / 'keep').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
