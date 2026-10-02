#!/usr/bin/env python3
"""Own one local environment probe and reap all of its descendants."""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_environment import _reap_owned

stopping = False


def stop(_signal, _frame):
    global stopping
    stopping = True


def main():
    parent = os.getppid()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0
            or libc.prctl(38, 1, 0, 0, 0) != 0
            or os.getppid() != parent or parent == 1 or len(sys.argv) < 2):
        return 125
    try:
        process = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True)
        while process.poll() is None and not stopping:
            time.sleep(.01)
        return 130 if stopping else process.returncode
    finally:
        _reap_owned()


if __name__ == '__main__':
    raise SystemExit(main())
