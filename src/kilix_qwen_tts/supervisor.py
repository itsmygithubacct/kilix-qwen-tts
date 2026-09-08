"""Dedicated Linux descendant owner; never reaps the provider's other children."""
from __future__ import annotations

import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

stopping = False
CLEANUP_MARKER = b"KILIX_REAPED_V1"


def stop(_signal, _frame):
    global stopping
    stopping = True


def reap_descendants():
    """A dedicated subreaper owns adoptees even after setsid/setpgid."""
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        # Only this supervisor's direct children can appear here. Killing
        # parents reparents any still-running descendants to this subreaper.
        children = [int(value) for value in children_file.read_text().split()]
        for child in children:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if reaped == 0:
                break
        time.sleep(0.005)


def main():
    completion = None
    if len(sys.argv) >= 3 and sys.argv[-2] == "--completion-fd":
        completion_fd = int(sys.argv[-1])
        if completion_fd < 3:
            return 125
        completion = socket.socket(fileno=completion_fd)
        if completion.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
            return 125
        os.set_inheritable(completion_fd, False)
        del sys.argv[-2:]
    guard_fd = None
    if len(sys.argv) >= 3 and sys.argv[-2] == "--guard-fd":
        guard_fd = int(sys.argv[-1])
        if guard_fd < 3 or (completion is not None and guard_fd == completion.fileno()):
            return 125
        os.fstat(guard_fd)
        os.set_inheritable(guard_fd, False)
        del sys.argv[-2:]
    parent = os.getppid()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER, PR_SET_PDEATHSIG, PR_SET_NO_NEW_PRIVS.
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0
            or libc.prctl(38, 1, 0, 0, 0) != 0):
        return 125
    if os.getppid() != parent or parent == 1:
        return 125
    payload = sys.stdin.buffer.read(65_537)
    if len(payload) > 65_536:
        return 125
    json.loads(payload)
    if len(sys.argv) == 3 and sys.argv[1] == "--launch-fd":
        descriptor = int(sys.argv[2])
        if descriptor < 3 or fcntl.fcntl(descriptor, 1034) & 15 != 15:
            return 125
        with os.fdopen(descriptor, "rb") as source:
            configuration = json.loads(source.read(65_537))
        command = configuration["command"]
        descriptors = configuration["descriptors"]
        if (type(command) is not list or not 1 <= len(command) <= 256
                or any(type(value) is not str for value in command)
                or command[0] != "/usr/bin/bwrap" or type(descriptors) is not list
                or len(descriptors) > 8 or any(type(fd) is not int or fd < 3 for fd in descriptors)):
            return 125
    elif len(sys.argv) == 1:
        descriptors = []
        command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("worker.py"))]
    else:
        return 125
    if guard_fd is not None:
        descriptors.append(guard_fd)
        if command[0] == "/usr/bin/bwrap":
            # Bubblewrap retains this same open description until its sandbox
            # exits. The trusted supervisor also keeps its own inherited copy.
            command = [command[0], "--sync-fd", str(guard_fd), *command[1:]]
    worker = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=sys.stdout.buffer, stderr=sys.stderr,
        pass_fds=tuple(descriptors), start_new_session=True,
    )
    try:
        for descriptor in descriptors:
            if descriptor != guard_fd:
                os.close(descriptor)
        assert worker.stdin is not None
        worker.stdin.write(payload)
        worker.stdin.close()
        while worker.poll() is None and not stopping:
            time.sleep(0.01)
        return 130 if stopping else worker.returncode
    finally:
        reap_descendants()
        if completion is not None:
            # Only this trusted supervisor emits the kernel-credentialed
            # cleanup message; the engine never inherits this descriptor.
            try:
                completion.send(CLEANUP_MARKER)
            except OSError:
                pass
            finally:
                completion.close()


if __name__ == "__main__":
    raise SystemExit(main())
