"""Trusted namespace setup: unpack a sealed bundle, seal mounts, drop caps.

Only the system Python runs this stage. Model and dependency code is imported
only after the private runtime filesystem is read-only and capabilities are
empty. Input contains regular files only; no tar extraction or host paths.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
from pathlib import Path
import struct
import sys

MAX_FILES = 40_000
MAX_BYTES = 7 * 1024**3
MAGIC = b"KQRT\x01"
RECORD = struct.Struct("!HQB")


def exact(source, size):
    value = source.read(size)
    if len(value) != size:
        raise ValueError("truncated runtime bundle")
    return value


def unpack(descriptor):
    if fcntl.fcntl(descriptor, 1034) & 15 != 15:
        raise ValueError("unsealed runtime bundle")
    if os.fstat(descriptor).st_size > MAX_BYTES + MAX_FILES * (4096 + RECORD.size) + 32:
        raise ValueError("oversized runtime bundle")
    count = total = 0
    with os.fdopen(descriptor, "rb") as source:
        if exact(source, len(MAGIC)) != MAGIC:
            raise ValueError("invalid runtime bundle")
        while True:
            name_size, size, executable = RECORD.unpack(exact(source, RECORD.size))
            if name_size == 0:
                if size or executable or source.read(1):
                    raise ValueError("invalid runtime bundle end")
                break
            if name_size > 4096 or size > 4 * 1024**3 or executable not in (0, 1):
                raise ValueError("invalid runtime bundle record")
            name = exact(source, name_size).decode("utf-8")
            relative = Path(name)
            if (relative.is_absolute() or str(relative) != name or ".." in relative.parts
                    or "\0" in name or relative.parts[0] not in {"python", "runtime", "provider", "prompt.pcm"}):
                raise ValueError("unsafe runtime bundle path")
            count += 1
            total += size
            if count > MAX_FILES or total > MAX_BYTES:
                raise ValueError("runtime bundle exceeds its bound")
            destination = Path("/opt") / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o555 if executable else 0o444)
            with os.fdopen(fd, "wb") as output:
                while size:
                    data = exact(source, min(size, 1024 * 1024))
                    output.write(data)
                    size -= len(data)


def prepare_mount():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    # bwrap's setup mount namespace belongs to its earlier user namespace.
    # Own a new mount namespace and a new tmpfs with no writable outer alias.
    if libc.unshare(0x20000) != 0:  # CLONE_NEWNS
        raise OSError(ctypes.get_errno(), "cannot own runtime mount namespace")
    if libc.mount(b"tmpfs", b"/opt", b"tmpfs", 6, f"size={MAX_BYTES + 64 * 1024**2},mode=0700".encode()) != 0:
        raise OSError(ctypes.get_errno(), "cannot allocate private runtime filesystem")
    return libc


def seal_mount_and_drop_capabilities(libc):
    # MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV | MS_RELATIME.
    if libc.mount(None, b"/opt", None, 4096 | 32 | 1 | 2 | 4 | (1 << 21), None) != 0:
        raise OSError(ctypes.get_errno(), "cannot seal runtime filesystem")
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "cannot prevent privilege acquisition")
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    if libc.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref((Data * 2)())) != 0:
        raise OSError(ctypes.get_errno(), "cannot drop bootstrap capabilities")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if any(int(status[key].strip(), 16) for key in ("CapInh", "CapPrm", "CapEff", "CapAmb")):
        raise ValueError("bootstrap retained capabilities")


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) < 3:
        return 125
    libc = prepare_mount()
    unpack(int(sys.argv[1]))
    seal_mount_and_drop_capabilities(libc)
    executable = "/opt/python/bin/python3.12"
    os.execv(executable, [executable, "-I", "-B", "/opt/provider/kilix_qwen_tts/supervisor.py"])
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
