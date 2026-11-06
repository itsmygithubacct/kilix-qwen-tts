"""Classify a speech worker stop that the worker itself recorded.

A generic nonzero exit stays unclassified. The host then keeps ENGINE_FAILED.
"""
from __future__ import annotations

import errno
import json
import os
import stat

RECORD = "failure.json"
CLASSIFIED_EXIT = 3
DISK_EXIT = 4
MAX_BYTES = 64
KINDS = {
    "oom": ("DEVICE_UNAVAILABLE", "speech worker ran out of memory"),
    "device_loss": ("DEVICE_UNAVAILABLE", "speech device is unavailable"),
    "corrupt": ("INVALID_REQUEST", "speech worker refused corrupt input"),
    "disk": ("INSUFFICIENT_DISK", "speech worker ran out of disk"),
}
CRASH = ("WORKER_CRASH", "speech worker crashed")
# SIGINT (2, status 130) is the supervisor's own stop status, so it is not a crash.
# SIGTERM is a crash only when the host has not already decided to cancel.
_CRASH_SIGNALS = {4, 6, 7, 8, 9, 11, 15, 31}


class CorruptInput(Exception):
    """The worker refused bytes that are not a usable input."""


def _disk_or_raise(kind, error):
    if kind == "disk" and getattr(error, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
        raise SystemExit(DISK_EXIT) from error
    raise error


def report(workspace, kind):
    """Record one failure kind and stop. A full disk cannot record itself."""
    if kind not in KINDS or type(workspace) is not str or not workspace or "\0" in workspace:
        raise ValueError("invalid failure report")
    try:
        descriptor = os.open(
            os.path.join(workspace, RECORD),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as error:
        _disk_or_raise(kind, error)
    try:
        payload = json.dumps({"kind": kind}, separators=(",", ":")).encode("ascii")
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except OSError as error:
                _disk_or_raise(kind, error)
            if written <= 0:
                _disk_or_raise(kind, OSError(errno.ENOSPC, "short failure record"))
            offset += written
    finally:
        os.close(descriptor)
    raise SystemExit(CLASSIFIED_EXIT)


def _record(workspace):
    if type(workspace) is not str or not workspace or "\0" in workspace:
        return None
    try:
        descriptor = os.open(
            os.path.join(workspace, RECORD),
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or not 1 <= info.st_size <= MAX_BYTES):
            return None
        payload = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if not 1 <= len(payload) <= MAX_BYTES:
        return None
    try:
        value = json.loads(payload.decode())
    except (UnicodeDecodeError, ValueError):
        return None
    if type(value) is not dict or set(value) != {"kind"} or value["kind"] not in KINDS:
        return None
    return KINDS[value["kind"]]


def classify(returncode, workspace):
    """Return a protocol code and message, or None when the exit stays generic."""
    if type(returncode) is not int:
        return None
    if returncode < 0:
        return CRASH
    if returncode > 128 and returncode - 128 in _CRASH_SIGNALS:
        return CRASH
    if returncode == DISK_EXIT:
        return KINDS["disk"]
    if returncode != CLASSIFIED_EXIT:
        return None
    return _record(workspace)


def guard(workspace, function, classify=None):
    """Turn a worker exception into one recorded stop. Other exceptions propagate."""
    try:
        return function()
    except SystemExit:
        raise
    except MemoryError:
        if type(workspace) is str:
            report(workspace, "oom")
        raise
    except OSError as error:
        if getattr(error, "errno", None) in (errno.ENOSPC, errno.EDQUOT) and type(workspace) is str:
            report(workspace, "disk")
        raise
    except CorruptInput:
        if type(workspace) is str:
            report(workspace, "corrupt")
        raise
    except Exception as error:
        kind = classify(error) if classify is not None else None
        if kind in KINDS and type(workspace) is str:
            report(workspace, kind)
        raise
