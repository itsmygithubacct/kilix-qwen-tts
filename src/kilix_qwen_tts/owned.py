"""Private owned-cleanup evidence and optional shared execution coordination."""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import uuid

from .protocol import ProtocolError

CLEANUP_MARKER = b"KILIX_REAPED_V1"
_CREDENTIALS = struct.Struct("iII")


class ExecutionPolicy:
    """Coordinate a selected device; this grants no hardware/profile admission."""
    def __init__(self, device: str, *, namespace: str | None = None):
        try:
            from voicelib import device_leases
        except ImportError as error:
            raise ProtocolError("PROVIDER_UNAVAILABLE", "shared execution lease API is unavailable") from error
        self.module = device_leases
        self.device = device
        self.namespace = namespace

    def convert(self, error):
        code = {"invalid-request": "INVALID_REQUEST", "queue-full": "BUSY",
                "cancelled": "CANCELED", "deadline": "DEADLINE_EXCEEDED",
                "unavailable": "PROVIDER_UNAVAILABLE", "lost-lease": "SUPERVISOR_FAILED"}.get(
                    error.code, "PROVIDER_UNAVAILABLE")
        return ProtocolError(code, "shared execution lease refused")

    def acquire(self, *, job_id, workload, deadline, cancelled, disconnected, progress=None):
        try:
            return self.module.acquire(job_id=job_id, workload=workload, device=self.device,
                deadline=deadline, cancelled=cancelled, disconnected=disconnected,
                progress=progress, namespace=self.namespace)
        except self.module.LeaseError as error:
            raise self.convert(error) from error


class OwnedExecution:
    """A supervisor must prove teardown before the caller acknowledges a lease."""
    def __init__(self, policy, *, job_id, workload, deadline, cancelled, disconnected, progress=None):
        self.policy = policy
        self.arguments = dict(job_id=job_id or uuid.uuid4().hex, workload=workload,
                              deadline=deadline, cancelled=cancelled, disconnected=disconnected, progress=progress)
        self.lease = None
        self.parent = self.child = None
        self.cleanup_complete = True  # No owned process has started.

    def __enter__(self):
        try:
            if self.policy is not None:
                self.lease = self.policy.acquire(**self.arguments)
            self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self.parent.setblocking(False)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        if self.lease is not None:
            try:
                self.lease.check()
            except self.policy.module.LeaseError as error:
                raise self.policy.convert(error) from error

    def spawn(self, command, **kwargs):
        descriptors = list(kwargs.get("pass_fds", ()))
        descriptors.append(self.child.fileno())
        if self.lease is not None:
            descriptors.append(self.lease.guard_fd)
        kwargs["pass_fds"] = tuple(descriptors)
        if self.lease is not None:
            command = [*command, "--guard-fd", str(self.lease.guard_fd)]
        command = [*command, "--completion-fd", str(self.child.fileno())]
        process = subprocess.Popen(command, **kwargs)
        self.cleanup_complete = False
        self.child.close()
        self.child = None
        return process

    def finish(self, process, stop_process):
        try:
            stop_process(process)
        finally:
            try:
                payload, ancillary, flags, _address = self.parent.recvmsg(
                    len(CLEANUP_MARKER) + 1, socket.CMSG_SPACE(_CREDENTIALS.size))
                credentials = [(level, kind, data) for level, kind, data in ancillary
                               if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
                valid = (payload == CLEANUP_MARKER and not flags
                         and len(ancillary) == len(credentials) == 1
                         and len(credentials[0][2]) == _CREDENTIALS.size
                         and _CREDENTIALS.unpack(credentials[0][2]) == (process.pid, os.geteuid(), os.getegid()))
                if valid:
                    trailing, extra_control, extra_flags, _address = self.parent.recvmsg(
                        1, socket.CMSG_SPACE(_CREDENTIALS.size), socket.MSG_DONTWAIT)
                    # An empty seqpacket record still has sender credentials;
                    # only an empty payload/control/flags tuple proves EOF.
                    if trailing or extra_control or extra_flags:
                        valid = False
            except (OSError, ValueError):
                valid = False
            self.cleanup_complete = valid
            if not valid:
                raise ProtocolError("SUPERVISOR_FAILED", "owned cleanup was not acknowledged")

    def __exit__(self, *_exc):
        if self.parent is not None:
            self.parent.close()
            self.parent = None
        if self.child is not None:
            self.child.close()
            self.child = None
        if self.lease is not None:
            try:
                self.lease.release(cleanup_complete=self.cleanup_complete)
            except self.policy.module.LeaseError as error:
                raise self.policy.convert(error) from error
            finally:
                self.lease = None



def from_options(options):
    """Explicit selection refuses if the shared implementation is unavailable."""
    device = getattr(options, "lease_device", None)
    namespace = getattr(options, "lease_namespace", None)
    if device is None:
        if namespace is not None:
            raise ProtocolError("INVALID_REQUEST", "lease namespace requires a selected device")
        return None
    return ExecutionPolicy(device, namespace=namespace)
