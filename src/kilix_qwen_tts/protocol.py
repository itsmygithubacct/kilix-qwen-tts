"""Bounded candidate transport mechanics for the local Qwen speech provider.

The module intentionally stops short of freezing F104 P1.  It implements the
parts of the private transport that are invariant regardless of the selected
speech engine: strict JSON decoding, typed request envelopes, descriptor
counting, peer credential checks, and bounded ``SOCK_SEQPACKET`` I/O.
"""

from __future__ import annotations

import array
import fcntl
import hashlib
import json
import math
import os
import socket
import stat
import struct
import tempfile
from dataclasses import dataclass
from typing import Any


PROTOCOL_SCHEMA = "kilix.qwen-tts.provider/candidate-v1"
WIRE_OPERATIONS = ("hello", "submit", "models", "status", "cancel", "unload")
FORBIDDEN_FIELDS = frozenset(
    {"path", "url", "command", "shell", "executable", "environment", "module", "import"}
)

MAX_CONTROL_FRAME_BYTES = 65_536
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 2_048
MAX_KEY_UTF8_BYTES = 64
MAX_NUMBER_TOKEN_BYTES = 64
MAX_INTEGER_BITS = 213
MAX_ID_UTF8_BYTES = 64
MAX_LANGUAGE_UTF8_BYTES = 64
MAX_DESCRIPTORS = 1

_U32 = struct.Struct("!I")
_PEERCRED = struct.Struct("3i")


class ProtocolError(ValueError):
    """Stable candidate transport refusal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ProtocolError(code, message)


def _json_constant(_value: str) -> None:
    raise ProtocolError("INVALID_REQUEST", "non-standard JSON constants are forbidden")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "INVALID_REQUEST", "duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _bounded_int(token: str) -> int:
    _require(len(token) <= MAX_NUMBER_TOKEN_BYTES, "LIMIT_EXCEEDED",
             "integer token exceeds its byte bound")
    value = int(token)
    _require(value.bit_length() <= MAX_INTEGER_BITS, "LIMIT_EXCEEDED",
             "integer value exceeds its bit bound")
    return value


def _bounded_float(token: str) -> float:
    _require(len(token) <= MAX_NUMBER_TOKEN_BYTES, "LIMIT_EXCEEDED",
             "number token exceeds its byte bound")
    value = float(token)
    _require(math.isfinite(value), "INVALID_REQUEST", "non-finite numbers are forbidden")
    return value


def _raw_depth_guard(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            _require(depth <= MAX_JSON_DEPTH, "LIMIT_EXCEEDED",
                     "JSON nesting exceeds its depth bound")
        elif character in "]}":
            depth -= 1
            _require(depth >= 0, "INVALID_REQUEST", "JSON containers are unbalanced")


def _utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ProtocolError("INVALID_REQUEST", "strings must contain Unicode scalar text") from error


def _validate_structure(value: Any) -> None:
    nodes = 0
    scalar_bytes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES, "LIMIT_EXCEEDED",
                 "JSON node population exceeds its bound")
        _require(depth <= MAX_JSON_DEPTH, "LIMIT_EXCEEDED",
                 "JSON nesting exceeds its depth bound")
        if isinstance(current, dict):
            _require(nodes + len(stack) + len(current) <= MAX_JSON_NODES,
                     "LIMIT_EXCEEDED", "JSON node population exceeds its bound")
            for key, child in current.items():
                _require(type(key) is str, "INVALID_REQUEST", "object keys must be strings")
                size = _utf8_size(key)
                _require(size <= MAX_KEY_UTF8_BYTES, "LIMIT_EXCEEDED",
                         "object key exceeds its byte bound")
                scalar_bytes += size
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            _require(nodes + len(stack) + len(current) <= MAX_JSON_NODES,
                     "LIMIT_EXCEEDED", "JSON node population exceeds its bound")
            stack.extend((child, depth + 1) for child in current)
        elif type(current) is int:
            _require(current.bit_length() <= MAX_INTEGER_BITS, "LIMIT_EXCEEDED",
                     "integer value exceeds its bit bound")
            scalar_bytes += len(str(current))
        elif type(current) is float:
            _require(math.isfinite(current), "INVALID_REQUEST", "non-finite numbers are forbidden")
            scalar_bytes += len(repr(current))
        elif type(current) is str:
            scalar_bytes += _utf8_size(current)
        else:
            _require(current is None or type(current) is bool, "INVALID_REQUEST",
                     "control frame contains a non-JSON value")
            scalar_bytes += 4
        _require(scalar_bytes <= MAX_CONTROL_FRAME_BYTES, "LIMIT_EXCEEDED",
                 "control scalar population exceeds its byte bound")


def decode_payload(payload: bytes) -> dict[str, Any]:
    """Decode one raw JSON payload with stable bounded failures."""

    _require(type(payload) is bytes, "INVALID_REQUEST", "control payload must be bytes")
    _require(bool(payload), "INVALID_REQUEST", "control payload must not be empty")
    _require(len(payload) <= MAX_CONTROL_FRAME_BYTES, "LIMIT_EXCEEDED",
             "control payload exceeds its byte bound")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ProtocolError("INVALID_REQUEST", "control payload is not valid UTF-8") from error
    _raw_depth_guard(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_json_constant,
            parse_int=_bounded_int,
            parse_float=_bounded_float,
        )
    except ProtocolError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ProtocolError("INVALID_REQUEST", "control payload is not valid JSON") from error
    _require(type(value) is dict, "INVALID_REQUEST", "control payload root must be an object")
    _validate_structure(value)
    return value


def encode_payload(value: dict[str, Any]) -> bytes:
    """Encode a validated JSON object using deterministic bytes."""

    _require(type(value) is dict, "INVALID_REQUEST", "control payload root must be an object")
    _validate_structure(value)
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as error:
        raise ProtocolError("INVALID_REQUEST", "control payload is not strict JSON") from error
    _require(len(payload) <= MAX_CONTROL_FRAME_BYTES, "LIMIT_EXCEEDED",
             "control payload exceeds its byte bound")
    return payload


def _identity(value: Any, field: str) -> str:
    _require(type(value) is str and bool(value) and value.strip() == value,
             "INVALID_REQUEST", f"{field} must be non-empty trimmed text")
    _require(_utf8_size(value) <= MAX_ID_UTF8_BYTES, "LIMIT_EXCEEDED",
             f"{field} exceeds its byte bound")
    return value


def _walk_forbidden(value: Any) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                _require(key.casefold() not in FORBIDDEN_FIELDS, "FORBIDDEN_FIELD",
                         "request contains a caller-controlled execution or path field")
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    request_id: str
    operation: str
    job_id: str | None
    arguments: dict[str, Any]
    deadline_ms: int

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> "ProviderRequest":
        from .interface import CandidateError, load_json, SPEC_PATH, validate_request
        try:
            validate_request(load_json(SPEC_PATH), value)
        except CandidateError as error:
            raise ProtocolError(error.code, "invalid speech request") from error
        return cls(value["request_id"], value["op"], value.get("job_id"),
                   value["args"], value["deadline_ms"])


def frame_payload(value: dict[str, Any]) -> bytes:
    payload = encode_payload(value)
    return _U32.pack(len(payload)) + payload


def unframe_payload(packet: bytes) -> dict[str, Any]:
    _require(type(packet) is bytes and len(packet) >= _U32.size,
             "INVALID_REQUEST", "control packet is truncated")
    declared = _U32.unpack_from(packet)[0]
    payload = packet[_U32.size:]
    _require(declared == len(payload), "INVALID_REQUEST",
             "control packet length prefix is inconsistent")
    return decode_payload(payload)


def send_packet(channel: socket.socket, value: dict[str, Any], descriptor: int | None = None) -> None:
    """Send exactly one bounded packet and at most 1/1 descriptor."""

    packet = frame_payload(value)
    ancillary: list[tuple[int, int, bytes]] = []
    if descriptor is not None:
        _require(type(descriptor) is int and descriptor >= 0, "DESCRIPTOR_MISMATCH",
                 "descriptor must be a non-negative integer")
        descriptors = array.array("i", [descriptor])
        ancillary.append((socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes()))
    try:
        sent = channel.sendmsg([packet], ancillary)
    except OSError as error:
        raise ProtocolError("TRANSPORT_ERROR", "control packet send failed") from error
    _require(sent == len(packet), "TRANSPORT_ERROR", "control packet send was partial")


def receive_packet(channel: socket.socket) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Receive one packet, refusing truncation and excess descriptors."""

    try:
        packet, ancillary, flags, _address = channel.recvmsg(
            MAX_CONTROL_FRAME_BYTES + _U32.size,
            socket.CMSG_SPACE(MAX_DESCRIPTORS * array.array("i").itemsize),
            getattr(socket, "MSG_CMSG_CLOEXEC", 0),
        )
    except OSError as error:
        raise ProtocolError("TRANSPORT_ERROR", "control packet receive failed") from error
    descriptors: list[int] = []
    try:
        for level, kind, data in ancillary:
            _require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS,
                     "DESCRIPTOR_MISMATCH", "unexpected ancillary data")
            values = array.array("i")
            usable = len(data) - (len(data) % values.itemsize)
            values.frombytes(data[:usable])
            descriptors.extend(values)
        _require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), "LIMIT_EXCEEDED",
                 "control packet or descriptor population was truncated")
        _require(len(descriptors) <= MAX_DESCRIPTORS, "DESCRIPTOR_MISMATCH",
                 "descriptor population exceeds 1/1")
        return unframe_payload(packet), tuple(descriptors)
    except Exception:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def verify_request_descriptors(
    request: ProviderRequest,
    descriptors: tuple[int, ...],
) -> int | None:
    """Bind a typed request to its exact descriptor population and audio digest."""

    _require(isinstance(request, ProviderRequest), "INVALID_REQUEST",
             "request must be a typed ProviderRequest")
    _require(type(descriptors) is tuple
             and all(type(descriptor) is int and descriptor >= 0 for descriptor in descriptors),
             "DESCRIPTOR_MISMATCH", "descriptor population is invalid")
    if request.operation != "submit" or request.arguments["mode"] != "prompt_clone":
        _require(not descriptors, "DESCRIPTOR_MISMATCH",
                 "non-submit operations forbid descriptors")
        return None
    _require(len(descriptors) == 1, "DESCRIPTOR_MISMATCH",
             "submit requires exactly 1/1 audio descriptor")
    descriptor = descriptors[0]
    audio = request.arguments["prompt_audio"]
    snapshot = -1
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "DESCRIPTOR_MISMATCH",
                 "audio descriptor is not a regular file")
        _require(before.st_uid == os.geteuid(), "UNAUTHORIZED_PEER",
                 "audio descriptor has another owner")
        _require(before.st_size == audio["byte_length"], "DESCRIPTOR_MISMATCH",
                 "audio descriptor length disagrees with metadata")
        digest = hashlib.sha256()
        writer = tempfile.TemporaryFile(prefix="kilix-qwen-prompt-")
        offset = 0
        try:
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                _require(bool(chunk), "DESCRIPTOR_MISMATCH", "audio descriptor ended early")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(writer.fileno(), view)
                    _require(written > 0, "TRANSPORT_ERROR", "audio snapshot write stalled")
                    view = view[written:]
                offset += len(chunk)
            snapshot = os.open(
                f"/proc/self/fd/{writer.fileno()}",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            _require(
                fcntl.fcntl(snapshot, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
                "TRANSPORT_ERROR",
                "audio snapshot is not read-only",
            )
        finally:
            writer.close()
        after = os.fstat(descriptor)
        _require((after.st_dev, after.st_ino, after.st_size)
                 == (before.st_dev, before.st_ino, before.st_size),
                 "DESCRIPTOR_MISMATCH", "audio descriptor changed during verification")
        _require(digest.hexdigest() == audio["sha256"], "DESCRIPTOR_MISMATCH",
                 "audio descriptor digest disagrees with metadata")
    except ProtocolError:
        if snapshot >= 0:
            os.close(snapshot)
        raise
    except OSError as error:
        if snapshot >= 0:
            os.close(snapshot)
        raise ProtocolError("TRANSPORT_ERROR", "audio descriptor verification failed") from error
    return snapshot


def peer_effective_uid(channel: socket.socket) -> int:
    """Return the kernel-asserted peer UID on Linux."""

    _require(hasattr(socket, "SO_PEERCRED"), "PLATFORM_UNSUPPORTED",
             "SO_PEERCRED is unavailable on this platform")
    try:
        credentials = channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
        _pid, uid, _gid = _PEERCRED.unpack(credentials)
    except (OSError, struct.error) as error:
        raise ProtocolError("TRANSPORT_ERROR", "peer credential query failed") from error
    return uid


def require_same_uid_peer(channel: socket.socket, expected_uid: int | None = None) -> int:
    expected = os.geteuid() if expected_uid is None else expected_uid
    _require(type(expected) is int and expected >= 0, "UNAUTHORIZED_PEER",
             "expected peer UID is invalid")
    observed = peer_effective_uid(channel)
    _require(observed == expected, "UNAUTHORIZED_PEER", "peer UID differs from provider owner")
    return observed
