"""Bounded append-only PCM records between an owned worker and its provider."""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import struct

from .protocol import ProtocolError, decode_payload, encode_payload

MAGIC = b"KILIX_PCM_V1\x00"
HEADER = struct.Struct("!cI")
MAX_PCM_BYTES = 96_000
MAX_RECORDS = 65_536
MAX_RESULT_BYTES = 65_536


def requested(value):
    extensions = value.get("extensions", {})
    if type(extensions) is not dict:
        raise ProtocolError("INVALID_REQUEST", "invalid PCM extension")
    if "x_pcm_stream_v1" not in extensions:
        return False
    selected = extensions["x_pcm_stream_v1"]
    if value.get("op") != "submit" or type(selected) is not bool:
        raise ProtocolError("INVALID_REQUEST", "invalid PCM extension")
    return selected


def chunk_metadata(sequence, frame_offset, pcm):
    return {"sequence": sequence, "frame_offset": frame_offset, "frame_count": len(pcm) // 2,
            "sample_format": "s16le", "sample_rate_hz": 24_000, "channels": 1,
            "audio": {"fd": 0, "byte_length": len(pcm), "sha256": hashlib.sha256(pcm).hexdigest()}}


class ClientStream:
    def __init__(self, maximum_duration_ms, callback=None):
        self.maximum_bytes = maximum_duration_ms * 48
        self.callback = callback
        self.records = []
        self.pcm_bytes = 0

    @staticmethod
    def refuse():
        raise ProtocolError("INVALID_RESPONSE", "invalid provider PCM stream")

    def append(self, metadata, descriptors):
        if (type(metadata) is not dict or set(metadata) != {
                "sequence", "frame_offset", "frame_count", "sample_format", "sample_rate_hz", "channels", "audio"}
                or len(descriptors) != 1 or len(self.records) >= MAX_RECORDS):
            self.refuse()
        for name in ("sequence", "frame_offset", "frame_count", "sample_rate_hz", "channels"):
            if type(metadata[name]) is not int:
                self.refuse()
        if (metadata["sequence"] != len(self.records) or metadata["frame_offset"] != self.pcm_bytes // 2
                or not 0 < metadata["frame_count"] <= 48_000 or metadata["frame_count"] % 24
                or metadata["sample_format"] != "s16le" or metadata["sample_rate_hz"] != 24_000
                or metadata["channels"] != 1):
            self.refuse()
        audio = metadata["audio"]
        if (type(audio) is not dict or set(audio) != {"fd", "byte_length", "sha256"}
                or type(audio["fd"]) is not int or audio["fd"] != 0
                or type(audio["byte_length"]) is not int
                or audio["byte_length"] != metadata["frame_count"] * 2
                or not 0 < audio["byte_length"] <= MAX_PCM_BYTES
                or self.pcm_bytes + audio["byte_length"] > self.maximum_bytes
                or type(audio["sha256"]) is not str or len(audio["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in audio["sha256"])):
            self.refuse()
        fd = descriptors[0]
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_size != audio["byte_length"]
                or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
            self.refuse()
        payload = os.pread(fd, audio["byte_length"] + 1, 0)
        if len(payload) != audio["byte_length"] or hashlib.sha256(payload).hexdigest() != audio["sha256"]:
            self.refuse()
        self.records.append(payload)
        self.pcm_bytes += len(payload)
        if self.callback is not None:
            self.callback(metadata["sequence"], metadata["frame_offset"], payload)

    def require_audio(self, canonical_wave):
        if not self.records or canonical_wave[44:] != b"".join(self.records):
            self.refuse()


def write_bytes(destination, payload):
    view = memoryview(payload)
    while view:
        count = destination.write(view)
        if type(count) is not int or not 0 < count <= len(view):
            raise ValueError("stream write made no progress")
        view = view[count:]
    destination.flush()


class WorkerStream:
    def __init__(self, destination):
        self.destination = destination
        self.records = 0
        self.ended = False
        write_bytes(destination, MAGIC)

    def pcm(self, payload):
        if (self.ended or type(payload) is not bytes
                or not 0 < len(payload) <= MAX_PCM_BYTES or len(payload) % 48
                or self.records >= MAX_RECORDS):
            raise ValueError("invalid PCM stream record")
        write_bytes(self.destination, HEADER.pack(b"P", len(payload)) + payload)
        self.records += 1

    def finish(self, result):
        if self.ended or not self.records:
            raise ValueError("invalid PCM stream completion")
        payload = encode_payload(result)
        write_bytes(self.destination, HEADER.pack(b"R", len(payload)) + payload)
        self.ended = True


class WorkerStreamReader:
    """Read complete records without waiting on a partial write or seeking it."""
    def __init__(self, descriptor, maximum_duration_ms, callback, check):
        if type(maximum_duration_ms) is not int or not 0 < maximum_duration_ms <= 900_000:
            raise ValueError("invalid stream duration")
        self.descriptor = descriptor
        self.maximum_bytes = maximum_duration_ms * 48
        self.callback, self.check = callback, check
        self.offset = 0
        self.observed_size = 0
        self.records = []
        self.pcm_bytes = 0
        self.result = None

    @staticmethod
    def refuse():
        raise ProtocolError("MALFORMED_WORKER_RESULT", "invalid worker PCM stream")

    def drain(self, *, final=False):
        self.check()
        size = os.fstat(self.descriptor).st_size
        if (size < self.observed_size
                or size > len(MAGIC) + self.maximum_bytes + HEADER.size * (MAX_RECORDS + 1) + MAX_RESULT_BYTES):
            self.refuse()
        self.observed_size = size
        if not self.offset:
            prefix = os.pread(self.descriptor, min(size, len(MAGIC)), 0)
            if prefix != MAGIC[:len(prefix)]:
                self.refuse()
            if len(prefix) < len(MAGIC):
                if final:
                    self.refuse()
                return
            self.offset = len(MAGIC)
        while self.offset < size:
            self.check()
            if self.result is not None:
                self.refuse()
            header = os.pread(self.descriptor, HEADER.size, self.offset)
            if len(header) != HEADER.size:
                break
            kind, length = HEADER.unpack(header)
            if kind == b"P":
                if (not 0 < length <= MAX_PCM_BYTES or length % 48
                        or len(self.records) >= MAX_RECORDS
                        or self.pcm_bytes + length > self.maximum_bytes):
                    self.refuse()
            elif kind == b"R":
                if not 0 < length <= MAX_RESULT_BYTES or not self.records:
                    self.refuse()
            else:
                self.refuse()
            if self.offset + HEADER.size + length > size:
                break
            payload = os.pread(self.descriptor, length, self.offset + HEADER.size)
            if len(payload) != length:
                self.refuse()
            self.offset += HEADER.size + length
            if kind == b"P":
                sequence = len(self.records)
                frame_offset = self.pcm_bytes // 2
                self.records.append(payload)
                self.pcm_bytes += length
                self.callback(sequence, frame_offset, payload)
            else:
                try:
                    self.result = decode_payload(payload)
                except ProtocolError:
                    self.refuse()
        if final and (self.offset != size or self.result is None):
            self.refuse()

    def require_audio(self, canonical_wave):
        if canonical_wave[44:] != b"".join(self.records):
            self.refuse()


def wave_from_pcm(pcm):
    if type(pcm) is not bytes or not pcm or len(pcm) % 48:
        raise ValueError("invalid canonical PCM")
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", len(pcm) + 36, b"WAVE",
                       b"fmt ", 16, 1, 1, 24_000, 48_000, 2, 16, b"data", len(pcm)) + pcm
