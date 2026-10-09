"""Bounded, process-scoped, memory-only speaker embeddings; never consent."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time

from .protocol import ProtocolError

MAGIC = b'KQPE\x01\0\0\0'
DIMENSIONS = 1024
ENCODING = struct.Struct('<1024f')
PAYLOAD_BYTES = len(MAGIC) + ENCODING.size
MAX_ENTRIES = 8
IDLE_SECONDS = 300
# These are execution identities, not the float32 CPU storage format. The
# pinned engine extracts speaker embeddings on model.device / model.dtype.
PRODUCERS = {'cpu-float32': ('cpu', 'float32'),
             'cuda0-bfloat16': ('cuda:0', 'bfloat16')}
PRODUCER_EXECUTION = 'qwen3-tts.create_voice_clone_prompt/x-vector-only/v1'


def producer_for_device(device):
    if type(device) is not str or device not in {'cpu', 'cuda'}:
        raise ValueError('unsupported embedding device')
    return 'cpu-float32' if device == 'cpu' else 'cuda0-bfloat16'


def producers_for_profile(profile):
    producer_for_device(profile)  # Refuse unknown profiles.
    return ('cpu-float32',) if profile == 'cpu' else tuple(PRODUCERS)


def validate_inputs(embeddings, profile):
    allowed = producers_for_profile(profile)
    if (type(embeddings) is not dict or len(embeddings) > len(allowed)
            or any(type(key) is not str or key not in allowed for key in embeddings)):
        raise ProtocolError('INVALID_REQUEST', 'invalid prompt cache inputs')
    for payload in embeddings.values():
        validate_embedding(payload)
    return embeddings


def validate_embedding(payload):
    if (type(payload) is not bytes or len(payload) != PAYLOAD_BYTES or not payload.startswith(MAGIC)
            or any(not math.isfinite(v) or abs(v) > 10_000 for v in ENCODING.unpack(payload[len(MAGIC):]))):
        raise ProtocolError('MALFORMED_WORKER_RESULT', 'invalid speaker embedding')
    return payload


def read_embedding(path, *, readonly_mount=False):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & (0o222 if readonly_mount else 0o077) or info.st_size != PAYLOAD_BYTES):
            raise ProtocolError('MALFORMED_WORKER_RESULT', 'unsafe speaker embedding')
        return validate_embedding(os.pread(fd, PAYLOAD_BYTES + 1, 0))
    finally:
        os.close(fd)


def peer_scope(channel):
    """A departed/unobservable peer gets no cache, with no PID-only fallback."""
    try:
        pid, uid, gid = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if pid <= 0 or uid != os.geteuid():
            return None
        with (Path('/proc') / str(pid) / 'stat').open('rb') as source:
            data = source.read(4097)
        if len(data) > 4096 or not data.startswith(str(pid).encode() + b' ('):
            return None
        fields = data.rsplit(b')', 1)[1].split()
        start = fields[19]
        if fields[0] in {b'Z', b'X'} or not start.isdigit() or len(start) > 32:
            return None
        return pid, uid, gid, start.decode('ascii')
    except (OSError, IndexError, ValueError):
        return None


def prompt_key(scope, manifest, arguments, producer):
    if type(producer) is not str or producer not in PRODUCERS:
        raise ValueError('unsupported embedding producer')
    if scope is None or arguments['mode'] != 'prompt_clone':
        return None
    consent = arguments['consent']
    # Every request must already have freshly validated consent and actual PCM.
    # A changed attestation timestamp does not change its permitted use or the
    # numerical speaker embedding. It is still bound into that job's result.
    device, compute_dtype = PRODUCERS[producer]
    identity = {'schema': 'kilix.qwen-tts.prompt-cache/v2', 'peer': scope,
                'runtime': manifest, 'audio': arguments['prompt_audio'],
                'producer': {'execution': PRODUCER_EXECUTION, 'device': device, 'compute_dtype': compute_dtype},
                'consent_scope': {k: consent[k] for k in ('schema', 'source_sha256', 'allowed_use', 'purpose')}}
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(b'KILIX_QWEN_PROMPT_CACHE_V2\0' + payload).digest()


class PromptCache:
    """At most eight 4,104-byte entries, discarded after five idle minutes."""
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self.hits = self.misses = 0

    def _expire(self, now):
        for key, (touched, _payload) in tuple(self._entries.items()):
            if now - touched >= IDLE_SECONDS:
                del self._entries[key]

    def expire(self):
        with self._lock:
            self._expire(self._clock())

    def get(self, key):
        if key is None:
            return None
        with self._lock:
            now = self._clock()
            self._expire(now)
            found = self._entries.pop(key, None)
            if found is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries[key] = (now, found[1])
            return found[1]

    def put(self, key, payload):
        payload = validate_embedding(payload)
        if key is None:
            return
        if type(key) is not bytes or len(key) != 32:
            raise ValueError('invalid prompt cache key')
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._entries.pop(key, None)
            self._entries[key] = (now, payload)
            while len(self._entries) > MAX_ENTRIES:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()
