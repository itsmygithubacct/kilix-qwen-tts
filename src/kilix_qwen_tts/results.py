"""Strict binding of a completed synthesis result to the requested job."""
from __future__ import annotations

import hashlib
import json
import struct

from .protocol import ProtocolError


def validate_result(result: dict, arguments: dict, payload: bytes) -> None:
    from .runtime import ENGINE_COMMIT, MODEL_CANDIDATES
    required = {'engine_id', 'engine_revision', 'model_id', 'model_revision', 'duration_ms', 'seed', 'audio'}
    if arguments.get('mode') == 'prompt_clone':
        required.add('conditioning')
    if type(result) is not dict or set(result) != required:
        raise ProtocolError('INVALID_RESPONSE', 'invalid result field population')
    model_id = result['model_id']
    candidate = MODEL_CANDIDATES.get(model_id) if type(model_id) is str else None
    if (candidate is None or result['model_revision'] != candidate[0] or arguments['mode'] != candidate[1]
            or arguments['output'] != {'sample_format': 's16le', 'sample_rate_hz': 24000, 'channels': 1}
            or arguments['model_id'] not in {'auto', model_id}
            or result['engine_id'] != 'qwen3-tts' or result['engine_revision'] != ENGINE_COMMIT
            or type(result['seed']) is not int or result['seed'] != arguments['seed']
            or type(result['duration_ms']) is not int
            or not 0 < result['duration_ms'] <= arguments['max_duration_ms']
            or type(result['audio']) is not dict or set(result['audio']) != {'byte_length', 'sha256'}
            or type(result['audio']['byte_length']) is not int
            or result['audio']['byte_length'] != len(payload)
            or result['audio']['sha256'] != hashlib.sha256(payload).hexdigest()):
        raise ProtocolError('INVALID_RESPONSE', 'result identity does not match the job')
    if arguments['mode'] == 'prompt_clone':
        consent = json.dumps(arguments['consent'], separators=(',', ':'), sort_keys=True).encode()
        expected = {'prompt_sha256': arguments['prompt_audio']['sha256'],
                    'consent_sha256': hashlib.sha256(consent).hexdigest()}
        if result['conditioning'] != expected:
            raise ProtocolError('INVALID_RESPONSE', 'unbound prompt or consent result')
    validate_wave(payload, result['duration_ms'])


def validate_wave(payload: bytes, duration_ms: int) -> None:
    if type(duration_ms) is not int or not 0 < duration_ms <= 900_000:
        raise ProtocolError('INVALID_RESPONSE', 'invalid canonical audio duration')
    size = duration_ms * 24 * 2
    # The provider emits one canonical PCM16 header. Comparing every field
    # also binds byte rate, block alignment, fmt size and the data chunk;
    # Python's general WAV reader deliberately ignores some of those fields.
    header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', size + 36, b'WAVE',
                         b'fmt ', 16, 1, 1, 24000, 48000, 2, 16, b'data', size)
    if len(payload) != size + 44 or payload[:44] != header:
        raise ProtocolError('INVALID_RESPONSE', 'invalid canonical WAV result')
