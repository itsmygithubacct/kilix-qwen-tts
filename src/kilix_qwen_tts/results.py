"""Strict binding of a completed synthesis result to the requested job."""
from __future__ import annotations

import hashlib
import io
import json
import wave

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
    try:
        with wave.open(io.BytesIO(payload), 'rb') as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 24000, 2, 'NONE'):
                raise ValueError('invalid audio format')
            if audio.getnframes() != result['duration_ms'] * 24 or len(payload) != 44 + audio.getnframes() * 2:
                raise ValueError('invalid canonical audio size or duration')
            if len(audio.readframes(audio.getnframes())) != audio.getnframes() * 2:
                raise ValueError('truncated audio')
    except (EOFError, ValueError, wave.Error) as error:
        raise ProtocolError('INVALID_RESPONSE', 'invalid canonical WAV result') from error
