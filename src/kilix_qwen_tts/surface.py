"""Pure Qwen-provider mechanics with no model or runtime dependency.

This module implements bounded PCM assembly, deterministic WAV framing,
consent binding, lifecycle transitions, and a fail-closed command shell.  It
does not load Qwen code or weights and is not the frozen P1 wire/catalog
contract or a P2 model/profile selection.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


SURFACE_SCHEMA = "kilix.qwen-tts.surface/candidate-v1"
CLI_COMMANDS = ("serve", "synthesize", "models", "status", "cancel", "unload")
RUNTIME_COMMANDS = ("serve", "synthesize", "cancel", "unload")
SAMPLE_FORMATS = ("s16le", "f32le")
SAMPLE_RATES_HZ = (24_000, 48_000)
CHANNELS = (1,)
SYNTHESIS_MODES = ("named_voice", "prompt_clone", "voice_design")
PROVIDER_REFUSAL = (
    "KILIX_QWEN_TTS_REFUSAL [RUNTIME_UNSELECTED] "
    "no Qwen runtime or release profile is selected"
)

MAX_AUDIO_CHUNK_FRAMES = 48_000
MAX_AUDIO_CHUNK_BYTES = 192_000
MAX_OUTPUT_DURATION_MS = 900_000
MAX_PROMPT_DURATION_MS = 30_000
MAX_PROMPT_AUDIO_BYTES = 11_520_000
MAX_ID_BYTES = 256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class SurfaceError(ValueError):
    """A stable candidate refusal with a machine-readable reason code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RuntimeUnselected(SurfaceError):
    def __init__(self) -> None:
        super().__init__("RUNTIME_UNSELECTED", PROVIDER_REFUSAL)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SurfaceError(code, message)


def _nonnegative_int(value: object, code: str, name: str) -> int:
    _require(type(value) is int and value >= 0, code,
             f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, code: str, name: str) -> int:
    _require(type(value) is int and value > 0, code,
             f"{name} must be a positive integer")
    return value


def _identity(value: object, code: str, name: str) -> str:
    _require(isinstance(value, str) and bool(value) and value.strip() == value,
             code, f"{name} must be non-empty trimmed text")
    _require(not _CONTROL_CHARACTERS.search(value), code,
             f"{name} contains a forbidden control character")
    _require(len(value.encode("utf-8")) <= MAX_ID_BYTES, code,
             f"{name} exceeds the candidate identity bound")
    return value


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_format: str
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        _require(self.sample_format in SAMPLE_FORMATS, "AUDIO_FORMAT",
                 "sample format is outside the 2/2 candidate population")
        _require(type(self.sample_rate_hz) is int and self.sample_rate_hz in SAMPLE_RATES_HZ,
                 "AUDIO_RATE", "sample rate is outside the 2/2 candidate population")
        _require(type(self.channels) is int and self.channels in CHANNELS,
                 "AUDIO_CHANNELS", "channel count is outside the 1/1 candidate population")

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self.sample_format == "s16le" else 4

    @property
    def bytes_per_frame(self) -> int:
        return self.bytes_per_sample * self.channels

    @property
    def wav_format_code(self) -> int:
        return 1 if self.sample_format == "s16le" else 3


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sequence: int
    frame_count: int
    pcm: bytes

    def __post_init__(self) -> None:
        _nonnegative_int(self.sequence, "CHUNK_SEQUENCE", "chunk sequence")
        frames = _positive_int(self.frame_count, "CHUNK_FRAMES", "chunk frame_count")
        _require(frames <= MAX_AUDIO_CHUNK_FRAMES, "CHUNK_FRAMES",
                 "chunk frame population exceeds the candidate bound")
        _require(type(self.pcm) is bytes and bool(self.pcm), "CHUNK_BYTES",
                 "chunk PCM must be non-empty immutable bytes")
        _require(len(self.pcm) <= MAX_AUDIO_CHUNK_BYTES, "CHUNK_BYTES",
                 "chunk PCM exceeds the candidate byte bound")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    engine_id: str
    model_id: str
    seed: int
    audio_format: AudioFormat
    chunks: tuple[AudioChunk, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_id", _identity(self.engine_id, "ENGINE_ID", "engine_id"))
        object.__setattr__(self, "model_id", _identity(self.model_id, "MODEL_ID", "model_id"))
        seed = _nonnegative_int(self.seed, "SEED", "seed")
        _require(seed < 2**63, "SEED", "seed exceeds the candidate bound")
        _require(isinstance(self.audio_format, AudioFormat), "AUDIO_FORMAT",
                 "audio_format must be an AudioFormat")
        _require(isinstance(self.chunks, tuple) and bool(self.chunks),
                 "EMPTY_AUDIO", "a final synthesis result requires at least 1/1 chunk")
        for expected, chunk in enumerate(self.chunks):
            _require(isinstance(chunk, AudioChunk), "CHUNK_TYPE",
                     "every result chunk must be an AudioChunk")
            _require(chunk.sequence == expected, "CHUNK_SEQUENCE",
                     "result chunks are not contiguous from sequence zero")
            _require(len(chunk.pcm) == chunk.frame_count * self.audio_format.bytes_per_frame,
                     "CHUNK_ALIGNMENT", "chunk bytes disagree with frame_count")
        _require(self.frame_count * 1000
                 <= MAX_OUTPUT_DURATION_MS * self.audio_format.sample_rate_hz,
                 "OUTPUT_DURATION", "result exceeds the candidate duration bound")

    @property
    def frame_count(self) -> int:
        return sum(chunk.frame_count for chunk in self.chunks)

    @property
    def pcm_bytes(self) -> bytes:
        return b"".join(chunk.pcm for chunk in self.chunks)

    @property
    def pcm_sha256(self) -> str:
        return hashlib.sha256(self.pcm_bytes).hexdigest()


class AudioStreamAssembler:
    """Validate ordered bounded chunks and seal an immutable result."""

    def __init__(self, audio_format: AudioFormat, max_duration_ms: int) -> None:
        _require(isinstance(audio_format, AudioFormat), "AUDIO_FORMAT",
                 "audio_format must be an AudioFormat")
        maximum = _positive_int(max_duration_ms, "OUTPUT_DURATION", "max_duration_ms")
        _require(maximum <= MAX_OUTPUT_DURATION_MS, "OUTPUT_DURATION",
                 "maximum output duration exceeds the candidate bound")
        self._format = audio_format
        self._max_duration_ms = maximum
        self._chunks: list[AudioChunk] = []
        self._frames = 0
        self._sealed = False

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def append(self, sequence: int, pcm: bytes) -> AudioChunk:
        _require(not self._sealed, "STREAM_SEALED", "audio stream is already sealed")
        _nonnegative_int(sequence, "CHUNK_SEQUENCE", "chunk sequence")
        _require(sequence == len(self._chunks), "CHUNK_SEQUENCE",
                 "chunk sequence must be contiguous from zero")
        _require(type(pcm) is bytes and bool(pcm), "CHUNK_BYTES",
                 "chunk PCM must be non-empty immutable bytes")
        _require(len(pcm) <= MAX_AUDIO_CHUNK_BYTES, "CHUNK_BYTES",
                 "chunk PCM exceeds the candidate byte bound")
        _require(len(pcm) % self._format.bytes_per_frame == 0,
                 "CHUNK_ALIGNMENT", "chunk PCM is not frame aligned")
        frames = len(pcm) // self._format.bytes_per_frame
        _require(0 < frames <= MAX_AUDIO_CHUNK_FRAMES, "CHUNK_FRAMES",
                 "chunk frame population exceeds the candidate bound")
        new_frames = self._frames + frames
        _require(new_frames * 1000 <= self._max_duration_ms * self._format.sample_rate_hz,
                 "OUTPUT_DURATION", "stream exceeds the requested duration bound")
        chunk = AudioChunk(sequence, frames, pcm)
        self._chunks.append(chunk)
        self._frames = new_frames
        return chunk

    def finish(self, *, engine_id: str, model_id: str, seed: int) -> SynthesisResult:
        _require(not self._sealed, "STREAM_SEALED", "audio stream is already sealed")
        _require(bool(self._chunks), "EMPTY_AUDIO", "cannot finish an empty audio stream")
        result = SynthesisResult(
            engine_id=engine_id,
            model_id=model_id,
            seed=seed,
            audio_format=self._format,
            chunks=tuple(self._chunks),
        )
        self._sealed = True
        return result


def _riff_chunk(kind: bytes, payload: bytes) -> bytes:
    _require(len(kind) == 4, "WAV_CHUNK", "RIFF chunk identifier must be 4/4 bytes")
    padding = b"\x00" if len(payload) % 2 else b""
    return kind + struct.pack("<I", len(payload)) + payload + padding


def render_wav(result: SynthesisResult) -> bytes:
    """Render a final result as deterministic RIFF/WAVE bytes."""

    _require(isinstance(result, SynthesisResult), "RESULT_TYPE",
             "result must be a final SynthesisResult")
    audio = result.audio_format
    bits = audio.bytes_per_sample * 8
    byte_rate = audio.sample_rate_hz * audio.bytes_per_frame
    fmt = struct.pack(
        "<HHIIHH",
        audio.wav_format_code,
        audio.channels,
        audio.sample_rate_hz,
        byte_rate,
        audio.bytes_per_frame,
        bits,
    )
    chunks = [_riff_chunk(b"fmt ", fmt)]
    if audio.sample_format == "f32le":
        chunks.append(_riff_chunk(b"fact", struct.pack("<I", result.frame_count)))
    chunks.append(_riff_chunk(b"data", result.pcm_bytes))
    body = b"WAVE" + b"".join(chunks)
    _require(len(body) <= 0xFFFFFFFF, "WAV_SIZE", "WAV exceeds the RIFF size bound")
    return b"RIFF" + struct.pack("<I", len(body)) + body


@dataclass(frozen=True, slots=True)
class PromptAudio:
    descriptor_index: int
    audio_format: AudioFormat
    duration_ms: int
    frame_count: int
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        _require(type(self.descriptor_index) is int and self.descriptor_index == 0,
                 "DESCRIPTOR_MISMATCH",
                 "prompt descriptor index must be 0 in the 1/1 allowed index population")
        _require(isinstance(self.audio_format, AudioFormat), "AUDIO_FORMAT",
                 "prompt audio_format must be an AudioFormat")
        duration = _positive_int(self.duration_ms, "PROMPT_DURATION", "prompt duration_ms")
        _require(duration <= MAX_PROMPT_DURATION_MS, "PROMPT_DURATION",
                 "prompt duration exceeds the candidate bound")
        frames = _positive_int(self.frame_count, "PROMPT_FRAMES", "prompt frame_count")
        size = _positive_int(self.byte_length, "PROMPT_BYTES", "prompt byte_length")
        _require(size <= MAX_PROMPT_AUDIO_BYTES, "PROMPT_BYTES",
                 "prompt bytes exceed the candidate bound")
        _require(size == frames * self.audio_format.bytes_per_frame,
                 "DESCRIPTOR_MISMATCH", "prompt bytes disagree with frame_count")
        _require(frames * 1000 == duration * self.audio_format.sample_rate_hz,
                 "DESCRIPTOR_MISMATCH", "prompt duration disagrees with frame_count")
        _require(isinstance(self.sha256, str) and _SHA256.fullmatch(self.sha256) is not None,
                 "PROMPT_DIGEST", "prompt SHA-256 is not canonical")


@dataclass(frozen=True, slots=True)
class ConsentBinding:
    schema: str
    source_sha256: str
    asserted_by_peer: bool
    allowed_use: str
    recorded_at: str

    def __post_init__(self) -> None:
        _require(self.schema == "kilix.voice.consent/candidate-v1",
                 "CONSENT_REQUIRED", "consent schema is incompatible")
        _require(isinstance(self.source_sha256, str)
                 and _SHA256.fullmatch(self.source_sha256) is not None,
                 "CONSENT_REQUIRED", "consent source digest is not canonical")
        _require(self.asserted_by_peer is True, "CONSENT_REQUIRED",
                 "authenticated peer did not attest consent")
        _require(self.allowed_use in {"this-project", "named-purpose"},
                 "CONSENT_REQUIRED", "consent allowed_use is invalid")
        _require(isinstance(self.recorded_at, str) and _UTC.fullmatch(self.recorded_at) is not None,
                 "CONSENT_REQUIRED", "consent timestamp is not canonical UTC")
        try:
            datetime.strptime(self.recorded_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise SurfaceError("CONSENT_REQUIRED", "consent timestamp is not a real UTC time") from error


def bind_prompt(prompt: PromptAudio, consent: ConsentBinding) -> str:
    _require(isinstance(prompt, PromptAudio), "PROMPT_TYPE",
             "prompt must be PromptAudio")
    _require(isinstance(consent, ConsentBinding), "CONSENT_REQUIRED",
             "consent must be ConsentBinding")
    _require(prompt.sha256 == consent.source_sha256, "CONSENT_REQUIRED",
             "consent is not bound to the prompt digest")
    return prompt.sha256


class JobState(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    STREAMING = "streaming"
    CANCEL_REQUESTED = "cancel-requested"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass(slots=True)
class JobLifecycle:
    state: JobState = JobState.QUEUED
    result: SynthesisResult | None = field(default=None, init=False)
    failure_code: str | None = field(default=None, init=False)

    def start_loading(self) -> JobState:
        _require(self.state is JobState.QUEUED, "JOB_TRANSITION",
                 "only a queued job can start loading")
        self.state = JobState.LOADING
        return self.state

    def start_streaming(self) -> JobState:
        _require(self.state is JobState.LOADING, "JOB_TRANSITION",
                 "only a loading job can start streaming")
        self.state = JobState.STREAMING
        return self.state

    def request_cancel(self) -> JobState:
        if self.state is JobState.QUEUED:
            self.state = JobState.CANCELED
        elif self.state in {JobState.LOADING, JobState.STREAMING}:
            self.state = JobState.CANCEL_REQUESTED
        elif self.state not in {JobState.CANCEL_REQUESTED, JobState.CANCELED}:
            raise SurfaceError("JOB_TERMINAL", "a terminal job cannot be canceled")
        return self.state

    def acknowledge_cancel(self) -> JobState:
        _require(self.state is JobState.CANCEL_REQUESTED, "JOB_TRANSITION",
                 "only a cancel-requested job can acknowledge cancellation")
        self.state = JobState.CANCELED
        return self.state

    def complete(self, result: SynthesisResult) -> JobState:
        _require(self.state is JobState.STREAMING, "JOB_TRANSITION",
                 "only a streaming job can complete")
        _require(isinstance(result, SynthesisResult), "RESULT_TYPE",
                 "completion requires a final SynthesisResult")
        self.result = result
        self.state = JobState.SUCCEEDED
        return self.state

    def fail(self, code: str) -> JobState:
        _require(self.state in {
            JobState.QUEUED, JobState.LOADING, JobState.STREAMING,
            JobState.CANCEL_REQUESTED,
        }, "JOB_TERMINAL", "a terminal job cannot fail again")
        self.failure_code = _identity(code, "FAILURE_CODE", "failure code")
        self.result = None
        self.state = JobState.FAILED
        return self.state


def status_payload() -> dict[str, Any]:
    return {
        "model_lines": {"selected": 0, "total": 2},
        "provider_state": "RUNTIME_UNSELECTED",
        "release_tier_slots": {"selected": 0, "total": 2},
        "schema": SURFACE_SCHEMA,
        "synthesis_modes": {"selected": 0, "total": 3},
    }


def models_payload() -> dict[str, Any]:
    return {
        "model_lines": {"selected": 0, "total": 2},
        "models": [],
        "release_tier_slots": {"selected": 0, "total": 2},
        "schema": SURFACE_SCHEMA,
        "synthesis_modes": {"selected": 0, "total": 3},
    }


def inspect_command(command: str) -> dict[str, Any]:
    _require(command in CLI_COMMANDS, "COMMAND",
             "command is outside the 6/6 candidate population")
    if command == "models":
        return models_payload()
    if command == "status":
        return status_payload()
    raise RuntimeUnselected()


def result_metadata(result: SynthesisResult) -> str:
    """Return deterministic metadata without transcript, prompt, or audio content."""

    payload = {
        "audio": {
            "channels": result.audio_format.channels,
            "frame_count": result.frame_count,
            "pcm_bytes": len(result.pcm_bytes),
            "pcm_sha256": result.pcm_sha256,
            "sample_format": result.audio_format.sample_format,
            "sample_rate_hz": result.audio_format.sample_rate_hz,
        },
        "engine": {"id": result.engine_id},
        "model": {"id": result.model_id},
        "schema": SURFACE_SCHEMA,
        "seed": result.seed,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
