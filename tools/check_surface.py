#!/usr/bin/env python3
"""Fail-closed checker for engine-neutral Qwen provider mechanics."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from kilix_qwen_tts.surface import (
    CLI_COMMANDS,
    MAX_AUDIO_CHUNK_BYTES,
    MAX_AUDIO_CHUNK_FRAMES,
    MAX_OUTPUT_DURATION_MS,
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    AudioChunk,
    AudioFormat,
    AudioStreamAssembler,
    ConsentBinding,
    JobLifecycle,
    JobState,
    PromptAudio,
    RuntimeUnselected,
    SurfaceError,
    SynthesisResult,
    bind_prompt,
    inspect_command,
    render_wav,
    result_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SurfaceError(code, message)


def expect_refusal(action, code: str) -> None:
    try:
        action()
    except SurfaceError as error:
        require(error.code == code, "WRONG_REASON",
                f"refused as {error.code}, expected {code}")
    else:
        raise SurfaceError("MUTATION_ACCEPTED", f"expected {code} refusal")


def synthetic_result(sample_format: str = "s16le"):
    audio_format = AudioFormat(sample_format, 24_000, 1)
    assembler = AudioStreamAssembler(audio_format, 1_000)
    assembler.append(0, b"\x00" * (240 * audio_format.bytes_per_frame))
    assembler.append(1, b"\x01" * (240 * audio_format.bytes_per_frame))
    return assembler.finish(
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        seed=7,
    )


def run() -> None:
    spec = json.loads(
        (ROOT / "contracts" / "provider-interface-candidate-v1.json").read_text(
            encoding="utf-8"
        )
    )
    require(CLI_COMMANDS == tuple(spec["cli_operations"]), "COMMANDS",
            "command population differs from 6/6")
    require(RUNTIME_COMMANDS == ("serve", "synthesize", "cancel", "unload"),
            "RUNTIME_COMMANDS", "runtime refusal population differs from 4/4")
    require(MAX_AUDIO_CHUNK_BYTES == spec["limits"]["audio_chunk_bytes"]
            and MAX_AUDIO_CHUNK_FRAMES == spec["limits"]["audio_chunk_frames"]
            and MAX_OUTPUT_DURATION_MS == spec["limits"]["output_duration_ms"],
            "LIMITS", "implementation limits differ from the interface candidate")

    formats = [
        AudioFormat(sample_format, sample_rate, 1)
        for sample_format in spec["audio"]["sample_formats"]
        for sample_rate in spec["audio"]["sample_rates_hz"]
    ]
    require(len(formats) == 4, "AUDIO_POPULATION",
            "audio format/rate population differs from 4/4")

    for command in ("models", "status"):
        payload = inspect_command(command)
        require(payload["model_lines"] == {"selected": 0, "total": 2},
                "PREMATURE_SELECTION", f"{command} selected a Qwen model line")
        require(payload["release_tier_slots"] == {"selected": 0, "total": 2},
                "PREMATURE_SELECTION", f"{command} selected a release tier")
        require(payload["synthesis_modes"] == {"selected": 0, "total": 3},
                "PREMATURE_SELECTION", f"{command} selected a synthesis mode")
    for command in RUNTIME_COMMANDS:
        try:
            inspect_command(command)
        except RuntimeUnselected as error:
            require(str(error) == PROVIDER_REFUSAL, "REFUSAL_TEXT",
                    f"{command} refusal text drifted")
        else:
            raise SurfaceError("RUNTIME_ADMITTED", f"{command} admitted an unselected runtime")

    s16 = synthetic_result("s16le")
    s16_wav = render_wav(s16)
    require(s16_wav[:4] == b"RIFF" and s16_wav[8:12] == b"WAVE",
            "WAV_HEADER", "integer WAV container header drifted")
    require(struct.unpack_from("<H", s16_wav, 20)[0] == 1,
            "WAV_FORMAT", "integer WAV format code drifted")
    f32 = synthetic_result("f32le")
    f32_wav = render_wav(f32)
    require(struct.unpack_from("<H", f32_wav, 20)[0] == 3 and b"fact" in f32_wav,
            "WAV_FORMAT", "float WAV format or fact chunk drifted")
    require(json.loads(result_metadata(s16))["audio"]["pcm_sha256"] == s16.pcm_sha256,
            "RESULT_METADATA", "result metadata digest drifted")

    digest = "01" * 32
    prompt = PromptAudio(0, AudioFormat("s16le", 24_000, 1), 1_000, 24_000, 48_000, digest)
    consent = ConsentBinding(
        "kilix.voice.consent/candidate-v1",
        digest,
        True,
        "this-project",
        "2026-08-29T00:00:00Z",
    )
    require(bind_prompt(prompt, consent) == digest, "CONSENT_BINDING",
            "prompt consent digest did not bind")

    success = JobLifecycle()
    require(success.start_loading() is JobState.LOADING, "LIFECYCLE", "load did not start")
    require(success.start_streaming() is JobState.STREAMING,
            "LIFECYCLE", "streaming did not start")
    require(success.complete(s16) is JobState.SUCCEEDED,
            "LIFECYCLE", "streaming did not complete")
    queued_cancel = JobLifecycle()
    require(queued_cancel.request_cancel() is JobState.CANCELED,
            "LIFECYCLE", "queued cancellation was not terminal")
    loading_cancel = JobLifecycle()
    loading_cancel.start_loading()
    require(loading_cancel.request_cancel() is JobState.CANCEL_REQUESTED,
            "LIFECYCLE", "loading cancellation was not requested")
    require(loading_cancel.acknowledge_cancel() is JobState.CANCELED,
            "LIFECYCLE", "cancellation was not acknowledged")
    failed = JobLifecycle()
    require(failed.fail("SYNTHETIC_FAILURE") is JobState.FAILED,
            "LIFECYCLE", "failure was not terminal")

    aligned = AudioStreamAssembler(AudioFormat("f32le", 24_000, 1), 1_000)
    gap = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
    frame_limit = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 10_000)
    duration = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1)
    duration.append(0, b"\x00" * 48)
    empty = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
    canceled = JobLifecycle()
    canceled.request_cancel()
    sealed = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
    sealed.append(0, b"\x00\x00")
    sealed.finish(engine_id="e", model_id="m", seed=0)
    out_of_order = (
        AudioChunk(0, 1, b"\x00\x00"),
        AudioChunk(2, 1, b"\x00\x00"),
    )
    controls = (
        (lambda: AudioFormat("u8", 24_000, 1), "AUDIO_FORMAT"),
        (lambda: aligned.append(0, b"\x00"), "CHUNK_ALIGNMENT"),
        (lambda: gap.append(1, b"\x00\x00"), "CHUNK_SEQUENCE"),
        (lambda: frame_limit.append(0, b"\x00" * (48_001 * 2)), "CHUNK_FRAMES"),
        (lambda: duration.append(1, b"\x00\x00"), "OUTPUT_DURATION"),
        (lambda: bind_prompt(prompt, ConsentBinding(
            "kilix.voice.consent/candidate-v1", "02" * 32, True,
            "this-project", "2026-08-29T00:00:00Z",
        )), "CONSENT_REQUIRED"),
        (lambda: empty.finish(engine_id="e", model_id="m", seed=0), "EMPTY_AUDIO"),
        (lambda: canceled.complete(s16), "JOB_TRANSITION"),
        (lambda: sealed.append(1, b"\x00\x00"), "STREAM_SEALED"),
        (lambda: SynthesisResult(
            "e", "m", 0, AudioFormat("s16le", 24_000, 1), out_of_order,
        ), "CHUNK_SEQUENCE"),
    )
    for action, expected_code in controls:
        expect_refusal(action, expected_code)

    print(
        "QWEN_IMPLEMENTATION_SURFACE: PASS "
        "(6/6 commands; 2/2 introspection commands; 4/4 runtime commands refused; "
        "4/4 audio format/rate combinations; 2/2 deterministic WAV encodings; "
        "1/1 consent-bound prompt; 7/7 lifecycle terminals/transitions; "
        "10/10 negative controls; 0/2 Qwen model lines selected; "
        "0/2 release tier slots selected; 0/3 synthesis modes selected)"
    )


if __name__ == "__main__":
    try:
        run()
    except (OSError, SurfaceError, TypeError, ValueError, json.JSONDecodeError) as error:
        code = error.code if isinstance(error, SurfaceError) else type(error).__name__
        print(f"QWEN_IMPLEMENTATION_SURFACE: FAIL [{code}] {error}", file=sys.stderr)
        raise SystemExit(1)
