#!/usr/bin/env python3
"""Fail-closed checker for engine-neutral Qwen provider mechanics."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from kilix_qwen_tts.surface import (
    CLI_COMMANDS,
    MAX_AUDIO_CHUNK_BYTES,
    MAX_AUDIO_CHUNK_FRAMES,
    MAX_AUDIO_CHUNKS,
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


def synthetic_chunk(sequence: int, pcm: bytes, audio_format: AudioFormat) -> AudioChunk:
    return AudioChunk(
        sequence=sequence,
        descriptor_count=1,
        descriptor_index=0,
        frame_count=len(pcm) // audio_format.bytes_per_frame,
        byte_length=len(pcm),
        sha256=hashlib.sha256(pcm).hexdigest(),
        pcm=pcm,
    )


def synthetic_result(sample_format: str = "s16le"):
    audio_format = AudioFormat(sample_format, 24_000, 1)
    assembler = AudioStreamAssembler(audio_format, 1_000)
    assembler.append(synthetic_chunk(
        0, b"\x00" * (240 * audio_format.bytes_per_frame), audio_format,
    ))
    assembler.append(synthetic_chunk(
        1, b"\x01" * (240 * audio_format.bytes_per_frame), audio_format,
    ))
    return assembler.finish(
        job_id="job-synthetic",
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        model_revision="synthetic-revision-not-selected",
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
            and MAX_AUDIO_CHUNKS == spec["limits"]["audio_chunks_per_job"]
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

    prompt_pcm = b"\x01\x02" * 24_000
    digest = hashlib.sha256(prompt_pcm).hexdigest()
    prompt = PromptAudio(
        0, AudioFormat("s16le", 24_000, 1), 1_000, 24_000, 48_000, digest,
    )
    consent = ConsentBinding(
        "kilix.voice.consent/candidate-v1",
        digest,
        True,
        "this-project",
        None,
        "2026-08-29T00:00:00Z",
    )
    binding = bind_prompt(
        prompt, consent, prompt_pcm,
        descriptor_count=1,
        peer_uid=1000,
        model_id="synthetic-model-not-selected",
        model_revision="synthetic-revision-not-selected",
        job_id="job-clone",
    )
    require(binding.prompt_sha256 == digest and binding.peer_uid == 1000,
            "CONSENT_BINDING", "prompt bytes, peer, and consent did not bind")

    clone_format = AudioFormat("s16le", 24_000, 1)
    clone = AudioStreamAssembler(clone_format, 1_000)
    clone.append(synthetic_chunk(0, b"\x00\x00", clone_format))
    clone_result = clone.finish(
        job_id="job-clone",
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        model_revision="synthetic-revision-not-selected",
        seed=7,
        prompt_binding=binding,
    )
    clone_metadata = json.loads(result_metadata(clone_result))
    require(clone_metadata["conditioning"]["consent_sha256"] == binding.consent_sha256,
            "CONSENT_PROVENANCE", "result provenance lost the consent binding")

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
    s16_format = AudioFormat("s16le", 24_000, 1)
    duration = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1)
    duration.append(synthetic_chunk(0, b"\x00" * 48, s16_format))
    empty = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
    canceled = JobLifecycle()
    canceled.request_cancel()
    sealed = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
    sealed.append(synthetic_chunk(0, b"\x00\x00", s16_format))
    sealed.finish(job_id="j", engine_id="e", model_id="m", model_revision="r", seed=0)
    bad_digest_pcm = b"\x00\x00"
    out_of_order = (
        synthetic_chunk(0, b"\x00\x00", s16_format),
        synthetic_chunk(2, b"\x00\x00", s16_format),
    )
    controls = (
        (lambda: AudioFormat("u8", 24_000, 1), "AUDIO_FORMAT"),
        (lambda: aligned.append(AudioChunk(
            0, 1, 0, 1, 1, hashlib.sha256(b"\x00").hexdigest(), b"\x00",
        )), "CHUNK_ALIGNMENT"),
        (lambda: gap.append(synthetic_chunk(1, b"\x00\x00", s16_format)),
         "CHUNK_SEQUENCE"),
        (lambda: AudioChunk(
            0, 1, 0, 48_001, 96_002,
            hashlib.sha256(b"\x00" * 96_002).hexdigest(), b"\x00" * 96_002,
        ), "CHUNK_FRAMES"),
        (lambda: duration.append(synthetic_chunk(1, b"\x00\x00", s16_format)),
         "OUTPUT_DURATION"),
        (lambda: bind_prompt(prompt, ConsentBinding(
            "kilix.voice.consent/candidate-v1", "02" * 32, True,
            "this-project", None, "2026-08-29T00:00:00Z",
        ), prompt_pcm, descriptor_count=1, peer_uid=1000,
            model_id="m",
            model_revision="r", job_id="j"), "CONSENT_REQUIRED"),
        (lambda: bind_prompt(
            prompt, consent, b"\x00" * len(prompt_pcm), descriptor_count=1,
            peer_uid=1000, model_id="m", model_revision="r", job_id="j",
        ), "DESCRIPTOR_MISMATCH"),
        (lambda: AudioChunk(
            0, 1, 0, 1, len(bad_digest_pcm), "00" * 32, bad_digest_pcm,
        ), "DESCRIPTOR_MISMATCH"),
        (lambda: empty.finish(
            job_id="j", engine_id="e", model_id="m", model_revision="r", seed=0,
        ), "EMPTY_AUDIO"),
        (lambda: canceled.complete(s16), "JOB_TRANSITION"),
        (lambda: sealed.append(synthetic_chunk(1, b"\x00\x00", s16_format)),
         "STREAM_SEALED"),
        (lambda: SynthesisResult(
            "j", "e", "m", "r", 0, s16_format, out_of_order,
        ), "CHUNK_SEQUENCE"),
        (lambda: SynthesisResult(
            "job-clone", "e", "other-model", "synthetic-revision-not-selected",
            0, s16_format, (synthetic_chunk(0, b"\x00\x00", s16_format),), binding,
        ), "CONSENT_REQUIRED"),
    )
    for action, expected_code in controls:
        expect_refusal(action, expected_code)

    print(
        "QWEN_IMPLEMENTATION_SURFACE: PASS "
        "(6/6 commands; 2/2 introspection commands; 4/4 runtime commands refused; "
        "4/4 audio format/rate combinations; 2/2 deterministic WAV encodings; "
        "1/1 byte/peer/model/result-bound prompt; 7/7 lifecycle terminals/transitions; "
        "13/13 negative controls; 0/2 Qwen model lines selected; "
        "0/2 release tier slots selected; 0/3 synthesis modes selected)"
    )


if __name__ == "__main__":
    try:
        run()
    except (OSError, SurfaceError, TypeError, ValueError, json.JSONDecodeError) as error:
        code = error.code if isinstance(error, SurfaceError) else type(error).__name__
        print(f"QWEN_IMPLEMENTATION_SURFACE: FAIL [{code}] {error}", file=sys.stderr)
        raise SystemExit(1)
