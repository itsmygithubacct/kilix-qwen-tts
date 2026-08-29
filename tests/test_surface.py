from __future__ import annotations

import contextlib
import io
import json
import struct
import unittest

from kilix_qwen_tts.cli import main
from kilix_qwen_tts.surface import (
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    AudioChunk,
    AudioFormat,
    AudioStreamAssembler,
    ConsentBinding,
    JobLifecycle,
    JobState,
    PromptAudio,
    SurfaceError,
    SynthesisResult,
    bind_prompt,
    render_wav,
    result_metadata,
)


def result_fixture(sample_format: str = "s16le") -> SynthesisResult:
    audio_format = AudioFormat(sample_format, 24_000, 1)
    assembler = AudioStreamAssembler(audio_format, 1_000)
    assembler.append(0, b"\x00" * (120 * audio_format.bytes_per_frame))
    assembler.append(1, b"\x01" * (120 * audio_format.bytes_per_frame))
    return assembler.finish(
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        seed=11,
    )


class AudioFormatTests(unittest.TestCase):
    def refusal(self, action, code: str) -> None:
        with self.assertRaises(SurfaceError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_all_format_rate_combinations(self) -> None:
        formats = [
            AudioFormat(sample_format, rate, 1)
            for sample_format in ("s16le", "f32le")
            for rate in (24_000, 48_000)
        ]
        self.assertEqual(len(formats), 4)

    def test_unknown_sample_format_is_refused(self) -> None:
        self.refusal(lambda: AudioFormat("u8", 24_000, 1), "AUDIO_FORMAT")

    def test_boolean_sample_rate_is_refused(self) -> None:
        self.refusal(lambda: AudioFormat("s16le", True, 1), "AUDIO_RATE")

    def test_stereo_is_refused(self) -> None:
        self.refusal(lambda: AudioFormat("s16le", 24_000, 2), "AUDIO_CHANNELS")


class AssemblerTests(unittest.TestCase):
    def refusal(self, action, code: str) -> None:
        with self.assertRaises(SurfaceError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_ordered_chunks_finish(self) -> None:
        result = result_fixture()
        self.assertEqual((len(result.chunks), result.frame_count, len(result.pcm_bytes)),
                         (2, 240, 480))

    def test_sequence_gap_is_refused(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
        self.refusal(lambda: assembler.append(1, b"\x00\x00"), "CHUNK_SEQUENCE")

    def test_misaligned_chunk_is_refused(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("f32le", 24_000, 1), 1_000)
        self.refusal(lambda: assembler.append(0, b"\x00"), "CHUNK_ALIGNMENT")

    def test_chunk_byte_bound_is_enforced(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 10_000)
        self.refusal(lambda: assembler.append(0, b"\x00" * 192_002), "CHUNK_BYTES")

    def test_chunk_frame_bound_is_enforced(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 10_000)
        self.refusal(lambda: assembler.append(0, b"\x00" * 96_002), "CHUNK_FRAMES")

    def test_duration_bound_is_enforced(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1)
        assembler.append(0, b"\x00" * 48)
        self.refusal(lambda: assembler.append(1, b"\x00\x00"), "OUTPUT_DURATION")

    def test_empty_stream_cannot_finish(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
        self.refusal(lambda: assembler.finish(engine_id="e", model_id="m", seed=0),
                     "EMPTY_AUDIO")

    def test_finished_stream_is_sealed(self) -> None:
        assembler = AudioStreamAssembler(AudioFormat("s16le", 24_000, 1), 1_000)
        assembler.append(0, b"\x00\x00")
        assembler.finish(engine_id="e", model_id="m", seed=0)
        self.refusal(lambda: assembler.append(1, b"\x00\x00"), "STREAM_SEALED")

    def test_direct_oversized_chunk_is_refused(self) -> None:
        self.refusal(lambda: AudioChunk(0, 48_001, b"\x00" * 96_002),
                     "CHUNK_FRAMES")

    def test_direct_result_requires_contiguous_chunks(self) -> None:
        chunks = (AudioChunk(0, 1, b"\x00\x00"), AudioChunk(2, 1, b"\x00\x00"))
        self.refusal(
            lambda: SynthesisResult(
                "e", "m", 0, AudioFormat("s16le", 24_000, 1), chunks,
            ),
            "CHUNK_SEQUENCE",
        )


class WavTests(unittest.TestCase):
    def test_integer_wav_header(self) -> None:
        wav = render_wav(result_fixture("s16le"))
        self.assertEqual((wav[:4], wav[8:12], struct.unpack_from("<H", wav, 20)[0]),
                         (b"RIFF", b"WAVE", 1))

    def test_float_wav_has_fact_chunk(self) -> None:
        wav = render_wav(result_fixture("f32le"))
        self.assertEqual(struct.unpack_from("<H", wav, 20)[0], 3)
        self.assertIn(b"fact", wav)

    def test_wav_is_deterministic(self) -> None:
        self.assertEqual(render_wav(result_fixture()), render_wav(result_fixture()))

    def test_metadata_contains_identity_and_digest_not_pcm(self) -> None:
        result = result_fixture()
        metadata = result_metadata(result)
        payload = json.loads(metadata)
        self.assertEqual(payload["engine"]["id"], "synthetic-engine-not-selected")
        self.assertEqual(payload["model"]["id"], "synthetic-model-not-selected")
        self.assertEqual(payload["audio"]["pcm_sha256"], result.pcm_sha256)
        self.assertNotIn("\x00", metadata)


class ConsentTests(unittest.TestCase):
    @staticmethod
    def prompt(digest: str = "01" * 32) -> PromptAudio:
        return PromptAudio(0, AudioFormat("s16le", 24_000, 1),
                           1_000, 24_000, 48_000, digest)

    @staticmethod
    def consent(digest: str = "01" * 32, *, asserted: bool = True,
                recorded_at: str = "2026-08-29T00:00:00Z") -> ConsentBinding:
        return ConsentBinding(
            "kilix.voice.consent/candidate-v1", digest, asserted,
            "this-project", recorded_at,
        )

    def test_valid_prompt_consent_binds(self) -> None:
        self.assertEqual(bind_prompt(self.prompt(), self.consent()), "01" * 32)

    def test_digest_mismatch_is_refused(self) -> None:
        with self.assertRaises(SurfaceError) as caught:
            bind_prompt(self.prompt(), self.consent("02" * 32))
        self.assertEqual(caught.exception.code, "CONSENT_REQUIRED")

    def test_unasserted_consent_is_refused(self) -> None:
        with self.assertRaises(SurfaceError) as caught:
            self.consent(asserted=False)
        self.assertEqual(caught.exception.code, "CONSENT_REQUIRED")

    def test_impossible_timestamp_is_refused(self) -> None:
        with self.assertRaises(SurfaceError) as caught:
            self.consent(recorded_at="2026-02-30T00:00:00Z")
        self.assertEqual(caught.exception.code, "CONSENT_REQUIRED")

    def test_prompt_frame_byte_mismatch_is_refused(self) -> None:
        with self.assertRaises(SurfaceError) as caught:
            PromptAudio(0, AudioFormat("s16le", 24_000, 1),
                        1_000, 24_000, 47_998, "01" * 32)
        self.assertEqual(caught.exception.code, "DESCRIPTOR_MISMATCH")


class LifecycleTests(unittest.TestCase):
    def test_success_requires_loading_and_streaming(self) -> None:
        job = JobLifecycle()
        self.assertIs(job.start_loading(), JobState.LOADING)
        self.assertIs(job.start_streaming(), JobState.STREAMING)
        self.assertIs(job.complete(result_fixture()), JobState.SUCCEEDED)
        self.assertIsNotNone(job.result)

    def test_queued_cancel_is_terminal(self) -> None:
        job = JobLifecycle()
        self.assertIs(job.request_cancel(), JobState.CANCELED)
        self.assertIsNone(job.result)

    def test_loading_cancel_requires_acknowledgement(self) -> None:
        job = JobLifecycle()
        job.start_loading()
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)
        self.assertIs(job.acknowledge_cancel(), JobState.CANCELED)

    def test_streaming_cancel_is_idempotent_while_pending(self) -> None:
        job = JobLifecycle()
        job.start_loading()
        job.start_streaming()
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)

    def test_terminal_success_cannot_be_canceled(self) -> None:
        job = JobLifecycle()
        job.start_loading()
        job.start_streaming()
        job.complete(result_fixture())
        with self.assertRaises(SurfaceError) as caught:
            job.request_cancel()
        self.assertEqual(caught.exception.code, "JOB_TERMINAL")

    def test_failure_retains_no_result(self) -> None:
        job = JobLifecycle()
        self.assertIs(job.fail("SYNTHETIC_FAILURE"), JobState.FAILED)
        self.assertIsNone(job.result)


class CommandTests(unittest.TestCase):
    def invoke(self, command: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([command])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_status_reports_zero_selection(self) -> None:
        code, stdout, stderr = self.invoke("status")
        self.assertEqual((code, stderr), (0, ""))
        payload = json.loads(stdout)
        self.assertEqual(payload["model_lines"], {"selected": 0, "total": 2})
        self.assertEqual(payload["release_tier_slots"], {"selected": 0, "total": 2})
        self.assertEqual(payload["synthesis_modes"], {"selected": 0, "total": 3})

    def test_models_reports_empty_population(self) -> None:
        code, stdout, stderr = self.invoke("models")
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["models"], [])

    def test_all_runtime_commands_refuse_verbatim(self) -> None:
        for command in RUNTIME_COMMANDS:
            with self.subTest(command=command):
                code, stdout, stderr = self.invoke(command)
                self.assertEqual((code, stdout), (69, ""))
                self.assertEqual(stderr, PROVIDER_REFUSAL + "\n")


if __name__ == "__main__":
    unittest.main()
