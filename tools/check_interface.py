#!/usr/bin/env python3
"""Validate the PREP7 Qwen provider-interface candidate and fixtures."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "contracts" / "provider-interface-candidate-v1.json"
VALID_DIR = ROOT / "contracts" / "fixtures" / "valid"
INVALID_DIR = ROOT / "contracts" / "fixtures" / "invalid"

EXPECTED_CLI = ("serve", "synthesize", "models", "status", "cancel", "unload")
EXPECTED_WIRE = ("hello", "submit", "models", "status", "cancel", "unload")
EXPECTED_EVENTS = (
    "hello", "accepted", "queued", "loading", "chunk", "progress",
    "result", "canceled", "error",
)
EXPECTED_MODES = {"named_voice", "prompt_clone", "voice_design"}
EXPECTED_ERRORS = (
    "INCOMPATIBLE_PROTOCOL", "INVALID_REQUEST", "FORBIDDEN_FIELD",
    "UNKNOWN_OPERATION", "UNAUTHORIZED_PEER",
    "LIMIT_EXCEEDED", "UNSUPPORTED_CAPABILITY", "CONSENT_REQUIRED",
    "MODEL_MISSING", "MODEL_BUSY", "DEVICE_UNAVAILABLE",
    "DEADLINE_EXCEEDED", "CANCELED", "WORKER_CRASH",
    "MALFORMED_WORKER_RESULT", "INSUFFICIENT_DISK",
    "OUTPUT_COMMIT_FAILED", "DESCRIPTOR_MISMATCH", "INTERNAL_ERROR",
)
EXPECTED_LIMITS = {
    "control_frame_bytes": 65536,
    "request_id_bytes": 64,
    "job_id_bytes": 64,
    "text_utf8_bytes": 16384,
    "instruction_utf8_bytes": 4096,
    "voice_design_utf8_bytes": 4096,
    "prompt_duration_ms": 30000,
    "prompt_audio_bytes": 11520000,
    "audio_chunk_frames": 48000,
    "audio_chunk_bytes": 192000,
    "descriptors_per_message": 1,
    "output_duration_ms": 900000,
    "deadline_ms": 3600000,
    "extension_fields": 32,
}
EXPECTED_AUDIO = {
    "sample_formats": ["s16le", "f32le"],
    "sample_rates_hz": [24000, 48000],
    "channels": [1],
    "descriptor_indices": [0],
}
HELLO_FIELDS = {
    "protocol_major", "protocol_minor", "max_control_frame_bytes",
    "max_audio_chunk_bytes", "max_descriptors_per_message",
}
SUBMIT_COMMON_FIELDS = {
    "task", "text", "model_id", "language", "instruction", "seed",
    "max_duration_ms", "mode", "output",
}
SUBMIT_REQUIRED_FIELDS = SUBMIT_COMMON_FIELDS - {"instruction"}
MODE_FIELDS = {
    "named_voice": {"voice_id"},
    "prompt_clone": {"prompt_fd", "prompt_audio", "consent"},
    "voice_design": {"description"},
}
REQUEST_FIELDS = SUBMIT_COMMON_FIELDS | set().union(*MODE_FIELDS.values()) | HELLO_FIELDS
TOP_LEVEL_FIELDS = {
    "schema", "type", "request_id", "op", "job_id", "deadline_ms",
    "extensions", "args",
}

TOKEN = re.compile(r"^[A-Za-z0-9._:+-]{1,64}$")
LANGUAGE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CandidateError(ValueError):
    """A stable refusal from the candidate validator."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise CandidateError(code, message)


def validate_spec(spec: dict[str, Any]) -> None:
    _require(spec.get("schema") == "kilix.qwen-tts.provider/candidate-v1",
             "SPEC_SCHEMA", "candidate schema identity drifted")
    _require(spec.get("status") == "PREP7_CANDIDATE_NOT_FROZEN",
             "SPEC_STATUS", "candidate must not claim a freeze")
    _require(spec.get("protocol") == {"major": 1, "minor": 0},
             "SPEC_PROTOCOL", "candidate protocol identity drifted")
    _require(spec.get("cli_operations") == list(EXPECTED_CLI),
             "SPEC_CLI", "CLI operation population differs from 6/6")
    _require(spec.get("wire_operations") == list(EXPECTED_WIRE),
             "SPEC_WIRE", "wire operation population differs from 6/6")
    _require(spec.get("event_types") == list(EXPECTED_EVENTS),
             "SPEC_EVENTS", "event population differs from 9/9")
    _require(set(spec.get("synthesis_modes", {})) == EXPECTED_MODES,
             "SPEC_MODES", "synthesis mode population differs from 3/3")
    _require(spec.get("error_codes") == list(EXPECTED_ERRORS),
             "SPEC_ERRORS", "error population differs from 19/19")

    transport = spec.get("transport", {})
    expected_transport = {
        "family": "AF_UNIX",
        "type": "SOCK_SEQPACKET",
        "peer_identity": "SO_PEERCRED_EFFECTIVE_UID",
        "runtime_directory_mode": "0700",
        "socket_mode": "0600",
        "control_framing": "U32_BE_LENGTH_PLUS_UTF8_JSON",
        "audio_transfer": "SCM_RIGHTS_DESCRIPTOR",
    }
    _require(transport == expected_transport, "SPEC_TRANSPORT",
             "transport or peer boundary drifted")

    limits = spec.get("limits", {})
    _require(limits == EXPECTED_LIMITS, "SPEC_LIMIT",
             "candidate limit population or value drifted")
    _require(spec.get("audio") == EXPECTED_AUDIO, "SPEC_AUDIO",
             "audio format or descriptor population drifted")

    clone = spec["synthesis_modes"]["prompt_clone"]
    _require("consent" in clone.get("required", []), "SPEC_CONSENT",
             "prompt cloning must require consent")
    _require(clone.get("consent_schema") == "kilix.voice.consent/candidate-v1",
             "SPEC_CONSENT", "consent schema identity drifted")

    forbidden = set(spec.get("forbidden_request_fields", []))
    allowed = set(spec.get("request_argument_fields", []))
    _require(forbidden.isdisjoint(allowed), "SPEC_FIELDS",
             "a forbidden caller-controlled field is allowlisted")
    _require(allowed == REQUEST_FIELDS
             and len(spec.get("request_argument_fields", [])) == len(REQUEST_FIELDS),
             "SPEC_FIELDS", "request argument population drifted")
    expected_operations = {
        "hello": {"job_id": "forbidden", "args": sorted(HELLO_FIELDS)},
        "submit": {"job_id": "required", "args": [
            "task", "text", "model_id", "language", "instruction", "seed",
            "max_duration_ms", "mode", "voice_id", "prompt_fd",
            "prompt_audio", "consent", "description", "output",
        ]},
        "models": {"job_id": "forbidden", "args": []},
        "status": {"job_id": "forbidden", "args": []},
        "cancel": {"job_id": "required", "args": []},
        "unload": {"job_id": "forbidden", "args": []},
    }
    observed_operations = spec.get("operation_requests", {})
    _require(set(observed_operations) == set(expected_operations),
             "SPEC_OPERATIONS", "operation request population differs from 6/6")
    for operation, expected in expected_operations.items():
        observed = observed_operations.get(operation, {})
        _require(observed.get("job_id") == expected["job_id"]
                 and set(observed.get("args", [])) == set(expected["args"])
                 and len(observed.get("args", [])) == len(expected["args"]),
                 "SPEC_OPERATIONS", f"request shape drifted for {operation}")


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _utf8_length(value: Any, field: str, maximum: int) -> str:
    _require(isinstance(value, str) and bool(value.strip()), "INVALID_REQUEST",
             f"{field} must be a non-empty string")
    _require(len(value.encode("utf-8")) <= maximum, "LIMIT_EXCEEDED",
             f"{field} exceeds its UTF-8 byte limit")
    return value


def _token(value: Any, field: str) -> str:
    _require(isinstance(value, str) and TOKEN.fullmatch(value) is not None,
             "INVALID_REQUEST", f"{field} is not a bounded token")
    return value


def _validate_output(spec: dict[str, Any], output: Any) -> None:
    _require(isinstance(output, dict), "INVALID_REQUEST", "output is required")
    _require(set(output) == {"sample_format", "sample_rate_hz", "channels"},
             "INVALID_REQUEST", "output field population is invalid")
    audio = spec["audio"]
    _require(output.get("sample_format") in audio["sample_formats"],
             "INVALID_REQUEST", "unsupported output sample format")
    _require(output.get("sample_rate_hz") in audio["sample_rates_hz"],
             "INVALID_REQUEST", "unsupported output sample rate")
    _require(output.get("channels") in audio["channels"],
             "INVALID_REQUEST", "unsupported output channel count")


def _validate_prompt(spec: dict[str, Any], args: dict[str, Any]) -> None:
    _require(isinstance(args.get("prompt_fd"), int)
             and not isinstance(args.get("prompt_fd"), bool)
             and args["prompt_fd"] in spec["audio"]["descriptor_indices"],
             "DESCRIPTOR_MISMATCH", "prompt descriptor index is invalid")
    prompt = args.get("prompt_audio")
    _require(isinstance(prompt, dict), "INVALID_REQUEST",
             "prompt_audio metadata is required")
    _require(set(prompt) == {
        "sample_format", "sample_rate_hz", "channels", "duration_ms",
        "frame_count", "byte_length", "sha256",
    }, "INVALID_REQUEST", "prompt_audio field population is invalid")
    output_view = {name: prompt[name] for name in ("sample_format", "sample_rate_hz", "channels")}
    _validate_output(spec, output_view)
    duration = prompt.get("duration_ms")
    frames = prompt.get("frame_count")
    size = prompt.get("byte_length")
    _require(isinstance(duration, int) and not isinstance(duration, bool)
             and 0 < duration <= spec["limits"]["prompt_duration_ms"],
             "LIMIT_EXCEEDED", "prompt duration is outside the candidate bound")
    _require(isinstance(frames, int) and not isinstance(frames, bool) and frames > 0,
             "INVALID_REQUEST", "prompt frame count is invalid")
    _require(isinstance(size, int) and not isinstance(size, bool)
             and 0 < size <= spec["limits"]["prompt_audio_bytes"],
             "LIMIT_EXCEEDED", "prompt byte length is outside the candidate bound")
    bytes_per_sample = {"s16le": 2, "f32le": 4}[prompt["sample_format"]]
    _require(size == frames * bytes_per_sample * prompt["channels"],
             "DESCRIPTOR_MISMATCH", "prompt byte length disagrees with PCM frames")
    _require(frames * 1000 == duration * prompt["sample_rate_hz"],
             "DESCRIPTOR_MISMATCH", "prompt duration disagrees with PCM frames")
    _require(isinstance(prompt.get("sha256"), str)
             and SHA256.fullmatch(prompt["sha256"]) is not None,
             "INVALID_REQUEST", "prompt SHA-256 is invalid")

    consent = args.get("consent")
    _require(isinstance(consent, dict), "CONSENT_REQUIRED",
             "prompt cloning requires a consent record")
    _require(set(consent) == {
        "schema", "source_sha256", "asserted_by_peer", "allowed_use", "recorded_at",
    }, "CONSENT_REQUIRED", "consent field population is invalid")
    _require(consent.get("schema") == "kilix.voice.consent/candidate-v1",
             "CONSENT_REQUIRED", "consent schema is incompatible")
    _require(consent.get("source_sha256") == prompt.get("sha256"),
             "CONSENT_REQUIRED", "consent is not bound to the prompt digest")
    _require(consent.get("asserted_by_peer") is True, "CONSENT_REQUIRED",
             "authenticated peer did not attest consent")
    _require(consent.get("allowed_use") in {"this-project", "named-purpose"},
             "CONSENT_REQUIRED", "consent allowed_use is invalid")
    _require(isinstance(consent.get("recorded_at"), str)
             and UTC.fullmatch(consent["recorded_at"]) is not None,
             "CONSENT_REQUIRED", "consent timestamp is not canonical UTC")


def validate_request(spec: dict[str, Any], request: Any) -> None:
    _require(isinstance(request, dict), "INVALID_REQUEST", "request is not an object")
    forbidden = set(spec["forbidden_request_fields"])
    used_forbidden = sorted(forbidden.intersection(_walk_keys(request)))
    _require(not used_forbidden, "FORBIDDEN_FIELD",
             f"forbidden request field present: {', '.join(used_forbidden)}")
    _require(set(request).issubset(TOP_LEVEL_FIELDS), "INVALID_REQUEST",
             "request has an unscoped top-level field")
    _require({"schema", "type", "request_id", "op", "deadline_ms", "args"}
             .issubset(request), "INVALID_REQUEST",
             "request is missing one or more required envelope fields")
    _require(request.get("schema") == spec["schema"], "INCOMPATIBLE_PROTOCOL",
             "request schema is incompatible")
    _require(request.get("type") == "request", "INVALID_REQUEST",
             "message type must be request")
    _token(request.get("request_id"), "request_id")
    op = request.get("op")
    _require(op in EXPECTED_WIRE, "UNKNOWN_OPERATION", "wire operation is unknown")
    deadline = request.get("deadline_ms")
    _require(isinstance(deadline, int) and not isinstance(deadline, bool)
             and 0 < deadline <= spec["limits"]["deadline_ms"],
             "LIMIT_EXCEEDED", "deadline is outside the candidate bound")
    extensions = request.get("extensions", {})
    _require(isinstance(extensions, dict), "INVALID_REQUEST",
             "extensions must be an object")
    _require(len(extensions) <= spec["limits"]["extension_fields"],
             "LIMIT_EXCEEDED", "too many extension fields")
    _require(all(isinstance(key, str) and key.startswith("x_") for key in extensions),
             "INVALID_REQUEST", "extension keys must begin with x_")
    args = request.get("args", {})
    _require(isinstance(args, dict), "INVALID_REQUEST", "args must be an object")

    job_policy = spec["operation_requests"][op]["job_id"]
    if job_policy == "required":
        _token(request.get("job_id"), "job_id")
    else:
        _require("job_id" not in request, "INVALID_REQUEST",
                 f"{op} forbids job_id")

    if op == "hello":
        _require(set(args) == HELLO_FIELDS, "INVALID_REQUEST",
                 "hello argument population is invalid")
        protocol = spec["protocol"]
        _require(args.get("protocol_major") == protocol["major"],
                 "INCOMPATIBLE_PROTOCOL", "protocol major is incompatible")
        minor = args.get("protocol_minor")
        _require(isinstance(minor, int) and not isinstance(minor, bool)
                 and minor >= protocol["minor"],
                 "INCOMPATIBLE_PROTOCOL", "protocol minor is incompatible")
        for field, limit in (
            ("max_control_frame_bytes", spec["limits"]["control_frame_bytes"]),
            ("max_audio_chunk_bytes", spec["limits"]["audio_chunk_bytes"]),
            ("max_descriptors_per_message", spec["limits"]["descriptors_per_message"]),
        ):
            value = args.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool)
                     and 0 < value <= limit,
                     "LIMIT_EXCEEDED", f"{field} is outside the candidate bound")
        return

    if op != "submit":
        _require(not args, "INVALID_REQUEST", f"{op} does not accept candidate args")
        return

    _require(SUBMIT_REQUIRED_FIELDS.issubset(args), "INVALID_REQUEST",
             "submit is missing one or more required arguments")
    _require(args.get("task") == "synthesize", "INVALID_REQUEST",
             "submit task must be synthesize")
    _utf8_length(args.get("text"), "text", spec["limits"]["text_utf8_bytes"])
    _token(args.get("model_id"), "model_id")
    _require(isinstance(args.get("language"), str)
             and LANGUAGE.fullmatch(args["language"]) is not None,
             "INVALID_REQUEST", "language is not a bounded language tag")
    seed = args.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool)
             and 0 <= seed < 2**63, "INVALID_REQUEST", "seed is invalid")
    maximum = args.get("max_duration_ms")
    _require(isinstance(maximum, int) and not isinstance(maximum, bool)
             and 0 < maximum <= spec["limits"]["output_duration_ms"],
             "LIMIT_EXCEEDED", "maximum output duration is outside the bound")
    if "instruction" in args:
        _utf8_length(args["instruction"], "instruction",
                     spec["limits"]["instruction_utf8_bytes"])
    _validate_output(spec, args.get("output"))

    mode = args.get("mode")
    _require(mode in EXPECTED_MODES, "UNSUPPORTED_CAPABILITY",
             "synthesis mode is unknown")
    allowed_mode_fields = SUBMIT_COMMON_FIELDS | MODE_FIELDS[mode]
    _require(set(args).issubset(allowed_mode_fields),
             "INVALID_REQUEST", "submit argument population is invalid for its mode")
    if mode == "named_voice":
        _token(args.get("voice_id"), "voice_id")
    elif mode == "prompt_clone":
        _validate_prompt(spec, args)
    else:
        _utf8_length(args.get("description"), "description",
                     spec["limits"]["voice_design_utf8_bytes"])


def run() -> None:
    spec = load_json(SPEC_PATH)
    validate_spec(spec)
    valid_paths = sorted(VALID_DIR.glob("*.json"))
    invalid_paths = sorted(INVALID_DIR.glob("*.json"))
    _require(len(valid_paths) == 8, "FIXTURE_POPULATION",
             "valid fixture population differs from 8/8")
    _require(len(invalid_paths) == 6, "FIXTURE_POPULATION",
             "refusal fixture population differs from 6/6")
    valid_operations = {
        load_json(path).get("op") for path in valid_paths
    }
    _require(valid_operations == set(EXPECTED_WIRE), "FIXTURE_OPERATION_COVERAGE",
             "valid fixtures cover fewer than 6/6 wire operations")

    for path in valid_paths:
        validate_request(spec, load_json(path))
    for path in invalid_paths:
        record = load_json(path)
        try:
            validate_request(spec, record.get("request"))
        except CandidateError as error:
            _require(error.code == record.get("expected_error"),
                     "FIXTURE_WRONG_REASON",
                     f"{path.name} refused as {error.code}, expected "
                     f"{record.get('expected_error')}")
        else:
            raise CandidateError("FIXTURE_ACCEPTED", f"{path.name} was accepted")

    print(
        "QWEN_INTERFACE_CANDIDATE: PASS "
        "(6/6 CLI operations; 6/6 wire operations; 9/9 message types; "
        "19/19 errors; 8/8 valid fixtures; 6/6 refusal fixtures)"
    )


if __name__ == "__main__":
    try:
        run()
    except (CandidateError, OSError, ValueError, json.JSONDecodeError) as error:
        code = error.code if isinstance(error, CandidateError) else type(error).__name__
        print(f"QWEN_INTERFACE_CANDIDATE: FAIL [{code}] {error}", file=sys.stderr)
        raise SystemExit(1)
