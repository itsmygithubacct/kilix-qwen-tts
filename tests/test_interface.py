from __future__ import annotations

import copy
import importlib.util
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_interface", ROOT / "tools" / "check_interface.py"
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
interface = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(interface)


class InterfaceCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = interface.load_json(interface.SPEC_PATH)
        cls.named = interface.load_json(
            interface.VALID_DIR / "named-voice.json"
        )
        cls.clone = interface.load_json(
            interface.VALID_DIR / "prompt-clone.json"
        )
        cls.hello = interface.load_json(
            interface.VALID_DIR / "hello.json"
        )

    def refusal(self, request, code: str) -> None:
        with self.assertRaises(interface.CandidateError) as caught:
            interface.validate_request(self.spec, request)
        self.assertEqual(caught.exception.code, code)

    def test_spec_population(self) -> None:
        interface.validate_spec(self.spec)

    def test_all_committed_fixtures(self) -> None:
        interface.run()

    def test_prompt_digest_must_match_consent(self) -> None:
        request = copy.deepcopy(self.clone)
        request["args"]["consent"]["source_sha256"] = "f" * 64
        self.refusal(request, "CONSENT_REQUIRED")

    def test_named_purpose_must_be_explicit(self) -> None:
        request = copy.deepcopy(self.clone)
        request["args"]["consent"]["allowed_use"] = "named-purpose"
        request["args"]["consent"]["purpose"] = None
        self.refusal(request, "CONSENT_REQUIRED")

    def test_unhashable_consent_use_is_stably_refused(self) -> None:
        for malformed in ([], {}):
            with self.subTest(malformed=malformed):
                request = copy.deepcopy(self.clone)
                request["args"]["consent"]["allowed_use"] = malformed
                self.refusal(request, "CONSENT_REQUIRED")

    def test_this_project_forbids_named_purpose(self) -> None:
        request = copy.deepcopy(self.clone)
        request["args"]["consent"]["purpose"] = "another-purpose"
        self.refusal(request, "CONSENT_REQUIRED")

    def test_impossible_consent_timestamp_is_refused(self) -> None:
        request = copy.deepcopy(self.clone)
        request["args"]["consent"]["recorded_at"] = "2026-02-30T00:00:00Z"
        self.refusal(request, "CONSENT_REQUIRED")

    def test_prompt_limit_is_enforced(self) -> None:
        with self.subTest("duration"):
            request = copy.deepcopy(self.clone)
            request["args"]["prompt_audio"]["duration_ms"] = 30001
            self.refusal(request, "LIMIT_EXCEEDED")
        with self.subTest("frame-byte consistency"):
            request = copy.deepcopy(self.clone)
            request["args"]["prompt_audio"]["byte_length"] += 2
            self.refusal(request, "DESCRIPTOR_MISMATCH")

    def test_output_channels_are_bounded(self) -> None:
        request = copy.deepcopy(self.named)
        request["args"]["output"]["channels"] = 2
        self.refusal(request, "INVALID_REQUEST")

    def test_boolean_output_numbers_are_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["args"]["output"]["channels"] = True
        self.refusal(request, "INVALID_REQUEST")

    def test_shell_field_is_refused_recursively(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_nested": {"command": "run"}}
        self.refusal(request, "FORBIDDEN_FIELD")

    def test_control_nesting_is_bounded(self) -> None:
        request = copy.deepcopy(self.named)
        nested = True
        for _ in range(self.spec["limits"]["control_json_depth"]):
            nested = [nested]
        request["extensions"] = {"x_nested": nested}
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_extreme_direct_control_nesting_is_stably_refused(self) -> None:
        request = copy.deepcopy(self.named)
        nested = True
        for _ in range(10_000):
            nested = [nested]
        request["extensions"] = {"x_nested": nested}
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_control_node_population_is_bounded(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {
            "x_nodes": [None] * self.spec["limits"]["control_json_nodes"]
        }
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_control_key_length_is_bounded(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {
            "x_" + "a" * self.spec["limits"]["control_key_utf8_bytes"]: True
        }
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_control_frame_size_is_bounded(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {
            "x_padding": "a" * self.spec["limits"]["control_frame_bytes"]
        }
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_raw_control_frame_size_is_bounded_before_decode(self) -> None:
        payload = b" " * (self.spec["limits"]["control_frame_bytes"] + 1)
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, payload)
        self.assertEqual(caught.exception.code, "LIMIT_EXCEEDED")

    def test_extreme_raw_control_nesting_is_stably_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_nested": "QWEN1_NESTING_MARKER"}
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        nested = b"[" * 10_000 + b"true" + b"]" * 10_000
        payload = payload.replace(b'"QWEN1_NESTING_MARKER"', nested, 1)
        self.assertLessEqual(len(payload), self.spec["limits"]["control_frame_bytes"])
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, payload)
        self.assertEqual(caught.exception.code, "LIMIT_EXCEEDED")

    def test_extreme_raw_integer_is_stably_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_integer": "QWEN1_INTEGER_MARKER"}
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        integer = b"9" * 50_000
        payload = payload.replace(b'"QWEN1_INTEGER_MARKER"', integer, 1)
        self.assertLessEqual(len(payload), self.spec["limits"]["control_frame_bytes"])
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, payload)
        self.assertEqual(caught.exception.code, "LIMIT_EXCEEDED")

    def test_extreme_direct_integer_is_stably_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_integer": 10 ** 50_000}
        self.refusal(request, "LIMIT_EXCEEDED")

    def test_duplicate_control_key_is_refused(self) -> None:
        payload = b'{"schema":"first","schema":"second"}'
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, payload)
        self.assertEqual(caught.exception.code, "INVALID_REQUEST")

    def test_invalid_control_utf8_is_refused(self) -> None:
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, b"\xff")
        self.assertEqual(caught.exception.code, "INVALID_REQUEST")

    def test_lone_surrogate_escape_is_stably_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_text": "QWEN1_SURROGATE_MARKER"}
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        payload = payload.replace(b'"QWEN1_SURROGATE_MARKER"', b'"\\ud800"', 1)
        with self.assertRaises(interface.CandidateError) as caught:
            interface.parse_control_frame(self.spec, payload)
        self.assertEqual(caught.exception.code, "INVALID_REQUEST")

    def test_non_finite_extension_number_is_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_number": math.inf}
        self.refusal(request, "INVALID_REQUEST")

    def test_compatible_extension_shape(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_review": {"unknown": True}}
        interface.validate_request(self.spec, request)

    def test_unscoped_extension_is_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"review": True}
        self.refusal(request, "INVALID_REQUEST")

    def test_all_wire_operations_have_valid_fixture(self) -> None:
        operations = {
            interface.load_json(path)["op"]
            for path in interface.VALID_DIR.glob("*.json")
        }
        self.assertEqual(operations, set(interface.EXPECTED_WIRE))

    def test_unknown_top_level_field_is_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["temperature"] = 0.7
        self.refusal(request, "INVALID_REQUEST")

    def test_unknown_submit_argument_is_refused(self) -> None:
        request = copy.deepcopy(self.named)
        request["args"]["temperature"] = 0.7
        self.refusal(request, "INVALID_REQUEST")

    def test_unhashable_request_enums_are_stably_refused(self) -> None:
        for field, code in (
            ("op", "UNKNOWN_OPERATION"),
            ("mode", "UNSUPPORTED_CAPABILITY"),
            ("sample_format", "INVALID_REQUEST"),
        ):
            for malformed in ([], {}):
                with self.subTest(field=field, malformed=malformed):
                    request = copy.deepcopy(self.named)
                    if field == "op":
                        request["op"] = malformed
                    elif field == "mode":
                        request["args"]["mode"] = malformed
                    else:
                        request["args"]["output"]["sample_format"] = malformed
                    self.refusal(request, code)

    def test_incompatible_hello_major_is_refused(self) -> None:
        request = copy.deepcopy(self.hello)
        request["args"]["protocol_major"] = 2
        self.refusal(request, "INCOMPATIBLE_PROTOCOL")

    def test_protocol_major_requires_exact_integer(self) -> None:
        for malformed in (True, 1.0):
            with self.subTest(malformed=malformed):
                request = copy.deepcopy(self.hello)
                request["args"]["protocol_major"] = malformed
                self.refusal(request, "INCOMPATIBLE_PROTOCOL")


if __name__ == "__main__":
    unittest.main()
