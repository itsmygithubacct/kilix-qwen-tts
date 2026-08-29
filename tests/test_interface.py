from __future__ import annotations

import copy
import importlib.util
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

    def test_shell_field_is_refused_recursively(self) -> None:
        request = copy.deepcopy(self.named)
        request["extensions"] = {"x_nested": {"command": "run"}}
        self.refusal(request, "FORBIDDEN_FIELD")

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

    def test_incompatible_hello_major_is_refused(self) -> None:
        request = copy.deepcopy(self.hello)
        request["args"]["protocol_major"] = 2
        self.refusal(request, "INCOMPATIBLE_PROTOCOL")


if __name__ == "__main__":
    unittest.main()
