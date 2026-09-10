#!/usr/bin/env python3
"""Validate the packaged request validator against the candidate fixtures."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kilix_qwen_tts.interface import (
    CandidateError, EXPECTED_WIRE, _require, load_json, parse_control_frame,
    validate_request, validate_spec,
)
ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "contracts" / "provider-interface-candidate-v1.json"
VALID_DIR = ROOT / "contracts" / "fixtures" / "valid"
INVALID_DIR = ROOT / "contracts" / "fixtures" / "invalid"


def run() -> None:

    spec = load_json(SPEC_PATH)
    validate_spec(spec)
    _require(SPEC_PATH.read_bytes() == (ROOT / "src/kilix_qwen_tts/provider-interface-candidate-v1.json").read_bytes(),
             "SPEC_MISMATCH", "packaged and contract request specifications differ")
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
        parse_control_frame(spec, path.read_bytes())
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
