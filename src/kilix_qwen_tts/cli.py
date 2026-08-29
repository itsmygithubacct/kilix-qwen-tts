"""Candidate command shell that refuses operations requiring a Qwen runtime."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .surface import CLI_COMMANDS, RuntimeUnselected, SurfaceError, inspect_command


def parser() -> argparse.ArgumentParser:
    candidate = argparse.ArgumentParser(prog="kilix-qwen-tts")
    commands = candidate.add_subparsers(dest="command", required=True)
    for command in CLI_COMMANDS:
        commands.add_parser(command)
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        payload = inspect_command(arguments.command)
    except RuntimeUnselected as error:
        print(str(error), file=sys.stderr)
        return 69
    except SurfaceError as error:
        print(f"KILIX_QWEN_TTS_REFUSAL [{error.code}] {error}", file=sys.stderr)
        return 64
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
