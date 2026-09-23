"""The stager binds a uv environment's symlinked interpreter as the runtime verifies it."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from kilix_qwen_tts.runtime import digest_file

_SPEC = importlib.util.spec_from_file_location(
    "stage_runtime", Path(__file__).resolve().parents[1] / "tools/stage_runtime.py")
stage_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_runtime)


class StageRuntimeTests(unittest.TestCase):
    def test_symlinked_environment_interpreter_binds_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "python-root/bin/python3.12"
            base.parent.mkdir(parents=True)
            base.write_bytes(b"interpreter")
            python = root / "environment/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(base)
            self.assertEqual(stage_runtime.interpreter_sha256(python),
                             digest_file(python, follow=True))
            self.assertEqual(stage_runtime.interpreter_sha256(python), digest_file(base))


if __name__ == "__main__":
    unittest.main()
