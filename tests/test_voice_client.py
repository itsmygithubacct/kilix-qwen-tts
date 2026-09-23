"""The executable client keeps WAV bytes off its metadata channel."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.cli import main


class VoiceClientTests(unittest.TestCase):
    def test_wav_stdout_binds_explicit_model_and_receipt_requirement(self):
        stdin = SimpleNamespace(buffer=io.BytesIO(b"Hello"))
        stdout = SimpleNamespace(buffer=io.BytesIO())
        stderr = io.StringIO()
        captured = []

        def client_request(_directory, request, _descriptor=None, **_kwargs):
            captured.append(request)
            return {"model_id": "qwen3-tts-0.6b-customvoice", "seed": 0}, b"RIFF-audio"

        with patch("kilix_qwen_tts.cli.sys.stdin", stdin), \
                patch("kilix_qwen_tts.cli.sys.stdout", stdout), \
                patch("kilix_qwen_tts.cli.sys.stderr", stderr), \
                patch("kilix_qwen_tts.service.runtime_directory", return_value=Path("/tmp")), \
                patch("kilix_qwen_tts.service.client_request", side_effect=client_request):
            status = main(["synthesize", "--wav-stdout", "--require-installed-asset",
                           "--model-id", "qwen3-tts-0.6b-customvoice",
                           "--voice-id", "Vivian"])
        self.assertEqual(status, 0)
        self.assertEqual(stdout.buffer.getvalue(), b"RIFF-audio")
        self.assertEqual(json.loads(stderr.getvalue())["model_id"],
                         "qwen3-tts-0.6b-customvoice")
        self.assertEqual(captured[0]["args"]["model_id"],
                         "qwen3-tts-0.6b-customvoice")
        self.assertEqual(captured[0]["extensions"],
                         {"x_require_installed_asset_v1": True})


if __name__ == "__main__":
    unittest.main()
