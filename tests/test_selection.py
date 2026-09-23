"""One socket selects each capability while retaining one owned worker slot."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import MODEL_CANDIDATES
from kilix_qwen_tts.selection import installed_runtimes
from kilix_qwen_tts.service import Service, client_request, request_value
import test_runtime as fixtures


@unittest.skipUnless(Path("/usr/bin/bwrap").exists(), "Linux namespace launcher required")
class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RuntimeTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.models = []
        for model_id in ("qwen3-tts-0.6b-base", "qwen3-tts-0.6b-customvoice", "qwen3-tts-1.7b-voicedesign"):
            model = SimpleNamespace(**vars(self.fixture.runtime))
            revision, mode = MODEL_CANDIDATES[model_id]
            model.model_id, model.model_revision, model.mode = model_id, revision, mode
            model.manifest = copy.deepcopy(model.manifest)
            model.manifest["model"] = {"id": model_id, "revision": revision}
            model.model_record = lambda selected=model: {"id": selected.model_id, "revision": selected.model_revision,
                                                         "capabilities": [selected.mode], "release_qualified": False}
            self.models.append(model)
        self.fixture.service = Service(self.models[0], self.fixture.ipc, additional_runtimes=self.models[1:])

    def start(self):
        self.fixture.thread = threading.Thread(target=self.fixture.service.serve)
        self.fixture.thread.start()
        self.assertTrue(self.fixture.service.ready.wait(3))

    def args(self, mode, text="speech"):
        args = self.fixture.arguments(text)
        args["mode"] = mode
        if mode != "prompt_clone":
            for key in ("prompt_fd", "prompt_audio", "consent"):
                del args[key]
        if mode == "named_voice":
            args["voice_id"] = "Vivian"
        if mode == "voice_design":
            args["description"] = "A clear calm voice."
        return args

    def submit(self, mode, **changes):
        args = self.args(mode)
        args.update(changes)
        with self.fixture.audio.open("rb") as audio:
            return client_request(self.fixture.ipc, request_value("submit", job_id="selected",
                args=args, timeout=5), audio.fileno() if mode == "prompt_clone" else None)

    def test_real_socket_all_three_modes_bind_selected_output(self):
        self.start()
        records = client_request(self.fixture.ipc, request_value("models"))["models"]
        self.assertEqual([row["id"] for row in records], [m.model_id for m in self.models])
        for model in self.models:
            result, audio = self.submit(model.mode)
            self.assertEqual(result["model_id"], model.model_id)
            self.assertEqual(result["model_revision"], model.model_revision)
            self.assertTrue(audio.startswith(b"RIFF"))
            self.assertEqual("conditioning" in result, model.mode == "prompt_clone")

    def test_explicit_mismatch_and_unsupported_instruction_do_not_fallback(self):
        self.start()
        for mode, changes in (
                ("named_voice", {"model_id": self.models[0].model_id}),
                ("voice_design", {"model_id": "qwen3-tts-1.7b-base"}),
                ("named_voice", {"instruction": "Speak softly."})):
            with self.subTest(mode=mode, changes=changes), self.assertRaises(ProtocolError) as caught:
                self.submit(mode, **changes)
            self.assertEqual(caught.exception.code, "UNSUPPORTED_CAPABILITY")
        result, _ = self.submit("voice_design", instruction="Speak softly.")
        self.assertEqual(result["model_id"], self.models[2].model_id)

    def test_receipt_required_submit_refuses_local_stage_before_job(self):
        self.start()
        args = self.args("named_voice")
        with self.assertRaises(ProtocolError) as caught:
            client_request(self.fixture.ipc, request_value(
                "submit", job_id="receipt-required", args=args, timeout=5,
                require_installed_asset=True))
        self.assertEqual(caught.exception.code, "UNSUPPORTED_CAPABILITY")
        self.assertFalse(self.fixture.service._jobs)
        self.models[1].model_source = object()
        self.assertIs(self.fixture.service.select_runtime(
            args, require_installed_asset=True), self.models[1])

    def test_receipt_requirement_is_only_for_submit(self):
        with self.assertRaises(ProtocolError):
            request_value("models", require_installed_asset=True)
        request = request_value("submit", job_id="receipt", args=self.args("named_voice"),
                                require_installed_asset=True)
        self.assertEqual(request["extensions"], {"x_require_installed_asset_v1": True})

    def test_busy_and_cancel_follow_the_active_selected_model(self):
        self.start()
        errors = []
        def job():
            try:
                self.submit("voice_design", text="escape")
            except ProtocolError as error:
                errors.append(error.code)
        thread = threading.Thread(target=job)
        thread.start()
        try:
            deadline = time.monotonic() + 3
            while not list(self.fixture.jobs.glob("*/ready")) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(list(self.fixture.jobs.glob("*/ready")))
            status = client_request(self.fixture.ipc, request_value("status"))
            self.assertEqual(status["model_id"], self.models[2].model_id)
            self.assertEqual(status["provider_state"], "busy")
            with self.assertRaises(ProtocolError) as caught:
                self.submit("named_voice")
            self.assertEqual(caught.exception.code, "BUSY")
            client_request(self.fixture.ipc, request_value("cancel", job_id="selected"))
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["CANCELED"])
            result, _ = self.submit("named_voice")
            self.assertEqual(result["model_id"], self.models[1].model_id)
        finally:
            self.fixture.stop()
            thread.join(3)

    def test_duplicate_or_malformed_runtime_population_refuses(self):
        for additional in ([self.models[0]], self.models * 2, [SimpleNamespace(model_id=[])],
                           [SimpleNamespace(model_id=self.models[1].model_id, model_revision="wrong", mode="named_voice")]):
            with self.subTest(additional=additional), self.assertRaises(ProtocolError):
                Service(self.models[0], self.fixture.ipc, additional_runtimes=additional)

    def test_index_validation_precedes_any_model_or_receipt_open(self):
        path = self.fixture.root / "index.json"
        valid = {"schema": "kilix.qwen-tts.runtime-set/v1", "runtimes": [
            {"root": str(self.fixture.installation), "asset_id": self.models[0].model_id, "snapshot_bytes": 1000}]}
        rows = []
        for changes in ({"asset_id": []}, {"snapshot_bytes": True}, {"root": "relative"},
                        {"root": "/invalid/../runtime"}, {"root": "/invalid//runtime"},
                        {"root": "/invalid/\x00runtime"},
                        {"snapshot_bytes": 0}, {"extra": "unknown"}, {"asset_id": "unknown"}):
            value = copy.deepcopy(valid)
            value["runtimes"][0].update(changes)
            rows.append(value)
        rows.extend(({"schema": valid["schema"], "runtimes": []},
                     {"schema": valid["schema"], "runtimes": valid["runtimes"] * 2},
                     {"schema": valid["schema"], "runtimes": valid["runtimes"] * 6}))
        with patch("kilix_qwen_tts.selection.InstalledModel", side_effect=AssertionError("must not open")):
            for row in rows:
                path.write_text(json.dumps(row))
                with self.subTest(row=row), self.assertRaises(ProtocolError):
                    with installed_runtimes(path, self.fixture.root):
                        self.fail("invalid index accepted")

    def test_index_fifo_symlink_and_oversize_refuse_without_opening_models(self):
        import os
        path = self.fixture.root / "special-index"
        baseline = len(os.listdir("/proc/self/fd"))
        with patch("kilix_qwen_tts.selection.InstalledModel", side_effect=AssertionError("must not open")):
            os.mkfifo(path, 0o600)
            with self.assertRaises(ProtocolError):
                with installed_runtimes(path, self.fixture.root):
                    self.fail("FIFO accepted")
            path.unlink()
            path.symlink_to(self.fixture.audio)
            with self.assertRaises(OSError):
                with installed_runtimes(path, self.fixture.root):
                    self.fail("symlink accepted")
            path.unlink()
            path.write_bytes(b"x" * 65_537)
            with self.assertRaises(ProtocolError):
                with installed_runtimes(path, self.fixture.root):
                    self.fail("oversize index accepted")
        self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_index_closes_all_preceding_sources_on_later_failure(self):
        path = self.fixture.root / "index.json"
        path.write_text(json.dumps({"schema": "kilix.qwen-tts.runtime-set/v1", "runtimes": [
            {"root": str(self.fixture.installation), "asset_id": model.model_id, "snapshot_bytes": 1000}
            for model in self.models]}))
        opened, closed = [], []
        @contextmanager
        def source(asset_id, *_args, **_kwargs):
            opened.append(asset_id)
            try:
                yield asset_id
            finally:
                closed.append(asset_id)
        def runtime(_path, *, model_source):
            if model_source == self.models[2].model_id:
                raise ProtocolError("INVALID_RUNTIME", "synthetic later startup failure")
            return model_source
        with patch("kilix_qwen_tts.selection.InstalledModel", side_effect=source), \
             patch("kilix_qwen_tts.selection.InstalledRuntime", side_effect=runtime):
            with self.assertRaises(ProtocolError):
                with installed_runtimes(path, self.fixture.root):
                    self.fail("failed third runtime accepted")
        self.assertEqual(opened, [model.model_id for model in self.models])
        self.assertEqual(closed, list(reversed(opened)))
