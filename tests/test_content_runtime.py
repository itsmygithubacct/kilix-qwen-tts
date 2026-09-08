"""Installed-model execution through the real namespace and socket service."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from kilix_qwen_tts import runtime as runtime_module
from kilix_qwen_tts.runtime import InstalledRuntime, RUNTIME_SCHEMA, ENGINE_COMMIT
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.service import client_request, request_value
import test_content as content_fixtures
import test_runtime as runtime_fixtures


@unittest.skipUnless(content_fixtures.CONTENT_AVAILABLE and Path("/usr/bin/bwrap").exists(),
                     "reviewed content API and Linux namespace launcher required")
class ContentRuntimeTests(unittest.TestCase):
    def setup_runtime(self):
        fixture = runtime_fixtures.RuntimeTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = fixture.manifest
        names = ("config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt", "speech_tokenizer/config.json",
                 "speech_tokenizer/model.safetensors", "speech_tokenizer/preprocessor_config.json")
        payloads = {"model/data": b"model bytes"}
        payloads.update({"model/" + name: b"synthetic " + name.encode() for name in names})
        manifest.update(schema=RUNTIME_SCHEMA, engine_revision=ENGINE_COMMIT, device="cpu",
                        files={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()})
        manifest["environment"].update(python=str(fixture.runtime.python_root / "bin/python3.12"),
                                        python_root=str(fixture.runtime.python_root),
                                        site_packages=str(fixture.runtime.site_packages))
        (fixture.installation / "runtime.json").write_text(json.dumps(manifest))
        (fixture.installation / "model/data").unlink()
        (fixture.installation / "model").rmdir()
        installed = self.enterContext(content_fixtures.PackagedFixture(payloads,
            asset_id=manifest["model"]["id"], revision=manifest["model"]["revision"]))
        source = installed.source()
        fixture.runtime = InstalledRuntime(fixture.installation, model_source=source)
        return fixture, installed, source

    def test_socket_worker_uses_installed_snapshot_after_path_replacement(self):
        fixture, installed, source = self.setup_runtime()
        original = source.open
        @contextmanager
        def replace_after_open(check):
            with original(check) as asset:
                (installed.selected / "model/data").write_bytes(b"later path replacement")
                yield asset
        source.open = replace_after_open
        fixture.start()
        baseline = len(os.listdir("/proc/self/fd"))
        spawn = runtime_module.subprocess.Popen
        observed = []
        def at_spawn(*args, **kwargs):
            targets = []
            for descriptor in Path("/proc/self/fd").iterdir():
                try:
                    targets.append(os.readlink(descriptor))
                except FileNotFoundError:
                    pass
            observed.append(sum("memfd:kilix-content-asset" in target for target in targets))
            return spawn(*args, **kwargs)
        with fixture.audio.open("rb") as audio, patch.object(runtime_module.subprocess, "Popen", side_effect=at_spawn):
            result, pcm = client_request(fixture.ipc, request_value("submit", job_id="installed",
                args=fixture.arguments(), timeout=5), audio.fileno())
        self.assertEqual(result["audio"]["sha256"], hashlib.sha256(pcm).hexdigest())
        self.assertEqual(observed, [0], "F100 model copies survived namespace spawn")
        self.assertFalse((fixture.installation / "model").exists())
        with fixture.audio.open("rb") as audio, self.assertRaises(ProtocolError):
            client_request(fixture.ipc, request_value("submit", job_id="changed",
                args=fixture.arguments(), timeout=5), audio.fileno())
        fixture.stop()
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_installed_engine_cancel_reaps_and_recovers(self):
        fixture, _installed, _source = self.setup_runtime()
        before_jobs = set(fixture.jobs.iterdir())
        fixture.start()
        errors = []
        def submit():
            try:
                with fixture.audio.open("rb") as audio:
                    client_request(fixture.ipc, request_value("submit", job_id="escaped",
                        args=fixture.arguments("escape"), timeout=5), audio.fileno())
            except ProtocolError as error:
                errors.append(error.code)
        thread = threading.Thread(target=submit)
        thread.start()
        try:
            deadline = time.monotonic() + 3
            while not list(fixture.jobs.glob("*/ready")) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(list(fixture.jobs.glob("*/ready")))
            client_request(fixture.ipc, request_value("cancel", job_id="escaped"))
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["CANCELED"])
            self.assertEqual(set(fixture.jobs.iterdir()), before_jobs)
            status = client_request(fixture.ipc, request_value("status"))
            self.assertEqual(status["provider_state"], "ready")
        finally:
            fixture.stop()
            thread.join(3)

    def test_bad_runtime_digest_refuses_before_namespace_spawn(self):
        fixture, _installed, source = self.setup_runtime()
        value = json.loads((fixture.installation / "runtime.json").read_text())
        value["files"]["model/data"] = "0" * 64
        (fixture.installation / "runtime.json").write_text(json.dumps(value))
        with patch.object(runtime_module.subprocess, "Popen", side_effect=AssertionError("must not spawn")):
            with self.assertRaises(ProtocolError):
                InstalledRuntime(fixture.installation, model_source=source)

    def test_spawn_failure_releases_every_content_and_bundle_descriptor(self):
        fixture, _installed, _source = self.setup_runtime()
        before_jobs = set(fixture.jobs.iterdir())
        baseline = len(os.listdir("/proc/self/fd"))
        with fixture.audio.open("rb") as audio:
            with patch.object(runtime_module.subprocess, "Popen", side_effect=OSError("synthetic spawn failure")):
                with self.assertRaises(OSError):
                    runtime_module.run_job(fixture.runtime, audio.fileno(), fixture.arguments(),
                        deadline=time.monotonic() + 3, cancel=threading.Event())
        self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)
        self.assertEqual(set(fixture.jobs.iterdir()), before_jobs)
