"""Real receipt authority integration with explicitly synthetic packaged bytes.

The fixture changes only the packaged catalog and its compiled digest. It does
not replace the production context, receipt decisions, installer or snapshots.
Run with the reviewed kilix-content package available on PYTHONPATH.
"""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts.content import InstalledModel, from_options
from kilix_qwen_tts.protocol import ProtocolError

try:
    import kilix_content as content
    from kilix_content import receipt, installed
    CONTENT_AVAILABLE = hasattr(content.Installer, "open_asset")
except ImportError:
    CONTENT_AVAILABLE = False

PROVIDER = "kilix-qwen-tts"
CONSUMER = "kilix.qwen-tts.runtime"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class PackagedFixture:
    def __init__(self, payloads=None, *, asset_id="synthetic-model", revision="revision1"):
        self.payloads = dict(payloads or {"model.bin": b"synthetic model bytes"})
        self.payloads["notices/LICENSE.txt"] = b"Synthetic fixture license, no model grant.\n"
        self.asset_id, self.revision = asset_id, revision

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="speech-content-")))
            size = sum(map(len, self.payloads.values()))
            record = {
                "schema": "kilix.content.asset/v1", "id": self.asset_id,
                "label": "Synthetic speech model fixture", "provider": PROVIDER,
                "stream": "F104", "version": self.revision,
                "files": [{"path": name, "bytes": len(data), "sha256": digest(data)}
                          for name, data in sorted(self.payloads.items())],
                "licenses": [{"decision": "informational", "id": "fixture-license",
                              "text_sha256": digest(self.payloads["notices/LICENSE.txt"])}],
                "compatibility": {"consumer_schema": CONSUMER, "minimum": 1, "maximum": 1},
                "sizes": {"download_bytes": 10, "installed_bytes": size, "temporary_bytes": size + 10},
                "source": {"mode": "mirrored", "archive_sha256": "3" * 64,
                           "mirrors": ["https://example.invalid/synthetic.tar"],
                           "provenance": {"original_url": "https://example.invalid/model",
                                          "project": "Synthetic test fixture", "revision": self.revision}},
            }
            self.spec = content.AssetSpec.from_mapping(record)
            original = receipt._packaged_bytes
            document = json.loads(original(receipt._CATALOG_RESOURCE, "test catalog"))
            document["assets"] = [self.spec.to_mapping()]
            self.catalog = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            self.stack.enter_context(patch.object(receipt, "_packaged_bytes", lambda name, label:
                self.catalog if name == receipt._CATALOG_RESOURCE else original(name, label)))
            self.stack.enter_context(patch.object(receipt, "_CATALOG_SHA256", digest(self.catalog)))
            self.stack.enter_context(patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "state")}))
            self.release = content.ReleaseContext.packaged()
            self.store = self.stack.enter_context(content.ReceiptStore.open_default())
            requirement = self.spec.licenses[0]
            decision = content.LicenseDecision.from_mapping({
                "schema": "kilix.install.license/v1", "kind": "decision",
                "artifact_ids": [self.asset_id], "decision_class": "informational",
                "license_id": requirement.license_id, "license_text_sha256": requirement.text_sha256,
                "outcome": "record", "presenter": "kilix-installer", "release": self.release.release_id,
            })
            self.store.record(decision, self.payloads["notices/LICENSE.txt"], self.release, [self.spec])
            self.data = self.root / "content"
            self.installer = content.Installer(str(self.data))
            self.selected = Path(self.installer.asset_destination(self.spec))
            self.selected.mkdir(mode=0o700, parents=True)
            for name, payload in self.payloads.items():
                path = self.selected / name
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                path.write_bytes(payload)
                path.chmod(0o600)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def source(self, **kwargs):
        options = dict(maximum_bytes=self.spec.installed_bytes, provider=PROVIDER, consumer_schema=CONSUMER)
        options.update(kwargs)
        source = self.stack.enter_context(InstalledModel(self.asset_id, self.data, **options))
        source.bind(self.asset_id, self.revision,
                    {name: digest(data) for name, data in self.payloads.items()
                     if not name.startswith("notices/")})
        return source


@unittest.skipUnless(CONTENT_AVAILABLE, "reviewed kilix-content installed API is required")
class ContentTests(unittest.TestCase):
    def test_immutable_exact_bytes_survive_paths_and_owner_close(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            baseline = len(os.listdir("/proc/self/fd"))
            with source.open(lambda: None) as asset:
                descriptor = source.descriptor(asset, "model.bin", lambda: None, verify_bytes=True)
                (fixture.selected / "model.bin").write_bytes(b"later replacement")
            try:
                self.assertEqual(os.pread(descriptor, 1000, 0), fixture.payloads["model.bin"])
                self.assertFalse(os.get_inheritable(descriptor))
                with self.assertRaises(OSError):
                    os.pwrite(descriptor, b"mutation", 0)
            finally:
                os.close(descriptor)
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("changed installed population was accepted")
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_missing_receipts_and_forged_release_refuse_before_copy(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            baseline = len(os.listdir("/proc/self/fd"))
            source.release = content.ReleaseContext.from_catalog(fixture.release.release_id, fixture.catalog)
            with patch.object(installed, "_create_memfd", side_effect=AssertionError("must not copy")):
                with self.assertRaises(ProtocolError):
                    with source.open(lambda: None):
                        self.fail("synthetic context was accepted")
                source.release = fixture.release
                with patch.dict(os.environ, {"XDG_STATE_HOME": str(fixture.root / "empty-state")}):
                    missing = fixture.source()
                with self.assertRaises(ProtocolError):
                    with missing.open(lambda: None):
                        self.fail("missing receipt was accepted")
                missing.close()
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_consumer_revision_population_and_budget_refusals(self):
        with PackagedFixture() as fixture:
            for options in ({"provider": "wrong-provider"}, {"consumer_schema": "wrong-schema"},
                            {"maximum_bytes": 1}, {"maximum_bytes": True},
                            {"maximum_bytes": 10**1000}):
                with self.subTest(options=options), self.assertRaises(ProtocolError):
                    fixture.source(**options)
            source = fixture.source()
            files = {"model.bin": digest(fixture.payloads["model.bin"])}
            for asset_id, revision, population in (("different", fixture.revision, files),
                    (fixture.asset_id, "different", files), (fixture.asset_id, fixture.revision, {}),
                    (fixture.asset_id, fixture.revision, {"model.bin": "0" * 64})):
                with self.subTest(asset_id=asset_id, revision=revision), self.assertRaises(ProtocolError):
                    source.bind(asset_id, revision, population)
            baseline = len(os.listdir("/proc/self/fd"))
            (fixture.selected / "unexpected").write_bytes(b"undeclared bytes")
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("extra file was accepted")
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_descriptor_metadata_seals_and_actual_digest_refuse(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            baseline = len(os.listdir("/proc/self/fd"))
            with source.open(lambda: None) as asset:
                for sealed, payload in ((False, fixture.payloads["model.bin"]),
                                        (True, b"x" * len(fixture.payloads["model.bin"]))):
                    writer = installed._create_memfd()
                    try:
                        os.fchmod(writer, 0o600)
                        os.write(writer, payload)
                        if sealed:
                            fcntl.fcntl(writer, 1033, 15)
                        def duplicate(_asset, _name):
                            return os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
                        with patch.object(content.InstalledAsset, "duplicate", duplicate):
                            with self.assertRaises(ProtocolError):
                                source.descriptor(asset, "model.bin", lambda: None, verify_bytes=True)
                    finally:
                        os.close(writer)
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_deadline_and_cancel_escape_held_receipt_lock(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            descriptor = os.open(f"/proc/self/fd/{fixture.store._lock_descriptor}", os.O_RDWR | os.O_CLOEXEC)
            baseline = len(os.listdir("/proc/self/fd"))
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                for code in ("DEADLINE_EXCEEDED", "CANCELED"):
                    start = time.monotonic()
                    errors = []
                    def check():
                        if time.monotonic() - start >= .025:
                            raise ProtocolError(code, "bounded test ending")
                    def operation():
                        try:
                            with source.open(check):
                                errors.append("accepted")
                        except ProtocolError as error:
                            errors.append(error.code)
                    thread = threading.Thread(target=operation)
                    thread.start()
                    thread.join(1)
                    self.assertFalse(thread.is_alive(), "operation waited for lock release")
                    self.assertEqual(errors, [code])
                self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_mid_copy_cancel_cleans_all_descriptors(self):
        with PackagedFixture({"model.bin": b"z" * (2 * 1024**2)}) as fixture:
            source = fixture.source()
            baseline = len(os.listdir("/proc/self/fd"))
            canceled = False
            original = os.read
            def read(descriptor, size):
                nonlocal canceled
                result = original(descriptor, size)
                if len(result) == 1024**2:
                    canceled = True
                return result
            def check():
                if canceled:
                    raise ProtocolError("CANCELED", "test canceled during snapshot")
            with patch.object(installed.os, "read", side_effect=read):
                with self.assertRaises(ProtocolError) as caught:
                    with source.open(check):
                        self.fail("canceled snapshot accepted")
            self.assertEqual(caught.exception.code, "CANCELED")
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_partial_cli_selection_and_empty_production_catalog(self):
        for values in (("asset", None, None), (None, Path("/unused"), 8), ("asset", Path("/unused"), None)):
            options = SimpleNamespace(installed_asset=values[0], content_root=values[1],
                                      model_snapshot_bytes=values[2], runtime_root=Path("/unused"))
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                from_options(options, provider=PROVIDER, consumer_schema=CONSUMER)
        # Real packaged population is currently empty. Never substitute a
        # development fixture or fall back to pathname models on refusal.
        if not content.verified_packaged_catalog().assets:
            with self.assertRaises(ProtocolError):
                InstalledModel("synthetic-model", Path("/unused"), maximum_bytes=1024,
                               provider=PROVIDER, consumer_schema=CONSUMER)

    def test_cli_startup_failure_closes_owned_receipt_store(self):
        from kilix_qwen_tts.cli import main
        with PackagedFixture() as fixture:
            before = len(os.listdir("/proc/self/fd"))
            missing_runtime = fixture.root / "missing-runtime"
            with patch("sys.stderr"):
                result = main(["serve", "--runtime-root", str(missing_runtime),
                               "--installed-asset", fixture.asset_id,
                               "--content-root", str(fixture.data),
                               "--model-snapshot-bytes", str(fixture.spec.installed_bytes)])
            self.assertEqual(result, 69)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
