"""Real receipt authority integration with explicitly synthetic packaged bytes.

The fixture changes only the packaged catalog and its compiled digest, and
points the receipt store at a private directory. The receipt it relies on is
minted through kilix-license's own agreement path, and coverage, the installer
layout and the snapshots are the production implementations. Run with the
reviewed kilix-content package (and its kilix-license) on PYTHONPATH.
"""
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilix_qwen_tts import content as provider_content
from kilix_qwen_tts.content import InstalledModel, from_options
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.sandbox import memory_file

try:
    import kilix_content as content
    from kilix_content import receipt
    import kilix_license as license
    CONTENT_AVAILABLE = (hasattr(content, "verified_packaged_catalog")
                         and hasattr(content.Installer, "asset_destination")
                         and hasattr(license, "require"))
except ImportError:
    CONTENT_AVAILABLE = False

PROVIDER = "kilix-qwen-tts"
CONSUMER = "kilix.qwen-tts.runtime"
# The fixture reuses a real determined licence record so the receipt is minted
# by the real authority; only the asset it covers is synthetic.
TEMPLATE_ASSET = "qwen3-tts-0.6b-customvoice"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def open_descriptors():
    return len(os.listdir("/proc/self/fd"))


class PackagedFixture:
    def __init__(self, payloads=None, *, asset_id="synthetic-model", revision="revision1", receipt=True):
        self.payloads = dict(payloads or {"model/model.bin": b"synthetic model bytes"})
        self.asset_id, self.revision, self.with_receipt = asset_id, revision, receipt

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="speech-content-")))
            template = content.verified_packaged_catalog().require_asset(TEMPLATE_ASSET).to_mapping()
            size = sum(map(len, self.payloads.values()))
            record = copy.deepcopy(template)
            record.update(id=self.asset_id, label=self.asset_id, version=self.revision,
                          files=[{"path": name, "bytes": len(data), "sha256": digest(data)}
                                 for name, data in sorted(self.payloads.items())],
                          sizes={"download_bytes": size, "installed_bytes": size, "temporary_bytes": size})
            record["source"]["fetch"] = [{"path": name, "url": "https://fixture.invalid/" + name}
                                         for name in sorted(self.payloads)]
            record["source"]["provenance"] = {"original_url": "https://fixture.invalid/model",
                                              "project": "Synthetic test fixture", "revision": self.revision}
            self.spec = content.AssetSpec.from_mapping(record)
            original = receipt._resource_bytes
            document = json.loads(original(receipt._CATALOG_RESOURCE))
            document["assets"] = [self.spec.to_mapping()]
            self.catalog = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            self.stack.enter_context(patch.object(receipt, "_resource_bytes", lambda name:
                self.catalog if name == receipt._CATALOG_RESOURCE else original(name)))
            self.stack.enter_context(patch.object(receipt, "_CATALOG_SHA256", digest(self.catalog)))
            self.receipts = self.root / "receipts"
            self.stack.enter_context(patch.dict(os.environ, {"KILIX_LICENSE_RECEIPTS": str(self.receipts)}))
            if self.with_receipt:
                self.mint_receipt()
            self.data = self.root / "content"
            self.selected = Path(content.Installer(str(self.data)).asset_destination(self.spec))
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

    def mint_receipt(self):
        records = license.RecordIndex(license.load_determined_records())
        record = records.by_digest(self.spec.licenses[0].record_digest)
        typed = license.typed_agreement_line(record) if record.expected_decision == "accept" else None
        agreement = license.capture_agreement(record, typed)
        license.ReceiptStore.shared().write(license.receipt_from_agreement(
            record, agreement, manifest_digest=self.spec.manifest_digest,
            release_digest=receipt.release_digest(), catalogue_digest=digest(self.catalog)))

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def population(self):
        return {name: digest(data) for name, data in self.payloads.items() if not name.startswith("notices/")}

    def source(self, **kwargs):
        options = dict(maximum_bytes=self.spec.installed_bytes, provider=PROVIDER, consumer_schema=CONSUMER)
        options.update(kwargs)
        source = self.stack.enter_context(InstalledModel(self.asset_id, self.data, **options))
        source.bind(self.asset_id, self.revision, self.population())
        return source


@unittest.skipUnless(CONTENT_AVAILABLE, "reviewed kilix-content installed API is required")
class ContentTests(unittest.TestCase):
    def test_immutable_exact_bytes_survive_paths_and_owner_close(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            baseline = open_descriptors()
            with source.open(lambda: None) as asset:
                descriptor = source.descriptor(asset, "model/model.bin", lambda: None, verify_bytes=True)
                (fixture.selected / "model/model.bin").write_bytes(b"later replacement")
            try:
                self.assertEqual(os.pread(descriptor, 1000, 0), fixture.payloads["model/model.bin"])
                self.assertFalse(os.get_inheritable(descriptor))
                with self.assertRaises(OSError):
                    os.pwrite(descriptor, b"mutation", 0)
            finally:
                os.close(descriptor)
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("changed installed population was accepted")
            payload = fixture.payloads["model/model.bin"]
            (fixture.selected / "model/model.bin").write_bytes(bytes(len(payload)))
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("same-size replacement bytes were accepted")
            self.assertEqual(open_descriptors(), baseline)

    def test_missing_or_foreign_receipt_refuses_before_copy(self):
        with PackagedFixture(receipt=False) as fixture:
            missing = fixture.source()
            baseline = open_descriptors()
            with patch("kilix_qwen_tts.sandbox.memory_file", side_effect=AssertionError("must not copy")):
                with self.assertRaises(ProtocolError):
                    with missing.open(lambda: None):
                        self.fail("missing receipt was accepted")
            self.assertEqual(open_descriptors(), baseline)
        # A genuine receipt for another manifest, filed under this asset's
        # name, still covers nothing: coverage reads the receipt, not the path.
        with PackagedFixture(receipt=False) as fixture:
            records = license.RecordIndex(license.load_determined_records())
            record = records.by_digest(fixture.spec.licenses[0].record_digest)
            typed = license.typed_agreement_line(record) if record.expected_decision == "accept" else None
            store = license.ReceiptStore.shared()
            foreign = license.receipt_from_agreement(
                record, license.capture_agreement(record, typed), manifest_digest="0" * 64,
                release_digest=receipt.release_digest(), catalogue_digest=digest(fixture.catalog))
            written = store.write(foreign)
            written.rename(store.path_for(record.digest, fixture.spec.manifest_digest))
            changed = fixture.source()
            with self.assertRaises(ProtocolError):
                with changed.open(lambda: None):
                    self.fail("a receipt for a different manifest was accepted")

    def test_consumer_revision_population_and_budget_refusals(self):
        with PackagedFixture() as fixture:
            for options in ({"provider": "wrong-provider"}, {"consumer_schema": "wrong-schema"},
                            {"maximum_bytes": 1}, {"maximum_bytes": True},
                            {"maximum_bytes": 10**1000}):
                with self.subTest(options=options), self.assertRaises(ProtocolError):
                    fixture.source(**options)
            source = fixture.source()
            files = fixture.population()
            for asset_id, revision, population in (("different", fixture.revision, files),
                    (fixture.asset_id, "different", files), (fixture.asset_id, fixture.revision, {}),
                    (fixture.asset_id, fixture.revision, {"model/model.bin": "0" * 64})):
                with self.subTest(asset_id=asset_id, revision=revision), self.assertRaises(ProtocolError):
                    source.bind(asset_id, revision, population)
            baseline = open_descriptors()
            (fixture.selected / "unexpected").write_bytes(b"undeclared bytes")
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("extra file was accepted")
            self.assertEqual(open_descriptors(), baseline)

    def test_symlinked_member_or_directory_refuses(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            target = fixture.root / "elsewhere.bin"
            target.write_bytes(fixture.payloads["model/model.bin"])
            member = fixture.selected / "model/model.bin"
            member.unlink()
            member.symlink_to(target)
            baseline = open_descriptors()
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("symlinked member was accepted")
            member.unlink()
            directory = fixture.selected / "model"
            directory.rmdir()
            elsewhere = fixture.root / "elsewhere"
            (elsewhere).mkdir()
            (elsewhere / "model.bin").write_bytes(fixture.payloads["model/model.bin"])
            directory.symlink_to(elsewhere, target_is_directory=True)
            with self.assertRaises(ProtocolError):
                with source.open(lambda: None):
                    self.fail("symlinked directory was accepted")
            self.assertEqual(open_descriptors(), baseline)

    def test_descriptor_metadata_seals_and_actual_digest_refuse(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            payload = fixture.payloads["model/model.bin"]
            baseline = open_descriptors()
            with source.open(lambda: None) as asset:
                for sealed, data in ((False, payload), (True, b"x" * len(payload))):
                    writer = memory_file()
                    try:
                        os.write(writer, data)
                        os.fchmod(writer, 0o400)
                        if sealed:
                            fcntl.fcntl(writer, 1033, 15)
                        def duplicate(_name):
                            return os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
                        with patch.object(asset, "duplicate", duplicate):
                            with self.assertRaises(ProtocolError):
                                source.descriptor(asset, "model/model.bin", lambda: None, verify_bytes=True)
                    finally:
                        os.close(writer)
            self.assertEqual(open_descriptors(), baseline)

    def test_deadline_and_cancel_win_before_receipt_lookup(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            baseline = open_descriptors()
            for code in ("DEADLINE_EXCEEDED", "CANCELED"):
                def check():
                    raise ProtocolError(code, "bounded test ending")
                with patch.object(license, "require", side_effect=AssertionError("must not look up")):
                    with self.assertRaises(ProtocolError) as caught:
                        with source.open(check):
                            self.fail("canceled job was accepted")
                self.assertEqual(caught.exception.code, code)
            self.assertEqual(open_descriptors(), baseline)

    def test_mid_copy_cancel_cleans_all_descriptors(self):
        with PackagedFixture({"model/model.bin": b"z" * (2 * 1024**2)}) as fixture:
            source = fixture.source()
            baseline = open_descriptors()
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
            with patch.object(provider_content.os, "read", side_effect=read):
                with self.assertRaises(ProtocolError) as caught:
                    with source.open(check):
                        self.fail("canceled snapshot accepted")
            self.assertEqual(caught.exception.code, "CANCELED")
            self.assertEqual(open_descriptors(), baseline)

    def test_snapshot_time_ceiling_refuses_and_cleans_up(self):
        with PackagedFixture({"model/model.bin": b"z" * (2 * 1024**2)}) as fixture:
            source = fixture.source()
            baseline = open_descriptors()
            with patch.object(provider_content, "SNAPSHOT_SECONDS", 0):
                with self.assertRaises(ProtocolError) as caught:
                    with source.open(lambda: None):
                        self.fail("snapshot past its ceiling was accepted")
            self.assertEqual(caught.exception.code, "INVALID_RUNTIME")
            self.assertEqual(open_descriptors(), baseline)

    def refuses(self, source, message, check=lambda: None):
        baseline = open_descriptors()
        with self.assertRaises(ProtocolError) as caught:
            with source.open(check):
                self.fail(message)
        self.assertEqual(open_descriptors(), baseline)
        return caught.exception

    def test_undeclared_directories_links_and_modes_refuse(self):
        cases = {
            "extra empty directory": lambda f: (f.selected / "extra").mkdir(),
            "extra directory symlink": lambda f: (f.selected / "extra").symlink_to("/etc"),
            "unreadable subdirectory": lambda f: ((f.selected / "hidden").mkdir(),
                                                  (f.selected / "hidden/undeclared.bin").write_bytes(b"x"),
                                                  (f.selected / "hidden").chmod(0)),
            "shared-writable member directory": lambda f: (f.selected / "model").chmod(0o777),
            "shared-writable member": lambda f: (f.selected / "model/model.bin").chmod(0o666),
            "hard-linked member": lambda f: os.link(f.selected / "model/model.bin", f.root / "outside.bin"),
            "symlinked asset parent": lambda f: (
                (f.data / "assets").rename(f.root / "real-assets"),
                (f.data / "assets").symlink_to(f.root / "real-assets")),
        }
        for label, damage in cases.items():
            with self.subTest(label), PackagedFixture() as fixture:
                source = fixture.source()
                damage(fixture)
                try:
                    self.refuses(source, label + " was accepted")
                finally:
                    hidden = fixture.selected / "hidden"
                    if hidden.exists() and not hidden.is_symlink():
                        hidden.chmod(0o700)

    def test_receipt_is_required_again_for_every_job(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            with source.open(lambda: None):
                pass
            for path in fixture.receipts.iterdir():
                path.unlink()
            self.refuses(source, "a revoked receipt still covered the next job")

    def test_cancel_during_a_refusal_reports_the_cancel(self):
        with PackagedFixture(receipt=False) as fixture:
            source = fixture.source()
            calls = []
            def check():
                calls.append(1)
                if len(calls) > 1:
                    raise ProtocolError("CANCELED", "canceled while refusing")
            self.assertEqual(self.refuses(source, "missing receipt accepted", check).code, "CANCELED")

    def test_members_swapped_or_grown_during_the_copy_refuse(self):
        original_open = InstalledModel._open_member
        original_fstat = os.fstat

        def swap_file(fixture):
            def open_member(root, name):
                member = fixture.selected / name
                copy_path = fixture.root / "swapped.bin"
                copy_path.write_bytes(member.read_bytes())
                member.unlink()
                member.symlink_to(copy_path)
                return original_open(root, name)
            return patch.object(InstalledModel, "_open_member", staticmethod(open_member))

        def swap_directory(fixture):
            def open_member(root, name):
                directory = fixture.selected / "model"
                moved = fixture.root / "moved-model"
                directory.rename(moved)
                directory.symlink_to(moved, target_is_directory=True)
                return original_open(root, name)
            return patch.object(InstalledModel, "_open_member", staticmethod(open_member))

        def grow(fixture):
            member = fixture.selected / "model/model.bin"
            identity = (member.stat().st_dev, member.stat().st_ino)
            def fstat(descriptor):
                info = original_fstat(descriptor)
                if (info.st_dev, info.st_ino) == identity:
                    # Only the member's own descriptor, after its size was taken.
                    with open(member, "ab") as handle:
                        handle.write(b"appended after fstat")
                return info
            return patch.object(provider_content.os, "fstat", fstat)

        for label, race in (("member swapped for a symlink", swap_file),
                            ("directory swapped for a symlink", swap_directory),
                            ("member grew after fstat", grow)):
            with self.subTest(label), PackagedFixture() as fixture:
                source = fixture.source()
                with race(fixture):
                    self.refuses(source, label + " was accepted")

    def test_time_ceiling_applies_inside_the_copy(self):
        with PackagedFixture({"model/model.bin": b"z" * (3 * 1024**2)}) as fixture:
            source = fixture.source()
            clock = [0.0]
            original_read = os.read
            def read(descriptor, size):
                clock[0] += provider_content.SNAPSHOT_SECONDS
                return original_read(descriptor, size)
            with patch.object(provider_content.time, "monotonic", lambda: clock[0]), \
                    patch.object(provider_content.os, "read", read):
                self.assertEqual(self.refuses(source, "copy past its ceiling accepted").code, "INVALID_RUNTIME")

    def test_inheritable_descriptor_refuses(self):
        with PackagedFixture() as fixture:
            source = fixture.source()
            with source.open(lambda: None) as asset:
                original = asset.duplicate
                def inheritable(name):
                    descriptor = original(name)
                    os.set_inheritable(descriptor, True)
                    return descriptor
                baseline = open_descriptors()
                with patch.object(asset, "duplicate", inheritable), self.assertRaises(ProtocolError):
                    source.descriptor(asset, "model/model.bin", lambda: None)
                self.assertEqual(open_descriptors(), baseline)

    def test_partial_cli_selection_and_unknown_production_asset(self):
        for values in (("asset", None, None), (None, Path("/unused"), 8), ("asset", Path("/unused"), None)):
            options = SimpleNamespace(installed_asset=values[0], content_root=values[1],
                                      model_snapshot_bytes=values[2], runtime_root=Path("/unused"))
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                from_options(options, provider=PROVIDER, consumer_schema=CONSUMER)
        # Never substitute a development fixture or fall back to pathname
        # models when the real packaged catalog does not name the asset.
        with self.assertRaises(ProtocolError):
            InstalledModel("synthetic-model", Path("/unused"), maximum_bytes=1024,
                           provider=PROVIDER, consumer_schema=CONSUMER)

    def test_cli_startup_failure_leaves_no_descriptors(self):
        from kilix_qwen_tts.cli import main
        with PackagedFixture() as fixture:
            before = open_descriptors()
            missing_runtime = fixture.root / "missing-runtime"
            with patch("sys.stderr"):
                result = main(["serve", "--runtime-root", str(missing_runtime),
                               "--installed-asset", fixture.asset_id,
                               "--content-root", str(fixture.data),
                               "--model-snapshot-bytes", str(fixture.spec.installed_bytes)])
            self.assertEqual(result, 69)
            self.assertEqual(open_descriptors(), before)
