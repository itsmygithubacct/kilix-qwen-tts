"""Optional model acquisition from the packaged content receipt authority.

Only the service owner selects this source. No wire request can supply a
catalog, release identity, receipt, model path or descriptor.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import time

from .protocol import ProtocolError


_READ_BLOCK = 1024 * 1024
SNAPSHOT_SECONDS = 120
_SEALS = 15  # F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE


class _Snapshot:
    """Sealed read-only memory copies of one installed population."""

    def __init__(self):
        self.descriptors: dict[str, int] = {}

    def duplicate(self, name: str) -> int:
        return os.open(f"/proc/self/fd/{self.descriptors[name]}", os.O_RDONLY | os.O_CLOEXEC)

    def close(self):
        while self.descriptors:
            os.close(self.descriptors.popitem()[1])


class InstalledModel:
    def __init__(self, asset_id: str, root: Path, *, maximum_bytes: int,
                 provider: str, consumer_schema: str):
        if (type(asset_id) is not str or not asset_id or len(asset_id) > 128
                or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 8 * 1024**3):
            raise ProtocolError("INVALID_RUNTIME", "invalid installed model selection or budget")
        try:
            import kilix_content as content
            import kilix_license as license
        except ImportError as error:
            raise ProtocolError("INVALID_RUNTIME", "installed content API is unavailable") from error
        if (not hasattr(content, "verified_packaged_catalog")
                or not hasattr(content.Installer, "asset_destination")
                or not all(hasattr(license, name) for name in
                           ("AssetRef", "ReceiptStore", "RecordIndex", "load_determined_records", "require"))):
            raise ProtocolError("INVALID_RUNTIME", "installed content API is unavailable")
        self._errors = (content.CatalogError, content.InstallError, license.LicenseError,
                        RuntimeError, OSError, ValueError, KeyError)
        try:
            self.spec = content.verified_packaged_catalog().require_asset(asset_id)
            if (self.spec.provider != provider or self.spec.stream != "F104"
                    or self.spec.consumer_schema != consumer_schema
                    or not self.spec.compatibility_minimum <= 1 <= self.spec.compatibility_maximum
                    or self.spec.installed_bytes > maximum_bytes or len(self.spec.licenses) != 1):
                raise ProtocolError("INVALID_RUNTIME", "installed model consumer or budget mismatch")
            self.reference = license.AssetRef(
                id=self.spec.asset_id, record_digest=self.spec.licenses[0].record_digest,
                manifest_digest=self.spec.manifest_digest)
            self.records = license.RecordIndex(license.load_determined_records())
            self.records.by_digest(self.reference.record_digest)
            self.store = license.ReceiptStore.shared()
            self.root = Path(root)
            directory = Path(content.Installer(str(root)).asset_destination(self.spec))
            self.relative = directory.relative_to(self.root).parts
            if not self.relative or any(part in {"", ".", ".."} for part in self.relative):
                raise ProtocolError("INVALID_RUNTIME", "installed model location is unsafe")
        except ProtocolError:
            raise
        except self._errors as error:
            raise ProtocolError("INVALID_RUNTIME", "installed model authority is unavailable") from error
        self._license = license
        self.maximum_bytes = maximum_bytes
        self._members = {item.path: item for item in self.spec.files}
        self._bound = False

    def close(self):
        """The receipt store keeps no open state; nothing is retained."""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def bind(self, asset_id: str, revision: str, files: dict[str, str]) -> None:
        """Require the runtime manifest and catalog to describe identical models."""
        population = {name: item.sha256 for name, item in self._members.items()
                      if not name.startswith("notices/")}
        if (self.spec.asset_id != asset_id or self.spec.version != revision
                or self.spec.provenance_revision != revision or population != files):
            raise ProtocolError("INVALID_RUNTIME", "installed model and runtime identities differ")
        self._bound = True

    @contextmanager
    def open(self, check):
        """Receipts and the full installed population are rechecked for every job."""
        if not self._bound:
            raise ProtocolError("INVALID_RUNTIME", "installed model has no runtime binding")
        check()
        snapshot = _Snapshot()
        try:
            try:
                self._license.require(self.reference, records=self.records, store=self.store)
                check()
                self._copy_population(snapshot, check)
            except ProtocolError:
                raise
            except self._errors as error:
                # Restore a currently effective provider cancellation/deadline
                # instead of hiding it as an authorization or integrity failure.
                check()
                raise ProtocolError("INVALID_RUNTIME", "installed model authorization or bytes failed") from error
            check()
            yield snapshot
        finally:
            snapshot.close()

    def _copy_population(self, snapshot: _Snapshot, job_check) -> None:
        from .sandbox import memory_file
        ceiling = time.monotonic() + SNAPSHOT_SECONDS

        def check():
            job_check()
            if time.monotonic() >= ceiling:
                raise ProtocolError("INVALID_RUNTIME", "installed model snapshot exceeded its time ceiling")
        root = self._open_asset_directory()
        try:
            found = set()
            ancestors = {str(parent) for name in self._members for parent in Path(name).parents
                         if str(parent) != "."}

            def unreadable(error):
                raise ProtocolError("INVALID_RUNTIME", "installed model directory is unreadable") from error

            for directory, dirs, names, directory_fd in os.fwalk(".", dir_fd=root, follow_symlinks=False,
                                                                  onerror=unreadable):
                check()
                for name in dirs:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    self._check_directory(info)
                    if os.path.normpath(os.path.join(directory, name)) not in ancestors:
                        raise ProtocolError("INVALID_RUNTIME", "undeclared installed model directory")
                for name in names:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise ProtocolError("INVALID_RUNTIME", "unsafe installed model member")
                    found.add(os.path.normpath(os.path.join(directory, name)))
            if found != set(self._members):
                raise ProtocolError("INVALID_RUNTIME", "installed model population differs")
            total = 0
            for name, item in sorted(self._members.items()):
                total += item.bytes
                if total > self.maximum_bytes:
                    raise ProtocolError("INVALID_RUNTIME", "installed model exceeds its snapshot budget")
                source = self._open_member(root, name)
                try:
                    info = os.fstat(source)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                            or info.st_mode & 0o022 or info.st_nlink != 1 or info.st_size != item.bytes):
                        raise ProtocolError("INVALID_RUNTIME", "unsafe installed model member")
                    copy = memory_file()
                    snapshot.descriptors[name] = copy
                    digest = hashlib.sha256()
                    remaining = item.bytes
                    while remaining:
                        check()
                        block = os.read(source, min(_READ_BLOCK, remaining))
                        if not block:
                            raise ProtocolError("INVALID_RUNTIME", "installed model snapshot ended early")
                        digest.update(block)
                        view = memoryview(block)
                        while view:
                            view = view[os.write(copy, view):]
                        remaining -= len(block)
                    if os.read(source, 1) or digest.hexdigest() != item.sha256:
                        raise ProtocolError("INVALID_RUNTIME", "installed model snapshot digest differs")
                finally:
                    os.close(source)
                os.fchmod(copy, 0o400)
                fcntl.fcntl(copy, 1033, _SEALS)
                os.lseek(copy, 0, os.SEEK_SET)
        finally:
            os.close(root)

    def _open_asset_directory(self) -> int:
        """The operator chooses the content root; nothing beneath it is followed."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory = os.open(self.root, flags)
        try:
            self._check_directory(os.fstat(directory))
            for part in self.relative:
                child = os.open(part, flags | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
                self._check_directory(os.fstat(directory))
            return directory
        except BaseException:
            os.close(directory)
            raise

    @staticmethod
    def _check_directory(info) -> None:
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, os.geteuid()}
                or info.st_mode & 0o022):
            raise ProtocolError("INVALID_RUNTIME", "unsafe installed model directory")

    @staticmethod
    def _open_member(root: int, name: str) -> int:
        parts = Path(name).parts
        if not parts or any(part in {"", ".", ".."} for part in parts) or Path(name).is_absolute():
            raise ProtocolError("INVALID_RUNTIME", "unsafe installed model member")
        parent = os.dup(root)
        try:
            InstalledModel._check_directory(os.fstat(parent))
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=parent)
                os.close(parent)
                parent = child
                InstalledModel._check_directory(os.fstat(parent))
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                           dir_fd=parent)
        finally:
            os.close(parent)

    def descriptor(self, asset, name: str, check, *, verify_bytes: bool = False) -> int:
        """Return one owned immutable descriptor; callers close it on every path."""
        if name not in self._members:
            raise ProtocolError("INVALID_RUNTIME", "unrecorded installed model member")
        descriptor = asset.duplicate(name)
        try:
            check()
            info = os.fstat(descriptor)
            item = self._members[name]
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_size != item.bytes or info.st_mode & 0o133
                    or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                    or not fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
                    or fcntl.fcntl(descriptor, 1034) & 15 != 15):
                raise ProtocolError("INVALID_RUNTIME", "unsafe installed model descriptor")
            if verify_bytes:
                digest = hashlib.sha256()
                offset = 0
                while offset < item.bytes:
                    check()
                    block = os.pread(descriptor, min(1024 * 1024, item.bytes - offset), offset)
                    if not block:
                        raise ProtocolError("INVALID_RUNTIME", "installed model snapshot ended early")
                    digest.update(block)
                    offset += len(block)
                if os.pread(descriptor, 1, offset) or digest.hexdigest() != item.sha256:
                    raise ProtocolError("INVALID_RUNTIME", "installed model snapshot digest differs")
            check()
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise


def from_options(options, *, provider: str, consumer_schema: str):
    """Select only the production packaged authority, never a caller catalog."""
    values = (options.installed_asset, options.content_root, options.model_snapshot_bytes)
    if not any(value is not None for value in values):
        return None
    if options.runtime_root is None or any(value is None for value in values):
        raise ProtocolError("INVALID_REQUEST", "installed model options require runtime, asset, root and budget")
    return InstalledModel(options.installed_asset, options.content_root,
                          maximum_bytes=options.model_snapshot_bytes,
                          provider=provider, consumer_schema=consumer_schema)
