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

from .protocol import ProtocolError


class InstalledModel:
    def __init__(self, asset_id: str, root: Path, *, maximum_bytes: int,
                 provider: str, consumer_schema: str):
        if (type(asset_id) is not str or not asset_id or len(asset_id) > 128
                or type(maximum_bytes) is not int or not 0 < maximum_bytes <= 8 * 1024**3):
            raise ProtocolError("INVALID_RUNTIME", "invalid installed model selection or budget")
        try:
            import kilix_content as content
            from kilix_content.receipt import ArtifactBinding
        except ImportError as error:
            raise ProtocolError("INVALID_RUNTIME", "installed content API is unavailable") from error
        if not hasattr(content.Installer, "open_asset"):
            raise ProtocolError("INVALID_RUNTIME", "installed content API is unavailable")
        self._errors = (content.CatalogError, content.ReceiptError, content.InstallError)
        try:
            self.spec = content.verified_packaged_catalog().require_asset(asset_id)
            self.release = content.ReleaseContext.packaged()
            if (self.spec.provider != provider or self.spec.stream != "F104"
                    or self.spec.consumer_schema != consumer_schema
                    or not self.spec.compatibility_minimum <= 1 <= self.spec.compatibility_maximum
                    or self.spec.installed_bytes > maximum_bytes):
                raise ProtocolError("INVALID_RUNTIME", "installed model consumer or budget mismatch")
            self.binding = ArtifactBinding.from_spec(self.spec)
            self.installer = content.Installer(str(root))
            self.store = content.ReceiptStore.open_default()
        except self._errors as error:
            raise ProtocolError("INVALID_RUNTIME", "installed model authority is unavailable") from error
        self._content = content
        self.maximum_bytes = maximum_bytes
        self._members = {item.path: item for item in self.spec.files}
        self._bound = False

    def close(self):
        self.store.close()

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
        """Full installed population and receipts are rechecked for every job."""
        if not self._bound:
            raise ProtocolError("INVALID_RUNTIME", "installed model has no runtime binding")
        check()
        def cancelled():
            # The provider's deadline/disconnect/cancel checker also runs during
            # both content receipt lock waits and each bounded member read.
            check()
            return False
        try:
            with self.installer.open_asset(
                    self.spec, self.store, self.release, maximum_bytes=self.maximum_bytes,
                    timeout=120, cancelled=cancelled) as asset:
                if (type(asset) is not self._content.InstalledAsset
                        or asset.binding != self.binding or asset.release != self.release
                        or asset.files != self.spec.files or not asset.receipts):
                    raise ProtocolError("INVALID_RUNTIME", "unbound installed model snapshot")
                check()
                yield asset
        except self._errors as error:
            # Content wraps I/O ValueErrors; restore a currently effective
            # provider cancellation/deadline instead of hiding it as integrity.
            check()
            raise ProtocolError("INVALID_RUNTIME", "installed model authorization or bytes failed") from error

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
