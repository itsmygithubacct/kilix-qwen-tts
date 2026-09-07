"""Launch Qwen from one sealed byte bundle in private Linux namespaces."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from .protocol import ProtocolError

BWRAP = "/usr/bin/bwrap"
MAX_FILES = 40_000
MAX_TREE_BYTES = 4 * 1024**3
MAX_BUNDLE_BYTES = 7 * 1024**3
BUNDLE_MAGIC = b"KQRT\x01"
RECORD = struct.Struct("!HQB")


def memory_file():
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(b"kilix-qwen-runtime", 3)
    if fd < 0:
        raise ProtocolError("INVALID_RUNTIME", "sealed runtime snapshots are unavailable")
    return fd


def _write(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise ProtocolError("INVALID_RUNTIME", "snapshot write stalled")
        view = view[written:]


def seal(fd):
    fcntl.fcntl(fd, 1033, 15)
    os.lseek(fd, 0, os.SEEK_SET)


def _copy(source: Path, destination_fd, check, header=lambda _info: None):
    digest = hashlib.sha256()
    source_fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(source_fd, "rb") as reader:
        info = os.fstat(reader.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022 or info.st_size > MAX_TREE_BYTES):
            raise ProtocolError("INVALID_RUNTIME", "unsafe runtime snapshot source")
        header(info)
        remaining = info.st_size
        while remaining:
            check()
            block = reader.read(min(remaining, 1024 * 1024))
            if not block:
                raise ProtocolError("INVALID_RUNTIME", "runtime snapshot source ended early")
            digest.update(block)
            _write(destination_fd, block)
            remaining -= len(block)
        if os.fstat(reader.fileno()).st_size != info.st_size:
            raise ProtocolError("INVALID_RUNTIME", "runtime snapshot source size changed")
    return digest.hexdigest(), bool(info.st_mode & 0o111)


def snapshot(source: Path, check):
    fd = memory_file()
    try:
        sha, executable = _copy(source, fd, check)
        seal(fd)
        return fd, sha, executable
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def launch(runtime, workspace: str, audio_fd: int | None, check):
    """Verify actual copied bytes, then unpack only that immutable population."""
    # System launcher/interpreter/libraries are root-owned inputs. No caller-
    # writable Python, package, model or prompt path is exposed to the worker.
    for executable in (BWRAP, "/usr/bin/python3"):
        info = Path(executable).stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ProtocolError("INVALID_RUNTIME", "unsafe namespace bootstrap")
    descriptors = []
    total_bytes = 0
    names = set()
    try:
        bundle = memory_file()
        descriptors.append(bundle)
        _write(bundle, BUNDLE_MAGIC)
        def add(source, destination):
            nonlocal total_bytes
            relative = Path(destination)
            encoded = destination.encode("utf-8")
            if (relative.is_absolute() or str(relative) != destination or ".." in relative.parts
                    or not encoded or len(encoded) > 4096 or destination in names
                    or len(names) >= MAX_FILES):
                raise ProtocolError("INVALID_RUNTIME", "unsafe runtime bundle path")
            names.add(destination)
            def header(info):
                nonlocal total_bytes
                total_bytes += info.st_size
                if total_bytes > MAX_BUNDLE_BYTES:
                    raise ProtocolError("LIMIT_EXCEEDED", "runtime snapshot exceeds its bound")
                _write(bundle, RECORD.pack(len(encoded), info.st_size, bool(info.st_mode & 0o111)) + encoded)
            return _copy(source, bundle, check, header)[0]
        def tree(root, prefix, expected, allow_links=False):
            digest = hashlib.sha256()
            for path in sorted(root.rglob("*")):
                check()
                if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                if path.is_symlink() and not (allow_links and path.is_file()):
                    raise ProtocolError("INVALID_RUNTIME", "unsafe runtime tree link")
                if path.is_dir():
                    continue
                relative = str(path.relative_to(root))
                sha = add(path, str(Path(prefix) / relative))
                if prefix == "python" and relative == "bin/python3.12" and sha != runtime.manifest["environment"]["python_sha256"]:
                    raise ProtocolError("INVALID_RUNTIME", "interpreter snapshot digest mismatch")
                digest.update(relative.encode() + b"\0" + bytes.fromhex(sha))
            if digest.hexdigest() != expected:
                raise ProtocolError("INVALID_RUNTIME", "runtime tree snapshot digest mismatch")
        environment = runtime.manifest["environment"]
        tree(runtime.python_root, "python", environment["python_root_sha256"], True)
        tree(runtime.site_packages, "python/lib/python3.12/site-packages", environment["site_packages_sha256"])
        for name, expected in runtime.manifest["files"].items():
            if add(runtime.root / name, "runtime/" + name) != expected:
                raise ProtocolError("INVALID_RUNTIME", "model snapshot digest mismatch")
        for path in sorted(Path(__file__).parent.glob("*.py")):
            add(path, "provider/kilix_qwen_tts/" + path.name)
        if audio_fd is not None:
            add(Path(f"/proc/self/fd/{audio_fd}"), "prompt.pcm")
        _write(bundle, RECORD.pack(0, 0, 0))
        seal(bundle)
        bootstrap, _, _ = snapshot(Path(__file__).with_name("bootstrap.py"), check)
        descriptors.append(bootstrap)
        command = [BWRAP, "--unshare-all", "--die-with-parent", "--as-pid-1", "--new-session",
                   "--cap-add", "CAP_SYS_ADMIN", "--ro-bind", "/usr", "/usr",
                   "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/opt",
                   "--bind", workspace, "/job", "--chdir", "/job",
                   "--ro-bind-data", str(bootstrap), "/bootstrap.py", "--",
                   "/usr/bin/python3", "-I", "-B", "/bootstrap.py", str(bundle)]
        # A dedicated host-side subreaper owns bwrap and its whole PID tree.
        # Returning from cancel cannot race namespace teardown or adopt the
        # embedding application's unrelated children.
        configuration = memory_file()
        descriptors.append(configuration)
        _write(configuration, json.dumps({"command": command, "descriptors": [bundle, bootstrap]},
                                        separators=(",", ":")).encode())
        seal(configuration)
        supervisor, _, _ = snapshot(Path(__file__).with_name("supervisor.py"), check)
        descriptors.append(supervisor)
        command = ["/usr/bin/python3", "-I", "-B", f"/proc/self/fd/{supervisor}",
                   "--launch-fd", str(configuration)]
        check()
        yield command, tuple(descriptors)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
