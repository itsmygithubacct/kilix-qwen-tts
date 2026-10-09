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
# A CUDA environment carries about 4.5 GiB of CUDA libraries. Its bundle and
# file-count bounds are separate; the bootstrap applies the same profile.
PROFILES = {"cpu": (MAX_BUNDLE_BYTES, MAX_FILES), "cuda": (14 * 1024**3, 60_000)}
# Only physical GPU 0 and the control/unified-memory nodes CUDA needs.
GPU_NODES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0")
# CUDA initialisation refuses (error 304) unless both kernel modules report
# their state; the worker sees these two read-only files and no other sysfs.
GPU_SYSFS = ("/sys/module/nvidia/initstate", "/sys/module/nvidia_uvm/initstate")
# The driver's user-space libraries are found below the read-only /usr bind.
# Distribution packaging (Debian's alternatives) may route them through
# /etc, which the namespace otherwise lacks: each such hop is bound read-only.
# libcuda is required; the PTX JIT compiler is bound when present.
DRIVER_LIBRARIES = ("/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                    "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1")
DRIVER_ROOT = "/usr/"
HOP_ROOT = "/etc/"
TRUSTED_UID = 0
BUNDLE_MAGIC = b"KQRT\x01"
RECORD = struct.Struct("!HQB")


def _character_device(path) -> bool:
    try:
        return stat.S_ISCHR(os.stat(path).st_mode)
    except OSError:
        return False


def _driver_hops(path):
    """The /etc symlink hops from a driver library to a root-owned /usr file."""
    hops = []
    for _ in range(8):
        try:
            info = os.lstat(path)
        except OSError:
            return None
        if stat.S_ISLNK(info.st_mode):
            target = os.path.normpath(os.path.join(os.path.dirname(path), os.readlink(path)))
            if not target.startswith((DRIVER_ROOT, HOP_ROOT)):
                return None
            if target.startswith(HOP_ROOT):
                hops.append(target)
            path = target
            continue
        if (stat.S_ISREG(info.st_mode) and path.startswith(DRIVER_ROOT)
                and info.st_uid == TRUSTED_UID and not info.st_mode & 0o022):
            return tuple(hops)
        return None
    return None


def driver_binds():
    """Read-only /etc hops the driver libraries need, or None without libcuda."""
    binds = []
    for index, library in enumerate(DRIVER_LIBRARIES):
        hops = _driver_hops(library)
        if hops is None:
            if index == 0:
                return None
            continue
        binds += [hop for hop in hops if hop not in binds]
    return tuple(binds)


def accelerator_nodes():
    """(host, sandbox) GPU 0 node pairs, only when every required node exists."""
    if (not all(_character_device(path) for path in GPU_NODES)
            or not all(os.path.isfile(path) for path in GPU_SYSFS)
            or driver_binds() is None):
        return ()
    return tuple((path, path) for path in GPU_NODES)


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
def launch(runtime, workspace: str, audio_fd: int | None, check, *, prompt_embeddings=None,
           gpu_nodes=()):
    """Verify actual copied bytes, then unpack only that immutable population."""
    profile = getattr(runtime, "device", "cpu")
    if profile not in PROFILES or type(gpu_nodes) is not tuple or (gpu_nodes and profile != "cuda"):
        raise ProtocolError("INVALID_RUNTIME", "unsupported runtime device profile")
    if gpu_nodes and (any(type(pair) is not tuple or len(pair) != 2 for pair in gpu_nodes)
                      or sorted(target for _source, target in gpu_nodes) != sorted(GPU_NODES)
                      or not all(type(source) is str and _character_device(source)
                                 for source, _target in gpu_nodes)):
        raise ProtocolError("INVALID_RUNTIME", "unsafe accelerator device nodes")
    bundle_limit, file_limit = PROFILES[profile]
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
                    or len(names) >= file_limit):
                raise ProtocolError("INVALID_RUNTIME", "unsafe runtime bundle path")
            names.add(destination)
            def header(info):
                nonlocal total_bytes
                total_bytes += info.st_size
                if total_bytes > bundle_limit:
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
        model_source = getattr(runtime, "model_source", None)
        if model_source is None:
            for name, expected in runtime.manifest["files"].items():
                if add(runtime.root / name, "runtime/" + name) != expected:
                    raise ProtocolError("INVALID_RUNTIME", "model snapshot digest mismatch")
        else:
            with model_source.open(check) as asset:
                for name, expected in runtime.manifest["files"].items():
                    descriptor = model_source.descriptor(asset, name, check)
                    try:
                        if add(Path(f"/proc/self/fd/{descriptor}"), "runtime/" + name) != expected:
                            raise ProtocolError("INVALID_RUNTIME", "installed model snapshot digest mismatch")
                    finally:
                        os.close(descriptor)
            # The bundle now owns the exact checked bytes. Release every F100
            # snapshot before process startup instead of retaining two models.
        for path in sorted(Path(__file__).parent.glob("*.py")):
            add(path, "provider/kilix_qwen_tts/" + path.name)
        if audio_fd is not None:
            add(Path(f"/proc/self/fd/{audio_fd}"), "prompt.pcm")
        if prompt_embeddings is not None:
            from .prompt_cache import validate_inputs
            for producer, payload in validate_inputs(prompt_embeddings, profile).items():
                name = f"prompt.{producer}.embedding"
                encoded = name.encode('ascii')
                total_bytes += len(payload)
                if total_bytes > bundle_limit or len(names) >= file_limit or name in names:
                    raise ProtocolError("LIMIT_EXCEEDED", "runtime snapshot exceeds its bound")
                names.add(name)
                _write(bundle, RECORD.pack(len(encoded), len(payload), False) + encoded + payload)
        _write(bundle, RECORD.pack(0, 0, 0))
        seal(bundle)
        bootstrap, _, _ = snapshot(Path(__file__).with_name("bootstrap.py"), check)
        descriptors.append(bootstrap)
        command = [BWRAP, "--unshare-all", "--die-with-parent", "--as-pid-1", "--new-session",
                   "--cap-add", "CAP_SYS_ADMIN", "--ro-bind", "/usr", "/usr",
                   "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64",
                   "--proc", "/proc", "--dev", "/dev"]
        # A GPU job sees only the checked NVIDIA nodes, at their host paths.
        for source, target in gpu_nodes:
            command += ["--dev-bind", source, target]
        if gpu_nodes:
            for path in GPU_SYSFS:
                command += ["--ro-bind-try", path, path]
            # Re-resolved here; a driver change since the offer only means
            # the worker cannot initialise CUDA and runs on the CPU.
            for hop in driver_binds() or ():
                command += ["--ro-bind-try", hop, hop]
        command += ["--tmpfs", "/tmp", "--dir", "/opt",
                    "--bind", workspace, "/job", "--chdir", "/job",
                    "--ro-bind-data", str(bootstrap), "/bootstrap.py", "--",
                    "/usr/bin/python3", "-I", "-B", "/bootstrap.py", str(bundle), profile]
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
        # The spawning parent releases its copies as soon as the supervisor
        # inherits them; keeping a multi-GiB bundle through inference would
        # retain an unnecessary second copy after private extraction.
        yield command, descriptors
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
