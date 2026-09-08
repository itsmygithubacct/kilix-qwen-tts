#!/usr/bin/env python3
"""Create a new CPU environment from the committed lock and pinned engine."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


_STOPPING = False
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _stop(_signal, _frame):
    global _STOPPING
    _STOPPING = True


def _reap_owned():
    """Only the dedicated build process adopts and reaps build descendants."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        for value in children.read_text().split():
            try:
                os.kill(int(value), signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if reaped == 0:
                break
        time.sleep(.005)


class Destination:
    """Pin a safe no-follow directory chain, including its newly owned leaf."""
    def __init__(self, path: Path):
        self.path = path
        self.descriptors = []
        self.entries = []
        self.created = False
        if not path.is_absolute() or not path.name or ".." in path.parts:
            raise ValueError("destination must be an absolute canonical path")

    @staticmethod
    def safe(info, *, leaf=False):
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, os.geteuid()}
                or (info.st_mode & 0o022 and not (
                    not leaf and info.st_uid == 0 and info.st_mode & stat.S_ISVTX))
                or (leaf and (info.st_uid != os.geteuid() or info.st_mode & 0o077))):
            raise ValueError("destination directory chain is unsafe")

    def __enter__(self):
        try:
            descriptor = os.open("/", _DIRECTORY)
            self.descriptors.append(descriptor)
            self.safe(os.fstat(descriptor))
            for index, name in enumerate(self.path.parts[1:]):
                leaf = index == len(self.path.parts) - 2
                parent = descriptor
                if leaf:
                    os.mkdir(name, 0o700, dir_fd=parent)
                    self.created = True
                else:
                    try:
                        os.mkdir(name, 0o700, dir_fd=parent)
                    except FileExistsError:
                        pass
                descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
                self.descriptors.append(descriptor)
                info = os.fstat(descriptor)
                self.safe(info, leaf=leaf)
                self.entries.append((parent, name, descriptor, info.st_dev, info.st_ino, leaf))
                self.check()
            return self
        except BaseException:
            try:
                self.remove()
            finally:
                self.close()
            raise

    def check(self):
        if _STOPPING:
            raise InterruptedError("build interrupted")
        for parent, name, descriptor, device, inode, leaf in self.entries:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            pinned = os.fstat(descriptor)
            self.safe(current, leaf=leaf)
            if (current.st_dev, current.st_ino) != (device, inode) or (
                    pinned.st_dev, pinned.st_ino) != (device, inode):
                raise ValueError("destination directory identity changed")

    def remove(self):
        # Do not traverse a replaced textual ancestor or remove its substitute.
        # shutil's fd-based implementation also refuses symlink substitution.
        if self.created and self.entries and self.entries[-1][-1]:
            parent, name, _descriptor, device, inode, _leaf = self.entries[-1]
            try:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (device, inode):
                shutil.rmtree(name, dir_fd=parent)

    def close(self):
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()

    def __exit__(self, kind, _value, _traceback):
        try:
            if kind is not None:
                self.remove()
        finally:
            self.close()


def _run(command, environment, destination, deadline, *, capture=False):
    """Wait for a command and all its owned descendants before continuing."""
    destination.check()
    with tempfile.TemporaryFile(dir=f"/proc/self/fd/{destination.descriptors[-1]}") as output:
        process = None
        try:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=output if capture else None,
                                       start_new_session=True)
            while process.poll() is None:
                destination.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU environment build deadline exceeded")
                if capture and os.fstat(output.fileno()).st_size > 65_536:
                    raise ValueError("environment probe output exceeds its bound")
                time.sleep(.02)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            _reap_owned()
            if process is not None:
                process.wait()
        destination.check()
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU environment build deadline exceeded")
        if capture:
            output.seek(0)
            payload = output.read(65_537)
            if len(payload) > 65_536:
                raise ValueError("environment probe output exceeds its bound")
            return payload
    return None


def _build(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error("timeout must be positive and at most 3600 seconds")
    deadline = time.monotonic() + args.timeout
    root = Path(__file__).resolve().parents[1]
    inputs = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("pyproject.toml", "uv.lock", ".python-version",
                           "tools/build_environment.py")}
    destination = args.destination.absolute()
    # Reject before invocation; mkdirat below remains the authoritative race-
    # free existing-leaf refusal through the validated parent descriptor.
    if destination.exists() or destination.is_symlink():
        parser.error("destination must not already exist")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("UV_", "PIP_", "PYTHON", "GIT_"))
                   and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    environment.update(UV_PROJECT_ENVIRONMENT=str(destination),
                       UV_PYTHON="3.12.8", UV_PYTHON_DOWNLOADS="never",
                       PYTHONDONTWRITEBYTECODE="1")
    command = [str(args.uv.absolute()), "sync", "--project", str(root),
               "--locked", "--no-install-project", "--no-editable"]
    if args.cache_directory is not None:
        command += ["--cache-dir", str(args.cache_directory.absolute())]
    if args.offline:
        command.append("--offline")
    with Destination(destination) as stage:
        _run(command + ["--only-group", "build"], environment, stage, deadline)
        _run(command + ["--group", "cpu", "--group", "build", "--no-build-isolation"],
             environment, stage, deadline)
        probe = (
            "import importlib.metadata,json,sys; "
            "assert sys.version_info[:3] == (3,12,8); "
            "d={p.metadata['Name'].lower().replace('_','-'):p.version "
            "for p in importlib.metadata.distributions()}; "
            "assert d['torch']=='2.6.0+cpu' and d['torchaudio']=='2.6.0+cpu'; "
            "assert 'gradio' not in d and 'kilix-qwen-tts' not in d; "
            "print(json.dumps(d,sort_keys=True))"
        )
        packages = json.loads(_run([str(destination / "bin/python"), "-I", "-B", "-c", probe],
                                  environment, stage, deadline, capture=True))
        if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
               for name, digest in inputs.items()):
            raise RuntimeError("build inputs changed during environment creation")
        receipt = {"schema": "kilix.qwen-tts.environment-build/v1",
                   "inputs": inputs, "packages": packages,
                   "python_version": "3.12.8", "offline": args.offline,
                   "uv_sha256": hashlib.sha256(args.uv.read_bytes()).hexdigest()}
        stage.check()
        descriptor = os.open("kilix-environment-build.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                             dir_fd=stage.descriptors[-1])
        with os.fdopen(descriptor, "w") as record:
            record.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            record.flush()
            os.fsync(record.fileno())
        stage.check()
        print(json.dumps({"status": "built", "packages": packages}, sort_keys=True))
        return 0


def main() -> int:
    if sys.argv[1:2] == ["--owned-build"]:
        parent = os.getppid()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _stop)
        libc = ctypes.CDLL(None, use_errno=True)
        if (libc.prctl(36, 1, 0, 0, 0) != 0
                or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0
                or libc.prctl(38, 1, 0, 0, 0) != 0
                or os.getppid() != parent or parent == 1):
            return 125
        try:
            return _build(sys.argv[2:])
        finally:
            _reap_owned()
    # Reaping is confined to a dedicated child, never an importing/embedding
    # process. An interrupted controller waits for cleanup before returning.
    process = subprocess.Popen(["/usr/bin/python3", "-I", "-S", str(Path(__file__).absolute()),
                                "--owned-build", *sys.argv[1:]], start_new_session=True)
    def interrupt(_signal, _frame):
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        return process.wait()
    except BaseException:
        interrupt(None, None)
        process.wait()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
