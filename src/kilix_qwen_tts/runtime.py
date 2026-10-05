"""An installed, digest-bound Qwen CPU or CUDA runtime and supervised jobs.

The installation manifest names files below the provider's own installation.
It is read only at service startup; job requests cannot choose files or tools.
It describes a local runtime, not a qualified release profile or an installer
licence receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import threading
import time
from typing import Callable

from .protocol import ProtocolError, decode_payload

RUNTIME_SCHEMA = "kilix.qwen-tts.runtime/v1"
ENGINE_COMMIT = "6cafe5582caea83df269c36b1ce62d953a9cc66b"
MAX_INPUT_BYTES = 11_520_000
MAX_RESULT_BYTES = 65_536
MAX_AUDIO_SECONDS = 900
# A "cuda" runtime has a CUDA torch environment. It is offered the GPU only
# when the host has the NVIDIA nodes, and otherwise runs on the CPU.
DEVICES = ("cpu", "cuda")


def digest_file(path: Path, check: Callable[[], None] = lambda: None, *,
                follow: bool = False) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
    if not follow:
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024**3:
            raise ProtocolError("INVALID_RUNTIME", "unsafe runtime file")
        remaining = before.st_size
        while remaining:
            check()
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise ProtocolError("INVALID_RUNTIME", "runtime file ended early")
            digest.update(block)
            remaining -= len(block)
        if os.fstat(source.fileno()).st_size != before.st_size:
            raise ProtocolError("INVALID_RUNTIME", "runtime file changed size")
    return digest.hexdigest()


def private_directory(path: Path, *, create: bool = False) -> Path:
    if not path.is_absolute():
        raise ProtocolError("INVALID_RUNTIME", "directory must be absolute")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o077):
        raise ProtocolError("INVALID_RUNTIME", "directory must be private and owned")
    return path


def tree_digest(root: Path, check: Callable[[], None] = lambda: None, *, allow_file_links=False) -> str:
    """Bind installed source/dependency files; bytecode is never snapshotted."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        check()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() and not (allow_file_links and path.is_file()):
            raise ProtocolError("INVALID_RUNTIME", "environment contains a symlink")
        if path.is_dir():
            continue
        info = path.stat() if allow_file_links else path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022):
            raise ProtocolError("INVALID_RUNTIME", "environment contains unsafe files")
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(bytes.fromhex(digest_file(path, check, follow=allow_file_links)))
    return digest.hexdigest()


MODEL_CANDIDATES = {
    "qwen3-tts-0.6b-base": ("5d83992436eae1d760afd27aff78a71d676296fc", "prompt_clone"),
    "qwen3-tts-0.6b-customvoice": ("85e237c12c027371202489a0ec509ded67b5e4b5", "named_voice"),
    "qwen3-tts-1.7b-base": ("fd4b254389122332181a7c3db7f27e918eec64e3", "prompt_clone"),
    "qwen3-tts-1.7b-customvoice": ("0c0e3051f131929182e2c023b9537f8b1c68adfe", "named_voice"),
    "qwen3-tts-1.7b-voicedesign": ("5ecdb67327fd37bb2e042aab12ff7391903235d3", "voice_design"),
}


class InstalledRuntime:
    def __init__(self, root: Path, *, model_source=None,
                 check: Callable[[], None] = lambda: None):
        check()
        self.root = private_directory(root)
        self.model_source = model_source
        path = root / "runtime.json"
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o022 or not 0 < info.st_size <= 65_536):
                raise ProtocolError("INVALID_RUNTIME", "unsafe runtime manifest")
            payload = source.read(info.st_size + 1)
            if len(payload) != info.st_size:
                raise ProtocolError("INVALID_RUNTIME", "runtime manifest changed size")
        value = decode_payload(payload)
        if (set(value) != {"schema", "engine_revision", "model", "files", "environment", "device"}
                or value["schema"] != RUNTIME_SCHEMA or value["engine_revision"] != ENGINE_COMMIT
                or type(value["device"]) is not str or value["device"] not in DEVICES):
            raise ProtocolError("INVALID_RUNTIME", "unsupported runtime identity or device")
        model = value["model"]
        if (type(model) is not dict or set(model) != {"id", "revision"}
                or type(model["id"]) is not str or model["id"] not in MODEL_CANDIDATES
                or model["revision"] != MODEL_CANDIDATES[model["id"]][0]):
            raise ProtocolError("INVALID_RUNTIME", "unsupported development model identity")
        files = value["files"]
        required = {"model/config.json", "model/generation_config.json", "model/model.safetensors",
                    "model/preprocessor_config.json", "model/tokenizer_config.json",
                    "model/vocab.json", "model/merges.txt", "model/speech_tokenizer/config.json",
                    "model/speech_tokenizer/model.safetensors", "model/speech_tokenizer/preprocessor_config.json"}
        if type(files) is not dict or not required <= set(files) or len(files) > 32:
            raise ProtocolError("INVALID_RUNTIME", "incomplete model file population")
        for name, expected in files.items():
            relative = Path(name)
            if (not relative.parts or str(relative) != name or relative.is_absolute()
                    or ".." in relative.parts or relative.parts[0] != "model"
                    or type(expected) is not str or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ProtocolError("INVALID_RUNTIME", "unsafe model file path")
            if model_source is not None:
                continue
            if any((root / Path(*relative.parts[:i])).is_symlink()
                   for i in range(1, len(relative.parts) + 1)):
                raise ProtocolError("INVALID_RUNTIME", "unsafe model file path")
            info = (root / name).lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o022 or type(expected) is not str
                    or digest_file(root / name, check) != expected):
                raise ProtocolError("INVALID_RUNTIME", "model file identity mismatch")
        if model_source is None:
            actual = {str(p.relative_to(root)) for p in (root / "model").rglob("*") if not p.is_dir()}
            if actual != set(files):
                raise ProtocolError("INVALID_RUNTIME", "unrecorded model file")
        else:
            model_source.bind(model["id"], model["revision"], files)
        environment = value["environment"]
        if (type(environment) is not dict
                or set(environment) != {"python", "python_sha256", "site_packages", "site_packages_sha256",
                                        "python_root", "python_root_sha256"}
                or any(type(v) is not str or not v for v in environment.values())):
            raise ProtocolError("INVALID_RUNTIME", "invalid environment identity")
        self.python = Path(environment["python"])
        self.site_packages = Path(environment["site_packages"])
        self.python_root = Path(environment["python_root"])
        if (not self.python.is_absolute() or not self.site_packages.is_absolute() or not self.python_root.is_absolute()
                or not os.access(self.python, os.X_OK)):
            raise ProtocolError("INVALID_RUNTIME", "invalid runtime interpreter")
        self.device = value["device"]
        self.last_device = None
        self.model_id = model["id"]
        self.model_revision = model["revision"]
        self.mode = MODEL_CANDIDATES[self.model_id][1]
        self.manifest = value
        self.verify_unchanged(check)

    def verify_unchanged(self, check: Callable[[], None] = lambda: None) -> None:
        if self.model_source is None:
            for name, digest in self.manifest["files"].items():
                if (self.root / name).is_symlink() or digest_file(self.root / name, check) != digest:
                    raise ProtocolError("INVALID_RUNTIME", "model changed; restart required")
        environment = self.manifest["environment"]
        if (digest_file(self.python, check, follow=True) != environment["python_sha256"]
                or tree_digest(self.site_packages, check) != environment["site_packages_sha256"]
                or tree_digest(self.python_root, check, allow_file_links=True) != environment["python_root_sha256"]):
            raise ProtocolError("INVALID_RUNTIME", "runtime environment changed; restart required")

    def model_record(self) -> dict:
        return {"id": self.model_id, "revision": self.model_revision,
                "engine_id": "qwen3-tts", "engine_revision": ENGINE_COMMIT,
                "installed": True, "release_qualified": False,
                "device": getattr(self, "device", "cpu"),
                "capabilities": [self.mode], "streaming": True,
                "asset_authority": ("kilix-content" if self.model_source is not None
                                    else "local-stage")}


def stop_process(process: subprocess.Popen) -> None:
    """Wait for the dedicated supervisor to reap every owned descendant."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise ProtocolError("SUPERVISOR_FAILED", "descendant cleanup did not complete") from error


def run_job(runtime: InstalledRuntime, audio_fd: int | None, args: dict, *,
            deadline: float, cancel: threading.Event,
            disconnected: Callable[[], bool] = lambda: False,
            execution_policy=None, job_id=None, progress=None, on_chunk=None,
            prompt_embedding=None, on_embedding=None) -> tuple[dict, bytes]:
    from .owned import OwnedExecution
    with OwnedExecution(execution_policy, job_id=job_id, workload="tts-utterance", deadline=deadline,
                        cancelled=cancel.is_set, disconnected=disconnected, progress=progress) as owner:
        return _run_owned_job(runtime, audio_fd, args, deadline=deadline, cancel=cancel,
                              disconnected=disconnected, owner=owner, on_chunk=on_chunk,
                              prompt_embedding=prompt_embedding, on_embedding=on_embedding)


def _run_owned_job(runtime, audio_fd, args, *, deadline, cancel, disconnected, owner, on_chunk=None,
                   prompt_embedding=None, on_embedding=None):
    if cancel.is_set() or disconnected():
        raise ProtocolError("CANCELED", "job canceled")
    if time.monotonic() >= deadline:
        raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
    def check():
        owner.check()
        if cancel.is_set() or disconnected():
            raise ProtocolError("CANCELED", "job canceled")
        if time.monotonic() >= deadline:
            raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
    check()
    if args["mode"] != runtime.mode or args["model_id"] not in {"auto", runtime.model_id}:
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "requested model capability is not installed")
    if args["output"] != {"sample_format": "s16le", "sample_rate_hz": 24000, "channels": 1}:
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "this runtime emits 24 kHz mono PCM16 WAV")
    if audio_fd is not None and not 0 < os.fstat(audio_fd).st_size <= MAX_INPUT_BYTES:
        raise ProtocolError("LIMIT_EXCEEDED", "prompt exceeds its bound")
    # File descriptors carry audio; the isolated worker's argv carries no text,
    # transcript, source filename, or model paths supplied by a client.
    from .sandbox import accelerator_nodes, launch
    profile = getattr(runtime, "device", "cpu")
    if profile not in DEVICES:
        raise ProtocolError("INVALID_RUNTIME", "unsupported runtime device")
    # The host, never a request, decides whether this job is offered the GPU.
    gpu_nodes = accelerator_nodes() if profile == "cuda" else ()
    offered = "cuda" if gpu_nodes else "cpu"
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
    if offered == "cuda":
        environment.update(CUDA_VISIBLE_DEVICES="0", CUDA_CACHE_DISABLE="1",
                           CUDA_MODULE_LOADING="LAZY")
    if (on_embedding is not None and (not callable(on_embedding) or args["mode"] != "prompt_clone")
            or prompt_embedding is not None and on_embedding is None):
        raise ProtocolError("INVALID_REQUEST", "invalid prompt cache selection")
    cache_options = {"prompt_embedding": prompt_embedding} if prompt_embedding is not None else {}
    if gpu_nodes:
        cache_options["gpu_nodes"] = gpu_nodes
    with tempfile.TemporaryDirectory(prefix="kilix-qwen-job-") as workspace, tempfile.TemporaryFile() as output, launch(runtime, workspace, audio_fd, check, **cache_options) as (command, descriptors):
        job = {"runtime": "/opt/runtime", "manifest": runtime.manifest,
               "audio_fd": None, "args": args, "workspace": "/job",
               "profile": profile, "device": offered}
        if on_embedding is not None:
            job.update(prompt_cache=True, prompt_cache_input=prompt_embedding is not None)
        stream = None
        if on_chunk is not None:
            from .streaming import WorkerStreamReader
            if not callable(on_chunk):
                raise ProtocolError("INVALID_REQUEST", "invalid PCM consumer")
            stream = WorkerStreamReader(output.fileno(), args["max_duration_ms"], on_chunk, check)
            job["streaming_pcm"] = True
        process = owner.spawn(
            command, stdin=subprocess.PIPE,
            stdout=output, stderr=subprocess.DEVNULL, env=environment,
            pass_fds=descriptors, start_new_session=True,
        )
        try:
            while descriptors:
                os.close(descriptors.pop())
            assert process.stdin is not None
            process.stdin.write(json.dumps(job, separators=(",", ":")).encode())
            process.stdin.close()
            while process.poll() is None:
                check()
                if stream is not None:
                    stream.drain()
                cancel.wait(0.025)
            if cancel.is_set() or disconnected():
                raise ProtocolError("CANCELED", "job canceled")
            if time.monotonic() >= deadline:
                raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
            if process.returncode != 0:
                raise ProtocolError("ENGINE_FAILED", "speech worker failed")
            if stream is not None:
                stream.drain(final=True)
                result = stream.result
            else:
                output.seek(0)
                payload = output.read(MAX_RESULT_BYTES + 1)
                if len(payload) > MAX_RESULT_BYTES:
                    raise ProtocolError("LIMIT_EXCEEDED", "transcript exceeds its bound")
                try:
                    result = decode_payload(payload)
                except (ProtocolError, ValueError, UnicodeDecodeError) as error:
                    raise ProtocolError("ENGINE_FAILED", "invalid worker result") from error
            required = {"engine_id", "engine_revision", "model_id", "model_revision", "duration_ms", "seed",
                        "audio", "device"}
            if args["mode"] == "prompt_clone":
                required.add("conditioning")
            # The worker reports where it ran; it may fall back to the CPU but
            # can never claim a GPU that this job was not offered.
            if (type(result) is not dict or set(result) != required
                    or type(result.get("device")) is not str or result["device"] not in {"cpu", offered}
                    or result.get("engine_revision") != ENGINE_COMMIT or result.get("engine_id") != "qwen3-tts"
                    or result.get("model_id") != runtime.model_id or result.get("model_revision") != runtime.model_revision
                    or type(result.get("seed")) is not int or result["seed"] != args["seed"]
                    or type(result.get("duration_ms")) is not int
                    or not 0 < result["duration_ms"] <= args["max_duration_ms"]
                    or type(result.get("audio")) is not dict or set(result["audio"]) != {"byte_length", "sha256"}):
                raise ProtocolError("ENGINE_FAILED", "unbound worker result")
            if args["mode"] == "prompt_clone":
                consent = json.dumps(args["consent"], separators=(",", ":"), sort_keys=True).encode()
                expected = {"prompt_sha256": args["prompt_audio"]["sha256"],
                            "consent_sha256": hashlib.sha256(consent).hexdigest()}
                if result["conditioning"] != expected:
                    raise ProtocolError("MALFORMED_WORKER_RESULT", "unbound conditioning result")
            audio_path = Path(workspace) / "output.wav"
            descriptor = os.open(audio_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(descriptor, "rb") as audio_file:
                info = os.fstat(audio_file.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                        or not 44 < info.st_size <= MAX_AUDIO_SECONDS * 24000 * 2 + 44):
                    raise ProtocolError("MALFORMED_WORKER_RESULT", "invalid synthesized audio size")
                payload = audio_file.read(info.st_size + 1)
            if (type(result["audio"]["byte_length"]) is not int or result["audio"]["byte_length"] != len(payload)
                    or result["audio"].get("sha256") != hashlib.sha256(payload).hexdigest()):
                raise ProtocolError("MALFORMED_WORKER_RESULT", "audio metadata does not match output")
            from .results import validate_wave
            try:
                validate_wave(payload, result["duration_ms"])
            except ProtocolError as error:
                raise ProtocolError("MALFORMED_WORKER_RESULT", "invalid synthesized WAV") from error
            if stream is not None:
                stream.require_audio(payload)
            if on_embedding is not None:
                from .prompt_cache import read_embedding
                check()
                on_embedding(read_embedding(Path(workspace) / "prompt.embedding"))
            # The client result contract is unchanged; the device stays local.
            runtime.last_device = result.pop("device")
            return result, payload
        finally:
            owner.finish(process, stop_process)
