"""Private seqpacket service for supervised, one-at-a-time synthesis."""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import select
import socket
import stat
import tempfile
import threading
import time
import uuid

from .protocol import (
    PROTOCOL_SCHEMA, ProtocolError, ProviderRequest, receive_packet,
    require_same_uid_peer, send_packet, verify_request_descriptors,
)
from .runtime import InstalledRuntime, MAX_INPUT_BYTES, MODEL_CANDIDATES, private_directory, run_job

SOCKET_NAME = "kilix-qwen-tts.sock"


def runtime_directory() -> Path:
    value = os.environ.get("XDG_RUNTIME_DIR")
    if not value:
        raise ProtocolError("INVALID_RUNTIME", "XDG_RUNTIME_DIR is required")
    return private_directory(Path(value))


def _closed(channel: socket.socket) -> bool:
    try:
        if not select.select([channel], [], [], 0)[0]:
            return False
        # Requests are single-packet operations: extra data is not another job.
        channel.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        return True
    except BlockingIOError:
        return False
    except OSError:
        return True


def _reply(request: ProviderRequest, kind: str, result: dict) -> dict:
    return {"schema": PROTOCOL_SCHEMA, "type": kind,
            "request_id": request.request_id, "job_id": request.job_id,
            "result": result}


class Service:
    def __init__(self, runtime: InstalledRuntime, directory: Path, *, additional_runtimes=(), execution_policy=None, prompt_cache=False):
        if type(prompt_cache) is not bool:
            raise ProtocolError("INVALID_REQUEST", "invalid prompt cache selection")
        from .prompt_cache import PromptCache
        self._prompt_cache = PromptCache() if prompt_cache else None
        self._cache_generation = 0
        self.runtime = runtime
        self.execution_policy = execution_policy
        self._unavailable = False
        if type(additional_runtimes) not in (tuple, list) or len(additional_runtimes) > 4:
            raise ProtocolError("INVALID_RUNTIME", "runtime selection exceeds its bound")
        self.runtimes = (runtime, *additional_runtimes)
        identities = set()
        for selected in self.runtimes:
            model_id = getattr(selected, "model_id", None)
            if (type(model_id) is not str or model_id not in MODEL_CANDIDATES
                    or model_id in identities
                    or (getattr(selected, "model_revision", None), getattr(selected, "mode", None))
                    != MODEL_CANDIDATES[model_id]):
                raise ProtocolError("INVALID_RUNTIME", "duplicate or invalid runtime selection")
            identities.add(model_id)
        self._active_model = None
        self.directory = private_directory(directory)
        self.path = directory / SOCKET_NAME
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self._mutex = threading.Lock()
        self._jobs: dict[str, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(8)
        self._threads: list[threading.Thread] = []

    def select_runtime(self, arguments):
        for selected in self.runtimes:
            if (arguments["mode"] == selected.mode
                    and arguments["model_id"] in {"auto", selected.model_id}
                    and not (selected.model_id == "qwen3-tts-0.6b-customvoice"
                             and arguments.get("instruction"))):
                return selected
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "no selected model supports this request")

    def stop(self) -> None:
        self.stopping.set()
        with self._mutex:
            self._cache_generation += 1
            if self._prompt_cache is not None:
                self._prompt_cache.clear()
            for cancellation in self._jobs.values():
                cancellation.set()

    def serve(self) -> None:
        if len(os.fsencode(self.path)) >= 108:
            raise ProtocolError("INVALID_RUNTIME", "socket path is too long")
        lock_fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        bound = None
        try:
            info = os.fstat(lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ProtocolError("INVALID_RUNTIME", "unsafe service lock")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ProtocolError("BUSY", "provider is already running") from error
            try:
                old = self.path.lstat()
            except FileNotFoundError:
                old = None
            if old is not None:
                if not stat.S_ISSOCK(old.st_mode) or old.st_uid != os.geteuid():
                    raise ProtocolError("INVALID_RUNTIME", "refusing existing non-socket path")
                # Do not replace a live endpoint created by another service.
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                try:
                    probe.settimeout(0.1)
                    probe.connect(str(self.path))
                except ConnectionRefusedError:
                    self.path.unlink()
                else:
                    raise ProtocolError("BUSY", "provider endpoint is live")
                finally:
                    probe.close()
            listener.bind(str(self.path))
            os.chmod(self.path, 0o600)
            current = self.path.lstat()
            bound = current.st_dev, current.st_ino
            listener.listen(8)
            listener.settimeout(0.2)
            self.ready.set()
            while not self.stopping.is_set():
                if self._prompt_cache is not None:
                    self._prompt_cache.expire()
                try:
                    channel, _ = listener.accept()
                except TimeoutError:
                    continue
                if not self._slots.acquire(blocking=False):
                    channel.close()
                    continue
                self._threads = [thread for thread in self._threads if thread.is_alive()]
                thread = threading.Thread(target=self._handle, args=(channel,), daemon=True)
                self._threads.append(thread)
                thread.start()
        finally:
            self.stop()
            listener.close()
            for thread in self._threads:
                thread.join()
            if self._prompt_cache is not None:
                self._prompt_cache.clear()
            if bound is not None:
                try:
                    current = self.path.lstat()
                    if (current.st_dev, current.st_ino) == bound:
                        self.path.unlink()
                except FileNotFoundError:
                    pass
            os.close(lock_fd)

    def _handle(self, channel: socket.socket) -> None:
        request = None
        descriptors = ()
        snapshot = None
        claimed = False
        started = time.monotonic()

        def finish_job():
            # A terminal event promises that another job can claim the worker.
            # Releasing here also prevents a suspended old sender's finally
            # from removing a later job that reuses the same client job ID.
            nonlocal snapshot, descriptors, claimed
            if snapshot is not None:
                os.close(snapshot)
                snapshot = None
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors = ()
            if claimed:
                with self._mutex:
                    self._jobs.pop(request.job_id, None)
                    self._active_model = None
                    claimed = False

        try:
            channel.settimeout(2)
            require_same_uid_peer(channel)
            value, descriptors = receive_packet(channel)
            request = ProviderRequest.from_payload(value)
            from .streaming import requested
            streaming = requested(value)
            deadline = started + request.deadline_ms / 1000
            if request.operation == "submit":
                if request.arguments.get("prompt_audio", {}).get("byte_length", 0) > MAX_INPUT_BYTES:
                    raise ProtocolError("LIMIT_EXCEEDED", "input exceeds runtime bound")
                language = request.arguments["language"].lower().split("-")[0]
                if language not in {"en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "auto"}:
                    raise ProtocolError("UNSUPPORTED_CAPABILITY", "language is not supported by this runtime")
                if request.arguments["mode"] == "prompt_clone" and "instruction" in request.arguments:
                    raise ProtocolError("UNSUPPORTED_CAPABILITY", "clone runtime cannot apply instructions")
                selected_runtime = self.select_runtime(request.arguments)
                with self._mutex:
                    if self._unavailable:
                        raise ProtocolError("SUPERVISOR_FAILED", "owned cleanup remains unproven")
                    if self._jobs or self.stopping.is_set():
                        raise ProtocolError("BUSY", "speech worker is busy")
                    cancellation = threading.Event()
                    self._jobs[request.job_id] = cancellation
                    self._active_model = selected_runtime.model_id
                    cache_generation = self._cache_generation
                    claimed = True
            snapshot = verify_request_descriptors(request, descriptors)
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors = ()
            if request.operation == "hello":
                result = {"protocol_major": 1, "protocol_minor": 0}
                kind = "hello"
            elif request.operation == "models":
                result = {"models": [selected.model_record() for selected in self.runtimes]}
                kind = "models"
            elif request.operation == "status":
                with self._mutex:
                    busy = bool(self._jobs)
                    unavailable = self._unavailable
                    model_id = self._active_model or self.runtime.model_id
                result = {"provider_state": "unavailable" if unavailable else "busy" if busy else "ready", "worker_active": busy,
                          "engine_id": "qwen3-tts", "model_id": model_id,
                          "release_qualified": False}
                kind = "status"
            elif request.operation == "cancel":
                with self._mutex:
                    cancellation = self._jobs.get(request.job_id)
                    if cancellation is not None:
                        cancellation.set()
                # This acknowledges a request; terminal CANCELED is emitted on
                # the original job only after its worker has been reaped.
                result = {"cancel_requested": cancellation is not None}
                kind = "canceled"
            elif request.operation == "unload":
                with self._mutex:
                    if self._unavailable:
                        raise ProtocolError("SUPERVISOR_FAILED", "owned cleanup remains unproven")
                    if self._jobs:
                        raise ProtocolError("BUSY", "cannot unload during a job")
                    self._cache_generation += 1
                    if self._prompt_cache is not None:
                        self._prompt_cache.clear()
                result = {"loaded": False}
                kind = "unloaded"
            else:
                def send_job_packet(kind, result, descriptor=None):
                    # SOCK_SEQPACKET sends one whole packet or no packet. A
                    # socket timeout transferred neither the packet nor its
                    # descriptor, so only that failure may be retried. Keep
                    # owned teardown reachable when a peer stops reading.
                    while True:
                        if cancellation.is_set() or self.stopping.is_set() or _closed(channel):
                            raise ProtocolError("CANCELED", "job canceled")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
                        channel.settimeout(min(0.05, remaining))
                        try:
                            send_packet(channel, _reply(request, kind, result), descriptor)
                            return
                        except ProtocolError as error:
                            if (error.code != "TRANSPORT_ERROR"
                                    or not isinstance(error.__cause__, TimeoutError)):
                                raise

                send_job_packet("accepted", {})
                def queued(status):
                    send_job_packet("queued", {
                        "state": status.state, "position": status.position,
                        "lease_version": status.version})
                def chunk(sequence, frame_offset, pcm):
                    from .streaming import chunk_metadata
                    with tempfile.TemporaryFile() as writer:
                        writer.write(pcm)
                        writer.flush()
                        descriptor = os.open(f"/proc/self/fd/{writer.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
                        try:
                            send_job_packet("chunk", chunk_metadata(sequence, frame_offset, pcm), descriptor)
                        finally:
                            os.close(descriptor)
                stream_options = {"on_chunk": chunk} if streaming else {}
                cache_key = None
                produced_embeddings = []
                if self._prompt_cache is not None and request.arguments["mode"] == "prompt_clone":
                    from .prompt_cache import peer_scope, prompt_key
                    cache_key = prompt_key(peer_scope(channel), selected_runtime.manifest, request.arguments)
                    if cache_key is not None:
                        stream_options.update(prompt_embedding=self._prompt_cache.get(cache_key),
                                              on_embedding=produced_embeddings.append)
                result, payload = run_job(selected_runtime, snapshot, request.arguments,
                                 deadline=deadline, cancel=cancellation,
                                 execution_policy=self.execution_policy, job_id=request.job_id, progress=queued,
                                 disconnected=lambda: self.stopping.is_set() or _closed(channel), **stream_options)
                if cache_key is not None:
                    if len(produced_embeddings) != 1:
                        raise ProtocolError("MALFORMED_WORKER_RESULT", "missing speaker embedding")
                # Canonical audio is returned in one read-only descriptor.
                with tempfile.TemporaryFile() as writer:
                    writer.write(payload)
                    writer.flush()
                    result_fd = os.open(f"/proc/self/fd/{writer.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
                    try:
                        metadata = result
                        finish_job()
                        send_job_packet("result", metadata, result_fd)
                        if cache_key is not None:
                            # A finished engine is not a successful terminal.
                            # Never retain an embedding after terminal refusal,
                            # or resurrect it after concurrent unload/stop.
                            # No service/cache lock spans the transport wait.
                            with self._mutex:
                                if (cache_generation == self._cache_generation
                                        and not self.stopping.is_set()
                                        and not cancellation.is_set()
                                        and time.monotonic() < deadline):
                                    self._prompt_cache.put(cache_key, produced_embeddings[0])
                    finally:
                        os.close(result_fd)
                return
            send_packet(channel, _reply(request, kind, result))
        except (ProtocolError, OSError, ValueError, KeyError) as error:
            if isinstance(error, ProtocolError) and error.code == "SUPERVISOR_FAILED":
                with self._mutex:
                    self._unavailable = True
            finish_job()
            if request is not None:
                code = error.code if isinstance(error, ProtocolError) else "PROVIDER_ERROR"
                event = _reply(request, "error", {})
                event.pop("result")
                event["error"] = {"code": code}
                try:
                    channel.settimeout(0.2)
                    send_packet(channel, event)
                except (ProtocolError, OSError):
                    pass
        finally:
            finish_job()
            channel.close()
            self._slots.release()


def request_value(operation: str, *, job_id: str | None = None,
                  args: dict | None = None, timeout: float = 300, stream: bool = False) -> dict:
    if type(stream) is not bool or (stream and operation != "submit"):
        raise ProtocolError("INVALID_REQUEST", "invalid PCM stream selection")
    value = {"schema": PROTOCOL_SCHEMA, "type": "request", "request_id": uuid.uuid4().hex,
             "op": operation, "deadline_ms": int(timeout * 1000), "args": args or {}}
    if job_id is not None:
        value["job_id"] = job_id
    if stream:
        value["extensions"] = {"x_pcm_stream_v1": True}
    ProviderRequest.from_payload(value)
    return value


def client_request(directory: Path, value: dict, descriptor: int | None = None, *, on_chunk=None) -> dict:
    from .streaming import ClientStream, MAX_RECORDS, requested
    ProviderRequest.from_payload(value)
    streaming = requested(value)
    if streaming and value["args"]["output"] != {"sample_format": "s16le", "sample_rate_hz": 24000, "channels": 1}:
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "PCM streaming requires 24 kHz mono PCM16")
    if on_chunk is not None and (not streaming or not callable(on_chunk)):
        raise ProtocolError("INVALID_REQUEST", "invalid PCM stream consumer")
    stream = ClientStream(value["args"]["max_duration_ms"], on_chunk) if streaming else None
    private_directory(directory)
    path = directory / SOCKET_NAME
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o177:
        raise ProtocolError("UNAUTHORIZED_PEER", "unsafe provider endpoint")
    # The job deadline stops execution. Allow bounded supervisor teardown and
    # the terminal error to arrive before treating the transport as failed.
    deadline = time.monotonic() + value["deadline_ms"] / 1000 + 6
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as channel:
        channel.settimeout(max(0.001, deadline - time.monotonic()))
        channel.connect(str(path))
        require_same_uid_peer(channel)
        send_packet(channel, value, descriptor)
        progress_count = 0
        for _ in range(MAX_RECORDS + 256 if streaming else 256):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded")
            channel.settimeout(remaining)
            event, descriptors = receive_packet(channel)
            try:
                if (event.get("schema") != PROTOCOL_SCHEMA
                        or event.get("request_id") != value["request_id"]
                        or event.get("job_id") != value.get("job_id")):
                    raise ProtocolError("INVALID_RESPONSE", "unbound provider response")
                if type(event.get("type")) is not str:
                    raise ProtocolError("INVALID_RESPONSE", "invalid event type")
                if event.get("type") == "error":
                    error = event.get("error")
                    if type(error) is not dict or descriptors:
                        raise ProtocolError("INVALID_RESPONSE", "invalid error response")
                    code = error.get("code", "PROVIDER_ERROR")
                    if (type(code) is not str or not 0 < len(code) <= 64
                            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in code)):
                        code = "PROVIDER_ERROR"
                    raise ProtocolError(code, "provider refused the request")
                if event.get("type") in {"accepted", "queued", "loading", "progress"}:
                    progress_count += 1
                    if progress_count > 255:
                        raise ProtocolError("LIMIT_EXCEEDED", "too many provider progress events")
                    if descriptors or value["op"] != "submit":
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "unexpected progress descriptor")
                    continue
                if event.get("type") == "chunk":
                    if stream is None:
                        raise ProtocolError("INVALID_RESPONSE", "unexpected PCM stream")
                    stream.append(event.get("result"), descriptors)
                    continue
                expected = {"hello": "hello", "models": "models", "status": "status",
                            "cancel": "canceled", "unload": "unloaded", "submit": "result"}
                if event.get("type") != expected.get(value["op"]):
                    raise ProtocolError("INVALID_RESPONSE", "terminal event does not match the operation")
                result = event.get("result")
                if type(result) is not dict:
                    raise ProtocolError("INVALID_RESPONSE", "invalid result envelope")
                if event.get("type") == "result":
                    metadata = result.get("audio")
                    if (len(descriptors) != 1 or type(metadata) is not dict
                            or type(metadata.get("byte_length")) is not int
                            or not 44 < metadata["byte_length"] <= 43_200_044):
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "invalid result audio")
                    fd = descriptors[0]
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                            or info.st_size != metadata["byte_length"]
                            or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "invalid audio file")
                    payload = os.pread(fd, info.st_size + 1, 0)
                    if (len(payload) != info.st_size
                            or hashlib.sha256(payload).hexdigest() != metadata.get("sha256")):
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "audio digest mismatch")
                    from .results import validate_result
                    validate_result(result, value["args"], payload)
                    if stream is not None:
                        stream.require_audio(payload)
                    return result, payload
                if descriptors or event.get("type") not in {"hello", "status", "models", "unloaded", "canceled"}:
                    raise ProtocolError("INVALID_RESPONSE", "unexpected provider response")
                return result
            finally:
                for received in descriptors:
                    os.close(received)
    raise ProtocolError("LIMIT_EXCEEDED", "provider event limit exceeded")
