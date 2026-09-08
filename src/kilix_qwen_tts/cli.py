"""Local synthesis client and explicitly installed CPU runtime service."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import uuid
import wave

from .surface import CLI_COMMANDS, RuntimeUnselected, SurfaceError, inspect_command
from .protocol import ProtocolError


def parser():
    candidate = argparse.ArgumentParser(prog="kilix-qwen-tts")
    commands = candidate.add_subparsers(dest="command", required=True)
    for command in CLI_COMMANDS:
        sub = commands.add_parser(command)
        if command == "serve":
            selected = sub.add_mutually_exclusive_group()
            selected.add_argument("--runtime-root", type=Path)
            selected.add_argument("--runtime-index", type=Path)
            sub.add_argument("--installed-asset")
            sub.add_argument("--content-root", type=Path)
            sub.add_argument("--model-snapshot-bytes", type=int)
            sub.add_argument("--prompt-cache", action="store_true",
                             help="retain up to eight process-scoped speaker embeddings for five idle minutes")
            sub.add_argument("--lease-device")
            sub.add_argument("--lease-namespace")
        elif command == "synthesize":
            sub.add_argument("--output", type=Path)
            sub.add_argument("--stream-pcm", action="store_true",
                             help="write incremental 24 kHz mono PCM16 to stdout; final metadata goes to stderr")
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--prompt", type=Path)
            mode.add_argument("--description-file", type=Path)
            sub.add_argument("--consent-asserted", action="store_true")
            sub.add_argument("--voice-id", default="Vivian")
            sub.add_argument("--language", default="en")
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--timeout", type=float, default=300)
            sub.add_argument("--max-duration-ms", type=int, default=60_000)
        elif command == "cancel":
            sub.add_argument("job_id", nargs="?")
    return candidate


def _prompt(path: Path) -> tuple[bytes, dict]:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or not 44 < info.st_size <= 1_500_000:
            raise ProtocolError("INVALID_REQUEST", "prompt must be a bounded owned WAV")
        with wave.open(source, "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (24000, 1, 2, "NONE"):
                raise ProtocolError("UNSUPPORTED_CAPABILITY", "prompt must be 24 kHz mono PCM16 WAV")
            frames = audio.getnframes()
            if not 24 <= frames <= 720_000:
                raise ProtocolError("LIMIT_EXCEEDED", "prompt must be at most 30 seconds")
            frames -= frames % 24
            pcm = audio.readframes(frames)
    if len(pcm) != frames * 2:
        raise ProtocolError("INVALID_REQUEST", "prompt ended early")
    metadata = {"sample_format": "s16le", "sample_rate_hz": 24000, "channels": 1,
                "duration_ms": frames // 24, "frame_count": frames,
                "byte_length": len(pcm), "sha256": hashlib.sha256(pcm).hexdigest()}
    return pcm, metadata


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .runtime import InstalledRuntime
        from .service import Service, client_request, request_value, runtime_directory
        if args.command == "serve":
            from .owned import from_options as execution_options
            execution_policy = execution_options(args)
        if args.command == "serve" and args.runtime_index is not None:
            if (args.content_root is None or args.installed_asset is not None
                    or args.model_snapshot_bytes is not None):
                raise ProtocolError("INVALID_REQUEST", "runtime index requires only the shared content root")
            from .selection import installed_runtimes
            with installed_runtimes(args.runtime_index, args.content_root) as runtimes:
                service = Service(runtimes[0], runtime_directory(), additional_runtimes=runtimes[1:],
                                  execution_policy=execution_policy, prompt_cache=args.prompt_cache)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda _sig, _frame: service.stop())
                service.serve()
            return 0
        if args.command == "serve":
            from .content import from_options
            model_source = from_options(args, provider="kilix-qwen-tts",
                                        consumer_schema="kilix.qwen-tts.runtime")
        if args.command == "serve" and args.runtime_root is not None:
            with model_source if model_source is not None else nullcontext():
                service = Service(InstalledRuntime(args.runtime_root, model_source=model_source), runtime_directory(),
                                  execution_policy=execution_policy, prompt_cache=args.prompt_cache)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda _sig, _frame: service.stop())
                service.serve()
            return 0
        if args.command == "synthesize" and (args.output is not None or args.stream_pcm):
            text = sys.stdin.buffer.read(16_385).decode("utf-8")
            request = {"task": "synthesize", "text": text, "model_id": "auto",
                       "language": args.language, "seed": args.seed,
                       "max_duration_ms": args.max_duration_ms,
                       "mode": "prompt_clone" if args.prompt is not None else "named_voice",
                       "output": {"sample_format": "s16le", "sample_rate_hz": 24000, "channels": 1}}
            with tempfile.TemporaryFile() as prompt_file:
                descriptor = None
                try:
                    if args.prompt is not None:
                        if not args.consent_asserted:
                            raise ProtocolError("CONSENT_REQUIRED", "prompt cloning requires explicit consent")
                        pcm, metadata = _prompt(args.prompt)
                        prompt_file.write(pcm)
                        prompt_file.flush()
                        descriptor = os.open(f"/proc/self/fd/{prompt_file.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
                        request.update(prompt_fd=0, prompt_audio=metadata,
                            consent={"schema": "kilix.voice.consent/candidate-v1",
                                     "source_sha256": metadata["sha256"], "asserted_by_peer": True,
                                     "allowed_use": "this-project", "purpose": None,
                                     "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                    elif args.description_file is not None:
                        request["mode"] = "voice_design"
                        description_fd = os.open(args.description_file, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
                        with os.fdopen(description_fd, "rb") as description:
                            if not stat.S_ISREG(os.fstat(description.fileno()).st_mode):
                                raise ProtocolError("INVALID_REQUEST", "description must be a regular file")
                            request["description"] = description.read(4097).decode("utf-8")
                    else:
                        request["voice_id"] = args.voice_id
                    def consume(_sequence, _frame_offset, pcm):
                        sys.stdout.buffer.write(pcm)
                        sys.stdout.buffer.flush()
                    result, audio = client_request(runtime_directory(), request_value(
                        "submit", job_id=uuid.uuid4().hex, args=request, timeout=args.timeout,
                        stream=args.stream_pcm), descriptor, on_chunk=consume if args.stream_pcm else None)
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            # CLI destinations are local user operations and never go on wire.
            if args.output is not None:
                descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
                try:
                    with os.fdopen(descriptor, "wb") as destination:
                        destination.write(audio)
                except BaseException:
                    args.output.unlink(missing_ok=True)
                    raise
            print(json.dumps(result, separators=(",", ":"), sort_keys=True),
                  file=sys.stderr if args.stream_pcm else sys.stdout)
            return 0
        if args.command in {"models", "status", "cancel", "unload"}:
            try:
                if args.command == "cancel" and args.job_id is None:
                    raise RuntimeUnselected()
                payload = client_request(runtime_directory(), request_value(
                    args.command, job_id=getattr(args, "job_id", None), timeout=5))
            except (FileNotFoundError, ConnectionRefusedError, RuntimeUnselected):
                payload = inspect_command(args.command)
            except ProtocolError as error:
                if error.code != "INVALID_RUNTIME":
                    raise
                payload = inspect_command(args.command)
        else:
            payload = inspect_command(args.command)
    except RuntimeUnselected as error:
        print(str(error), file=sys.stderr)
        return 69
    except (SurfaceError, ProtocolError, OSError, ValueError, OverflowError) as error:
        code = getattr(error, "code", "PROVIDER_UNAVAILABLE")
        print(f"KILIX_QWEN_TTS_REFUSAL [{code}] speech request failed", file=sys.stderr)
        return 69
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
