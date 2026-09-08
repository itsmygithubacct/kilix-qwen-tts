"""One offline Qwen job; supervised by the lightweight provider process."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilix_qwen_tts.runtime import ENGINE_COMMIT

LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
             "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
             "es": "Spanish", "it": "Italian", "auto": "Auto"}


def _offline(event: str, _arguments: tuple) -> None:
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
        raise PermissionError("network access is disabled in the speech worker")


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.umask(0o077)
    request = json.loads(sys.stdin.buffer.read(65_537))
    streaming = request.get("streaming_pcm", False)
    if type(streaming) is not bool:
        raise ValueError("invalid streaming selection")
    stream = None
    incremental = None
    if streaming:
        from kilix_qwen_tts.streaming import WorkerStream
        stream = WorkerStream(sys.stdout.buffer)
    args = request["args"]
    root = Path(request["runtime"])
    manifest = request["manifest"]
    workspace = Path(request["workspace"])
    os.environ.update(HF_HOME=str(workspace / "hf"), NUMBA_CACHE_DIR=str(workspace / "numba"),
                      XDG_CACHE_HOME=str(workspace / "cache"), MPLCONFIGDIR=str(workspace / "mpl"))
    # Native libraries use only local installed weights. Python network APIs
    # also refuse, even if a dependency ignores its offline environment flag.
    sys.addaudithook(_offline)
    import numpy as np
    import soundfile as sf
    import torch
    # Third-party imports/model generation may print status to stdout. The
    # worker reserves stdout for one bounded machine-readable result.
    from contextlib import ExitStack, redirect_stdout
    with ExitStack() as resources, redirect_stdout(sys.stderr):
        from qwen_tts import Qwen3TTSModel
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(args["seed"])
        model = Qwen3TTSModel.from_pretrained(
            str(root / "model"), device_map="cpu", dtype=torch.float32,
            attn_implementation="sdpa", local_files_only=True,
            use_safetensors=True, trust_remote_code=False,
        )
        if stream is not None:
            from kilix_qwen_tts.codec_stream import IncrementalCodes
            incremental = resources.enter_context(IncrementalCodes(model, stream.pcm, args["max_duration_ms"]))
        language = LANGUAGES[args["language"].lower().split("-")[0]]
        options = {"text": args["text"], "language": language,
                   "non_streaming_mode": True,
                   "max_new_tokens": max(1, args["max_duration_ms"] // 80)}
        conditioning = None
        if args["mode"] == "prompt_clone":
            descriptor = os.open("/opt/prompt.pcm", os.O_RDONLY | os.O_CLOEXEC)
            prompt = args["prompt_audio"]
            payload = os.pread(descriptor, prompt["byte_length"] + 1, 0)
            os.close(descriptor)
            if (len(payload) != prompt["byte_length"]
                    or hashlib.sha256(payload).hexdigest() != prompt["sha256"]):
                raise ValueError("prompt digest mismatch")
            dtype = "<i2" if prompt["sample_format"] == "s16le" else "<f4"
            audio = np.frombuffer(payload, dtype=dtype).astype(np.float32)
            if dtype == "<i2":
                audio /= 32768.0
            if not np.isfinite(audio).all():
                raise ValueError("non-finite prompt samples")
            consent = json.dumps(args["consent"], separators=(",", ":"), sort_keys=True).encode()
            conditioning = {"prompt_sha256": prompt["sha256"],
                            "consent_sha256": hashlib.sha256(consent).hexdigest()}
            waves, sample_rate = model.generate_voice_clone(
                **options, ref_audio=(audio, prompt["sample_rate_hz"]), x_vector_only_mode=True,
            )
        elif args["mode"] == "named_voice":
            waves, sample_rate = model.generate_custom_voice(
                **options, speaker=args["voice_id"], instruct=args.get("instruction", ""),
            )
        elif args["mode"] == "voice_design":
            instruction = args["description"]
            if args.get("instruction"):
                instruction += "\n" + args["instruction"]
            waves, sample_rate = model.generate_voice_design(**options, instruct=instruction)
        else:
            raise ValueError("unsupported synthesis mode")
    if len(waves) != 1 or sample_rate != 24000:
        raise ValueError("unexpected engine audio format")
    audio = np.asarray(waves[0])
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("invalid engine audio")
    duration_ms = (len(audio) * 1000 + sample_rate - 1) // sample_rate
    if duration_ms > args["max_duration_ms"] or duration_ms >= args["max_duration_ms"] - 80:
        raise ValueError("generation reached its duration bound")
    # Voicebox stores integral-millisecond canonical PCM WAV. Dropping fewer
    # than 24 trailing samples keeps the frame/duration contract exact.
    audio = audio[:len(audio) - len(audio) % 24]
    duration_ms = len(audio) // 24
    if not duration_ms:
        raise ValueError("empty engine audio")
    destination = workspace / "output.wav"
    if incremental is not None:
        from kilix_qwen_tts.streaming import wave_from_pcm
        destination.write_bytes(wave_from_pcm(incremental.finish(len(audio))))
    else:
        sf.write(destination, audio, sample_rate, format="WAV", subtype="PCM_16")
    payload = destination.read_bytes()
    result = {"engine_id": "qwen3-tts", "engine_revision": ENGINE_COMMIT,
              "model_id": manifest["model"]["id"], "model_revision": manifest["model"]["revision"],
              "duration_ms": duration_ms, "seed": args["seed"],
              "audio": {"sha256": hashlib.sha256(payload).hexdigest(), "byte_length": len(payload)}}
    if conditioning is not None:
        result["conditioning"] = conditioning
    if stream is not None:
        stream.finish(result)
    else:
        sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
