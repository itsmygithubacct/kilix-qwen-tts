"""One offline Qwen job; supervised by the lightweight provider process."""

from __future__ import annotations

from contextlib import nullcontext
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


def _generate_cached_clone(model, options, audio, cached, workspace, torch, device):
    """Choose a cache input by actual producer, again on every retry attempt."""
    from qwen_tts import VoiceClonePromptItem
    from kilix_qwen_tts.prompt_cache import (DIMENSIONS, ENCODING, MAGIC, PRODUCERS,
                                           producer_for_device, read_embedding, validate_embedding)
    producer = producer_for_device(device)
    expected_device, expected_dtype = PRODUCERS[producer]
    if (str(model.model.device) != expected_device
            or model.model.dtype != getattr(torch, expected_dtype)):
        raise ValueError("unsupported embedding producer")
    if model.model.config.speaker_encoder_config.enc_dim != DIMENSIONS:
        raise ValueError("unsupported speaker embedding shape")
    if producer in cached:
        saved = read_embedding(f"/opt/prompt.{producer}.embedding", readonly_mount=True)
        embedding = torch.tensor(ENCODING.unpack(saved[len(MAGIC):]), dtype=torch.float32, device="cpu")
    else:
        with torch.random.fork_rng(devices=[]):
            produced = model.create_voice_clone_prompt(ref_audio=audio, x_vector_only_mode=True)
        if (len(produced) != 1 or produced[0].ref_code is not None
                or produced[0].x_vector_only_mode is not True or produced[0].icl_mode is not False
                or produced[0].ref_text is not None):
            raise ValueError("invalid cached prompt shape")
        # The engine moves the embedding to the talker's device and dtype.
        embedding = produced[0].ref_spk_embedding.detach().to(device="cpu", dtype=torch.float32)
    if (tuple(embedding.shape) != (DIMENSIONS,) or embedding.dtype != torch.float32
            or embedding.device.type != "cpu"):
        raise ValueError("invalid cached prompt shape")
    items = [VoiceClonePromptItem(None, embedding, True, False, None)]
    saved = validate_embedding(MAGIC + ENCODING.pack(*embedding.tolist()))
    (workspace / "prompt.embedding").write_bytes(saved)
    waves, rate = model.generate_voice_clone(**options, voice_clone_prompt=items)
    return waves, rate, producer


def main() -> None:
    request = json.loads(sys.stdin.buffer.read(65_537))
    from kilix_qwen_tts.bootstrap import limits
    from kilix_qwen_tts.device import offer
    profile, offered = offer(request.get("profile", "cpu"), request.get("device", "cpu"))
    # The same ceilings the trusted bootstrap applied for this profile.
    for kind, value in limits(profile).items():
        resource.setrlimit(kind, (value, value))
    os.umask(0o077)
    streaming = request.get("streaming_pcm", False)
    if type(streaming) is not bool:
        raise ValueError("invalid streaming selection")
    stream = None
    if streaming:
        from kilix_qwen_tts.streaming import WorkerStream
        stream = WorkerStream(sys.stdout.buffer)
    args = request["args"]
    cache = request.get("prompt_cache", False)
    cached = request.get("prompt_cache_inputs", [])
    from kilix_qwen_tts.prompt_cache import producers_for_profile
    if (type(cache) is not bool or type(cached) is not list
            or any(type(item) is not str or item not in producers_for_profile(profile) for item in cached)
            or len(cached) > len(producers_for_profile(profile)) or len(set(cached)) != len(cached)
            or cached and not cache
            or cache and args["mode"] != "prompt_clone"):
        raise ValueError("invalid prompt cache selection")
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
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        from qwen_tts import Qwen3TTSModel
        from kilix_qwen_tts import device as devices
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        language = LANGUAGES[args["language"].lower().split("-")[0]]
        prompt_audio = None
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
            prompt_audio = np.frombuffer(payload, dtype=dtype).astype(np.float32)
            if dtype == "<i2":
                prompt_audio /= 32768.0
            if not np.isfinite(prompt_audio).all():
                raise ValueError("non-finite prompt samples")
            consent = json.dumps(args["consent"], separators=(",", ":"), sort_keys=True).encode()
            conditioning = {"prompt_sha256": prompt["sha256"],
                            "consent_sha256": hashlib.sha256(consent).hexdigest()}
        emitted = []

        def emit(pcm):
            emitted.append(True)
            stream.pcm(pcm)

        def load(device, dtype):
            return Qwen3TTSModel.from_pretrained(
                str(root / "model"), device_map="cuda:0" if device == "cuda" else "cpu",
                dtype=dtype, attn_implementation="sdpa", local_files_only=True,
                use_safetensors=True, trust_remote_code=False,
            )

        def generate(model, device):
            # A CPU retry must not inherit an advanced RNG from the CUDA
            # attempt. This does not assert numerical equivalence of devices.
            torch.manual_seed(args["seed"])
            producer = None
            incremental = None
            if stream is not None:
                from kilix_qwen_tts.codec_stream import IncrementalCodes
                incremental = IncrementalCodes(model, emit, args["max_duration_ms"])
            options = {"text": args["text"], "language": language,
                       "non_streaming_mode": True,
                       "max_new_tokens": max(1, args["max_duration_ms"] // 80)}
            with incremental if incremental is not None else nullcontext():
                if args["mode"] == "prompt_clone":
                    audio = (prompt_audio, args["prompt_audio"]["sample_rate_hz"])
                    if cache:
                        waves, rate, producer = _generate_cached_clone(model, options, audio, cached, workspace, torch, device)
                    else:
                        waves, rate = model.generate_voice_clone(
                            **options, ref_audio=audio, x_vector_only_mode=True,
                        )
                elif args["mode"] == "named_voice":
                    waves, rate = model.generate_custom_voice(
                        **options, speaker=args["voice_id"], instruct=args.get("instruction", ""),
                    )
                elif args["mode"] == "voice_design":
                    instruction = args["description"]
                    if args.get("instruction"):
                        instruction += "\n" + args["instruction"]
                    waves, rate = model.generate_voice_design(**options, instruct=instruction)
                else:
                    raise ValueError("unsupported synthesis mode")
            return waves, rate, incremental, producer

        used, (waves, sample_rate, incremental, producer) = devices.run(
            offered, torch, load, generate, retry_allowed=lambda: not emitted)
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
    result = {"engine_id": "qwen3-tts", "engine_revision": ENGINE_COMMIT, "device": used,
              "model_id": manifest["model"]["id"], "model_revision": manifest["model"]["revision"],
              "duration_ms": duration_ms, "seed": args["seed"],
              "audio": {"sha256": hashlib.sha256(payload).hexdigest(), "byte_length": len(payload)}}
    if conditioning is not None:
        result["conditioning"] = conditioning
    if cache:
        result["prompt_producer"] = producer
    if stream is not None:
        stream.finish(result)
    else:
        sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
