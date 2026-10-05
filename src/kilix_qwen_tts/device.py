"""Job-time CUDA selection with a CPU fallback.

Only the trusted host chooses whether a job is offered the GPU: it binds the
NVIDIA device nodes into the sandbox for a CUDA-profile runtime when every
node exists. The worker then uses CUDA only when torch can initialise it, and
otherwise runs the same job on the CPU. No wire request selects a device.
"""
from __future__ import annotations

DEVICES = ("cpu", "cuda")


def offer(profile: str, offered: str) -> tuple[str, str]:
    """A job's (profile, offered device); only a CUDA profile is offered CUDA."""
    if (type(profile) is not str or type(offered) is not str or profile not in DEVICES
            or offered not in DEVICES or offered == "cuda" and profile != "cuda"):
        raise ValueError("invalid device selection")
    return profile, offered


def select(offered: str, torch) -> str:
    """The device a job starts on: CUDA when offered and usable, else CPU."""
    if offered not in DEVICES:
        raise ValueError("invalid device selection")
    if offered == "cpu":
        return "cpu"
    try:
        usable = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:  # A driver/runtime mismatch must not fail the job.
        usable = False
    return "cuda" if usable else "cpu"


def dtype(device: str, torch):
    """CPU keeps float32; CUDA uses bfloat16 where supported, else float16."""
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def cuda_failure(error: BaseException, torch) -> bool:
    """A failure that the CPU can recover: CUDA memory or runtime errors."""
    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", None)
    if out_of_memory is not None and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and "CUDA" in str(error)


def release(torch) -> None:
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def run(offered: str, torch, load, generate, *, retry_allowed=lambda: True):
    """Return (device, result), retrying once on the CPU after a CUDA failure.

    ``load(device, dtype)`` returns a model and ``generate(model, device)``
    returns the result; ``generate`` must reseed so that a CPU retry is the
    same computation as a CPU-only job. A retry happens only while
    ``retry_allowed()`` is true, i.e. before any output was delivered.
    """
    device = select(offered, torch)
    if device == "cuda":
        model = None
        try:
            model = load("cuda", dtype("cuda", torch))
            return "cuda", generate(model, "cuda")
        except Exception as error:
            if not cuda_failure(error, torch) or not retry_allowed():
                raise
        finally:
            model = None
            release(torch)
    return "cpu", generate(load("cpu", torch.float32), "cpu")
