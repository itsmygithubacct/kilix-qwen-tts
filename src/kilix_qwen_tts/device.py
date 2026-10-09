"""Job-time CUDA selection with a CPU fallback.

Only the trusted host chooses whether a job is offered the GPU: it binds the
NVIDIA device nodes into the sandbox for a CUDA-profile runtime when every
node exists. The worker then uses CUDA only when torch can initialise it, and
otherwise runs the same job on the CPU. No wire request selects a device.
"""
from __future__ import annotations

import re

DEVICES = ("cpu", "cuda")

# Pinned torch 2.6: ATen/cuda/Exceptions.h (cuDNN / BLAS) and
# c10/cuda/{CUDAException,CUDACachingAllocator}.cpp (runtime / NVML).
# Status names for bad parameters, invalid values and licence failures are
# deliberately absent: those are not recoverable backend execution failures.
_BACKEND_STATUSES = r'(?:ALLOC_FAILED|NOT_INITIALIZED|ARCH_MISMATCH|MAPPING_ERROR|EXECUTION_FAILED|INTERNAL_ERROR|NOT_SUPPORTED)'
_BACKEND_FAILURE = re.compile(
    r'^(?:'
    r'CUDA error: (?:out of memory|no kernel image(?: is available for execution on the device)?'
    r'|initialization error|driver shutting down|unspecified launch failure'
    r'|the launch timed out and was terminated|all CUDA-capable devices are busy or unavailable'
    r'|CUDA-capable device\(s\) is/are busy or unavailable'
    r'|CUDA driver version is insufficient for CUDA runtime version|unknown error)(?=[.\n]|$)'
    r'|cuDNN error:\s*CUDNN_STATUS_' + _BACKEND_STATUSES + r'\b'
    r'|(?:CUDA error:\s*)?CUBLAS_STATUS_' + _BACKEND_STATUSES + r'\b'
    r'|NVML_SUCCESS == (?:r|DriverAPI::get\(\)->nvmlInit_v2_\(\)'
    r'|DriverAPI::get\(\)->nvmlDeviceGetHandleByPciBusId_v2_\(\s*pci_id,\s*&nvml_device\)) INTERNAL ASSERT FAILED\b'
    r'|CUDA driver initialization failed\b'
    r')', re.IGNORECASE)


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
    """CPU keeps float32; CUDA always uses bfloat16, the checkpoint's dtype.

    float16 overflows in this engine: a 0.6B CustomVoice generation stops
    with a device-side assert. Pre-Ampere GPUs emulate bfloat16, slower but
    correct.
    """
    return torch.float32 if device == "cpu" else torch.bfloat16


def cuda_failure(error: BaseException, torch) -> bool:
    """Recognize supported backend diagnostics, never an arbitrary RuntimeError.

    A matching failure permits one CPU attempt; it does not promise recovery.
    run() also requires that no PCM has been emitted by the CUDA attempt.
    """
    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", None)
    if out_of_memory is not None and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and _BACKEND_FAILURE.match(str(error)) is not None


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
    returns the result; ``generate`` must reseed for each attempt. A retry
    happens only while
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
