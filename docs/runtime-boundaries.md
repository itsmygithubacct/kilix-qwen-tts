# Runtime boundaries and CUDA

The sandbox, the socket service and the resource bounds, then the CUDA environment and its CPU fallback. Moved from the README.

## Runtime boundaries

Linux with unprivileged user namespaces, root-owned `/usr/bin/bwrap` and system
Python is required. The tested namespace launcher is bubblewrap 0.11.0. Its
system executables/shared libraries are trusted OS inputs. Caller-writable
Python, dependencies, models, provider code and prompts are copied into one
sealed memfd bundle; the bytes actually copied must match the installation.
A system-Python bootstrap creates a private tmpfs, unpacks bounded regular
files, makes the runtime mount read-only and drops its namespace capabilities
before importing model code. Network and PID namespaces are private. The
engine sees a private job directory and no home directory. A dedicated
host-side subreaper also owns launcher teardown, including descendants that
change process groups or sessions.

The service accepts same-UID `SOCK_SEQPACKET` clients at a private socket below
`XDG_RUNTIME_DIR`, bounds JSON bytes/depth/nodes, refuses duplicate keys and
non-finite numbers, and verifies descriptor metadata and prompt/consent
binding. A digest-bound read-only WAV descriptor carries the result. Synthesis
uses two CPU threads, float32, local safetensors and offline dependency flags;
no caller-selected executable, environment or model path is accepted on wire.
Output is integral-millisecond 24 kHz mono PCM16 WAV. Reaching the generation
duration limit fails instead of presenting capped audio as complete.

Each job snapshots up to 40,000 regular files and 7 GiB of runtime bytes; the
private extraction also consumes memory. These bounds are controls, not a
qualified RAM profile. Snapshot copy work observes cancellation/deadlines.
The trusted bootstrap applies hard CPU, address-space, file-size and descriptor
limits before executing the staged interpreter; all model code inherits them.
A crash, timeout, disconnect or canceled job releases its worker slot.

## CUDA runtimes and the CPU fallback

The `cuda` dependency group is the same engine on PyTorch `2.6.0+cu124`; it
conflicts with `cpu`, so one environment carries exactly one torch build. Build
it with `tools/build_environment.py --device cuda`; the build refuses an
environment whose reported torch build is not the requested one. Stage with
`--device cuda` (`stage_runtime.py` or `stage_installed_runtime.py`): staging
reads the environment's torch version and refuses a mismatch, and the runtime
manifest records `"device": "cuda"`. A `cpu` manifest needs the CPU build.

No wire request selects a device. For a `cuda` runtime the host offers the GPU
only when `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia0` are character
devices and both `/sys/module/nvidia/initstate` and
`/sys/module/nvidia_uvm/initstate` exist. The namespace then gains exactly
those three nodes (`--dev-bind`) and those two read-only sysfs files, which
CUDA initialisation requires; it sees no other device or sysfs entry, and
`CUDA_VISIBLE_DEVICES=0`. The driver's `libcuda.so.1` must resolve to a
root-owned file below `/usr`; where distribution packaging routes it through
`/etc` (Debian's alternatives), exactly those symlink hops are bound
read-only, and likewise for the optional PTX JIT compiler library. A `cpu` runtime is never offered the GPU.

Inside the job the worker uses CUDA only when torch can initialise it, in
bfloat16 (emulated, and slower, before Ampere); float16 overflows in this
engine and is never used. If CUDA is not
offered or not usable, or a CUDA memory/runtime error occurs before any PCM was
delivered, the same job runs on the CPU in float32 from the same seed, which is
the computation a CPU runtime performs. The worker reports the device it used;
the host refuses a GPU claim for a job that was not offered the GPU, keeps the
device out of the client result and records it on the runtime as
`last_device`. `models` reports each runtime's configured `device`.

A CUDA environment holds about 4.5 GiB of CUDA libraries, and every job copies
and verifies them like any other runtime byte. The `cuda` profile therefore
snapshots up to 60,000 files and 14 GiB, with a matching private tmpfs, and
bounds private writable data (`RLIMIT_DATA`, 24 GiB) instead of address space,
because CUDA reserves address space it never backs. The other ceilings are
unchanged. These are controls, not a qualified GPU or RAM profile.
