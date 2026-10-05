# kilix-qwen-tts

Local Qwen3-TTS synthesis over a private Unix socket. This development runtime
supports explicitly staged, digest-bound CPU installations. It is a review
candidate for Kilix 0.2.2; release profiles and installer authority remain open.

The pinned engine is Qwen3-TTS `6cafe5582caea83df269c36b1ce62d953a9cc66b`.
The runtime recognises the exact 0.6B/1.7B Base and CustomVoice snapshots and
the 1.7B VoiceDesign snapshot listed in `runtime.py`. A service can select one
installation or an installed-model index and route consent-attested prompt
cloning, named voices and voice design to their matching models while retaining
one worker slot. Real CPU development jobs cover 0.6B Base, 0.6B CustomVoice
and 1.7B VoiceDesign. Other paths require their own real-model validation.
Optional incremental PCM delivery is available for the pinned CPU decoder.
No GPU or measured streaming performance profile is qualified. Missing
installations and unsupported capabilities refuse.

## Run

The provider/client wheel has no inference dependencies. Its CPU environment
is a separate group in the committed `uv.lock`, using uv 0.12.5 and CPython
3.12.8 (a `cuda` group is described below). The lock pins the complete
dependency graph, CPU and CUDA 12.4 PyTorch wheels and
exact upstream engine commit. The upstream Gradio demonstration server is
explicitly excluded; the provider uses only the engine API. Build tools are
installed from the lock before third-party builds run without build isolation.

Build a new environment with the release-selected uv and an already acquired
Python 3.12.8 installation. With `--offline`, all dependency wheels and the
pinned Git source must already be in uv's cache; missing inputs refuse:

```sh
python3 tools/build_environment.py \
  --uv /absolute/pinned/uv \
  --destination /absolute/private/cpu-environment --offline
```

Omitting `--offline` permits dependency acquisition during this explicit build
step. Model files are acquired separately. Existing destinations refuse.
Destination ancestors must be real directories owned by the caller or root,
without group/world write access except root-owned sticky temporary directories.
The builder pins their identities and refuses symlinks or path replacement.
`--timeout` bounds the whole build (1,800 seconds by default, at most 3,600).
Cancellation reaps the build's descendants before removing its staging files.
Stage already acquired and reviewed model files, the exact source checkout,
and the resulting CPU environment:

```sh
python3 tools/stage_runtime.py \
  --destination /absolute/private/runtime \
  --snapshot /absolute/reviewed/model-snapshot \
  --artifact-record /absolute/reviewed/artifact-record.json \
  --environment /absolute/cpu-environment \
  --source-checkout /absolute/pinned/Qwen3-TTS \
  --model-id qwen3-tts-0.6b-base
PYTHONPATH=src python3 -m kilix_qwen_tts serve --runtime-root /absolute/private/runtime
```

The artifact record contains `revision` and `files` entries with relative
`path`, `bytes`, and `sha256`. Staging verifies installed Qwen source against
pinned Git objects and records the complete interpreter/dependency population.
It downloads nothing. Its `kilix.qwen-tts.runtime/v1` manifest is a development
byte binding, **not an F100 install/license receipt**.

To stage a model already installed through Kilix Content, use the same managed
environment with the packaged Content and Licence APIs available:

```sh
python3 tools/stage_installed_runtime.py \
  --destination /absolute/private/installed-runtime \
  --environment /absolute/cpu-environment \
  --source-checkout /absolute/pinned/Qwen3-TTS \
  --installed-asset qwen3-tts-0.6b-customvoice \
  --content-root /absolute/desktop-apps \
  --model-snapshot-bytes 3221225472
```

This creates only `runtime.json`. It verifies current licence coverage and the
complete installed asset before environment inspection and again before
completion; it never copies models or grants consent. Serve it with the same
`--installed-asset`, `--content-root`, and `--model-snapshot-bytes` options in
addition to `--runtime-root`. The provider checks coverage and seals the model
files for each job. Staging has a 300-second deadline (maximum 600 seconds),
observes interruption during hashing and probes, and removes its own new
destination on failure. Probe descendants are reaped by a dedicated supervisor.

The client reads synthesis text from standard input and writes a new private
WAV file. Prompt files must be owned, canonical 24 kHz mono PCM16 WAV, at most
30 seconds, with explicit consent:

```sh
printf '%s' 'Hello there.' | PYTHONPATH=src python3 -m kilix_qwen_tts synthesize \
  --prompt /absolute/prompt.wav --consent-asserted --output /absolute/new.wav
```

Use `--voice-id` with a CustomVoice installation or `--description-file` with
a VoiceDesign installation. The service supports `models`, `status`, `cancel`
and `unload`. CPU workers run for one job and unload when it ends. A live job
blocks a second submission with `BUSY`; cancellation completes only after
owned descendants and temporary job files have been removed. Text and source
paths do not appear in the worker command line or normal worker logs.

Kilix Voice can use this executable from its system-Python daemon without
importing the Python 3.12 inference environment. Its explicit named-voice
request uses `--model-id qwen3-tts-0.6b-customvoice`,
`--require-installed-asset`, and `--wav-stdout`: the verified WAV is written
to stdout and result metadata to stderr. The receipt requirement is also
enforced by the service on the submitted job, so an intervening provider
restart cannot turn a receipt-backed probe into a local-stage synthesis.

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
`CUDA_VISIBLE_DEVICES=0`. A `cpu` runtime is never offered the GPU.

Inside the job the worker uses CUDA only when torch can initialise it, with
bfloat16 where the GPU supports it and float16 otherwise. If CUDA is not
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

## Checks and qualification

```sh
make check
```

The existing interface/PCM/lifecycle checks remain alongside real Linux
namespace, descriptor, cancellation and malformed-result tests. Process tests
use an explicit fake engine; test counts do not establish neural quality.
Voicebox supports the same candidate protocol and canonical WAV result.

The interface candidate in `contracts/provider-interface-candidate-v1.json`
remains unchanged. The explicit PCM extension below adds a concrete incremental
CPU path; it does not qualify the wider interface. Production asset admission,
F106 selected resource profiles, shared accelerator admission, all-model and
GPU measurements, perceptual review and the combined soak remain required.
No model weights, environment binaries or user audio are committed here. The
product wrapper's source is licensed under MIT; see [LICENSE](LICENSE).
Upstream source, dependencies and model licenses remain separate.

## Installed model descriptors

The service can use the reviewed `kilix-content` 0.2.2 packaged catalog and
`kilix-license` receipt authority instead of local model paths. Provision both
shared packages in the provider environment, then select a catalog asset at
service startup:

```sh
kilix-qwen-tts serve --runtime-root /absolute/runtime \
  --installed-asset qwen3-tts-0.6b-customvoice \
  --content-root /absolute/installed-content \
  --model-snapshot-bytes 3000000000
```

The byte argument is an explicit snapshot ceiling, not measured hardware
admission. The runtime manifest still binds the exact model population and
interpreter/dependencies. Its model files need not exist under `runtime-root`
when this option is selected. The verified packaged catalog must contain the
matching provider, F104 stream, consumer version, model revision and file
digests; the asset is read from `kilix-content`'s own install location under
`--content-root`. No missing receipt or asset falls back to local model paths.

Every job first requires `kilix_license.require` to find a current receipt in
the shared store (`kilix_license.receipt_store_root()`) covering the asset's
licence record and manifest digest. It then reads the complete installed
population without following links, refuses any undeclared, missing, non-regular
or resized member, and copies each member into a sealed read-only memory file
whose bytes must match the catalog digest. Those sealed descriptors are checked
again and hashed into the immutable runtime bundle, then closed before process
startup. Snapshot reads share the job's cancellation/deadline checks, with an
additional 120-second snapshot ceiling. No receipt is created by this provider,
and no catalog, release identity or model path is accepted on wire.

Integration tests require `kilix-content` and `kilix-license` on `PYTHONPATH`;
without them, those optional tests report skips. Their synthetic packaged
catalog fixtures mint receipts through the real agreement path into a private
store and exercise the production coverage and snapshot implementation without
admitting models to a release. Installed descriptors do not establish a
qualified resource profile or full install-transaction authority.

For all three modes on one endpoint, pass `serve --runtime-index INDEX.json
--content-root /absolute/installed-content`. The index has this shape:

```json
{
  "schema": "kilix.qwen-tts.runtime-set/v1",
  "runtimes": [
    {
      "root": "/absolute/base-runtime",
      "asset_id": "qwen3-tts-0.6b-base",
      "snapshot_bytes": 3000000000
    },
    {
      "root": "/absolute/named-runtime",
      "asset_id": "qwen3-tts-0.6b-customvoice",
      "snapshot_bytes": 3000000000
    },
    {
      "root": "/absolute/design-runtime",
      "asset_id": "qwen3-tts-1.7b-voicedesign",
      "snapshot_bytes": 5000000000
    }
  ]
}
```

The owned regular index is bounded to 64 KiB and five unique models. Every
entry goes through the same packaged receipt binding. `auto` selects the first
listed model supporting the requested mode and instruction; an explicit model
must match exactly. The 0.6B CustomVoice model cannot apply style instructions.
An unsupported request refuses, and never chooses a different mode. `models`
lists the selected capabilities, while busy `status` names the active model.
All modes share one worker slot and release it only after owned teardown.
Models are loaded per job and are not retained idle between requests.


### Shared execution coordination

`serve --lease-device cpu-development` explicitly selects the optional
`voicelib.device_leases` v1 API. The same private default namespace coordinates
all participating providers; `--lease-namespace` selects a shared private
absolute namespace for isolated deployments or tests. A namespace requires a
device label. Selecting this policy requires the API to be installed and never
falls back to uncoordinated execution. The label grants coordination only;
it does not select a GPU, establish measured fit, or qualify a resource profile.
The explicitly selected CPU runtime continues to execute CPU inference.

A job waits for the shared grant before copying or allocating model bytes. Its
normal provider job slot remains occupied while queued; cancellation,
disconnection and the original deadline remain effective, with bounded queued
progress events. The grant is inherited by the dedicated supervisor and retained
through descendant teardown. Transcription also propagates it through the
decoder and inference worker; synthesis retains another copy in the namespace
launcher for the sandbox lifetime.

A private seqpacket channel carries exactly one cleanup acknowledgment from the
supervisor after all its owned descendants are reaped. The provider checks the
kernel sender PID/UID/GID and complete message before acknowledging the grant.
Normal success, engine errors and cancellation can release only after that
proof. A spawn that created no child can release directly. Missing, malformed
or wrong-sender proof makes the service unavailable and leaves shared ownership
quarantined, even after all guard descriptors disappear. An unload or service
restart does not reset persistent quarantine. There is no automatic recovery
from unproven ownership in this API. Unrelated embedding-process children are
never adopted or reaped by the provider.

Existing direct development execution remains available without this explicit
policy, and still requires supervisor cleanup proof. Its in-process unavailable
state cannot establish persistent coordination across provider restarts. Shared
lease, installed asset, hardware admission, microphone policy and release
qualification are separate requirements; passing local controls supplies none
of the unmeasured qualifications.


## Incremental PCM delivery

`kilix-qwen-tts synthesize --stream-pcm` writes headerless 24 kHz mono signed
16-bit little-endian PCM to standard output as generation proceeds. It uses the
same prompt, named-voice and design options as whole-result synthesis. The
final metadata goes to standard error. Add `--output /absolute/new.wav` to
commit a canonical WAV after successful completion. A failed or canceled
stream can already have emitted partial PCM; consumers must treat the final
result or error as authoritative and must not present partial output as a
completed artifact. A broken output pipe disconnects and cancels the job.

Python clients opt in with `request_value("submit", ..., stream=True)` and
`client_request(..., on_chunk=callback)`. The callback receives
`(sequence, frame_offset, pcm_bytes)` synchronously. The request carries
`extensions.x_pcm_stream_v1: true`; ordinary requests keep whole-result
delivery. Only 24 kHz mono PCM16 output is supported. Each `chunk` event has
exact sequence/frame metadata, byte length and SHA256 with one read-only
descriptor. Clients bound and copy the actual bytes before calling the
consumer. The final validated WAV body must equal the concatenated chunks.
Consumers should return promptly from callbacks; the original job deadline
also applies to transport and backpressure.

The worker observes complete 16-group code frames from the pinned 12 Hz
talker and decodes every 20 frames (1.6 seconds of audio), retaining 25 code
frames of left context. The last chunk can be shorter. Decoding preserves the
talker's random state. Worker records are length bounded; incomplete writes
never block the provider's cancellation/deadline checks. The final WAV uses
the exact emitted PCM, and terminal readiness still follows owned cleanup.

The rolling context can change samples compared with the whole-result
decoder. A retained development comparison measured these differences, so
streaming is not claimed to produce identical audio or to have passed listening
review. The current upstream API also performs its final decode after code
generation; that duplicate work has not been optimized away. The first real
0.6B named-voice CPU development run delivered five chunks, with the first at
71.774 seconds and completion at 149.748 seconds on a shared host. These are
observations from one draft run, not latency or memory qualification.


## Optional prompt embedding cache

`serve --prompt-cache` enables an in-memory LRU cache of at most eight
4,104-byte speaker embeddings. Entries expire after five idle minutes; the
service checks expiration while idle. `unload` and service shutdown discard
all entries. There is no persistent cache, model retention, raw recording cache
or unsafe tensor deserialization. This option defaults off.

Each entry is scoped to the connected peer's kernel PID/UID/GID and process
start identity, the complete selected runtime/model revision and byte digests,
actual prompt PCM metadata/digest, and its permitted-use scope. A missing or
exited peer identity bypasses caching. Different client processes cannot reuse
one another's entries. Every request still verifies its current consent and
actual prompt descriptor, and every job repeats installed-asset authority and
runtime byte checks. A new consent timestamp does not change the numerical
embedding or permitted-use scope; that request's fresh consent digest remains
bound to its own result. Text and seeds are not embedding inputs.

Only a validated final result with proved owned cleanup and a successfully
sent terminal packet can populate the cache. An unload or shutdown also
invalidates pending insertions from older handlers. Cache publication can
follow terminal delivery; an immediate next request may still miss. Cached
input is included in the same sealed runtime bundle and is read
only inside the private namespace. The worker accepts exactly 1,024 bounded
finite float32 values in a fixed binary encoding, and never loads an arbitrary
Python or pickle object. Prompt construction preserves the generation RNG.
The cache holds CPU data only; it does not establish a supported memory profile
or imply any measured GPU eviction behavior.

## Controlled client delivery

The Python client accepts `client_request(..., cancelled=callback)` for submit
operations. The callback must promptly return a boolean. Controlled calls keep
the requested deadline and observe cancellation during receive waits. On
cancellation the client attempts a short cancel request for that submitted job,
closes its own channel and descriptors, and raises `CANCELED`. This is local
cancellation: neither it nor a cancel ACK proves provider cleanup. Check provider
status before a successor job; busy or unavailable remains authoritative. No
background request thread is retained. Borrowed input descriptors stay open.
The client rechecks that deadline after validation, immediately before returning
output or invoking the PCM consumer. Expired output is refused with
`DEADLINE_EXCEEDED`. Legacy submit calls without a cancellation callback retain
their existing six-second supervisor grace; short control calls receive no grace.

Installed model directory chains must be owned by the current user or root
and must not permit group or other writes. The nominated Content root and all
held descendant directories are checked; replacing the root with a symlink
refuses. Member files still require current-user ownership and exact catalog
bytes. This directory policy does not establish hardware admission or release
qualification.
