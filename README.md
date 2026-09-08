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
No GPU or streaming profile is
advertised. Missing installations and unsupported capabilities refuse.

## Run

The provider/client wheel has no inference dependencies. Its CPU environment
is a separate group in the committed `uv.lock`, using uv 0.12.5 and CPython
3.12.8. The lock pins the complete dependency graph, CPU PyTorch wheels and
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

## Checks and qualification

```sh
make check
```

The existing interface/PCM/lifecycle checks remain alongside real Linux
namespace, descriptor, cancellation and malformed-result tests. Process tests
use an explicit fake engine; test counts do not establish neural quality.
Voicebox supports the same candidate protocol and canonical WAV result.

The interface candidate in `contracts/provider-interface-candidate-v1.json`
also describes future streaming mechanics. That interface is not a claim that
the present batch runtime implements streaming. Production asset admission,
F106 selected resource profiles, shared accelerator admission, all-model and
GPU measurements, perceptual review and the combined soak remain required.
No model weights, environment binaries or user audio are committed here. The
product wrapper's source-license grant remains an owner decision, separate
from upstream source and model licenses.

## Installed model descriptors

The service can use the reviewed `kilix-content` installed-asset API instead
of local model paths. Provision that shared package in the provider environment,
then select a catalog asset at service startup:

```sh
kilix-qwen-tts serve --runtime-root /absolute/runtime \
  --installed-asset qwen3-tts-0.6b-base \
  --content-root /absolute/installed-content \
  --model-snapshot-bytes 3000000000
```

The byte argument is an explicit snapshot ceiling, not measured hardware
admission. The runtime manifest still binds the exact model population and
interpreter/dependencies. Its model files need not exist under `runtime-root`
when this option is selected. The packaged catalog must contain the matching
provider, consumer version, model revision and file digests; every job requires
current durable license receipts and the complete installed population.
No missing receipt or asset falls back to local model paths.

Receipt storage opens once during service startup, under its bounded startup
lock policy, and closes after service shutdown. Per-job receipt waits and
snapshot reads share the job's cancellation/deadline checks, with an additional
120-second snapshot ceiling. Read-only sealed member descriptors are checked
and their actual bytes hashed into the immutable runtime bundle. The content
snapshots close before process startup. No receipt is created by this provider,
and no catalog, release identity or model path is accepted on wire.

Integration tests require the reviewed `kilix-content` package on `PYTHONPATH`;
without it, those optional tests report skips. Their synthetic packaged catalog
fixtures exercise the production authority and storage implementation without
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
