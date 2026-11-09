# Running the provider

Building an environment, staging a runtime and running the client. Moved from the README.

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
The interpreter must be a desktop-user-owned Python 3.12 tree, such as the
managed CPython 3.12.8 the builder uses; a system interpreter (for example a
venv whose `sys.base_prefix` is `/usr`) or another Python version refuses
before any population is hashed.
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
