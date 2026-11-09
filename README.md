# kilix-qwen-tts

Local Qwen3-TTS synthesis over a private Unix socket. This development runtime
supports explicitly staged, digest-bound CPU installations. It is a review
candidate for Kilix 0.2.2; release profiles and installer authority remain open.

**Status:** review candidate, not part of the Plebian-OS image. Kilix can stage
a pinned commit of this provider in the user's data directory with
`kilix tts provider prepare` (then `serve`, `status`, or `service` for login
startup), and fetches a pinned client when a user selects
`kilix tts --model qwen3-tts-0.6b-customvoice`. The three Qwen3-TTS model
snapshots are Kilix Content assets (`qwen3-tts-0.6b-base`,
`qwen3-tts-0.6b-customvoice`, `qwen3-tts-1.7b-voicedesign`) that download on
first use after their licence screen; no weights ship in the ISO or in this
repository.

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

## What it does

- Serves synthesis jobs to same-user clients over a private `SOCK_SEQPACKET`
  socket below `XDG_RUNTIME_DIR`, one job at a time.
- Runs each job in a bubblewrap namespace with no network, from a sealed
  in-memory copy of the interpreter, dependencies, engine and model.
- Supports prompt cloning (with an explicit consent assertion), named voices
  (CustomVoice) and voice design from a description (VoiceDesign).
- Reads models installed through Kilix Content and requires a current licence
  receipt for every job; it never downloads or grants consent itself.
- Optionally runs on CUDA, falling back to the CPU from the same seed.

## Quick start

The provider/client wheel has no inference dependencies; the inference
environment is a separate, locked build. The full sequence, with every option,
is in [docs/running.md](docs/running.md). In short:

```sh
python3 tools/build_environment.py \
  --uv /absolute/pinned/uv \
  --destination /absolute/private/cpu-environment --offline
python3 tools/stage_runtime.py \
  --destination /absolute/private/runtime \
  --snapshot /absolute/reviewed/model-snapshot \
  --artifact-record /absolute/reviewed/artifact-record.json \
  --environment /absolute/cpu-environment \
  --source-checkout /absolute/pinned/Qwen3-TTS \
  --model-id qwen3-tts-0.6b-base
PYTHONPATH=src python3 -m kilix_qwen_tts serve --runtime-root /absolute/private/runtime
printf '%s' 'Hello there.' | PYTHONPATH=src python3 -m kilix_qwen_tts synthesize \
  --prompt /absolute/prompt.wav --consent-asserted --output /absolute/new.wav
```

## Usage

| Command | What it does |
|---|---|
| `kilix-qwen-tts serve --runtime-root DIR` | Serve one staged runtime. |
| `kilix-qwen-tts serve --runtime-index INDEX.json --content-root DIR` | Serve several installed models on one endpoint. |
| `kilix-qwen-tts synthesize --prompt F.wav --consent-asserted --output F.wav` | Clone a consented prompt voice; text on stdin. |
| `kilix-qwen-tts synthesize --voice-id NAME` / `--description-file F` | Named voice (CustomVoice) or voice design (VoiceDesign). |
| `kilix-qwen-tts synthesize --stream-pcm` / `--wav-stdout` | Stream raw PCM, or write the verified WAV to stdout. |
| `kilix-qwen-tts models` / `status` / `cancel [JOB]` / `unload` | Query and control the running service. |
| `tools/build_environment.py`, `tools/stage_runtime.py`, `tools/stage_installed_runtime.py` | Build and stage runtimes. |
| `make check` | Interface, surface and unit checks. |

Useful `serve` options: `--installed-asset ID --content-root DIR
--model-snapshot-bytes N` (a Kilix Content install), `--prompt-cache`,
`--lease-device LABEL` and `--lease-namespace DIR`.

## How it works

```
client (kilix-tts, Voicebox, kilix-qwen-tts synthesize)
   |  same-UID SOCK_SEQPACKET, bounded JSON + read-only descriptors
   v
kilix-qwen-tts serve      licence receipt and byte checks per job
   |  sealed memfd bundle, dedicated subreaper supervisor
   v
bubblewrap namespace      private network/PID, read-only runtime, rlimits
   |
   v
worker: pinned Qwen3-TTS engine -> 24 kHz mono PCM16 WAV descriptor
```

The service accepts one job at a time. For each job it checks the licence
receipt and the installed model bytes against the Kilix Content catalog,
copies everything the engine will execute into one sealed memory file, and
starts a system-Python bootstrap inside bubblewrap that unpacks it to a private
tmpfs and drops capabilities before importing model code. The result comes back
as a digest-bound, read-only WAV descriptor. Cancellation, timeouts and crashes
release the slot only after a supervisor has reaped every owned descendant.

Details: [docs/runtime-boundaries.md](docs/runtime-boundaries.md) (sandbox,
bounds, CUDA), [docs/installed-models.md](docs/installed-models.md) (Kilix
Content assets, runtime index, shared device leases) and
[docs/streaming-and-cache.md](docs/streaming-and-cache.md). The wire interface
candidate is [contracts/provider-interface-candidate-v1.json](contracts/provider-interface-candidate-v1.json),
with fixtures under [contracts/fixtures/](contracts/fixtures/).

## Documentation

- [docs/running.md](docs/running.md): build an environment, stage a local or
  Kilix Content runtime, and run the client.
- [docs/runtime-boundaries.md](docs/runtime-boundaries.md): the namespace
  sandbox, socket bounds, resource ceilings, CUDA runtimes and the CPU fallback.
- [docs/installed-models.md](docs/installed-models.md): installed model
  descriptors, the runtime index and shared execution coordination.
- [docs/streaming-and-cache.md](docs/streaming-and-cache.md): incremental PCM
  delivery, the optional prompt embedding cache and controlled client calls.
- [docs/checks-and-qualification.md](docs/checks-and-qualification.md): what
  `make check` covers and what remains unqualified.

## Related repositories

- [kilix](https://github.com/itsmygithubacct/kilix#readme): `kilix tts provider`
  and the Qwen client installer that stage this provider.
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme): the
  catalog records for the Qwen3-TTS model assets.
- [kilix-license](https://github.com/itsmygithubacct/kilix-license#readme): the
  licence receipts every job requires.
- [kilix-voice](https://github.com/itsmygithubacct/kilix-voice#readme): Kilix
  Voice, which calls this provider for named-voice read-aloud.
- [kilix-voicebox](https://github.com/itsmygithubacct/kilix-voicebox#readme):
  the studio candidate that uses the same protocol.
- [kilix-system-monitor](https://github.com/itsmygithubacct/kilix-system-monitor#readme):
  source of the shared device-lease API.
- [kilix-transcribe](https://github.com/itsmygithubacct/kilix-transcribe#readme):
  the transcription provider design built the same way.

## Part of Plebian-OS

kilix-qwen-tts is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is not installed by default; Kilix stages a pinned commit when a user
prepares the Qwen provider. The map of every repository and how it ships is on
the [components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

No model weights, environment binaries or user audio are committed here. The
product wrapper's source is licensed under MIT; see [LICENSE](LICENSE).
Upstream source, dependencies and model licenses remain separate.
