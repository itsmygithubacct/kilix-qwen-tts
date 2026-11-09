# Installed model descriptors

Serving models installed through Kilix Content, the multi-model runtime index and shared execution coordination. Moved from the README.

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


## Shared execution coordination

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
