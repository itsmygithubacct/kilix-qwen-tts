# Streaming, prompt cache and client control

Optional incremental PCM delivery, the in-memory prompt embedding cache, and cancellable client calls. Moved from the README.

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
