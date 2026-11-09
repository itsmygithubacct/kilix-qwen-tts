# Checks and qualification

What `make check` covers and what remains unqualified. Moved from the README.

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
product wrapper's source is licensed under MIT; see [LICENSE](../LICENSE).
Upstream source, dependencies and model licenses remain separate.
