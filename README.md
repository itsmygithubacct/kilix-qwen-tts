# kilix-qwen-tts

This repository currently contains the PREP7 provider-interface candidate for
the local Qwen3-TTS service planned by Plebian OS / Kilix 0.2.1.

The candidate defines a bounded Unix-socket job surface for streaming
synthesis without selecting a model, runtime, content artifact, device
profile, or release package. It is deliberately reviewable before the F104 P1
entry dependencies arrive, but it is not the frozen P1 contract and it is not
a working speech provider.

The required command population is 6/6:

- `kilix-qwen-tts serve`
- `kilix-qwen-tts synthesize`
- `kilix-qwen-tts models`
- `kilix-qwen-tts status`
- `kilix-qwen-tts cancel`
- `kilix-qwen-tts unload`

The interface supports 3/3 planned synthesis modes: installed named voices,
consent-attested prompt cloning, and free-form voice design. A future P2
selection may admit only the modes supported by the selected exact model
presentations. The interface never turns a candidate capability into a claim
that a payload is installed or supported.

Run the complete local check with:

```sh
make check
```

That command validates all 6/6 operation request shapes, 19/19 declared error
codes, 8/8 valid fixtures, 6/6 refusal fixtures, and 12/12 unit tests using only
Python's standard library.

## Boundary

This candidate does not:

- enter or close F104 P1;
- copy Qwen source or model bytes;
- select either the lightweight or quality profile;
- define F100 installation/licensing fields or F106 resource-profile fields;
- download content, accept caller-selected paths or URLs, or execute caller
  strings;
- authorize a remote, push, tag, package, or release claim.

The exact executable candidate is
[`contracts/provider-interface-candidate-v1.json`](contracts/provider-interface-candidate-v1.json).
