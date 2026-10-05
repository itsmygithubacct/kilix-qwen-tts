"""Incremental decoding for the pinned Qwen 12 Hz codebook model on CPU or CUDA."""
from __future__ import annotations

import io


class IncrementalCodes:
    """Observe complete code frames and decode bounded causal windows.

    The pinned upstream decoder itself uses a 25-code left context when it
    divides long audio. This adapter emits 20-code windows during generation;
    its final WAV is the exact concatenation of these emitted PCM bytes.
    """
    def __init__(self, model, emit, maximum_duration_ms):
        import torch
        self.torch = torch
        self.talker = model.model.talker
        self.decoder = model.model.speech_tokenizer.model.decoder
        self.emit = emit
        self.maximum_codes = maximum_duration_ms // 80
        self.code_groups = self.decoder.config.num_quantizers
        self.codebook_size = self.decoder.config.codebook_size
        self.eos = model.model.config.talker_config.codec_eos_token_id
        # Code frames stay on the device that produced them.
        self.device_type = model.model.talker.device.type
        if (int(self.decoder.total_upsample) != 1920 or self.code_groups != 16
                or model.model.speech_tokenizer.get_output_sample_rate() != 24_000):
            raise ValueError("unsupported incremental decoder")
        self.context = []
        self.pending = []
        self.chunks = []
        self.code_count = 0
        self.ended = False
        self.handle = None

    def __enter__(self):
        self.handle = self.talker.register_forward_hook(self._observe)
        return self

    def __exit__(self, *_error):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _observe(self, _module, _arguments, output):
        state = output.hidden_states
        if type(state) is not tuple or len(state) != 2:
            raise ValueError("unexpected talker output")
        codes = state[1]
        if codes is None:  # The initial text/speaker prefill produces no audio.
            return
        torch = self.torch
        if (not isinstance(codes, torch.Tensor) or tuple(codes.shape) != (1, self.code_groups)
                or codes.dtype not in (torch.int32, torch.int64) or codes.device.type != self.device_type):
            raise ValueError("unexpected code frame")
        if int(codes[0, 0]) == self.eos:
            self.ended = True
            return
        if (self.ended or self.code_count >= self.maximum_codes
                or bool((codes < 0).any()) or bool((codes >= self.codebook_size).any())):
            raise ValueError("code frame exceeds decoder bounds")
        self.pending.append(codes.detach().clone())
        self.code_count += 1
        if len(self.pending) == 20:
            self._decode()

    def _decode(self):
        if not self.pending:
            return
        import numpy as np
        import soundfile as sf
        torch = self.torch
        population = self.context + self.pending
        codes = torch.stack(population, dim=-1)
        # Decoding must not consume the talker's sampling RNG, including if a
        # future admitted dependency adds a stochastic operation to decoding.
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            decoded = self.decoder(codes)
        if tuple(decoded.shape) != (1, 1, len(population) * 1920):
            raise ValueError("unexpected incremental audio shape")
        # A float32 copy, as the engine's own decode returns, on every device.
        audio = decoded[0, 0, len(self.context) * 1920:].detach().to(torch.float32).cpu().numpy()
        if not np.isfinite(audio).all():
            raise ValueError("non-finite incremental audio")
        container = io.BytesIO()
        sf.write(container, audio, 24_000, format="WAV", subtype="PCM_16")
        payload = container.getvalue()
        if len(payload) != 44 + len(self.pending) * 1920 * 2:
            raise ValueError("unexpected incremental WAV framing")
        pcm = payload[44:]
        self.emit(pcm)
        self.chunks.append(pcm)
        self.context = population[-25:]
        self.pending = []

    def finish(self, expected_frames):
        if not self.code_count or expected_frames != self.code_count * 1920:
            raise ValueError("incremental code population differs from final output")
        self._decode()
        return b"".join(self.chunks)
