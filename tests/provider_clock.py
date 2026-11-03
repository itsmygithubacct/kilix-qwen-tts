"""Advance provider deadlines after an observed fixture state, not startup."""
from contextlib import contextmanager
import time
from types import SimpleNamespace
from unittest.mock import patch


@contextmanager
def provider_clock():
    # Real sockets, workers and observers keep their ordinary clocks. Only the
    # provider modules (including the legacy client's cleanup grace) advance.
    offset = [0.0]
    clock = SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0], sleep=time.sleep)

    def advance(seconds):
        offset[0] += seconds

    with patch('kilix_qwen_tts.service.time', clock), patch('kilix_qwen_tts.runtime.time', clock):
        yield advance
