"""The worker's own CUDA-to-CPU retry wiring, with no torch, GPU or real model.

device.run() is tested on its own; these run the real worker.main() as a
subprocess against stand-in torch, soundfile and qwen_tts modules, so the gate
the worker hands to it (no CPU retry once PCM went out) and the per-attempt
reseed are exercised where they are wired, not only in the recorded GPU runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kilix_qwen_tts.streaming import HEADER, MAGIC

SRC = Path(__file__).resolve().parents[1] / 'src'

HARNESS = r'''
import json, os, sys, types, wave
from types import SimpleNamespace
import numpy as np

log = open(os.environ['WIRING_LOG'], 'a')
def note(*item):
    log.write(json.dumps(item) + '\n')
    log.flush()

class OutOfMemory(RuntimeError):
    pass

torch = types.ModuleType('torch')
torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             OutOfMemoryError=OutOfMemory, empty_cache=lambda: None)
torch.float32, torch.bfloat16 = 'float32', 'bfloat16'
torch.manual_seed = lambda seed: note('seed', seed)
torch.set_num_threads = torch.set_num_interop_threads = lambda count: None
sys.modules['torch'] = torch

soundfile = types.ModuleType('soundfile')
def write(path, audio, rate, format, subtype):
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate)
        output.writeframes((np.asarray(audio) * 32767).astype('<i2').tobytes())
soundfile.write = write
sys.modules['soundfile'] = soundfile

emit = []   # the worker's emit, handed over by the stand-in incremental decoder
FAIL = os.environ['WIRING_FAIL']

class Model:
    def __init__(self, device):
        self.device = device
    def generate_custom_voice(self, **options):
        note('generate', self.device)
        if emit and (self.device == 'cpu' or FAIL == 'after-pcm'):
            emit[0](b'\0' * 1920)
        if self.device == 'cuda' and FAIL != 'none':
            raise OutOfMemory('CUDA out of memory')
        return [np.zeros(24000, dtype=np.float32)], 24000

class Qwen3TTSModel:
    @staticmethod
    def from_pretrained(path, device_map, dtype, **options):
        device = 'cuda' if device_map == 'cuda:0' else 'cpu'
        note('load', device, dtype)
        return Model(device)

qwen_tts = types.ModuleType('qwen_tts')
qwen_tts.Qwen3TTSModel = Qwen3TTSModel
sys.modules['qwen_tts'] = qwen_tts

import kilix_qwen_tts.codec_stream as codec_stream
class Incremental:
    def __init__(self, model, emitter, maximum_duration_ms):
        emit[:] = [emitter]
    def __enter__(self):
        return self
    def __exit__(self, *error):
        emit.clear()
    def finish(self, frames):
        return b'\0\0' * frames
codec_stream.IncrementalCodes = Incremental

from kilix_qwen_tts import worker
worker.main()
'''


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'the worker needs numpy')
class WorkerRetryWiringTests(unittest.TestCase):
    def run_worker(self, fail, streaming):
        with tempfile.TemporaryDirectory(prefix='kq-wiring-') as tmp:
            root = Path(tmp)
            (root / 'harness.py').write_text(HARNESS)
            request = {'runtime': '/nonexistent', 'workspace': str(root), 'profile': 'cuda', 'device': 'cuda',
                       'streaming_pcm': streaming,
                       'manifest': {'model': {'id': 'qwen3-tts-0.6b-customvoice', 'revision': 'fixture'}},
                       'args': {'mode': 'named_voice', 'language': 'en', 'text': 'hello', 'voice_id': 'Vivian',
                                'seed': 7, 'max_duration_ms': 60000}}
            environment = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(SRC), 'PYTHONDONTWRITEBYTECODE': '1',
                           'WIRING_LOG': str(root / 'log'), 'WIRING_FAIL': fail}
            done = subprocess.run([sys.executable, '-B', str(root / 'harness.py')], input=json.dumps(request).encode(),
                                  capture_output=True, env=environment, timeout=60, cwd=tmp)
            notes = [json.loads(line) for line in (root / 'log').read_text().splitlines()]
        return done, notes

    @staticmethod
    def records(stdout):
        assert stdout[:len(MAGIC)] == MAGIC, stdout[:len(MAGIC)]
        kinds, offset = [], len(MAGIC)
        while offset < len(stdout):
            kind, length = HEADER.unpack_from(stdout, offset)
            kinds.append(kind)
            offset += HEADER.size + length
        return kinds

    def test_streaming_job_never_retries_on_the_cpu_after_pcm_went_out(self):
        done, notes = self.run_worker('after-pcm', streaming=True)
        self.assertNotEqual(done.returncode, 0, done.stderr.decode())
        self.assertIn(b'CUDA out of memory', done.stderr)
        self.assertEqual([item for item in notes if item[0] != 'seed'],
                         [['load', 'cuda', 'bfloat16'], ['generate', 'cuda']])
        # The consumer was sent one PCM record and no result, never CPU audio.
        self.assertEqual(self.records(done.stdout), [b'P'])

    def test_streaming_job_that_failed_before_any_pcm_retries_on_the_cpu(self):
        done, notes = self.run_worker('before-pcm', streaming=True)
        self.assertEqual(done.returncode, 0, done.stderr.decode())
        self.assertEqual([item for item in notes if item[0] != 'seed'],
                         [['load', 'cuda', 'bfloat16'], ['generate', 'cuda'],
                          ['load', 'cpu', 'float32'], ['generate', 'cpu']])
        self.assertEqual(self.records(done.stdout), [b'P', b'R'])

    def test_non_streaming_job_retries_on_the_cpu_and_reports_it(self):
        done, notes = self.run_worker('before-pcm', streaming=False)
        self.assertEqual(done.returncode, 0, done.stderr.decode())
        self.assertEqual([item for item in notes if item[0] != 'seed'],
                         [['load', 'cuda', 'bfloat16'], ['generate', 'cuda'],
                          ['load', 'cpu', 'float32'], ['generate', 'cpu']])
        self.assertEqual(json.loads(done.stdout)['device'], 'cpu')

    def test_every_attempt_starts_from_the_job_seed(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                done, notes = self.run_worker('before-pcm', streaming=streaming)
                self.assertEqual(done.returncode, 0, done.stderr.decode())
                self.assertEqual([item for item in notes if item[0] == 'seed'], [['seed', 7], ['seed', 7]])

    def test_a_cuda_job_that_succeeds_never_touches_the_cpu(self):
        done, notes = self.run_worker('none', streaming=False)
        self.assertEqual(done.returncode, 0, done.stderr.decode())
        self.assertEqual([item for item in notes if item[0] != 'seed'],
                         [['load', 'cuda', 'bfloat16'], ['generate', 'cuda']])
        self.assertEqual(json.loads(done.stdout)['device'], 'cuda')


if __name__ == '__main__':
    unittest.main()
