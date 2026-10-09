"""Private deterministic tensors/engine for real worker transport tests.

This is deliberately no physical CUDA, torch kernel, model or quality probe.
Only dependencies are replaced; worker.main, device.run and WorkerStream run.
The two numerical sentinels distinguish producers without asserting anything
about real model embeddings or audio equivalence.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import struct
import time
import types
import wave

sys.path.insert(0, '/opt/provider')


class Array:
    ndim = 1

    def __init__(self, values, dtype='float32'):
        self.values, self.dtype = list(values), dtype

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return Array(self.values[index], self.dtype) if isinstance(index, slice) else self.values[index]

    def __itruediv__(self, value):
        self.values = [v / value for v in self.values]
        return self

    def __mul__(self, value):
        return Array([v * value for v in self.values])

    def astype(self, dtype):
        return Array([int(v) if dtype == '<i2' else float(v) for v in self.values], dtype)

    def tobytes(self):
        assert self.dtype == '<i2'
        return struct.pack('<' + str(len(self.values)) + 'h', *self.values)


numpy = types.ModuleType('numpy')
numpy.float32 = 'float32'
numpy.frombuffer = lambda payload, dtype: Array(struct.unpack(
    '<' + str(len(payload) // (2 if dtype == '<i2' else 4)) + ('h' if dtype == '<i2' else 'f'), payload), dtype)
numpy.asarray = lambda values: values if isinstance(values, Array) else Array(values)
numpy.isfinite = lambda values: types.SimpleNamespace(all=lambda: all(math.isfinite(v) for v in values.values))
numpy.full = lambda count, value, dtype: Array([value] * count, dtype)
sys.modules['numpy'] = numpy
np = numpy

request_bytes = sys.stdin.buffer.read()
request = json.loads(request_bytes)
sys.stdin = io.TextIOWrapper(io.BytesIO(request_bytes))
text = request['args']['text']
workspace = Path(request['workspace'])
trace = (workspace / 'worker.trace').open('a', buffering=1)


def note(*row):
    trace.write(json.dumps(row) + '\n')


class Device:
    def __init__(self, name):
        self.name = name
        self.type = name.split(':')[0]

    def __str__(self):
        return self.name


class Tensor:
    def __init__(self, values, dtype, device):
        self.values = list(values)
        self.dtype = dtype
        self.device = Device(device)
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def to(self, *, device, dtype):
        return Tensor(self.values, dtype, device)

    def tolist(self):
        return self.values


class OutOfMemory(RuntimeError):
    pass


seed = [123456]


def manual_seed(value):
    seed[0] = value
    note('seed', value)


@contextlib.contextmanager
def fork_rng(devices):
    assert devices == []
    saved = seed[0]
    try:
        yield
    finally:
        seed[0] = saved


torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: not text.startswith('unavailable'),
                                   device_count=lambda: 1, OutOfMemoryError=OutOfMemory,
                                   empty_cache=lambda: note('release'))
torch.float32, torch.bfloat16 = 'float32', 'bfloat16'
torch.tensor = lambda values, dtype, device: Tensor(values, dtype, device)
torch.manual_seed = manual_seed
torch.random = types.SimpleNamespace(fork_rng=fork_rng)
torch.set_num_threads = torch.set_num_interop_threads = lambda count: None
sys.modules['torch'] = torch


def pcm(audio):
    return (np.asarray(audio) * 32767).astype('<i2').tobytes()


def write(path, audio, rate, format, subtype):
    assert (format, subtype) == ('WAV', 'PCM_16')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm(audio))


soundfile = types.ModuleType('soundfile')
soundfile.write = write
sys.modules['soundfile'] = soundfile
emitter = []


def failure(device, phase):
    if text == 'cpu-fails' and device == 'cpu':
        raise RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR')
    if device != 'cuda' or not text.startswith('fail-' + phase + '-'):
        return
    name = text[len('fail-' + phase + '-'):]
    errors = {
        'oom': OutOfMemory('CUDA out of memory'),
        'cudnn': RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR'),
        'cublas': RuntimeError('CUBLAS_STATUS_ALLOC_FAILED'),
        'nvml': RuntimeError('NVML_SUCCESS == r INTERNAL ASSERT FAILED at fixture'),
        'unrelated': RuntimeError('model configuration is invalid'),
        'user-cuda': RuntimeError('user selected CUDA setting invalid'),
        'bad-param': RuntimeError('cuDNN error: CUDNN_STATUS_BAD_PARAM'),
        'value': ValueError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR'),
    }
    raise errors[name]


class VoiceClonePromptItem:
    def __init__(self, code, embedding, xvector, icl, text):
        self.ref_code, self.ref_spk_embedding = code, embedding
        self.x_vector_only_mode, self.icl_mode, self.ref_text = xvector, icl, text


class Model:
    def __init__(self, device, dtype):
        self.device = device
        self.value = .125 if device == 'cuda' else .25
        self.model = types.SimpleNamespace(device=Device('cuda:0' if device == 'cuda' else 'cpu'),
            dtype='float32' if text == 'wrong-dtype' else dtype,
            config=types.SimpleNamespace(speaker_encoder_config=types.SimpleNamespace(enc_dim=1024)))
        if text == 'wrong-device':
            self.model.device = Device('cuda:1')

    def create_voice_clone_prompt(self, *, ref_audio, x_vector_only_mode):
        assert x_vector_only_mode is True and ref_audio[1] == 24000
        note('prompt', self.device, self.model.dtype)
        failure(self.device, 'prompt')
        seed[0] += 99  # Prompt work must not advance generation's RNG.
        return [VoiceClonePromptItem(None, Tensor([self.value] * 1024, self.model.dtype,
            str(self.model.device)), True, False, None)]

    def generate_voice_clone(self, *, voice_clone_prompt, **options):
        embedding = voice_clone_prompt[0].ref_spk_embedding
        assert embedding.dtype == 'float32' and embedding.device.type == 'cpu'
        note('generate', self.device, embedding.values[0], seed[0])
        audio = np.full(2400, embedding.values[0], dtype=np.float32)
        if text == 'wait-generate':
            (workspace / 'ready').write_text('ready')
            time.sleep(60)  # Reaped only through the product's owned teardown.
        if emitter and (self.device == 'cpu' or text.startswith('fail-after-pcm-')):
            emitter[0](pcm(audio))
            note('pcm', self.device, len(audio))
            if self.device == 'cuda' and text.startswith('fail-after-pcm-'):
                # Wait for the test client's actual chunk callback, not a
                # timing guess about the service's stdout polling interval.
                until = time.monotonic() + 6
                while not (workspace / 'pcm-accepted').exists():
                    if time.monotonic() >= until:
                        raise RuntimeError('private fixture PCM acknowledgement missing')
                    time.sleep(.005)
                note('pcm-ack', self.device)
        failure(self.device, 'after-pcm')
        failure(self.device, 'generate')
        return [audio], 24000


class Qwen3TTSModel:
    @staticmethod
    def from_pretrained(path, *, device_map, dtype, **options):
        assert options == {'attn_implementation': 'sdpa', 'local_files_only': True,
                           'use_safetensors': True, 'trust_remote_code': False}
        device = 'cuda' if device_map == 'cuda:0' else 'cpu'
        note('load', device, dtype)
        failure(device, 'load')
        return Model(device, dtype)


qwen_tts = types.ModuleType('qwen_tts')
qwen_tts.Qwen3TTSModel = Qwen3TTSModel
qwen_tts.VoiceClonePromptItem = VoiceClonePromptItem
sys.modules['qwen_tts'] = qwen_tts

import kilix_qwen_tts.codec_stream as codec_stream


class Incremental:
    def __init__(self, model, emit, maximum_duration_ms):
        self.model = model
        self.emit = emit
        self.sent = 0
        def send(payload):
            self.sent += len(payload) // 2
            emit(payload)
        emitter[:] = [send]

    def __enter__(self):
        return self

    def __exit__(self, *error):
        emitter.clear()

    def finish(self, frames):
        payload = pcm(np.full(frames, self.model.value, dtype=np.float32))
        if self.sent < frames:
            self.emit(payload[self.sent * 2:])
        return payload


codec_stream.IncrementalCodes = Incremental
from kilix_qwen_tts import worker

if text.startswith('meta-'):
    # Explicit corrupt-worker controls for the host's private result binding.
    output = sys.stdout
    sink = io.BytesIO()
    sys.stdout = io.TextIOWrapper(sink)
    worker.main()
    sys.stdout.flush()
    result = json.loads(sink.getvalue())
    if text == 'meta-missing':
        result.pop('prompt_producer', None)
    else:
        result['prompt_producer'] = {'meta-wrong': 'cpu-float32', 'meta-unknown': 'gpu',
                                     'meta-type': ['cuda0-bfloat16']}[text]
    output.write(json.dumps(result))
else:
    worker.main()
    if text == 'bad-embedding':
        (workspace / 'prompt.embedding').write_bytes(b'bad')
