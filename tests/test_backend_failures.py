"""Pinned backend diagnostics and unrelated-error negative controls.

No CUDA runtime is initialized. Signature evidence is from torch 2.6 headers
and CUDAException / CUDACachingAllocator, not physical recovery measurements.
"""
import unittest

from kilix_qwen_tts import device
from kilix_qwen_tts.protocol import ProtocolError
from test_device import OutOfMemory, fake_torch
import test_device as fixtures

POSITIVE = (
    'cuDNN error: CUDNN_STATUS_ALLOC_FAILED',
    'cuDNN error: CUDNN_STATUS_NOT_INITIALIZED',
    'cuDNN error: CUDNN_STATUS_ARCH_MISMATCH',
    'cuDNN error: CUDNN_STATUS_EXECUTION_FAILED',
    'cuDNN error: CUDNN_STATUS_INTERNAL_ERROR',
    'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.',
    'CUBLAS_STATUS_ALLOC_FAILED',
    'CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`',
    'CUBLAS_STATUS_ARCH_MISMATCH',
    'CUBLAS_STATUS_MAPPING_ERROR',
    'CUBLAS_STATUS_EXECUTION_FAILED',
    'CUBLAS_STATUS_INTERNAL_ERROR',
    'CUBLAS_STATUS_NOT_SUPPORTED',
    'CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`',
    'NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:950',
    'NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:920',
    'NVML_SUCCESS == DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_( pci_id, &nvml_device) INTERNAL ASSERT FAILED',
    'cuda error: out of memory\nCUDA kernel errors might be asynchronously reported',
    'CUDA error: no kernel image is available for execution on the device',
    'CUDA driver initialization failed, you might not have a CUDA gpu.',
)
NEGATIVE = (
    'model configuration is invalid',
    'user selected CUDA setting invalid',
    'CUDA configuration is invalid',
    'invalid model: CUDA is mentioned by user text',
    'canceled while CUDA was loading',
    'CUDA error: invalid argument',
    'CUDA error: invalid configuration argument',
    'CUDA error: invalid device ordinal',
    'CUDA error: device-side assert triggered',
    'cuDNN error: CUDNN_STATUS_BAD_PARAM',
    'cuDNN error: CUDNN_STATUS_SUCCESS',
    'cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_TYPO',
    'CUBLAS_STATUS_INVALID_VALUE',
    'CUBLAS_STATUS_LICENSE_ERROR',
    'CUBLAS_STATUS_SUCCESS',
    'CUBLAS_STATUS_ALLOC_FAILED_TYPO',
    'NVML metadata is missing',
    'NVML_SUCCESS != r INTERNAL ASSERT FAILED',
    'please handle cuDNN error: CUDNN_STATUS_INTERNAL_ERROR',
    'unknown backend error',
)


class BackendFailureTests(unittest.TestCase):
    def test_supported_signatures_and_types(self):
        for message in POSITIVE:
            with self.subTest(message=message):
                self.assertTrue(device.cuda_failure(RuntimeError(message), fake_torch()))
                self.assertFalse(device.cuda_failure(ValueError(message), fake_torch()))
        self.assertTrue(device.cuda_failure(OutOfMemory('allocator failure'), fake_torch()))

    def test_unrelated_configuration_admission_and_cancellation_errors(self):
        for message in NEGATIVE:
            with self.subTest(message=message):
                self.assertFalse(device.cuda_failure(RuntimeError(message), fake_torch()))
        for code in ('CANCELED', 'DEADLINE_EXCEEDED', 'CONSENT_REQUIRED', 'INVALID_RUNTIME', 'BUSY'):
            with self.subTest(code=code):
                self.assertFalse(device.cuda_failure(ProtocolError(code, POSITIVE[0]), fake_torch()))
        self.assertFalse(device.cuda_failure(KeyboardInterrupt(POSITIVE[0]), fake_torch()))

    def test_backend_failure_retries_once_before_output_in_load_or_generate(self):
        recorder = fixtures.SelectionTests()
        for message in POSITIVE:
            for phase in ('load', 'generate'):
                with self.subTest(message=message, phase=phase):
                    calls, load, generate = recorder.record([((phase, 'cuda'), RuntimeError(message))])
                    self.assertEqual(device.run('cuda', fake_torch(), load, generate), ('cpu', 'audio-cpu'))
                    self.assertEqual(calls[-2:], [('load', 'cpu', 'float32'), ('generate', 'model-cpu', 'cpu')])
                    self.assertEqual(sum(call == ('load', 'cpu', 'float32') for call in calls), 1)

    def test_no_retry_after_pcm_or_for_negative_runtime_errors(self):
        recorder = fixtures.SelectionTests()
        for message in (*POSITIVE, *NEGATIVE):
            with self.subTest(message=message):
                calls, load, generate = recorder.record([(('generate', 'cuda'), RuntimeError(message))])
                with self.assertRaises(RuntimeError):
                    device.run('cuda', fake_torch(), load, generate,
                               retry_allowed=lambda: message in NEGATIVE)
                self.assertNotIn(('load', 'cpu', 'float32'), calls)

    def test_cpu_failure_is_not_retried_again(self):
        recorder = fixtures.SelectionTests()
        calls, load, generate = recorder.record([
            (('generate', 'cuda'), RuntimeError(POSITIVE[0])),
            (('generate', 'cpu'), RuntimeError(POSITIVE[0]))])
        with self.assertRaises(RuntimeError):
            device.run('cuda', fake_torch(), load, generate)
        self.assertEqual(calls, [('load', 'cuda', 'bfloat16'), ('generate', 'model-cuda', 'cuda'),
                                 ('load', 'cpu', 'float32'), ('generate', 'model-cpu', 'cpu')])
