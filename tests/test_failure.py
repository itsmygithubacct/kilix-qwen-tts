"""Worker-stop classification without a device, a model, or a namespace."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch

from kilix_qwen_tts.device import cuda_failure, failure_kind
from kilix_qwen_tts.failure import CorruptInput, classify, guard, report
from kilix_qwen_tts.supervisor import exit_status


class _Torch:
    class cuda:
        class OutOfMemoryError(RuntimeError):
            pass


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kq-failure-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = self.temporary.name

    def test_exit_status_preserves_stop_and_signal_death(self):
        self.assertEqual(exit_status(-9, False), 137)
        self.assertEqual(exit_status(-15, False), 143)
        self.assertEqual(exit_status(0, False), 0)
        self.assertEqual(exit_status(1, False), 1)
        self.assertEqual(exit_status(3, False), 3)
        self.assertEqual(exit_status(None, True), 130)
        self.assertEqual(exit_status(-9, True), 130)
        self.assertEqual(exit_status(None, False), 125)

    def test_signals_and_disk_exit_do_not_need_a_record(self):
        for code in (-9, -15, 132, 134, 135, 136, 137, 139, 143, 159):
            with self.subTest(code=code):
                self.assertEqual(classify(code, self.workspace)[0], "WORKER_CRASH")
        self.assertIsNone(classify(130, self.workspace))
        self.assertIsNone(classify(1, self.workspace))
        self.assertIsNone(classify(0, self.workspace))
        self.assertEqual(classify(4, None),
                         ("INSUFFICIENT_DISK", "speech worker ran out of disk"))

    def test_only_a_small_regular_record_selects_a_code(self):
        report_path = Path(self.workspace) / "failure.json"
        report_path.write_text('{"kind":"oom"}')
        self.assertEqual(classify(3, self.workspace),
                         ("DEVICE_UNAVAILABLE", "speech worker ran out of memory"))
        self.assertIsNone(classify(1, self.workspace))
        report_path.write_text('{"kind":"oom","extra":1}')
        self.assertIsNone(classify(3, self.workspace))
        report_path.write_text("not-json")
        self.assertIsNone(classify(3, self.workspace))
        report_path.write_text('{"kind":"oom"}' + (" " * 64))
        self.assertGreater(report_path.stat().st_size, 64)
        self.assertIsNone(classify(3, self.workspace))
        report_path.unlink()
        report_path.symlink_to(self.workspace)
        self.assertIsNone(classify(3, self.workspace))
        report_path.unlink()
        os.mkfifo(report_path)
        started = time.monotonic()
        self.assertIsNone(classify(3, self.workspace))
        self.assertLess(time.monotonic() - started, 1)
        os.remove(report_path)
        report_path.write_text('{"kind":"device_loss"}')
        real = os.fstat

        def wrong_owner(descriptor):
            info = real(descriptor)
            return type("Stat", (), {
                "st_mode": info.st_mode, "st_uid": info.st_uid + 1, "st_size": info.st_size,
            })()

        with patch("kilix_qwen_tts.failure.os.fstat", side_effect=wrong_owner):
            self.assertIsNone(classify(3, self.workspace))
        self.assertTrue(stat.S_ISREG(report_path.stat().st_mode))
        self.assertEqual(classify(137, self.workspace)[0], "WORKER_CRASH")
        self.assertEqual(classify(4, self.workspace)[0], "INSUFFICIENT_DISK")

    def test_report_exits_and_a_full_disk_does_not_recurse(self):
        with self.assertRaises(SystemExit) as caught:
            report(self.workspace, "corrupt")
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text()),
                         {"kind": "corrupt"})
        self.assertEqual(Path(self.workspace, "failure.json").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            report(self.workspace, "oom")
        link = Path(self.workspace) / "linked"
        link.mkdir()
        os.symlink(link / "target", Path(link, "failure.json"))
        with self.assertRaises(OSError):
            report(str(link), "oom")
        self.assertFalse((link / "target").exists())

        def full(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "no space")

        with patch("kilix_qwen_tts.failure.os.open", side_effect=full):
            with self.assertRaises(SystemExit) as caught:
                report(self.workspace + "-missing", "disk")
            self.assertEqual(caught.exception.code, 4)
            with self.assertRaises(OSError):
                report(self.workspace + "-missing", "oom")

    def test_guard_records_memory_disk_corrupt_and_device_loss(self):
        def stop(error):
            raise error

        with self.assertRaises(SystemExit) as caught:
            guard(self.workspace, lambda: stop(MemoryError()))
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "oom")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(SystemExit) as caught:
            guard(self.workspace, lambda: stop(OSError(errno.ENOSPC, "full")))
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "disk")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(SystemExit):
            guard(self.workspace, lambda: stop(CorruptInput("bad")))
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "corrupt")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(SystemExit):
            guard(self.workspace, lambda: stop(RuntimeError("CUDA error: initialization error")),
                  classify=failure_kind)
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"],
                         "device_loss")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(RuntimeError):
            guard(self.workspace, lambda: stop(RuntimeError("CUDA error: unspecified launch failure")),
                  classify=failure_kind)
        self.assertFalse((Path(self.workspace) / "failure.json").exists())
        with self.assertRaises(MemoryError):
            guard(None, lambda: stop(MemoryError()))

    def test_failure_kind_is_narrower_than_the_cpu_retry_matcher(self):
        torch = _Torch()
        memory = _Torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.assertTrue(cuda_failure(memory, torch))
        self.assertEqual(failure_kind(memory), "oom")
        self.assertEqual(failure_kind(MemoryError()), "oom")
        self.assertEqual(failure_kind(RuntimeError("CUDA error: out of memory")), "oom")
        cublas = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        self.assertTrue(cuda_failure(cublas, torch))
        self.assertIsNone(failure_kind(cublas))
        located = RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED at fixture")
        self.assertTrue(cuda_failure(located, torch))
        self.assertIsNone(failure_kind(located))
        for text in (
            "CUDA error: no kernel image is available for execution on the device",
            "CUDA error: initialization error",
            "CUDA error: driver shutting down",
            "CUDA error: all CUDA-capable devices are busy or unavailable",
            "CUDA error: CUDA-capable device(s) is/are busy or unavailable",
            "CUDA error: unknown error",
            "CUDA driver initialization failed",
            "NVML_SUCCESS == r INTERNAL ASSERT FAILED",
        ):
            with self.subTest(text=text):
                error = RuntimeError(text)
                self.assertTrue(cuda_failure(error, torch))
                self.assertEqual(failure_kind(error), "device_loss")
        generic = RuntimeError("CUDA error: unspecified launch failure")
        self.assertTrue(cuda_failure(generic, torch))
        self.assertIsNone(failure_kind(generic))
        self.assertIsNone(failure_kind(ValueError("prompt digest mismatch")))


if __name__ == "__main__":
    unittest.main()
