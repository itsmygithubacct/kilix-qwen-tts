"""QT-L-02: digest_file does not follow model-file symlinks; _copy still reads /proc/self/fd."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from kilix_qwen_tts.runtime import digest_file
from kilix_qwen_tts.sandbox import _copy, memory_file


class NoFollowTests(unittest.TestCase):
    def test_digest_file_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.write_bytes(b"payload")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(OSError) as caught:
                digest_file(link)
            self.assertEqual(caught.exception.errno, errno.ELOOP)
            self.assertEqual(digest_file(target), hashlib.sha256(b"payload").hexdigest())

    def test_digest_file_follow_hashes_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.write_bytes(b"payload")
            link = root / "link"
            link.symlink_to(target)
            self.assertEqual(
                digest_file(link, follow=True),
                hashlib.sha256(b"payload").hexdigest(),
            )

    def test_copy_reads_proc_self_fd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_bytes(b"payload")
            source = os.open(target, os.O_RDONLY | os.O_CLOEXEC)
            self.addCleanup(os.close, source)
            destination = memory_file()
            self.addCleanup(os.close, destination)
            digest, executable = _copy(
                Path(f"/proc/self/fd/{source}"), destination, lambda: None
            )
            self.assertEqual(digest, hashlib.sha256(b"payload").hexdigest())
            self.assertFalse(executable)
            os.lseek(destination, 0, os.SEEK_SET)
            self.assertEqual(os.read(destination, 16), b"payload")
