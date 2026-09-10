"""QT-L-02: digest_file and sandbox._copy must not follow symlinks."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from kilix_qwen_tts.runtime import digest_file
from kilix_qwen_tts.sandbox import _copy


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

    def test_sandbox_copy_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.write_bytes(b"payload")
            link = root / "link"
            link.symlink_to(target)
            destination = os.memfd_create("qtl02-copy")
            self.addCleanup(os.close, destination)
            with self.assertRaises(OSError) as caught:
                _copy(link, destination, lambda: None)
            self.assertEqual(caught.exception.errno, errno.ELOOP)
