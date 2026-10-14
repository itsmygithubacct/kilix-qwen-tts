"""Producer key binding and exact bounded immutable cache input records."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kilix_qwen_tts import bootstrap, prompt_cache
from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.sandbox import memory_file, seal
from test_prompt_cache import EMBEDDING
import test_runtime as fixtures


class PromptIdentityTests(unittest.TestCase):
    def test_execution_device_compute_dtype_and_profile_bind_independently(self):
        fixture = fixtures.RuntimeTests()
        fixture.setUp()
        try:
            scope = (os.getpid(), os.geteuid(), os.getegid(), '123')
            args = fixture.arguments()
            cpu = prompt_cache.prompt_key(scope, fixture.manifest, args, 'cpu-float32')
            cuda = prompt_cache.prompt_key(scope, fixture.manifest, args, 'cuda0-bfloat16')
            self.assertNotEqual(cpu, cuda)
            for device, dtype in (('cpu', 'bfloat16'), ('cuda:0', 'float32')):
                with patch.dict(prompt_cache.PRODUCERS, {'cpu-float32': (device, dtype)}):
                    self.assertNotEqual(prompt_cache.prompt_key(scope, fixture.manifest, args, 'cpu-float32'), cpu)
            with patch.object(prompt_cache, 'PRODUCER_EXECUTION', 'changed-execution'):
                self.assertNotEqual(prompt_cache.prompt_key(scope, fixture.manifest, args, 'cpu-float32'), cpu)
            changed = copy.deepcopy(fixture.manifest)
            changed['device'] = 'cuda'
            self.assertNotEqual(prompt_cache.prompt_key(scope, changed, args, 'cpu-float32'), cpu)
            for value in ('gpu', None, 1, ['cpu-float32']):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    prompt_cache.prompt_key(scope, fixture.manifest, args, value)
        finally:
            fixture.doCleanups()

    def test_only_supported_profile_inputs_with_valid_payloads_are_bundled(self):
        self.assertEqual(prompt_cache.validate_inputs({'cpu-float32': EMBEDDING}, 'cpu'), {'cpu-float32': EMBEDDING})
        self.assertEqual(prompt_cache.validate_inputs({'cpu-float32': EMBEDDING, 'cuda0-bfloat16': EMBEDDING}, 'cuda'),
                         {'cpu-float32': EMBEDDING, 'cuda0-bfloat16': EMBEDDING})
        for inputs, profile in (({'cuda0-bfloat16': EMBEDDING}, 'cpu'), ({'gpu': EMBEDDING}, 'cuda'),
                                ({'../../outside': EMBEDDING}, 'cuda'), ({1: EMBEDDING}, 'cuda'),
                                ({'cpu-float32': b'bad'}, 'cuda'), ([EMBEDDING], 'cuda')):
            with self.subTest(inputs=repr(inputs)[:64], profile=profile), self.assertRaises(ProtocolError):
                prompt_cache.validate_inputs(inputs, profile)

    def unpack(self, root, name, payload, executable=0):
        descriptor = memory_file()
        encoded = name.encode()
        os.write(descriptor, bootstrap.MAGIC + bootstrap.RECORD.pack(len(encoded), len(payload), executable)
                 + encoded + payload + bootstrap.RECORD.pack(0, 0, 0))
        seal(descriptor)
        # The real unpacker writes only to our private /opt stand-in.
        original_path = Path
        with patch.object(bootstrap, 'Path', side_effect=lambda value: root if value == '/opt' else original_path(value)):
            bootstrap.unpack(descriptor)  # Owns and closes this sealed fd.

    def test_bootstrap_allows_exact_readonly_producer_files_and_refuses_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('prompt.cpu-float32.embedding', 'prompt.cuda0-bfloat16.embedding'):
                self.unpack(root, name, EMBEDDING)
                self.assertEqual((root / name).read_bytes(), EMBEDDING)
                self.assertEqual((root / name).stat().st_mode & 0o622, 0o400)
            for name, payload, executable in (
                    ('prompt.unknown.embedding', EMBEDDING, 0),
                    ('prompt.cpu-float32.embedding/child', EMBEDDING, 0),
                    ('prompt.cuda0-bfloat16.embedding/../escape', EMBEDDING, 0),
                    ('prompt.cpu-float32.embedding', EMBEDDING[:-1], 0),
                    ('prompt.cuda0-bfloat16.embedding', EMBEDDING, 1)):
                with self.subTest(name=name, length=len(payload), executable=executable), self.assertRaises(ValueError):
                    self.unpack(root, name, payload, executable)
