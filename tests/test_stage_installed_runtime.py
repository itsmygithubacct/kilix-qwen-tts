"""Installed-only staging retains authority and guarded transaction boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kilix_qwen_tts.protocol import ProtocolError
from kilix_qwen_tts.runtime import InstalledRuntime, MODEL_CANDIDATES
from kilix_qwen_tts import runtime
import test_content as content_fixtures
import test_runtime as runtime_fixtures

_SPEC = importlib.util.spec_from_file_location(
    'stage_installed_runtime', Path(__file__).resolve().parents[1]/'tools/stage_installed_runtime.py')
stager = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stager)


@unittest.skipUnless(content_fixtures.CONTENT_AVAILABLE, 'selected Content/Licence required')
class InstalledStagingTests(unittest.TestCase):
    def fixture(self, receipt=True):
        engine = runtime_fixtures.RuntimeTests('runTest')
        engine.setUp()
        self.addCleanup(engine.doCleanups)
        model_id = engine.manifest['model']['id']
        names = ('config.json', 'generation_config.json', 'model.safetensors',
                 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.json',
                 'merges.txt', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors',
                 'speech_tokenizer/preprocessor_config.json')
        fixture = self.enterContext(content_fixtures.PackagedFixture(
            {'model/'+name: ('synthetic '+name).encode() for name in names},
            asset_id=model_id, revision=MODEL_CANDIDATES[model_id][0], receipt=receipt))
        model = fixture.source()
        environment = dict(engine.manifest['environment'],
            python=str(engine.runtime.python_root/'bin/python3.12'),
            python_root=str(engine.runtime.python_root), site_packages=str(engine.runtime.site_packages))
        return engine, fixture, model, environment

    def run_stage(self, engine, model, destination, environment, callback=None):
        with patch.object(stager, 'environment_record',
                          side_effect=(lambda *a, **k: callback(*a)) if callback else None,
                          return_value=environment):
            return stager.stage(destination, engine.root/'environment', engine.root/'source', model)

    def test_installed_only_manifest_and_revocation_without_fallback(self):
        engine, fixture, model, environment = self.fixture()
        destination = engine.root/'staged'
        value = self.run_stage(engine, model, destination, environment)
        self.assertEqual(list(destination.iterdir()), [destination/'runtime.json'])
        self.assertEqual(json.loads((destination/'runtime.json').read_text()), value)
        InstalledRuntime(destination, model_source=model)
        with self.assertRaises(FileNotFoundError):
            InstalledRuntime(destination)
        model.store.path_for(model.reference.record_digest, model.reference.manifest_digest).unlink()
        with self.assertRaises(ProtocolError):
            with model.open(lambda: None):
                self.fail('revoked coverage was accepted after staging')

    def test_absent_receipt_refuses_before_environment_probe_or_destination(self):
        engine, _fixture, model, environment = self.fixture(receipt=False)
        destination = engine.root/'staged'
        with self.assertRaises(ProtocolError):
            self.run_stage(engine, model, destination, environment,
                           lambda *_: self.fail('unlicensed model probed environment'))
        self.assertFalse(destination.exists())

    def test_corrupt_installed_model_refuses_before_probe(self):
        engine, fixture, model, environment = self.fixture()
        (fixture.selected/'model/config.json').write_bytes(b'changed model')
        with self.assertRaises(ProtocolError):
            self.run_stage(engine, model, engine.root/'staged', environment,
                           lambda *_: self.fail('corrupt model probed environment'))
        self.assertFalse((engine.root/'staged').exists())

    def test_late_receipt_removal_rolls_back_owned_staging(self):
        engine, _fixture, model, environment = self.fixture()
        destination = engine.root/'staged'
        def revoke(*_args):
            model.store.path_for(model.reference.record_digest, model.reference.manifest_digest).unlink()
            return environment
        with self.assertRaises(ProtocolError):
            self.run_stage(engine, model, destination, environment, revoke)
        self.assertFalse(destination.exists())

    def test_replaced_ancestor_preserves_substitute_and_cleans_original(self):
        engine, _fixture, model, environment = self.fixture()
        parent, original, outside = (engine.root/name for name in ('parent','original','outside'))
        parent.mkdir(mode=0o700)
        outside.mkdir(mode=0o700)
        (outside/'sentinel').write_bytes(b'leave intact')
        def replace(*_args):
            parent.rename(original)
            parent.symlink_to(outside, target_is_directory=True)
            return environment
        with self.assertRaises(ValueError):
            self.run_stage(engine, model, parent/'staged', environment, replace)
        self.assertFalse((original/'staged').exists())
        self.assertFalse((outside/'staged').exists())
        self.assertEqual((outside/'sentinel').read_bytes(), b'leave intact')

    def test_cancellation_rolls_back_new_destination(self):
        engine, _fixture, model, environment = self.fixture()
        def cancelled(*_args):
            stager._CANCELLED = True
            return environment
        with patch.object(stager, '_CANCELLED', False):
            with self.assertRaises(InterruptedError):
                self.run_stage(engine, model, engine.root/'staged', environment, cancelled)
        self.assertFalse((engine.root/'staged').exists())

    def test_existing_destination_never_overwritten(self):
        engine, _fixture, model, environment = self.fixture()
        destination = engine.root/'staged'
        destination.mkdir(mode=0o700)
        (destination/'sentinel').write_bytes(b'prior generation')
        with self.assertRaises(FileExistsError):
            self.run_stage(engine, model, destination, environment)
        self.assertEqual((destination/'sentinel').read_bytes(), b'prior generation')

    def test_cancellation_during_runtime_rehash_rolls_back(self):
        engine, _fixture, model, environment = self.fixture()
        original = runtime.tree_digest
        calls = []
        def interrupted(path, check, **kwargs):
            calls.append(path)
            stager._CANCELLED = True
            return original(path, check, **kwargs)
        with patch.object(stager, '_CANCELLED', False), patch.object(runtime, 'tree_digest', interrupted):
            with self.assertRaises(InterruptedError):
                self.run_stage(engine, model, engine.root/'staged', environment)
        self.assertEqual(len(calls), 1)
        self.assertFalse((engine.root/'staged').exists())

    def test_cancellation_during_final_fsync_rolls_back(self):
        engine, _fixture, model, environment = self.fixture()
        original = os.fsync
        def interrupted(fd):
            import stat
            original(fd)
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                stager._CANCELLED = True
        with patch.object(stager, '_CANCELLED', False), patch.object(stager.os, 'fsync', interrupted):
            with self.assertRaises(InterruptedError):
                self.run_stage(engine, model, engine.root/'staged', environment)
        self.assertFalse((engine.root/'staged').exists())
