#!/usr/bin/env python3
"""Bind an owned CPU environment to receipt-covered Content without copying models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kilix_qwen_tts.content import InstalledModel
from kilix_qwen_tts.runtime import ENGINE_COMMIT, MODEL_CANDIDATES, RUNTIME_SCHEMA, InstalledRuntime
from build_environment import Destination
from stage_runtime import environment_record

_CANCELLED = False


def cancel(_signum, _frame):
    global _CANCELLED
    _CANCELLED = True


def stage(destination: Path, environment: Path, source_checkout: Path,
          model: InstalledModel, *, timeout: float = 300) -> dict:
    """Publish only a verified manifest into a pinned, newly owned directory."""
    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError('staging timeout must be positive and at most600 seconds')
    deadline = time.monotonic() + timeout

    def check():
        if _CANCELLED:
            raise InterruptedError('installed runtime staging interrupted')
        if time.monotonic() >= deadline:
            raise TimeoutError('installed runtime staging deadline exceeded')

    spec = model.spec
    if spec.asset_id not in MODEL_CANDIDATES or spec.version != MODEL_CANDIDATES[spec.asset_id][0]:
        raise ValueError('installed model is not a pinned Qwen candidate')
    files = {member.path: member.sha256 for member in spec.files
             if not member.path.startswith('notices/')}
    model.bind(spec.asset_id, spec.version, files)
    # Verify consent and the entire population before probing environment code
    # or creating a destination. Snapshots are closed, never copied into it.
    with model.open(check):
        pass
    with Destination(destination) as held:
        def checkpoint():
            check()
            held.check()
        bound = environment_record(environment, source_checkout, checkpoint, deadline=deadline)
        value = {'schema': RUNTIME_SCHEMA, 'engine_revision': ENGINE_COMMIT, 'device': 'cpu',
                 'model': {'id': spec.asset_id, 'revision': spec.version},
                 'files': files, 'environment': bound}
        descriptor = os.open('runtime.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=held.descriptors[-1])
        with os.fdopen(descriptor, 'w') as output:
            output.write(json.dumps(value, indent=2)+'\n')
            output.flush()
            os.fsync(output.fileno())
        checkpoint()
        InstalledRuntime(destination, model_source=model, check=checkpoint)
        # Recheck current consent/installed bytes immediately before publishing.
        with model.open(checkpoint):
            pass
        checkpoint()
        os.fsync(held.descriptors[-1])
        checkpoint()
    return value


def main(argv=None) -> int:
    global _CANCELLED
    _CANCELLED = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--source-checkout', type=Path, required=True)
    parser.add_argument('--installed-asset', choices=tuple(MODEL_CANDIDATES), required=True)
    parser.add_argument('--content-root', type=Path, required=True)
    parser.add_argument('--model-snapshot-bytes', type=int, required=True)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    if not args.destination.is_absolute() or not args.environment.is_absolute() or not args.source_checkout.is_absolute():
        parser.error('destination, environment and source checkout must be absolute paths')
    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with InstalledModel(args.installed_asset, args.content_root,
            maximum_bytes=args.model_snapshot_bytes, provider='kilix-qwen-tts',
            consumer_schema='kilix.qwen-tts.runtime') as model:
            stage(args.destination, args.environment, args.source_checkout, model, timeout=args.timeout)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print('Receipt-backed CPU runtime staged; release qualification remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
