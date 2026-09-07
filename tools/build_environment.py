#!/usr/bin/env python3
"""Create a new CPU environment from the committed lock and pinned engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("pyproject.toml", "uv.lock", ".python-version",
                           "tools/build_environment.py")}
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        parser.error("destination must not already exist")
    # Do not let a caller's ambient uv settings select another project, Python
    # release, index, dependency group, build backend or interpreter population.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("UV_", "PIP_", "PYTHON", "GIT_"))
                   and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    environment.update(UV_PROJECT_ENVIRONMENT=str(destination),
                       UV_PYTHON="3.12.8", UV_PYTHON_DOWNLOADS="never",
                       PYTHONDONTWRITEBYTECODE="1")
    command = [str(args.uv.absolute()), "sync", "--project", str(root),
               "--locked", "--no-install-project", "--no-editable"]
    if args.cache_directory is not None:
        command += ["--cache-dir", str(args.cache_directory.absolute())]
    if args.offline:
        command.append("--offline")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    identity = destination.stat()
    try:
        # Build tools are themselves locked before any third-party build runs.
        subprocess.run(command + ["--only-group", "build"], env=environment, check=True)
        subprocess.run(command + ["--group", "cpu", "--group", "build",
                                  "--no-build-isolation"], env=environment, check=True)
        python = destination / "bin/python"
        probe = (
            "import importlib.metadata,json,sys; "
            "assert sys.version_info[:3] == (3,12,8); "
            "d={p.metadata['Name'].lower().replace('_','-'):p.version "
            "for p in importlib.metadata.distributions()}; "
            "assert d['torch']=='2.6.0+cpu' and d['torchaudio']=='2.6.0+cpu'; "
            "assert 'gradio' not in d and 'kilix-qwen-tts' not in d; "
            "print(json.dumps(d,sort_keys=True))"
        )
        packages = json.loads(subprocess.check_output(
            [str(python), "-I", "-B", "-c", probe], env=environment, text=True))
        if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
               for name, digest in inputs.items()):
            raise RuntimeError("build inputs changed during environment creation")
        receipt = {"schema": "kilix.qwen-tts.environment-build/v1",
                   "inputs": inputs, "packages": packages,
                   "python_version": "3.12.8", "offline": args.offline,
                   "uv_sha256": hashlib.sha256(args.uv.read_bytes()).hexdigest()}
        record = destination / "kilix-environment-build.json"
        record.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        record.chmod(0o600)
        print(json.dumps({"status": "built", "packages": packages}, sort_keys=True))
        return 0
    except BaseException:
        current = destination.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            shutil.rmtree(destination)
        raise


if __name__ == "__main__":
    sys.exit(main())
