#!/usr/bin/env python3
"""Bind reviewed model/environment bytes for an offline development runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kilix_qwen_tts.runtime import (
    ENGINE_COMMIT, MODEL_CANDIDATES, RUNTIME_SCHEMA, InstalledRuntime, digest_file, tree_digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifact-record", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_CANDIDATES), required=True)
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        parser.error("destination must not already exist")
    revision = subprocess.check_output(["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != ENGINE_COMMIT:
        parser.error("source checkout does not match the pinned engine")
    record = json.loads(args.artifact_record.read_text())
    if record["revision"] != MODEL_CANDIDATES[args.model_id][0]:
        parser.error("artifact record has a different revision")
    python = args.environment.absolute() / "bin/python"
    # -I avoids inherited startup paths; -B preserves the bound environment.
    site = Path(subprocess.check_output([str(python), "-I", "-B", "-c",
                                        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    python_root = Path(subprocess.check_output([str(python), "-I", "-B", "-c",
                                               "import sys; print(sys.base_prefix)"], text=True).strip())
    tracked = subprocess.check_output(["git", "-C", str(args.source_checkout), "ls-tree", "-rz",
                                       "--name-only", ENGINE_COMMIT, "qwen_tts"]).decode().split("\0")
    sources = {name for name in tracked if name.endswith(".py")}
    installed_sources = {str(path.relative_to(site)) for path in (site / "qwen_tts").rglob("*.py")}
    if not sources or installed_sources != sources:
        parser.error("installed Qwen source population differs from pinned Git objects")
    for name in sorted(sources):
        original = subprocess.check_output(["git", "-C", str(args.source_checkout),
                                            "show", f"{ENGINE_COMMIT}:{name}"])
        if hashlib.sha256(original).hexdigest() != digest_file(site / name):
            parser.error("installed Qwen bytes differ from pinned Git objects")
    environment = {"python": str(python), "python_sha256": digest_file(python),
                   "site_packages": str(site), "site_packages_sha256": tree_digest(site),
                   "python_root": str(python_root),
                   "python_root_sha256": tree_digest(python_root, allow_file_links=True)}
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-stage-", dir=destination.parent) as temporary:
        root = Path(temporary) / "runtime"
        root.mkdir(mode=0o700)
        files = {}
        for item in record["files"]:
            name = Path(item["path"])
            if name.is_absolute() or ".." in name.parts:
                parser.error("unsafe model artifact path")
            target = root / "model" / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(args.snapshot / name, target)
            target.chmod(0o600)
            if target.stat().st_size != item["bytes"] or digest_file(target) != item["sha256"]:
                parser.error("model artifact bytes do not match the record")
            files[str(target.relative_to(root))] = item["sha256"]
        value = {"schema": RUNTIME_SCHEMA, "engine_revision": ENGINE_COMMIT, "device": "cpu",
                 "model": {"id": args.model_id, "revision": record["revision"]},
                 "files": files, "environment": environment}
        (root / "runtime.json").write_text(json.dumps(value, indent=2) + "\n")
        (root / "runtime.json").chmod(0o600)
        InstalledRuntime(root)
        destination.mkdir(mode=0o700)
        try:
            for source in root.iterdir():
                source.rename(destination / source.name)
        except BaseException:
            shutil.rmtree(destination)
            raise
    print("Development CPU runtime staged; release qualification remains required.")


if __name__ == "__main__":
    main()
