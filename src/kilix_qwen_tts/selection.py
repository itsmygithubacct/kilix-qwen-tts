"""Bounded startup selection of installed Qwen capabilities."""
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import stat

from .content import InstalledModel
from .protocol import ProtocolError, decode_payload
from .runtime import InstalledRuntime, MODEL_CANDIDATES


@contextmanager
def installed_runtimes(index: Path, content_root: Path):
    """Own receipt stores until every selected service worker has stopped."""
    descriptor = os.open(index, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022 or not 0 < info.st_size <= 65_536):
            raise ProtocolError("INVALID_RUNTIME", "unsafe runtime selection index")
        payload = source.read(info.st_size + 1)
        if len(payload) != info.st_size:
            raise ProtocolError("INVALID_RUNTIME", "runtime selection index changed size")
    value = decode_payload(payload)
    if (set(value) != {"schema", "runtimes"}
            or value["schema"] != "kilix.qwen-tts.runtime-set/v1"
            or type(value["runtimes"]) is not list or not 1 <= len(value["runtimes"]) <= 5):
        raise ProtocolError("INVALID_RUNTIME", "invalid runtime selection index")
    identities = set()
    for row in value["runtimes"]:
        if (type(row) is not dict or set(row) != {"root", "asset_id", "snapshot_bytes"}
                or type(row["root"]) is not str or "\x00" in row["root"]
                or not Path(row["root"]).is_absolute() or str(Path(row["root"])) != row["root"]
                or ".." in Path(row["root"]).parts
                or type(row["asset_id"]) is not str or row["asset_id"] not in MODEL_CANDIDATES
                or row["asset_id"] in identities
                or type(row["snapshot_bytes"]) is not int
                or not 0 < row["snapshot_bytes"] <= 8 * 1024**3):
            raise ProtocolError("INVALID_RUNTIME", "invalid runtime selection entry")
        identities.add(row["asset_id"])
    with ExitStack() as stack:
        runtimes = []
        for row in value["runtimes"]:
            model = stack.enter_context(InstalledModel(row["asset_id"], content_root,
                maximum_bytes=row["snapshot_bytes"], provider="kilix-qwen-tts",
                consumer_schema="kilix.qwen-tts.runtime"))
            runtime = InstalledRuntime(Path(row["root"]), model_source=model)
            runtimes.append(runtime)
        yield tuple(runtimes)
