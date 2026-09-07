"""The audit payload, shared through the object store.

A cache entry travels already (:mod:`reverberate.wave.vox_store`) and a run's
published files travel already (``w20_render.publish``). What did not was the
audit view's payload -- a directory of binary tiles per room -- so a second
checkout, or a second machine, opened the run page and found no grid to draw.

**It is a convenience, not the artefact.** The payload is *derived*: 1.7 GB for
the whole flat at 16 kHz, rebuilt from the cached voxelisation in thirteen
minutes by :mod:`reverberate.experiments.audit_view`. The grid it comes from is
the expensive thing -- 3.08 h of CPU for that one -- and it is what must be in
the store. This exists so a reader who only wants to *look* need not rebuild.

**Publication is atomic in the same sense as a cache entry's**: files land in a
sibling directory and are renamed into place only once every one of them
arrived, so a half-fetched payload never looks like a whole one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from reverberate.store import ObjectStore, digest_of_file

__all__ = ["INDEX", "fetch_payload", "publish_payload", "remote_prefix"]

#: The file whose presence means "a whole payload is here". Written last on
#: publication and fetched with the rest, for the reason ``vox_store`` gives:
#: writing the index first would advertise tiles that have not arrived.
INDEX = "rooms.json"


def remote_prefix(run: str) -> str:
    """Where ``run``'s payload lives in the store, under the project prefix."""
    return f"runs/{run}/voxels/"


def _files(root: Path) -> list[Path]:
    """Every file of a payload, index last."""
    found = sorted(p for p in root.rglob("*") if p.is_file())
    return [p for p in found if p.name != INDEX] + [p for p in found if p.name == INDEX]


def publish_payload(store: ObjectStore, run: str, root: Path) -> dict[str, str]:
    """Upload one run's payload directory, and return each file's digest.

    Idempotent: a file already in the store is not re-sent, which matters at
    a gigabyte and a half a run.
    """
    root = Path(root)
    if not (root / INDEX).is_file():
        raise ValueError(f"no {INDEX} at {root}, so there is no payload to publish")
    prefix = remote_prefix(run)
    digests: dict[str, str] = {}
    for path in _files(root):
        name = path.relative_to(root).as_posix()
        digests[name] = digest_of_file(path)
        key = f"{prefix}{name}"
        if name != INDEX and store.exists(key):
            continue
        store.put_file(key, path)
    return digests


def fetch_payload(store: ObjectStore, run: str, into: Path) -> Path | None:
    """Pull ``run``'s payload into ``into``, or return ``None`` if there is none.

    The index is read from the store first and names every file, so nothing
    here has to list a prefix: a store that can only ``get`` still works, and
    the fetch cannot half-succeed on a listing that raced a publication.
    """
    import json

    prefix = remote_prefix(run)
    into = Path(into)
    if not store.exists(f"{prefix}{INDEX}"):
        return None

    staging = into.with_name(f".{into.name}.fetch")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        index = staging / INDEX
        store.get_file(f"{prefix}{INDEX}", index)
        record = json.loads(index.read_text())
        for room in record.get("rooms", []):
            for tier in ("fine", "coarse"):
                for tile in room[tier]["tiles"]:
                    for field in ("corners_url", "index_url", "label_url"):
                        name = f"{room['dir']}/{tile[field]}"
                        target = staging / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        store.get_file(f"{prefix}{name}", target)
        if into.exists():
            shutil.rmtree(into, ignore_errors=True)
        staging.rename(into)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return into
