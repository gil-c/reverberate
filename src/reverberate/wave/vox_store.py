"""The voxelisation cache, shared through the object store.

The cache entry is the expensive artefact of this pipeline: minutes of CPU and
of the order of a gigabyte per room per band, and B0 paid for 52.3 minutes of it
on a machine rented for its GPU. The roadmap's W24 records that it "currently
lives per worktree with no sharing mechanism". This module is that mechanism,
and it is deliberately separate from :mod:`reverberate.wave.voxelise` so that
computing a voxelisation and sharing one remain two questions.

**The remote is the source of truth.** The lookup order is local, then remote,
then compute, and a computed entry is published. A rented instance therefore
starts from the same cache a laptop does, which is the whole point.

**An entry is published as its files, not as an archive.** The engine reads four
named files and the roadmap's own accounting is per file; tarring them would
hide a truncated member behind a whole-archive digest and make a partial fetch
impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

from reverberate.store import ObjectStore, digest_of_file
from reverberate.wave.voxelise import CACHE_FILES, CacheEntry, cache_root

__all__ = [
    "ENTRY_FILES",
    "fetch_entry",
    "publish_entry",
    "remote_prefix",
]

#: What is pushed and pulled. ``manifest.json`` last on publication and first on
#: fetch: it is what :attr:`CacheEntry.complete` looks for, so writing it before
#: the payload would advertise an entry that is not there.
ENTRY_FILES = (*CACHE_FILES, "manifest.json")


def remote_prefix(key: str) -> str:
    """Where entry ``key`` lives in the store, relative to the project prefix."""
    return f"vox/{key}/"


def publish_entry(store: ObjectStore, entry: CacheEntry) -> dict[str, str]:
    """Upload a complete local entry, and return each file's digest.

    Idempotent: a file already present with the same digest is not re-sent,
    which matters because these are gigabytes and the roadmap's own bandwidth
    arithmetic puts a terabyte at about 22 hours.
    """
    if not entry.complete:
        raise ValueError(f"refusing to publish an incomplete entry at {entry.path}")
    digests: dict[str, str] = {}
    for name in (*CACHE_FILES, "manifest.json"):
        key = f"{remote_prefix(entry.key)}{name}"
        local = entry.path / name
        digest = digest_of_file(local)
        digests[name] = digest
        if name != "manifest.json" and store.exists(key):
            continue
        store.put_file(key, local)
    return digests


def fetch_entry(store: ObjectStore, key: str) -> CacheEntry | None:
    """Pull entry ``key`` into the local cache, or return ``None`` if the store has none.

    Publication is atomic in the same sense as the local voxelisation: files
    land in a sibling directory and the entry is renamed into place only once
    every one of them arrived.
    """
    remote = remote_prefix(key)
    if not all(store.exists(f"{remote}{name}") for name in ENTRY_FILES):
        return None

    root = cache_root()
    staging = root / f".{key}.fetch"
    if staging.exists():
        _remove_tree(staging)
    staging.mkdir(parents=True)
    try:
        for name in ENTRY_FILES:
            store.get_file(f"{remote}{name}", staging / name)
        destination = root / key
        if destination.exists():
            _remove_tree(destination)
        staging.rename(destination)
    except Exception:
        _remove_tree(staging)
        raise

    manifest_file = root / key / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    return CacheEntry(path=root / key, key=key, manifest=manifest)


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
