"""Bring a backed-up part of ``data/`` back from the object store.

    PYTHONPATH=src .venv/bin/python scripts/restore_data.py runs/w42_gpu_hssd_0076
    PYTHONPATH=src .venv/bin/python scripts/restore_data.py --list

Every item listed in ``docs/data-layout.md`` has a manifest under
``reverberate/manifests/<item>.json``: each file's path under ``data/``, its key,
size and SHA-256, and any symlink the directory held. Restoring reads that manifest
and fetches each file with ``ObjectStore.get_file``, which checks the digest
before the file takes its name. A file already present with the right size is
skipped, so an interrupted restore resumes.

Credentials come from :func:`reverberate.store.shared_store`, as everywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from reverberate.store import shared_store

DATA = Path(__file__).resolve().parents[1] / "data"


def manifest_key(item: str) -> str:
    """Where the manifest of ``item`` (``runs/<run>``, ``vox/<key>``, ``external/...``) lives."""
    item = item.strip("/")
    if item.startswith("data/"):
        item = item[len("data/") :]
    if item.startswith("cache/vox/"):
        item = "vox/" + item[len("cache/vox/") :]
    return f"manifests/{item}.json"


def restore(item: str, data: Path) -> int:
    store = shared_store()
    if store is None:
        print("no store: the vault did not open", file=sys.stderr)
        return 2
    manifest = json.loads(store.get_bytes(manifest_key(item)))
    fetched = skipped = 0
    for entry in manifest["files"]:
        target = data / entry["path"]
        if target.is_file() and target.stat().st_size == entry["size"]:
            skipped += 1
            continue
        store.get_file(entry["key"], target)
        fetched += 1
    for link in manifest.get("symlinks", []):
        path = data / link["path"]
        if not path.exists() and not path.is_symlink():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(link["target"], path)
    print(f"{manifest['item']}: {fetched} fetched, {skipped} already present")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("item", nargs="?", help="e.g. runs/w42_gpu_hssd_0076 or cache/vox/<key>")
    parser.add_argument("--list", action="store_true", help="list the items that have a manifest")
    parser.add_argument("--data", type=Path, default=DATA, help="where data/ is (default: this checkout's)")
    args = parser.parse_args()
    if args.list:
        store = shared_store()
        if store is None:
            print("no store: the vault did not open", file=sys.stderr)
            return 2
        for key in store.list("manifests/"):
            print(key[len("manifests/") : -len(".json")])
        return 0
    if not args.item:
        parser.error("name an item, or pass --list")
    return restore(args.item, args.data)


if __name__ == "__main__":
    raise SystemExit(main())
