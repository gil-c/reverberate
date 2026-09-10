"""The assembled apartment, shared through the object store. See ADR 0011.

The remote is the source of truth and the local disk a read-through cache. The
two pools are published once for the whole dataset, under :data:`RENDER_PREFIX`
and :data:`SIM_PREFIX`; a scene under :func:`remote_prefix` carries only its own
four files. A fetch lands in a sibling directory and is renamed into place once
whole, so a half-fetched apartment never looks like one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reverberate.geometry import collider_cache
from reverberate.geometry.hssd_assets import resolve_asset
from reverberate.store import ObjectStore
from reverberate.viz import scene_pool

if TYPE_CHECKING:  # a runtime import would close the loop: scene_cache calls us
    from reverberate.viz.scene_cache import SceneEntry

__all__ = [
    "INDEX",
    "RENDER_PREFIX",
    "SIM_PREFIX",
    "fetch_entry",
    "fetch_index",
    "pool_files",
    "publish_entry",
    "publish_index",
    "publish_pool",
    "remote_prefix",
]

#: A scene's own files. ``entry.json`` last on publication: it is what marks a
#: scene complete, so writing it first would advertise one that is not there.
ENTRY_FILES = ("manifest.json", "shell_colour.glb", "shell_label.glb", "entry.json")

RENDER_PREFIX = "scene_pool/assets/"
SIM_PREFIX = "scene_pool/sim/"

#: The catalogue of storeys by short name. The app reads it instead of listing
#: the bucket, which would also show scenes whose meshes never landed.
INDEX = "scenes/index.json"


def remote_prefix(key: str) -> str:
    return f"scenes/{key}/"


def _pool_keys(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The render templates and the simulated templates one manifest needs."""
    instances = manifest.get("instances", [])
    render = {str(i["template"]) for i in instances}
    sim = {
        str(i["template"])
        for i in instances
        if str(i["collider_url"]).startswith(f"{scene_pool.SIM_DIR}/")
    }
    return render, sim


def publish_entry(store: ObjectStore, entry: SceneEntry, pools: bool = True) -> None:
    """Upload a complete local scene and, unless ``pools`` is false, its pool files.

    Idempotent. ``pools=False`` is for a caller that published the pools in one
    pass already, sparing an existence check per mesh per scene.
    """
    path = Path(entry.path)
    if not all((path / name).is_file() for name in ENTRY_FILES):
        raise ValueError(f"refusing to publish an incomplete scene at {path}")
    render, sim = _pool_keys(json.loads((path / "manifest.json").read_text()))
    if pools:
        for template in sorted(render):
            remote = f"{RENDER_PREFIX}{template}.glb"
            if not store.exists(remote):
                store.put_file(remote, (path / scene_pool.ASSET_DIR / f"{template}.glb").resolve())
        for template in sorted(sim):
            for local in collider_cache.entry_paths(template):
                if not store.exists(f"{SIM_PREFIX}{local.name}"):
                    store.put_file(f"{SIM_PREFIX}{local.name}", local)
    prefix = remote_prefix(entry.key)
    for name in ENTRY_FILES:
        if name == "entry.json" or not store.exists(f"{prefix}{name}"):
            store.put_file(f"{prefix}{name}", path / name)


def fetch_entry(store: ObjectStore, key: str, workers: int = 32) -> SceneEntry | None:
    """Pull scene ``key`` and the pool files it lacks, or ``None`` if the store has none.

    The manifest names every template, so nothing lists a prefix. Pool files are
    fetched on threads, and only those missing here: a flat places some 300
    meshes, one round trip each.
    """
    from reverberate.viz.scene_cache import SceneEntry, cache_root

    prefix = remote_prefix(key)
    if not all(store.exists(f"{prefix}{name}") for name in ENTRY_FILES):
        return None
    root = cache_root()
    staging = root / f".{key}.fetch"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for name in ENTRY_FILES:
            store.get_file(f"{prefix}{name}", staging / name)
        render, sim = _pool_keys(json.loads((staging / "manifest.json").read_text()))
        pool_dir = scene_pool.render_pool()
        wanted = [
            (f"{RENDER_PREFIX}{t}.glb", pool_dir / f"{t}.glb")
            for t in sorted(render)
            if not (pool_dir / f"{t}.glb").exists()
        ] + [
            (f"{SIM_PREFIX}{local.name}", local)
            for t in sorted(sim)
            for local in collider_cache.entry_paths(t)
            if not local.is_file()
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda job: store.get_file(*job), wanted))
        for t in sorted(render):
            scene_pool.link_into(staging, scene_pool.ASSET_DIR, f"{t}.glb", pool_dir / f"{t}.glb")
        for t in sorted(sim):
            mesh_file, _ = collider_cache.entry_paths(t)
            scene_pool.link_into(staging, scene_pool.SIM_DIR, f"{t}.glb", mesh_file)
        shutil.rmtree(root / key, ignore_errors=True)
        staging.rename(root / key)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    record = json.loads((root / key / "entry.json").read_text())
    return SceneEntry(path=root / key, key=key, manifest=record)


def publish_pool(store: ObjectStore, files: dict[str, Path], prefix: str, workers: int = 32) -> int:
    """Upload the pool files the store lacks. Returns how many went.

    One listing, then uploads on threads: they are bound by request latency,
    and an existence check per file across 47 000 made the first attempt take
    most of a day. Idempotent, so an interrupted run is resumed by rerunning.
    """
    known = set(store.list(prefix))
    missing = [
        (f"{prefix}{name}", path) for name, path in files.items() if f"{prefix}{name}" not in known
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: store.put_file(*job), missing))
    return len(missing)


def pool_files(
    hssd_root: Path, templates: Iterable[str]
) -> tuple[dict[str, Path], dict[str, Path]]:
    """The render and simulated files on this machine for these templates, by store name."""
    render: dict[str, Path] = {}
    sim: dict[str, Path] = {}
    for template in templates:
        asset = resolve_asset(hssd_root / "objects", template)
        if asset is not None:
            render[f"{template}.glb"] = asset.render
        mesh_file, record_file = collider_cache.entry_paths(template)
        if mesh_file.is_file() and record_file.is_file():
            sim[mesh_file.name] = mesh_file
            sim[record_file.name] = record_file
    return render, sim


def publish_index(store: ObjectStore, rows: list[dict[str, object]]) -> str:
    """Publish the catalogue. Written last, once every scene in it is stored."""
    payload = json.dumps({"scenes": rows}, indent=2, sort_keys=True).encode()
    return store.put_bytes(INDEX, payload)


def fetch_index(store: ObjectStore) -> list[dict[str, object]]:
    """The catalogue, or an empty list when nothing is published."""
    if not store.exists(INDEX):
        return []
    return list(json.loads(store.get_bytes(INDEX)).get("scenes", []))
