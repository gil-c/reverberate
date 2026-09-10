"""One template's simulated mesh on disk, shared by every apartment. See ADR 0011.

Turning a collision proxy into the mesh the solver receives -- a boolean union,
then :mod:`reverberate.geometry.carve` -- costs about a minute per template, and
the dataset places 15 684 of them across 53 021 instances. The mesh depends on
the template alone, so it is kept once, keyed on the code that decides it, and
``sim/<template>.glb`` in every assembled scene is a symlink to it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import trimesh

from reverberate.geometry.carve import CarveResult
from reverberate.settings import data_root

__all__ = [
    "CACHE_NAME",
    "SOURCES",
    "CachedCollider",
    "cache_root",
    "entry_paths",
    "load_entry",
    "stamp",
    "store_entry",
]

#: Subdirectory of the data root's cache holding per-template meshes.
CACHE_NAME = "colliders"

#: What decides a template's mesh: the file loaded, the union, the carve and the
#: closure test. Not the whole of ``sim_geometry``: a report string must not
#: discard fifteen thousand entries, which is why ``outer_surface`` is its own file.
SOURCES = (
    "geometry/carve.py",
    "geometry/hssd_assets.py",
    "geometry/orientation.py",
    "geometry/outer_surface.py",
)


def cache_root() -> Path:
    """``<data root>/cache/colliders``, created if missing."""
    path = data_root() / "cache" / CACHE_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def stamp() -> str:
    """A short hash over the source of every module in :data:`SOURCES`."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class CachedCollider:
    """One template's simulated mesh, and what the union and carve did to it."""

    mesh: trimesh.Trimesh
    #: Whether the boolean union of the convex bodies held. False means the
    #: mesh still carries its buried interior faces, which the scene report
    #: names rather than hides.
    merged: bool
    carve: CarveResult


def entry_paths(template: str) -> tuple[Path, Path]:
    """The mesh file and the record file for ``template``, under the stamp.

    Built by concatenation rather than by ``with_suffix``, which would read the
    stamp itself as the extension and hand back an unstamped name: entries made
    under two different rules would then be the same file.
    """
    base = cache_root() / f"{template}.{stamp()}"
    return Path(f"{base}.glb"), Path(f"{base}.json")


def load_entry(template: str) -> CachedCollider | None:
    """What is on disk for ``template``, or ``None`` when nothing is."""
    mesh_file, record_file = entry_paths(template)
    if not (mesh_file.is_file() and record_file.is_file()):
        return None
    # An unreadable entry is treated as absent, so the caller rebuilds and
    # overwrites it: an rsync interrupted with --partial left a zero-byte mesh
    # under its final name, and it took down every apartment placing it.
    try:
        record = json.loads(record_file.read_text())
        loaded = trimesh.load(mesh_file, force="mesh")
    except Exception:  # noqa: BLE001 - any unreadable entry is a missing one
        return None
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    return CachedCollider(
        mesh=loaded,
        merged=bool(record["merged"]),
        carve=CarveResult(mesh=loaded, **record["carve"]),
    )


def store_entry(template: str, mesh: trimesh.Trimesh, merged: bool, carve: CarveResult) -> None:
    """Write one template's entry, the record last since it marks the entry whole.

    Staged under a name carrying the process id: a restarted pool can retry a
    template still in flight, and a shared staging name made one writer rename
    a file the other had already moved.
    """
    mesh_file, record_file = entry_paths(template)
    exported = mesh.export(file_type="glb")
    assert isinstance(exported, bytes)
    staging = Path(f"{mesh_file}.{os.getpid()}.partial")
    staging.write_bytes(exported)
    staging.replace(mesh_file)
    record = {
        "merged": merged,
        "carve": {
            "carved": carve.carved,
            "reason": carve.reason,
            "collider_volume": carve.collider_volume,
            "carved_volume": carve.carved_volume,
            "volume_error": carve.volume_error,
            "pitch_m": carve.pitch_m,
            "leaked_at_m": carve.leaked_at_m,
        },
    }
    staging = Path(f"{record_file}.{os.getpid()}.partial")
    staging.write_text(json.dumps(record, indent=2))
    staging.replace(record_file)
