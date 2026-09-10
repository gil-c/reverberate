"""One copy of each mesh the dataset places, shared by every apartment. See ADR 0011.

The render pool holds HSSD's own glTF -- a symlink where the dataset is present,
the fetched file where it is not -- and the simulated pool is
:mod:`reverberate.geometry.collider_cache`. A scene's ``assets/`` and ``sim/``
are symlinks into them, so the manifest's relative URLs keep the shape the
browser resolves against.
"""

from __future__ import annotations

from pathlib import Path

from reverberate.geometry import collider_cache
from reverberate.geometry.hssd_assets import resolve_asset
from reverberate.settings import data_root

__all__ = [
    "ASSET_DIR",
    "RENDER_NAME",
    "SIM_DIR",
    "link_into",
    "render_path",
    "render_pool",
    "sim_path",
]

#: Subdirectory of the served scene holding render meshes, as the manifest's
#: URLs name it. Kept as it was: the app resolves ``assets/<file>`` against the
#: scene it is drawing.
ASSET_DIR = "assets"

#: Subdirectory of the served scene holding the meshes the solver receives.
SIM_DIR = "sim"

#: Subdirectory of the data root's cache holding the render pool.
RENDER_NAME = "scene_assets"


def render_pool() -> Path:
    """``<data root>/cache/scene_assets``, created if missing."""
    path = data_root() / "cache" / RENDER_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_path(hssd_root: Path, template: str) -> Path | None:
    """This template's render mesh in the pool, linked to HSSD when it is here.

    ``None`` when neither the pool nor the dataset has it, which is what a
    caller counts as unresolved rather than drawing a hole in silence.
    """
    pooled = render_pool() / f"{template}.glb"
    if pooled.exists():
        return pooled
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return None
    pooled.symlink_to(asset.render.resolve())
    return pooled


def sim_path(template: str) -> Path | None:
    """This template's simulated mesh in the collider pool, if it is built."""
    mesh_file, record_file = collider_cache.entry_paths(template)
    if mesh_file.is_file() and record_file.is_file():
        return mesh_file
    return None


def link_into(scene: Path, subdirectory: str, name: str, target: Path) -> str:
    """Expose one pool file under a scene, and return the URL the manifest uses.

    A symlink rather than a copy, for the reason the pools exist: the render
    meshes of one flat run to over a hundred megabytes, and copying them per
    scene would cost the dataset several hundred gigabytes of duplicates.
    """
    directory = scene / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / name
    if not link.exists():
        link.symlink_to(target.resolve())
    return f"{subdirectory}/{name}"
