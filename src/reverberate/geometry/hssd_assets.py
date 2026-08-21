"""Resolve HSSD template names to their asset files on disk.

The scene dataset config (``hssd-hab.scene_dataset_config.json``) declares two
object search paths, ``objects/*`` and ``objects/decomposed/*``, so a template
name is not simply a file under ``objects/<first hex char>/``. Three layouts
occur in practice and all three carry furniture that a room reconstruction
must account for:

``objects/<shard>/<template>.glb``
    The common case: one self contained object, sharded by the template's
    first character.
``objects/openings/<template>.glb``
    Doors and windows. Their template names are not hashes (``219-1``), so
    the shard rule does not apply to them at all.
``objects/decomposed/<base hash>/<base hash>_part_<n>.glb``
    Articulated objects split into parts, each placed as its own instance.

Resolving only the first layout silently drops roughly a quarter of a scene's
instances, doors and windows included, which is both visually obvious and
acoustically wrong (glass and wood are the extremes of the absorption range).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Marker separating an articulated object's base hash from its part index.
PART_MARKER = "_part_"


@dataclass(frozen=True)
class AssetPaths:
    """Where one template's meshes live, and which layout it came from."""

    render: Path
    collider: Path
    layout: str
    #: True when no ``.collider.glb`` exists and the render mesh is standing in
    #: for it. This is the dataset's own rule, not a workaround: an
    #: ``object_config.json`` without a ``collision_asset`` key means Habitat
    #: collides against the render asset itself. 1790 of HSSD's 16539 objects
    #: are in that case, doors and windows among them, and treating them as
    #: colliderless drops them from the simulation while leaving them visible.
    collider_is_render: bool


def candidate_directories(objects_dir: Path, template_name: str) -> list[tuple[Path, str]]:
    """Directories that may hold ``template_name``, most likely first."""
    candidates = [
        (objects_dir / template_name[0], "shard"),
        (objects_dir / "openings", "openings"),
    ]
    base_hash = template_name.split(PART_MARKER)[0]
    if base_hash != template_name:
        candidates.insert(0, (objects_dir / "decomposed" / base_hash, "decomposed"))
    else:
        candidates.append((objects_dir / "decomposed" / base_hash, "decomposed"))
    return candidates


def resolve_asset(objects_dir: Path, template_name: str) -> AssetPaths | None:
    """Find a template's render mesh, and the mesh to collide and simulate with.

    Returns ``None`` when no render mesh is found anywhere, so callers can
    count and report unresolved instances instead of losing them silently.
    """
    for directory, layout in candidate_directories(objects_dir, template_name):
        render = directory / f"{template_name}.glb"
        if not render.is_file():
            continue
        collider = directory / f"{template_name}.collider.glb"
        has_collider = collider.is_file()
        return AssetPaths(
            render=render,
            collider=collider if has_collider else render,
            layout=layout,
            collider_is_render=not has_collider,
        )
    return None


@lru_cache(maxsize=4)
def semantic_categories(metadata_csv: Path) -> dict[str, str]:
    """Map every object hash to its condensed semantic category, in one pass.

    Cached because the table has ~18k rows and callers ask per instance.
    """
    categories: dict[str, str] = {}
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 5 or not row[0]:
                continue
            category = row[3].strip() or row[4].strip()
            if category:
                categories[row[0]] = category
    return categories


def category_for_template(hssd_root: Path, template_name: str) -> str | None:
    """The condensed semantic category of a template.

    An articulated object's parts inherit their base object's category, since
    the metadata table is keyed by the base hash only.
    """
    table = semantic_categories(hssd_root / "metadata" / "hssd_obj_semantics_condensed.csv")
    if template_name in table:
        return table[template_name]
    base_hash = template_name.split(PART_MARKER)[0]
    return table.get(base_hash)
