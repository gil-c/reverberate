"""Per-room reconstruction from the HSSD (Habitat Synthetic Scene Dataset).

See ROADMAP.md, "BREAKTHROUGH" and "Immediate next steps for whoever picks up
Phase 1": HSSD's per-scene ``stages/*.glb`` render mesh is not watertight (0
of 168 sampled scenes), but each scene's
``semantics/scenes/<id>.semantic_config.json`` carries an authored floor plan
per room (``region_annotations[].poly_loop``), which extrudes into a closed,
watertight shell with no repair needed. Furniture placement comes from
``scenes/<id>.scene_instance.json``, matched to a room by testing whether the
object's translation falls inside that room's floor polygon (XZ plane, Y up).

This module is the "pure-ish" reconstruction step: given paths into an HSSD
checkout, it returns an assembled room as a watertight shell mesh plus a list
of placed, labelled furniture meshes. It does no simulation and no
decimation, both later pipeline stages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh
from shapely.geometry import Point, Polygon

__all__ = [
    "SEMANTIC_UNKNOWN",
    "RegionAnnotation",
    "ObjectInstance",
    "LabelledMesh",
    "RoomReconstruction",
    "load_region_annotations",
    "load_object_instances",
    "assign_objects_to_region",
    "quaternion_translation_scale_to_matrix",
    "extrude_region_shell",
    "load_semantic_lexicon",
    "object_semantic_category",
    "load_object_collider",
    "build_room",
    "list_region_names",
]

#: Category reported when an object's ``semantic_id`` has no entry in the
#: lexicon, or the lexicon itself was not supplied.
SEMANTIC_UNKNOWN = "unknown"

#: HSSD object directories are sharded by the first hex character of the
#: object hash, e.g. ``objects/1/<hash>.glb``. Articulated objects reference
#: a template like ``<hash>_part_3`` with no ``object_config.json`` of its
#: own; this strips the suffix before looking up files on disk.
_PART_SUFFIX_RE = re.compile(r"_part_\d+$")


@dataclass(frozen=True)
class RegionAnnotation:
    """One room's authored floor plan, read from ``region_annotations[]``."""

    name: str
    label: str
    poly_loop_xz: list[tuple[float, float]]
    floor_height: float
    extrusion_height: float


@dataclass(frozen=True)
class ObjectInstance:
    """One placed furniture instance, read from ``object_instances[]``."""

    template_name: str
    translation: npt.NDArray[np.float64]
    rotation_wxyz: npt.NDArray[np.float64]
    non_uniform_scale: npt.NDArray[np.float64]


@dataclass(frozen=True)
class LabelledMesh:
    """A placed, world-space furniture mesh tagged with its semantic category."""

    template_name: str
    category: str
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class RoomReconstruction:
    """A watertight room shell plus the furniture placed inside it."""

    region: RegionAnnotation
    shell: trimesh.Trimesh
    furniture: list[LabelledMesh]
    skipped_instances: list[str]


def _semantic_config_path(hssd_root: Path, scene_id: str) -> Path:
    return hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"


def _scene_instance_path(hssd_root: Path, scene_id: str) -> Path:
    return hssd_root / "scenes" / f"{scene_id}.scene_instance.json"


def load_region_annotations(hssd_root: Path, scene_id: str) -> list[RegionAnnotation]:
    """Read every room's authored floor plan for one HSSD scene."""
    data = json.loads(_semantic_config_path(hssd_root, scene_id).read_text())
    regions = []
    for entry in data["region_annotations"]:
        poly_xz = [(float(p[0]), float(p[2])) for p in entry["poly_loop"]]
        regions.append(
            RegionAnnotation(
                name=entry["name"],
                label=entry.get("label", ""),
                poly_loop_xz=poly_xz,
                floor_height=float(entry["floor_height"]),
                extrusion_height=float(entry["extrusion_height"]),
            )
        )
    return regions


def list_region_names(hssd_root: Path, scene_id: str) -> list[str]:
    """Convenience: the room names available in one scene, for CLI use."""
    return [r.name for r in load_region_annotations(hssd_root, scene_id)]


def load_object_instances(hssd_root: Path, scene_id: str) -> list[ObjectInstance]:
    """Read every placed furniture instance for one HSSD scene.

    Habitat quaternions are stored ``[w, x, y, z]``, matched against
    ``trimesh.transformations.quaternion_matrix`` which uses the same order.
    """
    data = json.loads(_scene_instance_path(hssd_root, scene_id).read_text())
    instances = []
    for entry in data.get("object_instances", []):
        instances.append(
            ObjectInstance(
                template_name=entry["template_name"],
                translation=np.asarray(entry["translation"], dtype=np.float64),
                rotation_wxyz=np.asarray(entry["rotation"], dtype=np.float64),
                non_uniform_scale=np.asarray(entry["non_uniform_scale"], dtype=np.float64),
            )
        )
    return instances


def assign_objects_to_region(
    region: RegionAnnotation, instances: list[ObjectInstance]
) -> list[ObjectInstance]:
    """Return the instances whose translation (XZ plane) falls inside the
    region's floor polygon.

    An instance that falls in no room, or that sits exactly on a shared wall
    between two rooms, is silently excluded here: it is the caller's job to
    decide, across all of a scene's regions, whether that is acceptable (see
    ROADMAP.md step 1 under "Immediate next steps"). This function only
    answers the question for one region at a time.
    """
    polygon = Polygon(region.poly_loop_xz)
    return [
        instance
        for instance in instances
        if polygon.contains(Point(float(instance.translation[0]), float(instance.translation[2])))
    ]


def quaternion_translation_scale_to_matrix(
    rotation_wxyz: npt.ArrayLike,
    translation: npt.ArrayLike,
    non_uniform_scale: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Build the 4x4 world transform ``T @ R @ S`` for one placed instance.

    Scale is applied first (in the object's local frame), then rotation, then
    translation, which matches how Habitat places a unit-scale collision mesh
    into a scene.
    """
    rotation = trimesh.transformations.quaternion_matrix(  # type: ignore[no-untyped-call]
        np.asarray(rotation_wxyz, dtype=np.float64)
    )
    scale = np.eye(4)
    scale[0, 0], scale[1, 1], scale[2, 2] = np.asarray(non_uniform_scale, dtype=np.float64)
    translate = trimesh.transformations.translation_matrix(  # type: ignore[no-untyped-call]
        np.asarray(translation, dtype=np.float64)
    )
    result: npt.NDArray[np.float64] = translate @ rotation @ scale
    return result


def extrude_region_shell(region: RegionAnnotation) -> trimesh.Trimesh:
    """Extrude a room's authored floor polygon into a closed shell.

    Watertight by construction whenever the polygon is simple and valid (see
    ROADMAP.md "BREAKTHROUGH", verified on 402/402 rooms across 30 scenes).
    """
    polygon = Polygon(region.poly_loop_xz)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    shell = trimesh.creation.extrude_polygon(polygon, region.extrusion_height)
    # extrude_polygon extrudes along +Z in the polygon's local 2D frame; the
    # polygon is built from (x, z) pairs, so its local Z axis is world Y (up).
    # Swap Y and Z, then shift to the room's real floor height.
    shell.vertices = shell.vertices[:, [0, 2, 1]]
    if shell.volume < 0:
        # Swapping two coordinate columns mirrors the mesh, which flips
        # face winding and therefore the outward-normal convention
        # pyroomacoustics relies on; restore it.
        shell.invert()
    shell.apply_translation((0.0, region.floor_height, 0.0))
    return shell


def load_semantic_lexicon(hssd_root: Path) -> dict[int, str]:
    """Load the ``semantic_id -> category name`` table shared across HSSD."""
    path = hssd_root / "semantics" / "hssd-hab_semantic_lexicon.json"
    data = json.loads(path.read_text())
    return {int(entry["id"]): str(entry["name"]) for entry in data["classes"]}


def _object_config_path(hssd_root: Path, base_hash: str) -> Path:
    return hssd_root / "objects" / base_hash[0] / f"{base_hash}.object_config.json"


def _strip_part_suffix(template_name: str) -> str:
    return _PART_SUFFIX_RE.sub("", template_name)


def object_semantic_category(hssd_root: Path, template_name: str, lexicon: dict[int, str]) -> str:
    """Look up an object instance's semantic category by its ``semantic_id``.

    Some ``template_name`` values carry an articulation ``_part_N`` suffix
    with no matching ``object_config.json`` of their own (the config is
    shared by the whole articulated object); the suffix is stripped before
    lookup.
    """
    base_hash = _strip_part_suffix(template_name)
    config_path = _object_config_path(hssd_root, base_hash)
    if not config_path.exists():
        return SEMANTIC_UNKNOWN
    config = json.loads(config_path.read_text())
    semantic_id = config.get("semantic_id")
    if semantic_id is None:
        return SEMANTIC_UNKNOWN
    return lexicon.get(int(semantic_id), SEMANTIC_UNKNOWN)


def load_object_collider(hssd_root: Path, template_name: str) -> trimesh.Trimesh | None:
    """Load one placed instance's watertight collision mesh, flattened.

    Returns ``None`` (rather than raising) when the referenced collider file
    does not exist, which happens for a minority of articulated
    ``_part_N`` instances (see ROADMAP.md): the caller is expected to skip
    and log these rather than fail the whole room.
    """
    base_hash = _strip_part_suffix(template_name)
    path = hssd_root / "objects" / base_hash[0] / f"{base_hash}.collider.glb"
    if not path.exists():
        return None
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    return loaded


def build_room(hssd_root: Path, scene_id: str, region_name: str) -> RoomReconstruction:
    """Reconstruct one named room from an HSSD scene: shell plus furniture.

    Raises ``KeyError`` if ``region_name`` is not a room in this scene.
    """
    regions = {r.name: r for r in load_region_annotations(hssd_root, scene_id)}
    if region_name not in regions:
        raise KeyError(
            f"no region named {region_name!r} in scene {scene_id!r}; available: {sorted(regions)}"
        )
    region = regions[region_name]
    shell = extrude_region_shell(region)

    all_instances = load_object_instances(hssd_root, scene_id)
    assigned = assign_objects_to_region(region, all_instances)
    lexicon = load_semantic_lexicon(hssd_root)

    furniture: list[LabelledMesh] = []
    skipped: list[str] = []
    for instance in assigned:
        collider = load_object_collider(hssd_root, instance.template_name)
        if collider is None:
            skipped.append(instance.template_name)
            continue
        transform = quaternion_translation_scale_to_matrix(
            instance.rotation_wxyz, instance.translation, instance.non_uniform_scale
        )
        placed = collider.copy()
        placed.apply_transform(transform)
        category = object_semantic_category(hssd_root, instance.template_name, lexicon)
        furniture.append(
            LabelledMesh(template_name=instance.template_name, category=category, mesh=placed)
        )

    return RoomReconstruction(
        region=region, shell=shell, furniture=furniture, skipped_instances=skipped
    )
