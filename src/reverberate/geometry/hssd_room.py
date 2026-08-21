"""Reconstruct individual HSSD rooms as watertight meshes with placed furniture.

Implements the approach documented in the project roadmap under "BREAKTHROUGH
(2026-08-21): per-room polygon extrusion solves watertightness cleanly": each
room's floor footprint (``poly_loop``) is taken from the scene's
``semantic_config.json`` and extruded to a closed prism, which is watertight
by construction regardless of the state of the underlying (broken) stage
mesh. Furniture instances are matched to rooms by a point in polygon test on
their translation, then placed using their already watertight collider mesh.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon


@dataclass(frozen=True)
class RoomRegion:
    """One room's authored floor plan, as read from semantic_config.json."""

    name: str
    label: str
    poly_loop: np.ndarray  # (N, 3) world coordinates, Y is up, ground plane is XZ
    floor_height: float
    extrusion_height: float

    @property
    def polygon_xz(self) -> Polygon:
        """The footprint polygon in world (x, z), for point in polygon tests."""
        return Polygon([(x, z) for x, _, z in self.poly_loop])

    def extrude(self) -> trimesh.Trimesh:
        """Build the watertight room shell as a vertical prism.

        ``trimesh.creation.extrude_polygon`` extrudes a Shapely (x, y)
        polygon along local +Z, producing vertices (local_x, local_y,
        local_z). We then rotate -90 degrees about the world X axis, which
        maps (local_x, local_y, local_z) -> (local_x, local_z, -local_y).
        For local_z (the extrusion height) to land on world +Y, and for
        local_y to land on world +Z without a sign flip (a pure rotation
        cannot swap two axes without also flipping one, since that swap is
        a reflection), the polygon is built with local_y = -world_z, so the
        rotation's "-local_y" term becomes +world_z as required. Verified
        numerically against a synthetic square with a known orientation.
        """
        polygon_2d = Polygon([(x, -z) for x, _, z in self.poly_loop])
        if not polygon_2d.is_valid:
            polygon_2d = polygon_2d.buffer(0)
        shell = trimesh.creation.extrude_polygon(polygon_2d, height=self.extrusion_height)
        rotate_local_to_world = trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
            angle=-np.pi / 2, direction=[1, 0, 0]
        )
        shell.apply_transform(rotate_local_to_world)
        shell.apply_translation([0.0, self.floor_height, 0.0])
        return shell


@dataclass(frozen=True)
class FurnitureInstance:
    """One placed furniture object, matched to a room."""

    template_name: str
    translation: np.ndarray  # (3,) world xyz
    rotation_wxyz: np.ndarray  # (4,) quaternion, scalar first (Habitat convention)
    non_uniform_scale: np.ndarray  # (3,)

    def transform_matrix(self) -> np.ndarray:
        w, x, y, z = self.rotation_wxyz
        rotation = trimesh.transformations.quaternion_matrix(  # type: ignore[no-untyped-call]
            [w, x, y, z]
        )
        scale = np.diag([*self.non_uniform_scale, 1.0])
        translation = trimesh.transformations.translation_matrix(  # type: ignore[no-untyped-call]
            self.translation
        )
        result: np.ndarray = translation @ rotation @ scale
        return result


def load_regions(semantic_config_path: Path) -> list[RoomRegion]:
    data = json.loads(semantic_config_path.read_text())
    regions = []
    for entry in data["region_annotations"]:
        regions.append(
            RoomRegion(
                name=entry["name"],
                label=entry["label"],
                poly_loop=np.array(entry["poly_loop"], dtype=float),
                floor_height=float(entry["floor_height"]),
                extrusion_height=float(entry["extrusion_height"]),
            )
        )
    return regions


def load_object_instances(scene_instance_path: Path) -> list[FurnitureInstance]:
    data = json.loads(scene_instance_path.read_text())
    instances = []
    for entry in data["object_instances"]:
        instances.append(
            FurnitureInstance(
                template_name=entry["template_name"],
                translation=np.array(entry["translation"], dtype=float),
                rotation_wxyz=np.array(entry["rotation"], dtype=float),
                non_uniform_scale=np.array(entry["non_uniform_scale"], dtype=float),
            )
        )
    return instances


def match_instances_to_regions(
    regions: list[RoomRegion], instances: list[FurnitureInstance]
) -> dict[int, list[FurnitureInstance]]:
    """Point-in-polygon match each instance's translation (XZ plane) to a region.

    An instance whose translation falls in no region, or in more than one, is
    assigned to the first containing region if any, else dropped. Both cases
    are counted so the caller can report them rather than silently losing
    furniture.
    """
    polygons = [region.polygon_xz for region in regions]
    assignment: dict[int, list[FurnitureInstance]] = {i: [] for i in range(len(regions))}
    unmatched = 0
    for instance in instances:
        x, _, z = instance.translation
        point = Point(x, z)
        matches = [i for i, polygon in enumerate(polygons) if polygon.contains(point)]
        if not matches:
            unmatched += 1
            continue
        assignment[matches[0]].append(instance)
    if unmatched:
        import logging

        logging.getLogger(__name__).info(
            "%d of %d furniture instances matched no room polygon", unmatched, len(instances)
        )
    return assignment


def load_object_category(objects_dir: Path, template_name: str) -> str | None:
    """Look up an object's condensed semantic category from the objects metadata CSV."""
    csv_path = objects_dir.parent / "metadata" / "hssd_obj_semantics_condensed.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            if row and row[0] == template_name:
                return row[3] or row[4] or None
    return None


def load_collider_mesh(objects_dir: Path, template_name: str) -> trimesh.Trimesh:
    """Load the watertight collision mesh for a furniture template.

    Colliders are sharded into subdirectories named after the template's
    first hex character (``objects/8/8bb3...collider.glb``).
    """
    shard = template_name[0]
    path = objects_dir / shard / f"{template_name}.collider.glb"
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a Trimesh from {path}, got {type(mesh)!r}")
    return mesh
