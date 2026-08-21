"""The geometry handed to pyroomacoustics, built from what the viewer shows.

The rule this module exists to enforce is "what you see is what is simulated".
There is deliberately no second reconstruction here: the shell is
``extrude_storey`` of the same walkable outline the viewer walks on, and each
obstacle is the same collider file, under the same instance matrix, as the
acoustic view draws. If the two ever disagree, it is a bug, and
``describe_geometry`` exists so that disagreement can be measured rather than
argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import trimesh

from reverberate.geometry.apartment import Storey, extrude_storey
from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import FurnitureInstance
from reverberate.geometry.materials import material_for_label
from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.viz.room_surfaces import shell_surface_labels


@dataclass
class GeometrySummary:
    """What the simulator is about to receive, in numbers a human can check."""

    shell_faces: int
    shell_volume: float
    shell_watertight: bool
    obstacle_count: int
    obstacle_faces: int
    unresolved: list[str]

    @property
    def total_walls(self) -> int:
        """pyroomacoustics builds one wall per triangle, so this is the real cost."""
        return self.shell_faces + self.obstacle_faces

    def summary(self) -> str:
        return (
            f"shell {self.shell_faces} faces ({self.shell_volume:.0f} m3, "
            f"watertight={self.shell_watertight}), {self.obstacle_count} obstacles "
            f"totalling {self.obstacle_faces} faces, {self.total_walls} pra walls"
        )


#: Face budget per furniture obstacle. pyroomacoustics builds one wall per
#: triangle, and an undecimated apartment comes to over 400k of them, which is
#: not simulable. Decimation therefore happens *here*, in the single place both
#: the simulator and the acoustic view read from, so that reducing the cost
#: never turns the picture into a flattering version of the real input.
OBSTACLE_FACE_BUDGET = 150


def decimate(mesh: trimesh.Trimesh, face_budget: int) -> trimesh.Trimesh:
    """Reduce an obstacle to the face budget, keeping its overall shape."""
    if len(mesh.faces) <= face_budget:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=face_budget)
    except Exception:
        # A mesh the decimator cannot handle is passed through whole rather
        # than dropped: an expensive obstacle is better than a missing one.
        return mesh


@lru_cache(maxsize=512)
def simulation_collider(hssd_root: Path, template: str, face_budget: int) -> trimesh.Trimesh | None:
    """The mesh that both the simulator and the acoustic view use for a template.

    Cached because a room usually places the same template several times, and
    decimation is the expensive part.
    """
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return None
    mesh = trimesh.load(asset.collider, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return decimate(mesh, face_budget)


def shell_assignments(storey: Storey, seed: int = 0) -> list[MeshMaterialAssignment]:
    """The apartment shell, split into floor, wall and ceiling materials.

    Split by face normal rather than given one material for the whole
    enclosure: carpet underfoot and plasterboard overhead sit at opposite ends
    of the absorption range, and averaging them away would flatten the signal
    the model is meant to learn.
    """
    rng = np.random.default_rng(seed)
    shell = extrude_storey(storey)
    labels = shell_surface_labels(shell)
    assignments = []
    for surface in ("floor", "wall", "ceiling"):
        selected = labels == surface
        if not selected.any():
            continue
        part = shell.submesh([np.flatnonzero(selected)], append=True)
        assert isinstance(part, trimesh.Trimesh)
        assignments.append(
            MeshMaterialAssignment(
                mesh=part,
                material=material_for_label(surface, rng),
                name=f"shell_{surface}",
            )
        )
    return assignments


def obstacle_assignments(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    seed: int = 0,
    face_budget: int = OBSTACLE_FACE_BUDGET,
) -> tuple[list[MeshMaterialAssignment], list[str]]:
    """Every piece of furniture, as its collider under its instance matrix.

    ``resolve_asset`` falls back to the render mesh when an object ships no
    ``.collider.glb``, which is HSSD's own rule and is what keeps doors and
    windows in the simulation instead of silently dropping them.
    """
    rng = np.random.default_rng(seed)
    assignments = []
    unresolved = []
    for index, instance in enumerate(instances):
        base = simulation_collider(hssd_root, instance.template_name, face_budget)
        if base is None:
            unresolved.append(instance.template_name)
            continue
        mesh = base.copy()
        mesh.apply_transform(instance.transform_matrix())
        category = category_for_template(hssd_root, instance.template_name) or "unknown"
        assignments.append(
            MeshMaterialAssignment(
                mesh=mesh,
                material=material_for_label(category, rng),
                name=f"{category}_{index}",
            )
        )
    return assignments, unresolved


def simulation_geometry(
    hssd_root: Path, storey: Storey, instances: list[FurnitureInstance], seed: int = 0
) -> tuple[list[MeshMaterialAssignment], GeometrySummary]:
    """Everything pyroomacoustics receives for one apartment storey."""
    shell = shell_assignments(storey, seed=seed)
    obstacles, unresolved = obstacle_assignments(hssd_root, instances, seed=seed)
    whole_shell = extrude_storey(storey)
    summary = GeometrySummary(
        shell_faces=sum(len(assignment.mesh.faces) for assignment in shell),
        shell_volume=float(whole_shell.volume),
        shell_watertight=bool(whole_shell.is_watertight),
        obstacle_count=len(obstacles),
        obstacle_faces=sum(len(assignment.mesh.faces) for assignment in obstacles),
        unresolved=unresolved,
    )
    return [*shell, *obstacles], summary


def build_pra_room(
    assignments: list[MeshMaterialAssignment], fs: int = 16000, max_order: int = 1
) -> pra.Room:
    """Hand the geometry to pyroomacoustics unchanged.

    ``max_order`` defaults low on purpose: with furniture colliders included,
    the image source model becomes impractical well before it becomes
    inaccurate (see the roadmap's note on max_order 3).
    """
    from reverberate.geometry.pra_room import build_room

    return build_room(assignments, fs=fs, max_order=max_order)
