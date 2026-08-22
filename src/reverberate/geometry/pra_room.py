"""Build a pyroomacoustics Room from reconstructed HSSD geometry and simulate it.

Follows the official ``examples/room_from_stl.py`` pattern: one
``pra.wall_factory`` call per triangle, for both the room shell and every
furniture obstacle, each with its own per-triangle absorption and scattering
coefficients. Uses the hybrid image source plus ray tracing mode required by
the roadmap (section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
import trimesh


@dataclass
class MeshMaterialAssignment:
    """One mesh (room shell or one furniture obstacle) with its material."""

    mesh: trimesh.Trimesh
    material: pra.Material
    name: str = ""
    #: Set when the mesh was decimated and its absorption rescaled to keep the
    #: obstacle's absorbing power. Carried here so the viewer and the audit
    #: panel quote the simulator's own figures rather than recomputing them.
    compensation: object | None = None


def walls_from_mesh(assignment: MeshMaterialAssignment) -> list[pra.wall.Wall]:
    """One pra wall per triangle, as in the official STL example."""
    absorption = np.asarray(assignment.material.energy_absorption["coeffs"])
    scattering = np.asarray(assignment.material.scattering["coeffs"])
    triangles = assignment.mesh.vertices[assignment.mesh.faces]  # (n_faces, 3, 3)
    return [
        pra.wall_factory(
            triangle.T,
            absorption,
            scattering,
            name=f"{assignment.name}_{i}" if assignment.name else "",
        )
        for i, triangle in enumerate(triangles)
    ]


def build_room(
    assignments: list[MeshMaterialAssignment],
    fs: int = 16000,
    max_order: int = 3,
    n_rays: int = 10000,
    receiver_radius: float = 0.15,
) -> pra.Room:
    walls: list[pra.wall.Wall] = []
    for assignment in assignments:
        walls.extend(walls_from_mesh(assignment))
    room = pra.Room(
        walls,
        fs=fs,
        max_order=max_order,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(n_rays=n_rays, receiver_radius=receiver_radius)
    return room


@dataclass
class SimulationResult:
    rt60_broadband: float
    room_volume: float
    room_surface_area: float
    mean_absorption: float
    sabine_rt60: float
    eyring_rt60: float


def sabine_rt60(volume: float, surface_area: float, mean_absorption: float) -> float:
    """Sabine formula, seconds. Guard against zero/near-total absorption."""
    if mean_absorption <= 0:
        return float("inf")
    return 0.161 * volume / (surface_area * mean_absorption)


def eyring_rt60(volume: float, surface_area: float, mean_absorption: float) -> float:
    """Eyring formula, seconds. More accurate than Sabine at high absorption."""
    if mean_absorption >= 1.0:
        return 0.0
    denom = -surface_area * np.log(1.0 - mean_absorption)
    if denom <= 0:
        return float("inf")
    return 0.161 * volume / float(denom)


def simulate_and_validate(
    assignments: list[MeshMaterialAssignment],
    source: np.ndarray,
    mic: np.ndarray,
    fs: int = 16000,
    max_order: int = 3,
    n_rays: int = 10000,
    room_volume: float | None = None,
) -> SimulationResult:
    """Run the hybrid ISM + ray tracing simulation and compute the Sabine/Eyring guard.

    ``room_volume`` should be passed explicitly whenever the shell is split
    across more than one ``MeshMaterialAssignment`` (for example floor, wall
    and ceiling assigned separately so each can carry its own material): each
    part's own ``.mesh.volume`` is not the enclosed air volume in that case,
    only the volume of a thin, degenerate slab. When omitted, the volume of
    ``assignments[0].mesh`` is used, which is only correct when the first
    assignment is the single, complete, watertight shell.
    """
    room = build_room(assignments, fs=fs, max_order=max_order, n_rays=n_rays)
    room.add_source(source)
    room.add_microphone_array(np.c_[mic])
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()

    rt60 = float(room.measure_rt60()[0, 0])

    total_area = 0.0
    weighted_absorption = 0.0
    for assignment in assignments:
        area = float(assignment.mesh.area)
        mean_coeff = float(np.mean(assignment.material.energy_absorption["coeffs"]))
        total_area += area
        weighted_absorption += area * mean_coeff
    mean_absorption = weighted_absorption / total_area if total_area else 0.0

    if room_volume is not None:
        volume = room_volume
    else:
        # Fallback for the common single-mesh-shell case (roadmap 5.1's own
        # example): the first assignment, by convention, carries the whole
        # enclosed shell. Callers with a split shell must pass room_volume.
        volume = float(assignments[0].mesh.volume) if assignments else 0.0

    return SimulationResult(
        rt60_broadband=rt60,
        room_volume=volume,
        room_surface_area=total_area,
        mean_absorption=mean_absorption,
        sabine_rt60=sabine_rt60(volume, total_area, mean_absorption),
        eyring_rt60=eyring_rt60(volume, total_area, mean_absorption),
    )
