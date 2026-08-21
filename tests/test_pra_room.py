"""Tests confirming pyroomacoustics accepts reconstructed geometry and that
per-triangle material assignment is actually taken into account (not
silently ignored). Synthetic box geometry only, fast and offline, per the
project's hard constraints (roadmap section 3): no downloaded dataset
dependency, whole file runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra
import pytest
import trimesh

from reverberate.geometry.pra_room import (
    MeshMaterialAssignment,
    eyring_rt60,
    sabine_rt60,
    simulate_and_validate,
    walls_from_mesh,
)


def _box_room(extents: tuple[float, float, float] = (4.0, 3.0, 5.0)) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=list(extents))
    # box.creation centres extents on the origin; shift so floor is y=0.
    box.apply_translation([0.0, extents[1] / 2.0, 0.0])
    assert box.is_watertight
    result: trimesh.Trimesh = box
    return result


def test_walls_from_mesh_creates_one_wall_per_triangle() -> None:
    box = _box_room()
    material = pra.Material("hard_surface")
    assignment = MeshMaterialAssignment(mesh=box, material=material, name="shell")

    walls = walls_from_mesh(assignment)

    assert len(walls) == len(box.faces)
    assert all(isinstance(w, pra.Wall) for w in walls)


def test_simulate_and_validate_loads_geometry_and_produces_a_rir() -> None:
    """Direct answer to "do our geometries load into pyroomacoustics": build
    a room from a Trimesh mesh via wall_factory, run the hybrid ISM + ray
    tracing simulation exactly as in the roadmap (section 5.1), and confirm
    it produces a finite, physically plausible RT60.
    """
    box = _box_room()
    material = pra.Material("hard_surface")
    assignment = MeshMaterialAssignment(mesh=box, material=material, name="shell")

    result = simulate_and_validate(
        [assignment],
        source=np.array([1.0, 1.5, 1.0]),
        mic=np.array([-1.0, 1.5, -1.0]),
        max_order=2,
        n_rays=2000,
    )

    assert np.isfinite(result.rt60_broadband)
    assert result.rt60_broadband > 0.0
    # Sanity: simulated RT60 should be in the same order of magnitude as the
    # analytic Sabine/Eyring estimate for this simple convex room.
    assert result.rt60_broadband == pytest.approx(result.sabine_rt60, rel=2.0)


def test_different_materials_per_face_change_the_simulated_rt60() -> None:
    """Direct answer to "can we assign different acoustic coefficients to
    different labels and does pyroomacoustics take them into account":
    the same geometry simulated with a reflective material must produce a
    materially longer RT60 than with an absorptive material.
    """
    box = _box_room()
    source = np.array([1.0, 1.5, 1.0])
    mic = np.array([-1.0, 1.5, -1.0])

    reflective = MeshMaterialAssignment(
        mesh=box, material=pra.Material("hard_surface"), name="shell"
    )
    absorptive = MeshMaterialAssignment(
        mesh=box, material=pra.Material("curtains_velvet"), name="shell"
    )

    result_reflective = simulate_and_validate(
        [reflective], source=source, mic=mic, max_order=2, n_rays=2000
    )
    result_absorptive = simulate_and_validate(
        [absorptive], source=source, mic=mic, max_order=2, n_rays=2000
    )

    assert result_reflective.mean_absorption < result_absorptive.mean_absorption
    # The reflective room must ring out noticeably longer. A generous
    # threshold (at least 30% longer) avoids flakiness from ray tracing
    # stochasticity while still being a meaningful regression guard.
    assert result_reflective.rt60_broadband > 1.3 * result_absorptive.rt60_broadband


def test_mixed_materials_on_different_meshes_are_each_respected() -> None:
    """Two separate meshes (e.g. room shell vs one piece of furniture) with
    different materials must both contribute their own coefficients, not
    have one material silently override the other.
    """
    shell = _box_room()
    # A small "obstacle" box entirely inside the shell, standing in for a
    # piece of furniture with its own, different material.
    obstacle = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    obstacle.apply_translation([0.0, 0.25, 0.0])

    source = np.array([1.0, 1.5, 1.0])
    mic = np.array([-1.0, 1.5, -1.0])

    all_hard = [
        MeshMaterialAssignment(mesh=shell, material=pra.Material("hard_surface"), name="shell"),
        MeshMaterialAssignment(
            mesh=obstacle, material=pra.Material("hard_surface"), name="obstacle"
        ),
    ]
    mixed = [
        MeshMaterialAssignment(mesh=shell, material=pra.Material("hard_surface"), name="shell"),
        MeshMaterialAssignment(
            mesh=obstacle, material=pra.Material("curtains_velvet"), name="obstacle"
        ),
    ]

    result_all_hard = simulate_and_validate(
        all_hard, source=source, mic=mic, max_order=1, n_rays=2000
    )
    result_mixed = simulate_and_validate(mixed, source=source, mic=mic, max_order=1, n_rays=2000)

    # Giving the small obstacle an absorptive material must raise the area
    # weighted mean absorption versus the all-hard-surface baseline, proving
    # the per-mesh material was actually used in the aggregate, not ignored.
    assert result_mixed.mean_absorption > result_all_hard.mean_absorption


@pytest.mark.parametrize(
    "volume, surface_area, mean_absorption, expected",
    [
        (100.0, 100.0, 0.161, pytest.approx(1.0, rel=1e-6)),
        (60.0, 94.0, 0.0, float("inf")),
    ],
)
def test_sabine_formula(
    volume: float, surface_area: float, mean_absorption: float, expected: float
) -> None:
    assert sabine_rt60(volume, surface_area, mean_absorption) == expected


def test_eyring_formula_approaches_sabine_at_low_absorption() -> None:
    volume, surface_area = 60.0, 94.0
    low_absorption = 0.02
    sabine = sabine_rt60(volume, surface_area, low_absorption)
    eyring = eyring_rt60(volume, surface_area, low_absorption)
    assert eyring == pytest.approx(sabine, rel=0.1)


def test_eyring_formula_zero_at_full_absorption() -> None:
    assert eyring_rt60(60.0, 94.0, 1.0) == 0.0
