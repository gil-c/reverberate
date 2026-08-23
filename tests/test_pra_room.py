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
    HIT_TARGET,
    MeshMaterialAssignment,
    build_room,
    eyring_rt60,
    sabine_rt60,
    simulate_and_validate,
    simulate_pairs,
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


def test_simulate_pairs_matches_pairs_and_does_not_cross_them() -> None:
    """Amortising the room over several pairs must not mix them up.

    pyroomacoustics computes a response for every source/microphone
    combination, so the diagonal has to be selected deliberately. Getting that
    wrong is silent: the responses look plausible, they are simply not the
    ones that were asked for. Two pairs at very different separations make the
    mistake visible, because the direct sound arrives at different times.
    """
    box = _box_room((6.0, 3.0, 6.0))
    assignments = [
        MeshMaterialAssignment(mesh=box, material=pra.Material(0.2), name="shell"),
    ]
    near = (np.array([-0.5, 1.2, 0.0]), np.array([0.5, 1.2, 0.0]))
    far = (np.array([-2.4, 1.2, -2.4]), np.array([2.4, 1.2, 2.4]))

    responses = simulate_pairs(assignments, [near, far], max_order=1, n_rays=500)

    assert len(responses) == 2
    assert responses[0].bands.direct_time < responses[1].bands.direct_time


def test_the_same_seed_reproduces_the_same_response() -> None:
    """The ray tracer is stochastic, and its randomness is global to the process.

    Two runs of an identical scene do not agree, because diffuse reflection
    draws from one generator inside libroom. Measured on this box, the scatter
    is of the same size as the effects the dataset exists to measure, so
    pinning it is not tidiness: without a seed, an A/B between two geometries
    reports noise and calls it a finding.
    """
    box = _box_room((5.0, 3.0, 4.0))
    assignments = [MeshMaterialAssignment(mesh=box, material=pra.Material(0.25), name="shell")]
    pair = (np.array([-1.5, 1.2, -1.0]), np.array([1.5, 1.2, 1.0]))

    first = simulate_pairs(assignments, [pair], max_order=2, n_rays=200, seed=7)
    second = simulate_pairs(assignments, [pair], max_order=2, n_rays=200, seed=7)

    assert np.allclose(first[0].rir, second[0].rir)


def test_different_seeds_give_different_responses() -> None:
    """The converse, so that the test above cannot pass by the ray tracer being
    accidentally deterministic; that would make the seed look effective when it
    was doing nothing."""
    box = _box_room((5.0, 3.0, 4.0))
    assignments = [MeshMaterialAssignment(mesh=box, material=pra.Material(0.25), name="shell")]
    pair = (np.array([-1.5, 1.2, -1.0]), np.array([1.5, 1.2, 1.0]))

    first = simulate_pairs(assignments, [pair], max_order=2, n_rays=200, seed=7)
    second = simulate_pairs(assignments, [pair], max_order=2, n_rays=200, seed=8)

    length = min(len(first[0].rir), len(second[0].rir))
    assert not np.allclose(first[0].rir[:length], second[0].rir[:length])


def test_the_image_source_part_is_unaffected_by_the_other_pairs() -> None:
    """Amortising must not change the deterministic half of the simulation.

    The ray traced half cannot be compared pair by pair, since adding a source
    shifts everyone's random draws. The image source half has no randomness at
    all, so it is the part where "one room, several pairs" can be held to
    giving exactly what "one room, one pair" gives — and it is the part that
    carries the early reflections.
    """
    box = _box_room((5.0, 3.0, 4.0))
    assignments = [MeshMaterialAssignment(mesh=box, material=pra.Material(0.25), name="shell")]
    pair = (np.array([-1.5, 1.2, -1.0]), np.array([1.5, 1.2, 1.0]))
    other = (np.array([-1.0, 1.2, 1.0]), np.array([1.0, 1.2, -1.0]))

    alone = _image_source_rir(assignments, [pair])
    together = _image_source_rir(assignments, [pair, other])

    length = min(len(alone), len(together))
    assert np.allclose(alone[:length], together[:length])


def test_simulate_pairs_returns_nothing_for_no_pairs() -> None:
    assert simulate_pairs([], []) == []


def _image_source_rir(
    assignments: list[MeshMaterialAssignment],
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """The first pair's response with ray tracing switched off after the build."""
    room = build_room(assignments, max_order=2, n_rays=200)
    room.simulator_state["rt_needed"] = False
    for source, _ in pairs:
        room.add_source(source)
    room.add_microphone_array(np.array([mic for _, mic in pairs]).T)
    room.image_source_model()
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=float)


def _shell(extents: tuple[float, float, float]) -> list[MeshMaterialAssignment]:
    mesh = _box_room(extents)
    return [MeshMaterialAssignment(mesh=mesh, material=pra.Material(0.25), name="shell")]


def test_the_ray_count_grows_with_the_room_volume() -> None:
    """A fixed ray count cannot be right for every room.

    The number of rays needed to reach a given hit rate per histogram bin is
    proportional to the volume, so hard-coding one makes the noise floor a
    function of room size: tuned on a small room it silently under-samples a
    large one, and the failure is invisible because an under-sampled room
    returns a plausible, merely wrong, decay time.
    """
    small = build_room(_shell((4.0, 3.0, 5.0))).rt_args["n_rays"]
    large = build_room(_shell((8.0, 3.0, 10.0))).rt_args["n_rays"]
    assert large > 3 * small


def test_the_ray_count_clears_the_libraries_own_minimum() -> None:
    """pyroomacoustics sizes for 20 hits per bin; that is below our threshold.

    Twenty is enough to draw one decay curve but not to tell two apart, so the
    sizing here deliberately exceeds the library default rather than accepting
    it. Measured on scene 102344049, the default left the spread of T30 across
    seeds at 6-14 % against an acceptance threshold of 5 %.
    """
    assignments = _shell((4.0, 3.0, 5.0))
    library_default = build_room(assignments, hit_target=20).rt_args["n_rays"]
    ours = build_room(assignments).rt_args["n_rays"]
    assert ours == pytest.approx(library_default * HIT_TARGET / 20, rel=0.01)


def test_a_larger_receiver_needs_fewer_rays() -> None:
    """The receiver radius is a cost knob, and the trade is quadratic.

    Rays are counted as heard when they cross a sphere around the microphone,
    so the hit rate goes as its cross-section. Measurement showed the answer is
    unchanged once converged, which is what makes widening the sphere a way to
    buy convergence rather than a way to bias the result.
    """
    assignments = _shell((4.0, 3.0, 5.0))
    narrow = build_room(assignments, receiver_radius=0.25).rt_args["n_rays"]
    wide = build_room(assignments, receiver_radius=1.0).rt_args["n_rays"]
    assert wide == pytest.approx(narrow / 16, rel=0.05)
