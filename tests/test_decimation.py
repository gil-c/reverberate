"""Tests for adaptive decimation and its section 5.3 validation.

The property under test is that the reduction may not quietly destroy the
things the simulation depends on. Surface area carries absorption, so losing
it is losing physics, not detail; and a flipped normal is worse than an error
because it makes the ray tracer hang rather than fail.

Synthetic geometry only: offline, fast, no dataset dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reverberate.geometry.decimation import (
    DETAIL_LEVELS,
    MAX_OBSTACLE_FACES,
    MIN_OBSTACLE_FACES,
    MIN_WAVELENGTH,
    NEAR_DISTANCE,
    DecimationReport,
    decimate_adaptive,
    decimate_to,
    face_budget,
    level_for,
    summarise,
    validate,
)


def sphere(radius: float = 0.5, subdivisions: int = 4) -> trimesh.Trimesh:
    """A dense watertight obstacle, the case a flat budget handled worst."""
    mesh: trimesh.Trimesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return mesh


def test_budget_scales_with_surface_area_not_with_a_constant() -> None:
    """A wardrobe and a vase must not receive the same budget: that was the
    flat-150 rule's core mistake."""
    small = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    large = trimesh.creation.box(extents=(2.0, 2.0, 2.0))

    assert face_budget(large, MIN_WAVELENGTH) > face_budget(small, MIN_WAVELENGTH)


def test_budget_is_bounded_at_both_ends() -> None:
    tiny = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    huge = trimesh.creation.box(extents=(50.0, 50.0, 50.0))

    assert face_budget(tiny, MIN_WAVELENGTH) == MIN_OBSTACLE_FACES
    assert face_budget(huge, MIN_WAVELENGTH) == MAX_OBSTACLE_FACES


def test_coarser_detail_length_buys_a_smaller_budget() -> None:
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))

    fine = face_budget(box, MIN_WAVELENGTH)
    coarse = face_budget(box, MIN_WAVELENGTH * 4.0)

    assert coarse < fine


def test_budget_refuses_a_nonsensical_detail_length() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        face_budget(sphere(), 0.0)


def test_another_room_is_always_the_coarsest_level() -> None:
    """A wall between listener and object has already diffused whatever it
    reflects, so distance within that room cannot make it matter more."""
    assert level_for(0.5, same_room=False) == DETAIL_LEVELS[2]
    assert level_for(50.0, same_room=False) == DETAIL_LEVELS[2]


def test_near_and_far_within_the_listener_room_differ() -> None:
    near = level_for(NEAR_DISTANCE - 0.1, same_room=True)
    far = level_for(NEAR_DISTANCE + 0.1, same_room=True)

    assert near == DETAIL_LEVELS[0]
    assert far == DETAIL_LEVELS[1]
    assert near.detail_length < far.detail_length


def test_no_level_resolves_below_the_physical_limit() -> None:
    """Section 5.3's argument only holds down to the 4 kHz wavelength; past it
    decimation would stop being physics."""
    assert min(level.detail_length for level in DETAIL_LEVELS) == MIN_WAVELENGTH


def test_validation_reports_area_loss_rather_than_hiding_it() -> None:
    original = sphere()
    reduced = decimate_to(original, 40)

    report = validate(original, reduced)

    assert report.faces_after < report.faces_before
    assert report.area_error > 0.0
    assert "area" in report.summary()


def test_a_flipped_mesh_fails_the_normals_check() -> None:
    """The check that matters most: an inverted mesh raises no error anywhere,
    it makes the ray tracer run for minutes on geometry it cannot resolve."""
    original = sphere()
    flipped = original.copy()
    flipped.invert()

    report = validate(original, flipped)

    assert not report.normals_consistent
    assert not report.acceptable()


def test_losing_watertightness_is_never_acceptable() -> None:
    report = DecimationReport(
        faces_before=1000,
        faces_after=100,
        area_error=0.0,
        volume_error=0.0,
        watertight_before=True,
        watertight_after=False,
        normals_consistent=True,
    )

    assert report.lost_watertightness
    assert not report.acceptable()


def test_excessive_area_loss_is_rejected() -> None:
    """The measured failure of the flat budget was up to 78 % of area lost."""
    report = DecimationReport(
        faces_before=3008,
        faces_after=150,
        area_error=0.78,
        volume_error=0.87,
        watertight_before=True,
        watertight_after=True,
        normals_consistent=True,
    )

    assert not report.acceptable()


def test_adaptive_decimation_keeps_area_within_tolerance() -> None:
    """The headline property: reduce, but never past the point where the
    obstacle stops absorbing what it should."""
    original = sphere(subdivisions=5)

    reduced, report = decimate_adaptive(original, DETAIL_LEVELS[2])

    assert report.acceptable()
    assert report.area_error <= 0.15
    assert len(reduced.faces) < len(original.faces)


def test_adaptive_decimation_backs_off_rather_than_damaging_the_mesh() -> None:
    """With an impossible tolerance the mesh must come back untouched, not
    silently wrecked."""
    original = sphere(subdivisions=5)

    reduced, report = decimate_adaptive(original, DETAIL_LEVELS[2], area_tolerance=0.0)

    assert len(reduced.faces) == len(original.faces)
    assert report.faces_after == report.faces_before


def test_a_mesh_already_under_budget_is_left_alone() -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    reduced, report = decimate_adaptive(box, DETAIL_LEVELS[0])

    assert len(reduced.faces) == len(box.faces)
    assert report.acceptable()


def test_finer_levels_keep_more_faces_than_coarser_ones() -> None:
    original = sphere(subdivisions=5)

    near, _ = decimate_adaptive(original, DETAIL_LEVELS[0])
    far, _ = decimate_adaptive(original, DETAIL_LEVELS[2])

    assert len(near.faces) >= len(far.faces)


def test_summary_reports_the_rejection_count_for_the_phase_1_report() -> None:
    good = DecimationReport(100, 50, 0.01, 0.02, True, True, True)
    bad = DecimationReport(100, 50, 0.80, 0.90, True, True, True)

    text = summarise([good, bad])

    assert "1 not within tolerance" in text
    assert summarise([]) == "no obstacles"


def test_decimate_to_never_returns_an_empty_mesh() -> None:
    original = sphere(subdivisions=2)

    reduced = decimate_to(original, 1)

    assert len(reduced.faces) > 0
    assert np.isfinite(reduced.area)
