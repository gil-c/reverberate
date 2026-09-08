"""Tests for the source-receiver ellipsoid.

Built on a synthetic scene whose answer is known: two boxes far apart, with the
source and the receiver both inside the first. A path budget smaller than the
gap must keep only the near box, and one larger than the round trip to the far
box must keep both. That is the whole criterion, and it is checked as geometry
rather than as a cost.

The other half is the claim the module deliberately does not make. It prices a
domain; it does not assert that the truncated response is exact, because that
needs two solves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh

from reverberate.experiments.w37_ellipsoid import (
    build,
    load_scene,
    path_budget_m,
    restrict,
)

SOUND_SPEED = 343.0
RATE = 0.917


def _box(centre: tuple[float, float, float], size: float = 1.0) -> trimesh.Trimesh:
    mesh = cast(trimesh.Trimesh, trimesh.creation.box(extents=(size, size, size)))
    mesh.apply_translation(centre)
    return mesh


def _scene() -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """A near box holding both foci, and a far box 20 m away."""
    return (
        [_box((0.0, 0.0, 0.0), 2.0), _box((20.0, 0.0, 0.0), 2.0)],
        np.array([-0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    )


def _write_model(path: Path) -> Path:
    meshes, src, rec = _scene()
    document = {
        "mats_hash": {
            f"box{index}": {
                "pts": mesh.vertices.tolist(),
                "tris": mesh.faces.tolist(),
                "sides": [2] * len(mesh.faces),
            }
            for index, mesh in enumerate(meshes)
        },
        "sources": [{"name": "s0", "xyz": src.tolist()}],
        "receivers": [{"name": "r0", "xyz": rec.tolist()}],
    }
    path.write_text(json.dumps(document))
    return path


def test_the_budget_is_c_times_the_window() -> None:
    assert path_budget_m(0.015, sound_speed_m_s=343.2) == pytest.approx(5.148)


def test_the_conservative_budget_is_the_stencil_corner_rule() -> None:
    """A factor of 1.73 in domain for 135 decibels nobody can hear."""
    plain = path_budget_m(0.015, sound_speed_m_s=343.2)
    corner = path_budget_m(0.015, sound_speed_m_s=343.2, conservative=True)
    assert corner / plain == pytest.approx(np.sqrt(3.0))


def test_a_negative_window_is_refused() -> None:
    with pytest.raises(ValueError):
        path_budget_m(-1.0)


def test_a_short_budget_keeps_only_the_geometry_near_the_foci() -> None:
    meshes, src, rec = _scene()
    span, triangles = restrict(meshes, src, rec, cut_m=5.0)
    assert triangles == len(meshes[0].faces)
    assert span[0] < 3.0


def test_a_long_budget_keeps_the_distant_geometry_too() -> None:
    meshes, src, rec = _scene()
    span, triangles = restrict(meshes, src, rec, cut_m=60.0)
    assert triangles == sum(len(mesh.faces) for mesh in meshes)
    assert span[0] > 20.0


def test_the_domain_is_the_box_of_what_survives_not_of_the_ellipsoid() -> None:
    """The distinction the module exists to keep.

    A 5 m budget draws a spheroid about 5 m across, but the only geometry
    inside it is a 2 m box, and that box is what the solver would be given.
    """
    meshes, src, rec = _scene()
    span, _ = restrict(meshes, src, rec, cut_m=5.0)
    assert float(np.max(span)) < 4.0


def test_culling_a_coarse_mesh_removes_nothing() -> None:
    """The trap, and the reason refinement comes before truncation.

    ``path_length_bound`` subtracts a triangle's longest edge, so it is a
    genuine lower bound and is biased towards keeping. A 2 m box has 2.8 m
    diagonals, so even a budget far shorter than the room keeps every face.
    An unrefined apartment shell is a handful of enormous triangles and the
    ellipsoid would buy nothing at all on it.
    """
    near, _, _ = _scene()
    coarse = [near[0]]
    src, rec = np.array([-0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])

    # Twelve triangles for the whole box, and a budget a tenth of its size
    # removes none of them: every edge is longer than the budget.
    _, kept_coarse = restrict(coarse, src, rec, cut_m=0.2)
    assert kept_coarse == len(coarse[0].faces)

    refined = [cast(trimesh.Trimesh, coarse[0].subdivide_to_size(max_edge=0.25))]
    _, kept_fine = restrict(refined, src, rec, cut_m=3.0)
    assert kept_fine < len(refined[0].faces)


def test_a_budget_that_reaches_nothing_at_all_is_refused() -> None:
    """Rather than returning an empty domain that would price as free."""
    meshes, src, rec = _scene()
    refined = [cast(trimesh.Trimesh, mesh.subdivide_to_size(max_edge=0.05)) for mesh in meshes]
    with pytest.raises(ValueError, match="keeps no geometry"):
        restrict(refined, src, rec, cut_m=0.5)


def test_restriction_never_grows_with_a_smaller_budget() -> None:
    meshes, src, rec = _scene()
    wide, wide_faces = restrict(meshes, src, rec, cut_m=60.0)
    narrow, narrow_faces = restrict(meshes, src, rec, cut_m=5.0)
    assert narrow_faces <= wide_faces
    assert np.all(narrow <= wide + 1e-9)


def test_the_saving_is_reported_in_cards_and_saturation_is_flagged(tmp_path: Path) -> None:
    report = build(
        _write_model(tmp_path / "scene.json"),
        tmp_path / "out",
        fmax_hz=2000.0,
        usd_per_hour_per_card=RATE,
        windows_ms=(5.0, 200.0),
    )
    short, long = report["windows"]
    assert short["point_share"] < long["point_share"]
    assert long["saturated"] is True
    assert short["saturated"] is False


def test_a_saturated_window_is_never_priced_above_the_full_scene(tmp_path: Path) -> None:
    """The pad can make a culled box round up past the whole one."""
    report = build(
        _write_model(tmp_path / "scene.json"),
        tmp_path / "out",
        fmax_hz=2000.0,
        usd_per_hour_per_card=RATE,
        windows_ms=(500.0,),
    )
    window = report["windows"][0]
    assert window["grid_points"] <= report["full_grid_points"]
    assert window["usd"] <= window["usd_full"] * 1.000001


def test_the_report_declines_the_claim_it_cannot_support(tmp_path: Path) -> None:
    report = build(
        _write_model(tmp_path / "scene.json"),
        tmp_path / "out",
        fmax_hz=2000.0,
        usd_per_hour_per_card=RATE,
        windows_ms=(20.0,),
    )
    assert "not made here" in report["not_claimed"]
    assert report["path_budget_rule"] == "c T"
    assert "1.004" in report["path_budget_evidence"]
    assert report["reference_run"] is None


def test_a_model_without_a_pair_of_foci_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"mats_hash": {}, "sources": [], "receivers": []}))
    with pytest.raises(ValueError, match="no source and receiver"):
        load_scene(path)


def test_the_hourly_rate_cannot_be_omitted(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build(_write_model(tmp_path / "scene.json"), tmp_path / "out", fmax_hz=2000.0)  # type: ignore[call-arg]
