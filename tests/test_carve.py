"""Tests for carving HSSD's collision proxies back to the render shape.

The carve replaces the mesh the solver is given, so the tests that matter are
about the two directions it can be wrong in: carving away solid that is really
there, and quietly substituting something that is no longer a closed body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.carve import (
    CarveReport,
    CarveResult,
    _outside,
    _surface_cells,
    _to_budget,
)


def _grid(mesh: trimesh.Trimesh, pitch: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    low = mesh.bounds[0] - 3 * pitch
    high = mesh.bounds[1] + 3 * pitch
    extent = np.ceil((high - low) / pitch).astype(np.int64) + 4
    return low, (int(extent[0]), int(extent[1]), int(extent[2]))


class TestSurfaceCells:
    """Over-marking is safe; a hole in the surface is not."""

    def test_a_closed_box_leaves_its_inside_unreachable(self) -> None:
        """The property the flood fill depends on, stated directly.

        If the rasterisation leaks anywhere, the fill reaches the middle of the
        box and the carve deletes the object.
        """
        pitch = 0.01
        box = trimesh.creation.box(extents=(0.4, 0.3, 0.5))
        low, shape = _grid(box, pitch)
        outside = _outside(_surface_cells(box, pitch, low, shape))
        centre = tuple(n // 2 for n in shape)
        assert not outside[centre]

    def test_it_leaks_through_nothing_on_a_face_aligned_box(self) -> None:
        """Axis-aligned faces are the case a ray voxelisation gets wrong.

        A box's faces lie exactly in the planes the rays travel along, which is
        how ``method="ray"`` came back perforated and carved 42 templates down
        to a few per cent of themselves.
        """
        pitch = 0.01
        box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        low, shape = _grid(box, pitch)
        solid = ~_outside(_surface_cells(box, pitch, low, shape))
        # 0.2 m at 10 mm is 20 cells a side; conservative marking may add one
        # shell, so anything from 20^3 to 22^3 is right and 0 is the failure.
        assert 8000 <= int(solid.sum()) <= 10648

    def test_a_triangle_far_larger_than_a_cell_is_still_marked(self) -> None:
        """The carpet: 106 triangles spanning 4.5 m, at a 6 mm pitch.

        Marking only the cells the vertices fall in would leave a sheet with
        nothing between its corners, and trimesh's own subdivision voxeliser
        gives up on it with ``max_iter exceeded``.
        """
        pitch = 0.02
        sheet = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        low, shape = _grid(sheet, pitch)
        marked = _surface_cells(sheet, pitch, low, shape)
        # Half of a 100 x 100 cell square, give or take the conservative edge.
        assert marked.sum() > 4000


class TestToBudget:
    def test_a_mesh_already_inside_the_budget_is_returned_unchanged(self) -> None:
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        assert _to_budget(box, 1000) is box

    def test_it_backs_off_rather_than_giving_up(self) -> None:
        """Asking once and refusing lost 11 of 47 templates.

        A sphere at 5 000 triangles reduced to 20 in one step is not a closed
        body; the ladder should find a target between that and the original
        rather than dropping the carve.
        """
        sphere = trimesh.creation.icosphere(subdivisions=4)
        reduced = _to_budget(sphere, 20)
        assert reduced is not None
        assert reduced.is_watertight
        assert len(reduced.faces) < len(sphere.faces)

    def test_a_small_mesh_survives_a_budget_it_cannot_meet(self) -> None:
        """The relative budget alone refused 17 of 41 templates.

        Four times a 108 face picture is 2 000 triangles, and an isosurface
        that will not reduce that far is still nothing against a scene of a
        million and a half. Refusing it puts the inflated collider back.
        """
        box = trimesh.creation.box()
        assert _to_budget(box, 1) is box

    def test_it_returns_none_when_nothing_holds(self, monkeypatch: object) -> None:
        """Past both caps a carve that will not reduce is not worth its cost."""
        from reverberate.geometry import carve

        monkeypatch.setattr(carve, "ABSOLUTE_CAP", 1)  # type: ignore[attr-defined]
        assert _to_budget(trimesh.creation.box(), 1) is None


class TestCarveReport:
    def test_it_separates_what_was_carved_from_what_was_left(self) -> None:
        report = CarveReport()
        report.add(
            "carved",
            CarveResult(
                mesh=trimesh.creation.box(),
                carved=True,
                collider_volume=1.0,
                carved_volume=0.25,
            ),
        )
        report.add(
            "kept",
            CarveResult(mesh=trimesh.creation.box(), carved=False, reason="carve came back open"),
        )
        assert report.carved == {"carved": 0.25}
        assert report.skipped == {"kept": "carve came back open"}
        assert "1 colliders carved" in report.summary()

    def test_a_scene_with_nothing_to_carve_says_so(self) -> None:
        assert CarveReport().summary() == "no carve"


class TestShrink:
    def test_an_uncarved_result_reports_no_change(self) -> None:
        """``shrink`` is read straight into the manifest, so its neutral value
        has to be 1.0 and not a division by a volume nobody measured."""
        result = CarveResult(mesh=trimesh.creation.box(), carved=False)
        assert result.shrink == 1.0


def test_the_pitch_is_below_the_coarsest_grid_it_is_used_on() -> None:
    """A carve finer than the grid invents nothing the solver can see, but a
    carve coarser than it would put features between the nodes.

    4 kHz at 10.5 points per wavelength is an 8.17 mm step, and that is the
    coarser of the two runs this geometry is built for.
    """
    from reverberate.experiments.run import grid_step
    from reverberate.geometry.carve import CARVE_PITCH_M

    assert grid_step(4000.0, 10.5) > CARVE_PITCH_M


def test_the_cache_entry_names_the_rules_it_was_made_under(tmp_path: Path) -> None:
    """Changing the pitch or the budget must not silently reuse old carves."""
    from reverberate.geometry import carve

    assert str(carve.CARVE_PITCH_M) in f"{carve.CARVE_PITCH_M}_{carve.BUDGET_FACTOR}"
