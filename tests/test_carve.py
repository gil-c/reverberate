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
    CARVE_PITCH_LADDER,
    CARVE_PITCH_M,
    _carve_uncached,
    _outside,
    _surface_cells,
    _tightened,
    _to_budget,
    pitch_for,
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

    def test_past_the_caps_the_closest_reduction_is_still_the_better_answer(
        self, monkeypatch: object
    ) -> None:
        """Refusing here means shipping the collider, and that is worse.

        A reduction whose volume missed the band by a fifth is not a good mesh.
        The collider it would be replaced by is out by an order of magnitude --
        one decoration's isosurface holds 0.028 m3 against a 0.325 m3 collider
        -- so between the two the reduction wins, and the miss is recorded in
        ``CarveResult.volume_error`` rather than hidden.
        """
        from reverberate.geometry import carve

        monkeypatch.setattr(carve, "ABSOLUTE_CAP", 1)  # type: ignore[attr-defined]
        monkeypatch.setattr(carve, "VOLUME_TOLERANCE", 0.0)  # type: ignore[attr-defined]
        reduced = _to_budget(trimesh.creation.box(), 1)
        assert reduced is not None
        assert len(reduced.faces) < 12


def _object_tree(root: Path, render: trimesh.Trimesh, collider: trimesh.Trimesh) -> None:
    directory = root / "objects" / "a"
    directory.mkdir(parents=True)
    for name, mesh in (("abc.glb", render), ("abc.collider.glb", collider)):
        exported = mesh.export(file_type="glb")
        assert isinstance(exported, bytes)
        (directory / name).write_bytes(exported)


class TestNothingToRemove:
    """The fill decides, and it counts cells.

    Comparing the isosurface's volume against the collider's instead would
    compare two different rasterisations: measured, that runs to +1 per cent on
    a sphere and -4 per cent on a small box, which is larger than the signal.
    """

    def test_an_object_with_no_air_inside_keeps_its_collider(self, tmp_path: Path) -> None:
        """A carve that removes nothing would be the collider resampled at 6 mm
        and decimated: approximate where the collider is exact, for no volume
        recovered. It was also the pipeline's one non-deterministic step, since
        whether that decimation stayed closed decided the geometry and the
        answer differs between platforms."""
        sphere = trimesh.creation.icosphere(subdivisions=4)
        _object_tree(tmp_path, sphere, sphere)

        result = _carve_uncached(tmp_path, "abc", sphere)
        assert not result.carved
        assert "no air inside" in result.reason
        assert len(result.mesh.faces) == len(sphere.faces)

    def test_a_gap_the_render_mesh_proves_is_still_carved(self, tmp_path: Path) -> None:
        """The guard above must not swallow the case the carve exists for.

        Two legs under one bounding block is the shape of every inflated
        collider in the bedroom: the fill walks in from outside and the block
        loses the space between them.
        """
        left = trimesh.creation.box(extents=(0.1, 0.4, 0.3))
        left.apply_translation([-0.25, 0.0, 0.0])
        right = trimesh.creation.box(extents=(0.1, 0.4, 0.3))
        right.apply_translation([0.25, 0.0, 0.0])
        render = trimesh.util.concatenate([left, right])
        assert isinstance(render, trimesh.Trimesh)
        collider = trimesh.creation.box(extents=(0.6, 0.4, 0.3))
        _object_tree(tmp_path, render, collider)

        result = _carve_uncached(tmp_path, "abc", collider)
        assert result.carved, result.reason
        assert result.shrink < 0.5
        assert result.mesh.is_watertight


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


class TestAlignment:
    """Where the carve puts the surface, against where the occupancy put it.

    The carve decides on a grid of cells and hands back a mesh, so the mapping
    from cell index to metres is part of the geometry. Half a cell of it is
    3 mm, which is a millimetre and a half of grid step at 16 kHz on each of
    three axes, and it moves the whole object rather than reshaping it -- the
    kind of error that shows up in a voxel view as a fringe on three faces and
    nothing at all on the other three.
    """

    def test_the_isosurface_lands_on_the_cells_it_was_built_from(self) -> None:
        """A block of cells has known bounds; the mesh must agree with them."""
        from scipy import ndimage
        from skimage import measure

        pitch = 0.006
        low = np.array([-0.1, -0.1, -0.1])
        kept = np.zeros((60, 60, 60), dtype=bool)
        kept[20:40, 20:40, 20:40] = True
        # Cell i spans [low + i*pitch, low + (i+1)*pitch), so the block does too.
        want_low = low + 20 * pitch
        want_high = low + 40 * pitch

        field = ndimage.gaussian_filter(np.pad(kept.astype(np.float32), 2), sigma=0.6)
        isosurface = measure.marching_cubes(field, level=0.5)  # type: ignore[no-untyped-call]
        vertices, faces = isosurface[0], isosurface[1]
        mesh = trimesh.Trimesh((vertices - 1.5) * pitch + low, faces, process=True)

        assert np.allclose(mesh.bounds[0], want_low, atol=1e-9)
        assert np.allclose(mesh.bounds[1], want_high, atol=1e-9)

    def test_a_carve_does_not_walk_off_its_collider(self, tmp_path: Path) -> None:
        """End to end, on the shape the carve exists for.

        Two legs under one bounding block: the carve removes the space between
        them, so the volume changes, but neither the top face nor the outer
        sides move -- they are the collider's own, and a mapping error would
        slide all three of them at once.
        """
        left = trimesh.creation.box(extents=(0.1, 0.4, 0.3))
        left.apply_translation([-0.25, 0.0, 0.0])
        right = trimesh.creation.box(extents=(0.1, 0.4, 0.3))
        right.apply_translation([0.25, 0.0, 0.0])
        render = trimesh.util.concatenate([left, right])
        assert isinstance(render, trimesh.Trimesh)
        collider = trimesh.creation.box(extents=(0.6, 0.4, 0.3))
        _object_tree(tmp_path, render, collider)

        result = _carve_uncached(tmp_path, "abc", collider)
        assert result.carved, result.reason
        # One carve cell of slack each way: the occupancy is decided on 6 mm
        # cells, so the surface can only land on one of their faces.
        offset = result.mesh.bounds - collider.bounds
        assert np.all(np.abs(offset) <= CARVE_PITCH_M + 1e-9), offset


class TestTightened:
    """Both sides of the subtraction measured the same way, and thin bodies kept."""

    def test_it_gives_back_the_cell_the_over_marking_took_from_the_air(self) -> None:
        """A solid slab with air on one side loses its outermost layer.

        That layer is the render surface's own marked shell: cells the surface
        passes through, part air and part solid, which the fill stopped in front
        of. The solid has already been eroded by the matching cell, so leaving
        this one in is what made every carve come back a cell fat.
        """
        solid = np.zeros((9, 9, 9), dtype=bool)
        solid[2:7, 2:7, 2:7] = True
        air = np.zeros((9, 9, 9), dtype=bool)
        air[0:2] = True  # reachable air up against the slab's -x face
        tight = _tightened(solid, air)
        assert not tight[2].any()
        assert tight[3:7, 2:7, 2:7].all()

    def test_a_sheet_one_cell_thick_keeps_its_cell(self) -> None:
        """A carpet, a curtain, a picture's canvas: all shell, no second layer.

        Growing the air into a sheet from both faces deletes it, and those are
        the objects the carve exists to recover, so a body that would lose more
        than half of itself is left alone.
        """
        solid = np.zeros((9, 9, 9), dtype=bool)
        solid[4, 2:7, 2:7] = True
        air = np.zeros((9, 9, 9), dtype=bool)
        air[0:4] = True
        air[5:] = True
        tight = _tightened(solid, air)
        assert tight[4, 2:7, 2:7].all()

    def test_a_thick_body_beside_a_sheet_is_judged_on_its_own(self) -> None:
        """One template can hold both, so the guard is per body and not per grid.

        A table is a slab on legs; deciding for the whole occupancy at once
        would either leave the slab fat or take the legs away.
        """
        solid = np.zeros((9, 9, 20), dtype=bool)
        solid[2:7, 2:7, 2:7] = True  # a block
        solid[4, 2:7, 12:17] = True  # and a sheet, not touching it
        air = np.zeros((9, 9, 20), dtype=bool)
        air[0:2] = True
        air[7:] = True
        tight = _tightened(solid, air)
        assert not tight[2, 2:7, 2:7].any()  # the block gave its cell up
        assert tight[4, 2:7, 12:17].all()  # the sheet kept its own


class TestVolumeBand:
    def test_a_reduction_that_inflates_the_shape_is_not_accepted(self) -> None:
        """Hard decimation moves the volume up as readily as down.

        This flat's curtain carves to 0.118 m3 and comes back 0.165 at 2 000
        triangles -- a fifth of the inflation the carve had just removed, put
        back by the mesh that was supposed to record it. A one sided floor took
        that silently.
        """
        from reverberate.geometry import carve

        sphere = trimesh.creation.icosphere(subdivisions=4)
        # A band of zero admits nothing, so the ladder runs out and the cap
        # decides -- which is the refusal path, not an inflated acceptance.
        with_no_band = 0.0
        original = carve.VOLUME_TOLERANCE
        try:
            carve.VOLUME_TOLERANCE = with_no_band
            assert _to_budget(sphere, 20) is sphere
        finally:
            carve.VOLUME_TOLERANCE = original


class TestPitchFor:
    """One pitch per template, from that template's own two meshes."""

    def test_a_small_object_is_carved_finer_than_a_large_one(self) -> None:
        """The point of the change: 6 mm was set by the largest object.

        Five per cent of a 30 cm lamp in every direction, and a fiftieth of a
        car. The lamp's isosurface at 2 mm is small enough to carry ten times
        over, so it pays nothing for the refinement.
        """
        small = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        large = trimesh.creation.box(extents=(4.5, 2.0, 2.0))
        assert pitch_for(small, small) < pitch_for(large, large)
        assert pitch_for(large, large) == CARVE_PITCH_M

    def test_it_never_goes_below_the_finest_grid_this_project_builds(self) -> None:
        """16 kHz at 10.5 points per wavelength is 2.043 mm.

        Deciding occupancy on a cell no solver here can read is detail invented
        for nobody.
        """
        from reverberate.experiments.run import grid_step

        tiny = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
        assert pitch_for(tiny, tiny) >= min(CARVE_PITCH_LADDER)
        assert min(CARVE_PITCH_LADDER) < grid_step(16000.0, 10.5)

    def test_it_does_not_read_fmax(self) -> None:
        """The same flat at 4 and at 16 kHz has to be the same geometry.

        Stated as a property of the signature rather than as a comment: there
        is no frequency to pass, so there is no way for one to leak in.
        """
        import inspect

        assert list(inspect.signature(pitch_for).parameters) == ["render", "collider"]

    def test_a_broad_surface_keeps_the_coarse_pitch_however_small_its_box(self) -> None:
        """The mesh ceiling, not the memory one.

        A crumpled sheet inside a small box has an isosurface that scales with
        its area, and past ISO_FACE_CAP no reduction of it holds the volume --
        which makes a finer pitch a worse carve, not a better one.
        """
        from reverberate.geometry import carve

        box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        wide = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        original = carve.FACES_PER_AREA
        try:
            # Same geometry, an area constant large enough that no rung fits:
            # the choice is the cap's, and it has to bind.
            carve.FACES_PER_AREA = 1e9
            assert pitch_for(wide, box) == CARVE_PITCH_M
        finally:
            carve.FACES_PER_AREA = original


class TestLeakFloor:
    def test_a_finer_pitch_that_collapses_the_carve_is_rejected(self, tmp_path: Path) -> None:
        """A leak is not a refinement, and the two are told apart by size.

        The flood fill is stopped by the render mesh's conservatively marked
        shell, and that shell is what plugs the mesh's own holes; the band
        narrows with the pitch, so a fine pitch is where a leak first appears.
        Simulated here by making the floor unreachable, which is the only way to
        exercise the branch on geometry that does not leak.
        """
        from reverberate.geometry import carve

        left = trimesh.creation.box(extents=(0.03, 0.12, 0.09))
        left.apply_translation([-0.075, 0.0, 0.0])
        right = trimesh.creation.box(extents=(0.03, 0.12, 0.09))
        right.apply_translation([0.075, 0.0, 0.0])
        render = trimesh.util.concatenate([left, right])
        assert isinstance(render, trimesh.Trimesh)
        collider = trimesh.creation.box(extents=(0.18, 0.12, 0.09))
        _object_tree(tmp_path, render, collider)

        original = carve.LEAK_FLOOR
        try:
            carve.LEAK_FLOOR = 2.0  # no fine pitch can ever clear this
            result = _carve_uncached(tmp_path, "abc", collider)
        finally:
            carve.LEAK_FLOOR = original
        assert result.carved, result.reason
        assert result.pitch_m == CARVE_PITCH_M
        assert result.leaked_at_m and result.leaked_at_m < CARVE_PITCH_M
