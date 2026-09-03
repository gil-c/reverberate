"""Tests for the voxel cloud the run page draws.

The picture is only worth having if it is honest about being a sample, and if
the points land where the geometry is. Both are asserted here, because a cloud
that quietly drew one node in thirty while looking complete, or that drew the
right count in the wrong frame, would be worse than no cloud at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.viz.vox_view import (
    _blocks_path,
    _dense_blocks,
    read_voxels,
    surface_of,
    write_voxel_payload,
)


def write_cache(root: Path, nodes: int = 1000) -> Path:
    """A cache entry shaped like PFFDTD's, small enough to reason about.

    The axis extents differ on purpose so ``transpose_order`` actually permutes
    something: a cubic grid would let a wrong frame pass unnoticed.
    """
    root.mkdir(parents=True, exist_ok=True)
    # cart_grid is the scene frame; the engine sorts axes by descending extent,
    # so (8, 4, 6) becomes (8, 6, 4) and axes 1 and 2 swap.
    xv = np.linspace(0.0, 0.7, 8)
    yv = np.linspace(0.0, 0.3, 4)
    zv = np.linspace(0.0, 0.5, 6)
    with h5py.File(root / "cart_grid.h5", "w") as handle:
        handle["h"] = 0.1
        handle["xv"], handle["yv"], handle["zv"] = xv, yv, zv

    nx, ny, nz = 8, 6, 4  # engine order
    rng = np.random.default_rng(0)
    index = rng.choice(nx * ny * nz, size=nodes, replace=False).astype(np.int64)
    with h5py.File(root / "vox_out.h5", "w") as handle:
        handle["Nx"], handle["Ny"], handle["Nz"] = nx, ny, nz
        handle["h"] = 0.1
        handle["bn_ixyz"] = np.sort(index)
        # A node with no adjacency left carries no material either: vox_scene
        # sets mat_bn to -1 for exactly those. A fixture that gave them one
        # would let a bug in the sealing rule pass.
        material = np.full(nodes, 3, dtype=np.int8)
        material[: nodes // 4] = -1
        handle["mat_bn"] = material
        adjacency = np.ones((nodes, 6), dtype=bool)
        adjacency[: nodes // 4] = False  # a quarter sealed
        handle["adj_bn"] = adjacency
    return root


def test_the_cloud_lands_inside_the_scene_box(tmp_path: Path) -> None:
    """Wrong axis order puts points outside the room and is otherwise silent."""
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=1000)

    assert cloud.positions.min(axis=0)[0] >= 0.0
    assert cloud.positions.max(axis=0)[0] <= 0.7 + 1e-6
    assert cloud.positions.max(axis=0)[1] <= 0.3 + 1e-6
    assert cloud.positions.max(axis=0)[2] <= 0.5 + 1e-6


def test_it_says_what_a_cube_stands_for(tmp_path: Path) -> None:
    """A block that did not admit to being one would be the worse failure."""
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=40)

    assert cloud.total_nodes == 192
    assert cloud.drawn <= 40
    assert cloud.cell_m > cloud.h_m
    assert "for 192 boundary nodes" in cloud.summary()


def test_a_corrupt_cache_is_rebuilt_not_raised(tmp_path: Path) -> None:
    """``np.load`` raises ``zipfile.BadZipFile`` on a truncated ``.npz``, which
    is neither ``OSError`` nor ``ValueError`` nor ``KeyError`` -- a half
    written cache entry (crash mid ``np.savez``, disk full) must still fall
    back to rebuilding rather than taking the viewer down.
    """
    root = write_cache(tmp_path / "vox", nodes=192)
    target_cubes = 100_000
    read_voxels(root, target_cubes=target_cubes)  # writes the cache entry
    path = _blocks_path(root, target_cubes)
    # A real npz (a zip archive), cut off mid file -- what a crash or a full
    # disk mid ``np.savez`` leaves behind. Arbitrary bytes are not this: those
    # never reach the zip reader and surface as a plain ``ValueError``, which
    # was already caught.
    written = path.read_bytes()
    path.write_bytes(written[: len(written) // 2])

    cloud = read_voxels(root, target_cubes=target_cubes)

    assert cloud.total_nodes == 192


def test_a_grid_that_fits_is_drawn_cell_by_cell(tmp_path: Path) -> None:
    """No aggregation is applied when none is needed to fit the budget."""
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=100_000)

    assert cloud.cell_m == pytest.approx(cloud.h_m)
    assert cloud.drawn == 192


def test_a_block_holding_any_material_is_not_called_sealed(tmp_path: Path) -> None:
    """A block at a surface holds both its sides.

    Calling it sealed because half its nodes are would paint every wall in the
    room the colour of the sealing, which is the one thing this view is for.
    """
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=8)

    assert cloud.drawn <= 8
    # The fixture seals a quarter of the nodes and spreads them over the grid,
    # so once blocks are this coarse every one of them also holds material.
    assert int(cloud.inert.sum()) == 0


def test_sealed_nodes_are_marked_so_they_can_be_drawn_apart(tmp_path: Path) -> None:
    """Sealing stops the solver carrying sound there; the picture must say so."""
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=100_000)

    assert 0 < int(cloud.inert.sum()) < cloud.drawn


def test_a_rigid_but_coupled_block_is_not_shown_as_a_material(tmp_path: Path) -> None:
    """A block with no material nodes at all is rigid whether or not it is
    also sealed. Gating the rigid label on ``block_inert`` (which additionally
    demands every node have lost adjacency) used to leave a rigid-but-still-
    coupled block -- the very defect patch 5 exists to fix -- looking like it
    commonly held material 0, ``argmax``'s answer on an all-zero row.

    Two blocks, so the scene also holds a real material and ``kinds`` is not
    degenerate: the first half of the grid is rigid and still fully coupled
    (no node has lost adjacency), the second half genuinely is material 0.
    """
    root = tmp_path / "vox"
    root.mkdir(parents=True)
    h = 0.1
    nx, ny, nz = 4, 2, 2  # span 2 along x makes exactly two blocks
    with h5py.File(root / "cart_grid.h5", "w") as handle:
        handle["h"] = h
        handle["xv"], handle["yv"], handle["zv"] = (
            np.arange(nx) * h,
            np.arange(ny) * h,
            np.arange(nz) * h,
        )

    total = nx * ny * nz
    material = np.zeros(total, dtype=np.int8)
    material[: total // 2] = -1  # the block at ix in {0, 1}: rigid
    material[total // 2 :] = 0  # the block at ix in {2, 3}: material 0
    with h5py.File(root / "vox_out.h5", "w") as handle:
        handle["Nx"], handle["Ny"], handle["Nz"] = nx, ny, nz
        handle["h"] = h
        handle["bn_ixyz"] = np.arange(total, dtype=np.int64)
        handle["mat_bn"] = material
        handle["adj_bn"] = np.ones((total, 6), dtype=bool)  # nobody has lost adjacency

    cloud = read_voxels(root, target_cubes=2)

    assert cloud.drawn == 2
    assert int(cloud.material[0]) == -1  # rigid, not material 0
    assert not bool(cloud.inert[0])  # coupled, so not sealed
    assert int(cloud.material[1]) == 0
    assert not bool(cloud.inert[1])


def test_a_boundary_block_does_not_draw_past_the_grid(tmp_path: Path) -> None:
    """A block is sized as if it held a full ``cell_m`` of native cells, which
    one straddling the far edge of an axis the grid's shape does not divide
    evenly by does not -- its corners must be clipped to the grid's own extent
    rather than drawn a whole block past it.
    """
    root = tmp_path / "vox"
    root.mkdir(parents=True)
    h = 0.1
    nx, ny, nz = 5, 4, 3  # none a multiple of the span this target forces
    xv, yv, zv = np.arange(nx) * h, np.arange(ny) * h, np.arange(nz) * h
    with h5py.File(root / "cart_grid.h5", "w") as handle:
        handle["h"] = h
        handle["xv"], handle["yv"], handle["zv"] = xv, yv, zv

    total = nx * ny * nz
    with h5py.File(root / "vox_out.h5", "w") as handle:
        handle["Nx"], handle["Ny"], handle["Nz"] = nx, ny, nz
        handle["h"] = h
        handle["bn_ixyz"] = np.arange(total, dtype=np.int64)
        handle["mat_bn"] = np.full(total, 3, dtype=np.int8)
        handle["adj_bn"] = np.ones((total, 6), dtype=bool)

    # One block for the whole grid, so its span overshoots every axis.
    cloud = read_voxels(root, target_cubes=1)
    surface = surface_of(cloud)
    corners = surface.corners.reshape(-1, 3).astype(np.float64)

    assert (corners >= cloud.bounds_lo - 1e-6).all()
    assert (corners <= cloud.bounds_hi + 1e-6).all()


def test_the_payload_is_binary_and_self_describing(tmp_path: Path) -> None:
    """Typed arrays, because the browser wants them and text is ten times."""
    cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=100_000)
    surface = surface_of(cloud)
    target = tmp_path / "site"

    record = write_voxel_payload(surface, ["carpet", "shell"], target)

    written = json.loads((target / "voxels.json").read_text())
    assert written == record
    assert (target / "voxels.f32").stat().st_size == surface.corners.size * 4
    assert (target / "voxels_index.u32").stat().st_size == surface.index.size * 4
    assert (target / "voxels_label.i16").stat().st_size == surface.label.size * 2
    assert record["labels"] == ["carpet", "shell"]
    assert record["quads"] == surface.quads


class TestSurface:
    """Merging must reduce the triangle count without moving a single face."""

    def test_the_merged_area_equals_the_visible_face_area(self, tmp_path: Path) -> None:
        """The one assertion that separates merging from dropping.

        A mesher that lost faces would also report a smaller triangle count and
        look like a better one.
        """
        cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=100_000)
        surface = surface_of(cloud)

        occupied = _dense_blocks(cloud)
        visible = 0
        for axis in range(3):
            faces = np.moveaxis(occupied, axis, 0)
            visible += int(
                (faces[:-1] & ~faces[1:]).sum()
                + (faces[1:] & ~faces[:-1]).sum()
                + faces[0].sum()
                + faces[-1].sum()
            )

        quads = surface.corners.reshape(-1, 4, 3)
        area = float(
            np.abs(np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])).sum(axis=1).sum()
        )
        # 1e-6 rather than exact: the corners are float32, which is what the
        # browser reads, and the tolerance has to be the data's not the maths'.
        assert area == pytest.approx(visible * cloud.cell_m**2, rel=1e-6)

    def test_it_draws_fewer_triangles_than_solid_cubes(self, tmp_path: Path) -> None:
        cloud = read_voxels(write_cache(tmp_path / "vox", nodes=192), target_cubes=100_000)
        surface = surface_of(cloud)

        assert surface.triangles < cloud.drawn * 12
        assert surface.index.size == surface.quads * 6

    def test_a_flat_face_of_one_material_becomes_one_quad(self) -> None:
        """The whole point: a floor is one rectangle, not one per cell."""
        from reverberate.viz.vox_view import _greedy_quads

        kind = np.full((6, 5), 3, dtype=np.int16)
        merged = _greedy_quads(kind)

        assert merged.shape[0] == 1
        assert list(merged[0]) == [0, 0, 6, 5, 3]

    def test_two_materials_do_not_merge_into_one_quad(self) -> None:
        """Merging across a material boundary would draw the wrong colour."""
        from reverberate.viz.vox_view import _greedy_quads

        kind = np.full((6, 5), 3, dtype=np.int16)
        kind[3:] = 7
        merged = _greedy_quads(kind)

        assert merged.shape[0] == 2
        assert sorted(int(row[4]) for row in merged) == [3, 7]

    def test_nothing_is_drawn_where_there_is_no_face(self) -> None:
        from reverberate.viz.vox_view import NO_FACE, _greedy_quads

        assert _greedy_quads(np.full((4, 4), NO_FACE, dtype=np.int16)).shape[0] == 0

    def test_reused_scratch_buffers_do_not_leak_between_slices(self) -> None:
        """``surface_of`` passes the same ``changed``/``ends`` buffers to every
        slice of an axis rather than allocating fresh ones. Both are fully
        overwritten before they are read, so a second call reusing a first
        call's buffers must answer only for its own, smaller ``kind`` -- not
        for whatever the first call's larger region left behind in them.
        """
        from reverberate.viz.vox_view import NO_FACE, _greedy_quads

        changed = np.empty((6, 5), dtype=bool)
        ends = np.empty((7, 5), dtype=np.int64)

        full = np.full((6, 5), 3, dtype=np.int16)
        first = _greedy_quads(full, changed, ends)
        assert list(first[0]) == [0, 0, 6, 5, 3]

        partial = np.full((6, 5), NO_FACE, dtype=np.int16)
        partial[2:4, 1:3] = 7
        second = _greedy_quads(partial, changed, ends)
        assert second.shape[0] == 1
        assert list(second[0]) == [2, 1, 4, 3, 7]

        # And unbuffered, for the same inputs, agrees -- proving the buffers
        # are what changed, not the algorithm.
        assert np.array_equal(second, _greedy_quads(partial))
