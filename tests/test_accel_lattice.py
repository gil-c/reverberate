"""The grid, the constants and the voxel lattice: upstream's numbers, not approximations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reverberate.accel.lattice import (
    cart_grid,
    lattice_for,
    sim_constants,
    tri_box_hits,
    triangle_index,
    voxel_triangles,
)
from reverberate.accel.scene import load_scene
from test_accel_scene import write_scene


class TestConstants:
    def test_the_storey_of_hssd_0076_at_8_khz(self) -> None:
        """The values its cache manifest records, to the last digit."""
        constants = sim_constants(20.0, 50.0, 8000.0, 10.5)
        assert constants.h == 0.004085714285714285
        assert constants.sr == 145637.9057415272

    def test_fcc_is_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            sim_constants(20.0, 50.0, 1000.0, 10.5, fcc=True)


class TestGrid:
    def test_the_offset_adds_ceil_plus_one_nodes(self) -> None:
        grid = cart_grid(0.25, 3.0, np.zeros(3), np.array([1.0, 2.0, 0.5]))
        # extents 2.5, 3.5, 2.0 over 0.25: 10, 14, 8 cells, one node more than cells
        assert grid.shape == (11, 15, 9)
        assert grid.xv[0] == -0.75
        assert grid.points == 11 * 15 * 9

    def test_an_offset_under_two_cells_is_refused(self) -> None:
        with pytest.raises(ValueError):
            cart_grid(0.1, 2.0, np.zeros(3), np.ones(3))


class TestLattice:
    def test_voxels_tile_the_grid_with_a_one_cell_halo(self) -> None:
        grid = cart_grid(0.05, 3.5, np.zeros(3), np.ones(3))
        lattice = lattice_for(grid, 6)
        assert lattice.nvox == int(np.prod(lattice.nvox_xyz))
        assert np.all(lattice.count >= 8)
        assert np.all(lattice.count < 16)
        # The cores tile the interior: every interior node is in exactly one core.
        covered = np.zeros(grid.shape, dtype=np.int32)
        for start, count in zip(lattice.start, lattice.count, strict=True):
            covered[
                start[0] + 1 : start[0] + count[0] - 1,
                start[1] + 1 : start[1] + count[1] - 1,
                start[2] + 1 : start[2] + count[2] - 1,
            ] += 1
        assert np.all(covered[1:-1, 1:-1, 1:-1] == 1)
        assert np.all(covered[0] == 0) and np.all(covered[-1] == 0)
        assert np.all(lattice.bmin[0] == np.array([grid.xv[0], grid.yv[0], grid.zv[0]]) - 0.025)

    def test_a_small_nh_is_refused(self) -> None:
        grid = cart_grid(0.05, 3.5, np.zeros(3), np.ones(3))
        with pytest.raises(ValueError):
            lattice_for(grid, 3)


class TestTriangleIndex:
    def test_candidates_are_ascending_and_the_box_test_keeps_a_subset(self, tmp_path: Path) -> None:
        scene = load_scene(write_scene(tmp_path / "box.json", extra_material=True))
        grid = cart_grid(0.05, 3.5, scene.bmin, scene.bmax)
        lattice = lattice_for(grid, 5)
        offsets, candidates = triangle_index(lattice, scene.pre)
        kept_offsets, kept = voxel_triangles(lattice, scene.pre, np)
        assert kept.size <= candidates.size
        for vox in range(lattice.nvox):
            mine = candidates[offsets[vox] : offsets[vox + 1]]
            assert np.all(np.diff(mine) > 0)
            got = kept[kept_offsets[vox] : kept_offsets[vox + 1]]
            assert np.all(np.diff(got) > 0)
            assert set(got.tolist()) <= set(mine.tolist())
        # Every triangle is held by some voxel; a voxel deep inside the bare
        # box, away from the walls and the two objects near its middle, holds none.
        assert set(kept.tolist()) == set(range(scene.triangles))
        bare = load_scene(write_scene(tmp_path / "bare.json"))
        bare_offsets, _ = voxel_triangles(lattice, bare.pre, np)
        inside = np.flatnonzero(
            np.all(lattice.bmin > 0.1, axis=1) & np.all(lattice.bmax < 0.9, axis=1)
        )
        assert inside.size > 0
        assert all(bare_offsets[v + 1] == bare_offsets[v] for v in inside)

    def test_the_box_test_agrees_with_geometry_on_a_plain_case(self) -> None:
        v = np.array([[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]])
        nor = np.array([[0.0, 0.0, 0.5]])
        cent = v.mean(axis=1)
        tbmin, tbmax = v.min(axis=1), v.max(axis=1)
        crossing = tri_box_hits(
            np.array([[0.1, 0.1, 0.4]]), np.array([[0.3, 0.3, 0.6]]), v, nor, cent, tbmin, tbmax, np
        )
        above = tri_box_hits(
            np.array([[0.1, 0.1, 0.6]]), np.array([[0.3, 0.3, 0.9]]), v, nor, cent, tbmin, tbmax, np
        )
        beside = tri_box_hits(
            np.array([[0.8, 0.8, 0.4]]), np.array([[0.9, 0.9, 0.6]]), v, nor, cent, tbmin, tbmax, np
        )
        assert crossing[0] and not above[0] and not beside[0]
