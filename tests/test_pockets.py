"""Air pockets on the grid are found, sized and sealed, and rooms are left alone."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.wave.pockets import ROOM_MIN_M3, census, seal


def a_grid(
    tmp_path: Path, *, step: float = 0.1, shape: tuple[int, int, int] = (40, 30, 20)
) -> Path:
    """A box room with one closed cabinet inside it, as boundary nodes.

    The room's walls are the outermost layer of boundary nodes. The cabinet is
    a hollow box of boundary nodes with air inside, which is exactly what the
    voxeliser makes of a mesh whose inside nothing sealed. Boundary nodes at
    the room side carry the shell material 0, the cabinet material 1, and every
    boundary node's adjacency points at its air neighbours.
    """
    nx, ny, nz = shape
    solid = np.zeros(shape, dtype=bool)
    # The cabinet: a hollow box 8 x 6 x 10 cells, walls one cell thick.
    solid[10:18, 10:16, 0:10] = True
    solid[11:17, 11:15, 1:9] = False
    material = np.full(shape, -1, dtype=np.int8)
    is_boundary = np.zeros(shape, dtype=bool)
    # Room walls: the outer layer.
    is_boundary[0, :, :] = is_boundary[-1, :, :] = True
    is_boundary[:, 0, :] = is_boundary[:, -1, :] = True
    is_boundary[:, :, 0] = is_boundary[:, :, -1] = True
    material[is_boundary] = 0
    # Cabinet walls.
    is_boundary |= solid
    material[solid] = 1
    air = ~is_boundary
    flat = np.flatnonzero(is_boundary.ravel())
    ix, iy, iz = np.unravel_index(flat, shape)
    adj = np.zeros((flat.size, 6), dtype=bool)
    for column, (dx, dy, dz) in enumerate(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ):
        jx, jy, jz = ix + dx, iy + dy, iz + dz
        inside = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
        hit = np.zeros(flat.size, dtype=bool)
        hit[inside] = air[jx[inside], jy[inside], jz[inside]]
        adj[:, column] = hit
    path = tmp_path / "vox_out.h5"
    with h5py.File(path, "w") as handle:
        for name, value in (("Nx", nx), ("Ny", ny), ("Nz", nz), ("Nb", flat.size)):
            handle.create_dataset(name, data=np.int64(value))
        handle.create_dataset("h", data=np.float64(step))
        handle.create_dataset("bn_ixyz", data=flat.astype(np.int64))
        handle.create_dataset("adj_bn", data=adj)
        handle.create_dataset("mat_bn", data=material.ravel()[flat])
        handle.create_dataset("saf_bn", data=np.ones(flat.size))
        for name, count in (("xv", nx), ("yv", ny), ("zv", nz)):
            handle.create_dataset(name, data=np.arange(count) * step)
    return path


def test_the_cabinet_s_inside_is_a_pocket_and_the_room_is_not(tmp_path: Path) -> None:
    grid = a_grid(tmp_path)
    found = census(grid)
    # The room and the cabinet's inside: two components of air.
    assert found.components == 2
    # Both touch live boundary nodes; only the small one is to be sealed.
    assert len(found.pockets) == 2
    assert len(found.to_seal()) == 1
    pocket = found.to_seal()[0]
    assert pocket.cells == 6 * 4 * 8
    assert pocket.volume_m3 == pytest.approx(6 * 4 * 8 * 0.1**3)
    assert pocket.span_m == pytest.approx((0.6, 0.4, 0.8))
    assert pocket.materials == (1,)
    assert pocket.sealed is True
    assert pocket.lowest_mode_hz == pytest.approx(343.2 / 1.6)
    # The room itself is bigger than a room's floor and is kept.
    assert found.largest[0]["volume_m3"] > ROOM_MIN_M3


def test_sealing_makes_the_pocket_static_and_keeps_the_grid_sorted(tmp_path: Path) -> None:
    grid = a_grid(tmp_path)
    found = census(grid)
    out = tmp_path / "sealed.h5"
    record = seal(grid, out, found.to_seal())
    assert record["pockets_sealed"] == 1
    assert record["cells_sealed"] == 6 * 4 * 8
    with h5py.File(out, "r") as handle:
        bn = handle["bn_ixyz"][:]
        adj = handle["adj_bn"][:]
        mat = handle["mat_bn"][:]
        assert int(handle["Nb"][()]) == bn.size
    assert np.all(np.diff(bn) > 0), "the engine wants sorted boundary nodes"
    assert bn.size == record["boundary_nodes_after"]
    # Every cell of the pocket is now a boundary node with nothing around it.
    shape = (40, 30, 20)
    inside = np.zeros(shape, dtype=bool)
    inside[11:17, 11:15, 1:9] = True
    pocket_index = np.flatnonzero(inside.ravel())
    where = np.searchsorted(bn, pocket_index)
    assert np.array_equal(bn[where], pocket_index)
    assert not adj[where].any()
    assert np.all(mat[where] == -1)
    # And the cabinet's walls no longer point inward: the bit towards a pocket
    # cell is clear, while the bit towards the room above the cabinet stays.
    walls = ~np.isin(bn, pocket_index)
    below = walls & np.isin(bn + 1, pocket_index)
    above = walls & np.isin(bn - 1, pocket_index)
    assert below.any() and above.any()
    assert not adj[below, 4].any() and not adj[above, 5].any()
    assert adj[above, 4].any(), "the top wall still faces the room"
    # A wall with no air side left carries no material: the engine asserts it.
    buried = ~adj.any(axis=1)
    assert buried.sum() > pocket_index.size, "some cabinet walls faced only the pocket"
    assert np.all(mat[buried] == -1)
    assert record["walls_made_static"] == int(buried.sum()) - pocket_index.size
    # The census of the sealed grid finds the room alone.
    after = census(out)
    assert after.to_seal() == []


def test_a_grid_too_big_for_this_machine_is_refused_before_it_is_read(tmp_path: Path) -> None:
    grid = a_grid(tmp_path)
    with pytest.raises(MemoryError, match="GB"):
        census(grid, max_cells=1000)
