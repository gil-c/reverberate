"""Tests for the tiered audit view: one room at the grid's step, the rest coarser.

The view is an audit, so the only property that really matters is that tiering
does not change the picture. Two things could: a room processed without its
neighbours would draw the faces between them, which the grid hides, and a room
too heavy to draw whole must be cut into tiles without a single face appearing
at a cut. Both are asserted against the same grid merged in one piece.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from shapely.geometry import Polygon

from reverberate.experiments.audit_view import (
    _scan_room,
    _tile,
    _write_tier,
    halo_for,
    membership_mask,
)
from reverberate.geometry.rooms import Partition, RoomPartition, _fill_unclaimed, _rasterise
from reverberate.viz.vox_view import blocks_from_nodes, read_grid_nodes, surface_of


def face_area(corners: np.ndarray) -> float:
    quads = corners.reshape(-1, 4, 3).astype(np.float64)
    return float(
        np.abs(np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])).sum(axis=1).sum()
    )


@pytest.fixture
def grid(tmp_path: Path) -> Path:
    """A cache entry with a solid block of nodes spanning two rooms.

    Wide along x so a partition down the middle actually cuts through the
    geometry, which is where every artefact this file hunts would appear.
    """
    root = tmp_path / "vox"
    root.mkdir(parents=True)
    h = 0.1
    nx, ny, nz = 24, 5, 7
    with h5py.File(root / "cart_grid.h5", "w") as handle:
        handle["h"] = h
        handle["xv"] = np.arange(nx) * h
        handle["yv"] = np.arange(ny) * h
        handle["zv"] = np.arange(nz) * h

    # Engine order is descending extent: (24, 7, 5).
    total = nx * ny * nz
    rng = np.random.default_rng(3)
    keep = np.sort(rng.choice(total, size=total * 2 // 3, replace=False)).astype(np.int64)
    with h5py.File(root / "vox_out.h5", "w") as handle:
        handle["Nx"], handle["Ny"], handle["Nz"] = nx, nz, ny
        handle["h"] = h
        handle["bn_ixyz"] = keep
        material = rng.integers(0, 3, keep.size).astype(np.int8)
        material[: keep.size // 5] = -1
        handle["mat_bn"] = material
        adjacency = np.ones((keep.size, 6), dtype=bool)
        adjacency[: keep.size // 5] = False
        handle["adj_bn"] = adjacency
    return root


@pytest.fixture
def split(grid: Path) -> Partition:
    """Two rooms, cut across the middle of the grid's x extent."""
    with h5py.File(grid / "cart_grid.h5", "r") as handle:
        xv, zv = handle["xv"][:], handle["zv"][:]
    rooms = [
        RoomPartition("west", "west", Polygon([(-1, -1), (1.1, -1), (1.1, 2), (-1, 2)]), ("west",)),
        RoomPartition("east", "east", Polygon([(1.2, -1), (4, -1), (4, 2), (1.2, 2)]), ("east",)),
    ]
    raster = _fill_unclaimed(_rasterise(rooms, np.asarray(xv), np.asarray(zv)))
    return Partition(rooms=tuple(rooms), raster=raster, xv=np.asarray(xv), zv=np.asarray(zv))


@pytest.mark.parametrize("span", [1, 2])
def test_tiering_draws_the_same_faces_as_one_untiered_merge(
    grid: Path, split: Partition, tmp_path: Path, span: int
) -> None:
    """The assertion the whole view rests on.

    Area, not quad count: putting the room in the merge key stops a rectangle
    spanning two rooms, so the partitioned picture is legitimately made of a
    few more, smaller quads. If any *face* were added or lost the area would
    move, and nothing else about the picture can.
    """
    subs, material, inert, axes, h_m, total, _ = read_grid_nodes(grid)
    whole = surface_of(blocks_from_nodes(subs, material, inert, None, span, axes, h_m, total))

    with h5py.File(grid / "vox_out.h5", "r") as handle:
        ny, nz = int(handle["Ny"][()]), int(handle["Nz"][()])
    shape = (axes[0].size, axes[1].size, axes[2].size)
    mask = membership_mask(split, halo_for(span))

    tiered = 0.0
    nodes = 0
    for index, room in enumerate(split.rooms):
        room_subs, room_material, room_inert = _scan_room(grid, mask, index, shape, ny, nz)
        room_of = split.raster[room_subs[0], room_subs[2]]
        plan = _write_tier(
            room_subs,
            room_material,
            room_inert,
            room_of,
            span,
            axes,
            h_m,
            index,
            tmp_path / room.name,
            "tier",
            10**9,
        )
        nodes += plan.nodes
        for entry in plan.files:
            corners = np.fromfile(tmp_path / room.name / str(entry["corners_url"]), np.float32)
            tiered += face_area(corners)

    assert nodes == total, "the partition must own every node of the grid"
    assert tiered == pytest.approx(face_area(whole.corners), rel=1e-6)


def test_a_room_too_heavy_to_draw_whole_is_cut_without_adding_a_face(
    grid: Path, split: Partition, tmp_path: Path
) -> None:
    """Tiles cut the finished quads, never the blocks.

    Cutting the blocks first would leave each piece's edge blocks unable to see
    their neighbours, so they would emit faces the grid does not have -- a
    picture of the cut. Cutting quads cannot: every quad goes to exactly one
    tile and none is reshaped.
    """
    subs, material, inert, axes, h_m, total, _ = read_grid_nodes(grid)
    with h5py.File(grid / "vox_out.h5", "r") as handle:
        ny, nz = int(handle["Ny"][()]), int(handle["Nz"][()])
    shape = (axes[0].size, axes[1].size, axes[2].size)
    mask = membership_mask(split, halo_for(1))
    room_subs, room_material, room_inert = _scan_room(grid, mask, 0, shape, ny, nz)
    room_of = split.raster[room_subs[0], room_subs[2]]

    args = (room_subs, room_material, room_inert, room_of, 1, axes, h_m, 0)
    whole = _write_tier(*args, tmp_path / "whole", "tier", 10**9)
    cut = _write_tier(*args, tmp_path / "cut", "tier", 40)

    assert len(whole.files) == 1
    assert len(cut.files) > 1
    assert cut.quads == whole.quads
    assert sum(int(entry["quads"]) for entry in cut.files) == whole.quads
    area = sum(
        face_area(np.fromfile(tmp_path / "cut" / str(entry["corners_url"]), np.float32))
        for entry in cut.files
    )
    assert area == pytest.approx(
        face_area(np.fromfile(tmp_path / "whole" / str(whole.files[0]["corners_url"]), np.float32))
    )


class TestTile:
    def test_a_payload_within_budget_is_one_tile(self) -> None:
        corners = np.zeros((10, 4, 3), dtype=np.float32)
        assignment, count = _tile(corners, budget=100, quads=10)
        assert count == 1
        assert set(assignment.tolist()) == {0}

    def test_the_cut_follows_the_longer_horizontal_axis(self) -> None:
        """A room is wider than it is tall, and cutting the height would make
        tiles a reader has to load in pairs to see one wall."""
        corners = np.zeros((100, 4, 3), dtype=np.float32)
        corners[:, :, 0] = np.linspace(0, 10, 100)[:, None]  # long in x
        corners[:, :, 2] = np.linspace(0, 1, 100)[:, None]  # short in z
        assignment, count = _tile(corners, budget=30, quads=100)

        assert count == 4
        # Four cuts along x and none along z, so the tile index must track x.
        assert (np.diff(assignment) >= 0).all()
