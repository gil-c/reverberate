"""Tests for the everyday-room partition the audit view draws against.

Two properties matter and neither is obvious from the code. A closet must join
the room it *opens into*, which W34 measured as disagreeing with the shared
boundary criterion on five of seven closets in the shipped scene. And the
partition must be **total**: a node with no owner vanishes from every tier of
the picture, which is the silent loss the whole view exists to prevent.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from reverberate.geometry.hssd_room import RoomRegion
from reverberate.geometry.rooms import (
    MIN_ROOM_AREA,
    RoomPartition,
    _fill_unclaimed,
    _rasterise,
    merge_closets,
)


def region(name: str, x0: float, z0: float, x1: float, z1: float) -> RoomRegion:
    loop = np.array([[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]], dtype=float)
    return RoomRegion(
        name=name,
        label=name.split(".")[0],
        poly_loop=loop,
        floor_height=0.0,
        extrusion_height=2.8,
    )


class TestMergeClosets:
    def test_a_closet_joins_the_room_its_doorway_reaches(self) -> None:
        """The two rooms touch the closet equally, so only the gap decides."""
        left = region("left", 0.0, 0.0, 3.0, 3.0)
        right = region("right", 3.0, 0.0, 6.0, 3.0)
        # A closet sitting on the wall between them, touching both by the same
        # length. Shared boundary cannot separate the two; the doorway can.
        closet = region("closet", 2.0, 3.0, 3.0 + 1.0, 3.0 + 1.5)
        assert closet.polygon_xz.area < MIN_ROOM_AREA
        doorway = Polygon([(2.2, 2.9), (2.8, 2.9), (2.8, 3.1), (2.2, 3.1)])  # into left

        rooms = merge_closets([left, right, closet], [doorway])

        joined = {room.name: room.regions for room in rooms}
        assert joined["left"] == ("left", "closet")
        assert joined["right"] == ("right",)

    def test_the_larger_room_wins_a_doorway_that_reaches_both(self) -> None:
        left = region("left", 0.0, 0.0, 3.0, 3.0)
        right = region("right", 3.0, 0.0, 9.0, 3.0)
        closet = region("closet", 2.5, 3.0, 3.5, 4.0)
        doorway = Polygon([(2.4, 2.9), (3.6, 2.9), (3.6, 3.1), (2.4, 3.1)])

        rooms = merge_closets([left, right, closet], [doorway])

        assert {r.name: r.regions for r in rooms}["right"] == ("right", "closet")

    def test_a_closet_no_doorway_reaches_falls_back_to_shared_boundary(self) -> None:
        """The fallback is the criterion W34 measured as ambiguous, so it is
        used only when the reliable one is silent -- never in preference."""
        left = region("left", 0.0, 0.0, 3.0, 3.0)
        right = region("right", 3.0, 0.0, 6.0, 3.0)
        closet = region("closet", 0.5, 3.0, 1.5, 4.0)  # wholly above left

        rooms = merge_closets([left, right, closet], [])

        assert {r.name: r.regions for r in rooms}["left"] == ("left", "closet")

    def test_a_region_at_the_threshold_stays_a_room_of_its_own(self) -> None:
        big = region("big", 0.0, 0.0, 10.0, 10.0)
        edge = region("edge", 10.0, 0.0, 12.0, MIN_ROOM_AREA / 2.0)

        rooms = merge_closets([big, edge], [])

        assert sorted(room.name for room in rooms) == ["big", "edge"]


class TestRaster:
    @staticmethod
    def _two_rooms() -> list[RoomPartition]:
        return [
            RoomPartition("a", "a", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), ("a",)),
            RoomPartition("b", "b", Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), ("b",)),
        ]

    def test_cells_inside_a_room_carry_its_index(self) -> None:
        xv = np.linspace(0.0, 3.0, 31)
        zv = np.linspace(0.0, 1.0, 11)

        raster = _rasterise(self._two_rooms(), xv, zv)

        # Interior cells, not corners: ``shapely.contains_xy`` is strict, so a
        # cell falling exactly on a room's edge is unclaimed here and picked up
        # by the nearest-room fill below. That is the right division of labour
        # and it is worth pinning, because the alternative -- ``covers`` -- would
        # hand a cell on a shared wall to whichever room was rasterised first.
        assert raster[3, 5] == 0
        assert raster[27, 5] == 1
        assert raster[0, 0] == -1  # a corner, exactly on the boundary
        assert raster[15, 5] == -1  # the gap between them

    def test_an_unclaimed_cell_takes_the_nearer_room(self) -> None:
        xv = np.linspace(0.0, 3.0, 31)
        zv = np.linspace(0.0, 1.0, 11)

        filled = _fill_unclaimed(_rasterise(self._two_rooms(), xv, zv))

        assert filled[12, 5] == 0  # nearer a, which ends at x = 1.0
        assert filled[18, 5] == 1  # nearer b, which starts at x = 2.0
