"""Tests for the everyday-room partition, which is what the simulator receives.

Four properties are worth pinning, and each is a trap the code fell into once.

A wall separates even when the hole in it is wide, which is the rule that keeps
a 32 m2 lounge out of the room next door. A door-width passage separates even
when the rest of the boundary is open, which is what keeps a dressing room its
own volume. A corridor must not carry a merge from one of its ends to the
other. And a merged room must come back **connected**: HSSD holds neighbouring
regions 0.15 m apart with the wall band, so a plain union of the parts of one
room is several disjoint prisms, which is not a volume anything can be
simulated in.

The partition must also be **total**: a node with no owner vanishes from every
tier of the audit picture, which is the silent loss that view exists to
prevent.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from reverberate.geometry.hssd_room import RoomRegion
from reverberate.geometry.rooms import (
    DOOR_MAX_M,
    MIN_ROOM_M2,
    RoomPartition,
    _fill_unclaimed,
    _rasterise,
    everyday_rooms,
    room_holding,
)

#: The wall band HSSD leaves between two neighbouring regions.
BAND = 0.15


def region(name: str, x0: float, z0: float, x1: float, z1: float, label: str = "") -> RoomRegion:
    loop = np.array([[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]], dtype=float)
    return RoomRegion(
        name=name,
        label=label or name.split(".")[0],
        poly_loop=loop,
        floor_height=0.0,
        extrusion_height=2.8,
    )


def wall_between(x: float, z0: float, z1: float, gaps: list[tuple[float, float]]) -> Polygon:
    """The band at ``x`` from ``z0`` to ``z1``, minus the passages in ``gaps``."""
    band = box(x - BAND / 2, z0, x + BAND / 2, z1)
    return band.difference(unary_union([box(x - BAND, a, x + BAND, b) for a, b in gaps]))


def _horizontal_wall(z: float, x0: float, x1: float, gaps: list[tuple[float, float]]) -> Polygon:
    """The same band, lying the other way."""
    band = box(x0, z - BAND / 2, x1, z + BAND / 2)
    return band.difference(unary_union([box(a, z - BAND, b, z + BAND) for a, b in gaps]))


def names(rooms: list[RoomPartition]) -> dict[str, tuple[str, ...]]:
    return {room.name: room.regions for room in rooms}


class TestTheRules:
    def test_a_wide_passage_in_a_wall_still_separates(self) -> None:
        """The trap: a 1.4 m opening is wider than any door, but the 5 m of wall
        either side of it is what a person calls a wall."""
        left = region("left", 0.0, 0.0, 3.0, 6.0)
        right = region("right", 3.0 + BAND, 0.0, 6.0 + BAND, 6.0)
        walls = wall_between(3.0 + BAND / 2, 0.0, 6.0, [(2.3, 3.7)])

        rooms = everyday_rooms([left, right], walls)

        assert sorted(names(rooms)) == ["left", "right"]

    def test_a_door_separates_even_where_the_rest_is_open(self) -> None:
        """The other half: a dressing room shut by a door keeps its own volume,
        whatever the dataset labels it."""
        bedroom = region("bedroom", 0.0, 0.0, 4.0, 4.0)
        dressing = region("closet", 4.0 + BAND, 0.0, 6.0 + BAND, 4.0)
        walls = wall_between(4.0 + BAND / 2, 0.0, 4.0, [(0.1, 0.1 + DOOR_MAX_M - 0.2)])

        rooms = everyday_rooms([bedroom, dressing], walls)

        assert sorted(names(rooms)) == ["bedroom", "closet"]

    def test_a_corridor_does_not_join_the_rooms_at_its_two_ends(self) -> None:
        """It is attached to the largest room it serves, and carries no merge
        from one end to the other."""
        big = region("living room", 0.0, 0.0, 8.0, 6.0)
        corridor = region("hallway", 8.0 + BAND, 0.0, 10.0 + BAND, 6.0, label="hallway")
        small = region("office", 10.0 + 2 * BAND, 0.0, 13.0 + 2 * BAND, 6.0)
        walls = unary_union(
            [
                wall_between(8.0 + BAND / 2, 0.0, 6.0, [(0.5, 5.5)]),
                wall_between(10.0 + 1.5 * BAND, 0.0, 6.0, [(0.5, 5.5)]),
            ]
        )

        rooms = everyday_rooms([big, corridor, small], walls)

        assert sorted(names(rooms)) == ["living room-hallway", "office"]

    def test_a_region_under_two_square_metres_is_not_a_room(self) -> None:
        """`102344403`'s toilet is 0.80 m2 behind a 0.53 m gap: too small to be a
        room, too narrow for the door rule to fold it in."""
        bathroom = region("bathroom", 0.0, 0.0, 4.0, 3.0)
        wc = region("toilet", 4.0 + BAND, 0.0, 4.0 + BAND + 0.9, 0.9, label="toilet")
        assert wc.polygon_xz.area < MIN_ROOM_M2
        walls = wall_between(4.0 + BAND / 2, 0.0, 3.0, [(0.1, 0.6)])

        rooms = everyday_rooms([bathroom, wc], walls)

        assert room_holding(rooms, "toilet").regions == ("bathroom", "toilet")

    def test_the_room_a_merge_makes_is_connected(self) -> None:
        """A plain union of two regions is two prisms: HSSD holds them 0.15 m
        apart with the wall band, and a disconnected shell is not a volume."""
        left = region("living room", 0.0, 0.0, 3.0, 6.0)
        right = region("kitchen", 3.0 + BAND, 0.0, 6.0 + BAND, 6.0)
        walls = wall_between(3.0 + BAND / 2, 0.0, 6.0, [(0.4, 5.6)])

        room = everyday_rooms([left, right], walls)[0]

        assert not isinstance(room.polygon, MultiPolygon)
        assert set(room.regions) == {"living room", "kitchen"}


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
