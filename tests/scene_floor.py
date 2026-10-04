"""A hand-built dwelling for the tests of ``reverberate.scenes``: two rooms and a door.

No HSSD: the floor is drawn here. A living room of 5 m by 4 m and a bedroom of
3.8 m by 4 m, joined by a door one metre wide; a couch and an armchair in the
first, a chair in the second. The storey's floor is at ``y = 0.3`` and not at
zero, so nothing that assumes zero passes by accident.
"""

from __future__ import annotations

from functools import lru_cache

from shapely.geometry import box
from shapely.ops import unary_union

from reverberate.experiments.w40_volume_field.plan import FURNITURE_CLEARANCE_M, WALL_SETBACK_M
from reverberate.scenes import (
    Floor,
    Layout,
    Parameters,
    Recipe,
    Room,
    SeatObject,
    build_layout,
    generate,
)
from reverberate.scenes.recipe import Dwelling

FLOOR_Y_M = 0.3


@lru_cache(maxsize=1)
def two_room_floor() -> Floor:
    living = box(0.0, 0.0, 5.0, 4.0)
    bedroom = box(5.2, 0.0, 9.0, 4.0)
    door = box(4.99, 1.5, 5.21, 2.5)
    walkable = unary_union([living, bedroom, door])
    furniture = (
        SeatObject("couch_0", "couch", box(1.0, 0.05, 3.0, 0.95), (0.0, 1.0)),
        SeatObject("seat_1", "seat", box(0.4, 2.6, 0.9, 3.1), (1.0, 0.0)),
        SeatObject("seat_2", "seat", box(7.4, 3.2, 7.9, 3.7), (0.0, -1.0)),
    )
    blocked = unary_union([item.footprint.buffer(FURNITURE_CLEARANCE_M) for item in furniture])
    return Floor(
        floor_y_m=FLOOR_Y_M,
        free=walkable.buffer(-WALL_SETBACK_M).difference(blocked),
        walkable=walkable,
        rooms=(Room("living room", living), Room("bedroom", bedroom)),
        objects=furniture,
    )


@lru_cache(maxsize=1)
def two_room_layout() -> Layout:
    return build_layout(Dwelling("two_rooms", "synthetic_two_rooms", FLOOR_Y_M), two_room_floor())


#: A short scene with every kind of source, small enough to draw in a moment.
SMALL = Parameters(
    duration_s=90.0,
    near_voice_count=(2, 2),
    far_voice_count=(2, 2),
    noise_count=(1, 1),
    dwell_s=(8.0, 25.0),
    listener_rest_s=(6.0, 20.0),
)


@lru_cache(maxsize=8)
def small_recipe(seed: int = 1) -> Recipe:
    return generate(
        two_room_layout(),
        SMALL,
        seed,
        allow_placeholder_clips=True,
        allow_placeholder_assets=True,
    )
