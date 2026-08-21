"""Tests for apartment assembly: rooms joined through the walls' own doorways."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from reverberate.geometry.apartment import (
    Storey,
    build_storey,
    clean_outline,
    extrude_storey,
    find_doorways,
    group_by_storey,
    wall_footprint,
)
from reverberate.geometry.hssd_room import RoomRegion


def room(x0: float, x1: float, z0: float, z1: float, floor: float = 0.0) -> RoomRegion:
    loop = np.array(
        [[x0, floor, z0], [x1, floor, z0], [x1, floor, z1], [x0, floor, z1]], dtype=float
    )
    return RoomRegion(
        name=f"room_{x0}_{z0}",
        label="bedroom",
        poly_loop=loop,
        floor_height=floor,
        extrusion_height=2.8,
    )


def wall_with_gap(
    x: float, z0: float, z1: float, gap_centre: float, gap: float = 0.9
) -> trimesh.Trimesh:
    """Two wall panels leaving a doorway between them, as a stage would model it."""
    lower = trimesh.creation.box(
        extents=(0.1, 2.8, gap_centre - gap / 2 - z0),
        transform=trimesh.transformations.translation_matrix(  # type: ignore[no-untyped-call]
            [x, 1.4, (z0 + gap_centre - gap / 2) / 2]
        ),
    )
    upper = trimesh.creation.box(
        extents=(0.1, 2.8, z1 - (gap_centre + gap / 2)),
        transform=trimesh.transformations.translation_matrix(  # type: ignore[no-untyped-call]
            [x, 1.4, (z1 + gap_centre + gap / 2) / 2]
        ),
    )
    combined = trimesh.util.concatenate([lower, upper])
    assert isinstance(combined, trimesh.Trimesh)
    return combined


def test_regions_are_grouped_into_storeys_by_floor_height() -> None:
    regions = [room(0, 4, 0, 4), room(0, 4, 5, 9), room(0, 4, 0, 4, floor=2.5)]
    storeys = group_by_storey(regions)
    assert sorted(len(group) for group in storeys.values()) == [1, 2]


def test_near_identical_floor_heights_are_the_same_storey() -> None:
    """Authored heights wobble slightly; a 5 cm difference is not a new floor."""
    storeys = group_by_storey([room(0, 4, 0, 4), room(5, 9, 0, 4, floor=0.05)])
    assert len(storeys) == 1


def test_a_gap_touching_only_one_room_is_not_a_doorway() -> None:
    """Otherwise the space outside the front door would count as a passage."""
    rooms = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    assert find_doorways(rooms, Polygon()) == []


def test_a_gap_touching_two_rooms_is_a_doorway() -> None:
    rooms = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        Polygon([(4.2, 0), (8, 0), (8, 4), (4.2, 4)]),
    ]
    assert len(find_doorways(rooms, Polygon())) == 1


def test_a_wall_between_two_rooms_blocks_the_passage() -> None:
    rooms = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        Polygon([(4.2, 0), (8, 0), (8, 4), (4.2, 4)]),
    ]
    wall = Polygon([(4.05, -1), (4.15, -1), (4.15, 5), (4.05, 5)])
    assert find_doorways(rooms, wall) == []


def test_wall_footprint_reads_the_stage_slice_not_the_room_interior() -> None:
    """Filling the section's closed loops would turn whole rooms into wall."""
    box = trimesh.creation.box(extents=(0.1, 2.8, 4.0))
    footprint = wall_footprint(box, 1.0)
    assert not footprint.is_empty
    # A 10 cm wall over 4 m is well under a square metre; a filled room would
    # be several square metres.
    assert footprint.area < 1.5


def test_two_rooms_joined_through_a_real_doorway_become_one_storey() -> None:
    regions = [room(0, 4, 0, 4), room(4.3, 8, 0, 4)]
    stage = wall_with_gap(x=4.15, z0=0.0, z1=4.0, gap_centre=2.0)
    storey = build_storey(regions, stage)
    assert storey.doorways == 1
    assert len(storey.polygons) == 1


def test_a_solid_wall_leaves_the_rooms_separate() -> None:
    regions = [room(0, 4, 0, 4), room(4.3, 8, 0, 4)]
    stage = trimesh.creation.box(
        extents=(0.1, 2.8, 6.0),
        transform=trimesh.transformations.translation_matrix([4.15, 1.4, 2.0]),  # type: ignore[no-untyped-call]
    )
    storey = build_storey(regions, stage)
    assert storey.doorways == 0
    assert len(storey.polygons) == 2


def test_extruded_storey_is_watertight_so_pyroomacoustics_can_use_it() -> None:
    regions = [room(0, 4, 0, 4), room(4.3, 8, 0, 4)]
    storey = build_storey(regions, wall_with_gap(4.15, 0.0, 4.0, 2.0))
    mesh = extrude_storey(storey)
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extruded_storey_sits_between_its_floor_and_ceiling() -> None:
    """Guards the axis convention: a mirrored extrusion would invert these."""
    regions = [room(0, 4, 0, 4, floor=2.5)]
    storey = build_storey(regions, trimesh.creation.box(extents=(0.01, 0.01, 0.01)))
    mesh = extrude_storey(storey)
    assert mesh.bounds[0][1] == pytest.approx(2.5)
    assert mesh.bounds[1][1] == pytest.approx(5.3)


def test_cleaning_preserves_the_area_it_tidies() -> None:
    rough = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]).union(
        Polygon([(3.999, 0), (4.001, 0), (4.001, 4), (3.999, 4)])
    )
    cleaned = clean_outline(rough)
    assert cleaned.area == pytest.approx(rough.area, rel=0.01)


def test_storey_summary_mentions_rooms_doorways_and_area() -> None:
    storey = Storey(
        floor_height=0.0,
        ceiling_height=2.8,
        walkable=MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]),
        rooms=[room(0, 2, 0, 2)],
        doorways=3,
    )
    assert "1 rooms" in storey.summary()
    assert "3 doorways" in storey.summary()
