"""Tests for where the source and the listener stand.

The properties that matter here are the two defects this module was written to
remove: a source must never be sampled inside a piece of furniture, and the
height of neither end may be a constant. Synthetic geometry only, so the whole
file runs offline and in well under a second, per the roadmap's hard
constraints.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon

from reverberate.geometry.apartment import Storey, build_storey
from reverberate.geometry.hssd_room import RoomRegion
from reverberate.geometry.placement import (
    APPLIANCE,
    REFERENCE_OMNI,
    SEATED_EAR_HEIGHT,
    SOURCE_ARCHETYPES,
    STANDING_EAR_HEIGHT,
    VOICE,
    choose_archetype,
    footprint_of,
    largest_free_room,
    sample_group,
    sample_pair,
    sampling_area,
)


def square_storey(size: float = 8.0) -> Storey:
    loop = np.array([[0, 0, 0], [size, 0, 0], [size, 0, size], [0, 0, size]], dtype=float)
    region = RoomRegion(
        name="room", label="bedroom", poly_loop=loop, floor_height=0.0, extrusion_height=2.8
    )
    tiny = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    return build_storey([region], tiny)


def sofa_at(x: float, z: float, size: float = 2.0) -> Polygon:
    """A square obstacle footprint, as ``furniture_footprints`` would produce."""
    half = size / 2.0
    return Polygon(
        [(x - half, z - half), (x + half, z - half), (x + half, z + half), (x - half, z + half)]
    )


def test_footprint_is_the_xz_hull_and_ignores_height() -> None:
    """A tall thin wardrobe and a low wide table block floor area by their
    footprint, not by their volume, so the hull is taken in XZ only."""
    box = trimesh.creation.box(extents=(2.0, 3.0, 1.0))

    footprint = footprint_of(box)

    assert footprint.bounds == pytest.approx((-1.0, -0.5, 1.0, 0.5))


def test_furniture_is_removed_from_the_sampling_area() -> None:
    storey = square_storey()
    sofa = sofa_at(4.0, 4.0)

    free = sampling_area(storey, footprints=[sofa], min_wall_distance=0.5)

    assert not free.contains(Point(4.0, 4.0))
    assert free.area < sampling_area(storey, min_wall_distance=0.5).area


def test_no_sampled_position_ever_lands_inside_furniture() -> None:
    """The point of the module: this is structural, not filtered afterwards.

    A single obstacle covering most of the room makes the failure mode likely
    enough that a plain rejection sampler over the walkable area would hit it.
    """
    storey = square_storey()
    sofa = sofa_at(4.0, 4.0, size=5.0)
    free = sampling_area(storey, footprints=[sofa], min_wall_distance=0.5)
    rng = np.random.default_rng(0)

    for _ in range(50):
        pair = sample_pair(storey, rng, area=free)
        for placement in (pair.source, pair.listener):
            point = Point(float(placement.position[0]), float(placement.position[2]))
            assert not sofa.contains(point)


def test_sampled_heights_are_not_constant_and_respect_the_archetype() -> None:
    storey = square_storey()
    rng = np.random.default_rng(0)

    heights = {
        float(sample_pair(storey, rng, archetype=APPLIANCE).source.position[1]) for _ in range(40)
    }

    assert len(heights) > 1, "an appliance drawn 40 times must not always sit at one height"
    assert all(APPLIANCE.min_height <= height <= APPLIANCE.max_height for height in heights)


def test_listener_height_is_a_standing_or_seated_ear_height() -> None:
    storey = square_storey()
    rng = np.random.default_rng(0)

    heights = {float(sample_pair(storey, rng).listener.position[1]) for _ in range(40)}

    assert heights == {SEATED_EAR_HEIGHT, STANDING_EAR_HEIGHT}


def test_heights_are_offset_by_the_storey_floor() -> None:
    """An upper storey's listener stands on *its* floor, not on the ground."""
    storey = square_storey()
    upper = Storey(
        floor_height=3.0,
        ceiling_height=storey.ceiling_height + 3.0,
        walkable=storey.walkable,
        rooms=storey.rooms,
        doorways=storey.doorways,
    )
    rng = np.random.default_rng(0)

    pair = sample_pair(upper, rng, archetype=REFERENCE_OMNI)

    assert float(pair.source.position[1]) == pytest.approx(3.0 + REFERENCE_OMNI.min_height)


def test_reference_omni_has_no_directivity_and_voice_does() -> None:
    """The control archetype must be genuinely omnidirectional: if it carried a
    directivity, it could not isolate geometry from orientation."""
    storey = square_storey()
    rng = np.random.default_rng(0)

    omni = sample_pair(storey, rng, archetype=REFERENCE_OMNI).source
    voice = sample_pair(storey, rng, archetype=VOICE).source

    assert omni.directivity() is None
    assert voice.directivity() is not None


def test_orientation_is_sampled_rather_than_fixed() -> None:
    storey = square_storey()
    rng = np.random.default_rng(0)

    azimuths = {round(sample_pair(storey, rng).source.azimuth, 6) for _ in range(30)}

    assert len(azimuths) > 25


def test_reference_omni_is_about_half_the_draws() -> None:
    """It is the control, so it must dominate; a rare control is a useless one."""
    rng = np.random.default_rng(0)

    draws = [choose_archetype(rng).name for _ in range(2000)]

    share = draws.count(REFERENCE_OMNI.name) / len(draws)
    assert 0.45 < share < 0.55
    assert set(draws) == {archetype.name for archetype in SOURCE_ARCHETYPES}


def test_same_room_constraint_is_honoured_across_two_rooms() -> None:
    """Inter-room pairs are how a response through a doorway gets sampled."""
    left = RoomRegion(
        name="left",
        label="bedroom",
        poly_loop=np.array([[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]], dtype=float),
        floor_height=0.0,
        extrusion_height=2.8,
    )
    right = RoomRegion(
        name="right",
        label="kitchen",
        poly_loop=np.array([[4, 0, 0], [8, 0, 0], [8, 0, 4], [4, 0, 4]], dtype=float),
        floor_height=0.0,
        extrusion_height=2.8,
    )
    storey = build_storey([left, right], trimesh.creation.box(extents=(0.01, 0.01, 0.01)))
    rng = np.random.default_rng(0)

    for _ in range(10):
        assert sample_pair(storey, rng, same_room=True).same_room
        assert not sample_pair(storey, rng, same_room=False).same_room


def test_sampling_area_refuses_a_storey_with_no_room_left() -> None:
    """Better a loud failure than a source quietly placed in a wall."""
    storey = square_storey(size=2.0)

    with pytest.raises(ValueError, match="no free floor area"):
        sampling_area(storey, footprints=[sofa_at(1.0, 1.0, size=4.0)], min_wall_distance=0.5)


def two_room_storey() -> Storey:
    """A large room and a small one, so "largest free" has something to choose."""
    big = RoomRegion(
        name="living",
        label="living room",
        poly_loop=np.array([[0, 0, 0], [8, 0, 0], [8, 0, 6], [0, 0, 6]], dtype=float),
        floor_height=0.0,
        extrusion_height=2.8,
    )
    small = RoomRegion(
        name="closet",
        label="bedroom",
        poly_loop=np.array([[8, 0, 0], [11, 0, 0], [11, 0, 2], [8, 0, 2]], dtype=float),
        floor_height=0.0,
        extrusion_height=2.8,
    )
    return build_storey([big, small], trimesh.creation.box(extents=(0.01, 0.01, 0.01)))


def test_a_group_puts_every_placement_in_one_room() -> None:
    """Twelve responses of one run share a room, so they share a decay to compare."""
    storey = two_room_storey()
    area = sampling_area(storey)
    group = sample_group(storey, np.random.default_rng(0), area, sources=2, receivers=6, seed=0)

    assert group.response_count == 12
    assert group.room == "living"
    for placement in group.sources + group.receivers:
        assert placement.room == "living"


def test_the_largest_free_room_is_chosen_by_free_floor_not_gross_area() -> None:
    """A big room filled with furniture is a worse candidate than a small empty one."""
    storey = two_room_storey()
    filled = sampling_area(storey, [sofa_at(4.0, 3.0, size=7.5)])
    assert largest_free_room(storey, filled) == "closet"
    assert largest_free_room(storey, sampling_area(storey)) == "living"


def test_no_placement_lands_inside_furniture() -> None:
    storey = two_room_storey()
    sofa = sofa_at(4.0, 3.0, size=3.0)
    area = sampling_area(storey, [sofa])
    group = sample_group(storey, np.random.default_rng(3), area, room="living", seed=3)

    for placement in group.sources + group.receivers:
        assert not sofa.contains(Point(placement.position[0], placement.position[2]))


def test_placements_are_kept_apart() -> None:
    """Two receivers inside one cell would return the same response twice."""
    storey = two_room_storey()
    group = sample_group(
        storey, np.random.default_rng(11), sampling_area(storey), min_separation=0.5, seed=11
    )
    points = np.array([p.position[[0, 2]] for p in group.sources + group.receivers], dtype=float)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 0.5


def test_a_separation_the_room_cannot_satisfy_is_refused() -> None:
    """Reported, not silently relaxed into overlapping placements."""
    storey = two_room_storey()
    with pytest.raises(ValueError, match="at least 20.0 m apart"):
        sample_group(
            storey, np.random.default_rng(0), sampling_area(storey), min_separation=20.0, seed=0
        )


def test_receivers_are_at_ear_height_and_sources_are_not_all_the_same() -> None:
    storey = two_room_storey()
    group = sample_group(
        storey, np.random.default_rng(5), sampling_area(storey), receivers=6, seed=5
    )
    heights = {round(float(r.position[1]), 3) for r in group.receivers}
    assert heights <= {STANDING_EAR_HEIGHT, SEATED_EAR_HEIGHT}
    assert len(heights) == 2, "every receiver was given the same posture"


def test_the_same_seed_gives_the_same_group() -> None:
    """The run record carries a seed, so the seed has to be sufficient."""
    storey = two_room_storey()
    area = sampling_area(storey)
    first = sample_group(storey, np.random.default_rng(42), area, seed=42)
    again = sample_group(storey, np.random.default_rng(42), area, seed=42)
    other = sample_group(storey, np.random.default_rng(43), area, seed=43)

    assert first.record() == again.record()
    assert first.record() != other.record()


def test_the_record_carries_the_positions_and_the_seed() -> None:
    storey = two_room_storey()
    record = sample_group(storey, np.random.default_rng(1), sampling_area(storey), seed=1).record()

    assert record["seed"] == 1
    assert record["room"] == "living"
    sources = record["sources"]
    receivers = record["receivers"]
    assert isinstance(sources, list) and isinstance(receivers, list)
    assert len(sources) == 2 and len(receivers) == 6
    assert len(sources[0]["position"]) == 3
    assert sources[0]["archetype"] is not None


def test_a_room_that_does_not_exist_is_named_in_the_error() -> None:
    storey = two_room_storey()
    with pytest.raises(ValueError, match="no room named 'kitchen'"):
        sample_group(storey, np.random.default_rng(0), sampling_area(storey), room="kitchen")
