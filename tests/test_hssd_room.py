"""Tests for HSSD room reconstruction (roadmap section 5.2, "per-room polygon
extrusion" breakthrough). Synthetic geometry only, no dependency on
downloaded HSSD data (hard constraint: tests run offline)."""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.geometry.hssd_room import (
    FurnitureInstance,
    RoomRegion,
    match_instances_to_regions,
)


def _square_region(name: str = "room", extrusion_height: float = 2.5) -> RoomRegion:
    # A simple 4m x 3m rectangular footprint in world (x, z), floor at y=0.
    poly_loop = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
        ]
    )
    return RoomRegion(
        name=name,
        label=name,
        poly_loop=poly_loop,
        floor_height=0.0,
        extrusion_height=extrusion_height,
    )


def test_extrude_produces_watertight_shell_with_correct_volume() -> None:
    region = _square_region()
    shell = region.extrude()

    assert shell.is_watertight
    assert shell.volume == pytest.approx(4.0 * 3.0 * 2.5, rel=1e-6)


def test_extrude_orientation_matches_world_axes() -> None:
    """The extruded shell must not be mirrored or have swapped axes.

    Verified against a concrete regression: an earlier version of this code
    used a rotation that flipped the Z axis, producing a mirrored shell
    whose vertices no longer aligned with the original poly_loop
    coordinates, which would silently misplace furniture matched by world
    coordinates.
    """
    region = _square_region()
    shell = region.extrude()

    assert shell.bounds[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert shell.bounds[1] == pytest.approx([4.0, 2.5, 3.0], abs=1e-6)

    floor_vertices = shell.vertices[np.isclose(shell.vertices[:, 1], 0.0, atol=1e-9)]
    floor_xz = {tuple(np.round(v[[0, 2]], 6)) for v in floor_vertices}
    expected_xz = {(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)}
    assert floor_xz == expected_xz


def test_extrude_with_floor_height_offset_is_translated() -> None:
    region = _square_region()
    region_upstairs = RoomRegion(
        name=region.name,
        label=region.label,
        poly_loop=region.poly_loop,
        floor_height=3.0,
        extrusion_height=region.extrusion_height,
    )
    shell = region_upstairs.extrude()
    assert shell.bounds[0][1] == pytest.approx(3.0, abs=1e-6)
    assert shell.bounds[1][1] == pytest.approx(5.5, abs=1e-6)


def test_match_instances_to_regions_uses_point_in_polygon() -> None:
    room_a = _square_region("room_a")
    room_b = RoomRegion(
        name="room_b",
        label="room_b",
        poly_loop=np.array(
            [
                [10.0, 0.0, 0.0],
                [14.0, 0.0, 0.0],
                [14.0, 0.0, 3.0],
                [10.0, 0.0, 3.0],
            ]
        ),
        floor_height=0.0,
        extrusion_height=2.5,
    )

    inside_a = FurnitureInstance(
        template_name="chair",
        translation=np.array([1.0, 0.5, 1.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.array([1.0, 1.0, 1.0]),
    )
    inside_b = FurnitureInstance(
        template_name="table",
        translation=np.array([12.0, 0.5, 1.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.array([1.0, 1.0, 1.0]),
    )
    outside_both = FurnitureInstance(
        template_name="lamp",
        translation=np.array([50.0, 0.5, 50.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.array([1.0, 1.0, 1.0]),
    )

    assignment = match_instances_to_regions([room_a, room_b], [inside_a, inside_b, outside_both])

    assert [inst.template_name for inst in assignment[0]] == ["chair"]
    assert [inst.template_name for inst in assignment[1]] == ["table"]


def test_furniture_instance_transform_matrix_applies_translation_rotation_scale() -> None:
    instance = FurnitureInstance(
        template_name="box",
        translation=np.array([1.0, 2.0, 3.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
        non_uniform_scale=np.array([2.0, 1.0, 1.0]),
    )
    matrix = instance.transform_matrix()
    point = np.array([1.0, 0.0, 0.0, 1.0])  # homogeneous
    transformed = matrix @ point
    # scale x by 2 -> (2, 0, 0), then translate by (1, 2, 3) -> (3, 2, 3)
    assert transformed[:3] == pytest.approx([3.0, 2.0, 3.0])
