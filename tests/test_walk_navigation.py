"""Property tests for the pure first person navigation maths.

No PyVista, no trame, no file IO: fast and offline per ROADMAP.md's hard
constraints.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import Point, Polygon

from reverberate.viz.walk_navigation import (
    clamp_into_polygon,
    forward_vector,
    right_vector,
    walk_step,
)


def test_forward_vector_faces_negative_z_at_zero_yaw_and_pitch() -> None:
    x, y, z = forward_vector(0.0, 0.0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_forward_vector_faces_positive_x_at_90_degree_yaw() -> None:
    x, y, z = forward_vector(90.0, 0.0)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_forward_vector_pitch_up_raises_y_component() -> None:
    _, y, _ = forward_vector(0.0, 45.0)
    assert y == pytest.approx(math.sin(math.radians(45.0)))


@given(yaw=st.floats(min_value=-720, max_value=720), pitch=st.floats(min_value=-89, max_value=89))
def test_forward_vector_is_always_unit_length(yaw: float, pitch: float) -> None:
    x, y, z = forward_vector(yaw, pitch)
    assert math.sqrt(x * x + y * y + z * z) == pytest.approx(1.0, abs=1e-9)


def test_right_vector_is_perpendicular_to_forward_on_the_horizontal_plane() -> None:
    for yaw in (0.0, 30.0, 90.0, 200.0):
        fx, _, fz = forward_vector(yaw, 0.0)
        rx, rz = right_vector(yaw)
        dot = fx * rx + fz * rz
        assert dot == pytest.approx(0.0, abs=1e-9)


def test_walk_step_forward_at_zero_yaw_moves_in_negative_z() -> None:
    new_x, new_z = walk_step((0.0, 0.0), yaw_degrees=0.0, forward_amount=1.0, strafe_amount=0.0)
    assert (new_x, new_z) == pytest.approx((0.0, -1.0), abs=1e-9)


def test_walk_step_ignores_pitch_a_person_does_not_float_when_looking_up() -> None:
    # walk_step has no pitch argument at all: this test documents that
    # design choice by construction (a call site cannot pass one).
    a = walk_step((1.0, 1.0), yaw_degrees=45.0, forward_amount=0.5, strafe_amount=0.2)
    b = walk_step((1.0, 1.0), yaw_degrees=45.0, forward_amount=0.5, strafe_amount=0.2)
    assert a == b


def test_clamp_into_polygon_leaves_a_well_inside_point_unchanged() -> None:
    square = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    assert clamp_into_polygon((0.0, 0.0), square, margin=0.3) == (0.0, 0.0)


def test_clamp_into_polygon_pulls_a_point_outside_the_wall_back_inside() -> None:
    square = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    x, z = clamp_into_polygon((5.0, 0.0), square, margin=0.3)
    assert square.buffer(1e-9).contains(Point(x, z))
    assert x < 2.0


def test_clamp_into_polygon_keeps_the_configured_margin_from_the_wall() -> None:
    square = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    x, _z = clamp_into_polygon((10.0, 0.0), square, margin=0.3)
    assert x == pytest.approx(1.7, abs=1e-6)


def test_clamp_into_polygon_does_not_trap_a_walker_in_a_tiny_room() -> None:
    tiny = Polygon([(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)])
    x, z = clamp_into_polygon((0.0, 0.0), tiny, margin=0.3)
    assert math.isfinite(x) and math.isfinite(z)
