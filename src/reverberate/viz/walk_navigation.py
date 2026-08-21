"""Pure first person navigation maths for the interior walkthrough viewer.

Kept free of PyVista, trame and file IO so it can be property tested in
isolation (see ``tests/test_walk_navigation.py``). The model is a person
walking on a fixed floor plane and turning their head: horizontal movement
and turning (yaw) are independent of where they are looking up or down
(pitch), exactly like a real pedestrian, and the walker cannot step through
walls: a move that would leave the room polygon is clamped back to the
nearest interior point at a minimum clearance from the boundary.
"""

from __future__ import annotations

import math

from shapely.geometry import Point, Polygon

__all__ = [
    "forward_vector",
    "right_vector",
    "walk_step",
    "clamp_into_polygon",
]


def forward_vector(yaw_degrees: float, pitch_degrees: float) -> tuple[float, float, float]:
    """Unit look direction for a head yawed and pitched from facing -Z.

    Yaw rotates left/right around the vertical (Y) axis, pitch tilts the
    head up (positive) or down (negative). Returned as ``(x, y, z)`` in the
    scene's Y-up, right-handed convention.
    """
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    return (
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        -math.cos(yaw) * math.cos(pitch),
    )


def right_vector(yaw_degrees: float) -> tuple[float, float]:
    """Unit horizontal strafe direction (XZ only), perpendicular to yaw."""
    yaw = math.radians(yaw_degrees)
    return (math.cos(yaw), math.sin(yaw))


def walk_step(
    position_xz: tuple[float, float],
    yaw_degrees: float,
    forward_amount: float,
    strafe_amount: float,
) -> tuple[float, float]:
    """Move on the horizontal plane, ignoring pitch (a person does not
    float up when looking up, or sink when looking down)."""
    fx, _fy, fz = forward_vector(yaw_degrees, 0.0)
    rx, rz = right_vector(yaw_degrees)
    x, z = position_xz
    return (
        x + fx * forward_amount + rx * strafe_amount,
        z + fz * forward_amount + rz * strafe_amount,
    )


def clamp_into_polygon(
    position_xz: tuple[float, float], polygon: Polygon, margin: float
) -> tuple[float, float]:
    """Keep a walker's position inside ``polygon`` with at least ``margin``
    clearance from any wall.

    If the requested position is already at a valid clearance, it is
    returned unchanged. Otherwise the nearest point on the polygon boundary
    is used, pulled inward by ``margin`` towards the polygon's centroid.
    This is a deliberately simple collision response (nudge back inside),
    not full physical contact resolution, which is enough for a walkthrough
    viewer.
    """
    point = Point(position_xz)
    inset = polygon.buffer(-margin) if margin > 0 else polygon
    if not inset.is_empty and inset.contains(point):
        return position_xz
    if inset.is_empty:
        # The room is smaller than twice the margin: fall back to the
        # unshrunk polygon so the walker is not trapped outside entirely.
        inset = polygon
    nearest = inset.exterior.interpolate(inset.exterior.project(point))
    return (float(nearest.x), float(nearest.y))
