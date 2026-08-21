"""Tests for the offline furniture placement audit."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reverberate.geometry.audit_placement import (
    PiecePlacement,
    RegionAudit,
    audit_region,
    footprint_xz,
)
from reverberate.geometry.hssd_room import FurnitureInstance, RoomRegion


def square_region(half_size: float = 2.0, floor_height: float = 0.0) -> RoomRegion:
    loop = np.array(
        [
            [-half_size, floor_height, -half_size],
            [half_size, floor_height, -half_size],
            [half_size, floor_height, half_size],
            [-half_size, floor_height, half_size],
        ]
    )
    return RoomRegion(
        name="room",
        label="bedroom",
        poly_loop=loop,
        floor_height=floor_height,
        extrusion_height=2.5,
    )


def unit_instance(translation: tuple[float, float, float]) -> FurnitureInstance:
    return FurnitureInstance(
        template_name="unit",
        translation=np.array(translation, dtype=float),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.ones(3),
    )


def test_footprint_uses_world_x_and_z_not_y() -> None:
    """A box stretched along Y must still have a small XZ footprint."""
    mesh = trimesh.creation.box(extents=(1.0, 10.0, 1.0))
    assert footprint_xz(mesh).area == pytest.approx(1.0)


def test_piece_fully_inside_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = run_audit(monkeypatch, [unit_instance((0.0, 0.5, 0.0))])
    assert audit.escaped == []


def test_piece_beyond_the_wall_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """This is exactly the failure mode a mirrored shell produced on real data."""
    audit = run_audit(monkeypatch, [unit_instance((10.0, 0.5, 0.0))])
    assert [placement.template_name for placement in audit.escaped] == ["unit"]


def test_mirroring_the_room_in_z_makes_placement_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the PR #6 regression: a Z-mirrored shell strands off-centre furniture.

    The room polygon is deliberately asymmetric in Z, so mirroring it (the
    effect of swapping coordinate columns rather than rotating) moves the room
    away from furniture that is correctly placed relative to the true polygon.
    """
    loop = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 5.0], [-1.0, 0.0, 5.0]])
    region = RoomRegion(
        name="room", label="bedroom", poly_loop=loop, floor_height=0.0, extrusion_height=2.5
    )
    mirrored = RoomRegion(
        name="room",
        label="bedroom",
        poly_loop=loop * np.array([1.0, 1.0, -1.0]),
        floor_height=0.0,
        extrusion_height=2.5,
    )
    instance = unit_instance((0.0, 0.5, 3.0))

    assert audit_region_with_unit_box(monkeypatch, region, instance).escaped == []
    assert len(audit_region_with_unit_box(monkeypatch, mirrored, instance).escaped) == 1


def test_below_floor_and_above_ceiling_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    sunk = run_audit(monkeypatch, [unit_instance((0.0, -1.0, 0.0))])
    assert len(sunk.sunk) == 1
    tall = run_audit(monkeypatch, [unit_instance((0.0, 3.0, 0.0))])
    assert len(tall.above_ceiling) == 1
    assert len(tall.floating) == 1


def run_audit(monkeypatch: pytest.MonkeyPatch, instances: list[FurnitureInstance]) -> RegionAudit:
    patch_collider(monkeypatch)
    from pathlib import Path

    return audit_region("scene", square_region(), instances, Path("unused"))


def audit_region_with_unit_box(
    monkeypatch: pytest.MonkeyPatch, region: RoomRegion, instance: FurnitureInstance
) -> RegionAudit:
    patch_collider(monkeypatch)
    from pathlib import Path

    return audit_region("scene", region, [instance], Path("unused"))


def patch_collider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reverberate.geometry.audit_placement.load_collider_mesh",
        lambda objects_dir, template_name: trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
    )


def test_escape_fraction_ignores_furniture_merely_touching_a_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A piece flush against the wall, clipping slightly through, is not a failure."""
    audit = run_audit(monkeypatch, [unit_instance((1.7, 0.5, 0.0))])
    assert audit.escaped == []
    assert isinstance(audit.placements[0], PiecePlacement)
