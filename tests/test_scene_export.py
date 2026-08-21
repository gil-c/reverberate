"""Tests for the glTF export of a reconstructed room."""

from __future__ import annotations

import numpy as np
import trimesh

from reverberate.geometry.hssd_room import RoomRegion
from reverberate.viz.scene_export import (
    absorption_colour,
    decimate_for_display,
    shell_surface_labels,
)


def square_region(height: float = 2.5) -> RoomRegion:
    loop = np.array([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]])
    return RoomRegion(
        name="room", label="living room", poly_loop=loop, floor_height=0.0, extrusion_height=height
    )


def test_shell_faces_split_into_floor_wall_and_ceiling() -> None:
    shell = square_region().extrude()
    labels = shell_surface_labels(shell)
    assert set(labels) == {"floor", "wall", "ceiling"}


def test_floor_and_ceiling_are_at_the_expected_heights() -> None:
    """Guards the axis convention: a mirrored shell puts the floor on top."""
    region = square_region()
    shell = region.extrude()
    labels = shell_surface_labels(shell)
    centres = shell.triangles_center[:, 1]
    assert centres[labels == "floor"].max() < centres[labels == "ceiling"].min()
    assert centres[labels == "floor"].mean() == region.floor_height
    assert centres[labels == "ceiling"].mean() == region.floor_height + region.extrusion_height


def test_absorption_colour_runs_from_blue_to_red() -> None:
    reflective = absorption_colour(0.0)
    absorptive = absorption_colour(1.0)
    assert reflective[2] > reflective[0]
    assert absorptive[0] > absorptive[2]
    assert absorption_colour(5.0).tolist() == absorptive.tolist()  # clipped, not wrapped


def test_decimation_reduces_faces_but_keeps_the_shape() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)
    reduced = decimate_for_display(sphere, face_budget=200)
    assert len(reduced.faces) < len(sphere.faces)
    assert reduced.bounds == pytest_approx(sphere.bounds)


def test_meshes_under_the_budget_are_returned_untouched() -> None:
    box = trimesh.creation.box()
    assert decimate_for_display(box, face_budget=1000) is box


def pytest_approx(value: np.ndarray) -> object:
    import pytest

    return pytest.approx(value, abs=0.05)
