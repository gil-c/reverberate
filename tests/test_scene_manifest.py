"""Tests for the room shell surfaces and the browser manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from reverberate.geometry.hssd_room import FurnitureInstance, RoomRegion
from reverberate.viz.label_palette import SHELL_LABEL_COLOURS, category_colour, rgba
from reverberate.viz.room_surfaces import (
    absorption_colour,
    shell_mesh,
    shell_surface_labels,
)
from reverberate.viz.scene_manifest import build_instances, column_major, link_asset


def square_region(height: float = 2.5) -> RoomRegion:
    loop = np.array([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]])
    return RoomRegion(
        name="room", label="living room", poly_loop=loop, floor_height=0.0, extrusion_height=height
    )


def test_shell_faces_split_into_floor_wall_and_ceiling() -> None:
    labels = shell_surface_labels(square_region().extrude())
    assert set(labels) == {"floor", "wall", "ceiling"}


def test_floor_sits_below_ceiling_at_the_authored_heights() -> None:
    """Guards the axis convention: a mirrored shell puts the floor on top."""
    region = square_region()
    shell = region.extrude()
    labels = shell_surface_labels(shell)
    centres = shell.triangles_center[:, 1]
    assert centres[labels == "floor"].max() < centres[labels == "ceiling"].min()
    assert centres[labels == "floor"].mean() == pytest.approx(region.floor_height)
    assert centres[labels == "ceiling"].mean() == pytest.approx(
        region.floor_height + region.extrusion_height
    )


def test_shell_mesh_colours_every_face_from_its_surface() -> None:
    shell = shell_mesh(square_region(), SHELL_LABEL_COLOURS)
    assert isinstance(shell.visual, trimesh.visual.ColorVisuals)
    colours = shell.visual.face_colors
    labels = shell_surface_labels(shell)
    for surface, colour in SHELL_LABEL_COLOURS.items():
        assert (colours[labels == surface] == rgba(colour)).all()


def test_absorption_colour_runs_from_blue_to_red_and_clips() -> None:
    reflective = absorption_colour(0.0)
    absorptive = absorption_colour(1.0)
    assert reflective[2] > reflective[0]
    assert absorptive[0] > absorptive[2]
    assert absorption_colour(5.0).tolist() == absorptive.tolist()


def test_category_colour_is_stable_and_distinguishes_categories() -> None:
    assert category_colour("sofa") == category_colour("sofa")
    assert category_colour("sofa") != category_colour("lamp")


def test_column_major_puts_translation_where_three_js_reads_it() -> None:
    """three.js reads translation from elements 12, 13, 14 of the flat array."""
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    flat = column_major(matrix)
    assert flat[12:15] == [1.0, 2.0, 3.0]
    assert len(flat) == 16


def test_link_asset_exposes_the_file_without_copying_it(tmp_path: Path) -> None:
    source = tmp_path / "object.glb"
    source.write_bytes(b"payload")
    url = link_asset(source, tmp_path / "site" / "assets")
    link = tmp_path / "site" / "assets" / "object.glb"
    assert url == "assets/object.glb"
    assert link.is_symlink()
    assert link.read_bytes() == b"payload"


def test_link_asset_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "object.glb"
    source.write_bytes(b"payload")
    target = tmp_path / "assets"
    assert link_asset(source, target) == link_asset(source, target)


def write_glb(path: Path) -> None:
    """A real GLB: the manifest now loads colliders to decimate them."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    exported = mesh.export(file_type="glb")
    assert isinstance(exported, bytes)
    path.write_bytes(exported)


def build_hssd_stub(root: Path) -> None:
    """A minimal HSSD tree: one object with a collider, one door without."""
    (root / "objects" / "a").mkdir(parents=True)
    write_glb(root / "objects" / "a" / "abc.glb")
    write_glb(root / "objects" / "a" / "abc.collider.glb")
    (root / "objects" / "openings").mkdir()
    write_glb(root / "objects" / "openings" / "219-1.glb")
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "hssd_obj_semantics_condensed.csv").write_text(
        "hash,art,pick,condensed,primary,,\nabc,No,No,sofa,sofa,,\n"
    )


def instance(template: str) -> FurnitureInstance:
    return FurnitureInstance(
        template_name=template,
        translation=np.zeros(3),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.ones(3),
    )


def test_a_door_is_simulated_from_its_render_mesh_rather_than_dropped(tmp_path: Path) -> None:
    """Doors have no collider file, and are acoustically the opposite of negligible."""
    build_hssd_stub(tmp_path)
    entries, report = build_instances(
        tmp_path, [instance("abc"), instance("219-1")], tmp_path / "site"
    )
    assert report.placed == 2
    assert report.render_as_collider == ["219-1"]
    assert report.layouts == {"shard": 1, "openings": 1}
    door = entries[1]
    # The door is simulated from its render mesh, and the browser is pointed
    # at the decimated copy the simulator will actually receive.
    assert door.collider_is_render
    assert door.collider_url == "sim/219-1.glb"


def test_manifest_counts_an_unresolvable_template_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    build_hssd_stub(tmp_path)
    entries, report = build_instances(tmp_path, [instance("missing")], tmp_path / "site")
    assert entries == []
    assert report.unresolved == ["missing"]


def test_known_category_gets_its_table_material_not_a_random_one(tmp_path: Path) -> None:
    build_hssd_stub(tmp_path)
    entries, _ = build_instances(tmp_path, [instance("abc")], tmp_path / "site")
    # "sofa" is in SEMANTIC_MATERIAL_TABLE, so its absorption is deterministic
    # and repeated runs must agree.
    again, _ = build_instances(tmp_path, [instance("abc")], tmp_path / "site")
    assert entries[0].absorption == again[0].absorption
    assert entries[0].category == "sofa"


def test_manifest_entries_serialise_to_json(tmp_path: Path) -> None:
    """The manifest crosses into the browser, so it must be plain JSON."""
    build_hssd_stub(tmp_path)
    entries, _ = build_instances(tmp_path, [instance("abc")], tmp_path / "site")
    from dataclasses import asdict

    json.dumps([asdict(entry) for entry in entries])


def test_manifest_entries_carry_what_the_solver_was_given() -> None:
    """Absorption and area, with no "before" and "after" pair.

    The compensation provenance this used to assert is gone with the
    compensation: there is no rescaling left to be transparent about.
    """
    from dataclasses import fields

    from reverberate.viz.scene_manifest import InstanceEntry

    names = {field.name for field in fields(InstanceEntry)}

    assert {"absorption", "area"} <= names
    assert (
        not {
            "base_absorption",
            "original_area",
            "reduced_area",
            "compensation_factor",
            "capped",
            "compensation_colour",
        }
        & names
    )
