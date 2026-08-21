"""Property and unit tests for HSSD room reconstruction.

Everything here uses synthetic JSON and geometry, never ``data/``: per
ROADMAP.md's hard constraints, the test suite runs offline and stays fast.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from reverberate.geometry.hssd_room import (
    ObjectInstance,
    RegionAnnotation,
    assign_objects_to_region,
    build_room,
    extrude_region_shell,
    load_object_instances,
    load_region_annotations,
    load_semantic_lexicon,
    object_semantic_category,
    quaternion_translation_scale_to_matrix,
)


def _square_region(name: str = "room", side: float = 4.0) -> RegionAnnotation:
    half = side / 2.0
    return RegionAnnotation(
        name=name,
        label=name,
        poly_loop_xz=[(-half, -half), (half, -half), (half, half), (-half, half)],
        floor_height=0.0,
        extrusion_height=2.5,
    )


def test_extrude_region_shell_is_watertight_with_plausible_volume() -> None:
    region = _square_region(side=4.0)
    shell = extrude_region_shell(region)
    assert shell.is_watertight
    assert shell.volume == pytest.approx(4.0 * 4.0 * 2.5, rel=1e-6)


def test_extrude_region_shell_uses_floor_height_as_the_base() -> None:
    region = RegionAnnotation(
        name="upstairs",
        label="upstairs",
        poly_loop_xz=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        floor_height=3.0,
        extrusion_height=2.5,
    )
    shell = extrude_region_shell(region)
    assert shell.bounds[0][1] == pytest.approx(3.0)
    assert shell.bounds[1][1] == pytest.approx(5.5)


def test_assign_objects_to_region_keeps_only_points_inside_polygon() -> None:
    region = _square_region(side=4.0)
    inside = ObjectInstance(
        template_name="inside",
        translation=np.array([0.5, 0.0, 0.5]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.array([1.0, 1.0, 1.0]),
    )
    outside = ObjectInstance(
        template_name="outside",
        translation=np.array([10.0, 0.0, 10.0]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        non_uniform_scale=np.array([1.0, 1.0, 1.0]),
    )
    assigned = assign_objects_to_region(region, [inside, outside])
    assert [i.template_name for i in assigned] == ["inside"]


def test_quaternion_translation_scale_to_matrix_identity() -> None:
    matrix = quaternion_translation_scale_to_matrix(
        rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        translation=[1.0, 2.0, 3.0],
        non_uniform_scale=[1.0, 1.0, 1.0],
    )
    point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
    assert point[:3] == pytest.approx([2.0, 2.0, 3.0])


def test_quaternion_translation_scale_to_matrix_applies_scale_then_rotation() -> None:
    # 90 degree rotation about Y: w=cos(45deg), y=sin(45deg).
    half_angle = math.radians(45.0)
    matrix = quaternion_translation_scale_to_matrix(
        rotation_wxyz=[math.cos(half_angle), 0.0, math.sin(half_angle), 0.0],
        translation=[0.0, 0.0, 0.0],
        non_uniform_scale=[2.0, 1.0, 1.0],
    )
    # A point on the scaled local +X axis (2, 0, 0) rotates to local +Z... a
    # 90 degree rotation about Y sends +X to -Z in a right handed frame.
    point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
    assert point[:3] == pytest.approx([0.0, 0.0, -2.0], abs=1e-9)


def test_object_semantic_category_strips_articulated_part_suffix(tmp_path: Path) -> None:
    hssd_root = tmp_path
    (hssd_root / "objects" / "a").mkdir(parents=True)
    (hssd_root / "objects" / "a" / "abc123.object_config.json").write_text(
        json.dumps({"semantic_id": 7})
    )
    lexicon = {7: "chair"}
    assert object_semantic_category(hssd_root, "abc123_part_2", lexicon) == "chair"
    assert object_semantic_category(hssd_root, "abc123", lexicon) == "chair"


def test_object_semantic_category_unknown_when_config_missing(tmp_path: Path) -> None:
    lexicon = {7: "chair"}
    assert object_semantic_category(tmp_path, "doesnotexist", lexicon) == "unknown"


def test_load_semantic_lexicon(tmp_path: Path) -> None:
    semantics_dir = tmp_path / "semantics"
    semantics_dir.mkdir()
    (semantics_dir / "hssd-hab_semantic_lexicon.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "unknown"}, {"id": 5, "name": "sofa"}]})
    )
    lexicon = load_semantic_lexicon(tmp_path)
    assert lexicon == {0: "unknown", 5: "sofa"}


def _write_scene(hssd_root: Path, scene_id: str) -> None:
    (hssd_root / "semantics" / "scenes").mkdir(parents=True, exist_ok=True)
    (hssd_root / "scenes").mkdir(parents=True, exist_ok=True)
    (hssd_root / "semantics" / "hssd-hab_semantic_lexicon.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "unknown"}]})
    )
    (hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json").write_text(
        json.dumps(
            {
                "region_annotations": [
                    {
                        "name": "bedroom",
                        "label": "bedroom",
                        "poly_loop": [
                            [-2.0, 0.0, -2.0],
                            [2.0, 0.0, -2.0],
                            [2.0, 0.0, 2.0],
                            [-2.0, 0.0, 2.0],
                        ],
                        "floor_height": 0.0,
                        "extrusion_height": 2.5,
                    }
                ]
            }
        )
    )
    (hssd_root / "scenes" / f"{scene_id}.scene_instance.json").write_text(
        json.dumps({"stage_instance": {}, "object_instances": []})
    )


def test_load_region_annotations_round_trips_poly_loop(tmp_path: Path) -> None:
    _write_scene(tmp_path, "s1")
    regions = load_region_annotations(tmp_path, "s1")
    assert len(regions) == 1
    assert regions[0].name == "bedroom"
    assert regions[0].poly_loop_xz[0] == (-2.0, -2.0)


def test_load_object_instances_empty(tmp_path: Path) -> None:
    _write_scene(tmp_path, "s1")
    assert load_object_instances(tmp_path, "s1") == []


def test_build_room_with_no_furniture_still_produces_watertight_shell(tmp_path: Path) -> None:
    _write_scene(tmp_path, "s1")
    room = build_room(tmp_path, "s1", "bedroom")
    assert room.shell.is_watertight
    assert room.furniture == []
    assert room.skipped_instances == []


def test_build_room_raises_on_unknown_region(tmp_path: Path) -> None:
    _write_scene(tmp_path, "s1")
    with pytest.raises(KeyError):
        build_room(tmp_path, "s1", "does-not-exist")
