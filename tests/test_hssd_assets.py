"""Tests for HSSD asset resolution across the dataset's three object layouts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from reverberate.geometry.hssd_assets import (
    candidate_directories,
    category_for_template,
    resolve_asset,
    semantic_categories,
)


@pytest.fixture
def objects_dir(tmp_path: Path) -> Path:
    """A miniature objects tree with all three layouts HSSD actually uses."""
    root = tmp_path / "objects"
    (root / "a").mkdir(parents=True)
    (root / "a" / "abc123.glb").write_bytes(b"render")
    (root / "a" / "abc123.collider.glb").write_bytes(b"collider")

    (root / "openings").mkdir()
    (root / "openings" / "219-1.glb").write_bytes(b"door")

    (root / "decomposed" / "deadbeef").mkdir(parents=True)
    (root / "decomposed" / "deadbeef" / "deadbeef_part_7.glb").write_bytes(b"part")
    (root / "decomposed" / "deadbeef" / "deadbeef_part_7.collider.glb").write_bytes(b"c")
    return root


def test_sharded_object_is_found_with_its_collider(objects_dir: Path) -> None:
    asset = resolve_asset(objects_dir, "abc123")
    assert asset is not None
    assert asset.layout == "shard"
    assert asset.collider.name == "abc123.collider.glb"
    assert not asset.collider_is_render


def test_door_in_openings_is_found_despite_not_being_a_hash(objects_dir: Path) -> None:
    """`219-1` has no hash shard, so the shard rule alone loses every door."""
    asset = resolve_asset(objects_dir, "219-1")
    assert asset is not None
    assert asset.layout == "openings"


def test_render_stands_in_as_collider(objects_dir: Path) -> None:
    asset = resolve_asset(objects_dir, "219-1")
    assert asset is not None
    assert asset.collider == asset.render
    assert asset.collider_is_render


def test_articulated_part_is_found_under_its_base_hash(objects_dir: Path) -> None:
    asset = resolve_asset(objects_dir, "deadbeef_part_7")
    assert asset is not None
    assert asset.layout == "decomposed"


def test_unknown_template_resolves_to_none_rather_than_raising(objects_dir: Path) -> None:
    assert resolve_asset(objects_dir, "nothing_here") is None


def test_decomposed_directory_is_tried_first_for_a_part(objects_dir: Path) -> None:
    layouts = [layout for _, layout in candidate_directories(objects_dir, "deadbeef_part_7")]
    assert layouts[0] == "decomposed"


def write_semantics_csv(path: Path, rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Object Hash", "Articulated", "Pickable", "Condensed", "Primary", "", ""])
        writer.writerows(rows)
    return path


def test_condensed_category_wins_over_primary(tmp_path: Path) -> None:
    csv_path = write_semantics_csv(
        tmp_path / "metadata" / "hssd_obj_semantics_condensed.csv",
        [["hash1", "No", "Yes", "sofa", "sectional sofa", "", ""]],
    )
    assert semantic_categories(csv_path)["hash1"] == "sofa"


def test_primary_category_used_when_condensed_is_blank(tmp_path: Path) -> None:
    csv_path = write_semantics_csv(
        tmp_path / "metadata" / "hssd_obj_semantics_condensed.csv",
        [["hash2", "No", "Yes", "", "footstool", "", ""]],
    )
    assert semantic_categories(csv_path)["hash2"] == "footstool"


def test_articulated_part_inherits_its_base_objects_category(tmp_path: Path) -> None:
    """The metadata table is keyed by base hash, so parts must fall back to it."""
    write_semantics_csv(
        tmp_path / "metadata" / "hssd_obj_semantics_condensed.csv",
        [["deadbeef", "Yes", "No", "cabinet", "cabinet", "", ""]],
    )
    assert category_for_template(tmp_path, "deadbeef_part_7") == "cabinet"
    assert category_for_template(tmp_path, "unknown_hash") is None
