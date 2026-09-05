"""Tests for publishing a voxelisation as a run page with no solve behind it.

The risk this module carries is not that it fails, it is that it succeeds and
the page then reads as a measurement. So the tests are about what it must
*refuse* to invent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from reverberate.experiments import grid_page


class _Entry:
    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self.path = path
        self.manifest = manifest


class _Geometry:
    volume_m3 = 38.1
    surface_area_m2 = 125.1
    mean_absorption = 0.29

    def record(self) -> dict[str, Any]:
        return {"volume_m3": 38.1, "surface_area_m2": 125.1, "mean_absorption": 0.29}


class _Theory:
    def record(self) -> dict[str, Any]:
        return {"sabine_rt60_s": 0.5, "eyring_rt60_s": 0.4}


@pytest.fixture
def models(tmp_path: Path) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "bedroom_only.json").write_text(json.dumps({"mats_hash": {}}))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": "102344022",
                "room": "bedroom.001",
                "sealed": {"sealed_volume_m3": 3.3},
                "sealed_full": {"sealed_volume_m3": 41.5},
                "scenes": [{"name": "bedroom_only", "file": "bedroom_only.json"}],
            }
        )
    )
    return directory


def _build(models: Path, out: Path, scene: str = "bedroom_only") -> dict[str, Any]:
    entry = _Entry(out / "cache", {"fmax": 4000.0, "h_m": 0.00817, "boundary_nodes": 3861276})
    with (
        mock.patch.object(grid_page, "entry_from_key", lambda key: entry),
        mock.patch.object(grid_page, "_sound_speed", lambda path: 343.2),
        mock.patch.object(grid_page, "room_geometry", lambda *a, **k: _Geometry()),
        mock.patch.object(grid_page, "theory", lambda *a: _Theory()),
    ):
        return grid_page.build(models, scene, "deadbeef", out)


class TestBuild:
    def test_it_writes_the_two_files_the_viewer_looks_for(
        self, models: Path, tmp_path: Path
    ) -> None:
        """``discover_runs`` needs both, and skips a directory with only one."""
        out = tmp_path / "page"
        _build(models, out)
        assert (out / "plan.json").is_file()
        assert (out / "report.json").is_file()
        plan = json.loads((out / "plan.json").read_text())
        assert plan["scene_id"] == "102344022"
        assert plan["room"] == "bedroom.001"

    def test_it_places_no_source_and_no_receiver(self, models: Path, tmp_path: Path) -> None:
        """The page draws a marker per source and per receiver.

        A plausible-looking pair nobody placed would read as the geometry
        having been measured somewhere, which is the one claim this page must
        not make.
        """
        out = tmp_path / "page"
        report = _build(models, out)
        assert report["placement"] == {"sources": [], "receivers": []}
        assert report["sources"] == []
        assert report["dry_voice"] is None

    def test_the_absence_of_a_solve_is_on_the_page(self, models: Path, tmp_path: Path) -> None:
        """An empty responses panel is ambiguous; ``omissions`` is not."""
        report = _build(models, tmp_path / "page")
        assert report["omissions"] == [grid_page.NO_SOLVE]
        assert "no solve" in grid_page.NO_SOLVE.lower()

    def test_the_grid_is_described_from_the_entry_and_not_the_arguments(
        self, models: Path, tmp_path: Path
    ) -> None:
        """A page cannot claim a grid it was not built from."""
        report = _build(models, tmp_path / "page")
        assert report["grid"]["boundary_nodes"] == 3861276
        assert report["grid"]["fmax"] == 4000.0

    def test_a_room_scene_takes_the_room_census_and_not_the_storey_one(
        self, models: Path, tmp_path: Path
    ) -> None:
        """The bedroom seals 3.3 m3 and the whole storey 41.5.

        Reading the wrong one would put the flat's figure on the room's page,
        where it is off by an order of magnitude and looks merely surprising.
        """
        report = _build(models, tmp_path / "page")
        assert report["sealed"]["sealed_volume_m3"] == 3.3
