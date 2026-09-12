"""Tests for what the walk-through app reads from a run.

The app must never open a run that was made for the old viewer, and the
synthetic run it is built against must have every shape the real one will.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate.viz.app_payload import (
    WALK_MANIFEST,
    WalkRun,
    build_run,
    discover_walk_runs,
    write_synthetic_run,
)
from reverberate.viz.field_payload import read_header


def test_a_run_without_the_walk_manifest_is_not_a_run_for_this_app(tmp_path: Path) -> None:
    """Every earlier run has a report.json and audio; none has walk.json.

    They stay on disk and out of the app, which is the owner's rule: nothing
    computed before the field exists is to be heard or drawn.
    """
    old = tmp_path / "w39_living_8k"
    old.mkdir()
    (old / "report.json").write_text("{}")
    (old / "audio").mkdir()
    write_synthetic_run(tmp_path / "synthetic", fields=False)

    assert [run.name for run in discover_walk_runs(tmp_path)] == ["synthetic"]


def test_the_synthetic_run_builds_into_a_site_with_its_mesh_linked(tmp_path: Path) -> None:
    run = write_synthetic_run(tmp_path / "run", centre=(1.0, 0.5, -2.0), fields=False)
    record = build_run(run, tmp_path / "site" / "runs" / "run")

    written = json.loads((tmp_path / "site" / "runs" / "run" / "run.json").read_text())
    assert written == record
    assert [s["id"] for s in record["sources"]] == ["S1", "S2", "S3", "S4", "S5"]
    mesh = record["meshes"]["4000"]
    assert [room["name"] for room in mesh["rooms"]] == ["living room-hallway-dining room-kitchen"]
    assert mesh["rooms"][0]["regions"] == ["living room", "hallway", "dining room", "kitchen"]
    assert mesh["url"] == "meshes/4000"
    link = tmp_path / "site" / "runs" / "run" / "meshes" / "4000"
    assert link.is_symlink()
    assert (link / "rooms.json").is_file()


def test_each_source_has_a_field_box_around_it_at_one_listening_height(tmp_path: Path) -> None:
    """The producer records one height, 1.70 m, on a 0.55 m lattice; the mock
    does the same around each of its five sources."""
    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    for source in run.sources:
        header = read_header(tmp_path / "run" / source["field"])
        lo = np.asarray(header.grid_origin_m)
        assert header.grid_shape[1] == 1 and header.grid_step_m[1] == 0.0
        assert lo[1] == pytest.approx(1.70)
        assert header.source_id == source["id"]
        assert header.samples == 57600
        assert abs(lo[0] - source["position"][0]) <= 1.1 + 1e-6
    rooms = json.loads((tmp_path / "run" / "audit_4k" / "voxels" / "rooms.json").read_text())
    room = rooms["rooms"][0]
    assert room["coarse"]["cell_m"] > room["fine"]["cell_m"]


def test_a_source_with_an_unknown_directivity_is_refused_by_name(tmp_path: Path) -> None:
    (tmp_path / WALK_MANIFEST).write_text(
        json.dumps(
            {
                "scene_id": "102344403",
                "sources": [{"id": "S1", "position": [0, 1, 0], "directivity": "laser"}],
            }
        )
    )
    with pytest.raises(ValueError, match="laser"):
        WalkRun.read(tmp_path)


def test_a_run_may_name_its_dwelling_and_gets_the_hssd_id_back(tmp_path: Path) -> None:
    """Runs are written under the project's names; files are stored under HSSD's."""
    (tmp_path / WALK_MANIFEST).write_text(json.dumps({"dwelling": "hssd_0002", "sources": []}))

    run = WalkRun.read(tmp_path)

    assert run.scene_id == "102344022"
    assert run.dwelling == "hssd_0002"
    assert write_synthetic_run(tmp_path / "s", fields=False).dwelling == "hssd_0002"


def test_a_source_field_is_unpacked_beside_the_run_and_linked(tmp_path: Path) -> None:
    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    record = build_run(run, tmp_path / "site" / "runs" / "run")

    field = record["sources"][0]["field"]
    assert field["url"] == "fields/S1" and field["order"] == 7
    site = tmp_path / "site" / "runs" / "run" / "fields" / "S1"
    assert (site / "index.json").is_file()
    assert (site / "field.h5").is_symlink()
    assert (site / "field.h5").resolve() == (tmp_path / "run" / "fields" / "S1.h5").resolve()


def test_check_run_names_what_the_app_would_refuse(tmp_path: Path) -> None:
    from reverberate.viz.app_payload import check_run

    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    assert check_run(run.path) == []
    (run.path / "fields" / "S2.h5").unlink()
    problems = check_run(run.path)
    assert any("S2" in problem and "absent" in problem for problem in problems)
    assert check_run(tmp_path / "nowhere") == [f"no {WALK_MANIFEST} in {tmp_path / 'nowhere'}"]
