"""Tests for what the walk-through app reads from a run.

The app must never open a run that was made for the old viewer, a run's
payloads must be linked into the site as the page expects them, and what the
app would refuse must be named before the first launch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverberate.viz.app_payload import (
    WALK_MANIFEST,
    WalkRun,
    build_run,
    check_run,
    discover_walk_runs,
    write_synthetic_run,
)


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


def test_a_run_builds_into_a_site_with_its_mesh_and_fields_linked(tmp_path: Path) -> None:
    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    site = tmp_path / "site" / "runs" / "run"
    record = build_run(run, site)

    assert json.loads((site / "run.json").read_text()) == record
    assert [s["id"] for s in record["sources"]] == ["S1", "S2"]
    field = record["sources"][0]["field"]
    assert field["url"] == "fields/S1" and field["order"] == 7
    assert (site / "fields" / "S1" / "index.json").is_file()
    linked = (site / "fields" / "S1" / "field.h5").resolve()
    assert linked == (run.path / "fields" / "S1.h5").resolve()
    mesh = record["meshes"]["4000"]
    assert mesh["url"] == "meshes/4000"
    assert mesh["rooms"][0]["regions"] == ["living room", "hallway", "dining room", "kitchen"]
    assert (site / "meshes" / "4000").is_symlink()
    assert (site / "meshes" / "4000" / "rooms.json").is_file()


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


def test_check_run_names_what_the_app_would_refuse(tmp_path: Path) -> None:
    """A missing field on one source must not hide the checks on the next."""
    run = write_synthetic_run(tmp_path / "run", field_step_m=1.1, field_box_m=2.2)
    assert check_run(run.path) == []
    (run.path / "fields" / "S1.h5").unlink()
    with (run.path / WALK_MANIFEST).open() as handle:
        manifest = json.load(handle)
    manifest["sources"][1]["id"] = "S9"
    (run.path / WALK_MANIFEST).write_text(json.dumps(manifest))

    problems = check_run(run.path)

    assert any("S1" in p and "absent" in p for p in problems)
    assert any("S9" in p and "source_id 'S2'" in p for p in problems)
    assert check_run(tmp_path / "nowhere") == [f"no {WALK_MANIFEST} in {tmp_path / 'nowhere'}"]


def test_a_mesh_of_another_flat_is_refused(tmp_path: Path) -> None:
    """A run's walk.json copied from another flat keeps that flat's grids; drawn,
    they would put the wrong walls around this run's sources."""
    run = write_synthetic_run(tmp_path / "run", fields=False)
    rooms = tmp_path / "run" / "audit_4k" / "voxels" / "rooms.json"
    index = json.loads(rooms.read_text())
    index["scene_id"] = "999"
    rooms.write_text(json.dumps(index))

    assert any("scene 999" in problem for problem in check_run(run.path))
    assert build_run(run, tmp_path / "site")["meshes"] == {}


def manifest(tmp_path: Path, **extra: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / WALK_MANIFEST).write_text(
        json.dumps(
            {"scene_id": "102344403", "sources": [{"id": "S1", "position": [0, 1, 0]}], **extra}
        )
    )
    return tmp_path


def test_a_run_says_what_made_it_and_defaults_to_the_wave_solver(tmp_path: Path) -> None:
    plain = WalkRun.read(manifest(tmp_path / "a"))
    assert plain.simulation == "wave" and plain.crossover_hz is None and plain.mirror == {}
    hybrid = WalkRun.read(
        manifest(
            tmp_path / "b",
            simulation="hybrid",
            crossover_hz=1000,
            mirror_solver={"scene": "mirror/scene", "paths": {"S1": "mirror/paths_S1.npz"}},
        )
    )
    assert hybrid.simulation == "hybrid" and hybrid.crossover_hz == 1000
    assert hybrid.mirror["paths"] == {"S1": "mirror/paths_S1.npz"}


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"simulation": "rays"}, "simulation"),
        ({"simulation": "hybrid"}, "crossover_hz"),
        ({"mirror_solver": {"paths": {}}}, "mirror_solver"),
    ],
)
def test_a_manifest_that_does_not_say_it_right_is_refused_and_skipped(
    tmp_path: Path, extra: dict[str, object], message: str
) -> None:
    run = manifest(tmp_path / "bad", **extra)
    with pytest.raises(ValueError, match=message):
        WalkRun.read(run)
    manifest(tmp_path / "good")
    assert [r.name for r in discover_walk_runs(tmp_path)] == ["good"]


def test_a_run_with_a_mirror_carries_its_audit_into_the_site(tmp_path: Path) -> None:
    from reverberate.mirror.files import write_paths
    from reverberate.mirror.geometry import write_derived
    from reverberate.mirror.ism import grow_tree, paths_for
    from test_mirror_ism import RECEIVER, SOURCE, box_scene

    run_dir = write_synthetic_run(tmp_path / "run", fields=False).path
    scene = box_scene(alpha=0.3)
    write_derived(scene, run_dir / "mirror" / "scene")
    write_paths(
        [paths_for(scene, grow_tree(scene, SOURCE), RECEIVER)], run_dir / "mirror" / "paths_S1.npz"
    )
    walk = json.loads((run_dir / WALK_MANIFEST).read_text())
    walk.update(
        simulation="hybrid",
        crossover_hz=1000,
        mirror_solver={"scene": "mirror/scene", "paths": {"S1": "mirror/paths_S1.npz"}},
    )
    (run_dir / WALK_MANIFEST).write_text(json.dumps(walk))
    record = build_run(WalkRun.read(run_dir), tmp_path / "site")
    assert record["simulation"] == "hybrid" and record["crossover_hz"] == 1000
    assert record["mirror"]["key"] == scene.key
    assert (tmp_path / "site" / record["mirror"]["paths"]["S1"]).is_file()
