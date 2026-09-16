"""The stage end to end on a box, with the twins: every file a campaign expects, written.

A solver model of a closed box, a mock reference field of a few points in
it, one source: the stage must derive the scene, write the mirror field in
the reference's format, the metrics with a judgement per point, the report,
and point ``walk.json`` at the mirror. Small and offline; the card is not
needed because every kernel has its twin.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.stage import MirrorSettings, load_every, run_mirror, write_every
from reverberate.viz.field_payload import check, mock_field
from test_accel_scene import write_scene


def a_run(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    models.mkdir()
    write_scene(models / "apartment_full.json")
    run = tmp_path / "run"
    (run / "field").mkdir(parents=True)
    reference = mock_field(
        run / "field" / "S1.h5",
        source_id="S1",
        source_position=[0.3, 0.5, 0.4],
        box_lo=[0.2, 0.0, 0.2],
        box_hi=[0.8, 1.0, 0.8],
        step_m=0.3,
        margin_m=0.0,
        heights_m=(0.5,),
        order=3,
        total_s=0.15,
    )
    # A reference with a direct sound at the path length, so the alignment
    # and the criteria have something to read; the mock's own content is noise.
    with __import__("h5py").File(reference, "a") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        direct = np.asarray(handle["direct_path_m"][...])
        for point in range(handle["ir"].shape[0]):
            block = np.asarray(handle["ir"][point]) * 1e-3
            at = int(round((direct[point] / 343.2 + 0.002) * rate))
            block[0, at] += 1.0
            block[3, at] += 1.0
            handle["ir"][point] = block
    (run / "walk.json").write_text(
        json.dumps(
            {
                "scene_id": "102344022",
                "sources": [
                    {"id": "S1", "name": "S1", "position": [0.3, 0.5, 0.4], "field": "field/S1.h5"}
                ],
            }
        )
    )
    return run, models


@pytest.mark.slow
def test_the_stage_writes_everything_and_its_two_phases_equal_it(tmp_path: Path) -> None:
    """One run in one place writes every file a campaign expects; then the card phase on a
    box with only the lattice, and the host phase at home, give the same field and summary."""
    run, models = a_run(tmp_path)
    settings = MirrorSettings(
        ism=IsmSettings(max_order=2, flutter_order=2, furniture_bounces=2),
        rays=RaySettings(rays=300, duration_s=0.15, bin_s=0.002, receiver_radius_m=0.15),
        workers=2,
    )
    source = {"name": "S1", "position": [0.3, 0.5, 0.4]}
    whole = run_mirror(run, models=models, source=source, settings=settings, say=lambda _: None)
    whole_field = (run / "field_mirror" / "S1.h5").read_bytes()
    assert whole["tree"]["images"] == 1 + 6 + 30
    assert (run / "mirror" / "scene.json").is_file()
    mirror = run / "field_mirror" / "S1.h5"
    assert check(mirror) == []
    metrics = json.loads((run / "mirror" / "metrics" / "S1.json").read_text())
    assert metrics["floor"]["recall_median"] == 0.88
    assert metrics["judged"] == len(metrics["points"]) > 0
    first = next(iter(metrics["points"].values()))
    assert "paths" in first and ("verdicts" in first or "silent" in first or "error" in first)
    manifest = json.loads((run / "walk.json").read_text())
    assert manifest["sources"][0]["field_mirror"] == "field_mirror/S1.h5"
    assert manifest["sources"][0]["metrics"] == "mirror/metrics/S1.json"
    assert manifest["mirror"]["key"] == whole["scene"]["key"]
    assert (run / "mirror" / "report_S1.json").is_file()
    assert whole["alignment"]["points_used"] > 0
    assert whole["paths"]["median"] >= 7

    # The card, somewhere without the field: only the lattice travels.
    box = tmp_path / "box"
    (box / "mirror").mkdir(parents=True)
    with h5py.File(run / "field" / "S1.h5", "r") as handle:
        np.savez(
            box / "mirror" / "lattice_S1.npz",
            positions=handle["positions"][...],
            sample_rate_hz=handle.attrs["sample_rate_hz"],
            order=handle.attrs["order"],
        )
    card = run_mirror(
        box, models=models, source=source, settings=settings, phase="card", say=lambda _: None
    )
    assert card["phase"] == "card" and "alignment" not in card
    assert (box / "mirror" / "card_S1.json").is_file()
    assert not (box / "field_mirror").exists()
    every = load_every(box / "mirror" / "paths_S1.npz")
    assert [p.count for p in every] == [
        p.count for p in load_every(run / "mirror" / "paths_S1.npz")
    ]
    again = tmp_path / "again.npz"
    write_every(every, again)
    np.testing.assert_array_equal(load_every(again)[0].points, every[0].points)

    # Home: the card's files beside the field, then the host phase.
    for name in ("card_S1.json", "paths_S1.npz", "histogram_S1.npz", "scene.npz", "scene.json"):
        shutil.copy(box / "mirror" / name, run / "mirror" / name)
    (run / "field_mirror" / "S1.h5").unlink()
    host = run_mirror(
        run, models=models, source=source, settings=settings, phase="host", say=lambda _: None
    )
    assert host["phase"] == "host"
    assert host["tree"] == whole["tree"] and host["alignment"] == whole["alignment"]
    assert host["summary"] == whole["summary"]
    assert (run / "field_mirror" / "S1.h5").read_bytes() == whole_field
