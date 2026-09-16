"""The stage end to end on a box, with the twins: every file a campaign expects, written.

A solver model of a closed box, a mock reference field of a few points in
it, one source: the stage must derive the scene, write the mirror field in
the reference's format, the metrics with a judgement per point, the report,
and point ``walk.json`` at the mirror. Small and offline; the card is not
needed because every kernel has its twin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.stage import MirrorSettings, run_mirror
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


def test_the_stage_writes_the_mirror_field_the_metrics_and_the_report(tmp_path: Path) -> None:
    run, models = a_run(tmp_path)
    settings = MirrorSettings(
        # The box's one label is not the shell, so its facets count as furniture.
        ism=IsmSettings(max_order=2, flutter_order=2, furniture_bounces=2),
        rays=RaySettings(rays=300, duration_s=0.15, bin_s=0.002, receiver_radius_m=0.15),
        workers=2,
    )
    report = run_mirror(
        run,
        models=models,
        source={"name": "S1", "position": [0.3, 0.5, 0.4]},
        settings=settings,
        say=lambda _: None,
    )
    assert report["tree"]["images"] == 1 + 6 + 30
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
    assert manifest["mirror"]["key"] == report["scene"]["key"]
    assert (run / "mirror" / "report_S1.json").is_file()
    assert report["alignment"]["points_used"] > 0
    assert report["paths"]["median"] >= 7
