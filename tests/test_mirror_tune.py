"""The calibration run reads the card phase and writes its file; the subset travels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.stage import MirrorSettings, load_every, run_mirror
from reverberate.mirror.tune import (
    calibrate_run,
    choose_points,
    read_references,
    write_reference_subset,
)
from test_mirror_stage import a_run


def test_the_calibration_run_reads_the_card_phase_and_writes_its_file(tmp_path: Path) -> None:
    run, models = a_run(tmp_path)
    settings = MirrorSettings(
        ism=IsmSettings(max_order=2, flutter_order=2, furniture_bounces=2),
        rays=RaySettings(rays=60, duration_s=0.15, bin_s=0.002, receiver_radius_m=0.15),
        workers=2,
    )
    source = {"name": "S1", "position": [0.3, 0.5, 0.4]}
    run_mirror(
        run, models=models, source=source, settings=settings, phase="card", say=lambda _: None
    )
    every = load_every(run / "mirror" / "paths_S1.npz")
    chosen = choose_points(every, 2)
    assert len(chosen) == 2 and all(np.any(every[i].order == 0) for i in chosen)

    # The same references from the field and from the subset that travels.
    subset = write_reference_subset(
        run / "field" / "S1.h5", run / "mirror" / "reference_subset_S1.h5", chosen, 1
    )
    from_field = read_references(run, "S1", chosen, 1)
    (run / "field" / "S1.h5").rename(run / "field" / "away.h5")
    from_subset = read_references(run, "S1", chosen, 1)
    for i in chosen:
        np.testing.assert_allclose(from_subset[i].signals, from_field[i].signals, rtol=1e-6)
        assert from_subset[i].signals.shape[0] == 4
    (run / "field" / "away.h5").rename(run / "field" / "S1.h5")
    subset.unlink()

    best, evaluations, target = calibrate_run(
        run,
        source=source,
        settings=settings,
        points=2,
        iterations=3,
        tied=True,
        order=1,
        say=lambda _: None,
    )
    assert target.is_file() and target.parent.name == "calibration"
    latest = json.loads((run / "mirror" / "calibration" / "latest.json").read_text())
    assert latest["file"] == target.name and latest["points"] == chosen
    assert len(evaluations) >= 3 and "tied" in best.note
    record = json.loads(target.read_text())
    assert record["trajectory"][0]["points"] == 2
