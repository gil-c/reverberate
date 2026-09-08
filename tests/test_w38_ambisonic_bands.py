"""Three W10 shaped band runs on three grids assemble into one run the viewer can open."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.experiments.w10_ambisonic import COMMS_NAME
from reverberate.experiments.w38_ambisonic_bands import BANDS, TRICK, assemble_run, outer_radius_for
from reverberate.spatial.array import design_array
from reverberate.spatial.encode import EncoderSettings
from reverberate.spatial.sh import scene_to_ambisonic
from reverberate.viz.run_view import SPATIAL_RUN_REPORT_KEYS, report_is_drawable
from reverberate.wave.comms import Grid

RATE = 48000.0
SOUND_SPEED = 343.2
#: The three bands' grid steps at 10.5 points per wavelength, and their fmax.
STEPS = {"low": 0.0327, "mid": 0.008171, "high": 0.002043}
FMAX = {"low": 1000.0, "mid": 4000.0, "high": 16000.0}
#: What each band was solved for: the low band the whole decay, the others less.
SECONDS = {"low": 0.12, "mid": 0.06, "high": 0.03}


def _grid(step: float) -> Grid:
    nodes = int(np.ceil(1.2 / step)) + 1
    axis = np.arange(nodes) * step
    return Grid(
        h=step,
        Ts=step / SOUND_SPEED / np.sqrt(3.0),
        l2=1 / 3,
        fcc_flag=0,
        xv=axis,
        yv=axis,
        zv=axis,
    )


def _band_run(out: Path, name: str, centre: np.ndarray, source: np.ndarray) -> dict:
    """One W10 room run on this band's own grid, holding a plane wave from the source."""
    step, fmax, seconds = STEPS[name], FMAX[name], SECONDS[name]
    design = design_array(
        centre, _grid(step), fit_order=6, outer_radius_m=outer_radius_for(step, 0.16)
    )
    run = out / name
    (run / "comms").mkdir(parents=True)
    (run / "source0").mkdir(parents=True)
    np.save(run / "array_positions.npy", design.positions)
    samples = int(seconds * RATE)
    plan = {
        "run": name,
        "scene_id": "test",
        "room": "box",
        "cache_key": "0" * 32,
        "geometry_sha256": "1" * 64,
        "fmax_hz": fmax,
        "grid_step_m": step,
        "sample_rate_hz": RATE,
        "sound_speed_m_s": SOUND_SPEED,
        "duration_s": seconds,
        "samples": samples,
        "source": [float(v) for v in source],
        "centre": [float(v) for v in centre],
        "extra_receivers": [],
        "clearance": {"boundary_nodes_in_ball": 0},
        "cost": {"grid_points": 1, "receivers": design.count},
        "array": design.record(),
        "encoder": EncoderSettings(max_frequency_hz=fmax).record(),
        "conditioning": {"effective_order": [7]},
    }
    (run / "plan.json").write_text(json.dumps(plan))

    unit = scene_to_ambisonic((source - centre)[None, :])[0]
    unit = unit / np.linalg.norm(unit)
    along = scene_to_ambisonic(design.offsets) @ unit
    rng = np.random.default_rng(hash(name) % 1000)
    time = np.arange(samples) / RATE
    frequency = np.fft.rfftfreq(samples, 1.0 / RATE)
    # A band-limited pulse at unit peak carries a spectral density that goes as
    # 1 / fmax, which is the level ratio the assembly has to measure and undo.
    pulse = np.fft.rfft(np.r_[1.0, np.zeros(samples - 1)]) * (frequency <= fmax)
    pressure = np.zeros((design.count, samples))
    for index in range(design.count):
        delay = 0.004 - along[index] / SOUND_SPEED
        wave = np.fft.irfft(pulse * np.exp(-2j * np.pi * frequency * delay), n=samples)
        pressure[index] = wave / (fmax / 16000.0)
        pressure[index] += 0.02 * rng.standard_normal(samples) * np.exp(-time / 0.03)
    with h5py.File(run / "source0" / "sim_outs.h5", "w") as handle:
        handle.create_dataset("u_out", data=pressure)
    with h5py.File(run / "comms" / COMMS_NAME, "w") as handle:
        handle.create_dataset("out_alpha", data=np.ones((design.count, 1)))
        handle.create_dataset("diff", data=np.int8(0))
    with h5py.File(run / "source0" / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(step))
        handle.create_dataset("SR", data=np.float64(RATE))
    return plan


def a_three_band_run(tmp_path: Path) -> Path:
    out = tmp_path / "w38_test"
    centre = np.full(3, 0.6)
    source = centre + np.array([1.5, 0.0, 0.0])
    plans = {name: _band_run(out, name, centre, source) for name in BANDS}
    plan = {
        "run": out.name,
        "scene_id": "test",
        "room": "box",
        "kind": "three-band ambisonic",
        "trick": TRICK,
        "centre": centre.tolist(),
        "source": source.tolist(),
        "sound_speed_m_s": SOUND_SPEED,
        "bands": {
            name: {
                "cache_key": "0" * 32,
                "fmax_hz": FMAX[name],
                "grid_step_m": STEPS[name],
                "duration_s": SECONDS[name],
            }
            for name in BANDS
        },
        "cache_key": "0" * 32,
        "model_json": None,
        "array": plans["high"]["array"],
        "conditioning": plans["high"]["conditioning"],
    }
    (out / "plan.json").write_text(json.dumps(plan))
    return out


def test_three_grids_assemble_into_one_drawable_run_that_points_at_the_source(
    tmp_path: Path,
) -> None:
    out = a_three_band_run(tmp_path)
    rendered: dict = {}
    report = assemble_run(
        out, air=None, order=3, fit_order=5, lowcut_hz=50.0, plain_decode=True, rendered=rendered
    )

    assert report["run"] == "w38_test"
    assert report["kind"] == "three-band ambisonic"
    assert all(key in report for key in SPATIAL_RUN_REPORT_KEYS)
    assert report_is_drawable(report)
    json.dumps(report)

    # Every band was levelled by the ratio its bandwidth predicts.
    gains = {row["band"]: row["gain_db"] for row in report["assembly"]["bands"]}
    assert gains["high"] == pytest.approx(0.0)
    assert gains["mid"] == pytest.approx(-12.04, abs=1.5)
    assert gains["low"] == pytest.approx(-24.08, abs=2.0)

    # The two short bands were continued to the low band's decay.
    assert [row["band"] for row in report["tails"]] == ["mid", "high"]
    assert rendered["ambisonic"].duration_s == pytest.approx(SECONDS["low"], abs=1e-3)

    # Each band expanded about its own node, and the report says how far off it sat.
    offsets = {row["band"]: row for row in report["centre_offsets"]}
    for name in BANDS:
        assert offsets[name]["offset_m"] <= offsets[name]["bound_m"]

    # And the assembled field still points at the source.
    usable = [row for row in report["direction_of_arrival"] if row["usable"]]
    assert usable, "no band usable for direction of arrival"
    assert max(row["error_deg"] for row in usable) < 15.0


def test_a_run_missing_one_band_is_refused(tmp_path: Path) -> None:
    out = a_three_band_run(tmp_path)
    plan = json.loads((out / "plan.json").read_text())
    del plan["bands"]["mid"]
    (out / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(KeyError):
        assemble_run(out, air=None, order=3, fit_order=5, lowcut_hz=50.0, plain_decode=True)


def test_the_array_grows_with_the_cell_so_it_exists_on_every_grid() -> None:
    assert outer_radius_for(0.002043, 0.16) == 0.16
    assert outer_radius_for(0.008171, 0.16) == 0.16
    assert outer_radius_for(0.0327, 0.16) == pytest.approx(0.3924)
