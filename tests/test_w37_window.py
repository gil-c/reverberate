"""Tests for the window sweep.

The experiment's claim is that a shorter computed window plus a synthetic tail
costs less and stays accurate, so the tests check the two halves separately:
that the price really is linear in the window, and that the accuracy is
reported with the spread that says whether it is a measurement at all.

They also check the two refusals that keep the comparison honest. Calibrating
one room's tail from another room's decay, or comparing two runs delivered at
different rates, both produce plausible numbers and mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.experiments.w37_window import build, mean_absorption_of
from reverberate.response import Provenance, ResponseSet, write_raw

SOUND_SPEED = 343.2
SAMPLE_RATE = 48000.0
SCENE = "f" * 64


def _report(bands: int = 7) -> dict[str, Any]:
    """A room of two materials, one absorbing and one not."""
    return {
        "cost": {"grid_points": 4_607_993_520, "steps": 291_276},
        "cache_key": "0" * 32,
        "room": {
            "per_class": [
                {
                    "label": "plasterboard",
                    "area_m2": 100.0,
                    "random_incidence_absorption": [0.10] * bands,
                },
                {
                    "label": "carpet_thick",
                    "area_m2": 25.0,
                    "random_incidence_absorption": [0.50] * bands,
                },
            ]
        },
    }


def _write_run(
    root: Path,
    name: str,
    *,
    fmax_hz: float,
    seconds: float,
    scene: str = SCENE,
    sample_rate: float = SAMPLE_RATE,
    t60_s: float = 0.30,
) -> Path:
    run_dir = root / name
    (run_dir / "responses").mkdir(parents=True)
    rng = np.random.default_rng(20250101)
    samples = int(seconds * sample_rate)
    times = np.arange(samples) / sample_rate
    ir = rng.standard_normal((2, samples)) * 10.0 ** (-60.0 * times / (20.0 * t60_s))
    write_raw(
        ResponseSet(
            ir=ir,
            sample_rate_hz=sample_rate,
            source_position=np.zeros(3),
            receiver_positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            provenance=Provenance(
                scene_sha256=scene,
                mats_hash="0" * 32,
                engine="cpu",
                band="wave",
                fmax_hz=fmax_hz,
                grid_step_m=343.2 / (fmax_hz * 10.5),
                points_per_wavelength=10.5,
                sound_speed_m_s=SOUND_SPEED,
                seed=20250101,
                run_id=name,
            ),
        ),
        run_dir / "responses" / "source0.h5",
    )
    (run_dir / "report.json").write_text(json.dumps(_report()))
    return run_dir


def _pair(root: Path, **kwargs: Any) -> tuple[Path, Path]:
    high = _write_run(root, "high", fmax_hz=16000.0, seconds=0.6)
    mid = _write_run(root, "mid", fmax_hz=4000.0, seconds=0.6, **kwargs)
    return high, mid


def _build(root: Path, **kwargs: Any) -> dict[str, Any]:
    high, mid = _pair(root)
    return build(
        high,
        mid,
        root / "out",
        usd_per_hour_per_card=0.917,
        windows_ms=(30.0, 60.0, 150.0),
        seeds=2,
        **kwargs,
    )


def test_absorption_is_area_weighted() -> None:
    """100 m2 at 0.10 and 25 m2 at 0.50 is 0.18, not 0.30."""
    absorption = mean_absorption_of(_report(), bands=7)
    assert absorption[0] == pytest.approx(0.18)


def test_absorption_is_extended_to_the_analysis_banks_top_band() -> None:
    """The catalogue stops at 8 kHz; the bank at 48 kHz runs to 16."""
    absorption = mean_absorption_of(_report(bands=7), bands=8)
    assert len(absorption) == 8
    assert 0.0 < absorption[7] <= absorption[6]


def test_a_room_with_no_surface_is_refused() -> None:
    empty = _report()
    for entry in empty["room"]["per_class"]:
        entry["area_m2"] = 0.0
    with pytest.raises(ValueError):
        mean_absorption_of(empty, bands=7)


def test_cost_is_linear_in_the_window(tmp_path: Path) -> None:
    """The whole reason the lever exists."""
    report = _build(tmp_path)
    windows = {entry["window_ms"]: entry for entry in report["windows"]}
    assert windows[60.0]["usd"] == pytest.approx(2.0 * windows[30.0]["usd"], rel=0.02)
    assert windows[150.0]["usd"] == pytest.approx(5.0 * windows[30.0]["usd"], rel=0.02)


def test_a_longer_window_leaves_less_for_the_tail_to_supply(tmp_path: Path) -> None:
    report = _build(tmp_path)
    late = {entry["window_ms"]: entry["bands"][-1]["late_energy_db"] for entry in report["windows"]}
    assert late[150.0] < late[60.0] < late[30.0]


def test_the_error_is_reported_with_its_seed_spread(tmp_path: Path) -> None:
    """An error below the noise floor is not a measurement, and must show it."""
    report = _build(tmp_path)
    for window in report["windows"]:
        for band in window["bands"]:
            assert "t30_error_sd_pct" in band
            assert band["t30_error_sd_pct"] >= 0.0


def test_every_band_is_reported_and_flagged_against_fmax(tmp_path: Path) -> None:
    report = _build(tmp_path)
    bands = report["windows"][0]["bands"]
    assert [band["centre_hz"] for band in bands] == [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    assert all(band["in_band"] for band in bands)


def test_the_report_carries_the_ledger_fields(tmp_path: Path) -> None:
    report = _build(tmp_path)
    assert report["reference_run"] == "high"
    assert report["mid_band_run"] == "mid"
    assert report["scene_sha256"] == SCENE
    assert report["cache_key"] == "0" * 32
    assert "synthesised" in report["trick"]
    assert report["seeds"] == 2
    assert (tmp_path / "out" / "report.json").is_file()


def test_the_transposition_records_which_bands_were_extrapolated(tmp_path: Path) -> None:
    """The uncertainty of the method lives in exactly this distinction."""
    report = _build(tmp_path)
    prediction = report["transposition"][0]
    by_centre = dict(zip(prediction["centres_hz"], prediction["measured"], strict=True))
    assert by_centre[4000] is True
    assert by_centre[16000] is False


def test_two_geometries_cannot_calibrate_each_other(tmp_path: Path) -> None:
    high = _write_run(tmp_path, "high", fmax_hz=16000.0, seconds=0.6)
    other = _write_run(tmp_path, "mid", fmax_hz=4000.0, seconds=0.6, scene="a" * 64)
    with pytest.raises(ValueError, match="different geometries"):
        build(high, other, tmp_path / "out", usd_per_hour_per_card=0.917, seeds=1)


def test_two_sample_rates_cannot_be_compared(tmp_path: Path) -> None:
    high = _write_run(tmp_path, "high", fmax_hz=16000.0, seconds=0.6)
    other = _write_run(tmp_path, "mid", fmax_hz=4000.0, seconds=0.6, sample_rate=24000.0)
    with pytest.raises(ValueError, match="sample rates"):
        build(high, other, tmp_path / "out", usd_per_hour_per_card=0.917, seeds=1)


def test_the_hourly_rate_cannot_be_omitted(tmp_path: Path) -> None:
    high, mid = _pair(tmp_path)
    with pytest.raises(TypeError):
        build(high, mid, tmp_path / "out")  # type: ignore[call-arg]


def test_another_rooms_grid_can_be_priced_with_this_rooms_error(tmp_path: Path) -> None:
    """Pricing the kitchen while measuring accuracy on the bedroom."""
    high, mid = _pair(tmp_path)
    report = build(
        high,
        mid,
        tmp_path / "out",
        usd_per_hour_per_card=0.917,
        windows_ms=(60.0,),
        seeds=1,
        grid_points=int(5.44e9),
    )
    assert report["grid_points"] == int(5.44e9)
    assert report["windows"][0]["cards"] == 1
