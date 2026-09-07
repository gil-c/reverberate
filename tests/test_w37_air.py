"""Tests for the air-absorption experiment.

The experiment's job is to report a correction honestly, so the tests are about
reporting as much as about arithmetic: that a band above the run's own ``fmax``
is flagged rather than quoted, that an unmeasurable decay comes back as null
rather than as a plausible number, and that the analytic prediction and the
measurement are both present because they disagree.

The response is synthetic and built here, so the expected decay is known by
construction rather than read off a stored run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate.air import Atmosphere
from reverberate.experiments.w37_air import (
    ReceiverDecay,
    analytic_t60_s,
    build,
    measure_run,
)
from reverberate.response import Provenance, ResponseSet, write_raw

SOUND_SPEED = 343.2
SAMPLE_RATE = 48000.0


def _provenance(fmax_hz: float, run_id: str) -> Provenance:
    return Provenance(
        scene_sha256="f" * 64,
        mats_hash="0" * 32,
        engine="cpu",
        band="wave",
        fmax_hz=fmax_hz,
        grid_step_m=343.2 / (fmax_hz * 10.5),
        points_per_wavelength=10.5,
        sound_speed_m_s=SOUND_SPEED,
        seed=20250101,
        run_id=run_id,
    )


def _response(fmax_hz: float = 16000.0, t60_s: float = 0.4, receivers: int = 3) -> ResponseSet:
    """Decaying broadband noise with a known reverberation time."""
    rng = np.random.default_rng(20250101)
    samples = int(1.2 * SAMPLE_RATE)
    times = np.arange(samples) / SAMPLE_RATE
    envelope = 10.0 ** (-60.0 * times / (20.0 * t60_s))
    ir = rng.standard_normal((receivers, samples)) * envelope
    return ResponseSet(
        ir=ir,
        sample_rate_hz=SAMPLE_RATE,
        source_position=np.zeros(3),
        receiver_positions=np.stack([np.array([1.0 + i, 0.0, 0.0]) for i in range(receivers)]),
        provenance=_provenance(fmax_hz, "synthetic"),
        room_volume_m3=38.1,
    )


def _measure(response: ResponseSet) -> list[ReceiverDecay]:
    return measure_run(
        response,
        Atmosphere(),
        fmax_hz=response.provenance.fmax_hz,
        sound_speed_m_s=SOUND_SPEED,
    )


def test_air_shortens_the_top_of_the_band_and_leaves_the_bottom_alone() -> None:
    receiver = _measure(_response())[0]
    by_centre = {band.centre_hz: band for band in receiver.bands}

    assert abs(by_centre[250].t30_change_pct) < 1.0
    assert by_centre[16000].t30_change_pct < -20.0


def test_the_correction_grows_monotonically_with_frequency() -> None:
    receiver = _measure(_response())[0]
    changes = [
        band.t30_change_pct for band in receiver.bands if band.centre_hz >= 1000 and band.in_band
    ]
    assert all(later <= earlier for earlier, later in zip(changes, changes[1:], strict=False))


def test_bands_above_fmax_are_flagged_rather_than_quoted() -> None:
    """A 4 kHz run must not be read as if it held a 16 kHz decay."""
    receiver = _measure(_response(fmax_hz=4000.0))[0]
    in_band = {band.centre_hz for band in receiver.bands if band.in_band}
    assert 4000 in in_band
    assert 8000 not in in_band
    assert 16000 not in in_band


def test_energy_is_never_gained_in_any_band() -> None:
    receiver = _measure(_response())[0]
    assert all(band.energy_lost_db <= 0.0 for band in receiver.bands)


def test_every_receiver_is_reported_separately() -> None:
    """The rule W30 exists to enforce: never one scalar."""
    receivers = _measure(_response(receivers=4))
    assert len(receivers) == 4
    assert [receiver.index for receiver in receivers] == [0, 1, 2, 3]
    assert receivers[1].distance_m > receivers[0].distance_m


def test_an_unmeasurable_decay_is_null_and_not_a_plausible_number() -> None:
    """JSON has no NaN, and a silent band must not come back as a duration."""
    silent = _response()
    quiet = np.array(silent.ir)
    quiet[:] = 0.0
    receiver = _measure(
        ResponseSet(
            ir=quiet,
            sample_rate_hz=silent.sample_rate_hz,
            source_position=silent.source_position,
            receiver_positions=silent.receiver_positions,
            provenance=silent.provenance,
        )
    )[0]
    record = receiver.record()
    assert all(band["t30_dry_s"] is None for band in record["bands"])
    json.dumps(record)


@pytest.mark.parametrize(
    ("centre_hz", "t60_dry_s", "expected_air_alone", "expected_combined"),
    [
        (1000.0, 0.252, 37.5, 0.250),
        (2000.0, 0.269, 17.7, 0.265),
        (4000.0, 0.276, 5.90, 0.264),
        (8000.0, 0.278, 1.66, 0.238),
        (16000.0, 0.291, 0.48, 0.181),
    ],
)
def test_the_analytic_prediction_reproduces_the_published_table(
    centre_hz: float,
    t60_dry_s: float,
    expected_air_alone: float,
    expected_combined: float,
) -> None:
    """Roadmap W30's air table, every row, and it is analytic rather than measured."""
    air_alone, combined = analytic_t60_s(
        t60_dry_s, centre_hz, Atmosphere(), sound_speed_m_s=SOUND_SPEED
    )
    assert air_alone == pytest.approx(expected_air_alone, rel=0.01)
    assert combined == pytest.approx(expected_combined, abs=0.001)


def test_the_analytic_prediction_reads_long_inside_a_wide_band() -> None:
    """The mechanism the report exists to show.

    ``m(f)`` goes as the square of frequency, so the top of an octave is
    absorbed four times as fast as its bottom, and a T30 measured on the
    filtered response falls short of the band-centre formula.
    """
    receiver = _measure(_response())[0]
    by_centre = {band.centre_hz: band for band in receiver.bands}
    assert by_centre[8000].analytic_excess_pct > 2.0
    assert abs(by_centre[250].analytic_excess_pct) < 1.0


def test_a_silent_band_still_gives_the_air_only_time(tmp_path: Path) -> None:
    air_alone, combined = analytic_t60_s(
        float("nan"), 16000.0, Atmosphere(), sound_speed_m_s=SOUND_SPEED
    )
    assert air_alone == pytest.approx(0.48, rel=0.01)
    assert not np.isfinite(combined)


def test_the_report_carries_the_ledger_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic_run"
    (run_dir / "responses").mkdir(parents=True)
    write_raw(_response(), run_dir / "responses" / "source0.h5")

    report = build([run_dir], tmp_path / "out")

    assert report["reference_run"] is None
    assert "exp(-m(f) c t)" in report["trick"]
    assert report["atmosphere"]["relative_humidity_pct"] == 50.0
    assert report["atmosphere"]["standard"] == "ISO 9613-1"
    assert report["scene_sha256"] == ["f" * 64]
    assert report["scene_sha256_shared"] is True
    assert (tmp_path / "out" / "report.json").is_file()


def test_a_mismatched_pair_of_geometries_is_flagged(tmp_path: Path) -> None:
    """Comparing two runs of different scenes is void, and must say so."""
    for index, digest in enumerate(("a" * 64, "b" * 64)):
        run_dir = tmp_path / f"run{index}"
        (run_dir / "responses").mkdir(parents=True)
        response = _response()
        write_raw(
            ResponseSet(
                ir=response.ir,
                sample_rate_hz=response.sample_rate_hz,
                source_position=response.source_position,
                receiver_positions=response.receiver_positions,
                provenance=Provenance(**{**vars(response.provenance), "scene_sha256": digest}),
            ),
            run_dir / "responses" / "source0.h5",
        )

    report = build([tmp_path / "run0", tmp_path / "run1"], tmp_path / "out")
    assert report["scene_sha256_shared"] is False
    assert len(report["scene_sha256"]) == 2


def test_a_declared_humidity_changes_the_answer(tmp_path: Path) -> None:
    """Why the atmosphere is recorded rather than assumed."""
    response = _response()
    dry = measure_run(
        response,
        Atmosphere(relative_humidity_pct=30.0),
        fmax_hz=16000.0,
        sound_speed_m_s=SOUND_SPEED,
    )[0]
    humid = measure_run(
        response,
        Atmosphere(relative_humidity_pct=80.0),
        fmax_hz=16000.0,
        sound_speed_m_s=SOUND_SPEED,
    )[0]
    top_dry = next(band for band in dry.bands if band.centre_hz == 16000)
    top_humid = next(band for band in humid.bands if band.centre_hz == 16000)
    assert top_dry.t30_air_s < top_humid.t30_air_s
