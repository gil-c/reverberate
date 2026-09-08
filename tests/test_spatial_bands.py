"""Three solves on three grids become one ambisonic response, and the seams are measured."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from reverberate import metrics
from reverberate.audio import Atmosphere
from reverberate.spatial.bands import (
    BandSolve,
    assemble,
    calibration_bands_hz,
    centre_offsets,
    extend,
    pad_order,
)
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count

RATE = 48000
SOUND_SPEED = 343.2


def _truth(order: int = 1, seconds: float = 0.25, seed: int = 0) -> Ambisonic:
    """A first order field: one decaying broadband noise with fixed channel ratios.

    The ratios are the direction, and they are what the assembly must keep.
    """
    rng = np.random.default_rng(seed)
    samples = int(seconds * RATE)
    time = np.arange(samples) / RATE
    noise = rng.standard_normal(samples) * np.exp(-time / 0.04)
    noise[0] = 1.0
    ratios = np.array([1.0, 0.6, -0.3, 0.45])[: channel_count(order)]
    return Ambisonic(
        signals=np.outer(ratios, noise),
        sample_rate_hz=float(RATE),
        order=order,
        centre=np.zeros(3),
    )


def _band(
    truth: Ambisonic, name: str, fmax: float, step: float, *, seconds: float | None = None
) -> BandSolve:
    """What a solve at ``fmax`` would deliver: the truth low passed and louder by 16 kHz / fmax."""
    sos = signal.butter(8, fmax / (RATE / 2), output="sos")
    signals = signal.sosfiltfilt(sos, truth.signals, axis=-1) * (16000.0 / fmax)
    if seconds is not None:
        signals = signals[:, : int(seconds * RATE)]
    ambisonic = Ambisonic(
        signals=np.ascontiguousarray(signals),
        sample_rate_hz=float(RATE),
        order=truth.order,
        centre=np.full(3, step / 3.0),
    )
    return BandSolve(name, ambisonic, fmax, step)


def _three(truth: Ambisonic) -> tuple[BandSolve, BandSolve, BandSolve]:
    return (
        _band(truth, "low", 1000.0, 0.0327),
        _band(truth, "mid", 4000.0, 0.0082),
        _band(truth, "high", 16000.0, 0.00204),
    )


def test_the_calibration_bands_sit_an_octave_under_the_lower_solve() -> None:
    assert calibration_bands_hz(1000.0, RATE) == (125.0, 500.0)
    assert calibration_bands_hz(4000.0, RATE) == (250.0, 2000.0)


def test_padding_adds_silent_channels_and_never_removes_any() -> None:
    truth = _truth(order=1)
    padded = pad_order(truth, 3)
    assert padded.signals.shape[0] == channel_count(3)
    assert np.array_equal(padded.signals[:4], truth.signals)
    assert not padded.signals[4:].any()
    assert pad_order(truth, 1) is truth
    with pytest.raises(ValueError, match="pad order"):
        pad_order(padded, 1)


def test_the_assembly_levels_each_solve_by_the_ratio_its_bandwidth_predicts() -> None:
    low, mid, high = _three(_truth())
    _, record = assemble(low, mid, high)
    rows = {row["band"]: row for row in record["bands"]}
    assert rows["high"]["gain_db"] == pytest.approx(0.0)
    assert rows["mid"]["gain_db"] == pytest.approx(-12.04, abs=0.5)
    assert rows["low"]["gain_db"] == pytest.approx(-24.08, abs=1.0)
    for check in record["level_checks"]:
        assert abs(check["measured_db"] - check["predicted_db"]) < 1.0


def test_the_assembled_field_is_the_truth_band_for_band_and_keeps_its_direction() -> None:
    truth = _truth()
    low, mid, high = _three(truth)
    assembled, record = assemble(low, mid, high)
    delay = record["split"]["group_delay_samples"]
    # Same energy per octave as the truth, which none of the three solves had alone.
    got = metrics.octave_filter(assembled.signals[0][delay:], RATE)
    want = metrics.octave_filter(truth.signals[0][: got.shape[1]], RATE)
    centres = metrics.band_centres(RATE)
    for band, centre in enumerate(centres):
        if 125 <= centre <= 8000:
            level = 10 * np.log10((got[band] ** 2).sum() / (want[band] ** 2).sum())
            assert abs(level) < 1.0, (centre, level)
    # The direction is the ratio between channels, and it survives every band.
    ratios = (
        assembled.signals[1:, delay : delay + 4000] @ assembled.signals[0, delay : delay + 4000]
    )
    ratios /= (
        assembled.signals[0, delay : delay + 4000] @ assembled.signals[0, delay : delay + 4000]
    )
    assert ratios == pytest.approx([0.6, -0.3, 0.45], abs=0.02)


def test_a_solve_at_the_wrong_level_is_refused_rather_than_summed() -> None:
    low, mid, high = _three(_truth())
    loud = BandSolve(
        "mid",
        Ambisonic(
            signals=mid.ambisonic.signals * 10.0 ** (10 / 20),
            sample_rate_hz=float(RATE),
            order=mid.ambisonic.order,
            centre=mid.ambisonic.centre,
        ),
        mid.fmax_hz,
        mid.grid_step_m,
    )
    with pytest.raises(ValueError, match="wrong level"):
        assemble(low, loud, high)


def test_bands_of_different_orders_assemble_at_the_highest() -> None:
    truth = _truth()
    low, mid, high = _three(truth)
    omni_only = Ambisonic(
        signals=low.ambisonic.signals[:1],
        sample_rate_hz=float(RATE),
        order=0,
        centre=low.ambisonic.centre,
    )
    low = BandSolve("low", omni_only, low.fmax_hz, low.grid_step_m)
    high = BandSolve("high", pad_order(high.ambisonic, 3), high.fmax_hz, high.grid_step_m)
    assembled, record = assemble(low, mid, high)
    assert assembled.order == 3
    assert record["order"] == 3
    assert assembled.signals.shape[0] == channel_count(3)


def test_the_centre_offsets_are_bounded_by_half_a_cell_diagonal() -> None:
    low, mid, high = _three(_truth())
    rows = centre_offsets([low, mid, high], np.zeros(3))
    for row, solve in zip(rows, (low, mid, high), strict=True):
        assert row["offset_m"] <= row["bound_m"]
        assert row["bound_m"] == pytest.approx(np.sqrt(3) * solve.grid_step_m / 2, abs=1e-6)


def test_the_tail_continues_every_channel_independently_at_one_decay() -> None:
    truth = _truth(seconds=0.5)
    low, mid, high = _three(truth)
    short = _band(truth, "high", 16000.0, 0.00204, seconds=0.08)
    extended, record = extend(
        short,
        0.5,
        calibration=mid,
        mean_absorption=np.full(len(metrics.band_centres(RATE)), 0.2),
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
        seed=3,
    )
    assert extended.ambisonic.signals.shape == (4, int(0.5 * RATE))
    # The computed part is carried through untouched before the crossfade.
    fade = int(0.010 * RATE)
    window = int(0.08 * RATE)
    assert np.array_equal(
        extended.ambisonic.signals[:, : window - fade], short.ambisonic.signals[:, : window - fade]
    )
    # The tail decays: the last tenth is quieter than the first tenth after the seam.
    tail = extended.ambisonic.signals[:, window:]
    tenth = tail.shape[1] // 10
    assert (tail[:, -tenth:] ** 2).sum() < 0.1 * (tail[:, :tenth] ** 2).sum()
    # Independent noise per channel: the W and X tails are uncorrelated, where the
    # computed part had them at a fixed ratio.
    w, x = tail[0, 2 * tenth :], tail[1, 2 * tenth :]
    coherence = abs(float(w @ x) / np.sqrt(float(w @ w) * float(x @ x)))
    assert coherence < 0.1
    assert record["total_s"] == pytest.approx(0.5)
    assert len(record["tail_t60_s"]) == len(metrics.band_centres(RATE))


def test_extending_to_the_computed_length_changes_nothing() -> None:
    truth = _truth()
    low, mid, high = _three(truth)
    same, record = extend(
        high,
        high.ambisonic.duration_s,
        calibration=mid,
        mean_absorption=np.full(len(metrics.band_centres(RATE)), 0.2),
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
        seed=0,
    )
    assert same.ambisonic is high.ambisonic
    assert record["computed_s"] == record["total_s"]


def test_a_tail_cannot_be_shorter_than_what_was_solved() -> None:
    truth = _truth()
    low, mid, high = _three(truth)
    with pytest.raises(ValueError, match="cannot be extended"):
        extend(
            high,
            0.01,
            calibration=mid,
            mean_absorption=np.full(len(metrics.band_centres(RATE)), 0.2),
            atmosphere=Atmosphere(),
            sound_speed_m_s=SOUND_SPEED,
            seed=0,
        )
