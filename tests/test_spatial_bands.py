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
    continue_from,
    decay_from_bands,
    extend,
    extend_spectrum,
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


def _decaying_field(
    t60_by_band: dict[int, float], seconds: float = 0.4, seed: int = 5
) -> Ambisonic:
    """A first order field whose octave bands decay at known rates, all directions alike."""
    rng = np.random.default_rng(seed)
    samples = int(seconds * RATE)
    time = np.arange(samples) / RATE
    bands = metrics.octave_filter(rng.standard_normal(samples), RATE)
    centres = metrics.band_centres(RATE)
    w = np.zeros(samples)
    for band, centre in enumerate(centres):
        w += bands[band] * 10.0 ** (-3.0 * time / t60_by_band[int(centre)])
    ratios = np.array([1.0, 0.6, -0.3, 0.45])
    return Ambisonic(
        signals=np.outer(ratios, w), sample_rate_hz=float(RATE), order=1, centre=np.zeros(3)
    )


def test_the_extension_leaves_the_solved_band_alone_and_fills_the_rest() -> None:
    truth = _decaying_field({c: 0.3 for c in metrics.band_centres(RATE)})
    solved = Ambisonic(
        signals=signal.sosfiltfilt(
            signal.butter(8, 8000 / (RATE / 2), output="sos"), truth.signals, axis=-1
        ),
        sample_rate_hz=float(RATE),
        order=1,
        centre=np.zeros(3),
    )
    t60 = np.full(len(metrics.band_centres(RATE)), 0.3)
    extended, record = extend_spectrum(
        solved, fmax_hz=8000.0, t60_s=t60, atmosphere=Atmosphere(), sound_speed_m_s=SOUND_SPEED
    )
    assert extended.signals.shape == solved.signals.shape
    # Below the template top nothing moved, to the leakage of one Hann frame.
    spectrum = np.abs(np.fft.rfft(extended.signals[0] - solved.signals[0]))
    had_spectrum = np.abs(np.fft.rfft(solved.signals[0]))
    below = np.fft.rfftfreq(solved.signals.shape[1], 1.0 / RATE) < 6000.0
    residual_db = 10 * np.log10((spectrum[below] ** 2).sum() / (had_spectrum[below] ** 2).sum())
    assert residual_db < -30.0
    # Above it there is now the truth's energy, where the low pass had left little.
    got = metrics.octave_filter(extended.signals[0], RATE)
    had = metrics.octave_filter(solved.signals[0], RATE)
    want = metrics.octave_filter(truth.signals[0], RATE)
    top = list(metrics.band_centres(RATE)).index(16000)
    assert (got[top] ** 2).sum() > 10.0 * (had[top] ** 2).sum()
    assert abs(10 * np.log10((got[top] ** 2).sum() / (want[top] ** 2).sum())) < 3.0
    assert record["synthesised_hz"][0] == pytest.approx(7200.0)


def test_the_synthesised_octave_decays_at_the_rate_it_was_told() -> None:
    centres = metrics.band_centres(RATE)
    truth_t60 = {c: 0.5 for c in centres}
    truth_t60[16000] = 0.15
    truth = _decaying_field(truth_t60, seconds=0.6)
    solved = Ambisonic(
        signals=signal.sosfiltfilt(
            signal.butter(8, 8000 / (RATE / 2), output="sos"), truth.signals, axis=-1
        ),
        sample_rate_hz=float(RATE),
        order=1,
        centre=np.zeros(3),
    )
    t60 = np.array([truth_t60[c] for c in centres], dtype=float)
    # The air term is inside those decay times already; a dry atmosphere keeps
    # the check about the rule and not about ISO 9613-1.
    extended, _ = extend_spectrum(
        solved,
        fmax_hz=8000.0,
        t60_s=t60,
        atmosphere=Atmosphere(humidity_percent=100.0),
        sound_speed_m_s=1e-9,
    )
    # Judged against the truth's own measurement, not the number it was built
    # from: the octave filter bank leaks the slower neighbour into this band,
    # for the truth and for the synthesis alike.
    measured = metrics.rt60_per_band(extended.signals[0], RATE)
    reference = metrics.rt60_per_band(truth.signals[0], RATE)
    top = list(centres).index(16000)
    assert measured[top] == pytest.approx(reference[top], rel=0.25)
    assert measured[top] < 0.7 * reference[top - 1]
    # And the direction survives: the channel ratios in the new octave are the old ones.
    hf = signal.sosfiltfilt(
        signal.butter(8, [10000 / (RATE / 2), 20000 / (RATE / 2)], btype="band", output="sos"),
        extended.signals,
        axis=-1,
    )
    ratios = hf[1:] @ hf[0] / (hf[0] @ hf[0])
    assert ratios == pytest.approx([0.6, -0.3, 0.45], abs=0.02)


def test_the_extension_refuses_a_ceiling_under_the_solved_band() -> None:
    truth = _decaying_field({c: 0.3 for c in metrics.band_centres(RATE)}, seconds=0.1)
    with pytest.raises(ValueError, match="nothing to synthesise"):
        extend_spectrum(
            truth,
            fmax_hz=8000.0,
            t60_s=np.full(8, 0.3),
            atmosphere=Atmosphere(),
            sound_speed_m_s=SOUND_SPEED,
            ceiling_hz=5000.0,
        )


def test_the_high_band_is_continued_by_the_mid_band_s_own_reflections() -> None:
    """After its window the high band carries the mid band's structure, lifted an octave."""
    truth = _decaying_field({c: 0.4 for c in metrics.band_centres(RATE)}, seconds=0.4)
    # A discrete reflection at 0.25 s in the mid band, after the high band's window.
    mid_signals = (
        signal.sosfiltfilt(
            signal.butter(8, 4000 / (RATE / 2), output="sos"), truth.signals, axis=-1
        )
        * 2.0
    )
    spike = int(0.25 * RATE)
    mid_signals[:, spike] += np.array([1.0, 0.6, -0.3, 0.45]) * 0.5
    mid = BandSolve(
        "mid",
        Ambisonic(signals=mid_signals, sample_rate_hz=float(RATE), order=1, centre=np.zeros(3)),
        4000.0,
        0.0082,
    )
    high = _band(truth, "high", 8000.0, 0.0041, seconds=0.1)
    decay = np.full(len(metrics.band_centres(RATE)), 0.4)
    continued, record = continue_from(
        high, mid, gain=0.5, t60_s=decay, atmosphere=Atmosphere(), sound_speed_m_s=SOUND_SPEED
    )
    assert record["continued"] is True
    assert continued.ambisonic.signals.shape[1] == mid.ambisonic.signals.shape[1]
    window, fade = int(0.1 * RATE), int(0.010 * RATE)
    assert np.array_equal(
        continued.ambisonic.signals[:, : window - fade], high.ambisonic.signals[:, : window - fade]
    )
    # The reflection shows up above 4 kHz after the window, where the high band had nothing.
    hf = signal.sosfiltfilt(
        signal.butter(8, [4500 / (RATE / 2), 7500 / (RATE / 2)], btype="band", output="sos"),
        continued.ambisonic.signals[0],
    )
    around = np.abs(hf[spike - 200 : spike + 200]).max()
    elsewhere = np.abs(hf[spike + 2000 : spike + 4000]).max()
    assert around > 3.0 * elsewhere
    # And it keeps the mid band's direction, which is the ratio between channels.
    seg = continued.ambisonic.signals[:, spike - 50 : spike + 50]
    ratios = seg[1:] @ seg[0] / (seg[0] @ seg[0])
    assert ratios == pytest.approx([0.6, -0.3, 0.45], abs=0.05)


def test_a_mid_band_no_longer_than_the_high_band_continues_nothing() -> None:
    truth = _truth(seconds=0.1)
    low, mid, high = _three(truth)
    same, record = continue_from(
        high,
        mid,
        gain=1.0,
        t60_s=np.full(8, 0.3),
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
    )
    assert record["continued"] is False
    assert same is high


def test_each_band_s_decay_comes_from_the_solve_that_can_measure_it() -> None:
    centres = metrics.band_centres(RATE)
    slow = _decaying_field({c: 0.9 for c in centres}, seconds=1.0, seed=1)
    fast = _decaying_field({c: 0.3 for c in centres}, seconds=0.4, seed=2)
    low = _band(slow, "low", 1000.0, 0.0327)
    mid = _band(fast, "mid", 4000.0, 0.0082)
    decay, record = decay_from_bands(
        low,
        mid,
        mean_absorption=np.full(8, 0.2),
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
    )
    source = dict(zip(centres, record["source"], strict=True))
    assert source[125] == "low" and source[500] == "low"
    assert source[1000] == "mid" and source[2000] == "mid"
    assert source[8000] == "transposed" and source[16000] == "transposed"
    by_band = dict(zip(centres, decay, strict=True))
    assert by_band[250] == pytest.approx(0.9, rel=0.2)
    assert by_band[2000] == pytest.approx(0.3, rel=0.2)


def test_a_given_decay_overrides_the_local_fit() -> None:
    truth = _truth(seconds=0.5)
    low, mid, high = _three(truth)
    short = _band(truth, "high", 16000.0, 0.00204, seconds=0.08)
    given = np.full(len(metrics.band_centres(RATE)), 0.2)
    _, record = extend(
        short,
        0.5,
        calibration=mid,
        mean_absorption=np.full(8, 0.2),
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
        seed=0,
        t60_s=given,
    )
    assert record["decay_given"] is True
    assert not any(record["from_local_fit"])
    assert record["tail_t60_s"] == [0.2] * 8


def test_the_seams_sit_under_the_solves_own_edges_by_default() -> None:
    low, mid, high = _three(_truth())
    _, record = assemble(low, mid, high)
    assert record["split"]["crossovers_hz"] == [800.0, 3200.0]
    _, explicit = assemble(low, mid, high, crossovers_hz=(1000.0, 4000.0))
    assert explicit["split"]["crossovers_hz"] == [1000.0, 4000.0]
