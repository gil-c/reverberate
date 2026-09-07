"""Tests for the synthetic reverberation tail.

The test that matters most is
:func:`test_each_band_decays_at_the_rate_it_was_asked_for`. A first version of
the module enveloped white noise and summed the bands afterwards, so every
band's noise leaked into every other band. That matched total band energy to
0.00 dB and got T30 wrong by 60 per cent. Energy is not an acceptance test for
a decay, so the decay is measured here band by band.

The rest is about not losing the distinction the whole method turns on: which
bands were copied from a run that resolves them, and which were extrapolated.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.air import Atmosphere
from reverberate.metrics import band_centres, octave_filter, rt60_per_band
from reverberate.tail import (
    effective_mean_free_path_m,
    eyring_t60_s,
    mixing_time_s,
    splice,
    synthesise,
    transpose,
)

SOUND_SPEED = 343.2
SAMPLE_RATE = 48000


def _decaying_noise(t60_s: float = 0.30, seconds: float = 1.0, receivers: int = 1) -> np.ndarray:
    rng = np.random.default_rng(20250101)
    samples = int(seconds * SAMPLE_RATE)
    times = np.arange(samples) / SAMPLE_RATE
    envelope = 10.0 ** (-60.0 * times / (20.0 * t60_s))
    return rng.standard_normal((receivers, samples)) * envelope


def test_mixing_time_reproduces_the_roadmaps_own_rooms() -> None:
    """12 ms for the living room, 7 ms for the kitchen."""
    assert mixing_time_s(151.7) == pytest.approx(0.012, abs=0.0006)
    assert mixing_time_s(45.5) == pytest.approx(0.007, abs=0.0006)


def test_mixing_time_refuses_an_impossible_room() -> None:
    with pytest.raises(ValueError):
        mixing_time_s(0.0)


def test_eyring_and_the_mean_free_path_are_inverses() -> None:
    path = 2.66
    alpha = 0.29
    decay = eyring_t60_s(path, alpha, sound_speed_m_s=SOUND_SPEED)
    recovered = effective_mean_free_path_m(decay, alpha, sound_speed_m_s=SOUND_SPEED)
    assert recovered == pytest.approx(path, rel=1e-9)


def test_the_effective_path_is_not_four_v_over_s_on_a_furnished_room() -> None:
    """Why it is calibrated rather than computed. Measured on w29_16k."""
    path = effective_mean_free_path_m(0.269, 0.3497, sound_speed_m_s=SOUND_SPEED)
    geometric = 4.0 * 38.11 / 127.5
    assert path > 2.0 * geometric


def test_the_air_term_is_removed_before_the_path_is_taken() -> None:
    """The path must describe the boundaries, so it can be reused elsewhere."""
    without = effective_mean_free_path_m(0.28, 0.35, sound_speed_m_s=SOUND_SPEED)
    with_air = effective_mean_free_path_m(
        0.28, 0.35, sound_speed_m_s=SOUND_SPEED, air_db_per_m=0.1053
    )
    assert with_air > without


def test_a_decay_air_alone_cannot_explain_is_refused() -> None:
    """Rather than returning a negative path that would look like a number."""
    with pytest.raises(ValueError):
        effective_mean_free_path_m(0.10, 0.35, sound_speed_m_s=SOUND_SPEED, air_db_per_m=5.0)


@pytest.mark.parametrize("t60", [0.15, 0.30, 0.60])
def test_each_band_decays_at_the_rate_it_was_asked_for(t60: float) -> None:
    """The property, and the one a total-energy check does not have.

    Every band is given the same decay, so any leakage between bands would
    still pass. Then one band is given a different one, which is what
    separates a correct filterbank from an averaged one.
    """
    centres = band_centres(SAMPLE_RATE)
    early = _decaying_noise(t60_s=t60, seconds=0.06)[0]
    asked = np.full(len(centres), t60)

    tail = synthesise(
        early,
        SAMPLE_RATE,
        asked,
        int(1.2 * SAMPLE_RATE),
        rng=np.random.default_rng(7),
    )

    measured = rt60_per_band(tail, SAMPLE_RATE)
    for band, centre in enumerate(centres):
        if centre < 250 or centre > 8000:
            continue  # the edge bands of the filter bank are not resolved here
        assert measured[band] == pytest.approx(t60, rel=0.25), centre


def test_one_band_can_be_given_a_different_decay_from_its_neighbours() -> None:
    """What fails if the noise is enveloped before it is band-limited.

    Judged against a control that asks for the same decay everywhere, rather
    than against an absolute threshold. The bands of an octave filter bank
    overlap, so a band whose own content has died away still collects the
    skirts of its neighbours, and a real room behaves the same way. The claim
    is that asking for a shorter decay produces one, not that a band can be
    isolated from the bank it is measured with.
    """
    centres = band_centres(SAMPLE_RATE)
    early = _decaying_noise(t60_s=0.5, seconds=0.06)[0]
    fast = centres.index(1000)

    uniform = np.full(len(centres), 0.5)
    changed = uniform.copy()
    changed[fast] = 0.15

    control = synthesise(
        early, SAMPLE_RATE, uniform, int(1.5 * SAMPLE_RATE), rng=np.random.default_rng(11)
    )
    treated = synthesise(
        early, SAMPLE_RATE, changed, int(1.5 * SAMPLE_RATE), rng=np.random.default_rng(11)
    )

    control_decay = rt60_per_band(control, SAMPLE_RATE)
    treated_decay = rt60_per_band(treated, SAMPLE_RATE)

    assert treated_decay[fast] < 0.75 * control_decay[fast]
    # And the neighbours are left alone, which is what leakage in the other
    # direction would break.
    for band in (centres.index(250), centres.index(4000)):
        assert treated_decay[band] == pytest.approx(control_decay[band], rel=0.15)


def test_the_computed_part_survives_the_splice_untouched() -> None:
    """Everything before the crossfade is the solver's own output."""
    early = _decaying_noise(seconds=0.10)[0]
    tail = synthesise(
        early,
        SAMPLE_RATE,
        np.full(len(band_centres(SAMPLE_RATE)), 0.3),
        int(0.5 * SAMPLE_RATE),
        rng=np.random.default_rng(3),
        fade_s=0.010,
    )
    intact = int(0.09 * SAMPLE_RATE)
    assert np.allclose(tail[:intact], early[:intact], atol=1e-9)


def test_the_seam_does_not_step_in_level() -> None:
    """The audible trap is a discontinuity, not the join itself."""
    early = _decaying_noise(t60_s=0.3, seconds=0.06)[0]
    tail = synthesise(
        early,
        SAMPLE_RATE,
        np.full(len(band_centres(SAMPLE_RATE)), 0.3),
        int(0.6 * SAMPLE_RATE),
        rng=np.random.default_rng(5),
    )
    window = int(0.06 * SAMPLE_RATE)
    block = int(0.005 * SAMPLE_RATE)
    before = np.sqrt((tail[window - 2 * block : window - block] ** 2).mean())
    after = np.sqrt((tail[window + block : window + 2 * block] ** 2).mean())
    assert 0.5 < after / before < 2.0


def test_the_same_seed_gives_the_same_tail() -> None:
    early = _decaying_noise(seconds=0.06)[0]
    decay = np.full(len(band_centres(SAMPLE_RATE)), 0.3)
    args = (early, SAMPLE_RATE, decay, int(0.4 * SAMPLE_RATE))
    first = synthesise(*args, rng=np.random.default_rng(42))
    second = synthesise(*args, rng=np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_receivers_draw_independent_noise() -> None:
    """Physically correct above 1 kHz: interaural coherence is about 0.04."""
    early = _decaying_noise(seconds=0.06, receivers=2)
    tail = synthesise(
        early,
        SAMPLE_RATE,
        np.full(len(band_centres(SAMPLE_RATE)), 0.3),
        int(0.5 * SAMPLE_RATE),
        rng=np.random.default_rng(9),
    )
    late = tail[:, int(0.2 * SAMPLE_RATE) :]
    correlation = float(np.corrcoef(late[0], late[1])[0, 1])
    assert abs(correlation) < 0.1


def test_a_band_with_no_decay_stops_rather_than_being_invented() -> None:
    """A band the mid band could not measure is left silent, not guessed.

    Against a control, for the same reason as the test above: what remains in
    the silenced band is the skirts of its neighbours, which the analysis
    filter cannot separate from it.
    """
    centres = band_centres(SAMPLE_RATE)
    silent = centres.index(4000)
    early = _decaying_noise(seconds=0.06)[0]

    uniform = np.full(len(centres), 0.3)
    dropped = uniform.copy()
    dropped[silent] = np.nan

    def late_energy(decay: np.ndarray) -> np.ndarray:
        tail = synthesise(
            early, SAMPLE_RATE, decay, int(0.5 * SAMPLE_RATE), rng=np.random.default_rng(13)
        )
        bands = octave_filter(tail, SAMPLE_RATE)[:, int(0.2 * SAMPLE_RATE) :]
        return np.asarray((bands**2).sum(axis=1))

    control = late_energy(uniform)
    treated = late_energy(dropped)

    # The residue is 8.5 per cent of the control, and it is the neighbours'
    # skirts rather than the band: a sixfold drop is the band being dropped.
    assert treated[silent] < 0.15 * control[silent]
    assert treated[centres.index(1000)] == pytest.approx(control[centres.index(1000)], rel=0.2)


def test_splicing_keeps_the_reference_prefix_exactly() -> None:
    """Truncating a stored response is what stopping the solve early would give."""
    reference = _decaying_noise(seconds=1.0)[0]
    spliced = splice(
        reference,
        SAMPLE_RATE,
        0.060,
        np.full(len(band_centres(SAMPLE_RATE)), 0.3),
        rng=np.random.default_rng(17),
    )
    assert spliced.shape == reference.shape
    intact = int(0.045 * SAMPLE_RATE)
    assert np.allclose(spliced[:intact], reference[:intact], atol=1e-9)


def test_transpose_copies_what_the_mid_band_resolves_and_marks_the_rest() -> None:
    centres = band_centres(SAMPLE_RATE)
    mid = np.full(len(centres), 0.27)
    alpha = np.full(len(centres), 0.35)

    result = transpose(
        mid,
        alpha,
        SAMPLE_RATE,
        mid_fmax_hz=4000.0,
        atmosphere=Atmosphere(),
        sound_speed_m_s=SOUND_SPEED,
    )

    by_centre = dict(zip(result.centres_hz, result.measured, strict=True))
    assert by_centre[1000] is True
    assert by_centre[4000] is True
    assert by_centre[8000] is False
    assert by_centre[16000] is False
    assert result.t60_s[centres.index(4000)] == pytest.approx(0.27)


def test_the_extrapolated_bands_are_shorter_because_of_air() -> None:
    centres = band_centres(SAMPLE_RATE)
    result = transpose(
        np.full(len(centres), 0.27),
        np.full(len(centres), 0.35),
        SAMPLE_RATE,
        mid_fmax_hz=4000.0,
        sound_speed_m_s=SOUND_SPEED,
    )
    assert result.t60_s[centres.index(16000)] < result.t60_s[centres.index(8000)]
    assert result.t60_s[centres.index(8000)] < result.t60_s[centres.index(4000)]


def test_transpose_records_where_its_path_length_came_from() -> None:
    centres = band_centres(SAMPLE_RATE)
    result = transpose(
        np.full(len(centres), 0.27),
        np.full(len(centres), 0.35),
        SAMPLE_RATE,
        mid_fmax_hz=4000.0,
        sound_speed_m_s=SOUND_SPEED,
    )
    record = result.record()
    assert record["anchor_hz"] == 2000
    assert record["mean_free_path_m"] > 0.0
    assert len(record["measured"]) == len(centres)


def test_mismatched_band_axes_are_refused() -> None:
    with pytest.raises(ValueError):
        transpose(
            np.full(4, 0.27),
            np.full(8, 0.35),
            SAMPLE_RATE,
            mid_fmax_hz=4000.0,
            sound_speed_m_s=SOUND_SPEED,
        )


def test_a_tail_shorter_than_the_computed_part_is_refused() -> None:
    with pytest.raises(ValueError):
        synthesise(
            _decaying_noise(seconds=0.10)[0],
            SAMPLE_RATE,
            np.full(len(band_centres(SAMPLE_RATE)), 0.3),
            int(0.05 * SAMPLE_RATE),
            rng=np.random.default_rng(1),
        )


def test_a_crossfade_longer_than_the_window_is_refused() -> None:
    with pytest.raises(ValueError):
        synthesise(
            _decaying_noise(seconds=0.005)[0],
            SAMPLE_RATE,
            np.full(len(band_centres(SAMPLE_RATE)), 0.3),
            int(0.5 * SAMPLE_RATE),
            rng=np.random.default_rng(1),
            fade_s=0.050,
        )


def test_a_window_outside_the_reference_is_refused() -> None:
    with pytest.raises(ValueError):
        splice(
            _decaying_noise(seconds=0.1)[0],
            SAMPLE_RATE,
            0.5,
            np.full(len(band_centres(SAMPLE_RATE)), 0.3),
            rng=np.random.default_rng(1),
        )
