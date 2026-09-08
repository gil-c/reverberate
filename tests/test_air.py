"""Tests for atmospheric absorption.

Two kinds of check, and the project needs both.

The coefficient is checked against **published figures** the roadmap already
quotes, so a change to the formula shows up as a disagreement with a number
that was written down before the code existed. Fitting the code and then
writing the expectation from its output would test nothing.

The application is checked against an **analytic case**: a pure tone at ``f``
must decay at exactly ``m(f) c`` decibels per second, whatever the
implementation does internally. That is the property, and it is what makes the
short-time transform an implementation detail rather than a second model.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.air import (
    Atmosphere,
    apply,
    attenuation_db_per_m,
    gain,
)
from reverberate.metrics import octave_filter, rt60_per_band

SOUND_SPEED = 343.2

#: The roadmap's own table, section W30, at 20 C and 50 per cent relative
#: humidity, in dB/m. Written down before this module existed.
ROADMAP_DB_PER_M = {
    1000.0: 0.0047,
    2000.0: 0.0099,
    4000.0: 0.0297,
    8000.0: 0.1053,
    16000.0: 0.3645,
}


@pytest.mark.parametrize(("frequency", "expected"), sorted(ROADMAP_DB_PER_M.items()))
def test_coefficient_matches_the_published_table(frequency: float, expected: float) -> None:
    measured = float(attenuation_db_per_m(frequency, Atmosphere()))
    assert measured == pytest.approx(expected, abs=5e-5)


def test_air_alone_gives_the_quoted_reverberation_time_at_16_khz() -> None:
    """Roadmap: air alone is a T60 of 0.48 s at 16 kHz."""
    coefficient = float(attenuation_db_per_m(16000.0, Atmosphere()))
    assert 60.0 / (coefficient * SOUND_SPEED) == pytest.approx(0.48, abs=0.005)


def test_the_43_db_per_500_ms_figure_is_the_humid_end_of_the_range() -> None:
    """Section 4.1's headline number is the 80 per cent case, the mildest."""
    humid = float(attenuation_db_per_m(16000.0, Atmosphere(relative_humidity_pct=80.0)))
    assert humid * SOUND_SPEED * 0.5 == pytest.approx(43.0, abs=0.5)


def test_humidity_moves_16_khz_by_nearly_a_factor_of_two() -> None:
    """Why humidity is declared per run rather than defaulted silently."""
    dry = float(attenuation_db_per_m(16000.0, Atmosphere(relative_humidity_pct=30.0)))
    humid = float(attenuation_db_per_m(16000.0, Atmosphere(relative_humidity_pct=80.0)))
    assert dry / humid == pytest.approx(1.85, abs=0.05)


def test_attenuation_rises_with_frequency() -> None:
    frequencies = np.array([125.0, 250.0, 500.0, 1e3, 2e3, 4e3, 8e3, 16e3])
    coefficients = attenuation_db_per_m(frequencies, Atmosphere())
    assert np.all(np.diff(coefficients) > 0.0)


def test_attenuation_is_negligible_in_the_low_bands() -> None:
    """Below 1 kHz a room-sized path loses under a tenth of a decibel."""
    assert float(attenuation_db_per_m(250.0, Atmosphere())) * 20.0 < 0.1


def test_zero_frequency_is_not_absorbed() -> None:
    assert float(attenuation_db_per_m(0.0, Atmosphere())) == 0.0


def test_gain_is_unity_at_the_origin_and_falls_after() -> None:
    at_zero = gain(16000.0, 0.0, sound_speed_m_s=SOUND_SPEED)
    later = gain(16000.0, 0.1, sound_speed_m_s=SOUND_SPEED)
    assert float(at_zero) == pytest.approx(1.0)
    assert 0.0 < float(later) < 1.0


def test_gain_broadcasts_frequency_against_time() -> None:
    surface = gain(
        np.array([1e3, 16e3])[:, None],
        np.array([0.0, 0.1, 0.2])[None, :],
        sound_speed_m_s=SOUND_SPEED,
    )
    assert surface.shape == (2, 3)
    assert np.all(surface[1] <= surface[0])


@pytest.mark.parametrize("frequency", [2000.0, 8000.0, 16000.0])
def test_a_pure_tone_decays_at_exactly_the_analytic_rate(frequency: float) -> None:
    """The property the whole module exists to have.

    A tone at ``f`` must lose ``m(f) c`` decibels every second. Measured on the
    block RMS of the filtered signal, well inside the record so neither edge is
    involved.
    """
    sample_rate = 48000.0
    times = np.arange(int(0.4 * sample_rate)) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency * times)

    absorbed = apply(tone, sample_rate, sound_speed_m_s=SOUND_SPEED)

    interior = absorbed[int(0.05 * sample_rate) : int(0.35 * sample_rate)]
    blocks = interior.reshape(-1, 480)
    decibels = 20.0 * np.log10(np.sqrt((blocks**2).mean(axis=1)))
    seconds = np.arange(len(decibels)) * 480 / sample_rate
    measured = float(np.polyfit(seconds, decibels, 1)[0])

    analytic = -float(attenuation_db_per_m(frequency, Atmosphere())) * SOUND_SPEED
    assert measured == pytest.approx(analytic, rel=0.01)


def test_the_frame_does_not_move_the_quantities_a_run_reports() -> None:
    """What fixes the default frame length, rather than taste.

    A total energy check is too weak here: it is dominated by the low bands,
    where the gain is almost one, so it would pass while the top of the band
    was badly wrong. The quantities this project actually publishes are
    per-octave, so those are what the frame must not move.

    Written after the W10 branch measured its own frame drifting on a long
    response. On a decay of the length this project simulates, the default and
    a frame sixteen times longer agree far inside the 3.1 per cent noise floor.
    """
    rng = np.random.default_rng(20250101)
    sample_rate = 48000
    times = np.arange(int(1.0 * sample_rate)) / sample_rate
    decay = rng.standard_normal(len(times)) * 10.0 ** (-60.0 * times / (20.0 * 0.29))

    coarse = apply(decay, float(sample_rate), sound_speed_m_s=SOUND_SPEED, frame=256)
    fine = apply(decay, float(sample_rate), sound_speed_m_s=SOUND_SPEED, frame=4096)

    length = min(len(coarse), len(fine))
    bands_coarse = octave_filter(coarse[:length], sample_rate)
    bands_fine = octave_filter(fine[:length], sample_rate)
    energy = 10.0 * np.log10((bands_coarse**2).sum(axis=1) / (bands_fine**2).sum(axis=1))
    assert float(np.max(np.abs(energy))) < 0.2

    decay_coarse = rt60_per_band(coarse[:length], sample_rate)
    decay_fine = rt60_per_band(fine[:length], sample_rate)
    usable = np.isfinite(decay_coarse) & np.isfinite(decay_fine)
    assert float(np.max(np.abs(decay_coarse[usable] / decay_fine[usable] - 1.0))) < 0.02


def _tone_level_db(seconds: float, frame: int, frequency: float = 16000.0) -> np.ndarray:
    """Block level of an absorbed tone, relative to its start, edges dropped."""
    sample_rate, block = 48000.0, 480
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency * times)
    out = apply(tone, sample_rate, sound_speed_m_s=SOUND_SPEED, frame=frame)
    keep = out[int(0.15 * sample_rate) : int((seconds - 0.15) * sample_rate)]
    usable = keep[: (len(keep) // block) * block].reshape(-1, block)
    level = 20.0 * np.log10(np.sqrt((usable**2).mean(axis=1)) + 1e-300)
    return np.asarray(level - level[0])


def test_the_gain_is_exact_wherever_there_is_signal_at_any_duration() -> None:
    """The property, replacing a test that asserted the wrong one.

    An earlier version claimed the default frame was "validated to about one
    second" and required it to drift past that. It does not drift: a slope
    fitted over a longer record reads shallow because the fit runs into the
    frame's leakage floor, which is the measurement meeting the floor and not
    the filter mis-applying the gain. Restricted to where the tone is still
    above -100 dB, the default holds at every duration out to two seconds.
    """
    analytic = -float(attenuation_db_per_m(16000.0, Atmosphere())) * SOUND_SPEED
    block_s = 480 / 48000.0

    for seconds in (1.0, 1.5, 2.0):
        level = _tone_level_db(seconds, 256)
        axis = np.arange(len(level)) * block_s
        above = level > -100.0
        assert above.sum() > 10, seconds
        slope = float(np.polyfit(axis[above], level[above], 1)[0])
        assert abs(slope / analytic - 1.0) < 0.005, seconds


def test_a_longer_frame_buys_dynamic_range_and_nothing_else() -> None:
    """What the frame is actually for, measured rather than assumed.

    Every frame follows the analytic line to -125 dB and then flattens on its
    own leakage floor. Doubling the frame lowers that floor by roughly 20 dB.
    A room response never reaches it, because its top band still holds content
    where a tone has gone.
    """
    analytic = -float(attenuation_db_per_m(16000.0, Atmosphere())) * SOUND_SPEED
    block_s = 480 / 48000.0
    floors = {}
    for frame in (256, 1024):
        level = _tone_level_db(2.5, frame)
        axis = np.arange(len(level)) * block_s
        # Both must be exact while the tone is well above any floor.
        early = np.abs(axis - 0.8).argmin()
        assert abs(level[early] - analytic * axis[early]) < 6.0, frame
        floors[frame] = float(level.min())

    assert floors[1024] < floors[256] - 20.0


def test_absorption_never_adds_energy() -> None:
    rng = np.random.default_rng(7)
    sample_rate = 48000.0
    noise = rng.standard_normal(int(0.2 * sample_rate))
    absorbed = apply(noise, sample_rate, sound_speed_m_s=SOUND_SPEED)
    assert float((absorbed**2).sum()) < float((noise**2).sum())


def test_the_top_of_the_band_loses_more_than_the_bottom() -> None:
    """The shape of the effect, not just its sign."""
    rng = np.random.default_rng(11)
    sample_rate = 48000.0
    samples = int(0.3 * sample_rate)
    times = np.arange(samples) / sample_rate
    low = np.sin(2.0 * np.pi * 500.0 * times) + 0.0 * rng.standard_normal(samples)
    high = np.sin(2.0 * np.pi * 16000.0 * times)

    lost_low = (
        1.0 - (apply(low, sample_rate, sound_speed_m_s=SOUND_SPEED) ** 2).sum() / (low**2).sum()
    )
    lost_high = (
        1.0 - (apply(high, sample_rate, sound_speed_m_s=SOUND_SPEED) ** 2).sum() / (high**2).sum()
    )
    assert lost_high > 10.0 * lost_low


def test_shape_is_preserved_for_one_and_many_receivers() -> None:
    sample_rate = 48000.0
    single = np.zeros(1000)
    single[0] = 1.0
    many = np.tile(single, (4, 1))

    assert apply(single, sample_rate, sound_speed_m_s=SOUND_SPEED).shape == (1000,)
    assert apply(many, sample_rate, sound_speed_m_s=SOUND_SPEED).shape == (4, 1000)


def test_receivers_are_absorbed_independently_and_identically() -> None:
    """A batch must give exactly what one at a time gives."""
    rng = np.random.default_rng(3)
    sample_rate = 48000.0
    batch = rng.standard_normal((3, 4096))

    together = apply(batch, sample_rate, sound_speed_m_s=SOUND_SPEED)
    apart = np.stack([apply(row, sample_rate, sound_speed_m_s=SOUND_SPEED) for row in batch])
    assert np.allclose(together, apart)


def test_atmosphere_records_what_a_report_must_declare() -> None:
    record = Atmosphere(temperature_c=18.0, relative_humidity_pct=35.0).record()
    assert record["relative_humidity_pct"] == 35.0
    assert record["temperature_c"] == 18.0
    assert record["standard"] == "ISO 9613-1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_humidity_pct": -1.0},
        {"relative_humidity_pct": 101.0},
        {"pressure_kpa": 0.0},
        {"temperature_c": -300.0},
    ],
)
def test_an_impossible_atmosphere_is_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Atmosphere(**kwargs)


def test_a_missing_sound_speed_cannot_be_defaulted() -> None:
    """It scales every attenuation, so it is keyword-only and required."""
    with pytest.raises(TypeError):
        apply(np.zeros(64), 48000.0)  # type: ignore[call-arg]


def test_a_negative_sound_speed_is_refused() -> None:
    with pytest.raises(ValueError):
        gain(1000.0, 1.0, sound_speed_m_s=-343.0)


@pytest.mark.parametrize("frame", [7, 9, 4])
def test_an_unusable_frame_length_is_refused(frame: int) -> None:
    with pytest.raises(ValueError):
        apply(np.zeros(1024), 48000.0, sound_speed_m_s=SOUND_SPEED, frame=frame)
