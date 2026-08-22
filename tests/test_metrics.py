"""Tests for the per-band metrics.

These are the instrument every other claim in the project is measured with, so
they are checked against responses whose answer is known by construction: an
exponentially decaying noise burst has a reverberation time you can dial in,
and a band-limited one decays only in its own band.

The point of the module is that a broadband number can hide a large per-band
error, so the tests that matter most are the ones showing a broadband average
staying still while individual bands move.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.acoustics import MIN_WAVELENGTH, OCTAVE_BANDS
from reverberate.metrics import (
    EARLY_WINDOW,
    JND_RT60_RELATIVE,
    clarity_per_band,
    compare,
    critical_distance,
    direct_sound,
    direct_to_reverberant_per_band,
    edt_per_band,
    energy_decay_curve,
    measure,
    octave_filter,
    rt60_per_band,
    schroeder_frequency,
    wavelength_of,
)

FS = 16000


def decaying_noise(rt60: float, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Noise with a known reverberation time, plus a direct-sound spike."""
    rng = np.random.default_rng(seed)
    samples = int(seconds * FS)
    time = np.arange(samples) / FS
    # -60 dB over rt60 seconds is a decay constant of ln(10^6) = 13.8155/2.
    response = rng.normal(size=samples) * np.exp(-6.9078 * time / rt60)
    response[0] += 5.0
    return response


def test_band_range_reaches_8_khz_and_sets_the_physical_floor() -> None:
    """The range and the decimation limit are the same decision, so they are
    checked together: moving the ceiling up an octave halves the floor."""
    assert OCTAVE_BANDS == (125, 250, 500, 1000, 2000, 4000, 8000)
    assert pytest.approx(0.0429, abs=1e-3) == MIN_WAVELENGTH
    assert wavelength_of(8000) == pytest.approx(MIN_WAVELENGTH)


def test_rt60_recovers_a_known_decay_in_every_band() -> None:
    for target in (0.5, 1.2):
        measured = rt60_per_band(decaying_noise(target), FS)

        assert len(measured) == len(OCTAVE_BANDS)
        assert np.all(np.abs(measured - target) / target < 0.15)


def test_a_broadband_average_can_hide_a_per_band_error() -> None:
    """The reason this module exists.

    A response made too reverberant in one band and too dead in another has
    almost the same broadband average as the original, while two bands are
    badly wrong. Only per-band metrics can see it.
    """
    reference = measure(decaying_noise(0.8), FS)
    bands = octave_filter(decaying_noise(0.8), FS)
    time = np.arange(bands.shape[1]) / FS
    # Speed up one band and slow another by the same factor.
    bands[1] *= np.exp(+2.0 * time)
    bands[5] *= np.exp(-2.0 * time)
    candidate = measure(bands.sum(axis=0), FS)

    difference = compare(reference, candidate)

    mean_error = abs(np.nanmean(candidate.rt60) - np.nanmean(reference.rt60)) / np.nanmean(
        reference.rt60
    )
    assert mean_error < np.nanmax(difference.rt60_relative_error)
    assert not difference.rt60_within_jnd


def test_edt_is_computed_and_tracks_the_decay() -> None:
    slow = edt_per_band(decaying_noise(1.5), FS)
    fast = edt_per_band(decaying_noise(0.4), FS)

    assert np.nanmean(slow) > np.nanmean(fast)


def test_decay_curve_starts_at_zero_and_falls_monotonically() -> None:
    curve = energy_decay_curve(decaying_noise(0.8))

    assert curve[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(curve) <= 1e-9)


def test_direct_sound_is_found_at_its_arrival() -> None:
    response = np.zeros(FS)
    response[300] = 1.0
    response[900] = 0.2

    level, arrival = direct_sound(response, FS)

    assert arrival == 300
    assert level == pytest.approx(0.0, abs=1e-6)


def test_clarity_rises_when_late_energy_is_removed() -> None:
    """C50 must respond to where the energy sits, not just how much there is."""
    reverberant = decaying_noise(1.5)
    dry = reverberant.copy()
    dry[int(EARLY_WINDOW * FS) * 2 :] *= 0.01

    assert np.nanmean(clarity_per_band(dry, FS)) > np.nanmean(clarity_per_band(reverberant, FS))


def test_drr_rises_with_a_stronger_direct_path() -> None:
    weak = decaying_noise(1.0)
    strong = weak.copy()
    strong[0] += 50.0

    assert np.nanmean(direct_to_reverberant_per_band(strong, FS)) > np.nanmean(
        direct_to_reverberant_per_band(weak, FS)
    )


def test_comparison_reports_the_worst_band_rather_than_an_average() -> None:
    reference = measure(decaying_noise(0.8), FS)
    candidate = measure(decaying_noise(0.8, seed=1), FS)

    difference = compare(reference, candidate)

    assert difference.worst_band in OCTAVE_BANDS
    assert "worst RT60" in difference.summary()


def test_identical_responses_sit_inside_every_jnd() -> None:
    metrics = measure(decaying_noise(0.9), FS)

    difference = compare(metrics, metrics)

    assert difference.within_jnd
    assert difference.direct_level_error_db == pytest.approx(0.0)


def test_a_single_failing_band_fails_the_whole_comparison() -> None:
    """One audible band is enough: a mean would forgive it, this must not."""
    reference = measure(decaying_noise(0.8), FS)
    candidate = measure(decaying_noise(0.8), FS)
    broken = np.array(candidate.rt60, dtype=float)
    broken[3] *= 1.0 + JND_RT60_RELATIVE * 4.0
    candidate = type(candidate)(**{**candidate.__dict__, "rt60": broken})

    assert not compare(reference, candidate).rt60_within_jnd


def test_schroeder_frequency_marks_where_a_room_stops_being_diffuse() -> None:
    """Below it a large low-band error may be the measurement losing meaning
    rather than the geometry being wrong."""
    small = schroeder_frequency(rt60=0.5, volume=30.0)
    large = schroeder_frequency(rt60=0.5, volume=800.0)

    assert small > large
    assert schroeder_frequency(rt60=0.0, volume=30.0) == float("inf")


def test_critical_distance_grows_with_a_deader_room() -> None:
    assert critical_distance(volume=200.0, rt60=0.4) > critical_distance(volume=200.0, rt60=1.6)


def test_metrics_carry_every_band_and_the_direct_sound() -> None:
    metrics = measure(decaying_noise(0.7), FS)

    assert metrics.bands == OCTAVE_BANDS
    assert metrics.decay_curves.shape[0] == len(OCTAVE_BANDS)
    assert len(metrics.c50) == len(metrics.drr) == len(OCTAVE_BANDS)
    assert metrics.direct_time >= 0.0
    assert "125 Hz" in metrics.summary()


def test_a_silent_response_does_not_pretend_to_have_measurements() -> None:
    """A response too short or too quiet to support a measurement must say so
    rather than return a number that looks like one."""
    silent = np.zeros(FS)

    level, arrival = direct_sound(silent, FS)

    assert level == float("-inf")
    assert arrival == 0
    assert np.all(np.isnan(rt60_per_band(silent, FS)))
