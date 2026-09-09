"""Tests for the three-band split and its recombination.

The property that matters is exact reconstruction. Three filters built by
subtraction sum to a delayed impulse by construction, and a test that checks it
to machine precision is what makes "no bump or notch at a crossover" a fact
rather than an intention. Three separately designed filters could not pass it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from reverberate.bands import (
    level_ratio,
    recombine,
    split_filters,
)

RATE = 48000.0


def test_the_three_filters_sum_to_a_delayed_impulse() -> None:
    """By construction, so it holds to machine precision or the design is wrong."""
    split = split_filters(RATE)
    total = split.low + split.mid + split.high
    impulse = np.zeros(split.taps)
    impulse[split.group_delay] = 1.0
    assert float(np.max(np.abs(total - impulse))) < 1e-12


def test_one_signal_through_all_three_bands_comes_back_unchanged() -> None:
    rng = np.random.default_rng(20250101)
    split = split_filters(RATE)
    signal_in = rng.standard_normal((2, 4000))

    out = recombine(signal_in, signal_in, signal_in, split)

    delay = split.group_delay
    end = 3500
    assert np.allclose(out[:, delay:end], signal_in[:, : end - delay], atol=1e-10)


def test_every_band_carries_the_same_group_delay() -> None:
    """Sample-accurate alignment is what lets three solves be added at all."""
    split = split_filters(RATE)
    assert len(split.low) == len(split.mid) == len(split.high) == split.taps
    assert split.group_delay == (split.taps - 1) // 2


def test_each_filter_passes_its_own_band_and_stops_the_others() -> None:
    split = split_filters(RATE, (1000.0, 4000.0))
    probes = {"low": 200.0, "mid": 2000.0, "high": 10000.0}
    kernels = {"low": split.low, "mid": split.mid, "high": split.high}

    for name, frequency in probes.items():
        _, response = signal.freqz(kernels[name], worN=[frequency], fs=RATE)
        passband = float(np.abs(response[0]))
        assert passband > 0.9, f"{name} should pass {frequency} Hz"
        for other, kernel in kernels.items():
            if other == name:
                continue
            _, blocked = signal.freqz(kernel, worN=[frequency], fs=RATE)
            assert float(np.abs(blocked[0])) < 0.1, f"{other} should stop {frequency} Hz"


def test_level_ratio_recovers_a_gain_it_was_given() -> None:
    """Two solves are not on one scale, and this is how they are put on one."""
    rng = np.random.default_rng(20250101)
    reference = rng.standard_normal((3, 8000))
    subject = reference * 0.25

    ratio = level_ratio(subject, reference, 48000, calibration_hz=(500.0, 2000.0))

    assert ratio == pytest.approx(4.0, rel=0.02)


def test_a_gain_applies_only_to_its_own_band() -> None:
    """Scaling the run that feeds the low band must not move the high one."""
    rng = np.random.default_rng(11)
    split = split_filters(RATE, (1000.0, 4000.0))
    signal_in = rng.standard_normal((1, 8000))

    plain = recombine(signal_in, signal_in, signal_in, split)
    lifted = recombine(signal_in, signal_in, signal_in, split, gains=(2.0, 1.0, 1.0))

    from reverberate.metrics import band_centres, octave_filter

    centres = band_centres(48000)
    before = octave_filter(plain[0], 48000)
    after = octave_filter(lifted[0], 48000)
    low = centres.index(250)
    high = centres.index(8000)
    assert (after[low] ** 2).sum() > 2.0 * (before[low] ** 2).sum()
    assert (after[high] ** 2).sum() == pytest.approx((before[high] ** 2).sum(), rel=0.01)
