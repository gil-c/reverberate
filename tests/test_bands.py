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

from reverberate.bands import DEFAULT_CROSSOVERS_HZ, recombine, split_filters

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


def test_bands_of_different_lengths_are_padded_not_truncated() -> None:
    """Three runs of three windows have three lengths, and the sum keeps the longest."""
    split = split_filters(RATE)
    long_band = np.ones((1, 2000))
    short_band = np.ones((1, 500))

    out = recombine(long_band, short_band, short_band, split)

    assert out.shape[1] == 2000


def test_a_one_dimensional_response_stays_one_dimensional() -> None:
    split = split_filters(RATE)
    single = np.zeros(1000)
    single[0] = 1.0
    assert recombine(single, single, single, split).ndim == 1


def test_mismatched_receiver_counts_are_refused() -> None:
    split = split_filters(RATE)
    with pytest.raises(ValueError, match="receiver counts"):
        recombine(np.zeros((2, 100)), np.zeros((3, 100)), np.zeros((2, 100)), split)


@pytest.mark.parametrize(
    "crossovers", [(4000.0, 1000.0), (0.0, 4000.0), (1000.0, 30000.0), (1000.0, 1000.0)]
)
def test_impossible_crossovers_are_refused(crossovers: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        split_filters(RATE, crossovers)


def test_an_even_tap_count_is_refused() -> None:
    """An even length has a half-sample delay and the complement stops being exact."""
    with pytest.raises(ValueError, match="odd"):
        split_filters(RATE, DEFAULT_CROSSOVERS_HZ, taps=1024)


def test_the_record_says_how_the_bank_was_built() -> None:
    record = split_filters(RATE).record()
    assert record["crossovers_hz"] == [1000.0, 4000.0]
    assert "sum to that impulse exactly" in record["construction"]
    assert record["group_delay_ms"] > 0.0
