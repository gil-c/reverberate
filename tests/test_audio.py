"""Tests for the path from what the engine wrote to what a person hears.

Each test constrains one of the four steps the module exists for, and each of
those steps is one this project has already got wrong somewhere: taking a single
interpolation node for a receiver, forgetting to integrate a differentiated
source, listening through the dispersive top of the grid's band, and resampling
before filtering.

Synthetic signals only, offline, well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reverberate.audio import (
    convolve,
    integrate_and_lowcut,
    lowpass,
    reduce_nodes,
    resample_to,
    write_wav,
)


def test_reduce_nodes_is_the_weighted_sum_not_the_first_node() -> None:
    """Eight nodes make one receiver. ``u_out[0]`` is a node, not a receiver."""
    weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    nodes = np.array(
        [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [10.0, 0.0], [20.0, 0.0], [40.0, 0.0]],
    )
    reduced = reduce_nodes(nodes, weights)
    assert reduced.shape == (2, 2)
    assert reduced[0, 0] == pytest.approx(0.5 * 1 + 0.3 * 2 + 0.2 * 4)
    assert reduced[1, 0] == pytest.approx(0.1 * 10 + 0.1 * 20 + 0.8 * 40)
    assert reduced[0, 0] != nodes[0, 0]


def test_reduce_nodes_preserves_a_constant_field() -> None:
    """Weights sum to 1, so a field equal everywhere must survive untouched."""
    weights = np.full((3, 8), 1 / 8)
    nodes = np.full((24, 16), 2.5)
    assert np.allclose(reduce_nodes(nodes, weights), 2.5)


def test_reduce_nodes_refuses_a_node_count_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match="do not match"):
        reduce_nodes(np.zeros((7, 4)), np.zeros((2, 3)))


def test_integration_recovers_a_differentiated_impulse() -> None:
    """The trap: reading a differentiated run without undoing it.

    A rectangular pulse differentiated becomes a doublet, which is near zero
    between its two spikes. Integrating must put the plateau back. The recovered
    plateau is not flat, because the high pass that stops the integrator
    drifting also droops a 2 ms pulse, and that droop is measured here rather
    than papered over: it is why the cut is at 10 Hz and not higher.
    """
    rate = 48_000.0
    ts = 1.0 / rate
    pulse = np.zeros((1, 4096))
    pulse[0, 500:600] = 1.0
    doublet = np.diff(pulse, axis=-1, prepend=0.0) / ts

    recovered = integrate_and_lowcut(doublet, ts, differentiated=True, fcut=10.0)
    peak = float(np.max(np.abs(recovered)))
    assert peak == pytest.approx(1.0, abs=0.01), "the integrator changed the level"
    assert int(np.argmax(np.abs(recovered[0]))) == 501, "the plateau starts where the pulse did"

    middle = recovered[0, 550] / peak
    assert middle > 0.6, f"the plateau collapsed to {middle:.2f}, the high pass is too aggressive"
    assert abs(doublet[0, 550]) < 1e-9, "the unintegrated doublet is empty here, which is the point"
    assert np.max(np.abs(recovered[0, 700:])) < 0.35, "the plateau did not end"


def test_a_signal_that_was_not_differentiated_is_only_high_passed() -> None:
    rate = 48_000.0
    time = np.arange(4096) / rate
    tone = np.sin(2 * np.pi * 1000 * time)[np.newaxis, :]
    filtered = integrate_and_lowcut(tone, 1.0 / rate, differentiated=False, fcut=10.0)
    settled = slice(1024, None)
    assert np.max(np.abs(filtered[0, settled])) == pytest.approx(1.0, abs=0.02)


def test_the_lowpass_removes_what_is_above_the_grid_band() -> None:
    rate = 48_000.0
    time = np.arange(8192) / rate
    below = np.sin(2 * np.pi * 1000 * time)
    above = np.sin(2 * np.pi * 12_000 * time)
    signals = np.stack([below, above])
    filtered = lowpass(signals, rate, fcut=4000.0)
    settled = slice(2048, -2048)
    assert np.max(np.abs(filtered[0, settled])) == pytest.approx(1.0, abs=0.05)
    assert np.max(np.abs(filtered[1, settled])) < 0.02


def test_the_lowpass_adds_no_group_delay() -> None:
    """Run forwards and backwards, so the timing of an arrival is untouched."""
    rate = 48_000.0
    impulse = np.zeros((1, 4096))
    impulse[0, 2048] = 1.0
    filtered = lowpass(impulse, rate, fcut=4000.0)
    assert int(np.argmax(np.abs(filtered[0]))) == 2048


def test_the_lowpass_refuses_an_odd_order() -> None:
    with pytest.raises(ValueError, match="even"):
        lowpass(np.zeros((1, 16)), 48_000.0, 4000.0, order=7)


def test_resampling_keeps_the_duration_and_the_tone() -> None:
    source_rate = 72_400.0
    time = np.arange(int(source_rate * 0.1)) / source_rate
    tone = np.sin(2 * np.pi * 1000 * time)[np.newaxis, :]
    resampled = resample_to(tone, source_rate, 48_000.0)
    assert resampled.shape[1] == pytest.approx(0.1 * 48_000, rel=0.01)
    settled = slice(2000, -2000)
    assert np.max(np.abs(resampled[0, settled])) == pytest.approx(1.0, abs=0.02)


def test_resampling_to_the_same_rate_copies_rather_than_filters() -> None:
    signals = np.random.default_rng(0).standard_normal((2, 64))
    assert np.array_equal(resample_to(signals, 48_000.0, 48_000.0), signals)


def test_convolution_with_a_delayed_impulse_is_a_delay() -> None:
    dry = np.array([1.0, 2.0, 3.0])
    ir = np.zeros(10)
    ir[4] = 1.0
    wet = convolve(dry, ir)
    assert wet.shape == (12,)
    assert np.allclose(wet[4:7], dry)


def test_convolution_keeps_the_whole_tail() -> None:
    """The tail is what makes separation hard; truncating it hides the point."""
    dry = np.ones(100)
    ir = np.zeros(5000)
    ir[4999] = 1.0
    assert convolve(dry, ir).shape[0] == 100 + 5000 - 1


def test_convolution_refuses_multichannel_input() -> None:
    with pytest.raises(ValueError, match="one dry signal"):
        convolve(np.zeros((2, 8)), np.zeros(8))


def test_one_gain_is_applied_to_every_channel_and_returned(tmp_path: Path) -> None:
    """Per channel normalisation would destroy the relative level between receivers."""
    soundfile = pytest.importorskip("soundfile")
    signals = np.stack([np.ones(64) * 0.5, np.ones(64) * 0.1])
    gain = write_wav(tmp_path / "out.wav", signals, 48_000.0)
    written, rate = soundfile.read(str(tmp_path / "out.wav"), always_2d=True)
    assert rate == 48_000
    assert written[:, 0].max() == pytest.approx(0.5 * gain, abs=1e-6)
    ratio = written[:, 0].max() / written[:, 1].max()
    assert ratio == pytest.approx(5.0, rel=1e-4)


def test_the_written_peak_leaves_the_asked_for_headroom(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    signals = np.array([[0.0, 3.0, -3.0, 0.0]])
    write_wav(tmp_path / "out.wav", signals, 48_000.0, headroom_db=6.0)
    written, _ = soundfile.read(str(tmp_path / "out.wav"), always_2d=True)
    assert np.max(np.abs(written)) == pytest.approx(10 ** (-6.0 / 20.0), abs=1e-6)


def test_a_silent_signal_does_not_divide_by_zero(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    assert write_wav(tmp_path / "silence.wav", np.zeros((2, 32)), 48_000.0) == 1.0
