"""Tests for the path from what the engine wrote to what a person hears.

Each test constrains one of the four steps the module exists for, and each of
those steps is one this project has already got wrong somewhere: taking a single
interpolation node for a receiver, forgetting to integrate a differentiated
source, listening through the dispersive top of the grid's band, and resampling
before filtering.

Synthetic signals only, offline, well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate import metrics
from reverberate.audio import (
    DECIBELS_PER_NEPER,
    Atmosphere,
    air_absorption_np_per_m,
    apply_air_absorption,
    convolve,
    frame_for,
    integrate_and_lowcut,
    lowpass,
    peak_gain,
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


def test_the_iso_coefficients_match_the_published_table() -> None:
    """20 C and 50 per cent relative humidity, the figures the roadmap quotes."""
    frequency = np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
    decibels = air_absorption_np_per_m(frequency) * 8.686
    assert np.allclose(decibels, [0.0047, 0.0099, 0.0297, 0.1053, 0.3645], atol=5e-5)


def test_humidity_moves_the_top_octave_by_a_factor_of_nearly_two() -> None:
    """It is a declared parameter of a response, not a detail."""
    top = np.array([16000.0])
    dry = air_absorption_np_per_m(top, humidity_percent=30.0)[0]
    damp = air_absorption_np_per_m(top, humidity_percent=80.0)[0]
    assert dry / damp == pytest.approx(1.85, rel=0.05)


def test_a_delayed_impulse_comes_back_with_exactly_its_own_air_gain() -> None:
    """The claim the filter makes: a sample at t has travelled c t, so its gain is known."""
    rate, samples, arrival = 48000.0, 32768, 0.05
    impulse = np.zeros((1, samples))
    impulse[0, int(arrival * rate)] = 1.0
    filtered = apply_air_absorption(impulse, rate, sound_speed_m_s=343.0)
    frequency = np.fft.rfftfreq(samples, 1.0 / rate)
    expected = np.exp(-air_absorption_np_per_m(frequency) * 343.0 * arrival)
    band = (frequency > 50.0) & (frequency < 20000.0)
    error_db = 20.0 * np.log10(np.abs(np.fft.rfft(filtered[0])[band]) / expected[band])
    assert np.max(np.abs(error_db)) < 0.1


def test_air_absorption_reconstructs_exactly_when_there_is_nothing_to_absorb() -> None:
    """The overlap add has to be transparent, or every response acquires a quiet gain."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2, 4096))
    assert np.allclose(apply_air_absorption(signal, 48000.0, sound_speed_m_s=0.0), signal)


def test_air_absorption_takes_the_expected_bite_out_of_a_measured_decay() -> None:
    """W29's own 0.29 s decay loses 38 per cent of its 16 kHz T60 and none of its 500 Hz."""
    rate = 48000.0
    time = np.arange(int(0.6 * rate)) / rate
    rng = np.random.default_rng(0)
    tail = rng.standard_normal((1, time.size)) * np.exp(-3.0 * np.log(10.0) * time / 0.29)
    before = metrics.rt60_per_band(tail[0], int(rate))
    after = metrics.rt60_per_band(apply_air_absorption(tail, rate)[0], int(rate))
    bands = metrics.band_centres(int(rate))
    change = after / before - 1.0
    assert change[bands.index(500)] == pytest.approx(0.0, abs=0.02)
    assert change[bands.index(16000)] == pytest.approx(-0.377, abs=0.03)


def test_a_frame_that_cannot_overlap_add_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        apply_air_absorption(np.zeros((1, 64)), 48000.0, frame=30)


def test_a_one_dimensional_response_stays_one_dimensional() -> None:
    out = apply_air_absorption(np.zeros(512), 48000.0)
    assert out.ndim == 1


def test_the_atmosphere_writes_itself_down_because_a_run_must_declare_it() -> None:
    """A triple of loose floats does not satisfy the roadmap's requirement."""
    record = Atmosphere(temperature_c=18.0, humidity_percent=35.0).record()
    assert record["humidity_percent"] == 35.0
    assert record["standard"] == "ISO 9613-1"
    assert json.dumps(record)


def test_humidity_changes_the_top_octave_by_the_factor_the_atmosphere_warns_about() -> None:
    top = np.array([16000.0])
    dry = Atmosphere(humidity_percent=30.0).attenuation_np_per_m(top)[0]
    damp = Atmosphere(humidity_percent=80.0).attenuation_np_per_m(top)[0]
    assert dry / damp == pytest.approx(1.85, abs=0.02)
    assert dry * DECIBELS_PER_NEPER == pytest.approx(0.466, abs=0.002)
    assert damp * DECIBELS_PER_NEPER == pytest.approx(0.252, abs=0.002)


def test_the_coefficient_is_in_nepers_so_the_conversion_cannot_go_missing() -> None:
    """Two correct implementations disagreed in the fifth digit on this rounding."""
    assert pytest.approx(20.0 / np.log(10.0)) == DECIBELS_PER_NEPER
    frequency = np.array([4000.0])
    nepers = air_absorption_np_per_m(frequency)[0]
    assert nepers * 8.686 == pytest.approx(0.0297, abs=5e-5)


def test_one_gain_over_several_blocks_keeps_the_level_between_them() -> None:
    """W30's defect: six receivers all written at 0.8913, distance inaudible."""
    loud = np.zeros((2, 16))
    loud[0, 1] = 1.0
    quiet = np.zeros((2, 16))
    quiet[0, 1] = 0.01
    gain = peak_gain(loud, quiet)
    assert np.max(np.abs(loud * gain)) / np.max(np.abs(quiet * gain)) == pytest.approx(100.0)
    assert np.max(np.abs(loud * gain)) == pytest.approx(10.0 ** (-1.0 / 20.0))


def test_peak_gain_survives_silence_and_empty_input() -> None:
    assert peak_gain(np.zeros(8)) == 1.0
    assert peak_gain() == 1.0


def test_a_passed_gain_overrides_the_per_file_one(tmp_path: Path) -> None:
    """What a caller writing a comparable set passes, so no file rescales itself."""
    soundfile = pytest.importorskip("soundfile")
    quiet = np.zeros((1, 32))
    quiet[0, 3] = 0.01
    used = write_wav(tmp_path / "q.wav", quiet, 48000.0, gain=0.5)
    assert used == 0.5
    samples, _ = soundfile.read(str(tmp_path / "q.wav"))
    assert np.max(np.abs(samples)) == pytest.approx(0.005, abs=1e-6)


@pytest.mark.parametrize(("duration", "arrival"), [(0.5, 0.45), (1.0, 0.9)])
def test_a_long_response_keeps_its_air_gain_exact(duration: float, arrival: float) -> None:
    """The defect a 50 ms test could not see, and which a 0.5 s room response has.

    The gain ``exp(-m(f) c t)`` grows steeper in frequency as ``t`` does: at 1 s
    it falls 125 dB across the band. A short frame resolves frequency too
    coarsely for that, and leakage from the loud bottom swamps the quiet top. A
    128 sample frame is 0.008 dB out at 50 ms and 0.79 dB out at 500 ms.
    """
    rate = 48000.0
    samples = int(duration * rate)
    impulse = np.zeros((1, samples))
    impulse[0, int(arrival * rate)] = 1.0
    filtered = apply_air_absorption(impulse, rate, sound_speed_m_s=343.0)
    frequency = np.fft.rfftfreq(samples, 1.0 / rate)
    expected = np.exp(-air_absorption_np_per_m(frequency) * 343.0 * arrival)
    # Only where the answer is above the numerical floor: past 1 s the top of
    # the band is genuinely gone, which is the physics rather than an error.
    band = (frequency > 50.0) & (frequency < 16000.0) & (expected > 1e-6)
    error = 20.0 * np.log10(np.abs(np.fft.rfft(filtered[0])[band]) / expected[band])
    assert np.max(np.abs(error)) < 0.1


def test_the_frame_grows_with_the_response_it_is_given() -> None:
    assert frame_for(0.06) == 256
    assert frame_for(0.5) == 512
    assert frame_for(1.0) == 1024
    assert frame_for(2.0) == 2048
    assert frame_for(0.5) < frame_for(2.0)


def test_a_frame_chosen_by_the_caller_is_not_corrected() -> None:
    """Trusted, because a caller who states one has a reason this cannot know."""
    signal = np.zeros((1, 4096))
    signal[0, 100] = 1.0
    assert apply_air_absorption(signal, 48000.0, frame=128).shape == signal.shape


def test_reconstruction_is_the_summed_normalisation_and_not_the_window() -> None:
    """Corrected after another session measured it: any window pair reconstructs.

    The exactness comes from dividing by the summed product of the analysis and
    synthesis windows. The square root Hann pair is kept because it splits the
    modification evenly between the two, not because it is uniquely exact.
    """
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((1, 8192))
    for frame in (128, 512, 2048):
        reconstructed = apply_air_absorption(signal, 48000.0, sound_speed_m_s=0.0, frame=frame)
        assert np.max(np.abs(reconstructed - signal)) < 1e-12
