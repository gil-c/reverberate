"""The renderer against pulses whose time, direction and level are known.

A single direct path must come out as one band limited pulse at its delay,
from its direction, at its gain; two paths with different per band gains
must read those gains through the criteria's own detector; and the interim
tail must decay at Eyring's time and carry Barron's energy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from reverberate.accel.backend import cuda_available
from reverberate.metrics import band_centres, rt60_per_band
from reverberate.mirror.criteria import CriteriaSettings, detect_reflections
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, paths_for
from reverberate.mirror.render import (
    RenderSettings,
    barron_reflected_ratio,
    eyring_t60_s,
    render,
    render_paths,
    storey_volume_m3,
)
from reverberate.spatial.validate import direct_arrival_sample, direction_of_arrival
from test_mirror_ism import RECEIVER, SOURCE, box_scene

C = 343.2


def one_path(length_m: float, direction: np.ndarray, gains: np.ndarray, order: int = 0) -> Paths:
    return Paths(
        receiver=RECEIVER,
        image=np.array([0], dtype=np.int32),
        order=np.array([order], dtype=np.int32),
        length_m=np.array([length_m]),
        direction=direction[None, :] / np.linalg.norm(direction),
        gain=gains[None, :],
        points=np.zeros((1, 5, 3)),
        sequence=np.full((1, 3), -1, dtype=np.int32),
    )


def test_a_single_path_is_a_pulse_at_its_delay_from_its_direction() -> None:
    settings = RenderSettings(order=3, duration_s=0.2)
    direction = np.array([0.0, 0.0, -1.0])  # scene frame: towards -z is ambisonic +y, the left
    paths = one_path(3.432, direction, np.full(7, 0.5))
    early = render_paths(paths, settings, C)
    assert direct_arrival_sample(early) == pytest.approx(0.010 * 48000, abs=2)
    doa = direction_of_arrival(early)
    assert doa.azimuth_deg == pytest.approx(90.0, abs=1.0)
    assert doa.elevation_deg == pytest.approx(0.0, abs=1.0)


def test_per_band_gains_are_read_back_by_the_detector() -> None:
    settings = RenderSettings(order=7, duration_s=0.3)
    direct = one_path(2.0, np.array([1.0, 0.0, 0.0]), np.full(7, 0.5))
    # A reflection 3 ms later, from above, 6 dB down at low bands and 12 dB down at high.
    gains = 0.5 * np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
    both = Paths(
        receiver=RECEIVER,
        image=np.array([0, 1], dtype=np.int32),
        order=np.array([0, 1], dtype=np.int32),
        length_m=np.array([2.0, 2.0 + 0.003 * C]),
        direction=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        gain=np.vstack([direct.gain[0], gains]),
        points=np.zeros((2, 5, 3)),
        sequence=np.full((2, 3), -1, dtype=np.int32),
    )
    found = detect_reflections(render_paths(both, settings, C), CriteriaSettings())
    assert len(found) == 2
    reflection = found[1]
    assert reflection.time_s == pytest.approx(0.003, abs=5e-5)
    assert reflection.elevation_deg == pytest.approx(90.0, abs=5.0)
    levels = dict(zip(reflection.bands.centres_hz, reflection.bands.levels_db, strict=True))
    assert levels[500] == pytest.approx(-6.0, abs=0.7)
    assert levels[4000] == pytest.approx(-12.0, abs=0.7)


def test_the_interim_tail_decays_at_eyring_s_time_with_barron_s_energy() -> None:
    scene = box_scene(alpha=0.3)
    volume = storey_volume_m3(scene)
    assert volume == pytest.approx(4.0 * 3.0 * 2.5)
    t60 = eyring_t60_s(scene, volume, C)
    # Uniform absorption 0.3 on 59 m2 and 30 m3: Eyring gives about 0.23 s.
    assert np.allclose(t60, 0.161 * volume / (-59.0 * np.log(0.7)), rtol=0.05)
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=1))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=1))
    settings = RenderSettings(order=1, duration_s=1.0, tail_from_s=0.015)
    response, record = render(paths, scene, settings, sound_speed_m_s=C, seed=1)
    assert record["tail"]["kind"].startswith("statistical")
    measured = rt60_per_band(response.signals[0], 48000)
    assert np.allclose(measured[2:6], t60[2:6], rtol=0.25)
    ratio = barron_reflected_ratio(1.65, t60, volume)
    assert np.all(ratio > 0.5)
    dry, _ = render(paths, scene, settings, sound_speed_m_s=C, with_tail=False)
    assert np.sum(response.signals[0] ** 2) > 1.5 * np.sum(dry.signals[0] ** 2)


def test_rendering_is_deterministic_for_a_seed() -> None:
    scene = box_scene()
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=1))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=1))
    settings = RenderSettings(order=1, duration_s=0.3)
    a, _ = render(paths, scene, settings, sound_speed_m_s=C, seed=3)
    b, _ = render(paths, scene, settings, sound_speed_m_s=C, seed=3)
    np.testing.assert_array_equal(a.signals, b.signals)


def test_the_histogram_tail_reads_through_the_bank_at_the_energy_asked() -> None:
    """A flat histogram tail, read back through the bank the criteria use, per band."""
    from reverberate.metrics import band_centres, octave_filter_rows
    from reverberate.mirror.rays import Histogram
    from reverberate.mirror.render import tail_from_histogram

    rate = 48000.0
    bins, bands, channels = 300, 7, 16
    energy = np.zeros((1, bins, bands))
    energy[0, 50:250] = 1e-4
    moments = np.zeros((1, bins, bands, channels))
    moments[0, :, :, 0] = energy[0]
    histogram = Histogram(
        energy=energy,
        moments=moments,
        hits=np.ones((1, bins), dtype=np.int64),
        bin_s=0.002,
        bands_hz=(125, 250, 500, 1000, 2000, 4000, 8000),
        order=3,
        rays=1,
    )
    centres = band_centres(int(rate))
    settings = RenderSettings(order=3, duration_s=0.6, tail_from_s=0.0)
    scale = np.ones(len(centres))
    tail, _ = tail_from_histogram(
        histogram,
        0,
        np.zeros(len(centres)),
        settings,
        sound_speed_m_s=343.2,
        start_s=0.0,
        seed=3,
        scale_per_band=scale,
    )
    read = octave_filter_rows(
        np.repeat(tail[0:1], len(centres), 0), int(rate), np.arange(len(centres))
    )
    got = np.sum(read**2, axis=1)
    asked = 200 * 1e-4
    # Every band from 250 Hz up reads what was asked within half a decibel.
    np.testing.assert_allclose(10 * np.log10(got[1:7] / asked), 0.0, atol=0.5)


gpu = pytest.mark.skipif(not cuda_available(), reason="needs a CUDA device and cupy")


@gpu
def test_the_card_renders_the_early_part_as_the_host_and_the_tail_at_its_energy() -> None:
    import cupy

    from reverberate.mirror.rays import Histogram
    from reverberate.mirror.render import early_signals, tail_from_histogram

    scene = box_scene()
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=2))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=2))
    settings = RenderSettings(order=3, duration_s=0.6, tail_from_s=0.0)
    host = early_signals(paths, settings, C)
    card = cupy.asnumpy(early_signals(paths, settings, C, cupy))
    np.testing.assert_allclose(card, host, rtol=0, atol=1e-9 * np.abs(host).max())

    rng = np.random.default_rng(1)
    energy = np.zeros((1, 300, 7))
    energy[0, 50:250] = 1e-4 * rng.uniform(0.5, 1.5, size=(200, 7))
    moments = rng.normal(size=(1, 300, 7, 16)) * energy[..., None] * 0.2
    moments[0, :, :, 0] = energy[0]
    histogram = Histogram(
        energy=energy,
        moments=moments,
        hits=np.ones((1, 300), dtype=np.int64),
        bin_s=0.002,
        bands_hz=(125, 250, 500, 1000, 2000, 4000, 8000),
        order=3,
        rays=1,
    )
    bands = len(band_centres(48000))
    common: dict[str, Any] = dict(
        sound_speed_m_s=C, start_s=0.0, seed=3, scale_per_band=np.ones(bands)
    )
    on_host, _ = tail_from_histogram(histogram, 0, np.zeros(bands), settings, **common)
    on_card, _ = tail_from_histogram(histogram, 0, np.zeros(bands), settings, xp=cupy, **common)
    on_card = cupy.asnumpy(on_card)
    # The draws differ, the law does not: the omni energy agrees within 0.3 dB, the first
    # order ones (where a few bursts' cross terms still show) within 1.5 dB.
    ratio = 10 * np.log10(np.sum(on_card**2, axis=1) / np.sum(on_host**2, axis=1))
    assert abs(ratio[0]) < 0.3
    np.testing.assert_allclose(ratio[1:4], 0.0, atol=1.5)


def test_smooth_over_time_keeps_the_mean_and_the_slope() -> None:
    """The moving mean removes the noise, keeps the total, and leaves the early bins."""
    from reverberate.mirror.render import smooth_over_time

    rng = np.random.default_rng(7)
    bins = 400
    decay = np.exp(-np.arange(bins) / 80.0)
    noisy = decay * rng.lognormal(0.0, 0.8, size=bins)
    values = noisy[:, None]

    flat = smooth_over_time(values, 10, first=50)[:, 0]
    # What comes before the window's first bin is untouched.
    np.testing.assert_allclose(flat[:50], noisy[:50], rtol=0, atol=0)
    # The smoothed part holds the same energy to within the window's own edges.
    assert abs(flat[60:-20].sum() / noisy[60:-20].sum() - 1.0) < 0.05

    # ... and fluctuates far less about the decay.
    def spread(x: np.ndarray) -> float:
        lg = np.log10(np.maximum(x[60:350], 1e-30))
        t = np.arange(lg.size)
        fit = np.polyval(np.polyfit(t, lg, 1), t)
        return float(10 * (lg - fit).std())

    assert spread(flat) < 0.45 * spread(noisy)

    grown = smooth_over_time(values, 100, first=0, fraction=0.1)[:, 0]
    # A window that grows leaves the first bins alone and averages the last hard.
    np.testing.assert_allclose(grown[:5], noisy[:5], rtol=1e-12, atol=0)
    assert spread(grown) < spread(noisy)
    early, late = noisy[:40], grown[:40]
    assert abs(np.log10(late.sum() / early.sum())) < 0.05
