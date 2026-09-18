"""The renderer against pulses whose time, direction and level are known.

A single direct path must come out as one band limited pulse at its delay,
from its direction, at its gain; two paths with different per band gains
must read those gains through the criteria's own detector; the histogram's
tail must read back through the bank at the energy asked; a point renders
the same twice for a seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from reverberate.compute import cuda_available
from reverberate.metrics import band_centres
from reverberate.mirror.calibration.criteria import CriteriaSettings
from reverberate.mirror.calibration.early import detect_reflections
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, paths_for
from reverberate.mirror.rays import RaySettings, trace
from reverberate.mirror.render import RenderSettings, early_signals, render_point
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.validate import direct_arrival_sample, direction_of_arrival
from test_mirror_ism import RECEIVER, SOURCE, box_scene

C = 343.2


def render_paths(paths: Paths, settings: RenderSettings, c: float) -> Ambisonic:
    return Ambisonic(
        early_signals(paths, settings, c), settings.sample_rate_hz, settings.order, paths.receiver
    )


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


def test_rendering_is_deterministic_for_a_seed() -> None:
    scene = box_scene()
    tree = grow_tree(scene, SOURCE, IsmSettings(max_order=1))
    paths = paths_for(scene, tree, RECEIVER, IsmSettings(max_order=1))
    settings = RenderSettings(order=1, duration_s=0.3)
    histogram = trace(scene, SOURCE, RECEIVER[None, :], RaySettings(rays=300, duration_s=0.3))

    def once() -> np.ndarray:
        response, record = render_point(
            0,
            paths,
            histogram,
            settings,
            tail_gain_db=np.zeros(7),
            receiver_radius_m=0.2,
            sound_speed_m_s=C,
            seed=3,
        )
        assert isinstance(record["tail"], dict)
        return response.signals

    np.testing.assert_array_equal(once(), once())


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
        settings,
        sound_speed_m_s=343.2,
        start_s=0.0,
        seed=3,
        bursts=settings.tail_bursts,
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
        sound_speed_m_s=C, start_s=0.0, seed=3, bursts=6, scale_per_band=np.ones(bands)
    )
    on_host, _ = tail_from_histogram(histogram, 0, settings, **common)
    on_card, _ = tail_from_histogram(histogram, 0, settings, xp=cupy, **common)
    on_card = cupy.asnumpy(on_card)
    # The draws differ, the law does not: the omni energy agrees within 0.3 dB, the first
    # order ones (where a few bursts' cross terms still show) within 1.5 dB.
    ratio = 10 * np.log10(np.sum(on_card**2, axis=1) / np.sum(on_host**2, axis=1))
    assert abs(ratio[0]) < 0.3
    np.testing.assert_allclose(ratio[1:4], 0.0, atol=1.5)
