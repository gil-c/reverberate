"""The renderer against pulses whose time, direction and level are known.

A single direct path must come out as one band limited pulse at its delay,
from its direction, at its gain; two paths with different per band gains
must read those gains through the criteria's own detector; and the interim
tail must decay at Eyring's time and carry Barron's energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.metrics import rt60_per_band
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
