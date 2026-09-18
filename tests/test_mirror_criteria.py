"""The criteria against responses whose answer is planted.

A reflection put into an order 7 response at a known time, direction and
level must come back with those numbers; a tail drawn as a diffuse field
must read flat across orders and sectors; a response judged against itself
must pass every criterion. Synthetic, offline, a few seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from reverberate.mirror.calibration.criteria import (
    Criteria,
    CriteriaSettings,
    beam_weights,
    sector_directions,
)
from reverberate.mirror.calibration.early import (
    detect_reflections,
    echogram,
    echogram_distance_db,
    match_reflections,
)
from reverberate.mirror.calibration.judge import aggregate, judge
from reverberate.mirror.calibration.tail import mixing_time_s, tail_statistics
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count, directions, real_sh

RATE = 48000.0


def planted(
    order: int = 7,
    seconds: float = 0.4,
    reflections: tuple[tuple[float, float, float, float], ...] = (),
    tail_from_s: float = 0.060,
    tail_db: float = -20.0,
    tail_t60_s: float = 0.4,
    tail_direction: tuple[float, float] | None = None,
    seed: int = 0,
    direct_s: float = 0.005,
) -> Ambisonic:
    """A direct sound from the front, ``(time_s, azimuth, elevation, level_db)`` reflections,
    and a decaying tail: diffuse by default, or from one direction."""
    rng = np.random.default_rng(seed)
    samples = int(seconds * RATE)
    signals = np.zeros((channel_count(order), samples))
    front = real_sh(order, directions(np.array(0.0), np.array(0.0))[None, :])[0]
    signals[:, int(direct_s * RATE)] += front
    for time_s, azimuth, elevation, level_db in reflections:
        unit = directions(np.array(np.radians(azimuth)), np.array(np.radians(elevation)))
        signals[:, int(round(time_s * RATE))] += (
            10 ** (level_db / 20.0) * real_sh(order, unit[None, :])[0]
        )
    start = int(tail_from_s * RATE)
    times = np.arange(samples - start) / RATE
    envelope = 10 ** (tail_db / 20.0) * 10 ** (-3.0 * times / tail_t60_s)
    if tail_direction is None:
        noise = rng.standard_normal((signals.shape[0], samples - start))
    else:
        unit = directions(
            np.array(np.radians(tail_direction[0])), np.array(np.radians(tail_direction[1]))
        )
        noise = real_sh(order, unit[None, :])[0][:, None] * rng.standard_normal(samples - start)
    signals[:, start:] += noise * envelope[None, :] / np.sqrt(signals.shape[0])
    return Ambisonic(signals, RATE, order, np.zeros(3))


PLANTED = ((0.0083, 60.0, 10.0, -6.0), (0.0127, -120.0, -20.0, -9.0), (0.0200, 170.0, 40.0, -14.0))


def test_a_beam_answers_one_on_its_axis_and_little_elsewhere() -> None:
    unit = directions(np.array(np.radians(30.0)), np.array(np.radians(10.0)))[None, :]
    weights = beam_weights(7, unit)[0]
    on_axis = float(weights @ real_sh(7, unit)[0])
    assert on_axis == pytest.approx(1.0, abs=1e-9)
    away = directions(np.array(np.radians(30.0 + 60.0)), np.array(np.radians(10.0)))[None, :]
    assert abs(float(weights @ real_sh(7, away)[0])) < 10 ** (-12.0 / 20.0)


def test_planted_reflections_are_found_with_their_time_direction_and_level() -> None:
    found = detect_reflections(planted(reflections=PLANTED))
    assert found[0].index == 0 and found[0].level_db == pytest.approx(0.0, abs=1e-6)
    assert found[0].azimuth_deg == pytest.approx(0.0, abs=2.0)
    assert len(found) == 1 + len(PLANTED)
    for got, (time_s, azimuth, elevation, level_db) in zip(found[1:], PLANTED, strict=True):
        assert got.time_s == pytest.approx(time_s - 0.005, abs=5e-5)
        assert got.azimuth_deg == pytest.approx(azimuth, abs=3.0)
        assert got.elevation_deg == pytest.approx(elevation, abs=3.0)
        assert got.level_db == pytest.approx(level_db, abs=0.5)
        assert np.allclose(got.bands.levels_db, level_db, atol=0.5)


def test_a_reflection_under_the_level_floor_is_neither_counted_nor_expected() -> None:
    quiet = (*PLANTED, (0.030, 20.0, 0.0, -18.0))
    found = detect_reflections(planted(reflections=quiet))
    assert len(found) == 1 + len(PLANTED)


def test_matching_reports_recall_precision_and_the_planted_errors() -> None:
    reference = detect_reflections(planted(reflections=PLANTED))
    moved = (
        (0.0083 + 0.0001, 65.0, 10.0, -6.0),
        (0.0127 - 0.0001, -120.0, -15.0, -9.5),
        (0.0350, -30.0, 0.0, -8.0),  # spurious: nothing at 35 ms in the reference
    )
    candidate = detect_reflections(planted(reflections=moved, seed=1))
    matching = match_reflections(reference, candidate)
    assert matching.recall == pytest.approx(2.0 / 3.0)
    assert matching.precision == pytest.approx(2.0 / 3.0)
    assert matching.unmatched_reference == (3,)
    assert matching.unmatched_candidate == (3,)
    assert np.median(matching.time_errors_s) == pytest.approx(1e-4, abs=3e-5)
    assert np.median(matching.direction_errors_deg) == pytest.approx(5.0, abs=2.0)
    assert 0.2 < float(np.median(matching.level_errors_db)) < 0.8
    record = matching.record()
    assert record["pairs"] == [[1, 1], [2, 2]]


def test_the_mixing_time_is_where_the_response_turns_gaussian() -> None:
    rng = np.random.default_rng(3)
    seconds, mix = 0.3, 0.030
    signal = np.zeros(int(seconds * RATE))
    signal[240] = 1.0
    sparse = rng.integers(300, int(mix * RATE), 12)
    signal[sparse] = 10 ** (-6.0 / 20.0) * rng.choice([-1.0, 1.0], sparse.size)
    start = int(mix * RATE)
    times = np.arange(signal.size - start) / RATE
    decay = 10 ** (-3 * times / 0.5)
    signal[start:] += 10 ** (-9.0 / 20.0) * rng.standard_normal(times.size) * decay
    measured = mixing_time_s(signal, RATE)
    assert 0.020 <= measured <= 0.045


def test_a_diffuse_tail_is_flat_across_orders_and_sectors() -> None:
    report = tail_statistics(planted(order=3, seconds=0.5, tail_from_s=0.020, tail_db=-10.0))
    assert np.all(np.abs(report.order_energy_db) < 0.7)
    assert np.all(np.abs(report.sector_energy_db) < 1.5)
    assert report.late_coherence < report.coherence_floor + 0.15


def test_a_tail_from_one_direction_is_anisotropic_in_that_sector() -> None:
    report = tail_statistics(
        planted(order=3, seconds=0.5, tail_from_s=0.020, tail_db=-10.0, tail_direction=(0.0, 90.0))
    )
    sectors = np.asarray(report.sector_energy_db)
    up = int(np.argmax(sector_directions()[:, 2]))
    assert int(np.argmax(sectors)) in (up, up + 1)
    assert sectors.max() - np.median(sectors) > 6.0


def test_echogram_distance_is_zero_on_itself_and_grows_with_a_moved_reflection() -> None:
    settings = CriteriaSettings()
    a = echogram(planted(order=3, reflections=PLANTED), settings)
    b = echogram(planted(order=3, reflections=PLANTED), settings)
    assert max(echogram_distance_db(a, b)) < 1e-9
    moved = (PLANTED[0], (0.0160, -120.0, -20.0, -9.0), PLANTED[2])
    c = echogram(planted(order=3, reflections=moved), settings)
    assert min(echogram_distance_db(a, c)) > 1.0
    assert a.energy.shape[0] == int(settings.echogram_s / settings.time_bin_s)


def test_a_response_passes_every_criterion_against_itself() -> None:
    response = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0)
    same = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0)
    report = judge(response, same)
    assert all(report.verdicts.values()), report.verdicts
    summary = aggregate([report, report])
    assert summary["points"] == 2
    assert summary["pass_fraction"]["recall"] == 1.0
    json.dumps(report.record())
    assert Criteria().record()["targets"]["recall"] == 0.9


def test_a_wrong_tail_fails_the_tail_criteria_and_not_the_reflections() -> None:
    reference = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0)
    longer = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0, tail_t60_s=0.8)
    report = judge(reference, longer)
    assert report.verdicts["recall"] and report.verdicts["direction_error_deg"]
    assert not report.verdicts["t30_relative"]


_SCRIPT = """
import hashlib, json, sys
sys.path.insert(0, {tests!r})
from test_mirror_criteria import PLANTED, planted
from reverberate.mirror.calibration.judge import judge
a = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0)
b = planted(order=3, reflections=PLANTED, tail_from_s=0.030, tail_db=-12.0, seed=5)
print(hashlib.sha256(json.dumps(judge(a, b).record(), sort_keys=True).encode()).hexdigest())
"""


@pytest.mark.slow
def test_the_judgement_is_the_same_in_two_processes() -> None:
    """Constraint 1: byte identical across processes, not merely within one."""
    tests = os.path.dirname(os.path.abspath(__file__))
    digests = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", _SCRIPT.format(tests=tests)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]
