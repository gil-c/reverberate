"""The ray tracer twin against what a box must give.

In a box of uniform absorption the histogram's decay must be Eyring's; the
direct ray hits must land in the bin of the direct path at the energy the
sphere's solid angle predicts; a receiver behind a slab must get no direct
hit; the uniform grid must return every triangle a brute force test finds;
and the same seed must give the same histogram to the bit in two processes.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from reverberate.mirror.rays import (
    RaySettings,
    UniformGrid,
    build_grid,
    philox_directions,
    trace,
)
from test_mirror_ism import RECEIVER, SIZE, SOURCE, box_scene

C = 343.2


def test_directions_are_uniform_on_the_sphere_and_keyed_by_ray() -> None:
    directions = philox_directions(4000, seed=1)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(np.abs(directions.mean(axis=0)) < 0.05)
    later = philox_directions(10, seed=1, start=100)
    np.testing.assert_array_equal(later, directions[100:110])


def test_the_grid_finds_every_triangle_a_segment_meets() -> None:
    scene = box_scene()
    grid = build_grid(scene.occluder_vertices, 0.5, scene.bmin, scene.bmax)
    assert isinstance(grid, UniformGrid)
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([4.1, 3.1, 2.6])  # through the far corner, out of the box
    candidates = set(grid.candidates(a, b).tolist())
    # The far corner's three walls are crossed: x = 4, y = 3 and z = 2.5, two
    # triangles each; the near walls are not on the way.
    assert {2, 3, 6, 7, 10, 11} <= candidates
    assert not ({0, 1} & candidates)
    # Every triangle of a box wall lies in the cells the wall spans; a segment
    # along the floor collects the floor's triangles.
    along_floor = set(
        grid.candidates(np.array([0.1, 0.05, 0.1]), np.array([3.9, 0.05, 2.4])).tolist()
    )
    assert {8, 9} <= along_floor


def test_direct_hits_land_in_the_direct_bin_at_the_solid_angle_s_share() -> None:
    scene = box_scene(alpha=0.99)
    settings = RaySettings(rays=3000, duration_s=0.02, bin_s=0.001, receiver_radius_m=0.5, seed=2)
    histogram = trace(scene, SOURCE, RECEIVER[None, :], settings)
    distance = float(np.linalg.norm(RECEIVER - SOURCE))
    # A ray enters the sphere between one radius before the centre and the
    # centre itself, so the direct hits spread over those bins.
    first = int((distance - settings.receiver_radius_m) / C / settings.bin_s)
    last = int(distance / C / settings.bin_s)
    expected = settings.rays * (settings.receiver_radius_m / distance) ** 2 / 4.0
    direct_hits = int(histogram.hits[0, first : last + 1].sum())
    assert direct_hits == pytest.approx(expected, rel=0.3)
    assert histogram.energy[0, first : last + 1, 0].sum() == pytest.approx(
        direct_hits / settings.rays
    )
    assert histogram.hits[0, :first].sum() == 0


def test_the_histogram_decays_at_eyring_s_time() -> None:
    alpha = 0.3
    scene = box_scene(alpha=alpha, scattering=0.5)
    settings = RaySettings(rays=300, duration_s=0.3, bin_s=0.005, receiver_radius_m=0.7, seed=3)
    histogram = trace(scene, SOURCE, RECEIVER[None, :], settings)
    energy = histogram.energy[0, :, 0]
    times = histogram.times_s
    # Fit the decay between 30 ms and 300 ms on a log scale.
    window = (times > 0.03) & (times < 0.25) & (energy > 0)
    slope, _ = np.polyfit(times[window], 10 * np.log10(energy[window]), 1)
    measured_t60 = -60.0 / slope
    volume = float(np.prod(SIZE))
    area = 2 * (SIZE[0] * SIZE[1] + SIZE[1] * SIZE[2] + SIZE[0] * SIZE[2])
    eyring = 24 * np.log(10) * volume / (-C * area * np.log(1 - alpha))
    assert measured_t60 == pytest.approx(eyring, rel=0.2)


def test_a_slab_stops_the_direct_ray_and_the_moments_point_at_the_source() -> None:
    open_box = box_scene(alpha=0.5)
    settings = RaySettings(rays=1500, duration_s=0.012, bin_s=0.001, receiver_radius_m=0.4, seed=4)
    histogram = trace(open_box, SOURCE, RECEIVER[None, :], settings)
    distance = float(np.linalg.norm(RECEIVER - SOURCE))
    at = int((distance - settings.receiver_radius_m) / C / settings.bin_s)
    moments = histogram.moments[0, at : at + 2, 0].sum(axis=0)
    # The first order moments over the omni one point from the receiver to the source.
    towards = (SOURCE - RECEIVER) / distance
    from_scene = np.array([towards[0], -towards[2], towards[1]])  # scene to ambisonic
    read = np.array([moments[3], moments[1], moments[2]]) / (np.sqrt(3.0) * moments[0])
    assert np.allclose(read, from_scene, atol=0.05)
    blocked = box_scene(
        alpha=0.5, obstacle=(np.array([1.75, 0.0, 0.0]), np.array([1.75, 3.0, 2.5]))
    )
    histogram = trace(blocked, SOURCE, RECEIVER[None, :], settings)
    assert histogram.hits[0, at : at + 2].sum() == 0


_SCRIPT = """
import hashlib, sys
sys.path.insert(0, {tests!r})
import numpy as np
from test_mirror_ism import RECEIVER, SOURCE, box_scene
from reverberate.mirror.rays import RaySettings, trace
settings = RaySettings(rays=60, duration_s=0.15, seed=5)
h = trace(box_scene(scattering=0.3), SOURCE, RECEIVER[None, :], settings)
print(hashlib.sha256(h.energy.tobytes() + h.moments.tobytes() + h.hits.tobytes()).hexdigest())
"""


def test_the_same_seed_gives_the_same_histogram_in_two_processes() -> None:
    tests = os.path.dirname(os.path.abspath(__file__))
    digests = []
    for seed in ("7", "4242"):
        out = subprocess.run(
            [sys.executable, "-c", _SCRIPT.format(tests=tests)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]
    assert len(digests[0]) == len(hashlib.sha256().hexdigest())
