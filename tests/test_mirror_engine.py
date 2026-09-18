"""The engine against its twins: the same paths and the same histogram, card or not.

Without a card the engine is the twin, so the fast layer checks that the
device split and the merge give the twin's answer. With a card (the ``gpu``
marker) the kernels are held to the twin: the same set of valid images per
receiver, the hit points to a nanometre, and the histogram counts to the
integer, bin by bin.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.accel.backend import cuda_available
from reverberate.mirror.engine import (
    device_count,
    facet_buckets,
    histogram_on_devices,
    paths_on_devices,
)
from reverberate.mirror.ism import (
    IsmSettings,
    _facet_arrays,
    grow_tree,
    occluder_grid,
    paths_for,
    ray_triangles,
)
from reverberate.mirror.rays import RaySettings, trace
from test_mirror_ism import RECEIVER, SOURCE, box_scene

gpu = pytest.mark.skipif(not cuda_available(), reason="needs a CUDA device and cupy")

RECEIVERS = np.array([RECEIVER, [1.5, 2.2, 0.9], [3.1, 0.7, 1.9]])


def test_without_a_card_the_engine_answers_with_the_twins() -> None:
    scene = box_scene(alpha=0.3, scattering=0.2)
    settings = IsmSettings(max_order=2, flutter_order=3)
    tree = grow_tree(scene, SOURCE, settings)
    if device_count():
        pytest.skip("a card is present; the twin fallback is not what runs")
    found = paths_on_devices(scene, tree, RECEIVERS, settings)
    grid = occluder_grid(scene)
    for receiver, paths in zip(RECEIVERS, found, strict=True):
        twin = paths_for(scene, tree, receiver, settings, grid=grid)
        np.testing.assert_array_equal(paths.image, twin.image)
        np.testing.assert_allclose(paths.points, twin.points)
    rays = RaySettings(rays=200, duration_s=0.1, seed=2)
    histogram = histogram_on_devices(scene, SOURCE, RECEIVERS, rays)
    twin_histogram = trace(scene, SOURCE, RECEIVERS, rays)
    np.testing.assert_array_equal(histogram.hits, twin_histogram.hits)
    np.testing.assert_array_equal(histogram.energy, twin_histogram.energy)


def test_a_facet_bucket_holds_every_triangle_its_probe_can_hit() -> None:
    """The kernel scans the bucket of the crossing; the twin scans the whole facet."""
    scene = box_scene(alpha=0.3, obstacle=(np.array([1.75, 0.5, 0.5]), np.array([1.75, 2.5, 2.0])))
    frame, shape, base, offsets, members = facet_buckets(scene)
    normals, _, _, _ = _facet_arrays(scene)
    rng = np.random.default_rng(3)
    for f, facet in enumerate(scene.facets):
        tris = scene.reflector_vertices[facet.triangles]
        lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
        n = normals[f]
        # Points on the plane over and around the facet, triangle edges included.
        points = lo + (hi - lo) * rng.uniform(-0.1, 1.1, size=(400, 3))
        points -= ((points - tris[0, 0]) @ n)[:, None] * n
        corners = tris.reshape(-1, 3)
        points = np.concatenate([points, corners, 0.5 * (corners[::3] + corners[1::3])])
        a = points - 1e-4 * n
        b = points + 1e-4 * n
        full = ray_triangles(a, b, tris)
        u, v = frame[f, 3:6], frame[f, 6:9]
        iu = np.clip(np.floor((points @ u - frame[f, 0]) / frame[f, 2]), 0, shape[f, 0] - 1)
        iv = np.clip(np.floor((points @ v - frame[f, 1]) / frame[f, 2]), 0, shape[f, 1] - 1)
        bucket = base[f] + iu.astype(int) * shape[f, 1] + iv.astype(int)
        for k in range(points.shape[0]):
            held = members[offsets[bucket[k]] : offsets[bucket[k] + 1]]
            assert set(held) <= set(facet.triangles.tolist())
            found = ray_triangles(a[k : k + 1], b[k : k + 1], scene.reflector_vertices[held])[0]
            assert found == full[k]


@gpu
def test_the_paths_kernel_validates_what_the_twin_validates() -> None:
    scene = box_scene(alpha=0.3, obstacle=(np.array([1.75, 0.5, 0.5]), np.array([1.75, 2.5, 2.0])))
    settings = IsmSettings(max_order=3, flutter_order=5)
    tree = grow_tree(scene, SOURCE, settings)
    grid = occluder_grid(scene)
    found = paths_on_devices(scene, tree, RECEIVERS, settings, grid=grid)
    for receiver, paths in zip(RECEIVERS, found, strict=True):
        twin = paths_for(scene, tree, receiver, settings, grid=grid)
        assert paths.count == twin.count > 10
        np.testing.assert_array_equal(paths.image, twin.image)
        np.testing.assert_allclose(paths.points, twin.points, atol=1e-9)
        np.testing.assert_allclose(paths.gain, twin.gain)


@gpu
def test_the_rays_kernel_counts_what_the_twin_counts() -> None:
    scene = box_scene(alpha=0.4, scattering=0.5)
    settings = RaySettings(rays=2000, duration_s=0.15, bin_s=0.002, receiver_radius_m=0.3, seed=9)
    histogram = histogram_on_devices(scene, SOURCE, RECEIVERS, settings)
    twin = trace(scene, SOURCE, RECEIVERS, settings)
    assert twin.hits.sum() > 100
    np.testing.assert_array_equal(histogram.hits, twin.hits)
    counts = histogram.energy * 2**40
    twin_counts = twin.energy * 2**40
    assert np.max(np.abs(counts - twin_counts)) <= 2.0
    assert np.max(np.abs(histogram.moments - twin.moments) * 2**40) <= 2.0


@gpu
@pytest.mark.parametrize("window_s", [0.0, 0.02])
def test_the_rays_kernel_skips_the_covered_rays_as_the_twin_does(window_s: float) -> None:
    scene = box_scene(alpha=0.4, scattering=0.3)
    settings = RaySettings(
        rays=2000,
        duration_s=0.15,
        bin_s=0.002,
        receiver_radius_m=0.3,
        seed=11,
        skip_specular_order=3,
        skip_window_s=window_s,
    )
    histogram = histogram_on_devices(scene, SOURCE, RECEIVERS, settings)
    twin = trace(scene, SOURCE, RECEIVERS, settings)
    assert twin.hits.sum() > 100
    np.testing.assert_array_equal(histogram.hits, twin.hits)
    assert np.max(np.abs(histogram.energy - twin.energy) * 2**40) <= 2.0
    assert np.max(np.abs(histogram.moments - twin.moments) * 2**40) <= 2.0


@gpu
def test_one_card_and_every_card_give_the_same_field() -> None:
    if device_count() < 2:
        pytest.skip("needs two cards")
    scene = box_scene(alpha=0.3, scattering=0.3)
    settings = IsmSettings(max_order=2)
    tree = grow_tree(scene, SOURCE, settings)
    one = paths_on_devices(scene, tree, RECEIVERS, settings, devices=[0])
    every = paths_on_devices(scene, tree, RECEIVERS, settings)
    for a, b in zip(one, every, strict=True):
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.points, b.points)
    rays = RaySettings(rays=4000, duration_s=0.1, seed=3)
    h1 = histogram_on_devices(scene, SOURCE, RECEIVERS, rays, devices=[0])
    h2 = histogram_on_devices(scene, SOURCE, RECEIVERS, rays)
    np.testing.assert_array_equal(h1.energy, h2.energy)
    np.testing.assert_array_equal(h1.moments, h2.moments)
