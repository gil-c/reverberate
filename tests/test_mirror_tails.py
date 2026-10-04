"""The tail along trajectories: where rays are traced from, what a step reads, what is kept.

Rails are sampled no farther apart than 0.80 m; a source between two samples
reads both with weights linear in arc length and a source on a station reads
it alone; a head reads the two nearest cells it sees; a histogram is traced
once per source position, scene and settings; and the table is the pack's
``tail`` group, whose step is an interpolation in energy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from reverberate.compute import Devices
from reverberate.mirror.moving import prepare
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import RaySettings, trace
from reverberate.mirror.render import band_pulse_energy
from reverberate.mirror.tails import (
    SPACING_M,
    TailCache,
    cell_weights,
    histograms,
    interpolation_error_db,
    source_weights,
    tail_key,
    tail_sites,
    tail_table,
)
from test_mirror_diffract import walled_box

SETTINGS = MirrorSettings(
    rays=RaySettings(rays=60, duration_s=0.06, bin_s=0.002, receiver_radius_m=0.3)
)
STATION = np.array([1.0, 1.2, 0.8])
RAIL = np.array([[0.5, 1.2, 0.5], [0.5, 1.2, 2.0], [1.5, 1.2, 2.0]])
CELLS = np.array([[1.0, 1.2, 1.0], [1.4, 1.2, 1.8], [3.0, 1.2, 1.0], [3.0, 1.2, 1.8]])


def test_rails_are_sampled_evenly_and_a_source_reads_the_samples_either_side() -> None:
    sites = tail_sites(STATION[None, :], [RAIL])
    assert sites.rail.tolist() == [-1, 0, 0, 0, 0, 0]
    gaps = np.diff(sites.arc_m[1:])
    assert np.allclose(gaps, gaps[0]) and gaps[0] <= SPACING_M + 1e-12
    assert sites.arc_m[-1] == 2.5
    np.testing.assert_allclose(sites.positions[1], RAIL[0])
    np.testing.assert_allclose(sites.positions[-1], RAIL[-1])
    # 0.9 m along the rail: between the samples at 0.625 and 1.25 m.
    moving = np.array([0.5, 1.2, 1.4])
    slots, weight = source_weights(np.stack([STATION, sites.positions[3], moving]), sites)
    assert slots.tolist() == [[0, -1], [3, -1], [2, 3]]
    np.testing.assert_allclose(weight, [0.0, 0.0, (0.9 - 0.625) / 0.625])


def test_a_head_reads_the_two_nearest_cells_it_sees() -> None:
    head = np.array([[1.0, 1.2, 1.0], [1.8, 1.2, 1.0], [2.1, 1.2, 1.3]])
    slots, weight = cell_weights(head, CELLS)
    assert slots.tolist() == [[0, -1], [0, 1], [1, 2]]
    np.testing.assert_allclose(weight[1], 0.8 / (0.8 + np.hypot(0.4, 0.8)))
    # With the scene, the cell behind the wall at x = 2 is passed over.
    ms = prepare(walled_box(), SETTINGS)
    slots, weight = cell_weights(head, CELLS, ms)
    assert slots.tolist() == [[0, -1], [0, 1], [2, 3]]
    assert np.all((weight >= 0.0) & (weight <= 0.5))


def test_a_histogram_is_traced_once_per_position_scene_and_settings(tmp_path: Path) -> None:
    catalogue = walled_box()
    cache = TailCache(tmp_path / "tails")
    first = histograms(
        catalogue, SETTINGS, STATION[None, :], CELLS, devices=Devices.host(1), cache=cache
    )
    assert (cache.hits, cache.misses) == (0, 1)
    scene = prepare(catalogue, SETTINGS).scene
    want = trace(scene, STATION, CELLS, SETTINGS.traced_rays())
    np.testing.assert_array_equal(first[0].energy, want.energy)
    # A new cache on the same directory reads it back.
    again = histograms(
        catalogue, SETTINGS, STATION[None, :], CELLS, cache=TailCache(tmp_path / "tails")
    )
    np.testing.assert_array_equal(again[0].energy, first[0].energy)
    np.testing.assert_array_equal(again[0].moments, first[0].moments)
    rays = SETTINGS.traced_rays()
    key = tail_key(scene, STATION, CELLS, rays)
    assert len(key) == 64 and (tmp_path / "tails" / f"{key}.npz").is_file()
    assert key != tail_key(scene, STATION + [0.001, 0, 0], CELLS, rays)
    assert key != tail_key(scene, STATION, CELLS[:3], rays)
    assert key != tail_key(scene, STATION, CELLS, replace(rays, rays=61))
    louder = replace(scene.materials, absorption=scene.materials.absorption * 0.5)
    assert key != tail_key(replace(scene, materials=louder), STATION, CELLS, rays)


def test_the_table_is_the_pack_s_and_a_step_is_a_sum_in_energy() -> None:
    catalogue = walled_box()
    ms = prepare(catalogue, SETTINGS)
    sites = tail_sites(STATION[None, :], [RAIL])
    source = np.array([[0.5, 1.2, 1.4], [0.5, 1.2, 1.4], STATION, STATION])
    listener = np.array([[1.8, 1.2, 1.0], [1.0, 1.2, 1.0], [1.8, 1.2, 1.0], [3.0, 1.2, 1.0]])
    audible = np.array([True, True, True, False])
    cache = TailCache()
    table = tail_table(
        ms,
        SETTINGS,
        source,
        listener,
        sites,
        CELLS,
        audible=audible,
        cache=cache,
        devices=Devices.host(1),
    )
    # Three sites are read (two rail samples and the station), none of the others is traced.
    assert cache.misses == 3
    assert table.hist.shape == (4, 2, 2)
    assert np.all(table.hist[3] == -1)
    assert np.all(table.hist[0] >= 0)
    # A second source slot exactly where the weight is not zero; the same for the cells.
    assert np.all((table.hist[:, 1, 0] >= 0) == (table.position_weight > 0))
    assert np.all((table.hist[:, 0, 1] >= 0) == (table.cell_weight > 0))
    assert table.energy.shape == (table.hist.max() + 1, 30, 7)
    assert table.moments.shape == (*table.energy.shape, 16)
    assert table.scale.shape == (table.energy.shape[0], 8)
    assert len(
        {(tuple(p), int(c)) for p, c in zip(table.hist_position, table.hist_cell, strict=True)}
    ) == len(table.hist_cell)
    rows, share = table.weights(0)
    assert rows.size == 4 and share.sum() == 1.0
    energy, moments = table.at(0)
    np.testing.assert_allclose(energy, np.tensordot(share, table.energy[rows], axes=1))
    assert moments.shape == (30, 7, 16)
    # A cell in view of its source is scaled on its own direct sound: the pulse's energy
    # over what a sphere of the receivers' radius catches of the rays at that distance.
    seen = int(table.hist[2, 0, 0])
    distance = float(np.linalg.norm(CELLS[table.hist_cell[seen]] - STATION))
    caught = 0.3**2 / (4.0 * max(distance, 1.05 * 0.3) ** 2)
    np.testing.assert_allclose(table.scale[seen], band_pulse_energy(48000.0) / distance**2 / caught)
    packed = table.pack()
    assert packed["energy"].dtype == packed["moments"].dtype == np.float32
    assert packed["hist"].dtype == np.int32 and packed["scale"].dtype == np.float64


def test_the_interpolation_error_is_a_level_per_band() -> None:
    low = np.ones((10, 7))
    high = 3.0 * np.ones((10, 7))
    truth = 2.0 * np.ones((10, 7))
    np.testing.assert_allclose(interpolation_error_db(low, high, truth, 0.5), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        interpolation_error_db(low, high, truth, 0.0, from_bin=4), 10 * np.log10(0.5)
    )
