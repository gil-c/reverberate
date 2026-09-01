"""Tests for W20's wiring: the plan, the cost and the artefacts.

The solver itself is not run here. What is tested is everything that decides
*whether* and *how* it runs, because those are the parts that can be wrong
silently: a cost estimate that says "no need to rent" when there is, a
placement that falls outside the grid the run will actually use, and a
theoretical figure quoted without the correction it was computed with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reverberate.experiments.w20_first_listen import (
    GPU_UPDATES_PER_SECOND,
    LOCAL_UPDATES_PER_SECOND,
    RENTAL_TRIGGER_S,
    _check_inside_grid,
    estimate,
)
from reverberate.experiments.w20_render import (
    NOT_BINAURAL,
    REALISED_ABSORPTION_FACTOR,
    responses_of_run,
    room_geometry,
    theory,
)
from reverberate.geometry.placement import PlacedGroup, Placement


def cache_entry(tmp_path: Path, **overrides: object) -> Path:
    manifest = {
        "grid_points": 75_682_368,
        "sample_rate_hz": 72_818.95,
        "bmin": [-2.4556, -0.0227, -2.4077],
        "bmax": [1.9224, 2.8014, 0.737],
        "h_m": 0.00817142857142857,
    }
    manifest.update(overrides)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def group_at(*positions: tuple[float, float, float]) -> PlacedGroup:
    placements = tuple(
        Placement(position=np.array(p, dtype=float), azimuth=0.0, room="bedroom.001")
        for p in positions
    )
    return PlacedGroup(room="bedroom.001", sources=placements[:1], receivers=placements[1:], seed=0)


def test_the_cost_is_point_updates_and_not_a_guess(tmp_path: Path) -> None:
    """The rental decision is arithmetic that can be checked line by line."""
    cost = estimate(cache_entry(tmp_path), duration_s=1.5, sources=2)

    assert cost.steps == round(1.5 * 72_818.95)
    expected = 75_682_368 * cost.steps * 2
    # The update count covers the whole experiment, as the times do. They
    # disagreed by exactly the source count until this was made a field.
    assert cost.updates == pytest.approx(expected)
    assert cost.record()["point_updates_per_source"] == pytest.approx(expected / 2)
    assert cost.local_s == pytest.approx(expected / LOCAL_UPDATES_PER_SECOND)
    assert cost.gpu_s == pytest.approx(expected / GPU_UPDATES_PER_SECOND)


def test_the_rental_flag_follows_the_agreed_trigger(tmp_path: Path) -> None:
    small = estimate(cache_entry(tmp_path, grid_points=1_000_000), duration_s=0.1, sources=1)
    assert small.local_s < RENTAL_TRIGGER_S
    assert small.record()["would_rent"] is False

    big = estimate(cache_entry(tmp_path, grid_points=75_682_368), 1.5, 2)
    assert big.local_s > RENTAL_TRIGGER_S
    assert big.record()["would_rent"] is True


def test_the_w20_room_is_past_the_trigger_on_the_corrected_throughput(
    tmp_path: Path,
) -> None:
    """The estimate that sent this run local said 636 s; it took about 5880 s.

    The constant it used was read off a column named ``engine_s`` in a sweep
    whose figures cannot have come from this machine's cores. With the
    throughput actually measured on the run, the same room is well past the
    rental trigger, which is the answer the decision should have had.
    """
    cost = estimate(cache_entry(tmp_path, grid_points=75_682_368), 1.5, 2)
    assert cost.local_s > 4 * RENTAL_TRIGGER_S
    assert cost.local_s / 2 == pytest.approx(5880, rel=0.15)


def test_the_cost_record_carries_the_trigger_it_was_judged_against(tmp_path: Path) -> None:
    record = estimate(cache_entry(tmp_path), 1.5, 2).record()
    assert record["rental_trigger_s"] == RENTAL_TRIGGER_S
    assert record["point_updates"] > 0


def test_a_placement_outside_the_voxelised_bounds_is_refused(tmp_path: Path) -> None:
    """The silent failure this guards: interpolation reading the grid edge."""
    entry = cache_entry(tmp_path)
    _check_inside_grid(group_at((0.0, 1.6, -1.0), (0.5, 1.2, -1.0)), entry)

    with pytest.raises(ValueError, match="outside the voxelised bounds"):
        _check_inside_grid(group_at((0.0, 1.6, -1.0), (5.0, 1.2, -1.0)), entry)


def test_the_margin_is_counted_in_cells(tmp_path: Path) -> None:
    """A position one cell inside the boundary is not far enough inside it."""
    entry = cache_entry(tmp_path)
    just_inside = group_at((1.9224 - 0.05, 1.6, -1.0))
    _check_inside_grid(just_inside, entry, margin_cells=2.0)
    with pytest.raises(ValueError, match="outside the voxelised bounds"):
        _check_inside_grid(just_inside, entry, margin_cells=20.0)


def test_theory_applies_the_measured_absorption_factor_and_says_so() -> None:
    """W3 measured a 0.89 shortfall; quoting Sabine without it compares to a
    room this pipeline does not build."""
    result = theory(volume_m3=38.1, surface_area_m2=70.0, mean_absorption=0.20)
    record = result.record()

    assert record["realised_absorption_factor"] == REALISED_ABSORPTION_FACTOR
    assert record["mean_absorption_realised"] == pytest.approx(0.20 * 0.89)
    # Less absorption means a longer tail, so the correction must lengthen it.
    naive = 0.161 * 38.1 / (70.0 * 0.20)
    assert result.sabine_s > naive
    assert result.sabine_s == pytest.approx(0.161 * 38.1 / (70.0 * 0.20 * 0.89))


def test_eyring_is_shorter_than_sabine_at_this_absorption() -> None:
    result = theory(volume_m3=38.1, surface_area_m2=70.0, mean_absorption=0.30)
    assert result.eyring_s < result.sabine_s


def test_the_assumption_travels_with_the_number() -> None:
    record = theory(38.1, 70.0, 0.2).record()
    assert "diffuse" in str(record["assumption"])


def test_the_not_binaural_warning_is_explicit() -> None:
    assert "not a binaural" in NOT_BINAURAL
    assert "pinna" in NOT_BINAURAL


def test_the_signal_path_runs_in_the_only_correct_order(tmp_path: Path) -> None:
    """Resampling before band limiting would alias the dispersive top of the
    grid's band into the audible range. This asserts the outcome of the order,
    on a synthetic run whose content above fmax is known."""
    h5py = pytest.importorskip("h5py")
    grid_rate = 72_818.95
    count = 8192
    time = np.arange(count) / grid_rate
    # One tone inside the band and one well above it, at equal amplitude.
    node = np.sin(2 * np.pi * 1000 * time) + np.sin(2 * np.pi * 20_000 * time)
    u_out = np.repeat(node[np.newaxis, :], 8, axis=0)

    with h5py.File(tmp_path / "sim_outs.h5", "w") as handle:
        handle.create_dataset("u_out", data=u_out)
    with h5py.File(tmp_path / "comms_out.h5", "w") as handle:
        handle.create_dataset("out_alpha", data=np.full((1, 8), 1 / 8))
        handle.create_dataset("diff", data=np.int8(0))
    with h5py.File(tmp_path / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(0.00817))
        handle.create_dataset("SR", data=np.float64(grid_rate))

    ir, rate = responses_of_run(tmp_path, fmax_hz=4000.0)

    assert rate == 48_000.0
    assert ir.shape[0] == 1
    spectrum = np.abs(np.fft.rfft(ir[0, 2000:-2000]))
    freqs = np.fft.rfftfreq(ir[0, 2000:-2000].size, 1 / 48_000.0)
    in_band = spectrum[np.argmin(np.abs(freqs - 1000))]
    # 20 kHz would alias to 20000 - 48000 + 48000 ... it simply must not be there.
    above = spectrum[freqs > 6000].max()
    assert above < in_band / 100, "content above the grid's band survived"


def _fitted_room(tmp_path: Path) -> Path:
    """A two node boundary with one rigid node, and a watertight unit shell."""
    import h5py

    run = tmp_path / "source0"
    run.mkdir()
    with h5py.File(run / "vox_out.h5", "w") as vox:
        vox.create_dataset("h", data=0.5)
        vox.create_dataset("mat_bn", data=np.array([0, 0, -1], dtype=np.int8))
        vox.create_dataset("saf_bn", data=np.array([1.0, 3.0, 8.0], dtype=float))
    # One purely resistive branch, so the absorption is a hand checkable number
    # and the test stays offline: fitting a real class needs PFFDTD's routine.
    triplets = np.array([[0.0, 4.0, 0.0]])
    with h5py.File(run / "sim_mats.h5", "w") as mats:
        mats.create_dataset("Nmat", data=1)
        mats.create_dataset("mat_00_DEF", data=np.asarray(triplets, dtype=float))

    box = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"mats_hash": {"shell": {"pts": box.tolist(), "tris": faces}}}))
    return model


def test_the_surface_is_the_one_the_solver_meets_not_the_one_the_mesh_drew(
    tmp_path: Path,
) -> None:
    model = _fitted_room(tmp_path)
    geometry = room_geometry(tmp_path / "source0", model)
    # Four material weighted cells of 0.25 m2, and eight rigid ones.
    assert geometry.surface_area_m2 == pytest.approx(1.0)
    assert geometry.rigid_area_m2 == pytest.approx(2.0)
    # The shell is a unit cube, so its own area is six and its volume one.
    assert geometry.shell_area_m2 == pytest.approx(6.0)
    assert geometry.volume_m3 == pytest.approx(1.0)


def test_rigid_boundary_is_excluded_from_absorption_but_still_reported(
    tmp_path: Path,
) -> None:
    model = _fitted_room(tmp_path)
    geometry = room_geometry(tmp_path / "source0", model)
    assert 0.0 < geometry.mean_absorption < 1.0
    assert geometry.record()["rigid_area_m2"] == pytest.approx(2.0)
    assert "rigid" in geometry.record()["note"]


def test_a_room_with_no_material_is_an_error_not_a_zero(tmp_path: Path) -> None:
    import h5py

    model = _fitted_room(tmp_path)
    with h5py.File(tmp_path / "source0" / "vox_out.h5", "r+") as vox:
        del vox["mat_bn"]
        vox.create_dataset("mat_bn", data=np.array([-1, -1, -1], dtype=np.int8))
    with pytest.raises(ValueError, match="no boundary node carries a material"):
        room_geometry(tmp_path / "source0", model)
