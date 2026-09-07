"""W37's own arithmetic, and the guards that stop money being spent on a defect.

Nothing here runs a solver. What it checks is the part of W37 that is a
calculation: the rehearsal box's timings, which decide whether the window it
solves contains a clean direct sound at all, and the clearance rule, which is
the one PFFDTD does not make and which would otherwise return a plausible field
that is not the one in the room.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.experiments.w37_ambisonic import (
    REHEARSAL,
    Rehearsal,
    array_at,
    cost_record,
    plan_record,
)
from reverberate.spatial.array import design_array
from reverberate.spatial.encode import EncoderSettings
from reverberate.wave.comms import Grid, engine_indices

SOUND_SPEED = 343.2


def a_grid(nodes: int, h: float) -> Grid:
    axis = np.arange(nodes) * h
    return Grid(
        h=h, Ts=h / SOUND_SPEED / np.sqrt(3.0), l2=1 / 3, fcc_flag=0, xv=axis, yv=axis, zv=axis
    )


def write_entry(directory: Path, grid: Grid, boundary: np.ndarray) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(grid.h))
        handle.create_dataset("Ts", data=np.float64(grid.Ts))
        handle.create_dataset("l2", data=np.float64(grid.l2))
        handle.create_dataset("fcc_flag", data=np.int8(grid.fcc_flag))
    with h5py.File(directory / "cart_grid.h5", "w") as handle:
        for name, values in (("xv", grid.xv), ("yv", grid.yv), ("zv", grid.zv)):
            handle.create_dataset(name, data=values)
    with h5py.File(directory / "vox_out.h5", "w") as handle:
        handle.create_dataset("bn_ixyz", data=np.sort(boundary))
    return directory


def test_the_rehearsal_box_gives_the_array_a_clean_direct_sound() -> None:
    """The whole point of the box: every node hears the source before any wall."""
    assert REHEARSAL.direct_last_sample < REHEARSAL.first_reflection_sample
    assert REHEARSAL.first_reflection_sample - REHEARSAL.direct_last_sample > 100
    assert REHEARSAL.samples > REHEARSAL.first_reflection_sample


def test_the_rehearsal_timings_are_the_geometry_and_not_a_guess() -> None:
    box = Rehearsal(half_width_cells=300, source_cells=150, array_radius_m=0.1)
    radius = box.array_radius_cells
    assert box.direct_last_sample == 150 + radius
    assert box.first_reflection_sample == 600 - 150 - radius
    assert box.side_m == pytest.approx(600 * box.step_m)


def test_a_smaller_box_loses_the_clean_window_and_the_numbers_say_so() -> None:
    """The failure this arithmetic exists to catch, made visible rather than solved."""
    cramped = Rehearsal(half_width_cells=120, source_cells=147)
    assert cramped.first_reflection_sample < cramped.direct_last_sample


def test_the_cost_is_stated_before_the_run_and_counts_the_output(tmp_path: Path) -> None:
    """Roadmap constraint: announce the figure before spending, not after."""
    record = cost_record(grid_points=4_610_307_520, samples=24_000, receivers=850)
    assert record["estimated_gpu_s"] > 1000.0
    assert record["sim_outs_bytes"] == 850 * 24_000 * 8


def test_an_array_in_free_air_is_accepted_and_says_how_much_it_checked(
    tmp_path: Path,
) -> None:
    grid = a_grid(400, 0.002)
    centre = np.full(3, 200 * 0.002)
    far_away = np.array([0], dtype=np.int64)
    entry = write_entry(tmp_path / "entry", grid, far_away)
    design, _, clearance = array_at(
        centre,
        entry,
        outer_radius_m=0.05,
        settings=EncoderSettings(fit_order=6),
    )
    assert clearance["boundary_nodes_in_ball"] == 0
    assert clearance["ball_nodes"] > design.count
    assert clearance["ball_radius_m"] == pytest.approx(design.radii.max() + 0.02, abs=5e-5)


def test_a_surface_inside_the_ball_is_refused_even_though_no_receiver_touches_it(
    tmp_path: Path,
) -> None:
    """The rule PFFDTD does not make, and the reason it has to exist here.

    The interior expansion is a solution of the homogeneous Helmholtz equation
    on the whole ball. A boundary inside it makes the model wrong rather than
    noisy, and the fit would return a plausible field that is not the one in
    the room. PFFDTD only refuses a receiver standing *on* a boundary node.
    """
    grid = a_grid(400, 0.002)
    centre = np.full(3, 200 * 0.002)
    nx, ny, nz = grid.shape
    # A node inside the ball but nowhere near a receiver: one cell off centre.
    intruder = engine_indices(np.array([201 * nz * ny + 200 * nz + 200], dtype=np.int64), grid)
    assert not np.isin(intruder, [0]).any()
    entry = write_entry(tmp_path / "entry", grid, intruder)
    with pytest.raises(ValueError, match="boundary nodes lie inside"):
        array_at(centre, entry, outer_radius_m=0.05, settings=EncoderSettings(fit_order=6))


def test_a_ball_reaching_the_edge_of_the_grid_is_refused(tmp_path: Path) -> None:
    grid = a_grid(60, 0.002)
    entry = write_entry(tmp_path / "entry", grid, np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="edge of the grid"):
        array_at(
            np.full(3, 30 * 0.002),
            entry,
            outer_radius_m=0.05,
            settings=EncoderSettings(fit_order=6),
        )


def test_the_plan_records_what_the_array_supports_before_anything_is_solved() -> None:
    """A rental is approved on this table, so it has to exist before the rental."""
    grid = a_grid(400, 0.0020428571428571427)
    design = design_array(np.full(3, 200 * grid.h), grid, fit_order=10)
    settings = EncoderSettings(order=7, fit_order=10)
    record = plan_record(
        design,
        settings,
        sound_speed_m_s=SOUND_SPEED,
        frequencies=np.array([1000.0, 16000.0]),
    )
    assert record["array"]["nodes"] == design.count
    assert record["encoder"]["gate_kr"] == 6.0
    assert record["conditioning"]["effective_order"] == [7, 7]
