"""The batched low band solver on ``numpy``: its arithmetic, its physics and its campaign.

Small grids, no card and no PFFDTD. The schemes are read against their own
dispersion relations and against the wave equation's closed forms (a plane
wave, a monopole in free air, the modes of a rigid box); the three
implementations of a step are read against each other to the bit; and the
campaign is the real one on a lossy shoebox written as a cache entry's files.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from scipy.signal import find_peaks, sosfilt

from reverberate.accel import dsp
from reverberate.accel.encode import encode_point, prepare_band
from reverberate.accel.pairs import KIND, PairsCampaign, encoder_record
from reverberate.spatial.encode import EncoderSettings, numerical_wavenumber
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.trace.engines import BatchedPairs
from reverberate.wave.comms import Grid, engine_indices, interp_weights, nearest_node
from reverberate.wave.lowband.box import TEST_BRANCHES, box_arrays, write_entry
from reverberate.wave.lowband.cli import main
from reverberate.wave.lowband.fit import FitOperator, level_scale
from reverberate.wave.lowband.harness import compare_responses, paper_numbers
from reverberate.wave.lowband.pairs import (
    Item,
    LowbandPairs,
    batch_capacity,
    node_indices,
    pack_batches,
)
from reverberate.wave.lowband.problem import (
    EngineArrays,
    Problem,
    _absorbing,
    build_problem,
    prune_branches,
    read_entry,
)
from reverberate.wave.lowband.reciprocity import solve_counts
from reverberate.wave.lowband.scheme import (
    CARTESIAN,
    FCC,
    Scheme,
    numbers,
    points_per_wavelength,
    velocity_error,
    wavenumber,
)
from reverberate.wave.lowband.solver import (
    Drive,
    NumpyStepper,
    State,
    drive_for,
    reference_step,
    solve,
)

C = 343.2
FMAX = 1500.0
SCHEMES = (CARTESIAN, FCC)


def node(grid: Grid, subs: tuple[int, int, int]) -> int:
    """The engine index of the node at these unrotated subscripts."""
    position = np.array(subs, dtype=float) * grid.h
    found, index = nearest_node(position, grid)
    assert np.allclose(found, position)
    return int(engine_indices(np.array([index]), grid)[0])


def seeds_of(grid: Grid, sources: np.ndarray) -> np.ndarray:
    return np.concatenate([engine_indices(interp_weights(s, grid)[1], grid) for s in sources])


def lossy_room(scheme: Scheme) -> tuple[EngineArrays, Grid, np.ndarray, list[np.ndarray]]:
    """A lossy room that straddles the face centred grid's fold, two sources, three receivers."""
    arrays, grid = box_arrays(scheme, (26, 22, 18), room=((4, 4, 4), (20, 17, 13)), lossy=True)
    sources = np.array([[8.3, 9.1, 7.6], [15.2, 12.4, 9.9]]) * grid.h
    nodes = np.array([node(grid, s) for s in ((12, 11, 9), (18, 6, 6), (6, 15, 11))])
    return arrays, grid, sources, [nodes, nodes]


def silent(batch: int = 1) -> Drive:
    """A drive that injects and records nothing."""
    return Drive(
        batch=batch,
        steps=1,
        inject_index=np.zeros(0, dtype=np.int64),
        inject_source=np.zeros(0, dtype=np.int32),
        inject_signal=np.zeros((0, 1), dtype=np.float32),
        record_index=np.zeros(0, dtype=np.int64),
        record_source=np.zeros(0, dtype=np.int32),
    )


def coordinates(problem: Problem) -> np.ndarray:
    """The place of every stored node in cells, ``[column, z, 3]``; the fold undone."""
    nx, ny, nz = problem.shape
    ix = (problem.columns // ny)[:, None] * np.ones(nz, dtype=np.int64)[None, :]
    iy = (problem.columns % ny)[:, None] * np.ones(nz, dtype=np.int64)[None, :]
    iz = np.arange(nz)[None, :] * np.ones(problem.columns.size, dtype=np.int64)[:, None]
    if problem.scheme.fcc:
        iy = np.where((ix + iy + iz) % 2 == 0, iy, 2 * (ny - 1) - 1 - iy)
    return np.stack([ix, iy, iz], axis=-1).astype(float)


class TestTheSchemesOnPaper:
    def test_each_scheme_holds_one_per_cent_at_the_points_pffdtd_runs_it_at(self) -> None:
        cartesian = velocity_error(CARTESIAN, 10.5)
        fcc = velocity_error(FCC, 7.7)
        assert cartesian["worst"] == pytest.approx(-0.0103, abs=2e-4)
        assert fcc["worst"] == pytest.approx(-0.0101, abs=2e-4)
        # Each is worst along its own direction and nearly exact along the other's.
        assert cartesian["axis"] == cartesian["worst"] and abs(cartesian["body_diagonal"]) < 1e-4
        assert fcc["body_diagonal"] == fcc["worst"] and abs(fcc["axis"]) < 1e-4

    def test_a_budget_gives_the_points_and_the_face_centred_grid_needs_fewer(self) -> None:
        assert points_per_wavelength(CARTESIAN, 0.01) == pytest.approx(10.63, abs=0.02)
        assert points_per_wavelength(FCC, 0.01) == pytest.approx(7.75, abs=0.02)
        assert points_per_wavelength(CARTESIAN, 0.02) == pytest.approx(7.63, abs=0.02)

    def test_the_face_centred_solve_is_twelve_times_fewer_node_updates(self) -> None:
        one = numbers(CARTESIAN, fmax_hz=FMAX, duration_s=1.2, reference_nodes=65.9e6)
        other = numbers(FCC, fmax_hz=FMAX, duration_s=1.2, reference_nodes=65.9e6)
        assert one["steps"] == 32769 and other["steps"] == 13874
        assert one["node_updates"] / other["node_updates"] == pytest.approx(11.98, abs=0.02)

    def test_the_cartesian_wavenumber_is_the_encoder_s(self) -> None:
        frequency = np.array([100.0, 1000.0, 1500.0])
        ours = wavenumber(CARTESIAN, frequency, 0.0218, C, courant=1.0 / np.sqrt(3.0))
        assert np.array_equal(ours, numerical_wavenumber(frequency, 0.0218, C))

    def test_what_a_source_costs_on_each_grid(self) -> None:
        rows = paper_numbers(
            fmax_hz=FMAX,
            duration_s=1.2,
            reference_nodes=65.9e6,
            reached_share=0.724,
            lossy_share=0.0403,
            branches=11,
            rows=9840,
        )
        by_name = {(r["scheme"], r["ppw"]): r for r in rows}
        assert (
            by_name[("cartesian", 10.5)]["bytes_per_source"]
            > 4 * by_name[("fcc", 7.7)]["bytes_per_source"]
        )
        assert all(r["sources_per_card"]["80 GB"] >= r["sources_per_card"]["16 GB"] for r in rows)


class TestAStep:
    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_a_plane_wave_advances_as_the_scheme_s_relation_says(self, scheme: Scheme) -> None:
        """Free field dispersion: one step of the solver is the relation, in any direction."""
        arrays, _ = box_arrays(scheme, (22, 20, 16), room=None, fmax_hz=FMAX)
        problem = build_problem(arrays, None)
        stepper = NumpyStepper(problem, silent())
        where = coordinates(problem)
        air = problem.kind[:-1] == 1
        for direction in ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.36, 0.48, 0.8)):
            unit = np.array(direction) / np.linalg.norm(direction)
            k = 2.0 * np.pi / 8.0 * unit  # eight cells a wavelength, in cells
            omega_t = 2.0 * np.arcsin(np.sqrt(scheme.relation(k, 1.0, arrays.courant)))
            phase = where @ k
            state = State.zeros(problem, 1, np)
            state.u1[: -problem.nz, 0] = np.cos(phase).ravel()
            state.u0[: -problem.nz, 0] = np.cos(phase + omega_t).ravel()
            stepper.step(state, 0, np.zeros((0, 1), dtype=np.float32))
            after = state.u1[: -problem.nz, 0].reshape(phase.shape)
            assert np.abs(after - np.cos(phase - omega_t))[air].max() < 2e-5
            # ... and that relation's speed is within the budget of the true one.
            speed = omega_t / (np.linalg.norm(k) * arrays.courant)
            assert abs(speed - 1.0) < 0.02

    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_a_batch_is_its_sources_solved_alone_bit_for_bit(self, scheme: Scheme) -> None:
        arrays, grid, sources, receivers = lossy_room(scheme)
        problem = build_problem(arrays, seeds_of(grid, sources))
        duration = 150 * grid.Ts
        together = solve(problem, drive_for(problem, grid, sources, receivers, duration), np)
        assert np.abs(together).max() > 0
        for b in range(2):
            alone = solve(
                problem, drive_for(problem, grid, sources[b : b + 1], [receivers[b]], duration), np
            )
            assert np.array_equal(alone, together[3 * b : 3 * b + 3])

    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_the_air_a_source_reaches_gives_what_the_whole_box_gives(self, scheme: Scheme) -> None:
        arrays, grid, sources, receivers = lossy_room(scheme)
        whole = build_problem(arrays, None)
        cut = build_problem(arrays, seeds_of(grid, sources))
        assert cut.reached < 0.3 * whole.reached and cut.column_count < whole.column_count
        # A closed room reaches no halo; this one straddles the fold, whose face is a copy.
        assert len(cut.copies) == (1 if scheme.fcc else 0)
        duration = 150 * grid.Ts
        assert np.array_equal(
            solve(whole, drive_for(whole, grid, sources, receivers, duration), np),
            solve(cut, drive_for(cut, grid, sources, receivers, duration), np),
        )

    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_the_vectorised_step_is_the_kernel_s_arithmetic_node_by_node(
        self, scheme: Scheme
    ) -> None:
        arrays, grid = box_arrays(scheme, (16, 14, 12), room=((3, 3, 3), (12, 10, 8)), lossy=True)
        sources = np.array([[6.3, 6.1, 6.0], [8.2, 6.4, 5.9]]) * grid.h
        nodes = np.array([node(grid, (8, 6, 6))])
        # The whole box, so that the halo's copies and the absorbing layer are stepped too.
        problem = build_problem(arrays, None)
        drive = drive_for(problem, grid, sources, [nodes, nodes], 10 * grid.Ts)
        stepper = NumpyStepper(problem, drive)
        fast, slow = stepper.state(), State.zeros(problem, 2, np)
        out_fast, out_slow = stepper.records(), stepper.records()
        for n in range(drive.steps):
            stepper.step(fast, n, out_fast)
            reference_step(problem, drive, slow, n, out_slow)
        assert np.abs(fast.u1).max() > 0 and np.abs(fast.vh).max() > 0
        for name in ("u0", "u1", "vh", "gh"):
            assert np.array_equal(getattr(fast, name), getattr(slow, name)), name
        assert np.array_equal(out_fast, out_slow)

    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_the_absorbing_layer_is_the_engine_s(self, scheme: Scheme) -> None:
        arrays, _ = box_arrays(scheme, (22, 20, 16), room=None)
        nx, ny, nz = arrays.shape
        nyf = 2 * (ny - 1) if scheme.fcc else ny
        nba = 2 * (nx * nyf + nx * nz + nyf * nz) - 12 * (nx + nyf + nz) + 56
        assert int((_absorbing(arrays.shape, arrays.fcc_flag) > 0).sum()) == (
            nba // 2 if scheme.fcc else nba
        )

    def test_a_source_on_a_wall_is_refused(self) -> None:
        arrays, grid, _, receivers = lossy_room(CARTESIAN)
        problem = build_problem(arrays, None)
        on_the_skin = np.array([[4.5, 9.0, 9.0]]) * grid.h
        with pytest.raises(ValueError, match="not plain air"):
            drive_for(problem, grid, on_the_skin, receivers[:1], 10 * grid.Ts)
        with pytest.raises(ValueError, match="boundary node"):
            build_problem(arrays, seeds_of(grid, on_the_skin))


def spectrum_of(record: np.ndarray, ts: float) -> tuple[np.ndarray, np.ndarray]:
    """A long record's magnitude spectrum with the differentiated drive's slope taken off."""
    length = 16 * record.size
    frequency = np.fft.rfftfreq(length, ts)
    return frequency, np.abs(np.fft.rfft(record * np.blackman(record.size), length)) / np.maximum(
        frequency, 1.0
    )


def box_modes(sizes_m: np.ndarray, below_hz: float) -> np.ndarray:
    found = [
        C / 2.0 * np.sqrt(sum((q / size) ** 2 for q, size in zip(m, sizes_m, strict=True)))
        for m in itertools.product(range(9), repeat=3)
        if any(m)
    ]
    return np.array(sorted(f for f in found if f <= below_hz))


def rigid_box(scheme: Scheme, steps: int) -> tuple[np.ndarray, np.ndarray, Grid]:
    """A rigid room of 20 by 15 by 11 nodes: its response's peaks under ``fmax``."""
    arrays, grid = box_arrays(scheme, (28, 24, 20), room=((4, 4, 4), (23, 18, 14)), fmax_hz=FMAX)
    source = np.array([8.0, 8.0, 8.0]) * grid.h
    problem = build_problem(arrays, seeds_of(grid, source[None]))
    listen = np.array([node(grid, (20 if scheme.fcc else 21, 16, 12))])
    record = solve(problem, drive_for(problem, grid, source[None], [listen], steps * grid.Ts), np)
    assert np.isfinite(record).all()
    late, early = record[0, -500:], record[0, 500:1000]
    assert (late**2).sum() < 2.0 * (early**2).sum(), "a rigid box neither grows nor decays"
    frequency, magnitude = spectrum_of(record[0].astype(float), grid.Ts)
    chosen = (frequency > 100.0) & (frequency <= FMAX)
    peaks, _ = find_peaks(magnitude[chosen], prominence=0.03 * magnitude[chosen].max())
    return frequency[chosen][peaks], np.array([20, 15, 11]) * grid.h, grid


class TestThePhysics:
    def test_a_rigid_box_rings_at_its_modes_on_the_cartesian_grid(self) -> None:
        """Walls half a step beyond the last nodes; every peak within the dispersion budget."""
        peaks, sizes, _ = rigid_box(CARTESIAN, 6000)
        modes = box_modes(sizes, FMAX * 1.02)
        assert peaks.size >= 15
        error = np.array([np.abs(modes - p).min() / p for p in peaks])
        # 1.03 per cent of dispersion at fmax, and a fifth of that for the record's resolution.
        assert error.max() < 0.012
        assert error[0] < 0.002, "the lowest mode, where the grid is fine"
        assert peaks[0] == pytest.approx(C / (2 * sizes[0]), rel=0.002)

    def test_on_the_face_centred_grid_the_walls_slow_what_runs_along_them(self) -> None:
        """What the engine's boundary costs there, measured: modes low by about ``0.8 h / L``.

        A node on a wall loses four of its twelve links, and with them a
        quarter of its stiffness along the wall, while it keeps a whole
        node's mass; the Cartesian node loses one link of six and none along
        the wall. On this box of 0.6 m, a fifth of whose nodes lie on a
        wall, the lowest mode is 4 per cent low; on one twice the size it is
        2 per cent low (measured, not run here). It is PFFDTD's boundary,
        which this solver reproduces, and it is why the face centred grid is
        a choice to be measured on a dwelling and not a default.
        """
        peaks, sizes, _ = rigid_box(FCC, 2600)
        lowest = C / (2 * sizes[0])
        assert 0.94 * lowest < peaks[0] < 0.975 * lowest
        modes = box_modes(sizes, FMAX * 1.02)
        error = np.array([np.abs(modes - p).min() / p for p in peaks])
        assert np.median(error) < 0.02

    @pytest.mark.parametrize(
        ("scheme", "direction", "near", "far"),
        [(CARTESIAN, (1, 0, 0), 6, 16), (FCC, (1, 1, 1), 4, 10)],
        ids=["cartesian", "fcc"],
    )
    def test_a_monopole_in_free_air_has_its_level_and_its_speed(
        self, scheme: Scheme, direction: tuple[int, int, int], near: int, far: int
    ) -> None:
        """``1 / 4 pi r`` at two distances along the scheme's worst direction."""
        size = 42
        arrays, grid = box_arrays(scheme, (size + 2, size, size - 2), room=None, fmax_hz=FMAX)
        centre = np.array([size // 2 + 1, size // 2, size // 2 - 1])
        if scheme.fcc and centre.sum() % 2:
            centre[0] += 1
        along = np.array(direction)
        nodes = np.array([node(grid, tuple(centre + d * along)) for d in (near, far)])
        distance = np.array([near, far]) * np.linalg.norm(along) * grid.h
        problem = build_problem(arrays, None)
        steps = int((distance[1] / C + 4.0 / FMAX) / grid.Ts)
        record = solve(
            problem, drive_for(problem, grid, (centre * grid.h)[None], [nodes], steps * grid.Ts), np
        ).astype(float)
        pressure = sosfilt(dsp.lowcut_sos(1 / grid.Ts, 40.0, 8, differentiated=True), record)
        frequency = np.fft.rfftfreq(8 * steps, grid.Ts)
        spectrum = np.fft.rfft(pressure, 8 * steps, axis=-1)
        band = (frequency >= 0.6 * FMAX) & (frequency <= FMAX)
        # The level of a unit source, on either grid: what one scale for both rests on.
        level = np.abs(spectrum[:, band]).mean(axis=1) * 4.0 * np.pi * distance
        assert 0.85 < level[0] < 1.3 and 0.85 < level[1] < 1.3
        # The speed between the two nodes, over the top of the band: the scheme's, within 1 %.
        k = 2.0 * np.pi * frequency[band] / C
        span = distance[1] - distance[0]
        lag = -np.angle(spectrum[1, band] / spectrum[0, band] * np.exp(1j * k * span))
        measured = float(np.mean(k / (k + lag / span) - 1.0))
        key = "axis" if direction == (1, 0, 0) else "body_diagonal"
        predicted = np.mean(
            [velocity_error(scheme, scheme.ppw, fraction=f)[key] for f in np.linspace(0.6, 1, 9)]
        )
        assert abs(measured - predicted) < 0.01 and -0.02 < measured < 0.01

    @pytest.mark.parametrize("scheme", SCHEMES, ids=lambda s: s.name)
    def test_a_lossy_room_decays(self, scheme: Scheme) -> None:
        arrays, grid, sources, receivers = lossy_room(scheme)
        problem = build_problem(arrays, seeds_of(grid, sources))
        record = solve(
            problem, drive_for(problem, grid, sources[:1], receivers[:1], 900 * grid.Ts), np
        )
        assert np.isfinite(record).all()
        # The drive is a differentiated impulse, which never stops; the response is its integral.
        rate = 1.0 / grid.Ts
        pressure = sosfilt(dsp.lowcut_sos(rate, 40.0, 8, differentiated=True), record.astype(float))
        pressure = dsp.sosfiltfilt(dsp.lowpass_sos(rate, FMAX), pressure, np)
        early, late = (pressure[:, 100:300] ** 2).sum(), (pressure[:, 700:900] ** 2).sum()
        assert late < 0.5 * early


class TestTheGrid:
    def test_a_source_reaches_its_room_and_nothing_outside(self) -> None:
        arrays, grid, sources, _ = lossy_room(CARTESIAN)
        cut = build_problem(arrays, seeds_of(grid, sources))
        room = 17 * 14 * 10
        assert cut.reached == room and cut.updated == room
        assert cut.record["lossy_nodes_reached"] == cut.lossy
        assert cut.bytes_per_source() == 8 * cut.nodes + 8 * cut.lossy * (3 + 1)
        # A node outside the room reads the column of zeros, whatever is solved.
        outside = cut.compact(np.array([node(grid, (2, 2, 2))]))
        assert outside[0] == -1

    def test_dropping_the_high_branches_is_measured_before_it_is_used(self) -> None:
        """They are not small: the branches are fitted together and none stands for a band."""
        kept, record = prune_branches([TEST_BRANCHES], FMAX, above=1.0)
        assert record["branches"] == [2] and kept[0].shape == (2, 3)
        assert record["worst_relative_admittance_change"] > 0.05
        _, whole = prune_branches([TEST_BRANCHES], FMAX, above=100.0)
        assert whole["worst_relative_admittance_change"] == 0.0


# --------------------------------------------------------------------------
# the campaign
# --------------------------------------------------------------------------

KEY = "lowgrid"
BAND_FMAX = 1000.0
DURATION = 0.008
ROOM = ((3, 3, 3), (36, 34, 32))


def write_bundle(bundle: Path, cells: np.ndarray, heard: list[list[int]]) -> None:
    bundle.mkdir(parents=True)
    spec = {
        "kind": KIND,
        "scene_id": "104862621_172226772",
        "dwelling": "hssd_0076",
        "models": "models/storey",
        "model_json": "models/storey/apartment_full.json",
        "storey_scene": "apartment_full",
        "materials": "models/materials",
        "bands": {"low": {"fmax_hz": BAND_FMAX, "duration_s": DURATION, "cache_key": KEY, "nh": 4}},
        "ppw": 10.5,
        "tc": 20.0,
        "rh": 50.0,
        "ram_gb": 64.0,
        "order": 2,
        "fit_order": 3,
        "points": 2,
        "pairs": sum(len(c) for c in heard),
        "solver": "the bundle's",
        "encoder": encoder_record(2, 3, BAND_FMAX),
    }
    (bundle / "campaign.json").write_text(json.dumps(spec))
    np.save(bundle / "sources.npy", SOURCES)
    np.save(bundle / "cells.npy", cells)
    (bundle / "heard_at.json").write_text(json.dumps(heard))


H = C / (BAND_FMAX * 10.5)
SOURCES = np.array([[6.3, 6.1, 5.6], [8.2, 30.4, 6.9], [31.5, 7.5, 27.2]]) * H
CELLS = np.array([[20.0, 19.0, 18.0], [21.0, 18.0, 17.0]]) * H
HEARD_AT = [[0, 1], [1], [0]]


class Driven(LowbandPairs):
    """The real campaign with the grid already voxelised, on ``numpy``."""

    def __post_init__(self) -> None:
        self.gpu = False
        super().__post_init__()

    def voxelise(self) -> dict[str, Any]:
        return {"low": {"cached": True}}


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A data root holding a lossy room as a grid's entry, and a bundle of four pairs."""
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    arrays, grid = box_arrays(CARTESIAN, (40, 38, 36), room=ROOM, fmax_hz=BAND_FMAX, lossy=True)
    write_entry(arrays, grid, tmp_path / "data" / "cache" / "vox" / KEY)
    write_bundle(tmp_path / "bundle", CELLS, HEARD_AT)
    return {"bundle": tmp_path / "bundle", "out": tmp_path / "out", "root": tmp_path}


def a_campaign(machine: dict[str, Path], out: str = "out", **options: Any) -> Driven:
    return Driven(
        bundle=machine["bundle"], out=machine["root"] / out, pffdtd_dir=machine["root"], **options
    )


class DrivenFcc(Driven):
    """The same on a face centred grid of its own key, already voxelised."""

    def grid_key(self) -> str:
        return "fccgrid"


class TestTheCampaign:
    def test_another_grid_is_another_key_and_the_face_centred_one_runs(
        self, machine: dict[str, Path]
    ) -> None:
        arrays, grid = box_arrays(FCC, (40, 38, 36), room=ROOM, fmax_hz=BAND_FMAX, lossy=True)
        write_entry(arrays, grid, machine["root"] / "data" / "cache" / "vox" / "fccgrid")
        np.save(machine["bundle"] / "sources.npy", SOURCES / H * grid.h)
        np.save(machine["bundle"] / "cells.npy", CELLS / H * grid.h)
        campaign = DrivenFcc(
            bundle=machine["bundle"], out=machine["out"], pffdtd_dir=machine["root"], scheme="fcc"
        )
        assert campaign.keys["low"] == "fccgrid" and campaign.cache.voxel_low_key == "fccgrid"
        report = campaign.run()
        assert report["pairs_solved"] == 4 and report["placed"]["refused"] == []
        assert campaign.problem_record["scheme"] == "fcc"
        record = campaign.cache.records()[campaign.key_of(0, 0)]
        assert "fcc at 7.7 points per wavelength" in record["solver"]
        assert record["scale"] == pytest.approx(level_scale(FCC, BAND_FMAX, grid.Ts, 7.7))
        response = campaign.cache.read(campaign.key_of(0, 0))
        assert response.shape == (9, 32) and np.abs(response[0]).max() > 0
        # Every node of an array exists on the face centred lattice.
        design = campaign.designs[0]
        assert design is not None
        assert (np.rint(design.positions / grid.h).sum(axis=1) % 2 == 0).all()

    def test_the_pairs_are_cached_and_a_batch_is_what_one_source_at_a_time_gives(
        self, machine: dict[str, Path]
    ) -> None:
        together = a_campaign(machine)
        report = together.run()
        assert report["pairs_solved"] == 4 and report["pairs_in_cache"] == 4
        assert [r["cells"] for r in report["solves"]] == [2, 1, 1]
        assert len(together.batches) == 1 and together.batches[0]["batch"] == 3
        assert together.problem_record["reached_nodes"] == 34 * 32 * 30
        alone = a_campaign(machine, out="alone", batch=1)
        alone.run()
        assert [b["batch"] for b in alone.batches] == [1, 1, 1]
        for source, cells in enumerate(HEARD_AT):
            for cell in cells:
                key = together.key_of(source, cell)
                response = together.cache.read(key)
                assert response.shape == (9, 32) and response.dtype == np.float32
                assert np.abs(response).max() > 0
                assert np.array_equal(response, alone.cache.read(alone.key_of(source, cell)))
        record = together.cache.records()[together.key_of(0, 0)]
        assert record["scale"] == pytest.approx(BAND_FMAX / 8000.0)
        assert "reverberate.wave.lowband/1 cartesian at 10.5" in record["solver"]
        ledger = together.ledger(1.74)
        assert ledger["solves"] == 3 and ledger["pairs"] == 4 and ledger["usd_per_source"] > 0

    def test_a_pair_s_key_names_the_solver_so_two_engines_never_share_one(
        self, machine: dict[str, Path]
    ) -> None:
        ours = a_campaign(machine)
        theirs = PairsCampaign(
            bundle=machine["bundle"],
            out=machine["root"] / "theirs",
            pffdtd_dir=machine["root"],
            gpu=False,
        )
        assert ours.key_of(0, 0) != theirs.key_of(0, 0)
        assert ours.keys["low"] == theirs.keys["low"] == KEY

    def test_a_second_run_solves_nothing_and_a_new_pair_solves_its_source_alone(
        self, machine: dict[str, Path]
    ) -> None:
        a_campaign(machine).run()
        assert a_campaign(machine).run()["pairs_solved"] == 0
        (machine["bundle"] / "heard_at.json").write_text(json.dumps([[0, 1], [0, 1], [0]]))
        again = a_campaign(machine)
        report = again.run()
        assert [r["source"] for r in report["solves"]] == [1] and report["pairs_in_cache"] == 5

    def test_a_source_heard_at_more_than_a_card_holds_is_solved_in_parts(
        self, machine: dict[str, Path]
    ) -> None:
        campaign = a_campaign(machine)
        campaign.voxelise()
        campaign.place()
        rows = int(campaign.cell_nodes(0).size)
        whole = campaign.items([0], 10 * rows)
        parts = campaign.items([0], rows)
        assert [i.cells for i in whole] == [(0, 1)] and [i.cells for i in parts] == [(0,), (1,)]

    def test_the_fit_as_an_operator_is_the_fit_as_a_solve(self) -> None:
        rng = np.random.default_rng(3)
        offsets = rng.uniform(-0.3, 0.3, size=(60, 3))
        offsets[0] = 0.0
        settings = EncoderSettings(order=2, fit_order=3, max_frequency_hz=BAND_FMAX)
        signals = rng.standard_normal((60, 48))

        def prepared() -> Any:
            return prepare_band(
                offsets,
                grid_step_m=H,
                sound_speed_m_s=C,
                settings=settings,
                samples=48,
                sample_rate_hz=4000.0,
                xp=np,
            )

        solved = encode_point(prepared(), signals, np)
        operator = FitOperator.prepare(prepared(), np)
        assert operator.parts[0].shape[1:] == (9, 60)
        applied = operator.apply(signals, np)
        assert np.abs(applied - solved).max() < 1e-9 * np.abs(solved).max()

    def test_one_scale_for_both_grids(self) -> None:
        assert level_scale(CARTESIAN, 1500.0, 1.0, 10.5) == 1500.0 / 8000.0
        ts = H / C * CARTESIAN.courant
        # Off the bundle's grid the scale is the ratio of the steps, the same figure.
        assert level_scale(CARTESIAN, BAND_FMAX, ts, 10.4) == pytest.approx(BAND_FMAX / 8000.0)
        assert level_scale(FCC, BAND_FMAX, 2 * ts, 7.7) == pytest.approx(BAND_FMAX / 16000.0)
        # The cache's contract: a unit source's impulse, one step of its grid heard as
        # 1 / (4 pi d), reads FIELD_UNIT_AT_1M at 48 kHz on any grid. The monopole test
        # above is what shows a step is that impulse on both grids.
        for scheme, step in ((CARTESIAN, ts), (FCC, 2.3 * ts), (CARTESIAN, 1.4 * ts)):
            unit = level_scale(scheme, BAND_FMAX, step, 9.0) * step * 48000.0 / (4.0 * np.pi)
            assert unit == pytest.approx(FIELD_UNIT_AT_1M, rel=2e-3)

    def test_batches_are_packed_by_the_card_s_memory(self, machine: dict[str, Path]) -> None:
        campaign = a_campaign(machine)
        campaign.voxelise()
        campaign.place()
        problem = build_problem(read_entry(campaign.entry_path), seeds_of(campaign.grid, SOURCES))
        steps = campaign.steps
        items = [Item(s, (0,), 100) for s in range(10)]
        one = problem.bytes_per_source() + 4.0 * steps * 100 + 32.0 * steps
        free = (3.5 * one + problem.bytes_shared()) / 0.8
        assert [len(b) for b in pack_batches(items, problem, steps, free)] == [3, 3, 3, 1]
        assert batch_capacity(problem, steps, free, rows=100) == 3
        assert batch_capacity(problem, steps, 1.0) == 1

    def test_a_cell_s_nodes_are_found_in_the_engine_s_space(self, machine: dict[str, Path]) -> None:
        campaign = a_campaign(machine)
        campaign.voxelise()
        campaign.place()
        design = campaign.designs[0]
        assert design is not None
        one_by_one = [
            int(engine_indices(np.array([nearest_node(p, campaign.grid)[1]]), campaign.grid)[0])
            for p in design.positions[:20]
        ]
        assert node_indices(design.positions[:20], campaign.grid).tolist() == one_by_one
        with pytest.raises(ValueError, match="not a node"):
            node_indices(design.positions[:1] + 0.3 * H, campaign.grid)

    def test_the_trace_s_engine_and_the_command_line(
        self, machine: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(LowbandPairs, "voxelise", Driven.voxelise)
        engine = BatchedPairs(machine["bundle"], machine["out"], gpu=False)
        assert engine.voxel_low_key == KEY and "lowband" in engine.solver
        centres = engine.place()
        assert len(centres) == 2 and centres[0] is not None
        found = engine.solve(HEARD_AT)
        assert found == {"solved": 4, "cached": 0, "source_positions_solved": 3}
        assert engine.cache.has(engine.key_of(1, 1)) and engine.report["batches"]
        code = main(
            [
                "cost",
                "--bundle",
                str(machine["bundle"]),
                "--out",
                str(machine["root"] / "cost"),
                "--cpu",
                "--rate",
                "1.74",
                "--batches",
                "1,2",
                "--steps",
                "20",
                "--cells",
                "1",
            ]
        )
        table = json.loads((machine["root"] / "cost" / "cost.json").read_text())
        assert code == 0 and [b["batch"] for b in table["batches"]] == [1, 2]
        assert table["batches"][0]["usd_per_source"] > 0 and table["pair"]["card_s_per_pair"] > 0


class TestTheMeasurements:
    def test_a_response_against_itself_has_no_error_and_a_scaled_one_its_level(self) -> None:
        rng = np.random.default_rng(1)
        reference = rng.standard_normal((2, 64, 4800)) * np.exp(-np.arange(4800) / 400.0)
        centres = np.zeros((2, 3))
        same = compare_responses(
            reference, reference, reference_centres=[(0.0, centres)], candidate_centres=centres
        )
        assert same["worst_band"]["error_db"] < -200 and same["worst_band"]["level_abs_db"] < 1e-9
        louder = compare_responses(
            reference,
            2.0 * reference,
            reference_centres=[(0.0, centres)],
            candidate_centres=centres,
        )
        assert louder["worst_band"]["level_abs_db"] == pytest.approx(6.02, abs=0.01)
        assert louder["worst_band"]["error_db"] == pytest.approx(0.0, abs=0.01)
        assert len(louder["degree_error_db"]["worst"]) == 8

    def test_which_side_to_solve_is_counted_not_argued(self) -> None:
        # Three positions heard at one cell each and seventy heard at a fourth cell.
        heard = [[0], [1], [2]] + [[3]] * 70
        counts = solve_counts(heard, 4, channels=64)
        assert counts["forward_solves"] == 73 and counts["reciprocal_solves"] == 256
        assert counts["mixed_solves"] == 67 and counts["mixed_reciprocal_cells"] == [3]
        assert counts["mixed_forward_positions"] == 3
        few = solve_counts([[0, 1], [1]], 2, channels=64)
        assert few["mixed_solves"] == few["forward_solves"] == 2


class TestTheCommands:
    def test_the_solver_is_verified_against_itself_without_a_card(self, tmp_path: Path) -> None:
        assert main(["verify", "--out", str(tmp_path), "--cpu", "--steps", "30"]) == 0
        report = json.loads((tmp_path / "verify.json").read_text())
        assert set(report) == {"cartesian", "fcc"}
        assert all(r["cut_equals_whole_on_numpy"] for r in report.values())
        assert report["fcc"]["reached"] < report["cartesian"]["reached"]

    def test_the_numbers_and_the_counts_are_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["numbers"]) == 0
        printed = capsys.readouterr().out
        assert '"scheme": "fcc"' in printed and '"80 GB"' in printed
        heard = tmp_path / "heard.json"
        heard.write_text(json.dumps([[0], [1], [2]] + [[3]] * 70))
        assert main(["counts", "--heard", str(heard)]) == 0
        assert json.loads(capsys.readouterr().out)["mixed_solves"] == 67

    def test_a_field_s_low_side_is_taken_as_a_reference(self, tmp_path: Path) -> None:

        rng = np.random.default_rng(5)
        ir = (rng.standard_normal((2, 64, 57600)) * np.exp(-np.arange(57600) / 4000.0)).astype(
            np.float32
        )
        with h5py.File(tmp_path / "S1.h5", "w") as handle:
            handle.create_dataset("ir", data=ir)
            handle.create_dataset("point_index", data=np.array([0, 2]))
            handle.create_dataset("positions", data=np.zeros((2, 3)))
            handle.attrs["sample_rate_hz"] = 48000.0
            handle.attrs["source_position"] = np.array([1.0, 1.5, 2.0])
        centres = [[0.0, 1.7, 0.1 * i] for i in range(3)]
        points = [[0.0, 1.7, 0.1 * i + 0.01] for i in range(3)]
        plan = {
            "points": points,
            "bands": {"low": {"centres": centres}, "mid": {"centres": centres}},
        }
        (tmp_path / "plan.json").write_text(json.dumps(plan))
        code = main(
            [
                "line",
                "--field",
                str(tmp_path / "S1.h5"),
                "--plan",
                str(tmp_path / "plan.json"),
                "--out",
                str(tmp_path / "L"),
            ]
        )
        assert code == 0
        with np.load(tmp_path / "L" / "line.npz") as held:
            assert held["stored"].shape == (2, 64, 4800) and np.abs(held["stored"]).max() > 0
            assert held["cells"][1].tolist() == points[2]
            assert held["centres_low"][1].tolist() == centres[2]
            # Nothing of the low side lies over the top of the crossover's ramp.
            spectrum = np.abs(np.fft.rfft(held["stored"][0, 0]))
            freqs = np.fft.rfftfreq(4800, 1 / 4000.0)
            assert spectrum[freqs > 1450].max() < 1e-4 * spectrum.max()
        assert np.load(tmp_path / "L" / "sources.npy").tolist() == [[1.0, 1.5, 2.0]]
        assert np.load(tmp_path / "L" / "cells.npy").shape == (2, 3)


class TestTheEstimate:
    def test_a_scene_is_priced_on_the_card_the_solver_was_measured_on(self) -> None:
        from reverberate.trace.plan import estimate as plan_estimate
        from reverberate.wave.lowband.pairs import (
            MEASURED_ON,
            MEASURED_RATE_USD_PER_HOUR,
            ONCE_S,
            PAIR_S,
            SOLVE_S_AT_1500,
            estimate,
        )

        priced = estimate(1646, 16529, fmax_hz=1500.0)
        assert priced["measured_on"] == MEASURED_ON
        assert priced["billed_rate_usd_per_hour"] == MEASURED_RATE_USD_PER_HOUR
        seconds = ONCE_S + 1646 * SOLVE_S_AT_1500 + 16529 * PAIR_S
        assert priced["seconds"] == pytest.approx(seconds, abs=0.1)
        assert priced["usd"] == pytest.approx(seconds / 3600 * MEASURED_RATE_USD_PER_HOUR, abs=0.01)
        # A solve goes as the fourth power of fmax, and several cards share the work.
        assert estimate(10, 0, fmax_hz=750.0)["stencil_s_per_source"] == pytest.approx(
            SOLVE_S_AT_1500 / 16, abs=0.01
        )
        four = estimate(1646, 16529, fmax_hz=1500.0, cards=4)
        assert four["seconds"] == pytest.approx(ONCE_S + (seconds - ONCE_S) / 4, abs=0.1)
        record = {
            "source_positions": 1646,
            "pairs": 16529,
            "tail_sites": 211,
            "tail_cells": 53,
            "step_pairs": 45023,
        }
        ours = plan_estimate(record, rate_usd_per_hour=0.136)
        theirs = plan_estimate(record, rate_usd_per_hour=1.74, low_engine="pffdtd")
        assert ours["measured_on"] == MEASURED_ON and ours["low_engine"] == "lowband"
        assert theirs["measured_on"] == "2 x A100" and theirs["usd"]["low"] > ours["usd"]["low"]
        with pytest.raises(ValueError, match="unknown low band engine"):
            plan_estimate(record, rate_usd_per_hour=1.0, low_engine="other")
