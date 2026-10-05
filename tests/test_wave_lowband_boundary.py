"""The low band solver's walls and its other waste: fewer branches, no outside air, a fit in spectra

Everything on ``numpy`` and small. The walls' refit is read against the
materials it stands for, the worst third octave; the air outside a
dwelling's outer walls is found in a grid built for it and cut off; the fit
made in its spectra is read against the chain of filters it replaces; and
the order of a queue's launches, the probe of a card and several recipes'
bundles as one are held to what they say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from reverberate.spatial.encode import EncoderSettings
from reverberate.wave.comms import Grid, engine_indices, interp_weights, nearest_node
from reverberate.wave.lowband import walls
from reverberate.wave.lowband.box import air_arrays, box_arrays
from reverberate.wave.lowband.fit import CellEncoder
from reverberate.wave.lowband.hostkernel import compiler, host_kernel
from reverberate.wave.lowband.outside import OPEN_BRANCHES, outside_air, reach_of, without
from reverberate.wave.lowband.pairs import (
    Item,
    last_short,
    memory_rate_of,
    merge_bundles,
    probe,
    solver_name,
)
from reverberate.wave.lowband.problem import build_problem
from reverberate.wave.lowband.scheme import CARTESIAN
from reverberate.wave.lowband.solver import CardStepper, drive_for, solve
from reverberate.wave.lowband.walls import (
    ABSORPTION_BAR,
    REFLECTION_BAR_DB,
    WallFit,
    admittance,
    errors,
    refit,
    refit_kept,
    third_octaves,
    worst_of,
)

#: Two materials of the catalogue as the solver is given them, eleven branches an octave
#: apart from 16 Hz to 16 kHz: the dwelling's shell, which two lossy nodes in three carry,
#: and a couch, the hardest to fit under 1500 Hz with fewer.
SHELL = np.array(
    [
        [0.549716452, 38.1613414, 5298.32414],
        [0.256505228, 35.6132094, 9889.08252],
        [0.074937595, 20.8086851, 11556.319],
        [0.237213671, 131.739071, 146325.318],
        [0.0742660951, 82.4888917, 183244.245],
        [0.0238552606, 52.9930653, 235441.985],
        [0.0107281432, 47.6638846, 423530.119],
        [0.00566877459, 50.3714038, 895177.002],
        [0.00247025914, 43.9002887, 1560350.75],
        [0.00126978315, 45.1319833, 3208257.9],
        [0.000521770467, 37.0906417, 5273260.13],
    ]
)
COUCH = np.array(
    [
        [2.03874611, 141.529849, 19650.0172],
        [2.76647945, 384.098261, 106656.476],
        [0.494575512, 137.333819, 76269.7601],
        [0.116780833, 64.8554461, 72036.2887],
        [0.0213202822, 23.6808795, 52605.6877],
        [0.00522099041, 11.5981246, 51529.1099],
        [0.00151766578, 6.74281139, 59915.0434],
        [0.00109165427, 9.70018428, 172387.133],
        [0.000535916123, 9.5240504, 338513.928],
        [0.000298912448, 10.6242641, 755237.788],
        [0.000116964862, 8.314579, 1182102.44],
    ]
)


class TestTheWallsFittedAgain:
    def test_the_band_s_third_octaves_are_cut_to_the_band(self) -> None:
        thirds = third_octaves(40.0, 1500.0)
        assert [round(c) for c, _, _ in thirds][:3] == [39, 50, 62]
        assert thirds[0][1] == 40.0 and thirds[-1][2] == 1500.0
        assert round(thirds[-1][0]) == 1587 and len(thirds) == 17

    def test_a_material_of_few_enough_branches_is_left_as_it_is(self) -> None:
        assert np.array_equal(refit(SHELL[:5], 7), SHELL[:5])
        with pytest.raises(ValueError, match="one branch"):
            refit(SHELL, 0)

    def test_two_branches_beyond_the_band_fold_into_the_others(self) -> None:
        """The branches resonant at 8 and 16 kHz are a spring under 1500 Hz: nine do for eleven."""
        fitted = refit(SHELL, 9)
        assert fitted.shape == (9, 3) and (fitted > 0).all()
        assert (np.diff(fitted[:, 2] / fitted[:, 0]) > 0).all()
        worst = worst_of(errors(SHELL, fitted))
        assert worst["admittance"] < 1e-3 and worst["reflection_db"] < -80.0
        assert max(worst["normal"], worst["random"]) < 1e-4
        # ... where taking them away without a fit is paid in the band.
        dropped = worst_of(errors(SHELL, SHELL[:9]))
        assert dropped["admittance"] > 20 * worst["admittance"]
        # Passive at every frequency, in the band or not: positive branches cannot be otherwise.
        everywhere = np.geomspace(1.0, 24000.0, 500)
        assert (admittance(fitted, everywhere).real > 0).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["shell", "couch"])
    def test_seven_branches_hold_a_wall_under_the_bars_and_six_do_not(self, name: str) -> None:
        """The table of ``docs/open-questions/solver-boundary.md``, its two rows that decide."""
        material = {"shell": SHELL, "couch": COUCH}[name]
        seven = worst_of(errors(material, refit(material, 7)))
        assert seven["reflection_db"] <= REFLECTION_BAR_DB - 9.0
        assert seven["admittance"] <= 0.007
        assert max(seven["normal"], seven["random"]) <= 0.25 * ABSORPTION_BAR
        six = worst_of(errors(material, refit(material, 6)))
        assert six["random"] > ABSORPTION_BAR
        assert six["reflection_db"] > REFLECTION_BAR_DB - 1.0

    def test_a_fit_is_kept_and_read_again_only_for_its_own_materials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kept = tmp_path / "state" / "walls.json"
        fit = WallFit(10)
        assert "10 branches fitted from 40 to 1500 Hz" in fit.name()
        calls = []

        def fitted(materials: list[np.ndarray], fit: WallFit) -> tuple[list[np.ndarray], Any]:
            calls.append(fit.branches)
            return [materials[0][: fit.branches] / 3.0], {"fit": fit.name()}

        monkeypatch.setattr(walls, "refit_all", fitted)
        first, record = refit_kept([SHELL], fit, kept)
        assert record["fit"] == fit.name() and len(calls) == 1
        again, _ = refit_kept([SHELL], fit, kept)
        # Read back, to the last bit of every number: the workers of a campaign hold one fit.
        assert len(calls) == 1 and np.array_equal(again[0], first[0])
        refit_kept([COUCH], fit, kept)
        refit_kept([COUCH], WallFit(9), kept)
        assert len(calls) == 3
        assert json.loads(kept.read_text())["fit"] == WallFit(9).name()

    def test_an_option_that_changes_a_response_changes_the_solver_s_name(self) -> None:
        plain = solver_name(CARTESIAN, 10.5)
        assert plain == "reverberate.wave.lowband/1 cartesian at 10.5 points per wavelength"
        names = {
            plain,
            solver_name(CARTESIAN, 10.5, walls=7),
            solver_name(CARTESIAN, 10.5, walls=6),
            solver_name(CARTESIAN, 10.5, outside="open"),
            solver_name(CARTESIAN, 10.5, outside="rigid"),
            solver_name(CARTESIAN, 10.5, fit="spectra"),
            solver_name(CARTESIAN, 10.5, walls=7, outside="open", fit="spectra"),
        }
        assert len(names) == 7 and all(name.startswith(plain) for name in names)
        with pytest.raises(ValueError, match="closed as"):
            solver_name(CARTESIAN, 10.5, outside="ajar")
        with pytest.raises(ValueError, match="a fit is"):
            solver_name(CARTESIAN, 10.5, fit="guess")


# --------------------------------------------------------------------------
# the air outside the outer walls
# --------------------------------------------------------------------------

SHAPE = (40, 34, 22)


def dwelling(opening: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A room in a shell: the room's air, the ring's between walls and shell, and the opening's.

    The shell holds nodes 3 to 36, 3 to 30 and 3 to 18. The ring is three
    nodes wide on four sides, the walls two nodes thick, and the opening is
    cut through the wall of low ``x`` from floor to ceiling.
    """
    inside = np.zeros(SHAPE, dtype=bool)
    inside[3:37, 3:31, 3:19] = True
    room = np.zeros(SHAPE, dtype=bool)
    room[8:32, 8:26, 3:19] = True
    ring = inside.copy()
    ring[6:34, 6:28, :] = False
    hole = np.zeros(SHAPE, dtype=bool)
    if opening:
        hole[6:8, 14:19, 3:19] = True
    return room, ring, hole


def node_of(grid: Grid, subs: tuple[int, int, int]) -> int:
    found, index = nearest_node(np.array(subs, dtype=float) * grid.h, grid)
    return int(engine_indices(np.array([index]), grid)[0])


class TestTheAirOutside:
    def test_the_ring_is_found_from_the_grid_and_nothing_of_the_room_with_it(self) -> None:
        room, ring, hole = dwelling()
        arrays, grid = air_arrays(room | ring | hole)
        source = np.array([20.3, 17.1, 10.6]) * grid.h
        seeds = engine_indices(interp_weights(source, grid)[1], grid)
        reached = reach_of(arrays, seeds)
        assert np.array_equal(reached, room | ring | hole)
        outside, record = outside_air(reached, grid.h, max_gap_m=5 * grid.h)
        assert np.array_equal(outside, ring)
        assert record["nodes"] == int(ring.sum())
        found = {face["face"]: face for face in record["faces"]}
        assert all(
            found[name]["ring"] and found[name]["gap_nodes"] == 3
            for name in ("x-", "x+", "y-", "y+")
        )
        assert not found["z-"]["ring"] and not found["z+"]["ring"]
        assert found["x-"]["first_node"] == 3
        # The rows that go on past the wall: the opening's own, and the two strips beside.
        assert found["x-"]["rows_past_the_wall"] == (5 + 6) * 16

    def test_a_dwelling_that_does_not_leak_has_no_ring(self) -> None:
        room, ring, hole = dwelling(opening=False)
        arrays, grid = air_arrays(room | ring)
        source = np.array([20.3, 17.1, 10.6]) * grid.h
        reached = reach_of(arrays, engine_indices(interp_weights(source, grid)[1], grid))
        assert np.array_equal(reached, room)
        outside, record = outside_air(reached, grid.h, max_gap_m=5 * grid.h)
        assert not outside.any() and not any(face["ring"] for face in record["faces"])

    def test_cut_off_it_is_never_reached_and_the_opening_is_a_wall_s_own_nodes(self) -> None:
        room, ring, hole = dwelling()
        arrays, grid = air_arrays(room | ring | hole)
        closed, _ = air_arrays(room | hole)
        source = np.array([20.3, 17.1, 10.6]) * grid.h
        seeds = engine_indices(interp_weights(source, grid)[1], grid)
        cut, record = without(arrays, ring, closure="rigid")
        assert record["closing_nodes"] == 5 * 16
        assert record["closing_m2"] == pytest.approx(5 * 16 * grid.h**2, abs=1e-3)
        problem = build_problem(cut, seeds)
        assert problem.reached == int((room | hole).sum())
        assert problem.reached == build_problem(arrays, seeds).reached - int(ring.sum())
        # Every node of the room and of the opening reads what it reads in a dwelling
        # built without a ring: the cut is the wall that model has there.
        kept = (room | hole).reshape(-1)
        mine = dict(zip(cut.bn_ixyz[kept[cut.bn_ixyz]], cut.adj_bn[kept[cut.bn_ixyz]], strict=True))
        theirs = dict(
            zip(
                closed.bn_ixyz[kept[closed.bn_ixyz]],
                closed.adj_bn[kept[closed.bn_ixyz]],
                strict=True,
            )
        )
        assert mine.keys() == theirs.keys()
        assert all(np.array_equal(mine[node], theirs[node]) for node in mine)

    def test_an_open_end_lets_a_duct_s_sound_out_and_a_rigid_one_keeps_it(self) -> None:
        """A duct of hard walls cut across: what meets an open cut squarely does not return."""
        air = np.zeros(SHAPE, dtype=bool)
        air[4:36, 14:20, 8:14] = True
        beyond = np.zeros(SHAPE, dtype=bool)
        beyond[30:36] = air[30:36]
        arrays, grid = air_arrays(air, branches=np.array([[0.0, 400.0, 0.0]]))
        source = np.array([[8.3, 16.4, 10.6]]) * grid.h
        seeds = engine_indices(interp_weights(source[0], grid)[1], grid)
        nodes = np.array([node_of(grid, s) for s in ((12, 16, 10), (20, 17, 11), (27, 16, 11))])
        late = {}
        for closure in ("open", "rigid"):
            cut, record = without(arrays, beyond, closure=closure)
            assert record["closure"] == closure and record["closing_nodes"] == 36
            problem = build_problem(cut, seeds)
            assert problem.reached == int((air & ~beyond).sum())
            records = solve(problem, drive_for(problem, grid, source, [nodes], 900 * grid.Ts), np)
            # Under the duct's first cross mode, 1300 Hz, a wave travels along it and meets
            # the cut squarely: that is what an admittance of one absorbs.
            along = sosfilt(butter(4, 600.0, fs=1.0 / grid.Ts, output="sos"), records, axis=1)
            late[closure] = float((along[:, 600:] ** 2).sum())
        assert np.array_equal(cut.materials[0], arrays.materials[0]) and len(cut.materials) == 1
        opened, _ = without(arrays, beyond, closure="open")
        assert np.array_equal(opened.materials[-1], OPEN_BRANCHES)
        # Every node of the cut carries the opening's material, its edge too.
        assert int((opened.mat_bn == 1).sum()) == 36
        # 45 dB less after seventeen lengths of the duct; the bound is 35.
        assert late["open"] < 10.0**-3.5 * late["rigid"]
        with pytest.raises(ValueError, match="closed as"):
            without(arrays, beyond, closure="ajar")


# --------------------------------------------------------------------------
# the fit in its spectra
# --------------------------------------------------------------------------


def a_cell(steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A cell's records as a room gives them: silence, a direct sound, a decay of 60 dB."""
    rng = np.random.default_rng(7)
    rate = 27307.107326536352
    time = np.arange(steps) / rate
    onset = 0.004
    span = steps / rate
    envelope = np.where(time > onset, 10.0 ** (-3.0 * (time - onset) / span), 0.0)
    smooth = rng.standard_normal((40, steps)) * envelope[None, :]
    smooth[:, int(onset * rate) + 3] += 30.0
    # The solver's source is differentiated, and so are its records.
    records = (np.diff(smooth, axis=1, prepend=0.0) * rate).astype(np.float32)
    offsets = rng.uniform(-0.3, 0.3, size=(40, 3))
    offsets[0] = 0.0
    return records, offsets, rate


class TestTheFitInItsSpectra:
    def test_it_is_the_filters_and_the_resampler_to_their_own_ripple(self) -> None:
        steps = 8192
        records, offsets, rate = a_cell(steps)
        out = {}
        for fit in ("time", "spectra"):
            encoder = CellEncoder(
                scheme=CARTESIAN,
                grid_rate_hz=rate,
                grid_step_m=0.0218,
                sound_speed_m_s=343.2,
                fmax_hz=1500.0,
                settings=EncoderSettings(order=2, fit_order=3, max_frequency_hz=1500.0),
                xp=np,
                samples=1200,
                scale=1.0,
                fit=fit,
            )
            out[fit] = encoder.cell(records, offsets).astype(np.float64)
            assert out[fit].shape == (9, 1200)
            assert encoder.cells(records, [offsets])[0].shape == (9, 1200)
        chain, spectra = out["time"], out["spectra"]
        difference = spectra - chain

        def level(values: np.ndarray) -> float:
            return float(10.0 * np.log10((values**2).sum()))

        # The whole response: 56 dB under it. Where the record is not ending, what is left
        # is the resampler's own gain, 0.006 dB over one in its passband, which the
        # spectra do not have: 62 dB under each window of 50 ms, read against its own energy.
        assert level(difference) - level(chain) < -55.0
        assert level(spectra) - level(chain) == pytest.approx(-0.006, abs=0.003)
        for start in (0, 200, 400):
            window = slice(start, start + 200)
            assert level(difference[:, window]) - level(chain[:, window]) < -60.0, start
        # In the record's last tenth of a second the two part: the chain stops its filters
        # with the record, the spectra let the low cut of 40 Hz ring on. The response is
        # 60 dB down by then, and the difference 56 dB under the response's start on this
        # record of 0.3 s (94 dB under it on a record of 1.2 s).
        assert level(difference[:, 800:]) - level(chain[:, :200]) < -50.0

    def test_a_fit_is_one_of_the_two(self) -> None:
        with pytest.raises(ValueError, match="a fit is one of"):
            CellEncoder(
                scheme=CARTESIAN,
                grid_rate_hz=27307.1,
                grid_step_m=0.0218,
                sound_speed_m_s=343.2,
                fmax_hz=1500.0,
                settings=EncoderSettings(order=2, fit_order=3, max_frequency_hz=1500.0),
                xp=np,
                samples=1200,
                scale=1.0,
                fit="guess",
            )


# --------------------------------------------------------------------------
# a run's order, its probe, and several recipes as one
# --------------------------------------------------------------------------


class TestARun:
    def test_the_longest_launches_go_first_and_the_last_are_of_one_source(self) -> None:
        launches = [[Item(s, (0,), 10) for s in range(8 * k, 8 * k + 8)] for k in range(6)]
        launches.insert(2, [Item(100, (0, 1, 2), 30), Item(101, (0,), 10)])
        ordered = last_short(launches, cards=2)
        sizes = [len(batch) for batch in ordered]
        assert sizes == [8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # Nothing is lost and nothing solved twice; launches of one length keep their order.
        assert sorted(i.source for b in ordered for i in b) == sorted(
            i.source for b in launches for i in b
        )
        assert [b[0].source for b in ordered[:5]] == [0, 8, 16, 24, 32]
        assert [b[0].source for b in ordered[5:]] == [40, 41, 42, 43, 44, 45, 46, 47, 100, 101]
        # A run of no more than two launches a card has no end to shorten.
        assert last_short(launches[:4], cards=2) == sorted(launches[:4], key=lambda b: -len(b))
        assert last_short([], cards=8) == []

    def test_a_probe_reads_the_step_against_its_bytes(self) -> None:
        arrays, _ = box_arrays(CARTESIAN, (26, 22, 18), room=((4, 4, 4), (20, 17, 13)), lossy=True)
        problem = build_problem(arrays, None)
        record = probe(problem, np, seconds=0.05, card="NVIDIA GeForce RTX 3090")
        assert record["steps"] >= 1 and record["updates_per_s"] > 0
        assert record["gb_per_s"] == pytest.approx(
            record["bytes_a_step"] * record["steps"] / record["seconds"] / 1e9, rel=0.2
        )
        # The host is measured and not judged: only a card has a memory rate to be held to.
        assert "slow" not in record
        assert memory_rate_of("NVIDIA GeForce RTX 3090") == 936.0
        assert memory_rate_of("GeForce RTX 3090 Ti") == 1008.0
        assert memory_rate_of("Tesla P100-PCIE-16GB") is None

    @pytest.mark.skipif(compiler() is None, reason="no C++ compiler on this host")
    def test_a_launch_steps_on_the_grid_another_stepper_put_on_the_card(self) -> None:
        arrays, grid = box_arrays(
            CARTESIAN, (26, 22, 18), room=((4, 4, 4), (20, 17, 13)), lossy=True
        )
        sources = np.array([[8.3, 9.1, 7.6], [15.2, 12.4, 9.9]]) * grid.h
        nodes = np.array([node_of(grid, s) for s in ((12, 11, 9), (18, 6, 6))])
        seeds = np.concatenate([engine_indices(interp_weights(s, grid)[1], grid) for s in sources])
        problem = build_problem(arrays, seeds)
        drive = drive_for(problem, grid, sources, [nodes, nodes], 40 * grid.Ts)
        from reverberate.wave.lowband.solver import Drive

        held = CardStepper(problem, Drive.nothing(), np, compile=host_kernel)
        launch = CardStepper(problem, drive, np, compile=host_kernel, shared=held)
        assert launch.mask is held.mask and launch.lossy is held.lossy
        assert np.array_equal(solve(problem, drive, np, stepper=launch), solve(problem, drive, np))
        with pytest.raises(ValueError, match="its own problem"):
            CardStepper(build_problem(arrays, None), drive, np, compile=host_kernel, shared=held)


def a_bundle(root: Path, sources: Any, cells: Any, heard: list[list[int]], **spec: Any) -> Path:
    root.mkdir(parents=True)
    (root / "models").mkdir()
    np.save(root / "sources.npy", np.asarray(sources, dtype=float))
    np.save(root / "cells.npy", np.asarray(cells, dtype=float))
    (root / "heard_at.json").write_text(json.dumps(heard))
    campaign = {
        "kind": "low-band-pairs",
        "dwelling": "hssd_0000",
        "bands": {"low": {"fmax_hz": 1500.0, "duration_s": 1.2, "cache_key": "grid"}},
        "ppw": 10.5,
        "tc": 20.0,
        "rh": 50.0,
        "order": 7,
        "fit_order": 10,
        "encoder": {"order": 7},
        "estimate": {"usd": 1.0},
        **spec,
    }
    (root / "campaign.json").write_text(json.dumps(campaign))
    return root


class TestSeveralRecipesAsOne:
    def test_a_position_two_recipes_share_is_solved_once_for_the_cells_of_both(
        self, tmp_path: Path
    ) -> None:
        rail = [[1.0, 1.2, 2.0], [1.08, 1.2, 2.0], [1.16, 1.2, 2.0]]
        one = a_bundle(
            tmp_path / "one", rail[:2], [[3.0, 1.2, 1.0], [3.4, 1.2, 1.0]], [[0, 1], [1]]
        )
        # The second recipe passes two of the same positions, to within the key's millimetre,
        # and hears them from cells of its own and from one of the first's.
        other = a_bundle(
            tmp_path / "other",
            [[1.0800004, 1.2, 2.0], rail[2]],
            [[5.0, 1.2, 1.0], [3.4, 1.2, 1.0]],
            [[0, 1], [0]],
        )
        record = merge_bundles([one, other], tmp_path / "union")
        assert record == {
            "recipes": 2,
            "source_positions_asked": 4,
            "source_positions": 3,
            "pairs_asked": 6,
            "pairs": 5,
            "cells": 3,
        }
        union = tmp_path / "union"
        assert np.allclose(np.load(union / "sources.npy"), rail)
        assert json.loads((union / "heard_at.json").read_text()) == [[0, 1], [1, 2], [2]]
        held = json.loads((union / "campaign.json").read_text())
        assert held["source_positions"] == 3 and held["pairs"] == 5 and held["points"] == 2
        assert "estimate" not in held and held["merged"]["recipes"] == 2
        assert (union / "models").resolve() == (one / "models").resolve()

    def test_bundles_of_two_grids_are_not_merged(self, tmp_path: Path) -> None:
        one = a_bundle(tmp_path / "one", [[1.0, 1.2, 2.0]], [[3.0, 1.2, 1.0]], [[0]])
        other = a_bundle(tmp_path / "other", [[1.0, 1.2, 2.0]], [[3.0, 1.2, 1.0]], [[0]], ppw=7.2)
        with pytest.raises(ValueError, match="first bundle's ppw"):
            merge_bundles([one, other], tmp_path / "union")
        with pytest.raises(ValueError, match="no bundle"):
            merge_bundles([], tmp_path / "union")
