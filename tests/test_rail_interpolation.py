"""A source between its solved positions: the weights, the plan, the pack and the engine.

The field is known in closed form throughout: a handful of plane waves for
the weights, a free monopole for the engine, whose low band at any place of
the source is :func:`reverberate.render.pack.monopole_low_response`. Nothing
is solved and nothing is read from the data root.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.compute import Devices
from reverberate.experiments.w44_interpolation.__main__ import main as experiment
from reverberate.experiments.w44_interpolation.rail_interpolation import (
    THIRDS,
    predict,
    score,
    third_octave_masks,
)
from reverberate.render.engine import Engine
from reverberate.render.low import SLOT_REACH, LowPart
from reverberate.render.pack import (
    Low,
    PackError,
    ScenePack,
    monopole_low_response,
    read_pack,
    synthetic_free_field,
    validate,
    write_pack,
)
from reverberate.scenes import Parameters, Recipe, rail_arc_lengths
from reverberate.scenes import validate as validate_recipe
from reverberate.scenes.recipe import Rail
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.spatial.rail import (
    KNOT_HZ,
    band_limited_weights,
    knots_hz,
    lagrange_weights,
    nearest_samples,
    residual_db,
)
from reverberate.spatial.translate import SOUND_SPEED_M_S
from reverberate.trace.bundle import build_bundle
from reverberate.trace.engines import FreeFieldPairs
from reverberate.trace.plan import Profile, make_plan, read_arcs, tracks_of
from reverberate.trace.run import Trace
from reverberate.viz.computed_api import low_at
from test_trace import assets, moving_recipe
from test_w44_interpolation import PlaneWaves

FS = 48000


def error_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(estimate - truth) ** 2) / np.sum(np.abs(truth) ** 2)))


# --------------------------------------------------------------------------
# the weights
# --------------------------------------------------------------------------


def along(pitch_m: float, count: int = 12) -> np.ndarray:
    """Positions ``pitch_m`` apart on a line off every axis."""
    way = np.array([0.6, 0.0, 0.8])
    return np.asarray(np.array([0.3, 1.7, -0.2]) + np.arange(count)[:, None] * pitch_m * way)


def read_at(pitch_m: float, share: float, freqs: np.ndarray, count: int = 8) -> float:
    """The error of ``count`` positions round a target ``share`` of the way through a gap."""
    field = PlaneWaves(count=7)
    line = along(pitch_m)
    arcs = np.arange(line.shape[0]) * pitch_m
    at = np.array([(5 + share) * pitch_m])
    target = line[5] + share * (line[6] - line[5])
    chosen = nearest_samples(arcs, at, count)[0]
    weights = band_limited_weights(line[chosen][None], target[None], freqs)[0]  # [slot, f]
    solved = np.stack([field.pressure(point, freqs) for point in line[chosen]])
    return error_db((weights * solved).sum(axis=0), field.pressure(target, freqs))


def test_positions_half_a_wavelength_apart_hold_the_field() -> None:
    top = np.array([1000.0, 1250.0, 1400.0])  # half a wavelength is 17, 14 and 12 cm
    # Measured: -57 dB at 8 cm, where the two round the target alone leave -22 dB.
    assert read_at(0.08, 0.5, top) < -50.0
    assert read_at(0.08, 0.5, top, count=2) > -25.0
    # At 12 cm, -44 dB at 1 kHz and -13 dB at 1400 Hz, whose half wavelength it is.
    assert read_at(0.12, 0.5, top[:1]) < -38.0
    assert -20.0 < read_at(0.12, 0.5, top[2:]) < -10.0
    # Past half a wavelength of the frequency nothing holds it (-5 dB), and well under it
    # the same positions do, however far apart: -95 dB and -61 dB.
    assert read_at(0.20, 0.5, np.array([1250.0])) > -6.0
    assert read_at(0.20, 0.5, np.array([400.0])) < -60.0
    assert read_at(0.32, 0.5, np.array([250.0])) < -50.0


def test_the_weights_change_with_frequency_and_are_the_position_itself_on_it() -> None:
    line = along(0.12, 8)
    freqs = np.array([100.0, 1000.0])
    middle = 0.5 * (line[3] + line[4])
    chosen = nearest_samples(np.arange(8) * 0.12, np.array([3.5 * 0.12]), 8)[0]
    assert chosen.tolist() == [3, 4, 2, 5, 1, 6, 0, 7]
    weights = band_limited_weights(line[chosen][None], middle[None], freqs)[0]
    assert weights.shape == (8, 2)
    # Each side alike, and nearly all of it from the two round the target.
    np.testing.assert_allclose(weights[0], weights[1], atol=1e-9)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=0.02)
    assert np.abs(weights[:, 0] - weights[:, 1]).max() > 0.02
    # On a position: that position alone at 1 kHz. At 100 Hz the eight are a twentieth of a
    # wavelength apart and say the same thing: the weights average them, and what they
    # give is still the field there.
    on = band_limited_weights(line[chosen][None], line[4][None], freqs)[0]
    np.testing.assert_allclose(on[:, 1], np.eye(8)[1], atol=2e-3)
    field = PlaneWaves(count=7)
    solved = np.stack([field.pressure(point, freqs) for point in line[chosen]])
    assert error_db((on * solved).sum(axis=0), field.pressure(line[4], freqs)) < -40.0
    # A slot that holds no position has no weight and changes nothing for the others.
    valid = np.array([[True] * 6 + [False] * 2])
    six = band_limited_weights(line[chosen][None], middle[None], freqs, valid=valid)[0]
    assert np.all(six[6:] == 0.0)
    np.testing.assert_allclose(
        six[:6], band_limited_weights(line[chosen[:6]][None], middle[None], freqs)[0]
    )


def test_an_end_of_the_rail_and_a_corner_are_read_by_the_same_weights() -> None:
    field = PlaneWaves(count=7)
    freqs = np.array([500.0, 1000.0])
    pitch = 0.12
    arcs = np.arange(10) * pitch
    # The first gap of a rail: the two round the target, then six on one side only.
    chosen = nearest_samples(arcs, np.array([0.5 * pitch]), 8)[0]
    assert chosen.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert nearest_samples(arcs[:3], np.array([0.5 * pitch]), 8)[0].tolist() == [0, 1, 2] + [-1] * 5
    line = along(pitch, 10)
    target = 0.5 * (line[0] + line[1])
    weights = band_limited_weights(line[chosen][None], target[None], freqs)[0]
    solved = np.stack([field.pressure(point, freqs) for point in line[chosen]])
    # Measured: -35 dB, against -20 dB for the two alone.
    assert error_db((weights * solved).sum(axis=0), field.pressure(target, freqs)) < -30.0
    assert error_db(0.5 * (solved[0] + solved[1]), field.pressure(target, freqs)) > -24.0
    # A right angle 0.30 m along, between the samples at 0.24 and 0.36 m of the path: on the
    # corner the source is on neither sample's line, and eight positions do no better than two.
    rail = Rail("r", "a", "b", ((0.0, 0.0), (0.30, 0.0), (0.30, 0.90)), pitch)
    turned, chord, left = corner_errors(rail, rail_arc_lengths(rail), 0.30, freqs)
    # Measured: -6.8 dB and -7.2 dB, and the weights say so themselves: -12.7 dB at 1 kHz.
    assert turned > -12.0 and chord > -12.0 and left[1] > -15.0
    # With the corner solved (:func:`read_arcs`) every place of the rail is between two
    # positions of its own line: each leg is cut into equal gaps of the pitch at most.
    arcs = read_arcs(rail)
    assert arcs.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, *(0.3 + 0.1125 * np.arange(1, 9))])
    # Measured, the middle of the gap before the corner, of the one after it, and of the
    # next: -22, -24 and -36 dB, where the two round the source alone leave -9 to -19 dB.
    for at, bound in ((0.25, -18.0), (0.35625, -20.0), (0.46875, -30.0)):
        turned, chord, left = corner_errors(rail, arcs, at, freqs)
        assert turned < bound and chord > turned + 2.0 and np.all(left < -25.0), at
    # No gap is a sliver: a rail 1.21 m long is eleven gaps of 11 cm, where its samples
    # end on one of 1 cm, two positions that say the same thing.
    straight = Rail("r", "a", "b", ((0.0, 0.0), (1.21, 0.0)), pitch)
    assert np.diff(read_arcs(straight)).tolist() == pytest.approx([0.11] * 11)
    assert np.diff(rail_arc_lengths(straight))[-1] == pytest.approx(0.01)


def corner_errors(
    rail: Rail, arcs: np.ndarray, at_m: float, freqs: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Eight positions' error at ``at_m`` of a rail, the two round it alone, the weights' own."""
    field = PlaneWaves(count=7)
    points = np.asarray(rail.points, dtype=float)
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])

    def place(arc: np.ndarray) -> np.ndarray:
        x, z = np.interp(arc, walked, points[:, 0]), np.interp(arc, walked, points[:, 1])
        return np.stack([x, np.full(np.shape(arc), 1.7), z], axis=-1)

    nodes, target = place(arcs), place(np.array(at_m))
    chosen = nearest_samples(arcs, np.array([at_m]), 8)[0]
    weights = band_limited_weights(nodes[chosen][None], target[None], freqs)[0]
    solved = np.stack([field.pressure(point, freqs) for point in nodes[chosen]])
    truth = field.pressure(target, freqs)
    pair = lagrange_weights(arcs[chosen[:2]] - at_m)
    return (
        error_db((weights * solved).sum(axis=0), truth),
        error_db(pair @ solved[:2], truth),
        residual_db(nodes[chosen][None], target[None], freqs)[0],
    )


def test_the_polynomial_through_two_positions_is_the_linear_reading() -> None:
    np.testing.assert_allclose(lagrange_weights(np.array([-0.02, 0.06])), [0.75, 0.25])
    cubic = lagrange_weights(np.array([-0.04, 0.04, -0.12, 0.12]))
    np.testing.assert_allclose(cubic, [0.5625, 0.5625, -0.0625, -0.0625])
    assert knots_hz(1500.0).tolist() == np.arange(0.0, 1501.0, KNOT_HZ).tolist()
    with pytest.raises(ValueError, match="at least one"):
        nearest_samples(np.zeros(0), np.array([0.0]), 4)


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------


def test_the_measurement_scores_an_interpolator_per_third_octave(tmp_path: Path) -> None:
    field = PlaneWaves(count=7)
    samples, points = 4096, 64
    freqs = np.fft.rfftfreq(samples, 1.0 / FS)
    shape = np.clip((1700.0 - freqs) / 200.0, 0.0, 1.0)
    positions = np.array([0.0, 1.7, 0.0]) + np.arange(points)[:, None] * np.array([0, 0, 0.02])
    path = tmp_path / "line.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = float(FS)
        f.attrs["source_position"] = np.array([-1.0, 1.7, -1.5])
        f["positions"] = positions
        ir = f.create_dataset("ir", (points, 1, samples), dtype="f4")
        for i, point in enumerate(positions):
            ir[i, 0] = np.fft.irfft(field.pressure(point, freqs) * shape, samples)
    assert experiment(["rail", "--field", str(path), "--out", str(tmp_path), "--pitches", "6"]) == 0
    summary = json.loads((tmp_path / "summary_rail.json").read_text())
    assert summary["thirds_hz"] == THIRDS
    mid = summary["regimes"]["mid"]
    assert mid["bands_hz"] == [800, 1000, 1250] and list(mid["pitches"]) == ["0.120"]
    at = mid["pitches"]["0.120"]
    # At 12 cm eight positions hold 1 kHz and two do not, measured at worst: -30 dB against
    # -14 dB, and a level within 0.05 dB against 1 dB.
    assert at["band8"]["inner_far"]["early_error"]["worst"][1] < -25.0
    assert at["linear2"]["inner_far"]["early_error"]["worst"][1] > -18.0
    assert at["band8"]["inner_far"]["early_level"]["worst"][1] < 0.2
    assert set(at["band8"]) >= {"inner_far", "end_far"}
    assert at["band8"]["end_far"]["early_masked"]["worst"][2] < -30.0

    bins = np.fft.rfftfreq(400, 1 / 4000.0)
    masks = third_octave_masks(bins)
    assert masks.shape == (12, bins.size) and masks[THIRDS.index(1000)][100]
    # A prediction that is the truth scores nothing wrong and the truth's own level.
    spectrum = np.ones((2, 151), dtype=complex)
    same = score(spectrum, spectrum, np.arange(151) * 10.0, 400)
    assert np.all(same["whole_error"] == -np.inf) and np.allclose(same["whole_level"], 0.0)
    with pytest.raises(ValueError, match="no interpolator"):
        predict(
            "spline",
            2,
            np.arange(3.0),
            np.zeros((3, 3)),
            np.array([0.5]),
            np.zeros((1, 3)),
            bins,
            1,
        )


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

#: A rail towards the listener, then a right angle: 0.80 m and 0.40 m, walked at 1 m/s.
RAIL_POINTS = np.array([[3.0, 0.3], [2.2, 0.3], [2.2, 0.7]])
WALK_S = 1.2


def on_rail(arc: np.ndarray, height: float) -> np.ndarray:
    lengths = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(RAIL_POINTS, axis=0), axis=1))]
    )
    x = np.interp(arc, lengths, RAIL_POINTS[:, 0])
    z = np.interp(arc, lengths, RAIL_POINTS[:, 1])
    return np.stack([x, np.full(np.shape(arc), height), z], axis=-1)


def walking(pitch_m: float | None, count: int = 2) -> ScenePack:
    """A monopole walking the rail past a listener on a cell, its low band as a pack holds it.

    With no pitch every step has the response of where the source is: the
    truth. With one the rail is solved every ``pitch_m`` and a step reads
    two positions linearly or, with a ``count`` over two, that many by the
    band limited weights.
    """
    base = synthetic_free_field(
        level="B", source=on_rail(np.array(0.0), 1.5), listener_start=(0, 1.5, 0), duration_s=WALK_S
    )
    header, crossover = base.header, base.crossover
    steps = header.steps
    arc = np.linspace(0.0, 1.2, steps)
    position = on_rail(arc, 1.5)
    cell = base.cells.position[0]
    pair = np.full((steps, 2, 2), -1, dtype=np.int32)
    weight = np.zeros(steps, dtype=np.float32)
    slot_pair = slot_weight = knots = None
    if pitch_m is None:
        solved = position
        pair[:, 0, 0] = np.arange(steps)
    else:
        rail = Rail("r", "a", "b", tuple(map(tuple, RAIL_POINTS)), pitch_m)
        arcs = read_arcs(rail) if count > 2 else rail_arc_lengths(rail)
        solved = on_rail(arcs, 1.5)
        lower = np.clip(np.searchsorted(arcs, arc, side="right") - 1, 0, arcs.size - 2)
        share = (arc - arcs[lower]) / (arcs[lower + 1] - arcs[lower])
        between = (share > 1e-9) & (share < 1.0 - 1e-9)
        pair[:, 0, 0] = np.where(share >= 1.0 - 1e-9, lower + 1, lower)
        pair[between, 1, 0] = lower[between] + 1
        weight[between] = share[between]
        if count > 2:
            knots = knots_hz(1500.0)
            chosen = nearest_samples(arcs, arc, count, lower=lower)
            slot_pair = np.full((steps, count, 2), -1, dtype=np.int32)
            slot_pair[:, 0, 0] = pair[:, 0, 0]
            slot_pair[between, :, 0] = chosen[between]
            slot_weight = np.zeros((steps, count, knots.size), dtype=np.float32)
            slot_weight[:, 0, :] = 1.0
            slot_weight[between] = band_limited_weights(
                solved[np.maximum(chosen[between], 0)],
                position[between],
                knots,
                valid=chosen[between] >= 0,
            )
    held = base.sources["s1"].low
    assert held is not None
    low = Low(
        ir=np.stack([monopole_low_response(point - cell, header, crossover) for point in solved]),
        pair_position=solved,
        pair_cell=np.zeros(len(solved), dtype=np.int32),
        pair_key=np.array([f"{row:064d}" for row in range(len(solved))], dtype="S64"),
        seam_db=np.zeros(len(solved), dtype=np.float32),
        onset_s=np.linalg.norm(solved - cell, axis=1) / header.sound_speed_m_s,
        pair=pair,
        position_weight=weight,
        cell=held.cell,
        mode=held.mode,
        slot_pair=slot_pair,
        slot_weight=slot_weight,
        slot_knots_hz=knots,
    )
    pack = replace(base, sources={"s1": replace(base.sources["s1"], position=position, low=low)})
    validate(pack)
    return pack


def low_of(pack: ScenePack, dry: np.ndarray) -> np.ndarray:
    return np.asarray(Engine(pack, {"s1": dry}).stem("s1", parts=("low",)))


def thirds_db(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """The error of the first channel over the truth's, per third octave from 200 Hz."""
    freqs = np.fft.rfftfreq(want.shape[-1], 1.0 / FS)
    masks = third_octave_masks(freqs)[THIRDS.index(200) :]
    wrong = np.abs(np.fft.rfft(got[0] - want[0])) ** 2
    return np.asarray(
        10 * np.log10((wrong @ masks.T) / (np.abs(np.fft.rfft(want[0])) ** 2 @ masks.T))
    )


def test_a_walking_monopole_is_its_truth_from_fewer_positions() -> None:
    rng = np.random.default_rng(11)
    dry = rng.standard_normal(int(WALK_S * FS))
    truth = low_of(walking(None), dry)
    today = thirds_db(low_of(walking(0.08), dry), truth)
    # Bands 200 250 315 400 500 630 800 1000 1250 Hz. Measured, two positions every 8 cm:
    # -45 -41 -37 -32 -29 -24 -21 -18 -15 dB.
    assert today[-2] > -20.0 and today[-1] > -17.0
    errors = {}
    for pitch in (0.08, 0.12, 0.16):
        errors[pitch] = thirds_db(low_of(walking(pitch, 8), dry), truth)
    # Eight positions, the rail's corner and both its ends included, measured. Every 8 cm:
    # -43 dB at 1250 Hz and under -50 dB below. Every 12 cm: -49 dB to 630 Hz, then -43,
    # -33 and -22 dB. Every 16 cm: -34 dB at 630 Hz, then -23, -10 and -3 dB: half a
    # wavelength is 16 cm at 1072 Hz.
    assert np.all(errors[0.08] < -40.0)
    assert np.all(errors[0.12][:-1] < -30.0) and errors[0.12][-1] < -19.0
    assert np.all(errors[0.16][:-2] < -20.0) and errors[0.16][-2] > -15.0
    # At 12 cm no band is worse than two positions every 8 cm, and 1 kHz is 15 dB better.
    assert np.all(errors[0.12] < today) and errors[0.12][-2] < today[-2] - 12.0
    # All 64 channels, on what the low side holds: -42 dB at 12 cm against -25 dB today.
    whole = {
        name: error_db(low_of(walking(*args), dry), truth)
        for name, args in (("today", (0.08,)), ("eight", (0.12, 8)))
    }
    assert whole["eight"] < -38.0 and whole["today"] > -30.0


def test_two_slots_weighted_linearly_are_the_first_rule_and_a_pack_without_is_untouched(
    tmp_path: Path,
) -> None:
    dry = np.random.default_rng(3).standard_normal(int(WALK_S * FS))
    plain = walking(0.08)
    low = plain.sources["s1"].low
    assert low is not None and low.slot_pair is None
    knots = knots_hz(1500.0)
    weights = np.stack([1.0 - low.position_weight, low.position_weight], axis=1)
    slotted = replace(
        plain,
        sources={
            "s1": replace(
                plain.sources["s1"],
                low=replace(
                    low,
                    slot_pair=low.pair.copy(),
                    slot_weight=np.repeat(weights[:, :, None], knots.size, axis=2),
                    slot_knots_hz=knots,
                ),
            )
        },
    )
    validate(slotted)
    first, second = low_of(plain, dry), low_of(slotted, dry)
    # Another length of transform, the same samples: 1e-12 of the peak.
    np.testing.assert_allclose(second, first, atol=1e-10 * np.abs(first).max())
    # The first rule's transform is as long as it was, and reads nothing more of the dry signal.
    part = Engine(plain, {"s1": dry}).source("s1").parts["low"]
    assert isinstance(part, LowPart) and (part.pad, part.n) == (0, 5184) and SLOT_REACH == 256
    # The file: no slot table unless the pack has them, and they come back as written.
    write_pack(tmp_path / "plain.h5", plain)
    write_pack(tmp_path / "slotted.h5", walking(0.12, 8))
    with h5py.File(tmp_path / "plain.h5", "r") as f:
        assert sorted(f["sources/s1/low"]) == sorted(
            ["cell", "ir", "mode", "onset_s", "pair", "pair_cell", "pair_key"]
            + ["pair_position", "position_weight", "seam_db"]
        )
    with read_pack(tmp_path / "slotted.h5", deep=True) as back:
        held = back.sources["s1"].low
        assert held is not None and held.slot_pair is not None and held.slot_weight is not None
        assert held.slot_pair.shape == (25, 8, 2) and held.slot_weight.shape == (25, 8, 31)
        want = walking(0.12, 8).sources["s1"].low
        assert want is not None and want.slot_weight is not None
        np.testing.assert_array_equal(held.slot_weight, want.slot_weight)
        # What the audit shows of a step: every position read, its weight at the crossover.
        step = int(np.flatnonzero(held.slot_pair[:, 7, 0] >= 0)[0])
        shown = low_at(back, back.sources["s1"], step)
        assert len(shown["cells"][0]["pairs"]) == 8 and "1000 Hz" in shown["exact"]
        assert sum(pair["weight"] for pair in shown["cells"][0]["pairs"]) == pytest.approx(
            1.0, abs=0.05
        )


def test_slots_that_are_not_the_pair_or_weigh_nothing_are_refused() -> None:
    pack = walking(0.12, 4)
    low = pack.sources["s1"].low
    assert low is not None and low.slot_pair is not None and low.slot_weight is not None

    def broken(**changes: Any) -> ScenePack:
        return replace(
            pack, sources={"s1": replace(pack.sources["s1"], low=replace(low, **changes))}
        )

    moving = int(np.flatnonzero(low.slot_pair[:, 1, 0] >= 0)[0])
    swapped = low.slot_pair.copy()
    swapped[moving, 0, 0], swapped[moving, 1, 0] = swapped[moving, 1, 0], swapped[moving, 0, 0]
    heavy = low.slot_weight.copy()
    heavy[moving, 0, 0] = 100.0
    ghost = low.slot_weight.copy()
    ghost[0, 3, :] = 0.5
    for changes, said in (
        ({"slot_pair": swapped}, "is not pair"),
        ({"slot_weight": heavy}, "passes"),
        ({"slot_weight": ghost}, "holds no row"),
        ({"slot_knots_hz": None}, "not all"),
        ({"slot_knots_hz": np.array([0.0, 50.0, 150.0])}, "expected|evenly"),
    ):
        with pytest.raises(PackError, match=said):
            validate(broken(**changes))


# --------------------------------------------------------------------------
# the recipe, the plan and the trace
# --------------------------------------------------------------------------


def at_pitch(recipe: Recipe, pitch_m: float) -> Recipe:
    return replace(recipe, rails=tuple(replace(rail, pitch_m=pitch_m) for rail in recipe.rails))


def test_a_recipe_may_ask_for_another_pitch_on_all_its_rails() -> None:
    recipe = moving_recipe()
    for pitch in (0.08, 0.12, 0.16):
        assert not [v for v in validate_recipe(at_pitch(recipe, pitch), None) if v.rule == 4]
    assert [v for v in validate_recipe(at_pitch(recipe, 0.30), None) if v.rule == 4]
    record = Parameters().record()
    record["rails"]["pitch_m"] = 0.12
    assert Parameters.from_record(record).rail_pitch_m == 0.12


def test_the_plan_reads_more_positions_only_when_asked() -> None:
    recipe = moving_recipe()
    plain = tracks_of(recipe)
    assert Profile().record() == {"seconds": None, "sources": None, "patch": False, "start_s": 0.0}
    assert Profile.from_record(Profile().record()) == Profile()
    voice = plain.sources["voice"]
    assert voice.rail_slot is None and plain.rail_knots_hz is None
    assert voice.read(5).tolist() == voice.slot[5].tolist()
    asked = Profile(rail_positions=4)
    assert Profile.from_record(asked.record()) == asked and asked.record()["rail_positions"] == 4
    with pytest.raises(ValueError, match="2 to 16"):
        Profile(rail_positions=1)
    tracks = tracks_of(recipe, asked)
    voice = tracks.sources["voice"]
    assert voice.rail_slot is not None and voice.rail_weight is not None
    assert tracks.rail_knots_hz is not None and tracks.rail_knots_hz[-1] == 1500.0
    # The first rule's two positions come first, and its tables are what they were.
    np.testing.assert_allclose(voice.weight, plain.sources["voice"].weight, atol=1e-12)
    between = voice.weight > 0.0
    assert between.sum() >= 4 and np.all(voice.rail_slot[between] >= 0)
    for step in np.flatnonzero(voice.audible):
        mine = tracks.positions[voice.slot[step][voice.slot[step] >= 0]]
        theirs = plain.positions[plain.sources["voice"].slot[step]]
        np.testing.assert_array_equal(mine, theirs[: len(mine)])
        np.testing.assert_array_equal(voice.rail_slot[step, :2], voice.slot[step])
    # At rest, and on a solved position, one position with a weight of one.
    alone = voice.audible & ~between
    assert np.all(voice.rail_slot[alone, 1:] == -1) and np.all(voice.rail_weight[alone, 0] == 1.0)
    np.testing.assert_allclose(voice.rail_weight[between][:, :, :21].sum(axis=1), 1.0, atol=0.05)
    # No weight is large: positions are never so close that it takes one to tell them apart.
    assert float(np.abs(voice.rail_weight).max()) < 2.0
    # Four positions round every step are more solves than two: the rail's ends are read
    # from further in.
    assert tracks.positions.shape[0] >= plain.positions.shape[0]
    held = assets()
    plan = make_plan(recipe, held.triangles, asked)
    assert plan.record["profile"]["rail_positions"] == 4
    assert plan.record["source_positions"] == tracks.positions.shape[0]
    assert "rail_positions" not in make_plan(recipe, held.triangles).record["profile"]


def test_a_trace_writes_the_positions_it_read_and_the_engine_renders_them(tmp_path: Path) -> None:
    recipe = at_pitch(moving_recipe(), 0.12)
    held = assets(16)
    plan = make_plan(recipe, held.triangles, Profile(rail_positions=4))
    build_bundle(tmp_path / "bundle", recipe, held, plan, allow_asset_mismatch=True)
    pairs = FreeFieldPairs(
        plan.tracks.positions, plan.all_cells, tmp_path / "out", gain=FIELD_UNIT_AT_1M
    )
    Trace(
        bundle=tmp_path / "bundle",
        out=tmp_path / "out",
        engine=pairs,
        gpu=False,
        devices=Devices.host(1),
        quiet=True,
        check_mode="read",
    ).run()
    with read_pack(tmp_path / "out" / "pack.h5") as pack:
        low = pack.sources["voice"].low
        assert low is not None and low.slot_pair is not None and low.slot_weight is not None
        assert low.slot_pair.shape[1:] == (4, 2) and low.slot_knots_hz is not None
        moving = low.position_weight > 0.0
        assert moving.any() and np.all(low.slot_pair[moving][:, :, 0] >= 0)
        # Every slot's row was solved from one of the rail's positions, 12 cm apart.
        rows = np.unique(low.slot_pair[moving][:, :, 0])
        gaps = np.diff(np.unique(np.round(low.pair_position[rows][:, 2], 3)))
        assert np.all(np.isin(np.round(gaps, 3), [0.12]))
        tap = pack.sources["tap"].low
        assert tap is not None and tap.slot_pair is not None
        assert np.all(tap.slot_pair[:, 1:] == -1)
        dry = np.random.default_rng(1).standard_normal(pack.header.samples)
        stem = Engine(pack, {"voice": dry, "tap": dry}).stem("voice", parts=("low",))
        assert np.all(np.isfinite(stem)) and float(np.abs(stem).max()) > 0.0
    assert SOUND_SPEED_M_S / (2 * 0.12) > 1414.0
