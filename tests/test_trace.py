"""The scene trace on a small room, with a monopole in free air where a card would solve.

A recipe is written by hand in the walled box of the mirror's tests: a voice
that walks a rail, a noise behind the wall, a listener who walks between two
rests. Its plan is made, its bundle built and its trace run on ``numpy``,
with :class:`reverberate.trace.engines.FreeFieldPairs` in place of the pairs
campaign: everything else, the cell rule, the batched trace, the onsets,
the rays, the levelling and the pack's writer, is the real code.

The last test is the reason for the levelling: at rest, the source on a
station and the head on a cell, the pack rendered by the signal engine is
the response :func:`reverberate.mirror.hybrid.blend` joins from the same wave
response and the mirror of the present pipeline.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reverberate.accel import pairs as pairs_module
from reverberate.accel.pairs import PairCache
from reverberate.audio import Atmosphere
from reverberate.compute import Devices
from reverberate.metrics import octave_bank, octave_filter_rows
from reverberate.mirror.geometry import write_derived
from reverberate.mirror.hybrid import Crossover, blend, seam_db
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.pipeline import trace as trace_lattice
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.render import render_point
from reverberate.render.compact import CompactIr
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import read_pack
from reverberate.scenes import (
    Recipe,
    low_band_source_positions,
    placeholder_assets,
    recipe_sha256,
)
from reverberate.spatial.lowband import (
    FIELD_UNIT_AT_1M,
    LOW_RATE_HZ,
    delayed,
    from_stored,
    pair_key,
    with_air,
)
from reverberate.spatial.translate import (
    MODE_EXACT,
    MODE_FUSED,
    MODE_TRANSLATED,
    choose_cells,
    serving_radius_m,
)
from reverberate.trace.assets import MirrorAssets, found_assets, mismatched
from reverberate.trace.bundle import build_bundle
from reverberate.trace.cli import main
from reverberate.trace.driver import (
    cost_records,
    describe,
    finish,
    launch,
    machine_holds,
    resume_command,
    stamp_cost,
)
from reverberate.trace.engines import CardPairs, FreeFieldPairs, cache_levers
from reverberate.trace.level import pair_low
from reverberate.trace.plan import (
    ORIGIN_DENSE,
    ORIGIN_HEAD,
    Plan,
    Profile,
    busiest_window,
    estimate,
    make_plan,
    tail_cells,
)
from reverberate.trace.run import Trace
from test_mirror_diffract import walled_box

C = 343.2
FS = 48000
DIGEST = "ab" * 32
CLIP = {"library": "ears", "name": "p001/sentences_01_regular", "sha256": DIGEST}


def assets(rays: int = 120) -> MirrorAssets:
    """The walled box as a dwelling's mirror: a signature, a lead of 4 ms, a gain of 0.02.

    The gain is the mirror's on the field's scale, under the field's unit
    (0.026) as that of hssd_0076 is (0.0144).
    """
    return MirrorAssets(
        catalogue=walled_box(),
        settings=MirrorSettings(
            rays=RaySettings(rays=rays, duration_s=0.12, bin_s=0.002, receiver_radius_m=0.3)
        ),
        signature=np.array([1.0, 0.35, -0.12, 0.04]),
        lead_s=0.0040073,
        gain=0.02,
    )


def station(name: str, x: float, z: float, kind: str = "stand") -> dict[str, Any]:
    seated = kind == "seat"
    return {
        "id": name,
        "kind": kind,
        "position": [x, 1.2 if seated else 1.7, z],
        "height": "seated" if seated else "standing",
        "room": "east" if x > 2.0 else "west",
        "facing_yaw_deg": 0.0,
    }


def dwell(name: str, start: float, end: float, height: str = "standing") -> dict[str, Any]:
    return {
        "type": "dwell",
        "station": name,
        "height": height,
        "start_s": start,
        "end_s": end,
        "facing": {"mode": "fixed", "yaw_deg": 0.0},
    }


def source(name: str, kind: str, segments: list[dict[str, Any]], end: float) -> dict[str, Any]:
    tree: dict[str, Any] = {
        "id": name,
        "kind": kind,
        "directivity": {"model": "voice_v1" if kind != "noise" else "omni", "enabled": False},
        "gain_db": 0.0,
        "turn_rate_deg_s": 180.0,
        "segments": segments,
        "activity": [
            {"start_s": 0.0, "end_s": end, "clip": CLIP, "clip_offset_s": 0.0, "gain_db": 0.0}
        ],
    }
    if kind == "noise":
        tree["subtype"] = "water"
    return tree


def recipe_of(
    duration: float,
    stations: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    keyframes: list[tuple[float, tuple[float, float, float]]],
    rails: list[dict[str, Any]] | None = None,
) -> Recipe:
    return Recipe.from_dict(
        {
            "schema": "reverberate.scene-recipe",
            "schema_version": 1,
            "dwelling": {"name": "box", "scene_id": "walled-box", "floor_y_m": 0.0},
            "assets": placeholder_assets().to_dict(),
            "seed": 7,
            "duration_s": duration,
            "output": {"order": 7, "sample_rate_hz": 48000},
            "atmosphere": {
                "temperature_c": 20.0,
                "humidity_percent": 50.0,
                "pressure_kpa": 101.325,
            },
            "heights": {"standing_m": 1.7, "seated_m": 1.2},
            "stations": stations,
            "rails": rails or [],
            "sources": sources,
            "listener": {
                "interpolation": "linear",
                "keyframes": [
                    {
                        "t_s": t,
                        "position": list(position),
                        "yaw_deg": 0.0,
                        "pitch_deg": 0.0,
                        "roll_deg": 0.0,
                    }
                    for t, position in keyframes
                ],
            },
            "generator": None,
        }
    )


def moving_recipe() -> Recipe:
    """Half a second: a voice walks its rail, a noise stays behind the wall, the head walks."""
    rail = {"id": "r", "a": "a", "b": "b", "points": [[0.6, 0.5], [0.6, 0.98]], "pitch_m": 0.08}
    travel = {
        "type": "travel",
        "rail": "r",
        "from": "a",
        "to": "b",
        "profile": "constant",
        "start_s": 0.15,
        "end_s": 0.5,
        "facing": {"mode": "travel"},
    }
    return recipe_of(
        0.5,
        [station("a", 0.6, 0.5), station("b", 0.6, 0.98), station("tap", 3.2, 1.0)],
        [
            source("voice", "near_voice", [dwell("a", 0.0, 0.15), travel], 0.5),
            source("tap", "noise", [dwell("tap", 0.0, 0.5)], 0.5),
        ],
        [
            (0.0, (1.3, 1.7, 0.9)),
            (0.1, (1.3, 1.7, 0.9)),
            (0.4, (1.3, 1.7, 1.2)),
            (0.5, (1.3, 1.7, 1.2)),
        ],
        [rail],
    )


def resting_recipe() -> Recipe:
    """Half a second: a voice at its station and the head on one cell."""
    return recipe_of(
        0.5,
        [station("a", 0.6, 0.8)],
        [source("voice", "near_voice", [dwell("a", 0.0, 0.5)], 0.5)],
        [(0.0, (1.4, 1.7, 1.3)), (0.5, (1.4, 1.7, 1.3))],
    )


def traced(
    tmp: Path, recipe: Recipe, *, rays: int = 120, low_levers: str | None = None, **engine: Any
) -> tuple[Trace, FreeFieldPairs, Plan]:
    """The bundle of ``recipe`` and its trace, not yet run, on the free field engine.

    ``low_levers`` is the bundle's: left out, the form a trace writes by
    default, the bins in 16 bits, in the pack and in the pair cache.
    """
    held = assets(rays)
    plan = make_plan(recipe, held.triangles)
    build_bundle(
        tmp / "bundle", recipe, held, plan, allow_asset_mismatch=True, low_levers=low_levers
    )
    # The cache form: on the geometric clock and on the field's scale.
    pairs = FreeFieldPairs(
        plan.tracks.positions,
        plan.all_cells,
        tmp / "out",
        **{"gain": FIELD_UNIT_AT_1M, **engine},
    )
    # As the machine's command does: the form the pairs are kept in is the bundle's.
    pairs.cache.levers = cache_levers(tmp / "bundle")
    trace = Trace(
        bundle=tmp / "bundle",
        out=tmp / "out",
        engine=pairs,
        gpu=False,
        devices=Devices.host(1),
        quiet=True,
        check_mode="full",
    )
    return trace, pairs, plan


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def test_the_plan_is_deterministic_and_no_audible_step_is_without_cells() -> None:
    recipe = moving_recipe()
    held = assets()
    plan = make_plan(recipe, held.triangles)
    again = make_plan(recipe, held.triangles)
    assert json.dumps(plan.record, sort_keys=True) == json.dumps(again.record, sort_keys=True)
    np.testing.assert_array_equal(plan.cells.position, again.cells.position)
    assert plan.heard_at == again.heard_at
    assert plan.recipe_sha256 == recipe_sha256(recipe)
    assert plan.record["fallback_steps"] == []
    cells, clearance = plan.cells.position, plan.cells.clearance_m
    read: set[tuple[int, int]] = set()
    for name, track in plan.tracks.sources.items():
        chosen = plan.low[name]
        assert track.audible.all() and not chosen.fallback.any()
        for step in range(plan.tracks.steps):
            # The library's rule itself, over every cell: no step raises, and it is the plan's.
            radius = serving_radius_m(cells, clearance, track.position[step])
            mode, picked = choose_cells(plan.tracks.listener[step], cells, radius)
            assert (mode, picked) == (int(chosen.mode[step]), tuple(chosen.cell[step]))
            for position in track.slot[step]:
                read.update((int(position), int(c)) for c in picked if position >= 0 and c >= 0)
    # A pair is planned exactly when some audible step reads it.
    planned = {(p, c) for p, listed in enumerate(plan.heard_at) for c in listed}
    assert planned == read and plan.pairs == len(read)
    modes = plan.record["modes"]
    assert modes["exact"] > 0 and modes["fused"] > 0
    # The walk reads two solved positions at a step, the rest one; the rail is solved every 8 cm.
    voice = plan.tracks.sources["voice"]
    assert (voice.weight[voice.moving] > 0).any() and not voice.weight[~voice.moving].any()
    assert plan.record["source_positions"] == plan.tracks.positions.shape[0] > 3
    priced = estimate(plan, rate_usd_per_hour=0.4)
    assert priced["billed_rate_usd_per_hour"] == 0.4 and priced["total_usd"] > 0.0
    assert priced["total_s"] == pytest.approx(sum(priced["seconds"].values()), abs=0.5)
    text = describe(plan, priced)
    assert "USD" in text and f"{plan.pairs} pairs" in text


def beside_the_path(distance_m: float) -> Recipe:
    """The head walks a metre in a second past a voice that stands ``distance_m`` from its path."""
    return recipe_of(
        1.0,
        [station("near", 1.3 - distance_m, 1.2)],
        [source("voice", "near_voice", [dwell("near", 0.0, 1.0)], 1.0)],
        [(0.0, (1.3, 1.7, 0.6)), (1.0, (1.3, 1.7, 1.6))],
    )


def test_a_near_source_makes_the_path_denser_and_a_step_no_cell_serves_is_flagged() -> None:
    held = assets()
    # At 0.33 m a cell is read from 5 cm: cells 0.15 m apart leave gaps, cells 0.10 m apart none.
    plan = make_plan(beside_the_path(0.33), held.triangles)
    cells = plan.record["cells"]
    assert cells["steps_refused_at_0.15_m"] > 0 and cells["dense_cells"] > 0
    assert cells["steps_refused_at_0.10_m"] == 0 and plan.record["fallback_steps"] == []
    assert (plan.cells.origin == ORIGIN_DENSE).sum() == cells["dense_cells"]
    # At 0.28 m the cells every 0.10 m are refused too: a cell on the head, where an array stands.
    plan = make_plan(beside_the_path(0.28), held.triangles)
    assert plan.record["cells"]["steps_refused_at_0.10_m"] > 0
    assert plan.record["cells"]["head_cells"] > 0 and plan.record["fallback_steps"] == []
    assert (plan.cells.origin == ORIGIN_HEAD).any()
    # At 0.20 m the mouth is inside the array's ball: the nearest cell alone, and the plan says so.
    plan = make_plan(beside_the_path(0.20), held.triangles)
    flagged = plan.record["fallback_steps"]
    assert flagged and all(name == "voice" for name, _ in flagged)
    chosen = plan.low["voice"]
    steps = [step for _, step in flagged]
    assert chosen.fallback[steps].all() and chosen.fallback.sum() == len(steps)
    assert set(chosen.mode[steps].tolist()) <= {MODE_EXACT, MODE_TRANSLATED}
    assert (chosen.cell[steps, 0] >= 0).all() and (chosen.cell[steps, 1] == -1).all()


def test_a_smoke_run_is_the_first_seconds_and_the_moving_sources_and_a_patch_is_882_cells() -> None:
    recipe = moving_recipe()
    held = assets()
    smoke = make_plan(recipe, held.triangles, Profile(seconds=0.3, sources=1))
    assert smoke.tracks.steps == 7 and list(smoke.tracks.sources) == ["voice"]
    assert smoke.record["pairs"] < make_plan(recipe, held.triangles).record["pairs"]
    # The window where most moves: the head from 0.1 s to 0.4 s, the voice from 0.15 s.
    start = busiest_window(recipe, 0.3, 1, every_s=0.1)
    assert start == pytest.approx(0.1)
    later = make_plan(recipe, held.triangles, Profile(seconds=0.3, sources=1, start_s=start))
    assert later.tracks.times[0] == pytest.approx(0.1) and later.tracks.steps == 7
    assert later.tracks.sources["voice"].moving[1:].all()
    patched = make_plan(recipe, held.triangles, Profile(patch=True))
    patch = patched.patch
    assert patch is not None and patch.cells.shape == (882, 3)
    assert sorted(set(np.round(patch.cells[:, 1], 3))) == [1.2, 1.7]
    assert patch.min_clearance_m >= 0.26
    assert np.linalg.norm(patched.tracks.positions[patch.source] - patch.centre) >= 1.2
    rows = set(range(patched.cells.count, patched.cells.count + 882))
    assert rows <= set(patched.heard_at[patch.source])
    assert patched.record["pairs"] == scene_pairs(patched) + 882
    assert patched.all_cells.shape[0] == patched.cells.count + 882
    # The receiver spheres of the tail: no two nearer than 0.80 m.
    chosen = tail_cells(patched.cells.position, patched.cells.kind)
    apart = np.linalg.norm(
        patched.cells.position[chosen][:, None] - patched.cells.position[chosen][None], axis=2
    )
    assert chosen.size >= 1 and (apart[np.triu_indices(chosen.size, 1)] >= 0.80).all()


def scene_pairs(plan: Plan) -> int:
    """The pairs of the scene's own cells, the patch's left out."""
    scene = plan.cells.count
    return sum(1 for listed in plan.heard_at for c in listed if c < scene)


def test_a_recipe_of_other_assets_is_named_by_key(tmp_path: Path) -> None:
    recipe = resting_recipe()
    held = assets()
    found = found_assets(recipe, held, voxel_low_key="free-field")
    assert mismatched(recipe, found) == [
        "export_sha256",
        "voxel_low_key",
        "mirror_scene_key",
        "calibration_key",
        "directivity.voice_v1",
    ]
    tree = recipe.to_dict()
    tree["assets"] = found
    assert mismatched(Recipe.from_dict(tree), found) == []
    held.save(tmp_path / "mirror")
    back = MirrorAssets.load(tmp_path / "mirror")
    assert back.catalogue.key == held.catalogue.key
    assert back.settings.record() == held.settings.record()
    assert (back.lead_s, back.gain) == (held.lead_s, held.gain)
    # The lead a pack carries is the whole number of samples the mirror's field was shifted by.
    assert back.pack_lead_s == pytest.approx(192 / 48000) and back.lead_samples == 192
    # The same from a run's mirror directory, whose report holds the alignment.
    write_derived(held.catalogue, tmp_path / "run" / "scene")
    np.save(tmp_path / "run" / "signature_S1.npy", held.signature)
    report = {"alignment": {"lead_s": 0.011, "gain": 0.02}}
    (tmp_path / "run" / "report_S1.json").write_text(json.dumps(report))
    found_there = MirrorAssets.from_run(tmp_path / "run")
    assert (found_there.lead_s, found_there.gain) == (0.011, 0.02)
    assert found_there.catalogue.key == held.catalogue.key
    assert MirrorAssets.from_run(tmp_path / "run", lead_s=0.004, gain=0.5).lead_samples == 192
    # A trace refuses the recipe unless its bundle allows the mismatch.
    plan = make_plan(recipe, held.triangles)
    build_bundle(tmp_path / "bundle", recipe, held, plan)
    pairs = FreeFieldPairs(plan.tracks.positions, plan.all_cells, tmp_path / "out")
    trace = Trace(tmp_path / "bundle", tmp_path / "out", engine=pairs, gpu=False, quiet=True)
    with pytest.raises(RuntimeError, match="calibration_key"):
        trace.run()
    assert (tmp_path / "out" / "campaign.failed").is_file() and pairs.solved == []


# --------------------------------------------------------------------------
# the trace
# --------------------------------------------------------------------------

#: How far the arrays stand from the cells asked for, as on a grid; and the cell that gets none.
OFF_THE_CELL = np.array([0.006, 0.0, 0.004])
REFUSED = 2


@pytest.fixture(scope="module")
def moving(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The moving scene traced in two runs: one stopped before the pack, one that finishes."""
    tmp = tmp_path_factory.mktemp("moving")
    recipe = moving_recipe()
    plan = make_plan(recipe, assets().triangles)
    centres: list[np.ndarray | None] = [cell + OFF_THE_CELL for cell in plan.all_cells]
    centres[REFUSED] = None
    first, first_pairs, _ = traced(tmp, recipe, rays=16, centres=centres)

    def stopped() -> Path:
        raise KeyboardInterrupt("the machine went away")

    first.write = stopped  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        first.run()
    failed = (tmp / "out" / "campaign.failed").is_file()
    second, second_pairs, _ = traced(tmp, recipe, rays=16, centres=centres)
    report = second.run()
    return {
        "tmp": tmp,
        "plan": plan,
        "centres": centres,
        "first_solved": list(first_pairs.solved),
        "first_report": first.report,
        "failed_then": failed,
        "second_solved": list(second_pairs.solved),
        "report": report,
        "pack": tmp / "out" / "pack.h5",
    }


def test_a_resumed_trace_recomputes_nothing_that_is_cached(moving: dict[str, Any]) -> None:
    first, report = moving["first_report"], moving["report"]
    out = moving["tmp"] / "out"
    assert moving["failed_then"] and not (out / "campaign.failed").exists()
    assert (out / "campaign.done").is_file()
    # The first run solved every pair, traced every table and levelled every pair ...
    assert len(moving["first_solved"]) == first["pairs"]["scene"] > 0
    assert first["low_pairs"]["solved"] == first["pairs"]["scene"]
    assert first["rays"]["histograms_traced"] > 0 and first["level"]["made"] > 0
    assert all("cached" not in record for record in first["paths"].values())
    # ... and the second none: the pairs, the paths, the histograms and the seams were on disk.
    assert moving["second_solved"] == []
    assert report["low_pairs"]["solved"] == 0
    assert report["low_pairs"]["cached"] == report["pairs"]["scene"]
    assert all(record == {"cached": True} for record in report["paths"].values())
    assert report["rays"]["histograms_traced"] == 0 and report["rays"]["histograms_read"] > 0
    assert report["level"]["made"] == 0 and report["level"]["read"] == report["pairs"]["scene"]
    assert set(report["timings_s"]) >= {"assign", "paths", "rays", "level", "write", "check"}
    # The noise behind the wall has a diffracted onset and no direct path: its pairs are not
    # read for the clock, whose pairs trail their direct sound by the lead.
    records = [json.loads(line) for line in (out / "level.jsonl").read_text().splitlines()]
    assert {record["direct"] for record in records} == {True, False}
    trail = report["level"]["trail_s_of_pairs_with_a_direct_path"]
    assert abs(trail["worst"] - report["level"]["lead_s"]) < 0.25e-3
    status = json.loads((out / "status.json").read_text())
    assert status["stage"] == "done"


def test_the_trace_writes_a_pack_the_reader_accepts_and_the_engine_renders(
    moving: dict[str, Any],
) -> None:
    plan, report = moving["plan"], moving["report"]
    with read_pack(moving["pack"], deep=True) as pack:
        h = pack.header
        assert (h.profile, h.steps, h.has_low, h.has_tail) == ("trace", 11, True, True)
        assert h.recipe_sha256 == plan.recipe_sha256 and sorted(pack.sources) == ["tap", "voice"]
        assert pack.mirror.lead_s == pytest.approx(192 / 48000)
        # A pack is physical: the mirror's gain is the alignment's over the field's unit.
        assert pack.mirror.alignment_gain == pytest.approx(0.02 / FIELD_UNIT_AT_1M)
        assert pack.air.enabled
        provenance = dict(h.provenance)
        assert provenance["cost"] == [] and provenance["solver"].startswith("free-field")
        assert provenance["low_pairs"]["cached"] == report["pairs"]["scene"]
        assert "calibration_key" in provenance["assets_mismatched"]
        # The cells are where the arrays stood, not where they were asked for.
        cells = pack.cells.position
        asked = plan.cells.position
        placed = np.arange(asked.shape[0]) != REFUSED
        np.testing.assert_allclose(cells[placed], (asked + OFF_THE_CELL)[placed])
        assert report["cells"]["without_an_array"] == 1
        assert report["cells"]["moved_m"]["worst"] == pytest.approx(
            np.linalg.norm(OFF_THE_CELL), abs=1e-4
        )
        voice, tap = pack.sources["voice"], pack.sources["tap"]
        for source in (voice, tap):
            assert source.low is not None and source.tail is not None
            low = source.low
            # No step reads the cell without an array; a head 7 mm off its cell is translated.
            assert REFUSED not in low.cell and REFUSED not in low.pair_cell
            assert MODE_TRANSLATED in low.mode and MODE_EXACT not in low.mode
            assert low.ir.shape[1:] == (64, 4800) and np.all(np.isfinite(low.seam_db))
            assert np.all(low.onset_s > pack.mirror.lead_s)
            # A step that reads one pair takes that pair's seam on top of the alignment's gain.
            one = (low.position_weight == 0.0) & (low.mode == MODE_TRANSLATED)
            assert one.any()
            np.testing.assert_allclose(
                source.level.high_gain_db[one],
                20.0 * np.log10(0.02 / FIELD_UNIT_AT_1M) + low.seam_db[low.pair[one, 0, 0]],
                atol=1e-5,
            )
            assert np.all(source.level.onset_s > pack.mirror.lead_s)
            assert np.all(source.tail.hist_cell < cells.shape[0])
        # The cell without an array leaves the two beside it 0.30 m apart. The noise is far and
        # reads both; the voice is 0.7 m away and, half way, may read neither: that one step
        # takes the nearest alone, and the pack says how many did.
        assert MODE_FUSED in tap.low.mode and MODE_FUSED not in voice.low.mode  # type: ignore[union-attr]
        assert report["fallback_steps"] == [["voice", 5]] and provenance["fallback_steps"] == 1
        # The voice walks its rail between two solved positions; the noise is behind the wall.
        assert voice.low is not None and (voice.low.position_weight > 0).sum() >= 4
        assert np.all(voice.early.kind <= 1) and np.any(tap.early.kind >= 2)
        assert not np.any(tap.early.kind == 0)
        assert (voice.kind, tap.kind, tap.subtype) == ("near_voice", "noise", "water")
        dry = np.random.default_rng(1).standard_normal(h.samples)
        engine = Engine(pack, {"voice": dry, "tap": dry})
        stem = engine.stem("voice")
    assert np.all(np.isfinite(stem)) and float(np.abs(stem).max()) > 0.0
    render = report["render"]
    assert render["finite"] and render["peak"] > 0.0 and render["card"] is False
    assert render["seconds"] == pytest.approx(0.5) and render["sources"] == 2


# --------------------------------------------------------------------------
# at rest, the validated hybrid
# --------------------------------------------------------------------------


def under_20_khz(signals: np.ndarray) -> np.ndarray:
    """The band both renderers hold: the engine fades its early part out from 22 to 24 kHz."""
    n = signals.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    keep = np.cos(0.5 * np.pi * np.clip((freqs - 20000.0) / 1000.0, 0.0, 1.0)) ** 2
    return np.asarray(np.fft.irfft(np.fft.rfft(signals, axis=-1) * keep, n, axis=-1))


def test_at_rest_the_pack_rendered_is_the_hybrid_of_the_present_pipeline(tmp_path: Path) -> None:
    """A source on its station, the head on a cell: ``render.engine`` against ``hybrid.blend``.

    The reference is made as a hybrid field is: the present pipeline's paths
    and histogram at the point, ``render_point``, the alignment's lead and
    gain, then ``blend`` with the wave response. The wave response is the
    free field engine's; the mirror's part is real.
    """
    recipe = resting_recipe()
    _, pairs, plan = traced(tmp_path, recipe)
    arguments = ["run", "--bundle", str(tmp_path / "bundle"), "--out", str(tmp_path / "out")]
    assert main([*arguments, "--free-field", "--cpu"]) == 0
    assert pairs.solved == [] and (tmp_path / "out" / "campaign.done").is_file()
    held = assets()
    settings = held.settings
    c = settings.sound_speed_m_s
    station_m, cell = plan.tracks.positions[0], plan.cells.position[0]
    lattice = trace_lattice(held.catalogue, station_m, cell[None, :], settings, Devices.host(1))
    # Order 2 is enough of the reference: its nine channels are the first nine of order 7.
    render = replace(settings.render, order=2, sample_rate_hz=float(FS))
    keywords: dict[str, Any] = {
        "tail_gain_db": np.asarray(settings.parameters.tail_gain_db, dtype=float),
        "receiver_radius_m": settings.rays.receiver_radius_m,
        "sound_speed_m_s": c,
        "seed": settings.seed,
        "signature": held.signature,
    }
    whole, _ = render_point(0, lattice.paths[0], lattice.histogram, render, **keywords)
    no_rays = replace(lattice.histogram, hits=np.zeros_like(lattice.histogram.hits))
    early_only, _ = render_point(0, lattice.paths[0], no_rays, render, **keywords)

    def aligned(signals: np.ndarray) -> np.ndarray:
        """As ``mirror.files.write_field`` writes a point: the lead, then the gain."""
        out = np.zeros_like(signals)
        out[:, held.lead_samples :] = signals[:, : signals.shape[1] - held.lead_samples]
        return np.asarray(out * held.gain)

    high, high_early = aligned(whole.signals), aligned(early_only.signals)
    # The wave response as a field holds it: the pair of the cache, the lead later.
    cached = pairs.cache.read(pairs.key_of(0, 0))
    aired = with_air(cached, LOW_RATE_HZ, Atmosphere(), sound_speed_m_s=c)
    wave = from_stored(delayed(aired, LOW_RATE_HZ, held.pack_lead_s), FS)
    crossover = Crossover()
    joined, record = blend(wave[:9], high, FS, crossover)
    # The hybrid taken apart by its own masks and window: its three parts add back to it.
    gain = 10.0 ** (seam_db(wave[0], high[0], FS, crossover) / 20.0)
    together = crossover.onset_window(wave[0], FS)
    power = crossover.masks(wave.shape[-1], FS, power=True)
    pressure = crossover.masks(wave.shape[-1], FS, power=False)

    def side(signals: np.ndarray, which: int) -> np.ndarray:
        spectrum = np.fft.rfft(signals * (1.0 - together), axis=-1) * power[which]
        spectrum = spectrum + np.fft.rfft(signals * together, axis=-1) * pressure[which]
        return np.asarray(np.fft.irfft(spectrum, n=signals.shape[-1], axis=-1))

    want = {
        "low": side(wave, 0),
        "early": side(high_early * gain, 1),
        "tail": side((high - high_early) * gain, 1),
    }
    peak = float(np.abs(joined).max())
    assert np.abs(want["low"][:9] + want["early"] + want["tail"] - joined).max() < 1e-12 * peak

    start = 4800
    click = np.zeros(int(0.5 * FS))
    click[start] = 1.0
    with read_pack(tmp_path / "out" / "pack.h5", deep=True) as pack:
        source = pack.sources["voice"]
        assert source.low is not None and set(source.low.mode.tolist()) == {MODE_EXACT}
        # The three scalars are the hybrid's own.
        seam = seam_db(wave[0], high[0], FS, crossover)
        assert float(source.low.seam_db[0]) == pytest.approx(seam, abs=1e-4)
        assert round(seam, 3) == record["seam_db"]
        onset = float(np.argmax(np.abs(wave[0]))) / FS
        assert float(source.low.onset_s[0]) == onset
        np.testing.assert_allclose(source.level.onset_s, onset, atol=1e-12)
        np.testing.assert_allclose(
            source.level.high_gain_db, 20.0 * np.log10(held.pack_gain) + seam, atol=1e-4
        )
        # The lead is in the response: the direct sound is its loudest sample, the lead after
        # its geometric time.
        straight = float(np.linalg.norm(station_m - cell)) / c
        assert onset == pytest.approx(straight + held.pack_lead_s, abs=2 / FS)
        # The pack is physical: over the crossover the direct sound is 1 / d of a unit source
        # (the seam aside), under it the pair over the field's unit.
        engine = Engine(pack, {"voice": click}, settings=RenderSettings(directivity=False))
        got = {
            name: engine.stem("voice", parts=(name,))[:, start:] * FIELD_UNIT_AT_1M for name in want
        }
    got["early"], got["tail"] = got["early"][:9], got["tail"][:9]
    count = click.size - start

    def padded(signals: np.ndarray) -> np.ndarray:
        out = np.zeros_like(joined)
        out[:, :count] = signals
        return out

    # The band under the crossover, all 64 channels: sample for sample. Measured 3.2e-5 of the peak.
    # The peak of the whole response on its 64 channels is the low part's, to 2e-4.
    assert np.abs(got["low"] - want["low"][:, :count]).max() < 1e-4 * np.abs(want["low"]).max()
    # The early part, under 20 kHz, sample for sample to the engine's own approximations (its
    # masks are filters of 85 ms, and a path takes the window's value at its arrival where
    # ``blend`` windows the samples). Measured, of the early part's peak: 1.2e-2 inside the
    # onset window, 1.7e-3 after it; the error's energy is -39.5 dB of the early part's.
    error = under_20_khz(padded(got["early"]) - want["early"])[:, : count - 2000]
    held_early = under_20_khz(want["early"])
    early_peak = float(np.abs(held_early).max())
    window_end = int(np.flatnonzero(together > 0.0)[-1]) + 1
    assert np.abs(error).max() < 2e-2 * early_peak
    assert np.abs(error[:, window_end:]).max() < 4e-3 * early_peak
    assert 10.0 * np.log10((error**2).sum() / (held_early**2).sum()) < -35.0
    # The tail is noise: two draws at the same energies, band by band. Measured 0.68 dB at worst.
    bands = octave_bank(FS).filters.shape[1]

    def read(signal: np.ndarray) -> np.ndarray:
        rows = octave_filter_rows(np.repeat(signal[None], bands, 0), FS, np.arange(bands))
        return np.asarray((rows**2).sum(axis=1))

    level = 10.0 * np.log10(read(got["tail"][0]) / read(want["tail"][0, :count]))
    assert np.abs(level).max() < 1.0, level


def test_pairs_on_another_clock_than_the_mirror_stop_the_trace(tmp_path: Path) -> None:
    """The first real pack: its pairs 10.67 ms before its mirror, and a line in the log."""
    held = assets()
    # The seeded fault: pairs that already carry the lead, as a field's low band does.
    trace, _, _ = traced(tmp_path, resting_recipe(), lead_s=held.pack_lead_s)
    with pytest.raises(RuntimeError, match="not on one clock"):
        trace.run()
    assert (tmp_path / "out" / "campaign.failed").is_file()
    assert not (tmp_path / "out" / "pack.h5").exists()


def test_a_pair_stored_without_its_lead_rings_round_to_its_end() -> None:
    """A source 0.8 m away: its masks ring before 2.3 ms, which a transform brings to the end."""
    source, cell = np.array([[0.6, 1.7, 0.8]]), np.array([[1.4, 1.7, 0.8]])
    pairs = FreeFieldPairs(source, cell, Path("unused"), gain=FIELD_UNIT_AT_1M)
    cached = pairs.response(0, 0)
    keywords: dict[str, Any] = {"sound_speed_m_s": C}

    def end_db(lead_s: float) -> float:
        stored, _, _ = pair_low(cached, Crossover(), Atmosphere(), lead_s=lead_s, **keywords)
        return float(10.0 * np.log10(np.sum(stored[0, -400:] ** 2) / np.sum(stored[0] ** 2)))

    # The last 100 ms of a response that has no room in it. Measured -44 dB without the lead
    # (on the nearest pair of the first real pack the level rose by 31 dB there) and -64 dB
    # with it.
    assert end_db(0.0) > -50.0
    assert end_db(512 / FS) < -60.0
    # On the pack's clock and scale: the onset the lead later, a unit source 1 / d at 0.8 m.
    stored, onset, aired = pair_low(
        cached, Crossover(), Atmosphere(), lead_s=512 / FS, unit_at_1m=FIELD_UNIT_AT_1M, **keywords
    )
    assert onset == pytest.approx(0.8 / C + 512 / FS, abs=1 / FS)
    # The seam is read on the pair on the pack's clock: silent while the lead lasts, but for
    # the ring of a delay that is not a whole sample (2e-3 of the peak, at the Nyquist
    # frequency, which the masks remove).
    assert aired.shape == (4800,) and np.abs(aired[:40]).max() < 1e-2 * np.abs(aired).max()
    # An impulse of one through the low pressure mask peaks at ``unit``: the pack holds it over d.
    low_mask = Crossover().masks(57600, FS, power=False)[0]
    unit = float(np.max(np.abs(np.fft.irfft(low_mask, n=57600))))
    # Less what the engine's own high pass from 80 to 160 Hz takes of it: 12 per cent.
    assert float(np.abs(from_stored(stored[:1])).max()) == pytest.approx(
        0.88 * unit / 0.8, rel=0.03
    )
    # The seeded fault: left on the field's scale, the same response is 31.6 dB under it.
    raw, _, _ = pair_low(cached, Crossover(), Atmosphere(), lead_s=512 / FS, **keywords)
    assert float(np.abs(raw).max() / np.abs(stored).max()) == pytest.approx(
        FIELD_UNIT_AT_1M, rel=1e-6
    )
    assert 20.0 * np.log10(FIELD_UNIT_AT_1M) == pytest.approx(-31.6, abs=0.05)


# --------------------------------------------------------------------------
# the driver and the command line
# --------------------------------------------------------------------------


def test_a_dry_run_prints_the_plan_and_its_cost_and_rents_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverberate.gpu import onebox
    from reverberate.scenes import save_recipe

    def rented(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a dry run rented")

    monkeypatch.setattr(onebox, "run", rented)
    recipe = moving_recipe()
    save_recipe(recipe, tmp_path / "recipe.json")
    assets().save(tmp_path / "mirror")
    arguments = ["rent", "--recipe", str(tmp_path / "recipe.json"), "--home", str(tmp_path / "h")]
    arguments += ["--mirror", str(tmp_path / "mirror"), "--rate", "0.4"]
    assert main([*arguments, "--dry-run", "--smoke", "0.3", "--smoke-sources", "1"]) == 0
    printed = capsys.readouterr().out
    assert "dry run: nothing built, nothing rented" in printed
    assert "cost at 0.4 USD/h" in printed and "sources: 1 (voice)" in printed
    assert "calibration_key" in printed and not (tmp_path / "h").exists()
    # The bundle alone: what the machine reads, and no pairs campaign without an export.
    arguments = ["bundle", "--recipe", str(tmp_path / "recipe.json"), "--out", str(tmp_path / "b")]
    assert main([*arguments, "--mirror", str(tmp_path / "mirror"), "--patch", "1.2", "1.3"]) == 0
    campaign = json.loads((tmp_path / "b" / "campaign.json").read_text())
    assert campaign["kind"] == "scene-trace" and campaign["trace"]["profile"]["patch"]
    # Priced for the batched low band solver, at the rate of the card it was measured on.
    assert campaign["estimate"]["billed_rate_usd_per_hour"] == 0.136
    assert campaign["estimate"]["low_engine"] == "lowband"
    assert campaign["estimate"]["measured_on"] == "1 x RTX 3080 20 GB"
    assert not (tmp_path / "b" / "pairs").exists()
    # The form the machine writes the pack in: the bins in 16 bits unless told, and said.
    assert campaign["trace"]["low_levers"] == "bins,int16"
    assert campaign["estimate"]["low_levers"] == "bins,int16"
    assert "the pack's low band is written as bins,int16 (--low-levers)" in capsys.readouterr().out
    assert main([*arguments, "--mirror", str(tmp_path / "mirror"), "--low-levers", "none"]) == 0
    campaign = json.loads((tmp_path / "b" / "campaign.json").read_text())
    assert campaign["trace"]["low_levers"] == "none" and "pair_cache" not in campaign["trace"]
    assert "low_levers" not in campaign["estimate"]
    assert "written as low/ir, the samples" in capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "b" / "trace").iterdir()) == [
        "mirror",
        "plan.json",
        "plan.npz",
        "positions.npy",
        "recipe.json",
    ]
    # A rental needs the storey's export, and says so before it asks for a machine.
    with pytest.raises(SystemExit, match="export"):
        launch(recipe, assets(), tmp_path / "h", say=lambda _: None)


def test_a_run_left_on_its_machine_is_told_how_to_resume_and_what_the_machine_holds() -> None:
    """Two scenes ended ``campaign.failed`` after their solves and the driver said one line."""
    words = ["rent", "--recipe", "R with space.json", "--home", "H", "--mirror", "M"]
    words += ["--gpus", "8", "--yes", "--max-hours", "10", "--instance", "7", "--plan-offers"]
    # The same words, without what rents, on the instance the last line names.
    assert resume_command(words) == (
        "python -m reverberate.trace rent --recipe 'R with space.json' --home H --mirror M"
        " --gpus 8 --max-hours 10 --instance {instance}"
    )
    asked: list[str] = []

    def machine(answer: str) -> Any:
        def run_on(_: Any, command: str, *, what: str, timeout: Any = None) -> str:
            asked.append(command)
            return answer

        return run_on

    # Scene A as it stood: every pair solved and levelled, no pack, stopped by the check.
    lines = machine_holds(
        "m",
        {"pairs": 16887},
        run_on=machine(
            "pairs 18219\nearly 15\ntails 848\nlevel 18219\nrows 0\npack 0\n"
            'failed RuntimeError("the low band and the mirror are not on one clock")\n'
        ),
    )
    assert "/root/campaign/out" in asked[0] and "campaign.failed" in asked[0]
    assert lines[0] == (
        "on the machine: 18219 pairs solved (the plan counts 16887), 15 early tables,"
        " 848 histograms, 18219 pairs levelled, 0 blocks of the pack's rows, no pack"
    )
    assert lines[1].startswith("a resume keeps all of it and makes again: the pack's write")
    assert "solves" not in lines[1] and "not on one clock" in lines[2]
    # A machine that died in its solves: what is missing is what a resume pays for.
    lines = machine_holds(
        "m", {"pairs": 16887}, run_on=machine("pairs 7831\nearly 0\ntails 0\nlevel 0\npack 0\n")
    )
    assert "the solves of about 9056 pairs; the early trace; the rays; the levelling" in lines[1]
    assert len(lines) == 2
    # Done, its pack on it: nothing but the fetch.
    lines = machine_holds(
        "m",
        {"pairs": 4},
        run_on=machine("pairs 4\nearly 3\ntails 9\nlevel 4\nrows 1\npack 9508553242\n"),
    )
    assert "a pack of 9.51 GB" in lines[0] and "nothing but the fetch" in lines[1]


def test_the_cost_records_carry_the_rate_and_are_stamped_into_the_pack(tmp_path: Path) -> None:
    import h5py

    report = {
        "device": {"gpu": "RTX 3090", "devices": 1},
        "timings_s": {"voxelise": 60.0, "plan": 30.0, "solve": 510.0, "paths": 12.0, "check": 8.0},
    }
    records = cost_records(
        report, billed_rate_usd_per_hour=0.36, instance=51, fetch_s=40.0, billed_s=1800.0
    )
    by_stage = {record["stage"]: record for record in records}
    assert by_stage["low"]["seconds"] == 600.0 and by_stage["transfer"]["seconds"] == 40.0
    assert by_stage["rental"]["seconds"] == 1800.0 - 600.0 - 12.0 - 8.0 - 40.0
    assert all(record["billed_rate_usd_per_hour"] == 0.36 for record in records)
    assert sum(record["usd"] for record in records) == pytest.approx(0.18, abs=1e-3)
    assert by_stage["low"]["card"] == "RTX 3090" and by_stage["low"]["instance"] == 51
    with h5py.File(tmp_path / "pack.h5", "w") as handle:
        handle.attrs["provenance_json"] = json.dumps({"cost": [], "solver": "x"})
    stamp_cost(tmp_path / "pack.h5", records)
    with h5py.File(tmp_path / "pack.h5", "r") as handle:
        held = json.loads(handle.attrs["provenance_json"])
    assert held["cost"] == records and held["solver"] == "x"


def test_the_machines_command_reads_that_a_bundle_is_a_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverberate.accel.cli import main as accel
    from reverberate.trace import run as run_module

    called: dict[str, Any] = {}
    monkeypatch.setattr(
        run_module, "run_trace", lambda bundle, out, **kw: called.update(bundle=bundle, **kw)
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "campaign.json").write_text(json.dumps({"kind": "scene-trace"}))
    arguments = ["campaign", "--bundle", str(tmp_path / "b"), "--out", str(tmp_path / "o")]
    assert accel([*arguments, "--cpu", "--solvers", "2"]) == 0
    assert called["bundle"] == tmp_path / "b" and called["gpu"] is False and called["solvers"] == 2


def test_the_card_engine_drives_the_pairs_campaign_stage_by_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Campaign:
        """What of ``PairsCampaign`` the trace reads, answering as one would."""

        def __init__(self, **given: Any) -> None:
            self.given = given
            self.cache = PairCache(tmp_path / "pairs", "grid")
            self.keys = {"low": "grid"}
            self.spec = {"solver": "an engine"}
            self.staged: list[str] = []
            self.heard_at: list[list[int]] = []

        def stage(self, name: str, work: Any) -> Any:
            self.staged.append(name)
            return work()

        def voxelise(self) -> dict[str, Any]:
            return {"low": {"cached": True}}

        def place(self) -> dict[str, Any]:
            return {"centres": [[0.1, 1.7, 0.2], None], "cells": 2}

        def solve(self) -> list[dict[str, Any]]:
            return [{"source": 0, "cells": 2}, {"source": 1, "cells": 0, "skipped": True}]

        def key_of(self, position: int, cell: int) -> str:
            return f"{position}-{cell}"

    monkeypatch.setattr(pairs_module, "PairsCampaign", Campaign)
    engine = CardPairs(tmp_path / "b", tmp_path / "o", pffdtd_dir=tmp_path, solvers=2)
    assert (engine.voxel_low_key, engine.solver) == ("grid", "an engine")
    campaign: Any = engine.campaign
    assert campaign.given["solvers"] == 2 and engine.key_of(3, 4) == "3-4"
    centres = engine.place()
    assert centres[1] is None and centres[0] is not None and centres[0].tolist() == [0.1, 1.7, 0.2]
    counts = engine.solve([[1, 0], [0]])
    assert campaign.heard_at == [[0, 1], [0]]
    assert campaign.staged == ["voxelise", "plan", "solve"]
    assert counts == {"solved": 2, "cached": 1, "source_positions_solved": 1}


def test_pairs_the_bundle_carries_are_not_solved_and_the_patch_is_solved_from_one_source(
    tmp_path: Path,
) -> None:
    recipe = moving_recipe()
    first, first_pairs, plan = traced(tmp_path / "a", recipe)
    first.assign(first_pairs.place())
    first.solve()
    assert first.report["low_pairs"] == {**first.report["low_pairs"], "carried": 0}
    assert len(first_pairs.solved) == plan.pairs > 0
    # A second machine, whose bundle carries what the first one's cache holds.
    second, second_pairs, _ = traced(tmp_path / "b", recipe)
    shutil.copytree(tmp_path / "a" / "out" / "pairs", tmp_path / "b" / "bundle" / "pairs_cache")
    second.assign(second_pairs.place())
    second.solve()
    assert second_pairs.solved == [] and second.report["low_pairs"]["carried"] == plan.pairs
    # The patch: its cells are heard from its one source, and are no cell of the scene.
    held = assets()
    patched = make_plan(recipe, held.triangles, Profile(patch=True))
    build_bundle(tmp_path / "c" / "bundle", recipe, held, patched, allow_asset_mismatch=True)
    pairs = FreeFieldPairs(patched.tracks.positions, patched.all_cells, tmp_path / "c" / "out")
    third = Trace(tmp_path / "c" / "bundle", tmp_path / "c" / "out", engine=pairs, gpu=False)
    third.journal.quiet = True
    third.assign(pairs.place())
    scene = patched.cells.count
    assert third.cells.shape[0] == scene and third.report["cells"]["patch"] == 882
    assert third.report["pairs"] == {"scene": plan.pairs, "all": plan.pairs + 882}
    assert set(range(scene, scene + 882)) <= set(third.heard_at[patched.patch.source])  # type: ignore[union-attr]


def test_a_seat_is_a_cell_of_its_own_and_a_rise_is_solved_on_its_vertical_rail() -> None:
    rise = {"type": "rise", "station": "chair", "to": "standing", "start_s": 0.2, "end_s": 0.6}
    recipe = recipe_of(
        0.8,
        [station("chair", 0.6, 0.8, "seat"), station("sofa", 1.4, 1.3, "seat")],
        [
            source(
                "voice",
                "far_voice",
                [dwell("chair", 0.0, 0.2, "seated"), rise, dwell("chair", 0.6, 0.8)],
                0.8,
            )
        ],
        [(0.0, (1.4, 1.2, 1.3)), (0.8, (1.4, 1.2, 1.3))],
    )
    plan = make_plan(recipe, assets().triangles)
    # The listener sits: one cell, on the seat, at the seated height, and a tail sphere on it.
    assert plan.cells.position.tolist() == [[1.4, 1.2, 1.3]]
    assert plan.cells.kind.tolist() == [1] and plan.tail_cells.tolist() == [0]
    assert plan.record["modes"] == {"exact": 17, "translated": 0, "fused": 0}
    # The voice rises through the rungs of its seat, 8 cm apart and the standing height.
    voice = plan.tracks.sources["voice"]
    heights = sorted(set(np.round(plan.tracks.positions[:, 1], 3)))
    assert heights == [1.2, 1.28, 1.36, 1.44, 1.52, 1.6, 1.68, 1.7]
    assert np.all(plan.tracks.positions[:, [0, 2]] == [0.6, 0.8])
    assert plan.record["source_positions"] == low_band_source_positions(recipe).count == 8
    assert voice.moving[4:12].all() and not voice.moving[:4].any() and not voice.moving[13:].any()
    assert (voice.weight[voice.moving] > 0).any() and plan.pairs == 8
    # Its rays leave from the seat and from the standing height above it.
    assert plan.record["tail_sites"] == 2 and plan.sites["voice"].count == 4


def test_what_came_home_is_checked_stamped_and_installed(
    moving: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = moving["tmp"] / "out"
    pulled = tmp_path / "home" / "pulled"
    pulled.mkdir(parents=True)
    shutil.copy2(out / "pack.h5", pulled / "pack.h5")
    shutil.copy2(out / "trace_report.json", pulled / "trace_report.json")
    shutil.copytree(out / "pairs", pulled / "pairs")
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    record = {"outcome": "done", "instance": 7, "hours_billed": 0.5, "cost_usd": 0.2, "fetch_s": 9}
    found = finish(tmp_path / "home", record)
    assert found["pack"]["steps"] == 11 and sorted(found["pack"]["sources"]) == ["tap", "voice"]
    assert found["pairs"] == {"installed": moving["report"]["pairs"]["scene"], "published": 0}
    assert found["usd"] == pytest.approx(0.2, abs=1e-3) and found["render"]["finite"]
    assert json.loads((tmp_path / "home" / "cost.json").read_text())["usd"] == found["usd"]
    with read_pack(pulled / "pack.h5") as pack:
        cost = pack.header.provenance["cost"]
    assert {r["stage"] for r in cost} >= {"paths", "rays", "level", "write", "transfer", "rental"}
    assert all(r["billed_rate_usd_per_hour"] == 0.4 and r["instance"] == 7 for r in cost)
    cache = PairCache.local("free-field")
    assert len(cache.records()) == moving["report"]["pairs"]["scene"]
    # Without a bill there is no rate, and a cost without its rate is not written.
    assert finish(tmp_path / "home", {"outcome": "deadline"})["cost"] == []


def test_the_one_command_bundles_rents_and_brings_the_pack_home(
    moving: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``launch`` with the rental replaced: what the machine is handed, and what comes home."""
    import reverberate.experiments.run as run_module
    from reverberate.gpu import onebox

    class Spec:
        key, ppw, nh = "key1500", 10.5, 4

    monkeypatch.setattr(run_module, "scene_spec", lambda models, scene, fmax: (Spec(), None, 0))
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    export = tmp_path / "earlier" / "storey"
    export.mkdir(parents=True)
    (export / "manifest.json").write_text("{}")
    (export / "apartment_full.json").write_text("{}")
    (export.parent / "materials").mkdir()
    tree = moving_recipe().to_dict()
    tree["dwelling"]["scene_id"] = "104862621_172226772"
    recipe = Recipe.from_dict(tree)
    held = assets()
    plan = make_plan(recipe, held.triangles)
    # One of the plan's pairs is in this machine's cache already, as the batched solver
    # keys it on the bundle's grid: the bundle carries it. The same pair as the present
    # engine keys it is another engine's and is not carried: a machine would not ask for it.
    from reverberate.wave.lowband.pairs import solver_name
    from reverberate.wave.lowband.scheme import CARTESIAN

    def keyed(solver: str) -> str:
        return pair_key(
            "key1500",
            plan.tracks.positions[0],
            plan.all_cells[plan.heard_at[0][0]],
            encoder=pairs_module.encoder_record(7, 10, 1500.0),
            solver=solver,
        )

    known, other = keyed(solver_name(CARTESIAN, 10.5)), keyed(pairs_module.SOLVER)
    for key in (known, other):
        PairCache.local("key1500").write(key, np.zeros((64, 4800), dtype=np.float32), {})
    home = tmp_path / "home"
    seen: dict[str, Any] = {}

    def rented(bundle: Path, where: Path, **given: Any) -> dict[str, Any]:
        """The machine: it is handed a trace's bundle, and leaves a pack and its report."""
        seen.update(given, bundle=bundle)
        (where / "pulled").mkdir(parents=True)
        shutil.copy2(moving["pack"], where / "pulled" / "pack.h5")
        shutil.copy2(moving["tmp"] / "out" / "trace_report.json", where / "pulled")
        return {"outcome": "done", "instance": 9, "hours_billed": 0.25, "cost_usd": 0.1}

    monkeypatch.setattr(onebox, "run", rented)
    said: list[str] = []
    with pytest.raises(SystemExit, match="other assets: export_sha256"):
        launch(recipe, held, home, models_from=export, yes=True, say=said.append)
    assert "bundle" not in seen
    result = launch(
        recipe,
        held,
        home,
        models_from=export,
        allow_asset_mismatch=True,
        yes=True,
        avoid=[41, 42],
        gpu="3090",
        max_dph=0.5,
        say=said.append,
    )
    assert seen["bundle"] == home / "bundle" and seen["avoid"] == [41, 42]
    assert (seen["gpu"], seen["max_dph"], seen["yes"], seen["fetch_cache"]) == (
        "3090",
        0.5,
        True,
        False,
    )
    campaign = json.loads((home / "bundle" / "campaign.json").read_text())
    # What ``onebox.campaign_need`` sizes the machine by is read from the trace's own campaign.
    assert campaign["kind"] == "scene-trace" and campaign["bands"]["low"]["cache_key"] == "key1500"
    assert campaign["model_json"] == "pairs/models/storey/apartment_full.json"
    assert (home / "bundle" / campaign["model_json"]).is_file()
    assert campaign["pairs_carried"] == 1 and campaign["trace"]["allow_asset_mismatch"] is True
    assert PairCache(home / "bundle" / "pairs_cache", "key1500").has(known)
    assert not PairCache(home / "bundle" / "pairs_cache", "key1500").has(other)
    assert campaign["trace"]["low"] == {
        "engine": "lowband",
        "scheme": "cartesian",
        "ppw": None,
        "voxel_low_key": "key1500",
        "solver": solver_name(CARTESIAN, 10.5),
        "bundle_grid": True,
    }
    assert campaign["trace"]["allowed_mismatch"] == []
    # The machine is told its engine, the pairs come home as the run lasts, the early
    # tables stay, and the offers are priced for this plan.
    assert seen["campaign_args"] == "--low-engine lowband" and seen["sync"] == ("pairs",)
    # The driver is given what its last lines say of a machine it leaves rented.
    assert seen["resume_command"] == "" and callable(seen["inventory"])
    priced = seen["predict"](
        SimpleNamespace(
            gpu_name="RTX 3090", num_gpus=4, gpu_ram_gb=24.0, dph_total=0.8, cpu_cores=36.0
        )
    )
    assert priced["queue"] and priced["seconds"]["transfer_pack"] > 0
    assert campaign["trace"]["low_levers"] == "bins,int16"
    assert (
        seen["leave"] == ()
        and seen["also"] == ()
        and callable(seen["predict"])
        and seen["hours"] is None
    )
    # With the present engine the bundle carries that engine's pair, and no offer is priced.
    launch(
        recipe,
        held,
        tmp_path / "home_pffdtd",
        models_from=export,
        allow_asset_mismatch=True,
        yes=True,
        low_engine="pffdtd",
        say=said.append,
    )
    carried = PairCache(tmp_path / "home_pffdtd" / "bundle" / "pairs_cache", "key1500")
    assert carried.has(other) and not carried.has(known) and seen["predict"] is None
    pairs = json.loads((home / "bundle" / "pairs" / "campaign.json").read_text())
    assert pairs["kind"] == pairs_module.KIND and pairs["pairs"] == plan.pairs
    assert np.array_equal(np.load(home / "bundle" / "pairs" / "cells.npy"), plan.all_cells)
    # Home: the pack is read, and its cost is the rental's bill at the rental's rate, to the
    # rounding of its records: each is written to 0.0001 USD, and how the bill splits between
    # them is how long each stage took on the machine that ran this test.
    assert result["home"]["pack"]["steps"] == 11
    assert result["home"]["usd"] == pytest.approx(0.1, abs=5e-4)
    with read_pack(home / "pulled" / "pack.h5") as pack:
        assert {r["billed_rate_usd_per_hour"] for r in pack.header.provenance["cost"]} == {0.4}
    assert any("cost at 1.74 USD/h" in line for line in said)


# --------------------------------------------------------------------------
# what a second machine does not do again
# --------------------------------------------------------------------------


def test_a_host_that_dies_loses_its_last_pairs_and_the_next_machine_solves_those_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two machines, one recipe: what came home from the first is carried to the second."""
    from reverberate.trace.bundle import carry_pairs

    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    recipe = moving_recipe()
    first, first_pairs, plan = traced(tmp_path / "a", recipe)
    first.assign(first_pairs.place())
    first.solve()
    solved = list(first_pairs.solved)
    assert len(solved) == plan.pairs > 3
    # The host dies. Home is what the last homecoming brought: every pair but the last
    # three, two of which never left and one of which was cut in its transfer.
    home = tmp_path / "home"
    shutil.copytree(tmp_path / "a" / "out" / "pairs", home / "pulled" / "pairs")
    arrived = PairCache(home / "pulled" / "pairs", "free-field")
    lost = [first_pairs.key_of(*pair) for pair in solved[-3:]]
    for key in lost[:2]:
        arrived.path(key).unlink()
    arrived.path(lost[2]).write_bytes(arrived.path(lost[2]).read_bytes()[:200])
    found = finish(home, {"outcome": "instance vanished", "instance": 7})
    assert found["pairs"] == {"installed": len(solved) - 3, "published": 0}
    assert found["pairs_damaged"] == 1 and found["pack"] is None
    # The second machine's bundle carries them under the keys its engine asks for ...
    second, second_pairs, _ = traced(tmp_path / "b", recipe)
    keys = [
        second_pairs.key_of(position, cell)
        for position, cells in enumerate(plan.heard_at)
        for cell in cells
    ]
    carried = carry_pairs(
        tmp_path / "b" / "bundle" / "pairs_cache", PairCache.local("free-field"), keys
    )
    assert carried == len(solved) - 3
    # ... and it solves the three that were lost, and no pair twice.
    second.assign(second_pairs.place())
    second.solve()
    assert sorted(second_pairs.solved) == sorted(solved[-3:])
    assert second.report["low_pairs"]["carried"] == len(solved) - 3
    assert not set(second_pairs.solved) & set(solved[:-3])
    for pair in solved:
        key = second_pairs.key_of(*pair)
        assert np.array_equal(second_pairs.cache.read(key), first_pairs.cache.read(key))


def test_the_low_band_is_written_a_pair_at_a_time_and_the_file_is_the_same(
    moving: dict[str, Any], tmp_path: Path
) -> None:
    import h5py

    from reverberate.render import pack as pack_module
    from reverberate.trace.run import PairRows

    # In the pack: each row is the pair of its key, through the air and the masks.
    held = assets()
    cache = PairCache(moving["tmp"] / "out" / "pairs", "free-field")
    rows = 0
    with read_pack(moving["pack"]) as pack:
        atmosphere = pack.air.atmosphere
        for source in pack.sources.values():
            assert source.low is not None
            ir = source.low.ir
            # The default form: the bins in 16 bits, 78 dB under the response and more.
            assert isinstance(ir, CompactIr) and ir.dtype == np.float32
            for row, key in enumerate(source.low.pair_key):
                want = pair_low(
                    cache.read(key.decode()),
                    pack.crossover,
                    atmosphere,
                    sound_speed_m_s=held.settings.sound_speed_m_s,
                    lead_s=held.pack_lead_s,
                    unit_at_1m=FIELD_UNIT_AT_1M,
                )[0]
                assert np.abs(ir[row] - want).max() < 2e-4 * np.abs(want).max()
                rows += 1
    assert rows > 0
    # The file: a table handed over a row at a time is, byte for byte, the table handed whole.
    table = np.random.default_rng(5).standard_normal((7, 64, 48)).astype(np.float32)
    asked: list[int] = []

    def row_of(index: int) -> np.ndarray:
        asked.append(index)
        return np.asarray(table[index])

    class Low:
        def __init__(self, ir: Any) -> None:
            self.ir = ir

    spec = {"ir": pack_module._LOW["ir"]}
    for name, value in (("whole", table), ("rows", PairRows((7, 64, 48), row_of))):
        with h5py.File(tmp_path / f"{name}.h5", "w") as handle:
            pack_module._write_group(handle.create_group("low"), spec, Low(value))
    assert (tmp_path / "whole.h5").read_bytes() == (tmp_path / "rows.h5").read_bytes()
    assert asked == list(range(7)), "each row asked once, in order: one pair is all that is held"
    with pytest.raises(ValueError, match="a row of"):
        PairRows((7, 64, 48), lambda index: table[index, :3])[0]


def test_the_pack_and_the_pair_cache_are_compact_unless_told_and_say_which(
    tmp_path: Path,
) -> None:
    """``--low-levers``: the bins in 16 bits by default, ``none`` for the samples as before."""
    held = assets()
    made: dict[str, Any] = {}
    for name, levers in (("compact", None), ("plain", "none"), ("decay", "bins,int16,decay=60")):
        trace, pairs, _ = traced(tmp_path / name, resting_recipe(), rays=16, low_levers=levers)
        trace.check_mode = "read"
        trace.run()
        told = json.loads((tmp_path / name / "bundle" / "campaign.json").read_text())["trace"]
        files = sorted(p.suffix for p in (tmp_path / name / "out" / "pairs").rglob("*.np[yz]"))
        with read_pack(tmp_path / name / "out" / "pack.h5") as pack:
            (source,) = pack.sources.values()
            assert source.low is not None
            made[name] = {
                "told": told,
                "files": set(files),
                "ir": np.asarray(source.low.ir[:]),
                "compact": isinstance(source.low.ir, CompactIr),
                "provenance": pack.header.provenance,
                "keys": [key.decode() for key in source.low.pair_key],
                "bytes": (tmp_path / name / "out" / "pack.h5").stat().st_size,
                "cache": pairs.cache,
            }
    compact, plain, decay = made["compact"], made["plain"], made["decay"]
    # The bundle says the form, the machine writes it, the pack's provenance repeats it.
    assert compact["told"]["low_levers"] == "bins,int16"
    assert compact["told"]["pair_cache"] == "bins,int16"
    assert compact["compact"] and compact["files"] == {".npz"}
    assert compact["provenance"]["low_levers"]["sample"] == "int16"
    assert compact["provenance"]["pair_cache"] == "bins,int16"
    # ``none``: ``low/ir`` and a cache of samples, and a provenance that names no lever.
    assert plain["told"]["low_levers"] == "none" and "pair_cache" not in plain["told"]
    assert not plain["compact"] and plain["files"] == {".npy"}
    assert "low_levers" not in plain["provenance"] and "pair_cache" not in plain["provenance"]
    cache = plain["cache"]
    for row, key in enumerate(plain["keys"]):
        want = pair_low(
            cache.read(key),
            Crossover(),
            Atmosphere(),
            sound_speed_m_s=held.settings.sound_speed_m_s,
            lead_s=held.pack_lead_s,
            unit_at_1m=FIELD_UNIT_AT_1M,
        )[0]
        assert np.array_equal(plain["ir"][row], np.asarray(want, dtype=np.float32))
    # One trace heard both ways: 70 dB and more under the response, in half the bytes.
    assert compact["keys"] == plain["keys"]
    error = np.abs(compact["ir"] - plain["ir"]).max() / np.abs(plain["ir"]).max()
    assert error < 3e-4, error
    assert compact["bytes"] < 0.6 * plain["bytes"]
    # A pack's own levers never reach the cache, which serves every pack of its dwelling.
    assert decay["told"]["low_levers"] == "bins,int16,decay=60"
    assert decay["told"]["pair_cache"] == "bins,int16"
    assert decay["provenance"]["low_levers"]["decay_db"] == 60.0
    with pytest.raises(ValueError, match="keeps every degree whole"):
        PairCache(tmp_path / "c", "grid", levers="bins,decay=60")


def test_a_pair_cache_reads_both_forms_and_a_compact_pair_is_the_pair(tmp_path: Path) -> None:
    source, cell = np.array([[0.6, 1.5, 0.8]]), np.array([[2.9, 1.6, 1.3]])
    response = FreeFieldPairs(source, cell, tmp_path / "ff", gain=FIELD_UNIT_AT_1M).response(0, 0)
    plain = PairCache(tmp_path / "cache", "grid")
    compact = PairCache(tmp_path / "cache", "grid", levers="bins,int16")
    plain.write("aa" + "0" * 62, response, {"solver": "s"})
    compact.write("bb" + "0" * 62, response, {"solver": "s"})
    assert plain.path("aa" + "0" * 62).suffix == ".npy"
    assert compact.path("bb" + "0" * 62).suffix == ".npz"
    # Either cache reads either pair: the form is the file's, not the reader's.
    for cache in (plain, compact):
        assert cache.has("aa" + "0" * 62) and cache.has("bb" + "0" * 62)
        assert np.array_equal(cache.read("aa" + "0" * 62), response)
        back = cache.read("bb" + "0" * 62)
        assert back.shape == response.shape and back.dtype == np.float32
        assert np.abs(back - response).max() < 1e-4 * np.abs(response).max()
        for key in ("aa" + "0" * 62, "bb" + "0" * 62):
            first = cache.first_channel(key)
            assert first.shape == (1, response.shape[1])
            assert np.array_equal(first[0], cache.read(key)[0])
    assert (
        compact.path("bb" + "0" * 62).stat().st_size
        < 0.52 * plain.path("aa" + "0" * 62).stat().st_size
    )
    # Into the pack's stored form, the compact pair is the plain one to 80 dB.
    stored = [
        pair_low(
            cache.read(key),
            Crossover(),
            Atmosphere(),
            sound_speed_m_s=C,
            lead_s=512 / FS,
            unit_at_1m=FIELD_UNIT_AT_1M,
        )
        for cache, key in ((plain, "aa" + "0" * 62), (compact, "bb" + "0" * 62))
    ]
    assert stored[0][1] == stored[1][1], "the onset did not move"
    assert np.abs(stored[0][0] - stored[1][0]).max() < 1e-4 * np.abs(stored[0][0]).max()
    # A pair goes from cache to cache in the form it is in, and is installed so.
    other = PairCache(tmp_path / "other", "grid")
    other.adopt("bb" + "0" * 62, compact.path("bb" + "0" * 62), {"solver": "s"})
    assert other.path("bb" + "0" * 62).suffix == ".npz"
    assert other.records()["bb" + "0" * 62]["solver"] == "s"


def test_another_low_grid_reuses_the_sources_early_tables_and_differs_by_one_key_alone(
    tmp_path: Path,
) -> None:
    """Two traces of one recipe whose arrays stand on other nodes, as on two grids."""
    from reverberate.trace.bundle import carry_early

    recipe = moving_recipe()
    plan = make_plan(recipe, assets().triangles)
    there: list[np.ndarray | None] = [cell + OFF_THE_CELL for cell in plan.all_cells]
    elsewhere: list[np.ndarray | None] = [
        cell + np.array([0.011, 0.0, 0.002]) for cell in plan.all_cells
    ]
    first, _, _ = traced(tmp_path / "a", recipe, rays=16, centres=there)
    first.check_mode = "read"
    first.run()
    assert all("cached" not in record for record in first.report["paths"].values())
    second, _, _ = traced(tmp_path / "b", recipe, rays=16, centres=elsewhere)
    second.check_mode = "read"
    assert carry_early(tmp_path / "b" / "bundle" / "early_cache", tmp_path / "a" / "out") == 3
    second.run()
    paths = second.report["paths"]
    # The sources' tables are the first run's; the pairs at rest stand on other centres.
    assert paths["voice"] == paths["tap"] == {"cached": True}
    assert "cached" not in paths["_pairs"]
    with (
        read_pack(tmp_path / "a" / "out" / "pack.h5") as a,
        read_pack(tmp_path / "b" / "out" / "pack.h5") as b,
    ):
        for name in a.sources:
            assert np.array_equal(a.sources[name].early.delay_s, b.sources[name].early.delay_s)
        assert not np.array_equal(a.cells.position, b.cells.position)
    with pytest.raises(SystemExit, match="holds no early table"):
        carry_early(tmp_path / "c", tmp_path / "nowhere")
    # A recipe of this dwelling's assets, traced on another low grid: that key alone may differ.
    held = assets()
    tree = recipe.to_dict()
    tree["assets"] = found_assets(recipe, held, voxel_low_key="the validated grid")
    ours = Recipe.from_dict(tree)
    build_bundle(tmp_path / "d" / "bundle", ours, held, make_plan(ours, held.triangles))
    spec_file = tmp_path / "d" / "bundle" / "campaign.json"
    spec = json.loads(spec_file.read_text())
    assert spec["trace"]["allowed_mismatch"] == [] and spec["trace"]["low"]["engine"] == "lowband"

    def refused(allowed: list[str]) -> list[str] | str:
        spec["trace"]["allowed_mismatch"] = allowed
        spec_file.write_text(json.dumps(spec))
        pairs = FreeFieldPairs(plan.tracks.positions, plan.all_cells, tmp_path / "d" / "out")
        trace = Trace(tmp_path / "d" / "bundle", tmp_path / "d" / "out", engine=pairs, gpu=False)
        trace.journal.quiet = True
        try:
            return trace.check_assets()
        except RuntimeError as error:
            return str(error)

    assert "voxel_low_key" in str(refused([]))
    assert refused(["voxel_low_key"]) == ["voxel_low_key"]
    assert "voxel_low_key" in str(refused(["calibration_key"]))
