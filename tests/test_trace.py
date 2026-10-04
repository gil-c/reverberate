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
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import read_pack
from reverberate.scenes import (
    Recipe,
    low_band_source_positions,
    placeholder_assets,
    recipe_sha256,
)
from reverberate.spatial.lowband import LOW_RATE_HZ, from_stored, pair_key, with_air
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
from reverberate.trace.driver import cost_records, describe, finish, launch, stamp_cost
from reverberate.trace.engines import CardPairs, FreeFieldPairs
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
    """The walled box as a dwelling's mirror: a signature, a lead of 4 ms, a gain of a half."""
    return MirrorAssets(
        catalogue=walled_box(),
        settings=MirrorSettings(
            rays=RaySettings(rays=rays, duration_s=0.12, bin_s=0.002, receiver_radius_m=0.3)
        ),
        signature=np.array([1.0, 0.35, -0.12, 0.04]),
        lead_s=0.0040073,
        gain=0.5,
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
    tmp: Path, recipe: Recipe, *, rays: int = 120, **engine: Any
) -> tuple[Trace, FreeFieldPairs, Plan]:
    """The bundle of ``recipe`` and its trace, not yet run, on the free field engine."""
    held = assets(rays)
    plan = make_plan(recipe, held.triangles)
    build_bundle(tmp / "bundle", recipe, held, plan, allow_asset_mismatch=True)
    pairs = FreeFieldPairs(
        plan.tracks.positions,
        plan.all_cells,
        tmp / "out",
        lead_s=held.pack_lead_s,
        gain=held.gain,
        **engine,
    )
    trace = Trace(
        bundle=tmp / "bundle",
        out=tmp / "out",
        engine=pairs,
        gpu=False,
        devices=Devices.host(1),
        quiet=True,
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
        assert pack.mirror.alignment_gain == 0.5 and pack.air.enabled
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
                20.0 * np.log10(0.5) + low.seam_db[low.pair[one, 0, 0]],
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
    cached = pairs.cache.read(pairs.key_of(0, 0))
    wave = from_stored(with_air(cached, LOW_RATE_HZ, Atmosphere(), sound_speed_m_s=c), FS)
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
            source.level.high_gain_db, 20.0 * np.log10(held.gain) + seam, atol=1e-4
        )
        engine = Engine(pack, {"voice": click}, settings=RenderSettings(directivity=False))
        got = {name: engine.stem("voice", parts=(name,))[:, start:] for name in want}
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
    # One of the plan's pairs is in this machine's cache already: the bundle carries it.
    known = pair_key(
        "key1500",
        plan.tracks.positions[0],
        plan.all_cells[plan.heard_at[0][0]],
        encoder=pairs_module.encoder_record(7, 10, 1500.0),
        solver=pairs_module.SOLVER,
    )
    PairCache.local("key1500").write(known, np.zeros((64, 4800), dtype=np.float32), {})
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
    pairs = json.loads((home / "bundle" / "pairs" / "campaign.json").read_text())
    assert pairs["kind"] == pairs_module.KIND and pairs["pairs"] == plan.pairs
    assert np.array_equal(np.load(home / "bundle" / "pairs" / "cells.npy"), plan.all_cells)
    # Home: the pack is read, and its cost is the rental's bill at the rental's rate.
    assert result["home"]["pack"]["steps"] == 11 and result["home"]["usd"] == pytest.approx(0.1)
    with read_pack(home / "pulled" / "pack.h5") as pack:
        assert {r["billed_rate_usd_per_hour"] for r in pack.header.provenance["cost"]} == {0.4}
    assert any("cost at 1.74 USD/h" in line for line in said)
