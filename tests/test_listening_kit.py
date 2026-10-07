"""The listening kit: the options that change a pack, priced, named and put side by side.

A variant is a name and the options it takes. What is checked here: that an
option left alone leaves the plan's price and records as they were; that a
low band solved for fewer seconds is stored at the pack's length and ends in
silence; that a set is priced on the whole scene and on an excerpt, with one
command a variant; and that a variant of another rail pitch is the same
scene under another recipe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.render.variant import label_of, measured_cost, read_variant, summary
from reverberate.scenes import load_recipe, save_recipe
from reverberate.scenes.describe import describe, low_band_positions
from reverberate.scenes.recipe import canonical_bytes, recipe_sha256
from reverberate.spatial.lowband import LOW_RATE_HZ, LOW_SAMPLES
from reverberate.trace import machines, variants
from reverberate.trace.cli import main, with_rays
from reverberate.trace.engines import FreeFieldPairs
from reverberate.trace.level import LOW_FADE_S, at_pack_length, pair_low
from reverberate.trace.plan import Profile, estimate, make_plan, tracks_of
from reverberate.wave.lowband import pairs as batched
from test_trace import assets, moving_recipe, resting_recipe, traced

RECORD = {
    "source_positions": 1646,
    "pairs": 16529,
    "tail_sites": 211,
    "tail_cells": 53,
    "step_pairs": 45023,
    "cells_a_position": [120] * 40 + [9] * 1606,
}


# --------------------------------------------------------------------------
# the low band solved for fewer seconds
# --------------------------------------------------------------------------


def test_a_response_solved_for_fewer_seconds_is_kept_at_the_pack_s_length() -> None:
    rng = np.random.default_rng(5)
    whole = rng.standard_normal((4, LOW_SAMPLES)).astype(np.float32)
    # The reference's response is the array it was: nothing is copied, nothing can differ.
    assert at_pack_length(whole) is whole
    short = whole[:, : round(0.8 * LOW_RATE_HZ)]
    kept = at_pack_length(short)
    fade = round(LOW_FADE_S * LOW_RATE_HZ)
    assert kept.shape == whole.shape and kept.dtype == np.float32 and fade == 80
    cut = short.shape[1]
    assert np.array_equal(kept[:, : cut - fade], short[:, : cut - fade])
    assert not kept[:, cut:].any() and kept[0, cut - 1] == 0.0
    # A raised cosine: half way down at its middle, and never above what was solved.
    ratio = kept[:, cut - fade : cut] / short[:, cut - fade :]
    assert ratio[0, fade // 2 - 1] == pytest.approx(0.5, abs=0.01)
    assert (np.diff(ratio[0]) < 0).all() and ratio.max() < 1.0
    with pytest.raises(ValueError, match="longer than the pack"):
        at_pack_length(np.zeros((1, LOW_SAMPLES + 1), dtype=np.float32))


def test_a_trace_of_shorter_responses_writes_a_pack_of_the_same_shape(tmp_path: Path) -> None:
    import h5py

    class Short(FreeFieldPairs):
        """An engine that solved 0.8 s: its responses end there."""

        def response(self, position: int, cell: int) -> np.ndarray:
            return super().response(position, cell)[:, : round(0.8 * LOW_RATE_HZ)]

    # ``low/ir`` as it was, read in the file itself: the samples, not their bins.
    trace, pairs, _ = traced(tmp_path, resting_recipe(), low_levers="none")
    trace.engine = Short(pairs.sources, pairs.cells, tmp_path / "out", gain=pairs.gain)
    trace.check_mode = "read"
    trace.run()
    with h5py.File(tmp_path / "out" / "pack.h5", "r") as pack:
        assert pack.attrs["low_samples"] == LOW_SAMPLES
        stored = np.asarray(pack["sources/voice/low/ir"])
    assert stored.shape[1:] == (64, LOW_SAMPLES)
    key = trace.pair_key[0]
    cached = trace.engine.cache.read(key)
    assert cached.shape == (64, round(0.8 * LOW_RATE_HZ))
    # What the pack holds is the padded response through the pack's own chain.
    from reverberate.audio import Atmosphere
    from reverberate.spatial.lowband import FIELD_UNIT_AT_1M

    expected = pair_low(
        at_pack_length(cached),
        trace.crossover,
        Atmosphere(**trace.recipe.atmosphere.to_dict()),
        sound_speed_m_s=trace.assets.settings.sound_speed_m_s,
        lead_s=trace.assets.pack_lead_s,
        unit_at_1m=FIELD_UNIT_AT_1M,
    )[0]
    assert np.array_equal(stored[0], expected)
    assert trace.report["level"]["end_db"] is not None


def test_the_estimate_goes_with_the_seconds_solved_and_is_untouched_without_them() -> None:
    reference = estimate(RECORD, rate_usd_per_hour=0.136)
    assert "low_seconds" not in reference and "rays" not in reference
    assert reference == estimate(RECORD, rate_usd_per_hour=0.136, low_seconds=None, rays=None)
    short = estimate(RECORD, rate_usd_per_hour=0.136, low_seconds=0.8)
    assert short["low_seconds"] == 0.8
    # A solve goes as the seconds; a pair's filters and what is done once do not.
    assert short["low"]["stencil_s_per_source"] == pytest.approx(
        reference["low"]["stencil_s_per_source"] * 0.8 / 1.2, abs=0.01
    )
    # A card holds the records of more cells when each is shorter: fewer solves twice over.
    assert batched.cells_a_solve(20.0) == batched.CELLS_A_SOLVE_MEASURED == 57
    assert batched.cells_a_solve(20.0, duration_s=0.8) == 85
    assert short["solves"] < reference["solves"] and short["extra_solves"] == 40
    assert reference["extra_solves"] == 80
    assert short["seconds"]["low"] < reference["seconds"]["low"] * 0.7
    # The pack is as long; the pair cache that goes home is not.
    assert short["pack_gb"] == reference["pack_gb"]
    assert short["pair_cache_gb"] == pytest.approx(reference["pair_cache_gb"] * 0.8 / 1.2, abs=0.01)
    for stage in ("paths", "rays", "level", "write", "transfer_pack"):
        assert short["seconds"][stage] == reference["seconds"][stage]
    with pytest.raises(ValueError, match="up to 1.2 s"):
        estimate(RECORD, rate_usd_per_hour=0.136, low_seconds=1.5)
    with pytest.raises(ValueError, match="only the batched solver"):
        estimate(RECORD, rate_usd_per_hour=1.74, low_engine="pffdtd", low_seconds=0.8)
    # An offered machine is priced the same way.
    offer: dict[str, Any] = {
        "gpu_name": "RTX 3090",
        "num_gpus": 4,
        "gpu_ram_gb": 24.0,
        "dph_total": 0.5,
    }
    whole, less = (
        machines.predict(RECORD, **offer, low_seconds=seconds) for seconds in (None, 0.8)
    )
    assert whole is not None and less is not None
    assert less["seconds"]["low"] < whole["seconds"]["low"] * 0.7 and less["usd"] < whole["usd"]


def test_fewer_rays_are_priced_in_proportion_and_carry_the_same_energy(tmp_path: Path) -> None:
    reference = estimate(RECORD, rate_usd_per_hour=0.136)
    half = estimate(RECORD, rate_usd_per_hour=0.136, rays=50_000)
    assert half["rays"] == 50_000
    assert half["seconds"]["rays"] == pytest.approx(reference["seconds"]["rays"] / 2, abs=0.1)
    assert half["seconds"]["low"] == reference["seconds"]["low"]
    held = assets(rays=4000)
    assert with_rays(held, None) is held and with_rays(held, 4000) is held
    fewer = with_rays(held, 2000)
    assert fewer.settings.rays.rays == 2000 and held.settings.rays.rays == 4000
    # Nothing else of the mirror moves: the calibration's key is the one the recipe names.
    assert fewer.settings.parameters.key == held.settings.parameters.key
    assert fewer.settings.rays.record() == {**held.settings.rays.record(), "rays": 2000}
    with pytest.raises(SystemExit, match="a thousand rays"):
        with_rays(held, 10)


# --------------------------------------------------------------------------
# the count of source positions
# --------------------------------------------------------------------------


def test_the_positions_counted_are_those_a_trace_of_the_same_rail_reading_solves() -> None:
    recipe = moving_recipe()
    for count in (2, 4, 8):
        plan = make_plan(recipe, assets().triangles, Profile(rail_positions=count))
        counted = low_band_positions(recipe, count)["audible"]
        assert counted == plan.record["source_positions"] == plan.tracks.positions.shape[0]
        assert f"{counted} where a source is audible" in describe(recipe, count)
    assert low_band_positions(recipe) == low_band_positions(recipe, 2)
    assert "positions read" not in describe(recipe)


# --------------------------------------------------------------------------
# the set
# --------------------------------------------------------------------------


def test_the_set_of_the_first_scene_names_the_reference_and_what_each_variant_takes() -> None:
    name, held = variants.load_set()
    assert name == "listening_v1" and held[0].name == "reference" and held[0].flags == {}
    by_name = {variant.name: variant for variant in held}
    assert list(by_name) == [
        "reference",
        "grid-7.2",
        "rail-10cm-x8",
        "rail-12cm-x8",
        "low-0.8s",
        "rays-50k",
        "all-cheap",
    ]
    assert by_name["all-cheap"].flags == {
        "low_ppw": 7.2,
        "rail_pitch_m": 0.12,
        "rail_positions": 8,
        "low_seconds": 0.8,
    }
    assert by_name["grid-7.2"].get("rail_positions") == 2 and all(v.says for v in held)
    with pytest.raises(ValueError, match="sets loudness"):
        variants.Variant("x", flags={"loudness": 3})
    with pytest.raises(ValueError, match="a file's name"):
        variants.Variant("a b")


def test_another_rail_pitch_is_the_same_scene_under_another_recipe() -> None:
    recipe = moving_recipe()
    wider, changed = variants.recipe_at_pitch(recipe, 0.12)
    assert changed["pitch_m"] == {"was": 0.08, "is": 0.12} and changed["rails"] == 1
    assert changed["recipe_sha256"]["was"] == recipe_sha256(recipe) != recipe_sha256(wider)
    one, other = (json.loads(canonical_bytes(r)) for r in (recipe, wider))
    assert [rail["pitch_m"] for rail in other["rails"]] == [0.12]
    for rail in (*one["rails"], *other["rails"]):
        del rail["pitch_m"]
    assert one == other
    # The same movements: every source and the head are where they were at every step.
    here, there = tracks_of(recipe), tracks_of(wider, Profile(rail_positions=8))
    assert np.array_equal(here.listener, there.listener)
    for name, track in here.sources.items():
        assert np.array_equal(track.position, there.sources[name].position)
        assert np.array_equal(track.audible, there.sources[name].audible)
    assert there.positions.shape[0] != here.positions.shape[0]


def test_a_window_is_proposed_where_a_source_speaks_on_the_move() -> None:
    recipe = moving_recipe()
    found = variants.listening_windows(recipe, 0.2, every_s=0.05, best=2)
    assert len(found) == 2 and found[0].score >= found[1].score
    best = found[0]
    # The voice walks from 0.15 s: a window before it holds none of that.
    assert best.start_s >= 0.1 and best.walking_s > 0.1
    still = variants.listening_windows(resting_recipe(), 0.2, every_s=0.05, best=1)[0]
    assert still.walking_s == still.passing_s == 0.0
    assert "audible on the move" in best.says() and "another room" in best.says()


def test_a_set_is_priced_on_the_scene_and_the_excerpt_with_a_command_a_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = moving_recipe()
    save_recipe(recipe, tmp_path / "recipe.json")
    assets().save(tmp_path / "mirror")
    arguments = ["variants", "--recipe", str(tmp_path / "recipe.json")]
    arguments += ["--mirror", str(tmp_path / "mirror"), "--home", str(tmp_path / "kit")]
    arguments += ["--window", "0.1", "0.3", "--sources", "1", "--pass", "--gpus 2 --yes"]
    assert main([*arguments, "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "dry run: no recipe and no variant.json written" in printed
    assert not (tmp_path / "kit").exists()
    assert main(arguments) == 0
    printed = capsys.readouterr().out
    assert "set listening_v1: 7 variants on 0.3 s from 0.1 s" in printed
    names = [v.name for v in variants.load_set()[1]]
    records = {name: read_variant(tmp_path / "kit" / name / "pulled" / "pack.h5") for name in names}
    assert all(record is not None for record in records.values())
    reference: dict[str, Any] = records["reference"] or {}
    cheap: dict[str, Any] = records["all-cheap"] or {}
    # The reference's command is the reference's recipe and its grid by name, which a
    # trace no longer takes unless told; a variant's, its own.
    home = tmp_path / "kit"
    assert reference["command"] == (
        f"python -m reverberate.trace rent --recipe {tmp_path / 'recipe.json'}"
        f" --home {home / 'reference'} --smoke 0.3 --smoke-start 0.1 --smoke-sources 1"
        f" --low-ppw 10.5 --mirror {tmp_path / 'mirror'} --gpus 2 --yes"
    )
    assert reference["command"] in printed and reference["recipe_changed"] is None
    assert (
        f"--recipe {home / 'all-cheap' / 'recipe.json'} --home {home / 'all-cheap'} --smoke 0.3"
        " --smoke-start 0.1 --smoke-sources 1 --low-ppw 7.2 --low-seconds 0.8"
        " --rail-positions 8 --mirror"
    ) in cheap["command"]
    assert "--rays 50000" in (records["rays-50k"] or {})["command"]
    # A variant of the rails has its recipe beside it, valid, and the table says what changed.
    wider = load_recipe(home / "all-cheap" / "recipe.json")
    assert (
        recipe_sha256(wider)
        == cheap["recipe_sha256"]
        == cheap["recipe_changed"]["recipe_sha256"]["is"]
    )
    assert {rail.pitch_m for rail in wider.rails} == {0.12}
    assert not (home / "grid-7.2" / "recipe.json").exists()
    assert "all-cheap: another recipe" in printed and "solved every 0.12 m and not 0.08" in printed
    # The whole scene's price is the plan's own, and a cheaper grid's is less.
    whole = make_plan(recipe, assets().triangles)
    priced = estimate(whole, rate_usd_per_hour=0.136)
    assert reference["predicted"]["scene"]["usd"] == priced["total_usd"]
    assert reference["predicted"]["scene"]["source_positions"] == whole.record["source_positions"]
    grid = (records["grid-7.2"] or {})["predicted"]["scene"]
    assert grid["low_usd"] < reference["predicted"]["scene"]["low_usd"]
    assert reference["predicted"]["excerpt"]["pairs"] <= reference["predicted"]["scene"]["pairs"]
    assert reference["window"] == {"start_s": 0.1, "seconds": 0.3, "sources": 1}
    # The window proposed, when none is given.
    assert main([*arguments[:7], "--window", "auto", "0.2", "--dry-run"]) == 0
    assert "windows proposed, the best first:" in capsys.readouterr().out


def test_a_pack_is_named_and_costed_by_the_file_beside_it(tmp_path: Path) -> None:
    home = tmp_path / "low-0.8s"
    (home / "pulled").mkdir(parents=True)
    pack = home / "pulled" / "pack.h5"
    assert read_variant(pack) is None and label_of(pack) == "low-0.8s"
    assert label_of(tmp_path / "A.h5") == "A"
    taken: set[str] = set()
    assert [label_of(tmp_path / "A.h5", taken) for _ in range(2)] == ["A", "A-2"]
    record = {
        "name": "low-0.8s",
        "says": "shorter",
        "flags": {"low_seconds": 0.8},
        "predicted": {"scene": {"usd": 5.07}, "excerpt": {"usd": 0.57}},
        "measured": None,
    }
    (home / "variant.json").write_text(json.dumps(record))
    assert (read_variant(pack) or {})["flags"] == {"low_seconds": 0.8}
    cost = [{"stage": "low", "seconds": 600.0, "usd": 0.31}, {"stage": "rays", "usd": 0.04}]
    assert measured_cost({}) is None
    assert measured_cost({"cost": cost}) == {"usd": 0.35, "seconds": 600.0, "records": 2}
    assert summary(pack, {"cost": cost}) == {
        "name": "low-0.8s",
        "says": "shorter",
        "flags": {"low_seconds": 0.8},
        "predicted_scene_usd": 5.07,
        "predicted_excerpt_usd": 0.57,
        "measured": {"usd": 0.35, "seconds": 600.0, "records": 2},
    }
    assert summary(tmp_path / "A.h5")["name"] is None
