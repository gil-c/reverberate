"""Tests for recipes of version 2 and the generator that writes them.

On the hand-built dwelling of ``scene_floor``, with three objects a fixed
source may belong to. What is held here is what the other lots and the
owner lean on: a version 1 recipe is what it was; a version 2 recipe says
who is of the listener's conversation and never contradicts itself; what
moves for nothing, a sway, a carried noise, the wearer's own voice, asks for
no wave solve; nobody is audible on the move but by footsteps; and every
rule of version 2 refuses the recipe that breaks it, by its number.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import (
    Fixture,
    Layout,
    Recipe,
    SocialParameters,
    build_layout,
    canonical_bytes,
    cost,
    generate,
    generate_social,
    listener_state,
    low_band_source_positions,
    parse_recipe,
    recipe_sha256,
    sample_times,
    source_state,
    validate,
    validate_text,
    wave_band,
)
from reverberate.scenes.describe import describe, gaze_share
from reverberate.scenes.kinematics import audible_steps
from reverberate.scenes.levels import (
    EFFORT_LEVEL_DB,
    conversation_snr,
    effort_of,
    in_conversation,
    lombard_level_db,
)
from reverberate.scenes.recipe import Dwell, Dwelling
from reverberate.scenes.validate import SWAY_RADIUS_M
from scene_floor import FLOOR_Y_M, SMALL, small_recipe, two_room_floor


@lru_cache(maxsize=1)
def social_layout() -> Layout:
    """The two rooms, with a television, a counter and a window."""
    floor = replace(
        two_room_floor(),
        fixtures=(
            Fixture("tv_7", "tv", (2.0, FLOOR_Y_M + 1.4, 3.7), (0.0, -1.0)),
            Fixture("kitchen_counter_8", "kitchen_counter", (4.6, FLOOR_Y_M + 1.05, 0.4), None),
            Fixture("window_9", "window", (8.7, FLOOR_Y_M + 1.4, 2.0), (-1.0, 0.0)),
        ),
    )
    return build_layout(Dwelling("two_rooms", "synthetic_two_rooms", FLOOR_Y_M), floor)


@lru_cache(maxsize=16)
def social_recipe(preset: str = "medium", seed: int = 1, duration_s: float = 60.0) -> Recipe:
    return generate_social(
        social_layout(),
        SocialParameters.preset(preset, duration_s),
        seed,
        allow_placeholder_clips=True,
        allow_placeholder_assets=True,
    )


@lru_cache(maxsize=1)
def moving_recipe() -> Recipe:
    """A scene in which somebody changes group: the first of a few seeds that has one."""
    for seed in range(1, 12):
        recipe = social_recipe("medium", seed)
        if any(len(source.roles) > 1 for source in recipe.sources):
            return recipe
    raise AssertionError("no seed moved anybody")


def rules(tree: dict[str, Any]) -> list[int]:
    return [v.rule for v in validate_text(json.dumps(tree), social_layout().floor)]


def tree_of(recipe: Recipe) -> dict[str, Any]:
    return copy.deepcopy(recipe.to_dict())


def source_of(tree: dict[str, Any], **wanted: Any) -> dict[str, Any]:
    return next(s for s in tree["sources"] if all(s.get(k) == v for k, v in wanted.items()))


# --------------------------------------------------------------------------
# the two versions
# --------------------------------------------------------------------------


def test_a_version_1_recipe_is_what_it_was() -> None:
    recipe = small_recipe()
    payload = canonical_bytes(recipe)
    assert b'"schema_version":1' in payload
    assert all(key not in payload for key in (b'"scene"', b'"sway"', b'"roles"', b'"level_spl'))
    assert recipe_sha256(parse_recipe(payload)) == recipe_sha256(recipe)
    # A key of version 2 is no key of version 1.
    tree = tree_of(recipe)
    tree["scene"] = {"calmness": 0.5, "snr_floor_db": 0.0}
    assert rules(tree) == [1]
    tree = tree_of(recipe)
    tree["schema_version"] = 3
    assert rules(tree) == [1]


def test_the_first_generator_does_not_see_the_fixtures() -> None:
    layout = social_layout()
    assert {s.kind for s in layout.fixtures} == {"fixture"} and len(layout.fixtures) == 3
    assert all(s.kind != "fixture" for s in layout.stations)
    again = generate(layout, SMALL, 1, allow_placeholder_clips=True, allow_placeholder_assets=True)
    assert recipe_sha256(again) == recipe_sha256(small_recipe())


@pytest.mark.parametrize("preset", ["quiet", "medium", "lively"])
def test_a_social_recipe_is_valid_and_keeps_its_identity(preset: str) -> None:
    recipe = social_recipe(preset)
    assert recipe.schema_version == 2
    assert validate(recipe, social_layout().floor) == []
    payload = canonical_bytes(recipe)
    assert b'"schema_version":2' in payload
    assert canonical_bytes(parse_recipe(payload)) == payload
    assert (
        recipe.scene is not None
        and recipe.scene.calmness == {"quiet": 0.9, "medium": 0.5, "lively": 0.15}[preset]
    )
    assert "calmness" in describe(recipe)


def test_the_same_seed_gives_the_same_bytes() -> None:
    parameters = SocialParameters.preset("medium", 60.0)
    assert SocialParameters.from_record(parameters.record()) == parameters

    def draw(seed: int) -> bytes:
        return canonical_bytes(
            generate_social(
                social_layout(),
                parameters,
                seed,
                allow_placeholder_clips=True,
                allow_placeholder_assets=True,
            )
        )

    assert draw(1) == canonical_bytes(social_recipe("medium", 1))
    assert draw(2) != draw(1)
    assert social_recipe("medium", 1).generator.parameters["calmness"] == 0.5  # type: ignore[union-attr]


def test_a_calm_scene_has_fewer_sources_and_a_quieter_voice() -> None:
    quiet, lively = social_recipe("quiet"), social_recipe("lively")
    assert len(quiet.sources) < len(lively.sources)

    def level(recipe: Recipe) -> float:
        said = [
            (source.level_spl_1m_db or 0.0) + interval.gain_db
            for source in recipe.sources
            if source.kind == "voice"
            for interval in source.activity
        ]
        return float(np.median(said))

    assert level(quiet) < level(lively)


# --------------------------------------------------------------------------
# roles: the training label
# --------------------------------------------------------------------------


def test_a_voice_is_of_the_conversation_exactly_while_it_shares_the_listeners_group() -> None:
    recipe = moving_recipe()
    assert {s.kind for s in recipe.sources} >= {"voice", "own_voice"}
    assert all(s.kind not in ("near_voice", "far_voice") for s in recipe.sources)
    mine = recipe.listener.conversation
    assert mine[0].start_s == 0.0 and mine[-1].end_s == recipe.duration_s
    times = np.arange(0.5, recipe.duration_s, 1.0)
    for source in recipe.sources:
        if source.kind != "voice":
            assert source.roles == ()
            continue
        inside = in_conversation(source, times)
        for t, said in zip(times, inside, strict=True):
            group = next(m.group for m in mine if m.start_s <= t < m.end_s)
            own = next(r.group for r in source.roles if r.start_s <= t < r.end_s)
            assert said == (own is not None and own == group)
    # A role that contradicts the groups is refused.
    tree = tree_of(recipe)
    voice = source_of(tree, kind="voice")
    voice["roles"][0]["role"] = (
        "outside" if voice["roles"][0]["role"] == "conversation" else "conversation"
    )
    assert 13 in rules(tree)
    tree = tree_of(recipe)
    source_of(tree, kind="voice")["roles"][-1]["end_s"] -= 1.0
    assert 13 in rules(tree)


def test_nobody_is_audible_on_the_move_but_by_footsteps() -> None:
    recipe = moving_recipe()
    walked = False
    for source in recipe.sources:
        if not source.segments:
            continue
        heard = audible_steps(recipe, source.id)
        state = source_state(recipe, source.id, sample_times(recipe)[heard])
        resting = [isinstance(source.segments[int(n)], Dwell) for n in np.unique(state.segment)]
        assert all(resting), source.id
        walked |= any(not isinstance(segment, Dwell) for segment in source.segments)
    assert walked
    # Footsteps are the mirror's alone (the owner, 2026-10-07): a footfall asks no solve.
    low = low_band_source_positions(recipe)
    assert set(low.kind) == {"station"}
    steps = [s for s in recipe.sources if s.subtype == "steps"]
    assert steps and not any(wave_band(s) for s in steps)
    for source in steps:
        where = source_state(recipe, source.id, sample_times(recipe)).position
        assert np.allclose(where[:, 1], FLOOR_Y_M + 0.05)
        # A footfall does not move while it sounds: a place a stride, not one every 8 cm.
        assert source.attach is not None and source.attach.stride_m == 0.64
        hops = np.linalg.norm(np.diff(where, axis=0), axis=1)
        assert np.all((hops < 1e-9) | (hops > 0.05))
        walked_m = float(hops.sum())
        heard = where[audible_steps(recipe, source.id)]
        places = np.unique(np.round(heard, 3), axis=0).shape[0]
        assert 0 < places <= walked_m / 0.64 + 3
        assert low.by_source[source.id] == ()


# --------------------------------------------------------------------------
# what moves for nothing
# --------------------------------------------------------------------------


def test_a_sway_moves_the_mouth_and_no_low_band_position() -> None:
    recipe = social_recipe("medium")
    times = sample_times(recipe)
    voice = next(s for s in recipe.sources if s.kind == "voice")
    state = source_state(recipe, voice.id, times)
    assert state.sway_m is not None
    reach = np.linalg.norm(state.mouth - state.position, axis=1)
    assert 0.002 < reach.max() <= SWAY_RADIUS_M
    head = listener_state(recipe, times)
    assert 0.002 < np.linalg.norm(head.head - head.position, axis=1).max() <= SWAY_RADIUS_M
    still = replace(
        recipe,
        sources=tuple(replace(s, sway=()) for s in recipe.sources),
        listener=replace(recipe.listener, sway=()),
    )
    assert np.array_equal(
        low_band_source_positions(still).positions, low_band_source_positions(recipe).positions
    )
    assert cost.counts(still)["cells"] == cost.counts(recipe)["cells"]
    # A sway further than the low band is held is refused.
    tree = tree_of(recipe)
    source_of(tree, kind="voice")["sway"][0]["amplitude_m"] = 0.2
    assert 14 in rules(tree)


def test_the_wearers_own_voice_is_at_the_mouth_and_asks_for_no_solve() -> None:
    recipe = social_recipe("medium")
    own = [s for s in recipe.sources if s.kind == "own_voice"]
    assert len(own) == 1 and own[0].activity and not own[0].segments
    times = sample_times(recipe)
    head = listener_state(recipe, times)
    state = source_state(recipe, "own_voice", times)
    assert np.allclose(np.linalg.norm(state.position - head.position, axis=1), 0.103, atol=1e-3)
    assert np.allclose(state.yaw_deg, head.yaw_deg)
    # In front of the head: along the way it faces, at no pitch.
    level = np.abs(head.pitch_deg) < 1.0
    ahead = np.stack([np.cos(np.radians(head.yaw_deg)), -np.sin(np.radians(head.yaw_deg))], axis=1)
    offset = (state.position - head.position)[:, [0, 2]]
    assert level.any() and np.all(np.sum(offset[level] * ahead[level], axis=1) > 0.085)
    assert low_band_source_positions(recipe).by_source["own_voice"] == ()
    tree = tree_of(recipe)
    tree["sources"].append({**source_of(tree, kind="own_voice"), "id": "second_voice"})
    assert 12 in rules(tree)
    tree = tree_of(recipe)
    source_of(tree, kind="own_voice")["attach"]["offset_m"] = [0.6, 0.0, 0.0]
    assert 12 in rules(tree)


def test_a_persons_noise_is_at_its_talkers_mouth_and_asks_for_no_solve() -> None:
    recipe = social_recipe("medium")
    low = low_band_source_positions(recipe)
    bodies = [
        s for s in recipe.sources if s.subtype == "body" and s.attach and s.attach.to != "listener"
    ]
    assert bodies
    times = sample_times(recipe)
    for body in bodies:
        assert body.attach is not None and not wave_band(body)
        carrier = source_state(recipe, body.attach.to, times)
        mine = source_state(recipe, body.id, times)
        # Where its talker's mouth is, sway and all; and the mirror's alone.
        assert np.array_equal(mine.mouth, carrier.mouth)
        assert low.by_source[body.id] == ()
    tree = tree_of(recipe)
    source_of(tree, subtype="body").pop("attach")
    assert rules(tree) and set(rules(tree)) <= {2, 12}


def test_a_fixed_source_is_by_its_object_and_does_not_move() -> None:
    recipe = social_recipe("lively")
    fixed = [
        s
        for s in recipe.sources
        if s.kind == "media_voice"
        or s.subtype in ("appliance", "water", "other", "music", "outside")
    ]
    assert fixed
    for source in fixed:
        assert len(source.segments) == 1 and isinstance(source.segments[0], Dwell)
    stations = {s.id: s for s in recipe.stations}
    for source in fixed:
        station = stations[source.segments[0].station]  # type: ignore[union-attr]
        if source.subtype == "television":
            assert station.kind == "fixture" and station.object == "tv_7"
            assert source.kind == "media_voice"
        if source.subtype == "outside":
            assert source.opening is not None and source.opening.object == station.object
    tree = tree_of(recipe)
    moved = next(
        s for s in tree["sources"] if s["kind"] in ("media_voice", "noise") and s["segments"]
    )
    other = next(s["id"] for s in tree["stations"] if s["kind"] == "stand")
    half = recipe.duration_s / 2
    first = moved["segments"][0]
    moved["segments"] = [
        {**first, "end_s": half},
        {**first, "start_s": half, "station": other, "height": "standing"},
    ]
    assert 12 in rules(tree)


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------


def test_an_effort_is_a_level_and_noise_raises_the_voice() -> None:
    assert [effort_of(level) for level in (40.0, 54.0, 60.0, 66.0, 75.0)] == list(EFFORT_LEVEL_DB)
    assert lombard_level_db(60.0, 30.0) == 60.0
    assert lombard_level_db(60.0, 55.0) == 65.0
    assert lombard_level_db(60.0, 95.0) == 72.0
    recipe = social_recipe("lively")
    for source in recipe.sources:
        assert source.level_spl_1m_db is not None
        for interval in source.activity:
            assert (interval.effort is not None) == (source.kind in ("voice", "own_voice"))
    tree = tree_of(recipe)
    said = source_of(tree, kind="voice")["activity"][0]
    said["effort"] = "whisper" if said["effort"] != "whisper" else "loud"
    assert 15 in rules(tree)


def test_no_turn_of_the_conversation_is_under_the_floor() -> None:
    for preset in ("quiet", "medium", "lively"):
        recipe = social_recipe(preset)
        assert recipe.scene is not None
        ratios = [row[2] for row in conversation_snr(recipe)]
        assert ratios and min(ratios) >= recipe.scene.snr_floor_db
    recipe = social_recipe("lively")
    tree = tree_of(recipe)
    loud = next(
        s for s in tree["sources"] if s["kind"] in ("media_voice", "noise") and s["segments"]
    )
    loud["level_spl_1m_db"] = 99.0
    assert 16 in rules(tree)


# --------------------------------------------------------------------------
# turns and the head
# --------------------------------------------------------------------------


def test_turns_follow_one_another_with_gaps_and_overlaps() -> None:
    recipe = social_recipe("medium", 3, 300.0)
    turns = sorted(
        (interval.start_s, interval.end_s, source.id)
        for source in recipe.sources
        if source.kind in ("voice", "own_voice")
        for interval in source.activity
        if interval.event == "turn"
        and (source.kind == "own_voice" or in_conversation(source, np.array([interval.start_s]))[0])
    )
    offsets = np.array(
        [b[0] - a[1] for a, b in zip(turns[:-1], turns[1:], strict=True) if a[2] != b[2]]
    )
    assert offsets.size > 30
    # Heldner and Edlund 2010: about two transfers in five overlap; the mode is a short gap.
    assert 0.15 < float((offsets < 0).mean()) < 0.55
    assert 0.0 < float(np.median(offsets)) < 0.5
    events = {i.event for s in recipe.sources for i in s.activity if i.event}
    assert events == {"turn", "backchannel", "laughter"}
    # No voice speaks over itself.
    for source in recipe.sources:
        ends = [i.end_s for i in source.activity]
        assert all(b.start_s >= a for a, b in zip(ends[:-1], source.activity[1:], strict=True))


def test_the_listener_looks_at_the_talker_about_six_times_in_ten() -> None:
    recipe = social_recipe("medium", 3, 300.0)
    assert abs(gaze_share(recipe) - 0.6) < 0.08
    modes = {interval.mode for interval in recipe.listener.gaze}
    assert {"talker", "reading"} <= modes and len(modes) >= 4
    # The head is never still for long: a keyframe about every second.
    assert len(recipe.listener.keyframes) > 0.6 * recipe.duration_s
    yaw = np.array([frame.yaw_deg for frame in recipe.listener.keyframes])
    assert np.ptp(yaw) > 30.0
    tree = tree_of(recipe)
    tree["listener"]["gaze"][1]["start_s"] = tree["listener"]["gaze"][0]["start_s"]
    assert 17 in rules(tree)


def test_a_talker_turns_towards_who_speaks() -> None:
    recipe = social_recipe("medium", 3, 300.0)
    voice = max((s for s in recipe.sources if s.kind == "voice"), key=lambda s: len(s.segments))
    assert len(voice.segments) > 5
    yaw = source_state(recipe, voice.id, sample_times(recipe)).yaw_deg
    assert np.ptp(yaw) > 15.0 and np.abs(np.diff(yaw)).max() <= voice.turn_rate_deg_s * 0.05 + 1.0


# --------------------------------------------------------------------------
# the cost
# --------------------------------------------------------------------------


def test_a_social_scene_is_a_few_positions_where_the_first_was_a_rail() -> None:
    first, second = cost.counts(small_recipe()), cost.counts(moving_recipe())
    assert second["source_positions"] == low_band_source_positions(moving_recipe()).count
    assert first["source_positions"] == low_band_source_positions(small_recipe()).count
    assert second["source_positions_by_kind"]["rail_samples"] == 0
    assert first["source_positions_by_kind"]["rail_samples"] > 10 * second["source_positions"] / 4
    assert second["cells"] >= 1 and second["pairs"] >= second["source_positions"]
    assert sum(second["cells_a_position"]) == second["pairs"]
