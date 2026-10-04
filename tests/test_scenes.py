"""Tests for moving scenes: the recipe, its rules, where things are, the generator.

Everything runs on the hand-built dwelling of ``scene_floor``: two rooms and a
door, drawn in a few lines, so nothing here needs HSSD or a network. The
properties that matter are the ones other lots lean on: the identity of a
recipe is the same bytes in two processes, every numbered rule refuses the
recipe that breaks it and says its number, and the kinematics are continuous
and within the speeds the format allows.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import (
    Parameters,
    Recipe,
    RecipeError,
    audible_steps,
    canonical_bytes,
    check,
    describe,
    generate,
    listener_state,
    low_band_positions,
    low_band_source_positions,
    parse_recipe,
    placeholder_assets,
    quantise,
    rail_arc_lengths,
    rail_length,
    rail_samples,
    recipe_sha256,
    sample_times,
    save_recipe,
    seat_rail_heights,
    source_state,
    validate,
    validate_text,
)
from reverberate.scenes.__main__ import main
from reverberate.scenes.generate import stream
from reverberate.scenes.kinematics import yaw_of_direction
from reverberate.scenes.recipe import Rail, Travel
from scene_floor import FLOOR_Y_M, SMALL, small_recipe, two_room_floor, two_room_layout

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "ab" * 32


# --------------------------------------------------------------------------
# a recipe written by hand on the two rooms
# --------------------------------------------------------------------------


def hand_tree() -> dict[str, Any]:
    """Thirty seconds: a voice that walks one rail, a second that stays, a listener at rest."""
    layout = two_room_layout()
    stations = {station.id: station for station in layout.stations}
    rail = next(
        rail
        for rail in layout.rails
        if len(rail.points) == 2
        and stations[rail.a].kind == "stand"
        and stations[rail.b].kind == "stand"
    )
    seat = next(station for station in layout.stations if station.kind == "seat")
    seat_rail = next(r for r in layout.rails if r.b == seat.id)
    far = max(
        (s for s in layout.stations if s.kind == "stand" and s.room == "bedroom"),
        key=lambda s: s.position[0],
    )
    clip = {"library": "ears", "name": "p001/sentences_01_regular", "sha256": DIGEST}
    walker = {
        "id": "walker",
        "kind": "near_voice",
        "directivity": {"model": "voice_v1", "enabled": True},
        "gain_db": 0.0,
        "turn_rate_deg_s": 180.0,
        "segments": [
            {
                "type": "dwell",
                "station": rail.a,
                "height": "standing",
                "start_s": 0.0,
                "end_s": 10.0,
                "facing": {"mode": "listener"},
            },
            {
                "type": "travel",
                "rail": rail.id,
                "from": rail.a,
                "to": rail.b,
                "profile": "smoothstep",
                "start_s": 10.0,
                "end_s": 14.0,
                "facing": {"mode": "travel"},
            },
            {
                "type": "dwell",
                "station": rail.b,
                "height": "standing",
                "start_s": 14.0,
                "end_s": 30.0,
                "facing": {"mode": "fixed", "yaw_deg": 90.0},
            },
        ],
        "activity": [
            {"start_s": 1.0, "end_s": 8.0, "clip": clip, "clip_offset_s": 0.0, "gain_db": 0.0},
            {"start_s": 12.0, "end_s": 20.0, "clip": clip, "clip_offset_s": 7.0, "gain_db": 0.0},
        ],
    }
    sitter = {
        "id": "sitter",
        "kind": "far_voice",
        "directivity": {"model": "voice_v1", "enabled": False},
        "gain_db": -3.0,
        "turn_rate_deg_s": 120.0,
        "segments": [
            {
                "type": "dwell",
                "station": seat.id,
                "height": "seated",
                "start_s": 0.0,
                "end_s": 20.0,
                "facing": {"mode": "fixed", "yaw_deg": seat.facing_yaw_deg},
            },
            {"type": "rise", "station": seat.id, "to": "standing", "start_s": 20.0, "end_s": 22.0},
            {
                "type": "travel",
                "rail": seat_rail.id,
                "from": seat.id,
                "to": seat_rail.a,
                "profile": "constant",
                "start_s": 22.0,
                "end_s": 28.0,
                "facing": {"mode": "travel"},
            },
            {
                "type": "dwell",
                "station": seat_rail.a,
                "height": "standing",
                "start_s": 28.0,
                "end_s": 30.0,
                "facing": {"mode": "fixed", "yaw_deg": 0.0},
            },
        ],
        "activity": [],
    }
    head = [far.position[0], far.position[1], far.position[2]]
    used = {rail.a, rail.b, seat.id, seat_rail.a, far.id}
    return {
        "schema": "reverberate.scene-recipe",
        "schema_version": 1,
        "dwelling": layout.dwelling.to_dict(),
        "assets": placeholder_assets().to_dict(),
        "seed": 7,
        "duration_s": 30.0,
        "output": {"order": 7, "sample_rate_hz": 48000},
        "atmosphere": {"temperature_c": 20.0, "humidity_percent": 50.0, "pressure_kpa": 101.325},
        "heights": {"standing_m": 1.7, "seated_m": 1.2},
        "stations": [s.to_dict() for s in layout.stations if s.id in used],
        "rails": [rail.to_dict(), seat_rail.to_dict()],
        "sources": [walker, sitter],
        "listener": {
            "interpolation": "linear",
            "keyframes": [
                {"t_s": 0.0, "position": head, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
                {"t_s": 30.0, "position": head, "yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            ],
        },
        "generator": None,
    }


def rules_of(tree: dict[str, Any]) -> set[int]:
    return {violation.rule for violation in validate_text(json.dumps(tree), two_room_floor())}


def test_the_hand_written_recipe_is_valid() -> None:
    assert validate_text(json.dumps(hand_tree()), two_room_floor()) == []


# --------------------------------------------------------------------------
# the format's own example, and the canonical form
# --------------------------------------------------------------------------


def documented_example() -> str:
    text = (ROOT / "docs" / "formats" / "scene-recipe.md").read_text()
    found = re.search(r"## Example.*?```json\n(.*?)```", text, flags=re.DOTALL)
    assert found is not None
    return found.group(1)


def test_the_documents_example_is_a_recipe_this_package_reads() -> None:
    """The contract and its reader agree, on the one recipe the contract shows."""
    recipe = parse_recipe(documented_example())

    assert validate(recipe) == []
    assert recipe.source("v1").segments[2] == Travel(
        "armchair_counter",
        "armchair",
        "counter",
        "smoothstep",
        10.5,
        15.5,
        recipe.source("v1").segments[2].facing,  # type: ignore[union-attr]
    )


def test_the_canonical_form_round_trips_and_ignores_indentation() -> None:
    recipe = parse_recipe(documented_example())
    payload = canonical_bytes(recipe)

    assert payload.endswith(b"}\n") and b" " not in payload.replace(b"living room", b"")
    assert parse_recipe(payload) == recipe
    assert canonical_bytes(parse_recipe(payload)) == payload
    indented = json.dumps(json.loads(payload), indent=4, sort_keys=False)
    assert recipe_sha256(parse_recipe(indented)) == recipe_sha256(recipe)
    assert list(json.loads(payload)) == sorted(json.loads(payload))


def test_a_generated_recipe_round_trips_through_its_file(tmp_path: Path) -> None:
    recipe = small_recipe(1)

    digest = save_recipe(recipe, tmp_path / "recipe.json")

    assert digest == recipe_sha256(recipe)
    assert parse_recipe((tmp_path / "recipe.json").read_bytes()) == recipe
    assert quantise(recipe) == recipe


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ((), ("surprise", 1)),
        (("dwelling",), ("storey", 2)),
        (("sources", 0), ("colour", "red")),
        (("sources", 0, "segments", 1), ("speed_m_s", 1.0)),
        (("listener", "keyframes", 0), ("height", "standing")),
        (("rails", 0), ("length_m", 1.2)),
    ],
)
def test_an_unknown_key_is_refused_at_every_level(path: tuple[Any, ...], value: Any) -> None:
    tree = hand_tree()
    node: Any = tree
    for step in path:
        node = node[step]
    node[value[0]] = value[1]

    with pytest.raises(RecipeError) as refused:
        parse_recipe(json.dumps(tree))

    assert refused.value.rule == 1 and value[0] in str(refused.value)


def test_a_float_written_as_an_integer_and_a_missing_key_are_rule_one() -> None:
    whole = hand_tree()
    whole["duration_s"] = 30
    missing = hand_tree()
    del missing["atmosphere"]["pressure_kpa"]
    tangent = hand_tree()
    tangent["sources"][0]["segments"][0]["facing"] = {"mode": "travel"}

    for tree in (whole, missing, tangent):
        assert rules_of(tree) == {1}


def test_numbers_are_rounded_to_their_step_and_rule_eleven_sees_the_rest() -> None:
    tree = hand_tree()
    tree["sources"][0]["gain_db"] = -0.004
    tree["listener"]["keyframes"][1]["yaw_deg"] = 90.004
    tree["stations"][0]["position"][0] += 0.0004

    assert 11 in rules_of(tree)
    rounded = quantise(Recipe.from_dict(tree))
    assert rounded.sources[0].gain_db == 0.0 and str(rounded.sources[0].gain_db) == "0.0"
    assert rounded.listener.keyframes[1].yaw_deg == 90.0
    assert 11 not in {v.rule for v in validate(rounded)}


# --------------------------------------------------------------------------
# every rule, with the recipe that breaks it
# --------------------------------------------------------------------------


def gap_in_segments(tree: dict[str, Any]) -> None:
    tree["sources"][0]["segments"][1]["start_s"] = 10.5


def scene_not_covered(tree: dict[str, Any]) -> None:
    tree["sources"][0]["segments"][-1]["end_s"] = 29.0


def activity_overlaps(tree: dict[str, Any]) -> None:
    tree["sources"][0]["activity"][1]["start_s"] = 7.5


def activity_outside(tree: dict[str, Any]) -> None:
    tree["sources"][0]["activity"][1]["end_s"] = 31.0


def keyframes_stop_early(tree: dict[str, Any]) -> None:
    tree["listener"]["keyframes"][1]["t_s"] = 29.0


def stand_in_the_wall(tree: dict[str, Any]) -> None:
    station = next(s for s in tree["stations"] if s["kind"] == "stand")
    station["position"][0] = 0.1
    for rail in tree["rails"]:
        for end, index in (("a", 0), ("b", -1)):
            if rail[end] == station["id"]:
                rail["points"][index][0] = 0.1


def stand_in_the_other_room(tree: dict[str, Any]) -> None:
    next(s for s in tree["stations"] if s["kind"] == "stand")["room"] = "bedroom"


def seat_far_from_its_object(tree: dict[str, Any]) -> None:
    next(s for s in tree["stations"] if s["kind"] == "seat")["object"] = "seat_2"


def seat_at_standing_height(tree: dict[str, Any]) -> None:
    next(s for s in tree["stations"] if s["kind"] == "seat")["height"] = "standing"


def station_twice(tree: dict[str, Any]) -> None:
    tree["stations"].append(copy.deepcopy(tree["stations"][0]))


def rail_at_another_pitch(tree: dict[str, Any]) -> None:
    tree["rails"][0]["pitch_m"] = 0.1


def rail_through_the_couch(tree: dict[str, Any]) -> None:
    tree["rails"][0]["points"].insert(1, [2.0, 0.5])


def rail_not_at_its_station(tree: dict[str, Any]) -> None:
    tree["rails"][0]["points"][0][0] += 0.05


def travel_from_the_wrong_end(tree: dict[str, Any]) -> None:
    travel = tree["sources"][0]["segments"][1]
    travel["from"], travel["to"] = travel["to"], travel["from"]


def seated_on_the_floor(tree: dict[str, Any]) -> None:
    tree["sources"][0]["segments"][0]["height"] = "seated"


def travel_while_seated(tree: dict[str, Any]) -> None:
    segments = tree["sources"][1]["segments"]
    del segments[1]
    segments[1]["start_s"] = 20.0


def dwell_somewhere_else(tree: dict[str, Any]) -> None:
    segments = tree["sources"][0]["segments"]
    segments[2]["station"] = segments[0]["station"]


def listener_in_the_wall(tree: dict[str, Any]) -> None:
    tree["listener"]["keyframes"][1]["position"][0] = 8.95


def listener_crouches(tree: dict[str, Any]) -> None:
    for frame in tree["listener"]["keyframes"]:
        frame["position"][1] = FLOOR_Y_M + 1.2


def listener_through_the_wall(tree: dict[str, Any]) -> None:
    """Both keyframes on the free floor, the straight line between them through a wall."""
    frames = tree["listener"]["keyframes"]
    frames[0]["position"] = [4.0, FLOOR_Y_M + 1.7, 3.5]
    frames[1]["position"] = [6.0, FLOOR_Y_M + 1.7, 3.5]


def travel_too_fast(tree: dict[str, Any]) -> None:
    segments = tree["sources"][0]["segments"]
    segments[1]["end_s"] = 10.5
    segments[2]["start_s"] = 10.5


def rise_too_fast(tree: dict[str, Any]) -> None:
    segments = tree["sources"][1]["segments"]
    segments[1]["end_s"] = 20.5
    segments[2]["start_s"] = 20.5


def listener_too_fast(tree: dict[str, Any]) -> None:
    frames = tree["listener"]["keyframes"]
    frames.insert(1, copy.deepcopy(frames[0]))
    frames[1]["t_s"] = 29.5
    frames[2]["position"][2] -= 1.0


def head_spins(tree: dict[str, Any]) -> None:
    frames = tree["listener"]["keyframes"]
    frames.insert(1, copy.deepcopy(frames[0]))
    frames[1]["t_s"] = 29.9


def listener_on_the_voice(tree: dict[str, Any]) -> None:
    station = next(
        s for s in tree["stations"] if s["id"] == tree["sources"][0]["segments"][2]["station"]
    )
    for frame in tree["listener"]["keyframes"]:
        frame["position"] = [station["position"][0] + 0.3, station["position"][1], 1.88]


def two_voices_on_one_spot(tree: dict[str, Any]) -> None:
    twin = copy.deepcopy(tree["sources"][0])
    twin["id"] = "twin"
    tree["sources"].append(twin)


def two_voices_meet_on_a_rail(tree: dict[str, Any]) -> None:
    twin = copy.deepcopy(tree["sources"][0])
    twin["id"] = "twin"
    first, travel, last = twin["segments"]
    first["station"], last["station"] = last["station"], first["station"]
    travel["from"], travel["to"] = travel["to"], travel["from"]
    tree["sources"].append(twin)


def clip_without_a_digest(tree: dict[str, Any]) -> None:
    tree["sources"][0]["activity"][0]["clip"] = {"library": "ears", "name": "x", "sha256": "abc"}


def directivity_without_a_table(tree: dict[str, Any]) -> None:
    tree["sources"][0]["directivity"]["model"] = "voice_v9"


def another_temperature(tree: dict[str, Any]) -> None:
    tree["atmosphere"]["temperature_c"] = 23.0


def off_the_millimetre(tree: dict[str, Any]) -> None:
    tree["listener"]["keyframes"][0]["position"][2] += 0.0004


@pytest.mark.parametrize(
    ("rule", "breaks"),
    [
        (2, gap_in_segments),
        (2, scene_not_covered),
        (2, activity_overlaps),
        (2, activity_outside),
        (2, keyframes_stop_early),
        (3, stand_in_the_wall),
        (3, stand_in_the_other_room),
        (3, seat_far_from_its_object),
        (3, seat_at_standing_height),
        (3, station_twice),
        (4, rail_at_another_pitch),
        (4, rail_through_the_couch),
        (4, rail_not_at_its_station),
        (5, travel_from_the_wrong_end),
        (5, seated_on_the_floor),
        (5, travel_while_seated),
        (5, dwell_somewhere_else),
        (6, listener_in_the_wall),
        (6, listener_crouches),
        (6, listener_through_the_wall),
        (7, travel_too_fast),
        (7, rise_too_fast),
        (7, listener_too_fast),
        (7, head_spins),
        (8, listener_on_the_voice),
        (8, two_voices_on_one_spot),
        (9, two_voices_meet_on_a_rail),
        (10, clip_without_a_digest),
        (10, directivity_without_a_table),
        (10, another_temperature),
        (11, off_the_millimetre),
    ],
)
def test_each_rule_refuses_the_recipe_that_breaks_it(rule: int, breaks: Any) -> None:
    tree = hand_tree()
    breaks(tree)

    found = validate_text(json.dumps(tree), two_room_floor())

    assert rule in {violation.rule for violation in found}
    assert all(str(violation).startswith(f"rule {violation.rule}: ") for violation in found)
    with pytest.raises(RecipeError) as refused:
        check(Recipe.from_dict(tree), two_room_floor())
    assert refused.value.rule == min(violation.rule for violation in found)


def test_without_the_floor_the_geometric_halves_are_skipped_not_passed() -> None:
    tree = hand_tree()
    stand_in_the_wall(tree)
    recipe = Recipe.from_dict(tree)

    assert validate(recipe) == []
    assert 3 in {violation.rule for violation in validate(recipe, two_room_floor())}


# --------------------------------------------------------------------------
# where things are
# --------------------------------------------------------------------------


def test_yaw_zero_faces_x_and_ninety_faces_minus_z() -> None:
    assert yaw_of_direction(1.0, 0.0) == pytest.approx(0.0)
    assert yaw_of_direction(0.0, -1.0) == pytest.approx(90.0)
    assert yaw_of_direction(-1.0, 0.0) == pytest.approx(180.0)


def test_a_rail_is_sampled_every_eight_centimetres_and_at_its_end() -> None:
    straight = Rail("r", "a", "b", ((0.0, 0.0), (1.0, 0.0)))
    exact = Rail("r", "a", "b", ((0.0, 0.0), (0.0, 0.4), (0.4, 0.4)))

    assert rail_arc_lengths(straight) == pytest.approx([*np.arange(13) * 0.08, 1.0])
    assert rail_samples(straight)[-1] == pytest.approx([1.0, 0.0])
    # 0.80 m is ten pitches: b is the tenth sample and is not written twice.
    assert len(rail_samples(exact)) == 11
    assert rail_samples(exact)[5] == pytest.approx([0.0, 0.4])
    assert rail_samples(exact)[-1] == pytest.approx([0.4, 0.4])
    assert rail_length(exact) == pytest.approx(0.8)


def test_a_seats_vertical_rail_runs_from_seated_to_standing() -> None:
    heights = seat_rail_heights(Recipe.from_dict(hand_tree()))

    assert heights[0] == pytest.approx(1.2) and heights[-1] == pytest.approx(1.7)
    assert np.diff(heights)[:-1] == pytest.approx(0.08)


def test_a_source_is_at_its_station_on_its_rail_and_between_two_samples() -> None:
    recipe = Recipe.from_dict(hand_tree())
    rail = recipe.rails[0]
    start, end = recipe.station(rail.a), recipe.station(rail.b)

    state = source_state(recipe, "walker", [0.0, 5.0, 10.0, 12.0, 14.0, 30.0])

    # The floor is at 0.3 m: a height is above it, a position is whole.
    assert state.height_m == pytest.approx(1.7)
    assert state.position[:3] == pytest.approx(np.tile(start.position, (3, 1)))
    assert state.position[4:] == pytest.approx(np.tile(end.position, (2, 1)))
    # Smoothstep: half the rail at half the time.
    middle = 0.5 * (np.array(start.position) + np.array(end.position))
    assert state.position[3] == pytest.approx(middle)
    assert list(state.segment) == [0, 0, 1, 1, 2, 2]
    assert state.sample_a[0] == -1 and state.sample_a[-1] == -1
    samples = rail_samples(rail)
    a, b, w = state.sample_a[3], state.sample_b[3], state.weight[3]
    assert b == a + 1
    assert (1 - w) * samples[a] + w * samples[b] == pytest.approx(state.position[3, [0, 2]])


def test_a_rise_goes_up_the_seats_rail() -> None:
    recipe = Recipe.from_dict(hand_tree())

    state = source_state(recipe, "sitter", [10.0, 20.0, 21.0, 22.0])

    assert state.height_m == pytest.approx([1.2, 1.2, 1.45, 1.7])
    rungs = seat_rail_heights(recipe)
    a, b, w = state.sample_a[2], state.sample_b[2], state.weight[2]
    assert (1 - w) * rungs[a] + w * rungs[b] == pytest.approx(1.45)
    seat = recipe.station(recipe.source("sitter").segments[0].station)  # type: ignore[union-attr]
    assert state.position[:, [0, 2]] == pytest.approx(np.tile(np.array(seat.xz), (4, 1)))


def test_a_source_turns_to_its_facing_no_faster_than_its_rate() -> None:
    recipe = Recipe.from_dict(hand_tree())
    times = np.arange(0.0, 30.0001, 0.05)

    state = source_state(recipe, "walker", times)
    head = listener_state(recipe, times).position

    # Facing the listener from the first instant, at rest.
    gaze = head[0] - state.position[0]
    assert state.yaw_deg[0] == pytest.approx(float(yaw_of_direction(gaze[0], gaze[2])))
    assert np.abs(np.diff(state.yaw_deg)).max() <= 180.0 * 0.05 + 1e-9
    # On the rail the yaw goes to the tangent; at the last station to 90.
    assert np.cos(np.radians(state.yaw_deg[-1] - 90.0)) == pytest.approx(1.0)


def test_the_listener_is_linear_between_keyframes() -> None:
    recipe = Recipe.from_dict(hand_tree())

    state = listener_state(recipe, [0.0, 10.0, 30.0])

    assert state.yaw_deg == pytest.approx([0.0, 30.0, 90.0])
    assert state.height_m == pytest.approx(1.7)
    assert list(state.keyframe) == [0, 0, 0]


@pytest.mark.parametrize("seed", [1, 2])
def test_a_generated_scene_moves_continuously_and_within_the_speeds(seed: int) -> None:
    """Sampled every 10 ms: nobody jumps, nobody exceeds 1.5 m/s or a turn rate."""
    recipe = small_recipe(seed)
    step = 0.01
    times = np.arange(0.0, recipe.duration_s + 1e-9, step)

    head = listener_state(recipe, times)
    assert np.linalg.norm(np.diff(head.position, axis=0), axis=1).max() <= 1.5 * step + 1e-9
    assert np.abs(np.diff(head.yaw_deg)).max() <= 360.0 * step + 1e-9
    assert head.height_m.min() >= 1.2 - 1e-9 and head.height_m.max() <= 1.7 + 1e-9
    for source in recipe.sources:
        state = source_state(recipe, source.id, times)
        moved = np.linalg.norm(np.diff(state.position, axis=0), axis=1)
        assert moved.max() <= 1.5 * step + 1e-9
        assert np.abs(np.diff(state.yaw_deg)).max() <= source.turn_rate_deg_s * step + 1e-6
        assert np.linalg.norm(state.position - head.position, axis=1).min() >= 0.5
        # Between two samples of a rail the source is within half a pitch of their mix.
        on_rail = np.flatnonzero([isinstance(source.segments[k], Travel) for k in state.segment])
        for index in on_rail[:: max(1, len(on_rail) // 40)]:
            segment = source.segments[state.segment[index]]
            assert isinstance(segment, Travel)
            samples = rail_samples(recipe.rail(segment.rail))
            w = state.weight[index]
            mix = (1 - w) * samples[state.sample_a[index]] + w * samples[state.sample_b[index]]
            assert np.linalg.norm(mix - state.position[index, [0, 2]]) <= 0.04


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------


def test_the_layout_has_seats_spots_a_door_and_one_connected_graph() -> None:
    layout = two_room_layout()
    kinds = [station.kind for station in layout.stations]
    stands = np.array([s.xz for s in layout.stations if s.kind == "stand"])

    assert kinds.count("seat") == 5 and kinds.count("waypoint") == 1
    apart = np.linalg.norm(stands[:, None] - stands[None], axis=2) + 10 * np.eye(len(stands))
    assert apart.min() >= 1.0 - 1e-6
    assert {s.position[1] for s in layout.stations if s.kind != "seat"} == {FLOOR_Y_M + 1.7}
    assert {s.position[1] for s in layout.stations if s.kind == "seat"} == {FLOOR_Y_M + 1.2}
    reached, frontier = {layout.stations[0].id}, [layout.stations[0].id]
    while frontier:
        here = frontier.pop()
        for rail in layout.rails:
            for one, other in ((rail.a, rail.b), (rail.b, rail.a)):
                if one == here and other not in reached:
                    reached.add(other)
                    frontier.append(other)
    assert reached == {station.id for station in layout.stations}
    assert max(rail_length(rail) for rail in layout.rails) <= layout.settings.max_rail_m


def test_every_station_and_rail_of_the_layout_passes_the_formats_rules() -> None:
    layout = two_room_layout()
    tree = hand_tree()
    tree["stations"] = [station.to_dict() for station in layout.stations]
    tree["rails"] = [rail.to_dict() for rail in layout.rails]

    assert validate_text(json.dumps(tree), layout.floor) == []


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_generated_recipe_is_valid_for_several_seeds(seed: int) -> None:
    recipe = small_recipe(seed)

    assert validate(recipe, two_room_floor()) == []
    assert [source.kind for source in recipe.sources] == [
        "near_voice",
        "near_voice",
        "far_voice",
        "far_voice",
        "noise",
    ]
    assert recipe.generator is not None
    assert Parameters.from_record(recipe.generator.parameters) == SMALL
    assert recipe.generator.parameters["clips"] == {"placeholder": True}
    # Only what the scene uses is written.
    named = {s.station for src in recipe.sources for s in src.segments if hasattr(s, "station")}
    assert named <= {station.id for station in recipe.stations}
    assert all(any(source.activity) for source in recipe.sources)


def test_two_seeds_give_two_scenes_and_one_seed_one() -> None:
    again = generate(
        two_room_layout(),
        SMALL,
        1,
        allow_placeholder_clips=True,
        allow_placeholder_assets=True,
    )

    assert canonical_bytes(again) == canonical_bytes(small_recipe(1))
    assert recipe_sha256(small_recipe(1)) != recipe_sha256(small_recipe(2))


def test_the_same_seed_gives_the_same_bytes_in_two_processes() -> None:
    """Under two hash seeds, so nothing leans on a set's or a dict's order."""
    script = (
        "from scene_floor import small_recipe\n"
        "from reverberate.scenes import recipe_sha256\n"
        "print(recipe_sha256(small_recipe(3)))\n"
    )
    digests = []
    for hash_seed in ("1", "2"):
        environment = {
            **os.environ,
            "PYTHONHASHSEED": hash_seed,
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")]),
        }
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        digests.append(done.stdout.strip())

    assert digests[0] == digests[1] == recipe_sha256(small_recipe(3))


def test_a_stream_depends_on_its_seed_and_label_only() -> None:
    assert stream(5, "listener").integers(1 << 30) == stream(5, "listener").integers(1 << 30)
    assert stream(5, "listener").integers(1 << 30) != stream(5, "source:near_1").integers(1 << 30)
    assert stream(5, "listener").integers(1 << 30) != stream(6, "listener").integers(1 << 30)


def test_placeholders_are_written_only_when_asked_for_by_name() -> None:
    with pytest.raises(ValueError, match="allow_placeholder_clips"):
        generate(two_room_layout(), SMALL, 1, allow_placeholder_assets=True)
    with pytest.raises(ValueError, match="allow_placeholder_assets"):
        generate(two_room_layout(), SMALL, 1, allow_placeholder_clips=True)

    clips = {a.clip.library for s in small_recipe(1).sources for a in s.activity}
    assert clips == {"placeholder"}


def test_the_parameters_tree_gives_the_parameters_back() -> None:
    record = Parameters().record()

    assert Parameters.from_record(json.loads(json.dumps(record))) == Parameters()
    counts = [record["sources"][kind]["count"][0] for kind in ("near_voice", "far_voice", "noise")]
    assert sum(counts) == 14 and record["duration_s"] == 1200.0


# --------------------------------------------------------------------------
# the summary and the command line
# --------------------------------------------------------------------------


def test_the_summary_names_every_source_and_counts_the_low_band_positions() -> None:
    recipe = small_recipe(1)

    text = describe(recipe)
    positions = low_band_positions(recipe)

    assert recipe_sha256(recipe) in text
    assert all(source.id in text for source in recipe.sources)
    assert "metres" in text and "active" in text and "visits:" in text
    assert positions["rail_samples"] > 0 and positions["stations"] > 0
    # Both counts are printed: every position passed through, and those a source is heard at.
    parts = ("stations", "rail_samples", "seat_rail_samples")
    assert positions["all"] == sum(positions[name] for name in parts)
    assert 0 < positions["audible"] <= positions["all"]
    assert f"low band source positions: {positions['all']} (" in text
    assert f"{positions['audible']} where a source is audible" in text
    assert "speech on the move:" in text


def test_a_position_is_solved_only_where_its_source_is_heard() -> None:
    recipe = Recipe.from_dict(hand_tree())
    every = low_band_source_positions(recipe, audible_only=False)
    heard = low_band_source_positions(recipe)
    walker = next(s for s in recipe.sources if s.id == "walker")
    rail = recipe.rail(next(s.rail for s in walker.segments if isinstance(s, Travel)))
    # A rail's two ends are its stations: counted once, with the rail.
    assert every.by_source["walker"] == tuple(range(len(rail_samples(rail))))
    assert set(every.kind[row] for row in every.by_source["walker"]) == {"rail"}
    assert "seat_rail" in every.kind
    # The sitter never speaks: nothing is solved for it. The walker speaks at its
    # first station and again from the middle of its walk.
    assert heard.by_source["sitter"] == ()
    assert 0 < heard.count < len(rail_samples(rail))
    held = {tuple(p) for p in every.positions.tolist()}
    assert {tuple(p) for p in heard.positions.tolist()} <= held
    # The steps the pack calls audible: in an interval, or 1.2 s after one.
    steps = audible_steps(recipe, "walker")
    times = sample_times(recipe)
    assert steps[times == 1.0].all() and steps[times == 9.2].all()
    assert not steps[times == 9.25].any() and not steps[times == 0.95].any()


def test_the_command_line_describes_and_validates_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "recipe.json"
    save_recipe(small_recipe(1), path)

    assert main(["describe", str(path)]) == 0
    assert "near_1" in capsys.readouterr().out
    assert main(["validate", str(path), "--no-floor"]) == 0
    said = capsys.readouterr().out
    assert "valid" in said and "rules 3, 4, 6 were checked in part" in said

    broken = small_recipe(1).to_dict()
    broken["atmosphere"]["temperature_c"] = 25.0
    path.write_text(json.dumps(broken))
    assert main(["validate", str(path), "--no-floor"]) == 1
    assert "rule 10" in capsys.readouterr().out
