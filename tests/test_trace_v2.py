"""A recipe of version 2 through the trace and the engine, in the walled box.

One scene of a second and a half, written by hand, holds every kind of
source a version 2 recipe can: a voice that sways and its breath, somebody
who walks by in silence and whose footsteps are heard, the wearer's own
voice, a noise at a fixture, the
noise of an open window and of a closed one, and a fixture too near a wall
for the wave solver. The plan is made, the bundle built and the trace run on
``numpy`` with a monopole in free air where a card would solve, as
``test_trace.py`` does; the pack is then rendered to stems.

What is held: which sources ask for a wave solve and which are the mirror's
alone; that the plan and :mod:`reverberate.scenes.cost` count the same
thing; that the mirror follows the mouth and the head with their sways
while the low band, the cells and the tail read where they stand; what a
stem of the mirror alone holds under the crossover; and that the labels come
out beside the stems.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from typing import Any

import numpy as np

from reverberate.scenes import Recipe, cost, low_band_source_positions, wave_band
from reverberate.scenes.kinematics import FLOOR_SOURCE_HEIGHT_M
from reverberate.trace.plan import (
    NO_SURFACES,
    SOURCE_CLEARANCE_STEPS,
    Profile,
    make_plan,
    near_a_surface,
    tail_sites_of,
    tracks_of,
)
from test_trace import CLIP, assets, dwell, station

DURATION = 1.5
#: The sources of the scene the mirror renders alone, and those the wave solver answers.
MIRROR_ONLY = ("own_voice", "street_closed", "talker_1_body", "walker_steps")
SOLVED = ("street_open", "talker_1", "tap")


def fixture(name: str, x: float, y: float, z: float, thing: str) -> dict[str, Any]:
    return {
        "id": name,
        "kind": "fixture",
        "position": [x, y, z],
        "height": "fixed",
        "room": "west" if x < 2.0 else "east",
        "facing_yaw_deg": 0.0,
        "object": thing,
    }


def interval(start: float, end: float, voiced: bool = False, gain: float = 0.0) -> dict[str, Any]:
    made: dict[str, Any] = {
        "start_s": start,
        "end_s": end,
        "clip": CLIP,
        "clip_offset_s": 0.0,
        "gain_db": gain,
    }
    if voiced:
        made.update(effort="normal", event="turn")
    return made


def emitter(
    name: str,
    kind: str,
    *,
    subtype: str | None = None,
    segments: list[dict[str, Any]] | None = None,
    activity: list[dict[str, Any]] | None = None,
    level: float = 60.0,
    gain: float = 0.0,
    **more: Any,
) -> dict[str, Any]:
    tree: dict[str, Any] = {
        "id": name,
        "kind": kind,
        "directivity": {"model": "omni", "enabled": False},
        "gain_db": gain,
        "level_spl_1m_db": level,
        "turn_rate_deg_s": 180.0,
        "segments": segments or [],
        "activity": activity if activity is not None else [interval(0.0, DURATION)],
        "sway": [],
        **more,
    }
    if subtype is not None:
        tree["subtype"] = subtype
    return tree


def fixed(name: str) -> list[dict[str, Any]]:
    return [dwell(name, 0.0, DURATION, "fixed")]


def v2_tree() -> dict[str, Any]:
    """The scene, as a tree: a test changes it before it is read."""
    rail = {"id": "r", "a": "c", "b": "d", "points": [[1.2, 0.4], [1.2, 1.2]], "pitch_m": 0.08}
    walk = {
        "type": "travel",
        "rail": "r",
        "from": "c",
        "to": "d",
        "profile": "constant",
        "start_s": 0.5,
        "end_s": DURATION,
        "facing": {"mode": "travel"},
    }
    nobody = [{"start_s": 0.0, "end_s": DURATION, "role": "outside"}]
    sway = [
        {"axis": "x", "amplitude_m": 0.02, "period_s": 0.9, "phase_deg": 30.0},
        {"axis": "yaw", "amplitude_deg": 4.0, "period_s": 1.3, "phase_deg": 0.0},
    ]
    talker = emitter(
        "talker_1",
        "voice",
        segments=[dwell("a", 0.0, DURATION)],
        activity=[interval(0.0, 0.9, voiced=True, gain=2.0)],
        level=58.0,
        gain=-2.0,
        sway=sway,
        roles=[
            {"start_s": 0.0, "end_s": 0.8, "role": "conversation", "group": "g1"},
            {"start_s": 0.8, "end_s": DURATION, "role": "outside"},
        ],
    )
    return {
        "schema": "reverberate.scene-recipe",
        "schema_version": 2,
        "scene": {"calmness": 0.5, "snr_floor_db": 0.0},
        "dwelling": {"name": "box", "scene_id": "walled-box", "floor_y_m": 0.0},
        "assets": {
            "export_sha256": "0" * 64,
            "voxel_low_key": "placeholder",
            "mirror_scene_key": "placeholder",
            "calibration_key": "placeholder",
            "directivity": {},
            "rooms_rule": "adr-0010",
        },
        "seed": 7,
        "duration_s": DURATION,
        "output": {"order": 7, "sample_rate_hz": 48000},
        "atmosphere": {"temperature_c": 20.0, "humidity_percent": 50.0, "pressure_kpa": 101.325},
        "heights": {"standing_m": 1.7, "seated_m": 1.2},
        "stations": [
            station("a", 0.6, 0.8),
            station("c", 1.2, 0.4),
            station("d", 1.2, 1.2),
            station("rest", 1.5, 2.0),
            fixture("tap_7", 3.2, 1.05, 1.0, "kitchen_counter_7"),
            fixture("window_8", 1.0, 1.4, 0.25, "window_8"),
            fixture("window_9", 0.25, 1.4, 2.4, "window_9"),
        ],
        "rails": [rail],
        "sources": [
            talker,
            emitter(
                "talker_1_body",
                "noise",
                subtype="body",
                level=32.0,
                attach={"to": "talker_1", "at": "mouth", "yaw_offset_deg": 40.0},
            ),
            # Nobody talks on the move: the walker is silent, and its steps are heard.
            emitter(
                "walker",
                "voice",
                segments=[dwell("c", 0.0, 0.5), walk],
                activity=[],
                roles=nobody,
            ),
            emitter(
                "walker_steps",
                "noise",
                subtype="steps",
                level=48.0,
                activity=[interval(0.5, DURATION)],
                attach={"to": "walker", "at": "floor", "yaw_offset_deg": 0.0, "stride_m": 0.3},
            ),
            emitter(
                "own_voice",
                "own_voice",
                activity=[interval(0.4, 1.0, voiced=True)],
                attach={
                    "to": "listener",
                    "at": "mouth",
                    "yaw_offset_deg": 0.0,
                    "offset_m": [0.09, 0.0, -0.05],
                },
            ),
            emitter("tap", "noise", subtype="appliance", segments=fixed("tap_7"), level=54.0),
            emitter(
                "street_open",
                "noise",
                subtype="outside",
                segments=fixed("window_9"),
                level=50.0,
                opening={"object": "window_9", "state": "open"},
            ),
            emitter(
                "street_closed",
                "noise",
                subtype="outside",
                segments=fixed("window_8"),
                level=34.0,
                opening={"object": "window_8", "state": "closed"},
            ),
        ],
        "listener": {
            "interpolation": "linear",
            "keyframes": [
                {
                    "t_s": t,
                    "position": list(position),
                    "yaw_deg": yaw,
                    "pitch_deg": 0.0,
                    "roll_deg": 0.0,
                }
                for t, position, yaw in (
                    (0.0, (1.5, 1.7, 2.0), 180.0),
                    (1.0, (1.5, 1.7, 2.0), 150.0),
                    (DURATION, (1.5, 1.7, 2.3), 150.0),
                )
            ],
            "sway": [{"axis": "z", "amplitude_m": 0.015, "period_s": 0.7, "phase_deg": 0.0}],
            "conversation": [{"start_s": 0.0, "end_s": DURATION, "group": "g1"}],
            "gaze": [{"start_s": 0.0, "end_s": 1.0, "mode": "talker", "target": "talker_1"}],
        },
        "generator": None,
    }


@lru_cache(maxsize=1)
def v2_recipe() -> Recipe:
    return Recipe.from_dict(v2_tree())


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def test_only_the_sources_the_wave_solver_answers_ask_for_a_position() -> None:
    recipe = v2_recipe()
    assert tuple(sorted(s.id for s in recipe.sources if not wave_band(s))) == MIRROR_ONLY
    plan = make_plan(recipe, assets().triangles)
    tracks = plan.tracks
    # Every source is traced; four of them by the mirror alone, and one is never heard.
    assert set(tracks.sources) == {*MIRROR_ONLY, *SOLVED, "walker"}
    assert not tracks.sources["walker"].audible.any()
    assert tuple(sorted(plan.record["mirror_only"])) == MIRROR_ONLY
    assert plan.record["near_a_surface"] == {} and plan.profile.mirror_only == ()
    for name in MIRROR_ONLY:
        track = tracks.sources[name]
        assert not track.low and track.audible.any()
        assert np.all(track.slot == -1) and not track.weight.any()
        assert np.all(plan.low[name].cell == -1) and not plan.low[name].mode.any()
    # A station each for the voice, the tap and the open window: the breath, the steps of
    # who walks by in silence and the wearer's voice add none.
    assert plan.record["source_positions"] == 3 == low_band_source_positions(recipe).count
    read = {int(r) for name in SOLVED for r in np.unique(tracks.sources[name].slot) if r >= 0}
    assert read == {0, 1, 2}
    assert plan.pairs == sum(len(cells) for cells in plan.heard_at) >= 3
    assert plan.record["fallback_steps"] == []


def test_the_plan_and_the_recipe_s_cost_are_one_count() -> None:
    """``scenes.cost`` is the plan made without the dwelling: where no wall matters, the plan."""
    recipe = v2_recipe()
    plan = make_plan(recipe, assets().triangles, low_ppw=7.2)
    counted = cost.counts(recipe)
    bare = make_plan(recipe, NO_SURFACES, low_ppw=7.2).record
    for key in ("source_positions", "pairs", "tail_sites", "step_pairs", "mirror_only"):
        assert counted[key] == bare[key] == plan.record[key], key
    assert counted["cells"] == bare["cells"]["cells"] == plan.cells.count
    assert counted["on_the_grid"]["ppw"] == 7.2
    priced = cost.predict(recipe, num_gpus=1, dph_total=0.173)
    assert priced["source_positions"] == 3 and priced["usd"] > 0.0
    # The grid named, the reference's: other nodes, and so other pairs to price.
    assert cost.counts(recipe, low_ppw=10.5)["on_the_grid"]["ppw"] == 10.5


def test_a_track_holds_where_the_low_band_reads_and_where_the_body_is() -> None:
    recipe = v2_recipe()
    tracks = tracks_of(recipe)
    talker = tracks.sources["talker_1"]
    # The voice: its station under the crossover, its sway above.
    assert talker.mouth is not None and talker.anchor is None
    reach = np.linalg.norm(talker.mouth - talker.position, axis=1)
    assert 0.005 < reach.max() <= 0.02 + 1e-9
    assert np.all(talker.position == talker.position[0])
    # The head: the keyframes for the cells, the sway for the mirror.
    assert tracks.head is not None
    assert 0.005 < np.abs(tracks.head - tracks.listener).max() <= 0.015 + 1e-9
    assert np.array_equal(tracks.heard_from, tracks.head)
    # The breath is at the talker's mouth, sway and all, and its tail reads the talker's place.
    body = tracks.sources["talker_1_body"]
    assert np.array_equal(body.traced_from, talker.mouth)
    assert np.array_equal(body.tail_from, talker.position)
    # A footfall is 5 cm above the floor and hops; its tail is its walker's.
    steps = tracks.sources["walker_steps"]
    assert np.allclose(steps.traced_from[:, 1], FLOOR_SOURCE_HEIGHT_M)
    hops = np.linalg.norm(np.diff(steps.traced_from, axis=0), axis=1)
    # Every stride of 0.30 m, and the rail's end.
    assert np.all((hops < 1e-9) | (hops > 0.15)) and (hops > 0.15).sum() == 3
    assert np.array_equal(steps.tail_from, tracks.sources["walker"].position)
    # The wearer's own voice is 0.10 m from the head's centre, and its tail is the head's.
    own = tracks.sources["own_voice"]
    away = np.linalg.norm(own.traced_from - tracks.heard_from, axis=1)
    assert np.allclose(away, np.hypot(0.09, 0.05), atol=1e-9)
    assert np.array_equal(own.tail_from, tracks.listener)


def test_the_tail_of_a_carried_source_is_cast_from_its_carrier_s_sites() -> None:
    recipe = v2_recipe()
    tracks = tracks_of(recipe)
    talker = tail_sites_of(recipe, tracks.sources["talker_1"])
    assert talker.count == 1 and np.allclose(talker.positions[0], [0.6, 1.7, 0.8])
    # The steps are heard on their walker's rail, at its height: a site every 0.80 m.
    steps = tail_sites_of(recipe, tracks.sources["walker_steps"])
    assert np.all(steps.rail >= 0) and np.allclose(steps.positions[:, 1], 1.7)
    assert np.allclose(steps.positions[[0, -1]], [[1.2, 1.7, 0.4], [1.2, 1.7, 1.2]])
    # Who is never heard casts no ray.
    assert tail_sites_of(recipe, tracks.sources["walker"]).count == 0
    body = tail_sites_of(recipe, tracks.sources["talker_1_body"])
    assert np.allclose(body.positions[0], talker.positions[0])
    # The wearer's voice: where the head rests, and its way while the voice still rings.
    own = tail_sites_of(recipe, tracks.sources["own_voice"])
    assert np.allclose(own.positions[0], [1.5, 1.7, 2.0]) and own.rail[0] == -1
    assert (own.rail >= 0).sum() == 2 and np.allclose(own.positions[-1], [1.5, 1.7, 2.3])
    # A fixture's is where it stands.
    tap = tail_sites_of(recipe, tracks.sources["tap"])
    assert tap.count == 1 and np.allclose(tap.positions[0], [3.2, 1.05, 1.0])


def test_a_fixture_too_near_a_surface_is_given_to_the_mirror_and_said() -> None:
    """The solver refuses a source that touches a wall's nodes: found here, before a rental."""
    tree = v2_tree()
    tap = next(s for s in tree["stations"] if s["id"] == "tap_7")
    tap["position"] = [3.94, 1.05, 1.0]  # 6 cm from the wall at x = 4
    recipe = Recipe.from_dict(tree)
    held = assets()
    bare = tracks_of(recipe)
    near = near_a_surface(bare, held.triangles, ppw=7.2)
    assert near == {"tap": 0.06}
    # Three steps of the grid at 7.2 points to 1500 Hz are 95 mm; at 10.5 points, 65 mm.
    assert near_a_surface(bare, held.triangles, ppw=10.5) == near
    assert SOURCE_CLEARANCE_STEPS * 343.2 / (1500.0 * 10.5) > 0.06
    plan = make_plan(recipe, held.triangles, low_ppw=7.2)
    assert plan.profile.mirror_only == ("tap",) and plan.record["near_a_surface"] == near
    assert not plan.tracks.sources["tap"].low and plan.record["source_positions"] == 2
    # The machine reads the profile and holds the same positions.
    again = Profile.from_record(json.loads(json.dumps(plan.profile.record())))
    assert again == plan.profile
    assert np.array_equal(tracks_of(recipe, again).positions, plan.tracks.positions)
    # A profile of a recipe that needs none says none, as every record before.
    assert "mirror_only" not in Profile().record()
    assert replace(Profile(), mirror_only=("b", "a")).mirror_only == ("a", "b")


def test_more_positions_a_step_leave_a_source_of_the_mirror_alone_without_any() -> None:
    recipe = v2_recipe()
    tracks = tracks_of(recipe, Profile(rail_positions=4))
    own = tracks.sources["own_voice"]
    assert own.rail_slot is not None and np.all(own.rail_slot == -1)
    assert own.rail_weight is not None and not own.rail_weight.any()
    assert not own.low and all(tracks.sources[name].low for name in SOLVED)
