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
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from reverberate.compute import Devices
from reverberate.gpu import onebox
from reverberate.mirror.moving import MovingSettings, prepare, trace_early
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.render import band_pulse_energy
from reverberate.mirror.tails import TailCache, histograms, tail_scale
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.labels import labels, labels_path, write_labels
from reverberate.render.pack import KIND_DIRECT, PackError, band_map, read_pack, validate
from reverberate.scenes import Recipe, cost, low_band_source_positions, wave_band
from reverberate.scenes.kinematics import FLOOR_SOURCE_HEIGHT_M
from reverberate.scenes.levels import VOICE_REFERENCE_DB
from reverberate.trace import machines, mirror_only
from reverberate.trace.driver import describe
from reverberate.trace.plan import (
    LOW_PPW,
    NO_SURFACES,
    SOURCE_CLEARANCE_STEPS,
    Profile,
    estimate,
    make_plan,
    near_a_surface,
    tail_sites_of,
    tracks_of,
)
from scene_floor import FLOOR_Y_M
from test_preflight import Client, Offer, a_bundle
from test_scenes_social import social_recipe
from test_trace import CLIP, assets, dwell, station, traced

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


def interval(
    start: float, end: float, voiced: bool = False, gain: float = 0.0, effort: str = "normal"
) -> dict[str, Any]:
    made: dict[str, Any] = {
        "start_s": start,
        "end_s": end,
        "clip": CLIP,
        "clip_offset_s": 0.0,
        "gain_db": gain,
    }
    if voiced:
        made.update(effort=effort, event="turn")
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
        # A raised voice: 58 dB its own, 5 dB more for the turn, the clip stored at 60.
        activity=[interval(0.0, 0.9, voiced=True, gain=5.0, effort="raised")],
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
    # And whoever rents reads it before renting.
    text = describe(plan, estimate(plan, rate_usd_per_hour=0.136, low_ppw=7.2))
    assert "by the mirror alone, no wave solve: 5 (" in text
    assert "too near a surface for the wave solver: tap at 0.06 m" in text
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


# --------------------------------------------------------------------------
# what the second generator writes, planned and priced without a machine
# --------------------------------------------------------------------------


def shell(x: float, y0: float, y1: float, z: float) -> np.ndarray:
    """The six faces of a box from the origin, ``[12, 3, 3]``: a dwelling's outer surfaces."""
    c = np.array([[a, b, d] for a in (0.0, x) for b in (y0, y1) for d in (0.0, z)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([c[[q[i], q[j], q[k]]] for q in quads for i, j, k in ((0, 1, 2), (0, 2, 3))])


@pytest.mark.parametrize("preset", ["quiet", "medium", "lively"])
def test_a_scene_of_the_second_generator_is_planned_and_priced_without_a_machine(
    preset: str, tmp_path: Path
) -> None:
    """Every kind of source the generator draws, through the plan and onto fake offers."""
    recipe = social_recipe(preset)
    plan = make_plan(recipe, shell(9.0, FLOOR_Y_M, FLOOR_Y_M + 2.6, 4.0), low_ppw=LOW_PPW)
    record = plan.record
    alone = {s.id for s in recipe.sources if not wave_band(s)}
    assert "own_voice" in alone and set(record["mirror_only"]) == alone
    assert {s.subtype for s in recipe.sources if s.id in alone} >= {"body"}
    # The plan and the recipe's cost count the same solves, sites and early positions; the
    # dwelling's walls can only add cells, where one of the listener's way stands by them.
    counted = cost.counts(recipe)
    assert record["source_positions"] == low_band_source_positions(recipe).count
    for key in ("source_positions", "tail_sites", "step_pairs", "audible_steps_total"):
        assert counted[key] == record[key], key
    assert counted["cells"] <= plan.cells.count
    # No source carried and no closed window among the solved, and each of those is read.
    solved = {name for name, track in plan.tracks.sources.items() if track.low}
    assert solved.isdisjoint(alone)
    assert all(
        (plan.tracks.sources[name].slot >= 0).any() == plan.tracks.sources[name].audible.any()
        for name in solved
    )
    # The offers, priced for this run on the grid a trace takes: one card is the machine.
    said: list[str] = []
    offers = [Offer(1, "RTX 3090", 1, 0.173, 24.0), Offer(2, "RTX 3090", 8, 1.382, 24.0)]
    need = onebox.campaign_need(a_bundle(tmp_path))
    for count in (1, 8):
        found = onebox.plan_rental(
            Client(offers),
            need,
            hours=None,
            predict=machines.predictor(record, low_ppw=LOW_PPW),
            max_dph=3.0,
            min_ram_gb=60.0,
            gpu="",
            say=said.append,
            min_gpus=count,
        )
        # Among hosts of one card or more the one card is the lowest total, and first.
        assert [o.id for o in found.offers] == ([1, 2] if count == 1 else [2])
    one, eight = (
        machines.predict(
            record,
            low_ppw=LOW_PPW,
            gpu_name="RTX 3090",
            num_gpus=cards,
            gpu_ram_gb=24.0,
            dph_total=rate,
        )
        for cards, rate in ((1, 0.173), (8, 1.382))
    )
    assert one is not None and eight is not None
    # A minute of scene: the rental's start and the grid's preparation are most of it.
    assert one["usd"] < 0.10 and one["hours"] < 0.5 and one["usd"] < eight["usd"]
    assert one["solves"] == record["source_positions"]


# --------------------------------------------------------------------------
# the mirror where version 2 puts a source: by a surface, on the floor, at the head
# --------------------------------------------------------------------------


def test_what_the_mirror_gives_a_source_by_a_surface_on_it_and_behind_it() -> None:
    """Nothing breaks: near a surface is that surface's reflection, behind one is silence."""
    held = assets()
    ms = prepare(held.catalogue, held.settings, MovingSettings())
    head = np.array([[1.5, 1.7, 2.0]])

    def heard(point: list[float]) -> tuple[Any, float]:
        source = np.array([point])
        onsets = onset_field(held.catalogue, np.concatenate([source, head]), sound_speed_m_s=C)
        table = trace_early(ms, source, head, onsets=onsets)
        rays = histograms(
            held.catalogue, held.settings, source, head, devices=Devices.host(1), cache=TailCache()
        )[0]
        return table, float(rays.energy.sum())

    # A footstep, 5 cm above the floor: its direct sound, and the floor's reflection 8 cm
    # of path behind it, at nearly its level. That comb is the floor's and is right.
    step, late = heard([1.2, FLOOR_SOURCE_HEIGHT_M, 0.8])
    first = np.sort(step.delay_s)[:2] * C
    assert (step.kind == KIND_DIRECT).sum() == 1 and late > 0.0
    assert first[0] == pytest.approx(np.linalg.norm([0.3, 1.65, 1.2]), abs=1e-9)
    assert 0.05 < first[1] - first[0] < 0.10
    # A fixture 0.25 m from a wall is a source like any other.
    fixture_table, _ = heard([0.25, 1.4, 1.5])
    assert fixture_table.path_id.size > 30
    # On a surface: that surface's own image is lost, and the rays that leave into it.
    on_floor, lost = heard([1.2, 0.0, 0.8])
    assert 0 < on_floor.path_id.size < step.path_id.size and 0.0 < lost < 0.6 * late
    # Behind a surface, outside the shell: no arrival and no late part. Not an error: a
    # trace says it of the steps it happens to (``steps_without_an_arrival``).
    outside, nothing = heard([-0.2, 1.4, 1.5])
    assert outside.path_id.size == 0 and nothing == 0.0
    # And a source no step hears has a table and no row in it.
    silent = trace_early(ms, np.array([[1.2, 1.7, 0.8]]), head, audible=np.array([False]))
    assert silent.path_id.size == 0 and silent.offsets.tolist() == [0, 0]


def test_the_tail_of_a_source_at_the_head_is_on_the_scale_of_a_cell_beyond_it() -> None:
    """The wearer's own voice: cast from the head, where a cell stands a centimetre away."""
    held = assets()
    ms = prepare(held.catalogue, held.settings, MovingSettings())
    head = np.array([1.5, 1.7, 2.0])
    cells = np.array([[1.509, 1.7, 2.0], [1.5, 1.7, 2.6]])
    rays = histograms(held.catalogue, held.settings, head[None, :], cells, devices=Devices.host(1))[
        0
    ]
    radius = held.settings.rays.receiver_radius_m
    given: dict[str, Any] = {"rate": 48000.0, "receiver_radius_m": radius}
    beyond = band_pulse_energy(48000.0)[:8] * 4.0 / radius**2
    at_head = tail_scale(ms, head, cells, rays, at_the_head=True, **given)
    np.testing.assert_allclose(at_head[0], beyond)
    np.testing.assert_allclose(at_head[1], beyond)
    # As any other source's it would be read on a direct sound of ``1 / d`` at 9 mm.
    other = tail_scale(ms, head, cells, rays, **given)
    assert np.allclose(other[0] / beyond, (1.05 * radius / 0.009) ** 2)


# --------------------------------------------------------------------------
# the trace: recipe to pack
# --------------------------------------------------------------------------

C = 343.2
FS = 48000


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The scene traced once, on ``numpy``, a monopole in free air where a card would solve."""
    tmp = tmp_path_factory.mktemp("v2")
    trace, _, plan = traced(tmp, v2_recipe())
    # The pack's structure is read back; the engine's two modules are another test's.
    trace.check_mode = "read"
    trace.run()
    return {"trace": trace, "plan": plan, "pack": tmp / "out" / "pack.h5"}


def test_the_pack_says_what_version_2_says_of_every_source(made: dict[str, Any]) -> None:
    with read_pack(made["pack"], deep=True) as pack:
        assert pack.header.has_low and pack.header.has_tail
        assert sorted(pack.header.provenance["mirror_only"]) == sorted(MIRROR_ONLY)
        # Every step that is heard holds an arrival.
        assert made["trace"].report["steps_without_an_arrival"] == {}
        said = {
            name: (s.kind, s.subtype, s.mirror_only_because, s.carried_by, s.carried_at)
            for name, s in pack.sources.items()
        }
        assert said == {
            "own_voice": ("own_voice", "", "carried", "listener", "mouth"),
            "street_closed": ("noise", "outside", "closed opening", "", ""),
            "street_open": ("noise", "outside", "", "", ""),
            "talker_1": ("voice", "", "", "", ""),
            "talker_1_body": ("noise", "body", "carried", "talker_1", "mouth"),
            "tap": ("noise", "appliance", "", "", ""),
            "walker": ("voice", "", "", "", ""),
            "walker_steps": ("noise", "steps", "carried", "walker", "floor"),
        }
        for name, source in pack.sources.items():
            # A source of the mirror alone has no band of the wave solver, and says so.
            assert source.mirror_only == (name in MIRROR_ONLY) == (source.low is None)
            assert source.crossed == (name not in MIRROR_ONLY)
            assert source.tail is not None
            assert source.direct == (name != "own_voice")
        assert pack.sources["talker_1"].level_spl_1m_db == 58.0
        assert pack.sources["street_closed"].level_spl_1m_db == 34.0
        opened = pack.sources["street_open"]
        assert (opened.opening_object, opened.opening_state) == ("window_9", "open")
        assert not opened.band_gain_db and opened.low is not None
        assert pack.sources["street_closed"].opening_state == "closed"
        # Who is never heard has a group and nothing in it.
        assert pack.sources["walker"].early.path_id.size == 0
        # A pack that hides a low band, or its absence, is refused.
        own = pack.sources["own_voice"]
        with pytest.raises(PackError, match="mirror_only and the low group disagree"):
            validate(
                replace(pack, sources={**pack.sources, "own_voice": replace(own, low=opened.low)})
            )
        with pytest.raises(PackError, match="mirror_only and the low group disagree"):
            validate(
                replace(pack, sources={**pack.sources, "street_open": replace(opened, low=None)})
            )


def test_the_mirror_follows_the_mouth_and_the_head_and_the_low_band_does_not(
    made: dict[str, Any],
) -> None:
    tracks = made["plan"].tracks
    with read_pack(made["pack"]) as pack:
        talker, track = pack.sources["talker_1"], tracks.sources["talker_1"]
        # The pack holds the head where the low band is moved to, and the mouth where it is.
        assert np.array_equal(pack.listener.position, tracks.listener)
        assert np.array_equal(talker.position, track.mouth)
        assert talker.low is not None
        # The solved position is the station, to the millimetre: the sway asked no solve.
        assert np.allclose(talker.low.pair_position, [0.6, 1.7, 0.8], atol=1e-9)
        # The direct sound of every step is the mouth to the head, sways and all.
        early = talker.early
        swayed, held = [], []
        for step in np.flatnonzero(talker.audible):
            rows = early.rows(int(step))
            direct = np.flatnonzero(early.kind[rows] == KIND_DIRECT)
            assert direct.size == 1
            swayed.append(np.linalg.norm(track.mouth[step] - tracks.head[step]) / C)
            held.append(np.linalg.norm(track.position[step] - tracks.listener[step]) / C)
            assert early.delay_s[rows][direct[0]] == pytest.approx(swayed[-1], abs=1e-9)
        # Which is not where the low band reads them: 3 cm between the two, 0.1 ms.
        assert 2e-5 < np.abs(np.asarray(swayed) - np.asarray(held)).max() < 2e-4


def test_the_wearers_voice_is_the_rooms_answer_and_a_closed_window_its_glazing(
    made: dict[str, Any],
) -> None:
    trace = made["trace"]
    with read_pack(made["pack"]) as pack:
        own = pack.sources["own_voice"]
        direct = own.early.kind == KIND_DIRECT
        # The direct path's row is there, at 0.10 m and at no gain; the room's are whole.
        assert direct.sum() == own.audible.sum() and not own.early.gain[direct].any()
        assert np.allclose(own.early.delay_s[direct], np.hypot(0.09, 0.05) / C, atol=1e-9)
        assert (own.early.gain[~direct] > 0.0).any()
        # Its tail is cast from the head's centre, where a cell stands: on a finite scale.
        assert own.tail is not None and np.all(np.isfinite(own.tail.scale))
        assert own.tail.scale.max() < 10.0 * pack.sources["tap"].tail.scale.max()  # type: ignore[union-attr]
        # A closed window: the glazing's index a band, about what it takes from traffic.
        closed = pack.sources["street_closed"]
        colour = np.asarray(closed.band_gain_db)
        assert colour == pytest.approx([4.06, 8.06, 0.06, -9.94, -11.94, -5.94, -5.94])
        bare = trace.early["street_closed"].pack()["gain"]
        assert np.allclose(closed.early.gain, bare * 10.0 ** (colour / 20.0), rtol=1e-6)
        assert closed.tail is not None
        picks = band_map(pack.header.bands_hz, pack.header.bank)
        assert np.allclose(
            closed.tail.scale, trace.tail["street_closed"]["scale"] * 10.0 ** (colour[picks] / 10.0)
        )
    assert mirror_only.traffic_reduction_db("double") == pytest.approx(25.06, abs=0.01)
    assert mirror_only.traffic_reduction_db("single") == pytest.approx(25.73, abs=0.01)
    assert mirror_only.pane_gain_db("single")[:3] == pytest.approx([8.73, 5.73, -0.27])
    with pytest.raises(ValueError, match="a glazing is one of"):
        mirror_only.pane_gain_db("triple")


# --------------------------------------------------------------------------
# the engine: pack to stems
# --------------------------------------------------------------------------


def _band(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    """The energy of ``signal`` between two frequencies."""
    sos = butter(6, [low_hz, high_hz], btype="bandpass", fs=FS, output="sos")
    return float(np.sum(sosfiltfilt(sos, signal) ** 2))


def test_a_stem_of_the_mirror_alone_is_whole_under_the_crossover(made: dict[str, Any]) -> None:
    """No hole under 1 kHz: the mirror's own low octave bands, with no crossover on them."""
    with read_pack(made["pack"]) as pack:
        samples = 10 * pack.header.step_samples
        dry = np.random.default_rng(3).standard_normal(pack.header.samples)
        engine = Engine(
            pack,
            {"talker_1": dry, "talker_1_body": dry},
            settings=RenderSettings(workers=1, directivity=False),
        )
        # The breath is at the talker's mouth: the same paths as the voice's.
        # The voice's own gain is -2 dB and the breath's none: taken out of the voice's.
        own = 10.0 ** (2.0 / 20.0)
        alone = engine.stem("talker_1_body", 0, samples, parts=("early",))[0]
        crossed = own * engine.stem("talker_1", 0, samples, parts=("early",))[0]
        wave = own * engine.stem("talker_1", 0, samples, parts=("low",))[0]
        assert "low" not in engine.source("talker_1_body").parts
    low, high = (250.0, 700.0), (5600.0, 11000.0)
    # From 4 kHz up a pair's own seam is not applied: the two are one mirror at one level.
    assert _band(alone, *high) == pytest.approx(_band(crossed, *high), rel=1e-3)
    # Under the crossover the crossed source's arrivals are masked away, 20 dB and more,
    # and its wave band stands there; the source of the mirror alone keeps its own.
    assert _band(crossed, *low) < 0.01 * _band(alone, *low)
    assert _band(wave, *low) > 30.0 * _band(crossed, *low)
    # What it keeps is on the scale of the band a solve would have given. The solve here
    # is a monopole in free air and the mirror's is the box, whose walls give back 7 dB:
    # over the free field, and by less than 10 dB.
    assert 1.0 < _band(alone, *low) / _band(wave, *low) < 10.0


def test_the_level_is_applied_once_and_the_labels_come_out_beside_the_stems(
    made: dict[str, Any], tmp_path: Path
) -> None:
    said = write_labels(labels_path(tmp_path / "scene.f32"), made["pack"])
    assert said == tmp_path / "scene.labels.json"
    document = json.loads(said.read_text())
    with read_pack(made["pack"]) as pack:
        assert document == json.loads(json.dumps(labels(pack)))
        assert [s["id"] for s in document["sources"]] == list(pack.sources)
        by_id = {s["id"]: s for s in document["sources"]}
        talker = by_id["talker_1"]
        # The training label: what the voice is to the listener, by interval.
        assert [(r["role"], r.get("group")) for r in talker["roles"]] == [
            ("conversation", "g1"),
            ("outside", None),
        ]
        turn = talker["activity"][0]
        assert (turn["effort"], turn["event"], turn["role"], turn["group"]) == (
            "raised",
            "turn",
            "conversation",
            "g1",
        )
        assert (talker["low_band"], talker["stem"]) == ("wave", "whole")
        assert (by_id["own_voice"]["low_band"], by_id["own_voice"]["stem"]) == ("mirror", "room")
        assert by_id["walker_steps"]["attach"]["at"] == "floor"
        assert by_id["street_closed"]["opening"] == {"object": "window_8", "state": "closed"}
        assert document["listener"]["conversation"][0]["group"] == "g1"
        # THE LEVEL. The source is 58 dB at 1 m and the turn 5 dB over it: 63, said once.
        assert (talker["level_spl_1m_db"], turn["level_spl_1m_db"]) == (58.0, 63.0)
        # The engine lays the source's gain and the interval's on the clip as stored, a
        # voice at 60 dB: 63 - 60, and no more. ``level_spl_1m_db`` is nobody's gain.
        clip = 0.01 * np.random.default_rng(5).standard_normal(2 * FS)
        steps = 10 * pack.header.step_samples
        settings = RenderSettings(workers=1, directivity=False)
        from_recipe = Engine(pack, clips=lambda _: (clip, float(FS)), settings=settings)
        laid = np.zeros(pack.header.samples)
        laid[: int(0.9 * FS)] = clip[: int(0.9 * FS)]
        by_hand = Engine(pack, {"talker_1": laid}, settings=settings)
        heard = from_recipe.stem("talker_1", 4800, steps)[0]
        unit = by_hand.stem("talker_1", 4800, steps)[0]
    applied_db = 10.0 * np.log10(np.sum(heard**2) / np.sum(unit**2))
    # ``by_hand`` holds the source's own gain of -2 dB already: the interval's 5 are left.
    assert applied_db == pytest.approx(5.0, abs=0.01)
    assert turn["level_spl_1m_db"] - VOICE_REFERENCE_DB == pytest.approx(-2.0 + 5.0)
