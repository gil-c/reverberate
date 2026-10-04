"""The scene view's arithmetic, through node: time, lanes, tracks, the transport.

The page draws a scene from tracks the server sampled and computes nothing
about it; what it does compute is where a time falls on the timeline, where a
lane is, and where a marker is between two samples. ``tests/js/scene.mjs``
runs those functions; here they are held to the kinematics' own answers on the
two rooms of ``scene_floor``, and to the events an audio stage will attach to.

Skipped where node is not installed, as the other JavaScript tests are.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import listener_state, source_state
from reverberate.scenes.validate import MAX_SPEED_M_S, MAX_TURN_DEG_S
from reverberate.viz.scene_api import tracks
from scene_floor import small_recipe

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "src" / "reverberate" / "viz" / "app" / "scene"
HARNESS = ROOT / "tests" / "js" / "scene.mjs"
STEP_S = 0.1

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """What the harness printed, on the small scene sampled as the page asks for it."""
    recipe = small_recipe(1)
    sampled = tracks(recipe, STEP_S)
    grid = np.array(sampled["t"])
    # Between the samples, off their middle, and in the short last interval.
    between = np.concatenate([grid[:-1] + 0.37 * np.diff(grid), [recipe.duration_s - 1e-3]])
    head = listener_state(recipe, between)
    given = {
        "tracks": sampled,
        "between": {
            "t": between.tolist(),
            "sources": [
                source_state(recipe, source.id, between).position.tolist()
                for source in recipe.sources
            ],
            "listener": head.position.tolist(),
            "listener_yaw": head.yaw_deg.tolist(),
        },
    }
    path = tmp_path_factory.mktemp("scene") / "given.json"
    path.write_text(json.dumps(given))
    done = subprocess.run(
        ["node", str(HARNESS), str(SCENE), str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_time_maps_to_a_pixel_and_back_through_zoom_and_pan(out: Any) -> None:
    """Twenty minutes over 900 pixels. Zooming keeps the time under the
    pointer under it, stops at five seconds, and comes back to the whole
    scene; a pan stops at the scene's ends; a pixel outside seeks to an end."""
    m = out["map"]
    assert m["ends"] == [0, 900, 600]
    assert m["outside"] == [0, 1200]
    assert m["roundTrip"] < 1e-9
    assert m["held"] == pytest.approx([400.0, 400.0])
    assert m["nearSpan"] == pytest.approx(120.0)
    assert m["tinySpan"] == m["minSpan"] == 5
    assert m["back"] == [0, 1200]
    assert m["late"] == pytest.approx([1080.0, 1200.0])
    # The ruler: round steps, never closer than the 64 pixels a label needs.
    assert m["step"] == 120 and m["nearStep"] == 10
    assert min(m["gaps"]) >= 64
    assert m["labels"] == ["0:00", "2:00", "4:00"]
    assert m["times"] == ["0:00", "1:15.3", "20:00"]


def test_lanes_stack_without_overlap_and_the_listener_has_one_band(out: Any) -> None:
    lanes = out["lanes"]
    assert lanes["ys"] == [4, 26, 48] and lanes["height"] == 70
    assert lanes["at"] == [None, "a", "a", "b", "listener", None]
    assert lanes["bandsInside"] and lanes["bandsApart"]
    assert lanes["listenerActivity"] is None and lanes["listenerBand"] == 20
    # Only the spans that touch the window are drawn.
    assert out["visible"] == [[10, 20, 30], [], [0]]
    seated, standing, travel, up, down, walk = out["posture"]
    assert (seated["from"], standing["from"]) == (0, 1)
    assert not seated["moving"] and travel["moving"] and walk["moving"]
    assert (up["from"], up["to"], down["from"], down["to"]) == (0, 1, 1, 0)


def test_a_marker_is_on_its_sample_and_near_the_truth_between_two(out: Any) -> None:
    """On a sample the page shows the server's number. Between two it draws a
    straight line, and nothing walks faster than rule 7 allows, so the line is
    within half a step of travel of where the kinematics put the source; the
    head's yaw likewise within half a step of the fastest turn."""
    worst = out["tracks"]["worst"]
    assert worst["atSample"] < 1e-9 and worst["yawAtSample"] < 1e-9
    # 1e-3: the samples are sent to the millimetre.
    assert worst["between"] <= MAX_SPEED_M_S * STEP_S / 2 + 1e-3
    assert worst["listener"] <= MAX_SPEED_M_S * STEP_S / 2 + 1e-3
    assert worst["head"] <= MAX_TURN_DEG_S * STEP_S / 2 + 1e-2

    found = out["tracks"]
    before, after, on = found["brackets"]
    assert before == {"i": 0, "w": 0}
    assert after == {"i": found["lastIndex"] - 1, "w": 1}
    assert on == {"i": 3, "w": 0}
    assert found["lastInterval"]["i"] == found["lastIndex"] - 1
    assert found["lastInterval"]["w"] == pytest.approx(0.5)
    assert found["active"] == [True, True, False, False, False]
    assert found["span"] == [0, True]


def test_a_trail_runs_from_the_past_to_the_marker(out: Any) -> None:
    found = out["tracks"]
    # Four seconds at ten samples a second, and the two interpolated ends.
    assert found["trailPoints"] == found["trailCount"] == 42
    assert found["trailFits"] and found["trailSame"]
    assert max(found["trailEnds"]) < 1e-9
    assert found["trailAtStart"] == 1, "at the scene's start nothing has been walked yet"


def test_the_recipes_yaw_faces_the_same_way_in_the_view_and_on_the_plan(out: Any) -> None:
    """A recipe's yaw 0 faces ``+x`` and +90 faces ``-z``. The viewport's zero
    faces ``-z``: the offset is applied once, and the plan agrees with it."""
    view, plan = np.array(out["yaw"]["view"]), np.array(out["yaw"]["plan"])
    assert np.abs(view - plan).max() < 1e-12
    assert np.abs(plan[:2] - [[1, 0], [0, -1]]).max() < 1e-12


def test_the_transport_says_what_an_audio_stage_needs_to_follow_it(out: Any) -> None:
    """The seam for sound. Every change of the clock is an event carrying the
    scene time, the speed and whether it plays; the time follows whatever
    clock it is given; the scene stops at its end and play starts it over."""
    found = out["transport"]
    assert found["speeds"] == [1, 4, 16]
    assert (found["afterTwo"], found["afterFour"], found["paused"]) == (2, 6, 6)
    assert found["carried"] == 59, "another clock takes over without a jump"
    assert found["ended"] == {"t": 60, "speed": 4, "playing": False, "duration": 60}
    assert found["again"] == 0
    assert found["said"] == [
        ["load", 0, 1, False],
        ["play", 0, 1, True],
        ["speed", 2, 4, True],
        ["pause", 6, 4, False],
        ["seek", 58, 4, False],
        ["play", 58, 4, True],
        ["end", 60, 4, False],
        ["pause", 60, 4, False],
        ["play", 0, 4, True],
        ["seek", 0, 4, True],
        ["speed", 0, 16, True],
    ]
    assert found["unsubscribed"]


def test_a_solo_wins_over_a_mute_and_a_new_scene_hears_everything(out: Any) -> None:
    assert out["mix"]["changes"] == ["a,b,c", "a,c", "b", "b,c", "c", "a,c", "a"]
    assert out["mix"]["state"] == {"solo": [], "mute": [], "audible": ["a"]}


def test_the_panel_spells_units_and_holds_values_in_their_limits(out: Any) -> None:
    labels = [(entry["label"], entry["unit"]) for entry in out["panel"]["labels"]]
    assert labels == [
        ("turn rate", "°/s"),
        ("speed", "m/s"),
        ("seated share", ""),
        ("count", ""),
        ("pitch", "°"),
        ("dwell", "s"),
    ]
    assert out["panel"]["held"] == [1.5, 0.1, 4]
