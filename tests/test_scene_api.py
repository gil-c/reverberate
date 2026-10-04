"""Tests for the scene view's endpoints, on the two rooms of ``scene_floor``.

The page draws what these return and computes nothing of its own, so what is
pinned here is that each answer is the package's: the schema is
``Parameters`` itself, a generated recipe is ``generate``'s to the byte, a
track is ``source_state`` and ``listener_state`` on the grid, and nothing is
written until a recipe is saved.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import (
    Parameters,
    canonical_bytes,
    describe,
    listener_state,
    low_band_positions,
    recipe_sha256,
    source_state,
)
from reverberate.viz.scene_api import (
    SceneError,
    SceneService,
    layout_payload,
    parameter_schema,
    parameters_from,
    tracks,
)
from reverberate.viz.serve_room import _handler_for, _Server
from scene_floor import SMALL, small_recipe, two_room_layout


def _flat(parameters: Parameters) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(parameters)))  # type: ignore[no-any-return]


def _leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix: tree}
    found: dict[str, Any] = {}
    for key, value in tree.items():
        found.update(_leaves(value, f"{prefix}.{key}" if prefix else key))
    return found


class _Site:
    """A served scene view: the real handler over the two rooms and a temporary folder."""

    def __init__(self, folder: Path) -> None:
        self.recipes = folder / "recipes"
        self.scenes = SceneService(None, self.recipes, layout_of=self._layout)
        builder = type("Builder", (), {"target": folder, "scenes": self.scenes})()
        self.server = _Server(("127.0.0.1", 0), _handler_for(builder))
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()

    @staticmethod
    def _layout(dwelling: str) -> Any:
        if dwelling != "two_rooms":
            raise KeyError(dwelling)
        return two_room_layout()

    def ask(self, path: str, body: Any = None) -> tuple[int, Any]:
        url = f"http://127.0.0.1:{self.server.server_address[1]}/api/scene/{path}"
        data = None if body is None else json.dumps(body).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())


@pytest.fixture
def site(tmp_path: Path) -> Iterator[_Site]:
    served = _Site(tmp_path)
    try:
        yield served
    finally:
        served.server.shutdown()
        served.server.server_close()


def test_the_schema_is_the_parameters_themselves() -> None:
    """A field added to ``Parameters`` reaches the panel without a line here:
    names, defaults and the path in a recipe's ``generator`` block are read
    off the dataclass and its record. Every default lies inside its limits."""
    schema = parameter_schema()["parameters"]
    defaults = Parameters()

    assert [entry["name"] for entry in schema] == [field.name for field in fields(Parameters)]
    assert {entry["path"] for entry in schema} == set(_leaves(defaults.record()))
    record = _leaves(defaults.record())
    for entry in schema:
        default = getattr(defaults, entry["name"])
        assert entry["default"] == (list(default) if isinstance(default, tuple) else default)
        assert record[entry["path"]] == entry["default"]
        values = entry["default"] if entry["kind"] == "range" else [entry["default"]]
        assert all(entry["min"] <= value <= entry["max"] for value in values), entry["name"]
    by_name = {entry["name"]: entry for entry in schema}
    assert by_name["near_voice_count"]["type"] == "integer"
    assert by_name["rail_pitch_m"]["fixed"], "rule 4 leaves no choice of the rails' pitch"
    assert by_name["speed_m_s"]["max"] == 1.5, "rule 7's speed"


def test_parameters_outside_the_schema_are_refused_by_name() -> None:
    assert parameters_from(_flat(SMALL)) == SMALL
    assert parameters_from({"duration_s": 60}) == Parameters(duration_s=60.0)
    for flat, said in (
        ({"dwell_s": [30.0, 20.0]}, "dwell_s"),
        ({"overlap_share": 1.5}, "overlap_share"),
        ({"near_voice_count": [1]}, "near_voice_count"),
        ({"loudness": 3}, "loudness"),
        ({"rail_pitch_m": 0.1}, "rail_pitch_m"),
    ):
        with pytest.raises(SceneError, match=said) as refused:
            parameters_from(flat)
        assert refused.value.status == 400


def test_a_layout_is_drawn_from_its_stations_rails_and_floor() -> None:
    layout = two_room_layout()
    payload = layout_payload(layout)

    assert [s["id"] for s in payload["stations"]] == [s.id for s in layout.stations]
    assert [r["id"] for r in payload["rails"]] == [r.id for r in layout.rails]
    assert [room["name"] for room in payload["rooms"]] == ["living room", "bedroom"]
    assert len(payload["seating"]) == 3
    # The free floor is one piece through the door, with the furniture cut out of it.
    assert len(payload["free"]) == 1
    xs = [x for x, _ in payload["free"][0]["exterior"]]
    assert min(xs) == pytest.approx(0.25) and max(xs) == pytest.approx(8.75)
    json.dumps(payload)


def test_generation_returns_the_generators_own_recipe_and_writes_nothing(site: _Site) -> None:
    """The same parameters and seed give the bytes ``generate`` gives, with the
    summary, the violations and the cost beside them; the panel's values come
    back from the recipe's ``generator`` block. Nothing reaches the disk."""
    expected = small_recipe(1)
    status, answer = site.ask(
        "generate", {"dwelling": "two_rooms", "seed": 1, "parameters": _flat(SMALL)}
    )

    assert status == 200
    assert answer["recipe"] == json.loads(canonical_bytes(expected))
    assert answer["canonical"].encode() == canonical_bytes(expected)
    assert answer["recipe_sha256"] == recipe_sha256(expected)
    assert answer["describe"] == describe(expected)
    assert answer["violations"] == [] and answer["floor_checked"]
    positions = low_band_positions(expected)
    assert answer["low_band_positions"] == {**positions, "total": sum(positions.values())}
    assert answer["parameters"] == _flat(SMALL)
    assert answer["placeholder_clips"] and answer["placeholder_assets"]
    assert not site.recipes.exists()


def test_a_generation_that_cannot_succeed_says_why(site: _Site) -> None:
    crowded = {**_flat(SMALL), "noise_count": [16, 16]}
    status, answer = site.ask(
        "generate", {"dwelling": "two_rooms", "seed": 1, "parameters": crowded}
    )
    assert status == 422 and "noises" in answer["error"]

    status, answer = site.ask("generate", {"dwelling": "two_rooms", "seed": -1})
    assert status == 400 and "seed" in answer["error"]
    status, answer = site.ask("generate", {"dwelling": "nowhere", "seed": 1})
    assert status == 404 and "nowhere" in answer["error"]


def test_tracks_are_the_kinematics_on_the_grid(site: _Site) -> None:
    """The page interpolates between these samples and never places anything
    itself, so each column must be ``source_state`` or ``listener_state`` at
    the grid's times, the grid must end on the scene's end, and the lanes
    must cover the scene."""
    recipe = small_recipe(1)
    status, answer = site.ask(
        "tracks", {"canonical": canonical_bytes(recipe).decode(), "step_s": 0.5}
    )

    assert status == 200
    times = np.array(answer["t"])
    assert times[0] == 0.0 and times[-1] == recipe.duration_s
    assert np.allclose(np.diff(times)[:-1], 0.5)
    assert [s["id"] for s in answer["sources"]] == [s.id for s in recipe.sources]
    for source, track in zip(recipe.sources, answer["sources"], strict=True):
        state = source_state(recipe, source.id, times)
        placed = np.stack([track["x"], track["y"], track["z"]], axis=-1)
        assert np.abs(placed - state.position).max() <= 5e-4
        assert np.abs(np.array(track["yaw_deg"]) - state.yaw_deg).max() <= 5e-3
        assert track["kind"] == source.kind
        assert track["activity"] == [[a.start_s, a.end_s] for a in source.activity]
        spans = track["movement"]
        assert spans[0]["start_s"] == 0.0 and spans[-1]["end_s"] == recipe.duration_s
        assert [s["type"] for s in spans] == [s.to_dict()["type"] for s in source.segments]
    head = listener_state(recipe, times)
    listener = answer["listener"]
    placed = np.stack([listener["x"], listener["y"], listener["z"]], axis=-1)
    assert np.abs(placed - head.position).max() <= 5e-4
    for name, column in (("yaw_deg", head.yaw_deg), ("pitch_deg", head.pitch_deg)):
        assert np.abs(np.array(listener[name]) - column).max() <= 5e-3
    spans = listener["movement"]
    assert spans[0]["start_s"] == 0.0 and spans[-1]["end_s"] == recipe.duration_s
    assert all(a["end_s"] == b["start_s"] for a, b in zip(spans[:-1], spans[1:], strict=True))
    assert {span["type"] for span in spans} <= {"rest", "walk", "rise"}
    assert tracks(recipe, 0.5) == answer

    text = canonical_bytes(recipe).decode()
    status, answer = site.ask("tracks", {"canonical": text, "step_s": 0.001})
    assert status == 400 and "step_s" in answer["error"]
    broken = json.dumps({**recipe.to_dict(), "sound": True})
    status, answer = site.ask("tracks", {"canonical": broken})
    assert status == 422 and "rule 1" in answer["error"]
    # A recipe is named by its text. The tree is what a browser would send,
    # with every whole float written as an integer: not a recipe.
    status, answer = site.ask("tracks", {"recipe": recipe.to_dict()})
    assert status == 400 and "canonical" in answer["error"]


def test_a_recipe_is_saved_once_asked_listed_and_read_back(site: _Site) -> None:
    """Saving is the one request that writes: the canonical bytes, under the
    dwelling's folder. A name already taken is kept unless the page says to
    replace it, and a loaded recipe restores the generator's panel."""
    recipe = small_recipe(1)
    body = {"canonical": canonical_bytes(recipe).decode()}
    assert site.ask("recipes/two_rooms") == (200, [])
    assert not site.recipes.exists()

    status, answer = site.ask("recipes/two_rooms/first", body)
    assert status == 200 and answer["recipe_sha256"] == recipe_sha256(recipe)
    assert (site.recipes / "two_rooms" / "first.json").read_bytes() == canonical_bytes(recipe)
    assert [p.name for p in site.recipes.rglob("*") if p.is_file()] == ["first.json"]

    status, answer = site.ask("recipes/two_rooms/first", body)
    assert status == 409
    assert site.ask("recipes/two_rooms/first", {**body, "overwrite": True})[0] == 200

    status, listed = site.ask("recipes/two_rooms")
    assert listed == [{"name": "first", "seed": 1, "duration_s": 90.0, "sources": 5}]
    status, loaded = site.ask("recipes/two_rooms/first")
    assert status == 200
    assert loaded["recipe_sha256"] == recipe_sha256(recipe)
    assert loaded["parameters"] == _flat(SMALL)
    assert loaded["violations"] == [] and loaded["floor_checked"]

    for path, code in (
        ("recipes/two_rooms/..%2Fescape", 400),
        ("recipes/Two Rooms/first", 400),
        ("recipes/other/first", 400),
    ):
        assert site.ask(path.replace(" ", "%20"), body)[0] == code
    assert site.ask("recipes/two_rooms/missing")[0] == 404
    assert site.ask("nothing")[0] == 404


def test_a_recipe_that_breaks_a_rule_is_shown_with_the_rules_number(site: _Site) -> None:
    """A recipe written by hand, or by an older generator, may break a rule:
    it is still loaded and drawn, with every violation and its number."""
    tree = small_recipe(1).to_dict()
    tree["atmosphere"]["temperature_c"] = 25.0
    tree["generator"] = None
    (site.recipes / "two_rooms").mkdir(parents=True)
    (site.recipes / "two_rooms" / "warm.json").write_text(json.dumps(tree))

    status, loaded = site.ask("recipes/two_rooms/warm")
    assert status == 200
    assert [violation["rule"] for violation in loaded["violations"]] == [10]
    assert loaded["parameters"] is None
