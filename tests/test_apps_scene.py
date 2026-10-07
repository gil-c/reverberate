"""The scene player and the recipe manager, on this side of the socket.

What the two pages are handed is held here to what it must be: a scene
whose every source is where the pack or the recipe puts it and is to the
listener what the labels say, frames that are the engine's own samples
rendered when they are asked for and their sum under a balance (a fader at
-20 dB is -20 dB), levels that come with the chunks and are nothing before,
a balance written only when it is saved, with what the generator's level
table needs; recipes listed, drawn, moved to a trash and never deleted. No
browser, no network, no worker process.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.apps import recipes
from reverberate.apps import scene as player
from reverberate.render.pack import read_pack, write_pack
from reverberate.scenes import (
    canonical_bytes,
    listener_state,
    recipe_sha256,
    source_state,
    validate,
)
from reverberate.viz.audit_demo import demo_pack
from reverberate.viz.parts.scene import STEP_S, cast, head_track, pack_scene, plan_of, recipe_scene
from reverberate.viz.parts.stems import SCENE, Streamed, trim
from scene_floor import small_recipe, two_room_layout
from test_scenes_social import social_layout, social_recipe

SECONDS = 3.0


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two sources in the free field round a walking listener whose head sweeps."""
    path = tmp_path_factory.mktemp("pack") / "free.h5"
    write_pack(path, demo_pack(profile="free", duration_s=SECONDS, sources=2, seed=3))
    return path


@pytest.fixture
def streamed(pack_path: Path, tmp_path: Path) -> Iterator[Streamed]:
    made = Streamed(pack_path, tmp_path / "cache", workers=1, processes=False, ahead_s=1.0)
    yield made
    made.close()


def rms_db(frames: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.asarray(frames, dtype=np.float64) ** 2)))


# --- who is what ----------------------------------------------------------------------------


def test_the_cast_of_a_recipe_is_its_people_then_its_noises_each_what_the_labels_say() -> None:
    recipe = social_recipe("medium", 1, 60.0)
    tree = json.loads(canonical_bytes(recipe))
    made = cast(tree)
    rows = {row["id"]: row for row in made["sources"]}
    assert set(rows) == {source.id for source in recipe.sources}
    kinds = [row["kind"] for row in made["sources"]]
    assert kinds == sorted(kinds, key=lambda kind: kind != "voice")
    own = next(row for row in made["sources"] if row["what"] == "own_voice")
    assert own["shape"] == "self" and own["label"] == "Your voice"
    assert all(interval["role"] == "conversation" for interval in own["intervals"])
    voices = [row for row in made["sources"] if row["what"] == "voice"]
    assert voices and all(row["shape"] == "head" for row in voices)
    assert [row["label"] for row in voices] == [f"Talker {n + 1}" for n in range(len(voices))]
    # Told apart by colour, and never by the listener's white.
    assert len({row["colour"] for row in voices}) == len(voices) <= 9
    assert own["colour"] not in {row["colour"] for row in made["sources"] if row is not own}
    # A voice's interval is of the conversation when the recipe's role says so as it starts.
    for source in recipe.sources:
        if source.kind != "voice":
            continue
        for interval, drawn in zip(source.activity, rows[source.id]["intervals"], strict=True):
            role = next(r for r in source.roles if r.start_s <= interval.start_s < r.end_s)
            assert drawn["role"] == role.role
            assert (drawn["start_s"], drawn["end_s"]) == (interval.start_s, interval.end_s)
    things = [row for row in made["sources"] if row["kind"] == "noise"]
    assert all(row["shape"] == "marker" for row in things)
    assert all(i["role"] == "noise" for row in things for i in row["intervals"])
    # The listener's spans name who he talks with, and nobody when he talks with nobody.
    for span, said in zip(recipe.listener.conversation, made["conversation"], strict=True):
        members = {
            source.id
            for source in recipe.sources
            if source.kind == "voice"
            and any(
                r.role == "conversation"
                and r.group == span.group
                and r.start_s <= span.start_s < r.end_s
                for r in source.roles
            )
        }
        assert set(said["members"]) == (members if span.group else set())
    assert len(made["gaze"]) == len(recipe.listener.gaze)
    looked = next(g for g in made["gaze"] if g["mode"] == "talker")
    assert looked["said"] == f"looking at {rows[looked['target']]['label']}"


def test_a_recipe_of_the_first_version_has_a_cast_too() -> None:
    made = cast(json.loads(canonical_bytes(small_recipe())))
    whats = {row["what"] for row in made["sources"]}
    assert whats == {"near_voice", "far_voice", "noise"}
    near = next(row for row in made["sources"] if row["what"] == "near_voice")
    far = next(row for row in made["sources"] if row["what"] == "far_voice")
    assert {i["role"] for i in near["intervals"]} == {"conversation"}
    assert {i["role"] for i in far["intervals"]} == {"outside"}
    assert made["conversation"] == [] and made["gaze"] == []


def test_a_recipe_is_drawn_where_its_kinematics_put_everything() -> None:
    recipe = social_recipe("medium", 1, 60.0)
    scene = recipe_scene(recipe, plan_of(social_layout()))
    assert scene["step_s"] == STEP_S and scene["duration_s"] == 60.0 and scene["floor"] is None
    times = np.array([0.0, 12.3, 47.9])
    index = np.round(times / STEP_S).astype(int)
    head = listener_state(recipe, times)
    drawn = np.array(scene["listener"]["position"])[index]
    assert np.allclose(drawn, head.position, atol=1e-3)
    assert np.allclose(np.array(scene["listener"]["yaw_deg"])[index], head.yaw_deg, atol=1e-2)
    for source in scene["sources"]:
        state = source_state(recipe, source["id"], times)
        assert np.allclose(np.array(source["position"])[index], state.position, atol=1e-3)
        assert np.allclose(np.array(source["yaw_deg"])[index], state.yaw_deg, atol=1e-2)
    # The dwelling: a floor a room, and the walkable floor whose boundary is the walls.
    plan = scene["plan"]
    assert plan["floor_y"] == social_layout().floor.floor_y_m
    assert [room["name"] for room in plan["rooms"]] == [r.name for r in social_layout().floor.rooms]
    assert plan["walls"] and len(plan["walls"][0]["exterior"]) >= 4


def test_a_pack_is_drawn_from_its_own_tables_and_its_head_is_not_turned_the_long_way(
    pack_path: Path,
) -> None:
    with read_pack(pack_path) as pack:
        scene = pack_scene(pack)
        every = round(STEP_S / pack.header.step_s)
        assert scene["step_s"] == pytest.approx(STEP_S) and scene["plan"] is None
        assert scene["floor"]["rooms"] == [] and scene["duration_s"] == SECONDS
        assert [s["id"] for s in scene["sources"]] == list(pack.sources)
        for source in scene["sources"]:
            held = np.asarray(pack.sources[source["id"]].position)[::every]
            assert np.allclose(np.array(source["position"]), held, atol=1e-3)
        assert np.allclose(
            np.array(scene["listener"]["position"]),
            np.asarray(pack.listener.position)[::every],
            atol=1e-3,
        )
        track = head_track(pack)
        assert track["step_s"] == pack.header.step_s and len(track["yaw_deg"]) == pack.header.steps
        assert np.allclose(track["yaw_deg"], np.asarray(pack.listener.orientation)[:, 0], atol=0.01)
        # A source's intervals are its recipe's, which the pack holds.
        trees = {tree["id"]: tree for tree in pack.recipe_json()["sources"]}
        for source in scene["sources"]:
            spans = [(i["start_s"], i["end_s"]) for i in source["intervals"]]
            assert spans == [(a["start_s"], a["end_s"]) for a in trees[source["id"]]["activity"]]


def test_a_head_that_passes_180_degrees_is_followed_the_short_way(pack_path: Path) -> None:
    from dataclasses import replace

    with read_pack(pack_path) as pack:
        orientation = np.asarray(pack.listener.orientation).copy()
        orientation[:, 0] = (
            (170.0 + 4.0 * np.arange(orientation.shape[0])) + 180.0
        ) % 360.0 - 180.0
        turned = replace(pack, listener=replace(pack.listener, orientation=orientation))
        yaw = np.array(head_track(turned)["yaw_deg"])
        assert np.allclose(np.diff(yaw), 4.0, atol=0.02) and yaw[0] == pytest.approx(170.0)


# --- the sound, rendered as it is asked for -------------------------------------------------


def test_a_scene_is_rendered_when_it_is_asked_for_and_is_the_engine_s_own_samples(
    streamed: Streamed,
) -> None:
    (item,) = streamed.describe()
    assert item["id"] == SCENE and item["kind"] == "ambisonic" and item["stems"] == ["s1", "s2"]
    assert item["frames"] == int(SECONDS * 48000) and item["level_db"] == -12.0
    session = streamed.session
    assert not any(stem.done.any() for stem in session.stems.values())
    # Nothing is known of a level before its chunk is rendered.
    assert set(streamed.levels(0.0, 1.0)["stems"]["s1"]) == {None}
    size = session.chunk_samples
    one = streamed.frames(SCENE, size + 100, 5000, {"s1": 1.0, "s2": 0.0})
    assert one.dtype == np.float32 and one.shape == (5000, 64)
    assert np.array_equal(one, session.stems["s1"].read(1)[100:5100])
    # The other source was not heard and is not what was waited for.
    assert session.stems["s1"].done[1]
    both = streamed.frames(SCENE, size - 200, 400)
    a = np.concatenate([session.stems["s1"].read(0)[-200:], session.stems["s1"].read(1)[:200]])
    b = np.concatenate([session.stems["s2"].read(0)[-200:], session.stems["s2"].read(1)[:200]])
    assert np.array_equal(both, (a.astype(np.float64) + b).astype(np.float32))
    # Past the scene's end are zeros; nothing heard is zeros and renders nothing.
    end = streamed.frames(SCENE, streamed.frames_total - 10, 30)
    assert end.shape == (30, 64) and not end[10:].any()
    assert not streamed.frames(SCENE, 0, 100, {"s1": 0.0, "s2": 0.0}).any()
    # The levels come with the chunks: the omnidirectional channel's, at the page's level.
    told = streamed.levels(0.5, 0.6)
    assert told["hop_s"] == 0.05 and told["first"] == 10 and len(told["stems"]["s1"]) == 2
    step = session.pack.header.step_samples
    omni = session.stems["s1"].read(1)[:step, 0].astype(np.float64)
    assert told["stems"]["s1"][0] == pytest.approx(rms_db(omni) - 12.0, abs=0.06)


def test_a_fader_at_minus_20_db_is_minus_20_db(streamed: Streamed) -> None:
    size = streamed.session.chunk_samples
    full = streamed.frames(SCENE, 0, size, {"s1": 1.0, "s2": 0.0})
    low = streamed.frames(SCENE, 0, size, {"s1": 10.0 ** (-20.0 / 20.0), "s2": 0.0})
    assert rms_db(full) > -120.0
    assert rms_db(low) - rms_db(full) == pytest.approx(-20.0, abs=1e-4)
    assert np.allclose(low, 0.1 * full.astype(np.float64), rtol=1e-6, atol=1e-12)


def test_the_stems_kept_are_forgotten_the_scene_opened_longest_ago_first(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    for age, name in enumerate(("a" * 16, "b" * 16, "c" * 16)):
        (cache / name).mkdir(parents=True)
        (cache / name / "stem.f32").write_bytes(b"\1" * 300_000)
        stamp = time.time() - 100.0 * (3 - age)
        os.utime(cache / name, (stamp, stamp))
    (cache / "carriers").mkdir()
    (cache / "packs.json").write_text("{}")
    assert trim(cache, budget_gb=1.0) == 0
    assert trim(cache, budget_gb=0.0007) >= 300_000
    assert {p.name for p in cache.iterdir()} == {"b" * 16, "c" * 16, "carriers", "packs.json"}
    # The folder named is kept whatever its age.
    trim(cache, budget_gb=0.0, keep="b" * 16)
    assert (cache / ("b" * 16)).is_dir() and not (cache / ("c" * 16)).exists()
    assert trim(tmp_path / "nowhere", budget_gb=1.0) == 0


# --- the scene player -----------------------------------------------------------------------


def test_the_scene_player_plays_its_pack_and_writes_a_balance_only_when_it_is_saved(
    pack_path: Path, tmp_path: Path
) -> None:
    assert player.balance_file(Path("/runs/H/pack.h5")) == Path("/runs/H/pack.balance.json")
    balance = tmp_path / "out" / "free.balance.json"
    server, streamed, media = player.build(
        pack_path,
        tmp_path / "decoders",
        cache=tmp_path / "cache",
        balance=balance,
        measured_head=tmp_path / "none.sofa",
        workers=9,
        processes=False,
    )
    try:
        assert streamed.service.workers == 4 and streamed.status()["nice"] == 10
        told = server.handle("GET", "/api/player").json()
        assert told["item"]["id"] == SCENE and told["duration_s"] == SECONDS
        assert told["balance_file"] == str(balance) and "No dataset" in told["note"]
        assert len(told["head"]["yaw_deg"]) == int(SECONDS / 0.05) + 1
        scene = server.handle("GET", "/api/scene").json()
        tracks = server.handle("GET", "/api/balance").json()
        assert [t["id"] for t in tracks["tracks"]] == [s["id"] for s in scene["sources"]]
        assert [t["colour"] for t in tracks["tracks"]] == [s["colour"] for s in scene["sources"]]
        assert [t["label"] for t in tracks["tracks"]] == [s["label"] for s in scene["sources"]]
        assert tracks["saved"] is None and tracks["unsaved"] is False
        assert b"<title>Scene player</title>" in server.handle("GET", "/").payload
        # The frames, as the page asks them: whole, then one source 20 dB down, the other muted.
        url = f"/api/frames?item={SCENE}&start=0&count=24000"
        whole = np.frombuffer(server.handle("GET", url).payload, dtype="<f4").reshape(-1, 64)
        body = {"sources": {"s1": {"gain_db": -20.0}, "s2": {"mute": True}}}
        set_ = server.handle("POST", "/api/balance", json.dumps(body).encode()).json()
        assert set_["version"] == 1 and set_["unsaved"] is True and not balance.exists()
        low = np.frombuffer(server.handle("GET", url + "&balance=1").payload, dtype="<f4").reshape(
            -1, 64
        )
        alone = streamed.session.stems["s1"].read(0)
        assert rms_db(low) - rms_db(alone) == pytest.approx(-20.0, abs=1e-4)
        assert whole.shape == alone.shape == (24000, 64)
        levels = server.handle("GET", "/api/levels?from=0&to=0.5").json()
        assert len(levels["stems"]["s1"]) == 10 and None not in levels["stems"]["s1"]
        assert server.handle("POST", "/api/warm?at=1.0", b"{}").json()["ahead_s"] == 8.0
        assert streamed.session.cursor == 2
        # Saved: the file holds what is heard, and what the generator's level table reads.
        kept = server.handle("POST", "/api/balance/save", b"{}").json()
        assert kept["saved"] == str(balance) and kept["unsaved"] is False
        document = json.loads(balance.read_text())
        assert document["schema"] == "reverberate.apps.balance"
        assert document["sources"]["s1"] == {"gain_db": -20.0, "mute": False, "solo": False}
        assert document["gains"] == {"s1": pytest.approx(0.1), "s2": 0.0}
        context = document["context"]
        assert context["application"] == "reverberate.apps.scene"
        assert context["pack"] == str(pack_path.resolve())
        assert context["levels"]["s1"]["fader_db"] == -20.0 and context["levels"]["s2"]["silent"]
        assert server.handle("POST", "/api/balance/nothing", b"{}").status == 404
        assert server.handle("GET", "/api/sonogram?item=scene").status == 409
    finally:
        streamed.close()
    # Opened again, the balance is the one that was saved.
    server, streamed, _ = player.build(
        pack_path,
        tmp_path / "decoders",
        cache=tmp_path / "cache",
        balance=balance,
        measured_head=tmp_path / "none.sofa",
        processes=False,
    )
    try:
        again = {t["id"]: t for t in server.handle("GET", "/api/balance").json()["tracks"]}
        assert again["s1"]["gain_db"] == -20.0 and again["s2"]["mute"] is True
    finally:
        streamed.close()


def test_a_balance_says_the_level_the_listener_would_have_given_each_source() -> None:
    sources: list[dict[str, Any]] = [
        {"id": "talker_1", "label": "Talker 1", "what": "voice", "level_spl_1m_db": 60.0},
        {"id": "music_2", "label": "Music", "what": "noise", "subtype": "music", "gain_db": -3.0},
        {"id": "tv_3", "label": "Television", "what": "media_voice", "level_spl_1m_db": 62.0},
    ]
    balance = {
        "talker_1": {"gain_db": 4.0, "mute": False, "solo": False},
        "music_2": {"gain_db": -9.0, "mute": False, "solo": False},
        "tv_3": {"gain_db": 0.0, "mute": True, "solo": False},
    }
    table = player.level_table(sources, balance)["levels"]
    assert table["talker_1"]["set_level_spl_1m_db"] == 64.0 and table["talker_1"]["kind"] == "voice"
    assert table["music_2"]["set_level_spl_1m_db"] is None
    assert table["music_2"]["recipe_gain_db"] == -3.0 and table["music_2"]["fader_db"] == -9.0
    # A source left silent says nothing of a level.
    assert table["tv_3"]["silent"] is True and table["tv_3"]["set_level_spl_1m_db"] is None
    balance["talker_1"]["solo"] = True
    assert player.level_table(sources, balance)["levels"]["music_2"]["silent"] is True


# --- the recipes ----------------------------------------------------------------------------


def test_the_recipes_of_a_folder_are_listed_with_what_matters_and_held_to_the_rules(
    tmp_path: Path,
) -> None:
    recipe = social_recipe("medium", 1, 60.0)
    (tmp_path / "kept").mkdir()
    (tmp_path / "kept" / "medium.json").write_bytes(canonical_bytes(recipe))
    (tmp_path / "small.json").write_bytes(canonical_bytes(small_recipe()))
    tree = json.loads(canonical_bytes(recipe))
    tree["duration_s"] = -1.0
    (tmp_path / "broken.json").write_text(json.dumps(tree))
    (tmp_path / "other.json").write_text('{"schema": "something else"}')
    server, held = recipes.build(tmp_path)
    assert held.names() == ["broken.json", "kept/medium.json", "small.json"]
    told = server.handle("GET", "/api/recipes").json()
    assert told["can_generate"] is False and told["presets"] == ["quiet", "medium", "lively"]
    assert told["dwellings"] == ["two_rooms"]
    rows = {row["name"]: row for row in told["recipes"]}
    good = rows["kept/medium.json"]
    assert good["sha256"] == recipe_sha256(recipe) and good["duration_s"] == 60.0
    voices = [s for s in recipe.sources if s.kind == "voice"]
    assert good["people"] == len(voices) and good["own_voice"] is True
    assert good["groups"] == len({r.group for s in voices for r in s.roles})
    assert good["calmness"] == 0.5 and good["version"] == 2
    assert len(good["noises"]) == len(recipe.sources) - len(voices) - 1
    assert rows["small.json"]["version"] == 1 and rows["small.json"]["calmness"] is None
    one = server.handle("GET", "/api/recipes/kept/medium.json").json()
    assert one["valid"] is True and one["violations"] == [] and one["floor_checked"] is False
    assert one["summary"].startswith(f"recipe {recipe_sha256(recipe)}")
    heard = one["speech_over_noise"]
    assert heard["turns"] > 0 and heard["least_db"] <= heard["median_db"]
    # The cost is the trace's own predictor's, on the machine of the first scene.
    assert one["cost"]["usd"] > 0.0 and one["cost"]["hours"] > 0.0 and one["cost"]["pairs"] > 0
    assert one["placeholders"] == ["clips", "the dwelling's computed assets"]
    assert server.handle("GET", "/api/recipes/small.json").json()["speech_over_noise"] is None
    bad = server.handle("GET", "/api/recipes/broken.json").json()
    assert bad["valid"] is False and bad["violations"]
    assert server.handle("GET", "/api/recipes/../secret.json").status == 404
    assert server.handle("GET", "/api/recipes/missing.json").status == 404
    # Nothing is generated without the dataset.
    body = json.dumps({"dwelling": "two_rooms", "seed": 1}).encode()
    assert server.handle("POST", "/api/generate", body).status == 409
    for page in ("/", "/viewer.html"):
        assert server.handle("GET", page).status == 200


def test_a_recipe_thrown_away_goes_to_the_trash_and_is_never_deleted(tmp_path: Path) -> None:
    recipe = social_recipe("quiet", 2, 60.0)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "quiet.json").write_bytes(canonical_bytes(recipe))
    server, held = recipes.build(tmp_path)
    body = json.dumps({"name": "sub/quiet.json"}).encode()
    answer = server.handle("POST", "/api/delete", body).json()
    moved = Path(answer["moved_to"])
    assert moved.parent == tmp_path / "trash" and moved.name.endswith("_sub__quiet.json")
    assert moved.read_bytes() == canonical_bytes(recipe)
    assert not (tmp_path / "sub" / "quiet.json").exists()
    # What is in the trash is not listed, opened or thrown away again.
    assert held.names() == []
    assert server.handle("POST", "/api/delete", body).status == 404
    again = json.dumps({"name": f"trash/{moved.name}"}).encode()
    assert server.handle("POST", "/api/delete", again).status == 404
    assert server.handle("GET", f"/api/view/trash/{moved.name}").status == 404
    assert server.handle("POST", "/api/delete", b"{}").status == 400
    assert moved.is_file()


def test_a_recipe_is_drawn_from_a_dwelling_a_seed_a_preset_and_a_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, held = recipes.build(tmp_path, hssd_root=tmp_path)
    layouts = {"two_rooms": social_layout()}
    monkeypatch.setattr(held, "_layout", lambda dwelling: layouts.get(dwelling))

    def draw(**body: Any) -> Any:
        return server.handle("POST", "/api/generate", json.dumps(body).encode())

    made = draw(dwelling="two_rooms", seed=4, preset="lively", duration_s=60)
    assert made.status == 200, made.payload
    report = made.json()
    assert report["name"] == "two_rooms_lively_seed4_60s.json"
    assert report["valid"] is True and report["floor_checked"] is True
    assert report["duration_s"] == 60.0 and report["calmness"] == 0.15 and report["seed"] == 4
    # Its clips are the library's; no recipe of the dwelling is here to say its assets.
    assert report["placeholders"] == ["the dwelling's computed assets"]
    written = held._recipe(report["name"])
    assert not validate(written, social_layout().floor)  # type: ignore[arg-type]
    assert held.names() == [report["name"]]
    # The same again is the same recipe, and is refused; what is not a draw is refused too.
    assert draw(dwelling="two_rooms", seed=4, preset="lively", duration_s=60).status == 409
    assert draw(dwelling="two_rooms", seed=4, preset="loud").status == 400
    assert draw(dwelling="two_rooms", seed=-1).status == 400
    assert draw(dwelling="two_rooms", seed=1, duration_s=5).status == 400
    assert draw(dwelling="../etc", seed=1).status == 400
    assert draw(dwelling="nowhere", seed=1).status == 422
    # The viewer's scene is the recipe's, with the dwelling's walls.
    scene = server.handle("GET", f"/api/view/{report['name']}").json()
    assert scene["name"] == report["name"] and scene["plan"]["walls"] and scene["note"] == ""
    assert scene["duration_s"] == 60.0
    assert {s["id"] for s in scene["sources"]} == {s.id for s in written.sources}  # type: ignore[union-attr]
    assert two_room_layout().dwelling.name == scene["dwelling"]
