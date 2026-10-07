"""The parts of the small applications, on this side of the socket.

What a page is handed is held here to what it must be: frames that are the
files' own samples and their sum under a balance, a sonogram whose scale is
a level and not a picture's, a blind test whose key the page cannot read and
whose probability is the binomial's, a balance that is kept, and a server
that serves a folder and nothing above it. No browser and no network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from reverberate.apps import compare
from reverberate.render.output import write_signal
from reverberate.render.pack import synthetic_free_field
from reverberate.viz.parts import balance as balances
from reverberate.viz.parts.blind import BlindTest, chance
from reverberate.viz.parts.media import Item, Library, ambisonic_stem, kit_folder
from reverberate.viz.parts.scene import scene_view
from reverberate.viz.parts.server import AppServer, Binary, HttpError, Request
from reverberate.viz.parts.sonogram import difference, levels_db, sonogram

FS = 48000.0
FRAMES = 4800


def stem(folder: Path, name: str, seed: int, *, channels: int = 64) -> np.ndarray:
    """A scene signal of noise at ``folder/name``; its samples, ``[channel, sample]``."""
    data = np.random.default_rng(seed).standard_normal((channels, FRAMES)).astype(np.float32) * 0.1
    write_signal(folder / f"{name}.f32", [data], sample_rate_hz=FS, order=7, sources=[name])
    return data


def ears(path: Path, seed: int) -> np.ndarray:
    data = np.random.default_rng(seed).uniform(-0.5, 0.5, (FRAMES, 2)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), data, int(FS), subtype="FLOAT")
    return data


@pytest.fixture
def kit(tmp_path: Path) -> dict[str, Any]:
    """A comparison of two variants as the listening kit lays it: order 7 and two ears."""
    folder = tmp_path / "AB"
    held: dict[str, np.ndarray] = {}
    listen: list[dict[str, Any]] = []
    for variant in ("ref", "cheap"):
        for index, source in enumerate(("near_1", "noise_1")):
            held[f"{variant}_{source}"] = stem(
                folder / "ambisonic", f"{variant}_{source}", len(held) + index
            )
            ears(folder / "listen" / f"{variant}_{source}.wav", 10 + len(held))
            listen.append(
                {
                    "variant": variant,
                    "source": source,
                    "signal": "clips",
                    "wav": f"listen/{variant}_{source}.wav",
                    "ambisonic": f"ambisonic/{variant}_{source}.json",
                }
            )
        held[f"{variant}_wav"] = ears(folder / "listen" / f"{variant}_mix.wav", 20 + len(held))
        listen.append(
            {
                "variant": variant,
                "source": "mix",
                "signal": "clips",
                "wav": f"listen/{variant}_mix.wav",
                "ambisonic": None,
            }
        )
        # Pink noise through one source, of which only the two ears were kept.
        ears(folder / "listen" / f"{variant}_near_1_pink.wav", 30 + len(held))
        listen.append(
            {
                "variant": variant,
                "source": "near_1",
                "signal": "pink",
                "wav": f"listen/{variant}_near_1_pink.wav",
                "ambisonic": None,
            }
        )
    bands = [round(1000.0 * 2.0 ** (k / 3.0), 1) for k in range(-10, 13)]
    document = {
        "schema": "reverberate.sound-check.variants",
        "reference": {"name": "ref"},
        "variants": {
            "cheap": {
                "less_the_reference_db": {
                    "near_1": {
                        "impulse": {"time_s": 1.0, "early_db": [1.0] * 23, "late_db": [None] * 23},
                        "clips_db": [0.5] * 23,
                        "worst_abs_db": {"early": {"under_the_crossover": 1.0}},
                    }
                }
            }
        },
        "window_s": [10.0, 10.1],
        "sources": ["near_1", "noise_1"],
        "third_octaves_hz": bands,
        "crossover_hz": 1000.0,
        "early_s": 0.05,
        "listen": listen,
        "signals": {"clips": "the recipe's clips", "pink": "pink noise"},
        "head": {"step_s": 0.05, "yaw_deg": [0.0, 10.0, 20.0], "pitch_deg": [0.0] * 3},
        "recipe_sha256": "0" * 64,
    }
    folder.mkdir(exist_ok=True)
    (folder / "variants.json").write_text(json.dumps(document))
    return {"folder": folder, "held": held, "document": document}


# --- the server ---------------------------------------------------------------------------


def test_a_server_answers_routes_and_files_and_serves_nothing_above_its_folder(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>page</p>")
    (static / "app.js").write_text("export {};")
    (tmp_path / "secret.txt").write_text("not served")
    server = AppServer("test", static)

    def echo(request: Request) -> dict[str, Any]:
        return {"parts": list(request.parts), "n": request.number("n", 1.0), "body": request.body}

    def refuse(request: Request) -> Any:
        raise HttpError(409, "not now")

    server.route("GET", "api/echo", echo)
    server.route("POST", "api/echo", echo)
    server.route("GET", "api/raw", lambda request: Binary(b"\x01\x02", headers={"X-Said": "so"}))
    server.route("GET", "api/refuse", refuse)
    assert server.handle("GET", "/").payload == b"<p>page</p>"
    assert server.handle("GET", "/app.js").kind == "text/javascript"
    # The components and the inspector's decoder are mounted for every application.
    assert b"createPlayer" in server.handle("GET", "/parts/player.js").payload
    assert b"createStreamDecoder" in server.handle("GET", "/reuse/scene/sound-decode.js").payload
    assert server.handle("GET", "/api/echo/a/b?n=3").json() == {
        "parts": ["a", "b"],
        "n": 3.0,
        "body": None,
    }
    assert server.handle("POST", "/api/echo", b'{"k": 1}').json()["body"] == {"k": 1}
    raw = server.handle("GET", "/api/raw")
    assert raw.payload == b"\x01\x02" and raw.headers["X-Said"] == "so"
    assert server.handle("GET", "/api/echo?n=x").status == 400
    assert server.handle("POST", "/api/echo", b"{not json").status == 400
    refused = server.handle("GET", "/api/refuse")
    assert refused.status == 409 and refused.json() == {"error": "not now"}
    assert server.handle("GET", "/missing.js").status == 404
    assert server.handle("POST", "/app.js").status == 404
    for climb in ("/../secret.txt", "/%2e%2e/secret.txt", "/parts/../../secret.txt"):
        assert server.handle("GET", climb).status == 404


# --- what is played -----------------------------------------------------------------------


def test_frames_are_the_files_own_samples_and_their_sum_under_a_balance(
    kit: dict[str, Any],
) -> None:
    read = kit_folder(kit["folder"])
    held = kit["held"]
    assert read.variants == ("ref", "cheap")
    # Order 7 where every variant has it: the mix is its sources' stems, summed here.
    assert set(read.sets["clips"]) == {"mix", "near_1", "noise_1"}
    name = read.sets["clips"]["mix"]["cheap"]
    item = read.library.item(name)
    assert (item.kind, item.channels, item.frames, item.level_db) == (
        "ambisonic",
        64,
        FRAMES,
        -12.0,
    )
    assert [s.id for s in item.stems] == ["near_1", "noise_1"]
    one = read.library.frames(read.sets["clips"]["near_1"]["ref"], 100, 50)
    assert one.dtype == np.float32 and np.array_equal(one, held["ref_near_1"][:, 100:150].T)
    both = read.library.frames(name, 0, FRAMES)
    summed = held["cheap_near_1"].astype(np.float64) + held["cheap_noise_1"].astype(np.float64)
    assert np.array_equal(both, summed.T.astype(np.float32))
    under = read.library.frames(name, 0, FRAMES, {"near_1": 0.5, "noise_1": 0.0})
    assert np.allclose(under, 0.5 * held["cheap_near_1"].T, atol=1e-7)
    # Past the end a chunk keeps the length asked for, in zeros.
    last = read.library.frames(name, FRAMES - 10, 30)
    assert last.shape == (30, 64) and not np.any(last[10:]) and np.any(last[:10])
    # A signal of which only two ears were written is played as written.
    pink = read.library.item(read.sets["pink"]["mix"]["ref"])
    assert (pink.kind, pink.channels, pink.level_db) == ("binaural", 2, 0.0)
    assert list(read.sets["pink"]) == ["mix"]
    # What is looked at is at the page's level: the omnidirectional channel, 12 dB down.
    seen = read.library.mono(read.sets["clips"]["near_1"]["ref"])
    assert np.allclose(seen, held["ref_near_1"][0] * 10.0 ** (-12.0 / 20.0), atol=1e-7)


def test_a_folder_written_before_it_listed_its_files_is_read_by_their_names(
    kit: dict[str, Any], tmp_path: Path
) -> None:
    folder = kit["folder"]
    document = {k: v for k, v in kit["document"].items() if k not in ("listen", "head")}
    document["files"] = {
        f"{variant}_{source}": f"/elsewhere/listen/{variant}_{source}.wav"
        for variant in ("ref", "cheap")
        for source in ("mix", "near_1", "noise_1")
    }
    (folder / "variants.json").write_text(json.dumps(document))
    read = kit_folder(folder)
    # A folder that was moved since: the files are those beside the document.
    assert set(read.sets["clips"]) == {"mix", "near_1", "noise_1"} and read.head is None
    item = read.library.item(read.sets["clips"]["mix"]["ref"])
    assert item.kind == "binaural"
    assert np.array_equal(read.library.frames(item.id, 0, FRAMES), kit["held"]["ref_wav"])
    with pytest.raises(SystemExit, match="not a comparison"):
        kit_folder(tmp_path)


def test_variants_of_two_lengths_are_refused_and_stems_of_two_shapes_too(
    kit: dict[str, Any],
) -> None:
    folder = kit["folder"]
    short = np.zeros((100, 2), dtype=np.float32)
    soundfile.write(str(folder / "listen" / "cheap_near_1_pink.wav"), short, int(FS))
    with pytest.raises(SystemExit, match="not of one length"):
        kit_folder(folder)
    a = ambisonic_stem(folder / "ambisonic" / "ref_near_1.json", "a")
    stem(folder / "ambisonic", "small", 1, channels=4)
    b = ambisonic_stem(folder / "ambisonic" / "small.json", "b")
    with pytest.raises(ValueError, match="not of one shape"):
        Library([Item("x", "x", (a, b))])


# --- what is looked at --------------------------------------------------------------------


def test_a_sonogram_reads_a_level_where_the_sound_is_whatever_else_it_holds() -> None:
    time = np.arange(int(FS)) / FS
    # A tone at the middle of a band, a little over 1 kHz.
    hz = float(50.0 * 2.0 ** (52.5 / 12.0))
    tone = np.sin(2.0 * np.pi * hz * time)
    made = sonogram(tone, FS)
    assert made.levels_db.shape == (100, 100) and made.hop_s == 0.01
    assert made.edges_hz[0] == 50.0 and made.edges_hz[12] == pytest.approx(100.0)
    assert made.centres_hz[52] == pytest.approx(hz)
    middle = made.levels_db[50].astype(float)
    # A sine of full scale is a mean square of one half: -3.01 dB, in the band it lies in.
    assert 10.0 * math.log10(float(np.sum(10.0 ** (middle / 10.0)))) == pytest.approx(
        -3.01, abs=0.02
    )
    loudest = int(np.argmax(middle))
    assert loudest == 52 and middle[loudest] == pytest.approx(-3.01, abs=0.2)
    # The window's skirt: three bands away it is 60 dB down.
    assert max(middle[:49].max(), middle[56:].max()) < -60.0
    # Half the amplitude is 6.02 dB down everywhere: nothing is normalised to the signal.
    half = sonogram(0.5 * tone, FS)
    assert half.levels_db[50, loudest] - middle[loudest] == pytest.approx(-6.02, abs=0.01)
    # White noise holds its power in proportion to each band's width.
    noise = np.random.default_rng(0).standard_normal(4 * int(FS)) * 0.1
    wide = sonogram(noise, FS)
    power = np.mean(10.0 ** (wide.levels_db[10:-10].astype(float) / 10.0), axis=0)
    width = np.diff(wide.edges_hz)
    density = 10.0 * np.log10(power / width / (0.01 / (FS / 2.0)))
    # A band under 100 Hz is one bin's share over four seconds: it scatters by a decibel.
    assert np.abs(density).max() < 1.5 and abs(float(np.mean(density))) < 0.1
    # Silence is silence, and a column stands at its own instant.
    late = np.concatenate([np.zeros(int(FS) // 2), tone[: int(FS) // 2]])
    seen = sonogram(late, FS).levels_db[:, loudest]
    assert seen[:45].max() < -100.0 and seen[55:].min() > -4.5


def test_a_difference_is_the_level_of_one_less_the_other_and_nothing_where_both_are_silent() -> (
    None
):
    time = np.arange(int(FS)) / FS
    tone = np.sin(2.0 * np.pi * 500.0 * time)
    a, b = sonogram(tone, FS), sonogram(2.0 * tone, FS)
    less = difference(b, a)
    band = int(np.argmax(a.levels_db[50]))
    assert less[50, band] == pytest.approx(6.02, abs=0.01)
    assert np.isnan(less[50, -1]) and np.isnan(less[50, 0])
    with pytest.raises(ValueError, match="two shapes"):
        difference(a, sonogram(tone[:1000], FS))
    # A source's level a hop: the rms of each stretch.
    levels = levels_db(np.concatenate([np.zeros(2400), 0.1 * np.ones(2400)]), FS, hop_s=0.05)
    assert levels.tolist() == pytest.approx([-160.0, -20.0])


# --- the blind test -----------------------------------------------------------------------


def test_the_probability_of_a_score_by_chance_is_the_binomial_s() -> None:
    assert chance(0, 16) == 1.0
    assert chance(16, 16) == pytest.approx(0.5**16)
    # Twelve right out of sixteen: the 3.8 per cent every ABX table gives.
    assert chance(12, 16) == pytest.approx(0.0384, abs=5e-5)
    assert chance(9, 10) == pytest.approx(11 / 1024)
    assert chance(3, 6, 1.0 / 3.0) == pytest.approx(1.0 - (64 + 6 * 32 + 15 * 16) / 729)
    with pytest.raises(ValueError, match="not a score"):
        chance(5, 4)


def test_an_abx_test_keeps_its_key_scores_the_answers_and_writes_them(tmp_path: Path) -> None:
    test = BlindTest("abx", {"ref": "item-ref", "cheap": "item-cheap"}, 6, tmp_path, seed=5)
    told = test.state()
    assert told["known"] == {"A": "ref", "B": "cheap"} and told["hidden"] == ["X"]
    # Nothing the page is told names what X is.
    assert "item" not in json.dumps(told) and "result" not in told
    assert test.resolve("A") == "item-ref" and test.resolve("B") == "item-cheap"
    drawn = []
    for trial in range(6):
        hidden = test.resolve("X")
        drawn.append(hidden)
        # Right for the first four, wrong for the last two.
        right = "A" if hidden == "item-ref" else "B"
        wrong = "B" if right == "A" else "A"
        told = test.answer(right if trial < 4 else wrong)
        assert ("result" in told) == (trial == 5)
    # X was drawn again and was not always the same.
    assert len(set(drawn)) == 2
    assert told["finished"] and told["result"]["right"] == 4
    assert told["result"]["chance"] == pytest.approx(chance(4, 6))
    written = json.loads(Path(told["saved"]).read_text())
    assert written["schema"] == "reverberate.apps.blind" and written["complete"] is True
    assert [t["right"] for t in written["trials"]] == [True] * 4 + [False] * 2
    assert all(t["x"] in ("ref", "cheap") and t["seconds"] >= 0 for t in written["trials"])
    assert written["seed"] == 5 and written["items"] == {"ref": "item-ref", "cheap": "item-cheap"}
    for refused in (lambda: test.answer("A"), lambda: test.resolve("X")):
        with pytest.raises(HttpError, match="over"):
            refused()
    # The same seed draws the same test.
    again = BlindTest("abx", {"ref": "item-ref", "cheap": "item-cheap"}, 6, tmp_path, seed=5)
    assert again.resolve("X") == drawn[0]
    with pytest.raises(HttpError, match="A or B"):
        again.answer("X")
    with pytest.raises(HttpError, match="of two"):
        BlindTest("abx", {"a": "1", "b": "2", "c": "3"}, 4, tmp_path)


def test_a_ranking_hides_every_name_and_tells_who_came_first_and_how_likely(
    tmp_path: Path,
) -> None:
    items = {"ref": "i1", "cheap": "i2", "cheaper": "i3"}
    test = BlindTest("rank", items, 4, tmp_path, seed=1)
    assert test.state()["hidden"] == ["X1", "X2", "X3"] and test.state()["known"] == {}
    orders = []
    for trial in range(3):
        shown = {name: test.resolve(name) for name in ("X1", "X2", "X3")}
        orders.append(tuple(shown.values()))
        # The listener always prefers the reference, then the cheaper one.
        order = sorted(shown, key=lambda name: ["i1", "i3", "i2"].index(shown[name]))
        with pytest.raises(HttpError, match="every hidden item once"):
            test.answer(order[:2])
        test.answer(order)
        del trial
    assert len(set(orders)) > 1
    told = test.stop()
    result = told["result"]
    assert result["first"] == {"ref": 3, "cheap": 0, "cheaper": 0}
    assert result["mean_rank"] == {"ref": 1.0, "cheap": 3.0, "cheaper": 2.0}
    # Three firsts out of three, a third each, for whichever of the three did best.
    assert result["chance"] == pytest.approx(3 * (1 / 3) ** 3)
    written = json.loads(Path(told["saved"]).read_text())
    assert written["complete"] is False and written["trials_planned"] == 4
    assert written["trials"][0]["ranking"] == ["ref", "cheaper", "cheap"]
    assert sorted(written["trials"][0]["shown"]) == ["X1", "X2", "X3"]


# --- the balance --------------------------------------------------------------------------


def test_a_balance_becomes_gains_and_is_kept(tmp_path: Path) -> None:
    names = ["near_1", "near_2", "noise_1"]
    cleaned = balances.clean(
        {"sources": {"near_1": {"gain_db": -6}, "noise_1": {"gain_db": 99, "mute": True}}}, names
    )
    assert cleaned["near_2"] == {"gain_db": 0.0, "mute": False, "solo": False}
    assert cleaned["noise_1"]["gain_db"] == 12.0
    gains = balances.gains(cleaned)
    assert gains == {"near_1": pytest.approx(10 ** (-6 / 20)), "near_2": 1.0, "noise_1": 0.0}
    cleaned["near_2"]["solo"] = True
    assert balances.gains(cleaned) == {"near_1": 0.0, "near_2": 1.0, "noise_1": 0.0}
    cleaned["near_2"]["gain_db"] = -60.0
    assert balances.gains(cleaned)["near_2"] == 0.0
    path = tmp_path / "kept" / "balance.json"
    document = balances.save(path, cleaned, {"scene": "one"})
    assert json.loads(path.read_text()) == document
    assert document["schema"] == "reverberate.apps.balance" and document["context"] == {
        "scene": "one"
    }
    assert balances.load(path, names) == cleaned
    # A source the file does not know is at 0 dB; one it knows and the scene has not is left.
    assert balances.load(path, ["near_1", "far_9"])["far_9"]["gain_db"] == 0.0
    refused: list[dict[str, Any]] = [
        {},
        {"sources": {"nobody": {}}},
        {"sources": {"near_1": {"gain_db": "loud"}}},
    ]
    for bad in refused:
        with pytest.raises(HttpError):
            balances.clean(bad, names)


# --- the routes, through the comparator ------------------------------------------------------


def test_the_comparator_serves_its_folder_its_differences_and_a_blind_test(
    kit: dict[str, Any], tmp_path: Path
) -> None:
    server, read, media = compare.build(
        kit["folder"], tmp_path / "decoders", measured_head=tmp_path / "none.sofa"
    )
    assert media.results == kit["folder"].parent / "AB_listening"
    assert b"<title>Compare</title>" in server.handle("GET", "/").payload
    # No measured head here: the page is told that order 7 is not decoded.
    assert server.handle("GET", "/decoders/decoders.json").json() == []
    told = server.handle("GET", "/api/kit").json()
    assert told["variants"] == ["ref", "cheap"] and told["head"]["yaw_deg"] == [0.0, 10.0, 20.0]
    less = told["differences"]
    assert less["reference"] == "ref" and less["crossover_hz"] == 1000.0
    assert less["by_variant"]["cheap"]["near_1"]["early"] == [1.0] * 23
    assert less["by_variant"]["cheap"]["near_1"]["late"] == [None] * 23
    name = told["sets"]["clips"]["mix"]["cheap"]
    answer = server.handle("GET", f"/api/frames?item={name}&start=10&count=20")
    assert answer.headers == {"X-Frames": "20", "X-Channels": "64"}
    assert np.array_equal(
        np.frombuffer(answer.payload, "<f4").reshape(20, 64), read.library.frames(name, 10, 20)
    )
    assert server.handle("GET", "/api/frames?item=nobody&start=0&count=1").status == 404
    assert server.handle("GET", f"/api/frames?item={name}&start=0&count=999999999").status == 413
    # The track list: a balance is saved, and the frames asked under it are mixed by it.
    balance = server.handle("GET", "/api/balance").json()
    assert [t["id"] for t in balance["tracks"]] == ["near_1", "noise_1"]
    assert [t["kind"] for t in balance["tracks"]] == ["voice", "noise"]
    assert balance["version"] == 0 and balance["saved"] is None
    body = json.dumps({"sources": {"near_1": {"gain_db": -6.0}, "noise_1": {"mute": True}}})
    balance = server.handle("POST", "/api/balance", body.encode()).json()
    assert balance["version"] == 1 and Path(balance["saved"]).is_file()
    under = server.handle("GET", f"/api/frames?item={name}&start=0&count={FRAMES}&balance=1")
    assert np.allclose(
        np.frombuffer(under.payload, "<f4").reshape(FRAMES, 64),
        10 ** (-6 / 20) * kit["held"]["cheap_near_1"].T,
        atol=1e-7,
    )
    assert server.handle("GET", f"/api/frames?item={name}&start=0&count=1&balance=7").status == 409
    # What is looked at.
    seen = server.handle("GET", f"/api/sonogram?item={name}")
    said = json.loads(seen.headers["X-Sonogram"])
    assert (said["columns"], said["bands"]) == (10, 100)
    assert len(seen.payload) == 10 * 100 * 4
    other = told["sets"]["clips"]["mix"]["ref"]
    less_seen = server.handle("GET", f"/api/sonogram?item={name}&less={other}")
    assert json.loads(less_seen.headers["X-Sonogram"])["less"] == other
    levels = server.handle("GET", f"/api/levels?item={name}").json()
    assert set(levels["stems"]) == {"near_1", "noise_1"} and len(levels["stems"]["near_1"]) == 2
    # A blind test: the page plays a name, the server knows what it is, and nothing is shown.
    assert server.handle("GET", "/api/blind").json() == {"kind": None}
    assert server.handle("GET", "/api/frames?item=blind:X&start=0&count=1").status == 409
    start = {"kind": "abx", "trials": 2, "seed": 3, "items": {"ref": other, "cheap": name}}
    state = server.handle("POST", "/api/blind/start", json.dumps(start).encode()).json()
    assert state["trial"] == 1 and not state["finished"]
    assert server.handle("GET", f"/api/sonogram?item={name}").status == 409
    for _ in range(2):
        heard = server.handle("GET", f"/api/frames?item=blind:X&start=0&count={FRAMES}")
        x = np.frombuffer(heard.payload, "<f4").reshape(FRAMES, 64)
        is_a = np.array_equal(x, read.library.frames(other, 0, FRAMES))
        assert is_a or np.array_equal(x, read.library.frames(name, 0, FRAMES))
        answer_body = json.dumps({"answer": "A" if is_a else "B"}).encode()
        state = server.handle("POST", "/api/blind/answer", answer_body).json()
    assert state["finished"] and state["result"]["right"] == 2
    written = json.loads(Path(state["saved"]).read_text())
    assert Path(state["saved"]).parent == media.results
    assert written["context"]["folder"] == str(kit["folder"].resolve())
    assert written["result"]["chance"] == 0.25
    assert server.handle("POST", "/api/blind/start", b'{"kind": "abx"}').status == 400
    assert server.handle("GET", f"/api/sonogram?item={name}").status == 200


def test_two_packs_side_by_side_are_read_as_a_and_b() -> None:
    document = {
        "third_octaves_hz": [1000.0],
        "crossover_hz": 1000.0,
        "difference_b_less_a_db": {
            "s1": {"impulse": {"early_db": [6.0], "late_db": [None]}, "clips_db": [6.0]}
        },
    }
    less = compare.differences(document)
    assert less["reference"] == "A" and less["by_variant"]["B"]["s1"]["clips"] == [6.0]


# --- the scene ----------------------------------------------------------------------------


def test_a_scene_is_drawn_from_the_pack_s_own_tables() -> None:
    pack = synthetic_free_field(level="B", duration_s=2.0)
    view = scene_view(pack, (0.5, 1.5), every=2)
    assert view["window_s"] == [0.5, 1.5] and view["step_s"] == pytest.approx(0.1)
    assert len(view["listener"]["position"]) == 11 == len(view["listener"]["yaw_deg"])
    (source,) = view["sources"]
    assert source["position"][0] == [2.0, 1.5, 0.0] and len(source["audible"]) == 11
    # The free field's one source is a noise, and every noise is the same grey.
    assert source["kind"] == "noise" and source["colour"] == "#8a94a6"
    # The floor is the cells the low band was solved at: two, 0.40 m apart, each once.
    cells = view["floor"]["cells"]
    assert len(cells) == 2 and abs(cells[1][0] - cells[0][0]) == pytest.approx(0.4)
    assert view["floor"]["y"] == 0.0 and len(view["floor"]["rooms"]) >= 1
    # The whole scene when no window is given.
    assert len(scene_view(pack, every=1)["listener"]["position"]) == pack.header.steps
