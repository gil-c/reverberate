"""Tests for the screen of a noise library.

No model and no network: the detections are written here, as the detector
script would have written them, for clips that are noise with a tone where
the "speech" is. What matters is what a scene leans on: a passage the
detectors found is gone from the file and nothing else is, the join is at
the level of its sides, the level the entry states is the file's, a
programme is routed and not deleted, a clip of which too little would
remain is rejected and said to be, and the screened file is made again
from its source and its entry alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reverberate.scenes import clips as clip_library
from reverberate.scenes import load_clip_library, screen

RATE = clip_library.RATE_HZ
FRAME_S = 0.032
TONE_HZ = 1000.0


# --------------------------------------------------------------------------
# detections, as scripts/clip_speech_detect.py writes them
# --------------------------------------------------------------------------


def _detection(
    duration: float,
    *,
    vad: list[tuple[float, float]] = (),  # type: ignore[assignment]
    tags: dict[str, float] | None = None,
    tagged: list[tuple[float, float]] | None = None,
    segments: list[dict[str, Any]] = (),  # type: ignore[assignment]
) -> dict[str, Any]:
    """A clip's detections: the voice detector on in ``vad``, the tagger's
    ``tags`` in the windows that meet ``tagged`` (all of them when ``None``)."""
    frames = [0.02] * int(duration / FRAME_S)
    for a, b in vad:
        for k in range(int(a / FRAME_S), min(int(b / FRAME_S) + 1, len(frames))):
            frames[k] = 0.95
    windows, start = [], 0.0
    while start < max(duration - 10.0, 0.0) + 1e-9:
        end = min(start + 10.0, duration)
        on = tagged is None or any(a < end and b > start for a, b in tagged)
        windows.append({"start_s": start, "end_s": end, "scores": dict(tags or {}) if on else {}})
        start += 5.0
    return {
        "duration_s": duration,
        "vad": {"frame_s": FRAME_S, "speech": frames},
        "tags": windows,
        "words": {"language": "en", "language_probability": 0.9, "segments": list(segments)},
    }


def _segment(words: list[tuple[float, float, str]], **said: float) -> dict[str, Any]:
    return {
        "start_s": words[0][0],
        "end_s": words[-1][1],
        "text": " ".join(w for _, _, w in words),
        "no_speech": said.get("no_speech", 0.05),
        "log_probability": said.get("log_probability", -0.3),
        "compression": said.get("compression", 1.4),
        "words": [[a, b, said.get("probability", 0.9), w] for a, b, w in words],
    }


# --------------------------------------------------------------------------
# passages
# --------------------------------------------------------------------------


def test_a_voice_is_marked_with_its_margins_and_two_near_ones_are_one() -> None:
    rules = screen.Rules()
    found = screen.passages(_detection(60.0, vad=[(10.0, 12.0), (12.6, 14.0), (40.0, 41.0)]), rules)
    assert len(found) == 2
    (a, b), (c, d) = found
    assert a == pytest.approx(10.0 - rules.pad_s, abs=FRAME_S)
    assert b == pytest.approx(14.0 + rules.pad_s, abs=2 * FRAME_S)
    assert c == pytest.approx(40.0 - rules.pad_s, abs=FRAME_S)
    assert d == pytest.approx(41.0 + rules.pad_s, abs=2 * FRAME_S)


def test_nothing_detected_is_no_passage() -> None:
    assert screen.passages(_detection(30.0)) == []
    assert screen.decide(_detection(30.0)).action == "kept"


def test_a_word_the_recogniser_is_sure_of_is_speech_where_the_voice_detector_heard_none() -> None:
    said = _segment([(20.0, 20.4, "good"), (20.5, 21.0, "morning")])
    found = screen.passages(_detection(60.0, segments=[said]))
    assert len(found) == 1
    assert found[0][0] < 20.0 and found[0][1] > 21.0


@pytest.mark.parametrize(
    "said",
    [{"no_speech": 0.9}, {"log_probability": -1.6}, {"compression": 12.0}, {"probability": 0.2}],
)
def test_what_a_recogniser_writes_on_a_noise_is_not_speech(said: dict[str, float]) -> None:
    # The syllable repeated a hundred times, the thanks that end a video.
    words = [(float(k), k + 0.5, "thanks") for k in range(5, 25)]
    detection = _detection(60.0, segments=[_segment(words, **said)])
    assert screen.sure_words(detection) == []
    assert screen.passages(detection) == []


def test_a_murmur_takes_its_window_whole_and_a_voice_in_it_keeps_its_own_edges() -> None:
    voice = {"Speech": 0.8}
    # The tagger alone: every window it calls speech, whole, with the margins.
    murmur = _detection(30.0, tags=voice, tagged=[(12.0, 14.0)])
    assert screen.passages(murmur) == [(4.7, 20.3)]
    # With a voice the other detector placed, the tagger's windows add nothing there.
    placed = _detection(30.0, vad=[(12.0, 14.0)], tags=voice, tagged=[(12.0, 14.0)])
    ((a, b),) = screen.passages(placed)
    assert 11.5 < a < 12.0 and 14.0 < b < 14.5


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


def test_a_clip_loses_its_passage_when_enough_remains_and_is_rejected_when_not() -> None:
    rules = screen.Rules()
    cut = screen.decide(_detection(30.0, vad=[(10.0, 13.0)]), "appliance", rules)
    assert cut.action == "cut"
    assert [round(b - a) for a, b in cut.pieces] == [10, 17]
    # Speech nearly all along: what is left is under the least a clip may keep.
    talk = screen.decide(_detection(30.0, vad=[(1.0, 27.0)]), "appliance", rules)
    assert talk.action == "rejected" and talk.pieces == ()
    # A piece of noise too short to stand between two passages goes with them.
    holes = screen.decide(_detection(40.0, vad=[(10.0, 12.0), (13.5, 15.0)]), "water", rules)
    assert len(holes.pieces) == 2


def test_a_programme_and_a_song_are_routed_and_not_cut() -> None:
    talk = _detection(60.0, vad=[(0.5, 59.0)])
    assert screen.decide(talk, "television").action == "routed"
    song = _detection(30.0, vad=[(5.0, 20.0)], tags={"Music": 0.9, "Singing": 0.7})
    assert screen.is_media(song)
    verdict = screen.decide(song, "other")
    assert verdict.action == "routed" and verdict.pieces == ((0.0, 30.0),)
    radio = _detection(30.0, tags={"Radio": 0.6, "Speech": 0.9})
    assert screen.decide(radio, "other").action == "routed"
    # Music with no one singing stays music; with a voice the tagger did not
    # call sung it is a presenter's piece, and is not cut either.
    piece = _detection(30.0, tags={"Music": 0.9})
    assert screen.decide(piece, "music").action == "kept"
    spoken = _detection(30.0, vad=[(5.0, 8.0)], tags={"Music": 0.9})
    assert screen.decide(spoken, "music").action == "routed"


# --------------------------------------------------------------------------
# removal
# --------------------------------------------------------------------------


def _noise(seed: int, seconds: float, level: float = 0.03) -> np.ndarray:
    return level * np.random.default_rng(seed).standard_normal(int(seconds * RATE))


def _tone(x: np.ndarray, a: float, b: float, level: float = 0.2) -> np.ndarray:
    """``x`` with a loud tone from ``a`` to ``b`` seconds: the voice in the noise."""
    out = x.copy()
    n = np.arange(int(a * RATE), int(b * RATE))
    out[n] += level * np.sin(2 * np.pi * TONE_HZ * n / RATE)
    return out


def _tone_level(x: np.ndarray) -> float:
    """The largest share of a 50 ms frame's power that lies at the tone."""
    frame = int(0.05 * RATE)
    k = int(round(TONE_HZ * frame / RATE))
    worst = 0.0
    for start in range(0, x.size - frame, frame):
        spectrum = np.abs(np.fft.rfft(x[start : start + frame])) ** 2
        worst = max(worst, float(spectrum[k - 1 : k + 2].sum() / spectrum.sum()))
    return worst


def test_the_passage_is_gone_and_the_join_is_at_the_level_of_its_sides() -> None:
    x = _tone(_noise(1, 20.0), 8.0, 11.0)
    assert _tone_level(x) > 0.9
    out = screen.remove(x, [(0.0, 7.5), (11.5, 20.0)], crossfade_s=0.25)
    assert out.size == int((7.5 + 8.5 - 0.25) * RATE)
    assert _tone_level(out) < 0.05
    # Equal power: through the join the level is that of the noise.
    whole = float(np.sqrt(np.mean(out**2)))
    join = out[int(7.25 * RATE) : int(7.5 * RATE)]
    assert 20 * np.log10(float(np.sqrt(np.mean(join**2))) / whole) == pytest.approx(0.0, abs=0.5)
    # Nothing else moved: before the join, the source's own samples.
    np.testing.assert_array_equal(out[: int(7.25 * RATE)], x[: int(7.25 * RATE)])


def test_a_loop_cut_at_its_end_loops_again_and_a_piece_too_short_is_refused() -> None:
    x = _noise(2, 12.0)
    out = screen.remove(x, [(0.0, 9.0)], crossfade_s=0.25, loop=True)
    assert out.size == int(8.75 * RATE)
    # The seam: the last sample runs into the first as two neighbours do.
    steps = np.abs(np.diff(out))
    assert abs(out[0] - out[-1]) < 4 * float(np.percentile(steps, 99))
    with pytest.raises(ValueError, match="shorter than two crossfades"):
        screen.remove(x, [(0.0, 0.3), (5.0, 12.0)], crossfade_s=0.25)


# --------------------------------------------------------------------------
# a library through the screen
# --------------------------------------------------------------------------


def _library(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """A library of four noises and a voice on the disk, and its detections."""
    made = {
        "noise/fan": ("appliance", _tone(_noise(3, 30.0), 12.0, 15.0), True),
        "noise/tap": ("water", _noise(4, 20.0), True),
        "noise/cafe": ("other", _tone(_noise(5, 20.0), 1.0, 18.5), False),
        "noise/television_01": ("television", _tone(_noise(6, 20.0), 0.5, 19.5), False),
    }
    clips = []
    for name, (subtype, samples, loop) in made.items():
        samples = samples * 10 ** (-30 / 20) / float(np.sqrt(np.mean(samples**2)))
        pcm = clip_library._pcm(samples)
        payload = clip_library.wav_bytes(pcm)
        clip_library._write(clip_library.clip_path(root, "lib", name), payload)
        clips.append(
            {
                "name": name,
                "kind": "noise",
                "subtype": subtype,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "duration_s": round(pcm.size / RATE, 3),
                "loop": loop,
                "level": {
                    "measure": "rms",
                    "spl_1m_db": 56.0,
                    **clip_library._levels(pcm, "noise", ()),
                },
                "licence": "CC0 1.0",
                "credit": "made here",
                "what": "noise",
                "origin": {"parts": [], "process": {}},
            }
        )
    clips.append({"name": "voice/s01_01", "kind": "voice", "speaker": "s01", "sha256": "0" * 64})
    manifest = {
        "schema": clip_library.SCHEMA,
        "schema_version": clip_library.SCHEMA_VERSION,
        "library": "lib",
        "sample_rate_hz": RATE,
        "level": {"voice_active_dbfs": -26.0, "full_scale_spl_1m_db": 86.0},
        "datasets": {},
        "clips": clips,
    }
    detections = {
        "schema": screen.DETECTIONS_SCHEMA,
        "schema_version": 1,
        "detectors": {"vad": "written by the test"},
        "clips": {
            "noise/fan": _detection(30.0, vad=[(12.0, 15.0)]),
            "noise/tap": _detection(20.0),
            "noise/cafe": _detection(20.0, vad=[(1.0, 18.5)]),
            "noise/television_01": _detection(20.0, vad=[(0.5, 19.5)]),
        },
    }
    return manifest, detections


def test_a_library_through_the_screen(tmp_path: Path) -> None:
    manifest, detections = _library(tmp_path)
    screened = screen.screen_library(manifest, detections, tmp_path, "lib_screened", tmp_path)
    by = {clip["name"]: clip for clip in screened["clips"]}
    # The voices are not taken; the cafe is all talk and is rejected, by name.
    assert sorted(by) == ["noise/fan", "noise/tap", "noise/television_01"]
    (gone,) = screened["screen"]["rejected"]
    assert gone["name"] == "noise/cafe" and gone["action"] == "rejected"
    assert gone["evidence"]["speech_share"] > 0.8
    assert screened["screen"]["source_library"] == "lib"
    assert screened["screen"]["rules"]["crossfade_s"] == screen.Rules().crossfade_s

    # The fan lost its three seconds and their margins, and nothing else.
    fan = by["noise/fan"]
    assert fan["screen"]["action"] == "cut"
    assert fan["duration_s"] == pytest.approx(30.0 - 3.0 - 2 * 0.3 - 2 * 0.25, abs=0.1)
    assert fan["sha256"] != manifest["clips"][0]["sha256"]
    assert fan["screen"]["source"] == {
        "library": "lib",
        "name": "noise/fan",
        "sha256": manifest["clips"][0]["sha256"],
    }
    path = clip_library.clip_path(tmp_path, "lib_screened", "noise/fan")
    pcm = screen._read(path)
    assert _tone_level(pcm.astype(float)) < 0.05
    # The tap is the file it was, and the programme too, routed.
    assert by["noise/tap"]["sha256"] == manifest["clips"][1]["sha256"]
    assert by["noise/tap"]["screen"]["action"] == "kept"
    television = by["noise/television_01"]
    assert television["screen"]["route"] == screen.MEDIA_VOICE
    assert television["subtype"] == "television"
    assert television["sha256"] == manifest["clips"][3]["sha256"]

    # The screened library is a library: its levels are what its entries
    # state, and the generator reads it.
    reports = clip_library.check(screened, tmp_path)
    assert [r.failures for r in reports] == [(), (), ()]
    out = tmp_path / "lib_screened.json"
    out.write_text(clip_library.dumps(screened))
    assert len(load_clip_library(out).noises("appliance")) == 1
    assert "cut" in screen.format_table(screened)


def test_a_screened_clip_is_made_again_from_its_source_and_its_entry(tmp_path: Path) -> None:
    manifest, detections = _library(tmp_path / "in")
    screened = screen.screen_library(manifest, detections, tmp_path / "in", "s", tmp_path / "a")
    # No detections here: the entry says what was done.
    done = screen.rebuild(json.loads(json.dumps(screened)), tmp_path / "in", tmp_path / "b")
    assert set(done.values()) == {"built"}
    for clip in screened["clips"]:
        one = clip_library.clip_path(tmp_path / "a", "s", clip["name"]).read_bytes()
        two = clip_library.clip_path(tmp_path / "b", "s", clip["name"]).read_bytes()
        assert one == two and hashlib.sha256(one).hexdigest() == clip["sha256"]
    assert set(screen.rebuild(screened, tmp_path / "in", tmp_path / "b").values()) == {"kept"}
    # A source that is not the file the clip was screened from is refused.
    fan = clip_library.clip_path(tmp_path / "in", "lib", "noise/fan")
    fan.write_bytes(clip_library.wav_bytes(clip_library._pcm(_noise(9, 30.0))))
    clip_library.clip_path(tmp_path / "b", "s", "noise/fan").unlink()
    with pytest.raises(ValueError, match="not the file"):
        screen.rebuild(screened, tmp_path / "in", tmp_path / "b")


def test_nothing_passes_unheard(tmp_path: Path) -> None:
    manifest, detections = _library(tmp_path)
    del detections["clips"]["noise/tap"]
    with pytest.raises(ValueError, match="no detection of noise/tap"):
        screen.screen_library(manifest, detections, tmp_path, "s")
    manifest, detections = _library(tmp_path)
    detections["clips"]["noise/tap"] = _detection(10.0)
    with pytest.raises(ValueError, match="were heard of"):
        screen.screen_library(manifest, detections, tmp_path, "s")
    with pytest.raises(ValueError, match="a name of its own"):
        screen.screen_library(manifest, detections, tmp_path, "lib")


def test_the_command_decides_without_writing_and_then_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, detections = _library(tmp_path / "clips")
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "d.json").write_text(json.dumps(detections))
    common = [
        "screen",
        "--manifest",
        str(tmp_path / "m.json"),
        "--detections",
        str(tmp_path / "d.json"),
        "--library",
        "s",
        "--out",
        str(tmp_path / "s.json"),
        "--root",
        str(tmp_path / "clips"),
    ]
    assert screen.main([*common, "--dry-run"]) == 0
    assert "rejected" in capsys.readouterr().out
    assert not (tmp_path / "s.json").exists() and not (tmp_path / "clips" / "s").exists()
    assert screen.main(common) == 0
    assert "1.5 min of noise before" in capsys.readouterr().out
    screened = json.loads((tmp_path / "s.json").read_text())
    assert len(screened["clips"]) == 3
    for clip in screened["clips"]:
        clip_library.clip_path(tmp_path / "clips", "s", clip["name"]).unlink()
    assert (
        screen.main(
            ["rebuild", "--manifest", str(tmp_path / "s.json"), "--root", str(tmp_path / "clips")]
        )
        == 0
    )
