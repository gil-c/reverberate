"""A scene's sound on the page, through node: the clock, the fetch plan, the decode.

The page renders nothing of a scene's sound: it receives the signal engine's
order 7 frames, turns them by the head and decodes them. What it does own is
held here to what it must be: the timeline's clock is the samples that
sounded and nothing else; chunks are asked in order and an answer of a
stream that was replaced is dropped; and the stream's decode is, sample for
sample, the page's decode of a field (``audio/decode.js``), under a head, a
change of head and a change of mix. ``tests/js/sound.mjs`` runs them.

Skipped where node is not installed, as the other JavaScript tests are.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "reverberate" / "viz" / "app"
HARNESS = ROOT / "tests" / "js" / "sound.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

#: Float32 arithmetic through two transforms of 1024 points, relative to the peak.
EXACT = 2e-6
BLOCK = 512


@pytest.fixture(scope="module")
def out() -> Any:
    done = subprocess.run(
        ["node", str(HARNESS), str(APP)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_scene_time_is_the_samples_that_sounded(out: Any) -> None:
    clock = out["clock"]
    # Play was pressed at 10 s and nothing has sounded: the timeline has not moved.
    assert clock["heldWhileWaiting"] == 10
    assert clock["atBegin"] == 10
    # 4800 samples later it is a tenth of a second on, to the sample.
    assert clock["afterTenthOfASecond"] == pytest.approx(10.1, abs=1e-9)
    # Between two reports the picture runs on, by 30 ms at the very most.
    assert clock["betweenReports"] == pytest.approx(10.104, abs=1e-9)
    assert clock["aheadIsBounded"] == pytest.approx(10.13, abs=1e-9)
    # Waiting for a render: three seconds of the context's time move nothing.
    assert clock["stalled"][0] == clock["stalled"][1] == pytest.approx(10.2, abs=1e-9)


def test_a_seek_is_not_moved_by_the_stream_it_left(out: Any) -> None:
    clock = out["clock"]
    assert clock["afterSeek"] == 40
    assert clock["halfASecondOn"] == pytest.approx(40.5, abs=1e-9)
    assert clock["backwards"] == 0


def test_chunks_are_asked_in_order_and_a_replaced_stream_is_dropped(out: Any) -> None:
    plan = out["plan"]
    assert plan["first"] == [1, 2]  # from the chunk under the cursor, two at a time
    assert plan["none"] == []
    assert plan["mayGoEmpty"] is False and plan["mayGo"] is True
    assert plan["second"] == [3] and plan["third"] == [4]  # four ahead and no further
    assert plan["buffered"] == 3
    # Having run dry, the stream goes on only with enough in hand.
    assert plan["starvedNeedsMore"] == [False, True]
    assert plan["retried"] == [4]
    assert plan["stale"] is False
    assert plan["again"] == [1, 2]
    assert plan["generations"] == [1, 2]
    assert plan["endGoes"] is True and plan["pastEnd"] == []


def test_the_stream_decodes_as_the_page_decodes_a_field(out: Any) -> None:
    decode = out["decode"]
    assert decode["waitsForGo"] is False
    assert decode["preroll"] == BLOCK  # one block is made before any sounds
    assert decode["left"] < EXACT and decode["right"] < EXACT
    assert decode["chunking"] == 0  # where the chunks are cut changes no sample
    assert decode["played"] == 8 * BLOCK  # the whole scene, counted once
    assert decode["ended"] is False and decode["endState"] is True


def test_running_dry_stops_the_count_and_loses_nothing(out: Any) -> None:
    dry = out["dry"]
    assert dry["starved"] is True
    assert dry["playedWhenDry"] == 3 * BLOCK
    assert dry["heldSilent"] is True  # the samples came, the word to go did not
    assert dry["heard"] == 8 * BLOCK
    assert dry["before"] < EXACT and dry["after"] < EXACT


def test_a_new_head_and_a_new_mix_are_one_block_of_crossfade(out: Any) -> None:
    changes = out["changes"]
    assert max(changes[name] for name in ("before", "between", "after")) < EXACT
    assert changes["headFade"] < EXACT and changes["mixFade"] < EXACT
    assert changes["spent"] == 1  # the old head's filters go back to their thread
    assert changes["staleRefused"] is False


def test_the_heads_matrix_is_the_recipes(out: Any) -> None:
    head = out["head"]
    assert head["noRoll"] == 0  # without roll, `headMatrix` of audio/sh.js
    assert head["orthonormal"] < 1e-12
    assert head["front"] == pytest.approx([1, 0, 0], abs=1e-12)
    assert head["leftEarUp"] > 0  # a positive roll lowers the right ear


def test_an_arrival_is_drawn_where_its_sound_came_from(out: Any) -> None:
    glyphs = out["glyphs"]
    # A line as long as the sound travelled, along its direction from the head.
    assert glyphs["ends"][0] == pytest.approx([1 + 3.432, 1.5, 2])
    assert glyphs["ends"][1] == pytest.approx([1, 1.5, 2 - 6.864])
    # The strongest largest, 20 dB under it half way, 60 dB under it the smallest.
    assert glyphs["radii"] == pytest.approx([0.14, 0.085, 0.03])
    assert glyphs["orders"] == [0, 1, 6]  # a diffracted path has its own colour
    assert glyphs["shell"] == pytest.approx([1.25, 1.5, 2, 1, 2.2, 2])
    spans = out["spans"]
    assert spans["seconds"] == [[0, 1.5], [2.5, 3.2]]
    assert spans["common"] == [[2, 4], [6, 7]] and spans["none"] == []
    assert spans["meter"] == [0, 0, 0.5, 1, 0]


def test_the_default_level_keeps_a_whole_scene_under_full_scale(out: Any) -> None:
    from reverberate.render.check import measure

    level = out["level"]
    # One number on the page and in the files the checker writes for listening.
    assert level["default"] == measure.PAGE_DEFAULT_LEVEL_DB == -12
    assert level["fullScaleSpl"] == measure.FULL_SCALE_SPL_DB
    assert level["gains"][0] == pytest.approx(measure.PAGE_DEFAULT_GAIN)
    # The control is kept, from -60 to +20 dB; what is not a number is the default.
    assert level["range"] == [-60, 20]
    assert level["gains"][1] == 1 and level["gains"][2] == level["gains"][3] == level["gains"][0]
    assert level["gains"][4] == pytest.approx(10) and level["gains"][5] == pytest.approx(0.001)
    # Three of fourteen sources peaked at -3.8 dB at 0 dB. Fourteen, their powers added,
    # are 2.9 dB past full scale there and 9.1 dB under it at the default.
    whole = level["wholeSceneAtDefault"]
    assert whole["clipping"] is False and whole["events"] == 0
    assert whole["peakDb"] == pytest.approx(-9.11, abs=0.01)
    assert level["headroomDb"] == pytest.approx(9.11, abs=0.01)


def test_an_output_past_full_scale_is_said_and_nothing_is_limited(out: Any) -> None:
    level = out["level"]
    # The largest sample of the two ears since the audio thread last said.
    assert level["peak"] == 0.75 and level["peakHeld"] == pytest.approx(0.9)
    # At 0 dB the same scene passes full scale: said at once, with by how much ...
    assert level["atZero"]["clipping"] is True and level["atZero"]["events"] == 1
    assert level["atZero"]["overDb"] == pytest.approx(2.89, abs=0.01)
    # ... for three seconds after the last excursion, and no longer.
    assert level["heldAfter"] is True and level["released"]["clipping"] is False
    assert level["released"]["events"] == 1  # what happened stays counted
    # An excursion is counted once however long it lasts; the worst is kept.
    assert level["events"] == 3 and level["worstDb"] == pytest.approx(6.02, abs=0.01)
    # A new level forgets the old one's; exactly full scale is not past it.
    assert level["afterReset"] == {"clipping": False, "events": 0, "peakDb": None, "overDb": 0}
    assert level["atFullScale"] is False
    # The page says, and does not limit: no compressor or shaper stands in the output.
    source = (APP / "scene" / "sound.js").read_text()
    assert "createDynamicsCompressor" not in source and "createWaveShaper" not in source
    assert "node.connect(gain).connect(context.destination)" in source
    assert "value: DEFAULT_LEVEL_DB" in source and "clip.report(message.peak" in source
    worklet = (APP / "scene" / "sound.worklet.js").read_text()
    assert "peak: this.peak" in worklet and "peakOf(output[0]" in worklet


def test_the_packs_of_one_scene_are_offered_in_one_order_with_their_names_and_costs(
    out: Any,
) -> None:
    held = out["variants"]
    # Rails solved at another pitch are another recipe and the same scene.
    assert held["sameSceneOtherRecipe"] is True and held["otherScene"] is False
    assert held["withoutDigest"] == [True, False, False]
    # The reference first, then by name; two of one name are told apart by their ids.
    assert held["order"] == ["aaaaaa", "bbbbbb", "cccccc", "eeeeee", "ffffff"]
    assert held["labels"] == [
        "reference",
        "all-cheap",
        "low-0.8s · cccccc",
        "low-0.8s · eeeeee",
        "pulled",
    ]
    # A scene with one pack offers no choice.
    assert held["alone"] == [] and held["none"] == []
    assert held["names"] == ["reference", "pulled"]
    assert held["costs"] == [
        "whole scene 7.54 USD predicted, this pack 0.91 USD billed",
        "whole scene 5.07 USD predicted, this pack 0.57 USD predicted",
        "",
    ]
    assert held["saving"] == [76, 0, None]
    assert held["title"] == [
        "what all-cheap is",
        "low_ppw = 7.2, rail_pitch_m = 0.12",
        "whole scene 1.81 USD predicted, this pack 0.18 USD predicted",
        "76 % less than reference on the whole scene",
    ]
    assert held["referenceTitle"][1] == "the reference: no option"
    assert held["bareTitle"] == ["no variant.json beside this pack"]
    assert held["steps"] == ["bbbbbb", "ffffff", "bbbbbb", None]


def test_the_sound_bar_switches_among_the_packs_of_a_scene_at_one_instant() -> None:
    source = (APP / "scene" / "sound.js").read_text()
    # The row is filled from the packs of the scene of the pack heard, and a press opens one.
    assert 'seg("variants", []' in source and "scenePacks(packs, pack)" in source
    assert "open(button.dataset.value)" in source
    # A pack of the same scene under another recipe leaves the tracks, and the instant, alone.
    assert "!(shown && sameScene(shown, pack))" in source
    fold = (APP / "scene" / "computed.js").read_text()
    assert "B is not a pack of the same scene as A" in fold and "oneScene(about, aboutB)" in fold
