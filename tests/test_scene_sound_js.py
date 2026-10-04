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
