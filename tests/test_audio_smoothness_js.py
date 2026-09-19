"""The walk-through engine's rules, and its smoothness harness, through node.

Three things, each something this engine once got wrong:

- ``audio/ring.js``, which decides which slot a response goes to and every
  slot's gain. Fades that were re-ramped at every swap approached zero and
  never got there: a ring choosing the slot free longest collapsed three
  slots into two, and every early response was cut short. Pinned under the
  intervals the app really produces.
- The rules of ``audio/spatial.js`` and ``audio/engine.js`` that are plain
  functions: how far the listener may be from a cell, when the late part
  moves, how the propagation delay ramps.
- ``tests/js/smoothness.mjs``, which walks a field offline and measures how
  far the binaural filter moves from frame to frame. It is run by hand on a
  real field (ADR 0013 records the numbers); here, on a mock one, that it
  reads zero when nothing moves and that each decision it measures still
  pays.

Skipped where node is not installed, as the other JavaScript tests are.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reverberate.viz.decoders import export_decoders
from reverberate.viz.field_payload import build_site, mock_field
from test_spatial_binaural import TestAMeasuredHead as _Head

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "src" / "reverberate" / "viz" / "app" / "audio"
HARNESS = ROOT / "tests" / "js" / "smoothness.mjs"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _node(tmp_path: Path, script: str, *args: str) -> Any:
    path = tmp_path / "script.mjs"
    path.write_text(script)
    out = subprocess.run(["node", str(path), *args], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


RING = """
const { createRing } = await import(process.argv[2] + "/ring.js");
const { EARLY_SLOTS, FADE_S } = await import(process.argv[2] + "/engine.js");
const ring = createRing(EARLY_SLOTS, FADE_S);
// The intervals between early responses measured in the app, milliseconds:
// forty as scheduled, and the pairs the worker makes of it when a decode runs
// long and the next is released at once.
const gaps = [42, 38, 66, 18, 40, 53, 29, 42, 58, 24, 89, 4, 61, 22, 78, 4, 65, 53, 7, 71, 15];
let t = 0;
let audible = 0;
let sumError = 0;
const used = new Set();
for (let k = 0; k < 500; k++) {
  const before = Array.from({ length: EARLY_SLOTS }, (_, i) => ring.gainAt(i, t));
  const { slot } = ring.take(t);
  used.add(slot);
  if (before[slot] > 1e-6) audible++;
  const next = t + gaps[k % gaps.length] / 1000;
  // After the first response has had one fade to rise to its share.
  if (t > FADE_S) {
    for (let s = t; s < next; s += 0.0005) {
      let sum = 0;
      for (let i = 0; i < EARLY_SLOTS; i++) sum += ring.gainAt(i, s);
      sumError = Math.max(sumError, Math.abs(sum - 1));
    }
  }
  t = next;
}
ring.silence(t);
let left = 0;
for (let i = 0; i < EARLY_SLOTS; i++) left = Math.max(left, ring.gainAt(i, t + FADE_S));
console.log(JSON.stringify({ slots: EARLY_SLOTS, used: used.size, audible, sumError, left }));
"""


@needs_node
def test_a_response_never_lands_on_a_slot_that_is_still_sounding(tmp_path: Path) -> None:
    """Every slot used, none given a response while it sounds, gains summing
    to one throughout, and silence reached in one fade -- as the engine is
    configured, under the intervals the app produces."""
    got = _node(tmp_path, RING, str(AUDIO))
    assert got["used"] == got["slots"]
    assert got["audible"] == 0
    # Linear interpolation between curves sampled on different grids: a few
    # thousandths of a decibel, not a fade that loses its way.
    assert got["sumError"] < 1e-3
    assert got["left"] == 0


CELLS = """
const { createLattice } = await import(process.argv[2] + "/lattice.js");
// A row of cells 0.4 m apart along x, with a gap where cells 3 to 8 would be:
// its middle, at 2.2 m, is 1.4 m from the cells either side.
const cells = [0, 1, 2, 9, 10].map((i) => [i, 0, 0]);
const row = createLattice({
  grid_origin_m: [0, 0, 0],
  grid_step_m: [0.4, 0, 0.4],
  grid_shape: [11, 1, 1],
  cell_index: cells,
  positions: cells.map(([i]) => [0.4 * i, 0, 0]),
});
// A corner of one shell further than the middle of a face of the next: at a
// 40 cm lattice, the cell four across and four up is 2.26 m away, the cell
// five across 2.0 m, and the nearest within the 2.2 m hold is the second.
const corners = [[4, 0, 4], [5, 0, 0]];
const shells = createLattice({
  grid_origin_m: [0, 0, 0],
  grid_step_m: [0.4, 0, 0.4],
  grid_shape: [11, 1, 11],
  cell_index: corners,
  positions: corners.map(([i, , k]) => [0.4 * i, 0, 0.4 * k]),
});
console.log(JSON.stringify({
  onACell: row.cellAt(0.8, 0, 0, 1.1),
  inTheGapNearCell2: row.cellAt(1.3, 0, 0, 1.1),
  midGapWithinReach: row.cellAt(2.2, 0, 0, 1.1),
  midGapWithinHold: row.cellAt(2.2, 0, 0, 2.2),
  acrossShells: shells.cellAt(0, 0, 0, 2.2),
}));
"""


@needs_node
def test_the_cell_a_listener_is_given(tmp_path: Path) -> None:
    """The nearest held cell within reach, and a longer reach to keep it."""
    got = _node(tmp_path, CELLS, str(AUDIO))
    assert got["onACell"] == 2
    assert got["inTheGapNearCell2"] == 2
    # 1.4 m from the nearest cell: out of reach to be given one, within the
    # distance a listener who already had one keeps it, so the edge of the
    # solved air cannot chatter.
    assert got["midGapWithinReach"] is None
    assert got["midGapWithinHold"] is not None
    # The nearest cell, not the first shell's: a corner of shell four at
    # 2.26 m lost to a face of shell five at 2.0 m, which is within the hold.
    assert got["acrossShells"] == 1


TAIL = """
const { tailWait } = await import(process.argv[2] + "/spatial.js");
const origin = { x: 0, y: 0, z: 0 };
const tail = { position: 0, at: origin, fetchedAt: 0 };
const rooms = ["a", "a", "b"];
const at = (x) => ({ x, y: 0, z: 0 });
const wait = (seconds) => (Number.isFinite(seconds) ? Number(seconds.toFixed(3)) : "never");
console.log(JSON.stringify({
  first: wait(tailWait({ position: null, at: null, fetchedAt: -Infinity }, origin, 0, rooms, 0)),
  aMetreOn: wait(tailWait(tail, at(1), 1, rooms, 5)),
  threeMetresOn: wait(tailWait(tail, at(3), 1, rooms, 5)),
  threeMetresOnTooSoon: wait(tailWait(tail, at(3), 1, rooms, 0.3)),
  intoAnotherRoom: wait(tailWait(tail, at(0.5), 2, rooms, 5)),
}));
"""


@needs_node
def test_when_the_late_part_moves(tmp_path: Path) -> None:
    """Every two metres or at a new room, and never sooner than 0.7 s. A move
    that is due but too soon says when it will be due, so the page can come
    back for it: a listener who steps through a door and stops sends no more
    poses to trigger it."""
    assert _node(tmp_path, TAIL, str(AUDIO)) == {
        "first": 0,
        "aMetreOn": "never",
        "threeMetresOn": 0,
        "threeMetresOnTooSoon": 0.4,
        "intoAnotherRoom": 0,
    }


RAMP = """
const { propagationRamp } = await import(process.argv[2] + "/engine.js");
const c = 343.2;
console.log(JSON.stringify({
  aStep: propagationRamp(1 / c, 1.05 / c, 0),
  aStepFromAnOldPose: propagationRamp(1 / c, 1.05 / c, 0.05),
  acrossTheRoom: propagationRamp(1 / c, 4 / c, 0),
}));
"""


@needs_node
def test_how_the_delay_line_ramps(tmp_path: Path) -> None:
    """A ramp lands a constant after its pose was taken, so an older pose is
    given a shorter one and the delay slides at the walking speed; a change
    nobody could have walked is a jump."""
    got = _node(tmp_path, RAMP, str(AUDIO))
    assert got["aStep"]["seconds"] > got["aStepFromAnOldPose"]["seconds"] > 0
    assert got["acrossTheRoom"] == {"jump": True}


HARNESS_SCRIPT = """
const harness = await import(process.argv[2]);
const { rows } = await harness.run({
  fieldDirectory: process.argv[3],
  decoderDirectory: process.argv[4],
  decoderName: "measured",
  seconds: 2,
  cases: { walk: harness.CASES.walk },
  variants: harness.VARIANTS,
});
console.log(JSON.stringify(Object.fromEntries(rows.map((row) => [row.variant, row]))));
"""


@needs_node
def test_the_harness_reads_zero_standing_still_and_each_decision_pays(tmp_path: Path) -> None:
    """On a mock field, so only with room to spare: a test that turns on a few
    per cent fails on a busy laptop rather than on a regression. The worklet
    is not among them, because a mock field's early response is a handful of
    mirror images that crossfade cleanly on any convolver; what it is worth
    is pinned in ``test_partitioned_js.py``."""
    field = mock_field(
        tmp_path / "field.h5",
        source_id="S1",
        source_position=(1.0, 1.2, 1.0),
        box_lo=(0.0, 0.0, 0.0),
        box_hi=(6.0, 2.6, 4.0),
        order=3,
        step_m=0.4,
        total_s=0.4,
    )
    build_site(field, tmp_path / "site")
    head = _Head().a_file(tmp_path)
    export_decoders(tmp_path / "decoders", measured_path=head, order=3, filter_length=64)
    rows = _node(
        tmp_path, HARNESS_SCRIPT, str(HARNESS), str(tmp_path / "site"), str(tmp_path / "decoders")
    )
    # Every variant runs, so none of those the ADR's table is drawn from rots.
    assert set(rows) >= {"shipped", "frozen", "before", "lattice"}
    frozen = rows["frozen"]
    assert frozen["levelJerkDb"] == 0
    assert frozen["timbreStepDb"] == 0
    assert frozen["itdJerkSamples"] == 0
    shipped = rows["shipped"]
    # The delay line stops the arrival stepping from cell to cell.
    assert shipped["arrivalJerkSamples"] * 10 < rows["noDelayLine"]["arrivalJerkSamples"]
    # The late part by the two metres stops the reverberation restarting.
    assert shipped["reverbJerkDb"] * 4 < rows["tailEveryCell"]["reverbJerkDb"]
    # And all of it together, against the page before.
    assert shipped["levelJerkDb"] * 5 < rows["before"]["levelJerkDb"]
