"""The walk-through engine's rules, through node.

The ones that are plain functions: which cell a listener is given, and when
the late part of the response moves. Each is something this engine once got
wrong -- a listener fell silent in every hole of the solved air, and the late
part restarted its reverberation at every cell.

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
AUDIO = ROOT / "src" / "reverberate" / "viz" / "app" / "audio"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _node(tmp_path: Path, script: str, *args: str) -> Any:
    path = tmp_path / "script.mjs"
    path.write_text(script)
    out = subprocess.run(["node", str(path), *args], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


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
