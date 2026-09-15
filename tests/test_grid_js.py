"""The page's admission of fine grid tiles, through node.

`viz/app/grid.js::admit` decides which tiles of the solver's grid are drawn at
the grid's own step: those within reach of the listener, nearest first, and
never more quads than the budget. Skipped where node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app"

SCRIPT = """
const { admit } = await import(process.argv[2]);
const at = { getComponent: (axis) => [0, 0, 0][axis] };
const tile = (x, quads) => ({ file: { bounds: [[x, 0, 0], [x + 1, 1, 1]], quads } });
const tiles = [tile(9, 1), tile(0, 5), tile(1.5, 5), tile(2.5, 5)];
const near = admit(tiles, at, 3, 100);
const capped = admit(tiles, at, 3, 8);
const alone = admit([tile(0, 50)], at, 3, 8);
console.log(JSON.stringify({
  near: [...near.wanted].sort(), nearQuads: near.quads,
  capped: [...capped.wanted].sort(), aloneQuads: alone.quads,
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tiles_are_admitted_by_reach_then_budget(tmp_path: Path) -> None:
    (tmp_path / "check.mjs").write_text(SCRIPT)
    result = subprocess.run(
        ["node", str(tmp_path / "check.mjs"), str(APP / "grid.js")],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["near"] == [1, 2, 3], "the tile 9 m away is out of reach"
    assert out["nearQuads"] == 15
    assert out["capped"] == [1], "the budget stops at the first tile that would break it"
    assert out["aloneQuads"] == 50, "the nearest tile is always drawn, budget or not"
