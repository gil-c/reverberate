"""The page's spherical harmonics against the library's, through node.

`viz/app/audio/sh.js` re-implements `real_sh` and the rotation. The library is
the oracle: harmonics at random directions must agree to rounding, and the
general rotation for a pure yaw must agree with `rotate_yaw`. The head matrix
is checked the only way a convention can be: a direction on the camera's left
must land on the head frame's own left. Skipped where node is not installed;
it is on the laptop the app is developed on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reverberate.spatial.sh import real_sh, rotate_yaw

APP = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app"

SCRIPT = """
import fs from "node:fs";
const sh = await import(process.argv[2]);
const f = JSON.parse(fs.readFileSync(process.argv[3]));
const worst = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const apply = (blocks, x) => {
  const out = new Float64Array(x.length);
  for (const { offset, size, matrix } of blocks) {
    for (let i = 0; i < size; i++) {
      let sum = 0;
      for (let j = 0; j < size; j++) sum += matrix[i * size + j] * x[offset + j];
      out[offset + i] = sum;
    }
  }
  return out;
};
let sh_err = 0;
f.directions.forEach((d, i) => {
  sh_err = Math.max(sh_err, worst(Array.from(sh.realSH(f.order, ...d)), f.sh[i]));
});
const c = Math.cos(-f.yaw), s = Math.sin(-f.yaw);
const blocks = sh.rotationBlocks(f.order, [c, -s, 0, s, c, 0, 0, 0, 1]);
const yaw_err = worst(Array.from(apply(blocks, f.coefficients)), f.rotated);
// A direction on the camera's left, in the ambisonic frame (x, -z, y) of the
// scene, brought into the head's frame, must point along +y: the head's left.
// `f'(d) = f(M d)` maps head directions to the world, so the world comes
// back through the transpose.
const M = sh.headMatrix(f.camera_yaw, 0);
const v = f.left_ambisonic;
const head = [0, 1, 2].map((i) => M[i] * v[0] + M[3 + i] * v[1] + M[6 + i] * v[2]);
console.log(JSON.stringify({ sh_err, yaw_err, head }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_pages_harmonics_and_rotation_match_the_library(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    directions = rng.standard_normal((6, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coefficients = rng.standard_normal(64)
    camera_yaw = 0.7
    # viewport.js: forward is (-sin yaw, 0, -cos yaw); left is up cross forward.
    forward = np.array([-np.sin(camera_yaw), 0.0, -np.cos(camera_yaw)])
    left = np.cross([0.0, 1.0, 0.0], forward)
    fixture = {
        "camera_yaw": camera_yaw,
        "left_ambisonic": [left[0], -left[2], left[1]],
        "order": 7,
        "directions": directions.tolist(),
        "sh": real_sh(7, directions).tolist(),
        "coefficients": coefficients.tolist(),
        "yaw": 0.7,
        "rotated": rotate_yaw(coefficients, 0.7, 7).tolist(),
    }
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))
    (tmp_path / "check.mjs").write_text(SCRIPT)
    result = subprocess.run(
        [
            "node",
            str(tmp_path / "check.mjs"),
            str(APP / "audio" / "sh.js"),
            str(tmp_path / "fixture.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["sh_err"] < 1e-12
    assert out["yaw_err"] < 1e-12
    assert np.allclose(out["head"], [0.0, 1.0, 0.0], atol=1e-12)
