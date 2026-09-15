"""The page's direction diagram, through node.

`viz/app/audio/analysis.js::polar` reads where energy arrives from out of an
ambisonic covariance. A plane wave encoded from one azimuth must peak there,
and a level change must move the absolute plots by as much. Skipped where
node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app"

SCRIPT = """
const { realSH } = await import(process.argv[2] + "/sh.js");
const { analyse } = await import(process.argv[2] + "/analysis.js");
const order = 7, samples = 2048, azimuth = (2 * Math.PI * 100) / 360;
const gains = realSH(order, Math.cos(azimuth), Math.sin(azimuth), 0);
const wave = (scale) => Array.from(gains, (g) => {
  const c = new Float32Array(samples);
  for (let t = 0; t < 64; t++) c[100 + t] = scale * g * Math.sin(t / 3);
  return c;
});
const loud = analyse(wave(1), order, 512);
const quiet = analyse(wave(0.1), order, 512);
let peak = 0;
loud.polar.whole.forEach((v, k) => { if (v > loud.polar.whole[peak]) peak = k; });
console.log(JSON.stringify({
  peakDeg: (360 * peak) / loud.polar.whole.length,
  energyDrop: loud.energyDb - quiet.energyDb,
  peakDrop: loud.peakDb - quiet.peakDb,
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_direction_diagram_peaks_where_the_wave_comes_from(tmp_path: Path) -> None:
    (tmp_path / "check.mjs").write_text(SCRIPT)
    result = subprocess.run(
        ["node", str(tmp_path / "check.mjs"), str(APP / "audio")],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["peakDeg"] == 100
    assert abs(out["energyDrop"] - 20) < 1e-3, "levels are absolute, not normalised"
    assert abs(out["peakDrop"] - 20) < 1e-3
