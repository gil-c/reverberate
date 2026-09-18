"""The early convolver of the page, against direct convolution, through node.

`viz/app/audio/partitioned.js` is the arithmetic of the `AudioWorklet` that
convolves the early part of every response: uniformly partitioned
overlap-save, with one history of the input shared by every filter. What
it exists for is the second test here: when a response takes over from
another, the output is exactly the crossfade of the two complete
convolutions, each with the input's whole past. A `ConvolverNode` given a new
buffer starts with no past instead, and that was the crackle it replaced.

Skipped where node is not installed, as the other JavaScript tests are.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

AUDIO = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app" / "audio"

SCRIPT = """
const { createPartitionedConvolver, partitionFilter, BLOCK } = await import(process.argv[2]);
const { createRing } = await import(process.argv[3]);
let s = 5;
const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
const rate = 48000, blocks = 160, n = blocks * BLOCK;
const x = Float32Array.from({ length: n }, rnd);
// Lengths that are not whole blocks, the second longer than the first, so
// the history has to grow when it arrives.
const filter = (length) =>
  [0, 1].map(() => Float32Array.from({ length }, (_, i) => rnd() * Math.exp(-i / 600)));
const direct = (h) => {
  const y = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    let sum = 0;
    for (let k = 0; k < h.length && k <= t; k++) sum += h[k] * x[t - k];
    y[t] = sum;
  }
  return y;
};
const A = filter(900);
const B = filter(1500);
const yA = A.map(direct);
const yB = B.map(direct);

function run(switchAt) {
  const convolver = createPartitionedConvolver({ sampleRate: rate });
  const gains = createRing(6, 0.12);
  const out = [new Float32Array(n), new Float32Array(n)];
  for (let b = 0; b < blocks; b++) {
    const time = (b * BLOCK) / rate;
    if (b === 0) { convolver.setFilter(partitionFilter(A), time); gains.take(time); }
    if (b === switchAt) { convolver.setFilter(partitionFilter(B), time); gains.take(time); }
    const at = (a) => a.subarray(b * BLOCK, (b + 1) * BLOCK);
    convolver.process(at(x), at(out[0]), at(out[1]), time);
  }
  return { out, gains };
}

// One filter: once its first fade in is over, plain linear convolution.
const single = run(-1);
let worst = 0, scale = 0;
for (let ear = 0; ear < 2; ear++) {
  for (let t = Math.ceil(0.13 * rate); t < n; t++) {
    worst = Math.max(worst, Math.abs(single.out[ear][t] - yA[ear][t]));
    scale = Math.max(scale, Math.abs(yA[ear][t]));
  }
}
const plain = worst / scale;

// A switch half way: the two complete convolutions, crossfaded by the ring.
const switched = run(80);
worst = 0; scale = 0;
for (let ear = 0; ear < 2; ear++) {
  for (let t = Math.ceil(0.13 * rate); t < n; t++) {
    const g = switched.gains;
    const want = g.gainAt(0, t / rate) * yA[ear][t] + g.gainAt(1, t / rate) * yB[ear][t];
    worst = Math.max(worst, Math.abs(switched.out[ear][t] - want));
    scale = Math.max(scale, Math.abs(want));
  }
}
console.log(JSON.stringify({ plain, crossfade: worst / scale }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_early_convolver_is_linear_convolution_and_switches_with_its_past(
    tmp_path: Path,
) -> None:
    """Exact to single precision, before and across a change of filter."""
    script = tmp_path / "partitioned.mjs"
    script.write_text(SCRIPT)
    out = subprocess.run(
        ["node", str(script), str(AUDIO / "partitioned.js"), str(AUDIO / "ring.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["plain"] < 1e-5, got
    # The whole point: the incoming filter is applied to the input's past, so
    # the switch is the ideal crossfade and nothing is switched on.
    assert got["crossfade"] < 1e-4, got
