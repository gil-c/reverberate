"""The page's decode, against a direct one and against itself.

The first test decodes a response the slow and obvious way -- rotate the
coefficients sample by sample, convolve each with its decoder filter -- and
asks the page's to agree: it rotates spectra, computes only the half of each
below the Nyquist bin, and starts at the decoder's lead.

The rest: `viz/app/audio/decode.js` decodes a response in two parts, an early part
re-decoded at every head update and a late part decoded rarely, which meet in
a crossfade. ADR 0013 says that at rest their sum is the decode of the whole
response. For a long time it was not: the crossfade was put on the two ears
after the decode, and the binaural decoder spreads every sample over its
taps, centred half way along them, so the fade removed from the early part
field samples from well before the seam that the late part never had. Near
the seam the sum missed by half the response. Found by this test, and
fixed by fading the field itself.

The decoder here is the library's own, exported as the page reads it, so it
is centred as the real one is. Skipped where node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reverberate.viz.decoders import export_decoders

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "src" / "reverberate" / "viz" / "app" / "audio"

DIRECT = """
const { createDecoder } = await import(process.argv[2]);
const { headMatrix, rotationBlocks } = await import(process.argv[3]);
const order = 3, channels = 16, taps = 256, samples = 300;
let s = 7;
const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
const field = Array.from({ length: channels }, () => Float32Array.from({ length: samples }, rnd));
// A decoder centred as the real ones are, so that it has a lead to take off.
const filters = Float32Array.from({ length: 2 * channels * taps }, (_, i) => {
  const tap = i % taps;
  return rnd() * Math.exp(-(((tap - taps / 2) / 10) ** 2));
});
const decoder = createDecoder();
decoder.setDecoder({ order, channels, taps, filters });
decoder.keep({ key: "k", channels: field });
const head = headMatrix(0.7, -0.3);
const got = decoder.renderEarly({ key: "k", head });
const rotated = Array.from({ length: channels }, () => new Float64Array(samples));
for (const { offset, size, matrix } of rotationBlocks(order, head)) {
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const w = matrix[i * size + j];
      for (let t = 0; t < samples; t++) rotated[offset + i][t] += w * field[offset + j][t];
    }
  }
}
let worst = 0, scale = 0;
for (let ear = 0; ear < 2; ear++) {
  const want = new Float64Array(samples + taps - 1);
  for (let ch = 0; ch < channels; ch++) {
    const f = filters.subarray((ear * channels + ch) * taps, (ear * channels + ch + 1) * taps);
    for (let t = 0; t < samples; t++) {
      for (let k = 0; k < taps; k++) want[t + k] += rotated[ch][t] * f[k];
    }
  }
  for (let i = 0; i < got[ear].length; i++) {
    worst = Math.max(worst, Math.abs(got[ear][i] - want[i + decoder.lead]));
    scale = Math.max(scale, Math.abs(want[i + decoder.lead]));
  }
}
console.log(JSON.stringify({ lead: decoder.lead, error: worst / scale }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_decode_is_the_direct_decode_from_the_lead_on(tmp_path: Path) -> None:
    """To single precision, from the decoder's lead to the end."""
    script = tmp_path / "direct.mjs"
    script.write_text(DIRECT)
    out = subprocess.run(
        ["node", str(script), str(AUDIO / "decode.js"), str(AUDIO / "sh.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["lead"] > 0
    assert got["error"] < 1e-5, got


SCRIPT = """
import fs from "node:fs";
const { createDecoder } = await import(process.argv[2]);
const { headMatrix } = await import(process.argv[3]);
const record = JSON.parse(fs.readFileSync(process.argv[4] + ".json"));
const bytes = fs.readFileSync(process.argv[4] + ".f32");
const filters = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const decoder = { order: record.order, channels: record.channels, taps: record.taps, filters };
const channels = record.channels, samples = 2500, early = 900, fade = 64, start = early - fade;
let s = 3;
const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5);
const field = Array.from({ length: channels }, () =>
  Float32Array.from({ length: samples }, (_, i) => rnd() * Math.exp(-i / 400))
);
const head = headMatrix(0.4, 0.2);
const make = () => { const d = createDecoder(); d.setDecoder(decoder); return d; };
const whole = make();
whole.keep({ key: "w", channels: field });
const all = whole.renderEarly({ key: "w", head });
const e = make();
const head_ = field.map((c) => c.subarray(0, early));
e.keep({ key: "e", channels: head_, fadeOutFrom: start, fadeOut: fade });
const parts = e.renderEarly({ key: "e", head });
const l = make();
l.keep({ key: "l", channels: field.map((c) => c.subarray(start)), fadeIn: fade });
const plan = l.lateSteps({ key: "l", head });
for (const step of plan.steps) step();
// The late part sits `start` after the early part, less what the early part
// had cut from its front: the decoder's lead.
let worst = 0, scale = 0;
for (let ear = 0; ear < 2; ear++) {
  const sum = new Float64Array(all[ear].length);
  for (let k = 0; k < parts[ear].length; k++) sum[k] += parts[ear][k];
  for (let k = 0; k < plan.ears[ear].length; k++) {
    const at = k + start - e.lead;
    if (at >= 0 && at < sum.length) sum[at] += plan.ears[ear][k];
  }
  for (let k = 0; k < all[ear].length - record.taps; k++) {
    worst = Math.max(worst, Math.abs(sum[k] - all[ear][k]));
    scale = Math.max(scale, Math.abs(all[ear][k]));
  }
}
console.log(JSON.stringify({ lead: e.lead, error: worst / scale }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_early_and_late_parts_sum_to_the_whole_response(tmp_path: Path) -> None:
    """Exact to single precision, the decoder's lead taken off both alike."""
    export_decoders(tmp_path / "decoders", order=3, filter_length=512)
    script = tmp_path / "seam.mjs"
    script.write_text(SCRIPT)
    out = subprocess.run(
        [
            "node",
            str(script),
            str(AUDIO / "decode.js"),
            str(AUDIO / "sh.js"),
            str(tmp_path / "decoders" / "sphere"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    # A centred decoder has a lead to take off; if it did not, the check
    # below would not be testing the placement of the late part at all.
    assert got["lead"] > 0
    assert got["error"] < 1e-5, got


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_solver_latency_is_where_the_response_starts(tmp_path: Path) -> None:
    """The pulse delay a solver puts before the direct sound, less the guard."""
    script = tmp_path / "latency.mjs"
    script.write_text(
        """
const { solverLatency, LATENCY_GUARD } = await import(process.argv[2]);
const omni = new Float32Array(4000);
// The direct sound 300 samples after its geometric arrival at 1000, a little
// run-up in front of it, then a room.
for (let i = 1280; i < 1300; i++) omni[i] = 0.001;
omni[1300] = 1;
for (let i = 1400; i < 4000; i++) omni[i] = 0.2 * Math.exp(-(i - 1400) / 500) * Math.sin(i);
console.log(JSON.stringify({ latency: solverLatency(omni, 1000), guard: LATENCY_GUARD }));
"""
    )
    out = subprocess.run(
        ["node", str(script), str(AUDIO / "decode.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    # The run-up at a thousandth of the peak is under the 40 dB threshold;
    # the onset is the direct sound itself.
    assert got["latency"] == 300 - got["guard"]
    assert np.isfinite(got["latency"])
