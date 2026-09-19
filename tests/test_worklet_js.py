"""The early convolver's two shells, through node with a stand-in Web Audio.

`audio/engine.js` loads the `AudioWorklet`, and a page that cannot -- served
over plain http to another machine, where there is no `AudioWorklet` at all,
or with a module that fails to load -- must say so rather than play the late
part alone in silence. `audio/early.worklet.js` runs on the audio thread,
where a thrown error stops the processor for good and every allocation is a
collection pause waiting to happen. Neither needs a browser to be checked:
the few calls they make of Web Audio are stood in for here.

Skipped where node is not installed, as the other JavaScript tests are.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

AUDIO = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app" / "audio"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _node(tmp_path: Path, script: str) -> Any:
    path = tmp_path / "script.mjs"
    path.write_text(script)
    out = subprocess.run(
        ["node", str(path), str(AUDIO)], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


ENGINE = """
const param = () => ({
  value: 0,
  cancelScheduledValues() {},
  setValueCurveAtTime() {},
  setValueAtTime() {},
  linearRampToValueAtTime() {},
});
const node = (extra = {}) => ({ connect() {}, ...extra });
function standIn(audioWorklet) {
  globalThis.AudioContext = class {
    constructor() {
      Object.assign(this, { state: "running", currentTime: 0, sampleRate: 48000, destination: {} });
      this.audioWorklet = audioWorklet;
    }
    createGain() { return node({ gain: param() }); }
    createDelay() { return node({ delayTime: param() }); }
    createConvolver() { return node({}); }
  };
}
globalThis.location = { origin: "http://192.168.1.2:8765" };
const { createEngine } = await import(process.argv[2] + "/engine.js");
const heard = async (audioWorklet) => {
  standIn(audioWorklet);
  const errors = [];
  const engine = createEngine({ onError: (message) => errors.push(message) });
  engine.setVolume("S1", 1);
  engine.setEarly("S1", { partitions: 1, re: new Float32Array(258), im: new Float32Array(258) });
  await new Promise((resolve) => setTimeout(resolve, 20));
  return errors;
};
console.log(JSON.stringify({
  insecure: await heard(undefined),
  failing: await heard({ addModule: () => Promise.reject(new Error("module failed")) }),
}));
"""


@needs_node
def test_a_page_that_cannot_load_the_early_convolver_says_so(tmp_path: Path) -> None:
    got = _node(tmp_path, ENGINE)
    [insecure] = got["insecure"]
    assert "https or localhost" in insecure
    assert got["failing"] == ["early convolver: module failed"]


WORKLET = """
const posted = [];
let registered = null;
globalThis.sampleRate = 48000;
globalThis.currentTime = 0;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    const postMessage = (message, transfer = []) =>
      posted.push({ type: message.type, transferred: transfer.length });
    this.port = { postMessage };
  }
};
globalThis.registerProcessor = (name, processor) => { registered = processor; };
await import(process.argv[2] + "/early.worklet.js");
const { partitionFilter } = await import(process.argv[2] + "/partitioned.js");
const ear = () => new Float32Array(300).fill(0.1);
const filter = () => partitionFilter([ear(), ear()]);
const ears = (size) => [new Float32Array(size), new Float32Array(size)];

// Two filters before a block: the newer is used, both are given back.
const processor = new registered({ processorOptions: { slots: 6, fade: 0.12 } });
processor.port.onmessage({ data: { type: "filter", filter: filter() } });
processor.port.onmessage({ data: { type: "filter", filter: filter() } });
const input = [[new Float32Array(128).fill(1)]];
let alive = processor.process(input, [ears(128)]);
const spent = posted.filter((p) => p.type === "spent").map((p) => p.transferred);
let sounding = 0;
for (let b = 0; b < 20; b++) {
  const out = [ears(128)];
  globalThis.currentTime = (b + 1) * 128 / 48000;
  alive = processor.process(input, out) && alive;
  sounding = Math.max(sounding, Math.abs(out[0][0][127]));
}

// A block of another size: said once, silence, and still alive.
posted.length = 0;
const odd = new registered({ processorOptions: { slots: 6, fade: 0.12 } });
const first = odd.process([[new Float32Array(256)]], [ears(256)]);
const second = odd.process([[new Float32Array(256)]], [ears(256)]);
const errors = posted.filter((p) => p.type === "error").length;
console.log(JSON.stringify({ alive, spent, sounding, odd: { alive: first && second, errors } }));
"""


@needs_node
def test_the_worklet_takes_the_newest_filter_gives_buffers_back_and_never_dies(
    tmp_path: Path,
) -> None:
    got = _node(tmp_path, WORKLET)
    assert got["alive"] is True
    # Both filters' two buffers went back to the thread that made them: the
    # first when the second replaced it, the second once copied in.
    assert got["spent"] == [2, 2]
    assert got["sounding"] > 0
    assert got["odd"] == {"alive": True, "errors": 1}
