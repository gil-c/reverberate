"""The plots' axes, through node.

`viz/app/axes.js` graduates the spectrogram and the decay beside their
canvases: compact labels (250, 1k, 16k; 0, 0.2; -30), a tick at each, and
the unit once at the far end of its axis, where no tick label may sit.
Skipped where node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "reverberate" / "viz" / "app"

SCRIPT = """
const axes = await import(process.argv[2]);
const texts = (marks) => marks.map((m) => m.text);
console.log(JSON.stringify({
  hz: [250, 1000, 1500, 16000, 22050].map(axes.compactHz),
  frequency: texts(axes.frequencyMarks(24000)),
  frequencyAt: axes.frequencyMarks(24000).map((m) => m.at),
  time: texts(axes.timeMarks(1.2)),
  longTime: texts(axes.timeMarks(3.0)),
  level: texts(axes.levelMarks(90)),
  levelAt: axes.levelMarks(90).map((m) => m.at),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_axes_are_compact_and_name_their_unit_once_at_the_end(tmp_path: Path) -> None:
    (tmp_path / "check.mjs").write_text(SCRIPT)
    result = subprocess.run(
        ["node", str(tmp_path / "check.mjs"), str(APP / "axes.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    found = json.loads(result.stdout)
    assert found["hz"] == ["250", "1k", "1.5k", "16k", "22.1k"]
    assert found["frequency"] == ["0", "4k", "8k", "12k", "16k", "20k", "Hz"]
    # Zero at the bottom of the canvas, the unit at the top, where 24k is.
    assert found["frequencyAt"][0] == 1 and found["frequencyAt"][-1] == 0
    assert found["time"] == ["0", "0.2", "0.4", "0.6", "0.8", "1", "s"]
    assert found["longTime"] == ["0", "0.5", "1", "1.5", "2", "2.5", "s"]
    assert found["level"] == ["0", "-30", "-60", "dB"]
    assert found["levelAt"] == [0, 1 / 3, 2 / 3, 1]
