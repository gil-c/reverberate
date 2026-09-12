"""The exported decoder is the library's own, laid out as the page reads it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reverberate.viz.decoders import export_decoders


def test_the_sphere_decoder_is_exported_ear_channel_tap(tmp_path: Path) -> None:
    [record] = export_decoders(tmp_path, order=1, filter_length=64)
    taps = np.fromfile(tmp_path / "sphere.f32", dtype="<f4")

    assert record["ears"] == 2 and record["channels"] == 4 and record["taps"] == 64
    assert taps.size == 2 * 4 * 64
    filters = taps.reshape(2, 4, 64)
    # The W channel reaches both ears alike; the Y channel (left-right) does not.
    assert np.allclose(filters[0, 0], filters[1, 0], atol=1e-6)
    assert not np.allclose(filters[0, 1], filters[1, 1], atol=1e-3)
    listed = json.loads((tmp_path / "decoders.json").read_text())
    assert [entry["name"] for entry in listed] == ["sphere"]
