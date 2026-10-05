"""What a pack was computed on, as the page turns it into pictures, through node.

The page computes nothing of a grid, a facet or a ray: it draws what
``viz/computed_api.py`` sends. What it does own is held here: an answer's
arrays are read back in place, a slice is a pixel a node with one colour a
material (the walls' colour), merged squares keep their area, a ray's legs
carry the colour of the energy left, and an image path is drawn through the
corners the server gave and through nothing else. ``tests/js/computed.mjs``
runs them.

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
APP = ROOT / "src" / "reverberate" / "viz" / "app"
HARNESS = ROOT / "tests" / "js" / "computed.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def out() -> Any:
    done = subprocess.run(
        ["node", str(HARNESS), str(APP)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_an_answer_s_arrays_are_read_back_in_place(out: Any) -> None:
    held = out["unpack"]
    assert held["classes"] == [1, 0, 5]
    assert held["corners"] == [1.5, -2.25]
    assert held["material"] == [-1, 7]
    assert held["refused"] is True  # a type the page does not know is not guessed


def test_a_slice_is_a_pixel_a_node_and_a_colour_a_material(out: Any) -> None:
    held = out["slice"]
    assert held["length"] == 4 * 6 and held["opaque"] is True
    assert held["notReached"] != held["air"]
    assert held["sameMaterialSameColour"] and held["otherMaterialOtherColour"]
    assert held["isTheWallsColour"] is True
    assert held["census"] == [[0, 1], [1, 1], [2, 1], [7, 2], [8, 1]]
    # A pixel is a node's own cell: the picture overhangs the end nodes by half a step.
    assert held["rect"] == {"x0": -2.25, "z0": 9.75, "x1": 0.75, "z1": 11.75}
    assert held["diff"] == held["diffNames"]


def test_merged_squares_keep_their_area_and_their_material(out: Any) -> None:
    held = out["quads"]
    assert held["index"] == [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
    assert held["area"] == pytest.approx(7.0)
    assert held["rigid"] == [250, 250, 250]
    assert held["everyCornerOneColour"] is True
    assert held["triangleColours"] == [0, 0, 1] * 3 + [1, 0, 0] * 3
    assert all(out["facets"].values())  # a doubled sheet is marked, apart from a two-sided facet


def test_a_ray_s_legs_carry_the_energy_left_and_a_path_its_corners(out: Any) -> None:
    rays = out["rays"]
    assert rays["legs"] == 3  # a leg between two vertices of one ray, none from ray to ray
    assert rays["positions"] == [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2]
    assert rays["firstLegFull"] and rays["secondLegDimmer"] and rays["dimmerIsDarker"]
    paths = out["paths"]
    assert paths["segments"] == 6 and paths["orders"] == [0, 2, 2, 2, 1, 1]
    assert paths["missing"] == 1  # a row the server could not find again is not invented
    assert [c["facet"] for c in paths["corners"]] == [7, 3, 7]
    assert paths["hit"] == [[7, 2], [3, 1]]
    assert out["decay"] == [[100, 0], [200, 10], [300, 80]]  # dB under the peak, floored


def test_the_words_say_millimetres_links_and_the_method(out: Any) -> None:
    words = out["words"]
    assert words["mm"] == "21.8 mm" and words["key"] == "a6422ab46225…"
    assert words["opening"] == "at x 2.00 z 1.45, across x: A 18 links (900.0 mm), B none"
    assert words["diff"][0] == "air: A 28.8 m³, B 14.2 m³"
    assert "largest wall displacement 40.0 mm" in words["diff"][2]
    assert words["diff"][-1].startswith("method: ") and "resolution 50.0 mm" in words["diff"][-1]
