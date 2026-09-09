"""The room rules against the real scenes, so a nudged threshold cannot pass.

Marked ``slow`` and skipped without the HSSD download, so a checkout with no
data still runs the suite. `make test-slow` runs them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reverberate.experiments.small_objects import isolated_storey
from reverberate.geometry.apartment import build_apartment, extrude_storey
from reverberate.geometry.rooms import room_holding, rooms_of_scene

HSSD_ROOT = Path("/Users/gilles/Developer/reverberate/data/raw/hssd-hab")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HSSD_ROOT.exists(), reason="the HSSD download is not on this machine"),
]


@pytest.mark.parametrize(
    ("scene_id", "regions", "rooms"),
    [("102344403", 20, 15), ("102344022", 19, 10), ("102344094", 10, 4)],
)
def test_the_rules_make_these_rooms_of_these_regions(
    scene_id: str, regions: int, rooms: int
) -> None:
    storey = build_apartment(HSSD_ROOT, scene_id)[0]

    assert (len(storey.rooms), len(storey.everyday)) == (regions, rooms)


def test_the_room_a_run_receives_is_the_whole_open_space() -> None:
    """`w39_living_8k` sealed 296.1 m3: the `living room` region of `102344403`
    alone, with a rigid wall across the 6.97 m opening onto its kitchen."""
    storey = build_apartment(HSSD_ROOT, "102344403")[0]

    shell = extrude_storey(isolated_storey(storey, "living room"))

    assert shell.is_watertight
    assert shell.volume == pytest.approx(461.8, abs=2.0)


def test_a_wall_with_a_wide_opening_still_separates() -> None:
    """`lounge` opens onto `living room` by 1.43 m -- wider than a door -- but
    the boundary carries 3.35 m of wall. The counterpart to the merge above:
    without this the rules would swallow the whole floor."""
    rooms = rooms_of_scene(HSSD_ROOT, "102344403")

    assert room_holding(rooms, "lounge").regions == ("lounge",)
