"""The diffracted onset: round a wall through its doorway, later and duller than the line.

A box cut in two by a wall with a doorway: the receiver behind the wall has
no line of sight to the source. The onset must be found, pass through the
doorway (longer than the straight line, no shorter than the path by the
door's edge), lose more at high frequencies than at low, and arrive from the
doorway's side.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from reverberate.mirror.diffract import DiffractionSettings, diffracted_paths, maekawa_db
from test_mirror_ism import box_scene

# The wall at x = 2 spans the box but for a doorway at 2.0 <= z <= 2.5 (the
# box is 4 x 3 x 2.5), from the floor to y = 2.1.
WALL_X = 2.0
DOOR = (2.0, 2.5, 2.1)


def _rect(x: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    a = [x, y0, z0]
    b = [x, y1, z0]
    c = [x, y1, z1]
    d = [x, y0, z1]
    return np.asarray([[a, b, c], [a, c, d]], dtype=float)


def walled_box() -> Any:
    scene = box_scene(alpha=0.2)
    z0, z1, top = DOOR
    wall = np.concatenate(
        [
            _rect(WALL_X, 0.0, 3.0, 0.0, z0),  # beside the doorway
            _rect(WALL_X, top, 3.0, z0, z1),  # above it
        ]
    )
    occluders = np.concatenate([scene.occluder_vertices, wall])
    labels = np.concatenate([scene.occluder_label, np.zeros(len(wall), dtype=np.int16)])
    return replace(scene, occluder_vertices=occluders, occluder_label=labels)


def test_maekawa_is_the_shadow_boundary_at_no_detour_and_grows_with_frequency() -> None:
    bands = np.array([125.0, 1000.0, 8000.0])
    at_edge = maekawa_db(0.0, bands, 343.0, 24.0)
    np.testing.assert_allclose(at_edge, 10 * np.log10(3.0))
    deep = maekawa_db(0.5, bands, 343.0, 24.0)
    assert deep[0] < deep[1] < deep[2] <= 24.0


@pytest.mark.parametrize("cell_m", [0.1])
def test_the_onset_goes_through_the_doorway(cell_m: float) -> None:
    scene = walled_box()
    source = np.array([1.0, 1.2, 0.8])
    receivers = np.array([[3.0, 1.2, 0.8], [1.5, 1.2, 1.0]])
    onsets, record = diffracted_paths(
        scene,
        source,
        receivers,
        [0],
        sound_speed_m_s=343.2,
        settings=DiffractionSettings(cell_m=cell_m),
    )
    assert record["found"] == 1 and 0 in onsets and 1 not in onsets
    path = onsets[0]
    straight = float(np.linalg.norm(receivers[0] - source))
    # Round the door's near jamb at z = 2.0: the shortest path in the plane y = 1.2.
    jamb = np.array([WALL_X, 1.2, DOOR[0]])
    by_jamb = float(np.linalg.norm(jamb - source) + np.linalg.norm(receivers[0] - jamb))
    assert path.length_m[0] > straight + 0.5
    assert by_jamb - 0.05 <= path.length_m[0] <= by_jamb + 0.4
    loss = -20.0 * np.log10(path.gain[0] * path.length_m[0])
    assert np.all(np.diff(loss) >= -1e-9) and loss[0] > 4.0 and loss[-1] > loss[0] + 6.0
    # It arrives from the doorway's side: towards +z and back towards the wall.
    assert path.direction[0, 2] > 0.5 and path.direction[0, 0] < 0.0
