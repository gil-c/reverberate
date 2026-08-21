"""Unit tests for the pure parts of the interior walkthrough viewer.

Only ``classify_shell_faces`` is tested here without any mesh/data
dependency; the PyVista/trame scene assembly is exercised only via manual
smoke testing (see the module docstring), since it needs a real HSSD
checkout and a display/off-screen renderer, both out of scope for the fast
offline suite required by ROADMAP.md.
"""

from __future__ import annotations

import numpy as np

from reverberate.viz.interior_walkthrough import classify_shell_faces


def test_classify_shell_faces_identifies_floor_ceiling_and_wall() -> None:
    normals = np.array(
        [
            [0.0, -1.0, 0.0],  # floor: points down
            [0.0, 1.0, 0.0],  # ceiling: points up
            [1.0, 0.0, 0.0],  # wall: horizontal normal
            [0.0, 0.6, 0.6],  # ambiguous but steep enough up: ceiling
        ]
    )
    labels = classify_shell_faces(normals)
    assert list(labels) == ["floor", "ceiling", "wall", "ceiling"]
