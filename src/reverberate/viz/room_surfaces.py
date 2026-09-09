"""The room shell we reconstruct, and how its surfaces are coloured.

The shell is the one piece of room geometry that is *not* dataset geometry:
HSSD authors a floor polygon per region, and this project extrudes it into a
closed prism (see the roadmap's watertightness breakthrough). It therefore has
no appearance of its own, and every colour here is a convention chosen to make
an interpretation checkable, not a claim about how the room really looks.
"""

from __future__ import annotations

import numpy as np
import trimesh

from reverberate.viz.label_palette import SHELL_LABEL_COLOURS, SHELL_RENDER_COLOURS

__all__ = [
    "SHELL_LABEL_COLOURS",
    "SHELL_RENDER_COLOURS",
    "VERTICAL_NORMAL_THRESHOLD",
    "absorption_colour",
    "shell_surface_labels",
]

#: A face whose outward normal is this close to vertical is floor or ceiling
#: rather than wall. The shell is a prism, so its faces are either exactly
#: vertical or exactly horizontal and the threshold is not delicate.
VERTICAL_NORMAL_THRESHOLD = 0.9


def absorption_colour(mean_absorption: float) -> np.ndarray:
    """Blue (reflective) to red (absorptive), as an RGBA byte colour.

    A deliberately simple two-colour ramp rather than a perceptual colormap:
    the question this view answers is "did this surface get a plausible
    absorption", which only needs an ordered scale.
    """
    fraction = float(np.clip(mean_absorption, 0.0, 1.0))
    return np.array([int(255 * fraction), 40, int(255 * (1.0 - fraction)), 255], dtype=np.uint8)


def shell_surface_labels(shell: trimesh.Trimesh) -> np.ndarray:
    """Label each shell face floor, ceiling or wall from its outward normal.

    The room shell is one extruded prism, so without this split the whole
    enclosure would take a single material. That matters acoustically: carpet
    underfoot and plasterboard overhead are the two ends of the absorption
    range, and averaging them away would flatten the very signal the model is
    meant to learn.
    """
    vertical = shell.face_normals[:, 1]
    labels = np.full(len(shell.faces), "wall", dtype=object)
    labels[vertical > VERTICAL_NORMAL_THRESHOLD] = "ceiling"
    labels[vertical < -VERTICAL_NORMAL_THRESHOLD] = "floor"
    return labels
