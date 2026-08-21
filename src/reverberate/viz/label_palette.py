"""Deterministic colours for the semantic label view.

The label view answers "did this object get the category we think it did",
so the only requirement on a colour is that it is stable for a category and
easy to tell apart from its neighbours. Colours are therefore derived from a
hash of the category name: no table to maintain, no drift between runs, and
a category that appears in two rooms looks the same in both.
"""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np

#: Fixed tones for the parts of the room shell we synthesise ourselves.
#: These are *not* measured or textured surfaces, they are a legible
#: convention, and the room shell is our own extrusion rather than dataset
#: geometry, so no claim of realism is made or implied here.
SHELL_LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "floor": (150, 110, 70),
    "wall": (215, 210, 200),
    "ceiling": (245, 245, 245),
}

#: Plausible interior tones for the same surfaces in the colour view. Also an
#: approximation: HSSD ships no material for a room shell that we extruded
#: ourselves, so these are a stand in, and should be described as such rather
#: than presented as a render of the dataset.
SHELL_RENDER_COLOURS: dict[str, tuple[int, int, int]] = {
    "floor": (140, 105, 75),
    "wall": (225, 222, 214),
    "ceiling": (250, 250, 250),
}


def category_colour(category: str) -> tuple[int, int, int]:
    """A stable, vivid RGB colour for a semantic category name."""
    digest = hashlib.sha1(category.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    saturation = 0.55 + (digest[1] / 255.0) * 0.35
    value = 0.65 + (digest[2] / 255.0) * 0.3
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(red * 255), int(green * 255), int(blue * 255))


def rgba(colour: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    return np.array([*colour, alpha], dtype=np.uint8)
