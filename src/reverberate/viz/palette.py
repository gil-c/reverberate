"""Shared, dataset agnostic colour palette for semantic category rendering.

Kept separate from any one viewer so both the MIDI-3D mesh viewer and the
HSSD interior walkthrough (see :mod:`reverberate.viz.interior_walkthrough`)
assign the same category the same colour, deterministically, without either
module depending on the other.
"""

from __future__ import annotations

__all__ = ["color_for_category", "muted_color_for_category"]

#: Fixed, deterministic colour palette (Tableau 20, hex), so the same
#: category always gets the same colour across renders and reports.
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]


def color_for_category(category: str, known_categories: list[str]) -> str:
    """Return a stable hex colour for ``category``.

    ``known_categories`` must be the same sorted list every time the same
    scene is rendered, so a category keeps its colour across mode toggles
    and re-renders within one session.
    """
    index = known_categories.index(category)
    return _PALETTE[index % len(_PALETTE)]


def muted_color_for_category(category: str, known_categories: list[str]) -> str:
    """A desaturated variant of :func:`color_for_category`, blended 55%
    towards white.

    Used for the "colour" render mode, where flat vivid category colours
    would look nothing like a real room. HSSD's furniture textures cannot be
    reliably reduced to one representative colour (many use small palette
    atlases where a whole-texture average is dominated by unrelated
    background pixels, verified empirically), so this is a deliberate,
    documented approximation rather than a real material colour: same hue
    per category as the label view, muted towards a plausible interior
    tone, not photorealistic.
    """
    hex_color = color_for_category(category, known_categories)
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    blend = 0.55
    r = round(r + (255 - r) * blend)
    g = round(g + (255 - g) * blend)
    b = round(b + (255 - b) * blend)
    return f"#{r:02x}{g:02x}{b:02x}"
