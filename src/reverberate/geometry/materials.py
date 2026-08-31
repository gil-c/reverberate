"""Semantic label to acoustic material (roadmap section 5.5).

This module is now a thin adapter over :mod:`reverberate.materials`, which
holds the real data in checked-in CSV files. It used to carry a 23-entry table
plus a random fallback pool, which meant about 95 % of HSSD's 409 condensed
categories got a coefficient drawn at random from four materials. That is how
pillows ended up reflective, and it is not a defensible input to a model whose
whole subject is absorption.

Two behaviours changed as a result, deliberately:

- **Lookup is exact**, on the condensed category name. The previous substring
  match was wrong in both directions: "bed" matched a bedside table, and any
  label containing "clock" took a clock's material regardless of what it was.
- **An unknown category raises** instead of being randomised. The fix belongs
  in the mapping file where it can be reviewed, not in a runtime fallback that
  hides the gap.

``rng`` is still accepted so existing callers do not all have to change at
once, but it is unused: the assignment is now deterministic, which also means
two runs of the dataset generator agree on what a room is made of.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

from reverberate.materials import (
    UnknownCategoryError,
    class_for_category,
    material_for_category,
)

__all__ = [
    "UnknownCategoryError",
    "class_for_category",
    "material_for_label",
]


def material_for_label(label: str | None, rng: np.random.Generator | None = None) -> pra.Material:
    """Resolve a semantic label to a pyroomacoustics Material.

    Raises :class:`UnknownCategoryError` for a category with no entry, rather
    than inventing one.
    """
    del rng  # deterministic now; see the module docstring
    return material_for_category(label)
