"""Semantic label to pyroomacoustics material mapping (roadmap section 5.5).

Two mechanisms, as specified in the brief:

1. ``SEMANTIC_MATERIAL_TABLE``: a hand written lookup from a coarse semantic
   label to an entry in ``pyroomacoustics.materials_absorption_table``.
2. ``random_material_for_label``: for labels not in the table, sample from a
   plausible per category set, seeded and recorded, so the assignment is both
   reproducible and a deliberate source of augmentation.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

# Coarse label -> materials database key. Labels are matched case
# insensitively against HSSD's condensed semantic categories and region
# labels. This is deliberately small: unknown labels fall back to the
# randomised assignment below rather than growing this table indefinitely.
SEMANTIC_MATERIAL_TABLE: dict[str, str] = {
    "wall": "plasterboard",
    "floor": "carpet_cotton",
    "ceiling": "ceiling_fibre_absorber",
    "window": "glass_window",
    "door": "wood_1.6cm",
    "sofa": "curtains_cotton_0.5",
    "couch": "curtains_cotton_0.5",
    "chair": "wood_1.6cm",
    "bed": "curtains_cotton_0.5",
    "table": "wood_1.6cm",
    "desk": "wood_1.6cm",
    "cabinet": "wood_1.6cm",
    "shelf": "wood_1.6cm",
    "bookshelf": "wood_1.6cm",
    "wardrobe": "wood_1.6cm",
    "clock": "hard_surface",
    "wall_clock": "hard_surface",
    "mirror": "glass_3mm",
    "rug": "carpet_cotton",
    "carpet": "carpet_cotton",
    "curtain": "curtains_cotton_0.5",
    "lamp": "hard_surface",
    "lighting": "hard_surface",
}

# Plausible fallback set for unknown labels, sampled per instance so the
# absorption input is never a constant the model could ignore.
_FALLBACK_MATERIAL_POOL = (
    "hard_surface",
    "wood_1.6cm",
    "rough_concrete",
    "curtains_cotton_0.5",
)


def material_for_label(label: str | None, rng: np.random.Generator) -> pra.Material:
    """Resolve a semantic label to a pyroomacoustics Material.

    Falls back to a seeded random choice from a plausible pool when the
    label is missing or not in the hand written table, per roadmap 5.5.
    """
    key = None
    if label:
        normalised = label.strip().lower().replace(" ", "_")
        key = SEMANTIC_MATERIAL_TABLE.get(normalised)
        if key is None:
            # try a loose substring match, e.g. "wall_clock" -> "clock"
            for table_label, table_key in SEMANTIC_MATERIAL_TABLE.items():
                if table_label in normalised or normalised in table_label:
                    key = table_key
                    break
    if key is None:
        key = rng.choice(_FALLBACK_MATERIAL_POOL)
    return pra.make_materials(key)[0] if isinstance(key, tuple) else pra.Material(key)
