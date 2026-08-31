"""The material catalogue: absorption per class, and the solver's impedance.

Section 6.1 of the roadmap describes two faces of one source. Face A is
absorption per octave band, which the metrics, the tail model and validation
consume, and lives in :mod:`~reverberate.materials.db`. Face B is the
impedance filter the wave solver consumes, which is fitted from face A by
:mod:`~reverberate.materials.impedance`. Between them sits
:mod:`~reverberate.materials.extrapolation`, which is where the two octaves
above the last published measurement come from.
"""

from __future__ import annotations

from reverberate.materials.db import (
    AcousticClass,
    CategoryAssignment,
    UnknownCategoryError,
    absorption_for_category,
    acoustic_classes,
    category_assignments,
    class_for_category,
    coverage,
    material_for_category,
    solver_absorption_for_category,
)

__all__ = [
    "AcousticClass",
    "CategoryAssignment",
    "UnknownCategoryError",
    "absorption_for_category",
    "acoustic_classes",
    "category_assignments",
    "class_for_category",
    "coverage",
    "material_for_category",
    "solver_absorption_for_category",
]
