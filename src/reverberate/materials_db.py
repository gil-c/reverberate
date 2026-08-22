"""Absorption coefficients for every semantic category in the dataset.

What this replaces: a hand-written table of **23 entries against HSSD's 409
condensed categories**, with everything else falling through to a random draw
from four materials. That is how pillows ended up reflective. Randomising the
absorption of 95 % of categories is not augmentation, it is noise injected into
the one input a model about absorption most depends on.

The data lives in two checked-in CSV files next to this module rather than in
Python, because it is authored knowledge that should be reviewable and
diffable on its own terms:

- ``acoustic_classes.csv``: the materials, with a coefficient per octave band
  from 125 Hz to 8 kHz, a scattering coefficient, and the provenance of each
  row. ``provenance`` records whether the values were taken from a published
  measurement, which sources agreed, or whether the row was interpolated from
  a neighbouring material by reasoning about what the object is made of.
- ``category_materials.csv``: every one of the 409 categories mapped to a
  class, marked ``curated`` where the mapping was made by hand and ``derived``
  where it follows from the category name. All 409 are covered, so nothing is
  randomised and nothing is silently dropped.

**Unmatched labels are reported, never randomised.** A category with no entry
raises, because it is a gap in a checked-in file that someone can fix, not
noise to paper over at runtime.

**A note on 8 kHz.** Published absorption tables almost universally stop at
4 kHz. The 8 kHz column is therefore carried over from 4 kHz rather than
invented: absorption curves are flat or gently falling by then for the porous
materials that dominate, and repeating the last measured value is the
assumption that adds least. It is recorded here rather than hidden because it
is the weakest row of data in the file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

from reverberate.acoustics import OCTAVE_BANDS

DATA_DIR = Path(__file__).parent / "data"
CLASSES_FILE = DATA_DIR / "acoustic_classes.csv"
CATEGORIES_FILE = DATA_DIR / "category_materials.csv"


@dataclass(frozen=True)
class AcousticClass:
    """One material, with its coefficients and where they came from."""

    name: str
    absorption: tuple[float, ...]
    scattering: float
    provenance: str
    notes: str

    @property
    def measured(self) -> bool:
        """Whether these numbers come from a published measurement.

        The distinction is worth keeping visible: an interpolated row is a
        judgement about what an object is made of, and a reader should be able
        to tell it from a measurement without reading the notes.
        """
        return self.provenance.startswith("measured")

    @property
    def cross_checked(self) -> bool:
        """Whether more than one independent source agreed on these values."""
        return "crosscheck:" in self.provenance

    def material(self) -> pra.Material:
        """The pyroomacoustics material, on this project's octave bands."""
        return pra.Material(
            energy_absorption={
                "coeffs": list(self.absorption),
                "center_freqs": list(OCTAVE_BANDS),
            },
            scattering={
                "coeffs": [self.scattering] * len(OCTAVE_BANDS),
                "center_freqs": list(OCTAVE_BANDS),
            },
        )


@dataclass(frozen=True)
class CategoryAssignment:
    """Which class a semantic category was given, and how that was decided."""

    category: str
    material_class: str
    assignment: str
    objects: int

    @property
    def curated(self) -> bool:
        return self.assignment == "curated"


@lru_cache(maxsize=1)
def acoustic_classes() -> dict[str, AcousticClass]:
    """Every material class, keyed by name."""
    classes: dict[str, AcousticClass] = {}
    with CLASSES_FILE.open() as handle:
        for row in csv.DictReader(handle):
            name = row["material_class"]
            classes[name] = AcousticClass(
                name=name,
                absorption=tuple(float(row[f"a{band}"]) for band in OCTAVE_BANDS),
                scattering=float(row["scattering"]),
                provenance=row["provenance"],
                notes=row["notes"],
            )
    return classes


@lru_cache(maxsize=1)
def category_assignments() -> dict[str, CategoryAssignment]:
    """Every semantic category, keyed by its exact condensed name."""
    assignments: dict[str, CategoryAssignment] = {}
    with CATEGORIES_FILE.open() as handle:
        for row in csv.DictReader(handle):
            category = row["hssd_category"]
            assignments[category] = CategoryAssignment(
                category=category,
                material_class=row["material_class"],
                assignment=row["assignment"],
                objects=int(row["objects"]),
            )
    return assignments


def normalise(label: str) -> str:
    """Category names as the mapping file spells them."""
    return label.strip().lower().replace(" ", "_")


class UnknownCategoryError(KeyError):
    """A semantic category with no entry in the mapping file.

    Raised rather than silently defaulting, because the fix belongs in the
    checked-in file where it can be reviewed, not in a fallback that would make
    the gap invisible.
    """


def class_for_category(label: str | None) -> AcousticClass:
    """The material class for one semantic category, by exact lookup.

    Exact rather than substring: the old loose match was wrong in both
    directions, matching "bed" against a bedside table and "clock" against a
    wall clock's mounting.
    """
    if not label:
        raise UnknownCategoryError("no semantic category given")
    assignment = category_assignments().get(normalise(label))
    if assignment is None:
        raise UnknownCategoryError(
            f"{label!r} is not in {CATEGORIES_FILE.name}; add it there rather than "
            "guessing at runtime"
        )
    return acoustic_classes()[assignment.material_class]


def material_for_category(label: str | None) -> pra.Material:
    """The pyroomacoustics material for one semantic category."""
    return class_for_category(label).material()


def absorption_for_category(label: str | None) -> np.ndarray:
    """The per-band absorption coefficients for one semantic category."""
    return np.asarray(class_for_category(label).absorption, dtype=float)


def coverage() -> dict[str, int]:
    """How the mapping was arrived at, for the dataset report.

    Reported rather than assumed: "409 categories, 174 curated" is a claim a
    reader can weigh, and it makes it obvious when a new dataset drops in
    categories nobody has looked at.
    """
    assignments = category_assignments().values()
    return {
        "categories": len(list(assignments)),
        "curated": sum(1 for a in assignments if a.curated),
        "derived": sum(1 for a in assignments if not a.curated),
        "objects": sum(a.objects for a in assignments),
        "classes": len(acoustic_classes()),
    }
