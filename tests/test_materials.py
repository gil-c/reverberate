"""Tests for the material database.

The defect these encode is the one they replaced: 23 hand-written entries
against 409 semantic categories, with the rest drawn at random from four
materials, which is how a pillow ended up reflective. So the properties under
test are total coverage, exact lookup, loud failure on a gap, and physically
ordered coefficients.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.geometry.materials import material_for_label
from reverberate.materials_db import (
    UnknownCategoryError,
    absorption_for_category,
    acoustic_classes,
    category_assignments,
    class_for_category,
    coverage,
)


def test_every_semantic_category_in_the_dataset_is_covered() -> None:
    """95 % of categories used to fall through to a random draw."""
    report = coverage()

    assert report["categories"] == len(category_assignments())
    assert report["categories"] >= 409
    assert report["curated"] + report["derived"] == report["categories"]
    assert all(a.material_class for a in category_assignments().values())


def test_every_category_points_at_a_class_that_exists() -> None:
    classes = acoustic_classes()

    missing = {
        a.material_class for a in category_assignments().values() if a.material_class not in classes
    }

    assert missing == set()


def test_coefficients_are_physical_and_cover_every_band() -> None:
    for material in acoustic_classes().values():
        assert len(material.absorption) == len(OCTAVE_BANDS)
        assert all(0.0 <= value <= 1.0 for value in material.absorption)
        assert 0.0 <= material.scattering <= 1.0


def test_every_class_agrees_on_band_count() -> None:
    """A single 6-band material once made a whole apartment fail to build with
    "All walls should have the same number of frequency bands"."""
    counts = {
        len(material.material().energy_absorption["coeffs"])
        for material in acoustic_classes().values()
    }

    assert counts == {len(OCTAVE_BANDS)}


def test_a_pillow_is_absorptive_and_a_tile_is_not() -> None:
    """The headline regression: soft things must come out soft."""
    pillow = absorption_for_category("pillow")
    tile = absorption_for_category("fireplace")

    assert pillow.mean() > 0.3
    assert tile.mean() < 0.1


def test_a_plant_is_nearly_transparent_at_low_frequency() -> None:
    """Leaves are centimetres across against a 2.7 m wavelength at 125 Hz, so
    foliage must not be modelled as a broadband absorber."""
    plant = absorption_for_category("plant")

    assert plant[0] < 0.05
    assert plant[-1] > plant[0]


def test_curtains_and_carpet_absorb_more_with_frequency() -> None:
    """Porous absorbers are weak where the wavelength is long. A material whose
    curve does not rise is a sign the row was entered flat by mistake."""
    for category in ("curtain", "carpet"):
        coefficients = absorption_for_category(category)
        assert coefficients[-1] > coefficients[0]


def test_lookup_is_exact_rather_than_substring() -> None:
    """The old loose match sent a bedside table to a bed's material."""
    assert class_for_category("nightstand").name == class_for_category("table").name

    with pytest.raises(UnknownCategoryError):
        class_for_category("bedside table that does not exist")


def test_an_unknown_category_raises_instead_of_being_randomised() -> None:
    with pytest.raises(UnknownCategoryError, match="add it there"):
        class_for_category("not_a_real_category")

    with pytest.raises(UnknownCategoryError):
        class_for_category(None)


def test_assignment_is_deterministic() -> None:
    """Two runs of the dataset generator must agree on what a room is made of."""
    first = material_for_label("seat", np.random.default_rng(0))
    second = material_for_label("seat", np.random.default_rng(999))

    assert first.energy_absorption["coeffs"] == second.energy_absorption["coeffs"]


def test_provenance_distinguishes_measurement_from_judgement() -> None:
    """An interpolated row is a judgement about what a thing is made of, and a
    reader must be able to tell it from a published measurement."""
    classes = acoustic_classes()

    assert any(material.measured for material in classes.values())
    assert any(not material.measured for material in classes.values())
    assert any(material.cross_checked for material in classes.values())
    assert all(material.provenance for material in classes.values())
    assert all(material.notes for material in classes.values())


def test_the_8_khz_column_is_carried_over_rather_than_invented() -> None:
    """Published tables stop at 4 kHz. Repeating the last measured value is the
    assumption that adds least, and it is recorded rather than hidden."""
    for material in acoustic_classes().values():
        assert material.absorption[-1] == pytest.approx(material.absorption[-2])


def test_material_round_trips_into_pyroomacoustics() -> None:
    material = material_for_label("couch")

    assert material.energy_absorption["center_freqs"] == list(OCTAVE_BANDS)
    assert len(material.scattering["coeffs"]) == len(OCTAVE_BANDS)
