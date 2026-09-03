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

from reverberate.acoustics import OCTAVE_BANDS, SOLVER_BANDS
from reverberate.geometry.materials import material_for_label
from reverberate.materials import (
    UnknownCategoryError,
    absorption_for_category,
    acoustic_classes,
    category_assignments,
    class_for_category,
    coverage,
)
from reverberate.materials.db import CLASSES_FILE
from reverberate.materials.extrapolation import (
    MAX_ABSORPTION,
    MEASURED_BANDS,
    MIN_OCTAVE_RATIO,
    PLATEAU_TOP_RATIO,
    extend_high_bands,
    layer_model_fit,
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


def test_the_file_stores_only_what_is_measured() -> None:
    """The 8 kHz column used to be the 4 kHz value repeated, in the file, as
    though it were data. Everything above 4 kHz is now derived instead."""
    header = (CLASSES_FILE.read_text().splitlines()[0]).split(",")

    assert [column for column in header if column.startswith("a")] == [
        f"a{band}" for band in MEASURED_BANDS
    ]
    assert "a8000" not in header
    assert "a16000" not in header


def test_the_bands_above_4_khz_follow_the_measured_trend() -> None:
    """The rule W4 settled on: the material's own 2-to-4 kHz ratio, applied
    once per octave, clipped to at most 1 and at least the ratio floor."""
    for material in acoustic_classes().values():
        extension = material.high_bands
        measured = material.measured_absorption

        assert MIN_OCTAVE_RATIO <= extension.applied_ratio <= 1.0
        first = min(measured[-1] * extension.applied_ratio, MAX_ABSORPTION)
        second_ratio = (
            PLATEAU_TOP_RATIO if extension.applied_ratio >= 1.0 else extension.applied_ratio
        )
        assert extension.values[0] == pytest.approx(first)
        assert extension.values[1] == pytest.approx(min(first * second_ratio, MAX_ABSORPTION))


def test_nothing_gains_absorption_above_the_last_measurement() -> None:
    """A class still rising at 4 kHz is approaching a plateau, not
    accelerating. Continuing the rise two octaves invents the number."""
    for material in acoustic_classes().values():
        top = material.solver_absorption[-3:]

        assert top[1] <= top[0] + 1e-12
        assert top[2] <= top[1] + 1e-12


def test_a_rising_class_holds_at_8_khz_then_gives_way_at_16() -> None:
    """Carpet is measured to peak near 6.5 kHz and fall after it, so a plateau
    is defensible for one octave above the last measurement and not for two."""
    rising = extend_high_bands(np.array([0.05, 0.06, 0.13, 0.18, 0.24, 0.35]))

    assert rising.held and rising.clipped
    assert rising.values[0] == pytest.approx(0.35)
    assert rising.values[1] == pytest.approx(0.35 * PLATEAU_TOP_RATIO)


def test_a_falling_class_keeps_its_own_ratio_in_both_octaves() -> None:
    """The ceiling only overrules a class that was not already falling."""
    falling = extend_high_bands(np.array([0.19, 0.37, 0.56, 0.67, 0.61, 0.59]))

    assert not falling.held
    assert falling.values[1] < falling.values[0] < 0.59
    ratio = 0.59 / 0.61
    assert falling.values == pytest.approx((0.59 * ratio, 0.59 * ratio**2))


def test_a_single_measured_ratio_is_not_projected_without_bound() -> None:
    """A ratio taken from two values rounded to two decimals is not precise
    enough to be raised to the second power unbounded."""
    collapsing = extend_high_bands(np.array([0.5, 0.5, 0.5, 0.5, 0.50, 0.10]))

    assert collapsing.applied_ratio == pytest.approx(MIN_OCTAVE_RATIO)
    assert collapsing.values == pytest.approx((0.08, 0.064))


def test_the_low_bands_repeat_the_lowest_measurement() -> None:
    """No source measures below 125 Hz. Held, and said so, not modelled."""
    for material in acoustic_classes().values():
        assert material.solver_absorption[:4] == pytest.approx(
            (material.measured_absorption[0],) * 4
        )


def test_the_solver_curve_is_the_eleven_bands_pffdtd_asserts_on() -> None:
    """``fit_to_Sabs_oct_11`` asserts ``Sabs.size == 11`` and hard-codes the
    centres, so a curve of any other length never reaches the solver."""
    assert len(SOLVER_BANDS) == 11
    assert SOLVER_BANDS[0] == pytest.approx(15.625)
    assert SOLVER_BANDS[-1] == pytest.approx(16000.0)

    for material in acoustic_classes().values():
        assert len(material.solver_absorption) == 11
        assert all(0.0 <= value <= MAX_ABSORPTION for value in material.solver_absorption)


def test_the_two_faces_of_a_material_agree_where_they_overlap() -> None:
    """Face A feeds the metrics and face B feeds the solver. If the 8 kHz value
    differed between them, nothing downstream would say so."""
    for material in acoustic_classes().values():
        assert material.absorption == pytest.approx(
            material.measured_absorption + (material.solver_absorption[-2],)
        )


def test_the_layer_model_does_not_describe_the_soft_classes() -> None:
    """The reason W4 does not extrapolate with Delany-Bazley. If this ever
    stops holding, the physical model becomes the better rule and this
    module's decision should be revisited."""
    upholstery = acoustic_classes()["upholstery"]

    fit = layer_model_fit(np.asarray(upholstery.measured_absorption))

    assert fit.rms_residual > 0.05
    assert fit.high_bands[0] > upholstery.measured_absorption[-1] + 0.2


def test_material_round_trips_into_pyroomacoustics() -> None:
    material = material_for_label("couch")

    assert material.energy_absorption["center_freqs"] == list(OCTAVE_BANDS)
    assert len(material.scattering["coeffs"]) == len(OCTAVE_BANDS)
