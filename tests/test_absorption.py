"""Tests for absorption compensation.

The property the whole scheme rests on is that absorbing power α·S survives
decimation. The property that keeps it honest is that when it *cannot* survive
— because a coefficient cannot exceed 1 — the loss is counted rather than
swallowed. Both are tested here, plus the band-count invariant that a whole
apartment once tripped over.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra
import pytest

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.geometry.absorption import (
    MAX_ABSORPTION,
    MAX_COMPENSATION_FACTOR,
    absorbing_power,
    audit,
    band_weights,
    compensate,
)


def test_absorbing_power_is_mean_alpha_times_area() -> None:
    material = pra.Material(0.25)

    assert absorbing_power(material, 8.0) == pytest.approx(2.0)


def test_compensation_preserves_absorbing_power_exactly() -> None:
    """The headline property: a quarter of the surface, four times the alpha."""
    material = pra.Material(0.2)
    before = absorbing_power(material, 12.0)

    entry = compensate(material, original_area=12.0, reduced_area=3.0)

    assert entry.applied_factor == pytest.approx(4.0)
    assert absorbing_power(entry.material, 3.0) == pytest.approx(before)


def test_a_mesh_that_kept_its_area_is_left_alone() -> None:
    material = pra.Material(0.3)

    entry = compensate(material, original_area=5.0, reduced_area=5.0)

    assert entry.applied_factor == 1.0
    assert not entry.compensated
    assert not entry.capped
    assert entry.material is material


def test_coefficients_are_never_pushed_above_one() -> None:
    """A surface cannot absorb more than everything that reaches it."""
    material = pra.Material(0.5)

    entry = compensate(material, original_area=10.0, reduced_area=2.0)

    coefficients = np.asarray(entry.material.energy_absorption["coeffs"])
    assert np.all(coefficients <= MAX_ABSORPTION)
    assert entry.capped
    assert entry.capped_bands == len(coefficients)


def test_capping_is_reported_and_loses_power_visibly() -> None:
    """Capping is the one case the scheme cannot handle, so it must show."""
    material = pra.Material(0.5)
    before = absorbing_power(material, 10.0)

    entry = compensate(material, original_area=10.0, reduced_area=2.0)
    after = absorbing_power(entry.material, 2.0)

    assert after < before
    assert entry.capped
    assert "capped" in entry.summary()


def test_the_compensation_factor_is_bounded() -> None:
    """Past the ceiling, scaling stops being a correction and becomes invention."""
    material = pra.Material(0.02)

    entry = compensate(material, original_area=100.0, reduced_area=1.0)

    assert entry.requested_factor == pytest.approx(100.0)
    assert entry.applied_factor == MAX_COMPENSATION_FACTOR


def test_scattering_is_untouched_by_compensation() -> None:
    """Scattering describes how a surface redirects what it does not absorb,
    which decimation does not change."""
    material = pra.Material(0.2, 0.15)

    entry = compensate(material, original_area=8.0, reduced_area=2.0)

    assert list(entry.material.scattering["coeffs"]) == list(material.scattering["coeffs"])


def test_compensation_keeps_the_band_count() -> None:
    """A whole apartment once failed with "All walls should have the same
    number of frequency bands"; compensation must not reintroduce that."""
    material = pra.Material("curtains_velvet")

    entry = compensate(material, original_area=10.0, reduced_area=4.0)

    assert len(entry.material.energy_absorption["coeffs"]) == len(
        material.energy_absorption["coeffs"]
    )
    assert list(entry.material.energy_absorption["center_freqs"]) == list(
        material.energy_absorption["center_freqs"]
    )


def test_degenerate_areas_do_not_produce_a_nonsense_factor() -> None:
    material = pra.Material(0.2)

    assert compensate(material, 0.0, 1.0).applied_factor == 1.0
    assert compensate(material, 1.0, 0.0).applied_factor == 1.0


def test_audit_shows_no_loss_when_nothing_is_capped() -> None:
    materials = [pra.Material(0.1), pra.Material(0.2)]
    entries = [
        compensate(materials[0], 10.0, 5.0),
        compensate(materials[1], 8.0, 4.0),
    ]

    report = audit(entries, materials)

    assert report.power_error == pytest.approx(0.0, abs=1e-9)
    assert report.compensated == 2
    assert report.capped == 0


def test_audit_surfaces_the_loss_when_capping_bites() -> None:
    materials = [pra.Material(0.6)]
    entries = [compensate(materials[0], 10.0, 2.0)]

    report = audit(entries, materials)

    assert report.capped == 1
    assert report.power_error > 0.0
    assert "capped" in report.summary()


def test_audit_refuses_mismatched_inputs() -> None:
    """Pairing entries with the wrong materials would silently misreport."""
    with pytest.raises(ValueError):
        audit([compensate(pra.Material(0.2), 4.0, 2.0)], [])


def octave_material(coefficient: float = 0.1) -> pra.Material:
    """A seven-band material, which is what the material table produces."""
    return pra.Material(
        energy_absorption={
            "coeffs": [coefficient] * len(OCTAVE_BANDS),
            "center_freqs": list(OCTAVE_BANDS),
        },
        scattering={"coeffs": [0.1] * len(OCTAVE_BANDS), "center_freqs": list(OCTAVE_BANDS)},
    )


def test_band_weights_are_one_where_the_feature_is_at_least_a_wavelength() -> None:
    # 2.744 m is the wavelength at 125 Hz, so every band sees a feature that big.
    weights = band_weights(2.744, len(OCTAVE_BANDS))

    assert np.allclose(weights, 1.0)


def test_band_weights_taper_towards_the_low_bands() -> None:
    # A 10 cm feature: invisible at 125 Hz (274 cm), fully seen at 4 kHz (8.6 cm).
    weights = band_weights(0.10, len(OCTAVE_BANDS))

    assert weights[0] == pytest.approx(0.10 / 2.744, rel=1e-3)
    assert weights[-1] == 1.0
    assert np.all(np.diff(weights) >= 0.0)


def test_band_weights_leave_a_scalar_material_alone() -> None:
    """A material with one coefficient has no frequency to weigh against.

    Weighting it would mean inventing one, and the honest default is the
    previous behaviour rather than a silent promotion to multi-band.
    """
    assert np.allclose(band_weights(0.01, 1), 1.0)


def test_compensation_is_weighted_down_at_low_frequency() -> None:
    """The point of the whole change: a small feature is not restored at 125 Hz.

    A plant's lost leaf area contributed absorption where the wavelength was
    comparable to a leaf. Crediting it with the same absorption at 125 Hz, where
    the wavelength is 274 cm, is inventing absorption the object never had.
    """
    material = octave_material(0.1)

    entry = compensate(material, original_area=8.0, reduced_area=2.0, feature_size=0.05)

    coefficients = np.asarray(entry.material.energy_absorption["coeffs"])
    # 5 cm against 274 cm is a weight of 0.018, so a factor of 4 arrives as 1.05.
    assert coefficients[0] == pytest.approx(0.1 * (1.0 + 3.0 * 0.05 / 2.744))
    assert coefficients[-1] == pytest.approx(0.4)
    assert np.all(np.diff(coefficients) >= 0.0)


def test_an_unweighted_compensation_is_unchanged() -> None:
    """Omitting the feature size must reproduce the previous behaviour exactly."""
    material = octave_material(0.1)

    weighted = compensate(material, original_area=8.0, reduced_area=2.0, feature_size=0.0)

    assert np.allclose(weighted.material.energy_absorption["coeffs"], 0.4)
    assert weighted.applied_factor == pytest.approx(4.0)


def test_weighting_removes_capping_the_flat_factor_caused() -> None:
    """Most capping was the low bands, and the low bands are what weighting spares.

    A coefficient that saturates is absorbing power genuinely lost, so a scheme
    that avoids saturating in the first place is strictly better than one that
    counts how often it does.
    """
    material = octave_material(0.3)

    flat = compensate(material, original_area=10.0, reduced_area=2.0)
    weighted = compensate(material, original_area=10.0, reduced_area=2.0, feature_size=0.05)

    assert flat.capped_bands > 0
    assert weighted.capped_bands < flat.capped_bands


def test_audit_separates_intentional_low_band_shortfall_from_capping() -> None:
    """The audit has to tell a correction apart from a defect.

    Both look like "less absorbing power than the table says". Only the top
    band distinguishes them: nothing is ever weighted away at 8 kHz, so a
    shortfall there is capping and a shortfall at 125 Hz alone is the weighting
    doing its job.
    """
    material = octave_material(0.1)
    entry = compensate(material, original_area=8.0, reduced_area=2.0, feature_size=0.05)

    report = audit([entry], [material])
    error = report.band_power_error

    assert error is not None
    assert error[0] < -0.5
    assert report.top_band_error == pytest.approx(0.0, abs=1e-9)
