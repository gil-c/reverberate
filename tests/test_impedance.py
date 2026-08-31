"""Tests for face B: the impedance filter the wave solver actually reads.

Two of these need PFFDTD itself, which is not pip-installable and is not
present in CI, so they skip unless ``PFFDTD_PYTHON`` points at the pinned
checkout. The passivity check does not need it: passivity is a property of a
triplet, and the property that matters is that a bad triplet is *caught*, which
is exactly what cannot be tested by only ever feeding it good ones.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reverberate.acoustics import SOLVER_BANDS
from reverberate.materials import acoustic_classes
from reverberate.materials.impedance import (
    admittance,
    curve_digest,
    fit_material,
    normal_incidence_absorption,
    passivity,
)

pffdtd = pytest.mark.skipif(
    not os.environ.get("PFFDTD_PYTHON"),
    reason="needs PFFDTD's python/ directory; see scripts/build_pffdtd.sh",
)


def test_a_passive_triplet_passes() -> None:
    """One RLC branch with non-negative parts, which is what the fit produces."""
    result = passivity(np.array([[1e-3, 50.0, 1e5]]))

    assert result.passive
    assert result.min_real_admittance > 0.0
    assert result.max_reflection_magnitude <= 1.0


def test_an_active_triplet_is_caught() -> None:
    """The check has to be able to fail, or it is decoration.

    A negative resistance is a boundary that returns more energy than it
    receives, which is how the solver diverges, and it must be visible here
    rather than as a NaN field two hours into a rented run.
    """
    result = passivity(np.array([[1e-3, -50.0, 1e5]]))

    assert not result.passive
    assert result.min_real_admittance < 0.0
    assert result.max_reflection_magnitude > 1.0


def test_a_rigid_boundary_reflects_everything() -> None:
    """Sanity on the admittance algebra itself, against a known answer."""
    absorption = normal_incidence_absorption(
        np.array([[0.0, 1e12, 0.0]]), np.asarray(SOLVER_BANDS, dtype=float)
    )

    assert absorption == pytest.approx(np.zeros(len(SOLVER_BANDS)), abs=1e-9)


def test_a_matched_boundary_absorbs_everything() -> None:
    """Unit specific admittance is the anechoic case."""
    absorption = normal_incidence_absorption(
        np.array([[0.0, 1.0, 0.0]]), np.asarray(SOLVER_BANDS, dtype=float)
    )

    assert absorption == pytest.approx(np.ones(len(SOLVER_BANDS)))


def test_branches_add_in_parallel() -> None:
    """Admittances sum; impedances do not. Getting this backwards would make
    every multi-branch material quietly too reflective."""
    first = np.array([[0.0, 4.0, 0.0]])
    second = np.array([[0.0, 4.0, 0.0]])
    frequency = np.array([1000.0])

    both = admittance(np.vstack([first, second]), frequency)

    assert both.real == pytest.approx(admittance(first, frequency).real * 2)


def test_the_cache_key_is_the_curve_and_nothing_else() -> None:
    """The fit is a Nelder-Mead solve per material and the curve is the only
    input, so a cached file is valid exactly while the curve is unchanged."""
    curve = np.asarray(acoustic_classes()["upholstery"].solver_absorption)
    changed = curve.copy()
    changed[-1] += 0.01

    assert curve_digest(curve) == curve_digest(curve.copy())
    assert curve_digest(curve) != curve_digest(changed)


@pffdtd
def test_every_class_fits_to_a_passive_filter(tmp_path: Path) -> None:
    """The trap that costs a rented run: a boundary the solver diverges on."""
    for material in acoustic_classes().values():
        fit = fit_material(material, tmp_path)

        assert fit.triplets.shape == (len(SOLVER_BANDS), 3)
        assert np.all(fit.triplets >= 0.0)
        assert fit.passivity.passive, f"{material.name} is not passive"


@pffdtd
def test_the_fitted_filter_says_what_the_catalogue_says(tmp_path: Path) -> None:
    """End to end, in the units the catalogue is written in.

    The tolerance is loose on purpose, and it is the measurement: a locally
    reactive boundary with a real target admittance cannot reproduce a strong
    absorber exactly, and the report quotes the gap per class rather than
    hiding it behind a tighter assertion on an easier material.
    """
    for name in ("ceramic_tile", "wood_panel", "carpet_thick"):
        fit = fit_material(acoustic_classes()[name], tmp_path)

        assert np.max(np.abs(fit.band_error)) < 0.1
        assert np.max(np.abs(fit.model_error)) < 0.1
