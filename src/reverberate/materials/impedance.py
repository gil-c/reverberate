"""Face B: the impedance filter the wave solver reads, and its two traps.

The solver does not take an absorption coefficient. It takes a boundary
admittance as a sum of RLC branches, one triplet ``(D, E, F)`` per branch,
which it turns into a filter running on every boundary node. PFFDTD ships the
routine that fits those triplets to eleven octave-band absorption coefficients,
so this module does not reimplement it; it feeds the catalogue in, keeps the
result, and then checks the two things the fit does not check itself.

**Passivity.** The branch impedance is ``jw D + E + F / jw``, so a triplet with
all three parts non-negative has a non-negative real part everywhere by
construction, and the parallel sum of such branches does too. That is an
argument, not a measurement, and the solver diverging is expensive enough that
it is worth measuring: :func:`passivity` evaluates the admittance on a dense
frequency vector and reports the worst real part actually seen, along with
whether the reflection magnitude ever exceeds one, which is the same statement
in the units the solver's stability depends on.

**Phase.** An absorption coefficient carries no phase, so the phase of every
material here comes from the resonant model the fit assumes and not from any
measurement. Nothing in this file can improve on that. The one exception is
the porous classes, whose high bands come from a Delany-Bazley layer in
:mod:`reverberate.materials.extrapolation`, where the phase is grounded in a
flow resistivity; that phase is used to choose the absorption values, but the
solver still receives them through the same phase-free fit as everything else,
so the exception buys less than it sounds like it does. It is recorded so the
report can say so rather than implying the fit knows more than it does.

The fitted coefficients are cached content-addressed on the eleven-band curve,
because the fit is a Nelder-Mead solve per material and the curve is the only
thing that determines its result.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.acoustics import SOLVER_BANDS
from reverberate.materials.db import AcousticClass, acoustic_classes
from reverberate.materials.extrapolation import random_incidence_absorption
from reverberate.settings import interim_dir, pffdtd_python

#: Where the admittance is evaluated when checking a fit. Wider than the
#: solver's band of interest on purpose: passivity has to hold everywhere the
#: filter runs, not only where the material was specified.
CHECK_FREQUENCIES = np.logspace(np.log10(1.0), np.log10(24000.0), 4000)


def _load_adm_funcs() -> Any:
    """PFFDTD's material routines, imported from the pinned checkout."""
    path = str(pffdtd_python())
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("materials.adm_funcs")


def curve_digest(absorption: np.ndarray) -> str:
    """Content address of one eleven-band curve."""
    rounded = np.round(np.asarray(absorption, dtype=float), 6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def admittance(triplets: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """Normalised surface admittance of a set of RLC branches."""
    triplets = np.atleast_2d(np.asarray(triplets, dtype=float))
    jw = 1j * 2.0 * np.pi * np.asarray(frequency, dtype=float)
    d, e, f = triplets.T
    branch = jw[:, None] * d[None, :] + e[None, :] + f[None, :] / jw[:, None]
    return np.asarray(np.sum(1.0 / branch, axis=-1))


def normal_incidence_absorption(triplets: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """What the fitted filter actually absorbs at normal incidence."""
    admittances = admittance(triplets, frequency)
    reflection = (1.0 - admittances) / (1.0 + admittances)
    return np.asarray(1.0 - np.abs(reflection) ** 2)


@dataclass(frozen=True)
class Passivity:
    """The outcome of the check that decides whether the solver survives."""

    min_real_admittance: float
    max_reflection_magnitude: float
    frequency_of_worst: float

    @property
    def passive(self) -> bool:
        """A positive real part everywhere, and no reflection gain."""
        return self.min_real_admittance > 0.0 and self.max_reflection_magnitude <= 1.0 + 1e-9


def passivity(triplets: np.ndarray, frequency: np.ndarray = CHECK_FREQUENCIES) -> Passivity:
    """Measure passivity of a fitted filter rather than assume it."""
    admittances = admittance(triplets, frequency)
    real = np.real(admittances)
    reflection = np.abs((1.0 - admittances) / (1.0 + admittances))
    worst = int(np.argmin(real))
    return Passivity(
        min_real_admittance=float(real[worst]),
        max_reflection_magnitude=float(np.max(reflection)),
        frequency_of_worst=float(frequency[worst]),
    )


def _paris_inverted_admittance(absorption: np.ndarray) -> np.ndarray:
    """The routine's own target, in its own terms.

    ``fit_to_Sabs_oct_11`` does not fit the tabulated Sabine coefficient. It
    inverts the Paris formula first, with ``convert_Sabs_to_Yn``, and then fits
    the *normal-incidence* absorption of the resulting real admittance. So
    comparing the fitted filter's normal-incidence absorption against the
    catalogue's Sabine number measures that conversion, which is a deliberate
    part of the model, rather than measuring the fit. Both are worth knowing
    and this project needs them separated, so the conversion is redone here
    with PFFDTD's own function.
    """
    adm_funcs = _load_adm_funcs()
    return np.asarray(
        [adm_funcs.convert_Sabs_to_Yn(float(value)) for value in absorption], dtype=float
    )


@dataclass(frozen=True)
class ImpedanceFit:
    """One material as the solver will see it."""

    name: str
    path: Path
    triplets: np.ndarray
    target: np.ndarray
    passivity: Passivity

    @property
    def achieved(self) -> np.ndarray:
        """Normal-incidence absorption of the fitted filter, per solver band."""
        return normal_incidence_absorption(self.triplets, np.asarray(SOLVER_BANDS, dtype=float))

    @property
    def fit_target(self) -> np.ndarray:
        """What the routine was actually asked to hit, in normal-incidence terms."""
        admittances = _paris_inverted_admittance(self.target)
        reflection = (1.0 - admittances) / (1.0 + admittances)
        return 1.0 - np.abs(reflection) ** 2

    @property
    def band_error(self) -> np.ndarray:
        """Fit quality: achieved minus asked, per band, at normal incidence.

        The routine minimises absolute error over a log-spaced frequency vector
        rather than interpolating the band centres, so this is not expected to
        be zero. It is reported because a material whose band error is large is
        one the solver is not being told what the catalogue says.
        """
        return np.asarray(self.achieved - self.fit_target)

    @property
    def achieved_random_incidence(self) -> np.ndarray:
        """The fitted filter, averaged back over incidence angle.

        End to end: this is directly comparable with the catalogue's own
        coefficients, because both are random-incidence quantities. The gap
        between this and :attr:`target` is what a locally reactive boundary
        with a real admittance costs, and it is a property of the model rather
        than of the fit.
        """
        admittances = admittance(self.triplets, np.asarray(SOLVER_BANDS, dtype=float))
        return random_incidence_absorption(1.0 / admittances)

    @property
    def model_error(self) -> np.ndarray:
        """Random-incidence absorption of the filter minus the catalogue's."""
        return np.asarray(self.achieved_random_incidence - self.target)

    def summary(self) -> dict[str, Any]:
        return {
            "material_class": self.name,
            "file": self.path.name,
            "branches": int(self.triplets.shape[0]),
            "target_absorption": [round(float(v), 4) for v in self.target],
            "fit_target_absorption": [round(float(v), 4) for v in self.fit_target],
            "achieved_absorption": [round(float(v), 4) for v in self.achieved],
            "achieved_random_incidence": [
                round(float(v), 4) for v in self.achieved_random_incidence
            ],
            "max_band_error": round(float(np.max(np.abs(self.band_error))), 4),
            "rms_band_error": round(float(np.sqrt(np.mean(self.band_error**2))), 4),
            "max_model_error": round(float(np.max(np.abs(self.model_error))), 4),
            "passive": self.passivity.passive,
            "min_real_admittance": float(f"{self.passivity.min_real_admittance:.6g}"),
            "max_reflection_magnitude": round(self.passivity.max_reflection_magnitude, 6),
        }


def fit_material(material: AcousticClass, output_dir: Path | None = None) -> ImpedanceFit:
    """Fit one class's eleven-band curve with PFFDTD's own routine."""
    directory = output_dir if output_dir is not None else interim_dir("materials")
    directory.mkdir(parents=True, exist_ok=True)
    target = np.asarray(material.solver_absorption, dtype=float)
    path = directory / f"{material.name}_{curve_digest(target)}.h5"

    adm_funcs = _load_adm_funcs()
    if not path.is_file():
        adm_funcs.fit_to_Sabs_oct_11(target, path)
    triplets = np.atleast_2d(adm_funcs.read_mat_DEF(path))

    return ImpedanceFit(
        name=material.name,
        path=path,
        triplets=triplets,
        target=target,
        passivity=passivity(triplets),
    )


def fit_all(output_dir: Path | None = None) -> list[ImpedanceFit]:
    """Fit every class in the catalogue, in catalogue order."""
    return [fit_material(material, output_dir) for material in acoustic_classes().values()]


def write_manifest(fits: list[ImpedanceFit], output_dir: Path | None = None) -> Path:
    """Record what was fitted, so a solver run can say which file it used."""
    directory = output_dir if output_dir is not None else interim_dir("materials")
    manifest = {
        "solver_bands": list(SOLVER_BANDS),
        "pffdtd_python": str(pffdtd_python()),
        "materials": [fit.summary() for fit in fits],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
