"""How the catalogue reaches 16 kHz, given tables that stop at 4 kHz.

The architecture runs to 16 kHz and the published absorption data does not.
Every table this catalogue draws on stops at 4 kHz, so two octaves have to come
from somewhere. Until now the 8 kHz column was the 4 kHz value repeated, and a
test asserted it. That was defensible across one octave. Across two it states,
silently, that a 12 mm rug and a 100 mm mattress do the same thing at 16 kHz.

**The physical model was tried first and it does not survive its own
residual.** The obvious answer is to fit a rigid-backed Delany-Bazley layer to
each porous class and read 8 and 16 kHz off it: two parameters, flow
resistivity and thickness, against six measured bands, so the fit is
over-determined and can be judged. Judged, it fails. Over the soft classes the
residual on the *measured* bands runs to about 0.1 in absorption units, and the
disagreement is shape rather than noise: a layer model rises monotonically
towards its plateau, while several of these classes are already falling between
2 and 4 kHz, which is what upholstery and mattress measurements do. A model
that cannot reproduce the last measured octave has no standing to predict the
next two. Its extrapolations were also uniformly optimistic, putting upholstery
near 0.95 at 8 kHz against a measured 0.59 at 4 kHz.

So the model is kept, as :func:`layer_model_fit`, but as a *diagnostic* the
report quotes rather than as the source of any number. It is run over all 27
classes, not over a hand-picked subset, so no partition of the catalogue into
"porous" and "not" has to be defended.

**The rule that is used instead continues the material's own measured trend.**
The ratio between the last two measured bands, 2 and 4 kHz, is applied once per
octave above 4 kHz, clipped:

- **never above 1.** A porous absorber still rising at 4 kHz is approaching a
  plateau, not accelerating; continuing a rise two octaves would take
  ``curtain_light`` from a measured 0.35 to an invented 0.75. Clipping at the
  plateau degenerates to the old carry-over rule exactly where the old rule was
  defensible.
- **never below** :data:`MIN_OCTAVE_RATIO`. This bounds how far a single
  measured ratio, itself the quotient of two values rounded to two decimals,
  is allowed to be projected.

The rule is uniform: no class is special-cased, and each class's own data
decides how far it moves. Most of the catalogue holds its 4 kHz value because
its own measurements say to.

**The low bands are held, not modelled.** PFFDTD's fitting routine wants 11
octave bands from 16 Hz and the catalogue starts at 125 Hz, so 15.6, 31.25 and
62.5 Hz repeat the 125 Hz value. Nothing better is available: no source table
measures below 125 Hz, and Delany-Bazley there is outside its own stated
validity range for every flow resistivity fitted here. Those bands are also
nearly inert in a project that stops caring below a domestic room's Schroeder
frequency. It is stated rather than hidden, exactly as the 4 kHz carry-over
was.

Reference for the layer model: Delany and Bazley, *Acoustical properties of
fibrous absorbent materials*, Applied Acoustics 3 (1970). The random-incidence
average is the Paris formula, the same average PFFDTD's ``convert_Sabs_to_Yn``
inverts, so the modelled and the tabulated quantity are the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from reverberate.acoustics import AIR_DENSITY, SOLVER_BANDS, SPEED_OF_SOUND

#: Bands the catalogue actually measures, in Hz.
MEASURED_BANDS: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000)

#: Bands above the last measurement, in Hz. What this module is for.
HIGH_BANDS: tuple[float, ...] = tuple(band for band in SOLVER_BANDS if band > MEASURED_BANDS[-1])

#: Bands below the first measurement, in Hz. Held, see the module docstring.
LOW_BANDS: tuple[float, ...] = tuple(band for band in SOLVER_BANDS if band < MEASURED_BANDS[0])

#: Floor on the per-octave ratio above 4 kHz. A measured ratio is the quotient
#: of two numbers rounded to two decimals, so a class whose coefficient falls
#: sharply in the last measured octave is not allowed to fall that sharply
#: twice more on the strength of it.
MIN_OCTAVE_RATIO = 0.8

#: The locally reactive model cannot represent more than this, and PFFDTD's
#: ``convert_Sabs_to_Yn`` clips to it with a warning. Clip here instead, so the
#: clip is visible in our own report rather than in the solver's stdout.
MAX_ABSORPTION = 0.9512


@dataclass(frozen=True)
class HighBandExtension:
    """How one class's top two octaves were arrived at."""

    measured_ratio: float
    applied_ratio: float
    values: tuple[float, ...]

    @property
    def clipped(self) -> bool:
        """Whether the material's own ratio was overruled by a bound."""
        return not bool(np.isclose(self.measured_ratio, self.applied_ratio))

    @property
    def held(self) -> bool:
        """Whether this class ends up carrying 4 kHz forward after all."""
        return bool(np.isclose(self.applied_ratio, 1.0))


def extend_high_bands(measured: np.ndarray) -> HighBandExtension:
    """Continue the measured 2-to-4 kHz trend across the two bands above it."""
    measured = np.asarray(measured, dtype=float)
    last, previous = float(measured[-1]), float(measured[-2])
    ratio = 1.0 if previous <= 0.0 else last / previous
    applied = float(np.clip(ratio, MIN_OCTAVE_RATIO, 1.0))
    values = tuple(float(np.clip(last * applied**n, 0.0, MAX_ABSORPTION)) for n in (1, 2))
    return HighBandExtension(measured_ratio=ratio, applied_ratio=applied, values=values)


def extend_to_solver_bands(measured: np.ndarray) -> tuple[np.ndarray, HighBandExtension]:
    """Turn six measured bands into the eleven the solver is fitted on."""
    measured = np.asarray(measured, dtype=float)
    if measured.size != len(MEASURED_BANDS):
        raise ValueError(f"expected {len(MEASURED_BANDS)} measured bands, got {measured.size}")
    extension = extend_high_bands(measured)
    curve = np.concatenate(
        [np.repeat(measured[0], len(LOW_BANDS)), measured, np.asarray(extension.values)]
    )
    return np.asarray(np.clip(curve, 0.0, MAX_ABSORPTION)), extension


def _delany_bazley_surface_impedance(
    frequency: np.ndarray, flow_resistivity: float, thickness: float
) -> np.ndarray:
    """Normalised surface impedance of a rigid-backed porous layer."""
    ratio = AIR_DENSITY * frequency / flow_resistivity
    characteristic = 1.0 + 0.0571 * ratio**-0.754 - 1j * 0.087 * ratio**-0.732
    wavenumber = (2.0 * np.pi * frequency / SPEED_OF_SOUND) * (
        1.0 + 0.0978 * ratio**-0.700 - 1j * 0.189 * ratio**-0.595
    )
    return -1j * characteristic / np.tan(wavenumber * thickness)


def random_incidence_absorption(surface_impedance: np.ndarray, samples: int = 512) -> np.ndarray:
    """Paris-formula average of the oblique-incidence absorption.

    Tabulated coefficients are random-incidence, so fitting to normal incidence
    would fit the model to a different quantity than the data and then blame the
    difference on the material.
    """
    angle = np.linspace(0.0, np.pi / 2.0, samples)
    cosine = np.cos(angle)[None, :]
    reflection = (surface_impedance[:, None] * cosine - 1.0) / (
        surface_impedance[:, None] * cosine + 1.0
    )
    oblique = 1.0 - np.abs(reflection) ** 2
    return np.asarray(np.trapezoid(oblique * np.sin(2.0 * angle)[None, :], angle, axis=1))


def delany_bazley_absorption(
    frequency: np.ndarray, flow_resistivity: float, thickness: float
) -> np.ndarray:
    """Random-incidence absorption of a rigid-backed porous layer."""
    impedance = _delany_bazley_surface_impedance(
        np.asarray(frequency, dtype=float), flow_resistivity, thickness
    )
    return np.asarray(np.clip(random_incidence_absorption(impedance), 0.0, MAX_ABSORPTION))


@dataclass(frozen=True)
class LayerModelFit:
    """A Delany-Bazley layer fitted to one class's measured bands.

    Diagnostic only. Nothing in the catalogue is taken from it; the report
    quotes its residual as the evidence for not taking anything from it.
    """

    flow_resistivity: float
    thickness: float
    residual: tuple[float, ...]

    @property
    def rms_residual(self) -> float:
        """Absorption units, over the measured bands. The number to judge on."""
        return float(np.sqrt(np.mean(np.asarray(self.residual) ** 2)))

    @property
    def max_residual(self) -> float:
        return float(np.max(np.abs(np.asarray(self.residual))))

    @property
    def high_bands(self) -> tuple[float, ...]:
        """What the layer model would have said above 4 kHz."""
        values = delany_bazley_absorption(
            np.asarray(HIGH_BANDS), self.flow_resistivity, self.thickness
        )
        return tuple(float(value) for value in values)

    @property
    def validity_span(self) -> tuple[float, float]:
        """``rho f / sigma`` at 125 Hz and at 16 kHz.

        Delany and Bazley state their regression holds from 0.01 to 1.0.
        """
        return (
            AIR_DENSITY * MEASURED_BANDS[0] / self.flow_resistivity,
            AIR_DENSITY * SOLVER_BANDS[-1] / self.flow_resistivity,
        )


def layer_model_fit(measured: np.ndarray) -> LayerModelFit:
    """Fit flow resistivity and thickness to one class's measured bands."""
    frequency = np.asarray(MEASURED_BANDS, dtype=float)
    target = np.asarray(measured, dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        flow_resistivity, thickness = np.exp(parameters)
        return np.asarray(delany_bazley_absorption(frequency, flow_resistivity, thickness) - target)

    best: LayerModelFit | None = None
    # Multi-start on a coarse grid: the residual has local minima, because a
    # thin dense layer and a thick light one pass through the same few points,
    # and a single start from the middle lands in the wrong one for the carpets.
    for flow_resistivity in (5e3, 2e4, 5e4, 2e5):
        for thickness in (0.005, 0.02, 0.05, 0.15):
            solution = least_squares(
                residual,
                np.log([flow_resistivity, thickness]),
                bounds=(np.log([1e3, 0.002]), np.log([1e6, 0.30])),
            )
            fitted = LayerModelFit(
                float(np.exp(solution.x[0])),
                float(np.exp(solution.x[1])),
                tuple(float(value) for value in solution.fun),
            )
            if best is None or fitted.rms_residual < best.rms_residual:
                best = fitted
    assert best is not None
    return best
