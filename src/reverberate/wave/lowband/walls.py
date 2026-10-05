"""A wall's impedance fitted again for the low band alone: fewer branches, the same wall.

A material reaches the solver as eleven branches, a mass, a resistance and a
stiffness in series each, resonant an octave apart from 16 Hz to 16 kHz
(:mod:`reverberate.materials.impedance`): a fit for a solver that runs to
the top of hearing. The low band runs from 40 Hz to 1500 Hz. A lossy node
keeps two states a branch and moves them every step, and those states are
most of what a step moves (``docs/open-questions/performance-audit.md``): a
branch that the band does not need is paid at every lossy node of every
step.

Here each material's own admittance, as its eleven branches give it, is
fitted again over the band by ``M`` branches whose three numbers are free
and positive (:func:`refit`). Positive numbers are all the solver asks: a
branch ``D s + E + F / s`` with ``D, E, F >= 0`` is passive, their parallel
sum is, and the scheme's update of such a branch is stable for any of them,
so nothing has to be proven again about stability. What has to be shown is
that the wall is the same wall: :func:`errors` reads the fit against the
eleven branches a third octave at a time, in admittance, in the reflection
at normal incidence and in absorption at normal and at random incidence,
and the worst band is the figure.

**The fitter.** A branch is held by its resonance, its quality and its
conductance at resonance, in logarithms, which keeps it positive and well
conditioned. The start is the eleven branches themselves; branches are then
taken away one at a time, each time the one whose loss the others make up
best after a fit of the rest (the best three candidates are fitted, the
best kept), down to ``M``. Each fit is a least squares of the error
relative to the admittance's own modulus, followed by Lawson's reweighting
towards the least worst error, since the worst frequency is what is read.
Nothing is drawn at random: the same branches give the same fit.

**The bar.** Absorption coefficients are octave values known to a few
hundredths (the catalogue's own sources disagree by that much), and the
solver is held against the present engine at -30 dB in the worst third
octave over the 50 ms after the onset. A reflection's error of ``e`` at
each of a handful of bounces stays 10 dB under that bar when ``e`` is under
-46 dB: :data:`REFLECTION_BAR_DB`. :func:`smallest_count` gives the fewest
branches that hold every material under it, and under a hundredth of
absorption. Whether the card's comparison then passes is the proof; until
it does the solver's default is the materials as they are.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "ABSORPTION_BAR",
    "FIT_VERSION",
    "REFLECTION_BAR_DB",
    "WallFit",
    "admittance",
    "errors",
    "refit",
    "refit_all",
    "refit_kept",
    "smallest_count",
    "third_octaves",
    "worst_of",
]

#: In a pair's key: a change of the fitter that changes its numbers changes this.
FIT_VERSION = 1
#: The worst error of a reflection at normal incidence a refit may make, dB re the incident wave.
REFLECTION_BAR_DB = -46.0
#: The worst change of an absorption coefficient a refit may make: a quarter of what is known.
ABSORPTION_BAR = 0.01

#: Frequencies a fit is made on, a third octave: dense enough that a resonance is not stepped over.
POINTS_A_THIRD = 9
#: The bounds of a branch's resonance, Hz, and of its quality.
RESONANCE_HZ = (0.5, 2.0e5)
QUALITY = (1e-3, 1e3)
LAWSON_ROUNDS = 12


@dataclass(frozen=True)
class WallFit:
    """What a campaign's walls are fitted with: the count of branches and the band."""

    branches: int
    fmin_hz: float = 40.0
    fmax_hz: float = 1500.0

    def name(self) -> str:
        """As a pair's key says it: two fits never meet in a cache."""
        return (
            f"walls of {self.branches} branches fitted from {self.fmin_hz:g} to"
            f" {self.fmax_hz:g} Hz (reverberate.wave.lowband.walls/{FIT_VERSION})"
        )


def admittance(triplets: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """The wall's normalised admittance: the sum of ``1 / (D s + E + F / s)`` over its branches."""
    triplets = np.atleast_2d(np.asarray(triplets, dtype=np.float64))
    s = 2j * np.pi * np.asarray(frequency, dtype=np.float64)
    d, e, f = triplets[:, 0:1], triplets[:, 1:2], triplets[:, 2:3]
    return np.asarray(np.sum(1.0 / (d * s[None, :] + e + f / s[None, :]), axis=0))


def third_octaves(fmin_hz: float, fmax_hz: float) -> list[tuple[float, float, float]]:
    """The third octaves that meet the band: ``(centre, low, high)``, each cut to the band."""
    out = []
    for n in range(-30, 14):
        centre = 1000.0 * 2.0 ** (n / 3.0)
        low, high = centre * 2.0 ** (-1.0 / 6.0), centre * 2.0 ** (1.0 / 6.0)
        low, high = max(low, fmin_hz), min(high, fmax_hz)
        if high > low * 1.02:
            out.append((centre, low, high))
    return out


def _grid(fmin_hz: float, fmax_hz: float) -> np.ndarray:
    thirds = 3.0 * np.log2(fmax_hz / fmin_hz)
    return np.asarray(np.geomspace(fmin_hz, fmax_hz, int(np.ceil(thirds * POINTS_A_THIRD)) + 1))


def _to_parameters(triplets: np.ndarray) -> np.ndarray:
    """``D E F`` as the logarithms of a branch's resonance, quality and conductance."""
    d, e, f = (np.maximum(triplets[:, k], 1e-300) for k in range(3))
    resonance = np.clip(np.sqrt(f / d) / (2.0 * np.pi), *RESONANCE_HZ)
    quality = np.clip(np.sqrt(d * f) / e, *QUALITY)
    return np.asarray(np.log(np.stack([resonance, quality, 1.0 / e], axis=1)))


def _to_triplets(parameters: np.ndarray) -> np.ndarray:
    resonance, quality, conductance = np.exp(parameters).T
    e = 1.0 / conductance
    omega = 2.0 * np.pi * resonance
    return np.asarray(np.stack([quality * e / omega, e, quality * e * omega], axis=1))


def _model(parameters: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    resonance, quality, conductance = np.exp(parameters.reshape(-1, 3)).T
    ratio = frequency[None, :] / resonance[:, None]
    return np.asarray(
        np.sum(conductance[:, None] / (1.0 + 1j * quality[:, None] * (ratio - 1.0 / ratio)), axis=0)
    )


def _jacobian(parameters: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """The model's derivatives by its parameters, ``[frequency, branch, 3]`` complex."""
    resonance, quality, conductance = np.exp(parameters.reshape(-1, 3)).T
    ratio = frequency[None, :] / resonance[:, None]
    detuned = 1j * quality[:, None] * (ratio - 1.0 / ratio)
    branch = conductance[:, None] / (1.0 + detuned)
    by_resonance = branch * 1j * quality[:, None] * (ratio + 1.0 / ratio) / (1.0 + detuned)
    by_quality = -branch * detuned / (1.0 + detuned)
    return np.asarray(np.stack([by_resonance, by_quality, branch], axis=2).transpose(1, 0, 2))


def _fit(
    start: np.ndarray, frequency: np.ndarray, target: np.ndarray, *, rounds: int
) -> tuple[np.ndarray, float]:
    """The branches nearest ``target`` from ``start``, and their worst relative error."""
    scale = 1.0 / np.abs(target)
    lower = np.tile(np.log([RESONANCE_HZ[0], QUALITY[0], 1e-9]), start.shape[0])
    upper = np.tile(np.log([RESONANCE_HZ[1], QUALITY[1], 1e9]), start.shape[0])
    weights = np.ones(frequency.size)
    best, best_worst = start, np.inf
    x = np.clip(start.reshape(-1), lower + 1e-9, upper - 1e-9)
    for _ in range(max(1, rounds)):

        def residual(p: np.ndarray, held: np.ndarray = weights) -> np.ndarray:
            error = (_model(p, frequency) - target) * scale * held
            return np.asarray(np.concatenate([error.real, error.imag]))

        def jacobian(p: np.ndarray, held: np.ndarray = weights) -> np.ndarray:
            slope = _jacobian(p, frequency).reshape(frequency.size, -1) * (scale * held)[:, None]
            return np.asarray(np.concatenate([slope.real, slope.imag], axis=0))

        solved = least_squares(
            residual,
            x,
            jac=jacobian,
            bounds=(lower, upper),
            x_scale="jac",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=400,
        )
        x = solved.x
        error = np.abs(_model(x, frequency) - target) * scale
        worst = float(error.max())
        if worst < best_worst:
            best, best_worst = x.reshape(-1, 3).copy(), worst
        # Lawson: the frequencies that err most weigh more in the next fit.
        weights = weights * np.sqrt(error / max(float(error.mean()), 1e-300))
        weights = weights / weights.mean()
    return best, best_worst


def refit(
    triplets: np.ndarray,
    branches: int,
    *,
    fmin_hz: float = 40.0,
    fmax_hz: float = 1500.0,
) -> np.ndarray:
    """A material's ``D E F`` as ``branches`` branches that give its admittance over the band.

    A material that has no more than ``branches`` is returned as it is. The
    result's rows are in the order of their resonances, and every number is
    positive.
    """
    triplets = np.atleast_2d(np.asarray(triplets, dtype=np.float64))
    if branches < 1:
        raise ValueError("a wall has one branch or more")
    if triplets.shape[0] <= branches:
        return triplets.copy()
    frequency = _grid(fmin_hz, fmax_hz)
    target = admittance(triplets, frequency)
    held = _to_parameters(triplets)
    scale = 1.0 / np.abs(target)
    while held.shape[0] > branches:
        # Which branch the others miss least, read before any fit; the best three are fitted.
        each = [
            float((np.abs(_model(np.delete(held, j, axis=0), frequency) - target) * scale).max())
            for j in range(held.shape[0])
        ]
        tried = []
        for j in np.argsort(each)[:3]:
            tried.append(_fit(np.delete(held, int(j), axis=0), frequency, target, rounds=2))
        held = min(tried, key=lambda found: found[1])[0]
    held, _ = _fit(held, frequency, target, rounds=LAWSON_ROUNDS)
    out = _to_triplets(held)
    return np.asarray(out[np.argsort(out[:, 2] / out[:, 0])])


def _absorptions(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflection and absorption at normal incidence, and absorption at random incidence."""
    reflection = (1.0 - y) / (1.0 + y)
    cosine = np.cos(np.linspace(0.0, np.pi / 2.0, 181))[None, :]
    sine2 = np.sin(2.0 * np.arccos(cosine))
    oblique = 1.0 - np.abs((cosine - y[:, None]) / (cosine + y[:, None])) ** 2
    random = np.trapezoid(oblique * sine2, np.arccos(cosine[0]), axis=1)
    return reflection, 1.0 - np.abs(reflection) ** 2, np.asarray(random)


def errors(
    reference: np.ndarray,
    fitted: np.ndarray,
    *,
    fmin_hz: float = 40.0,
    fmax_hz: float = 1500.0,
) -> list[dict[str, float]]:
    """A refit against the material it stands for, the worst of each third octave of the band.

    ``admittance`` is the error's modulus over the admittance's;
    ``reflection_db`` the error of the reflection at normal incidence, dB
    re the incident wave; ``normal`` and ``random`` the change of the
    absorption coefficient at normal and at random incidence, as it is.
    """
    out = []
    for centre, low, high in third_octaves(fmin_hz, fmax_hz):
        frequency = np.geomspace(low, high, 25)
        a, b = admittance(reference, frequency), admittance(fitted, frequency)
        (ra, na, da), (rb, nb, db) = _absorptions(a), _absorptions(b)
        out.append(
            {
                "hz": round(centre, 1),
                "admittance": float((np.abs(b - a) / np.abs(a)).max()),
                "reflection_db": float(20.0 * np.log10(max(np.abs(rb - ra).max(), 1e-12))),
                "normal": float(np.abs(nb - na).max()),
                "random": float(np.abs(db - da).max()),
            }
        )
    return out


def worst_of(table: list[dict[str, float]]) -> dict[str, float]:
    """The worst third octave of each column of :func:`errors`."""
    return {
        name: float(max(row[name] for row in table))
        for name in ("admittance", "reflection_db", "normal", "random")
    }


def refit_all(materials: list[np.ndarray], fit: WallFit) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Every material of an entry fitted again, and the record of what it cost each."""
    fitted, rows = [], []
    for triplets in materials:
        found = refit(triplets, fit.branches, fmin_hz=fit.fmin_hz, fmax_hz=fit.fmax_hz)
        fitted.append(found)
        rows.append(worst_of(errors(triplets, found, fmin_hz=fit.fmin_hz, fmax_hz=fit.fmax_hz)))
    record = {
        "fit": fit.name(),
        "branches": [int(f.shape[0]) for f in fitted],
        "worst": {
            name: float(max((row[name] for row in rows), default=0.0))
            for name in ("admittance", "reflection_db", "normal", "random")
        }
        if rows
        else {},
        "materials": rows,
    }
    return fitted, record


def refit_kept(
    materials: list[np.ndarray], fit: WallFit, kept: Path | str | None
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """:func:`refit_all`, read from ``kept`` where that file holds this fit of these materials.

    The file is written whole or not at all. It names the fit and a digest
    of the materials it was made from: another fit, or other materials, is
    fitted again and written over it.
    """
    digest = hashlib.sha256(
        b"".join(np.ascontiguousarray(m, dtype=np.float64).tobytes() for m in materials)
    ).hexdigest()
    path = Path(kept) if kept is not None else None
    if path is not None and path.is_file():
        held = json.loads(path.read_text())
        if held.get("fit") == fit.name() and held.get("materials_sha256") == digest:
            return [np.asarray(m, dtype=np.float64) for m in held["triplets"]], held["record"]
    fitted, record = refit_all(materials, fit)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial.write_text(
            json.dumps(
                {
                    "fit": fit.name(),
                    "materials_sha256": digest,
                    "triplets": [m.tolist() for m in fitted],
                    "record": record,
                }
            )
        )
        partial.replace(path)
    return fitted, record


def smallest_count(
    materials: list[np.ndarray],
    counts: tuple[int, ...] = (5, 6, 7, 8),
    *,
    fmin_hz: float = 40.0,
    fmax_hz: float = 1500.0,
) -> tuple[int | None, dict[int, dict[str, float]]]:
    """The fewest branches that hold every material under both bars, and each count's worst."""
    found: dict[int, dict[str, float]] = {}
    chosen = None
    for count in sorted(counts):
        _, record = refit_all(materials, WallFit(count, fmin_hz, fmax_hz))
        found[count] = record["worst"]
        passes = (
            record["worst"]["reflection_db"] <= REFLECTION_BAR_DB
            and max(record["worst"]["normal"], record["worst"]["random"]) <= ABSORPTION_BAR
        )
        if passes and chosen is None:
            chosen = count
    return chosen, found
