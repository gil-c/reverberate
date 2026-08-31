"""Populate the catalogue's top two octaves and fit the solver's impedances.

Usage, with PFFDTD's python directory on hand::

    PFFDTD_PYTHON=/path/to/pffdtd/python python -m reverberate.materials

Writes one HDF5 filter per class, a manifest, and a report, under
``<data root>/interim/materials``. Nothing here touches geometry or a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reverberate.acoustics import SOLVER_BANDS
from reverberate.materials.db import acoustic_classes, coverage
from reverberate.materials.extrapolation import (
    MAX_ABSORPTION,
    MEASURED_BANDS,
    MIN_OCTAVE_RATIO,
    layer_model_fit,
)
from reverberate.materials.impedance import ImpedanceFit, fit_all, write_manifest
from reverberate.settings import interim_dir


def _band_label(frequency: float) -> str:
    return f"{frequency / 1000:g}k" if frequency >= 1000 else f"{frequency:g}"


def _extrapolation_table() -> str:
    lines = [
        "| class | 2 kHz | 4 kHz | measured ratio | applied | 8 kHz | 16 kHz |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for material in acoustic_classes().values():
        extension = material.high_bands
        note = " (clipped)" if extension.clipped else ""
        lines.append(
            f"| {material.name} "
            f"| {material.measured_absorption[-2]:.3f} "
            f"| {material.measured_absorption[-1]:.3f} "
            f"| {extension.measured_ratio:.3f} "
            f"| {extension.applied_ratio:.3f}{note} "
            f"| {extension.values[0]:.3f} | {extension.values[1]:.3f} |"
        )
    return "\n".join(lines)


def _layer_model_table() -> str:
    """The diagnostic that rules the layer model out, over every class."""
    lines = [
        "| class | fitted sigma (Pa s/m2) | d (mm) | rms resid | max resid | "
        "model 8 kHz | model 16 kHz | rho f / sigma at 125 Hz .. 16 kHz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for material in acoustic_classes().values():
        fit = layer_model_fit(np.asarray(material.measured_absorption))
        low, high = fit.validity_span
        model_high = fit.high_bands
        lines.append(
            f"| {material.name} | {fit.flow_resistivity:.3g} | {fit.thickness * 1000:.1f} "
            f"| {fit.rms_residual:.3f} | {fit.max_residual:.3f} "
            f"| {model_high[0]:.3f} | {model_high[1]:.3f} "
            f"| {low:.4f} .. {high:.2f} |"
        )
    return "\n".join(lines)


def _impedance_table(fits: list[ImpedanceFit]) -> str:
    lines = [
        "| class | branches | max fit error | rms fit error | max model error | passive | "
        "min Re(Y) | max abs(R) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fit in fits:
        summary = fit.summary()
        lines.append(
            f"| {summary['material_class']} | {summary['branches']} "
            f"| {summary['max_band_error']:.3f} | {summary['rms_band_error']:.3f} "
            f"| {summary['max_model_error']:.3f} "
            f"| {'yes' if summary['passive'] else 'NO'} "
            f"| {summary['min_real_admittance']:.3g} "
            f"| {summary['max_reflection_magnitude']:.4f} |"
        )
    return "\n".join(lines)


def build_report(fits: list[ImpedanceFit]) -> str:
    """The report W4 owes: which classes are extrapolated, by what rule."""
    classes = acoustic_classes()
    moved = [m for m in classes.values() if m.extrapolated]
    held = [m for m in classes.values() if not m.extrapolated]
    layer_fits = {
        name: layer_model_fit(np.asarray(m.measured_absorption)) for name, m in classes.items()
    }
    soft = [name for name, m in classes.items() if max(m.measured_absorption) >= 0.3]
    soft_residuals = [layer_fits[name].rms_residual for name in soft]
    worst = max(fits, key=lambda fit: float(np.max(np.abs(fit.band_error))))
    worst_model = max(fits, key=lambda fit: float(np.max(np.abs(fit.model_error))))
    failures = [fit.name for fit in fits if not fit.passivity.passive]
    largest_held = max(m.solver_absorption[-1] for m in held)
    worst_line = (
        f"Worst fit error: `{worst.name}`, "
        f"{float(np.max(np.abs(worst.band_error))):.3f} in absorption units. "
        f"Worst model error: `{worst_model.name}`, "
        f"{float(np.max(np.abs(worst_model.model_error))):.3f}."
    )
    upholstery_line = (
        f"It puts `upholstery` at {layer_fits['upholstery'].high_bands[0]:.2f} at 8 kHz, "
        f"against a measured {classes['upholstery'].measured_absorption[-1]:.2f} at 4 kHz."
    )
    carpet_line = (
        f"It puts `carpet_thick` at {layer_fits['carpet_thick'].high_bands[0]:.2f}, "
        f"against a measured {classes['carpet_thick'].measured_absorption[-1]:.2f}."
    )
    passivity_line = (
        f"All {len(fits)} materials are passive."
        if not failures
        else "NOT PASSIVE: " + ", ".join(failures)
    )

    return f"""# W4. Materials to 16 kHz

Section 6.1. The catalogue stores what is measured, 125 Hz to 4 kHz for
{len(classes)} classes, and derives every band outside that range. Two octaves
above have to be derived because the architecture runs to 16 kHz and every
source table stops at 4 kHz. Three bands below have to be derived because
PFFDTD's `fit_to_Sabs_oct_11` asks for eleven octave bands from 16 Hz and
asserts on the count.

- Solver bands: {", ".join(_band_label(f) for f in SOLVER_BANDS)} Hz.
- Measured bands: {", ".join(_band_label(f) for f in MEASURED_BANDS)} Hz.

## What changed, and what it replaced

The 8 kHz column used to be the 4 kHz value repeated, and a test asserted it.
Carrying one octave that way was defensible. Carrying two was not, so the
column is gone from `acoustic_classes.csv` and everything above 4 kHz is now
derived in one place, from a rule that is the same for every class.

## The physical model was tried first, and it failed on the measured bands

A rigid-backed Delany-Bazley layer is the obvious way to reach 16 kHz: fit
flow resistivity and thickness to the six measured bands, two parameters
against six points, then read off 8 and 16 kHz. Over-determined, so it can be
judged.

Judged, it does not hold up. Across the {len(soft)} classes with any
appreciable absorption the rms residual on the *measured* bands runs from
{min(soft_residuals):.3f} to {max(soft_residuals):.3f} in absorption units, and
the disagreement is shape rather than scatter: the layer model rises
monotonically towards its plateau, while `upholstery`, `mattress`,
`soft_furnishing`, `plush`, `curtain_heavy`, `human` and `generic_soft` are
already falling between 2 and 4 kHz. Its extrapolations are correspondingly
optimistic. {upholstery_line} {carpet_line}

A model that cannot reproduce the last measured octave has no standing to
predict the next two, so **nothing in the catalogue is taken from it**. It is
kept as a diagnostic and reported below, over all {len(classes)} classes
rather than a hand-picked subset, so that no partition of the catalogue into
"porous" and "not" has to be defended.

{_layer_model_table()}

The `rho f / sigma` column is Delany-Bazley's own validity ratio, stated to
hold from 0.01 to 1.0. Several fits sit outside it at both ends, which is a
second, independent reason not to build the catalogue on this model.

## The rule that is used: continue the material's own measured trend

The ratio between the last two measured bands, 2 and 4 kHz, applied once per
octave above 4 kHz, clipped to at most 1 and at least {MIN_OCTAVE_RATIO}.

- **Never above 1.** A class still rising at 4 kHz is approaching a plateau,
  not accelerating. Continuing the rise would take `curtain_light` from a
  measured 0.35 to about 0.75 at 16 kHz, which nothing measured supports.
- **Never below {MIN_OCTAVE_RATIO}.** The ratio is the quotient of two values
  rounded to two decimals, so it is not precise enough to be projected
  further than that.

The rule is uniform and no class is special-cased. The data then decides:

- **{len(held)} of {len(classes)} classes hold their 4 kHz value**, because
  their own last measured octave is flat or rising. The largest coefficient
  anywhere in the held range is {largest_held:.2f}, and for the hard classes
  the whole extrapolated range is narrower than the spread between two
  published tables for the same material.
- **{len(moved)} classes fall**, by their own measured ratio, between
  {min(m.high_bands.applied_ratio for m in moved):.3f} and
  {max(m.high_bands.applied_ratio for m in moved):.3f} per octave.

{_extrapolation_table()}

**The low bands are held, not modelled.** 16, 31.5 and 63 Hz repeat 125 Hz.
No source table measures below 125 Hz, Delany-Bazley there is outside its
validity range for every flow resistivity fitted above, and the bands are
nearly inert in a project that stops caring below a domestic room's Schroeder
frequency. Stated rather than hidden, exactly as the 4 kHz carry-over was.

## Impedance fit, per class

Fitted with PFFDTD's own `fit_to_Sabs_oct_11` at the pinned commit: eleven RLC
branches per material, one resonance per octave at half-octave bandwidth, with
only the branch magnitudes free. The routine minimises absolute absorption
error over a log-spaced frequency vector rather than interpolating the band
centres, so a fit error of a few hundredths is the routine working as
written.

Two errors are reported and they mean different things. **Fit error** is the
fitted filter's normal-incidence absorption against what the routine was asked
to hit, which is not the catalogue's number: `fit_to_Sabs_oct_11` inverts the
Paris formula first, with `convert_Sabs_to_Yn`, and fits the real admittance
that comes out. That inversion is done here with PFFDTD's own function, so the
column measures the fit alone. **Model error** is the fitted filter averaged
back over incidence angle, against the catalogue coefficient, and is therefore
the end-to-end figure: what a locally reactive boundary with a real target
admittance costs, which is a property of the model and not of the fit.

{_impedance_table(fits)}

{worst_line}

## Passivity

{passivity_line}

The check is not the argument that non-negative `D`, `E`, `F` implies a
non-negative real part. It is an evaluation of the admittance on 4000
log-spaced frequencies from 1 Hz to 24 kHz, wider than the solver's band of
interest because the filter runs everywhere and not only where the material
was specified, reporting the worst real part seen and the largest reflection
magnitude.

**The strongest absorbers are the ones the boundary model cannot reach.**
Model error is largest for `mattress`, `acoustic_ceiling`, `soft_furnishing`
and `upholstery`, and it is not a fitting failure: a locally reactive boundary
driven by a real target admittance cannot reproduce a coefficient near 1, which
is the same limit PFFDTD announces when it clips any input above
{MAX_ABSORPTION}. Three of the catalogue's measured values sit at or above that
clip, all of them on `acoustic_ceiling`. Where a scene's reverberation is
carried by one of these classes, the solver is a few hundredths less absorbing
than the catalogue says, in the direction of a slightly livelier room.

## What the report says about these numbers

**Above 4 kHz nothing in this catalogue is measured.** For a tile that is
uninteresting: the coefficient is 0.02 and the extrapolated range is narrower
than the disagreement between two published tables. For a carpet or a sofa it
is a projection of a measured trend with a stated per-octave ratio, and it
should be quoted as one, per class, from the table above.

**No fidelity claim above 4 kHz can be sized yet.** W3 has not run, so the
solver's own noise floor is unknown and there is nothing to compare the size
of this uncertainty against. A statement of the form "the 16 kHz band is
within the JND" is not interpretable until it has.

**Phase is not measured for any material here.** An absorption coefficient
carries none, so every phase in these filters comes from the resonant model
the fit assumes. Delany-Bazley would have grounded phase in a flow
resistivity, which is the one thing lost by rejecting it, but it would have
grounded it in a fit that misses the measured magnitudes by up to
{max(soft_residuals):.2f}, and the solver receives the result through the same
phase-free routine either way.

## Coverage

{coverage()}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write filters, manifest and report (default <data root>/interim/materials)",
    )
    arguments = parser.parse_args(argv)
    output_dir = arguments.output_dir or interim_dir("materials")
    output_dir.mkdir(parents=True, exist_ok=True)

    fits = fit_all(output_dir)
    write_manifest(fits, output_dir)
    report = output_dir / "report.md"
    report.write_text(build_report(fits))
    print(f"{len(fits)} materials fitted; report at {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
