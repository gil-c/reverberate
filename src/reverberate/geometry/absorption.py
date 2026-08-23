"""Keeping an obstacle's absorbing power after its surface has been decimated.

Decimation removes surface area. Absorption is applied per unit area, so a
sofa reduced to a quarter of its surface absorbs a quarter of what it should,
and the room comes out too reverberant. Preserving the area instead costs
roughly four times the triangles, which makes the simulation slower rather
than faster: measured at 36 118 to 168 484 obstacle faces on scene
``102344049``, against a run already dominated by one ``wall_factory`` call
per triangle.

The way out is that neither the diffuse field nor the analytic estimates
integrate area on its own. What they integrate is **absorbing power**, the
product α·S: Sabine and Eyring both put RT60 in terms of S·α, and the ray
tracer accumulates energy removed per reflection, which is the same product.
So the obstacle can keep its reduced surface and have its coefficient scaled
by ``original_area / reduced_area``: the product is preserved exactly, and the
wall count stays low.

**What this does not do, and must never be presented as doing**: it restores
the *statistical* absorption, not the specular geometry. A compensated sofa
removes the right amount of energy from the room but no longer returns the
same early reflections. That is the trade section 5.3's wavelength argument
already accepts, but it is a trade, and it belongs in the report.

**Where it breaks, and why that is reported rather than hidden**: a coefficient
cannot exceed 1, because a surface cannot absorb more than everything that
reaches it. When the required factor would push a band past 1 the band is
capped and absorbing power really is lost. Those objects are counted, flagged
to the viewer and totalled in :class:`AbsorptionAudit`, because they are
exactly the places where the dataset stops telling the truth.
**Compensation is weighted per band, because the lost surface has a size.**
Decimation and enveloping remove detail of a measured characteristic size
``d`` — the deviation the envelope was accepted at. A feature only ever
reflected where the wavelength was comparable to it or shorter, so restoring
its absorbing power at 125 Hz, where the wavelength is 274 cm, credits the
object with absorption it never provided. The weight is ``min(1, d / lambda)``:
full compensation once the wavelength is at or below the feature, tapering
linearly below it. Linear rather than the fourth power a Rayleigh scattering
argument would give, because the exact transition is not known and the linear
form errs towards the previous behaviour rather than towards a large
unvalidated change. Weighting also removes most of the capping, since the
bands that used to saturate are the low ones, where the coefficients are
smallest and the factor is now nearest to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra

from reverberate.acoustics import wavelengths

#: A coefficient may not exceed this: nothing absorbs more than all of the
#: energy arriving at it.
MAX_ABSORPTION = 1.0

#: Compensation beyond this factor is refused. Past it the obstacle has lost
#: so much surface that scaling the coefficient stops being a correction and
#: becomes an invention, and the honest move is to keep more geometry.
MAX_COMPENSATION_FACTOR = 8.0


def band_weights(feature_size: float, band_count: int) -> np.ndarray:
    """How much of the compensation each band should receive, in [0, 1].

    ``feature_size`` is the characteristic size, in metres, of the surface
    detail that was removed — in practice the deviation the envelope was
    accepted at, which is measured rather than assumed. A band whose wavelength
    is at or below that size sees the detail and gets the full factor; below it
    the detail is progressively invisible and so is its absorption.

    ``band_count`` is honoured rather than assumed to be seven: a material
    built from a single number carries one coefficient, and compensating it as
    if it were an octave band would silently promote it.
    """
    if feature_size <= 0:
        return np.ones(band_count)
    if band_count != len(wavelengths()):
        # A scalar (or otherwise non-octave) material has no frequency to weigh
        # against, so weighting it would be an invention. Full compensation is
        # the previous, conservative behaviour.
        return np.ones(band_count)
    return np.clip(feature_size / wavelengths(), 0.0, 1.0)


def _band_dict(coeffs: list[float], source: dict[str, object]) -> dict[str, object]:
    """Rebuild a coefficient dict, carrying band centres only when there are any.

    A material built from a single number has no ``center_freqs`` at all, while
    a table material has seven. Passing the key through unconditionally turns
    the scalar case into a ``KeyError``, and inventing band centres for it
    would silently promote it to multi-band, which is exactly the mismatch that
    once made a whole apartment fail with "All walls should have the same
    number of frequency bands".
    """
    rebuilt: dict[str, object] = {"coeffs": coeffs}
    centres = source.get("center_freqs")
    if centres is not None:
        rebuilt["center_freqs"] = list(centres)  # type: ignore[call-overload]
    return rebuilt


def absorbing_power(material: pra.Material, area: float) -> float:
    """The quantity that is actually conserved: mean α times surface, in m².

    This is what Sabine and Eyring consume and what the ray tracer removes per
    reflection, which is why it is the thing worth preserving when the
    geometry underneath it changes.
    """
    return float(np.mean(material.energy_absorption["coeffs"])) * float(area)


def absorbing_power_per_band(material: pra.Material, area: float) -> np.ndarray:
    """Absorbing power band by band, in m².

    The mean above hides exactly what per-band compensation changes: weighting
    the low bands down and leaving the high ones alone moves the average, and
    an average is not enough to tell that apart from a band that capped.
    """
    return np.asarray(material.energy_absorption["coeffs"], dtype=float) * float(area)


@dataclass(frozen=True)
class CompensatedMaterial:
    """A material rescaled for a decimated obstacle, with its own provenance.

    Everything needed to explain the number to a human is carried here rather
    than recomputed elsewhere, so the viewer, the audit panel and the
    simulator all quote the same figures.
    """

    material: pra.Material
    base_key: str
    original_area: float
    reduced_area: float
    requested_factor: float
    band_factors: np.ndarray
    capped_bands: int
    feature_size: float = 0.0

    @property
    def applied_factor(self) -> float:
        """The largest factor any band received.

        The worst case rather than the mean, because it is the number that says
        how far this obstacle's coefficients were moved from their tabulated
        values, and that is what a reader is entitled to challenge.
        """
        if self.band_factors.size == 0:
            return 1.0
        return float(np.max(self.band_factors))

    @property
    def capped(self) -> bool:
        """Whether any band hit the ceiling, i.e. absorbing power was lost."""
        return self.capped_bands > 0

    @property
    def compensated(self) -> bool:
        return self.applied_factor > 1.0

    def summary(self) -> str:
        state = f", {self.capped_bands} band(s) capped" if self.capped else ""
        spread = ""
        if self.band_factors.size > 1 and not np.allclose(self.band_factors, self.band_factors[0]):
            spread = f" (x{float(np.min(self.band_factors)):.2f} at 125 Hz)"
        return (
            f"{self.base_key}: {self.original_area:.2f} -> {self.reduced_area:.2f} m2, "
            f"x{self.applied_factor:.2f}{spread}{state}"
        )


def compensate(
    material: pra.Material,
    original_area: float,
    reduced_area: float,
    base_key: str = "",
    max_factor: float = MAX_COMPENSATION_FACTOR,
    feature_size: float = 0.0,
) -> CompensatedMaterial:
    """Scale a material's absorption so the obstacle keeps its absorbing power.

    ``feature_size`` is the characteristic size, in metres, of the detail the
    reduction removed. Pass it and the factor is weighted per band by
    :func:`band_weights`, so a plant's lost leaf area is restored at 8 kHz and
    left alone at 125 Hz. Leaving it at zero compensates every band equally,
    which is what the caller wants when the removed detail has no known size.

    Scattering coefficients are left alone: they describe how a surface
    redirects the energy it does not absorb, which is a property of the
    material rather than of how much of it survived decimation.
    """
    coefficients = np.asarray(material.energy_absorption["coeffs"], dtype=float)
    if original_area <= 0 or reduced_area <= 0:
        return CompensatedMaterial(
            material,
            base_key,
            original_area,
            reduced_area,
            1.0,
            np.ones(coefficients.size),
            0,
            feature_size,
        )

    requested = original_area / reduced_area
    applied = float(np.clip(requested, 1.0, max_factor))
    weights = band_weights(feature_size, coefficients.size)
    factors = 1.0 + (applied - 1.0) * weights
    if np.allclose(factors, 1.0):
        return CompensatedMaterial(
            material,
            base_key,
            original_area,
            reduced_area,
            requested,
            np.ones(coefficients.size),
            0,
            feature_size,
        )

    scaled = coefficients * factors
    capped_bands = int(np.sum(scaled > MAX_ABSORPTION))
    scaled = np.clip(scaled, 0.0, MAX_ABSORPTION)

    rescaled = pra.Material(
        energy_absorption=_band_dict(scaled.tolist(), material.energy_absorption),
        scattering=_band_dict(list(material.scattering["coeffs"]), material.scattering),
    )
    return CompensatedMaterial(
        material=rescaled,
        base_key=base_key,
        original_area=original_area,
        reduced_area=reduced_area,
        requested_factor=requested,
        band_factors=factors,
        capped_bands=capped_bands,
        feature_size=feature_size,
    )


@dataclass(frozen=True)
class AbsorptionAudit:
    """Whether the scheme actually held, in numbers rather than by assertion.

    Where compensation is unweighted, ``power_before`` and ``power_after``
    agree by construction and any gap is capping or a refused factor.

    Once compensation is weighted per band the two are **meant** to differ, and
    only in one direction: the low bands are deliberately left short because
    the detail that was removed never absorbed anything there. That makes the
    scalar totals no longer sufficient on their own, so the per-band figures
    are carried too. A shortfall confined to the low bands is the correction
    working; a shortfall at 8 kHz is capping and is a defect.
    """

    power_before: float
    power_after: float
    obstacles: int
    compensated: int
    capped: int
    band_power_before: np.ndarray | None = None
    band_power_after: np.ndarray | None = None

    @property
    def power_error(self) -> float:
        if self.power_before == 0:
            return 0.0
        return abs(self.power_after - self.power_before) / self.power_before

    @property
    def band_power_error(self) -> np.ndarray | None:
        """Per-band shortfall, signed, as a fraction of the tabulated power."""
        if self.band_power_before is None or self.band_power_after is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            error = (self.band_power_after - self.band_power_before) / self.band_power_before
        return np.asarray(np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    @property
    def top_band_error(self) -> float:
        """The shortfall at 8 kHz, where nothing should ever be weighted away.

        This is the number that says whether capping is happening, separated
        from the intentional low-band shortfall that would otherwise mask it.
        """
        error = self.band_power_error
        if error is None or error.size == 0:
            return self.power_after / self.power_before - 1.0 if self.power_before else 0.0
        return float(error[-1])

    def summary(self) -> str:
        bands = ""
        error = self.band_power_error
        if error is not None and error.size:
            bands = f", {error[0]:+.1%} at 125 Hz and {error[-1]:+.1%} at 8 kHz"
        return (
            f"absorbing power {self.power_before:.1f} -> {self.power_after:.1f} m2 "
            f"({self.power_error:+.1%}){bands}, {self.compensated} of {self.obstacles} obstacles "
            f"compensated, {self.capped} capped"
        )


def audit(
    entries: list[CompensatedMaterial], base_materials: list[pra.Material]
) -> AbsorptionAudit:
    """Total absorbing power before and after compensation.

    ``base_materials`` are the untouched materials, paired with ``entries``, so
    "before" means the obstacle at full surface with its tabulated coefficient
    rather than anything derived from the decimated mesh.

    Per-band totals are only accumulated where every obstacle carries the same
    number of bands, which is the case for the material table but not for a
    material built from a single number; a mixed list falls back to the scalar
    totals rather than summing arrays of different lengths.
    """
    before = sum(
        absorbing_power(material, entry.original_area)
        for material, entry in zip(base_materials, entries, strict=True)
    )
    after = sum(absorbing_power(entry.material, entry.reduced_area) for entry in entries)

    band_before: np.ndarray | None = None
    band_after: np.ndarray | None = None
    widths = {entry.band_factors.size for entry in entries}
    if len(widths) == 1 and widths != {0}:
        band_before = np.sum(
            [
                absorbing_power_per_band(material, entry.original_area)
                for material, entry in zip(base_materials, entries, strict=True)
            ],
            axis=0,
        )
        band_after = np.sum(
            [absorbing_power_per_band(entry.material, entry.reduced_area) for entry in entries],
            axis=0,
        )

    return AbsorptionAudit(
        power_before=float(before),
        power_after=float(after),
        obstacles=len(entries),
        compensated=sum(1 for entry in entries if entry.compensated),
        capped=sum(1 for entry in entries if entry.capped),
        band_power_before=band_before,
        band_power_after=band_after,
    )
