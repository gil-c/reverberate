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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra

#: A coefficient may not exceed this: nothing absorbs more than all of the
#: energy arriving at it.
MAX_ABSORPTION = 1.0

#: Compensation beyond this factor is refused. Past it the obstacle has lost
#: so much surface that scaling the coefficient stops being a correction and
#: becomes an invention, and the honest move is to keep more geometry.
MAX_COMPENSATION_FACTOR = 8.0


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
    applied_factor: float
    capped_bands: int

    @property
    def capped(self) -> bool:
        """Whether any band hit the ceiling, i.e. absorbing power was lost."""
        return self.capped_bands > 0

    @property
    def compensated(self) -> bool:
        return self.applied_factor > 1.0

    def summary(self) -> str:
        state = f", {self.capped_bands} band(s) capped" if self.capped else ""
        return (
            f"{self.base_key}: {self.original_area:.2f} -> {self.reduced_area:.2f} m2, "
            f"x{self.applied_factor:.2f}{state}"
        )


def compensate(
    material: pra.Material,
    original_area: float,
    reduced_area: float,
    base_key: str = "",
    max_factor: float = MAX_COMPENSATION_FACTOR,
) -> CompensatedMaterial:
    """Scale a material's absorption so the obstacle keeps its absorbing power.

    Scattering coefficients are left alone: they describe how a surface
    redirects the energy it does not absorb, which is a property of the
    material rather than of how much of it survived decimation.
    """
    if original_area <= 0 or reduced_area <= 0:
        return CompensatedMaterial(material, base_key, original_area, reduced_area, 1.0, 1.0, 0)

    requested = original_area / reduced_area
    applied = float(np.clip(requested, 1.0, max_factor))
    if applied <= 1.0:
        return CompensatedMaterial(
            material, base_key, original_area, reduced_area, requested, 1.0, 0
        )

    coefficients = np.asarray(material.energy_absorption["coeffs"], dtype=float)
    scaled = coefficients * applied
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
        applied_factor=applied,
        capped_bands=capped_bands,
    )


@dataclass(frozen=True)
class AbsorptionAudit:
    """Whether the scheme actually held, in numbers rather than by assertion.

    By construction ``power_before`` and ``power_after`` should agree. Any gap
    is capping or a refused factor, which is the one thing this approach cannot
    do, so the gap is measured and shown rather than assumed away.
    """

    power_before: float
    power_after: float
    obstacles: int
    compensated: int
    capped: int

    @property
    def power_error(self) -> float:
        if self.power_before == 0:
            return 0.0
        return abs(self.power_after - self.power_before) / self.power_before

    def summary(self) -> str:
        return (
            f"absorbing power {self.power_before:.1f} -> {self.power_after:.1f} m2 "
            f"({self.power_error:+.1%}), {self.compensated} of {self.obstacles} obstacles "
            f"compensated, {self.capped} capped"
        )


def audit(
    entries: list[CompensatedMaterial], base_materials: list[pra.Material]
) -> AbsorptionAudit:
    """Total absorbing power before and after compensation.

    ``base_materials`` are the untouched materials, paired with ``entries``, so
    "before" means the obstacle at full surface with its tabulated coefficient
    rather than anything derived from the decimated mesh.
    """
    before = sum(
        absorbing_power(material, entry.original_area)
        for material, entry in zip(base_materials, entries, strict=True)
    )
    after = sum(absorbing_power(entry.material, entry.reduced_area) for entry in entries)
    return AbsorptionAudit(
        power_before=float(before),
        power_after=float(after),
        obstacles=len(entries),
        compensated=sum(1 for entry in entries if entry.compensated),
        capped=sum(1 for entry in entries if entry.capped),
    )
