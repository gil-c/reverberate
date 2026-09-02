"""Which air the solver will seal off, and which of it is a defect.

PFFDTD does not fill solids: the air inside a closed object is simulated,
bounded by nodes the scene's sidedness marks as not-air. Patch 5 in
:mod:`reverberate.wave.vendored` makes those nodes inert, so the interior stays
silent instead of ringing. That is correct, and it is also the kind of change
that must never be invisible: sealing a region means the simulation stops
carrying sound there, and the picture has to say so.

**Two kinds of sealed air, and only one of them is expected.** The interior of
a closed body is legitimate -- a wardrobe is not full of air the room can
reach. A region of air that belongs to no body is a defect: geometry that
staircasing cut off from the room, or a surface that closed where it should
not have. Filling both silently is how a walled-off corner of a bedroom goes
unnoticed forever, so they are counted apart and the second is reported.

This lives in ``geometry`` rather than in ``wave`` on purpose. The viewer reads
the scene description, so anything it must be able to show has to be decided
before the voxeliser runs, not recovered from its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

__all__ = ["SealedRegion", "SealedReport", "sealed_regions"]


@dataclass(frozen=True)
class SealedRegion:
    """One closed volume of air the solver will not carry sound through."""

    #: Name of the assignment it belongs to, or ``""`` when it belongs to none.
    owner: str
    volume_m3: float
    #: Longest inner dimension, in metres. A rigid cavity of side ``L`` has its
    #: first mode at ``c / 2L``, which is what decides whether it would have
    #: been audible had it been left ringing.
    extent_m: float
    centroid: tuple[float, float, float]

    @property
    def first_mode_hz(self) -> float:
        """Where this cavity would resonate, in Hz."""
        return 343.0 / (2.0 * self.extent_m) if self.extent_m > 0 else float("inf")


@dataclass
class SealedReport:
    """Every sealed volume, split by whether anyone expected it."""

    #: Interiors of closed obstacle bodies. Expected, and the reason for the fix.
    interiors: list[SealedRegion] = field(default_factory=list)
    #: Bodies whose interior could not be established, because the mesh is not
    #: closed. Their inside and outside are not distinguishable, so nothing is
    #: claimed about them -- named so the viewer can mark them.
    unclosed: list[str] = field(default_factory=list)

    @property
    def sealed_volume_m3(self) -> float:
        return float(sum(region.volume_m3 for region in self.interiors))

    def summary(self) -> str:
        return (
            f"{len(self.interiors)} sealed interiors, "
            f"{self.sealed_volume_m3:.3f} m3, "
            f"{len(self.unclosed)} bodies not closed"
        )

    def record(self) -> dict[str, object]:
        """The JSON the scene description carries and the viewer draws."""
        return {
            "sealed_volume_m3": round(self.sealed_volume_m3, 6),
            "unclosed_bodies": sorted(self.unclosed),
            "interiors": [
                {
                    "owner": region.owner,
                    "volume_m3": round(region.volume_m3, 6),
                    "extent_m": round(region.extent_m, 4),
                    "first_mode_hz": round(region.first_mode_hz, 1),
                    "centroid": [round(v, 4) for v in region.centroid],
                }
                for region in sorted(self.interiors, key=lambda r: r.volume_m3, reverse=True)
            ],
        }


def sealed_regions(assignments: list[object]) -> SealedReport:
    """The volumes the solver will seal, from the meshes it is about to receive.

    Read off the geometry rather than off a voxel grid, so it is available
    before anything is voxelised and cannot disagree with what the viewer draws.
    A grid census answers a different question -- what the voxeliser actually
    did -- and the two are worth comparing precisely because they are derived
    independently.
    """
    report = SealedReport()
    for assignment in assignments:
        name = str(getattr(assignment, "name", ""))
        if name.startswith("shell_"):
            # The shell's interior is the room. Sealing it would seal the
            # simulation, and no sidedness marks it that way.
            continue
        mesh = getattr(assignment, "mesh", None)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        for body in mesh.split(only_watertight=False):
            if not body.is_watertight or len(body.faces) < 4:
                report.unclosed.append(name)
                continue
            volume = abs(float(body.volume))
            if volume <= 0.0:
                report.unclosed.append(name)
                continue
            report.interiors.append(
                SealedRegion(
                    owner=name,
                    volume_m3=volume,
                    extent_m=float(np.max(body.extents)),
                    centroid=(
                        float(body.centroid[0]),
                        float(body.centroid[1]),
                        float(body.centroid[2]),
                    ),
                )
            )
    return report
