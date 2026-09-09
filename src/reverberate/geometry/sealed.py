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

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import trimesh

from reverberate.geometry.orientation import is_closed

__all__ = ["REPORT_MIN_VOLUME_M3", "SealedRegion", "SealedReport", "sealed_regions"]

#: Smallest interior :meth:`SealedReport.record` lists one by one, in cubic
#: metres. One litre.
#:
#: Not a display preference: the record is embedded in the run page and the
#: page shows eight rows. A carve on a 2 mm cell splits a plant into thousands
#: of closed leaves, and every one of them is an interior -- measured on this
#: flat, **11 278 regions of which 10 899 are under a tenth of a litre and
#: together hold 24.6 litres of 26 840**. Listing them took the report from
#: 0.18 MB to 2.54 MB for rows nobody reads.
#:
#: A litre is where the question stops being answerable anyway. Such a cavity is
#: about 10 cm across and would have rung at 1.7 kHz, where a tenth of a litre
#: of air holds no energy worth naming; the whole point of the row is the
#: 125 Hz boom of a wardrobe, and that is three orders of magnitude away. What
#: is dropped is *counted*, in ``interiors_omitted``, so the total is never
#: implied by a truncated list.
REPORT_MIN_VOLUME_M3 = 0.001


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
    #: Bodies whose interior could not be established: either the mesh is not
    #: closed, so inside and outside are not distinguishable, or its convex
    #: decomposition could not be unioned into one solid, so its bodies still
    #: overlap and a per-body volume would double-count the overlap. Nothing
    #: is claimed about them either way -- named so the viewer can mark them.
    #:
    #: One entry per *body*, so an assignment appears once for each of its own
    #: that could not be judged. :meth:`record` publishes the distinct owners
    #: and the per-owner counts separately, because the two answer different
    #: questions and conflating them made the page say something false: a
    #: carve on a 2 mm cell splits a christmas tree into thousands of closed
    #: twigs and a few thousand slivers of under four faces, and the raw length
    #: of this list then read as "32 300 bodies are not closed" for 195 objects.
    unclosed: list[str] = field(default_factory=list)

    @property
    def sealed_volume_m3(self) -> float:
        return float(sum(region.volume_m3 for region in self.interiors))

    def summary(self) -> str:
        return (
            f"{len(self.interiors)} sealed interiors, "
            f"{self.sealed_volume_m3:.3f} m3, "
            f"{len(self.unclosed)} bodies not closed over "
            f"{len(set(self.unclosed))} objects"
        )

    def record(self) -> dict[str, object]:
        """The JSON the scene description carries and the viewer draws."""
        counts = Counter(self.unclosed)
        ordered = sorted(self.interiors, key=lambda r: r.volume_m3, reverse=True)
        listed = [r for r in ordered if r.volume_m3 >= REPORT_MIN_VOLUME_M3]
        omitted = ordered[len(listed) :]
        return {
            "sealed_volume_m3": round(self.sealed_volume_m3, 6),
            # Every region, so a reader counting the list below does not mistake
            # it for the total. The page prints this, not ``len(interiors)``.
            "interior_count": len(self.interiors),
            "interiors_omitted": {
                "count": len(omitted),
                "volume_m3": round(sum(r.volume_m3 for r in omitted), 6),
                "below_m3": REPORT_MIN_VOLUME_M3,
            },
            # The distinct owners, so a count of this list is a count of
            # objects. The viewer prints its length in a sentence about
            # objects, and one name per unjudged body made that sentence
            # wrong by two orders of magnitude.
            "unclosed_bodies": sorted(counts),
            # And how many bodies each of them contributed, for a reader who
            # wants to know whether an object is one open shell or a thousand
            # slivers the carve left behind.
            "unclosed_body_counts": dict(sorted(counts.items())),
            "interiors": [
                {
                    "owner": region.owner,
                    "volume_m3": round(region.volume_m3, 6),
                    "extent_m": round(region.extent_m, 4),
                    "first_mode_hz": round(region.first_mode_hz, 1),
                    "centroid": [round(v, 4) for v in region.centroid],
                }
                for region in listed
            ],
        }


def sealed_regions(
    assignments: list[object], unmerged: frozenset[str] = frozenset()
) -> SealedReport:
    """The volumes the solver will seal, from the meshes it is about to receive.

    Read off the geometry rather than off a voxel grid, so it is available
    before anything is voxelised and cannot disagree with what the viewer draws.
    A grid census answers a different question -- what the voxeliser actually
    did -- and the two are worth comparing precisely because they are derived
    independently.

    ``unmerged`` names the assignments whose convex bodies could not be
    unioned into one outer surface (see
    :func:`~reverberate.geometry.sim_geometry.obstacle_assignments`), so they
    still carry their original, overlapping convex decomposition. Per-body
    volumes on a mesh like that are not a claim this function can make: the
    bodies interpenetrate by construction, so summing them double-counts the
    overlap. Such an assignment is reported as unclosed rather than as a set
    of interiors that would silently misstate the sealed volume.
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
        if name in unmerged:
            report.unclosed.append(name)
            continue
        for body in mesh.split(only_watertight=False):
            # Closed, not edge-manifold: the same distinction
            # :func:`~reverberate.geometry.orientation.is_closed` was introduced
            # for. A body the solver seals is one whose inside is
            # distinguishable from its outside, and a handful of non-manifold
            # edges on a carved isosurface does not change that. Asking for
            # watertightness here would report every carved object as unclosed
            # while the solver went on sealing it, which is the picture and the
            # simulation disagreeing -- the one thing this module exists to
            # prevent.
            if not is_closed(body) or len(body.faces) < 4:
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
