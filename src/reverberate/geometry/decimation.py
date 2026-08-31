"""How much of an obstacle's geometry the simulator actually needs.

The previous rule was a flat 150 faces per obstacle regardless of what the
obstacle was. Measured on twelve real HSSD colliders, that cost **75 %, 78 %,
50 % and 37 % of surface area** on the worst cases, up to 90 % of volume, and
broke watertightness on three of twelve. Surface area is not a cosmetic
property here: absorption is applied per unit area, and Sabine puts RT60
proportional to V/(S·α), so removing three quarters of a sofa's surface
removes three quarters of its absorption and biases the room towards sounding
too reverberant. A flat budget also treats a wardrobe and a vase alike.

Two things replace it.

**A budget derived from the object's own size and from physics.** Section 5.3
of the brief argues that detail much smaller than the shortest wavelength of
interest does not reflect specularly, it scatters, and scattering is already
modelled by the scattering coefficient. At 8 kHz, the top of the band range in
``reverberate.acoustics``, that wavelength is about 4.3 cm, and that is the
floor: decimating past it stops being the argument the README makes and starts
being damage. The budget is therefore the number of
triangles of roughly that edge length needed to cover the object's surface.

**Levels of detail, so distance can pay for itself.** What is far away, or
behind a wall in another room, can be very coarse: by the time it contributes,
the energy is diffuse and filtered, and no listener resolves the shape of a
chair two rooms away. What is close keeps its detail. ``DETAIL_LEVELS`` is
that ladder, and it is data rather than logic so the viewer can be handed the
same table and pick the same level for the same listener position.

Every reduction is checked, per section 5.3, and the checks are the point: a
flipped normal after decimation raises no error anywhere, it just makes the
ray tracer run for minutes on geometry it cannot resolve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from reverberate.acoustics import MIN_WAVELENGTH

#: No obstacle is reduced below this, whatever its size. A handful of triangles
#: cannot enclose a volume, and an obstacle that has lost its volume has lost
#: the surface that absorbs.
MIN_OBSTACLE_FACES = 24

#: Ceiling per obstacle, so that one pathological asset cannot dominate the
#: wall count on its own.
MAX_OBSTACLE_FACES = 4000


@dataclass(frozen=True)
class DetailLevel:
    """One rung of the level-of-detail ladder.

    ``detail_length`` is the smallest feature, in metres, worth keeping at this
    level. ``MIN_WAVELENGTH`` is the finest rung on purpose: nothing is ever
    resolved below the physical limit, however close it is.
    """

    name: str
    detail_length: float


#: The ladder, finest first. Selection is by where the obstacle sits relative
#: to the listener, and lives in ``level_for`` so that Python and the viewer
#: can be shown to agree.
DETAIL_LEVELS: tuple[DetailLevel, ...] = (
    DetailLevel("near", MIN_WAVELENGTH),
    DetailLevel("room", MIN_WAVELENGTH * 2.0),
    DetailLevel("far", MIN_WAVELENGTH * 4.0),
)

#: Distance, in metres, beyond which an obstacle in the listener's own room
#: drops from "near" to "room".
NEAR_DISTANCE = 4.0


def level_for(distance: float, same_room: bool) -> DetailLevel:
    """Which rung an obstacle sits on, given where the listener is.

    Being in another room dominates distance: a wall between the listener and
    the object has already turned whatever it reflects into diffuse energy.
    """
    if not same_room:
        return DETAIL_LEVELS[2]
    if distance <= NEAR_DISTANCE:
        return DETAIL_LEVELS[0]
    return DETAIL_LEVELS[1]


def face_budget(mesh: trimesh.Trimesh, detail_length: float) -> int:
    """How many triangles it takes to cover this surface at this resolution.

    Proportional to the object's own area rather than fixed, so a wardrobe and
    a vase are not given the same budget. Two triangles tile a square of side
    ``detail_length``, hence the factor of two.
    """
    if detail_length <= 0:
        raise ValueError("detail_length must be positive")
    area = float(mesh.area)
    budget = math.ceil(2.0 * area / (detail_length**2))
    return int(min(max(budget, MIN_OBSTACLE_FACES), MAX_OBSTACLE_FACES))


@dataclass(frozen=True)
class DecimationReport:
    """The section 5.3 validation, as numbers rather than as a claim."""

    faces_before: int
    faces_after: int
    area_error: float
    volume_error: float
    watertight_before: bool
    watertight_after: bool
    normals_consistent: bool

    @property
    def lost_watertightness(self) -> bool:
        return self.watertight_before and not self.watertight_after

    def acceptable(self, area_tolerance: float = 0.15, volume_tolerance: float = 0.25) -> bool:
        """Whether the reduction kept the properties the simulation depends on.

        Area is held to a tighter tolerance than volume deliberately: it is
        area that carries absorption. Volume matters for the shell, and an
        obstacle's volume mostly matters because losing it means the surface
        went with it.
        """
        return (
            not self.lost_watertightness
            and self.normals_consistent
            and self.area_error <= area_tolerance
            and self.volume_error <= volume_tolerance
        )

    def summary(self) -> str:
        return (
            f"{self.faces_before} -> {self.faces_after} faces, "
            f"area {self.area_error:+.1%}, volume {self.volume_error:+.1%}, "
            f"watertight {self.watertight_before}->{self.watertight_after}, "
            f"normals {'ok' if self.normals_consistent else 'INCONSISTENT'}"
        )


def _relative_error(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return abs(after - before) / abs(before)


def validate(original: trimesh.Trimesh, reduced: trimesh.Trimesh) -> DecimationReport:
    """The three checks section 5.3 asks for, on one reduction.

    The normals check is not a formality. A mesh whose winding was flipped made
    the ray tracer run for over seven minutes on **twelve triangles** instead
    of failing, so an unchecked flip does not surface as an error, it surfaces
    as a job that never finishes.
    """
    return DecimationReport(
        faces_before=len(original.faces),
        faces_after=len(reduced.faces),
        area_error=_relative_error(float(original.area), float(reduced.area)),
        volume_error=_relative_error(float(original.volume), float(reduced.volume)),
        watertight_before=bool(original.is_watertight),
        watertight_after=bool(reduced.is_watertight),
        normals_consistent=bool(reduced.is_winding_consistent) and float(reduced.volume) >= 0.0,
    )


def decimate_to(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh:
    """Reduce to a face count, or return the mesh unchanged if it cannot be.

    ``simplify_quadric_decimation`` exposes no seed. It is believed
    deterministic, being a greedy quadric collapse, but that is an assumption
    rather than a measurement, so the cross-process determinism test in
    ``tests/test_decimation.py`` exercises this path end to end instead of
    trusting it.
    """
    if len(mesh.faces) <= budget:
        return mesh
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=budget)
    except Exception:
        # An obstacle the decimator cannot handle is passed through whole
        # rather than dropped: an expensive obstacle beats a missing one.
        return mesh
    if not isinstance(reduced, trimesh.Trimesh) or len(reduced.faces) == 0:
        return mesh
    return reduced


def decimate_adaptive(
    mesh: trimesh.Trimesh,
    level: DetailLevel,
    area_tolerance: float = 0.15,
    volume_tolerance: float = 0.25,
) -> tuple[trimesh.Trimesh, DecimationReport]:
    """Reduce an obstacle as far as this level allows, but no further than is safe.

    The budget is a target, not an order. If meeting it costs more area or
    volume than the tolerance permits, the reduction is backed off step by step
    and the least-reduced acceptable result is kept. A reduction that cannot be
    made acceptable at any step leaves the mesh untouched, which is the honest
    outcome: better a costly obstacle than a cheap one that no longer absorbs
    what it should.
    """
    budget = face_budget(mesh, level.detail_length)
    if len(mesh.faces) <= budget:
        return mesh, validate(mesh, mesh)

    attempt = budget
    while attempt < len(mesh.faces):
        reduced = decimate_to(mesh, attempt)
        report = validate(mesh, reduced)
        if report.acceptable(area_tolerance, volume_tolerance):
            return reduced, report
        attempt *= 2

    return mesh, validate(mesh, mesh)


def summarise(reports: list[DecimationReport]) -> str:
    """Dataset-level view, for the Phase 1 report's rejection statistics."""
    if not reports:
        return "no obstacles"
    before = sum(report.faces_before for report in reports)
    after = sum(report.faces_after for report in reports)
    unacceptable = sum(1 for report in reports if not report.acceptable())
    worst_area = max(report.area_error for report in reports)
    return (
        f"{len(reports)} obstacles, {before} -> {after} faces "
        f"({100.0 * after / before:.0f}% kept), worst area loss {worst_area:.1%}, "
        f"{unacceptable} not within tolerance"
    )


def mean_edge_length(mesh: trimesh.Trimesh) -> float:
    """Average triangle edge, in metres. Used to check the physical floor."""
    if len(mesh.faces) == 0:
        return 0.0
    return float(np.mean(mesh.edges_unique_length))


def deviation(
    original: trimesh.Trimesh, reduced: trimesh.Trimesh, samples: int = 2000, seed: int = 0
) -> float:
    """How far the reduced mesh strays from the original, in metres.

    The 95th percentile of the distance from points sampled on the original
    surface to the reduced one. A percentile rather than a maximum because a
    single sliver triangle should not veto a reduction that is otherwise well
    inside the physical limit.

    ``seed`` is not decoration. Without it ``sample_surface`` draws from OS
    entropy, and since this number is the accept-or-reject test in
    :func:`decimate_within`, two processes handed the same scene accepted
    different face budgets: 608 098 triangles against 614 330 on one apartment.
    The sample is the only randomness in the geometry pipeline, so pinning it
    makes the export reproducible across processes and not merely within one.
    """
    if len(reduced.faces) == 0 or len(original.faces) == 0:
        return float("inf")
    points, _ = trimesh.sample.sample_surface(original, samples, seed=seed)
    _, distances, _ = trimesh.proximity.closest_point(reduced, points)  # type: ignore[no-untyped-call]
    return float(np.percentile(np.abs(distances), 95))


def decimate_within(
    mesh: trimesh.Trimesh,
    max_deviation: float = MIN_WAVELENGTH / 2.0,
    candidates: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048),
    seed: int = 0,
) -> tuple[trimesh.Trimesh, float]:
    """Reduce to the fewest triangles that still stay within ``max_deviation``.

    This is section 5.3's argument applied directly, instead of through a face
    count derived from surface area. That derivation assumed the object had to
    be tessellated uniformly at the wavelength, which is not what the argument
    says: it bounds the smallest *feature* worth keeping, and a flat panel has
    no features at all, so it needs two triangles however large it is.
    Measured on real furniture, the area-derived rule asked for 866 to 4000
    faces where 100 already sat between 0.6 cm and 2.5 cm of a 8.6 cm limit.

    Returns the reduced mesh and its measured deviation, so the caller can
    record what the reduction actually cost rather than trusting the target.
    """
    if len(mesh.faces) <= min(candidates):
        return mesh, 0.0
    for target in candidates:
        if target >= len(mesh.faces):
            break
        reduced = decimate_to(mesh, target)
        if len(reduced.faces) >= len(mesh.faces):
            continue
        error = deviation(mesh, reduced, seed=seed)
        if error <= max_deviation:
            return reduced, error
    return mesh, 0.0
