"""The outer surface of an obstacle, as sound actually meets it.

HSSD ships collision meshes as convex decompositions: one bed is **323
separate convex bodies**, a car 254, and 80 % of a scene's instances are in
that form. Summing their triangle areas counts every face where two convex
pieces meet, and those faces are *inside* the solid, where sound never
reaches. Measured on one apartment, that inflates furniture surface from
698 m² to 1316 m², a factor of 1.9, and on the worst object from 20 m² to
138 m². Every downstream quantity built on that area is wrong by the same
factor: the face budget, the absorbing power, and the compensation meant to
conserve it.

Taking the convex hull of the whole object fixes the area but flattens genuine
concavity, and a bookshelf or a corner sofa loses a cavity that matters. So the
envelope is fitted rather than assumed: the hull is tried first, its deviation
from the real surface is *measured*, and the object is only split into more
hulls when that measurement says it has to be. Convexity is thus a conclusion
about each object, not a blanket assumption about furniture.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import trimesh

from reverberate.acoustics import MIN_WAVELENGTH
from reverberate.geometry.decimation import decimate_within, deviation

#: How far an envelope may sit from the real surface, in metres. Half the
#: shortest wavelength of interest: detail finer than this scatters rather than
#: reflects specularly, which is section 5.3's whole argument.
MAX_ENVELOPE_DEVIATION = MIN_WAVELENGTH / 2.0

#: Most parts an object may be split into before the attempt is abandoned.
#: Past this the envelope stops being cheaper than the decomposition it
#: replaces, which is the only reason it exists.
MAX_ENVELOPE_PARTS = 8


@dataclass(frozen=True)
class Envelope:
    """An obstacle's outer surface, with what it cost to get there."""

    mesh: trimesh.Trimesh
    parts: int
    deviation: float
    original_area: float
    original_faces: int
    bodies: int

    @property
    def area(self) -> float:
        """The area sound actually meets, free of buried inter-body faces."""
        return float(self.mesh.area)

    @property
    def area_ratio(self) -> float:
        """How much of the raw area was interior. 1.0 means none of it was."""
        if self.original_area == 0:
            return 1.0
        return self.area / self.original_area

    def summary(self) -> str:
        return (
            f"{self.bodies} bodies, {self.original_faces} faces, "
            f"{self.original_area:.1f} m2 -> {self.parts} part(s), "
            f"{len(self.mesh.faces)} faces, {self.area:.1f} m2 "
            f"(deviation {self.deviation * 100:.1f} cm)"
        )


def _grouped_hulls(mesh: trimesh.Trimesh, parts: int) -> trimesh.Trimesh | None:
    """Split the convex bodies into ``parts`` clusters and hull each cluster.

    Clustering is on body centroids, which is what recovers gross concavity: a
    corner sofa's two arms end up in different clusters and keep the space
    between them, while the dozens of small pieces making up one arm collapse
    into a single hull.
    """
    bodies = mesh.split(only_watertight=False)
    if len(bodies) <= 1 or parts >= len(bodies):
        return None
    centroids = np.array([body.centroid for body in bodies], dtype=float)
    labels = _cluster(centroids, parts)

    hulls = []
    for label in range(parts):
        selected = [body for body, own in zip(bodies, labels, strict=True) if own == label]
        if not selected:
            continue
        points = np.vstack([body.vertices for body in selected])
        if len(points) < 4:
            continue
        try:
            hulls.append(trimesh.Trimesh(vertices=points).convex_hull)
        except Exception:
            return None
    if not hulls:
        return None
    combined = trimesh.util.concatenate(hulls)
    return combined if isinstance(combined, trimesh.Trimesh) else None


def _cluster(points: np.ndarray, groups: int, iterations: int = 25) -> np.ndarray:
    """Plain Lloyd's algorithm, seeded deterministically.

    Deliberately not scipy or sklearn: this runs on a handful of points per
    object and the result must be reproducible across machines, since the
    geometry it produces is what gets simulated.
    """
    centres = points[np.linspace(0, len(points) - 1, groups).astype(int)]
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for group in range(groups):
            member = points[labels == group]
            if len(member):
                centres[group] = member.mean(axis=0)
    return labels


def acoustic_envelope(
    mesh: trimesh.Trimesh,
    max_deviation: float = MAX_ENVELOPE_DEVIATION,
    max_parts: int = MAX_ENVELOPE_PARTS,
) -> Envelope:
    """The surface to simulate for this obstacle, and how far it strays.

    Tries the single convex hull first and accepts it only if it is measurably
    close enough, then splits into more parts until it is, and finally gives up
    and keeps the original mesh rather than shipping a shape that misrepresents
    the object. Whatever comes back is reduced to the fewest triangles that
    stay within the same limit.
    """
    original_area = float(mesh.area)
    original_faces = len(mesh.faces)
    bodies = int(mesh.body_count)

    candidates: list[tuple[int, trimesh.Trimesh]] = []
    with contextlib.suppress(Exception):
        candidates.append((1, mesh.convex_hull))
    parts = 2
    while parts <= max_parts:
        grouped = _grouped_hulls(mesh, parts)
        if grouped is not None:
            candidates.append((parts, grouped))
        parts *= 2

    for count, candidate in candidates:
        # Sampled on the *candidate* and measured to the real mesh, not the
        # other way round. Sampling the decomposition would put points on the
        # faces buried between convex pieces, whose distance to any outer
        # envelope is half a piece thick, so a perfectly good envelope would be
        # rejected by the very interior geometry it exists to discard. This
        # direction asks the question that matters instead: is every point of
        # the envelope close to some real surface, or is it bulging into space
        # the object does not occupy? A hull can only ever contain the mesh, so
        # nothing is missed by not testing the other direction.
        error = deviation(candidate, mesh)
        if error <= max_deviation:
            reduced, _ = decimate_within(candidate, max_deviation)
            return Envelope(
                mesh=reduced,
                parts=count,
                deviation=error,
                original_area=original_area,
                original_faces=original_faces,
                bodies=bodies,
            )

    # Nothing convex represented this object closely enough. Keeping the real
    # mesh is expensive, and that is the honest cost of an object whose shape
    # genuinely cannot be approximated this way.
    reduced, error = decimate_within(mesh, max_deviation)
    return Envelope(
        mesh=reduced,
        parts=0,
        deviation=error,
        original_area=original_area,
        original_faces=original_faces,
        bodies=bodies,
    )
