"""Which side of a triangle the air is on, decided once, in the scene.

The solver does not treat a surface as a surface. It marks the grid nodes on
one side of each triangle as boundary nodes carrying that triangle's material,
and PFFDTD's per-triangle ``sides`` field is what says which side. Read from
its own source rather than from its README, the values are:

===== ======================================================================
value meaning in ``pffdtd/python/common/room_geo.py`` and ``vox_scene.py``
===== ======================================================================
0     unmarked; the node is forced rigid
1     back side only; nodes on the normal's positive side are made rigid
2     front side only; nodes on the normal's negative side are made rigid
3     both sides; the material applies whichever way the wave arrives
===== ======================================================================

**This corrects the premise B0 and B1 were run on.** Those exports wrote ``2``
for every triangle in the belief that it meant "two sided". It does not: it
means *front side only*, and it is therefore the strictest possible statement
about geometry whose normals were never checked. Two consequences follow, and
they point in opposite directions to the ones assumed.

*The timings were not an upper bound.* ``sides`` never enters the adjacency
computation, only the material marking, so no boundary node was ever
over-counted and no run was made artificially slow by it.

*The absorption silently was.* Every triangle whose HSSD normal happened to
point away from the air had its own boundary nodes marked ``-1``, which is
rigid. That surface kept its area in every report and contributed no absorption
at all to the solver, with nothing anywhere raising an error.

So orientation is derived here, once, and recorded in the scene description
rather than left to whatever winding an asset shipped with. The test is
whether the mesh answers the question at all: a closed, consistently wound mesh
has a genuine inside, its normals can be pointed at the air, and it earns
:data:`FRONT`. Anything open or inconsistently wound has no defensible normal,
and gets :data:`BOTH`, which is the honest answer and also the safe one, since
a surface active on both sides can never be accidentally rigid.

Closed is :func:`is_closed` and not ``trimesh``'s ``is_watertight``, which asks
for edge-manifoldness on top of closure and was throwing away most of this
project's carved geometry over a few dozen non-manifold edges. That function
carries the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

__all__ = [
    "BACK",
    "BOTH",
    "FRONT",
    "UNMARKED",
    "OrientedMesh",
    "is_closed",
    "orient_for_air",
]

#: PFFDTD's sidedness codes. Named, because ``2`` reads like "two sided" and is
#: not, which is precisely the mistake this module exists to stop repeating.
UNMARKED = 0
BACK = 1
FRONT = 2
BOTH = 3

#: Where the air is relative to the mesh. Furniture is a solid standing in air,
#: so the air is outside it; a room shell is a box containing air, so the air is
#: inside it.
AirSide = Literal["outside", "inside"]


@dataclass(frozen=True)
class OrientedMesh:
    """A mesh whose normals face the air, and the claim that entitles it to."""

    mesh: trimesh.Trimesh
    sides: np.ndarray
    authoritative: bool

    @property
    def summary(self) -> str:
        if self.authoritative:
            return f"{len(self.sides)} faces oriented into the air (sides={FRONT})"
        return f"{len(self.sides)} faces of unknown orientation (sides={BOTH})"


def is_closed(mesh: trimesh.Trimesh) -> bool:
    """Whether the mesh encloses a volume and agrees with itself which side is in.

    This is the question the pipeline actually needs answered, and it is *not*
    ``trimesh.Trimesh.is_watertight``. Trimesh calls a mesh watertight when
    every edge is used exactly twice, which is edge-manifoldness -- a stricter
    property, and one a closed surface can fail without having a hole anywhere.

    That difference was quietly deciding the geometry. A marching cubes surface
    decimated hard keeps every edge paired but grows a handful of edges shared
    by three or more faces, at the diagonal cell junctions the blur in
    :mod:`reverberate.geometry.carve` rounds off but does not eliminate.
    Measured on this flat's curtain: at 2 000 triangles, **0 boundary edges,
    72 non-manifold ones**, winding consistent, volume within a per cent of the
    isosurface it came from. Nothing is open; the object is closed and the
    normals are coherent. ``is_watertight`` said no, so the carve was discarded
    and the solver was given the collider -- a 0.561 m3 slab in place of a
    0.165 m3 curtain. Across the flat that rule refused **46 of 135 templates,
    89 placed instances, 41.6 m3 of collider**.

    Two conditions, and both are needed. No boundary edge means the surface has
    no rim to leak through, so inside and outside are distinguishable. One
    consistent winding means the normals all point the same way relative to
    that inside, so :func:`orient_for_air` can turn them towards the air and
    the signed volume has a meaning. A few non-manifold edges leave the inside
    ambiguous along those edges alone, which against a million paired ones is
    not a claim worth throwing an object away over.
    """
    if len(mesh.faces) == 0:
        return False
    boundary = trimesh.grouping.group_rows(  # type: ignore[no-untyped-call]
        mesh.edges_sorted, require_count=1
    )
    return bool(len(boundary) == 0 and mesh.is_winding_consistent)


def orient_for_air(mesh: trimesh.Trimesh, air_side: AirSide) -> OrientedMesh:
    """Point every normal at the air, and say so per face.

    Returns a copy whenever it changes anything, so a cached template mesh is
    never mutated under a caller that is still holding it.

    The winding is repaired before the mesh is judged. ``fix_normals`` makes an
    otherwise sound mesh consistent and outward-facing, and a mesh that only
    needed repairing is not one whose orientation is unknowable. What survives
    as unknowable is the genuinely open or self-inconsistent geometry, which is
    most of an HSSD convex decomposition, and it is marked as such rather than
    guessed at.
    """
    if len(mesh.faces) == 0:
        return OrientedMesh(mesh=mesh, sides=np.zeros(0, dtype=np.int8), authoritative=False)

    candidate = mesh
    if not is_closed(mesh):
        candidate = mesh.copy()
        candidate.fix_normals()

    if not is_closed(candidate):
        return OrientedMesh(
            mesh=mesh,
            sides=np.full(len(mesh.faces), BOTH, dtype=np.int8),
            authoritative=False,
        )

    # Normals out of the solid, which is where the air is for an obstacle. A
    # shell holds its air on the inside, so its winding is inverted; the
    # alternative, keeping the winding and writing BACK, would leave the scene
    # file carrying two conventions at once.
    #
    # ``invert`` rather than ``fix_normals`` for the negative-volume case. The
    # winding is already consistent by the time we get here, so all that is
    # wanted is its other sign, and inverting is the operation that always
    # delivers it. ``fix_normals`` reasons over the face adjacency graph
    # instead, and on a mesh with a few non-manifold edges -- which
    # :func:`is_closed` now admits -- it can decline to flip anything at all:
    # measured on this flat's carved curtain, volume -0.16537 before and
    # -0.16537 after, so the normals would have stayed pointing into the solid
    # and the whole object would have been marked rigid.
    if float(candidate.volume) < 0.0:
        candidate = candidate.copy()
        candidate.invert()
    if air_side == "inside":
        candidate = candidate.copy()
        candidate.invert()

    return OrientedMesh(
        mesh=candidate,
        sides=np.full(len(candidate.faces), FRONT, dtype=np.int8),
        authoritative=True,
    )
