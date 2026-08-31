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
whether the mesh answers the question at all: a watertight, consistently wound
mesh has a genuine inside, its normals can be pointed at the air, and it earns
:data:`FRONT`. Anything open or inconsistently wound has no defensible normal,
and gets :data:`BOTH`, which is the honest answer and also the safe one, since
a surface active on both sides can never be accidentally rigid.
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
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        candidate = mesh.copy()
        candidate.fix_normals()

    if not (candidate.is_watertight and candidate.is_winding_consistent):
        return OrientedMesh(
            mesh=mesh,
            sides=np.full(len(mesh.faces), BOTH, dtype=np.int8),
            authoritative=False,
        )

    # ``fix_normals`` leaves normals pointing out of the solid, which is where
    # the air is for an obstacle. A shell holds its air on the inside, so its
    # winding is inverted; the alternative, keeping the winding and writing
    # BACK, would leave the scene file carrying two conventions at once.
    if float(candidate.volume) < 0.0:
        candidate = candidate.copy()
        candidate.fix_normals()
    if air_side == "inside":
        candidate = candidate.copy()
        candidate.invert()

    return OrientedMesh(
        mesh=candidate,
        sides=np.full(len(candidate.faces), FRONT, dtype=np.int8),
        authoritative=True,
    )
