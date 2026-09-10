"""Reduce a collision proxy to the one surface the air can reach.

Pulled out of :mod:`reverberate.geometry.sim_geometry` when the per-template
mesh gained a disk cache: that cache keys on the source of everything that
decides what a template's mesh is, and hashing the whole of ``sim_geometry``
would throw away the pool -- fifteen thousand templates -- for an edit to a
report string. What decides the mesh is this function, the carve behind it and
the asset resolution in front of it, and each now sits in a file of its own.
"""

from __future__ import annotations

import trimesh

from reverberate.geometry.orientation import is_closed

__all__ = ["outer_surface"]


def outer_surface(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """The union of a collider's convex bodies, and whether the union succeeded.

    HSSD ships colliders as convex decompositions whose bodies interpenetrate.
    Every contact between two bodies leaves a pair of faces *inside* the solid,
    where sound never reaches, and the voxeliser's adjacency graph around them
    is a tangle of shards rather than one sealed surface.

    That is not cosmetic. PFFDTD deliberately does not fill solids -- it builds
    an adjacency graph so it can accept non-watertight scenes -- so the air
    inside every object is simulated, bounded by nodes marked rigid, which is a
    cavity with no absorption at all and a correspondingly enormous Q. Measured
    on one bedroom: 403 such pockets holding 3.77 m3, 11 per cent of the room's
    own air, and 211 of them resonating between 1 and 4 kHz at a mean size of
    8.6 cm. The responses show the consequence as narrow lines near 2 kHz that
    only emerge below -25 dB and drag the late decay to three times the early
    one.

    The union removes the buried faces exactly rather than approximately. It
    conserves volume to the digit, so nothing about the shape is given up; only
    the interior area goes, which is the area that was never reachable.

    Returns the original mesh unchanged, and False, when the boolean engine
    cannot do it, or when it returns something that is not actually one sealed
    solid: an obstacle with its buried faces is still better than no obstacle,
    and the caller reports the count rather than hiding it. Closure is checked
    here rather than assumed, because "conserves volume to the digit" is a
    claim about a *closed* solid, and a union that comes back open has not
    earned it. Closure by
    :func:`~reverberate.geometry.orientation.is_closed`, which is the claim
    being made; ``trimesh``'s ``is_watertight`` would add edge-manifoldness on
    top of it and discard unions that are sealed.
    """
    if mesh.body_count <= 1:
        return mesh, True
    try:
        united = trimesh.boolean.union(list(mesh.split(only_watertight=False)))
    except Exception:
        return mesh, False
    if not isinstance(united, trimesh.Trimesh) or len(united.faces) == 0:
        return mesh, False
    if not is_closed(united):
        return mesh, False
    return united, True
