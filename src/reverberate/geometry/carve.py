"""Carve HSSD's collision proxies back to the shape the render mesh proves.

HSSD ships every object twice: a render mesh, and a ``.collider.glb`` that is a
convex decomposition of it. The solver is given the collider, because it is the
only one of the two that can answer "which side is the air on" -- not one render
mesh in this apartment's bedroom is watertight, and they run to 28 668
disconnected shells apiece. So the collider stays. What it costs is that every
concave or hollow shape arrives at the grid as a solid lump.

**Measured on ``bedroom.001`` of ``102344022``, unioned collider volume against
render volume:** basket 176x, decoration 49x, lamp 39x and 24x, wardrobe 26x,
carpet 12x, flowerpot 8.8x, curtain 4.5x, plant 3.1 to 3.8x. Over the whole
room, **2.02x**. That is not a rounding error on a proxy, it is a different
object: a lampshade simulated as a solid cone, a basket as a block of wood, a
wardrobe as a slab. It changes surface area, it changes the volume patch 5
seals, and it is what makes the voxel view look inflated beside the mesh view.

**The carve.** The render mesh cannot say which side is solid, but it can say
where there is *air*: any cell reachable from outside the object without
crossing one of its surfaces is air, whatever the collider claims. So voxelise
the collider solid, voxelise the render surface, flood fill the complement of
the surface from the grid's border, and remove from the solid everything that
fill reached. What is left is the collider minus the air the render mesh proves
is there, and marching cubes turns it back into one closed surface -- which is
exactly the property the collider was being kept for.

**It is never a silent substitution.** A carve that comes back empty or open is
discarded and the plain collider is used. :class:`CarveReport` names every
template in each case, and names separately the ones whose shipped mesh sits
outside :data:`VOLUME_TOLERANCE` of its own isosurface. The whole point of the
change is that the picture stops lying; a fallback nobody can see would be the
same fault in the other direction.

**The triangle budget is not optional.** Marching cubes on one wardrobe at 6 mm
returns 207 312 triangles against the collider's 5 848, and voxelisation cost is
driven by triangle count. Each carve is decimated to a multiple of the collider
it replaces, so the scene grows by a bounded factor rather than by whatever the
isosurface happened to need.

*But the budget must not be a way of losing the object.* Reaching it was gated
on ``trimesh``'s ``is_watertight``, which asks for edge-manifoldness on top of
closure, and a hard-decimated isosurface fails that on a few dozen edges out of
a million while remaining perfectly closed. Measured on this flat, that rule
discarded the carve for **46 of 135 templates -- 89 placed instances holding
41.6 m3 of collider**: the christmas tree went into the grid as a 0.95 m3 solid
cone where its carve is 0.13, the carpet as 0.83 m3 where its carve is 0.18, the
curtain as 0.56 against 0.17. Closure is now
:func:`~reverberate.geometry.orientation.is_closed` and the ladder in
:func:`_to_budget` spans the factor of eight its own docstring always claimed.
All 46 come back. Over the flat's 231 placed instances that is **54.55 m3 of
simulated obstacle down to 36.38**, for **2 165 150 obstacle triangles up to
3 218 826** -- half again as much voxelisation, against a third of the volume
that was never there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from reverberate.geometry.hssd_assets import resolve_asset
from reverberate.geometry.orientation import is_closed
from reverberate.settings import data_root

__all__ = [
    "CARVE_PITCH_M",
    "CarveReport",
    "CarveResult",
    "carve_collider",
]

#: Coarsest side of the cell the carve is decided on, in metres, and the pitch
#: every template falls back to. 6 mm sits below the 8.17 mm grid step at 4 kHz,
#: so the carve never invents detail the coarser of this project's two grids
#: cannot see, and it is what every template used before the pitch became a
#: per-template choice.
CARVE_PITCH_M = 0.006

#: The pitches a template may be carved on, coarsest first. Chosen per template
#: by :func:`pitch_for` and **never by ``fmax``**: the bedroom at 16 kHz and the
#: apartment at 4 kHz must be the *same* geometry, or nothing measured on one
#: says anything about the other. Every rung is a function of the template's own
#: bounds and its own render mesh, so one template gets one pitch whatever grid
#: is later laid over it.
#:
#: 2 mm is the floor because the finest grid this project builds is 2.043 mm, at
#: 16 kHz and 10.5 points per wavelength. A carve finer than that would decide
#: occupancy no solver here can read.
CARVE_PITCH_LADDER = (0.006, 0.004, 0.003, 0.002)

#: The most faces the carve will let an isosurface reach, which is what bounds
#: the pitch from below on a large object. Past about two million the reduction
#: to a scene-affordable budget stops holding the volume: measured, the carpet's
#: 1 865 356 faces came back 0.5 per cent small at 2 000 and the curtain's
#: 1 646 378 came back **40 per cent large**. A finer pitch that cannot be
#: carried by a mesh is not a finer carve, it is a worse one.
ISO_FACE_CAP = 2_000_000

#: Isosurface faces per square metre of render surface, times pitch squared --
#: the constant that lets :func:`pitch_for` predict an isosurface without
#: building one. Measured over nine templates of this flat at 6 mm: 0.98
#: (dresser), 1.22 (couch), 1.24 (fridge), 1.49 (piano), 1.94 (christmas tree),
#: 2.00 (carpet), 2.03 (plant), 2.39 (car), 4.30 (curtain). The top of that
#: range is used, because this number decides whether to spend a pitch and
#: over-estimating costs a refinement while under-estimating costs the mesh.
FACES_PER_AREA = 4.5

#: How much of the coarse carve's volume a finer pitch must keep to be believed.
#: The flood fill is stopped by the conservatively marked shell of the render
#: mesh, and that shell is what plugs the mesh's own holes -- HSSD's render
#: meshes are wide open, the fridge's 1 676 faces carrying 1 416 boundary edges.
#: The plugging band narrows with the pitch, so a finer pitch is where a leak
#: would first appear, and a leak does not look like a refinement: it collapses
#: the carve by three orders of magnitude (measured at 0.979 m3 to 0.001 on that
#: fridge, with a rasteriser tight enough to let go). Refinement between rungs
#: runs to a factor of two either way, so a tenth is far below any real one and
#: far above any leak.
LEAK_FLOOR = 0.10

#: Largest occupancy array a single template may use, in cells. Past it the
#: carve is refused rather than retried at a coarser pitch: a second pitch is a
#: second geometry, and this module's whole discipline is that a substitution is
#: never silent. No template in this apartment reaches it -- all 257 cache
#: entries are at ``CARVE_PITCH_M`` -- so the branch that coarsened was never
#: once exercised, and an unexercised branch that changes the shape is worse
#: than a refusal that names itself.
MAX_CELLS = 120_000_000

#: Triangles a carve may keep, as a multiple of the collider it replaces.
#: Marching cubes returns 35x on a wardrobe; the scene cannot afford that and
#: does not need it, because the shape is already decided by the occupancy.
BUDGET_FACTOR = 4.0

#: Floor under the budget, so a collider that is a twelve-triangle box does not
#: force its carve down to forty-eight triangles and lose the shape.
BUDGET_FLOOR = 2000

#: Triangles a carve may keep regardless of the collider it replaces, when no
#: reduction of it stays closed. An object at this size is not what makes a
#: scene expensive, and refusing it leaves a 39x inflated lampshade in the grid
#: to save a rounding error on the triangle count.
ABSOLUTE_CAP = 25_000

#: How far a decimated carve's volume may sit from the isosurface it reduces,
#: as a fraction. The isosurface is the shape the occupancy decided; the mesh
#: only carries it, so a reduction that restates it a fifth larger is not a
#: cheaper version of the carve, it is a different object -- and larger is the
#: direction this whole module exists to fight. :func:`_to_budget` climbs its
#: ladder until a rung lands inside this band, spending triangles to keep the
#: shape; when none does it returns the closest rung anyway and
#: :attr:`CarveResult.volume_error` records the miss, because the alternative
#: is the collider and the collider is wrong by an order of magnitude rather
#: than by a fifth.
VOLUME_TOLERANCE = 0.10

#: Above this share of the collider's own cells, the fill has removed nothing
#: worth the substitution and the collider is kept. Counted in cells rather than
#: in volume: both sides then come from the same rasteriser, so the comparison
#: carries none of the isosurface's resampling error, which runs to several per
#: cent on a small object and in either direction. See :func:`_carve_uncached`.
KEEP_FRACTION = 0.98


@dataclass
class CarveResult:
    """One template's carve, and whether it is usable."""

    mesh: trimesh.Trimesh
    #: True when ``mesh`` is the carve; False when it is the untouched collider.
    carved: bool
    #: Why the carve was not used, empty when it was.
    reason: str = ""
    collider_volume: float = 0.0
    carved_volume: float = 0.0
    #: How far the shipped mesh's volume sits from the isosurface it was
    #: reduced from, as a fraction. Zero when nothing was decimated. Above
    #: :data:`VOLUME_TOLERANCE` the mesh is still the better of the two
    #: available answers, and this is what says so out loud rather than leaving
    #: it to be discovered.
    volume_error: float = 0.0
    #: The cell the occupancy was decided on, in metres. One per template, and
    #: never a function of ``fmax``. See :func:`pitch_for`.
    pitch_m: float = CARVE_PITCH_M
    #: The pitch whose carve was rejected as a leak, when one was, else 0.0.
    #: A finer pitch that comes back three orders of magnitude smaller has not
    #: refined the shape, it has let the flood fill inside; recorded rather
    #: than silently stepped over. See :data:`LEAK_FLOOR`.
    leaked_at_m: float = 0.0

    @property
    def shrink(self) -> float:
        """How much of the collider's volume the carve kept, 1.0 when unchanged."""
        if self.collider_volume <= 0.0:
            return 1.0
        return self.carved_volume / self.collider_volume


@dataclass
class CarveReport:
    """What the carve did across a scene, for the manifest and the run page."""

    carved: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    #: Templates whose shipped mesh sits outside :data:`VOLUME_TOLERANCE` of
    #: its own isosurface, and by how much. Named rather than counted only: the
    #: mesh is still much closer to the object than the collider it replaces,
    #: but it is the one place the carve's shape is decided by the decimator
    #: rather than by the occupancy, and that has to be readable.
    strained: dict[str, float] = field(default_factory=dict)
    #: The pitch each carved template was decided on, in millimetres. Not a
    #: uniform number any more, so the manifest has to carry it: two templates
    #: of the same scene can be carved on different cells, and which one an
    #: object got is the difference between a chunky lamp and a clean one.
    pitch_mm: dict[str, float] = field(default_factory=dict)
    #: Templates where a finer pitch was tried and rejected as a leak, and the
    #: pitch that was rejected, in millimetres.
    leaked: dict[str, float] = field(default_factory=dict)

    def add(self, template: str, result: CarveResult) -> None:
        if result.carved:
            self.carved[template] = round(result.shrink, 4)
            self.pitch_mm[template] = round(result.pitch_m * 1000, 2)
            if result.volume_error > VOLUME_TOLERANCE:
                self.strained[template] = round(result.volume_error, 4)
        else:
            self.skipped[template] = result.reason
        if result.leaked_at_m:
            self.leaked[template] = round(result.leaked_at_m * 1000, 2)

    def summary(self) -> str:
        if not self.carved and not self.skipped:
            return "no carve"
        kept = np.mean(list(self.carved.values())) if self.carved else 1.0
        strained = f", {len(self.strained)} off their isosurface" if self.strained else ""
        pitches = sorted(set(self.pitch_mm.values()))
        cell = (
            f", on {'/'.join(f'{p:g}' for p in pitches)} mm cells"
            if len(pitches) > 1
            else f", on {pitches[0]:g} mm cells"
            if pitches
            else ""
        )
        return (
            f"{len(self.carved)} colliders carved to {kept:.0%} of their volume, "
            f"{len(self.skipped)} left as they are{strained}{cell}"
        )


def grid_for(
    render: trimesh.Trimesh, collider: trimesh.Trimesh, pitch: float
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """The occupancy grid one pitch asks for: its origin, its shape, its cells.

    Three cells of margin each way, because :func:`_surface_cells` marks a
    triangle's whole bounding box and :func:`_outside` fills from the border --
    a surface touching the border would have nothing outside it to fill from.
    """
    low = np.minimum(render.bounds[0], collider.bounds[0]) - 3 * pitch
    high = np.maximum(render.bounds[1], collider.bounds[1]) + 3 * pitch
    extent = np.ceil((high - low) / pitch).astype(np.int64) + 4
    shape = (int(extent[0]), int(extent[1]), int(extent[2]))
    return low, shape, float(shape[0]) * shape[1] * shape[2]


def pitch_for(render: trimesh.Trimesh, collider: trimesh.Trimesh) -> float:
    """The finest pitch this template can be carved on, from its own two meshes.

    **Why the pitch was ever one number.** It had to be independent of ``fmax``,
    or the same flat at 4 and at 16 kHz would be two different objects and
    nothing measured on one would say anything about the other. That is still
    true, and nothing here reads ``fmax``: the answer is a function of the
    template's bounds and its render area alone, so a template has one pitch
    for every grid ever laid over it.

    **Why one number was still wrong.** 6 mm is a different statement about a
    2.3 m christmas tree than about a 30 cm lamp. On the lamp it is five per
    cent of the object in every direction -- the occupancy is fifty cells across
    and the carve is visibly chunky -- while the isosurface it produces is
    small enough to carry ten times over. The pitch was being set by the largest
    object in the dataset and paid for by the smallest.

    **Two ceilings, and they bind on different objects.**

    *Memory.* :data:`MAX_CELLS` is what a template's occupancy array may cost,
    and it is what stops a car or a carpet going below 6 mm at all.

    *The mesh.* An isosurface is only useful if the triangle budget can carry
    it, and past :data:`ISO_FACE_CAP` it cannot -- see that constant for the two
    measurements that fix it. The size is predicted rather than built:
    marching cubes emits of the order of one face per boundary cell face, so it
    scales as the render surface over the pitch squared, and
    :data:`FACES_PER_AREA` is that constant measured across this flat.

    So the objects that gain are the small ones, which is where the coarse pitch
    was worst and the refinement is cheapest. A christmas tree stays at 6 mm,
    and correctly: its limit is the budget its 1.9 million faces have to reduce
    to, not the cell they were decided on.
    """
    area = float(render.area)
    for pitch in sorted(CARVE_PITCH_LADDER):
        if pitch >= CARVE_PITCH_M:
            break
        _, _, cells = grid_for(render, collider, pitch)
        if cells > MAX_CELLS:
            continue
        if area > 0.0 and FACES_PER_AREA * area / pitch**2 > ISO_FACE_CAP:
            continue
        return pitch
    return CARVE_PITCH_M


def _surface_cells(
    mesh: trimesh.Trimesh, pitch: float, origin: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    """Every cell ``mesh``'s surface passes through, erring on the side of more.

    Conservative on purpose, and that direction is not arbitrary. This grid is
    about to be flood filled from the outside, and the fill is stopped by
    marked cells. A cell marked that should not be stops the fill one cell
    early, which leaves solid in place; a cell *missed* opens a hole, the fill
    pours into the object and the carve eats it. Over-marking costs a
    millimetre, under-marking costs the object.

    Two rejected alternatives, both measured. ``trimesh``'s ``method="ray"``
    casts along the three axes only, so a face nearly parallel to one of them
    comes back perforated: it leaked on 42 of 47 templates, keeping 2 to 3 per
    cent of the collider where the true figure is 6 to 16. Its default
    ``method="subdivide"`` does not leak, but it raises ``max_iter exceeded`` on
    this apartment's carpet -- 106 triangles spanning 4.5 m -- and takes 60 to
    77 s on a wardrobe.

    So subdivide only as far as the cell, then mark each triangle's whole
    bounding box. With every edge under two cells a triangle's box is at most
    three cells across, so the over-mark is bounded by one cell and the cost is
    bounded by the surface area rather than by the triangulation it arrived in.
    """
    subdivided: tuple[np.ndarray, np.ndarray] = trimesh.remesh.subdivide_to_size(  # type: ignore[no-untyped-call]
        mesh.vertices, mesh.faces, max_edge=2.0 * pitch, max_iter=24
    )
    vertices, faces = subdivided
    corners = vertices[faces]
    low = np.floor((corners.min(axis=1) - origin) / pitch).astype(np.int64)
    high = np.floor((corners.max(axis=1) - origin) / pitch).astype(np.int64)
    extent = np.asarray(shape, dtype=np.int64)
    np.clip(low, 0, extent - 1, out=low)
    np.clip(high, 0, extent - 1, out=high)

    occupied = np.zeros(shape, dtype=bool)
    spans = high - low + 1
    # The boxes are at most three cells a side, so enumerating them outright is
    # 27 writes per triangle at worst and needs no per-triangle Python.
    for dx in range(int(spans[:, 0].max())):
        for dy in range(int(spans[:, 1].max())):
            for dz in range(int(spans[:, 2].max())):
                use = (dx < spans[:, 0]) & (dy < spans[:, 1]) & (dz < spans[:, 2])
                if not use.any():
                    continue
                cell = low[use] + (dx, dy, dz)
                occupied[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    return occupied


def _eroded(solid: np.ndarray) -> np.ndarray:
    """``solid`` less the shell that conservative marking added, when it can spare it.

    :func:`_surface_cells` deliberately over-marks, so a solid derived from it
    is about a cell too big in every direction; left in, the carve comes back
    *larger* than the collider it is shrinking, measured at 114 per cent on one
    decoration.

    Applied to the collider's solid alone, never to the carve. Eroding the
    finished carve takes the cell off a second time, and on a body that is
    already a shell -- a wardrobe with open sides, a lampshade, a picture board
    -- there is no second cell to give: it took one wardrobe down to 0.1 per
    cent of its collider, which trades an object that is too fat for one the
    8.17 mm grid may not resolve at all. Swapping inflated for missing is not a
    fix.

    And not at all when the solid is thin enough that a cell is most of it, for
    the same reason.
    """
    thinner = ndimage.binary_erosion(solid)
    return thinner if thinner.sum() >= 0.5 * solid.sum() else solid


def _tightened(solid: np.ndarray, air: np.ndarray) -> np.ndarray:
    """``solid`` less the air, with the cell the conservative marking hid given back.

    :func:`_eroded` takes off the shell that over-marking added to the
    *collider*. The render mesh was over-marked by exactly the same rule and
    nothing ever gave that cell back, so the two sides of the subtraction were
    not symmetric: the solid was measured to its true face and the air was
    measured one cell short of its own, everywhere. Every carve came out a cell
    fat, and on an object whose surface is mostly shell that cell is most of the
    object -- this flat's christmas tree kept 0.325 m3 where the same fill run
    symmetrically keeps 0.171.

    So grow the air by the one cell it was denied. What that adds is precisely
    the outer layer of the render surface's own marked shell: cells the surface
    passes through, which are part air and part solid, and which the fill
    stopped in front of. Half of such a cell is air on average, and a grid
    cannot hold half a cell, so the choice is which way to round -- and rounding
    both sides the same way is the only one that leaves no bias.

    Body by body, and never past half of one. A sheet one cell thick is all
    shell, and growing the air into it from both faces at once deletes it: a
    picture's canvas, a curtain, a carpet. Those are exactly the objects the
    carve exists to recover, so a connected body that would lose more than half
    of itself keeps its fat cell instead, which is the same bound
    :func:`_eroded` uses and for the same reason. Measured over this flat's
    twelve largest refused templates, the guard fires on the carpet, the
    curtain and one decoration -- all three sheets -- and lets the tree lose
    47 per cent, the plant 26 and the piano 17.
    """
    kept: np.ndarray = solid & ~air
    tight: np.ndarray = solid & ~ndimage.binary_dilation(air)
    labels, count = ndimage.label(kept)
    if count == 0:
        return kept
    before = np.bincount(labels.ravel(), minlength=count + 1)
    after = np.bincount(labels.ravel(), weights=tight.ravel(), minlength=count + 1)
    fat = np.flatnonzero(after < 0.5 * before)
    fat = fat[fat > 0]
    if fat.size == 0:
        return tight
    grown: np.ndarray = tight | (np.isin(labels, fat) & kept)
    return grown


def _outside(surface: np.ndarray) -> np.ndarray:
    """Cells reachable from the grid's border without crossing ``surface``."""
    labels, _ = ndimage.label(~surface)
    border = np.unique(
        np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    return np.isin(labels, border[border > 0])


def _to_budget(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh | None:
    """``mesh`` decimated towards ``budget`` triangles, or None if none holds.

    Closure is the whole reason the collider is used at all, so a reduction
    that loses it is not a cheaper version of the mesh, it is a different kind
    of object. Closure, and not ``trimesh``'s ``is_watertight``: that also
    demands edge-manifoldness, which a hard-decimated isosurface fails on a few
    dozen edges while remaining perfectly closed, and demanding it here refused
    **46 of this flat's 135 templates** -- 89 placed instances holding 41.6 m3
    of collider, the christmas tree and the carpet and the curtain among them.
    See :func:`~reverberate.geometry.orientation.is_closed`.

    Asking for the budget once and giving up refused 11 of 47 templates: a
    marching cubes surface is uniform, and taking 46 000 triangles to 2 000 in
    one step pinches it open somewhere almost every time. So the budget is a
    target, not a cliff -- back off by doubling and take the first reduction
    that stays closed and keeps the volume. Four steps is a factor of eight,
    past which the mesh is not worth the triangles.

    **The volume is a second condition, and it is two sided.** The isosurface
    is the shape the occupancy decided; the mesh only carries it, so a
    reduction that restates it a fifth larger is not a cheaper carve, it is a
    different object -- and larger is the direction this module exists to
    fight. This flat's curtain carves to 0.118 m3 and comes back 0.165 at 2 000
    triangles. So each rung is asked to land inside
    :data:`VOLUME_TOLERANCE`, and the ladder is what pays for it: a rung that
    misses is refused and the next one up is asked instead, spending triangles
    to keep the shape.

    When no rung lands inside the band, the closest one is returned anyway
    rather than nothing. What "nothing" means here is the plain collider, and
    on the templates that reach this line the collider is worse by an order of
    magnitude, not by a fifth: one decoration's isosurface holds 0.028 m3
    against a 0.325 m3 collider. Trading a shape that is 20 per cent off for
    one that is 1 060 per cent off is not a defence of accuracy. The caller
    records how far off it was, so a reader can see which meshes are in this
    case -- see :attr:`CarveResult.volume_error`.
    """
    if len(mesh.faces) <= budget:
        return mesh
    import fast_simplification

    volume = abs(float(mesh.volume))
    best: trimesh.Trimesh | None = None
    best_error = float("inf")
    # Four rungs spanning a factor of eight, which is what the paragraph above
    # has always described. The code stopped at 2x, so the back-off it promised
    # was really a third of one -- worth naming, because the far end of this
    # ladder is where a large isosurface is caught.
    for target in (budget, budget * 2, budget * 4, budget * 8):
        if target >= len(mesh.faces):
            # Nothing left to ask for: the mesh is already under this target, so
            # the ladder is exhausted and the caller decides whether the
            # undecimated carve is small enough to keep.
            break
        vertices, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), target_count=target
        )
        reduced = trimesh.Trimesh(vertices, faces, process=True)
        reduced.update_faces(reduced.nondegenerate_faces())
        reduced.remove_unreferenced_vertices()
        if len(reduced.faces) == 0 or not is_closed(reduced):
            continue
        # Closed is not enough on its own. A pair of triangles back to back has
        # no boundary edge and one consistent winding while enclosing nothing:
        # reducing a twelve face box to one triangle returns a "closed" body of
        # zero volume, and the band's own floor is what rejects it.
        error = abs(abs(float(reduced.volume)) - volume) / volume if volume > 0.0 else float("inf")
        if error <= VOLUME_TOLERANCE:
            return reduced
        if error < best_error:
            best, best_error = reduced, error

    # No rung kept the volume. An isosurface that will not simplify is usually a
    # genuinely intricate shape rather than a broken one, and undecimated it is
    # the exact shape the occupancy decided -- better than any reduction of it,
    # so it is offered before one. Keep it when it is small enough to afford.
    #
    # Two caps, because the budget is relative and the cost is not. A budget of
    # 4x a 108 face picture is 2 000 triangles and refuses a 12 000 triangle
    # carve that the scene would never notice; measured on this bedroom,
    # relative-only refused 17 of 41 templates and left them inflated. So also
    # allow anything under ABSOLUTE_CAP outright: past that a single object
    # starts to matter against a scene of a million and a half.
    if len(mesh.faces) <= max(int(1.25 * budget), ABSOLUTE_CAP):
        return mesh

    # Too many triangles to ship whole, and no reduction of it kept the volume.
    # The closest one is still the better of the two answers left: on the
    # templates that reach this line the collider is out by an order of
    # magnitude, not by a fifth. ``None`` is kept for the case where nothing
    # closed came back at all, which is the only one where there is no carve to
    # choose between.
    return best


@dataclass
class _Occupancy:
    """One pitch's answer: what the collider fills, and what survives the fill."""

    solid: np.ndarray | None
    kept: np.ndarray
    low: np.ndarray


def _occupancy(render: trimesh.Trimesh, collider: trimesh.Trimesh, pitch: float) -> _Occupancy:
    """Rasterise both meshes at ``pitch`` and subtract the air from the solid."""
    low, shape, _ = grid_for(render, collider, pitch)
    # The collider is closed, so its solid is everything the fill cannot reach.
    # Deriving it the same way as the render's air keeps one rasteriser and one
    # fill in the module rather than two conventions that have to agree.
    solid = _eroded(~_outside(_surface_cells(collider, pitch, low, shape)))
    if not solid.any():
        return _Occupancy(solid=None, kept=solid, low=low)
    air = _outside(_surface_cells(render, pitch, low, shape))
    # The solid was eroded by the cell the marking added to it, and the air is
    # grown by the cell the same marking took from it, so both sides of the
    # subtraction are measured the same way. See :func:`_tightened`.
    return _Occupancy(solid=solid, kept=_tightened(solid, air), low=low)


def _carve_uncached(hssd_root: Path, template: str, collider: trimesh.Trimesh) -> CarveResult:
    """The carve itself. See the module docstring."""
    base = CarveResult(
        mesh=collider,
        carved=False,
        collider_volume=abs(float(collider.volume)),
    )
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        base.reason = "asset not resolved"
        return base
    if asset.collider_is_render:
        # There is one mesh, and it is already the one the solver is given.
        base.reason = "no separate collider to carve"
        return base
    render = trimesh.load(asset.render, force="mesh")
    if not isinstance(render, trimesh.Trimesh) or len(render.faces) == 0:
        base.reason = "render mesh unreadable"
        return base

    pitch = pitch_for(render, collider)
    base.pitch_m = pitch
    low, shape, cells = grid_for(render, collider, CARVE_PITCH_M)
    if cells > MAX_CELLS:
        base.reason = (
            f"too large to carve at {CARVE_PITCH_M * 1000:.0f} mm ({cells / 1e6:.0f}M cells)"
        )
        return base

    try:
        occupancy = _occupancy(render, collider, pitch)
        # A finer pitch is where a leak would first show, because the band of
        # conservatively marked cells that plugs the render mesh's own holes
        # narrows with it. So the coarse carve is computed as well and the fine
        # one is measured against it -- see :data:`LEAK_FLOOR`. Only when a
        # finer pitch was actually chosen: at 6 mm there is nothing to compare
        # against and the coarse pass is the pass.
        if pitch < CARVE_PITCH_M:
            reference = _occupancy(render, collider, CARVE_PITCH_M)
            fine = occupancy.kept.sum() * pitch**3
            coarse = reference.kept.sum() * CARVE_PITCH_M**3
            if fine < LEAK_FLOOR * coarse:
                base.leaked_at_m = pitch
                base.pitch_m = pitch = CARVE_PITCH_M
                occupancy = reference
    except Exception as error:  # trimesh raises several unrelated types here
        base.reason = f"rasterisation failed: {type(error).__name__}"
        return base
    if occupancy.solid is None:
        base.reason = "empty occupancy"
        return base
    solid, kept, low = occupancy.solid, occupancy.kept, occupancy.low
    if not kept.any():
        base.reason = "carve removed everything"
        return base

    # A carve that removed nothing is not a carve. What would come back is the
    # collider resampled at the carve pitch and then decimated: approximate
    # where the collider is exact, and no volume recovered. This is also the
    # only place the pipeline was not deterministic -- with no air to remove,
    # whether the decimation stayed closed decided the geometry, and that
    # answer differs between platforms.
    removed = 1.0 - float(kept.sum()) / float(solid.sum())
    if removed <= 1.0 - KEEP_FRACTION:
        base.reason = f"the render mesh proves no air inside ({removed:.1%} removed)"
        return base

    try:
        # Marching cubes on a hard 0/1 volume produces non-manifold edges
        # wherever two cells meet only along a diagonal, and trimesh then calls
        # the result open -- which lost 5 of 21 templates outright. Blurring by
        # well under a cell rounds those junctions off without moving any
        # surface a whole cell, and the isosurface comes back closed.
        field = ndimage.gaussian_filter(np.pad(kept.astype(np.float32), 2), sigma=0.6)
        isosurface: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        isosurface = measure.marching_cubes(field, level=0.5)  # type: ignore[no-untyped-call]
        vertices, faces = isosurface[0], isosurface[1]
    except Exception as error:
        base.reason = f"marching cubes failed: {type(error).__name__}"
        return base
    # Index space back to metres, and the half cell matters. A cell whose index
    # is ``i`` covers ``[low + i*pitch, low + (i+1)*pitch)`` -- that is what
    # ``_surface_cells`` floors into -- so the sample that stands for it sits at
    # its *centre*, ``low + (i + 0.5)*pitch``. ``np.pad(..., 2)`` puts that
    # sample at field index ``i + 2``, so index ``v`` is at
    # ``low + (v - 1.5)*pitch``. Subtracting two whole cells instead moved every
    # carved object half a carve cell towards the origin corner: 3 mm on all
    # three axes, which is one and a half grid steps at 16 kHz. It showed up as
    # a fringe of boundary nodes on each object's -x, -y and -z faces and a
    # surface cut flush on the opposite three -- measured on the 68 carved
    # templates of this flat as a median centroid shift of (-2.9, -2.0, -2.7) mm
    # against their own colliders.
    carved = trimesh.Trimesh((vertices - 1.5) * pitch + low, faces, process=True)
    if not is_closed(carved):
        base.reason = "carve came back open"
        return base

    budget = max(BUDGET_FLOOR, int(BUDGET_FACTOR * len(collider.faces)))
    reduced = _to_budget(carved, budget)
    if reduced is None:
        base.reason = f"cannot reach {budget} triangles and stay closed"
        return base

    base.mesh = reduced
    base.carved = True
    base.carved_volume = abs(float(reduced.volume))
    isosurface_volume = abs(float(carved.volume))
    if isosurface_volume > 0.0:
        base.volume_error = abs(base.carved_volume - isosurface_volume) / isosurface_volume
    return base


def _stamp() -> str:
    """What a cache entry was made under, in one filename-safe string.

    The module's own source is in it, not only its constants. An earlier
    version of this file leaked through the flood fill and carved objects down
    to three per cent of themselves; the fix changed no constant, so entries
    written by the broken version answered for the fixed one and the repair
    looked like it had not worked. The voxelisation cache keys on its patched
    voxeliser's source for exactly this reason -- see
    ``reverberate.wave.voxelise.SceneSpec.key`` -- and this is the same hazard.
    """
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    return f"{CARVE_PITCH_M}_{BUDGET_FACTOR}_{BUDGET_FLOOR}_{MAX_CELLS}_{digest}"


def _cache_dir() -> Path:
    path = data_root() / "cache" / "carve"
    path.mkdir(parents=True, exist_ok=True)
    return path


def carve_collider(hssd_root: Path, template: str, collider: trimesh.Trimesh) -> CarveResult:
    """The carve for ``template``, from disk when it has been computed before.

    Cached on disk as well as in memory because the apartment places 137 unique
    templates and a wardrobe takes a minute; a second assembly of the same scene
    should not pay for it again. The entry records the pitch and the budget it
    was made under, so changing either invalidates it rather than serving a
    carve made under different rules.
    """
    stamp = _stamp()
    entry = _cache_dir() / f"{template}.{stamp}.json"
    mesh_file = entry.with_suffix(".glb")
    if entry.is_file():
        record = json.loads(entry.read_text())
        if not record["carved"]:
            return CarveResult(mesh=collider, carved=False, **record["stats"])
        loaded = trimesh.load(mesh_file, force="mesh")
        if isinstance(loaded, trimesh.Trimesh):
            return CarveResult(mesh=loaded, carved=True, **record["stats"])

    result = _carve_uncached(hssd_root, template, collider)
    stats = {
        "reason": result.reason,
        "collider_volume": result.collider_volume,
        "carved_volume": result.carved_volume,
        "volume_error": result.volume_error,
        "pitch_m": result.pitch_m,
        "leaked_at_m": result.leaked_at_m,
    }
    if result.carved:
        result.mesh.export(mesh_file)
    entry.write_text(json.dumps({"carved": result.carved, "stats": stats}, indent=2))
    return result
