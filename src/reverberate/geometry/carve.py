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

**It is never a silent substitution.** A carve that comes back empty, open, or
that the triangle budget cannot be reduced to without opening, is discarded and
the plain collider is used. :class:`CarveReport` names every template in each
case, and :func:`carve_summary` puts the counts where the manifest can carry
them. The whole point of the change is that the picture stops lying; a fallback
nobody can see would be the same fault in the other direction.

**The triangle budget is not optional.** Marching cubes on one wardrobe at 6 mm
returns 207 312 triangles against the collider's 5 848, and voxelisation cost is
driven by triangle count. Each carve is decimated to a multiple of the collider
it replaces, so the scene grows by a bounded factor rather than by whatever the
isosurface happened to need.
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
from reverberate.settings import data_root

__all__ = [
    "CARVE_PITCH_M",
    "CarveReport",
    "CarveResult",
    "carve_collider",
]

#: Side of the cell the carve is decided on, in metres. One fixed value for
#: every scene and every ``fmax``, deliberately: the bedroom at 16 kHz and the
#: apartment at 4 kHz must be the *same* geometry, or nothing measured on one
#: says anything about the other. 6 mm sits below the 8.17 mm grid step at
#: 4 kHz, so the carve never invents detail the coarser of the two grids cannot
#: see, and above the point where a template's occupancy array stops fitting in
#: memory.
CARVE_PITCH_M = 0.006

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

    def add(self, template: str, result: CarveResult) -> None:
        if result.carved:
            self.carved[template] = round(result.shrink, 4)
        else:
            self.skipped[template] = result.reason

    def summary(self) -> str:
        if not self.carved and not self.skipped:
            return "no carve"
        kept = np.mean(list(self.carved.values())) if self.carved else 1.0
        return (
            f"{len(self.carved)} colliders carved to {kept:.0%} of their volume, "
            f"{len(self.skipped)} left as they are"
        )


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

    Watertightness is the whole reason the collider is used at all, so a
    reduction that loses it is not a cheaper version of the mesh, it is a
    different kind of object.

    Asking for the budget once and giving up refused 11 of 47 templates: a
    marching cubes surface is uniform, and taking 46 000 triangles to 2 000 in
    one step pinches it open somewhere almost every time. So the budget is a
    target, not a cliff -- back off by doubling and take the first reduction
    that stays closed. Four steps is a factor of eight, past which the mesh is
    not worth the triangles and the collider is the better answer.
    """
    if len(mesh.faces) <= budget:
        return mesh
    import fast_simplification

    volume = abs(float(mesh.volume))
    for target in (budget, int(budget * 1.4), budget * 2):
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
        if len(reduced.faces) == 0 or not reduced.is_watertight:
            continue
        # Watertight is not enough on its own. Trimesh calls a mesh watertight
        # when every edge is used twice, and a pair of triangles back to back
        # satisfies that while enclosing nothing: reducing a twelve face box to
        # one triangle returns a "closed" body of zero volume. Half the volume
        # is a wide bound -- decimation of an isosurface loses a per cent or so
        # -- but it is the one that separates a simplified solid from a sheet.
        if abs(float(reduced.volume)) >= 0.5 * volume:
            return reduced
    # No reduction held. An isosurface that will not simplify is usually a
    # genuinely intricate shape rather than a broken one, so keep it undecimated
    # when it is small enough to afford, and let the collider stand otherwise.
    #
    # Two caps, because the budget is relative and the cost is not. A budget of
    # 4x a 108 face picture is 2 000 triangles and refuses a 12 000 triangle
    # carve that the scene would never notice; measured on this bedroom,
    # relative-only refused 17 of 41 templates and left them inflated. So also
    # allow anything under ABSOLUTE_CAP outright: past that a single object
    # starts to matter against a scene of a million and a half.
    if len(mesh.faces) <= max(int(1.25 * budget), ABSOLUTE_CAP):
        return mesh
    return None


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

    pitch = CARVE_PITCH_M
    low = np.minimum(render.bounds[0], collider.bounds[0]) - 3 * pitch
    high = np.maximum(render.bounds[1], collider.bounds[1]) + 3 * pitch
    cells = np.prod(np.ceil((high - low) / pitch) + 4)
    if cells > MAX_CELLS:
        base.reason = f"too large to carve at {pitch * 1000:.0f} mm ({cells / 1e6:.0f}M cells)"
        return base
    extent = np.ceil((high - low) / pitch).astype(np.int64) + 4
    shape: tuple[int, int, int] = (int(extent[0]), int(extent[1]), int(extent[2]))

    try:
        # The collider is closed, so its solid is everything the fill cannot
        # reach. Deriving it the same way as the render's air keeps one
        # rasteriser and one fill in the module rather than two conventions
        # that have to agree.
        solid = _eroded(~_outside(_surface_cells(collider, pitch, low, shape)))
        air = _outside(_surface_cells(render, pitch, low, shape))
    except Exception as error:  # trimesh raises several unrelated types here
        base.reason = f"rasterisation failed: {type(error).__name__}"
        return base
    if not solid.any():
        base.reason = "empty occupancy"
        return base

    # One erosion undoes the shell that conservative marking added. The
    # rasterisation deliberately over-marks so the fill cannot leak, which grows
    # the solid outward by about a cell; left in, it puts the carve *above* the
    # collider it is supposed to be shrinking -- measured at 114 per cent on one
    # decoration. Eroding by the same cell it was grown by restores the volume
    # and keeps the error symmetric instead of one-sided.
    kept = solid & ~air
    if not kept.any():
        base.reason = "carve removed everything"
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
    carved = trimesh.Trimesh(vertices * pitch + low - 2 * pitch, faces, process=True)
    if not carved.is_watertight:
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
    }
    if result.carved:
        result.mesh.export(mesh_file)
    entry.write_text(json.dumps({"carved": result.carved, "stats": stats}, indent=2))
    return result
