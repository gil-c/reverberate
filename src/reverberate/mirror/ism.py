"""Image sources on the derived facets, validated leg by leg against the occluders.

The numpy twin of the card's kernel. An image tree is grown level by level
over the reflecting facets: a facet mirrors every image of the level below
that lies on its air side, a facet never follows itself, and a sequence
holds at most ``furniture_bounces`` furniture facets, so the tree stays a
few thousand to a few tens of thousands of images on a storey. That tree
depends on the source alone and is shared by every receiver.

For one receiver an image is a reflection when its path exists: walking back
from the receiver, each leg must cross its facet inside the facet's own
triangles, and no occluder triangle may cut the leg, shrunk by ``epsilon_m``
at both ends so a surface the leg starts or ends on does not cut it. The
first leg's facet test discards most of the tree cheaply, and only the
survivors pay for the occlusion tests, which is the order the kernel keeps.

What comes out per receiver is what the renderer and the audit need: the
path length, the arrival direction at the receiver (towards the image, the
last leg's own line), the per band pressure gain (spherical spreading and
the facets' reflection factors), the facet sequence and the points of the
path. Air absorption is not here: the project applies it as post processing
(ADR 0009), and the mirror does the same on the rendered signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.rays import UniformGrid, triangle_grid

__all__ = [
    "ImageTree",
    "IsmSettings",
    "Paths",
    "dot3",
    "grow_tree",
    "norm3",
    "occluder_grid",
    "paths_for",
    "ray_triangles",
    "sheet_layers",
]

#: A facet that reflects on both sides, PFFDTD's ``3``.
BOTH_SIDES = 3
#: What :attr:`IsmSettings.coincident_facets` may be.
COINCIDENT_FACETS = ("once", "twice")
#: Two layers are one sheet when their planes and their extents agree to this, m.
SHEET_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class IsmSettings:
    """What is a choice in the image source model."""

    #: Longest facet sequence.
    max_order: int = 3
    #: At most this many furniture facets in one sequence; the rest are shell.
    furniture_bounces: int = 1
    sound_speed_m_s: float = 343.2
    #: Legs are shrunk by this at both ends before the occlusion test.
    epsilon_m: float = 0.005
    #: A tree larger than this is refused rather than grown for hours.
    max_images: int = 500_000
    #: Images farther than ``sound_speed * window_s`` from every receiver are
    #: pruned with their descendants: a descendant's unfolded path passes
    #: through its parent's image and can only be longer.
    window_s: float = 0.080
    #: Beyond ``max_order``, a sequence that ends on two parallel facets
    #: facing each other goes on alternating between them up to this order:
    #: the flutter between floor and ceiling, or two walls, whose fourth and
    #: fifth bounces the reference still holds above -15 dB while the
    #: general tree at that order would be millions of images.
    flutter_order: int = 6
    #: A sheet the model holds as two layers on one plane, each reflecting on
    #: both sides (:func:`sheet_layers`): ``"once"`` reads each layer on the
    #: side it faces, so a reflection on the sheet is one path; ``"twice"``
    #: lets both layers mirror a source on either side, and every path that
    #: reflects there is found under two names, 6 dB too loud. ``"twice"`` is
    #: what every field before 2026-10-05 was traced with.
    coincident_facets: str = "once"

    def __post_init__(self) -> None:
        if self.coincident_facets not in COINCIDENT_FACETS:
            raise ValueError(
                f"coincident_facets is one of {COINCIDENT_FACETS}, not {self.coincident_facets!r}"
            )

    def record(self) -> dict[str, Any]:
        return {
            "max_order": self.max_order,
            "furniture_bounces": self.furniture_bounces,
            "sound_speed_m_s": self.sound_speed_m_s,
            "epsilon_m": self.epsilon_m,
            "max_images": self.max_images,
            "window_s": self.window_s,
            "flutter_order": self.flutter_order,
            "coincident_facets": self.coincident_facets,
        }


@dataclass(frozen=True)
class ImageTree:
    """Every image of one source: position, order, parent and facet sequence."""

    source: np.ndarray
    positions: np.ndarray
    order: np.ndarray
    parent: np.ndarray
    #: ``[image, max_order]``, the facets in bounce order, ``-1`` padded.
    sequence: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class Paths:
    """The reflections one receiver hears: a subset of the tree, with their attributes."""

    receiver: np.ndarray
    image: np.ndarray
    order: np.ndarray
    length_m: np.ndarray
    #: Unit vectors from the receiver towards the image, scene frame.
    direction: np.ndarray
    #: ``[path, band]`` pressure gain: ``1 / length`` times the reflection factors.
    gain: np.ndarray
    #: ``[path, max_order + 2, 3]``: source, the hit points in bounce order, receiver;
    #: unused rows repeat the receiver.
    points: np.ndarray
    sequence: np.ndarray

    @property
    def count(self) -> int:
        return int(self.image.shape[0])

    def time_s(self, sound_speed_m_s: float) -> np.ndarray:
        return np.asarray(self.length_m / sound_speed_m_s)


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def dot3(a: Any, b: Any) -> Any:
    """The scalar product over a last axis of three, written out: ``(a0 b0 + a1 b1) + a2 b2``.

    Every quantity that decides whether a path exists is computed through
    this function, here and in :mod:`reverberate.mirror.moving`. A matrix
    product or an ``einsum`` gives the same sum in an order, and with or
    without a fused multiply, that the linear algebra library and the
    processor choose; three products and two sums are the same bits
    everywhere. A leg that grazes a facet's edge is then a path on every
    machine or on none, and for the tree as for the batched trace.
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def norm3(a: Any) -> Any:
    """The length over a last axis of three: the square root of :func:`dot3` of a vector."""
    return np.sqrt(a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1] + a[..., 2] * a[..., 2])


# --------------------------------------------------------------------------
# the tree
# --------------------------------------------------------------------------


def sheet_layers(scene: DerivedScene) -> np.ndarray:
    """Per facet: it reflects both ways and a second such facet lies on it, facing the other way.

    An unoriented sheet (a pane, a panel, a door leaf) comes from the model
    as two layers of the same triangles, one facing each way, both marked as
    reflecting on both sides. Each layer then mirrors a source on either
    side to the same image, the two paths cross the same point of the same
    triangles, and one reflection is rendered twice. Two facets are the two
    layers of one sheet when their normals are opposite, their planes one,
    and their boxes and areas the same to :data:`SHEET_TOLERANCE_M`. Read on
    the side it faces, each layer gives the sheet's reflection to its own
    side and to no other, which is the sheet.
    """
    count = len(scene.facets)
    layer = np.zeros(count, dtype=bool)
    both = np.flatnonzero([f.sides == BOTH_SIDES for f in scene.facets])
    if both.size < 2:
        return layer
    normals = np.asarray([scene.facets[i].normal for i in both], dtype=float)
    offsets = np.asarray([scene.facets[i].offset for i in both], dtype=float)
    areas = np.asarray([scene.facets[i].area for i in both], dtype=float)
    boxes = np.zeros((both.size, 6))
    for row, i in enumerate(both):
        vertices = scene.reflector_vertices[scene.facets[i].triangles].reshape(-1, 3)
        boxes[row] = np.concatenate([vertices.min(axis=0), vertices.max(axis=0)])
    opposite = normals @ normals.T < -1.0 + 1e-6
    one_plane = np.abs(offsets[:, None] + offsets[None, :]) < SHEET_TOLERANCE_M
    one_box = np.abs(boxes[:, None, :] - boxes[None, :, :]).max(axis=2) < SHEET_TOLERANCE_M
    one_area = np.abs(areas[:, None] - areas[None, :]) <= 1e-3 * np.maximum(areas[:, None], 1e-12)
    layer[both] = np.any(opposite & one_plane & one_box & one_area, axis=1)
    return layer


def _facet_arrays(
    scene: DerivedScene, coincident_facets: str = "once"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per facet: its normal, its offset, whether it is furniture, whether it reflects both ways.

    With ``coincident_facets`` ``"once"`` a layer of a two layer sheet
    (:func:`sheet_layers`) reflects on the side it faces alone.
    """
    normals = np.asarray([f.normal for f in scene.facets], dtype=float).reshape(-1, 3)
    offsets = np.asarray([f.offset for f in scene.facets], dtype=float)
    furniture = np.asarray([f.kind == "furniture" for f in scene.facets], dtype=bool)
    both = np.asarray([f.sides == BOTH_SIDES for f in scene.facets], dtype=bool)
    if coincident_facets == "once":
        both &= ~sheet_layers(scene)
    return normals, offsets, furniture, both


def _facet_spheres(scene: DerivedScene) -> tuple[np.ndarray, np.ndarray]:
    """A bounding sphere per facet: centre and radius over its triangles."""
    centres = np.zeros((len(scene.facets), 3))
    radii = np.zeros(len(scene.facets))
    for i, facet in enumerate(scene.facets):
        vertices = scene.reflector_vertices[facet.triangles].reshape(-1, 3)
        centres[i] = vertices.mean(axis=0)
        radii[i] = float(np.max(np.linalg.norm(vertices - centres[i], axis=1)))
    return centres, radii


def _in_front(
    scene: DerivedScene,
    normals: np.ndarray,
    offsets: np.ndarray,
    coincident_facets: str = "once",
) -> np.ndarray:
    """``[i, j]``: some vertex of facet ``j`` lies strictly on the air side of facet ``i``'s plane.

    A path that has just reflected on facet ``i`` runs on its air side, so it
    can only reach a facet that has some of itself there.
    """
    count = len(scene.facets)
    front = np.zeros((count, count), dtype=bool)
    behind = np.zeros((count, count), dtype=bool)
    for j, facet in enumerate(scene.facets):
        vertices = scene.reflector_vertices[facet.triangles].reshape(-1, 3)
        heights = vertices @ normals.T - offsets[None, :]  # [vertex, i]
        front[:, j] = np.any(heights > 1e-9, axis=0)
        behind[:, j] = np.any(heights < -1e-9, axis=0)
    # A facet that reflects on both sides sends sound to either of them.
    both = _facet_arrays(scene, coincident_facets)[3]
    return front | (both[:, None] & behind)


def _through_the_beam(
    parents: np.ndarray,
    last: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """``[parent, facet]``: the facet's sphere meets the cone from the image through the last facet.

    The cone has its apex at the parent image, its axis towards the last
    facet's sphere centre and the half angle that sphere subtends; a facet
    whose own sphere lies wholly outside the cone cannot be reached through
    the last facet. Conservative: it never prunes a reachable facet.
    """
    axis = centres[last] - parents  # [parent, 3]
    distance = norm3(axis)
    axis = axis / np.maximum(distance, 1e-12)[:, None]
    # An apex inside the sphere sees the facet in every direction of its air
    # side: the cone is the whole space and only the air side test prunes. A
    # large ceiling's sphere holds every image made under it.
    inside = distance <= radii[last]
    half_angle = np.where(
        inside, np.pi, np.arcsin(np.clip(radii[last] / np.maximum(distance, 1e-12), 0.0, 1.0))
    )
    to_facet = centres[None, :, :] - parents[:, None, :]  # [parent, facet, 3]
    reach = norm3(to_facet)
    cosine = dot3(to_facet, axis[:, None, :]) / np.maximum(reach, 1e-12)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    subtended = np.arcsin(np.clip(radii[None, :] / np.maximum(reach, 1e-12), 0.0, 1.0))
    return np.asarray(angle <= half_angle[:, None] + subtended)


def _box_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance from each point to an axis aligned box, zero inside it."""
    gap = np.maximum(np.maximum(lo[None, :] - points, points - hi[None, :]), 0.0)
    return np.asarray(norm3(gap))


def grow_tree(
    scene: DerivedScene,
    source: np.ndarray,
    settings: IsmSettings | None = None,
    *,
    region: tuple[np.ndarray, np.ndarray] | None = None,
) -> ImageTree:
    """Every image of ``source`` up to ``max_order``, in level order, the source first.

    ``region`` is the box the receivers lie in; images whose unfolded path to
    it exceeds the window are pruned with their descendants. Without it the
    tree is complete.
    """
    settings = settings or IsmSettings()
    source = np.asarray(source, dtype=float).reshape(3)
    reach = settings.sound_speed_m_s * settings.window_s
    normals, offsets, furniture, both = _facet_arrays(scene, settings.coincident_facets)
    count = normals.shape[0]
    centres, radii = _facet_spheres(scene)
    front = (
        _in_front(scene, normals, offsets, settings.coincident_facets)
        if count
        else np.zeros((0, 0), dtype=bool)
    )
    positions = [source[None, :]]
    orders = [np.zeros(1, dtype=np.int32)]
    parents = [np.full(1, -1, dtype=np.int32)]
    width = max(settings.max_order, settings.flutter_order)
    sequences = [np.full((1, width), -1, dtype=np.int32)]
    furniture_used = [np.zeros(1, dtype=np.int32)]
    level_start = 0
    total = 1
    level_done = 0
    for level in range(1, settings.max_order + 1):
        parent_pos = positions[-1]
        parent_seq = sequences[-1]
        parent_furn = furniture_used[-1]
        if count == 0 or parent_pos.shape[0] == 0:
            break
        # Signed distance of every parent to every facet's plane: [parent, facet].
        height = dot3(parent_pos[:, None, :], normals[None, :, :]) - offsets[None, :]
        allowed = (height > 0.0) | both[None, :]
        # A facet never follows itself, and the next facet must have some of
        # itself on the air side of the last one and inside the beam the last
        # one opens from the image.
        last = parent_seq[:, level - 2] if level >= 2 else np.full(parent_pos.shape[0], -1)
        allowed &= np.arange(count)[None, :] != last[:, None]
        if level >= 2:
            allowed &= front[last]
            allowed &= _through_the_beam(parent_pos, last, centres, radii)
        # The furniture budget.
        allowed &= (parent_furn[:, None] + furniture[None, :].astype(np.int32)) <= (
            settings.furniture_bounces
        )
        rows, cols = np.nonzero(allowed)
        if rows.size == 0:
            break
        mirrored = parent_pos[rows] - 2.0 * height[rows, cols][:, None] * normals[cols]
        if region is not None:
            within = _box_distance(mirrored, region[0], region[1]) <= reach
            rows, cols, mirrored = rows[within], cols[within], mirrored[within]
            if rows.size == 0:
                break
        seq = parent_seq[rows].copy()
        seq[:, level - 1] = cols
        positions.append(mirrored)
        orders.append(np.full(rows.size, level, dtype=np.int32))
        parents.append((rows + level_start).astype(np.int32))
        sequences.append(seq)
        furniture_used.append(parent_furn[rows] + furniture[cols].astype(np.int32))
        level_start = total
        total += rows.size
        level_done = level
        if total > settings.max_images:
            raise ValueError(
                f"the image tree passes {settings.max_images} images at order {level};"
                " lower the order or the reflector count"
            )
    # The flutter: past the general tree, a sequence ending on two shell
    # facets that face each other keeps alternating between them.
    while level_done >= 2 and level_done < settings.flutter_order:
        parent_pos = positions[-1]
        parent_seq = sequences[-1]
        last = parent_seq[:, level_done - 1]
        before = parent_seq[:, level_done - 2]
        facing = (last >= 0) & (before >= 0)
        facing &= ~furniture[np.maximum(last, 0)] & ~furniture[np.maximum(before, 0)]
        facing &= dot3(normals[np.maximum(last, 0)], normals[np.maximum(before, 0)]) < -0.99
        rows = np.flatnonzero(facing)
        if rows.size == 0:
            break
        cols = np.asarray(before[rows], dtype=np.int64)
        height = dot3(parent_pos[rows], normals[cols]) - offsets[cols]
        keep = (height > 0.0) | both[cols]
        rows, cols, height = rows[keep], cols[keep], height[keep]
        if rows.size == 0:
            break
        mirrored = parent_pos[rows] - 2.0 * height[:, None] * normals[cols]
        if region is not None:
            within = _box_distance(mirrored, region[0], region[1]) <= reach
            rows, cols, mirrored = rows[within], cols[within], mirrored[within]
            if rows.size == 0:
                break
        level_done += 1
        seq = parent_seq[rows].copy()
        seq[:, level_done - 1] = cols
        positions.append(mirrored)
        orders.append(np.full(rows.size, level_done, dtype=np.int32))
        parents.append((rows + level_start).astype(np.int32))
        sequences.append(seq)
        furniture_used.append(furniture_used[-1][rows])
        level_start = total
        total += rows.size
    return ImageTree(
        source=source,
        positions=np.concatenate(positions),
        order=np.concatenate(orders),
        parent=np.concatenate(parents),
        sequence=np.concatenate(sequences),
    )


# --------------------------------------------------------------------------
# rays against triangles
# --------------------------------------------------------------------------


def ray_triangles(
    origins: np.ndarray, ends: np.ndarray, triangles: np.ndarray, *, strict: float = 0.0
) -> np.ndarray:
    """Whether each segment ``origins[i] -> ends[i]`` hits any of ``triangles``.

    Moller and Trumbore, vectorised over segments and triangles at once, so
    the caller keeps the product of the two counts small. ``strict`` shrinks
    the segment's parameter range to ``[strict, 1 - strict]``.
    """
    origins = np.atleast_2d(origins)
    ends = np.atleast_2d(ends)
    if triangles.shape[0] == 0 or origins.shape[0] == 0:
        return np.zeros(origins.shape[0], dtype=bool)
    direction = ends - origins  # [s, 3]
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    # [s, t, 3]
    p = np.cross(direction[:, None, :], e2[None, :, :])
    det = dot3(e1[None, :, :], p)
    parallel = np.abs(det) < 1e-12
    inv = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
    t_vec = origins[:, None, :] - v0[None, :, :]
    u = dot3(t_vec, p) * inv
    q = np.cross(t_vec, e1[None, :, :])
    v = dot3(direction[:, None, :], q) * inv
    t = dot3(e2[None, :, :], q) * inv
    hit = (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    hit &= (t >= strict) & (t <= 1.0 - strict)
    return np.asarray(hit.any(axis=1))


def _segment_hits(
    origins: np.ndarray,
    ends: np.ndarray,
    triangles: np.ndarray,
    grid: UniformGrid,
    *,
    epsilon_m: float,
) -> np.ndarray:
    """Occlusion of many segments, each tested against the triangles of the cells it crosses.

    Every segment is shrunk by ``epsilon_m`` at both ends, so a surface it
    starts or ends on does not cut it; a segment shorter than twice that
    has nothing left to test and is clear. The card does the same, leg by
    leg.
    """
    out = np.zeros(origins.shape[0], dtype=bool)
    for i in range(origins.shape[0]):
        length = float(norm3(ends[i] - origins[i]))
        if length <= 2.0 * epsilon_m:
            continue
        strict = epsilon_m / length
        near = grid.candidates(origins[i], ends[i])
        if near.size == 0:
            continue
        # In blocks, so the [segment, triangle] product stays in memory.
        for start in range(0, near.size, 65536):
            block = triangles[near[start : start + 65536]]
            if ray_triangles(origins[i : i + 1], ends[i : i + 1], block, strict=strict)[0]:
                out[i] = True
                break
    return out


def _plane_crossing(
    origins: np.ndarray, ends: np.ndarray, normal: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter and point where each segment crosses the plane; NaN when it does not."""
    heights_a = dot3(origins, normal[None, :]) - offset
    heights_b = dot3(ends, normal[None, :]) - offset
    denominator = heights_a - heights_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denominator) > 1e-12, heights_a / denominator, np.nan)
    inside = (t > 0.0) & (t < 1.0)
    t = np.where(inside, t, np.nan)
    points = origins + t[:, None] * (ends - origins)
    return t, points


# --------------------------------------------------------------------------
# paths of one receiver
# --------------------------------------------------------------------------


def occluder_grid(scene: DerivedScene, cell_m: float = 0.25) -> UniformGrid:
    """The uniform grid of the scene's occluders, built once and handed to every receiver."""
    return triangle_grid(scene.occluder_vertices, cell_m, scene.bmin, scene.bmax)


def paths_for(
    scene: DerivedScene,
    tree: ImageTree,
    receiver: np.ndarray,
    settings: IsmSettings | None = None,
    *,
    grid: UniformGrid | None = None,
) -> Paths:
    """The reflections of ``receiver`` from the tree, each validated leg by leg.

    ``grid`` is :func:`occluder_grid` of the scene; built here when not given,
    which a caller with many receivers should not let happen.
    """
    settings = settings or IsmSettings()
    receiver = np.asarray(receiver, dtype=float).reshape(3)
    normals, offsets, _, _ = _facet_arrays(scene)
    occluders = scene.occluder_vertices
    if grid is None:
        grid = occluder_grid(scene)
    n_images = tree.count
    max_order = tree.sequence.shape[1]
    # Points of every candidate path: [image, max_order + 2, 3], receiver-filled.
    points = np.repeat(receiver[None, None, :], n_images, axis=0)
    points = np.repeat(points, max_order + 2, axis=1)
    points[:, 0] = tree.source
    alive = np.ones(n_images, dtype=bool)
    # Walk back from the receiver: the current image of every path, its "far end".
    current = np.arange(n_images)
    far = tree.positions.copy()
    near = np.repeat(receiver[None, :], n_images, axis=0)
    for step in range(max_order):
        # Paths whose order is at least (max_order - step) still have a facet to cross.
        bounce = tree.order - 1 - step  # index into the sequence of the facet crossed now
        active = alive & (bounce >= 0)
        if not active.any():
            break
        idx = np.flatnonzero(active)
        facet_of = tree.sequence[idx, bounce[idx]]
        hit_points = np.full((idx.size, 3), np.nan)
        for facet in np.unique(facet_of):
            rows = idx[facet_of == facet]
            local = np.flatnonzero(facet_of == facet)
            _, crossing = _plane_crossing(near[rows], far[rows], normals[facet], offsets[facet])
            ok = ~np.isnan(crossing[:, 0])
            if ok.any():
                triangles = scene.reflector_vertices[scene.facets[facet].triangles]
                inside = np.zeros(rows.size, dtype=bool)
                # The crossing point lies on the plane: test a short segment
                # through it along the normal against the facet's triangles.
                probe_a = crossing[ok] - 1e-4 * normals[facet]
                probe_b = crossing[ok] + 1e-4 * normals[facet]
                inside[ok] = ray_triangles(probe_a, probe_b, triangles)
                ok &= inside
            alive[rows[~ok]] = False
            hit_points[local[ok]] = crossing[ok]
        still = idx[alive[idx]]
        if still.size == 0:
            break
        # The leg from the near end to the hit point must be clear.
        hits = hit_points[alive[idx]]
        blocked = _segment_hits(near[still], hits, occluders, grid, epsilon_m=settings.epsilon_m)
        alive[still[blocked]] = False
        kept = still[~blocked]
        hits = hits[~blocked]
        # Record the hit as the path's point after the source: bounce order
        # position is (order - step), counted from the source side.
        slot = tree.order[kept] - step
        points[kept, slot] = hits
        # Continue from the hit towards the parent image.
        near[kept] = hits
        current[kept] = tree.parent[current[kept]]
        far[kept] = tree.positions[current[kept]]
    # The last leg, to the source itself, for every surviving path.
    kept = np.flatnonzero(alive)
    if kept.size:
        blocked = _segment_hits(
            near[kept], far[kept], occluders, grid, epsilon_m=settings.epsilon_m
        )
        alive[kept[blocked]] = False
    kept = np.flatnonzero(alive)
    lengths = norm3(tree.positions[kept] - receiver[None, :])
    direction = (tree.positions[kept] - receiver[None, :]) / np.maximum(lengths, 1e-12)[:, None]
    gain = _gains(scene, tree.sequence[kept], lengths)
    return Paths(
        receiver=receiver,
        image=kept.astype(np.int32),
        order=tree.order[kept],
        length_m=lengths,
        direction=direction,
        gain=gain,
        points=points[kept],
        sequence=tree.sequence[kept],
    )


def _gains(scene: DerivedScene, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``[path, band]`` pressure gains: spreading times the reflection factors of the sequence."""
    absorption = scene.materials.absorption
    scattering = scene.materials.scattering
    labels = np.asarray([f.label for f in scene.facets], dtype=int)
    factor = (
        np.sqrt(np.clip(1.0 - absorption, 0.0, 1.0))
        * np.sqrt(np.clip(1.0 - scattering, 0.0, 1.0))[:, None]
    )  # [label, band]
    per_facet = factor[labels] if labels.size else np.zeros((0, absorption.shape[1]))
    gain = np.ones((sequences.shape[0], absorption.shape[1]))
    for k in range(sequences.shape[1]):
        used = sequences[:, k] >= 0
        if used.any():
            gain[used] *= per_facet[sequences[used, k]]
    return np.asarray(gain / np.maximum(lengths, 1e-12)[:, None])
