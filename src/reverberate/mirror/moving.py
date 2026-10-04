"""The early arrivals of a scene that moves: one table per source, a block of rows per step.

:func:`reverberate.mirror.pipeline.trace` answers one fixed source over a
lattice. A moving scene asks, at every 50 ms step, what a listener that has
moved hears of a source that has moved, and wants the answer as a table of
arrivals (``docs/formats/scene-pack.md``, group ``early``), not as a rendered
response. :func:`trace_early` is that table for one source.

**What is computed when.**

- *Once per scene* (:func:`prepare`): the facets' planes, bounding spheres
  and mutual visibility, their buckets of triangles, the occluders' grid, the
  materials the image sources read.
- *Once per source anchor* (:func:`grow_candidates`): the facet sequences an
  image tree can hold for any source within ``slack`` of the anchor. Anchors
  stand on a grid of :attr:`MovingSettings.source_pitch_m`, so a station is
  one anchor and a rail a few. An image is its facet sequence wherever the
  source stands, and the sequences change slowly: the tree is not grown
  again at each step.
- *Once per pair of anchors*, the source's and the listener's: the short
  list of sequences whose legs can cross their facets at all from anywhere in
  the two cells, a walk of the unfolded path with the cells' size as its
  margin (:func:`_walk`). Most of the tree leaves here.
- *Once per step*: the same walk over the short list with the images where
  the step puts them, through each image's affine map, which leaves the few
  hundred sequences that nearly are paths; then, for those alone, what the
  present pipeline computes: the images mirrored facet by facet from the
  source, the tree's own tests at that position (the air side of each facet,
  the beam, the window), so the step's tree is the one
  :func:`reverberate.mirror.ism.grow_tree` would grow there, and the legs,
  each crossing its facet inside the facet's triangles and clear of the
  occluders (:func:`_validate`). The steps are not taken one at a time: both
  stages run on flat arrays of (step, image) pairs on ``xp``, numpy on the
  host, cupy on a card.

A step whose source and listener stand where an earlier step's did is not
computed again.

**At rest** the table is the present pipeline's: the same paths in the same
order with the same lengths, directions and gains, so
:func:`render_early` reproduces its early part sample for sample. The
diffracted onset of a step without a direct path
(:mod:`reverberate.mirror.moving_onset`) is the pipeline's too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from reverberate.compute import to_numpy
from reverberate.mirror.engine import facet_buckets
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import (
    IsmSettings,
    Paths,
    _box_distance,
    _facet_arrays,
    _facet_spheres,
    _gains,
    _in_front,
    occluder_grid,
)
from reverberate.mirror.parameters import apply_parameters, image_scene
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.render import RenderSettings, early_signals

__all__ = [
    "KIND_DIFFRACTED",
    "KIND_DIFFRACTED_REFLECTED",
    "KIND_DIRECT",
    "KIND_SPECULAR",
    "Candidates",
    "EarlyTable",
    "MovingScene",
    "MovingSettings",
    "grow_candidates",
    "path_id",
    "prepare",
    "render_early",
    "trace_early",
]

#: The pack's ``kind`` of a row.
KIND_DIRECT = 0
KIND_SPECULAR = 1
KIND_DIFFRACTED = 2
KIND_DIFFRACTED_REFLECTED = 3


@dataclass(frozen=True)
class MovingSettings:
    """What the batched trace chooses. None of it changes a path: only the time."""

    #: Pitch of the grid the source's anchors stand on. A candidate tree
    #: serves every source position in its anchor's cell.
    source_pitch_m: float = 0.5
    #: Pitch of the listener's cells: a short list of sequences serves every
    #: step whose source and listener stay in one pair of cells.
    listener_pitch_m: float = 0.5
    #: Cell of the occluders' grid the legs are walked through.
    occluder_cell_m: float = 0.10
    #: (step, image) pairs sieved at once.
    pairs_per_block: int = 1_000_000
    #: Sieved pairs validated at once: the occluders are walked cell by cell,
    #: every leg in step, and a walk costs the same for a thousand legs as for ten.
    pairs_per_validation: int = 300_000
    #: Candidate trees kept, one per source anchor.
    anchors_kept: int = 16
    #: Short lists kept, one per pair of anchors: a listener that comes back
    #: to a cell finds its list.
    lists_kept: int = 512
    #: (segment, triangle) tests made at once.
    tests_per_block: int = 4_000_000


# --------------------------------------------------------------------------
# once per scene
# --------------------------------------------------------------------------


@dataclass
class MovingScene:
    """What does not depend on where anything stands. Built by :func:`prepare`."""

    #: The scene the paths are validated on, and the one their gains are read from.
    scene: DerivedScene
    images: DerivedScene
    #: The catalogue itself: what the diffracted onset is computed on.
    catalogue: DerivedScene
    ism: IsmSettings
    moving: MovingSettings
    normals: np.ndarray
    offsets: np.ndarray
    furniture: np.ndarray
    both: np.ndarray
    centres: np.ndarray
    radii: np.ndarray
    front: np.ndarray
    #: Per facet, the rectangle of its buckets in its own plane:
    #: ``[facet, 10]`` origin u, origin v, extent u, extent v, axis u, axis v.
    rectangle: np.ndarray
    host: dict[str, np.ndarray]
    _device: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    _candidates: dict[tuple[Any, ...], Candidates] = field(default_factory=dict, repr=False)
    _lists: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def width(self) -> int:
        """Columns of a facet sequence: the longest the tree can hold."""
        return max(self.ism.max_order, self.ism.flutter_order)

    def on(self, xp: Any) -> dict[str, Any]:
        """The arrays the validation reads, on ``xp``; moved there once."""
        held = self._device.get(id(xp))
        if held is None:
            held = {name: xp.asarray(array) for name, array in self.host.items()}
            self._device[id(xp)] = held
        return held


def prepare(
    catalogue: DerivedScene,
    settings: MirrorSettings | None = None,
    moving: MovingSettings | None = None,
) -> MovingScene:
    """Everything of the scene the steps share, from the catalogue and the mirror's settings."""
    settings = settings or MirrorSettings()
    moving = moving or MovingSettings()
    scene = apply_parameters(catalogue, settings.parameters)
    images = image_scene(catalogue, settings.parameters)
    ism = replace(settings.ism, sound_speed_m_s=settings.sound_speed_m_s)
    normals, offsets, furniture, both = _facet_arrays(scene)
    count = normals.shape[0]
    centres, radii = _facet_spheres(scene)
    front = _in_front(scene, normals, offsets) if count else np.zeros((0, 0), dtype=bool)
    frame, shape, base, bucket_offsets, bucket_members = facet_buckets(scene)
    grid = occluder_grid(scene, moving.occluder_cell_m)
    rectangle = np.zeros((count, 10))
    if count:
        rectangle[:, 0:2] = frame[:, 0:2]
        rectangle[:, 2:4] = shape * frame[:, 2:3]
        rectangle[:, 4:10] = frame[:, 3:9]
    reflectors = np.asarray(scene.reflector_vertices, dtype=float).reshape(-1, 3, 3)
    occluders = np.asarray(scene.occluder_vertices, dtype=float).reshape(-1, 3, 3)
    host = {
        "normals": np.ascontiguousarray(normals, dtype=float),
        "offsets": np.ascontiguousarray(offsets, dtype=float),
        "both": np.ascontiguousarray(both),
        "centres": centres,
        "radii": radii,
        "rectangle": rectangle,
        "frame": np.ascontiguousarray(frame, dtype=float),
        "shape": np.ascontiguousarray(shape, dtype=np.int64),
        "base": np.ascontiguousarray(base, dtype=np.int64),
        "bucket_offsets": np.ascontiguousarray(bucket_offsets, dtype=np.int64),
        "bucket_members": np.ascontiguousarray(bucket_members, dtype=np.int64),
        "reflector_v0": np.ascontiguousarray(reflectors[:, 0]),
        "reflector_e1": np.ascontiguousarray(reflectors[:, 1] - reflectors[:, 0]),
        "reflector_e2": np.ascontiguousarray(reflectors[:, 2] - reflectors[:, 0]),
        "occluder_v0": np.ascontiguousarray(occluders[:, 0]),
        "occluder_e1": np.ascontiguousarray(occluders[:, 1] - occluders[:, 0]),
        "occluder_e2": np.ascontiguousarray(occluders[:, 2] - occluders[:, 0]),
        "grid_origin": np.asarray(grid.origin, dtype=float),
        "grid_shape": np.asarray(grid.shape, dtype=np.int64),
        "grid_cell": np.asarray(float(grid.cell_m)),
        "cell_offsets": np.asarray(grid.offsets, dtype=np.int64),
        "cell_members": np.asarray(grid.members, dtype=np.int64),
    }
    return MovingScene(
        scene=scene,
        images=images,
        catalogue=catalogue,
        ism=ism,
        moving=moving,
        normals=host["normals"],
        offsets=host["offsets"],
        furniture=np.asarray(furniture, dtype=bool),
        both=host["both"],
        centres=centres,
        radii=radii,
        front=front,
        rectangle=rectangle,
        host=host,
    )


# --------------------------------------------------------------------------
# once per source anchor
# --------------------------------------------------------------------------


@dataclass
class Candidates:
    """The sequences a tree may hold for a source within ``slack_m`` of ``anchor``.

    In the order :func:`reverberate.mirror.ism.grow_tree` grows them: by
    level, and within a level by sequence. The tree of any source position in
    the cell is a subset in the same order.
    """

    anchor: np.ndarray
    slack_m: float
    #: Where each image stands when the source is on the anchor.
    positions: np.ndarray
    order: np.ndarray
    parent: np.ndarray
    #: ``[image, width]``, ``-1`` padded.
    sequence: np.ndarray
    #: How many of the sequence's facets the general rule added; the rest are flutter.
    general: np.ndarray
    #: An image is an affine function of its source: ``rotation @ source + translation``.
    rotation: np.ndarray
    translation: np.ndarray
    _device: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def count(self) -> int:
        return int(self.order.shape[0])

    def on(self, xp: Any) -> dict[str, Any]:
        """The tree's arrays on ``xp``; moved there once."""
        held = self._device.get(id(xp))
        if held is None:
            held = {
                "positions": xp.asarray(self.positions),
                "order": xp.asarray(self.order.astype(np.int64)),
                "parent": xp.asarray(self.parent.astype(np.int64)),
                "sequence": xp.asarray(self.sequence.astype(np.int64)),
                "general": xp.asarray(self.general.astype(np.int64)),
                "rotation": xp.asarray(self.rotation.reshape(-1, 9)),
                "translation": xp.asarray(self.translation),
            }
            self._device[id(xp)] = held
        return held


def _beam_with_slack(
    parents: np.ndarray, last: np.ndarray, centres: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """:func:`reverberate.mirror.ism._through_the_beam` with an apex that may have moved.

    ``radii`` are already grown by the slack: moving the apex by a vector is
    moving both spheres by its opposite. An apex inside the facet's own
    sphere sees it in every direction, which the tree's test does not grant:
    this one must hold every sequence that one can.
    """
    axis = centres[last] - parents
    distance = np.linalg.norm(axis, axis=1)
    axis = axis / np.maximum(distance, 1e-12)[:, None]
    inside = distance <= radii[last]
    half_angle = np.where(
        inside, np.pi, np.arcsin(np.clip(radii[last] / np.maximum(distance, 1e-12), 0.0, 1.0))
    )
    to_facet = centres[None, :, :] - parents[:, None, :]
    reach = np.linalg.norm(to_facet, axis=2)
    cosine = np.einsum("pfk,pk->pf", to_facet, axis) / np.maximum(reach, 1e-12)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    subtended = np.where(
        reach <= radii[None, :],
        np.pi,
        np.arcsin(np.clip(radii[None, :] / np.maximum(reach, 1e-12), 0.0, 1.0)),
    )
    # A hair of slack of its own: the step's test is in other arithmetic.
    return np.asarray(angle <= half_angle[:, None] + subtended + 1e-9)


def grow_candidates(
    ms: MovingScene,
    anchor: np.ndarray,
    slack_m: float,
    region: tuple[np.ndarray, np.ndarray] | None = None,
) -> Candidates:
    """Every sequence the tree of a source within ``slack_m`` of ``anchor`` can hold.

    :func:`reverberate.mirror.ism.grow_tree` with each of its tests that reads
    the source's position loosened by the slack: an image moves as far as its
    source does, a mirror being an isometry. The tests that do not read it
    (a facet never follows itself, the furniture budget, which facets face
    which) are the tree's own.
    """
    ism = ms.ism
    anchor = np.asarray(anchor, dtype=float).reshape(3)
    slack = float(slack_m)
    reach = ism.sound_speed_m_s * ism.window_s + slack
    normals, offsets, furniture, both = ms.normals, ms.offsets, ms.furniture, ms.both
    count = normals.shape[0]
    radii = ms.radii + slack
    width = ms.width
    positions = [anchor[None, :]]
    orders = [np.zeros(1, dtype=np.int32)]
    parents = [np.full(1, -1, dtype=np.int32)]
    sequences = [np.full((1, width), -1, dtype=np.int32)]
    generals = [np.zeros(1, dtype=np.int32)]
    furniture_used = [np.zeros(1, dtype=np.int32)]
    rotations = [np.eye(3)[None, :, :]]
    translations = [np.zeros((1, 3))]
    level_start = 0
    total = 1
    level_done = 0

    def mirror_maps(rows: np.ndarray, cols: np.ndarray) -> None:
        """The children's affine maps: a mirror in the facet after the parent's own map."""
        normal = normals[cols]
        householder = np.eye(3)[None, :, :] - 2.0 * normal[:, :, None] * normal[:, None, :]
        rotations.append(householder @ rotations[-1][rows])
        translations.append(
            np.einsum("nij,nj->ni", householder, translations[-1][rows])
            + 2.0 * offsets[cols][:, None] * normal
        )

    def within(rows: np.ndarray, cols: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
        if region is None:
            return np.ones(rows.size, dtype=bool)
        return np.asarray(_box_distance(mirrored, region[0], region[1]) <= reach)

    def refuse(level: int) -> None:
        if total > ism.max_images:
            raise ValueError(
                f"the candidate tree passes {ism.max_images} images at order {level};"
                " lower the order, the reflector count or the source pitch"
            )

    for level in range(1, ism.max_order + 1):
        parent_pos = positions[-1]
        parent_seq = sequences[-1]
        parent_furn = furniture_used[-1]
        if count == 0 or parent_pos.shape[0] == 0:
            break
        height = parent_pos @ normals.T - offsets[None, :]
        allowed = (height > -slack) | both[None, :]
        last = parent_seq[:, level - 2] if level >= 2 else np.full(parent_pos.shape[0], -1)
        allowed &= np.arange(count)[None, :] != last[:, None]
        if level >= 2:
            allowed &= ms.front[last]
            allowed &= _beam_with_slack(parent_pos, last, ms.centres, radii)
        allowed &= (parent_furn[:, None] + furniture[None, :].astype(np.int32)) <= (
            ism.furniture_bounces
        )
        rows, cols = np.nonzero(allowed)
        if rows.size == 0:
            break
        mirrored = parent_pos[rows] - 2.0 * height[rows, cols][:, None] * normals[cols]
        keep = within(rows, cols, mirrored)
        rows, cols, mirrored = rows[keep], cols[keep], mirrored[keep]
        if rows.size == 0:
            break
        seq = parent_seq[rows].copy()
        seq[:, level - 1] = cols
        mirror_maps(rows, cols)
        positions.append(mirrored)
        orders.append(np.full(rows.size, level, dtype=np.int32))
        parents.append((rows + level_start).astype(np.int32))
        sequences.append(seq)
        generals.append(np.full(rows.size, level, dtype=np.int32))
        furniture_used.append(parent_furn[rows] + furniture[cols].astype(np.int32))
        level_start = total
        total += rows.size
        level_done = level
        refuse(level)
    general_levels = level_done
    while level_done >= 2 and level_done < ism.flutter_order:
        parent_pos = positions[-1]
        parent_seq = sequences[-1]
        last = parent_seq[:, level_done - 1]
        before = parent_seq[:, level_done - 2]
        facing = (last >= 0) & (before >= 0)
        facing &= ~furniture[np.maximum(last, 0)] & ~furniture[np.maximum(before, 0)]
        facing &= (
            np.einsum("ij,ij->i", normals[np.maximum(last, 0)], normals[np.maximum(before, 0)])
            < -0.99
        )
        rows = np.flatnonzero(facing)
        if rows.size == 0:
            break
        cols = np.asarray(before[rows], dtype=np.int64)
        height = np.einsum("ij,ij->i", parent_pos[rows], normals[cols]) - offsets[cols]
        keep = (height > -slack) | both[cols]
        rows, cols, height = rows[keep], cols[keep], height[keep]
        if rows.size == 0:
            break
        mirrored = parent_pos[rows] - 2.0 * height[:, None] * normals[cols]
        keep = within(rows, cols, mirrored)
        rows, cols, mirrored = rows[keep], cols[keep], mirrored[keep]
        if rows.size == 0:
            break
        level_done += 1
        seq = parent_seq[rows].copy()
        seq[:, level_done - 1] = cols
        mirror_maps(rows, cols)
        positions.append(mirrored)
        orders.append(np.full(rows.size, level_done, dtype=np.int32))
        parents.append((rows + level_start).astype(np.int32))
        sequences.append(seq)
        generals.append(np.full(rows.size, general_levels, dtype=np.int32))
        level_start = total
        total += rows.size
        refuse(level_done)
    return Candidates(
        anchor=anchor,
        slack_m=slack,
        positions=np.concatenate(positions),
        order=np.concatenate(orders),
        parent=np.concatenate(parents),
        sequence=np.concatenate(sequences),
        general=np.concatenate(generals),
        rotation=np.concatenate(rotations),
        translation=np.concatenate(translations),
    )


# --------------------------------------------------------------------------
# once per pair of anchors, and once per step: the sieve
# --------------------------------------------------------------------------

#: The margin of the step's own sieve, whose positions are the step's: rounding alone, m.
SIEVE_MARGIN_M = 1e-6


def _walk(
    xp: Any,
    held: dict[str, Any],
    tree: dict[str, Any],
    width: int,
    job: Any,
    image: Any,
    near: Any,
    source: Any,
    margin: float,
) -> tuple[Any, Any]:
    """The (job, image) pairs whose unfolded path can cross its facets, sorted by job then image.

    A sieve, not a validation: nothing it keeps is a path until
    :func:`_validate` says so, and it drops nothing that function would keep.

    The unfolded path is walked back from ``near``, the listener of each
    pair, towards the image. Its crossing of a facet's plane, ``x0``, is
    where the true crossing would be if neither end of the leg had moved.
    When both ends move by at most ``r``, the crossing moves by at most
    ``r (1 + L / |ha - hb|)`` and its place along the leg by ``r / |ha - hb|``,
    ``L`` the leg's length and ``ha``, ``hb`` its ends' heights over the plane.
    A pair is dropped when ``x0`` is farther than that from the rectangle of
    the facet's triangles, or from the leg itself.

    Two uses. With ``source`` ``None`` the images stand where the tree's
    anchor puts them, ``near`` is a listener's anchor and ``margin`` the
    size of the two cells: what is kept is every sequence any source and
    listener of those cells can hear. With ``source`` the position of each
    job's source, the images are the step's own, through their affine maps,
    and ``margin`` covers rounding alone: what is kept is nearly the step's
    paths, the occluders aside.
    """
    order = tree["order"]
    count = int(order.shape[0])
    current = image
    reach = xp.full(int(job.shape[0]), float(margin))
    kept_job = []
    kept_image = []
    for step in range(width + 1):
        if int(job.shape[0]) == 0:
            break
        bounce = order[image] - 1 - step
        through = bounce < 0
        kept_job.append(job[through])
        kept_image.append(image[through])
        active = xp.flatnonzero(~through)
        if int(active.shape[0]) == 0:
            break
        job, image, current = job[active], image[active], current[active]
        near, reach = near[active], reach[active]
        facet = tree["sequence"][image, bounce[active]]
        if source is None:
            far = tree["positions"][current]
        else:
            at = source[job]
            turn = tree["rotation"][current]
            shift = tree["translation"][current]
            far = xp.stack(
                [
                    turn[:, 3 * k] * at[:, 0]
                    + turn[:, 3 * k + 1] * at[:, 1]
                    + turn[:, 3 * k + 2] * at[:, 2]
                    + shift[:, k]
                    for k in range(3)
                ],
                axis=1,
            )
        normal = held["normals"][facet]
        ha = _dot(near, normal) - held["offsets"][facet]
        hb = _dot(far, normal) - held["offsets"][facet]
        span = xp.abs(ha - hb)
        bounded = span > 1e-9
        safe = xp.where(bounded, span, 1.0)
        t = ha / xp.where(bounded, ha - hb, 1.0)
        leg = far - near
        crossing = near + t[:, None] * leg
        along = reach / safe
        moved = reach * (1.0 + _norm(xp, leg) / safe)
        rect = held["rectangle"][facet]
        u = _dot(crossing, rect[:, 4:7]) - rect[:, 0]
        v = _dot(crossing, rect[:, 7:10]) - rect[:, 1]
        ok = ~bounded | (
            (t > -along)
            & (t < 1.0 + along)
            & (u >= -moved)
            & (u <= rect[:, 2] + moved)
            & (v >= -moved)
            & (v <= rect[:, 3] + moved)
        )
        # A crossing nothing bounds says nothing of the next leg either.
        landed = xp.where(bounded[:, None], crossing, far)
        moved = xp.where(bounded, xp.maximum(moved, margin), xp.inf)
        keep = xp.flatnonzero(ok)
        job, image = job[keep], image[keep]
        near, reach = landed[keep], moved[keep]
        current = tree["parent"][current[keep]]
    job = xp.concatenate(kept_job)
    image = xp.concatenate(kept_image)
    sort = xp.argsort(job * count + image)
    return job[sort], image[sort]


# --------------------------------------------------------------------------
# once per step: the tree's own tests, then the legs
# --------------------------------------------------------------------------


def _dot(a: Any, b: Any) -> Any:
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _norm(xp: Any, a: Any) -> Any:
    return xp.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2])


def _members(xp: Any, offsets: Any, members: Any, cells: Any) -> tuple[Any, Any]:
    """For each of ``cells``, every member of its row: the owner's rank and the member."""
    start = offsets[cells]
    counts = offsets[cells + 1] - start
    ends = xp.cumsum(counts)
    total = int(ends[-1]) if counts.shape[0] else 0
    if total == 0:
        empty = xp.zeros(0, dtype=xp.int64)
        return empty, empty
    flat = xp.arange(total, dtype=xp.int64)
    if xp is np:
        owner = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    else:
        # cupy repeats by one count only.
        owner = xp.searchsorted(ends, flat, side="right")
    rank = flat - (ends - counts)[owner]
    return owner, members[start[owner] + rank]


def _segment_triangles(
    xp: Any,
    origin: Any,
    direction: Any,
    v0: Any,
    e1: Any,
    e2: Any,
    low: Any,
    high: Any,
) -> Any:
    """Moller and Trumbore, one segment against one triangle per row.

    The arithmetic of :func:`reverberate.mirror.ism.ray_triangles`.
    """
    px = direction[:, 1] * e2[:, 2] - direction[:, 2] * e2[:, 1]
    py = direction[:, 2] * e2[:, 0] - direction[:, 0] * e2[:, 2]
    pz = direction[:, 0] * e2[:, 1] - direction[:, 1] * e2[:, 0]
    det = e1[:, 0] * px + e1[:, 1] * py + e1[:, 2] * pz
    parallel = xp.abs(det) < 1e-12
    inv = xp.where(parallel, 0.0, 1.0 / xp.where(parallel, 1.0, det))
    tx = origin[:, 0] - v0[:, 0]
    ty = origin[:, 1] - v0[:, 1]
    tz = origin[:, 2] - v0[:, 2]
    u = (tx * px + ty * py + tz * pz) * inv
    qx = ty * e1[:, 2] - tz * e1[:, 1]
    qy = tz * e1[:, 0] - tx * e1[:, 2]
    qz = tx * e1[:, 1] - ty * e1[:, 0]
    v = (direction[:, 0] * qx + direction[:, 1] * qy + direction[:, 2] * qz) * inv
    t = (e2[:, 0] * qx + e2[:, 1] * qy + e2[:, 2] * qz) * inv
    return (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= low) & (t <= high)


def _any_member_hit(
    xp: Any,
    held: dict[str, Any],
    which: str,
    cells: Any,
    origin: Any,
    direction: Any,
    low: Any,
    high: Any,
    block: int,
) -> Any:
    """Per row, whether its segment hits any triangle of its cell's row of ``which``."""
    rows = int(cells.shape[0])
    out = xp.zeros(rows, dtype=bool)
    if rows == 0:
        return out
    offsets = held["bucket_offsets" if which == "reflector" else "cell_offsets"]
    members = held["bucket_members" if which == "reflector" else "cell_members"]
    owner, triangle = _members(xp, offsets, members, cells)
    for first in range(0, int(owner.shape[0]), block):
        o = owner[first : first + block]
        tri = triangle[first : first + block]
        hit = _segment_triangles(
            xp,
            origin[o],
            direction[o],
            held[f"{which}_v0"][tri],
            held[f"{which}_e1"][tri],
            held[f"{which}_e2"][tri],
            low[o],
            high[o],
        )
        out[o[hit]] = True
    return out


def _inside_facet(xp: Any, held: dict[str, Any], point: Any, facet: Any, block: int) -> Any:
    """Whether each point of a facet's plane lies in one of the facet's triangles.

    A short probe through the point along the normal, against the triangles
    of the point's bucket: the twin's test, on the kernel's buckets.
    """
    normal = held["normals"][facet]
    frame = held["frame"][facet]
    shape = held["shape"][facet]
    pu = _dot(point, frame[:, 3:6])
    pv = _dot(point, frame[:, 6:9])
    iu = xp.clip(xp.floor((pu - frame[:, 0]) / frame[:, 2]).astype(xp.int64), 0, shape[:, 0] - 1)
    iv = xp.clip(xp.floor((pv - frame[:, 1]) / frame[:, 2]).astype(xp.int64), 0, shape[:, 1] - 1)
    bucket = held["base"][facet] + iu * shape[:, 1] + iv
    start = point - 1e-4 * normal
    stop = point + 1e-4 * normal
    rows = int(point.shape[0])
    return _any_member_hit(
        xp,
        held,
        "reflector",
        bucket,
        start,
        stop - start,
        xp.zeros(rows),
        xp.ones(rows),
        block,
    )


def _blocked(xp: Any, held: dict[str, Any], a: Any, b: Any, epsilon_m: float, block: int) -> Any:
    """Whether each segment ``a -> b``, shrunk by ``epsilon_m`` at both ends, is cut by an occluder.

    :func:`reverberate.mirror.ism._segment_hits` for every segment at once:
    all the segments step together through the cells they cross, each
    tested against the triangles of the cell it is in, and leave the walk
    when they are cut or arrive.
    """
    rows = int(a.shape[0])
    out = xp.zeros(rows, dtype=bool)
    if rows == 0:
        return out
    direction = b - a
    length = _norm(xp, direction)
    seg = xp.flatnonzero(length > 2.0 * epsilon_m)
    if int(seg.shape[0]) == 0:
        return out
    a = a[seg]
    direction = direction[seg]
    strict = epsilon_m / length[seg]
    origin = held["grid_origin"]
    shape = held["grid_shape"]
    cell_m = held["grid_cell"]
    cell = xp.clip(xp.floor((a - origin) / cell_m).astype(xp.int64), 0, shape - 1)
    last = xp.clip(xp.floor((a + direction - origin) / cell_m).astype(xp.int64), 0, shape - 1)
    step = (direction > 0).astype(xp.int64) - (direction < 0).astype(xp.int64)
    moving = step != 0
    safe = xp.where(moving, direction, 1.0)
    boundary = origin + (cell + (step > 0)) * cell_m
    t_max = xp.where(moving, (boundary - a) / safe, xp.inf)
    t_delta = xp.where(moving, cell_m / xp.abs(safe), xp.inf)
    for _ in range(int(shape.sum()) * 3 + 1):
        flat = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
        hit = _any_member_hit(xp, held, "occluder", flat, a, direction, strict, 1.0 - strict, block)
        out[seg[hit]] = True
        done = hit | xp.all(cell == last, axis=1)
        here = xp.arange(int(cell.shape[0]))
        axis = xp.argmin(t_max, axis=1)
        done |= t_max[here, axis] > 1.0
        cell[here, axis] += step[here, axis]
        entered = cell[here, axis]
        done |= (entered < 0) | (entered >= shape[axis])
        t_max[here, axis] += t_delta[here, axis]
        keep = xp.flatnonzero(~done)
        if int(keep.shape[0]) == 0:
            break
        seg, a, direction, strict = seg[keep], a[keep], direction[keep], strict[keep]
        cell, last, step = cell[keep], last[keep], step[keep]
        t_max, t_delta = t_max[keep], t_delta[keep]
    return out


def _validate(
    xp: Any,
    ms: MovingScene,
    source: Any,
    listener: Any,
    sequence: Any,
    order: Any,
    general: Any,
    *,
    region: tuple[np.ndarray, np.ndarray] | None,
    epsilon_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs that are paths: their indices, their images and their first reflection points.

    A pair is one source position, one listener position and one facet
    sequence. First the tree's own tests where the source stands, level by
    level as the tree is grown: the image below must lie on the air side of
    the facet, the facet inside the beam the facet before it opens, the
    image within the window of the region. Then the twin's walk back from
    the listener (:func:`reverberate.mirror.ism.paths_for`). Returned on the
    host; a pair of order zero has the listener as its first point.
    """
    held = ms.on(xp)
    block = ms.moving.tests_per_block
    pairs = int(source.shape[0])
    width = int(sequence.shape[1])
    reach = ms.ism.sound_speed_m_s * ms.ism.window_s
    chain = xp.zeros((pairs, width + 1, 3))
    chain[:, 0] = source
    alive = xp.ones(pairs, dtype=bool)
    position = source
    if region is not None:
        lo = xp.asarray(np.asarray(region[0], dtype=float))
        hi = xp.asarray(np.asarray(region[1], dtype=float))
    for level in range(width):
        use = order > level
        facet = xp.where(use, sequence[:, level], 0)
        normal = held["normals"][facet]
        height = _dot(position, normal) - held["offsets"][facet]
        ok = (height > 0.0) | held["both"][facet]
        if level >= 1:
            before = xp.where(use, sequence[:, level - 1], 0)
            axis = held["centres"][before] - position
            distance = _norm(xp, axis)
            axis = axis / xp.maximum(distance, 1e-12)[:, None]
            radius = held["radii"][before]
            half = xp.where(
                distance <= radius,
                np.pi,
                xp.arcsin(xp.clip(radius / xp.maximum(distance, 1e-12), 0.0, 1.0)),
            )
            to_facet = held["centres"][facet] - position
            far = _norm(xp, to_facet)
            cosine = _dot(to_facet, axis) / xp.maximum(far, 1e-12)
            angle = xp.arccos(xp.clip(cosine, -1.0, 1.0))
            subtended = xp.arcsin(xp.clip(held["radii"][facet] / xp.maximum(far, 1e-12), 0.0, 1.0))
            ok &= (angle <= half + subtended) | (level >= general)
        mirrored = position - 2.0 * height[:, None] * normal
        if region is not None:
            gap = xp.maximum(xp.maximum(lo[None, :] - mirrored, mirrored - hi[None, :]), 0.0)
            ok &= _norm(xp, gap) <= reach
        alive &= ok | ~use
        position = xp.where(use[:, None], mirrored, position)
        chain[:, level + 1] = position
    # The walk back, every facet first and the occluders after. A path is
    # one whose legs all cross their facets and are all clear: the twin tests
    # a leg's occluders as soon as it has crossed, the answer is the same in
    # any order, and few of the pairs that cross their first facet cross
    # them all, so the occluders, which cost, are asked last and of few.
    index = xp.flatnonzero(alive)
    count = int(index.shape[0])
    # ``points[:, k]`` is where leg k starts: the listener, then each hit.
    points = xp.zeros((count, width + 1, 3))
    points[:, 0] = listener[index]
    for step in range(width):
        if int(index.shape[0]) == 0:
            break
        bounce = order[index] - 1 - step
        active = xp.flatnonzero(bounce >= 0)
        if int(active.shape[0]) == 0:
            break
        pair = index[active]
        facet = sequence[pair, bounce[active]]
        far = chain[pair, order[pair] - step]
        start = points[active, step]
        normal = held["normals"][facet]
        ha = _dot(start, normal) - held["offsets"][facet]
        hb = _dot(far, normal) - held["offsets"][facet]
        denominator = ha - hb
        crosses = xp.abs(denominator) > 1e-12
        t = ha / xp.where(crosses, denominator, 1.0)
        ok = crosses & (t > 0.0) & (t < 1.0)
        hit = start + t[:, None] * (far - start)
        crossing = xp.flatnonzero(ok)
        inside = _inside_facet(xp, held, hit[crossing], facet[crossing], block)
        ok[crossing[~inside]] = False
        points[active[ok], step + 1] = hit[ok]
        keep = xp.ones(int(index.shape[0]), dtype=bool)
        keep[active[~ok]] = False
        index, points = index[keep], points[keep]
    legs = order[index]
    for step in range(width + 1):
        # Leg ``step`` runs from its start to the next hit, or to the source.
        active = xp.flatnonzero(legs >= step)
        if int(active.shape[0]) == 0:
            break
        last = legs[active] == step
        end = xp.where(last[:, None], chain[index[active], 0], points[active, min(step + 1, width)])
        cut = _blocked(xp, held, points[active, step], end, epsilon_m, block)
        keep = xp.ones(int(index.shape[0]), dtype=bool)
        keep[active[cut]] = False
        index, points, legs = index[keep], points[keep], legs[keep]
    here = xp.arange(int(index.shape[0]))
    first = points[here, legs]
    image = chain[index, order[index]]
    return (
        to_numpy(index).astype(np.int64),
        to_numpy(image),
        to_numpy(first),
    )


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

_IDS: dict[bytes, int] = {}


def path_id(
    kind: int,
    facets: np.ndarray | tuple[int, ...] = (),
    edges: np.ndarray | tuple[int, ...] | None = None,
    *,
    rank: int | None = None,
) -> int:
    """A path's identity, the same at every step it exists (``scene-pack.md``).

    The first eight bytes, little endian, of the SHA-256 of: the kind as one
    byte; the facets in bounce order as little endian ``int32``; and, for a
    diffracted path, ``-1`` then the edges it bends round in order from the
    source. A diffracted path whose corners are not all on edges has no such
    name: it gives ``rank``, its rank by delay among the step's such paths,
    and is named by ``-1, -1, rank`` in place of the edges.
    """
    words = [int(f) for f in facets]
    if rank is not None:
        words += [-1, -1, int(rank)]
    elif edges is not None:
        words += [-1, *(int(e) for e in edges)]
    key = bytes([int(kind)]) + np.asarray(words, dtype="<i4").tobytes()
    found = _IDS.get(key)
    if found is None:
        found = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        _IDS[key] = found
    return found


@dataclass(frozen=True)
class EarlyTable:
    """One source's arrivals at every step: the pack's ``early`` group, in double precision.

    The pack keeps the directions and the gains in single precision and the
    delay alone; the table also keeps what the present renderer reads, the
    length, the facet sequence and ``rank``, the row's place in the present
    pipeline's own order, so that :meth:`paths` hands that renderer what the
    pipeline would have. :meth:`pack` is what the pack stores.
    """

    #: ``[step + 1]``: step ``k`` owns rows ``offsets[k]`` to ``offsets[k + 1]``.
    offsets: np.ndarray
    path_id: np.ndarray
    delay_s: np.ndarray
    #: Unit vectors from the listener towards where the sound comes from.
    arrival: np.ndarray
    #: Unit vectors from the source along the path's first leg.
    departure: np.ndarray
    #: ``[row, band]`` pressure gain: omnidirectional, without air.
    gain: np.ndarray
    order: np.ndarray
    kind: np.ndarray
    length_m: np.ndarray
    #: ``[row, width]`` the facets in bounce order, ``-1`` padded.
    sequence: np.ndarray
    rank: np.ndarray
    source: np.ndarray
    listener: np.ndarray
    sound_speed_m_s: float
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.offsets.shape[0]) - 1

    def rows(self, step: int) -> slice:
        return slice(int(self.offsets[step]), int(self.offsets[step + 1]))

    def pack(self) -> dict[str, np.ndarray]:
        """The datasets of ``/sources/<id>/early``, in the pack's types."""
        return {
            "offsets": self.offsets.astype(np.int64),
            "path_id": self.path_id.astype(np.uint64),
            "delay_s": self.delay_s.astype(np.float64),
            "arrival": self.arrival.astype(np.float32),
            "departure": self.departure.astype(np.float32),
            "gain": self.gain.astype(np.float32),
            "order": self.order.astype(np.uint8),
            "kind": self.kind.astype(np.uint8),
        }

    def paths(self, step: int) -> tuple[Paths, Paths | None]:
        """The step's image paths and its diffracted onset, as the present pipeline holds them.

        Each in the pipeline's own order. ``points`` holds the source and the
        listener alone: the reflection points between them are not kept.
        """
        rows = np.arange(int(self.offsets[step]), int(self.offsets[step + 1]))
        onset = self.kind[rows] >= KIND_DIFFRACTED

        def of(chosen: np.ndarray) -> Paths:
            chosen = chosen[np.argsort(self.rank[chosen], kind="stable")]
            points = np.repeat(self.listener[step][None, None, :], chosen.size, axis=0)
            points = np.repeat(points, 2, axis=1)
            points[:, 0] = self.source[step]
            return Paths(
                receiver=self.listener[step].copy(),
                image=np.full(chosen.size, -1, dtype=np.int32),
                order=self.order[chosen].astype(np.int32),
                length_m=self.length_m[chosen],
                direction=self.arrival[chosen],
                gain=self.gain[chosen],
                points=points,
                sequence=self.sequence[chosen],
            )

        return of(rows[~onset]), (of(rows[onset]) if bool(onset.any()) else None)


def render_early(
    table: EarlyTable, step: int, settings: RenderSettings, xp: Any = np
) -> np.ndarray:
    """One step's arrivals through the present renderer: ``[channel, sample]``.

    :func:`reverberate.mirror.render.early_signals` of the image paths, plus
    that of the diffracted onset where there is one, as
    :func:`reverberate.mirror.render.render_point` adds them. No tail, no
    air, no signature.
    """
    images, onset = table.paths(step)
    signals = early_signals(images, settings, table.sound_speed_m_s, xp)
    if onset is not None:
        signals = signals + early_signals(onset, settings, table.sound_speed_m_s, xp)
    return to_numpy(signals)


@dataclass
class _Rows:
    """The rows of the distinct (source, listener) positions, before they are laid on steps."""

    job: list[np.ndarray] = field(default_factory=list)
    columns: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, job: np.ndarray, **columns: np.ndarray) -> None:
        self.job.append(np.asarray(job, dtype=np.int64))
        for name, values in columns.items():
            self.columns.setdefault(name, []).append(np.asarray(values))

    def stacked(self, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        parts = self.columns.get(name, [])
        if not parts:
            return np.zeros((0, *shape), dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)


def _anchor_keys(positions: np.ndarray, pitch_m: float) -> np.ndarray:
    return np.asarray(np.rint(positions / pitch_m), dtype=np.int64)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(
        vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12), dtype=float
    )


def trace_early(
    ms: MovingScene,
    source: np.ndarray,
    listener: np.ndarray,
    *,
    audible: np.ndarray | None = None,
    region: tuple[np.ndarray, np.ndarray] | None = None,
    onsets: Any = None,
    xp: Any = np,
) -> EarlyTable:
    """One source's early arrivals at every step of a trajectory.

    ``source`` and ``listener`` are ``[step, 3]``, the mouth and the head's
    centre at each step, scene frame. A step that is not ``audible`` has no
    rows and is not computed.

    ``region`` is the box whose distance prunes the image tree, as
    :func:`reverberate.mirror.ism.grow_tree` reads it: the present pipeline
    passes its lattice's box grown by 0.5 m, and a trace that must agree
    with a field of that pipeline passes the same. Without it the tree is
    whole.

    ``onsets`` is the scene's :class:`reverberate.mirror.moving_onset.OnsetField`;
    with it a step without a direct path gets its diffracted onset, as the
    present pipeline gives one to a point the source does not see. Without
    it such a step holds its reflections alone.

    ``xp`` is the array module the steps are validated on.
    """
    source = np.atleast_2d(np.asarray(source, dtype=float))
    listener = np.atleast_2d(np.asarray(listener, dtype=float))
    if source.shape != listener.shape or source.shape[1] != 3:
        raise ValueError(f"source {source.shape} and listener {listener.shape} must be [step, 3]")
    steps = source.shape[0]
    heard = np.ones(steps, dtype=bool) if audible is None else np.asarray(audible, dtype=bool)
    if heard.shape != (steps,):
        raise ValueError(f"audible {heard.shape} must be [{steps}]")
    width = ms.width
    bands = int(ms.images.materials.absorption.shape[1])
    c = ms.ism.sound_speed_m_s
    # The distinct positions: a pair at rest is one job however long it rests.
    both = np.concatenate([source, listener], axis=1)[heard]
    if both.shape[0]:
        jobs, job_of = np.unique(both, axis=0, return_inverse=True)
        job_of = np.asarray(job_of).reshape(-1)
    else:
        jobs, job_of = np.zeros((0, 6)), np.zeros(0, dtype=np.int64)
    job_source, job_listener = jobs[:, :3], jobs[:, 3:]
    record: dict[str, Any] = {
        "steps": int(steps),
        "audible": int(heard.sum()),
        "jobs": int(jobs.shape[0]),
        "anchors": 0,
        "anchor_pairs": 0,
        "sieved": 0,
        "pairs": 0,
    }
    rows = _Rows()
    _image_rows(ms, job_source, job_listener, region, xp, rows, record)
    direct = np.zeros(jobs.shape[0], dtype=bool)
    if rows.job:
        order_all = rows.stacked("order", (), np.int64)
        direct[np.concatenate(rows.job)[order_all == 0]] = True
    record["without_direct"] = int((~direct).sum())
    if onsets is not None and not bool(direct.all()):
        from reverberate.mirror.moving_onset import onset_rows

        record["diffraction"] = onset_rows(
            ms, onsets, job_source, job_listener, np.flatnonzero(~direct), xp, rows
        )
    # The jobs' rows, laid on the steps that asked for them.
    job = np.concatenate(rows.job) if rows.job else np.zeros(0, dtype=np.int64)
    by_job = np.argsort(job, kind="stable")
    counts = np.bincount(job, minlength=jobs.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)])
    step_job = np.full(steps, -1, dtype=np.int64)
    step_job[heard] = job_of
    step_count = np.where(step_job >= 0, counts[np.maximum(step_job, 0)], 0)
    offsets = np.concatenate([[0], np.cumsum(step_count)]).astype(np.int64)
    total = int(offsets[-1])
    step_of = np.repeat(np.arange(steps), step_count)
    within = np.arange(total) - offsets[step_of]
    take = by_job[starts[step_job[step_of]] + within] if total else np.zeros(0, dtype=np.int64)
    ids = rows.stacked("path_id", (), np.uint64)[take]
    # Within a step, by identity.
    sort = np.lexsort((ids, step_of))
    take, ids = take[sort], ids[sort]
    same = (ids[1:] == ids[:-1]) & (step_of[1:] == step_of[:-1])
    if bool(same.any()):
        raise ValueError(f"step {int(step_of[1:][same][0])} holds one path identity twice")
    length = rows.stacked("length_m", (), float)[take]
    return EarlyTable(
        offsets=offsets,
        path_id=ids,
        delay_s=length / c,
        arrival=rows.stacked("arrival", (3,), float)[take],
        departure=rows.stacked("departure", (3,), float)[take],
        gain=rows.stacked("gain", (bands,), float)[take],
        order=rows.stacked("order", (), np.int64)[take].astype(np.uint8),
        kind=rows.stacked("kind", (), np.int64)[take].astype(np.uint8),
        length_m=length,
        sequence=rows.stacked("sequence", (width,), np.int32)[take],
        rank=rows.stacked("rank", (), np.int64)[take].astype(np.int32),
        source=source,
        listener=listener,
        sound_speed_m_s=float(c),
        record=record,
    )


def _candidates_of(
    ms: MovingScene, key: tuple[int, ...], region: tuple[np.ndarray, np.ndarray] | None
) -> Candidates:
    pitch = ms.moving.source_pitch_m
    name = (
        key,
        pitch,
        None if region is None else np.asarray(region, dtype=float).tobytes(),
    )
    found = ms._candidates.get(name)
    if found is None:
        # Half the cell's diagonal: the farthest a source of this cell is from its anchor.
        slack = 0.5 * np.sqrt(3.0) * pitch * (1.0 + 1e-9)
        found = grow_candidates(ms, np.asarray(key, dtype=float) * pitch, slack, region)
        while len(ms._candidates) >= max(1, ms.moving.anchors_kept):
            ms._candidates.pop(next(iter(ms._candidates)))
        ms._candidates[name] = found
    return found


def _short_lists(
    ms: MovingScene,
    candidates: Candidates,
    name: tuple[Any, ...],
    keys: np.ndarray,
    xp: Any,
) -> list[np.ndarray]:
    """Per listener cell of ``keys``: the candidates a source and a listener of the two cells can
    hear. Kept from one trace to the next.
    """
    moving = ms.moving
    pitch = moving.listener_pitch_m
    margin = max(0.5 * np.sqrt(3.0) * pitch * (1.0 + 1e-9), candidates.slack_m)
    names = [(name, tuple(int(v) for v in key)) for key in keys]
    missing = [k for k, n in enumerate(names) if n not in ms._lists]
    count = candidates.count
    per = max(1, moving.pairs_per_block // max(1, count))
    held, tree = ms.on(xp), candidates.on(xp)
    found: dict[int, np.ndarray] = {}
    for first in range(0, len(missing), per):
        share = missing[first : first + per]
        anchors = xp.asarray(keys[share].astype(float) * pitch)
        job = xp.repeat(xp.arange(len(share), dtype=xp.int64), count)
        image = xp.tile(xp.arange(count, dtype=xp.int64), len(share))
        job, image = _walk(xp, held, tree, ms.width, job, image, anchors[job], None, margin)
        host_job, host_image = to_numpy(job), to_numpy(image)
        bounds = np.searchsorted(host_job, np.arange(len(share) + 1))
        for k, which in enumerate(share):
            found[which] = host_image[bounds[k] : bounds[k + 1]]
    out = [found[k] if k in found else ms._lists[names[k]] for k in range(len(names))]
    for k, n in enumerate(names):
        ms._lists.pop(n, None)
        ms._lists[n] = out[k]
    while len(ms._lists) > max(moving.lists_kept, len(names)):
        ms._lists.pop(next(iter(ms._lists)))
    return out


def _image_rows(
    ms: MovingScene,
    job_source: np.ndarray,
    job_listener: np.ndarray,
    region: tuple[np.ndarray, np.ndarray] | None,
    xp: Any,
    rows: _Rows,
    record: dict[str, Any],
) -> None:
    """The image paths of every job, block of pairs by block, into ``rows``."""
    jobs = job_source.shape[0]
    if jobs == 0:
        return
    moving = ms.moving
    held = ms.on(xp)
    source_key = _anchor_keys(job_source, moving.source_pitch_m)
    listener_key = _anchor_keys(job_listener, moving.listener_pitch_m)
    groups, group_of = np.unique(
        np.concatenate([source_key, listener_key], axis=1), axis=0, return_inverse=True
    )
    group_of = np.asarray(group_of).reshape(-1)
    members = np.argsort(group_of, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(group_of, minlength=groups.shape[0]))])
    record["anchor_pairs"] += int(groups.shape[0])

    # What the sieve kept, from every anchor, until there is enough to validate at once.
    waiting: list[tuple[np.ndarray, Any, Any, Any, Any, Any]] = []
    waiting_pairs = 0

    def block(candidates: Candidates, parts: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Sieve the jobs of ``parts`` against their short lists."""
        nonlocal waiting_pairs
        tree = candidates.on(xp)
        block_jobs = np.concatenate([share for share, _ in parts])
        starts = np.concatenate([[0], np.cumsum([share.size for share, _ in parts])])
        job = np.concatenate(
            [
                np.repeat(np.arange(starts[k], starts[k + 1]), short.size)
                for k, (_, short) in enumerate(parts)
            ]
        )
        image = np.concatenate([np.tile(short, share.size) for share, short in parts])
        record["sieved"] += int(job.size)
        at = xp.asarray(job_source[block_jobs])
        ears = xp.asarray(job_listener[block_jobs])
        job_x = xp.asarray(job.astype(np.int64))
        job_x, image_x = _walk(
            xp,
            held,
            tree,
            ms.width,
            job_x,
            xp.asarray(image.astype(np.int64)),
            ears[job_x],
            at,
            SIEVE_MARGIN_M,
        )
        waiting.append(
            (
                block_jobs[to_numpy(job_x)],
                at[job_x],
                ears[job_x],
                tree["sequence"][image_x],
                tree["order"][image_x],
                tree["general"][image_x],
            )
        )
        waiting_pairs += int(job_x.shape[0])
        if waiting_pairs >= moving.pairs_per_validation:
            validate()

    def validate() -> None:
        """Validate what waits, and lay the paths on ``rows``."""
        nonlocal waiting_pairs
        if not waiting:
            return
        pair_job = np.concatenate([w[0] for w in waiting])
        sequence_x = xp.concatenate([w[3] for w in waiting])
        order_x = xp.concatenate([w[4] for w in waiting])
        record["pairs"] += int(pair_job.size)
        index, image_at, first = _validate(
            xp,
            ms,
            xp.concatenate([w[1] for w in waiting]),
            xp.concatenate([w[2] for w in waiting]),
            sequence_x,
            order_x,
            xp.concatenate([w[5] for w in waiting]),
            region=region,
            epsilon_m=ms.ism.epsilon_m,
        )
        waiting.clear()
        waiting_pairs = 0
        if index.size == 0:
            return
        job = pair_job[index]
        receiver = job_listener[job]
        # The twin's own expressions, on the host: its lengths, directions and gains.
        lengths = np.linalg.norm(image_at - receiver, axis=1)
        arrival = (image_at - receiver) / np.maximum(lengths, 1e-12)[:, None]
        order = to_numpy(order_x)[index]
        sequence = to_numpy(sequence_x)[index].astype(np.int32)
        gain = _gains(ms.images, sequence, lengths)
        departure = _unit(first - job_source[job])
        # Pairs were laid job by job, candidates in the tree's order: so are these.
        new = np.concatenate([[True], job[1:] != job[:-1]])
        rank = np.arange(job.size) - np.maximum.accumulate(np.where(new, np.arange(job.size), 0))
        ids = np.asarray(
            [
                path_id(KIND_DIRECT if o == 0 else KIND_SPECULAR, q[:o])
                for o, q in zip(order, sequence, strict=True)
            ],
            dtype=np.uint64,
        )
        rows.add(
            job,
            path_id=ids,
            length_m=lengths,
            arrival=arrival,
            departure=departure,
            gain=gain,
            order=order,
            kind=np.where(order == 0, KIND_DIRECT, KIND_SPECULAR),
            sequence=sequence,
            rank=rank,
        )

    anchors, anchor_of = np.unique(groups[:, :3], axis=0, return_inverse=True)
    anchor_of = np.asarray(anchor_of).reshape(-1)
    record["anchors"] += int(anchors.shape[0])
    for a in range(anchors.shape[0]):
        key = tuple(int(v) for v in anchors[a])
        candidates = _candidates_of(ms, key, region)
        name = (
            key,
            moving.source_pitch_m,
            moving.listener_pitch_m,
            None if region is None else np.asarray(region, dtype=float).tobytes(),
        )
        mine = np.flatnonzero(anchor_of == a)
        lists = _short_lists(ms, candidates, name, groups[mine, 3:], xp)
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        held_pairs = 0
        for g, short in zip(mine, lists, strict=True):
            its = members[bounds[g] : bounds[g + 1]]
            per = max(1, moving.pairs_per_block // max(1, short.size))
            for first in range(0, its.size, per):
                share = its[first : first + per]
                if pending and held_pairs + share.size * short.size > moving.pairs_per_block:
                    block(candidates, pending)
                    pending, held_pairs = [], 0
                pending.append((share, short))
                held_pairs += share.size * short.size
        if pending:
            block(candidates, pending)
    validate()
