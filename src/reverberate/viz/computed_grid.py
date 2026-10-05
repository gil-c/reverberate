"""The wave solver's staircase, drawable: walls, slices, and what two grids disagree on.

A grid record (:class:`reverberate.trace.computed.GridRecord`) says, node by
node, what the low band solver read of a dwelling: the air a source reaches,
the boundary nodes, the links each may not read, its material. Sixty million
nodes do not go to a browser, so three views are derived here, each exact in
what it shows and each saying what it leaves out:

- **the walls** (:func:`walls`): one square per link a reached boundary node
  may not read, half a step from the node, which is where the solver's wall
  is; coplanar squares of one material are merged into rectangles
  (:func:`merge_faces`), which changes how many are drawn and not where. A
  box bounds what is sent;
- **a slice** (:func:`slice_classes`): one layer of nodes, whole, a byte a
  node: not reached, air, rigid boundary, or lossy boundary with its
  material. Nothing is resampled;
- **two grids** (:func:`compare`): the coarser staircase read at the finer
  one's nodes. Where the air differs, how far a wall moved along its normal,
  which regions open or seal, and (:func:`opening_changes`) which openings
  change width.

A wall in a dwelling's model is often a sheet without thickness: both sides
are air and no node is solid. So a wall is read off the **links** and never
off the nodes alone: an opening is a run of open links, a displaced wall is
a cut link that moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.trace.computed import DIRECTIONS, GridRecord

__all__ = [
    "AIR",
    "FIRST_MATERIAL",
    "NOT_REACHED",
    "RIGID",
    "Comparison",
    "absorbing_layer",
    "compare",
    "diff_slice",
    "diff_surfaces",
    "facts",
    "layer_of",
    "merge_faces",
    "opening_changes",
    "openings",
    "quads_of",
    "slice_classes",
    "walls",
]

#: The classes of a slice's bytes; a lossy boundary node is ``FIRST_MATERIAL + material``.
NOT_REACHED, AIR, RIGID, FIRST_MATERIAL = 0, 1, 2, 3

#: The classes of a difference slice.
BOTH_CLOSED, BOTH_AIR, ONLY_A, ONLY_B, OUTSIDE = 0, 1, 2, 3, 4

#: A wall of the other grid is looked for this far along the normal, m.
WALL_SEARCH_M = 0.25
#: An opening is at most this wide and this deep, m: a doorway, a gap; not a corridor.
OPENING_WIDTH_M = 1.3
OPENING_DEPTH_M = 0.6
#: Two openings are one when their centres are this near, m.
OPENING_MATCH_M = 0.3


def layer_of(grid: GridRecord, y_m: float) -> int:
    """The layer of nodes nearest a height, kept inside the grid."""
    index = int(np.rint((float(y_m) - float(grid.origin_m[1])) / grid.step_m))
    return int(np.clip(index, 0, grid.shape[1] - 1))


def absorbing_layer(grid: GridRecord) -> tuple[np.ndarray, np.ndarray]:
    """The box of the absorbing nodes: the layer one node inside the grid's own box."""
    low = grid.origin_m + grid.step_m
    high = grid.origin_m + grid.step_m * (np.asarray(grid.shape) - 2)
    return np.asarray(low), np.asarray(high)


def facts(grid: GridRecord) -> dict[str, Any]:
    """What a page says of a grid: its place, its step, its counts."""
    low, high = absorbing_layer(grid)
    rigid = int(np.count_nonzero(grid.material < 0))
    return {
        "key": grid.key,
        "shape": list(grid.shape),
        "origin_m": [float(v) for v in grid.origin_m],
        "step_m": grid.step_m,
        "sample_rate_hz": 1.0 / grid.time_step_s if grid.time_step_s > 0 else None,
        "engine_axis_order": list(grid.order),
        "nodes": grid.nodes,
        "reached": int(np.count_nonzero(grid.reached)),
        "air_m3": float(np.count_nonzero(grid.reached)) * grid.step_m**3,
        "boundary_nodes": int(grid.boundary.size),
        "rigid_boundary_nodes": rigid,
        "lossy_boundary_nodes": int(grid.boundary.size) - rigid,
        "cut_links": int(_cut_count(grid.adjacency)),
        "labels": list(grid.labels),
        "seeds": grid.seeds,
        "absorbing_layer_m": [[float(v) for v in low], [float(v) for v in high]],
    }


def _cut_count(adjacency: np.ndarray) -> int:
    return int(sum(np.count_nonzero((adjacency >> d) & 1 == 0) for d in range(6)))


# --------------------------------------------------------------------------
# a slice
# --------------------------------------------------------------------------


def _layer_boundary(grid: GridRecord, layer: int) -> np.ndarray:
    """Rows of the boundary arrays that lie on a layer of constant ``y``."""
    nx, ny, nz = grid.shape
    rows = [np.zeros(0, dtype=np.int64)]
    # A layer is one run of flat indices per ``x``; the boundary is ascending.
    starts = (np.arange(nx, dtype=np.int64) * ny + layer) * nz
    first = np.searchsorted(grid.boundary, starts)
    last = np.searchsorted(grid.boundary, starts + nz)
    for a, b in zip(first, last, strict=True):
        if b > a:
            rows.append(np.arange(a, b, dtype=np.int64))
    return np.concatenate(rows)


def slice_classes(grid: GridRecord, y_m: float) -> tuple[np.ndarray, int]:
    """One layer of nodes as classes, ``[nz, nx]`` uint8 (rows ``z``, columns ``x``), and its index.

    ``NOT_REACHED``, ``AIR``, ``RIGID`` or ``FIRST_MATERIAL + material``: the
    layer itself, no node left out and none resampled.
    """
    layer = layer_of(grid, y_m)
    nx, ny, nz = grid.shape
    classes = np.where(grid.reached[:, layer, :], AIR, NOT_REACHED).astype(np.uint8)
    rows = _layer_boundary(grid, layer)
    flat = grid.boundary[rows]
    material = grid.material[rows].astype(np.int16)
    classes[flat // (ny * nz), flat % nz] = np.where(
        material < 0, RIGID, np.minimum(FIRST_MATERIAL + material, 255)
    ).astype(np.uint8)
    return np.ascontiguousarray(classes.T), layer


def _open_links(grid: GridRecord, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Whether each horizontal link of a layer is open: along ``x``, then along ``z``.

    ``[nx - 1, nz]`` and ``[nx, nz - 1]``.

    A link is open when both its nodes are reached and neither is told not
    to read the other.
    """
    nx, ny, nz = grid.shape
    reached = grid.reached[:, layer, :]
    bits = np.full((nx, nz), 0x3F, dtype=np.uint8)
    rows = _layer_boundary(grid, layer)
    flat = grid.boundary[rows]
    bits[flat // (ny * nz), flat % nz] = grid.adjacency[rows]
    along_x = reached[:-1] & reached[1:] & ((bits[:-1] & 1) > 0) & ((bits[1:] & 2) > 0)
    along_z = (
        reached[:, :-1] & reached[:, 1:] & ((bits[:, :-1] & 16) > 0) & ((bits[:, 1:] & 32) > 0)
    )
    return along_x, along_z


# --------------------------------------------------------------------------
# faces
# --------------------------------------------------------------------------


def merge_faces(
    w: np.ndarray, u: np.ndarray, v: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Unit squares at whole ``(w, u, v)`` as the fewest rectangles of one ``w`` and one value.

    Runs along ``v`` first, then equal runs of consecutive ``u``. Returns
    ``(w, u0, u1, v0, v1, value)``, the bounds inclusive. The rectangles
    cover exactly the squares given: their areas sum to the squares' count.
    """
    w, u, v, value = (np.asarray(a, dtype=np.int64) for a in (w, u, v, value))
    if w.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty, empty, empty
    order = np.lexsort((v, u, value, w))
    w, u, v, value = w[order], u[order], v[order], value[order]
    new = np.ones(w.size, dtype=bool)
    new[1:] = (
        (w[1:] != w[:-1]) | (value[1:] != value[:-1]) | (u[1:] != u[:-1]) | (v[1:] != v[:-1] + 1)
    )
    start = np.flatnonzero(new)
    end = np.append(start[1:], w.size) - 1
    rw, rm, ru, v0, v1 = w[start], value[start], u[start], v[start], v[end]
    order = np.lexsort((ru, v1, v0, rm, rw))
    rw, rm, ru, v0, v1 = rw[order], rm[order], ru[order], v0[order], v1[order]
    new = np.ones(rw.size, dtype=bool)
    new[1:] = (
        (rw[1:] != rw[:-1])
        | (rm[1:] != rm[:-1])
        | (v0[1:] != v0[:-1])
        | (v1[1:] != v1[:-1])
        | (ru[1:] != ru[:-1] + 1)
    )
    start = np.flatnonzero(new)
    end = np.append(start[1:], rw.size) - 1
    return rw[start], ru[start], ru[end], v0[start], v1[start], rm[start]


def quads_of(
    grid: GridRecord, subs: np.ndarray, direction: int, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The squares half a step from nodes ``subs`` along ``direction``, merged: corners and values.

    ``[quad, 4, 3]`` float32 in metres and ``[quad]`` int16.
    """
    axis, sign = direction // 2, 1.0 - 2.0 * (direction % 2)
    others = [a for a in range(3) if a != axis]
    w, u0, u1, v0, v1, kept = merge_faces(
        subs[:, axis], subs[:, others[0]], subs[:, others[1]], value
    )
    corners = np.zeros((w.size, 4, 3))
    corners[:, :, axis] = (w + 0.5 * sign)[:, None]
    low_u, high_u, low_v, high_v = u0 - 0.5, u1 + 0.5, v0 - 0.5, v1 + 0.5
    corners[:, :, others[0]] = np.stack([low_u, high_u, high_u, low_u], axis=1)
    corners[:, :, others[1]] = np.stack([low_v, low_v, high_v, high_v], axis=1)
    corners = grid.origin_m[None, None, :] + grid.step_m * corners
    return corners.astype(np.float32), kept.astype(np.int16)


def _box_subs(grid: GridRecord, low: np.ndarray | None, high: np.ndarray | None) -> tuple[Any, Any]:
    shape = np.asarray(grid.shape)
    first = np.zeros(3, dtype=np.int64)
    last = shape - 1
    if low is not None:
        first = np.clip(np.ceil((np.asarray(low) - grid.origin_m) / grid.step_m), 0, shape - 1)
    if high is not None:
        last = np.clip(np.floor((np.asarray(high) - grid.origin_m) / grid.step_m), 0, shape - 1)
    return first.astype(np.int64), last.astype(np.int64)


def walls(
    grid: GridRecord, low: np.ndarray | None = None, high: np.ndarray | None = None
) -> dict[str, Any]:
    """The solver's walls about the nodes inside a box: merged squares and their materials.

    ``corners`` ``[quad, 4, 3]`` float32, ``material`` ``[quad]`` int16
    (``-1`` rigid), ``faces`` the cut links drawn and ``quads`` the
    rectangles they merged into. Without a box, the whole grid.
    """
    first, last = _box_subs(grid, low, high)
    subs = grid.subs(grid.boundary)
    inside = np.all((subs >= first[None, :]) & (subs <= last[None, :]), axis=1)
    subs, adjacency, material = subs[inside], grid.adjacency[inside], grid.material[inside]
    corners, values, faces = [], [], 0
    for direction in range(6):
        cut = (adjacency >> direction) & 1 == 0
        faces += int(cut.sum())
        c, m = quads_of(grid, subs[cut], direction, material[cut])
        corners.append(c)
        values.append(m)
    held = np.concatenate(corners)
    return {
        "corners": held,
        "material": np.concatenate(values),
        "faces": faces,
        "quads": int(held.shape[0]),
        "box_m": [
            [float(v) for v in grid.origin_m + grid.step_m * first],
            [float(v) for v in grid.origin_m + grid.step_m * last],
        ],
    }


# --------------------------------------------------------------------------
# two grids
# --------------------------------------------------------------------------


def _read_at(a: GridRecord, b: GridRecord) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per axis: the node of ``b`` nearest each node of ``a``, and whether ``b`` has one there.

    A node of ``a`` counts only where both grids are past their halo and
    their absorbing layer (two nodes in), so the boxes' own edges are not
    read as a difference.
    """
    index, valid = [], []
    for axis in range(3):
        mine = np.arange(a.shape[axis])
        other = np.rint((a.origin_m[axis] + a.step_m * mine - b.origin_m[axis]) / b.step_m).astype(
            np.int64
        )
        valid.append(
            (mine >= 2) & (mine <= a.shape[axis] - 3) & (other >= 2) & (other <= b.shape[axis] - 3)
        )
        index.append(np.clip(other, 0, b.shape[axis] - 1))
    return index, valid


def _runs(subs: np.ndarray, shape: tuple[int, int, int], axis: int) -> np.ndarray:
    """Per node of a sparse set, the length of the run of the set through it along ``axis``."""
    others = [a for a in range(3) if a != axis]
    line = subs[:, others[0]] * shape[others[1]] + subs[:, others[1]]
    order = np.lexsort((subs[:, axis], line))
    along, held = subs[order, axis], line[order]
    new = np.ones(order.size, dtype=bool)
    new[1:] = (held[1:] != held[:-1]) | (along[1:] != along[:-1] + 1)
    run = np.cumsum(new) - 1
    lengths = np.bincount(run)
    out = np.zeros(order.size, dtype=np.int64)
    out[order] = lengths[run]
    return out


def _regions(
    mask: np.ndarray, grid: GridRecord, thick_nodes: int, most: int = 12
) -> list[dict[str, Any]]:
    """The parts of a difference thicker than a moved wall, largest first.

    A node is in such a part when the difference runs for more than
    ``thick_nodes`` through it along every axis; the parts are the connected
    sets of those nodes, and each one's volume is that of its own nodes.
    """
    from scipy import ndimage

    subs = np.argwhere(mask)
    if subs.shape[0] == 0:
        return []
    thinnest = np.minimum.reduce([_runs(subs, grid.shape, axis) for axis in range(3)])
    thick = subs[thinnest > thick_nodes]
    if thick.shape[0] == 0:
        return []
    low, high = thick.min(axis=0), thick.max(axis=0)
    box = np.zeros(tuple(high - low + 1), dtype=bool)
    box[tuple((thick - low[None, :]).T)] = True
    labels, count = ndimage.label(box)
    found = []
    volumes = np.bincount(labels.reshape(-1))[1:]
    centres = ndimage.center_of_mass(box, labels, np.arange(1, count + 1))
    for row in np.argsort(volumes)[::-1][:most]:
        centre = grid.origin_m + grid.step_m * (np.asarray(centres[row]) + low)
        found.append(
            {
                "nodes": int(volumes[row]),
                "volume_m3": float(volumes[row]) * grid.step_m**3,
                "centre_m": [round(float(v), 3) for v in centre],
            }
        )
    return found


def _cut_faces(grid: GridRecord, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """The cut links along one axis as link subscripts: ``[n, 3]``, the link from node to node + 1.

    A link is cut when either of its nodes is told not to read the other;
    each is returned once.
    """
    subs = grid.subs(grid.boundary)
    forward = (grid.adjacency >> (2 * axis)) & 1 == 0
    backward = (grid.adjacency >> (2 * axis + 1)) & 1 == 0
    back = subs[backward].copy()
    back[:, axis] -= 1
    links = np.concatenate([subs[forward], back])
    links = links[(links[:, axis] >= 0) & (links[:, axis] < grid.shape[axis] - 1)]
    shape = grid.shape
    flat = np.unique((links[:, 0] * shape[1] + links[:, 1]) * shape[2] + links[:, 2])
    return np.stack(np.unravel_index(flat, shape), axis=1), flat


def _displacement(mine: GridRecord, other: GridRecord, axis: int) -> dict[str, Any]:
    """How far along ``axis`` each wall square of ``mine`` is from the nearest of ``other``.

    The square's centre is read on the other grid's links through the same
    place; the nearest cut link within :data:`WALL_SEARCH_M` gives the
    distance, and a square with none is counted apart.
    """
    links, _ = _cut_faces(mine, axis)
    _, theirs = _cut_faces(other, axis)
    if links.shape[0] == 0:
        return {"faces": 0, "matched": 0, "distance_m": np.zeros(0), "where": np.zeros((0, 3))}
    held = np.zeros(other.nodes, dtype=bool)
    held[theirs] = True
    held = held.reshape(other.shape)
    centre = mine.origin_m[None, :] + mine.step_m * links.astype(float)
    centre[:, axis] += 0.5 * mine.step_m
    there = (centre - other.origin_m[None, :]) / other.step_m
    there[:, axis] -= 0.5
    base = np.rint(there).astype(np.int64)
    others = [a for a in range(3) if a != axis]
    inside = np.ones(links.shape[0], dtype=bool)
    for a in others:
        inside &= (base[:, a] >= 2) & (base[:, a] <= other.shape[a] - 3)
    inside &= (base[:, axis] >= 2) & (base[:, axis] <= other.shape[axis] - 4)
    reach = int(np.ceil(WALL_SEARCH_M / other.step_m))
    best = np.full(links.shape[0], np.inf)
    for shift in range(-reach, reach + 1):
        index = base.copy()
        index[:, axis] += shift
        ok = inside & (index[:, axis] >= 0) & (index[:, axis] < other.shape[axis] - 1)
        hit = np.zeros(links.shape[0], dtype=bool)
        hit[ok] = held[index[ok, 0], index[ok, 1], index[ok, 2]]
        away = np.abs((index[:, axis] - there[:, axis]) * other.step_m)
        best = np.where(hit & (away < best), away, best)
    found = np.isfinite(best) & inside
    return {
        "faces": int(inside.sum()),
        "matched": int(found.sum()),
        "distance_m": best[found],
        "where": centre[found],
    }


@dataclass(frozen=True)
class Comparison:
    """Two grids of one dwelling, the second read at the first's nodes."""

    a: GridRecord
    b: GridRecord
    #: ``[nx, ny, nz]`` of ``a``: where both grids are read, and where each alone has air.
    valid: np.ndarray
    only_a: np.ndarray
    only_b: np.ndarray
    numbers: dict[str, Any]


def compare(a: GridRecord, b: GridRecord) -> Comparison:
    """What two staircases of one dwelling disagree on.

    **The method.** ``b``'s air is read at ``a``'s node centres, each from
    the node of ``b`` nearest it, so the lattice of the comparison is
    ``a``'s and its resolution ``a``'s step: give the finer grid as ``a``.
    A volume that differs is counted in ``a``'s cells. A wall's
    displacement is measured on the links, along the wall's normal, from
    each grid to the other; a wall of one grid with none of the other within
    :data:`WALL_SEARCH_M` is counted apart and not as a displacement.
    """
    index, valid_axes = _read_at(a, b)
    b_air = b.reached[np.ix_(index[0], index[1], index[2])]
    valid = (
        valid_axes[0][:, None, None] & valid_axes[1][None, :, None] & valid_axes[2][None, None, :]
    )
    only_a = a.reached & ~b_air & valid
    only_b = ~a.reached & b_air & valid
    cell = a.step_m**3
    thick = int(np.ceil(2.0 * max(a.step_m, b.step_m) / a.step_m))
    walls_moved: dict[str, Any] = {}
    worst, worst_at, every = 0.0, None, []
    for name, mine, other in (("a_to_b", a, b), ("b_to_a", b, a)):
        faces = matched = 0
        for axis in range(3):
            found = _displacement(mine, other, axis)
            faces += found["faces"]
            matched += found["matched"]
            every.append(found["distance_m"])
            if found["distance_m"].size and float(found["distance_m"].max()) > worst:
                at = int(np.argmax(found["distance_m"]))
                worst, worst_at = float(found["distance_m"][at]), found["where"][at]
        walls_moved[name] = {
            "faces": faces,
            "with_a_counterpart": matched,
            "without_m2": float(faces - matched) * mine.step_m**2,
        }
    distances = np.concatenate(every) if every else np.zeros(0)
    numbers: dict[str, Any] = {
        "method": (
            "B's air read at A's node centres, each from B's nearest node; "
            "walls compared link by link along their normal"
        ),
        "resolution_m": a.step_m,
        "a": {"key": a.key, "step_m": a.step_m, "nodes": a.nodes},
        "b": {"key": b.key, "step_m": b.step_m, "nodes": b.nodes},
        "air_a_m3": float(np.count_nonzero(a.reached & valid)) * cell,
        "air_b_m3": float(np.count_nonzero(b_air & valid)) * cell,
        "air_only_in_a_m3": float(np.count_nonzero(only_a)) * cell,
        "air_only_in_b_m3": float(np.count_nonzero(only_b)) * cell,
        "compared_m3": float(np.count_nonzero(valid)) * cell,
        "walls": walls_moved,
        "wall_search_m": WALL_SEARCH_M,
        "largest_wall_displacement_m": worst,
        "largest_wall_displacement_at_m": None
        if worst_at is None
        else [round(float(v), 3) for v in worst_at],
        "median_wall_displacement_m": float(np.median(distances)) if distances.size else 0.0,
        "regions_thicker_than_m": thick * a.step_m,
        "regions_air_only_in_a": _regions(only_a, a, thick),
        "regions_air_only_in_b": _regions(only_b, a, thick),
    }
    numbers["differs_m3"] = numbers["air_only_in_a_m3"] + numbers["air_only_in_b_m3"]
    return Comparison(a=a, b=b, valid=valid, only_a=only_a, only_b=only_b, numbers=numbers)


def diff_slice(held: Comparison, y_m: float) -> tuple[np.ndarray, int]:
    """One layer of the comparison, ``[nz, nx]`` of ``a``: closed or air in both, or air in one."""
    layer = layer_of(held.a, y_m)
    classes = np.where(held.a.reached[:, layer, :], BOTH_AIR, BOTH_CLOSED).astype(np.uint8)
    classes[held.only_a[:, layer, :]] = ONLY_A
    classes[held.only_b[:, layer, :]] = ONLY_B
    classes[~held.valid[:, layer, :]] = OUTSIDE
    return np.ascontiguousarray(classes.T), layer


def diff_surfaces(
    held: Comparison, low: np.ndarray | None = None, high: np.ndarray | None = None
) -> dict[str, Any]:
    """The skin of the two differing volumes inside a box, merged: value 0 only A, 1 only B."""
    first, last = _box_subs(held.a, low, high)
    window = tuple(slice(int(f), int(g) + 1) for f, g in zip(first, last, strict=True))
    corners, values = [], []
    for value, mask in enumerate((held.only_a, held.only_b)):
        part = mask[window]
        subs = np.argwhere(part)
        if subs.shape[0] == 0:
            continue
        for direction, step in enumerate(DIRECTIONS):
            beside = subs + np.asarray(step)[None, :]
            inside = np.all((beside >= 0) & (beside < np.asarray(part.shape)[None, :]), axis=1)
            bare = np.ones(subs.shape[0], dtype=bool)
            bare[inside] = ~part[tuple(beside[inside].T)]
            c, m = quads_of(
                held.a, subs[bare] + first[None, :], direction, np.full(int(bare.sum()), value)
            )
            corners.append(c)
            values.append(m)
    if not corners:
        return {"corners": np.zeros((0, 4, 3), np.float32), "value": np.zeros(0, np.int16)}
    return {"corners": np.concatenate(corners), "value": np.concatenate(values)}


# --------------------------------------------------------------------------
# openings
# --------------------------------------------------------------------------


def _gaps(open_links: np.ndarray, step_m: float) -> list[dict[str, Any]]:
    """Openings among links ``[line, along]``: short runs of open links, grouped over lines."""
    from scipy import ndimage

    padded = np.zeros((open_links.shape[0], open_links.shape[1] + 2), dtype=bool)
    padded[:, 1:-1] = open_links
    edge = np.diff(padded.astype(np.int8), axis=1)
    line, start = np.nonzero(edge == 1)
    _, stop = np.nonzero(edge == -1)
    length = stop - start
    # A run that touches the grid's edge is not bounded on that side.
    bounded = (start > 0) & (stop < open_links.shape[1])
    short = bounded & (length * step_m <= OPENING_WIDTH_M)
    mask = np.zeros(open_links.shape, dtype=np.int32)
    for row, first, count in zip(line[short], start[short], length[short], strict=True):
        mask[row, first : first + count] = count
    labels, count = ndimage.label(mask > 0)
    found = []
    for piece in ndimage.find_objects(labels):
        if piece is None:
            continue
        rows, columns = piece
        depth = (rows.stop - rows.start) * step_m
        if depth > OPENING_DEPTH_M:
            continue
        widths = mask[piece][mask[piece] > 0]
        found.append(
            {
                "line": 0.5 * (rows.start + rows.stop - 1),
                "along": 0.5 * (columns.start + columns.stop - 1),
                "width_nodes": int(widths.min()),
                "depth_m": depth,
            }
        )
    return found


def openings(grid: GridRecord, y_m: float) -> list[dict[str, Any]]:
    """The openings of one layer: runs of open links no wider than a doorway, through a wall.

    Per opening: ``across``, the axis the sound crosses it along (``x`` or
    ``z``); its centre; its width in links and in metres. A derived view: a
    run of at most :data:`OPENING_WIDTH_M` of open links bounded by cut ones,
    no deeper than :data:`OPENING_DEPTH_M`.
    """
    layer = layer_of(grid, y_m)
    along_x, along_z = _open_links(grid, layer)
    found = []
    for across, links in (("x", along_x), ("z", along_z.T)):
        for gap in _gaps(links, grid.step_m):
            if across == "x":
                x = grid.origin_m[0] + grid.step_m * (gap["line"] + 0.5)
                z = grid.origin_m[2] + grid.step_m * gap["along"]
            else:
                z = grid.origin_m[2] + grid.step_m * (gap["line"] + 0.5)
                x = grid.origin_m[0] + grid.step_m * gap["along"]
            found.append(
                {
                    "across": across,
                    "centre_m": [round(float(x), 3), round(float(z), 3)],
                    "width_nodes": gap["width_nodes"],
                    "width_m": gap["width_nodes"] * grid.step_m,
                    "depth_m": gap["depth_m"],
                }
            )
    return found


def opening_changes(a: GridRecord, b: GridRecord, y_m: float) -> dict[str, Any]:
    """The openings of a layer in two grids, paired by place: which change width, open or seal."""
    mine, theirs = openings(a, y_m), openings(b, y_m)
    taken: set[int] = set()
    rows = []
    for one in mine:
        best, best_d = None, OPENING_MATCH_M
        for k, other in enumerate(theirs):
            if k in taken or other["across"] != one["across"]:
                continue
            d = float(np.hypot(*(np.asarray(one["centre_m"]) - np.asarray(other["centre_m"]))))
            if d <= best_d:
                best, best_d = k, d
        if best is not None:
            taken.add(best)
        rows.append(
            {
                "centre_m": one["centre_m"],
                "across": one["across"],
                "a": one,
                "b": theirs[best] if best is not None else None,
            }
        )
    for k, other in enumerate(theirs):
        if k not in taken:
            rows.append(
                {"centre_m": other["centre_m"], "across": other["across"], "a": None, "b": other}
            )
    changed = [
        row
        for row in rows
        if row["a"] is None
        or row["b"] is None
        or abs(row["a"]["width_m"] - row["b"]["width_m"]) > 1e-9
    ]
    changed.sort(
        key=lambda row: (
            -abs(
                (row["a"]["width_m"] if row["a"] else 0.0)
                - (row["b"]["width_m"] if row["b"] else 0.0)
            )
        )
    )
    return {
        "y_m": float(y_m),
        "layer_a_m": float(a.origin_m[1] + a.step_m * layer_of(a, y_m)),
        "layer_b_m": float(b.origin_m[1] + b.step_m * layer_of(b, y_m)),
        "openings_a": len(mine),
        "openings_b": len(theirs),
        "changed": changed,
        "rule": (
            f"a run of open links at most {OPENING_WIDTH_M} m wide between cut links, "
            f"at most {OPENING_DEPTH_M} m deep; paired within {OPENING_MATCH_M} m"
        ),
    }
