"""A hierarchy of boxes over the triangles, and the nearest hit of a ray through it.

The rays of the tail looked their triangles up in a uniform grid of 0.10 m:
a segment walked 28 cells and tested 140 triangles of a scene of 1.48 million
(``docs/open-questions/performance-audit.md``). A hierarchy of bounding boxes
is the structure for a scene of small triangles in large rooms: a segment
goes down a tree of 21 levels or so, visits some tens of nodes and tests a
handful of triangles.

**What is built.** A binary tree, each node holding the boxes of its two
children (so one read of 64 bytes decides both), cut by the surface area
heuristic over 16 bins of the centroids, breadth first and a whole level at a
time, in numpy. A child is a node, a leaf of 1 to :data:`LEAF_MOST` items
that are contiguous in :attr:`Bvh.order`, or nothing. The build is
deterministic: no random number, stable sorts, the lower index on a tie.

**What a traversal may change, and may not.** The boxes are single
precision, rounded outwards and padded (:func:`pad_of`), and the test of a
box is written so that it never refuses a box the ray in double precision
enters: it only ever adds candidates. The test of a triangle is the grid
twin's own, in double precision, and of two triangles hit at the same
distance the lower index is kept, which is what the twin's ``argmin`` over
its sorted candidates keeps. So **the nearest hit does not depend on the
structure**: the hierarchy returns the triangle and the distance the grid
returns, and the histograms are the same to the integer.

:func:`nearest` is the specification of the traversal the C text of
:mod:`reverberate.mirror.tracer` runs on a host and on a card: a stack a
ray, the nearer child first, the farther pushed with its entry distance and
dropped at the pop when a hit has come nearer. It advances every ray of a
batch by one node at a time, so it visits the nodes the C text visits and
its counts of nodes and of triangle tests are that text's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BINS",
    "EMPTY",
    "LEAF",
    "LEAF_MOST",
    "STACK",
    "Bvh",
    "build",
    "nearest",
    "pad_of",
    "triangle_boxes",
]

#: A segment of more items than this is cut; a leaf holds at most as many. On the storey
#: of 1.48 million triangles a segment tests 7.0, 8.5, 12.3 and 21.0 triangles with
#: leaves of 1, 2, 4 and 8, and visits 48, 45, 41 and 38 nodes: a triangle's test is
#: double precision where a node's is single, so the leaves are kept small.
LEAF = 2
#: What a leaf's count is written on: four bits.
LEAF_MOST = 16
#: Bins of the surface area heuristic along each axis.
BINS = 16
#: From this depth on a segment is cut in two halves, so the tree stays under :data:`STACK`.
HALVE_FROM = 48
#: The depth of the stack the traversals hold; a tree deeper than this is refused.
STACK = 96
#: The link of a child that is not there.
EMPTY = -(2**31)
#: A box's entry distance is taken a little short and its exit a little long.
SHRINK = np.float32(0.999998)
GROW = np.float32(1.000002)


def pad_of(lo: np.ndarray, hi: np.ndarray) -> float:
    """How far the single precision boxes are grown, m, for a scene between ``lo`` and ``hi``.

    A ray's origin and direction are rounded to single precision for the
    boxes, and so are the differences the test takes: together some units in
    the last place of the largest coordinate, which is 2e-6 m at 20 m. The
    pad is sixteen of them and never under 50 micrometres, which is nothing
    against a triangle of two centimetres and makes the test of a box
    conservative for any segment inside the scene.
    """
    largest = float(max(np.max(np.abs(lo), initial=0.0), np.max(np.abs(hi), initial=0.0), 1.0))
    return float(max(5e-5, 16.0 * float(np.spacing(np.float32(largest)))))


def triangle_boxes(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The boxes of ``[n, 3, 3]`` triangles."""
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    return triangles.min(axis=1), triangles.max(axis=1)


@dataclass(frozen=True)
class Bvh:
    """The tree: per node its two children's boxes and links, and the items in leaf order."""

    #: ``[node, 2, 6]`` single precision: each child's low corner, then its high corner.
    boxes: np.ndarray
    #: ``[node, 2]``: a node's index; a leaf as ``~(first << 4 | count - 1)``; :data:`EMPTY`.
    links: np.ndarray
    #: ``[item]``: the item at each place of the leaf order.
    order: np.ndarray
    depth: int
    pad_m: float

    @property
    def nodes(self) -> int:
        return int(self.links.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        """What a store keeps: :meth:`from_arrays` is the inverse."""
        return {
            "boxes": self.boxes,
            "links": self.links,
            "order": self.order,
            "shape": np.asarray([self.depth], dtype=np.int64),
            "pad_m": np.asarray([self.pad_m], dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, Any]) -> Bvh:
        return cls(
            boxes=arrays["boxes"],
            links=arrays["links"],
            order=arrays["order"],
            depth=int(arrays["shape"][0]),
            pad_m=float(arrays["pad_m"][0]),
        )

    def packed(self) -> np.ndarray:
        """``[node, 16]`` single precision words, 64 bytes a node: what the C text reads.

        Twelve floats of the two boxes, then the two links as the integers
        they are, in the same words, and two words of nothing.
        """
        out = np.zeros((self.nodes, 16), dtype=np.float32)
        out[:, :12] = self.boxes.reshape(self.nodes, 12)
        out.view(np.int32)[:, 12:14] = self.links
        return out

    def leaves(self) -> tuple[np.ndarray, np.ndarray]:
        """Every leaf's first place in the order and its count."""
        code = ~self.links[(self.links < 0) & (self.links != EMPTY)].astype(np.int64)
        return code >> 4, (code & 15) + 1


def _area(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Half the surface of boxes, zero for one that holds nothing (``lo > hi``)."""
    d = np.maximum(hi - lo, 0.0)
    held = np.all(hi >= lo, axis=-1)
    return np.asarray(
        np.where(held, d[..., 0] * d[..., 1] + d[..., 1] * d[..., 2] + d[..., 2] * d[..., 0], 0.0)
    )


def _leaf_link(first: np.ndarray, count: np.ndarray) -> np.ndarray:
    return np.asarray(~((first.astype(np.int64) << 4) | (count.astype(np.int64) - 1)))


def build(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    leaf: int = LEAF,
    bins: int = BINS,
    pad_m: float | None = None,
) -> Bvh:
    """The tree of the items whose boxes are ``lo`` and ``hi`` (``[n, 3]`` each)."""
    lo = np.ascontiguousarray(lo, dtype=np.float64).reshape(-1, 3)
    hi = np.ascontiguousarray(hi, dtype=np.float64).reshape(-1, 3)
    if not 1 <= leaf <= LEAF_MOST:
        raise ValueError(f"a leaf holds 1 to {LEAF_MOST} items, asked {leaf}")
    n = int(lo.shape[0])
    if n >= 2**27:
        raise ValueError(f"a leaf's first place is written on 27 bits, {n} items are too many")
    pad = pad_of(lo, hi) if pad_m is None else float(pad_m)
    centre = 0.5 * (lo + hi)
    perm = np.arange(n, dtype=np.int64)
    node_boxes: list[np.ndarray] = []
    node_links: list[np.ndarray] = []
    count_nodes = 0
    depth = 0
    if n <= leaf:
        box = np.full((1, 2, 6), 0.0)
        links = np.full((1, 2), EMPTY, dtype=np.int64)
        if n:
            box[0, 0, :3], box[0, 0, 3:] = lo.min(axis=0), hi.max(axis=0)
            links[0, 0] = _leaf_link(np.asarray([0]), np.asarray([n]))[0]
        node_boxes.append(box)
        node_links.append(links)
        start = np.zeros(0, dtype=np.int64)
        size = np.zeros(0, dtype=np.int64)
    else:
        start = np.asarray([0], dtype=np.int64)
        size = np.asarray([n], dtype=np.int64)
    # A level at a time: every segment still to cut is one node, numbered in the order of
    # its first place, and its two halves are leaves or the next level's segments.
    while start.size:
        depth += 1
        segments = int(start.size)
        offsets = np.concatenate([[0], np.cumsum(size)])
        which = np.repeat(np.arange(segments), size)
        place = np.repeat(start - offsets[:-1], size) + np.arange(int(offsets[-1]))
        items = perm[place]
        c = centre[items]
        heads = offsets[:-1]
        c_lo = np.minimum.reduceat(c, heads, axis=0)
        c_hi = np.maximum.reduceat(c, heads, axis=0)
        extent = c_hi - c_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(extent > 0.0, bins / extent, 0.0)
        binned = np.minimum(((c - c_lo[which]) * scale[which]).astype(np.int64), bins - 1)
        item_lo, item_hi = lo[items], hi[items]
        best_cost = np.full(segments, np.inf)
        best_axis = np.zeros(segments, dtype=np.int64)
        best_plane = np.zeros(segments, dtype=np.int64)
        for axis in range(3):
            key = which * bins + binned[:, axis]
            held = np.bincount(key, minlength=segments * bins).reshape(segments, bins)
            flat_lo = np.full((segments * bins, 3), np.inf)
            flat_hi = np.full((segments * bins, 3), -np.inf)
            for k in range(3):
                np.minimum.at(flat_lo[:, k], key, item_lo[:, k])
                np.maximum.at(flat_hi[:, k], key, item_hi[:, k])
            b_lo = flat_lo.reshape(segments, bins, 3)
            b_hi = flat_hi.reshape(segments, bins, 3)
            left_n = np.cumsum(held, axis=1)[:, :-1]
            right_n = np.cumsum(held[:, ::-1], axis=1)[:, ::-1][:, 1:]
            left_area = _area(
                np.minimum.accumulate(b_lo, axis=1)[:, :-1],
                np.maximum.accumulate(b_hi, axis=1)[:, :-1],
            )
            right_area = _area(
                np.minimum.accumulate(b_lo[:, ::-1], axis=1)[:, ::-1][:, 1:],
                np.maximum.accumulate(b_hi[:, ::-1], axis=1)[:, ::-1][:, 1:],
            )
            cost = left_area * left_n + right_area * right_n
            cost = np.where((left_n > 0) & (right_n > 0), cost, np.inf)
            plane = np.argmin(cost, axis=1)
            found = cost[np.arange(segments), plane]
            better = found < best_cost
            best_cost = np.where(better, found, best_cost)
            best_axis = np.where(better, axis, best_axis)
            best_plane = np.where(better, plane, best_plane)
        rank = np.arange(int(offsets[-1])) - heads[which]
        left = binned[np.arange(binned.shape[0]), best_axis[which]] <= best_plane[which]
        # Where no plane parts the centroids (they are one point), or the tree is deep
        # already, the segment is halved in the order it stands in.
        halved = ~np.isfinite(best_cost) | (depth >= HALVE_FROM)
        left = np.where(halved[which], rank < (size[which] + 1) // 2, left)
        arranged = np.argsort(which * 2 + (~left).astype(np.int64), kind="stable")
        perm[place] = items[arranged]
        left_n = np.bincount(which[left], minlength=segments).astype(np.int64)
        child_start = np.stack([start, start + left_n], axis=1).reshape(-1)
        child_size = np.stack([left_n, size - left_n], axis=1).reshape(-1)
        child_heads = np.stack([heads, heads + left_n], axis=1).reshape(-1)
        box = np.concatenate(
            [
                np.minimum.reduceat(item_lo[arranged], child_heads, axis=0),
                np.maximum.reduceat(item_hi[arranged], child_heads, axis=0),
            ],
            axis=1,
        )
        cut = child_size > leaf
        links = _leaf_link(child_start, child_size)
        links[cut] = count_nodes + segments + np.arange(int(np.count_nonzero(cut)))
        node_boxes.append(box.reshape(segments, 2, 6))
        node_links.append(links.reshape(segments, 2))
        count_nodes += segments
        start, size = child_start[cut], child_size[cut]
    # The first node made is the root; with a root that is a leaf it is the only one.
    if depth > STACK - 8:
        raise ValueError(f"the tree is {depth} deep, the traversal's stack holds {STACK}")
    boxes = np.concatenate(node_boxes)
    grown = np.empty(boxes.shape, dtype=np.float32)
    grown[..., :3] = np.nextafter((boxes[..., :3] - pad).astype(np.float32), np.float32(-np.inf))
    grown[..., 3:] = np.nextafter((boxes[..., 3:] + pad).astype(np.float32), np.float32(np.inf))
    return Bvh(
        boxes=grown,
        links=np.concatenate(node_links).astype(np.int32),
        order=perm.astype(np.int32),
        depth=max(depth, 1),
        pad_m=pad,
    )


# --------------------------------------------------------------------------
# the traversal
# --------------------------------------------------------------------------


def _slab(
    box: np.ndarray, origin: np.ndarray, inverse: np.ndarray, limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per ray, the entry distance into its box and whether the box is to be entered.

    Single precision throughout. ``fmin`` and ``fmax`` pass over the
    not-a-number an origin on a face gives a ray that runs along it, which
    then leaves that face out of the test.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        low = (box[:, :3] - origin) * inverse
        high = (box[:, 3:] - origin) * inverse
    near = np.fmin(low, high)
    far = np.fmax(low, high)
    t_near = np.fmax(np.fmax(near[:, 0], near[:, 1]), near[:, 2])
    t_far = np.fmin(np.fmin(far[:, 0], far[:, 1]), far[:, 2])
    t_near = np.fmax(t_near, np.float32(0.0)) * SHRINK
    t_far = t_far * GROW
    return t_near, (t_near <= t_far) & (t_near <= limit)


def moller_trumbore(
    origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row by row, the distance at which a ray hits its triangle, and whether it does.

    The grid twin's ``_nearest_hit`` with every product written out: a hit
    is inside the triangle, edges included, and farther than a micrometre.
    """
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    px = dy * e2[:, 2] - dz * e2[:, 1]
    py = dz * e2[:, 0] - dx * e2[:, 2]
    pz = dx * e2[:, 1] - dy * e2[:, 0]
    det = e1[:, 0] * px + e1[:, 1] * py + e1[:, 2] * pz
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tx, ty, tz = origin[:, 0] - v0[:, 0], origin[:, 1] - v0[:, 1], origin[:, 2] - v0[:, 2]
    u = (tx * px + ty * py + tz * pz) * inv
    qx = ty * e1[:, 2] - tz * e1[:, 1]
    qy = tz * e1[:, 0] - tx * e1[:, 2]
    qz = tx * e1[:, 1] - ty * e1[:, 0]
    v = (dx * qx + dy * qy + dz * qz) * inv
    t = (e2[:, 0] * qx + e2[:, 1] * qy + e2[:, 2] * qz) * inv
    return t, ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)


def nearest(
    tree: Bvh,
    ordered: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    exclude: np.ndarray | None = None,
    *,
    counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per ray, the distance to the first triangle it hits and that triangle; ``inf`` and -1.

    ``ordered`` are the triangles in the tree's leaf order (``triangles[tree.order]``);
    ``exclude`` is per ray a triangle not to hit, the one it leaves. The
    triangle returned is an index of the scene, not of the order. Of two
    triangles at one distance the lower index is returned. ``counts``, when
    given, gains the nodes visited and the triangles tested.
    """
    origins = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 3)
    rays = int(origins.shape[0])
    skip = np.full(rays, -1, dtype=np.int64) if exclude is None else np.asarray(exclude, np.int64)
    origin32 = origins.astype(np.float32)
    with np.errstate(divide="ignore"):
        inverse = np.float32(1.0) / directions.astype(np.float32)
    best = np.full(rays, np.inf)
    best_triangle = np.full(rays, -1, dtype=np.int64)
    limit = np.full(rays, np.inf, dtype=np.float32)
    node = np.zeros(rays, dtype=np.int64)
    top = np.zeros(rays, dtype=np.int64)
    stack_node = np.zeros((rays, STACK), dtype=np.int64)
    stack_near = np.zeros((rays, STACK), dtype=np.float32)
    alive = np.ones(rays, dtype=bool)
    visited = tested = 0
    links_of = tree.links.astype(np.int64)
    while True:
        active = np.flatnonzero(alive)
        if active.size == 0:
            break
        at = node[active]
        inner = at >= 0
        pop = np.zeros(active.size, dtype=bool)
        a = active[inner]
        if a.size:
            visited += int(a.size)
            boxes = tree.boxes[at[inner]]
            links = links_of[at[inner]]
            near0, hit0 = _slab(boxes[:, 0], origin32[a], inverse[a], limit[a])
            near1, hit1 = _slab(boxes[:, 1], origin32[a], inverse[a], limit[a])
            hit0 &= links[:, 0] != EMPTY
            hit1 &= links[:, 1] != EMPTY
            both = hit0 & hit1
            swap = both & (near1 < near0)
            go = np.where(hit0 & ~swap, links[:, 0], links[:, 1])
            pushed = a[both]
            stack_node[pushed, top[pushed]] = np.where(swap, links[:, 0], links[:, 1])[both]
            stack_near[pushed, top[pushed]] = np.where(swap, near0, near1)[both]
            top[pushed] += 1
            any_hit = hit0 | hit1
            node[a[any_hit]] = go[any_hit]
            pop[np.flatnonzero(inner)[~any_hit]] = True
        leaves = active[~inner]
        if leaves.size:
            code = ~at[~inner]
            first, held = code >> 4, (code & 15) + 1
            owner = np.repeat(leaves, held)
            slot = np.repeat(first, held) + (
                np.arange(int(held.sum())) - np.repeat(np.cumsum(held) - held, held)
            )
            tested += int(slot.size)
            t, hit = moller_trumbore(origins[owner], directions[owner], ordered[slot])
            triangle = tree.order[slot].astype(np.int64)
            hit &= triangle != skip[owner]
            owner, t, triangle = owner[hit], t[hit], triangle[hit]
            # The nearest of a ray's hits in this leaf, the lower index on a tie, against
            # what the ray holds already under the same rule.
            ranked = np.lexsort((triangle, t, owner))
            owner, t, triangle = owner[ranked], t[ranked], triangle[ranked]
            head = np.ones(owner.size, dtype=bool)
            head[1:] = owner[1:] != owner[:-1]
            owner, t, triangle = owner[head], t[head], triangle[head]
            better = (t < best[owner]) | ((t == best[owner]) & (triangle < best_triangle[owner]))
            owner, t, triangle = owner[better], t[better], triangle[better]
            best[owner] = t
            best_triangle[owner] = triangle
            limit[owner] = t.astype(np.float32) * GROW
            pop[~inner] = True
        waiting = active[pop]
        while waiting.size:
            empty = top[waiting] == 0
            alive[waiting[empty]] = False
            waiting = waiting[~empty]
            top[waiting] -= 1
            near = stack_near[waiting, top[waiting]]
            taken = near <= limit[waiting]
            node[waiting[taken]] = stack_node[waiting[taken], top[waiting[taken]]]
            waiting = waiting[~taken]
    if counts is not None:
        counts["nodes"] = counts.get("nodes", 0) + visited
        counts["tests"] = counts.get("tests", 0) + tested
        counts["segments"] = counts.get("segments", 0) + rays
    return best, best_triangle
