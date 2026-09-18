"""The sound that reaches a receiver round an obstacle: the shortest path and its loss.

A receiver in a room the source does not see (a bedroom off the living
room) has no direct path and, at order 3, often no reflection either; the
reference reaches it first through the doorway, by diffraction. The mirror
had nothing there until its first ray, a broadband diffuse burst, so the
onset, its colour and the timing of everything after it were wrong at 185 of
the 437 points of hssd_0076.

Here the diffracted onset is one path:

- the storey's occluders are sampled into an occupancy grid, grown by one
  cell so a one-cell wall cannot leak along a diagonal;
- the geodesic from the source's cell to every cell is Dijkstra's on the
  26-connected free cells;
- a receiver's chain of cells back to the source is pulled tight: from the
  receiver, the farthest cell of the chain it sees is the last edge, and so
  on to the source; the legs' lengths are the path's length;
- each edge loses Maekawa's ``10 log10(3 + 20 N)`` dB per band, capped at
  24 dB, with ``N = 2 delta f / c`` and ``delta`` the detour that edge adds;
- the path arrives from the last edge.

It is an estimate of the first arrival's time, direction and colour, not a
wave solution; the report says how many points received one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import Paths

__all__ = [
    "DiffractionSettings",
    "Edges",
    "Occupancy",
    "diffracted_paths",
    "diffracting_edges",
    "edge_paths",
    "maekawa_db",
    "occupancy_of",
]


@dataclass(frozen=True)
class DiffractionSettings:
    """What the diffracted onset chooses."""

    cell_m: float = 0.10
    #: Cells the occupancy is grown by, so thin walls stay closed.
    grow_cells: int = 1
    #: Least samples per triangle edge when the occluders are rasterised.
    samples_per_edge: int = 4
    #: Cells kept free round the source and each receiver.
    clear_cells: int = 1
    #: A pulled corner adding less detour than this is the grid's, not an edge.
    min_detour_m: float = 0.05
    #: Reflection order of the edge's own image tree. The edge the sound bends
    #: round is a source in the receiver's room, and that room reflects it:
    #: on 0076 the reference gives a point behind a doorway a floor and a
    #: ceiling bounce 3 to 6 dB under its onset, and the mirror had neither.
    #: Zero keeps the onset alone.
    reflections: int = 1
    #: Edges closer together than this share one image tree.
    cluster_m: float = 0.25
    #: Paths kept per receiver, the shortest first.
    max_paths: int = 24
    #: Shortest edge kept as a diffracting one. Under about a third of a
    #: metre an edge bends too little of the band the mirror answers.
    min_edge_m: float = 0.30
    #: An edge whose faces absorb more than this at 1 kHz is left out: a
    #: curtain's hem bends a field that is not there.
    max_edge_absorption: float = 0.60
    #: How much longer than the straight line a single edge path may be.
    #: Measured on the 32 shadowed points of hssd_0076: the share of criteria
    #: met is 0.230 with no edge at all, 0.303 at 0.5 m, 0.309 at 1.5 m,
    #: 0.297 at 3 m and 0.289 at 8 m. Long ways round spread the early part
    #: in time, which costs early decay time and seam more than their own
    #: directions gain.
    max_edge_detour_m: float = 1.5
    #: Whether every selected edge is tried, beside the geodesic.
    edges: bool = True
    #: The geodesic walks a grid, so its corners sit at cell centres and its
    #: length, arrival and direction carry the grid's step. Each corner
    #: within this of a selected edge is pulled onto it. Measured on 40
    #: shadowed points of hssd_0076 (2026-09-18, after the bend rule below
    #: was fixed): criteria met 0.2297 against 0.2250 without, the onset's
    #: level error 2.96 against 4.07 dB, its direction 1.47 against 1.63
    #: degrees. A corner that stands on an edge is a bend whatever kink is
    #: left once the chain is pulled tight.
    #: How far a corner may be from an edge and still be taken as that edge.
    snap_m: float = 0.30

    def record(self) -> dict[str, Any]:
        return {
            "cell_m": self.cell_m,
            "grow_cells": self.grow_cells,
            "samples_per_edge": self.samples_per_edge,
            "clear_cells": self.clear_cells,
            "min_detour_m": self.min_detour_m,
            "reflections": self.reflections,
            "cluster_m": self.cluster_m,
            "max_paths": self.max_paths,
            "min_edge_m": self.min_edge_m,
            "max_edge_absorption": self.max_edge_absorption,
            "max_edge_detour_m": self.max_edge_detour_m,
            "edges": self.edges,
            "snap_m": self.snap_m,
        }


@dataclass
class Occupancy:
    """A boolean grid of the cells the occluders take, and where it sits."""

    blocked: np.ndarray
    origin: np.ndarray
    cell_m: float

    def cell_of(self, points: np.ndarray) -> np.ndarray:
        index = np.floor((np.atleast_2d(points) - self.origin) / self.cell_m).astype(int)
        return np.asarray(np.clip(index, 0, np.asarray(self.blocked.shape) - 1))

    def centre_of(self, index: np.ndarray) -> np.ndarray:
        return np.asarray(self.origin + (np.atleast_2d(index) + 0.5) * self.cell_m)

    def flat(self, index: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.blocked.shape
        index = np.atleast_2d(index)
        return np.asarray((index[:, 0] * ny + index[:, 1]) * nz + index[:, 2])

    def sees(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether the segment from ``a`` to ``b`` crosses no blocked cell."""
        length = float(np.linalg.norm(b - a))
        steps = max(2, int(np.ceil(length / (0.4 * self.cell_m))) + 1)
        t = np.linspace(0.0, 1.0, steps)[:, None]
        cells = self.cell_of(a[None, :] * (1.0 - t) + b[None, :] * t)
        return not bool(np.any(self.blocked[cells[:, 0], cells[:, 1], cells[:, 2]]))


def _surface_samples(
    triangles: np.ndarray, cell_m: float, least: int, chunk: int = 200_000
) -> np.ndarray:
    """Points on every triangle, on a barycentric lattice finer than half a cell.

    A triangle's lattice has ``k`` steps per edge, ``k`` from its longest
    edge, at least ``least``: the export's small triangles take the least,
    a box's wall takes as many as it needs.
    """
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    ).max(axis=1)
    steps_of = np.clip(np.ceil(edges / (0.5 * cell_m)), least, 4096).astype(int)
    out = []
    for k in np.unique(steps_of):
        lattice = [(i / k, j / k) for i in range(k + 1) for j in range(k + 1 - i)]
        weights = np.asarray([(1.0 - u - v, u, v) for u, v in lattice], dtype=np.float32)
        group = triangles[steps_of == k]
        size = max(1, chunk // max(1, len(lattice) // 15))
        for begin in range(0, group.shape[0], size):
            tri = group[begin : begin + size].astype(np.float32)
            out.append(np.einsum("sk,tkd->tsd", weights, tri).reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def occupancy_of(
    scene: DerivedScene,
    lo: np.ndarray,
    hi: np.ndarray,
    settings: DiffractionSettings | None = None,
) -> Occupancy:
    """The occluders of ``scene`` inside ``[lo, hi]`` as a grown boolean grid."""
    settings = settings or DiffractionSettings()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    shape = tuple(int(v) for v in np.ceil((hi - lo) / settings.cell_m))
    blocked = np.zeros(shape, dtype=bool)
    triangles = scene.occluder_vertices.reshape(-1, 3, 3)
    points = _surface_samples(triangles, settings.cell_m, settings.samples_per_edge)
    index = np.floor((points - lo) / settings.cell_m).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(shape)), axis=1)
    index = index[inside]
    blocked[index[:, 0], index[:, 1], index[:, 2]] = True
    if settings.grow_cells > 0:
        blocked = ndimage.binary_dilation(
            blocked, structure=np.ones((3, 3, 3), dtype=bool), iterations=settings.grow_cells
        )
    return Occupancy(blocked, lo, settings.cell_m)


def _clear(occupancy: Occupancy, points: np.ndarray, radius: int) -> None:
    shape = np.asarray(occupancy.blocked.shape)
    for c in occupancy.cell_of(points):
        a = np.maximum(c - radius, 0)
        b = np.minimum(c + radius + 1, shape)
        occupancy.blocked[a[0] : b[0], a[1] : b[1], a[2] : b[2]] = False


def _graph(occupancy: Occupancy) -> csr_matrix:
    """The 26-connected free cells, edges weighted by their length."""
    free = ~occupancy.blocked
    nx, ny, nz = free.shape
    ids = np.arange(free.size).reshape(free.shape)
    rows = []
    cols = []
    weights = []
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue  # each undirected pair once
                xs = slice(0, nx - dx)
                xd = slice(dx, nx)
                ys = slice(max(0, -dy), ny - max(0, dy))
                yd = slice(max(0, dy), ny - max(0, -dy))
                zs = slice(max(0, -dz), nz - max(0, dz))
                zd = slice(max(0, dz), nz - max(0, -dz))
                both = free[xs, ys, zs] & free[xd, yd, zd]
                a = ids[xs, ys, zs][both]
                b = ids[xd, yd, zd][both]
                rows.append(a)
                cols.append(b)
                weights.append(
                    np.full(a.size, occupancy.cell_m * float(np.sqrt(dx * dx + dy * dy + dz * dz)))
                )
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.concatenate(weights)
    return csr_matrix((w, (r, c)), shape=(free.size, free.size))


def maekawa_db(
    detour_m: float, bands_hz: np.ndarray, sound_speed_m_s: float, cap_db: float = 24.0
) -> np.ndarray:
    """Maekawa's insertion loss of one edge per band, for a detour of ``detour_m``.

    Capped at 24 dB, Maekawa's own practical limit. On hssd_0076 caps of 24,
    32, 40 and 60 dB read the same: the onset sits some 20 dB under the
    peak there, and the judgement's first arrival is another one.
    """
    fresnel = 2.0 * max(detour_m, 0.0) * np.asarray(bands_hz, dtype=float) / sound_speed_m_s
    return np.asarray(np.minimum(10.0 * np.log10(3.0 + 20.0 * fresnel), cap_db))


def _detours(corners: list[np.ndarray]) -> list[float]:
    """Per corner, how much longer its two legs are than the line across them."""
    out = [0.0] * len(corners)
    for k in range(1, len(corners) - 1):
        out[k] = float(
            np.linalg.norm(corners[k] - corners[k - 1])
            + np.linalg.norm(corners[k + 1] - corners[k])
            - np.linalg.norm(corners[k + 1] - corners[k - 1])
        )
    return out


def _pull(occupancy: Occupancy, chain: np.ndarray) -> list[np.ndarray]:
    """The chain of points from receiver to source, pulled tight: the corners kept."""
    corners = [chain[0]]
    at = 0
    last = len(chain) - 1
    while at < last:
        anchor = chain[at]
        step = last
        while step > at + 1 and not occupancy.sees(anchor, chain[step]):
            step -= 1
        corners.append(chain[step])
        at = step
    return corners


# --------------------------------------------------------------------------
# the edges themselves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Edges:
    """The scene's diffracting edges: where they are, how long, whose they are."""

    #: ``[edge, 3]`` the two ends.
    a: np.ndarray
    b: np.ndarray
    #: ``[edge]`` the label the edge's facet carries, and the facet itself.
    label: np.ndarray
    facet: np.ndarray
    length_m: np.ndarray

    @property
    def count(self) -> int:
        return int(self.a.shape[0])

    def record(self) -> dict[str, Any]:
        from collections import Counter

        return {
            "edges": self.count,
            "length_m_total": round(float(self.length_m.sum()), 2),
            "length_m_median": round(float(np.median(self.length_m)), 3) if self.count else None,
            "by_label": dict(Counter(int(v) for v in self.label).most_common(8)),
        }


def _lines_of(
    facet_triangles: np.ndarray, min_length_m: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The facet's own boundary, as the longest segment on each of its lines.

    A facet is a merged plane of many triangles; an edge shared by two of
    them is inside it, and one held by a single triangle is its rim. The rim
    is made of many short collinear pieces, so the pieces on one line are
    taken together and only the whole is kept.
    """
    tri = facet_triangles
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    key = np.round(np.concatenate([np.minimum(a, b), np.maximum(a, b)], axis=1), 5)
    view = np.ascontiguousarray(key).view([("", key.dtype)] * 6).ravel()
    unique, counts = np.unique(view, return_counts=True)
    rim = unique[counts == 1].view(key.dtype).reshape(-1, 6)
    if rim.shape[0] == 0:
        return []
    p, q = rim[:, :3], rim[:, 3:]
    direction = q - p
    length = np.linalg.norm(direction, axis=1)
    alive = length > 1e-9
    p, q, direction, length = p[alive], q[alive], direction[alive], length[alive]
    unit = direction / length[:, None]
    # One sense per line, so a piece and its reverse land together.
    flip = (unit[:, 0] < -1e-9) | ((np.abs(unit[:, 0]) < 1e-9) & (unit[:, 1] < -1e-9))
    unit[flip] *= -1.0
    foot = p - np.einsum("ij,ij->i", p, unit)[:, None] * unit
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i in range(unit.shape[0]):
        marker = (tuple(np.round(unit[i], 3)), tuple(np.round(foot[i], 2)))
        groups.setdefault(marker, []).append(i)
    out = []
    for members in groups.values():
        axis = unit[members[0]]
        along = np.concatenate([p[members] @ axis, q[members] @ axis])
        low, high = float(along.min()), float(along.max())
        if high - low < min_length_m:
            continue
        origin = foot[members[0]]
        out.append((origin + low * axis, origin + high * axis))
    return out


def diffracting_edges(scene: DerivedScene, settings: DiffractionSettings | None = None) -> Edges:
    """The edges worth bending sound round, chosen by their length and their material.

    Geometry first: an edge is the rim of a reflector facet, taken whole
    along each of its lines, and it is kept when it is at least
    ``min_edge_m`` long. Below that it bends too little of the band the
    mirror answers, and there are thousands of them.

    Material second: an edge whose facet absorbs more than
    ``max_edge_absorption`` at 1 kHz is dropped. What a curtain's hem bends
    is a field the curtain has already taken.

    On hssd_0076 this leaves 709 edges of 19 805 triangle rim pieces, 1 425 m
    of them the shell's own: the doorways, the window reveals and the wall
    returns, which is what a point in another room hears first.
    """
    settings = settings or DiffractionSettings()
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    at_1k = int(np.argmin(np.abs(bands - 1000.0)))
    ends_a: list[np.ndarray] = []
    ends_b: list[np.ndarray] = []
    labels: list[int] = []
    facets: list[int] = []
    for index, facet in enumerate(scene.facets):
        absorption = float(scene.materials.absorption[facet.label][at_1k])
        if absorption > settings.max_edge_absorption:
            continue
        for low, high in _lines_of(scene.reflector_vertices[facet.triangles], settings.min_edge_m):
            ends_a.append(low)
            ends_b.append(high)
            labels.append(int(facet.label))
            facets.append(index)
    if not ends_a:
        empty = np.zeros((0, 3))
        return Edges(empty, empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    a = np.asarray(ends_a, dtype=float)
    b = np.asarray(ends_b, dtype=float)
    return Edges(
        a=a,
        b=b,
        label=np.asarray(labels, dtype=np.int64),
        facet=np.asarray(facets, dtype=np.int64),
        length_m=np.linalg.norm(b - a, axis=1),
    )


def _aperture(a: np.ndarray, b: np.ndarray, source: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """The point on each segment ``a -> b`` with the shortest way from source to receiver.

    The sum of the two legs is convex along the segment, so a golden section
    over the whole of it finds the one point, for every edge at once.
    """
    lo = np.zeros(a.shape[0])
    hi = np.ones(a.shape[0])
    phi = 0.5 * (np.sqrt(5.0) - 1.0)

    def total(t: np.ndarray) -> np.ndarray:
        point = a + t[:, None] * (b - a)
        legs = np.linalg.norm(point - source, axis=1) + np.linalg.norm(point - receiver, axis=1)
        return np.asarray(legs, dtype=float)

    left = hi - phi * (hi - lo)
    right = lo + phi * (hi - lo)
    f_left, f_right = total(left), total(right)
    for _ in range(24):
        take = f_left < f_right
        hi = np.where(take, right, hi)
        lo = np.where(take, lo, left)
        left = hi - phi * (hi - lo)
        right = lo + phi * (hi - lo)
        f_left, f_right = total(left), total(right)
    t = 0.5 * (lo + hi)
    return np.asarray(a + t[:, None] * (b - a), dtype=float)


def edge_paths(
    scene: DerivedScene,
    edges: Edges,
    source: np.ndarray,
    receiver: np.ndarray,
    *,
    grid: Any,
    sound_speed_m_s: float,
    settings: DiffractionSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One diffracted arrival per edge the receiver can see the source round.

    Returns the lengths, the directions, the per band gains and the points
    the sound bent at, for the paths that survive: the edge's own aperture
    point is found, the two legs are tested against the occluders, and
    Maekawa's loss for that detour is applied. A shadowed point in the
    reference hears its first sound round several edges at once, and the
    geodesic can only give it one.
    """
    from reverberate.mirror.ism import _segment_hits

    nothing = (np.zeros(0), np.zeros((0, 3)), np.zeros((0, len(OCTAVE_BANDS))), np.zeros((0, 3)))
    if edges.count == 0:
        return nothing
    straight = float(np.linalg.norm(receiver - source))
    point = _aperture(edges.a, edges.b, source, receiver)
    first = np.linalg.norm(point - source, axis=1)
    second = np.linalg.norm(point - receiver, axis=1)
    total = first + second
    near = total <= straight + settings.max_edge_detour_m
    near &= second > 1e-3
    if not bool(np.any(near)):
        return nothing
    picked = np.flatnonzero(near)
    order = picked[np.argsort(total[picked])]
    clear_a = ~_segment_hits(
        np.repeat(source[None, :], order.size, axis=0),
        point[order],
        scene.occluder_vertices,
        grid,
        epsilon_m=0.01,
    )
    keep = order[clear_a]
    if keep.size == 0:
        return nothing
    clear_b = ~_segment_hits(
        point[keep],
        np.repeat(receiver[None, :], keep.size, axis=0),
        scene.occluder_vertices,
        grid,
        epsilon_m=0.01,
    )
    keep = keep[clear_b]
    if keep.size == 0:
        return nothing
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    lengths = total[keep]
    towards = point[keep] - receiver
    directions = towards / np.maximum(np.linalg.norm(towards, axis=1, keepdims=True), 1e-9)
    gains = np.zeros((keep.size, bands.size))
    for i, index in enumerate(keep):
        loss = maekawa_db(float(total[index] - straight), bands, sound_speed_m_s)
        gains[i] = 10.0 ** (-loss / 20.0) / max(float(total[index]), 1e-3)
    return lengths, directions, gains, point[keep]


def _nearest_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The point of each segment ``a -> b`` closest to ``point``."""
    along = b - a
    length2 = np.einsum("ij,ij->i", along, along)
    t = np.where(
        length2 > 1e-18,
        np.einsum("ij,ij->i", point[None, :] - a, along) / np.maximum(length2, 1e-18),
        0.0,
    )
    return np.asarray(a + np.clip(t, 0.0, 1.0)[:, None] * along)


def _on_segment(a: np.ndarray, b: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """The point of one segment with the shortest way from ``before`` to ``after``."""
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    lo, hi = 0.0, 1.0

    def total(t: float) -> float:
        point = a + t * (b - a)
        return float(np.linalg.norm(point - before) + np.linalg.norm(point - after))

    left, right = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f_left, f_right = total(left), total(right)
    for _ in range(20):
        if f_left < f_right:
            hi, right, f_right = right, left, f_left
            left = hi - phi * (hi - lo)
            f_left = total(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + phi * (hi - lo)
            f_right = total(right)
    return np.asarray(a + 0.5 * (lo + hi) * (b - a))


def snap_to_edges(
    corners: list[np.ndarray], edges: Edges, settings: DiffractionSettings, sweeps: int = 3
) -> tuple[list[np.ndarray], list[bool]]:
    """The geodesic's corners pulled onto the edges they stand near, then tightened.

    A corner from the grid sits at a cell centre, so the way round is as
    long as the grid is coarse and arrives from a direction the grid chose.
    Each interior corner is matched to the nearest selected edge, within
    ``snap_m``; a corner with no edge near it is left where it is, since the
    grid may have gone round an occluder that carries no reflector facet at
    all. What is matched is then swept back and forth, each point taking the
    place on its own edge that shortens its two legs, which is the shortest
    way round that sequence of edges.

    Returns the corners and, per corner, whether it ended on an edge. The
    caller needs the second: tightening the chain takes the kink out of a
    bend, and a bend whose kink is gone is still a bend if it stands on an
    edge. Counting bends by the kink they have left loses one of every two
    on 0076, and 10 dB of loss with them.
    """
    if edges.count == 0 or len(corners) < 3:
        return corners, [False] * len(corners)
    held: list[int | None] = [None] * len(corners)
    out = [np.asarray(c, dtype=float) for c in corners]
    for k in range(1, len(corners) - 1):
        feet = _nearest_on_segment(out[k], edges.a, edges.b)
        gaps = np.linalg.norm(feet - out[k], axis=1)
        best = int(np.argmin(gaps))
        if float(gaps[best]) <= settings.snap_m:
            held[k] = best
            out[k] = feet[best]
    if not any(h is not None for h in held):
        return corners, [False] * len(corners)
    for _ in range(sweeps):
        for k in range(1, len(out) - 1):
            edge = held[k]
            if edge is None:
                continue
            out[k] = _on_segment(edges.a[edge], edges.b[edge], out[k - 1], out[k + 1])
    return out, [h is not None for h in held]


def diffracted_paths(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    indices: list[int],
    *,
    sound_speed_m_s: float,
    settings: DiffractionSettings | None = None,
    say: Any = None,
) -> tuple[dict[int, Paths], dict[str, Any]]:
    """For each receiver in ``indices``: the diffracted onset as a one-path ``Paths``.

    Receivers the geodesic cannot reach are left out. The path's ``order`` is
    0 for the rendering (it carries no reflection) but it is not a direct
    path: the caller keeps it apart.
    """
    settings = settings or DiffractionSettings()
    source = np.asarray(source, dtype=float)
    receivers = np.asarray(receivers, dtype=float)
    chosen = receivers[indices] if indices else np.zeros((0, 3))
    # The whole storey: the way round may leave the receivers' box.
    lo = np.minimum(np.minimum(receivers.min(axis=0), source), scene.bmin) - 0.3
    hi = np.maximum(np.maximum(receivers.max(axis=0), source), scene.bmax) + 0.3
    occupancy = occupancy_of(scene, lo, hi, settings)
    _clear(occupancy, np.vstack([source[None, :], chosen]), settings.clear_cells)
    selected = diffracting_edges(scene, settings)
    graph = _graph(occupancy)
    start = int(occupancy.flat(occupancy.cell_of(source))[0])
    distance, predecessor = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    shape = occupancy.blocked.shape
    out: dict[int, Paths] = {}
    secondary: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    lengths = []
    for index in indices:
        receiver = receivers[index]
        node = int(occupancy.flat(occupancy.cell_of(receiver))[0])
        if not np.isfinite(distance[node]):
            continue
        nodes = [node]
        while nodes[-1] != start and predecessor[nodes[-1]] >= 0:
            nodes.append(int(predecessor[nodes[-1]]))
        cells = np.stack(np.unravel_index(np.asarray(nodes), shape), axis=1)
        chain = occupancy.centre_of(cells)
        chain[0] = receiver
        chain[-1] = source
        corners = _pull(occupancy, chain)
        loose = _detours(corners)
        corners, on_edge = snap_to_edges(corners, selected, settings)
        legs = [
            float(np.linalg.norm(b - a)) for a, b in zip(corners[:-1], corners[1:], strict=True)
        ]
        length = float(sum(legs))
        loss = np.zeros(len(bands))
        edges = 0
        tight = _detours(corners)
        for k in range(1, len(corners) - 1):
            # A corner on an edge is a bend whatever kink it has left: pulling
            # the chain tight takes the kink out and not the wall. Its detour
            # keeps the grid's slack, the only reading that still knows how
            # deep in the shadow the receiver is (on 40 shadowed points of
            # 0076: 0.2297 of criteria met, against 0.2297 for the tightened
            # detour alone and 0.2281 for one loss over the whole way round).
            detour = max(tight[k], loose[k]) if on_edge[k] else tight[k]
            if detour < settings.min_detour_m and not on_edge[k]:
                # A corner the grid made, not an edge the sound bends round.
                continue
            edges += 1
            loss += maekawa_db(detour, bands, sound_speed_m_s)
        if edges == 0:
            # Pulled straight within the grid's slack: the shadow boundary's 4.8 dB.
            loss += maekawa_db(0.0, bands, sound_speed_m_s)
        towards = corners[1] - receiver
        direction = towards / max(float(np.linalg.norm(towards)), 1e-9)
        gain = 10.0 ** (-loss / 20.0) / max(length, 1e-3)
        points = np.repeat(receiver[None, None, :], 5, axis=1)
        points[0, 0] = source
        out[index] = Paths(
            receiver=receiver.copy(),
            image=np.asarray([-1], dtype=np.int64),
            order=np.asarray([0], dtype=np.int64),
            length_m=np.asarray([length]),
            direction=direction[None, :],
            gain=gain[None, :],
            points=points,
            sequence=np.full((1, 3), -1, dtype=np.int64),
        )
        # The last corner is a source in the receiver's own room; keep what the
        # room needs to reflect it: where it stands, how far the sound already
        # travelled to reach it, and what the edges took out on the way.
        secondary[index] = (np.asarray(corners[1], dtype=float), length - legs[0], loss)
        lengths.append(length - float(np.linalg.norm(receiver - source)))
    reflected = 0
    trees = 0
    edge_record: dict[str, Any] = {}
    if settings.edges:
        out, secondary, edge_record = _through_the_edges(
            scene, source, receivers, out, secondary, sound_speed_m_s, settings
        )
    if settings.reflections > 0 and secondary:
        out, trees, reflected = _reflect_the_edges(scene, receivers, out, secondary, settings)
    record = {
        "settings": settings.record(),
        "grid": list(shape),
        "blocked_fraction": round(float(occupancy.blocked.mean()), 4),
        "asked": len(indices),
        "found": len(out),
        "detour_m_median": round(float(np.median(lengths)), 3) if lengths else None,
        "edge_trees": trees,
        "reflected_paths": reflected,
        **edge_record,
    }
    if say is not None:
        say(
            f"diffraction: {len(out)} of {len(indices)} points, grid {shape}, "
            f"{trees} edge trees, {reflected} reflected paths"
        )
    return out, record


def _through_the_edges(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    out: dict[int, Paths],
    secondary: dict[int, tuple[np.ndarray, float, np.ndarray]],
    sound_speed_m_s: float,
    settings: DiffractionSettings,
) -> tuple[dict[int, Paths], dict[int, tuple[np.ndarray, float, np.ndarray]], dict[str, Any]]:
    """Every selected edge tried for every shadowed receiver, beside the geodesic.

    The geodesic gives one way round and therefore one direction; the
    reference's first sound arrives round several edges at once, which is
    why the onset's direction on hssd_0076 was unbiased and yet spread
    +-35 degrees in elevation. Here each edge that both the source and the
    receiver can see contributes its own arrival, at its own time, from its
    own direction. The geodesic is kept only when it is shorter than every
    single edge path, which is where it earns its place: a way round two
    corners that no one edge gives.
    """
    from reverberate.mirror.ism import occluder_grid

    edges = diffracting_edges(scene, settings)
    grid = occluder_grid(scene)
    bands = len(OCTAVE_BANDS)
    found = 0
    per_point: list[int] = []
    for index in sorted(out):
        lengths, directions, gains, apertures = edge_paths(
            scene,
            edges,
            source,
            receivers[index],
            grid=grid,
            sound_speed_m_s=sound_speed_m_s,
            settings=settings,
        )
        if lengths.size == 0:
            per_point.append(0)
            continue
        # Two facets can share one rim; one arrival is enough.
        marker = np.round(np.column_stack([lengths, directions]), 3)
        _, first = np.unique(marker, axis=0, return_index=True)
        pick = np.sort(first)
        lengths, directions, gains = lengths[pick], directions[pick], gains[pick]
        apertures = apertures[pick]
        keep = np.argsort(lengths)[: settings.max_paths]
        lengths, directions, gains = lengths[keep], directions[keep], gains[keep]
        apertures = apertures[keep]
        onset = out[index]
        geodesic = float(onset.length_m[0])
        rows = lengths.size
        nearest = int(np.argmin(lengths))
        aperture = apertures[nearest]
        before = float(np.linalg.norm(aperture - source))
        loss_db = -20.0 * np.log10(np.maximum(gains[nearest] * float(lengths[nearest]), 1e-12))
        if geodesic < float(lengths.min()) - 0.01:
            lengths = np.concatenate([onset.length_m[:1], lengths])
            directions = np.vstack([onset.direction[:1], directions])
            gains = np.vstack([onset.gain[:1], gains])
        # One barrier's worth of sound, shared over the ways round it. Maekawa's
        # loss is a whole barrier's answer; giving it to each of a doorway's
        # edges and adding them counts the same sound several times (on 0076
        # that cost 0.20 of early decay time and 2.3 dB of seam).
        share = np.sqrt(np.sum(onset.gain[0] ** 2) / max(float(np.sum(gains**2)), 1e-300))
        gains = gains * share
        points = np.repeat(receivers[index][None, None, :], 5, axis=1)
        points = np.repeat(points, lengths.size, axis=0)
        points[:, 0] = source
        out[index] = Paths(
            receiver=receivers[index].copy(),
            image=np.full(lengths.size, -1, dtype=np.int64),
            order=np.zeros(lengths.size, dtype=np.int64),
            length_m=lengths,
            direction=directions,
            gain=gains,
            points=points,
            sequence=np.full((lengths.size, 3), -1, dtype=np.int64),
        )
        # The room reflects the nearest edge, so the tree moves there too.
        secondary[index] = (aperture, before, loss_db[:bands])
        found += 1
        per_point.append(int(rows))
    return (
        out,
        secondary,
        {
            "edges_kept": edges.count,
            "edges_record": edges.record(),
            "points_with_edge_paths": found,
            "edge_paths_median": float(np.median(per_point)) if per_point else 0.0,
        },
    )


def _reflect_the_edges(
    scene: DerivedScene,
    receivers: np.ndarray,
    out: dict[int, Paths],
    secondary: dict[int, tuple[np.ndarray, float, np.ndarray]],
    settings: DiffractionSettings,
) -> tuple[dict[int, Paths], int, int]:
    """Give each diffracted onset the reflections of its own edge in the receiver's room.

    The corner the sound last bent round is a point source standing in the
    doorway. Its image tree, grown to ``settings.reflections``, gives the
    receiver the floor, the ceiling and the near walls of the room it is
    actually in, which the source's own tree cannot reach at all. Every
    such path carries the edges' loss and spreads over the whole way, the
    leg before the edge included, so the onset itself comes back unchanged.

    Edges within ``settings.cluster_m`` of each other share one tree, which
    is what makes this affordable: on 0076 the 185 points with no direct
    path stand behind some twenty doorways.
    """
    from reverberate.mirror.ism import IsmSettings, grow_tree, occluder_grid, paths_for

    grid = occluder_grid(scene)
    ism = IsmSettings(max_order=settings.reflections, flutter_order=0)
    order = sorted(secondary)
    taken: list[np.ndarray] = []
    groups: list[list[int]] = []
    for index in order:
        point = secondary[index][0]
        for k, centre in enumerate(taken):
            if float(np.linalg.norm(point - centre)) <= settings.cluster_m:
                groups[k].append(index)
                break
        else:
            taken.append(point)
            groups.append([index])
    reflected = 0
    for centre, members in zip(taken, groups, strict=True):
        tree = grow_tree(scene, centre, ism)
        for index in members:
            edge, before_m, loss_db = secondary[index]
            found = paths_for(scene, tree, receivers[index], ism, grid=grid)
            keep = found.order > 0
            if not bool(np.any(keep)):
                continue
            after = found.length_m[keep]
            total = after + before_m
            # One point source at the edge: the spreading is over the whole
            # way, not over the leg after it, and the edges' loss applies.
            scale = (after / np.maximum(total, 1e-6))[:, None] * 10.0 ** (-loss_db / 20.0)[None, :]
            onset = out[index]
            reflected_points = _shift_points(found.points[keep], edge)
            width = max(onset.points.shape[1], reflected_points.shape[1])
            columns = max(onset.sequence.shape[1], found.sequence.shape[1])
            merged = Paths(
                receiver=onset.receiver,
                image=np.concatenate([onset.image, found.image[keep]]),
                order=np.concatenate([onset.order, found.order[keep]]),
                length_m=np.concatenate([onset.length_m, total]),
                direction=np.vstack([onset.direction, found.direction[keep]]),
                gain=np.vstack([onset.gain, found.gain[keep] * scale]),
                points=np.concatenate(
                    [_widen(onset.points, width), _widen(reflected_points, width)], axis=0
                ),
                sequence=np.concatenate(
                    [_pad(onset.sequence, columns), _pad(found.sequence[keep], columns)]
                ),
            )
            keep_n = min(settings.max_paths, merged.length_m.size)
            pick = np.argsort(merged.length_m)[:keep_n]
            out[index] = Paths(
                receiver=merged.receiver,
                image=merged.image[pick],
                order=merged.order[pick],
                length_m=merged.length_m[pick],
                direction=merged.direction[pick],
                gain=merged.gain[pick],
                points=merged.points[pick],
                sequence=merged.sequence[pick],
            )
            reflected += int(keep_n) - 1
    return out, len(taken), reflected


def _widen(points: np.ndarray, width: int) -> np.ndarray:
    """``[path, point, 3]`` padded to ``width`` points by repeating the last one.

    A path's points end at the receiver, so repeating the last one adds legs
    of zero length and changes nothing the renderer reads.
    """
    if points.shape[1] >= width:
        return points
    extra = np.repeat(points[:, -1:, :], width - points.shape[1], axis=1)
    return np.concatenate([points, extra], axis=1)


def _pad(sequence: np.ndarray, columns: int) -> np.ndarray:
    """``[path, order]`` padded with -1, the empty facet."""
    if sequence.shape[1] >= columns:
        return sequence
    fill = np.full((sequence.shape[0], columns - sequence.shape[1]), -1, dtype=sequence.dtype)
    return np.concatenate([sequence, fill], axis=1)


def _shift_points(points: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """The path's own points with the edge written where the source would be."""
    out = np.array(points, dtype=float, copy=True)
    out[:, 0] = edge
    return out
