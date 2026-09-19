"""The edges a shadowed receiver hears the source round, and the paths through them.

Diffracting edges are the rims of the reflecting facets, merged into lines,
kept when long enough and hard enough to bend the band the mirror answers.
For a receiver, each edge's aperture point is where the path from source to
receiver over the edge is shortest; the geodesic's corners within
:data:`SNAP_M` of an edge are pulled onto it; and the corner the sound bends
round is given its own image tree in the receiver's room.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import Paths
from reverberate.mirror.occupancy import DiffractionSettings, maekawa_db
from reverberate.mirror.rays import UniformGrid

__all__ = ["Edges", "diffracting_edges", "edge_paths", "snap_to_edges"]

#: Shortest edge kept as a diffracting one, m.
MIN_EDGE_M = 0.30
#: An edge whose faces absorb more than this at 1 kHz is left out.
MAX_EDGE_ABSORPTION = 0.60
#: How far a geodesic corner may be from an edge and still be taken as it, m.
SNAP_M = 0.30
#: Corners closer together than this share one image tree, m.
CLUSTER_M = 0.25
#: Paths kept per receiver, the shortest first.
MAX_PATHS = 24


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
    is made of many short collinear pieces, so the touching pieces on one
    line are taken together; a gap along the line, such as a doorway, ends
    one edge and starts the next.
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
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for members in groups.values():
        axis = unit[members[0]]
        origin = foot[members[0]]
        # Pieces that touch make one edge; a gap (a doorway) starts another.
        start = np.minimum(p[members] @ axis, q[members] @ axis)
        stop = np.maximum(p[members] @ axis, q[members] @ axis)
        order = np.argsort(start)
        low, high = float(start[order[0]]), float(stop[order[0]])
        runs = []
        for k in order[1:]:
            if start[k] <= high + 1e-3:
                high = max(high, float(stop[k]))
            else:
                runs.append((low, high))
                low, high = float(start[k]), float(stop[k])
        runs.append((low, high))
        out.extend((origin + a * axis, origin + b * axis) for a, b in runs if b - a >= min_length_m)
    return out


def diffracting_edges(
    scene: DerivedScene,
    *,
    min_length_m: float = MIN_EDGE_M,
    max_absorption: float = MAX_EDGE_ABSORPTION,
) -> Edges:
    """The edges worth bending sound round: rims of reflector facets, by length and material.

    An edge is kept when it is at least ``min_length_m`` long (shorter ones
    bend too little of the mirror's band) and its facet absorbs at most
    ``max_absorption`` at 1 kHz (what a curtain's hem bends, the curtain has
    already taken).
    """
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    at_1k = int(np.argmin(np.abs(bands - 1000.0)))
    ends_a: list[np.ndarray] = []
    ends_b: list[np.ndarray] = []
    labels: list[int] = []
    facets: list[int] = []
    for index, facet in enumerate(scene.facets):
        absorption = float(scene.materials.absorption[facet.label][at_1k])
        if absorption > max_absorption:
            continue
        for low, high in _lines_of(scene.reflector_vertices[facet.triangles], min_length_m):
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
    corners: list[np.ndarray], edges: Edges, within_m: float = SNAP_M, sweeps: int = 3
) -> tuple[list[np.ndarray], list[bool]]:
    """The geodesic's corners pulled onto the edges they stand near, then tightened.

    A corner from the grid sits at a cell centre, so the way round is as
    long as the grid is coarse and arrives from a direction the grid chose.
    Each interior corner is matched to the nearest selected edge, within
    ``within_m``; a corner with no edge near it is left where it is, since the
    grid may have gone round an occluder that carries no reflector facet at
    all. What is matched is then swept back and forth, each point taking the
    place on its own edge that shortens its two legs, which is the shortest
    way round that sequence of edges.

    Returns the corners and, per corner, whether it ended on an edge. The
    caller needs the second: tightening the chain takes the kink out of a
    bend, and a bend whose kink is gone is still a bend if it stands on an
    edge.
    """
    if edges.count == 0 or len(corners) < 3:
        return corners, [False] * len(corners)
    held: list[int | None] = [None] * len(corners)
    out = [np.asarray(c, dtype=float) for c in corners]
    for k in range(1, len(corners) - 1):
        feet = _nearest_on_segment(out[k], edges.a, edges.b)
        gaps = np.linalg.norm(feet - out[k], axis=1)
        best = int(np.argmin(gaps))
        if float(gaps[best]) <= within_m:
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


def _through_the_edges(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    out: dict[int, Paths],
    secondary: dict[int, tuple[np.ndarray, float, np.ndarray]],
    sound_speed_m_s: float,
    settings: DiffractionSettings,
    edges: Edges,
    grid: UniformGrid,
) -> tuple[dict[int, Paths], dict[int, tuple[np.ndarray, float, np.ndarray]], dict[str, Any]]:
    """Every selected edge tried for every shadowed receiver, beside the geodesic.

    The geodesic gives one way round and one direction; a wave's first sound
    arrives round several edges at once. Each edge that both the source and
    the receiver can see contributes its own arrival, time and direction. The
    geodesic is kept only when it is shorter than every single edge path: a
    way round two corners that no one edge gives.
    """
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
        keep = np.argsort(lengths)[:MAX_PATHS]
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
        # edges and adding them counts the same sound several times.
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
    grid: UniformGrid,
) -> tuple[dict[int, Paths], int, int]:
    """Give each diffracted onset the reflections of its own edge in the receiver's room.

    The corner the sound last bent round is a point source standing in the
    doorway. Its image tree, grown to ``settings.reflections``, gives the
    receiver the floor, the ceiling and the near walls of the room it is
    actually in, which the source's own tree cannot reach at all. Every
    such path carries the edges' loss and spreads over the whole way, the
    leg before the edge included, so the onset itself comes back unchanged.

    Edges within ``CLUSTER_M`` of each other share one tree: the points in
    shadow stand behind a few doorways, not hundreds of edges.
    """
    from reverberate.mirror.ism import IsmSettings, grow_tree, paths_for

    ism = IsmSettings(max_order=settings.reflections, flutter_order=0)
    order = sorted(secondary)
    taken: list[np.ndarray] = []
    groups: list[list[int]] = []
    for index in order:
        point = secondary[index][0]
        for k, centre in enumerate(taken):
            if float(np.linalg.norm(point - centre)) <= CLUSTER_M:
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
            keep_n = min(MAX_PATHS, merged.length_m.size)
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
            # The onset's own paths come first in ``merged``; the rest are reflections.
            reflected += int(np.count_nonzero(pick >= onset.count))
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
