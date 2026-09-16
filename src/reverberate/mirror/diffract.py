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

__all__ = ["DiffractionSettings", "Occupancy", "diffracted_paths", "maekawa_db", "occupancy_of"]


@dataclass(frozen=True)
class DiffractionSettings:
    """What the diffracted onset chooses."""

    cell_m: float = 0.10
    #: Cells the occupancy is grown by, so thin walls stay closed.
    grow_cells: int = 1
    #: Least samples per triangle edge when the occluders are rasterised.
    samples_per_edge: int = 4
    #: Cap of one edge's loss, dB (Maekawa's own practical limit).
    max_loss_db: float = 24.0
    #: Cells kept free round the source and each receiver.
    clear_cells: int = 1
    #: A pulled corner adding less detour than this is the grid's, not an edge.
    min_detour_m: float = 0.05

    def record(self) -> dict[str, Any]:
        return {
            "cell_m": self.cell_m,
            "grow_cells": self.grow_cells,
            "samples_per_edge": self.samples_per_edge,
            "max_loss_db": self.max_loss_db,
            "clear_cells": self.clear_cells,
            "min_detour_m": self.min_detour_m,
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
    detour_m: float, bands_hz: np.ndarray, sound_speed_m_s: float, cap_db: float
) -> np.ndarray:
    """Maekawa's insertion loss of one edge per band, for a detour of ``detour_m``."""
    fresnel = 2.0 * max(detour_m, 0.0) * np.asarray(bands_hz, dtype=float) / sound_speed_m_s
    return np.asarray(np.minimum(10.0 * np.log10(3.0 + 20.0 * fresnel), cap_db))


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
    graph = _graph(occupancy)
    start = int(occupancy.flat(occupancy.cell_of(source))[0])
    distance, predecessor = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    shape = occupancy.blocked.shape
    out: dict[int, Paths] = {}
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
        legs = [
            float(np.linalg.norm(b - a)) for a, b in zip(corners[:-1], corners[1:], strict=True)
        ]
        length = float(sum(legs))
        loss = np.zeros(len(bands))
        edges = 0
        for k in range(1, len(corners) - 1):
            detour = legs[k - 1] + legs[k] - float(np.linalg.norm(corners[k + 1] - corners[k - 1]))
            if detour < settings.min_detour_m:
                # A corner the grid made, not an edge the sound bends round.
                continue
            edges += 1
            loss += maekawa_db(detour, bands, sound_speed_m_s, settings.max_loss_db)
        if edges == 0:
            # Pulled straight within the grid's slack: the shadow boundary's 4.8 dB.
            loss += maekawa_db(0.0, bands, sound_speed_m_s, settings.max_loss_db)
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
        lengths.append(length - float(np.linalg.norm(receiver - source)))
    record = {
        "settings": settings.record(),
        "grid": list(shape),
        "blocked_fraction": round(float(occupancy.blocked.mean()), 4),
        "asked": len(indices),
        "found": len(out),
        "detour_m_median": round(float(np.median(lengths)), 3) if lengths else None,
    }
    if say is not None:
        say(f"diffraction: {len(out)} of {len(indices)} points, grid {shape}")
    return out, record
