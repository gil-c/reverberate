"""The occluders as a grid of cells, the shortest way round them, and the loss of each bend.

The storey's occluders are sampled into a boolean grid, grown by one cell so
a one cell wall cannot leak along a diagonal; the geodesic from the source is
Dijkstra's on the 26-connected free cells; a receiver's chain back to the
source is pulled tight, the cells it keeps being the bends. Each bend loses
Maekawa's ``10 log10(3 + 20 N)`` dB per band, ``N = 2 delta f / c``, capped at 24 dB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix

from reverberate.mirror.geometry import DerivedScene

__all__ = ["DiffractionSettings", "Occupancy", "maekawa_db", "occupancy_of"]

#: Cells the occupancy is grown by, so thin walls stay closed.
GROW_CELLS = 1
#: Least samples per triangle edge when the occluders are rasterised.
SAMPLES_PER_EDGE = 4
#: Cells kept free round the source and each receiver.
CLEAR_CELLS = 1
#: A pulled corner adding less detour than this is the grid's, not a bend, m.
MIN_DETOUR_M = 0.05


@dataclass(frozen=True)
class DiffractionSettings:
    """What the diffracted onset chooses."""

    #: Occupancy grid cell, m. Finer grids leak through thin walls.
    cell_m: float = 0.10
    #: Reflection order of the corner's own image tree: the corner the sound
    #: bends round is a source in the receiver's room, and that room reflects
    #: it. Zero keeps the onset alone.
    reflections: int = 1
    #: Whether every selected edge is tried beside the geodesic.
    edges: bool = True
    #: How much longer than the straight line a single edge path may be, m.
    #: Long ways round spread the early part in time.
    max_edge_detour_m: float = 1.5

    def record(self) -> dict[str, Any]:
        return asdict(self)


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
    points = _surface_samples(triangles, settings.cell_m, SAMPLES_PER_EDGE)
    index = np.floor((points - lo) / settings.cell_m).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(shape)), axis=1)
    index = index[inside]
    blocked[index[:, 0], index[:, 1], index[:, 2]] = True
    if GROW_CELLS > 0:
        blocked = ndimage.binary_dilation(
            blocked, structure=np.ones((3, 3, 3), dtype=bool), iterations=GROW_CELLS
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

    Capped at ``cap_db``, Maekawa's own practical limit of 24 dB.
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
