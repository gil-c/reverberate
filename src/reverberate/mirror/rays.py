"""Stochastic rays over the derived surfaces, into directional histograms per receiver.

The tail of the mirror. Rays leave the source in directions drawn from a
counter based generator keyed on the ray's own index, so a ray is the same
whatever the batch or the card that traces it, carry one energy per octave
band, lose ``alpha`` of it at every surface, bounce specularly or, with the
surface's scattering probability, in a Lambert direction about the normal,
and die when their time runs out. On the way each ray reports to every
receiver sphere it crosses: the energy it carries lands in the receiver's
histogram at the arrival time, per band, and on the low order harmonics of
the arrival direction, so the tail keeps its anisotropy.

The direct ray hits are in the histogram too, and they are its scale: the
renderer sets the tail so that the histogram's direct energy equals the
rendered direct pulse's, band by band, which needs no detector theory.

The surfaces are looked up through a uniform grid of cells, each holding the
triangles whose box meets it, walked cell by cell along a segment. The same
grid serves the image source model's occlusion tests. The twin here traces
one ray at a time in Python and exists to test the kernel and to run the
small scenes of the test suite; the storey runs on the card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.mirror.geometry import DerivedScene
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic

__all__ = [
    "Histogram",
    "RaySettings",
    "UniformGrid",
    "build_grid",
    "philox_directions",
    "trace",
]


@dataclass(frozen=True)
class RaySettings:
    """What is a choice in the tracer."""

    rays: int = 100_000
    duration_s: float = 1.2
    #: Histogram time bin.
    bin_s: float = 0.002
    #: Order of the directional moments kept per bin.
    order: int = 3
    #: Radius of every receiver sphere.
    receiver_radius_m: float = 0.20
    sound_speed_m_s: float = 343.2
    #: A ray whose energy in every band falls under this fraction of its
    #: start is dropped.
    energy_floor: float = 1e-6
    #: Cell edge of the uniform grid.
    cell_m: float = 0.25
    seed: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "rays": self.rays,
            "duration_s": self.duration_s,
            "bin_s": self.bin_s,
            "order": self.order,
            "receiver_radius_m": self.receiver_radius_m,
            "sound_speed_m_s": self.sound_speed_m_s,
            "energy_floor": self.energy_floor,
            "cell_m": self.cell_m,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Histogram:
    """Energy per receiver, time bin and band, and its directional moments."""

    #: ``[receiver, bin, band]`` energy, in units of the source's total energy.
    energy: np.ndarray
    #: ``[receiver, bin, band, channel]`` energy weighted harmonics of the
    #: arrival directions, ACN and N3D, in the ambisonic frame.
    moments: np.ndarray
    #: ``[receiver, bin]`` how many ray crossings the bin holds, all bands.
    hits: np.ndarray
    bin_s: float
    bands_hz: tuple[int, ...]
    order: int
    rays: int

    @property
    def times_s(self) -> np.ndarray:
        return np.asarray((np.arange(self.energy.shape[1]) + 0.5) * self.bin_s)


# --------------------------------------------------------------------------
# the uniform grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformGrid:
    """Triangles binned by the cells their boxes meet, in compressed rows."""

    origin: np.ndarray
    cell_m: float
    shape: tuple[int, int, int]
    #: ``[cells + 1]`` offsets into ``members``.
    offsets: np.ndarray
    members: np.ndarray

    def cell_of(self, point: np.ndarray) -> np.ndarray:
        index = np.floor((point - self.origin) / self.cell_m).astype(int)
        return np.asarray(np.clip(index, 0, np.asarray(self.shape) - 1))

    def flat(self, index: np.ndarray) -> np.ndarray:
        return np.asarray(
            (index[..., 0] * self.shape[1] + index[..., 1]) * self.shape[2] + index[..., 2]
        )

    def triangles_in(self, flat: int) -> np.ndarray:
        return np.asarray(self.members[self.offsets[flat] : self.offsets[flat + 1]])

    def cells_along(self, a: np.ndarray, b: np.ndarray) -> list[int]:
        """The flat indices of the cells a segment passes through, in order (3D DDA)."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        direction = b - a
        length = float(np.linalg.norm(direction))
        cell = self.cell_of(a)
        last = self.cell_of(b)
        out = [int(self.flat(cell))]
        if length == 0.0:
            return out
        step = np.where(direction > 0, 1, np.where(direction < 0, -1, 0))
        # Distance along the segment to the next cell boundary on each axis.
        boundary = self.origin + (cell + np.where(step > 0, 1, 0)) * self.cell_m
        with np.errstate(divide="ignore", invalid="ignore"):
            t_max = np.where(step != 0, (boundary - a) / direction, np.inf)
            t_delta = np.where(step != 0, self.cell_m / np.abs(direction), np.inf)
        for _ in range(sum(self.shape) * 3):
            if np.array_equal(cell, last):
                break
            axis = int(np.argmin(t_max))
            if t_max[axis] > 1.0:
                break
            cell = cell.copy()
            cell[axis] += step[axis]
            if not (0 <= cell[axis] < self.shape[axis]):
                break
            t_max[axis] += t_delta[axis]
            out.append(int(self.flat(cell)))
        return out

    def candidates(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The union of the triangles in every cell a segment crosses."""
        cells = self.cells_along(a, b)
        if not cells:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate([self.triangles_in(c) for c in cells]))


def build_grid(triangles: np.ndarray, cell_m: float, lo: np.ndarray, hi: np.ndarray) -> UniformGrid:
    """Bin ``triangles`` (``[n, 3, 3]``) into cells of ``cell_m`` covering ``lo`` to ``hi``."""
    lo = np.asarray(lo, dtype=float) - cell_m
    hi = np.asarray(hi, dtype=float) + cell_m
    counts_per_axis = np.maximum(np.ceil((hi - lo) / cell_m), 1).astype(int)
    shape = (int(counts_per_axis[0]), int(counts_per_axis[1]), int(counts_per_axis[2]))
    if triangles.shape[0] == 0:
        return UniformGrid(
            lo, cell_m, shape, np.zeros(int(np.prod(shape)) + 1, dtype=int), np.zeros(0, dtype=int)
        )
    tmin = np.floor((triangles.min(axis=1) - lo) / cell_m).astype(int)
    tmax = np.floor((triangles.max(axis=1) - lo) / cell_m).astype(int)
    limit = np.asarray(shape) - 1
    tmin = np.clip(tmin, 0, limit)
    tmax = np.clip(tmax, 0, limit)
    spans = tmax - tmin + 1
    counts = np.prod(spans, axis=1)
    total = int(counts.sum())
    cell_ids = np.empty(total, dtype=int)
    tri_ids = np.empty(total, dtype=int)
    cursor = 0
    # Most triangles fit one cell; the loop is over the few that span many.
    for t in np.argsort(counts, kind="stable"):
        n = int(counts[t])
        if n == 1:
            cell_ids[cursor] = (tmin[t, 0] * shape[1] + tmin[t, 1]) * shape[2] + tmin[t, 2]
            tri_ids[cursor] = t
            cursor += 1
            continue
        xs = np.arange(tmin[t, 0], tmax[t, 0] + 1)
        ys = np.arange(tmin[t, 1], tmax[t, 1] + 1)
        zs = np.arange(tmin[t, 2], tmax[t, 2] + 1)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        cell_ids[cursor : cursor + n] = ((grid_x * shape[1] + grid_y) * shape[2] + grid_z).ravel()
        tri_ids[cursor : cursor + n] = t
        cursor += n
    order = np.lexsort((tri_ids, cell_ids))
    cell_ids = cell_ids[order]
    tri_ids = tri_ids[order]
    offsets = np.zeros(int(np.prod(shape)) + 1, dtype=int)
    np.add.at(offsets, cell_ids + 1, 1)
    offsets = np.cumsum(offsets)
    return UniformGrid(lo, cell_m, shape, offsets, tri_ids)


# --------------------------------------------------------------------------
# the rays
# --------------------------------------------------------------------------


def philox_directions(count: int, seed: int, start: int = 0) -> np.ndarray:
    """Unit vectors for rays ``start`` to ``start + count``, each from its own counter.

    numpy's Philox with the ray index as the key: ray ``i`` draws the same
    two uniforms whichever batch it is traced in.
    """
    out = np.empty((count, 3))
    for i in range(count):
        gen = np.random.Generator(np.random.Philox(key=seed + 0, counter=[start + i, 0, 0, 0]))
        u, v = gen.random(2)
        z = 1.0 - 2.0 * u
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * np.pi * v
        out[i] = (r * np.cos(phi), r * np.sin(phi), z)
    return out


def _ray_uniforms(seed: int, ray: int, bounce: int, count: int) -> np.ndarray:
    """The uniforms of one bounce of one ray, from the same counter family."""
    gen = np.random.Generator(np.random.Philox(key=seed, counter=[ray, bounce + 1, 0, 0]))
    return np.asarray(gen.random(count))


def _nearest_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    candidates: np.ndarray,
    exclude: int,
) -> tuple[float, int]:
    """Distance and index of the first triangle the ray hits among ``candidates``."""
    if candidates.size == 0:
        return np.inf, -1
    tri = triangles[candidates]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    p = np.cross(direction[None, :], e2)
    det = np.einsum("tk,tk->t", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    t_vec = origin[None, :] - v0
    u = np.einsum("tk,tk->t", t_vec, p) * inv
    q = np.cross(t_vec, e1)
    v = np.einsum("k,tk->t", direction, q) * inv
    t = np.einsum("tk,tk->t", e2, q) * inv
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
    hit &= candidates != exclude
    if not hit.any():
        return np.inf, -1
    best = int(np.argmin(np.where(hit, t, np.inf)))
    return float(t[best]), int(candidates[best])


def _lambert(normal: np.ndarray, u: float, v: float) -> np.ndarray:
    """A direction about ``normal`` with a cosine distribution."""
    z = np.sqrt(u)
    r = np.sqrt(max(0.0, 1.0 - u))
    phi = 2.0 * np.pi * v
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return np.asarray(r * np.cos(phi) * t1 + r * np.sin(phi) * t2 + z * normal)


def _sphere_crossings(
    origin: np.ndarray, direction: np.ndarray, length: float, centres: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Receivers whose sphere the segment enters, and the distance along it to the entry."""
    rel = centres - origin[None, :]
    along = rel @ direction
    perp2 = np.sum(rel**2, axis=1) - along**2
    inside_r = perp2 <= radius**2
    half = np.sqrt(np.maximum(radius**2 - perp2, 0.0))
    entry = along - half
    # Entered along this segment: the entry lies on it, or the origin is
    # already inside the sphere (then the crossing counts at the origin).
    starts_inside = np.sum(rel**2, axis=1) <= radius**2
    entry = np.where(starts_inside, 0.0, entry)
    crossed = inside_r & (entry >= 0.0) & (entry <= length) & ~(starts_inside & (along < 0))
    return np.flatnonzero(crossed), entry[crossed]


def trace(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings | None = None,
    *,
    grid: UniformGrid | None = None,
) -> Histogram:
    """The twin: every ray in turn, in Python. For tests and small scenes."""
    settings = settings or RaySettings()
    source = np.asarray(source, dtype=float).reshape(3)
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    triangles = scene.occluder_vertices
    labels = scene.occluder_label
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    absorption = scene.materials.absorption  # [label, band]
    scattering = scene.materials.scattering
    bands = absorption.shape[1]
    if grid is None:
        pad = settings.sound_speed_m_s * settings.duration_s
        lo = np.minimum(scene.bmin, np.minimum(receivers.min(axis=0), source)) - 0.0
        hi = np.maximum(scene.bmax, np.maximum(receivers.max(axis=0), source)) + 0.0
        del pad
        grid = build_grid(triangles, settings.cell_m, lo, hi)
    reach = settings.sound_speed_m_s * settings.duration_s
    bins = int(np.ceil(settings.duration_s / settings.bin_s))
    channels = channel_count(settings.order)
    energy = np.zeros((receivers.shape[0], bins, bands))
    moments = np.zeros((receivers.shape[0], bins, bands, channels))
    hits = np.zeros((receivers.shape[0], bins), dtype=np.int64)
    directions = philox_directions(settings.rays, settings.seed)
    for ray in range(settings.rays):
        position = source.copy()
        direction = directions[ray]
        carried = np.full(bands, 1.0 / settings.rays)
        travelled = 0.0
        last_triangle = -1
        for bounce in range(100_000):
            candidates = grid.candidates(position, position + direction * (reach - travelled))
            distance, triangle = _nearest_hit(
                position, direction, triangles, candidates, last_triangle
            )
            segment = min(distance, reach - travelled)
            which, entries = _sphere_crossings(
                position, direction, segment, receivers, settings.receiver_radius_m
            )
            if which.size:
                arrival = scene_to_ambisonic((-direction)[None, :])
                harmonics = real_sh(settings.order, arrival)[0]
                for r, entry in zip(which, entries, strict=True):
                    at = int((travelled + entry) / settings.sound_speed_m_s / settings.bin_s)
                    if at < bins:
                        energy[r, at] += carried
                        moments[r, at] += carried[:, None] * harmonics[None, :]
                        hits[r, at] += 1
            if triangle < 0 or travelled + distance >= reach:
                break
            travelled += distance
            position = position + direction * distance
            label = int(labels[triangle])
            carried = carried * (1.0 - absorption[label])
            if np.all(carried < settings.energy_floor / settings.rays):
                break
            normal = normals[triangle]
            if float(normal @ direction) > 0.0:
                normal = -normal
            u, v, w = _ray_uniforms(settings.seed, ray, bounce, 3)
            if w < scattering[label]:
                direction = _lambert(normal, u, v)
            else:
                direction = direction - 2.0 * float(direction @ normal) * normal
            last_triangle = triangle
    return Histogram(
        energy=energy,
        moments=moments,
        hits=hits,
        bin_s=settings.bin_s,
        bands_hz=scene.materials.bands_hz,
        order=settings.order,
        rays=settings.rays,
    )
