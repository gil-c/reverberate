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

**Everything here is written so the card can reproduce it to the bit.** The
generator is an integer hash of (seed, ray, bounce, draw); directions come
from rejection sampling with nothing but products, sums and square roots;
and the histogram accumulates in 64 bit integers at a fixed scale, so the
order in which rays add to a bin does not change the sum. The surfaces and
the receivers are looked up through a uniform grid of cells walked cell by
cell along a segment. The twin here traces one ray at a time in Python and
exists to test the kernel and to run the small scenes of the test suite;
the storey runs on the card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.mirror.geometry import DerivedScene
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic

__all__ = [
    "HISTOGRAM_SCALE",
    "Histogram",
    "RaySettings",
    "UniformGrid",
    "build_grid",
    "harmonics3",
    "hash_uniform",
    "ray_directions",
    "trace",
    "triangle_grid",
]

#: Energies are accumulated as integers of this scale: 2^40 per unit of the
#: source's energy, so a ray of one millionth carries a million counts.
HISTOGRAM_SCALE = float(2**40)


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
    #: A ray whose bounces so far are all specular, on reflector facets, at
    #: most this many, is what the image tree renders already: it is not
    #: counted when it crosses a receiver. Zero counts everything.
    skip_specular_order: int = 0
    #: ... and with at most this many of those bounces on furniture, as the
    #: tree's ``furniture_bounces`` rule.
    skip_furniture_bounces: int = 1
    #: ... and arriving within this many seconds, the image tree's window;
    #: zero skips them whenever they arrive.
    skip_window_s: float = 0.0

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
            "skip_specular_order": self.skip_specular_order,
            "skip_furniture_bounces": self.skip_furniture_bounces,
            "skip_window_s": self.skip_window_s,
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

    @classmethod
    def from_counts(
        cls,
        energy_counts: np.ndarray,
        moment_counts: np.ndarray,
        hits: np.ndarray,
        *,
        bin_s: float,
        bands_hz: tuple[int, ...],
        order: int,
        rays: int,
    ) -> Histogram:
        """The integer accumulators back to energies."""
        return cls(
            energy=np.asarray(energy_counts, dtype=np.float64) / HISTOGRAM_SCALE,
            moments=np.asarray(moment_counts, dtype=np.float64) / HISTOGRAM_SCALE,
            hits=np.asarray(hits, dtype=np.int64),
            bin_s=bin_s,
            bands_hz=bands_hz,
            order=order,
            rays=rays,
        )


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------

_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_BOUNCE = np.uint64(0xD6E8FEB86659FD93)
_DRAW = np.uint64(0xA0761D6478BD642F)


def _mix(x: np.ndarray) -> np.ndarray:
    """MurmurHash3's 64 bit finaliser, on unsigned arrays that wrap."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(33))
        x = x * _M1
        x = x ^ (x >> np.uint64(33))
        x = x * _M2
        x = x ^ (x >> np.uint64(33))
    return np.asarray(x, dtype=np.uint64)


def hash_uniform(seed: int, ray: np.ndarray, bounce: np.ndarray, draw: np.ndarray) -> np.ndarray:
    """Uniforms in ``[0, 1)`` for (ray, bounce, draw) triples, the same on every machine.

    53 bits of the hash over ``2^53``, so the value is exactly representable
    and the card's integer arithmetic gives the same double.
    """
    ray_u = np.asarray(ray, dtype=np.uint64)
    bounce_u = np.asarray(bounce, dtype=np.uint64)
    draw_u = np.asarray(draw, dtype=np.uint64)
    with np.errstate(over="ignore"):
        key = _mix(np.uint64(seed) * _GOLDEN + ray_u)
        state = _mix(key ^ (bounce_u * _BOUNCE) ^ (draw_u * _DRAW))
    return np.asarray((state >> np.uint64(11)).astype(np.float64) / float(2**53))


def ray_directions(count: int, seed: int, start: int = 0, *, bounce: int = 0) -> np.ndarray:
    """Unit vectors for rays ``start`` to ``start + count``, by rejection in the cube.

    Draws 0, 1, 2 of the ray's stream are one candidate; a candidate outside
    the unit ball, or too close to its centre to normalise, moves to the
    next three draws. Nothing but products, sums and one square root.
    """
    rays = np.arange(start, start + count, dtype=np.uint64)
    out = np.zeros((count, 3))
    pending = np.ones(count, dtype=bool)
    for attempt in range(64):
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        base = 3 * attempt
        bounces = np.full(idx.size, bounce)
        u = hash_uniform(seed, rays[idx], bounces, np.full(idx.size, base))
        v = hash_uniform(seed, rays[idx], bounces, np.full(idx.size, base + 1))
        w = hash_uniform(seed, rays[idx], bounces, np.full(idx.size, base + 2))
        p = np.stack([2.0 * u - 1.0, 2.0 * v - 1.0, 2.0 * w - 1.0], axis=1)
        norm2 = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1] + p[:, 2] * p[:, 2]
        ok = (norm2 <= 1.0) & (norm2 > 1e-8)
        out[idx[ok]] = p[ok] / np.sqrt(norm2[ok])[:, None]
        pending[idx[ok]] = False
    return out


def _lambert(normal: np.ndarray, seed: int, ray: int, bounce: int) -> np.ndarray:
    """A direction about ``normal`` with a cosine distribution: a point of the disk, lifted.

    Draws 1, 2, 3, 4 ... of the bounce (draw 0 decides the bounce's kind); a
    point outside the disk moves to the next two draws.
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 = t1 / np.sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2])
    t2 = np.cross(normal, t1)
    one = np.array([ray])
    at = np.array([bounce])
    for attempt in range(64):
        base = 1 + 2 * attempt
        a = 2.0 * float(hash_uniform(seed, one, at, np.array([base]))[0]) - 1.0
        b = 2.0 * float(hash_uniform(seed, one, at, np.array([base + 1]))[0]) - 1.0
        r2 = a * a + b * b
        if r2 <= 1.0:
            z = np.sqrt(1.0 - r2)
            return np.asarray(a * t1 + b * t2 + z * normal)
    return np.asarray(normal)


# --------------------------------------------------------------------------
# the uniform grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformGrid:
    """Items binned by the cells their boxes meet, in compressed rows."""

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

    def members_in(self, flat: int) -> np.ndarray:
        return np.asarray(self.members[self.offsets[flat] : self.offsets[flat + 1]])

    def triangles_in(self, flat: int) -> np.ndarray:
        return self.members_in(flat)

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
        """The union of the members of every cell a segment crosses."""
        cells = self.cells_along(a, b)
        if not cells:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate([self.members_in(c) for c in cells]))


def build_grid(
    boxes_min: np.ndarray, boxes_max: np.ndarray, cell_m: float, lo: np.ndarray, hi: np.ndarray
) -> UniformGrid:
    """Bin items by their boxes (``[n, 3]`` each) into cells of ``cell_m`` from ``lo`` to ``hi``."""
    lo = np.asarray(lo, dtype=float) - cell_m
    hi = np.asarray(hi, dtype=float) + cell_m
    counts_per_axis = np.maximum(np.ceil((hi - lo) / cell_m), 1).astype(int)
    shape = (int(counts_per_axis[0]), int(counts_per_axis[1]), int(counts_per_axis[2]))
    cells = int(np.prod(shape))
    if boxes_min.shape[0] == 0:
        return UniformGrid(
            lo, cell_m, shape, np.zeros(cells + 1, dtype=int), np.zeros(0, dtype=int)
        )
    tmin = np.floor((boxes_min - lo) / cell_m).astype(int)
    tmax = np.floor((boxes_max - lo) / cell_m).astype(int)
    limit = np.asarray(shape) - 1
    tmin = np.clip(tmin, 0, limit)
    tmax = np.clip(tmax, 0, limit)
    spans = tmax - tmin + 1
    counts = np.prod(spans, axis=1)
    total = int(counts.sum())
    cell_ids = np.empty(total, dtype=int)
    item_ids = np.empty(total, dtype=int)
    single = counts == 1
    n_single = int(single.sum())
    cursor = 0
    if n_single:
        t = np.flatnonzero(single)
        cell_ids[:n_single] = (tmin[t, 0] * shape[1] + tmin[t, 1]) * shape[2] + tmin[t, 2]
        item_ids[:n_single] = t
        cursor = n_single
    for item in np.flatnonzero(~single):
        n = int(counts[item])
        xs = np.arange(tmin[item, 0], tmax[item, 0] + 1)
        ys = np.arange(tmin[item, 1], tmax[item, 1] + 1)
        zs = np.arange(tmin[item, 2], tmax[item, 2] + 1)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        cell_ids[cursor : cursor + n] = ((grid_x * shape[1] + grid_y) * shape[2] + grid_z).ravel()
        item_ids[cursor : cursor + n] = int(item)
        cursor += n
    order = np.lexsort((item_ids, cell_ids))
    cell_ids = cell_ids[order]
    item_ids = item_ids[order]
    offsets = np.zeros(cells + 1, dtype=int)
    np.add.at(offsets, cell_ids + 1, 1)
    offsets = np.cumsum(offsets)
    return UniformGrid(lo, cell_m, shape, offsets, item_ids)


def grid_like(grid: UniformGrid, boxes_min: np.ndarray, boxes_max: np.ndarray) -> UniformGrid:
    """Other items binned into the cells of an existing grid: same origin, cell and shape."""
    cell = grid.cell_m
    lo = grid.origin + cell
    hi = grid.origin + cell * (np.asarray(grid.shape, dtype=float) - 1.0)
    other = build_grid(boxes_min, boxes_max, cell, lo, hi)
    if other.shape != grid.shape or not np.allclose(other.origin, grid.origin):
        raise RuntimeError(f"grid_like built {other.shape} at {other.origin}, not {grid.shape}")
    return other


def triangle_grid(
    triangles: np.ndarray, cell_m: float, lo: np.ndarray, hi: np.ndarray
) -> UniformGrid:
    """The grid of ``[n, 3, 3]`` triangles."""
    if triangles.shape[0] == 0:
        return build_grid(np.zeros((0, 3)), np.zeros((0, 3)), cell_m, lo, hi)
    return build_grid(triangles.min(axis=1), triangles.max(axis=1), cell_m, lo, hi)


# --------------------------------------------------------------------------
# the rays
# --------------------------------------------------------------------------


def harmonics3(direction: np.ndarray) -> np.ndarray:
    """N3D real harmonics to order 3 of one unit vector, ACN order, as the kernel evaluates them.

    The same polynomials in the same order of operations as the CUDA source,
    so a deposit rounds to the same integer on the card and on the host.
    Checked against :func:`reverberate.spatial.sh.real_sh` by the tests.
    """
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    s3 = np.sqrt(3.0)
    s5 = np.sqrt(5.0)
    s7 = np.sqrt(7.0)
    s15 = np.sqrt(15.0)
    out = np.empty(16)
    out[0] = 1.0
    out[1] = s3 * y
    out[2] = s3 * z
    out[3] = s3 * x
    out[4] = s15 * x * y
    out[5] = s15 * y * z
    out[6] = s5 * 0.5 * (3.0 * z * z - 1.0)
    out[7] = s15 * x * z
    out[8] = s15 * 0.5 * (x * x - y * y)
    out[9] = s7 * np.sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y)
    out[10] = s7 * np.sqrt(15.0) * x * y * z
    out[11] = s7 * np.sqrt(3.0 / 8.0) * y * (5.0 * z * z - 1.0)
    out[12] = s7 * 0.5 * z * (5.0 * z * z - 3.0)
    out[13] = s7 * np.sqrt(3.0 / 8.0) * x * (5.0 * z * z - 1.0)
    out[14] = s7 * np.sqrt(15.0) * 0.5 * z * (x * x - y * y)
    out[15] = s7 * np.sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y)
    return out


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


def reflector_of_triangles(
    scene: DerivedScene, *, planar_m: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """Per occluder triangle: the reflector facet it lies on (or -1), and whether it is furniture.

    A triangle belongs to a facet when it carries the facet's label, its
    centroid is within ``planar_m`` of the facet's plane and its normal is
    parallel to it. What the image tree can mirror on is exactly this set.
    """
    tris = scene.occluder_vertices.reshape(-1, 3, 3)
    if tris.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    centroids = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(norms, 1e-12)[:, None]
    labels = np.asarray(scene.occluder_label, dtype=int)
    reflector = np.full(tris.shape[0], -1, dtype=np.int32)
    furniture = np.zeros(tris.shape[0], dtype=np.int32)
    for k, facet in enumerate(scene.facets):
        n = np.asarray(facet.normal, dtype=float)
        on = (
            (labels == facet.label)
            & (np.abs(centroids @ n - facet.offset) <= planar_m)
            & (np.abs(normals @ n) >= 0.99)
        )
        reflector[on & (reflector < 0)] = k
        if facet.kind == "furniture":
            furniture[on] = 1
    return reflector, furniture


def trace(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings | None = None,
    *,
    grid: UniformGrid | None = None,
    ray_start: int = 0,
    ray_count: int | None = None,
) -> Histogram:
    """The twin: every ray in turn, in Python. For tests and small scenes.

    ``ray_start`` and ``ray_count`` trace a share of the rays, as a card
    does: each ray keeps its own index, stream and energy ``1 / rays``, so
    the shares' histograms sum to the whole one.
    """
    settings = settings or RaySettings()
    count = settings.rays - ray_start if ray_count is None else ray_count
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
        lo = np.minimum(scene.bmin, np.minimum(receivers.min(axis=0), source))
        hi = np.maximum(scene.bmax, np.maximum(receivers.max(axis=0), source))
        grid = triangle_grid(triangles, settings.cell_m, lo, hi)
    reach = settings.sound_speed_m_s * settings.duration_s
    bins = int(np.ceil(settings.duration_s / settings.bin_s))
    channels = channel_count(settings.order)
    energy = np.zeros((receivers.shape[0], bins, bands), dtype=np.int64)
    moments = np.zeros((receivers.shape[0], bins, bands, channels), dtype=np.int64)
    hits = np.zeros((receivers.shape[0], bins), dtype=np.int64)
    directions = ray_directions(count, settings.seed, start=ray_start)
    floor = settings.energy_floor / settings.rays
    one = np.array([0])
    tri_reflector, tri_furniture = reflector_of_triangles(scene)
    skip_reach = (
        settings.sound_speed_m_s * settings.skip_window_s if settings.skip_window_s > 0 else np.inf
    )
    for ray in range(ray_start, ray_start + count):
        position = source.copy()
        direction = directions[ray - ray_start]
        carried = np.full(bands, 1.0 / settings.rays)
        travelled = 0.0
        last_triangle = -1
        specular_only = True
        furniture_bounces = 0
        for bounce in range(100_000):
            candidates = grid.candidates(position, position + direction * (reach - travelled))
            distance, triangle = _nearest_hit(
                position, direction, triangles, candidates, last_triangle
            )
            segment = min(distance, reach - travelled)
            covered = (
                specular_only
                and 1 <= bounce <= settings.skip_specular_order
                and furniture_bounces <= settings.skip_furniture_bounces
            )
            which, entries = (
                (np.zeros(0, dtype=int), np.zeros(0))
                if covered
                else _sphere_crossings(
                    position, direction, segment, receivers, settings.receiver_radius_m
                )
            )
            if covered and skip_reach < np.inf:
                # A covered ray arriving after the tree's window is the rays' again.
                which, entries = _sphere_crossings(
                    position, direction, segment, receivers, settings.receiver_radius_m
                )
                keep = travelled + entries > skip_reach
                which, entries = which[keep], entries[keep]
            if which.size:
                arrival = scene_to_ambisonic((-direction)[None, :])[0]
                harmonics = (
                    harmonics3(arrival)[:channels]
                    if settings.order <= 3
                    else real_sh(settings.order, arrival[None, :])[0]
                )
                counts = np.rint(carried * HISTOGRAM_SCALE).astype(np.int64)
                weighted = np.rint(carried[:, None] * harmonics[None, :] * HISTOGRAM_SCALE).astype(
                    np.int64
                )
                for r, entry in zip(which, entries, strict=True):
                    at = int((travelled + entry) / settings.sound_speed_m_s / settings.bin_s)
                    if at < bins:
                        energy[r, at] += counts
                        moments[r, at] += weighted
                        hits[r, at] += 1
            if triangle < 0 or travelled + distance >= reach:
                break
            travelled += distance
            position = position + direction * distance
            label = int(labels[triangle])
            carried = carried * (1.0 - absorption[label])
            if np.all(carried < floor):
                break
            normal = normals[triangle]
            if float(normal @ direction) > 0.0:
                normal = -normal
            kind = float(
                hash_uniform(settings.seed, np.array([ray]), np.array([bounce + 1]), one)[0]
            )
            if tri_furniture[triangle]:
                furniture_bounces += 1
            if kind < scattering[label]:
                direction = _lambert(normal, settings.seed, ray, bounce + 1)
                specular_only = False
            else:
                direction = direction - 2.0 * float(direction @ normal) * normal
                if tri_reflector[triangle] < 0:
                    specular_only = False
            last_triangle = triangle
    return Histogram.from_counts(
        energy,
        moments,
        hits,
        bin_s=settings.bin_s,
        bands_hz=scene.materials.bands_hz,
        order=settings.order,
        rays=settings.rays,
    )
