"""The rays of the tail through a hierarchy of boxes: the twin, and one C text for host and card.

:mod:`reverberate.mirror.rays` is the tracer as it was specified: a ray at a
time in Python, its triangles looked up in a uniform grid. This module traces
the same rays through the tree of :mod:`reverberate.mirror.bvh`, three ways
that are held to one another:

- :func:`trace_tree`, **the twin**, in numpy and double precision: every ray
  of a batch advanced a segment at a time. It is :func:`rays.trace` with the
  grid replaced, and gives its histogram: the same crossings in the same
  bins, the energies to the count (the moments to a count or two of 2^40,
  where a direction differs in its last bit).
- :data:`CORE_SOURCE`, **the C text**, compiled by the host's compiler and
  called through ``ctypes`` (:func:`counts_on_host`), and by ``cupy`` for a
  card, a thread a ray (:data:`DEVICE_SOURCE`). In double precision it is the
  twin operation for operation, compiled without contraction on both.
- the same text with ``RT_SINGLE``: positions, directions and energies in
  single precision, the triangle test of Woop, Benthin and Wald (2013), which
  is watertight in any precision because two triangles compute the function
  of the edge they share from the same two products. Its histograms are not
  the double precision's count for count: a ray that grazes an edge may go
  to the other side. They are the same tail statistically, and that is what
  is proven of it (``docs/open-questions/ray-tracer.md``).

**The boxes are single precision in every mode and the answer does not
depend on them**: a box's test only adds candidates (:mod:`.bvh`), so "double"
means the hits of double precision arithmetic, found by a walk in single.

Nothing is installed: the shared object is built as
:mod:`reverberate.mirror.native` builds its own, under a name made of the
text, the flags and the compiler.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.bvh import BINS, LEAF, Bvh, build, nearest, triangle_boxes
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.native import FLAGS, _cache_root, _compiler
from reverberate.mirror.rays import (
    HISTOGRAM_SCALE,
    Histogram,
    RaySettings,
    hash_uniform,
    ray_directions,
    reflector_of_triangles,
)
from reverberate.mirror.shared import Store, name_of
from reverberate.spatial.sh import channel_count, scene_to_ambisonic

__all__ = [
    "CORE_SOURCE",
    "DEVICE_SOURCE",
    "HOST_SOURCE",
    "STATS",
    "STRUCTURE_VARIABLE",
    "TracerScene",
    "available",
    "counts_on_host",
    "first_hits",
    "device_source",
    "prepare",
    "receiver_tree",
    "structure",
    "trace_tree",
]

#: ``tree`` or ``grid``: what the rays look their triangles up in. The tree, since a card
#: showed its histograms the grid's (an RTX 3090 Ti, 2026-10-05: 20 sites of a storey,
#: ``docs/open-questions/ray-tracer.md``); the grid's kernel stays, eighteen times slower.
STRUCTURE_VARIABLE = "REVERBERATE_RAY_STRUCTURE"

#: What a trace counts beside its histograms, in the order the C text writes them.
STATS = (
    "segments",
    "nodes",
    "tests",
    "escapes",
    "receiver_nodes",
    "deposits",
    "ended_by_reach",
    "ended_by_floor",
)

#: The version of what :func:`prepare` keeps in a store: part of the entry's name.
_KEPT = 1


def structure(asked: str | None = None) -> str:
    """The structure in use: ``asked``, else :data:`STRUCTURE_VARIABLE`, else the tree."""
    named = asked or os.environ.get(STRUCTURE_VARIABLE) or "tree"
    if named not in ("grid", "tree"):
        raise ValueError(f"the rays' structure is 'grid' or 'tree', not {named!r}")
    return named


# --------------------------------------------------------------------------
# the scene as the rays read it
# --------------------------------------------------------------------------


@dataclass
class TracerScene:
    """A scene's triangles under their tree: what does not depend on materials or sources."""

    tree: Bvh
    #: ``[triangle, 9]`` double precision, in the tree's leaf order.
    triangles: np.ndarray
    #: ``[triangle, 2]`` in leaf order: the scene's index, then the label with bit 16 set
    #: for furniture and bit 17 for a triangle that lies on no reflector facet.
    meta: np.ndarray
    #: The middle of the scene: single precision coordinates are taken from it.
    centre: np.ndarray
    _kept: dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> np.ndarray:
        """The tree as the C text reads it, 64 bytes a node."""
        if "nodes" not in self._kept:
            self._kept["nodes"] = np.ascontiguousarray(self.tree.packed())
        found: np.ndarray = self._kept["nodes"]
        return found

    @property
    def single(self) -> np.ndarray:
        """``[triangle, 9]`` single precision, from the scene's middle, in leaf order."""
        if "single" not in self._kept:
            shifted = self.triangles.reshape(-1, 3, 3) - self.centre[None, None, :]
            self._kept["single"] = np.ascontiguousarray(shifted.reshape(-1, 9), dtype=np.float32)
        found: np.ndarray = self._kept["single"]
        return found

    @property
    def nodes_single(self) -> np.ndarray:
        """:attr:`nodes` with the boxes taken from the scene's middle, rounded outwards again."""
        if "nodes_single" not in self._kept:
            boxes = self.tree.boxes.astype(np.float64)
            moved = np.empty(boxes.shape, dtype=np.float32)
            down, up = np.float32(-np.inf), np.float32(np.inf)
            moved[..., :3] = np.nextafter((boxes[..., :3] - self.centre).astype(np.float32), down)
            moved[..., 3:] = np.nextafter((boxes[..., 3:] - self.centre).astype(np.float32), up)
            out = self.nodes.copy()
            out[:, :12] = moved.reshape(-1, 12)
            self._kept["nodes_single"] = out
        found: np.ndarray = self._kept["nodes_single"]
        return found

    def of_scene(self) -> dict[str, np.ndarray]:
        """Per triangle in the scene's own order: label, furniture, off every reflector."""
        if "of_scene" not in self._kept:
            index = self.meta[:, 0].astype(np.int64)
            word = self.meta[:, 1].astype(np.int64)
            count = int(index.size)
            label = np.zeros(count, dtype=np.int64)
            furniture = np.zeros(count, dtype=bool)
            off = np.zeros(count, dtype=bool)
            label[index] = word & 0xFFFF
            furniture[index] = (word >> 16) & 1 == 1
            off[index] = (word >> 17) & 1 == 1
            self._kept["of_scene"] = {"label": label, "furniture": furniture, "off": off}
        found: dict[str, np.ndarray] = self._kept["of_scene"]
        return found


def prepare(scene: DerivedScene, store: Store | None = None) -> TracerScene:
    """The scene's tree and its triangles in leaf order, built once and kept in ``store``.

    The entry's name holds the triangles, their labels and the facets they
    are held against, and no material: a calibration changes absorptions and
    finds the same tree.
    """
    triangles = np.ascontiguousarray(scene.occluder_vertices, dtype=np.float64).reshape(-1, 3, 3)
    labels = np.ascontiguousarray(scene.occluder_label, dtype=np.int64)

    def made() -> dict[str, Any]:
        lo, hi = triangle_boxes(triangles)
        tree = build(lo, hi)
        reflector, furniture = reflector_of_triangles(scene)
        word = (
            labels | (furniture.astype(np.int64) << 16) | ((reflector < 0).astype(np.int64) << 17)
        )
        order = tree.order.astype(np.int64)
        meta = np.stack([order, word[order]], axis=1).astype(np.int32)
        return {**tree.arrays(), "meta": meta}

    if labels.size and int(labels.max()) >= 1 << 16:
        raise ValueError("a triangle's label is written on 16 bits")
    if store is None:
        arrays = made()
    else:
        facets = [
            [int(f.label), f.kind, float(f.offset), *[float(v) for v in f.normal]]
            for f in scene.facets
        ]
        name = name_of("rays_tree", _KEPT, LEAF, BINS, triangles, labels, facets)
        arrays, _ = store.make("rays_tree", name, made)
    tree = Bvh.from_arrays(arrays)
    centre = 0.5 * (np.asarray(scene.bmin, dtype=float) + np.asarray(scene.bmax, dtype=float))
    if triangles.shape[0]:
        centre = 0.5 * (triangles.min(axis=(0, 1)) + triangles.max(axis=(0, 1)))
    return TracerScene(
        tree=tree,
        triangles=np.ascontiguousarray(triangles[np.asarray(tree.order)].reshape(-1, 9)),
        meta=np.ascontiguousarray(arrays["meta"], dtype=np.int32),
        centre=np.asarray(centre, dtype=np.float64),
    )


def receiver_tree(receivers: np.ndarray, radius_m: float) -> Bvh:
    """The tree of the receivers' spheres: which of them a segment can enter."""
    receivers = np.asarray(receivers, dtype=float).reshape(-1, 3)
    return build(receivers - radius_m, receivers + radius_m, leaf=2)


# --------------------------------------------------------------------------
# the twin
# --------------------------------------------------------------------------


def _harmonics3(direction: np.ndarray) -> np.ndarray:
    """:func:`rays.harmonics3` row by row, the same products in the same order."""
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    s3, s5, s7, s15 = np.sqrt(3.0), np.sqrt(5.0), np.sqrt(7.0), np.sqrt(15.0)
    out = np.empty((direction.shape[0], 16))
    out[:, 0] = 1.0
    out[:, 1] = s3 * y
    out[:, 2] = s3 * z
    out[:, 3] = s3 * x
    out[:, 4] = s15 * x * y
    out[:, 5] = s15 * y * z
    out[:, 6] = s5 * 0.5 * (3.0 * z * z - 1.0)
    out[:, 7] = s15 * x * z
    out[:, 8] = s15 * 0.5 * (x * x - y * y)
    out[:, 9] = s7 * np.sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y)
    out[:, 10] = s7 * np.sqrt(15.0) * x * y * z
    out[:, 11] = s7 * np.sqrt(3.0 / 8.0) * y * (5.0 * z * z - 1.0)
    out[:, 12] = s7 * 0.5 * z * (5.0 * z * z - 3.0)
    out[:, 13] = s7 * np.sqrt(3.0 / 8.0) * x * (5.0 * z * z - 1.0)
    out[:, 14] = s7 * np.sqrt(15.0) * 0.5 * z * (x * x - y * y)
    out[:, 15] = s7 * np.sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y)
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        axis=1,
    )


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2])


def _lambert(normal: np.ndarray, seed: int, ray: np.ndarray, bounce: int) -> np.ndarray:
    """:func:`rays._lambert` for many rays at one bounce."""
    helper = np.where(
        (np.abs(normal[:, 0]) < 0.9)[:, None],
        np.array([1.0, 0.0, 0.0])[None, :],
        np.array([0.0, 1.0, 0.0])[None, :],
    )
    t1 = _cross(normal, helper)
    t1 = t1 / np.sqrt(_dot(t1, t1))[:, None]
    t2 = _cross(normal, t1)
    out = normal.copy()
    pending = np.ones(ray.size, dtype=bool)
    for attempt in range(64):
        idx = np.flatnonzero(pending)
        if idx.size == 0:
            break
        base = 1 + 2 * attempt
        at = np.full(idx.size, bounce)
        a = 2.0 * hash_uniform(seed, ray[idx], at, np.full(idx.size, base)) - 1.0
        b = 2.0 * hash_uniform(seed, ray[idx], at, np.full(idx.size, base + 1)) - 1.0
        r2 = a * a + b * b
        ok = r2 <= 1.0
        z = np.sqrt(np.where(ok, 1.0 - r2, 0.0))
        made = a[:, None] * t1[idx] + b[:, None] * t2[idx] + z[:, None] * normal[idx]
        out[idx[ok]] = made[ok]
        pending[idx[ok]] = False
    return out


def trace_tree(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings | None = None,
    *,
    held: TracerScene | None = None,
    ray_start: int = 0,
    ray_count: int | None = None,
    stats: dict[str, int] | None = None,
) -> Histogram:
    """The twin through the tree: :func:`rays.trace`, every ray advanced a segment at a time.

    ``stats``, when given, gains the counts of :data:`STATS` the C text
    counts: segments, nodes visited, triangles tested, rays that met no
    triangle at all.
    """
    settings = settings or RaySettings()
    if settings.precision != "double":
        raise ValueError("the twin is double precision; single precision is the C text's")
    if settings.order > 3:
        raise ValueError(f"the tree's tracer holds harmonics to order 3, asked {settings.order}")
    held = held or prepare(scene)
    count = settings.rays - ray_start if ray_count is None else ray_count
    source = np.asarray(source, dtype=float).reshape(3)
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    triangles = np.asarray(scene.occluder_vertices, dtype=float).reshape(-1, 3, 3)
    ordered = held.triangles.reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    of_scene = held.of_scene()
    absorption = scene.materials.absorption
    scattering = scene.materials.scattering
    bands = absorption.shape[1]
    reach = settings.sound_speed_m_s * settings.duration_s
    bins = int(np.ceil(settings.duration_s / settings.bin_s))
    bin_length = settings.sound_speed_m_s * settings.bin_s
    channels = channel_count(settings.order)
    radius2 = settings.receiver_radius_m**2
    energy = np.zeros((receivers.shape[0], bins, bands), dtype=np.int64)
    moments = np.zeros((receivers.shape[0], bins, bands, channels), dtype=np.int64)
    hits = np.zeros((receivers.shape[0], bins), dtype=np.int64)
    floor = settings.energy_floor / settings.rays
    skip_reach = (
        settings.sound_speed_m_s * settings.skip_window_s if settings.skip_window_s > 0 else np.inf
    )
    ray = np.arange(ray_start, ray_start + count, dtype=np.int64)
    direction = ray_directions(count, settings.seed, start=ray_start)
    position = np.repeat(source[None, :], count, axis=0)
    carried = np.full((count, bands), 1.0 / settings.rays)
    travelled = np.zeros(count)
    last = np.full(count, -1, dtype=np.int64)
    specular_only = np.ones(count, dtype=bool)
    furniture_bounces = np.zeros(count, dtype=np.int64)
    alive = np.arange(count)
    counted: dict[str, int] = {}
    tally = dict.fromkeys(STATS, 0)
    # Rays by receivers at a time: no more than a few million crossings tested at once.
    block = max(1, 2_000_000 // max(receivers.shape[0], 1))
    for bounce in range(100_000):
        if alive.size == 0:
            break
        o, d = position[alive], direction[alive]
        distance, triangle = nearest(held.tree, ordered, o, d, last[alive], counts=counted)
        segment = np.minimum(distance, reach - travelled[alive])
        covered = (
            specular_only[alive]
            & (1 <= bounce <= settings.skip_specular_order)
            & (furniture_bounces[alive] <= settings.skip_furniture_bounces)
        )
        for first in range(0, alive.size, block):
            rows = slice(first, first + block)
            rel = receivers[None, :, :] - o[rows, None, :]
            dd = d[rows]
            along = (
                rel[:, :, 0] * dd[:, None, 0]
                + rel[:, :, 1] * dd[:, None, 1]
                + rel[:, :, 2] * dd[:, None, 2]
            )
            rel2 = rel[:, :, 0] * rel[:, :, 0] + rel[:, :, 1] * rel[:, :, 1]
            rel2 = rel2 + rel[:, :, 2] * rel[:, :, 2]
            perp2 = rel2 - along * along
            half = np.sqrt(np.maximum(radius2 - perp2, 0.0))
            starts_inside = rel2 <= radius2
            entry = np.where(starts_inside, 0.0, along - half)
            crossed = (
                (perp2 <= radius2)
                & (entry >= 0.0)
                & (entry <= segment[rows, None])
                & ~(starts_inside & (along < 0.0))
            )
            arrival = travelled[alive[rows], None] + entry
            # What the image tree renders already is not counted, within its window.
            crossed &= ~covered[rows, None] | (arrival > skip_reach)
            at = (arrival / bin_length).astype(np.int64)
            crossed &= at < bins
            who, where = np.nonzero(crossed)
            if who.size == 0:
                continue
            tally["deposits"] += int(who.size)
            mine = carried[alive[rows]][who]
            harmonics = _harmonics3(scene_to_ambisonic(-dd[who]))[:, :channels]
            np.add.at(
                energy, (where, at[who, where]), np.rint(mine * HISTOGRAM_SCALE).astype(np.int64)
            )
            np.add.at(
                moments,
                (where, at[who, where]),
                np.rint(mine[:, :, None] * harmonics[:, None, :] * HISTOGRAM_SCALE).astype(
                    np.int64
                ),
            )
            np.add.at(hits, (where, at[who, where]), 1)
        tally["escapes"] += int(np.count_nonzero(triangle < 0))
        with np.errstate(invalid="ignore"):
            on = (triangle >= 0) & (travelled[alive] + distance < reach)
        tally["ended_by_reach"] += int(np.count_nonzero(~on & (triangle >= 0)))
        alive, triangle, distance = alive[on], triangle[on], distance[on]
        travelled[alive] += distance
        position[alive] = position[alive] + direction[alive] * distance[:, None]
        label = of_scene["label"][triangle]
        carried[alive] = carried[alive] * (1.0 - absorption[label])
        on = ~np.all(carried[alive] < floor, axis=1)
        tally["ended_by_floor"] += int(np.count_nonzero(~on))
        alive, triangle, label = alive[on], triangle[on], label[on]
        normal = normals[triangle]
        d = direction[alive]
        normal = np.where((_dot(normal, d) > 0.0)[:, None], -normal, normal)
        kind = hash_uniform(
            settings.seed, ray[alive], np.full(alive.size, bounce + 1), np.zeros(alive.size, int)
        )
        furniture_bounces[alive] += of_scene["furniture"][triangle]
        scattered = kind < scattering[label]
        mirrored = d - (2.0 * _dot(d, normal))[:, None] * normal
        if bool(scattered.any()):
            mirrored[scattered] = _lambert(
                normal[scattered], settings.seed, ray[alive[scattered]], bounce + 1
            )
        direction[alive] = mirrored
        specular_only[alive[scattered | of_scene["off"][triangle]]] = False
        last[alive] = triangle
    if stats is not None:
        tally["segments"] = counted.get("segments", 0)
        tally["nodes"] = counted.get("nodes", 0)
        tally["tests"] = counted.get("tests", 0)
        for name, value in tally.items():
            stats[name] = stats.get(name, 0) + value
    return Histogram.from_counts(
        energy,
        moments,
        hits,
        bin_s=settings.bin_s,
        bands_hz=scene.materials.bands_hz,
        order=settings.order,
        rays=settings.rays,
    )


# --------------------------------------------------------------------------
# the C text
# --------------------------------------------------------------------------

#: The tracer: plain C that is also C++, so that a card compiles the same lines. The text it
#: is appended to gives ``RT_FN`` (what a function is declared with), ``RT_SINGLE`` (0 or 1),
#: ``RT_ADD`` (how a count is added to a histogram), ``rt_bits`` (a float's bits as an
#: integer) and the infinities.
CORE_SOURCE = r"""
#define RT_STACK 96
#define RT_EMPTY (-2147483647 - 1)
#define RT_SHRINK 0.999998f
#define RT_GROW 1.000002f
#define RT_SCALE 1099511627776.0   /* 2^40, HISTOGRAM_SCALE */
#if RT_SINGLE
typedef float rt_real;
#define RT_R(x) ((float)(x))
#define RT_SQRT(x) sqrtf(x)
#define RT_ABS(x) fabsf(x)
#define RT_INF RT_INF_F
/* A triangle parallel to the one a ray leaves, hit nearer than this to that one's plane,
   is the ray's own surface, which single precision may put on either side of the origin:
   it is not hit. Metres, and the cosine from which two triangles are parallel. */
#define RT_PLANE 2e-5f
#define RT_PARALLEL 0.999f
#else
typedef double rt_real;
#define RT_R(x) (x)
#define RT_SQRT(x) sqrt(x)
#define RT_ABS(x) fabs(x)
#define RT_INF RT_INF_D
#endif

typedef struct {
    const float* nodes;           /* [node, 16]: two boxes, two links, two words of nothing */
    const rt_real* tris;          /* [triangle, 9] in leaf order */
    const int* meta;              /* [triangle, 2] in leaf order: scene index, label and flags */
    const float* rec_nodes;       /* the receivers' tree */
    const int* rec_order;         /* [receiver] in its leaf order */
    const rt_real* receivers;     /* [receiver, 3] */
    const double* absorption;     /* [label, band] */
    const double* scattering;     /* [label] */
    const double* source;         /* [3] */
    unsigned long long seed;
    double reach, bin_length, radius, floor_energy, skip_reach;
    int total_rays, bins, bands, channels, skip_order, skip_furniture;
} RtShot;

RT_FN unsigned long long rt_mix(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* The twin's ``hash_uniform``: 53 bits of the hash over 2^53. */
RT_FN double rt_uniform53(
    unsigned long long seed, unsigned long long ray, unsigned long long bounce,
    unsigned long long draw)
{
    unsigned long long key = rt_mix(seed * 0x9E3779B97F4A7C15ULL + ray);
    unsigned long long state =
        rt_mix(key ^ (bounce * 0xD6E8FEB86659FD93ULL) ^ (draw * 0xA0761D6478BD642FULL));
    return (double)(state >> 11) / 9007199254740992.0;
}

/* The same draw as the arithmetic takes it: all 53 bits in double precision, the 24
   highest of them in single, which is still under one. */
RT_FN rt_real rt_uniform(
    unsigned long long seed, unsigned long long ray, unsigned long long bounce,
    unsigned long long draw)
{
#if RT_SINGLE
    unsigned long long key = rt_mix(seed * 0x9E3779B97F4A7C15ULL + ray);
    unsigned long long state =
        rt_mix(key ^ (bounce * 0xD6E8FEB86659FD93ULL) ^ (draw * 0xA0761D6478BD642FULL));
    return (float)(state >> 40) * 5.9604644775390625e-8f;
#else
    return rt_uniform53(seed, ray, bounce, draw);
#endif
}

RT_FN rt_real rt_dot(const rt_real* a, const rt_real* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

RT_FN void rt_cross(const rt_real* a, const rt_real* b, rt_real* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* The twin's ``harmonics3``: N3D real harmonics to order 3, ACN order. */
RT_FN void rt_harmonics(rt_real x, rt_real y, rt_real z, rt_real* out) {
    rt_real s3 = RT_SQRT(RT_R(3.0)), s5 = RT_SQRT(RT_R(5.0));
    rt_real s7 = RT_SQRT(RT_R(7.0)), s15 = RT_SQRT(RT_R(15.0));
    out[0] = RT_R(1.0);
    out[1] = s3 * y;
    out[2] = s3 * z;
    out[3] = s3 * x;
    out[4] = s15 * x * y;
    out[5] = s15 * y * z;
    out[6] = s5 * RT_R(0.5) * (RT_R(3.0) * z * z - RT_R(1.0));
    out[7] = s15 * x * z;
    out[8] = s15 * RT_R(0.5) * (x * x - y * y);
    out[9] = s7 * RT_SQRT(RT_R(5.0 / 8.0)) * y * (RT_R(3.0) * x * x - y * y);
    out[10] = s7 * RT_SQRT(RT_R(15.0)) * x * y * z;
    out[11] = s7 * RT_SQRT(RT_R(3.0 / 8.0)) * y * (RT_R(5.0) * z * z - RT_R(1.0));
    out[12] = s7 * RT_R(0.5) * z * (RT_R(5.0) * z * z - RT_R(3.0));
    out[13] = s7 * RT_SQRT(RT_R(3.0 / 8.0)) * x * (RT_R(5.0) * z * z - RT_R(1.0));
    out[14] = s7 * RT_SQRT(RT_R(15.0)) * RT_R(0.5) * z * (x * x - y * y);
    out[15] = s7 * RT_SQRT(RT_R(5.0 / 8.0)) * x * (x * x - RT_R(3.0) * y * y);
}

/* Whether a ray enters a box before ``limit``, and where: :func:`bvh._slab`. Single
   precision; the box is padded and the distances are taken short and long, so a box the
   ray in double precision enters is never refused. */
RT_FN int rt_slab(
    const float* b, const float* o, const float* inv, float limit, float* near_out)
{
    float lo = (b[0] - o[0]) * inv[0], hi = (b[3] - o[0]) * inv[0];
    float t_near = fminf(lo, hi), t_far = fmaxf(lo, hi);
    lo = (b[1] - o[1]) * inv[1]; hi = (b[4] - o[1]) * inv[1];
    t_near = fmaxf(t_near, fminf(lo, hi)); t_far = fminf(t_far, fmaxf(lo, hi));
    lo = (b[2] - o[2]) * inv[2]; hi = (b[5] - o[2]) * inv[2];
    t_near = fmaxf(t_near, fminf(lo, hi)); t_far = fminf(t_far, fmaxf(lo, hi));
    t_near = fmaxf(t_near, 0.0f) * RT_SHRINK;
    t_far = t_far * RT_GROW;
    *near_out = t_near;
    return t_near <= t_far && t_near <= limit;
}

#if RT_SINGLE
/* Woop, Benthin and Wald (2013): the ray's own frame, in which it runs along z. */
typedef struct { int kx, ky, kz; float sx, sy, sz; } RtFrame;

RT_FN void rt_frame(const float* d, RtFrame* f) {
    int kz = 0;
    if (fabsf(d[1]) > fabsf(d[kz])) kz = 1;
    if (fabsf(d[2]) > fabsf(d[kz])) kz = 2;
    int kx = kz == 2 ? 0 : kz + 1;
    int ky = kx == 2 ? 0 : kx + 1;
    if (d[kz] < 0.0f) { int swap = kx; kx = ky; ky = swap; }
    f->kx = kx; f->ky = ky; f->kz = kz;
    f->sx = d[kx] / d[kz];
    f->sy = d[ky] / d[kz];
    f->sz = 1.0f / d[kz];
}

/* Whether the ray hits the triangle, the distance, and the hit's weights on the three
   corners. The three edge functions are each a difference of two products of the sheared
   corners: the triangle on the other side of an edge takes the same two products, so a
   ray is never outside both, and one that is on the edge to the last bit is settled in
   double precision.

   ``plane`` is the unit normal of the surface the ray leaves, on the side it leaves by,
   and ``leaving`` its cosine with the ray (both zero for a ray from the source). A hit is
   refused when it lies within RT_PLANE of that surface on a triangle parallel to it: the
   surface itself, met again because the origin is a rounding behind it. A wall that meets
   the surface at an angle is hit however near, which a bound on the distance alone would
   let the ray through at every corner of a room. */
RT_FN int rt_hit(
    const float* o, const RtFrame* f, const float* tri, const float* plane, float leaving,
    float* t_out, float* w)
{
    float a[3], b[3], c[3];
    for (int k = 0; k < 3; ++k) {
        a[k] = tri[k] - o[k]; b[k] = tri[3 + k] - o[k]; c[k] = tri[6 + k] - o[k];
    }
    float ax = a[f->kx] - f->sx * a[f->kz], ay = a[f->ky] - f->sy * a[f->kz];
    float bx = b[f->kx] - f->sx * b[f->kz], by = b[f->ky] - f->sy * b[f->kz];
    float cx = c[f->kx] - f->sx * c[f->kz], cy = c[f->ky] - f->sy * c[f->kz];
    float u = cx * by - cy * bx;
    float v = ax * cy - ay * cx;
    float s = bx * ay - by * ax;
    if (u == 0.0f || v == 0.0f || s == 0.0f) {
        u = (float)((double)cx * (double)by - (double)cy * (double)bx);
        v = (float)((double)ax * (double)cy - (double)ay * (double)cx);
        s = (float)((double)bx * (double)ay - (double)by * (double)ax);
    }
    if ((u < 0.0f || v < 0.0f || s < 0.0f) && (u > 0.0f || v > 0.0f || s > 0.0f)) return 0;
    float det = u + v + s;
    if (det == 0.0f) return 0;
    float t = (u * (f->sz * a[f->kz]) + v * (f->sz * b[f->kz]) + s * (f->sz * c[f->kz])) / det;
    if (!(t > 0.0f)) return 0;
    if (t * leaving < RT_PLANE) {
        float e1[3], e2[3], n[3];
        for (int k = 0; k < 3; ++k) { e1[k] = tri[3 + k] - tri[k]; e2[k] = tri[6 + k] - tri[k]; }
        rt_cross(e1, e2, n);
        float along = fabsf(rt_dot(n, plane));
        if (along > RT_PARALLEL * sqrtf(rt_dot(n, n))) return 0;
    }
    *t_out = t;
    w[0] = u / det; w[1] = v / det; w[2] = s / det;
    return 1;
}
#else
/* Moller and Trumbore as the twin's ``moller_trumbore``: a unit direction, a hit inside
   the triangle, edges included, and farther than ``t_min``. */
RT_FN int rt_hit(
    const double* o, const double* d, const double* tri, double t_min, double* t_out)
{
    double e1[3], e2[3], p[3], tv[3], q[3];
    for (int k = 0; k < 3; ++k) { e1[k] = tri[3 + k] - tri[k]; e2[k] = tri[6 + k] - tri[k]; }
    rt_cross(d, e2, p);
    double det = rt_dot(e1, p);
    if (!(fabs(det) > 1e-12)) return 0;
    double inv = 1.0 / det;
    for (int k = 0; k < 3; ++k) tv[k] = o[k] - tri[k];
    double u = rt_dot(tv, p) * inv;
    rt_cross(tv, e1, q);
    double v = rt_dot(d, q) * inv;
    double t = rt_dot(e2, q) * inv;
    if (!(u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > t_min)) return 0;
    *t_out = t;
    return 1;
}
#endif

/* The first triangle a ray hits: its place in the leaf order, or -1, and the distance.
   :func:`bvh.nearest`: a stack, the nearer child first, the farther pushed with its entry
   distance and dropped at the pop when a hit has come nearer; of two triangles at one
   distance, the lower index of the scene. In single precision ``w`` takes the hit's
   weights on the triangle's corners. */
RT_FN int rt_nearest(
    const RtShot* s, const rt_real* o, const rt_real* d, int last, const rt_real* plane,
    rt_real* t_out, float* w, long long* nodes, long long* tests)
{
    float of[3], inv[3];
    for (int k = 0; k < 3; ++k) { of[k] = (float)o[k]; inv[k] = 1.0f / (float)d[k]; }
#if RT_SINGLE
    RtFrame frame;
    rt_frame(d, &frame);
    float leaving = rt_dot(plane, d);
#endif
    rt_real best = RT_INF;
    int best_slot = -1, best_index = -1;
    float limit = RT_INF_F;
    int stack_node[RT_STACK];
    float stack_near[RT_STACK];
    int top = 0, node = 0;
    for (;;) {
        if (node >= 0) {
            const float* n = s->nodes + 16 * (long long)node;
            int l0 = rt_bits(n[12]), l1 = rt_bits(n[13]);
            float n0 = 0.0f, n1 = 0.0f;
            int h0 = l0 != RT_EMPTY && rt_slab(n, of, inv, limit, &n0);
            int h1 = l1 != RT_EMPTY && rt_slab(n + 6, of, inv, limit, &n1);
            *nodes += 1;
            if (h0 && h1) {
                if (n1 < n0) { stack_node[top] = l0; stack_near[top] = n0; node = l1; }
                else { stack_node[top] = l1; stack_near[top] = n1; node = l0; }
                ++top;
                continue;
            }
            if (h0) { node = l0; continue; }
            if (h1) { node = l1; continue; }
        } else {
            int code = ~node;
            int first = code >> 4, count = (code & 15) + 1;
            *tests += count;
            for (int i = 0; i < count; ++i) {
                int slot = first + i;
                int index = s->meta[2 * (long long)slot];
                if (index == last) continue;
                rt_real t;
#if RT_SINGLE
                float weights[3];
                const float* tri = s->tris + 9 * (long long)slot;
                if (!rt_hit(o, &frame, tri, plane, leaving, &t, weights)) continue;
#else
                /* The twin's bound: a hit farther than a micrometre. */
                if (!rt_hit(o, d, s->tris + 9 * (long long)slot, 1e-6, &t)) continue;
#endif
                if (t < best || (t == best && index < best_index)) {
                    best = t; best_slot = slot; best_index = index;
                    limit = (float)t * RT_GROW;
#if RT_SINGLE
                    w[0] = weights[0]; w[1] = weights[1]; w[2] = weights[2];
#endif
                }
            }
        }
        for (;;) {
            if (top == 0) { *t_out = best; return best_slot; }
            --top;
            if (stack_near[top] <= limit) { node = stack_node[top]; break; }
        }
    }
}

/* The receivers whose sphere a segment enters, each given the ray's energy at the time it
   enters: the twin's crossings, over the receivers the spheres' own tree finds. */
RT_FN void rt_deposit(
    const RtShot* s, const rt_real* pos, const rt_real* dir, rt_real segment,
    rt_real travelled, int covered, const rt_real* carried,
    long long* energy_out, long long* moments_out, long long* hits_out,
    long long* nodes, long long* deposits)
{
    float of[3], inv[3];
    for (int k = 0; k < 3; ++k) { of[k] = (float)pos[k]; inv[k] = 1.0f / (float)dir[k]; }
    float limit = (float)segment * RT_GROW;
    rt_real radius2 = RT_R(s->radius) * RT_R(s->radius);
    rt_real harmonics[16];
    int have = 0;
    int stack_node[RT_STACK];
    int top = 0, node = 0;
    for (;;) {
        if (node >= 0) {
            const float* n = s->rec_nodes + 16 * (long long)node;
            int l0 = rt_bits(n[12]), l1 = rt_bits(n[13]);
            float n0, n1;
            int h0 = l0 != RT_EMPTY && rt_slab(n, of, inv, limit, &n0);
            int h1 = l1 != RT_EMPTY && rt_slab(n + 6, of, inv, limit, &n1);
            *nodes += 1;
            if (h0 && h1) { stack_node[top++] = l1; node = l0; continue; }
            if (h0) { node = l0; continue; }
            if (h1) { node = l1; continue; }
        } else {
            int code = ~node;
            int first = code >> 4, count = (code & 15) + 1;
            for (int i = 0; i < count; ++i) {
                int r = s->rec_order[first + i];
                const rt_real* c = s->receivers + 3 * (long long)r;
                rt_real rel[3] = {c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]};
                rt_real along = rt_dot(rel, dir);
                rt_real rel2 = rt_dot(rel, rel);
                rt_real perp2 = rel2 - along * along;
                rt_real left = radius2 - perp2;
                rt_real half = RT_SQRT(left > RT_R(0.0) ? left : RT_R(0.0));
                int starts_inside = rel2 <= radius2;
                rt_real entry = starts_inside ? RT_R(0.0) : along - half;
                if (!(perp2 <= radius2 && entry >= RT_R(0.0) && entry <= segment)) continue;
                if (starts_inside && along < RT_R(0.0)) continue;
                rt_real arrival = travelled + entry;
                /* What the image tree renders already is not counted, within its window. */
                if (covered && !(arrival > RT_R(s->skip_reach))) continue;
                int at = (int)(arrival / RT_R(s->bin_length));
                if (at >= s->bins) continue;
                if (!have) {
                    /* Arrival direction, scene frame to ambisonic: (x, y, z) -> (x, -z, y). */
                    rt_harmonics(-dir[0], dir[2], -dir[1], harmonics);
                    have = 1;
                }
                long long base = ((long long)r * s->bins + at) * s->bands;
                for (int b = 0; b < s->bands; ++b) {
                    RT_ADD(energy_out + base + b, llrint((double)carried[b] * RT_SCALE));
                    long long mbase = (base + b) * s->channels;
                    for (int ch = 0; ch < s->channels; ++ch)
                        RT_ADD(moments_out + mbase + ch,
                               llrint((double)(carried[b] * harmonics[ch]) * RT_SCALE));
                }
                RT_ADD(hits_out + (long long)r * s->bins + at, 1LL);
                *deposits += 1;
            }
        }
        if (top == 0) return;
        node = stack_node[--top];
    }
}

/* One ray's whole life: the twin's loop over its bounces. ``stats`` are this ray's own
   eight counts, added by the caller. */
RT_FN void rt_ray(
    const RtShot* s, int ray, long long* energy_out, long long* moments_out,
    long long* hits_out, long long* stats)
{
    /* The twin's ``ray_directions``: rejection in the cube, draws 0, 1, 2, ... */
    double first[3] = {0.0, 0.0, 1.0};
    for (int attempt = 0; attempt < 64; ++attempt) {
        int base = 3 * attempt;
        double u = rt_uniform53(s->seed, ray, 0, base);
        double v = rt_uniform53(s->seed, ray, 0, base + 1);
        double w = rt_uniform53(s->seed, ray, 0, base + 2);
        double p[3] = {2.0 * u - 1.0, 2.0 * v - 1.0, 2.0 * w - 1.0};
        double norm2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if (norm2 <= 1.0 && norm2 > 1e-8) {
            double length = sqrt(norm2);
            for (int k = 0; k < 3; ++k) first[k] = p[k] / length;
            break;
        }
    }
    rt_real pos[3], dir[3], carried[8];
    for (int k = 0; k < 3; ++k) { pos[k] = RT_R(s->source[k]); dir[k] = RT_R(first[k]); }
    for (int b = 0; b < s->bands; ++b) carried[b] = RT_R(1.0 / (double)s->total_rays);
    rt_real reach = RT_R(s->reach), floor_energy = RT_R(s->floor_energy);
    rt_real travelled = RT_R(0.0);
    /* The unit normal of the surface the ray leaves, on its side: none yet. */
    rt_real plane[3] = {RT_R(0.0), RT_R(0.0), RT_R(0.0)};
    int last = -1, specular_only = 1, furniture_bounces = 0;
    for (int bounce = 0; bounce < 100000; ++bounce) {
        rt_real best;
        float weights[3] = {0.0f, 0.0f, 0.0f};
        int slot = rt_nearest(s, pos, dir, last, plane, &best, weights, stats + 1, stats + 2);
        stats[0] += 1;
        rt_real remaining = reach - travelled;
        rt_real segment = best < remaining ? best : remaining;
        /* The twin's ``covered``: what the image tree renders already is not counted. */
        int covered = specular_only && bounce >= 1 && bounce <= s->skip_order
            && furniture_bounces <= s->skip_furniture;
        if (!(covered && !(travelled + segment > RT_R(s->skip_reach))))
            rt_deposit(s, pos, dir, segment, travelled, covered, carried,
                       energy_out, moments_out, hits_out, stats + 4, stats + 5);
        if (slot < 0) { stats[3] += 1; break; }
        if (travelled + best >= reach) { stats[6] += 1; break; }
        travelled += best;
        const rt_real* tri = s->tris + 9 * (long long)slot;
#if RT_SINGLE
        /* The hit as the triangle's own point: on its plane to the rounding of a
           coordinate, however long the segment was. */
        for (int k = 0; k < 3; ++k)
            pos[k] = weights[0] * tri[k] + weights[1] * tri[3 + k] + weights[2] * tri[6 + k];
#else
        for (int k = 0; k < 3; ++k) pos[k] = pos[k] + dir[k] * best;
#endif
        int word = s->meta[2 * (long long)slot + 1];
        int label = word & 0xFFFF;
        int any = 0;
        for (int b = 0; b < s->bands; ++b) {
            carried[b] = carried[b] * RT_R(1.0 - s->absorption[label * s->bands + b]);
            if (carried[b] >= floor_energy) any = 1;
        }
        if (!any) { stats[7] += 1; break; }
        rt_real e1[3], e2[3], n[3];
        for (int k = 0; k < 3; ++k) { e1[k] = tri[3 + k] - tri[k]; e2[k] = tri[6 + k] - tri[k]; }
        rt_cross(e1, e2, n);
        rt_real nn = RT_SQRT(rt_dot(n, n));
        if (nn < RT_R(1e-12)) nn = RT_R(1e-12);
        for (int k = 0; k < 3; ++k) n[k] = n[k] / nn;
        if (rt_dot(n, dir) > RT_R(0.0)) for (int k = 0; k < 3; ++k) n[k] = -n[k];
        rt_real kind = rt_uniform(s->seed, ray, bounce + 1, 0);
        if ((word >> 16) & 1) furniture_bounces += 1;
        if (kind < RT_R(s->scattering[label])) {
            specular_only = 0;
            /* The twin's ``_lambert``. */
            rt_real helper[3] = {RT_R(1.0), RT_R(0.0), RT_R(0.0)};
            if (!(RT_ABS(n[0]) < RT_R(0.9))) { helper[0] = RT_R(0.0); helper[1] = RT_R(1.0); }
            rt_real t1[3], t2[3];
            rt_cross(n, helper, t1);
            rt_real t1n = RT_SQRT(rt_dot(t1, t1));
            for (int k = 0; k < 3; ++k) t1[k] = t1[k] / t1n;
            rt_cross(n, t1, t2);
            int found = 0;
            for (int attempt = 0; attempt < 64 && !found; ++attempt) {
                int base = 1 + 2 * attempt;
                rt_real a = RT_R(2.0) * rt_uniform(s->seed, ray, bounce + 1, base) - RT_R(1.0);
                rt_real bb = RT_R(2.0) * rt_uniform(s->seed, ray, bounce + 1, base + 1) - RT_R(1.0);
                rt_real r2 = a * a + bb * bb;
                if (r2 <= RT_R(1.0)) {
                    rt_real z = RT_SQRT(RT_R(1.0) - r2);
                    for (int k = 0; k < 3; ++k) dir[k] = a * t1[k] + bb * t2[k] + z * n[k];
                    found = 1;
                }
            }
            if (!found) for (int k = 0; k < 3; ++k) dir[k] = n[k];
        } else {
            rt_real dn = rt_dot(dir, n);
            for (int k = 0; k < 3; ++k) dir[k] = dir[k] - RT_R(2.0) * dn * n[k];
            if ((word >> 17) & 1) specular_only = 0;
        }
        last = s->meta[2 * (long long)slot];
#if RT_SINGLE
        /* A direction keeps its length to a rounding a bounce: put back to one. */
        rt_real length = RT_SQRT(rt_dot(dir, dir));
        for (int k = 0; k < 3; ++k) dir[k] = dir[k] / length;
#endif
        for (int k = 0; k < 3; ++k) plane[k] = n[k];
    }
}
"""

_HOST_HEAD = r"""
#include <math.h>
#include <string.h>
#define RT_FN static inline
#define RT_INF_F ((float)INFINITY)
#define RT_INF_D ((double)INFINITY)
#define RT_ADD(target, value) (*(target) += (value))
static inline int rt_bits(float x) { int i; memcpy(&i, &x, sizeof i); return i; }
"""

_HOST_TAIL = r"""
/* Rays ``ray_start`` to ``ray_start + ray_count``, one after the other. */
void RT_NAME(
    const RtShot* s, int ray_start, int ray_count, long long* energy_out,
    long long* moments_out, long long* hits_out, long long* stats_out)
{
    for (int local = 0; local < ray_count; ++local) {
        long long stats[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        rt_ray(s, ray_start + local, energy_out, moments_out, hits_out, stats);
        for (int k = 0; k < 8; ++k) stats_out[k] += stats[k];
    }
}

/* The first triangle each of ``count`` rays hits, as the scene numbers it, and how far. */
void RT_FIRST(
    const RtShot* s, int count, const rt_real* origins, const rt_real* directions,
    rt_real* t_out, int* index_out)
{
    for (int i = 0; i < count; ++i) {
        long long nodes = 0, tests = 0;
        float weights[3];
        rt_real plane[3] = {RT_R(0.0), RT_R(0.0), RT_R(0.0)};
        int slot = rt_nearest(
            s, origins + 3 * i, directions + 3 * i, -1, plane, t_out + i, weights,
            &nodes, &tests);
        index_out[i] = slot < 0 ? -1 : s->meta[2 * (long long)slot];
    }
}
"""


def _host_part(single: bool) -> str:
    """The text once for a precision, its names apart: C has one name a function."""
    text = CORE_SOURCE
    suffix = "_s" if single else "_d"
    for name in (
        "rt_real",
        "RtShot",
        "RtFrame",
        "rt_mix",
        "rt_uniform53",
        "rt_uniform",
        "rt_dot",
        "rt_cross",
        "rt_harmonics",
        "rt_slab",
        "rt_frame",
        "rt_hit",
        "rt_nearest",
        "rt_deposit",
        "rt_ray",
    ):
        text = _renamed(text, name, name + suffix)
    tail = _HOST_TAIL
    for name in ("RtShot", "rt_real", "rt_ray", "rt_nearest"):
        tail = _renamed(tail, name, name + suffix)
    macros = ["STACK", "EMPTY", "SHRINK", "GROW", "SCALE", "R", "SQRT", "ABS", "INF", "PLANE"]
    macros += ["PARALLEL"]
    macros += ["SINGLE", "NAME", "FIRST"]
    return (
        f"#define RT_SINGLE {int(single)}\n#define RT_NAME rt_trace{suffix}\n"
        f"#define RT_FIRST rt_first{suffix}\n"
        + text
        + tail
        + "".join(f"#undef RT_{macro}\n" for macro in macros)
    )


def _renamed(text: str, name: str, to: str) -> str:
    import re

    return re.sub(rf"\b{name}\b", to, text)


#: What a host's compiler is given: the text in double precision, then in single.
HOST_SOURCE = _HOST_HEAD + _host_part(False) + _host_part(True)

_DEVICE_HEAD = r"""
#define RT_FN __device__ __forceinline__
#define RT_INF_F (__int_as_float(0x7f800000))
#define RT_INF_D (__longlong_as_double(0x7ff0000000000000LL))
#define RT_ADD(target, value) \
    atomicAdd((unsigned long long*)(target), (unsigned long long)(value))
#define rt_bits(x) __float_as_int(x)
"""

_DEVICE_TAIL = r"""
/* A thread a ray. */
extern "C" __global__ void rays_tree(
    const float* __restrict__ nodes, const rt_real* __restrict__ tris,
    const int* __restrict__ meta, const float* __restrict__ rec_nodes,
    const int* __restrict__ rec_order, const rt_real* __restrict__ receivers,
    const double* __restrict__ absorption, const double* __restrict__ scattering,
    const double* __restrict__ source, unsigned long long seed,
    double reach, double bin_length, double radius, double floor_energy, double skip_reach,
    int total_rays, int bins, int bands, int channels, int skip_order, int skip_furniture,
    int ray_start, int ray_count,
    long long* __restrict__ energy_out, long long* __restrict__ moments_out,
    long long* __restrict__ hits_out, long long* __restrict__ stats_out)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= ray_count) return;
    RtShot s;
    s.nodes = nodes; s.tris = tris; s.meta = meta; s.rec_nodes = rec_nodes;
    s.rec_order = rec_order; s.receivers = receivers; s.absorption = absorption;
    s.scattering = scattering; s.source = source; s.seed = seed;
    s.reach = reach; s.bin_length = bin_length; s.radius = radius;
    s.floor_energy = floor_energy; s.skip_reach = skip_reach;
    s.total_rays = total_rays; s.bins = bins; s.bands = bands; s.channels = channels;
    s.skip_order = skip_order; s.skip_furniture = skip_furniture;
    long long stats[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rt_ray(&s, ray_start + local, energy_out, moments_out, hits_out, stats);
    for (int k = 0; k < 8; ++k) if (stats[k]) RT_ADD(stats_out + k, stats[k]);
}
"""


def device_source(single: bool) -> str:
    """What ``cupy`` compiles for a card, under :data:`reverberate.compute.KERNEL_OPTIONS`."""
    return f"#define RT_SINGLE {int(single)}\n" + _DEVICE_HEAD + CORE_SOURCE + _DEVICE_TAIL


#: The card's text in double precision; :func:`device_source` gives either.
DEVICE_SOURCE = device_source(False)

_POINTER = ctypes.c_void_p


class _Shot(ctypes.Structure):
    _fields_ = [
        *[
            (name, _POINTER)
            for name in (
                "nodes",
                "tris",
                "meta",
                "rec_nodes",
                "rec_order",
                "receivers",
                "absorption",
                "scattering",
                "source",
            )
        ],
        ("seed", ctypes.c_ulonglong),
        *[
            (name, ctypes.c_double)
            for name in ("reach", "bin_length", "radius", "floor_energy", "skip_reach")
        ],
        *[
            (name, ctypes.c_int)
            for name in ("total_rays", "bins", "bands", "channels", "skip_order", "skip_furniture")
        ],
    ]


_state: dict[str, Any] = {"library": None, "why": None, "tried": False}


def _build() -> ctypes.CDLL:
    """The shared object of :data:`HOST_SOURCE`: found under its name, or built there."""
    compiler = _compiler()
    if compiler is None:
        raise OSError("no C compiler on this machine (cc, gcc, clang, or CC)")
    version = subprocess.run(
        [*compiler.split(), "--version"], capture_output=True, text=True, check=False
    ).stdout
    name = hashlib.sha256(
        "\n".join([HOST_SOURCE, " ".join(FLAGS), compiler, version]).encode()
    ).hexdigest()[:24]
    root = _cache_root()
    target = root / f"rays_{name}.so"
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            source = Path(scratch) / "rays.c"
            source.write_text(HOST_SOURCE)
            built = Path(scratch) / "rays.so"
            done = subprocess.run(
                [*compiler.split(), *FLAGS, str(source), "-o", str(built), "-lm"],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0 or not built.is_file():
                raise OSError(f"{compiler} did not build the rays: {done.stderr[-800:]}")
            # Whole or not at all: another process of the machine may be building it too.
            built.replace(target)
    held = ctypes.CDLL(str(target))
    for entry in (held.rt_trace_d, held.rt_trace_s):
        entry.restype = None
        entry.argtypes = [
            ctypes.POINTER(_Shot),
            ctypes.c_int,
            ctypes.c_int,
            *[_POINTER] * 4,
        ]
    for entry in (held.rt_first_d, held.rt_first_s):
        entry.restype = None
        entry.argtypes = [ctypes.POINTER(_Shot), ctypes.c_int, *[_POINTER] * 4]
    return held


def library() -> ctypes.CDLL | None:
    """The host's build of the text, made at the first call; ``None`` where there is none."""
    if os.environ.get("REVERBERATE_NO_NATIVE"):
        return None
    if not _state["tried"]:
        _state["tried"] = True
        try:
            _state["library"] = _build()
        except OSError as error:
            _state["why"] = str(error)
    found: ctypes.CDLL | None = _state["library"]
    return found


def available() -> bool:
    """Whether this process can run the C text on its host."""
    return library() is not None


def shot_arrays(
    scene: DerivedScene,
    held: TracerScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings,
) -> dict[str, np.ndarray]:
    """The arrays of one source's launch as the C text reads them, in ``settings``' precision.

    In single precision every coordinate is taken from the scene's middle,
    where a single holds a micrometre.
    """
    single = settings.precision == "single"
    real = np.float32 if single else np.float64
    shift = held.centre if single else np.zeros(3)
    receivers = np.atleast_2d(np.asarray(receivers, dtype=float))
    spheres = receiver_tree(receivers - shift[None, :], settings.receiver_radius_m)
    return {
        "nodes": held.nodes_single if single else held.nodes,
        "tris": held.single if single else held.triangles,
        "meta": held.meta,
        "rec_nodes": np.ascontiguousarray(spheres.packed()),
        "rec_order": np.ascontiguousarray(spheres.order, dtype=np.int32),
        "receivers": np.ascontiguousarray(receivers - shift[None, :], dtype=real),
        "absorption": np.ascontiguousarray(scene.materials.absorption, dtype=np.float64),
        "scattering": np.ascontiguousarray(scene.materials.scattering, dtype=np.float64),
        "source": np.ascontiguousarray(np.asarray(source, dtype=float).reshape(3) - shift),
    }


def shot_scalars(scene: DerivedScene, settings: RaySettings) -> dict[str, Any]:
    """The numbers of a launch, by the names of the C text's ``RtShot``."""
    bands = int(scene.materials.absorption.shape[1])
    channels = channel_count(settings.order)
    # The text holds a ray's bands and harmonics in fixed arrays.
    if bands > 8 or channels > 16:
        raise ValueError(f"the rays hold 8 bands and order 3, asked {bands} and {settings.order}")
    if settings.precision not in ("double", "single"):
        raise ValueError(f"a precision is 'double' or 'single', not {settings.precision!r}")
    return {
        "seed": int(settings.seed),
        "reach": float(settings.sound_speed_m_s * settings.duration_s),
        "bin_length": float(settings.sound_speed_m_s * settings.bin_s),
        "radius": float(settings.receiver_radius_m),
        "floor_energy": float(settings.energy_floor / settings.rays),
        "skip_reach": float(
            settings.sound_speed_m_s * settings.skip_window_s
            if settings.skip_window_s > 0
            else np.inf
        ),
        "total_rays": int(settings.rays),
        "bins": int(np.ceil(settings.duration_s / settings.bin_s)),
        "bands": bands,
        "channels": channels,
        "skip_order": int(settings.skip_specular_order),
        "skip_furniture": int(settings.skip_furniture_bounces),
    }


def counts_on_host(
    scene: DerivedScene,
    held: TracerScene,
    source: np.ndarray,
    receivers: np.ndarray,
    settings: RaySettings,
    *,
    ray_start: int = 0,
    ray_count: int | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The integer histograms of a share of the rays, by the C text on this host's core."""
    built = library()
    if built is None:
        raise OSError(f"the rays' C text is not built here: {_state['why']}")
    arrays = shot_arrays(scene, held, source, receivers, settings)
    scalars = shot_scalars(scene, settings)
    count = settings.rays - ray_start if ray_count is None else ray_count
    cells = int(arrays["receivers"].shape[0])
    energy = np.zeros((cells, scalars["bins"], scalars["bands"]), dtype=np.int64)
    moments = np.zeros(
        (cells, scalars["bins"], scalars["bands"], scalars["channels"]), dtype=np.int64
    )
    hits = np.zeros((cells, scalars["bins"]), dtype=np.int64)
    counted = np.zeros(len(STATS), dtype=np.int64)
    shot = _Shot(**{name: int(a.ctypes.data) for name, a in arrays.items()}, **scalars)
    entry = built.rt_trace_s if settings.precision == "single" else built.rt_trace_d
    started = time.time()
    entry(
        ctypes.byref(shot),
        int(ray_start),
        int(count),
        int(energy.ctypes.data),
        int(moments.ctypes.data),
        int(hits.ctypes.data),
        int(counted.ctypes.data),
    )
    if stats is not None:
        spent = int(round((time.time() - started) * 1e6))
        stats["kernel_us"] = stats.get("kernel_us", 0) + spent
        for name, value in zip(STATS, counted, strict=True):
            stats[name] = stats.get(name, 0) + int(value)
    return energy, moments, hits


def first_hits(
    held: TracerScene, origins: np.ndarray, directions: np.ndarray, *, precision: str = "double"
) -> tuple[np.ndarray, np.ndarray]:
    """Per ray, by the C text: the distance to the first triangle it hits, and the triangle.

    :func:`reverberate.mirror.bvh.nearest` as the text runs it, in either
    precision: what the tests hold the text's walk and its triangle tests
    against, a ray at a time.
    """
    built = library()
    if built is None:
        raise OSError(f"the rays' C text is not built here: {_state['why']}")
    single = precision == "single"
    real = np.float32 if single else np.float64
    shift = held.centre if single else np.zeros(3)
    start = np.ascontiguousarray(np.atleast_2d(np.asarray(origins, dtype=float)) - shift, real)
    along = np.ascontiguousarray(np.atleast_2d(np.asarray(directions, dtype=float)), real)
    nodes = held.nodes_single if single else held.nodes
    tris = held.single if single else held.triangles
    shot = _Shot(
        nodes=int(nodes.ctypes.data), tris=int(tris.ctypes.data), meta=int(held.meta.ctypes.data)
    )
    distance = np.zeros(start.shape[0], dtype=real)
    index = np.zeros(start.shape[0], dtype=np.int32)
    entry = built.rt_first_s if single else built.rt_first_d
    entry(
        ctypes.byref(shot),
        int(start.shape[0]),
        int(start.ctypes.data),
        int(along.ctypes.data),
        int(distance.ctypes.data),
        int(index.ctypes.data),
    )
    return distance.astype(np.float64), index.astype(np.int64)
