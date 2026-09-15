"""PFFDTD's ``VoxScene.calc_adj`` on the card, byte for byte.

For every node of the grid the voxeliser answers three things: along which
of its six legs a triangle crosses within one cell (the adjacency), which
triangle is nearest (the material and the surface area factor), and whether
a triangle passes so close to the node itself that it is sealed. Upstream
answers them one voxel at a time, one triangle at a time, vectorised over
the voxel's nodes, in twelve processes that spill to disk and are merged
afterwards; on the storey of hssd_0076 at 8 kHz that is 27 minutes of ray
tests and 9 of merging on 40 cores.

The card answers them one thread block per ``(voxel, triangle)`` pair, each
thread taking nodes of the pair's sub-box in turn, and the answers are
combined with order-independent atomics: ``or`` on the six non-adjacency
bits and the sealed bit, ``min`` on the nearest distance, then ``min`` on
the triangle index among the triangles at that distance. Upstream's "first
triangle wins a tie" is "lowest index wins a tie", because it visits them
ascending, and a minimum does not care who computed it first. So the result
does not depend on the block schedule, and it does not depend on how the
voxels are cut into slabs either -- the same fact the CPU's slabbed path
rests on.

**Two things that look like optimisations are not, and are kept.** Upstream
skips a direction for a whole voxel when no node's leg reaches the triangle
within one cell *before* it looks at the nodes that reach it within one cell
and a millionth; that shortcut can change the answer at a node whose hit
falls in that millionth, so the kernel reduces the same "any" over the same
nodes before it marks. And the ray direction is normalised through
``|d| + eps``, which makes a unit vector ``1 / (1 + eps)`` long; the kernel
carries the same factor.

:func:`adjacency_numpy` is the twin: upstream's loop transcribed, run in the
tests against small scenes and against upstream itself when its interpreter
is available. :func:`adjacency_cuda` is the kernel. Both return the same
arrays, in scene space; :func:`finish` applies upstream's sides rule and
surface area factors, :func:`engine_space` its rotation and sort, and
:func:`write_entry` the four files a cache entry holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.accel.backend import raw_kernel, to_numpy
from reverberate.accel.lattice import (
    CartGrid,
    Lattice,
    SimConstants,
    cart_grid,
    lattice_for,
    voxel_triangles,
)
from reverberate.accel.lattice import sim_constants as constants_for
from reverberate.accel.scene import EPS, Scene, dotv, load_scene, normalise

__all__ = [
    "R_EPS",
    "BoundaryNodes",
    "EngineSpace",
    "VoxelisationReport",
    "adjacency_cuda",
    "adjacency_numpy",
    "engine_space",
    "finish",
    "voxelise_scene",
    "write_entry",
]

#: ``vox_scene.R_EPS``: relative to the grid step, the band of a near hit.
R_EPS = 1e-6
#: ``tri_ray_intersection_vec``'s coplanarity epsilon.
CP_EPS = 1e-6
#: The six leg directions, upstream's order: +x, -x, +y, -y, +z, -z.
VV = np.array(
    [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)
#: Boundary nodes per slab on the card: 17 bytes a node of working set.
NODES_PER_SLAB = 120_000_000
#: Non-adjacency bits, then the sealed bit, in ``flags``.
NB_BIT = 1 << 6


@dataclass(frozen=True)
class BoundaryNodes:
    """What the adjacency pass emits, in scene (unrotated) space.

    ``flat`` is ``ix * Ny * Nz + iy * Nz + iz`` over the unrotated grid;
    ``adj`` is ``[node, 6]`` with ``True`` meaning adjacent, as upstream;
    ``tidx`` the nearest triangle, ``-1`` for a node sealed with no hit.
    """

    flat: np.ndarray
    adj: np.ndarray
    tidx: np.ndarray

    @property
    def count(self) -> int:
        return int(self.flat.size)

    @staticmethod
    def concatenate(parts: list[BoundaryNodes]) -> BoundaryNodes:
        if not parts:
            return BoundaryNodes(
                np.zeros(0, np.int64), np.zeros((0, 6), bool), np.zeros(0, np.int32)
            )
        return BoundaryNodes(
            flat=np.concatenate([p.flat for p in parts]),
            adj=np.concatenate([p.adj for p in parts]),
            tidx=np.concatenate([p.tidx for p in parts]),
        )


# --------------------------------------------------------------------------
# the twin: upstream's loop, transcribed
# --------------------------------------------------------------------------


def _leg_distances(
    xyz: np.ndarray, ray_un: np.ndarray, vvh_k: np.ndarray, tri: dict[str, np.ndarray], d_eps: float
) -> np.ndarray:
    """``tri_ray_intersection_vec`` for one triangle and many rays, then ``t - h``."""
    ray_o = xyz - vvh_k
    beta = dotv(ray_un, tri["unor"])
    fail = np.abs(beta) < CP_EPS
    beta = np.where(fail, -EPS, beta)
    t = dotv(tri["unor"], tri["cent"] - ray_o) / beta
    fail |= t < 0
    pop = ray_o + ray_un * t[:, None]
    a, b, c = tri["v"][0], tri["v"][1], tri["v"][2]
    fail |= dotv(pop - 0.5 * (a + b), tri["eab_unor"]) > d_eps
    fail |= dotv(pop - 0.5 * (b + c), tri["ebc_unor"]) > d_eps
    fail |= dotv(pop - 0.5 * (c + a), tri["eca_unor"]) > d_eps
    return np.where(fail, np.inf, t)


def adjacency_numpy(
    scene: Scene,
    grid: CartGrid,
    lattice: Lattice,
    offsets: np.ndarray,
    tri_ids: np.ndarray,
    voxels: np.ndarray | None = None,
) -> BoundaryNodes:
    """``calc_adj``'s ``process_voxel`` over ``voxels`` (default: every non-empty one)."""
    h = grid.h
    hf = h
    vvh = h * VV
    uvv = VV
    d_eps = 1.0e-3 * h
    axes = (grid.xv, grid.yv, grid.zv)
    ny, nz = grid.shape[1], grid.shape[2]
    pre = scene.pre
    if voxels is None:
        voxels = np.flatnonzero(np.diff(offsets) > 0)
    parts: list[BoundaryNodes] = []
    for vox in voxels:
        candidates = tri_ids[offsets[vox] : offsets[vox + 1]]
        if candidates.size == 0:
            continue
        start = lattice.start[vox]
        count = lattice.count[vox]
        shape = (int(count[0]), int(count[1]), int(count[2]))
        ix, iy, iz = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        xyz = np.c_[
            axes[0][start[0] + ix.ravel()],
            axes[1][start[1] + iy.ravel()],
            axes[2][start[2] + iz.ravel()],
        ]
        n = xyz.shape[0]
        ndist = np.full(n, np.inf)
        adj = np.ones((n, 6), dtype=bool)
        nb = np.zeros(n, dtype=bool)
        tidx = np.full(n, -1, dtype=np.int32)
        bp = np.zeros(n, dtype=bool)
        in_mask = np.zeros(shape, dtype=bool)
        in_mask[1:-1, 1:-1, 1:-1] = True
        in_mask_flat = in_mask.ravel()
        for tri_index in candidates:
            tri = {
                "v": pre.v[tri_index],
                "unor": pre.unor[tri_index],
                "cent": pre.cent[tri_index],
                "bmin": pre.bmin[tri_index],
                "bmax": pre.bmax[tri_index],
                "eab_unor": pre.eab_unor[tri_index],
                "ebc_unor": pre.ebc_unor[tri_index],
                "eca_unor": pre.eca_unor[tri_index],
            }
            bb = np.all(xyz >= tri["bmin"] - hf * (1 + R_EPS), axis=-1) & np.all(
                xyz <= tri["bmax"] + hf * (1 + R_EPS), axis=-1
            )
            if not np.any(bb):
                continue
            dtp = np.full(n, np.inf)
            dtp[bb] = dotv(tri["unor"], tri["cent"] - xyz[bb])
            ray_mask = np.abs(dtp) <= hf * (1 + R_EPS)
            if not np.any(ray_mask):
                continue
            tnb = np.zeros(n, dtype=bool)
            for k in range(6):
                ray_un = normalise(uvv[k][None, :])[0]
                hit = np.full(n, np.inf)
                hit[ray_mask] = _leg_distances(xyz[ray_mask], ray_un, vvh[k], tri, d_eps)
                hit -= hf
                hit[hit < -R_EPS * hf] = np.inf
                tnb |= np.abs(hit) <= R_EPS * hf
                hit[tnb] = np.abs(hit[tnb])
                nb |= tnb
                if not np.any(hit <= hf):
                    continue
                hit[hit > (1 + R_EPS) * hf] = np.inf
                ii0 = np.flatnonzero(hit <= (1 + R_EPS) * hf)
                adj[ii0, k] = False
                bp[ii0] = True
                nearer = hit[ii0] < ndist[ii0]
                ndist[ii0[nearer]] = hit[ii0[nearer]]
                tidx[ii0[nearer]] = tri_index
            adj[nb, :] = False
        adj[~in_mask_flat, :] = True
        bp[~in_mask_flat] = False
        tidx[~in_mask_flat] = -1
        qq = np.flatnonzero(np.any(~adj, axis=-1))
        if not np.array_equal(qq, np.flatnonzero(bp)):
            raise AssertionError("boundary nodes and hit nodes disagree, as upstream asserts")
        flat = (
            (start[0] + ix.ravel()[qq]) * (ny * nz)
            + (start[1] + iy.ravel()[qq]) * nz
            + (start[2] + iz.ravel()[qq])
        ).astype(np.int64)
        parts.append(BoundaryNodes(flat=flat, adj=adj[qq], tidx=tidx[qq]))
    return BoundaryNodes.concatenate(parts)


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------

KERNEL = r"""
#define R_EPS 1.0e-6
#define CP_EPS 1.0e-6
#define DBL_EPS 2.220446049250313e-16
#define NB_BIT 64u

struct Tri {
    double v[9];
    double unor[3];
    double cent[3];
    double bmin[3];
    double bmax[3];
    double eab[3];
    double ebc[3];
    double eca[3];
};

__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// tri_ray_intersection_vec for one ray from (xyz - h*dir) along dir, then t - h,
// with upstream's near-hit handling. Returns +inf for no hit.
__device__ double leg(const double* xyz, int k, const Tri& t, double h, double d_eps,
                      bool* near) {
    double dir[3] = {0.0, 0.0, 0.0};
    int axis = k / 2;
    double sign = (k % 2 == 0) ? 1.0 : -1.0;
    dir[axis] = sign;
    // normalise(ray_d): divide by (|d| + eps), |d| = sqrt(1*1 + 0*0 + 0*0) = 1
    double norm = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    double un[3] = {dir[0] / (norm + DBL_EPS), dir[1] / (norm + DBL_EPS),
                    dir[2] / (norm + DBL_EPS)};
    double vvh[3] = {h * dir[0], h * dir[1], h * dir[2]};
    double ro[3] = {xyz[0] - vvh[0], xyz[1] - vvh[1], xyz[2] - vvh[2]};
    double beta = dot3(un, t.unor);
    bool fail = fabs(beta) < CP_EPS;
    if (fail) beta = -DBL_EPS;
    double d[3] = {t.cent[0] - ro[0], t.cent[1] - ro[1], t.cent[2] - ro[2]};
    double tt = dot3(t.unor, d) / beta;
    fail = fail || (tt < 0.0);
    double pop[3] = {ro[0] + un[0] * tt, ro[1] + un[1] * tt, ro[2] + un[2] * tt};
    const double* a = t.v;
    const double* b = t.v + 3;
    const double* c = t.v + 6;
    double e[3];
    e[0] = pop[0] - 0.5 * (a[0] + b[0]);
    e[1] = pop[1] - 0.5 * (a[1] + b[1]);
    e[2] = pop[2] - 0.5 * (a[2] + b[2]);
    fail = fail || (dot3(e, t.eab) > d_eps);
    e[0] = pop[0] - 0.5 * (b[0] + c[0]);
    e[1] = pop[1] - 0.5 * (b[1] + c[1]);
    e[2] = pop[2] - 0.5 * (b[2] + c[2]);
    fail = fail || (dot3(e, t.ebc) > d_eps);
    e[0] = pop[0] - 0.5 * (c[0] + a[0]);
    e[1] = pop[1] - 0.5 * (c[1] + a[1]);
    e[2] = pop[2] - 0.5 * (c[2] + a[2]);
    fail = fail || (dot3(e, t.eca) > d_eps);
    double inf = __longlong_as_double(0x7ff0000000000000LL);
    double hd = fail ? inf : tt;
    hd -= h;
    if (hd < -R_EPS * h) hd = inf;
    *near = fabs(hd) <= R_EPS * h;
    if (*near) hd = fabs(hd);
    return hd;
}

extern "C" __global__ void adjacency(
    const double* __restrict__ xv, const double* __restrict__ yv, const double* __restrict__ zv,
    const double* __restrict__ tris,
    const int* __restrict__ pair_slot, const int* __restrict__ pair_tri, int npairs,
    const int* __restrict__ slot_start, const int* __restrict__ slot_count,
    const long long* __restrict__ slot_offset,
    double h, int mode,
    unsigned int* __restrict__ flags, unsigned long long* __restrict__ ndist,
    int* __restrict__ tidx)
{
    __shared__ Tri t;
    __shared__ int lo[3], hi[3], start[3], count[3];
    __shared__ long long base;
    __shared__ int s_any;
    __shared__ int tri_index;
    int pair = blockIdx.x;
    if (pair >= npairs) return;
    if (threadIdx.x == 0) {
        int slot = pair_slot[pair];
        tri_index = pair_tri[pair];
        const double* src = tris + (long long)tri_index * 30;
        double* dst = (double*)&t;
        for (int i = 0; i < 30; ++i) dst[i] = src[i];
        for (int a = 0; a < 3; ++a) {
            start[a] = slot_start[slot * 3 + a];
            count[a] = slot_count[slot * 3 + a];
        }
        base = slot_offset[slot];
        double margin = h * (1.0 + R_EPS);
        const double* axes[3] = {xv, yv, zv};
        for (int a = 0; a < 3; ++a) {
            int l = count[a], hh = -1;
            for (int i = 0; i < count[a]; ++i) {
                double x = axes[a][start[a] + i];
                bool in = (x >= t.bmin[a] - margin) && (x <= t.bmax[a] + margin);
                if (in) { if (l == count[a]) l = i; hh = i; }
            }
            lo[a] = l; hi[a] = hh;
        }
        s_any = 0;
    }
    __syncthreads();
    if (hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2]) return;
    int sx = hi[0] - lo[0] + 1, sy = hi[1] - lo[1] + 1, sz = hi[2] - lo[2] + 1;
    int nsub = sx * sy * sz;
    double margin = h * (1.0 + R_EPS);
    double d_eps = 1.0e-3 * h;
    int cx = count[0] - 2, cy = count[1] - 2, cz = count[2] - 2;
    for (int k = 0; k < 6; ++k) {
        int any = 0;
        for (int n = threadIdx.x; n < nsub; n += blockDim.x) {
            int lz = lo[2] + n % sz;
            int ly = lo[1] + (n / sz) % sy;
            int lx = lo[0] + n / (sz * sy);
            double xyz[3] = {xv[start[0] + lx], yv[start[1] + ly], zv[start[2] + lz]};
            bool bb = true;
            for (int a = 0; a < 3; ++a)
                bb = bb && (xyz[a] >= t.bmin[a] - margin) && (xyz[a] <= t.bmax[a] + margin);
            if (!bb) continue;
            double dd[3] = {t.cent[0] - xyz[0], t.cent[1] - xyz[1], t.cent[2] - xyz[2]};
            double dtp = dot3(t.unor, dd);
            if (!(fabs(dtp) <= margin)) continue;
            bool near;
            double hd = leg(xyz, k, t, h, d_eps, &near);
            bool core = lx >= 1 && lx <= count[0] - 2 && ly >= 1 && ly <= count[1] - 2
                        && lz >= 1 && lz <= count[2] - 2;
            if (near && core && mode == 0) {
                long long node = base + ((long long)(lx - 1) * cy + (ly - 1)) * cz + (lz - 1);
                atomicOr(&flags[node], NB_BIT);
            }
            if (hd <= h) any = 1;
        }
        if (any) s_any = 1;
        __syncthreads();
        if (s_any) {
            for (int n = threadIdx.x; n < nsub; n += blockDim.x) {
                int lz = lo[2] + n % sz;
                int ly = lo[1] + (n / sz) % sy;
                int lx = lo[0] + n / (sz * sy);
                bool core = lx >= 1 && lx <= count[0] - 2 && ly >= 1 && ly <= count[1] - 2
                            && lz >= 1 && lz <= count[2] - 2;
                if (!core) continue;
                double xyz[3] = {xv[start[0] + lx], yv[start[1] + ly], zv[start[2] + lz]};
                bool bb = true;
                for (int a = 0; a < 3; ++a)
                    bb = bb && (xyz[a] >= t.bmin[a] - margin) && (xyz[a] <= t.bmax[a] + margin);
                if (!bb) continue;
                double dd[3] = {t.cent[0] - xyz[0], t.cent[1] - xyz[1], t.cent[2] - xyz[2]};
                double dtp = dot3(t.unor, dd);
                if (!(fabs(dtp) <= margin)) continue;
                bool near;
                double hd = leg(xyz, k, t, h, d_eps, &near);
                if (!(hd <= (1.0 + R_EPS) * h)) continue;
                long long node = base + ((long long)(lx - 1) * cy + (ly - 1)) * cz + (lz - 1);
                if (mode == 0) {
                    atomicOr(&flags[node], 1u << k);
                    atomicMin(&ndist[node], (unsigned long long)__double_as_longlong(hd));
                } else {
                    if ((unsigned long long)__double_as_longlong(hd) == ndist[node])
                        atomicMin(&tidx[node], tri_index);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) s_any = 0;
        __syncthreads();
    }
}
"""


def _slabs(lattice: Lattice, offsets: np.ndarray, nodes_per_slab: int) -> list[np.ndarray]:
    """Non-empty voxels cut into runs whose core nodes fit the card's working set."""
    nonempty = np.flatnonzero(np.diff(offsets) > 0)
    cores = np.prod(lattice.core[nonempty], axis=1)
    slabs: list[np.ndarray] = []
    current: list[int] = []
    held = 0
    for vox, size in zip(nonempty, cores, strict=True):
        if current and held + int(size) > nodes_per_slab:
            slabs.append(np.asarray(current, dtype=np.int64))
            current, held = [], 0
        current.append(int(vox))
        held += int(size)
    if current:
        slabs.append(np.asarray(current, dtype=np.int64))
    return slabs


def adjacency_cuda(
    scene: Scene,
    grid: CartGrid,
    lattice: Lattice,
    offsets: np.ndarray,
    tri_ids: np.ndarray,
    *,
    nodes_per_slab: int = NODES_PER_SLAB,
    threads: int = 256,
    say: Any = None,
) -> BoundaryNodes:
    """Every boundary node of the grid, from the kernel, slab by slab."""
    import cupy

    kernel = raw_kernel(KERNEL, "adjacency")
    xv, yv, zv = (cupy.asarray(a, dtype=cupy.float64) for a in (grid.xv, grid.yv, grid.zv))
    tris = cupy.asarray(scene.pre.packed())
    ny, nz = grid.shape[1], grid.shape[2]
    inf_bits = np.uint64(0x7FF0000000000000)
    parts: list[BoundaryNodes] = []
    slabs = _slabs(lattice, offsets, nodes_per_slab)
    started = time.time()
    for number, voxels in enumerate(slabs):
        start = lattice.start[voxels].astype(np.int32)
        count = lattice.count[voxels].astype(np.int32)
        core = lattice.core[voxels]
        sizes = np.prod(core, axis=1)
        slot_offset = np.zeros(voxels.size + 1, dtype=np.int64)
        np.cumsum(sizes, out=slot_offset[1:])
        total = int(slot_offset[-1])
        counts = np.diff(offsets)[voxels]
        pair_slot = np.repeat(np.arange(voxels.size, dtype=np.int32), counts)
        pair_tri = np.concatenate([tri_ids[offsets[v] : offsets[v + 1]] for v in voxels]).astype(
            np.int32
        )
        npairs = int(pair_tri.size)

        flags = cupy.zeros(total, dtype=cupy.uint32)
        ndist = cupy.full(total, inf_bits, dtype=cupy.uint64)
        tidx = cupy.full(total, np.iinfo(np.int32).max, dtype=cupy.int32)
        args_static = (
            xv,
            yv,
            zv,
            tris,
            cupy.asarray(pair_slot),
            cupy.asarray(pair_tri),
            np.int32(npairs),
            cupy.asarray(start.reshape(-1)),
            cupy.asarray(count.reshape(-1)),
            cupy.asarray(slot_offset[:-1]),
            np.float64(grid.h),
        )
        for mode in (0, 1):
            kernel((npairs,), (threads,), (*args_static, np.int32(mode), flags, ndist, tidx))
        cupy.cuda.Stream.null.synchronize()

        boundary = cupy.flatnonzero(flags != 0)
        if boundary.size:
            bits = flags[boundary]
            sealed = (bits & NB_BIT) != 0
            bits = cupy.where(sealed, cupy.uint32(0x3F), bits & cupy.uint32(0x3F))
            adj = cupy.stack([((bits >> k) & 1) == 0 for k in range(6)], axis=1)
            near = tidx[boundary]
            near = cupy.where(near == np.iinfo(np.int32).max, cupy.int32(-1), near)
            slot = cupy.searchsorted(cupy.asarray(slot_offset), boundary, side="right") - 1
            local = boundary - cupy.asarray(slot_offset)[slot]
            core_d = cupy.asarray(core)
            cz = core_d[slot, 2]
            cy = core_d[slot, 1]
            lz = local % cz
            ly = (local // cz) % cy
            lx = local // (cz * cy)
            start_d = cupy.asarray(start.astype(np.int64))
            ix = start_d[slot, 0] + 1 + lx
            iy = start_d[slot, 1] + 1 + ly
            iz = start_d[slot, 2] + 1 + lz
            flat = ix * (ny * nz) + iy * nz + iz
            parts.append(
                BoundaryNodes(
                    flat=to_numpy(flat).astype(np.int64),
                    adj=to_numpy(adj).astype(bool),
                    tidx=to_numpy(near).astype(np.int32),
                )
            )
        del flags, ndist, tidx
        if say is not None:
            say(
                f"  slab {number + 1}/{len(slabs)}: {voxels.size} voxels, {npairs} pairs,"
                f" {sum(p.count for p in parts)} boundary nodes so far,"
                f" {time.time() - started:.0f} s"
            )
    return BoundaryNodes.concatenate(parts)


# --------------------------------------------------------------------------
# sides, surface area factors, engine space, the files
# --------------------------------------------------------------------------


def _out_of_memory(error: BaseException) -> bool:
    """cupy's ``OutOfMemoryError`` without importing cupy: the host finishes the chunk."""
    return "memory" in repr(error).lower()


#: Boundary nodes a chunk of :func:`finish` holds on the device: about 200
#: bytes a node of working set, so 2 GB, which any card has beside a slab.
FINISH_CHUNK = 10_000_000


def finish(
    nodes: BoundaryNodes, scene: Scene, grid: CartGrid, xp: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``calc_adj`` after the merge: materials with sides, and the area factors.

    Returns ``(bn_ixyz, adj_bn, mat_bn, saf_bn)`` in scene space, on the host.
    Every row is its own computation, so the nodes are taken in chunks and
    a card short of memory falls back to the host for the rest.
    """
    if nodes.count == 0:
        return (
            np.zeros(0, np.int64),
            np.zeros((0, 6), bool),
            np.zeros(0, np.int8),
            np.zeros(0, np.float64),
        )
    if int(np.unique(nodes.flat).size) != nodes.count:
        raise AssertionError("a boundary node was emitted twice")
    if xp is not np and nodes.count > FINISH_CHUNK:
        parts = []
        for start in range(0, nodes.count, FINISH_CHUNK):
            piece = BoundaryNodes(
                flat=nodes.flat[start : start + FINISH_CHUNK],
                adj=nodes.adj[start : start + FINISH_CHUNK],
                tidx=nodes.tidx[start : start + FINISH_CHUNK],
            )
            try:
                parts.append(_finish_block(piece, scene, grid, xp))
            except Exception as short:
                if not _out_of_memory(short):
                    raise
                parts.append(_finish_block(piece, scene, grid, np))
        return tuple(np.concatenate([part[k] for part in parts]) for k in range(4))  # type: ignore[return-value]
    return _finish_block(nodes, scene, grid, xp)


def _finish_block(
    nodes: BoundaryNodes, scene: Scene, grid: CartGrid, xp: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, nz = grid.shape[1], grid.shape[2]
    flat = xp.asarray(nodes.flat)
    adj = xp.asarray(nodes.adj)
    tidx = xp.asarray(nodes.tidx.astype(np.int64))
    iz = flat % nz
    iy = (flat - iz) // nz % ny
    ix = ((flat - iz) // nz - iy) // ny
    xv, yv, zv = (xp.asarray(a) for a in (grid.xv, grid.yv, grid.zv))
    xyz = xp.stack([xv[ix], yv[iy], zv[iz]], axis=1)
    cent = xp.asarray(scene.pre.cent)[tidx]
    unor = xp.asarray(scene.pre.unor)[tidx]
    dv = dotv(xyz - cent, unor)
    mat_ind = xp.asarray(scene.mat_ind)
    mat_side = xp.asarray(scene.mat_side)
    mat_bn = mat_ind[tidx]
    side = mat_side[tidx]
    back_side = ((dv > 0) & (side == 1)) | ((dv < 0) & (side == 2))
    mat_bn = xp.where(back_side, xp.int8(-1), mat_bn).astype(xp.int8)
    adj = adj & ~back_side[:, None]
    mat_bn = xp.where(xp.all(~adj, axis=-1), xp.int8(-1), mat_bn).astype(xp.int8)
    uvv = xp.asarray(VV)
    saf_bn = xp.zeros(nodes.count, dtype=xp.float64)
    for j in range(0, 6, 2):
        saf = xp.abs(dotv(uvv[j], unor))
        saf_bn = saf_bn + (~adj[:, j] | ~adj[:, j + 1]) * saf
    return to_numpy(flat), to_numpy(adj), to_numpy(mat_bn), to_numpy(saf_bn)


@dataclass(frozen=True)
class EngineSpace:
    """``vox_out.h5``'s content after ``rotate_sim_data`` and ``sort_sim_data``."""

    bn_ixyz: np.ndarray
    adj_bn: np.ndarray
    mat_bn: np.ndarray
    saf_bn: np.ndarray
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray
    shape: tuple[int, int, int]
    order: np.ndarray


def engine_space(
    grid: CartGrid,
    bn_ixyz: np.ndarray,
    adj_bn: np.ndarray,
    mat_bn: np.ndarray,
    saf_bn: np.ndarray,
    xp: Any,
) -> EngineSpace:
    """Axes in descending extent, the biggest outermost, and every row sorted."""
    if xp is not np:
        try:
            return _engine_space(grid, bn_ixyz, adj_bn, mat_bn, saf_bn, xp)
        except Exception as short:
            if not _out_of_memory(short):
                raise
            return _engine_space(grid, bn_ixyz, adj_bn, mat_bn, saf_bn, np)
    return _engine_space(grid, bn_ixyz, adj_bn, mat_bn, saf_bn, np)


def _engine_space(
    grid: CartGrid,
    bn_ixyz: np.ndarray,
    adj_bn: np.ndarray,
    mat_bn: np.ndarray,
    saf_bn: np.ndarray,
    xp: Any,
) -> EngineSpace:
    nx, ny, nz = grid.shape
    tr = np.argsort(np.array([nx, ny, nz]))[::-1]
    dims = np.array([nx, ny, nz])[tr]
    flat = xp.asarray(bn_ixyz)
    iz = flat % nz
    iy = (flat - iz) // nz % ny
    ix = ((flat - iz) // nz - iy) // ny
    subs = [ix, iy, iz]
    moved = (subs[tr[0]] * dims[1] + subs[tr[1]]) * dims[2] + subs[tr[2]]
    ivv = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    jj = np.array([int(np.flatnonzero(np.all(v[tr] == ivv, axis=-1))[0]) for v in ivv])
    ia = np.argsort(jj)
    adj = xp.asarray(adj_bn)[:, xp.asarray(ia)]
    order = xp.argsort(moved)
    axes = [grid.xv, grid.yv, grid.zv]
    return EngineSpace(
        bn_ixyz=to_numpy(moved[order]).astype(np.int64),
        adj_bn=to_numpy(adj[order]).astype(bool),
        mat_bn=to_numpy(xp.asarray(mat_bn)[order]).astype(np.int8),
        saf_bn=to_numpy(xp.asarray(saf_bn)[order]).astype(np.float64),
        xv=axes[tr[0]],
        yv=axes[tr[1]],
        zv=axes[tr[2]],
        shape=(int(dims[0]), int(dims[1]), int(dims[2])),
        order=tr,
    )


def write_entry(
    out_dir: Path,
    space: EngineSpace,
    grid: CartGrid,
    constants: SimConstants,
    mat_files: dict[str, str],
    mat_folder: Path,
    mat_str: list[str],
) -> None:
    """The four files of a cache entry: ``vox_out``, ``cart_grid``, ``sim_consts``, ``sim_mats``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / "vox_out.h5", "w") as handle:
        handle.create_dataset("bn_ixyz", data=space.bn_ixyz)
        handle.create_dataset("adj_bn", data=space.adj_bn)
        handle.create_dataset("mat_bn", data=space.mat_bn)
        handle.create_dataset("saf_bn", data=space.saf_bn)
        handle.create_dataset("xv", data=space.xv)
        handle.create_dataset("yv", data=space.yv)
        handle.create_dataset("zv", data=space.zv)
        handle.create_dataset("h", data=np.float64(grid.h))
        handle.create_dataset("Nx", data=np.int64(space.shape[0]))
        handle.create_dataset("Ny", data=np.int64(space.shape[1]))
        handle.create_dataset("Nz", data=np.int64(space.shape[2]))
        handle.create_dataset("Nb", data=np.int64(space.bn_ixyz.size))
    compressed = {"compression": "gzip", "compression_opts": 9}
    with h5py.File(out_dir / "cart_grid.h5", "w") as handle:
        handle.create_dataset("xv", data=grid.xv, **compressed)
        handle.create_dataset("yv", data=grid.yv, **compressed)
        handle.create_dataset("zv", data=grid.zv, **compressed)
        handle.create_dataset("h", data=np.float64(grid.h))
    with h5py.File(out_dir / "sim_consts.h5", "w") as handle:
        handle.create_dataset("c", data=np.float64(constants.c))
        handle.create_dataset("h", data=np.float64(constants.h))
        handle.create_dataset("Ts", data=np.float64(constants.ts))
        handle.create_dataset("SR", data=np.float64(constants.sr))
        handle.create_dataset("l", data=np.float64(constants.courant))
        handle.create_dataset("l2", data=np.float64(constants.l2))
        handle.create_dataset("fcc_flag", data=np.int8(constants.fcc))
        handle.create_dataset("Tc", data=np.float64(constants.tc))
        handle.create_dataset("rh", data=np.float64(constants.rh))
    labels = [m for m in mat_str if m != "_RIGID"]
    labels.sort()
    if labels != sorted(mat_files):
        raise ValueError(f"materials {labels} do not match the files given {sorted(mat_files)}")
    defs = []
    for label in labels:
        with h5py.File(Path(mat_folder) / mat_files[label], "r") as handle:
            defs.append(np.asarray(handle["DEF"][()]))
    with h5py.File(out_dir / "sim_mats.h5", "w") as handle:
        handle.create_dataset("Nmat", data=np.int8(len(defs)))
        branches = np.zeros(len(defs), dtype=np.int8)
        for index, definition in enumerate(defs):
            if definition.ndim != 2 or definition.shape[1] != 3:
                raise ValueError(f"{labels[index]}: DEF must be [branch, 3]")
            handle.create_dataset(f"mat_{index:02d}_DEF", data=definition)
            branches[index] = definition.shape[0]
        handle.create_dataset("Mb", data=branches)


@dataclass(frozen=True)
class VoxelisationReport:
    """What one voxelisation produced, in the keys the CPU child reports."""

    timings: dict[str, float]
    h_m: float
    sample_rate_hz: float
    grid_shape: list[int]
    grid_points: int
    boundary_nodes: int
    triangles: int
    bmin: list[float]
    bmax: list[float]
    backend: str
    pockets: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "timings": self.timings,
            "slabs": 1,
            "voxelisation_s": round(
                sum(self.timings.get(k, 0.0) for k in ("vox_grid_fill_s", "vox_scene_adj_s")), 3
            ),
            "h_m": self.h_m,
            "sample_rate_hz": self.sample_rate_hz,
            "grid_shape": self.grid_shape,
            "grid_points": self.grid_points,
            "boundary_nodes": self.boundary_nodes,
            "triangles": self.triangles,
            "bmin": self.bmin,
            "bmax": self.bmax,
            "voxeliser": f"reverberate.accel ({self.backend})",
            "numpy": np.__version__,
            "pockets": self.pockets,
        }


def voxelise_scene(
    model_json: Path,
    out_dir: Path,
    *,
    mat_folder: Path,
    mat_files: dict[str, str],
    fmax: float,
    ppw: float,
    nh: int,
    tc: float = 20.0,
    rh: float = 50.0,
    xp: Any,
    offset: float = 3.5,
    say: Any = None,
    nodes_per_slab: int = NODES_PER_SLAB,
    seal_pockets: bool = True,
    pockets_max_cells: int = 2_000_000_000,
) -> VoxelisationReport:
    """Everything ``_child_voxelise`` does for a Cartesian grid, on ``xp``.

    ``xp`` is ``cupy`` for the card and ``numpy`` for the twin; with numpy the
    adjacency runs upstream's loop, which is what the tests compare against.
    """
    say = say or (lambda message: None)
    timings: dict[str, float] = {}
    t0 = time.time()
    scene = load_scene(model_json)
    timings["room_geo_s"] = round(time.time() - t0, 3)
    constants = constants_for(tc, rh, fmax, ppw)
    grid = cart_grid(constants.h, offset, scene.bmin, scene.bmax)
    lattice = lattice_for(grid, nh)
    say(
        f"grid {grid.shape} ({grid.points:.3g} nodes), {scene.triangles} triangles,"
        f" {lattice.nvox} voxels of {nh}"
    )
    t0 = time.time()
    offsets, tri_ids = voxel_triangles(lattice, scene.pre, xp)
    timings["vox_grid_fill_s"] = round(time.time() - t0, 3)
    say(
        f"fill: {tri_ids.size} (voxel, triangle) pairs in"
        f" {int(np.count_nonzero(np.diff(offsets)))} non-empty voxels,"
        f" {timings['vox_grid_fill_s']:.1f} s"
    )
    t0 = time.time()
    if xp is np:
        nodes = adjacency_numpy(scene, grid, lattice, offsets, tri_ids)
    else:
        nodes = adjacency_cuda(
            scene, grid, lattice, offsets, tri_ids, nodes_per_slab=nodes_per_slab, say=say
        )
    bn_ixyz, adj_bn, mat_bn, saf_bn = finish(nodes, scene, grid, xp)
    timings["vox_scene_adj_s"] = round(time.time() - t0, 3)
    say(f"adjacency: {nodes.count} boundary nodes, {timings['vox_scene_adj_s']:.1f} s")
    t0 = time.time()
    space = engine_space(grid, bn_ixyz, adj_bn, mat_bn, saf_bn, xp)
    timings["to_engine_space_s"] = round(time.time() - t0, 3)
    t0 = time.time()
    write_entry(out_dir, space, grid, constants, mat_files, mat_folder, scene.mat_str)
    timings["vox_scene_save_s"] = round(time.time() - t0, 3)
    # The chain's last word on a grid, applied here as it applies it: every
    # air component smaller than a room is sealed, and a grid too large to
    # label on this host is left as it is and the record says so.
    t0 = time.time()
    pockets_record: dict[str, Any] = {"sealed": False, "why": "not asked for"}
    if seal_pockets:
        from reverberate.wave.pockets import seal_in_place

        pockets_record = seal_in_place(Path(out_dir) / "vox_out.h5", max_cells=pockets_max_cells)
    timings["pockets_s"] = round(time.time() - t0, 3)
    say(
        f"pockets: {pockets_record.get('seal', pockets_record).get('pockets_sealed', 0)} sealed"
        f" ({timings['pockets_s']:.1f} s)"
        if pockets_record.get("sealed", True)
        else f"pockets: not sealed, {pockets_record.get('why')}"
    )
    return VoxelisationReport(
        timings=timings,
        h_m=constants.h,
        sample_rate_hz=constants.sr,
        grid_shape=list(grid.shape),
        grid_points=grid.points,
        boundary_nodes=int(pockets_record.get("seal", {}).get("boundary_nodes_after", nodes.count)),
        triangles=scene.triangles,
        bmin=[float(v) for v in scene.bmin],
        bmax=[float(v) for v in scene.bmax],
        backend="numpy" if xp is np else "cupy",
        pockets=pockets_record,
    )


def read_vox_out(path: Path) -> dict[str, np.ndarray]:
    """Every dataset of a ``vox_out.h5``, for comparisons."""
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name][()]) for name in handle}


def entries_identical(a: Path, b: Path) -> dict[str, Any]:
    """Compare two cache entries dataset by dataset; the report names any difference."""
    report: dict[str, Any] = {}
    for name in ("vox_out.h5", "cart_grid.h5", "sim_consts.h5", "sim_mats.h5"):
        left = read_vox_out(Path(a) / name)
        right = read_vox_out(Path(b) / name)
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                report[f"{name}/{key}"] = "missing on one side"
                continue
            x, y = left[key], right[key]
            if x.shape != y.shape:
                report[f"{name}/{key}"] = f"shape {x.shape} vs {y.shape}"
            elif not np.array_equal(x, y):
                differing = int(np.count_nonzero(x != y)) if x.shape == y.shape else -1
                report[f"{name}/{key}"] = f"{differing} of {x.size} elements differ"
    report["identical"] = not any(k != "identical" for k in report)
    return report
