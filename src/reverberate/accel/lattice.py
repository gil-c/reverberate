"""The grid, the voxel lattice over it, and which triangles each voxel holds.

Three of PFFDTD's classes, reduced to arrays: ``SimConsts`` (the grid step
and the sample rate from ``fmax`` and the points per wavelength),
``CartGrid`` (the node coordinates along each axis, with a 3.5 cell offset
round the scene) and ``VoxGrid`` (cubes of ``nh`` cells with a one-cell halo,
each holding the triangles whose box meets its own). Every formula is the
upstream one in the upstream order, because the node coordinates and the
voxel boxes are compared against triangle geometry to the last bit.

The triangle index is this project's patch 6 to ``vox_grid_base.py``: every
triangle binned into the voxels its bounding box spans, then the exact
Schwarz-Seidel box test on each candidate pair. Both are vectorised over
pairs here, on whichever array module the caller passes, and the result is
the same CSR list ``fill`` produced: each voxel's triangles ascending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.accel.backend import to_numpy
from reverberate.accel.scene import Triangles, dotv

__all__ = [
    "CartGrid",
    "Lattice",
    "SimConstants",
    "cart_grid",
    "lattice_for",
    "sim_constants",
    "tri_box_hits",
    "triangle_index",
    "voxel_triangles",
]


@dataclass(frozen=True)
class SimConstants:
    """``fdtd.sim_consts.SimConsts`` for a Cartesian grid."""

    c: float
    h: float
    ts: float
    sr: float
    courant: float
    l2: float
    tc: float
    rh: float
    fcc: bool = False


def sim_constants(
    tc: float, rh: float, fmax: float, ppw: float, *, fcc: bool = False
) -> SimConstants:
    """The constants of a run from ``fmax`` and ``PPW``, upstream's own arithmetic."""
    if fcc:
        raise NotImplementedError("the accelerated voxeliser handles the Cartesian grid only")
    c = 343.2 * np.sqrt(tc / 20)
    l2 = 1 / 3
    lam = np.sqrt(l2)
    lam *= 0.999
    l2 = lam * lam
    h = c / (fmax * ppw)
    ts = h / c * lam
    sr = 1 / ts
    return SimConstants(
        c=float(c),
        h=float(h),
        ts=float(ts),
        sr=float(sr),
        courant=float(lam),
        l2=float(l2),
        tc=tc,
        rh=rh,
    )


@dataclass(frozen=True)
class CartGrid:
    """``voxelizer.cart_grid.CartGrid``: node coordinates along each axis."""

    h: float
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return int(self.xv.size), int(self.yv.size), int(self.zv.size)

    @property
    def points(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz


def cart_grid(h: float, offset: float, bmin: np.ndarray, bmax: np.ndarray) -> CartGrid:
    """``CartGrid(h, offset, bmin, bmax)``: ``offset`` cells of margin, ``ceil`` plus one nodes."""
    if offset <= 2.0:
        raise ValueError("the grid needs an offset above two cells for its halo")
    xyzmin0 = np.asarray(bmin, dtype=float) - offset * h
    xyzmax0 = np.asarray(bmax, dtype=float) + offset * h
    sizes = np.asarray(np.ceil((xyzmax0 - xyzmin0) / h), dtype=np.int64) + 1
    nx, ny, nz = (int(sizes[0]), int(sizes[1]), int(sizes[2]))
    xv, yv, zv = np.ogrid[0.0:nx, 0.0:ny, 0.0:nz]
    return CartGrid(
        h=float(h),
        xv=xv.ravel() * h + xyzmin0[0],
        yv=yv.ravel() * h + xyzmin0[1],
        zv=zv.ravel() * h + xyzmin0[2],
    )


@dataclass(frozen=True)
class Lattice:
    """``VoxGrid``'s cubes as arrays, ``[voxel, 3]``, in ``VoxGrid``'s own order.

    ``start`` is the first node of the voxel including its halo, ``count``
    the nodes along each axis including the halo (``nh + 2`` except along
    the last row, which runs to the grid's edge). ``bmin``/``bmax`` are the
    box the triangle test uses: half a cell beyond the halo nodes.
    """

    nh: int
    nvox_xyz: tuple[int, int, int]
    start: np.ndarray
    count: np.ndarray
    bmin: np.ndarray
    bmax: np.ndarray

    @property
    def nvox(self) -> int:
        return int(self.start.shape[0])

    @property
    def core(self) -> np.ndarray:
        """Nodes of the core along each axis, ``[voxel, 3]``: the count without its halo."""
        return np.asarray(self.count - 2, dtype=np.int64)


def lattice_for(grid: CartGrid, nh: int) -> Lattice:
    """``VoxGrid(room_geo, cart_grid, Nh=nh)`` without the million Python objects."""
    if nh <= 3:
        raise ValueError("nh must be above 3, as upstream asserts")
    shape = np.asarray(grid.shape, dtype=np.int64)
    if not np.any(shape >= nh):
        raise ValueError(f"nh {nh} exceeds every axis of {tuple(shape)}")
    nvox_xyz = np.asarray(np.floor((shape - 2) / nh), dtype=np.int64)
    axes = (grid.xv, grid.yv, grid.zv)
    starts_per_axis = []
    lasts_per_axis = []
    for axis in range(3):
        voxels_along = int(nvox_xyz[axis])
        starts = np.arange(voxels_along, dtype=np.int64) * nh
        lasts = starts + nh + 1
        lasts[-1] = int(shape[axis]) - 1
        starts_per_axis.append(starts)
        lasts_per_axis.append(lasts)
    sx, sy, sz = np.meshgrid(*starts_per_axis, indexing="ij")
    lx, ly, lz = np.meshgrid(*lasts_per_axis, indexing="ij")
    start = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)
    last = np.stack([lx.ravel(), ly.ravel(), lz.ravel()], axis=1)
    bmin = np.stack([axes[d][start[:, d]] for d in range(3)], axis=1) - 0.5 * grid.h
    bmax = np.stack([axes[d][last[:, d]] for d in range(3)], axis=1) + 0.5 * grid.h
    count = last - start + 1
    if np.any(count < nh + 2) or np.any(count >= 2 * (nh + 2)):
        raise AssertionError("a voxel's size fell outside upstream's own bounds")
    return Lattice(
        nh=nh,
        nvox_xyz=(int(nvox_xyz[0]), int(nvox_xyz[1]), int(nvox_xyz[2])),
        start=start,
        count=count,
        bmin=bmin,
        bmax=bmax,
    )


def triangle_index(
    lattice: Lattice, pre: Triangles, *, chunk_pairs: int = 8_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """Patch 6's ``build_triangle_index``: candidate ``(voxel, triangle)`` pairs as CSR.

    Voxel ``i``'s candidates are ``tri_ids[offsets[i]:offsets[i + 1]]``,
    ascending by triangle index. A candidate is a triangle whose bounding
    box can touch the voxel's; the exact test is :func:`tri_box_hits`.
    """
    nvox = lattice.nvox
    ntris = pre.count
    if nvox < 2 or ntris < 1:
        return np.zeros(nvox + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    vox_bmin = lattice.bmin.astype(np.float64)
    vox_bmax = lattice.bmax.astype(np.float64)
    axes = [np.unique(vox_bmin[:, d]) for d in range(3)]
    shape = np.array([axis.size for axis in axes], dtype=np.int64)
    if int(np.prod(shape)) != nvox:
        raise AssertionError("the voxel boxes do not form a regular lattice")
    origin = np.array([axis[0] for axis in axes], dtype=np.float64)
    step = np.ones(3, dtype=np.float64)
    for d, axis in enumerate(axes):
        if axis.size < 2:
            continue
        delta = np.diff(axis)
        if not np.allclose(delta, delta[0], rtol=0.0, atol=1e-6):
            raise AssertionError("the voxel lattice is not uniform")
        step[d] = delta[0]
    subs = [np.searchsorted(axes[d], vox_bmin[:, d]) for d in range(3)]
    flat = (subs[0] * shape[1] + subs[1]) * shape[2] + subs[2]
    if not np.array_equal(flat, np.arange(nvox)):
        raise AssertionError("the voxels are not ordered x-major")
    width = (vox_bmax - vox_bmin).max(axis=0)

    tri_bmin = pre.bmin.astype(np.float64)
    tri_bmax = pre.bmax.astype(np.float64)
    slack = 1e-6
    lo = np.ceil((tri_bmin - width - origin) / step - slack).astype(np.int64)
    hi = np.floor((tri_bmax - origin) / step + slack).astype(np.int64)
    np.clip(lo, 0, shape - 1, out=lo)
    np.clip(hi, 0, shape - 1, out=hi)
    spans = hi - lo + 1
    counts = spans[:, 0] * spans[:, 1] * spans[:, 2]
    total = int(counts.sum())
    starts = np.zeros(ntris + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    def pairs_of(first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        offs = np.arange(first, last, dtype=np.int64)
        tri = np.searchsorted(starts, offs, side="right") - 1
        within = offs - starts[tri]
        sx = spans[tri]
        ix = lo[tri, 0] + within // (sx[:, 1] * sx[:, 2])
        iy = lo[tri, 1] + (within // sx[:, 2]) % sx[:, 1]
        iz = lo[tri, 2] + within % sx[:, 2]
        return tri, (ix * shape[1] + iy) * shape[2] + iz

    per_vox = np.zeros(nvox, dtype=np.int64)
    for first in range(0, total, chunk_pairs):
        _, vox_ids = pairs_of(first, min(first + chunk_pairs, total))
        per_vox += np.bincount(vox_ids, minlength=nvox)
    offsets = np.zeros(nvox + 1, dtype=np.int64)
    np.cumsum(per_vox, out=offsets[1:])
    tri_ids = np.empty(total, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for first in range(0, total, chunk_pairs):
        tri, vox_ids = pairs_of(first, min(first + chunk_pairs, total))
        order = np.argsort(vox_ids, kind="stable")
        tri = tri[order]
        vox_ids = vox_ids[order]
        edges = np.flatnonzero(np.diff(vox_ids)) + 1
        run_start = np.zeros(vox_ids.size, dtype=np.int64)
        run_start[edges] = edges
        np.maximum.accumulate(run_start, out=run_start)
        tri_ids[cursor[vox_ids] + np.arange(vox_ids.size) - run_start] = tri
        cursor += np.bincount(vox_ids, minlength=nvox)
    return offsets, tri_ids


def tri_box_hits(
    bbmin: Any, bbmax: Any, v: Any, nor: Any, cent: Any, tbmin: Any, tbmax: Any, xp: Any
) -> Any:
    """``tri_box_intersection_vec`` over pairs: box ``i`` against triangle ``i``.

    Every argument is ``[pair, ...]`` on ``xp``. Returns ``[pair]`` bool.
    """
    p = bbmin
    dp = bbmax - bbmin
    fail1 = xp.any((tbmin > bbmax) | (bbmin > tbmax), axis=-1)
    c = xp.where(nor > 0, dp, xp.zeros_like(dp))
    d1 = dotv(nor, c - cent)
    d2 = dotv(nor, (dp - c) - cent)
    n_dot_p = dotv(nor, p)
    fail2 = (n_dot_p + d1) * (n_dot_p + d2) > 0
    fail3 = xp.zeros(fail2.shape, dtype=bool)
    for q in (0, 1, 2):
        xq = q % 3
        yq = (q + 1) % 3
        zq = (q + 2) % 3
        for i in (0, 1, 2):
            j = (i + 1) % 3
            ei_x = v[:, j, xq] - v[:, i, xq]
            ei_y = v[:, j, yq] - v[:, i, yq]
            vixy_x = 0.5 * (v[:, j, xq] + v[:, i, xq])
            vixy_y = 0.5 * (v[:, j, yq] + v[:, i, yq])
            ne_x = -ei_y
            ne_y = ei_x
            flip = nor[:, zq] < 0
            ne_x = xp.where(flip, ne_x * -1, ne_x)
            ne_y = xp.where(flip, ne_y * -1, ne_y)
            dpx = dp[:, xq] * ne_x
            dpy = dp[:, yq] * ne_y
            deixy = (
                -(ne_x * vixy_x + ne_y * vixy_y)
                + xp.where(dpx > 0, dpx, xp.zeros_like(dpx))
                + xp.where(dpy > 0, dpy, xp.zeros_like(dpy))
            )
            fail3 |= ((ne_x * p[:, xq] + ne_y * p[:, yq]) + deixy) < 0
    return ~(fail1 | fail2 | fail3)


def voxel_triangles(
    lattice: Lattice, pre: Triangles, xp: Any, *, chunk_pairs: int = 4_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """``VoxGrid.fill``: each voxel's triangles by the exact box test, as CSR on the host.

    The candidate pairs come from :func:`triangle_index`; the box test runs
    on ``xp`` a chunk at a time. Ascending triangle order inside a voxel is
    preserved, which is the order ``calc_adj`` visits them in and the order
    the nearest-triangle tie rule depends on.
    """
    offsets, tri_ids = triangle_index(lattice, pre)
    total = int(tri_ids.size)
    if total == 0:
        return offsets, tri_ids
    vox_of_pair = np.repeat(np.arange(lattice.nvox, dtype=np.int64), np.diff(offsets))
    keep = np.zeros(total, dtype=bool)
    v = xp.asarray(pre.v)
    nor = xp.asarray(pre.nor)
    cent = xp.asarray(pre.cent)
    tbmin = xp.asarray(pre.bmin)
    tbmax = xp.asarray(pre.bmax)
    lat_bmin = xp.asarray(lattice.bmin)
    lat_bmax = xp.asarray(lattice.bmax)
    for first in range(0, total, chunk_pairs):
        last = min(first + chunk_pairs, total)
        tri = xp.asarray(tri_ids[first:last])
        vox = xp.asarray(vox_of_pair[first:last])
        hits = tri_box_hits(
            lat_bmin[vox], lat_bmax[vox], v[tri], nor[tri], cent[tri], tbmin[tri], tbmax[tri], xp
        )
        keep[first:last] = to_numpy(hits)
    kept_ids = tri_ids[keep]
    kept_vox = vox_of_pair[keep]
    new_offsets = np.zeros(lattice.nvox + 1, dtype=np.int64)
    np.cumsum(np.bincount(kept_vox, minlength=lattice.nvox), out=new_offsets[1:])
    return new_offsets, kept_ids
