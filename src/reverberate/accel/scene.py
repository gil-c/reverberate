"""PFFDTD's ``RoomGeo`` and ``tris_precompute``, transcribed operation for operation.

The GPU voxeliser has to reproduce the CPU one byte for byte, and the CPU one
starts from these arrays: the triangles collapsed over the materials in
alphabetical order, the ones under ``area_eps`` pruned, and the per-triangle
quantities -- centroid, area-scaled normal, unit normal, the three outward
edge normals, the bounding box -- computed in one exact sequence of float64
operations. A different order of the same operations rounds differently,
and a rounding difference on an edge normal is a node that changes side.

So nothing here is simplified. ``normalise`` still divides by the norm
*plus* machine epsilon, the scaled normal is still the mean of three cross
products, and ``dotv`` sums its three products left to right, which is what
``numpy.sum`` does over three elements. Every line names the upstream one it
stands for.

Runs under this project's interpreter (numpy 2), not PFFDTD's, and reads only
the scene JSON. ``tests/test_accel_scene.py`` compares it with PFFDTD's own
``RoomGeo`` when that interpreter is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EPS",
    "Scene",
    "Triangles",
    "dotv",
    "load_scene",
    "normalise",
    "rotation_az_el",
    "tris_precompute",
    "vecnorm",
]

#: ``np.finfo(float).eps``, which ``myfuncs.normalise`` adds to every norm.
EPS = float(np.finfo(np.float64).eps)

#: ``RoomGeo(area_eps=1e-6)``: triangles under this area are pruned.
AREA_EPS = 1e-6


def dotv(a: Any, b: Any) -> Any:
    """``myfuncs.dotv``: ``np.sum(a*b, axis=-1)`` over three components, left to right."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def vecnorm(v: Any) -> Any:
    """``myfuncs.vecnorm``: ``sqrt(dotv(v, v))``."""
    xp = _module_of(v)
    return xp.sqrt(dotv(v, v))


def normalise(v: Any) -> Any:
    """``myfuncs.normalise``: ``v / (|v| + eps)``, the epsilon included."""
    return v / (vecnorm(v) + EPS)[..., None]


def _module_of(array: Any) -> Any:
    if hasattr(array, "get"):
        import cupy

        return cupy
    return np


def rotation_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    """``myfuncs.rotate_az_el_deg``: ``Rz(az) @ Ry(-el)``."""
    thy = np.deg2rad(-el_deg)
    thz = np.deg2rad(az_deg)
    ry = np.array(
        [[np.cos(thy), 0, np.sin(thy)], [0, 1, 0], [-np.sin(thy), 0, np.cos(thy)]], dtype=float
    )
    rz = np.array(
        [[np.cos(thz), -np.sin(thz), 0], [np.sin(thz), np.cos(thz), 0], [0, 0, 1]], dtype=float
    )
    return np.asarray(rz @ ry, dtype=float)


@dataclass(frozen=True)
class Triangles:
    """``tris_precompute``'s structured array, one field per attribute.

    Every array is ``[triangle, ...]`` float64. ``v`` is ``[triangle, 3, 3]``:
    the three vertices ``a, b, c``.
    """

    v: np.ndarray
    nor: np.ndarray
    unor: np.ndarray
    eab_unor: np.ndarray
    ebc_unor: np.ndarray
    eca_unor: np.ndarray
    cent: np.ndarray
    bmin: np.ndarray
    bmax: np.ndarray
    area: np.ndarray

    @property
    def count(self) -> int:
        return int(self.v.shape[0])

    def take(self, indices: np.ndarray) -> Triangles:
        return Triangles(**{name: getattr(self, name)[indices] for name in _TRIANGLE_FIELDS})

    def packed(self) -> np.ndarray:
        """``[triangle, 30]``: what the CUDA kernel reads, in :data:`PACK_ORDER`."""
        return np.concatenate(
            [
                self.v.reshape(self.count, 9),
                self.unor,
                self.cent,
                self.bmin,
                self.bmax,
                self.eab_unor,
                self.ebc_unor,
                self.eca_unor,
            ],
            axis=1,
        ).astype(np.float64)


_TRIANGLE_FIELDS = (
    "v",
    "nor",
    "unor",
    "eab_unor",
    "ebc_unor",
    "eca_unor",
    "cent",
    "bmin",
    "bmax",
    "area",
)

#: Column layout of :meth:`Triangles.packed`, mirrored by the kernel source.
PACK_ORDER = ("v", "unor", "cent", "bmin", "bmax", "eab_unor", "ebc_unor", "eca_unor")


def tris_precompute(pts: np.ndarray, tris: np.ndarray) -> Triangles:
    """``common.tris_precompute.tris_precompute``, line for line."""
    a = pts[tris[:, 0], :]
    b = pts[tris[:, 1], :]
    c = pts[tris[:, 2], :]
    ab = b - a
    bc = c - b
    ca = a - c
    cent = (a + b + c) / 3.0
    nor = (np.cross(ab, -ca) + np.cross(bc, -ab) + np.cross(ca, -bc)) / 3.0
    area = 0.5 * vecnorm(nor)
    eab_unor = normalise(np.cross(ab, nor))
    ebc_unor = normalise(np.cross(bc, nor))
    eca_unor = normalise(np.cross(ca, nor))
    unor = normalise(nor)
    bmin = np.min(np.stack([a, b, c], axis=2), axis=2)
    bmax = np.max(np.stack([a, b, c], axis=2), axis=2)
    return Triangles(
        v=np.concatenate((a[:, None, :], b[:, None, :], c[:, None, :]), axis=1),
        nor=nor,
        unor=unor,
        eab_unor=eab_unor,
        ebc_unor=ebc_unor,
        eca_unor=eca_unor,
        cent=cent,
        bmin=bmin,
        bmax=bmax,
        area=area,
    )


@dataclass(frozen=True)
class Scene:
    """``RoomGeo`` after ``load_json``, ``collapse_tris`` and ``prune_by_area``."""

    pts: np.ndarray
    tris: np.ndarray
    #: int8, ``-1`` for ``_RIGID``.
    mat_ind: np.ndarray
    mat_side: np.ndarray
    #: Alphabetical, ``_RIGID`` last when present.
    mat_str: list[str]
    #: Materials excluding ``_RIGID``.
    nmat: int
    pre: Triangles
    bmin: np.ndarray
    bmax: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray

    @property
    def triangles(self) -> int:
        return int(self.tris.shape[0])


def load_scene(
    model_json: Path | str,
    *,
    az_el: tuple[float, float] = (0.0, 0.0),
    bmin: np.ndarray | None = None,
    bmax: np.ndarray | None = None,
    area_eps: float = AREA_EPS,
) -> Scene:
    """``RoomGeo(json, az_el, area_eps, bmin, bmax)`` as arrays.

    The rotation is skipped when it is the identity: ``pts @ I`` is exact,
    and skipping it keeps a 1.6 million triangle scene from paying for a
    matrix product that changes nothing.
    """
    data = json.loads(Path(model_json).read_text())
    mats_dict: dict[str, Any] = data["mats_hash"]
    mat_str = sorted(mats_dict.keys())
    nmat = len(mat_str)
    if "_RIGID" in mat_str:
        mat_str.remove("_RIGID")
        mat_str.append("_RIGID")
        nmat -= 1
    rotate = any(float(v) != 0.0 for v in az_el)
    rotation = rotation_az_el(*az_el) if rotate else None

    low = np.array([np.inf, np.inf, np.inf]) if bmin is None else np.asarray(bmin, dtype=float)
    high = -np.array([np.inf, np.inf, np.inf]) if bmax is None else np.asarray(bmax, dtype=float)
    points: dict[str, np.ndarray] = {}
    faces: dict[str, np.ndarray] = {}
    for mat in mat_str:
        pts = np.asarray(mats_dict[mat]["pts"], dtype=np.float64)
        if rotation is not None:
            pts = pts @ rotation
        points[mat] = pts
        faces[mat] = np.asarray(mats_dict[mat]["tris"], dtype=np.int64)
    for mat in mat_str:
        low = np.min(np.r_[points[mat], low[None, :]], axis=0)
        high = np.max(np.r_[points[mat], high[None, :]], axis=0)

    sources = np.atleast_2d(np.asarray([s["xyz"] for s in data["sources"]], dtype=np.float64))
    receivers = np.atleast_2d(np.asarray([r["xyz"] for r in data["receivers"]], dtype=np.float64))
    if rotation is not None:
        sources = sources @ rotation
        receivers = receivers @ rotation

    # collapse_tris
    all_pts = np.concatenate([points[mat] for mat in mat_str], axis=0)
    offsets = np.r_[0, np.cumsum([points[mat].shape[0] for mat in mat_str])[:-1]]
    tris = np.concatenate(
        [faces[mat] + offset for mat, offset in zip(mat_str, offsets, strict=True)], axis=0
    )
    if tris.shape[0] < 4:
        raise ValueError(f"{model_json}: a scene needs at least four triangles")
    mat_ind = np.concatenate(
        [np.ones(faces[mat].shape[0], dtype=np.int8) * index for index, mat in enumerate(mat_str)],
        axis=0,
    )
    mat_ind[mat_ind == nmat] = -1
    mat_side = np.concatenate([np.asarray(mats_dict[mat]["sides"]) for mat in mat_str], axis=0)
    if not np.all(mat_side[mat_ind == -1] == 0):
        raise ValueError("every _RIGID triangle must have sidedness 0")
    pre = tris_precompute(all_pts, tris)

    # prune_by_area
    small = np.nonzero(pre.area < area_eps)[0]
    if small.size:
        keep = np.ones(tris.shape[0], dtype=bool)
        keep[small] = False
        tris = tris[keep]
        mat_ind = mat_ind[keep]
        mat_side = mat_side[keep]
        pre = pre.take(np.flatnonzero(keep))
    return Scene(
        pts=all_pts,
        tris=tris,
        mat_ind=mat_ind,
        mat_side=np.asarray(mat_side),
        mat_str=mat_str,
        nmat=nmat,
        pre=pre,
        bmin=low,
        bmax=high,
        sources=sources,
        receivers=receivers,
    )
