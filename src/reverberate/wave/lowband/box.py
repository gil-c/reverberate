"""A shoebox room written as the engine's arrays, on either grid: what the solver is proved on.

No voxeliser is needed for a room whose walls lie between node planes: a
node of the room reads the neighbours that are in the room, a node outside
reads those outside, and every link across a wall is cut on both sides, as
a voxelised surface cuts them. The arrays are the engine's (``vox_out.h5``
after PFFDTD's rotation and, on the face centred grid, its fold), so the
solver is given here exactly what it is given for a dwelling, and the grid
is the :class:`reverberate.wave.comms.Grid` sources and receivers are placed
with.

Used by the tests on ``numpy`` and by ``python -m reverberate.wave.lowband
verify`` on a card.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from reverberate.accel.lattice import sim_constants
from reverberate.wave.comms import Grid
from reverberate.wave.lowband.problem import EngineArrays
from reverberate.wave.lowband.scheme import BACK_OFF, Scheme

__all__ = ["TEST_BRANCHES", "box_arrays", "constants_of", "write_entry"]

#: Three branches of a fitted material of the project, ``D E F``: a wall that absorbs.
TEST_BRANCHES = np.array(
    [
        [4.24161203e-01, 1.17781161e02, 6.54109887e04],
        [1.96274083e-02, 2.18005694e01, 4.84286889e04],
        [9.50257301e-04, 8.44376390e00, 1.50058618e05],
    ]
)

#: PFFDTD's fold exchanges these neighbours of a node mirrored in ``y``.
_FOLD_SWAPS = ((0, 6), (1, 7), (2, 9), (3, 8))


def constants_of(scheme: Scheme, fmax_hz: float, ppw: float | None = None) -> dict[str, float]:
    """``SimConsts`` for either grid: the step, the time step and the Courant number."""
    points = scheme.ppw if ppw is None else ppw
    if not scheme.fcc:
        held = sim_constants(20.0, 50.0, fmax_hz, points)
        return {"c": held.c, "h": held.h, "ts": held.ts, "l": held.courant, "l2": held.l2}
    c = 343.2
    lam = float(np.sqrt(1.0) * BACK_OFF)
    h = c / (fmax_hz * points)
    return {"c": c, "h": float(h), "ts": float(h / c * lam), "l": lam, "l2": lam * lam}


def box_arrays(
    scheme: Scheme,
    shape: tuple[int, int, int],
    *,
    room: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
    fmax_hz: float = 1500.0,
    ppw: float | None = None,
    lossy: bool = False,
    branches: np.ndarray | None = None,
) -> tuple[EngineArrays, Grid]:
    """The engine's arrays of a box of ``shape`` nodes holding one room, and its grid.

    ``shape`` is strictly descending, so PFFDTD's rotation leaves it alone,
    and its ``y`` is even on the face centred grid, which is folded.
    ``room`` is the first and the last node of the room along each axis, at
    least three nodes inside the box; ``None`` is no room, free air out to
    the absorbing layer. Walls are rigid unless ``lossy``, which gives every
    node of the room's skin material 0 with ``branches``.
    """
    nx, ny, nz = shape
    if not nx > ny > nz:
        raise ValueError("the box's shape must be strictly descending")
    fcc = scheme.fcc
    if fcc and ny % 2:
        raise ValueError("the face centred grid needs an even y")
    held = constants_of(scheme, fmax_hz, ppw)
    grid = Grid(
        h=held["h"],
        Ts=held["ts"],
        l2=held["l2"],
        fcc_flag=2 if fcc else 0,
        xv=np.arange(nx) * held["h"],
        yv=np.arange(ny) * held["h"],
        zv=np.arange(nz) * held["h"],
    )
    neighbours = np.array(scheme.neighbours, dtype=np.int64)
    if room is None:
        subs = np.zeros((0, 3), dtype=np.int64)
        adjacency = np.zeros((0, len(scheme.neighbours)), dtype=bool)
        inside_node = np.zeros(0, dtype=bool)
    else:
        lo, hi = np.asarray(room[0]), np.asarray(room[1])
        if (lo < 3).any() or (hi > np.array(shape) - 4).any() or (hi <= lo).any():
            raise ValueError("the room must lie at least three nodes inside the box")
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        subs = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
        if fcc:
            subs = subs[subs.sum(axis=1) % 2 == 0]
        subs = subs[((subs >= 1) & (subs <= np.array(shape) - 2)).all(axis=1)]

        def in_room(points: np.ndarray) -> np.ndarray:
            return np.asarray(((points >= lo) & (points <= hi)).all(axis=-1))

        inside_node = in_room(subs)
        same_side = in_room(subs[:, None, :] + neighbours[None, :, :]) == inside_node[:, None]
        on_skin = ~same_side.all(axis=1)
        subs, adjacency, inside_node = subs[on_skin], same_side[on_skin], inside_node[on_skin]
    material = np.where(inside_node & lossy, 0, -1).astype(np.int8)
    saf = (~adjacency).sum(axis=1).astype(np.float64)
    engine_shape = (nx, ny // 2 + 1, nz) if fcc else (nx, ny, nz)
    if fcc:
        # PFFDTD's fold: the upper half in y mirrored onto the lower, its neighbours exchanged.
        mirrored = subs[:, 1] >= ny / 2
        folded_y = np.where(mirrored, ny - subs[:, 1] - 1, subs[:, 1])
        index = (subs[:, 0] * engine_shape[1] + folded_y) * nz + subs[:, 2]
        for a, b in _FOLD_SWAPS:
            held_a = adjacency[mirrored, a].copy()
            adjacency[mirrored, a] = adjacency[mirrored, b]
            adjacency[mirrored, b] = held_a
    else:
        index = (subs[:, 0] * ny + subs[:, 1]) * nz + subs[:, 2]
    order = np.argsort(index, kind="stable")
    arrays = EngineArrays(
        shape=engine_shape,
        fcc_flag=2 if fcc else 0,
        courant=held["l"],
        courant2=held["l2"],
        ts=held["ts"],
        h=held["h"],
        bn_ixyz=index[order],
        adj_bn=adjacency[order],
        mat_bn=material[order],
        saf_bn=saf[order],
        materials=[TEST_BRANCHES if branches is None else np.asarray(branches, dtype=float)]
        if lossy
        else [],
    )
    return arrays, grid


def write_entry(arrays: EngineArrays, grid: Grid, entry_dir: Path) -> Path:
    """The arrays as a cache entry's files: what PFFDTD's engine and a campaign read."""
    entry_dir = Path(entry_dir)
    entry_dir.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = arrays.shape
    with h5py.File(entry_dir / "sim_consts.h5", "w") as handle:
        handle.create_dataset("c", data=np.float64(343.2))
        handle.create_dataset("h", data=np.float64(arrays.h))
        handle.create_dataset("Ts", data=np.float64(arrays.ts))
        handle.create_dataset("SR", data=np.float64(1.0 / arrays.ts))
        handle.create_dataset("l", data=np.float64(arrays.courant))
        handle.create_dataset("l2", data=np.float64(arrays.courant2))
        handle.create_dataset("fcc_flag", data=np.int8(arrays.fcc_flag))
        handle.create_dataset("Tc", data=np.float64(20.0))
        handle.create_dataset("rh", data=np.float64(50.0))
    with h5py.File(entry_dir / "cart_grid.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(grid.h))
        for name, values in (("xv", grid.xv), ("yv", grid.yv), ("zv", grid.zv)):
            handle.create_dataset(name, data=values)
    with h5py.File(entry_dir / "vox_out.h5", "w") as handle:
        for name, value in (("Nx", nx), ("Ny", ny), ("Nz", nz), ("Nb", arrays.bn_ixyz.size)):
            handle.create_dataset(name, data=np.int64(value))
        handle.create_dataset("bn_ixyz", data=arrays.bn_ixyz.astype(np.int64))
        handle.create_dataset("adj_bn", data=arrays.adj_bn.astype(bool))
        handle.create_dataset("mat_bn", data=arrays.mat_bn.astype(np.int8))
        handle.create_dataset("saf_bn", data=arrays.saf_bn.astype(np.float64))
    with h5py.File(entry_dir / "sim_mats.h5", "w") as handle:
        handle.create_dataset("Nmat", data=np.int8(len(arrays.materials)))
        handle.create_dataset(
            "Mb", data=np.asarray([m.shape[0] for m in arrays.materials], dtype=np.int8)
        )
        for index, rows in enumerate(arrays.materials):
            handle.create_dataset(f"mat_{index:02d}_DEF", data=np.asarray(rows, dtype=np.float64))
    (entry_dir / "manifest.json").write_text(
        json.dumps({"scheme": arrays.scheme.name, "shape": list(arrays.shape)})
    )
    return entry_dir
