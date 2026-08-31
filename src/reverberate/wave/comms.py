"""Rebuilding ``comms_out.h5`` alone, without rerunning the voxeliser.

Roadmap section 11 and task W8, item 3. PFFDTD's ``sim_setup`` does two
unrelated things in one call: it voxelises the scene, which costs 52 minutes of
the 75 minute B0 session, and it places one source and its receivers on the
grid, which costs milliseconds. There is no supported way to redo the second
without the first, so a dataset of many source and receiver pairs in one room
pays for the same voxelisation once per pair. That is the whole economics of the
split pipeline, so this module builds the missing path.

**What makes this delicate is not the interpolation, it is the index space.**
``sim_setup`` writes ``comms_out.h5`` in the Cartesian grid's own index space,
then ``rotate_sim_data`` permutes the axes into the order the engine wants,
``fold_fcc_sim_data`` folds the FCC subgrid onto itself, and ``sort_sim_data``
sorts every index array. Those three passes rewrite ``vox_out.h5`` in place, so
a cached voxelisation is already in engine space and a freshly computed comms
file is not. Everything below exists to put new source and receiver points into
exactly the space the cached voxelisation is already in:

- the axis permutation is recovered from ``cart_grid.h5``, which
  :func:`~reverberate.wave.voxelise.voxelise` keeps precisely because
  ``rotate_sim_data`` never touches it and it is the only surviving record of
  the original ``Nx, Ny, Nz``;
- the fold and the sort are reproduced here rather than called, because
  PFFDTD's versions mutate ``vox_out.h5`` as a side effect and would corrupt a
  cache entry that is already folded and sorted.

The result is verified against ``sim_setup``'s own output, dataset by dataset,
in ``tests/test_wave_comms.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

__all__ = [
    "ENGINE_FILES",
    "Grid",
    "SIGNAL_TYPES",
    "SignalType",
    "fold_fcc",
    "interp_weights",
    "load_grid",
    "source_signal",
    "transpose_order",
    "write_comms",
]

#: The four files the CUDA engine reads. It reads nothing else, and in
#: particular not ``cart_grid.h5``, which is why only these are shipped.
ENGINE_FILES = ("sim_consts.h5", "vox_out.h5", "comms_out.h5", "sim_mats.h5")

SignalType = Literal["impulse", "hann10", "hann20", "dhann30", "hann5ms"]

#: Source signals, spelled as PFFDTD's ``sim_comms`` spells them.
SIGNAL_TYPES: tuple[str, ...] = ("impulse", "hann10", "hann20", "dhann30", "hann5ms")

#: The eight corners of the cell a point falls in, in PFFDTD's order. The order
#: matters: it fixes the order of ``out_alpha``'s columns, which the engine
#: pairs positionally with ``out_ixyz``.
_CORNER_OFFSETS = np.array(
    [
        [0, 0, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-1, -1, 0],
        [-1, 0, -1],
        [0, -1, -1],
        [-1, -1, -1],
    ]
)


@dataclass(frozen=True)
class Grid:
    """The grid a set of comms points must land on.

    Read from ``sim_consts.h5`` and ``cart_grid.h5``, both of which a cached
    voxelisation keeps. ``xv``, ``yv`` and ``zv`` are the node coordinates along
    each axis **before** any rotation, which is the space
    :func:`interp_weights` works in.
    """

    h: float
    Ts: float
    l2: float
    fcc_flag: int
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray

    @property
    def fcc(self) -> bool:
        """Whether the scheme is face centred cubic rather than Cartesian."""
        return self.fcc_flag > 0

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(Nx, Ny, Nz)`` of the unrotated grid."""
        return self.xv.size, self.yv.size, self.zv.size


def load_grid(data_dir: Path | str) -> Grid:
    """Read the grid a cached voxelisation was built on.

    ``fcc_flag`` is taken from ``sim_consts.h5``, where PFFDTD writes 2 rather
    than 1 once the subgrid has been folded; both mean FCC.
    """
    data_dir = Path(data_dir)
    with h5py.File(data_dir / "sim_consts.h5", "r") as handle:
        h = float(handle["h"][()])
        ts = float(handle["Ts"][()])
        l2 = float(handle["l2"][()])
        fcc_flag = int(handle["fcc_flag"][()])
    with h5py.File(data_dir / "cart_grid.h5", "r") as handle:
        xv = np.asarray(handle["xv"][()], dtype=np.float64)
        yv = np.asarray(handle["yv"][()], dtype=np.float64)
        zv = np.asarray(handle["zv"][()], dtype=np.float64)
    return Grid(h=h, Ts=ts, l2=l2, fcc_flag=fcc_flag, xv=xv, yv=yv, zv=zv)


def interp_weights(position: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear weights and flat indices for one point, in unrotated space.

    A transcription of ``SimComms.get_linear_interp_weights``. The two
    assertions PFFDTD makes are kept, because they are the only check that a
    point actually lies inside the grid: weights that do not sum to one, or that
    do not reproduce the position, mean the point is outside and the run would
    be meaningless rather than merely wrong.
    """
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"expected one xyz point, got shape {position.shape}")
    axes = [grid.xv, grid.yv, grid.zv]
    nx, ny, nz = grid.shape

    base = np.empty(3, dtype=np.int64)
    alpha_xyz = np.zeros(3, dtype=np.float64)
    for j in range(3):
        ahead = np.flatnonzero(axes[j] >= position[j])
        if ahead.size == 0 or axes[j][0] > position[j]:
            raise ValueError(f"point {position.tolist()} is outside the grid on axis {j}")
        base[j] = ahead[0]
        alpha_xyz[j] = (axes[j][base[j]] - position[j]) / grid.h

    offsets = _CORNER_OFFSETS.copy()
    if grid.fcc:
        # The FCC grid only carries every other node, so the surrounding cell is
        # twice as wide and the corner must be nudged onto the subgrid first.
        offsets *= 2
        if np.sum(base) % 2 == 1:
            base[int(np.argmin(alpha_xyz))] += 1
        for j in range(3):
            alpha_xyz[j] = (axes[j][base[j]] - position[j]) / (2 * grid.h)

    alpha8 = np.ones(8, dtype=np.float64)
    xyz8 = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        for j in range(3):
            xyz8[i, j] = axes[j][base[j] + offsets[i, j]]
            alpha8[i] *= (1 - alpha_xyz[j]) if offsets[i, j] == 0 else alpha_xyz[j]

    if not np.allclose(np.sum(alpha8), 1):
        raise ValueError(f"interpolation weights for {position.tolist()} do not sum to one")
    if not np.allclose(np.sum(alpha8 * xyz8.T, -1), position):
        raise ValueError(f"interpolation does not reproduce {position.tolist()}")

    corners = base + offsets
    ixyz8 = corners @ np.array([nz * ny, nz, 1], dtype=np.int64)
    return alpha8, ixyz8


def source_signal(duration: float, ts: float, sig_type: SignalType = "impulse") -> np.ndarray:
    """The unscaled input signal, sample for sample as ``sim_comms`` builds it."""
    if sig_type not in SIGNAL_TYPES:
        raise ValueError(f"unknown signal type {sig_type!r}, expected one of {SIGNAL_TYPES}")
    nt = int(np.ceil(duration / ts))
    signal = np.zeros(nt, dtype=np.float64)
    if sig_type == "impulse":
        signal[0] = 1.0
    elif sig_type in ("hann10", "hann20"):
        n_window = 10 if sig_type == "hann10" else 20
        n = np.arange(n_window)
        signal[:n_window] = 0.5 * (1.0 - np.cos(2 * np.pi * n / n_window))
    elif sig_type == "dhann30":
        n = np.arange(30)
        signal[:30] = np.cos(np.pi * n / 30) * np.sin(np.pi * n / 30)
    else:  # hann5ms
        n_window = int(np.ceil(5e-3 / ts))
        n = np.arange(n_window)
        signal[:n_window] = 0.5 * (1.0 - np.cos(2 * np.pi * n / n_window))
    return signal


def _differentiate(signals: np.ndarray, ts: float) -> np.ndarray:
    """Bilinear differentiator, undone after the run.

    Single precision runs need it as a safeguard against DC instability. Written
    out as a recurrence rather than through ``scipy.signal.lfilter`` so this
    module needs nothing but numpy and h5py.
    """
    b = 2.0 / ts * np.array([1.0, -1.0])
    out = np.empty_like(signals)
    previous_in = np.zeros(signals.shape[0])
    previous_out = np.zeros(signals.shape[0])
    for n in range(signals.shape[-1]):
        current = b[0] * signals[:, n] + b[1] * previous_in - previous_out
        out[:, n] = current
        previous_in = signals[:, n]
        previous_out = current
    return out


def transpose_order(shape: tuple[int, int, int]) -> np.ndarray:
    """The axis permutation ``rotate_sim_data`` picks for a grid of this shape.

    Descending extent, so the contiguous last axis is the smallest and the
    engine's inner loop is the cheapest. Taken from the **unrotated** shape,
    which is why the cache keeps ``cart_grid.h5``.
    """
    return np.argsort(np.asarray(shape))[::-1]


def _reindex(ixyz: np.ndarray, shape: tuple[int, int, int], order: np.ndarray) -> np.ndarray:
    """Move flat indices from the unrotated grid into the permuted one."""
    if np.array_equal(order, np.array([0, 1, 2])):
        return ixyz
    nx, ny, nz = shape
    iz = ixyz % nz
    iy = (ixyz - iz) // nz % ny
    ix = ((ixyz - iz) // nz - iy) // ny
    subs = [ix, iy, iz]
    permuted_shape = [shape[i] for i in order]
    permuted_subs = [subs[i] for i in order]
    nyt, nzt = permuted_shape[1], permuted_shape[2]
    return np.asarray(permuted_subs).T @ np.array([nzt * nyt, nzt, 1], dtype=np.int64)


def fold_fcc(ixyz: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Fold indices onto the half grid, as ``fold_fcc_sim_data`` folds them.

    ``shape`` is the rotated grid, before folding. The FCC subgrid occupies half
    the Cartesian nodes, so the upper half in ``y`` is mirrored onto the lower.
    """
    nx, ny, nz = shape
    if ny % 2:
        raise ValueError(f"FCC needs an even Ny, got {ny}")
    ny_half = ny // 2 + 1
    iz = ixyz % nz
    iy = (ixyz - iz) // nz % ny
    ix = ((ixyz - iz) // nz - iy) // ny
    mirrored = np.where(iy >= ny / 2, ny - iy - 1, iy)
    folded = np.c_[ix, mirrored, iz] @ np.array([nz * ny_half, nz, 1], dtype=np.int64)
    return np.asarray(folded, dtype=np.int64)


def _engine_indices(ixyz: np.ndarray, grid: Grid) -> np.ndarray:
    """Take unrotated flat indices all the way into the engine's index space."""
    order = transpose_order(grid.shape)
    rotated = _reindex(np.atleast_1d(ixyz), grid.shape, order)
    if not grid.fcc:
        return rotated
    rotated_shape = tuple(int(grid.shape[i]) for i in order)
    return fold_fcc(rotated, rotated_shape)  # type: ignore[arg-type]


def write_comms(
    data_dir: Path | str,
    source: np.ndarray,
    receivers: np.ndarray,
    duration: float,
    *,
    sig_type: SignalType = "impulse",
    diff_source: bool = True,
    out_path: Path | str | None = None,
    compress: int | None = None,
    check_clashes: bool = True,
) -> Path:
    """Write a ``comms_out.h5`` for a cached voxelisation, and nothing else.

    ``data_dir`` is a cache entry: ``sim_consts.h5``, ``cart_grid.h5`` and, for
    the clash check, ``vox_out.h5``. Nothing in it is modified. This is item 3
    of W8 and the reason one voxelisation can serve many source and receiver
    pairs.

    ``diff_source`` must match the engine precision, exactly as in a full
    ``sim_setup`` call: true for the single precision binaries, false for double.
    """
    data_dir = Path(data_dir)
    grid = load_grid(data_dir)
    receivers = np.atleast_2d(np.asarray(receivers, dtype=np.float64))
    if receivers.size == 0:
        raise ValueError("at least one receiver is required")

    in_alpha, in_ixyz = interp_weights(np.asarray(source, dtype=np.float64), grid)
    out_alpha = np.zeros((receivers.shape[0], 8), dtype=np.float64)
    out_ixyz = np.zeros((receivers.shape[0], 8), dtype=np.int64)
    for row, receiver in enumerate(receivers):
        out_alpha[row], out_ixyz[row] = interp_weights(receiver, grid)

    in_sigs = in_alpha[:, None] * source_signal(duration, grid.Ts, sig_type)[None, :]
    in_sigs *= (0.5 * grid.l2 / grid.h) if grid.fcc else (grid.l2 / grid.h)
    if diff_source:
        in_sigs = _differentiate(in_sigs, grid.Ts)

    in_ixyz = _engine_indices(in_ixyz, grid)
    out_ixyz = _engine_indices(out_ixyz.reshape(-1), grid)

    if check_clashes:
        _check_for_clashes(data_dir, in_ixyz, out_ixyz)

    # sort_sim_data's ordering, reproduced. out_alpha is deliberately not
    # reordered: the engine applies out_reorder to the signals it writes, and
    # out_alpha is indexed in the caller's original receiver order.
    order_in = np.argsort(in_ixyz)
    in_ixyz = in_ixyz[order_in]
    in_sigs = in_sigs[order_in]
    order_out = np.argsort(out_ixyz)
    out_ixyz = out_ixyz[order_out]
    out_reorder = np.argsort(order_out)

    out_path = Path(out_path) if out_path is not None else data_dir / "comms_out.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keywords = {} if compress is None else {"compression": "gzip", "compression_opts": compress}
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("in_ixyz", data=in_ixyz, **keywords)
        handle.create_dataset("out_ixyz", data=out_ixyz, **keywords)
        handle.create_dataset("out_alpha", data=out_alpha, **keywords)
        handle.create_dataset("out_reorder", data=out_reorder, **keywords)
        handle.create_dataset("in_sigs", data=in_sigs, **keywords)
        handle.create_dataset("Ns", data=np.int64(in_ixyz.size))
        handle.create_dataset("Nr", data=np.int64(out_ixyz.size))
        handle.create_dataset("Nt", data=np.int64(in_sigs.shape[-1]))
        handle.create_dataset("diff", data=np.int8(diff_source))
    return out_path


def _check_for_clashes(data_dir: Path, in_ixyz: np.ndarray, out_ixyz: np.ndarray) -> None:
    """Refuse points that land on a boundary node.

    The scheme only supports sources and receivers in regular air nodes. PFFDTD
    checks this inside ``sim_setup``; since the point of this module is not to
    call ``sim_setup``, the check comes with it. A boundary node here would
    otherwise show up as a silently wrong impulse response after the GPU has
    already been paid for.
    """
    vox = data_dir / "vox_out.h5"
    if not vox.exists():
        return
    with h5py.File(vox, "r") as handle:
        bn_ixyz = np.asarray(handle["bn_ixyz"][...])
    for name, ixyz in (("source", in_ixyz), ("receiver", out_ixyz)):
        unique = np.unique(ixyz)
        clashing = np.intersect1d(unique, bn_ixyz, assume_unique=True)
        if clashing.size:
            raise ValueError(
                f"{name} interpolation touches {clashing.size} boundary nodes; "
                "move the point away from the surface"
            )
