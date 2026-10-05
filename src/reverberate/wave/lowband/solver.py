"""The time stepping: many sources at once on one grid, receivers kept where they are computed.

A step is the engine's (``cpu_engine.h``), in the engine's order: the halo's
copies, the air, the absorbing layer, the boundary nodes with their
frequency dependent branches, the receivers read, the sources added, the two
fields exchanged. What differs is what it is run over:

- **a batch of sources.** Each source is its own two fields, and the grid
  with its boundary is shared. A field is stored ``[node, source]``, the
  source the contiguous axis, so one launch of a kernel updates every
  source's copy of a node from the same coefficients: one read of the node's
  kind and of its column's neighbours serves the batch;
- **the reached columns alone** (:mod:`reverberate.wave.lowband.problem`);
- **single precision throughout**, the engine's own for a campaign;
- **the receivers stay on the card**: a record is one node of one source,
  one sample a step, written into a buffer on the device that the fit reads
  there. Nothing goes to the host and nothing to a file.

Three implementations of the same arithmetic, operation for operation:
:class:`NumpyStepper` (vectorised, what a machine without a card and the
tests run), :func:`reference_step` (one node at a time, a transcription of
the kernels below, slow and obvious) and the CUDA kernels of
:class:`CardStepper`, compiled without fused multiply-add as every kernel of
this project is. The first two are compared bit for bit by the tests, and
the third with the first on the card by ``python -m reverberate.wave.lowband
verify``.

The arithmetic is the CPU engine's, not the card engine's: PFFDTD's CUDA air
kernel sums its neighbours pairwise and fuses the last products, so the two
agree to rounding and not to the bit, as PFFDTD's own two binaries do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.compute import raw_kernel
from reverberate.wave.comms import (
    Grid,
    _differentiate,
    engine_indices,
    interp_weights,
    source_signal,
)
from reverberate.wave.lowband.problem import Problem

__all__ = [
    "CardStepper",
    "Drive",
    "NumpyStepper",
    "SPOOL_STEPS",
    "State",
    "drive_for",
    "node_masks",
    "reference_step",
    "solve",
    "steps_for",
]

#: The most branches a kernel's thread holds, as ``MMb`` of ``fdtd_data.h`` bounds a material.
MAX_BRANCHES = 16


def steps_for(duration_s: float, ts: float) -> int:
    """The steps of a window: ``ceil(duration / Ts)``, the engine's ``Nt``."""
    return int(np.ceil(duration_s / ts))


@dataclass
class Drive:
    """What a batch is driven with and where it is read: sources in, records out.

    A row of ``inject`` is one grid node of one source with its signal; a row
    of ``record`` one grid node of one source. ``*_index`` are compact node
    indices of the problem and ``*_source`` the source's place in the batch.
    """

    batch: int
    steps: int
    inject_index: np.ndarray
    inject_source: np.ndarray
    inject_signal: np.ndarray
    record_index: np.ndarray
    record_source: np.ndarray


def drive_for(
    problem: Problem,
    grid: Grid,
    sources: np.ndarray,
    receivers: list[np.ndarray],
    duration_s: float,
) -> Drive:
    """The drive of a batch: each source as ``write_comms`` places it, each receiver a node.

    ``sources`` is ``[source, 3]`` in the scene's frame. A source is spread
    over the eight nodes round it with the engine's trilinear weights and
    given the differentiated impulse a single precision engine is given, on
    the engine's scale. ``receivers[s]`` are the engine flat indices of the
    nodes source ``s`` is read at.
    """
    sources = np.asarray(sources, dtype=float).reshape(-1, 3)
    if len(receivers) != sources.shape[0]:
        raise ValueError(f"{len(receivers)} receiver lists for {sources.shape[0]} sources")
    steps = steps_for(duration_s, grid.Ts)
    signal = source_signal(duration_s, grid.Ts, "impulse")
    scale = (0.5 * grid.l2 / grid.h) if grid.fcc else (grid.l2 / grid.h)
    index, of_source, signals = [], [], []
    for s, position in enumerate(sources):
        alpha, ixyz = interp_weights(position, grid)
        in_sigs = alpha[:, None] * signal[None, :]
        in_sigs *= scale
        in_sigs = _differentiate(in_sigs, grid.Ts)
        compact = problem.compact(engine_indices(ixyz, grid))
        if (compact < 0).any() or problem.kind.reshape(-1)[compact].min() == 0:
            raise ValueError(f"source {position.tolist()} touches a node that is not plain air")
        index.append(compact)
        of_source.append(np.full(compact.size, s, dtype=np.int32))
        signals.append(in_sigs.astype(np.float32))
    record_index, record_source = [], []
    for s, nodes in enumerate(receivers):
        compact = problem.compact(np.asarray(nodes, dtype=np.int64))
        # A receiver in a column no source reaches reads the column of zeros.
        compact = np.where(compact < 0, problem.column_count * problem.nz, compact)
        record_index.append(compact)
        record_source.append(np.full(compact.size, s, dtype=np.int32))
    return Drive(
        batch=int(sources.shape[0]),
        steps=steps,
        inject_index=np.concatenate(index).astype(np.int64),
        inject_source=np.concatenate(of_source),
        inject_signal=np.concatenate(signals),
        record_index=np.concatenate(record_index).astype(np.int64)
        if record_index
        else np.zeros(0, dtype=np.int64),
        record_source=np.concatenate(record_source)
        if record_source
        else np.zeros(0, dtype=np.int32),
    )


@dataclass
class State:
    """A batch's fields, ``[node, source]``, and its branches, ``[source, branch, node]``.

    The branches are stored a source, then a branch, then the lossy nodes in
    their order: a kernel's neighbouring threads are neighbouring nodes, and
    read neighbouring floats. Stored ``[node, branch, source]`` the same
    kernel took 2.1 ms a step for one source on an RTX 3080 and 15 ms a
    source for eight.
    """

    u0: Any
    u1: Any
    vh: Any
    gh: Any

    @classmethod
    def zeros(cls, problem: Problem, batch: int, xp: Any) -> State:
        shape = (problem.nodes, batch)
        branches = (batch, problem.max_branches, problem.lossy)
        return cls(
            u0=xp.zeros(shape, dtype=xp.float32),
            u1=xp.zeros(shape, dtype=xp.float32),
            vh=xp.zeros(branches, dtype=xp.float32),
            gh=xp.zeros(branches, dtype=xp.float32),
        )


# --------------------------------------------------------------------------
# numpy: the vectorised path
# --------------------------------------------------------------------------

_ONE = np.float32(1.0)
_TWO = np.float32(2.0)


class NumpyStepper:
    """The step on ``numpy``, every node of a kind at once, in the kernels' order."""

    def __init__(self, problem: Problem, drive: Drive) -> None:
        self.problem, self.drive = problem, drive
        nz = problem.nz
        kind = problem.kind.reshape(-1)
        self.air = np.flatnonzero(kind)
        self.air_neighbours = problem.neighbours_of(self.air // nz, self.air % nz)
        q = kind[self.air].astype(np.float32) - _ONE
        self.absorbing = np.flatnonzero(q > 0)
        self.lq = (problem.l32 * q[self.absorbing]).astype(np.float32)
        self.bn_neighbours = problem.neighbours_of(problem.bn_column, problem.bn_z)
        bits = problem.bn_adjacency
        count = np.zeros(bits.size, dtype=np.float32)
        self.coefficient = []
        for j in range(len(problem.stencil)):
            bit = ((bits >> np.uint16(j)) & np.uint16(1)).astype(np.float32)
            count += bit
            self.coefficient.append((problem.a2 * bit).astype(np.float32))
        self.b1 = (_TWO - problem.sl2 * count).astype(np.float32)
        self.lossy_rows = np.flatnonzero(problem.bn_lossy >= 0)
        material = problem.lossy_material
        self.quads = problem.quads[material] if material.size else np.zeros((0, 1, 4), np.float32)
        self.lo2kbg = ((problem.lo2 * problem.lossy_ssaf) * problem.beta[material]).astype(
            np.float32
        )
        self.fac = (((_TWO * problem.lo2) * problem.lossy_ssaf) / (_ONE + self.lo2kbg)).astype(
            np.float32
        )
        self.inject_at = (drive.inject_index, drive.inject_source)
        self.record_at = (drive.record_index, drive.record_source)

    def state(self) -> State:
        return State.zeros(self.problem, self.drive.batch, np)

    def records(self, steps: int | None = None) -> np.ndarray:
        width = self.drive.steps if steps is None else int(steps)
        return np.zeros((self.drive.record_index.size, width), dtype=np.float32)

    def step(self, state: State, n: int, out: np.ndarray, column: int | None = None) -> None:
        self.read(state, n if column is None else column, out)
        self.halo(state)
        self.update(state)
        self.inject(state, n)

    def read(self, state: State, column: int, out: np.ndarray) -> None:
        """The receivers, into ``column`` of ``out``."""
        out[:, column] = state.u1[self.record_at]

    def halo(self, state: State) -> None:
        """The box's own halo: the engine's copies, in its order."""
        u1 = state.u1
        for dst, src in self.problem.copies:
            u1[dst] = u1[src]

    def update(self, state: State) -> None:
        """Every node written a step: the air, the absorbing layer, the boundary."""
        p = self.problem
        u0, u1 = state.u0, state.u1
        # the air, and its absorbing layer
        previous = u0[self.air]
        partial = p.a1 * u1[self.air] - previous
        for neighbour in self.air_neighbours:
            partial += p.a2 * u1[neighbour]
        if self.absorbing.size:
            lq = self.lq[:, None]
            partial[self.absorbing] = (partial[self.absorbing] + lq * previous[self.absorbing]) / (
                _ONE + lq
            )
        u0[self.air] = partial
        # the boundary nodes: rigid first, then the branches of the lossy ones
        if p.bn_index.size:
            previous = u0[p.bn_index]
            partial = self.b1[:, None] * u1[p.bn_index] - previous
            for coefficient, neighbour in zip(self.coefficient, self.bn_neighbours, strict=True):
                partial += coefficient[:, None] * u1[neighbour]
            rows = self.lossy_rows
            if rows.size:
                q = self.quads
                before = previous[rows]
                lo2kbg = self.lo2kbg[:, None]
                x = (partial[rows] + lo2kbg * before) / (_ONE + lo2kbg)
                fac = self.fac[:, None]
                for m in range(p.max_branches):
                    x = x - fac * (
                        (_TWO * q[:, m, 2])[:, None] * state.vh[:, m].T
                        - q[:, m, 3][:, None] * state.gh[:, m].T
                    )
                du = x - before
                for m in range(p.max_branches):
                    vh1 = state.vh[:, m].T
                    gh1 = state.gh[:, m].T
                    vh0 = (q[:, m, 0][:, None] * du + q[:, m, 1][:, None] * vh1) - (
                        _TWO * q[:, m, 3]
                    )[:, None] * gh1
                    state.gh[:, m] = (gh1 + (vh0 + vh1) / _TWO).T
                    state.vh[:, m] = vh0.T
                partial[rows] = x
            u0[p.bn_index] = partial

    def inject(self, state: State, n: int) -> None:
        """The sources' sample of step ``n``, and the two fields exchanged."""
        u0, u1 = state.u0, state.u1
        np.add.at(u0, self.inject_at, self.drive.inject_signal[:, n])
        state.u0, state.u1 = u1, u0

    def finish(self) -> None:
        return None


# --------------------------------------------------------------------------
# one node at a time: what a kernel's thread does
# --------------------------------------------------------------------------


def reference_step(problem: Problem, drive: Drive, state: State, n: int, out: np.ndarray) -> None:
    """One step, every node and every source in its own scalar pass: the kernels, in Python."""
    p = problem
    f32 = np.float32
    nz, batch = p.nz, drive.batch
    u0, u1 = state.u0, state.u1
    for row in range(drive.record_index.size):
        out[row, n] = u1[drive.record_index[row], drive.record_source[row]]
    for dst, src in p.copies:
        for pair in range(dst.size):
            for b in range(batch):
                u1[dst[pair], b] = u1[src[pair], b]
    kind = p.kind.reshape(-1)
    for column in range(p.column_count):
        for z in range(nz):
            at = column * nz + z
            k = int(kind[at])
            if k == 0:
                continue
            for b in range(batch):
                previous = u0[at, b]
                partial = f32(p.a1 * u1[at, b]) - previous
                for slot, dz in p.stencil:
                    other = column if slot < 0 else int(p.lateral[column, slot])
                    partial = f32(partial + f32(p.a2 * u1[other * nz + z + dz, b]))
                if k > 1:
                    lq = f32(p.l32 * f32(k - 1))
                    partial = f32(f32(partial + f32(lq * previous)) / f32(_ONE + lq))
                u0[at, b] = partial
    for nb in range(p.bn_index.size):
        column, z = int(p.bn_column[nb]), int(p.bn_z[nb])
        at = column * nz + z
        adjacency = int(p.bn_adjacency[nb])
        b1 = f32(_TWO - f32(p.sl2 * f32(bin(adjacency).count("1"))))
        row = int(p.bn_lossy[nb])
        for b in range(batch):
            previous = u0[at, b]
            partial = f32(f32(b1 * u1[at, b]) - previous)
            for j, (slot, dz) in enumerate(p.stencil):
                other = column if slot < 0 else int(p.lateral[column, slot])
                bit = f32((adjacency >> j) & 1)
                partial = f32(partial + f32(f32(p.a2 * bit) * u1[other * nz + z + dz, b]))
            if row >= 0:
                k = int(p.lossy_material[row])
                ssaf = p.lossy_ssaf[row]
                lo2kbg = f32(f32(p.lo2 * ssaf) * p.beta[k])
                fac = f32(f32(f32(_TWO * p.lo2) * ssaf) / f32(_ONE + lo2kbg))
                x = f32(f32(partial + f32(lo2kbg * previous)) / f32(_ONE + lo2kbg))
                held_v = [state.vh[b, m, row] for m in range(int(p.branches[k]))]
                held_g = [state.gh[b, m, row] for m in range(int(p.branches[k]))]
                for m in range(int(p.branches[k])):
                    _, _, bdh, bfh = p.quads[k, m]
                    x = f32(
                        x - f32(fac * f32(f32(f32(_TWO * bdh) * held_v[m]) - f32(bfh * held_g[m])))
                    )
                du = f32(x - previous)
                for m in range(int(p.branches[k])):
                    bb, bd, _, bfh = p.quads[k, m]
                    vh0 = f32(
                        f32(f32(bb * du) + f32(bd * held_v[m])) - f32(f32(_TWO * bfh) * held_g[m])
                    )
                    state.gh[b, m, row] = f32(held_g[m] + f32(f32(vh0 + held_v[m]) / _TWO))
                    state.vh[b, m, row] = vh0
                partial = x
            u0[at, b] = partial
    for row in range(drive.inject_index.size):
        where = (drive.inject_index[row], drive.inject_source[row])
        u0[where] = f32(u0[where] + drive.inject_signal[row, n])
    state.u0, state.u1 = u1, u0


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

#: A node's mask on the card: its adjacency in the low twelve bits, the absorbing layer's
#: ``Q`` above them, and a flag on a boundary node. Zero is a node that is never written.
MASK_Q_SHIFT = 12
MASK_BOUNDARY = 0x4000

_AIR_KERNEL = r"""
extern "C" __global__ void lowband_air(
    float* __restrict__ u0, const float* __restrict__ u1,
    const unsigned short* __restrict__ mask, const int* __restrict__ lateral,
    int nz, int batch, int laterals, float a1, float a2, float sl2, float l)
{
    int column = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    int nzb = nz * batch;
    if (t >= nzb) return;
    unsigned int k = mask[(long long)column * nz + t / batch];
    if (k == 0) return;
    const int* nb = lateral + (long long)column * laterals;
    long long own = (long long)column * nzb;
    float previous = u0[own + t];
    float partial;
    if (k & 0x4000u) {
        // a boundary node's rigid update: the neighbours its adjacency allows
        unsigned int adjacency = k & 0x0FFFu;
        float b1 = 2.0f - sl2 * (float)__popc(adjacency);
        partial = b1 * u1[own + t] - previous;
%(boundary)s
    } else {
        partial = a1 * u1[own + t] - previous;
%(air)s
        unsigned int q = (k >> 12) & 3u;
        if (q > 0) {
            float lq = l * (float)q;
            partial = (partial + lq * previous) / (1.0f + lq);
        }
    }
    u0[own + t] = partial;
}
"""

_LOSSY_KERNEL = r"""
extern "C" __global__ void lowband_lossy(
    float* __restrict__ u0, float* __restrict__ before,
    const int* __restrict__ lossy_column, const int* __restrict__ lossy_z,
    const int* __restrict__ lossy_material, const float* __restrict__ lossy_ssaf,
    const signed char* __restrict__ branches, const float* __restrict__ quads,
    const float* __restrict__ beta, float* __restrict__ vh, float* __restrict__ gh,
    int nz, int batch, int max_branches, int lossy, float lo2)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lossy) return;
    int b = blockIdx.y;
    long long at = ((long long)lossy_column[row] * nz + lossy_z[row]) * batch + b;
    long long held = (long long)b * lossy + row;
    float partial = u0[at];
    float previous = before[held];
    int k = lossy_material[row];
    float ssaf = lossy_ssaf[row];
    float lo2kbg = lo2 * ssaf * beta[k];
    float fac = 2.0f * lo2 * ssaf / (1.0f + lo2kbg);
    float x = (partial + lo2kbg * previous) / (1.0f + lo2kbg);
    const float* q = quads + (long long)k * max_branches * 4;
    int mb = branches[k];
    long long first = (long long)b * max_branches * lossy + row;
    float held_v[%(most)d], held_g[%(most)d];
    for (int m = 0; m < mb; ++m) {
        held_v[m] = vh[first + (long long)m * lossy];
        held_g[m] = gh[first + (long long)m * lossy];
        x -= fac * (2.0f * q[m * 4 + 2] * held_v[m] - q[m * 4 + 3] * held_g[m]);
    }
    float du = x - previous;
    for (int m = 0; m < mb; ++m) {
        float vh0 = q[m * 4] * du + q[m * 4 + 1] * held_v[m] - 2.0f * q[m * 4 + 3] * held_g[m];
        gh[first + (long long)m * lossy] = held_g[m] + (vh0 + held_v[m]) / 2.0f;
        vh[first + (long long)m * lossy] = vh0;
    }
    u0[at] = x;
    before[held] = x;
}
"""

_SMALL_KERNELS = r"""
extern "C" __global__ void lowband_copy(
    float* __restrict__ u1, const long long* __restrict__ dst, const long long* __restrict__ src,
    int count, int batch)
{
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= count) return;
    int b = blockIdx.y;
    u1[dst[pair] * batch + b] = u1[src[pair] * batch + b];
}

extern "C" __global__ void lowband_inject(
    float* __restrict__ u0, const long long* __restrict__ at, const float* __restrict__ signal,
    int rows, long long steps, long long n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    u0[at[row]] += signal[row * steps + n];
}

extern "C" __global__ void lowband_record(
    const float* __restrict__ u1, const long long* __restrict__ at, float* __restrict__ out,
    int rows, long long steps, long long n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    out[row * steps + n] = u1[at[row]];
}
"""


def _stencil_source(problem: Problem, *, boundary: bool) -> str:
    """The neighbours' lines of a kernel, in the engine's order."""
    lines = []
    for j, (slot, dz) in enumerate(problem.stencil):
        base = "own" if slot < 0 else f"(long long)nb[{slot}] * nzb"
        shift = "" if dz == 0 else (" + batch" if dz > 0 else " - batch")
        weight = f"a2 * (float)((adjacency >> {j}) & 1u)" if boundary else "a2"
        lines.append(f"        partial += {weight} * u1[{base} + t{shift}];")
    return "\n".join(lines)


def node_masks(problem: Problem) -> np.ndarray:
    """Every stored node's mask for the card, ``[column, z]`` uint16; see :data:`MASK_BOUNDARY`."""
    full = (1 << len(problem.stencil)) - 1
    kind = problem.kind.astype(np.uint16)
    mask = np.where(kind > 0, full | ((kind - (kind > 0)) << MASK_Q_SHIFT), 0).astype(np.uint16)
    flat = mask.reshape(-1)
    flat[problem.bn_index] = problem.bn_adjacency | np.uint16(MASK_BOUNDARY)
    return mask


class CardStepper:
    """The step on a card: every source of the batch in each launch.

    The air's kernel walks every stored node and writes a boundary node's
    rigid update too, from the adjacency in its mask: the node's neighbours
    are then read in the stream of its column, not gathered one node at a
    time. What is left for the lossy nodes is their branches, and the value
    each held two steps before, which the air's kernel has by then
    overwritten and is therefore kept beside the branches. As a kernel of
    its own over the boundary nodes, gathering seven values a node, the
    boundary was 2.0 ms of a 3.2 ms step on an RTX 3080.
    """

    THREADS = 256

    def __init__(self, problem: Problem, drive: Drive, xp: Any) -> None:
        if problem.max_branches > MAX_BRANCHES:
            raise ValueError(f"a material has more than {MAX_BRANCHES} branches")
        self.problem, self.drive, self.xp = problem, drive, xp
        p, batch = problem, drive.batch
        self.air = raw_kernel(
            _AIR_KERNEL
            % {
                "air": _stencil_source(p, boundary=False),
                "boundary": _stencil_source(p, boundary=True),
            },
            "lowband_air",
        )
        self.lossy_kernel = raw_kernel(_LOSSY_KERNEL % {"most": MAX_BRANCHES}, "lowband_lossy")
        self.copy = raw_kernel(_SMALL_KERNELS, "lowband_copy")
        self.inject_kernel = raw_kernel(_SMALL_KERNELS, "lowband_inject")
        self.record = raw_kernel(_SMALL_KERNELS, "lowband_record")
        self.mask = xp.asarray(node_masks(p).reshape(-1))
        self.lateral = xp.asarray(np.ascontiguousarray(p.lateral).reshape(-1))
        rows = np.flatnonzero(p.bn_lossy >= 0)
        self.lossy = tuple(
            xp.asarray(a) for a in (p.bn_column[rows], p.bn_z[rows], p.lossy_material, p.lossy_ssaf)
        )
        self.materials = (
            xp.asarray(p.branches),
            xp.asarray(np.ascontiguousarray(p.quads).reshape(-1)),
            xp.asarray(p.beta),
        )
        # What each lossy node held one and two steps before, ``[source, node]``.
        self.before = [
            xp.zeros((batch, p.lossy), dtype=xp.float32),
            xp.zeros((batch, p.lossy), dtype=xp.float32),
        ]
        self.copies = [(xp.asarray(dst), xp.asarray(src), int(dst.size)) for dst, src in p.copies]
        self.inject_at = xp.asarray(drive.inject_index * batch + drive.inject_source)
        self.inject_signal = xp.asarray(np.ascontiguousarray(drive.inject_signal).reshape(-1))
        self.record_at = xp.asarray(drive.record_index * batch + drive.record_source)
        threads = self.THREADS
        self.air_grid = (p.column_count, (p.nz * batch + threads - 1) // threads)
        self.lossy_grid = ((p.lossy + threads - 1) // threads, batch)

    def state(self) -> State:
        return State.zeros(self.problem, self.drive.batch, self.xp)

    def records(self, steps: int | None = None) -> Any:
        width = self.drive.steps if steps is None else int(steps)
        return self.xp.zeros((self.drive.record_index.size, width), dtype=self.xp.float32)

    def launch_air(self, state: State) -> None:
        p, i32, f32 = self.problem, np.int32, np.float32
        self.air(
            self.air_grid,
            (self.THREADS,),
            (
                state.u0,
                state.u1,
                self.mask,
                self.lateral,
                i32(p.nz),
                i32(self.drive.batch),
                i32(len(p.lateral_offsets)),
                f32(p.a1),
                f32(p.a2),
                f32(p.sl2),
                f32(p.l32),
            ),
        )

    def launch_lossy(self, state: State) -> None:
        p, i32 = self.problem, np.int32
        self.lossy_kernel(
            self.lossy_grid,
            (self.THREADS,),
            (
                state.u0,
                self.before[0],
                *self.lossy,
                *self.materials,
                state.vh,
                state.gh,
                i32(p.nz),
                i32(self.drive.batch),
                i32(p.max_branches),
                i32(p.lossy),
                np.float32(p.lo2),
            ),
        )
        # The value just written is the one held a step before, next step.
        self.before.reverse()

    def step(self, state: State, n: int, out: Any, column: int | None = None) -> None:
        self.read(state, n if column is None else column, out)
        self.halo(state)
        self.update(state)
        self.inject(state, n)

    def read(self, state: State, column: int, out: Any) -> None:
        """The receivers, into ``column`` of ``out``, whose width is its own."""
        threads, i32, i64 = self.THREADS, np.int32, np.int64
        rows = int(self.drive.record_index.size)
        if rows:
            self.record(
                ((rows + threads - 1) // threads,),
                (threads,),
                (state.u1, self.record_at, out, i32(rows), i64(out.shape[1]), i64(column)),
            )

    def halo(self, state: State) -> None:
        """The box's own halo: the engine's copies, in its order."""
        batch, threads, i32 = self.drive.batch, self.THREADS, np.int32
        for dst, src, count in self.copies:
            self.copy(
                ((count + threads - 1) // threads, batch),
                (threads,),
                (state.u1, dst, src, i32(count), i32(batch)),
            )

    def update(self, state: State) -> None:
        """Every node written a step: the air with the rigid boundary, then the branches."""
        p = self.problem
        if p.column_count:
            self.launch_air(state)
        if p.lossy:
            self.launch_lossy(state)

    def inject(self, state: State, n: int) -> None:
        """The sources' sample of step ``n``, and the two fields exchanged."""
        threads, i32, i64 = self.THREADS, np.int32, np.int64
        injected = int(self.drive.inject_index.size)
        if injected:
            self.inject_kernel(
                ((injected + threads - 1) // threads,),
                (threads,),
                (
                    state.u0,
                    self.inject_at,
                    self.inject_signal,
                    i32(injected),
                    i64(self.drive.steps),
                    i64(n),
                ),
            )
        state.u0, state.u1 = state.u1, state.u0

    def finish(self) -> None:
        self.xp.cuda.Stream.null.synchronize()


#: Steps of its records a device keeps before they are brought to the host (``records_on``).
SPOOL_STEPS = 512


def solve(
    problem: Problem,
    drive: Drive,
    xp: Any,
    *,
    say: Any = None,
    report_every_s: float = 30.0,
    timing: dict[str, Any] | None = None,
    records_on: str = "device",
    spool_steps: int = SPOOL_STEPS,
    stepper: Any = None,
) -> Any:
    """Every step of a batch; the records, ``[record, step]`` single precision.

    ``records_on`` says where they are kept. ``"device"``: on ``xp``, the
    whole of them, 4 bytes a node a step, which on the grid to 1500 Hz is
    129 MB a cell and the larger part of what a launch holds. ``"host"``:
    the device keeps ``spool_steps`` steps of them and hands each block to
    the host's memory while the next is computed, so that a launch holds on
    its card its sources' fields and that block alone. The numbers are the
    same single precision values either way, to the bit: a block is a copy.

    ``timing``, when given, receives the seconds of the stepping alone and
    the node updates it made: the reached nodes written a step, times the
    sources, times the steps. ``stepper`` is the one to step with, where
    the caller holds another than the device's own (a grid in slabs).
    """
    if records_on not in ("device", "host"):
        raise ValueError(f"records are kept on the device or on the host, not {records_on!r}")
    if stepper is None:
        stepper = NumpyStepper(problem, drive) if xp is np else CardStepper(problem, drive, xp)
    state = stepper.state()
    spooled = records_on == "host"
    width = max(1, min(int(spool_steps), drive.steps)) if spooled else drive.steps
    out = stepper.records(width)
    rows = int(drive.record_index.size)
    held = np.zeros((rows, drive.steps), dtype=np.float32) if spooled else None
    started = last = time.time()
    for n in range(drive.steps):
        column = n % width
        stepper.step(state, n, out, column)
        if held is not None and (column == width - 1 or n == drive.steps - 1):
            held[:, n - column : n + 1] = records_to_host(out[:, : column + 1])
        if say is not None and time.time() - last > report_every_s:
            last = time.time()
            say(f"step {n + 1}/{drive.steps} at {last - started:.0f} s")
    stepper.finish()
    if timing is not None:
        seconds = time.time() - started
        updates = float(problem.updated) * drive.batch * drive.steps
        timing.update(
            {
                "seconds": seconds,
                "node_updates": updates,
                "updates_per_s": updates / max(seconds, 1e-9),
                "batch": drive.batch,
                "steps": drive.steps,
                "records_on": records_on,
            }
        )
    return out if held is None else held


def records_to_host(block: Any) -> np.ndarray:
    """A block of records as the host holds it: the one place records leave a device."""
    return np.asarray(block.get() if hasattr(block, "get") else block)
