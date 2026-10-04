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
    "State",
    "drive_for",
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
    """A batch's fields, ``[node, source]``, and its branches, ``[node, branch, source]``."""

    u0: Any
    u1: Any
    vh: Any
    gh: Any

    @classmethod
    def zeros(cls, problem: Problem, batch: int, xp: Any) -> State:
        shape = (problem.nodes, batch)
        branches = (problem.lossy, problem.max_branches, batch)
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

    def records(self) -> np.ndarray:
        return np.zeros((self.drive.record_index.size, self.drive.steps), dtype=np.float32)

    def step(self, state: State, n: int, out: np.ndarray) -> None:
        p = self.problem
        u0, u1 = state.u0, state.u1
        out[:, n] = u1[self.record_at]
        for dst, src in p.copies:
            u1[dst] = u1[src]
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
                        (_TWO * q[:, m, 2])[:, None] * state.vh[:, m]
                        - q[:, m, 3][:, None] * state.gh[:, m]
                    )
                du = x - before
                for m in range(p.max_branches):
                    vh1 = state.vh[:, m]
                    vh0 = (q[:, m, 0][:, None] * du + q[:, m, 1][:, None] * vh1) - (
                        _TWO * q[:, m, 3]
                    )[:, None] * state.gh[:, m]
                    state.gh[:, m] = state.gh[:, m] + (vh0 + vh1) / _TWO
                    state.vh[:, m] = vh0
                partial[rows] = x
            u0[p.bn_index] = partial
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
                held_v = [state.vh[row, m, b] for m in range(int(p.branches[k]))]
                held_g = [state.gh[row, m, b] for m in range(int(p.branches[k]))]
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
                    state.gh[row, m, b] = f32(held_g[m] + f32(f32(vh0 + held_v[m]) / _TWO))
                    state.vh[row, m, b] = vh0
                partial = x
            u0[at, b] = partial
    for row in range(drive.inject_index.size):
        where = (drive.inject_index[row], drive.inject_source[row])
        u0[where] = f32(u0[where] + drive.inject_signal[row, n])
    state.u0, state.u1 = u1, u0


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

_AIR_KERNEL = r"""
extern "C" __global__ void lowband_air(
    float* __restrict__ u0, const float* __restrict__ u1,
    const unsigned char* __restrict__ kind, const int* __restrict__ lateral,
    int nz, int batch, int laterals, float a1, float a2, float l)
{
    int column = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    int nzb = nz * batch;
    if (t >= nzb) return;
    unsigned char k = kind[(long long)column * nz + t / batch];
    if (k == 0) return;
    const int* nb = lateral + (long long)column * laterals;
    long long own = (long long)column * nzb;
    float previous = u0[own + t];
    float partial = a1 * u1[own + t] - previous;
%(stencil)s
    if (k > 1) {
        float lq = l * (float)(k - 1);
        partial = (partial + lq * previous) / (1.0f + lq);
    }
    u0[own + t] = partial;
}
"""

_BOUNDARY_KERNEL = r"""
extern "C" __global__ void lowband_boundary(
    float* __restrict__ u0, const float* __restrict__ u1,
    const int* __restrict__ bn_column, const int* __restrict__ bn_z,
    const unsigned short* __restrict__ bn_adjacency, const int* __restrict__ bn_lossy,
    const int* __restrict__ lateral,
    const int* __restrict__ lossy_material, const float* __restrict__ lossy_ssaf,
    const signed char* __restrict__ branches, const float* __restrict__ quads,
    const float* __restrict__ beta, float* __restrict__ vh, float* __restrict__ gh,
    int count, int nz, int batch, int laterals, int max_branches,
    float a2, float sl2, float lo2)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    int b = blockIdx.y;
    int column = bn_column[node];
    long long nzb = (long long)nz * batch;
    long long t = (long long)bn_z[node] * batch + b;
    long long own = (long long)column * nzb;
    const int* nb = lateral + (long long)column * laterals;
    unsigned int adjacency = bn_adjacency[node];
    float b1 = 2.0f - sl2 * (float)__popc(adjacency);
    float previous = u0[own + t];
    float partial = b1 * u1[own + t] - previous;
%(stencil)s
    int row = bn_lossy[node];
    if (row >= 0) {
        int k = lossy_material[row];
        float ssaf = lossy_ssaf[row];
        float lo2kbg = lo2 * ssaf * beta[k];
        float fac = 2.0f * lo2 * ssaf / (1.0f + lo2kbg);
        float x = (partial + lo2kbg * previous) / (1.0f + lo2kbg);
        const float* q = quads + (long long)k * max_branches * 4;
        int mb = branches[k];
        long long first = (long long)row * max_branches * batch + b;
        float held_v[%(most)d], held_g[%(most)d];
        for (int m = 0; m < mb; ++m) {
            held_v[m] = vh[first + (long long)m * batch];
            held_g[m] = gh[first + (long long)m * batch];
            x -= fac * (2.0f * q[m * 4 + 2] * held_v[m] - q[m * 4 + 3] * held_g[m]);
        }
        float du = x - previous;
        for (int m = 0; m < mb; ++m) {
            float vh0 = q[m * 4] * du + q[m * 4 + 1] * held_v[m] - 2.0f * q[m * 4 + 3] * held_g[m];
            gh[first + (long long)m * batch] = held_g[m] + (vh0 + held_v[m]) / 2.0f;
            vh[first + (long long)m * batch] = vh0;
        }
        partial = x;
    }
    u0[own + t] = partial;
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
        lines.append(f"    partial += {weight} * u1[{base} + t{shift}];")
    return "\n".join(lines)


class CardStepper:
    """The step on a card: five kernels a step, every source of the batch in each launch."""

    THREADS = 256

    def __init__(self, problem: Problem, drive: Drive, xp: Any) -> None:
        if problem.max_branches > MAX_BRANCHES:
            raise ValueError(f"a material has more than {MAX_BRANCHES} branches")
        self.problem, self.drive, self.xp = problem, drive, xp
        p, batch = problem, drive.batch
        self.air = raw_kernel(
            _AIR_KERNEL % {"stencil": _stencil_source(p, boundary=False)}, "lowband_air"
        )
        self.boundary = raw_kernel(
            _BOUNDARY_KERNEL % {"stencil": _stencil_source(p, boundary=True), "most": MAX_BRANCHES},
            "lowband_boundary",
        )
        self.copy = raw_kernel(_SMALL_KERNELS, "lowband_copy")
        self.inject = raw_kernel(_SMALL_KERNELS, "lowband_inject")
        self.record = raw_kernel(_SMALL_KERNELS, "lowband_record")
        self.kind = xp.asarray(p.kind.reshape(-1))
        self.lateral = xp.asarray(np.ascontiguousarray(p.lateral).reshape(-1))
        self.bn = tuple(xp.asarray(a) for a in (p.bn_column, p.bn_z, p.bn_adjacency, p.bn_lossy))
        self.lossy = (xp.asarray(p.lossy_material), xp.asarray(p.lossy_ssaf))
        self.materials = (
            xp.asarray(p.branches),
            xp.asarray(np.ascontiguousarray(p.quads).reshape(-1)),
            xp.asarray(p.beta),
        )
        self.copies = [(xp.asarray(dst), xp.asarray(src), int(dst.size)) for dst, src in p.copies]
        self.inject_at = xp.asarray(drive.inject_index * batch + drive.inject_source)
        self.inject_signal = xp.asarray(np.ascontiguousarray(drive.inject_signal).reshape(-1))
        self.record_at = xp.asarray(drive.record_index * batch + drive.record_source)
        threads = self.THREADS
        self.air_grid = (p.column_count, (p.nz * batch + threads - 1) // threads)
        self.bn_grid = ((int(p.bn_index.size) + threads - 1) // threads, batch)

    def state(self) -> State:
        return State.zeros(self.problem, self.drive.batch, self.xp)

    def records(self) -> Any:
        return self.xp.zeros(
            (self.drive.record_index.size, self.drive.steps), dtype=self.xp.float32
        )

    def step(self, state: State, n: int, out: Any) -> None:
        p, batch, threads = self.problem, self.drive.batch, self.THREADS
        i32, f32, i64 = np.int32, np.float32, np.int64
        steps = i64(self.drive.steps)
        rows = int(self.drive.record_index.size)
        if rows:
            self.record(
                ((rows + threads - 1) // threads,),
                (threads,),
                (state.u1, self.record_at, out, i32(rows), steps, i64(n)),
            )
        for dst, src, count in self.copies:
            self.copy(
                ((count + threads - 1) // threads, batch),
                (threads,),
                (state.u1, dst, src, i32(count), i32(batch)),
            )
        laterals = i32(len(p.lateral_offsets))
        if p.column_count:
            self.air(
                self.air_grid,
                (threads,),
                (
                    state.u0,
                    state.u1,
                    self.kind,
                    self.lateral,
                    i32(p.nz),
                    i32(batch),
                    laterals,
                    f32(p.a1),
                    f32(p.a2),
                    f32(p.l32),
                ),
            )
        if p.bn_index.size:
            self.boundary(
                self.bn_grid,
                (threads,),
                (
                    state.u0,
                    state.u1,
                    *self.bn,
                    self.lateral,
                    *self.lossy,
                    *self.materials,
                    state.vh,
                    state.gh,
                    i32(p.bn_index.size),
                    i32(p.nz),
                    i32(batch),
                    laterals,
                    i32(p.max_branches),
                    f32(p.a2),
                    f32(p.sl2),
                    f32(p.lo2),
                ),
            )
        injected = int(self.drive.inject_index.size)
        self.inject(
            ((injected + threads - 1) // threads,),
            (threads,),
            (state.u0, self.inject_at, self.inject_signal, i32(injected), steps, i64(n)),
        )
        state.u0, state.u1 = state.u1, state.u0

    def finish(self) -> None:
        self.xp.cuda.Stream.null.synchronize()


def solve(
    problem: Problem,
    drive: Drive,
    xp: Any,
    *,
    say: Any = None,
    report_every_s: float = 30.0,
    timing: dict[str, Any] | None = None,
) -> Any:
    """Every step of a batch; the records, ``[record, step]`` single precision on ``xp``.

    ``timing``, when given, receives the seconds of the stepping alone and
    the node updates it made: the reached nodes written a step, times the
    sources, times the steps.
    """
    stepper: Any = NumpyStepper(problem, drive) if xp is np else CardStepper(problem, drive, xp)
    state = stepper.state()
    out = stepper.records()
    started = last = time.time()
    for n in range(drive.steps):
        stepper.step(state, n, out)
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
            }
        )
    return out
