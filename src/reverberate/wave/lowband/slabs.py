"""One solve across several devices: the grid in slabs along ``x``, two planes a cut a step.

Independent solves on independent cards already divide a campaign's time by
the cards, so this is for what one card cannot hold, not for speed: a small
card, a larger dwelling, a higher ``fmax``. The stored columns are in the
order of their ``x``, so a slab is a run of them: the columns it owns, and
one plane of columns either side that it reads and never writes, its halo.
Every neighbour of a node is at most one plane away in ``x`` on both grids,
so one plane is enough.

A step is the solver's own, in the solver's order, with one exchange in it:

1. every slab reads its receivers and makes the box's own copies
   (:meth:`~reverberate.wave.lowband.solver.NumpyStepper.halo`) on the
   columns it owns;
2. **each cut's two planes are exchanged**: a slab's halo takes the plane
   its neighbour owns, as that neighbour holds it after its copies
   (:func:`exchange`, through :func:`move`, the one place a field crosses
   from a device to another);
3. every slab updates the nodes it owns, adds its sources and turns its two
   fields.

A slab is a :class:`~reverberate.wave.lowband.problem.Problem` of its own
(:func:`cut`), whose halo columns are of the kind that is never written, so
the steppers of :mod:`reverberate.wave.lowband.solver` run on it as they are
and the arithmetic of a node is the single solve's, operation for operation.
The records and the fields are the single solve's to the bit, which the
tests hold on ``numpy`` for both grids; on cards ``python -m
reverberate.wave.lowband slabs`` measures it and what the exchange costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from reverberate.wave.lowband.problem import Problem
from reverberate.wave.lowband.solver import (
    SPOOL_STEPS,
    CardStepper,
    Drive,
    NumpyStepper,
    records_to_host,
)

__all__ = [
    "Slab",
    "bounds_for",
    "cut",
    "exchange",
    "exchange_bytes",
    "move",
    "slabs_of",
    "solve_slabbed",
]

#: The box's own copies along ``x`` read the third plane from each end: a cut is inside them.
EDGE_PLANES = 3


@dataclass
class Slab:
    """A run of the problem's columns as a problem of its own, and what it gives and takes."""

    problem: Problem
    drive: Drive
    #: The rows of the whole drive's records this slab reads, in its own order.
    record_rows: np.ndarray
    #: The first stored column of the whole problem this slab holds, halo included.
    first_column: int
    #: The slab's own columns, as columns of the whole problem: ``[start, stop)``.
    own: tuple[int, int]
    #: The planes of ``x`` it owns: ``[start, stop)``.
    planes: tuple[int, int]

    def rows(self, start: int, stop: int) -> slice:
        """The nodes of the whole problem's columns ``start`` to ``stop``, as its own rows."""
        nz = self.problem.nz
        return slice((start - self.first_column) * nz, (stop - self.first_column) * nz)


def _planes(problem: Problem) -> np.ndarray:
    return np.asarray(problem.columns // int(problem.shape[1]), dtype=np.int64)


def bounds_for(problem: Problem, slabs: int) -> list[tuple[int, int]]:
    """The planes of ``x`` each of ``slabs`` slabs owns, the nodes written a step shared evenly.

    A cut is at least :data:`EDGE_PLANES` planes from either end of the box,
    where the box's copies along ``x`` are made inside one slab.
    """
    nx = int(problem.shape[0])
    if slabs < 1:
        raise ValueError("a grid is cut into one slab or more")
    if slabs == 1:
        return [(0, nx)]
    if nx < 2 * EDGE_PLANES * slabs:
        raise ValueError(f"a box of {nx} planes is too thin for {slabs} slabs")
    written = np.count_nonzero(problem.kind[: problem.column_count], axis=1).astype(np.float64)
    np.add.at(written, problem.bn_column, 1.0)
    per_plane = np.bincount(_planes(problem), weights=written, minlength=nx)
    share = np.cumsum(per_plane) / max(float(per_plane.sum()), 1.0)
    cuts: list[int] = []
    for k in range(1, slabs):
        at = int(np.searchsorted(share, k / slabs)) + 1
        low = (cuts[-1] if cuts else 0) + EDGE_PLANES
        cuts.append(int(min(max(at, low), nx - EDGE_PLANES * (slabs - k))))
    edges = [0, *cuts, nx]
    return [(edges[k], edges[k + 1]) for k in range(slabs)]


def cut(problem: Problem, drive: Drive, planes: tuple[int, int]) -> Slab:
    """The slab that owns the planes ``[start, stop)`` of ``x``, with its halo and its drive."""
    if max(abs(dx) for dx, _ in problem.lateral_offsets) > 1:
        raise NotImplementedError("a neighbour more than one plane away needs a wider halo")
    nz = problem.nz
    x = _planes(problem)
    start, stop = int(planes[0]), int(planes[1])
    first, last = (int(np.searchsorted(x, v)) for v in (start - 1, stop + 1))
    own = (int(np.searchsorted(x, start)), int(np.searchsorted(x, stop)))
    held = last - first
    column_of = np.full_like(problem.column_of, -1)
    column_of[problem.columns[first:last]] = np.arange(held, dtype=column_of.dtype)
    lateral = np.full((held + 1, problem.lateral.shape[1]), held, dtype=problem.lateral.dtype)
    across = problem.lateral[first:last]
    inside = (across >= first) & (across < last)
    lateral[:held] = np.where(inside, across - first, held)
    mine = np.zeros(held, dtype=bool)
    mine[own[0] - first : own[1] - first] = True
    # A node of the slab's own columns reads no column beyond the halo.
    beyond = (~inside & (across < problem.column_count))[mine]
    if bool(beyond.any()):
        raise RuntimeError("a column of the slab reads beyond its halo")
    kind = np.zeros((held + 1, nz), dtype=problem.kind.dtype)
    kind[:held][mine] = problem.kind[first:last][mine]

    def local(index: np.ndarray) -> np.ndarray:
        return np.asarray(index - first * nz, dtype=np.int64)

    def owned(index: np.ndarray) -> np.ndarray:
        column = np.asarray(index) // nz
        return np.asarray((column >= own[0]) & (column < own[1]))

    copies = []
    for dst, src in problem.copies:
        keep = owned(dst)
        if not bool(keep.any()):
            continue
        source = src[keep] // nz
        if bool(((source < first) | (source >= last)).any()):
            raise RuntimeError("a copy of the box's halo reads beyond the slab")
        copies.append((local(dst[keep]), local(src[keep])))
    boundary = (problem.bn_column >= own[0]) & (problem.bn_column < own[1])
    lossy_rows = problem.bn_lossy[boundary]
    kept = lossy_rows[lossy_rows >= 0]
    bn_lossy = np.full(lossy_rows.size, -1, dtype=problem.bn_lossy.dtype)
    bn_lossy[lossy_rows >= 0] = np.arange(kept.size, dtype=problem.bn_lossy.dtype)
    part = replace(
        problem,
        column_of=column_of,
        columns=problem.columns[first:last],
        lateral=lateral,
        kind=kind,
        copies=copies,
        bn_index=local(problem.bn_index[boundary]),
        bn_column=(problem.bn_column[boundary] - first).astype(problem.bn_column.dtype),
        bn_z=problem.bn_z[boundary],
        bn_adjacency=problem.bn_adjacency[boundary],
        bn_lossy=bn_lossy,
        lossy_material=problem.lossy_material[kept],
        lossy_ssaf=problem.lossy_ssaf[kept],
        record={"planes": [start, stop], "columns": held, "own_columns": own[1] - own[0]},
    )
    injected = owned(drive.inject_index)
    # A receiver no source reaches reads the whole problem's column of zeros: the first
    # slab reads its own.
    nowhere = drive.record_index >= problem.column_count * nz
    read = owned(drive.record_index) | (nowhere & (start == 0))
    record_index = np.where(nowhere[read], held * nz, local(drive.record_index[read]))
    return Slab(
        problem=part,
        drive=Drive(
            batch=drive.batch,
            steps=drive.steps,
            inject_index=local(drive.inject_index[injected]),
            inject_source=drive.inject_source[injected],
            inject_signal=drive.inject_signal[injected],
            record_index=np.asarray(record_index, dtype=np.int64),
            record_source=drive.record_source[read],
        ),
        record_rows=np.flatnonzero(read),
        first_column=first,
        own=own,
        planes=(start, stop),
    )


def slabs_of(problem: Problem, drive: Drive, slabs: int) -> list[Slab]:
    """``problem`` and ``drive`` in ``slabs`` slabs along ``x``; one slab is the whole."""
    return [cut(problem, drive, planes) for planes in bounds_for(problem, slabs)]


def exchange_bytes(problem: Problem, batch: int, slabs: list[Slab]) -> int:
    """Bytes that cross the cuts a step: both planes of each, every source of the batch."""
    x = _planes(problem)
    total = 0
    for left in slabs[:-1]:
        at = left.planes[1]
        for plane in (at - 1, at):
            total += int(np.count_nonzero(x == plane)) * problem.nz * batch * 4
    return total


def move(piece: Any, xp: Any, device: int | None = None, *, through_host: bool = False) -> Any:
    """A part of a field as the device that will read it holds it.

    The one place a field leaves a device. On ``numpy`` it is the part
    itself. Between cards it is a copy made on ``device``, from the other
    card's memory where the machine lets cards talk, or through the host's
    with ``through_host``.
    """
    if xp is np:
        return piece
    with xp.cuda.Device(device):
        if through_host:
            return xp.asarray(piece.get())
        return piece.copy()


def exchange(
    slabs: list[Slab],
    states: list[Any],
    xp: Any,
    devices: list[int | None],
    *,
    through_host: bool = False,
) -> None:
    """Each cut's two planes of the present field, from the slab that owns each to the other."""
    x_of = [_planes_of(slab) for slab in slabs]
    for k in range(len(slabs) - 1):
        left, right = slabs[k], slabs[k + 1]
        at = left.planes[1]
        # The plane the right slab owns first is the left slab's halo, and the reverse.
        for owner, reader, plane, o, r in (
            (right, left, at, k + 1, k),
            (left, right, at - 1, k, k + 1),
        ):
            start, stop = _columns_of_plane(owner, x_of[o], plane)
            if stop == start:
                continue
            piece = move(
                states[o].u1[owner.rows(start, stop)], xp, devices[r], through_host=through_host
            )
            if xp is np:
                states[r].u1[reader.rows(start, stop)] = piece
            else:
                with xp.cuda.Device(devices[r]):
                    states[r].u1[reader.rows(start, stop)] = piece


def _planes_of(slab: Slab) -> np.ndarray:
    return _planes(slab.problem)


def _columns_of_plane(slab: Slab, x: np.ndarray, plane: int) -> tuple[int, int]:
    """The whole problem's columns of one plane of ``x``, as the slab holds them."""
    start = slab.first_column + int(np.searchsorted(x, plane))
    stop = slab.first_column + int(np.searchsorted(x, plane + 1))
    return start, stop


def solve_slabbed(
    problem: Problem,
    drive: Drive,
    xp: Any,
    *,
    slabs: int = 2,
    devices: list[int] | None = None,
    spool_steps: int = SPOOL_STEPS,
    through_host: bool = False,
    timing: dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
) -> np.ndarray:
    """:func:`reverberate.wave.lowband.solver.solve` with the grid in ``slabs`` along ``x``.

    The records, ``[record, step]`` single precision on the host: each
    device keeps ``spool_steps`` steps of its own and hands them over block
    by block. ``devices`` are the cards, one a slab; without them every slab
    is on the present device, which is how ``numpy`` proves the cut.
    ``timing`` receives the seconds of the stepping, of which the exchange,
    and the bytes exchanged a step. ``fields``, when given, receives the
    present field of every stored node at the end, ``[node, source]``, for a
    test to hold against the single solve's.
    """
    parts = slabs_of(problem, drive, slabs)
    on: list[int | None] = [None] * len(parts)
    if xp is not np:
        on = list(devices) if devices else [int(xp.cuda.Device().id)] * len(parts)
        if len(on) != len(parts):
            raise ValueError(f"{len(parts)} slabs for {len(on)} devices")

    def held_by(device: int | None) -> Any:
        return _Here() if xp is np else xp.cuda.Device(device)

    steppers: list[Any] = []
    states: list[Any] = []
    blocks: list[Any] = []
    width = max(1, min(int(spool_steps), drive.steps))
    for part, device in zip(parts, on, strict=True):
        with held_by(device):
            stepper: Any = (
                NumpyStepper(part.problem, part.drive)
                if xp is np
                else CardStepper(part.problem, part.drive, xp)
            )
            steppers.append(stepper)
            states.append(stepper.state())
            blocks.append(stepper.records(width))
    out = np.zeros((drive.record_index.size, drive.steps), dtype=np.float32)
    exchange_s = 0.0
    started = time.time()
    for n in range(drive.steps):
        column = n % width
        for stepper, state, block, device in zip(steppers, states, blocks, on, strict=True):
            with held_by(device):
                stepper.read(state, column, block)
                stepper.halo(state)
        t0 = time.time()
        exchange(parts, states, xp, on, through_host=through_host)
        exchange_s += time.time() - t0
        for stepper, state, device in zip(steppers, states, on, strict=True):
            with held_by(device):
                stepper.update(state)
                stepper.inject(state, n)
        if column == width - 1 or n == drive.steps - 1:
            for part, block in zip(parts, blocks, strict=True):
                out[part.record_rows, n - column : n + 1] = records_to_host(block[:, : column + 1])
    for stepper, device in zip(steppers, on, strict=True):
        with held_by(device):
            stepper.finish()
    if timing is not None:
        seconds = time.time() - started
        updates = float(problem.updated) * drive.batch * drive.steps
        timing.update(
            {
                "seconds": seconds,
                "exchange_s": exchange_s,
                "exchange_bytes_a_step": exchange_bytes(problem, drive.batch, parts),
                "node_updates": updates,
                "updates_per_s": updates / max(seconds, 1e-9),
                "batch": drive.batch,
                "steps": drive.steps,
                "slabs": [dict(part.problem.record) for part in parts],
            }
        )
    if fields is not None:
        whole = np.zeros((problem.nodes, drive.batch), dtype=np.float32)
        for part, state in zip(parts, states, strict=True):
            rows = slice(part.own[0] * problem.nz, part.own[1] * problem.nz)
            whole[rows] = records_to_host(state.u1[part.rows(*part.own)])
        fields["u1"] = whole
    return out


class _Here:
    """The present device, where there is no other: what ``numpy`` steps on."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *held: object) -> None:
        return None
