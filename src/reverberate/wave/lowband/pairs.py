"""The pairs campaign on the batched solver: the same bundle, the same cache, another engine.

:class:`LowbandPairs` is :class:`reverberate.accel.pairs.PairsCampaign` with
its solve replaced. The bundle, the grid's voxelisation, the arrays stood at
the cells, the cache and its keys are the campaign's own, so the trace reads
the pairs exactly as it reads those of the present engine. What changes:

- the source positions are solved in batches on each card, the grid cut to
  what they reach (:mod:`reverberate.wave.lowband.problem`), and no engine
  process, comms file, pressure file or log exists per source;
- a batch is packed by the card's memory: a source costs its two fields and
  its boundary states, and four bytes a step for every node it is read at.
  A source heard at more cells than a card holds records for is solved more
  than once, each time for some of its cells;
- the records are filtered, resampled and fitted on the card
  (:mod:`reverberate.wave.lowband.fit`) and each response is written in the
  cache form under its pair's key.

**The grid may be another than the bundle's.** ``scheme`` and ``ppw`` name
it: left alone it is the bundle's Cartesian grid at 10.5 points per
wavelength, voxelised on the card as a campaign's is, and the result is the
present engine's to rounding. The face centred grid, or the Cartesian one at
other points per wavelength, is voxelised under its own key (PFFDTD's
voxeliser on the host for the first, which the card's does not make), and
its pairs are cached under that key: a pair's key names the grid and the
solver, so the two never meet in a cache.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np

from reverberate.accel.pairs import PairCache, PairsCampaign
from reverberate.spatial.lowband import LOW_RATE_HZ
from reverberate.wave.comms import Grid, engine_indices, interp_weights, load_grid
from reverberate.wave.lowband.fit import FITS, CellEncoder, level_scale
from reverberate.wave.lowband.outside import CLOSURES
from reverberate.wave.lowband.problem import Problem, load_problem
from reverberate.wave.lowband.scheme import CARTESIAN, SCHEMES, Scheme
from reverberate.wave.lowband.solver import (
    BOUNDARIES,
    SPOOL_STEPS,
    CardStepper,
    Drive,
    drive_for,
    solve,
    step_bytes,
    steps_for,
)
from reverberate.wave.lowband.walls import DEFAULT_BRANCHES, WallFit

__all__ = [
    "SOLVER",
    "Item",
    "LowbandPairs",
    "batch_bytes",
    "batch_capacity",
    "estimate",
    "MEMORY_GB_S",
    "PROBE_S",
    "halves",
    "last_short",
    "merge_bundles",
    "node_indices",
    "pack_batches",
    "probe",
    "solver_name",
]

#: The engine's name in a pair's key; a change between the grid and the cache form changes it.
SOLVER = "reverberate.wave.lowband/1"

#: Of a card's free memory, what a batch may take; the rest is the fit's working arrays.
MEMORY_SHARE = 0.8
#: The most sources of one launch: the boundary kernel's second grid axis.
BATCH_LIMIT = 4096
#: The most sources of a launch a queue hands out. A source costs the same card seconds
#: alone or among others (3.2 ms a step a source on an RTX 3080 at eight), and what is
#: left at a run's end for the last cards is one launch each: a short one.
LAUNCH_SOURCES = 8
#: Of the host's memory, what the records of the launches running at once may take.
HOST_RECORDS_SHARE = 0.5


#: Measured on one RTX 3080 20 GB at 0.136 USD/h, instance 54201838, 2026-10-05, on the
#: bundle's grid of hssd_0076 to 1500 Hz (47.4 M reached nodes of 63.3 M, 2.69 M lossy nodes
#: of 11 branches, 32 769 steps): card seconds a source position, card seconds a pair
#: (filters, resampling, fit, the file), and what is done once (the grid cut to its sources'
#: reach, the fit's operator).
MEASURED_ON = "1 x RTX 3080 20 GB"
MEASURED_RATE_USD_PER_HOUR = 0.136
SOLVE_S_AT_1500 = 110.0
PAIR_S = 0.31
ONCE_S = 95.0
REACHED_NODES_AT_1500 = 47_371_003
STEPS_AT_1500 = 32_769
#: The grid those were measured on, and how a solve goes with the points per wavelength of
#: another Cartesian grid: the same card took 36 s a source position at 7.2 points against
#: 110 s at 10.5, a power of 2.96. Fewer nodes and fewer steps would give 4; the boundary's
#: share does not thin as the air does.
MEASURED_PPW = 10.5
PPW_EXPONENT = 2.96
#: The measured card's memory as an offer states it (GiB), and the cells whose records one
#: solve held on it: the realistic recipe's 1646 positions took 1763 solves there.
MEASURED_CARD_GIB = 20.0
CELLS_A_SOLVE_MEASURED = 57
#: The seconds those solves simulated: a solve, and a cell's records, go as them.
MEASURED_DURATION_S = 1.2
#: Bytes of record a cell takes on the measured grid: 984 nodes, 4 bytes a step.
CELL_RECORD_BYTES_AT_1500 = 984 * 4.0 * STEPS_AT_1500


def cells_a_solve(
    card_gib: float,
    *,
    ppw: float | None = None,
    fmax_hz: float = 1500.0,
    duration_s: float = MEASURED_DURATION_S,
) -> int:
    """The cells one solve can be read at on a card of ``card_gib``; 0 when it holds none.

    The batch's share of the card (:data:`MEMORY_SHARE`) less what the
    grid, one source and the fit take, over a cell's records. What they
    take is not counted from the grid here: it is the one figure that
    makes this give the 57 cells measured on a 20 GiB card, 9.8 GB, scaled
    as the nodes of another grid. A cell's array has the same number of
    nodes on every grid (its radius is twelve steps) and its records go as
    the steps, which go as the grid's step and as the seconds simulated.
    """
    points = MEASURED_PPW if ppw is None else float(ppw)
    grid = (points / MEASURED_PPW) * (fmax_hz / 1500.0)
    record = CELL_RECORD_BYTES_AT_1500 * grid * (float(duration_s) / MEASURED_DURATION_S)
    share = MEMORY_SHARE * MEASURED_CARD_GIB * 2.0**30
    taken = (share - CELLS_A_SOLVE_MEASURED * CELL_RECORD_BYTES_AT_1500) * grid**3
    return int(max(0.0, MEMORY_SHARE * float(card_gib) * 2.0**30 - taken) // record)


def solves_needed(
    cells_a_position: list[int],
    card_gib: float,
    *,
    ppw: float | None = None,
    duration_s: float = MEASURED_DURATION_S,
) -> int | None:
    """Solves for positions heard at these many cells each; ``None`` on a card too small."""
    held = cells_a_solve(card_gib, ppw=ppw, duration_s=duration_s)
    if held < 1:
        return None
    return int(sum(-(-int(count) // held) for count in cells_a_position if count > 0))


def estimate(
    sources: int,
    pairs: int,
    *,
    fmax_hz: float,
    duration_s: float = 1.2,
    rate_usd_per_hour: float = MEASURED_RATE_USD_PER_HOUR,
    cards: int = 1,
    ppw: float | None = None,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Seconds and USD of ``sources`` solves and ``pairs`` responses on the batched solver.

    The terms are this module's constants, measured on :data:`MEASURED_ON`;
    another card needs its own, and ``rate_usd_per_hour`` is then that
    card's. A solve goes as the fourth power of ``fmax`` and as the window;
    ``cards`` of the measured kind each run their own batches. The keys are
    those of :func:`reverberate.accel.pairs.estimate`.

    ``sources`` is the number of solves: a source position heard at more
    cells than a card holds records for is solved more than once, which
    the caller counts. ``ppw`` prices another Cartesian grid than the
    measured one by :data:`PPW_EXPONENT`; ``speed`` is the card's
    throughput over the measured card's.
    """
    scale = fmax_hz / 1500.0
    points = MEASURED_PPW if ppw is None else float(ppw)
    grid = (points / MEASURED_PPW) ** PPW_EXPONENT
    solve_s = SOLVE_S_AT_1500 * scale**4 * duration_s / 1.2 * grid / speed
    pair_s = PAIR_S / speed
    seconds = ONCE_S + (sources * solve_s + pairs * pair_s) / max(1, cards)
    per_second = rate_usd_per_hour / 3600.0
    return {
        "fmax_hz": fmax_hz,
        "ppw": points,
        "grid_nodes": REACHED_NODES_AT_1500 * scale**3 * (points / MEASURED_PPW) ** 3,
        "steps": STEPS_AT_1500 * scale * duration_s / 1.2 * points / MEASURED_PPW,
        "solves": int(sources),
        "stencil_s_per_source": round(solve_s, 2),
        "per_cell_s": {"filters_and_fit_s": round(pair_s, 4)},
        "cell_s": round(pair_s, 4),
        "prepare_s": ONCE_S,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600.0, 3),
        "billed_rate_usd_per_hour": rate_usd_per_hour,
        "usd": round(seconds * per_second, 2),
        "usd_per_source": round(solve_s / max(1, cards) * per_second, 5),
        "usd_per_pair": round(pair_s / max(1, cards) * per_second, 6),
        "cache_gb": round(pairs * 64 * duration_s * LOW_RATE_HZ * 4 / 1e9, 2),
        "measured_on": MEASURED_ON,
        "cards": cards,
        "solver": SOLVER,
    }


def cores_lent() -> int:
    """The cores this process may really use: the cgroup's quota where there is one."""
    import os

    count = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count - 2)


def solver_name(
    scheme: Scheme,
    ppw: float,
    *,
    walls: int | None = DEFAULT_BRANCHES,
    outside: str | None = None,
    fit: str = "time",
) -> str:
    """What a pair's key and a pack's provenance say solved it.

    Each option that changes a response adds its own words, so that pairs
    made with it are other pairs: the walls fitted again with ``walls``
    branches, which a campaign's are unless told ``None``
    (:mod:`reverberate.wave.lowband.walls`), the air outside the outer walls
    cut off and its openings closed as ``outside`` says
    (:mod:`reverberate.wave.lowband.outside`), the fit made in its spectra
    (:class:`reverberate.wave.lowband.fit.SpectralChain`). Where a card
    updates the boundary is not in it: the bits are the same.
    """
    name = f"{SOLVER} {scheme.name} at {ppw:g} points per wavelength"
    if walls is not None:
        name += f", {WallFit(int(walls)).name()}"
    if outside is not None:
        if outside not in CLOSURES:
            raise ValueError(f"an opening is closed as one of {CLOSURES}, not {outside!r}")
        name += f", the air outside the outer walls cut off, its openings {outside} (outside/1)"
    if fit not in FITS:
        raise ValueError(f"a fit is one of {FITS}, not {fit!r}")
    if fit == "spectra":
        name += ", fitted in its spectra (spectra/1)"
    return name


#: What a card's memory moves a second, GB, as its maker states it: the scale a kernel's
#: bytes are read against. A card that is not here is measured and not judged.
MEMORY_GB_S = {
    "RTX 3080": 760.0,
    "RTX 3080 Ti": 912.0,
    "RTX 3090": 936.0,
    "RTX 3090 Ti": 1008.0,
    "RTX 4080": 717.0,
    "RTX 4090": 1008.0,
    "RTX A5000": 768.0,
    "RTX A6000": 768.0,
    "A100": 1555.0,
}
#: Of a card's memory rate, what the solver's kernels moved where they were measured: 517 GB/s
#: of the RTX 3090's 936 on the first whole scene, 453 and 394 of the RTX 3080's 760.
MEASURED_SHARE = 0.55
#: Under this share of what the bytes predict, a card is said to be slow.
PROBE_FLOOR = 0.7
#: Seconds of steps a worker times on its card before its first launch.
PROBE_S = 10.0


def memory_rate_of(card: str) -> float | None:
    """The memory rate of a card by its name as the driver gives it, GB/s; ``None`` unknown."""
    plain = " ".join(str(card).replace("NVIDIA", "").replace("GeForce", "").split())
    for name in sorted(MEMORY_GB_S, key=len, reverse=True):
        if name.lower() in plain.lower():
            return MEMORY_GB_S[name]
    return None


def probe(
    problem: Problem,
    xp: Any,
    *,
    seconds: float = PROBE_S,
    boundary: str = "apart",
    card: str = "",
    stepper: Any = None,
) -> dict[str, Any]:
    """A few seconds of the campaign's own step on its own grid, one source, against its bytes.

    The node updates a second, the bytes of memory they are
    (:func:`reverberate.wave.lowband.solver.step_bytes`), and, for a card
    whose memory rate is known, what the bytes predict at the share the
    kernels were measured at. ``slow`` says the card made under
    :data:`PROBE_FLOOR` of that: a card to give back, or a grid the kernel
    is poor on, and either way a figure to read before hours are spent.
    The fields are zero and stay zero: a step costs the same.
    """
    from reverberate.wave.lowband.solver import NumpyStepper

    nothing = Drive.nothing()
    if stepper is None:
        stepper = (
            NumpyStepper(problem, nothing)
            if xp is np
            else CardStepper(problem, nothing, xp, boundary=boundary)
        )
    state = stepper.state()
    out = stepper.records(1)
    for _ in range(2):  # the kernels compiled, the pages touched
        stepper.step(state, 0, out, 0)
    stepper.finish()
    started = time.time()
    steps = 0
    while time.time() - started < seconds:
        for _ in range(1 if xp is np else 32):
            stepper.step(state, 0, out, 0)
            steps += 1
        stepper.finish()
    elapsed = max(time.time() - started, 1e-9)
    moved = step_bytes(problem, boundary)
    record: dict[str, Any] = {
        "card": card,
        "boundary": boundary,
        "seconds": round(elapsed, 2),
        "steps": steps,
        "updates_per_s": float(problem.updated) * steps / elapsed,
        "bytes_a_step": moved["step"],
        "walls_share_of_bytes": round(float(moved["walls_share"]), 4),
        "gb_per_s": moved["step"] * steps / elapsed / 1e9,
    }
    rate = None if xp is np else memory_rate_of(card)
    if rate is not None:
        expected = MEASURED_SHARE * rate
        record["card_gb_per_s"] = rate
        record["expected_gb_per_s"] = expected
        record["expected_updates_per_s"] = expected * 1e9 / moved["step"] * problem.updated
        record["of_expected"] = record["gb_per_s"] / expected
        record["slow"] = bool(record["of_expected"] < PROBE_FLOOR)
    return record


def last_short(launches: list[list[Item]], cards: int) -> list[list[Item]]:
    """The launches in the order a queue should hand them out: the longest first, the last short.

    A launch's time goes as its sources, each the same card seconds alone
    or among others. When the queue is empty every card finishes what it
    holds while the others wait, half a launch each in the mean: so the
    launches of most sources go first, and the last two a card are parted
    into launches of one source, which leaves each card idle half a source
    position at the end and not half of eight. Among launches of one
    length the order given is kept. A run of no more than two launches a
    card has no end to speak of and is handed out as it is.
    """
    ordered = sorted(launches, key=lambda batch: -len(batch))
    tail: list[list[Item]] = []
    wanted = 2 * max(1, cards)
    if len(ordered) <= wanted:
        return ordered
    while ordered and len(tail) < wanted:
        batch = ordered.pop()
        tail = [[item] for item in batch] + tail
    return ordered + tail


def merge_bundles(bundles: list[Path], out: Path) -> dict[str, Any]:
    """Several recipes' pairs bundles of one dwelling as one: each position solved once.

    Two recipes of a dwelling share most of their source positions, which
    lie on its rails and stations, and almost none of their pairs, because
    a listening cell is where that recipe's listener passed
    (``docs/open-questions/performance-audit.md``, section 6). A pair's key
    holds its two positions in whole millimetres and no recipe, so a
    campaign over the union writes every recipe's pairs under the names
    each recipe's trace will ask for. Positions are one when they are the
    same to the millimetre, the key's own rule; a source position is heard
    at the union of the cells any recipe hears it at.

    The bundles must be of one grid, one window and one encoder. The models
    are the first bundle's, linked and not copied.
    """
    bundles = [Path(b) for b in bundles]
    out = Path(out)
    if not bundles:
        raise ValueError("no bundle to merge")
    specs = [json.loads((b / "campaign.json").read_text()) for b in bundles]
    same = ("kind", "dwelling", "bands", "ppw", "tc", "rh", "order", "fit_order", "encoder")
    for bundle, spec in zip(bundles[1:], specs[1:], strict=True):
        for name in same:
            if spec.get(name) != specs[0].get(name):
                raise ValueError(f"{bundle} is not of the first bundle's {name}")

    def millimetres(positions: np.ndarray) -> list[tuple[int, ...]]:
        return [tuple(round(float(v) * 1000.0) for v in row) for row in positions]

    sources: dict[tuple[int, ...], int] = {}
    cells: dict[tuple[int, ...], int] = {}
    source_m: list[np.ndarray] = []
    cell_m: list[np.ndarray] = []
    heard: list[set[int]] = []
    asked_positions = asked_pairs = 0
    for bundle in bundles:
        own_sources = np.load(bundle / "sources.npy").reshape(-1, 3)
        own_cells = np.load(bundle / "cells.npy").reshape(-1, 3)
        own_heard = json.loads((bundle / "heard_at.json").read_text())
        cell_of = []
        for key, position in zip(millimetres(own_cells), own_cells, strict=True):
            if key not in cells:
                cells[key] = len(cell_m)
                cell_m.append(position)
            cell_of.append(cells[key])
        for key, position, listed in zip(
            millimetres(own_sources), own_sources, own_heard, strict=True
        ):
            if key not in sources:
                sources[key] = len(source_m)
                source_m.append(position)
                heard.append(set())
            heard[sources[key]].update(cell_of[int(c)] for c in listed)
            asked_positions += 1
            asked_pairs += len(listed)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sources.npy", np.asarray(source_m, dtype=float).reshape(-1, 3))
    np.save(out / "cells.npy", np.asarray(cell_m, dtype=float).reshape(-1, 3))
    listed_all = [sorted(found) for found in heard]
    (out / "heard_at.json").write_text(json.dumps(listed_all))
    models = out / "models"
    if not models.exists():
        models.symlink_to((bundles[0] / "models").resolve(), target_is_directory=True)
    pairs = sum(len(found) for found in listed_all)
    record = {
        "recipes": len(bundles),
        "source_positions_asked": asked_positions,
        "source_positions": len(source_m),
        "pairs_asked": asked_pairs,
        "pairs": pairs,
        "cells": len(cell_m),
    }
    campaign = dict(specs[0])
    campaign.update(
        {
            "points": max((len(found) for found in listed_all), default=0),
            "source_positions": len(source_m),
            "cells": len(cell_m),
            "pairs": pairs,
            "merged": {**record, "bundles": [str(b) for b in bundles]},
        }
    )
    campaign.pop("estimate", None)
    (out / "campaign.json").write_text(json.dumps(campaign, indent=1))
    return record


def low_grid(
    models: Path,
    storey_scene: str,
    fmax_hz: float,
    *,
    scheme: str = CARTESIAN.name,
    ppw: float | None = None,
    bundle_ppw: float = 10.5,
    walls: int | None = DEFAULT_BRANCHES,
    outside: str | None = None,
    fit: str = "time",
) -> tuple[Any, str]:
    """The grid's spec and the solver's name, as a campaign of these options keys its pairs.

    Read on the laptop as on the machine, from the export alone: a bundle
    carries the pairs a machine will ask for only if it names them as that
    machine will. Left alone, the bundle's Cartesian grid at ``bundle_ppw``;
    another scheme or other points per wavelength is another spec, whose
    ``key`` is another grid's.
    """
    from reverberate.experiments.run import scene_spec
    from reverberate.wave.remote_voxelise import grid_shape_of
    from reverberate.wave.voxelise import nh_for

    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}, expected one of {sorted(SCHEMES)}")
    held = SCHEMES[scheme]
    points = float(ppw) if ppw is not None else held.ppw
    scene, _, _ = scene_spec(models, storey_scene, float(fmax_hz))
    if held.fcc or points != float(bundle_ppw):
        shape = grid_shape_of(scene.model_json, float(fmax_hz), points)
        scene = replace(scene, ppw=points, fcc=held.fcc, nh=nh_for(shape))
    return scene, solver_name(held, points, walls=walls, outside=outside, fit=fit)


def node_indices(positions: np.ndarray, grid: Grid) -> np.ndarray:
    """Engine flat indices of nodes given by their own coordinates, all at once."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    subs = []
    for axis, values in enumerate((grid.xv, grid.yv, grid.zv)):
        index = np.rint((positions[:, axis] - values[0]) / grid.h).astype(np.int64)
        if (index < 0).any() or (index >= values.size).any():
            raise ValueError("a node lies outside the grid")
        if not np.allclose(values[index], positions[:, axis], rtol=0.0, atol=1e-6 * grid.h):
            raise ValueError("a receiver is not a node of the grid")
        subs.append(index)
    nx, ny, nz = grid.shape
    return np.asarray(engine_indices((subs[0] * ny + subs[1]) * nz + subs[2], grid))


@dataclass(frozen=True)
class Item:
    """One solve of one source position: the cells it is read at, and its records' rows."""

    source: int
    cells: tuple[int, ...]
    rows: int


def batch_capacity(problem: Problem, steps: int, free_bytes: float, *, rows: int = 0) -> int:
    """Sources a card holds at once, each read at ``rows`` nodes; at least one."""
    per_source = problem.bytes_per_source() + 4.0 * steps * (rows + 8)
    usable = MEMORY_SHARE * free_bytes - problem.bytes_shared()
    return int(max(1, min(BATCH_LIMIT, usable // per_source)))


def pack_batches(
    items: list[Item], problem: Problem, steps: int, free_bytes: float
) -> list[list[Item]]:
    """The items in batches a card holds: the largest records first, each batch filled.

    Sorted by rows so that a batch's sources are alike, and filled until the
    next would pass the card's share; an item is never split here.
    """
    budget = MEMORY_SHARE * free_bytes - problem.bytes_shared()
    fixed = problem.bytes_per_source() + 32.0 * steps
    batches: list[list[Item]] = []
    current: list[Item] = []
    used = 0.0
    for item in sorted(items, key=lambda i: (-i.rows, i.source)):
        need = fixed + 4.0 * steps * item.rows
        if current and (used + need > budget or len(current) >= BATCH_LIMIT):
            batches.append(current)
            current, used = [], 0.0
        current.append(item)
        used += need
    if current:
        batches.append(current)
    return batches


def batch_bytes(batch: list[Item], problem: Problem, steps: int) -> float:
    """What a launch holds on its card besides the grid: its sources' fields and its records."""
    fixed = problem.bytes_per_source() + 32.0 * steps
    return float(sum(fixed + 4.0 * steps * item.rows for item in batch))


def halves(batch: list[Item], rows_of: Any) -> list[list[Item]] | None:
    """A launch its card does not hold, as two that ask for less; ``None`` when nothing is less.

    Several sources are parted first, which halves their fields and their
    records; one source alone is then solved twice, each time for half of
    its cells. ``rows_of`` gives a cell's nodes. One source read at one
    cell cannot be made smaller.
    """
    if len(batch) > 1:
        middle = len(batch) // 2
        return [batch[:middle], batch[middle:]]
    item = batch[0]
    if len(item.cells) < 2:
        return None
    middle = len(item.cells) // 2
    parts = (item.cells[:middle], item.cells[middle:])
    return [[Item(item.source, part, sum(int(rows_of(c)) for c in part))] for part in parts]


@dataclass
class LowbandPairs(PairsCampaign):
    """The pairs of a bundle on the batched solver, one worker a card."""

    scheme: str = CARTESIAN.name
    ppw: float | None = None
    #: Sources at once, at most; ``None`` is what the card's memory holds.
    batch: int | None = None
    #: Where a launch's records are kept (:func:`reverberate.wave.lowband.solver.solve`):
    #: on the device, or on the host as a queue's launches keep them.
    records_on: str = "device"
    #: Where a card updates a lossy node's branches: the same bits either way, and the
    #: kernel apart the faster on a card (:class:`reverberate.wave.lowband.solver.CardStepper`).
    boundary: str = "apart"
    #: Branches a material is fitted again with for the band; ``None`` is the materials as
    #: they are (:mod:`reverberate.wave.lowband.walls`). In the pairs' keys.
    walls: int | None = DEFAULT_BRANCHES
    #: ``"open"`` or ``"rigid"``: the air between the outer walls and the shell cut off
    #: (:mod:`reverberate.wave.lowband.outside`); ``None`` is the grid as exported. In the keys.
    outside: str | None = None
    #: ``"time"`` or ``"spectra"``: how the records reach the fit
    #: (:class:`reverberate.wave.lowband.fit.SpectralChain`). In the keys.
    fit: str = "time"
    #: Seconds of steps each worker times on its card before its first launch; 0 is none.
    probe_s: float = PROBE_S

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}, expected one of {sorted(SCHEMES)}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"the boundary is updated in one of {BOUNDARIES}")
        self.grid_scheme = SCHEMES[self.scheme]
        self.grid_ppw = float(self.ppw) if self.ppw is not None else self.grid_scheme.ppw
        self.bundle_grid = not self.grid_scheme.fcc and self.grid_ppw == float(self.spec["ppw"])
        self.spec["solver"] = solver_name(
            self.grid_scheme, self.grid_ppw, walls=self.walls, outside=self.outside, fit=self.fit
        )
        self.probes: list[dict[str, Any]] = []
        if not self.bundle_grid:
            key = self.grid_key()
            self.spec["bands"]["low"]["cache_key"] = key
            self.cache = PairCache(self.out / "pairs", key)
        self.batches: list[dict[str, Any]] = []

    # ---- the grid --------------------------------------------------------------------------

    def grid_key(self) -> str:
        """The cache key of this campaign's grid, where it is not the bundle's."""
        return str(self.scene_spec().key)

    def scene_spec(self) -> Any:
        """The grid's spec: the bundle's, with this campaign's scheme and points per wavelength."""
        scene, _ = low_grid(
            self.models,
            str(self.spec["storey_scene"]),
            float(self.spec["bands"]["low"]["fmax_hz"]),
            scheme=self.scheme,
            ppw=self.grid_ppw,
            bundle_ppw=float(self.spec["ppw"]),
        )
        return scene

    def voxelise(self) -> dict[str, Any]:
        """The campaign's own for the bundle's grid; any other is made under its own key."""
        if self.bundle_grid:
            return super().voxelise()
        from reverberate.wave.voxelise import entry_for, voxelise

        scene = self.scene_spec()
        entry = entry_for(scene)
        if entry.complete:
            self.say(f"voxelise low: entry {scene.key} already installed")
            return {"low": {"cached": True}}
        t0 = time.time()
        if self.grid_scheme.fcc:
            # The card's voxeliser makes the Cartesian grid only: PFFDTD's own, on the host,
            # on the cores the machine really lends (a container counts its host's).
            entry = voxelise(scene, nprocs=cores_lent())
            record: dict[str, Any] = dict(entry.manifest)
            record["computed_on"] = "host"
        else:
            import shutil

            from reverberate.accel.voxelise import voxelise_scene
            from reverberate.wave.remote_voxelise import install_entry

            staging = self.out / "vox" / "low.partial"
            if staging.exists():
                shutil.rmtree(staging)
            report = voxelise_scene(
                scene.model_json,
                staging,
                mat_folder=scene.mat_folder,
                mat_files=dict(scene.mat_files),
                fmax=scene.fmax,
                ppw=scene.ppw,
                nh=int(scene.nh or 0),
                tc=scene.tc,
                rh=scene.rh,
                xp=self.xp,
                say=lambda m: self.say(f"  vox low | {m}"),
            )
            record = report.record()
            record["computed_on"] = "gpu" if self.xp is not np else "host"
            install_entry(scene, staging, record, time.time() - t0)
        self.say(
            f"voxelise low: {self.grid_scheme.name} at {self.grid_ppw:g} points per wavelength"
            f" in {(time.time() - t0) / 60:.1f} min, key {scene.key}"
        )
        return {"low": record}

    # ---- the arrays ------------------------------------------------------------------------

    def place(self) -> dict[str, Any]:
        record = super().place()
        self.grid = load_grid(self.entry_path)
        self.steps = steps_for(self.durations_s["low"], self.grid.Ts)
        self.scale = level_scale(
            self.grid_scheme, self.fmax_hz["low"], float(self.grid.Ts), self.grid_ppw
        )
        self.nodes_of_cell: dict[int, np.ndarray] = {}
        return record

    def cell_nodes(self, cell: int) -> np.ndarray:
        """The engine indices of a cell's array, in the array's order."""
        if cell not in self.nodes_of_cell:
            design = self.designs[cell]
            assert design is not None
            self.nodes_of_cell[cell] = node_indices(design.positions, self.grid)
        return self.nodes_of_cell[cell]

    def encoder_on(self, xp: Any) -> CellEncoder:
        return CellEncoder(
            scheme=self.grid_scheme,
            grid_rate_hz=1.0 / float(self.grid.Ts),
            grid_step_m=float(self.grid.h),
            sound_speed_m_s=self.sound_speed_m_s(),
            fmax_hz=self.fmax_hz["low"],
            settings=self.encoder_settings,
            xp=xp,
            samples=round(self.durations_s["low"] * LOW_RATE_HZ),
            scale=self.scale,
            fit=self.fit,
        )

    # ---- the solve -------------------------------------------------------------------------

    def items(self, wanted: list[int], rows_limit: int) -> list[Item]:
        """Every solve to make: a source's cells together unless their records pass the limit."""
        items = []
        for source in wanted:
            chunk: list[int] = []
            rows = 0
            for cell in self.todo(source):
                count = int(self.cell_nodes(cell).size)
                if chunk and rows + count > rows_limit:
                    items.append(Item(source, tuple(chunk), rows))
                    chunk, rows = [], 0
                chunk.append(cell)
                rows += count
            if chunk:
                items.append(Item(source, tuple(chunk), rows))
        return items

    def free_bytes(self, xp: Any) -> float:
        """What this device may hold: the card's free memory, or a part of the host's."""
        if xp is np:
            from reverberate.accel.solve import host_memory_gb

            return 0.25 * host_memory_gb() * 1e9
        from reverberate.compute import card_free_bytes

        free = card_free_bytes(xp)
        return float(free)

    def release(self, xp: Any) -> None:
        """Give the device back what the pool holds free, before a launch and after a refusal.

        The pool keeps a block it was given until every part of it is free.
        A launch's records are one block of gigabytes; left in the pool, the
        next launch's small arrays are cut from it and hold the whole of it,
        and that launch's own records are then asked of a device that no
        longer has them. So measured on 8 x RTX 3090 (2026-10-05): a launch
        of 10.8 GB of records refused with 15.0 GB held, of which 8.4 GB
        were the launch before's.
        """
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()

    def fits(self, problem: Problem, batch: list[Item], xp: Any) -> bool:
        """Whether this card, as it is now, holds a launch: its share of what is really free."""
        usable = MEMORY_SHARE * self.free_bytes(xp) - problem.bytes_shared()
        return batch_bytes(batch, problem, self.steps) <= usable

    def parted(self, batch: list[Item]) -> list[list[Item]] | None:
        """:func:`halves` of a launch, without the pairs a first try already wrote."""
        left = [
            Item(item.source, cells, sum(int(self.cell_nodes(c).size) for c in cells))
            for item in batch
            for cells in [
                tuple(c for c in item.cells if not self.cache.has(self.key_of(item.source, c)))
            ]
            if cells
        ]
        if not left:
            return []
        return halves(left, lambda cell: self.cell_nodes(cell).size)

    def run_batch(
        self, problem: Problem, batch: list[Item], encoder: CellEncoder, xp: Any
    ) -> dict[str, Any]:
        """One launch: the batch solved, every cell of it encoded and cached.

        With the campaign's ``records_on`` ``"host"`` the card holds the
        fields and a block of the records, and the fit is handed the records
        back a few cells at a time: the same responses, to the bit.
        """
        records_on = self.records_on
        receivers = [np.concatenate([self.cell_nodes(c) for c in item.cells]) for item in batch]
        sources = np.asarray([self.sources[item.source] for item in batch], dtype=float)
        drive = drive_for(problem, self.grid, sources, receivers, self.durations_s["low"])
        timing: dict[str, Any] = {}
        t0 = time.time()
        stepper = None
        if xp is not np:
            # The grid and the kernels are on the card for the campaign: a launch brings
            # its sources and its records, and nothing of the grid again.
            stepper = CardStepper(
                problem, drive, xp, boundary=self.boundary, shared=self.grid_on_card(problem, xp)
            )
        records = solve(
            problem,
            drive,
            xp,
            say=lambda m: self.say(f"    {m}"),
            timing=timing,
            records_on=records_on,
            stepper=stepper,
        )
        del stepper
        solve_s = time.time() - t0
        if records_on == "host" and xp is not np:
            # The fields are freed with the stepper: the fit's arrays take their place.
            xp.get_default_memory_pool().free_all_blocks()
        t0 = time.time()
        spans = [(item.source, cell) for item in batch for cell in item.cells]
        designs = [self.designs[cell] for _, cell in spans]
        responses = encoder.cells(
            records, [d.positions - d.centre for d in designs if d is not None]
        )
        for (source, cell), design, response in zip(spans, designs, responses, strict=True):
            assert design is not None
            self.cache.write(
                self.key_of(source, cell),
                response,
                {
                    "source_m": [float(v) for v in self.sources[source]],
                    "cell_m": [float(v) for v in self.cells[cell]],
                    "centre_m": [float(v) for v in design.centre],
                    "fmax_hz": self.fmax_hz["low"],
                    "scale": self.scale,
                    "solver": str(self.spec["solver"]),
                },
            )
        pairs = len(spans)
        del records
        return {
            "sources": [item.source for item in batch],
            "cells": [len(item.cells) for item in batch],
            "batch": len(batch),
            "pairs": pairs,
            "rows": int(drive.record_index.size),
            "solve_s": round(solve_s, 3),
            "encode_s": round(time.time() - t0, 3),
            "updates_per_s": timing["updates_per_s"],
            "node_updates": timing["node_updates"],
        }

    def grid_on_card(self, problem: Problem, xp: Any) -> Any:
        """The grid as this thread's card holds it, uploaded once and kept for the campaign."""
        held: dict[tuple[int, int], CardStepper] | None = getattr(self, "_grids", None)
        if held is None:
            held = {}
            self._grids = held
        key = (id(problem), threading.get_ident())
        if key not in held:
            held[key] = CardStepper(problem, Drive.nothing(), xp, boundary=self.boundary)
        return held[key]

    def probed(self, problem: Problem, xp: Any, card: int | None = None) -> dict[str, Any] | None:
        """:func:`probe` of this worker's card, said and kept; ``None`` where it is not asked."""
        if self.probe_s <= 0 or xp is np:
            return None
        name = str(self.status.get("device", {}).get("gpu") or "")
        record = probe(
            problem,
            xp,
            seconds=self.probe_s,
            boundary=self.boundary,
            card=name,
            stepper=self.grid_on_card(problem, xp),
        )
        record["device"] = card
        said = (
            f"probe: {record['updates_per_s']:.3g} node updates/s,"
            f" {record['gb_per_s']:.0f} GB/s of memory"
        )
        if "of_expected" in record:
            said += (
                f", {100 * record['of_expected']:.0f} % of what the bytes predict on a"
                f" {name} ({record['expected_updates_per_s']:.3g})"
            )
            if record["slow"]:
                said += ": THIS CARD IS SLOW, under 70 % of its bytes"
        self.say(said)
        with self._status_lock:
            self.probes.append(record)
        state = self.out / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / f"probe.{os.getpid()}.{card if card is not None else 0}.json").write_text(
            json.dumps(record, indent=1)
        )
        return record

    def wanted_sources(self) -> list[int]:
        """The source positions with a pair still to make."""
        return [s for s in range(self.sources.shape[0]) if self.todo(s)]

    def problem_for(self, wanted: list[int]) -> Problem:
        """The grid cut to what the sources of ``wanted`` reach, and its record."""
        seeds = np.concatenate(
            [
                engine_indices(interp_weights(np.asarray(p, dtype=float), self.grid)[1], self.grid)
                for p in self.sources[wanted]
            ]
        )
        t0 = time.time()
        state = self.out / "state"
        problem = load_problem(
            self.entry_path,
            seeds,
            walls=None if self.walls is None else WallFit(int(self.walls)),
            walls_file=state / "walls.json",
            outside=self.outside,
        )
        self.problem_record = dict(problem.record)
        self.problem_record.update(
            {
                "updated_nodes": problem.updated,
                "stored_nodes": problem.nodes,
                "bytes_per_source": problem.bytes_per_source(),
                "bytes_shared": problem.bytes_shared(),
                "bytes_a_step": step_bytes(problem, self.boundary)["step"],
                "walls_share_of_bytes": round(
                    float(step_bytes(problem, self.boundary)["walls_share"]), 4
                ),
                "steps": self.steps,
                "prepare_s": round(time.time() - t0, 1),
            }
        )
        return problem

    # ---- the solve as jobs of a queue: a launch a job, on whichever card is free ------------

    def fit_bytes(self) -> float:
        """What the fit's operator holds on a card for the campaign, bytes; an upper figure.

        A matrix a fitted bin, the channels kept by an array's nodes, in
        double precision: 1.8 GB on the grid to 1500 Hz.
        """
        nodes = max((d.count for d in self.designs if d is not None), default=0)
        keep = (int(self.spec["order"]) + 1) ** 2
        bins = 2.0 * self.fmax_hz["low"] * self.durations_s["low"]
        return float(bins * keep * nodes * 8.0)

    def launch_bytes(self, problem: Problem, batch: list[Item]) -> float:
        """What a launch whose records go to the host holds on its card."""
        rows = sum(item.rows for item in batch)
        return float(
            problem.bytes_shared()
            + len(batch) * (problem.bytes_per_source() + 32.0 * self.steps)
            + 4.0 * SPOOL_STEPS * rows
            + self.fit_bytes()
        )

    def launches(self, free_bytes: list[float], host_bytes: float) -> list[dict[str, Any]]:
        """Every launch still to make, each sized for the smallest card: what a queue hands out.

        ``free_bytes`` is what each card has free; none is a machine without
        a card. A launch's records are brought to the host while it runs
        (:func:`reverberate.wave.lowband.solver.solve`), so a card holds the
        grid, its sources' fields and the fit, and a launch that fits the
        smallest card fits every card: whichever card is free takes the
        next, and none waits for a launch of its size. A source costs the
        same card seconds alone or among others, so a smaller launch loses
        nothing but its start. The records of the launches that run at once
        share :data:`HOST_RECORDS_SHARE` of the host's memory.

        Each launch is ``name``, ``items`` (a source position and the cells
        it is read at), ``bytes`` on a card and ``pairs``.
        """
        wanted = self.wanted_sources()
        self.say(
            f"solve: {len(wanted)} source position(s) to solve,"
            f" {self.sources.shape[0] - len(wanted)} wholly cached"
        )
        if not wanted:
            return []
        problem = self.problem_for(wanted)
        self.say(f"solve: grid cut to its sources' reach, {json.dumps(self.problem_record)}")
        state = self.out / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "solve.json").write_text(json.dumps({"wanted": wanted}))
        cards = max(1, len(free_bytes))
        smallest = min(free_bytes) if free_bytes else self.free_bytes(np)
        rows_limit = int(max(1, HOST_RECORDS_SHARE * host_bytes / cards // (4.0 * self.steps)))
        fixed = problem.bytes_shared() + self.fit_bytes() + 4.0 * SPOOL_STEPS * rows_limit
        per_source = problem.bytes_per_source() + 32.0 * self.steps
        at_once = int((MEMORY_SHARE * smallest - fixed) // per_source)
        at_once = max(1, min(LAUNCH_SOURCES, at_once, self.batch or LAUNCH_SOURCES))
        batches: list[list[Item]] = []
        current: list[Item] = []
        rows = 0
        # In the positions' order: the pairs of neighbouring positions come home together,
        # and what waits for a block of them (the levelling, the pack's rows) starts while
        # the cards still solve. The records are on the host: a launch need not be of a size.
        items = sorted(self.items(wanted, rows_limit), key=lambda i: i.source)
        for item in [*items, None]:
            full = item is None or len(current) >= at_once or rows + item.rows > rows_limit
            if current and full:
                batches.append(current)
                current, rows = [], 0
            if item is not None:
                current.append(item)
                rows += item.rows
        # ... and then the longest launches first and the last ones short, so that the cards
        # wait half a source position at the end, and not half a launch of eight.
        found = [self.launch_of(problem, batch) for batch in last_short(batches, cards)]
        self.say(
            f"solve: {sum(len(f['items']) for f in found)} solve(s) in {len(found)} launch(es)"
            f" of {at_once} at most, sized for a card with {smallest / 1e9:.1f} GB free;"
            f" records on the host, {rows_limit} rows a launch"
        )
        return found

    def launch_of(self, problem: Problem, batch: list[Item]) -> dict[str, Any]:
        """A launch as a queue's job: named by what it solves, so a rerun names it again."""
        import hashlib

        items: list[tuple[int, list[int]]] = [
            (int(item.source), [int(c) for c in item.cells]) for item in batch
        ]
        name = hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]
        return {
            "name": name,
            "items": items,
            "bytes": self.launch_bytes(problem, batch),
            "updates": float(problem.updated) * len(batch) * self.steps,
            "pairs": [(source, cell) for source, cells in items for cell in cells],
        }

    def ready(self) -> None:
        """What a worker's launches share, made once: the arrays, the grid's cut, the fit."""
        if getattr(self, "_ready", None) is not None:
            return
        if not hasattr(self, "designs"):
            self.place()
        wanted = json.loads((self.out / "state" / "solve.json").read_text())["wanted"]
        problem = self.problem_for([int(s) for s in wanted])
        encoder = self.encoder_on(self.xp)
        design = next(d for d in self.designs if d is not None)
        encoder.operator_for(design.positions - design.centre)
        encoder.prepare_for(self.steps)
        self.probed(problem, self.xp)
        self._ready = (problem, encoder)

    def run_launch(self, items: list[Any]) -> dict[str, Any]:
        """One launch of a queue, on this process's device; ``parted`` where it does not fit.

        Held against what the device has free now. A launch it does not
        hold comes back as two that ask for less (:func:`halves`), for the
        queue to hand out again; one source read at one cell cannot be
        made smaller and is a ``MemoryError``.
        """
        self.ready()
        problem, encoder = self._ready
        xp = self.xp
        # Without the pairs a first try, or another run, already wrote.
        batch = [
            Item(source, cells, sum(int(self.cell_nodes(c).size) for c in cells))
            for source, cells in (
                (
                    int(source),
                    tuple(int(c) for c in asked if not self.cache.has(self.key_of(int(source), c))),
                )
                for source, asked in items
            )
            if cells
        ]
        if not batch:
            return {
                "batch": 0,
                "pairs": 0,
                "sources": [],
                "cells": [],
                "rows": 0,
                "solve_s": 0.0,
                "encode_s": 0.0,
                "updates_per_s": 0.0,
                "node_updates": 0.0,
            }
        self.release(xp)
        # The fit is on the device already: what is free is free beside it.
        usable = MEMORY_SHARE * (self.free_bytes(xp) + self.fit_bytes())
        if xp is np or self.launch_bytes(problem, batch) <= usable:
            try:
                self.records_on = "host"
                return self.run_batch(problem, batch, encoder, xp)
            except MemoryError:
                self.release(xp)
        parts = halves(batch, lambda cell: self.cell_nodes(cell).size)
        if parts is None:
            raise MemoryError("one source read at one cell is more than this device holds")
        return {"parted": [self.launch_of(problem, part) for part in parts]}

    def gathered(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The launches' records as one a source position, and ``lowband.json``."""
        self.batches = records
        per_source: dict[int, dict[str, Any]] = {}
        for record in records:
            share = 1.0 / max(1, len(record["sources"]))
            for source, cells in zip(record["sources"], record["cells"], strict=True):
                held = per_source.setdefault(
                    source, {"source": source, "cells": 0, "engine_s": 0.0, "encode_s": 0.0}
                )
                held["cells"] += int(cells)
                held["engine_s"] = round(held["engine_s"] + record["solve_s"] * share, 3)
                held["encode_s"] = round(held["encode_s"] + record["encode_s"] * share, 3)
        (self.out / "lowband.json").write_text(
            json.dumps(
                {
                    "grid": getattr(self, "problem_record", {}),
                    "probes": self.probes,
                    "batches": records,
                },
                indent=1,
            )
        )
        return [per_source[s] for s in sorted(per_source)]

    def solve(self) -> list[dict[str, Any]]:
        """Every source position with a pair to make, in batches, on every card."""
        wanted = self.wanted_sources()
        self.say(
            f"solve: {len(wanted)} source position(s) to solve,"
            f" {self.sources.shape[0] - len(wanted)} wholly cached"
        )
        if not wanted:
            return []
        problem = self.problem_for(wanted)
        self.say(f"solve: grid cut to its sources' reach, {json.dumps(self.problem_record)}")
        cards = self.card_indices()
        records: list[dict[str, Any]] = []
        failures: list[BaseException] = []
        queue: Queue[list[Item]] = Queue()
        started = time.time()
        planned = threading.Event()

        def work(card: int | None) -> None:
            try:
                xp = self.xp
                if card is not None:
                    xp.cuda.Device(card).use()
                encoder = self.encoder_on(xp)
                design = next(d for d in self.designs if d is not None)
                operator = encoder.operator_for(design.positions - design.centre)
                with self._status_lock:
                    if not planned.is_set():
                        # Planned on the first card to be ready, with what it has left.
                        free = self.free_bytes(xp) - operator.bytes_on_device
                        per_row = 4.0 * self.steps
                        rows_limit = int(
                            max(
                                1,
                                (
                                    MEMORY_SHARE * free
                                    - problem.bytes_shared()
                                    - problem.bytes_per_source()
                                )
                                // per_row,
                            )
                        )
                        batches = pack_batches(
                            self.items(wanted, rows_limit), problem, self.steps, free
                        )
                        if self.batch:
                            batches = [
                                b[i : i + self.batch]
                                for b in batches
                                for i in range(0, len(b), self.batch)
                            ]
                        for held in batches:
                            queue.put(held)
                        self.say(
                            f"solve: {sum(len(b) for b in batches)} solve(s) in"
                            f" {len(batches)} batch(es) on {max(1, len(cards))} card(s),"
                            f" up to {max(len(b) for b in batches)} at once"
                        )
                        self.total_batches = len(batches)
                        planned.set()
                # The resampler's weights stay for the campaign: made now, on a pool that holds
                # nothing, they are a block of their own and not a part of a launch's.
                encoder.prepare_for(self.steps)
                self.probed(problem, xp, card)
                while not failures:
                    try:
                        batch = queue.get_nowait()
                    except Empty:
                        return
                    self.release(xp)
                    # The plan was made on one card, once. Each launch is held against what
                    # its own card has free now, and parted where that is less.
                    parts = None if self.fits(problem, batch, xp) else self.parted(batch)
                    if parts is None:
                        try:
                            record = self.run_batch(problem, batch, encoder, xp)
                        except MemoryError as error:
                            self.release(xp)
                            parts = self.parted(batch)
                            if parts is None:
                                raise
                            self.say(
                                f"a launch of {len(batch)} source(s) and"
                                f" {sum(len(i.cells) for i in batch)} cell(s) was refused its"
                                f" memory ({str(error)[:120]}); parted and tried again"
                            )
                    elif parts:
                        self.say(
                            f"a launch of {len(batch)} source(s) and"
                            f" {sum(len(i.cells) for i in batch)} cell(s) is more than card"
                            f" {card} has free; parted"
                        )
                    if parts is not None:
                        with self._status_lock:
                            self.total_batches += len(parts) - 1
                            self.parted_launches = getattr(self, "parted_launches", 0) + 1
                        for part in parts:
                            queue.put(part)
                        continue
                    record["card"] = card
                    with self._status_lock:
                        records.append(record)
                        elapsed = time.time() - started
                        left = self.total_batches - len(records)
                        self.set_status(
                            job=f"{len(records)}/{self.total_batches}",
                            pairs=sum(int(r["pairs"]) for r in records),
                            remaining_s=round(elapsed / len(records) * left, 1),
                        )
                    self.say(
                        f"batch of {record['batch']}: {record['pairs']} pair(s), solve"
                        f" {record['solve_s']} s at {record['updates_per_s']:.3g} updates/s,"
                        f" encode {record['encode_s']} s"
                    )
                with self._status_lock:
                    self.prepare_s = getattr(self, "prepare_s", 0.0) + encoder.prepare_s
            except BaseException as error:  # noqa: BLE001 - reported by the campaign's thread
                failures.append(error)

        if len(cards) > 1:
            threads = [threading.Thread(target=work, args=(card,)) for card in cards]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            work(cards[0] if cards else None)
        if self.xp is not np:
            # What the pool still holds is given back: another engine may follow on this card.
            self.xp.get_default_memory_pool().free_all_blocks()
        if failures:
            raise failures[0]
        # One record a source position, as the campaign of the present engine gives them.
        return self.gathered(records)

    def card_indices(self) -> list[int]:
        """The cards a worker is started on; none without one."""
        if self.xp is np:
            return []
        if self.devices:
            return [int(i) for i in self.devices.split(",")]
        return list(range(int(self.status["device"].get("devices") or 1)))

    def ledger(self, rate_usd_per_hour: float) -> dict[str, Any]:
        """What the solve cost: per source position and per pair, at an hourly rate."""
        solves = sum(int(b["batch"]) for b in self.batches)
        pairs = sum(int(b["pairs"]) for b in self.batches)
        solve_s = sum(float(b["solve_s"]) for b in self.batches)
        encode_s = sum(float(b["encode_s"]) for b in self.batches)
        cards = max(1, len(self.card_indices()))
        updates = sum(float(b["node_updates"]) for b in self.batches)
        per_hour = rate_usd_per_hour / 3600.0 / cards
        return {
            "solver": str(self.spec["solver"]),
            "grid": getattr(self, "problem_record", {}),
            "cards": cards,
            "billed_rate_usd_per_hour": rate_usd_per_hour,
            "solves": solves,
            "pairs": pairs,
            "node_updates_per_s_a_card": updates / max(solve_s, 1e-9),
            "solve_s_per_source": solve_s / max(1, solves) / cards,
            "encode_s_per_pair": encode_s / max(1, pairs) / cards,
            "usd_per_source": solve_s / max(1, solves) * per_hour,
            "usd_per_pair": encode_s / max(1, pairs) * per_hour,
            "prepare_s": round(getattr(self, "prepare_s", 0.0), 1),
        }
