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
from reverberate.wave.lowband.fit import CellEncoder, level_scale
from reverberate.wave.lowband.problem import Problem, load_problem
from reverberate.wave.lowband.scheme import CARTESIAN, SCHEMES, Scheme
from reverberate.wave.lowband.solver import drive_for, solve, steps_for

__all__ = [
    "SOLVER",
    "Item",
    "LowbandPairs",
    "batch_capacity",
    "node_indices",
    "pack_batches",
    "solver_name",
]

#: The engine's name in a pair's key; a change between the grid and the cache form changes it.
SOLVER = "reverberate.wave.lowband/1"

#: Of a card's free memory, what a batch may take; the rest is the fit's working arrays.
MEMORY_SHARE = 0.8
#: The most sources of one launch: the boundary kernel's second grid axis.
BATCH_LIMIT = 4096


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


def solver_name(scheme: Scheme, ppw: float) -> str:
    """What a pair's key and a pack's provenance say solved it."""
    return f"{SOLVER} {scheme.name} at {ppw:g} points per wavelength"


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


@dataclass
class LowbandPairs(PairsCampaign):
    """The pairs of a bundle on the batched solver, one worker a card."""

    scheme: str = CARTESIAN.name
    ppw: float | None = None
    #: Sources at once, at most; ``None`` is what the card's memory holds.
    batch: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}, expected one of {sorted(SCHEMES)}")
        self.grid_scheme = SCHEMES[self.scheme]
        self.grid_ppw = float(self.ppw) if self.ppw is not None else self.grid_scheme.ppw
        self.bundle_grid = not self.grid_scheme.fcc and self.grid_ppw == float(self.spec["ppw"])
        self.spec["solver"] = solver_name(self.grid_scheme, self.grid_ppw)
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
        from reverberate.experiments.run import scene_spec
        from reverberate.wave.remote_voxelise import grid_shape_of
        from reverberate.wave.voxelise import nh_for

        fmax = float(self.spec["bands"]["low"]["fmax_hz"])
        scene, _, _ = scene_spec(self.models, self.spec["storey_scene"], fmax)
        if self.bundle_grid:
            return scene
        shape = grid_shape_of(scene.model_json, fmax, self.grid_ppw)
        return replace(scene, ppw=self.grid_ppw, fcc=self.grid_scheme.fcc, nh=nh_for(shape))

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
        free, _ = xp.cuda.Device().mem_info
        return float(free)

    def run_batch(
        self, problem: Problem, batch: list[Item], encoder: CellEncoder, xp: Any
    ) -> dict[str, Any]:
        """One launch: the batch solved, every cell of it encoded and cached."""
        receivers = [np.concatenate([self.cell_nodes(c) for c in item.cells]) for item in batch]
        sources = np.asarray([self.sources[item.source] for item in batch], dtype=float)
        drive = drive_for(problem, self.grid, sources, receivers, self.durations_s["low"])
        timing: dict[str, Any] = {}
        t0 = time.time()
        records = solve(problem, drive, xp, say=lambda m: self.say(f"    {m}"), timing=timing)
        solve_s = time.time() - t0
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

    def solve(self) -> list[dict[str, Any]]:
        """Every source position with a pair to make, in batches, on every card."""
        wanted = [s for s in range(self.sources.shape[0]) if self.todo(s)]
        self.say(
            f"solve: {len(wanted)} source position(s) to solve,"
            f" {self.sources.shape[0] - len(wanted)} wholly cached"
        )
        if not wanted:
            return []
        seeds = np.concatenate(
            [
                engine_indices(interp_weights(np.asarray(p, dtype=float), self.grid)[1], self.grid)
                for p in self.sources[wanted]
            ]
        )
        t0 = time.time()
        problem = load_problem(self.entry_path, seeds)
        self.problem_record = dict(problem.record)
        self.problem_record.update(
            {
                "updated_nodes": problem.updated,
                "stored_nodes": problem.nodes,
                "bytes_per_source": problem.bytes_per_source(),
                "steps": self.steps,
                "prepare_s": round(time.time() - t0, 1),
            }
        )
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
                while not failures:
                    try:
                        batch = queue.get_nowait()
                    except Empty:
                        return
                    record = self.run_batch(problem, batch, encoder, xp)
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
        self.batches = records
        # One record a source position, as the campaign of the present engine gives them.
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
            json.dumps({"grid": self.problem_record, "batches": records}, indent=1)
        )
        return [per_source[s] for s in sorted(per_source)]

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
