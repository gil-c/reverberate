"""What a machine gives a trace: its cards as they are, its cores, its memory. Read, not named.

A rented machine is an offer's words until it is asked: a card's label does
not always match its memory, a container counts its host's cores and is
lent a fifth of them, and two cards of one name do not step a grid at one
rate. :class:`Machine` is what the machine itself answers:

- its cards, by ``nvidia-smi`` (no context is opened to count them), each
  with what it really has free;
- the cores it may really use (:func:`reverberate.compute.usable_cores`)
  and the memory its container is given;
- once a worker holds a card, two seconds of the work itself
  (:func:`measure`): the wave solver's own step on a box of free air, in
  node updates a second, and a transform of the size the fit makes. No
  table of card names is read: a card the project never saw is measured as
  any other.

:meth:`Machine.workers` is the trace's pool on that machine: a process a
card, and a process a core that is left. :func:`predict` is the run's wall
time and price on it, said before the long work starts, from the launches
really planned and the rates really measured.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from reverberate.compute import (
    card_free_bytes,
    card_limit_bytes,
    cuda_available,
    usable_cores,
)
from reverberate.trace.pool import WorkerSpec

__all__ = [
    "REFERENCE",
    "Card",
    "Machine",
    "measure",
    "predict",
    "stencil_rate",
    "transform_seconds",
]

#: The box of free air a card steps to be measured, nodes a side; and a host's.
CARD_BOX = (258, 254, 250)
HOST_BOX = (66, 62, 58)
#: The transform measured: records of this many samples, this many at once.
TRANSFORM = (64, 32768)

#: The host the stages a host does were measured on, with what :func:`measure` says of it.
#: A stage's seconds on another host go as its figures against these.
REFERENCE: dict[str, Any] = {
    "host": "the laptop, one performance core at 3.6 GHz, alone (2026-10-05)",
    "host_updates_per_s": 1.14e8,
    "host_transform_s": 0.0052,
    #: Host seconds a unit, one process, ``numpy``, on a 20 s window of the realistic
    #: scene (hssd_0076, 630 positions, 284 pairs): a (source, head) position of the early
    #: trace, a pair levelled, a pair's 64 channels through the air and the masks.
    "paths_job_s": 0.026,
    "level_pair_s": 0.076,
    "row_pair_s": 0.014,
    #: The solver on the grid to 1500 Hz against the same card's box of free air: the
    #: boundary and its branches. Measured on 4 x RTX 3090 (instance 54299322, 2026-10-05):
    #: 1.72e10 node updates a second a card in launches of 8 against 3.74e10 on the box.
    "solve_over_box": 0.46,
    #: Card seconds a launch besides its steps (the stepper and its drive made, the fit of
    #: its cells), and seconds before a card's first launch steps (the launches planned,
    #: the worker's own grid and fit): 8.6 and 55 on that machine.
    "launch_s": 8.6,
    "start_s": 55.0,
}


@dataclass(frozen=True)
class Card:
    """One card as the machine says it: its place, its name, its memory in bytes."""

    index: int
    name: str = ""
    total_bytes: float = 0.0
    free_bytes: float = 0.0
    #: The solver's step on a box of free air, node updates a second; ``None`` until measured.
    updates_per_s: float | None = None
    transform_s: float | None = None

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_gb": round(self.total_bytes / 1e9, 2),
            "free_gb": round(self.free_bytes / 1e9, 2),
            "updates_per_s": self.updates_per_s,
            "transform_s": self.transform_s,
        }


def _smi() -> list[Card]:
    """The cards ``nvidia-smi`` lists, without opening a context on any; none when it is absent."""
    try:
        said = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in said.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            cards.append(
                Card(int(parts[0]), parts[1], float(parts[2]) * 2**20, float(parts[3]) * 2**20)
            )
    return cards


def _threads_a_core() -> int:
    """The threads of one core as ``/proc/cpuinfo`` says them (siblings over cores); 1 unknown."""
    try:
        with open("/proc/cpuinfo") as handle:
            said = handle.read()
    except OSError:
        return 1
    found: dict[str, int] = {}
    for line in said.splitlines():
        name, _, value = line.partition(":")
        if name.strip() in ("siblings", "cpu cores") and value.strip().isdigit():
            found.setdefault(name.strip(), int(value))
    if found.get("cpu cores") and found.get("siblings"):
        return max(1, found["siblings"] // found["cpu cores"])
    return 1


@dataclass(frozen=True)
class Machine:
    """The cards, the cores and the memory a trace may use."""

    cards: tuple[Card, ...] = ()
    cores: int = 1
    ram_bytes: float = 0.0
    host_updates_per_s: float | None = None
    host_transform_s: float | None = None
    #: Threads the host shows for each of its cores; 1 where it does not say.
    threads_a_core: int = 1

    @classmethod
    def detect(cls, *, gpu: bool | None = None, devices: str | None = None) -> Machine:
        """This machine. ``gpu`` false is its host alone; ``devices`` keeps those cards."""
        from reverberate.accel.solve import host_memory_gb

        cards: list[Card] = []
        if gpu is not False and cuda_available():
            cards = _smi()
            if not cards:
                import cupy

                cards = [Card(i) for i in range(int(cupy.cuda.runtime.getDeviceCount()))]
            # The cards this process was shown, where it was shown some and not all.
            shown = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            for told in (shown, devices or ""):
                kept = {int(v) for v in told.split(",") if v.strip().isdigit()}
                if kept:
                    cards = [card for card in cards if card.index in kept]
        limit = card_limit_bytes()
        if limit is not None:
            # A smaller card stood in for: every card is counted as holding that much.
            cards = [
                replace(card, free_bytes=min(card.free_bytes or limit, limit)) for card in cards
            ]
        if gpu and not cards:
            raise RuntimeError("a card was asked for and this machine shows none")
        return cls(
            tuple(cards), usable_cores(), host_memory_gb() * 1e9, threads_a_core=_threads_a_core()
        )

    @classmethod
    def fake(
        cls, free_gb: tuple[float, ...] = (), *, cores: int = 4, ram_gb: float = 16.0
    ) -> Machine:
        """A machine that is not there, for a plan and for the tests: cards by their free memory."""
        cards = tuple(
            Card(i, f"fake {gb:g} GB", gb * 1e9, gb * 1e9, 1.4e10, 0.01)
            for i, gb in enumerate(free_gb)
        )
        return cls(cards, cores, ram_gb * 1e9, 1.0e8, 0.05)

    def workers(self, host: int | None = None) -> list[WorkerSpec]:
        """The pool: a process a card, and ``host`` more (the cores the cards' leave)."""
        if host is None:
            # A process a core, not a process a thread of a core: on 16 cores of two threads
            # the host's stages took 101 s with 16 workers and 117 s with 28 (2026-10-05).
            whole = max(1, int(self.cores / max(1, self.threads_a_core)))
            # A card's process takes most of a thread while its card works: a core's second
            # thread where there is one, a core of its own where there is not.
            taken = len(self.cards) if self.threads_a_core < 2 else 0
            host = max(1, whole - taken) if self.cards else max(1, whole - 1)
        specs = [
            WorkerSpec(index=k, card=card.index, free_bytes=card.free_bytes)
            for k, card in enumerate(self.cards)
        ]
        specs += [WorkerSpec(index=len(specs) + k) for k in range(max(0, int(host)))]
        return specs or [WorkerSpec(index=0)]

    def measured(self, said: list[dict[str, Any]]) -> Machine:
        """This machine with what its workers measured (:func:`measure`), a record a worker."""
        cards = list(self.cards)
        host: list[dict[str, Any]] = []
        for record in said:
            if record.get("card") is None:
                host.append(record)
                continue
            for k, card in enumerate(cards):
                if card.index == int(record["card"]):
                    cards[k] = replace(
                        card,
                        name=card.name or str(record.get("name", "")),
                        free_bytes=float(record.get("free_bytes") or card.free_bytes),
                        total_bytes=float(record.get("total_bytes") or card.total_bytes),
                        updates_per_s=record.get("updates_per_s"),
                        transform_s=record.get("transform_s"),
                    )
        updates = [float(r["updates_per_s"]) for r in host if r.get("updates_per_s")]
        transform = [float(r["transform_s"]) for r in host if r.get("transform_s")]
        return replace(
            self,
            cards=tuple(cards),
            host_updates_per_s=float(np.median(updates)) if updates else self.host_updates_per_s,
            host_transform_s=float(np.median(transform)) if transform else self.host_transform_s,
        )

    def record(self) -> dict[str, Any]:
        return {
            "cards": [card.record() for card in self.cards],
            "cores": self.cores,
            "threads_a_core": self.threads_a_core,
            "ram_gb": round(self.ram_bytes / 1e9, 1),
            "host_updates_per_s": self.host_updates_per_s,
            "host_transform_s": self.host_transform_s,
        }


def stencil_rate(xp: Any, seconds: float = 1.0, shape: tuple[int, int, int] | None = None) -> float:
    """Node updates a second of the wave solver's own step on a box of free air, one source."""
    from reverberate.wave.lowband.box import box_arrays
    from reverberate.wave.lowband.problem import build_problem
    from reverberate.wave.lowband.scheme import CARTESIAN
    from reverberate.wave.lowband.solver import CardStepper, NumpyStepper, drive_for

    shape = shape or (HOST_BOX if xp is np else CARD_BOX)
    arrays, grid = box_arrays(CARTESIAN, shape, room=None)
    problem = build_problem(arrays)
    centre = (np.asarray(shape, dtype=float) // 2 + 0.3) * grid.h
    drive = drive_for(problem, grid, centre[None, :], [np.zeros(0, np.int64)], 64 * grid.Ts)
    stepper: Any = NumpyStepper(problem, drive) if xp is np else CardStepper(problem, drive, xp)
    state = stepper.state()
    out = stepper.records(1)
    for n in range(2):  # the kernels compiled and the pages touched
        stepper.step(state, n, out, 0)
    stepper.finish()
    started = time.time()
    steps = 0
    while time.time() - started < seconds:
        for _ in range(1 if xp is np else 16):
            stepper.step(state, steps % drive.steps, out, 0)
            steps += 1
        stepper.finish()
    return float(problem.updated) * steps / max(time.time() - started, 1e-9)


def transform_seconds(xp: Any, repeats: int = 3) -> float:
    """Seconds of one real transform of :data:`TRANSFORM` in double precision, the best of a few."""
    block = xp.asarray(np.random.default_rng(0).standard_normal(TRANSFORM))
    best = float("inf")
    for _ in range(repeats + 1):
        started = time.time()
        spectrum = xp.fft.rfft(block, axis=-1)
        float(abs(spectrum[0, 1]))  # the device has answered
        best = min(best, time.time() - started)
    return best


def measure(card: int | None, xp: Any, seconds: float = 1.0) -> dict[str, Any]:
    """What a worker says of the device it holds: its memory and :func:`stencil_rate` of it."""
    record: dict[str, Any] = {"card": card}
    if xp is not np:
        device = xp.cuda.Device()
        free, total = device.mem_info
        properties = xp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        record.update(
            {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "free_bytes": card_free_bytes(xp),
                "total_bytes": float(total),
            }
        )
    if seconds > 0.0:
        record["updates_per_s"] = stencil_rate(xp, seconds)
        record["transform_s"] = transform_seconds(xp)
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
            record["free_bytes"] = card_free_bytes(xp)
    return record


def predict(
    counts: dict[str, Any],
    machine: Machine,
    *,
    host_workers: int,
    rate_usd_per_hour: float | None = None,
    solve_updates_per_s: list[float] | None = None,
) -> dict[str, Any]:
    """Wall seconds, and USD at a rate, of what is left of a trace on this machine.

    ``counts`` is the work as planned on the machine: ``node_updates`` of
    the launches to make, ``pairs`` to fit, ``sites`` of rays and the
    seconds of one on one card, and the host's units (``paths_jobs``,
    ``level_pairs``, ``row_pairs``). The cards' work and the host's overlap:
    the wall is the longer of the two, and what one process does after
    them. ``solve_updates_per_s`` is each card's rate on the run's own grid
    where it was measured; without it the box's rate stands for it, times
    :data:`REFERENCE` ``solve_over_box`` where that is known, and the
    prediction says it is not calibrated.
    """
    cards = [card for card in machine.cards if card.updates_per_s]
    calibrated = bool(solve_updates_per_s)
    if solve_updates_per_s:
        rates = [float(v) for v in solve_updates_per_s]
    else:
        over = REFERENCE["solve_over_box"]
        calibrated = over is not None
        rates = [float(card.updates_per_s or 0.0) * float(over or 1.0) for card in cards]
    if not rates:
        rates = [float(machine.host_updates_per_s or REFERENCE["host_updates_per_s"])]
    held = max(1, len(machine.cards))
    solve_s = float(counts.get("node_updates", 0.0)) / max(sum(rates), 1e-9)
    if solve_s > 0.0:
        solve_s += float(counts.get("launches", 0)) * float(REFERENCE["launch_s"]) / held
        solve_s += float(REFERENCE["start_s"])
    fit_s = float(counts.get("pairs", 0)) * float(counts.get("pair_s", 0.0)) / held
    rays_s = float(counts.get("sites", 0)) * float(counts.get("site_s", 0.0)) / held
    slow = 1.0
    if machine.host_updates_per_s and machine.host_transform_s:
        slow = 0.5 * (
            REFERENCE["host_updates_per_s"] / machine.host_updates_per_s
            + machine.host_transform_s / REFERENCE["host_transform_s"]
        )
    host_s = slow * sum(
        float(counts.get(name, 0)) * float(REFERENCE[unit])
        for name, unit in (
            ("paths_jobs", "paths_job_s"),
            ("level_pairs", "level_pair_s"),
            ("row_pairs", "row_pair_s"),
        )
    )
    seconds = {
        "solve": round(solve_s, 1),
        "fit": round(fit_s, 1),
        "rays": round(rays_s, 1),
        "host": round(host_s / max(1, host_workers), 1),
        "write": round(float(counts.get("write_s", 0.0)), 1),
    }
    wall = max(solve_s + fit_s + rays_s, host_s / max(1, host_workers)) + seconds["write"]
    record: dict[str, Any] = {
        "seconds": seconds,
        "wall_s": round(wall, 1),
        "hours": round(wall / 3600.0, 3),
        "calibrated": calibrated,
        "host_slowness": round(slow, 2),
        "cards": len(machine.cards),
        "host_workers": host_workers,
    }
    if rate_usd_per_hour is not None:
        record["usd"] = round(wall / 3600.0 * float(rate_usd_per_hour), 2)
        record["rate_usd_per_hour"] = float(rate_usd_per_hour)
    return record
