"""What the signal engine costs: a pack as dense as a real one, rendered and timed part by part.

No traced pack exists yet, so the benchmark builds one of the size
``scene-pack.md`` estimates (16 arrivals a step, a solved source position
every 8 cm of rail, a histogram every 0.80 m) with random tables in it: the
``synthetic-density`` profile. Its render means nothing; its cost is the
cost of a real pack of that size, because the engine's work depends on how
many rows, pairs and histograms a step holds and not on what they hold.

Two scenes bound a real one. **At rest**: the source at a station, the
listener on a seat, so every step reads the same pair and the same
histogram. **Moving**: the source on a rail at 1.5 m/s, the listener
walking at 1 m/s between cells, so every step reads new pairs, a new
operator and a histogram with new weights.

``python -m reverberate.render benchmark`` prints seconds of compute per
second of scene, for one source, per part. With ``--processes 1,4,8`` it
also renders stems as the audit does, a run at a time in that many
processes of one thread each, and prints what they render together, what a
process holds, and how long a scene of twenty minutes and fourteen sources
that each sound half the time would take. ``--save`` keeps the report and
``--against`` prints it beside an earlier one. Nothing here asserts.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import resource
import sys
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from reverberate.audio import Atmosphere
from reverberate.metrics import band_centres
from reverberate.mirror.directivity import omni, voice_v1
from reverberate.mirror.hybrid import Crossover
from reverberate.render.engine import PARTS, Engine, RenderSettings
from reverberate.render.pack import (
    STEP_S,
    Air,
    Cells,
    Early,
    Header,
    Level,
    Listener,
    Low,
    Mirror,
    ScenePack,
    Source,
    Tail,
    synthetic_recipe,
    tail_seed,
    validate,
)

__all__ = ["SCENE_SOURCE_S", "Repeated", "density_pack", "measure", "scaling", "table"]

#: Seconds of stem in the scene the format sizes: twenty minutes, fourteen sources,
#: each sounding half the time.
SCENE_SOURCE_S = 20 * 60 * 14 / 2


class Repeated:
    """A table whose rows repeat a few: the size of a real one without its memory."""

    def __init__(self, base: np.ndarray, rows: int) -> None:
        self.base = base
        self.shape = (rows, *base.shape[1:])
        self.dtype = base.dtype

    def __getitem__(self, row: int) -> np.ndarray:
        return np.asarray(self.base[int(row) % self.base.shape[0]])


def density_pack(
    *,
    duration_s: float = 5.0,
    moving: bool = True,
    arrivals: int = 16,
    sources: int = 1,
    bins: int = 600,
    seed: int = 0,
) -> ScenePack:
    """A pack of the format's estimated density, random inside; see the module."""
    rng = np.random.default_rng(seed)
    steps = int(round(duration_s / STEP_S)) + 1
    names = [f"s{i + 1}" for i in range(sources)]
    recipe = synthetic_recipe(duration_s, names)
    header = Header(
        profile="synthetic-density",
        recipe_sha256=hashlib.sha256(recipe).hexdigest(),
        dwelling="none",
        scene_id="density",
        duration_s=float(duration_s),
        steps=steps,
        bank_bands_hz=band_centres(48000),
        has_low=True,
        has_tail=True,
        provenance={
            "recipe_sha256": hashlib.sha256(recipe).hexdigest(),
            "code_version": "benchmark",
            "cost": [],
        },
    )
    c = header.sound_speed_m_s
    times = np.arange(steps) * STEP_S
    pitch = 0.40
    travelled = (1.0 if moving else 0.0) * times
    listener = np.stack([travelled, np.full(steps, 1.7), np.zeros(steps)], axis=1)
    cell_count = int(np.ceil(travelled[-1] / pitch)) + 2
    centres = np.stack(
        [pitch * np.arange(cell_count), np.full(cell_count, 1.7), np.zeros(cell_count)], axis=1
    )
    cells = Cells(
        position=centres,
        kind=np.zeros(cell_count, dtype=np.uint8),
        lattice_index=np.stack(
            [np.arange(cell_count), np.zeros(cell_count), np.zeros(cell_count)], axis=1
        ).astype(np.int32),
        clearance_m=np.full(cell_count, 0.65, dtype=np.float32),
        room=("living",) * cell_count,
        layers_y_m=(1.7,),
    )
    crossover = Crossover()
    bands = len(header.bands_hz)
    # A few distinct responses, band limited as the format says, repeated over the pairs.
    freqs = np.fft.rfftfreq(header.low_samples, 1.0 / header.low_sample_rate_hz)
    mask, _ = crossover.masks(header.low_samples, header.low_sample_rate_hz, power=True)
    decay = np.exp(-np.arange(header.low_samples) / (0.08 * header.low_sample_rate_hz))
    shaped = np.fft.rfft(rng.standard_normal((8, header.channels, header.low_samples)) * decay)
    base_ir = np.fft.irfft(shaped * mask * (freqs > 40.0), header.low_samples).astype(np.float32)
    # Histograms: an exponential decay from 10 ms, slightly anisotropic.
    bin_times = (np.arange(bins) + 0.5) * 0.002
    made: dict[str, Source] = {}
    for index, name in enumerate(names):
        speed = 1.5 if moving else 0.0
        position = np.stack(
            [3.0 + index + 0.0 * times, np.full(steps, 1.6), 2.0 + speed * times], axis=1
        )
        rail = 0.08
        along = speed * times / rail
        slot = np.floor(along).astype(int)
        weight = (along - slot).astype(np.float32)
        # The arrivals: images that travel with the source, some born and dying.
        offsets_image = rng.uniform(-6.0, 6.0, (arrivals, 3)) * np.array([1.0, 0.3, 1.0])
        offsets_image[0] = 0.0
        ids = np.sort(rng.integers(1, 2**62, arrivals).astype(np.uint64))
        life = rng.uniform(0.0, duration_s, (arrivals, 2))
        alive = np.ones((steps, arrivals), dtype=bool)
        flicker = np.arange(arrivals) >= (3 * arrivals) // 4
        if moving:
            alive[:, flicker] = (times[:, None] < life[None, flicker, 0]) | (
                times[:, None] > life[None, flicker, 1]
            )
        image = position[:, None, :] + offsets_image[None, :, :]
        seen = image - listener[:, None, :]
        distance = np.linalg.norm(seen, axis=2)
        arrival = seen / distance[:, :, None]
        order = np.minimum(rng.integers(0, 4, arrivals), 3)
        order[0] = 0
        reflect = rng.uniform(0.5, 0.9, (arrivals, bands)) ** order[:, None]
        gain = reflect[None, :, :] / distance[:, :, None]
        departure = -arrival
        keep = alive.ravel()
        early = Early(
            offsets=np.concatenate([[0], np.cumsum(alive.sum(axis=1))]).astype(np.int64),
            path_id=np.tile(ids, steps)[keep],
            delay_s=(distance / c).ravel()[keep],
            arrival=arrival.reshape(-1, 3)[keep].astype(np.float32),
            departure=departure.reshape(-1, 3)[keep].astype(np.float32),
            gain=gain.reshape(-1, bands)[keep].astype(np.float32),
            order=np.tile(order, steps)[keep].astype(np.uint8),
            kind=np.tile(np.minimum(order, 1), steps)[keep].astype(np.uint8),
        )
        # The low band: two solved positions either side of the source, by one or two cells.
        nearest = np.clip(np.round(travelled / pitch).astype(int), 0, cell_count - 1)
        other = np.clip(np.where(travelled >= nearest * pitch, nearest + 1, nearest - 1), 0, None)
        other = np.where(other == nearest, nearest + 1, other)
        exact = np.abs(travelled - nearest * pitch) <= 0.001
        mode = np.where(exact, 1, 3).astype(np.uint8)
        cell = np.stack([nearest, np.where(mode == 3, other, -1)], axis=1).astype(np.int32)
        rows: dict[tuple[int, int], int] = {}
        pair = np.full((steps, 2, 2), -1, dtype=np.int32)
        for k in range(steps):
            for a in range(2 if weight[k] > 0.0 else 1):
                for b in range(2 if mode[k] == 3 else 1):
                    key = (int(slot[k]) + a, int(cell[k, b]))
                    pair[k, a, b] = rows.setdefault(key, len(rows))
        keys = sorted(rows, key=rows.__getitem__)
        pair_position = np.stack([position[0] + np.array([0.0, 0.0, rail * p]) for p, _ in keys])
        low = Low(
            ir=Repeated(base_ir, len(keys)),
            pair_position=pair_position,
            pair_cell=np.array([cell_row for _, cell_row in keys], dtype=np.int32),
            pair_key=np.array(
                [hashlib.sha256(f"{name}:{p}:{q}".encode()).hexdigest() for p, q in keys],
                dtype="S64",
            ),
            seam_db=np.zeros(len(keys), dtype=np.float32),
            onset_s=np.full(len(keys), 0.01),
            pair=pair,
            position_weight=weight,
            cell=cell,
            mode=mode,
        )
        # The tail: a histogram every 0.80 m of rail and on every second cell.
        coarse = 0.80
        s_along = speed * times / coarse
        s_slot = np.floor(s_along).astype(int)
        s_weight = (s_along - s_slot).astype(np.float32)
        c_along = travelled / coarse
        c_slot = np.floor(c_along).astype(int)
        c_weight = (c_along - c_slot).astype(np.float32)
        hist_rows: dict[tuple[int, int], int] = {}
        hist = np.full((steps, 2, 2), -1, dtype=np.int32)
        for k in range(steps):
            for a in range(2 if s_weight[k] > 0.0 else 1):
                for b in range(2 if c_weight[k] > 0.0 else 1):
                    key = (int(s_slot[k]) + a, int(c_slot[k]) + b)
                    hist[k, a, b] = hist_rows.setdefault(key, len(hist_rows))
        count = len(hist_rows)
        t60 = rng.uniform(0.3, 0.6, (count, 1, bands))
        energy = (
            1e-3
            * np.exp(-13.8 * bin_times[None, :, None] / t60)
            * (bin_times[None, :, None] > 0.01)
        ).astype(np.float32)
        moments = np.zeros((count, bins, bands, 16), dtype=np.float32)
        moments[..., 0] = energy
        moments[..., 1:4] = 0.2 * energy[..., None] * rng.standard_normal((count, 1, 1, 3))
        tail = Tail(
            energy=energy,
            moments=moments,
            scale=np.ones((count, len(header.bank))),
            hist_position=np.zeros((count, 3)),
            hist_cell=np.zeros(count, dtype=np.int32),
            hist=hist,
            position_weight=s_weight,
            cell_weight=c_weight,
        )
        made[name] = Source(
            id=name,
            kind="near_voice",
            position=position,
            yaw_deg=np.zeros(steps, dtype=np.float32),
            audible=np.ones(steps, dtype=bool),
            early=early,
            level=Level(np.zeros(steps, dtype=np.float32), distance[:, 0] / c),
            low=low,
            tail=tail,
            directivity_model="voice",
            directivity_enabled=True,
            tail_seed=tail_seed(seed, name),
        )
    signature = np.exp(-np.arange(128) / 6.0) * rng.standard_normal(128)
    signature[0] = 1.0
    pack = ScenePack(
        header=header,
        recipe=recipe,
        listener=Listener(listener, np.zeros((steps, 3), dtype=np.float32)),
        cells=cells,
        sources=made,
        mirror=Mirror(
            signature=signature / np.abs(np.fft.rfft(signature, 1024)).max(),
            lead_s=0.0021,
            lowcut_hz=40.0,
            receiver_radius_m=0.25,
        ),
        crossover=crossover,
        air=Air(Atmosphere(), enabled=True),
        directivity={"voice": voice_v1(), "omni": omni()},
    )
    validate(pack)
    return pack


def measure(
    *, duration_s: float = 5.0, workers: int = -1, gpu: bool = False, seed: int = 0
) -> dict[str, Any]:
    """Seconds of compute per second of scene, one source, per part, at rest and moving.

    Each part is rendered alone over the whole scene by an engine of its
    own, after a first run of steps that pays what is paid once (the
    carrier, the filters, the first operator) and is reported apart.
    """
    report: dict[str, Any] = {
        "duration_s": duration_s,
        "cores": os.cpu_count(),
        "workers": workers,
        "gpu": gpu,
        "sources": 1,
        "unit": "seconds of compute per second of scene, per source",
    }
    dry = np.random.default_rng(seed).standard_normal(int(round(duration_s * 48000)))
    for label, moving in (("rest", False), ("moving", True)):
        pack = density_pack(duration_s=duration_s, moving=moving, seed=seed)
        source = pack.sources["s1"]
        assert source.low is not None and source.tail is not None
        scene: dict[str, Any] = {
            "arrivals_per_step": round(float(np.diff(source.early.offsets).mean()), 2),
            "pairs": int(source.low.pair_cell.shape[0]),
            "histograms": int(source.tail.hist_cell.shape[0]),
        }
        settings = RenderSettings(workers=workers)
        first = settings.chunk_steps * pack.header.step_samples
        total = 0.0
        for part in PARTS:
            engine = Engine(pack, {"s1": dry}, settings=settings, gpu=gpu)
            started = time.perf_counter()
            engine.stem("s1", 0, first, parts=(part,))
            once = time.perf_counter() - started
            started = time.perf_counter()
            engine.stem("s1", first, pack.header.samples, parts=(part,))
            spent = time.perf_counter() - started
            rest_s = (pack.header.samples - first) / pack.header.sample_rate_hz
            scene[part] = round(spent / rest_s, 4)
            scene[f"{part}_first_run_s"] = round(once, 3)
            total += spent / rest_s
        scene["total"] = round(total, 4)
        report[label] = scene
    return report


def _stems(moving: bool, duration_s: float, seed: int, ready: Any, go: Any, done: Any) -> None:
    """One process of :func:`scaling`: a stem a run at a time, as the audit's worker renders."""
    pack = density_pack(duration_s=duration_s, moving=moving, seed=seed)
    h = pack.header
    dry = np.random.default_rng(seed).standard_normal(int(round(duration_s * h.sample_rate_hz)))
    settings = RenderSettings(workers=1, chunks_held=1)
    engine = Engine(pack, {"s1": dry}, settings=settings)
    run = settings.chunk_steps * h.step_samples
    engine.stem("s1", 0, run)
    ready.put(None)
    go.wait()
    started = time.perf_counter()
    for first in range(run, h.samples, run):
        engine.stem("s1", first, min(first + run, h.samples)).astype(np.float32)
    spent = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resident = peak if sys.platform == "darwin" else peak * 1024
    done.put((spent / ((h.samples - run) / h.sample_rate_hz), resident))


def scaling(
    *,
    duration_s: float = 5.0,
    processes: Iterable[int] = (1, 4, 8),
    scenes: Iterable[str] = ("rest", "moving"),
    seed: int = 0,
) -> dict[str, Any]:
    """Whole stems rendered by several processes at once, one thread each.

    Every process builds the pack and renders its one source, the first run
    apart, then all start together. Per scene and per number of processes:
    the seconds of compute a second of scene costs one process, the seconds
    of stem all render in a second of wall clock, the largest resident size
    of a process (which holds the benchmark's pack, 0.28 GB, besides the
    engine), and the minutes the scene of :data:`SCENE_SOURCE_S` would take.
    """
    context = multiprocessing.get_context("spawn")
    report: dict[str, Any] = {}
    counts = list(processes)
    for label in scenes:
        moving = label == "moving"
        rows = []
        for count in counts:
            ready, done = context.Queue(), context.Queue()
            go = context.Event()
            workers = [
                context.Process(target=_stems, args=(moving, duration_s, seed, ready, go, done))
                for _ in range(count)
            ]
            for worker in workers:
                worker.start()
            for _ in workers:
                ready.get()
            go.set()
            results = [done.get() for _ in workers]
            for worker in workers:
                worker.join()
            cost = [seconds for seconds, _ in results]
            together = float(sum(1.0 / seconds for seconds in cost))
            rows.append(
                {
                    "processes": count,
                    "compute_s_per_scene_s": round(float(np.mean(cost)), 4),
                    "stem_s_per_wall_s": round(together, 2),
                    "resident_mb": round(max(size for _, size in results) / 1e6),
                    "scene_minutes": round(SCENE_SOURCE_S / together / 60.0, 1),
                }
            )
        report[label] = rows
    return report


def table(after: dict[str, Any], before: dict[str, Any] | None = None) -> str:
    """The report as lines of text, beside an earlier one when given."""
    lines = []
    reports = [("before", before), ("after", after)] if before else [("", after)]
    lines.append("seconds of compute per second of scene, one source, one process")
    lines.append(f"{'':16s}" + "".join(f"{part:>9s}" for part in (*PARTS, "total")))
    for label in ("rest", "moving"):
        for name, report in reports:
            assert report is not None
            scene = report[label]
            cells = "".join(f"{scene[part]:9.3f}" for part in (*PARTS, "total"))
            lines.append(f"{label + ' ' + name:16s}" + cells)
    if "scaling" in after:
        lines.append("")
        lines.append("stems a run at a time, processes of one thread each")
        lines.append(
            f"{'':16s}{'processes':>10s}{'s/s each':>10s}{'stem s/s':>10s}{'MB each':>9s}"
            f"{'scene min':>11s}"
        )
        for label in ("rest", "moving"):
            for name, report in reports:
                assert report is not None
                for row in report.get("scaling", {}).get(label, []):
                    lines.append(
                        f"{label + ' ' + name:16s}{row['processes']:10d}"
                        f"{row['compute_s_per_scene_s']:10.3f}{row['stem_s_per_wall_s']:10.2f}"
                        f"{row['resident_mb']:9d}{row['scene_minutes']:11.1f}"
                    )
    return "\n".join(lines)
