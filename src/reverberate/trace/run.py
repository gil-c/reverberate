"""The trace on the machine: a bundle in, ``pack.h5`` out, resumed from what is on disk.

The stages, each leaving what the next and a rerun look for:

1. ``voxelise``, ``plan``: the low grid and an array at every cell
   (:class:`reverberate.accel.pairs.PairsCampaign`); the centres the arrays
   really got are the pack's ``/cells``;
2. ``assign``: the cell rule on those centres (:func:`reverberate.trace.plan.assign`),
   which names the pairs;
3. ``solve``: the pairs not in the cache, a launch of the wave solver a job;
4. ``paths``: the batched early trace of every source over its audible
   steps, and of every pair at rest for the levelling, with the diffracted
   onsets (``early/<source>.npz``), a block of steps a job;
5. ``rays``: the histograms at the tail's sites over the tail's cells, in
   their cache (``tails/``, a (site, cell) an entry, the dwelling's own under
   ``REVERBERATE_MIRROR_STORE``), a site a job, and each source's table;
6. ``level``: a pair's seam and onset (``level.jsonl``, a line a pair), the
   pairs of some cells a job;
7. ``rows``: a pair as the pack stores it, a block of pairs a job;
8. ``write``: the pack, a source at a time, through
   :class:`reverberate.render.pack.PackWriter`;
9. ``check``: the pack read back, and, for a smoke run or when the bundle
   asks (``check``: ``full``), read back whole and rendered by the signal
   engine on the host and on the card, which must agree to 1e-6 of the peak
   and say which device computed. The whole scene pays the read alone.

**Stages 3 to 7 are the jobs of one queue** (:mod:`reverberate.trace.pool`,
:meth:`Trace.work`), pulled by a process a card and a process a core that is
left (:mod:`reverberate.trace.resources`). A job waits for the jobs whose
files it reads and for nothing else, so the stages overlap: the cards cast
the rays and solve while the host's cores trace the early paths, level the
pairs whose launches are home and make their rows. What only a card does
well (the solver, the rays) is a card's; the early trace, the levelling and
the rows are the cores', on ``numpy`` on every machine, the early trace's
inner loops in a compiled text that decides as ``numpy`` does
(:mod:`reverberate.mirror.native`). What the early trace's workers would
each prepare is made once and mapped by all (:mod:`reverberate.mirror.shared`,
the run's ``mirror_store``). A job's result is a file under the job's name and a stage's merge
reads them in the jobs' order: **the pack is the same bytes, but for its
date and its seconds, whatever the cards and the cores and whichever job
failed and was made again** (:func:`pack_digest`, ``tests/test_trace_pool.py``).

The stages' seconds, and the seconds of what they are made of, are in
``trace_report.json`` (``timings_s``, ``seconds``, ``pool``): the cost ledger
of ``docs/adr/0016-appendix-trace-cost.md`` is read from there, and
``docs/adr/0016-appendix-every-card-every-core.md`` is the queue's own.

``status.json``, ``campaign.log`` and the two markers are a campaign's, so
:mod:`reverberate.gpu.onebox` watches a trace as it watches a field.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.accel.pairs import PairCache
from reverberate.audio import Atmosphere
from reverberate.compute import (
    Devices,
    card_free_bytes,
    device_report,
    to_numpy,
    usable_cores,
    xp_for,
)
from reverberate.metrics import band_centres
from reverberate.mirror import native
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.moving import (
    KIND_DIRECT,
    EarlyTable,
    MovingSettings,
    prepare,
    trace_early,
)
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.render import TailNoise
from reverberate.mirror.shared import Store
from reverberate.mirror.tails import (
    TailCache,
    TailTable,
    shared_scene,
    site_key,
    sites_read,
    tail_key,
    tail_table,
)
from reverberate.mirror.tracer import structure as ray_structure
from reverberate.render.compact import Levers
from reverberate.render.pack import (
    Air,
    Cells,
    Early,
    Header,
    Level,
    Listener,
    Low,
    Mirror,
    PackWriter,
    Source,
    Tail,
    default_fusion,
    read_pack,
    tail_seed,
)
from reverberate.render.seam import (
    GIVEN,
    MEDIAN,
    SEAM_CONSTANT_DB,
    TAPER,
    band_levels,
    scene_constant_db,
    seam_record,
    taper_of,
)
from reverberate.scenes import canonical_bytes, load_recipe
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M, LOW_RATE_HZ
from reverberate.spatial.translate import clearance_m
from reverberate.trace.assets import MirrorAssets, directivity_models, found_assets, mismatched
from reverberate.trace.clock import CLOCK_S, read_direct, verdict
from reverberate.trace.computed import write_as_computed
from reverberate.trace.engines import CardPairs, PairsEngine
from reverberate.trace.level import (
    at_pack_length,
    first_arrival_s,
    mirror_omni,
    pair_anchor_s,
    pair_low,
    pair_omni,
    pair_seam_db,
    step_levels,
)
from reverberate.trace.plan import (
    RAYS_MEASURED,
    RAYS_SITE_CELL_S,
    RAYS_SITE_S,
    Profile,
    Tracks,
    assign,
    pairs_of,
    tail_cells,
    tail_sites_of,
    tracks_of,
)
from reverberate.trace.pool import CARD, HOST, PARENT, THREAD, Job, Pool, WorkerSpec
from reverberate.trace.resources import Machine, measure, predict

__all__ = [
    "KIND",
    "PAIRS_BLOCK",
    "PATHS_BLOCK",
    "Journal",
    "Trace",
    "make_worker",
    "pack_digest",
    "render_check",
    "run_trace",
]

#: What a trace's ``campaign.json`` says it is, and the command line reads.
KIND = "scene-trace"
#: (step, image) pairs validated at once on a card: the appendix's figure, 5000 steps and 1 GB.
CARD_PAIRS_PER_VALIDATION = 3_000_000
#: The engine's two array modules must agree to this share of the peak (V4).
RENDER_TOLERANCE = 1e-6
#: Seconds of the pack the check renders, the sources it renders (those heard longest in
#: those seconds), and the block it renders them in. The host's render is the cost: six
#: seconds of one core a second of a moving source on the first card box, whose check of
#: a minute of fourteen sources did not end in thirty-five minutes.
CHECK_SECONDS = 10.0
CHECK_SOURCES = 3
CHECK_BLOCK_S = 5.0
#: What the ``check`` stage does: the whole pack read and rendered on both modules, or the
#: pack's structure read. A smoke run takes the first unless its bundle says otherwise.
CHECK_FULL = "full"
CHECK_READ = "read"
#: The region of the early trace is rounded outwards to this, m.
REGION_STEP_M = 0.25
#: Audible steps of one source a job of the early trace holds, and pairs a job of the
#: levelling, of the pack's rows or of the pairs' own early table holds. They are the
#: jobs' bounds and never the workers': the same on every machine, so that what is merged
#: is the same whoever computed it.
PATHS_BLOCK = 256
PAIRS_BLOCK = 64
#: The table of the pairs at rest among the sources' early tables.
PAIRS_TABLE = "_pairs"
#: Histograms a worker's process keeps in memory: a site's is tens of megabytes.
TAILS_KEPT = 12
#: Where the mirror's shared preparation is kept instead of the run's own ``mirror_store``:
#: a directory that outlives the run, for a second scene of the same dwelling.
STORE_VARIABLE = "REVERBERATE_MIRROR_STORE"

_EARLY_FIELDS = (
    "offsets",
    "path_id",
    "delay_s",
    "arrival",
    "departure",
    "gain",
    "order",
    "kind",
    "length_m",
    "sequence",
    "rank",
    "source",
    "listener",
)


@dataclass
class Journal:
    """The log, the status and the timings of a run that has no campaign to keep them."""

    out: Path
    started: float = field(default_factory=time.time)
    status: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    quiet: bool = False

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.status = {
            "stage": "start",
            "started": self.started,
            "device": device_report(),
            "cpus": os.cpu_count(),
            "usable_cpus": usable_cores(),
        }

    def say(self, message: str) -> None:
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - self.started) / 3600:.2f} h"
            f" | {message}"
        )
        if not self.quiet:
            print(line, flush=True)
        with (self.out / "campaign.log").open("a") as handle:
            handle.write(line + "\n")

    def set_status(self, **fields: Any) -> None:
        self.status.update(fields)
        self.status["updated"] = time.time()
        self.status["elapsed_s"] = round(time.time() - self.started, 1)
        path = self.out / "status.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        tmp.replace(path)

    def stage(self, name: str, work: Any, **status: Any) -> Any:
        self.set_status(stage=name, stage_started=time.time(), **status)
        self.say(f"{name}: start")
        t0 = time.time()
        result = work()
        self.timings[name] = round(self.timings.get(name, 0.0) + time.time() - t0, 1)
        self.say(f"{name}: done in {(time.time() - t0) / 60:.1f} min")
        return result


@dataclass(frozen=True)
class PairRows:
    """A source's ``low/ir`` as the pack's writer reads it: a row when asked, never the table.

    The writer copies a table that is not an array a row at a time
    (:mod:`reverberate.render.pack`), so one pair, 1.2 MB, is all that is
    held. The table held whole was 5.6 GB for 4586 pairs, and on a machine
    whose page cache stood at its container's limit its copy took 164 ms a
    pair with the card idle, against 5 ms for the pair's own work.
    """

    shape: tuple[int, int, int]
    row: Any

    def __getitem__(self, index: int) -> np.ndarray:
        found = np.asarray(self.row(int(index)))
        if found.shape != self.shape[1:]:
            raise ValueError(f"a row of {found.shape}, not of {self.shape[1:]}")
        return found


def _digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _save_early(path: Path, table: EarlyTable, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        digest=np.asarray(digest),
        sound_speed_m_s=np.asarray(table.sound_speed_m_s),
        record=np.asarray(json.dumps(table.record, default=str)),
        **{name: getattr(table, name) for name in _EARLY_FIELDS},
    )
    partial.replace(path)


def _load_early(path: Path, digest: str) -> EarlyTable | None:
    if not path.is_file():
        return None
    with np.load(path) as held:
        if str(held["digest"]) != digest:
            return None
        return EarlyTable(
            **{name: np.asarray(held[name]) for name in _EARLY_FIELDS},
            sound_speed_m_s=float(held["sound_speed_m_s"]),
            record=json.loads(str(held["record"])),
        )


def _early_is(path: Path, digest: str) -> bool:
    """Whether ``path`` holds a table of this digest, without reading the table."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as held:
            return str(held["digest"]) == digest
    except (OSError, ValueError, KeyError):
        return False


def _blocks(heard: np.ndarray, size: int) -> list[tuple[int, int]]:
    """Every step, in consecutive runs that hold ``size`` audible steps at most.

    The bounds depend on the steps and on ``size`` alone: a table is cut
    the same way on every machine, whatever computes its blocks.
    """
    total = int(heard.shape[0])
    if total == 0:
        return [(0, 0)]
    count = np.cumsum(np.asarray(heard, dtype=np.int64))
    edges = {0, total}
    for filled in range(size, int(count[-1]), size):
        edges.add(int(np.searchsorted(count, filled, side="right")))
    cuts = sorted(edges)
    return list(zip(cuts[:-1], cuts[1:], strict=True))


_ROW_FIELDS = tuple(name for name in _EARLY_FIELDS if name not in ("offsets", "source", "listener"))


def _joined(parts: list[EarlyTable], source: np.ndarray, head: np.ndarray) -> EarlyTable:
    """A table from the tables of its consecutive blocks: the rows one after the other."""
    offsets = [np.zeros(1, dtype=np.int64)]
    base = 0
    for part in parts:
        offsets.append(np.asarray(part.offsets[1:], dtype=np.int64) + base)
        base += int(part.offsets[-1])
    record: dict[str, Any] = {"blocks": len(parts)}
    for part in parts:
        for key, value in part.record.items():
            if isinstance(value, int) and not isinstance(value, bool):
                record[key] = int(record.get(key, 0)) + value
    # What sieved and validated the blocks: the compiled text, numpy, or some of each.
    engines = sorted({str(part.record["engine"]) for part in parts if "engine" in part.record})
    if engines:
        record["engine"] = "; ".join(engines)
    return EarlyTable(
        offsets=np.concatenate(offsets),
        **{name: np.concatenate([getattr(part, name) for part in parts]) for name in _ROW_FIELDS},
        source=np.atleast_2d(np.asarray(source, dtype=float)),
        listener=np.atleast_2d(np.asarray(head, dtype=float)),
        sound_speed_m_s=parts[0].sound_speed_m_s,
        record=record,
    )


def _first_channel(cache: PairCache, key: str) -> np.ndarray:
    """Channel 0 of a pair in the cache, ``[1, sample]``: the file mapped, 19 kB of it read."""
    if not cache.has(key):
        raise KeyError(f"no pair {key} under {cache.directory}: not in the cache after the solve")
    return cache.first_channel(key)


#: What of a pack's provenance says when and on what it was made, and not what it holds.
_VOLATILE = ("created_utc", "timings_s", "device", "cost", "low_pairs")


def pack_digest(path: Path) -> str:
    """A digest of everything a pack holds but when and on what it was made.

    Every dataset's bytes, type and shape and every attribute, in the
    file's own order of names; of the provenance, all but the date, the
    seconds, the device and the count of what was solved against what was
    found. Two runs of one bundle have one digest whatever machine, however
    many workers and in whatever order their jobs ended, and the tests hold
    them to it.
    """
    import h5py

    digest = hashlib.sha256()

    def attributes(held: Any) -> None:
        for key in sorted(held.attrs):
            value = held.attrs[key]
            if key == "provenance_json":
                kept = {k: v for k, v in json.loads(value).items() if k not in _VOLATILE}
                value = json.dumps(kept, sort_keys=True)
            digest.update(key.encode())
            digest.update(value.encode() if isinstance(value, str) else np.asarray(value).tobytes())

    def visit(name: str, held: Any) -> None:
        digest.update(name.encode())
        attributes(held)
        if not isinstance(held, h5py.Dataset):
            return
        digest.update(f"{held.dtype}{held.shape}".encode())
        if held.dtype.kind == "O":
            digest.update(json.dumps([_text_of(v) for v in held[()].ravel()]).encode())
        elif held.ndim and held.size * held.dtype.itemsize > 2**28:
            for row in range(held.shape[0]):
                digest.update(np.ascontiguousarray(held[row]).tobytes())
        else:
            digest.update(np.ascontiguousarray(held[()]).tobytes())

    with h5py.File(path, "r") as handle:
        attributes(handle)
        handle.visititems(visit)
    return digest.hexdigest()


def _text_of(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass
class _WorkerJournal:
    """What a worker's process says: a log of its own, and no status but the run's."""

    out: Path
    index: int
    status: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def say(self, message: str) -> None:
        log = Path(self.out) / "workers" / f"{self.index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")

    def set_status(self, **fields: Any) -> None:
        self.status.update(fields)

    def stage(self, name: str, work: Any, **status: Any) -> Any:
        self.say(f"{name}: start")
        return work()


def make_worker(
    spec: WorkerSpec,
    *,
    bundle: str,
    out: str,
    engine_told: dict[str, Any],
    pffdtd_dir: str = "/root/pffdtd",
) -> Trace:
    """A worker's own trace of the run, built in its process from the files of the run.

    The bundle is read again, not handed over: the page cache shares its
    files between the workers, and nothing of the scene crosses a pipe. The
    worker measures the device it holds (:func:`reverberate.trace.resources.
    measure`) before it says it is ready.
    """
    from reverberate.trace.engines import build_engine

    card = spec.card is not None
    engine = build_engine(engine_told, Path(bundle), Path(out), gpu=card, pffdtd_dir=pffdtd_dir)
    trace = Trace(
        bundle=Path(bundle),
        out=Path(out),
        engine=engine,
        gpu=card,
        pffdtd_dir=Path(pffdtd_dir),
        quiet=True,
        worker=spec,
    )
    trace.measured = measure(spec.card, trace.xp, seconds=float(engine_told.get("measure_s", 1.0)))
    return trace


@dataclass
class Trace:
    """One recipe's pack, made on this machine from a bundle."""

    bundle: Path
    out: Path
    #: Where the pairs come from; the bundle's campaign on a card unless given.
    engine: PairsEngine | None = None
    gpu: bool | None = None
    devices: Devices | None = None
    pffdtd_dir: Path = Path("/root/pffdtd")
    card_devices: str | None = None
    solvers: int | None = None
    quiet: bool = False
    #: :data:`CHECK_FULL` or :data:`CHECK_READ`; the bundle's, or the profile's, unless given.
    check_mode: str | None = None
    #: The machine as it was read; left out, this one is asked when the work starts.
    machine: Machine | None = None
    #: Host processes beside the cards'. ``None`` is the cores the cards leave; 0 is no
    #: process at all, every job in this one, which is how the tests and one core run.
    workers: int | None = 0
    #: What a worker's process builds its own engine from (:func:`make_worker`); without
    #: it there is no other process, whatever ``workers`` says.
    engine_told: dict[str, Any] | None = None
    #: In a worker's process: which worker this is.
    worker: WorkerSpec | None = None
    #: Failures a test asks of the queue (:mod:`reverberate.trace.pool`).
    faults: dict[str, Any] | None = None
    #: The run stops before its long work where it is predicted to last longer, hours.
    max_hours: float | None = None
    #: USD an hour of this machine, for the prediction's price; the bundle's unless given.
    rate_usd_per_hour: float | None = None

    def __post_init__(self) -> None:
        self.bundle, self.out = Path(self.bundle), Path(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec = json.loads((self.bundle / "campaign.json").read_text())
        if self.spec.get("kind") != KIND:
            raise ValueError(f"{self.bundle} is not the bundle of a scene trace")
        told = dict(self.spec["trace"])
        held = self.bundle / "trace"
        self.recipe = load_recipe(held / "recipe.json")
        self.recipe_bytes = canonical_bytes(self.recipe)
        self.profile = Profile.from_record(told["profile"])
        self.crossover = Crossover(**told.get("crossover", {}))
        self.told = told
        self.assets = MirrorAssets.load(held / "mirror")
        with np.load(held / "plan.npz") as plan:
            self.asked = np.asarray(plan["cells"], dtype=float)
            self.kind = np.asarray(plan["kind"], dtype=np.uint8)
            self.patch_cells = np.asarray(plan["patch_cells"], dtype=float).reshape(-1, 3)
            self.patch_source = int(plan["patch_source"])
        self.tracks: Tracks = tracks_of(self.recipe, self.profile)
        self.xp = xp_for(self.gpu)
        if self.devices is None:
            self.devices = Devices.detect() if self.xp is not np else Devices.host(1)
        if self.engine is None:
            self.engine = CardPairs(
                self.bundle / "pairs",
                self.out,
                pffdtd_dir=self.pffdtd_dir,
                devices=self.card_devices,
                gpu=self.gpu,
                solvers=self.solvers,
            )
        campaign = getattr(self.engine, "campaign", None)
        self.journal: Any = (
            campaign if campaign is not None else Journal(self.out, quiet=self.quiet)
        )
        if self.worker is not None:
            # A worker says what it does in a log of its own and leaves the run's status alone.
            self.journal = _WorkerJournal(self.out, self.worker.index)
            if campaign is not None:
                campaign.say = self.journal.say
                campaign.set_status = self.journal.set_status
        self._assigned = False
        self._owner: dict[tuple[int, int], str] | None = None
        self._rows_files: dict[int, Path] = {}
        self._rows_open: tuple[int, Any] | None = None
        self.measured: dict[str, Any] | None = None
        if self.check_mode is None:
            self.check_mode = str(
                told.get("check")
                or (CHECK_FULL if self.profile.seconds is not None else CHECK_READ)
            )
        if self.check_mode not in (CHECK_FULL, CHECK_READ):
            raise ValueError(f"a trace checks {CHECK_FULL!r} or {CHECK_READ!r}")
        self.report: dict[str, Any] = {"kind": KIND, "profile": self.profile.record()}
        #: Seconds of what the stages are made of, by stage: the ledger's units.
        self.seconds: dict[str, dict[str, float]] = {}
        sources = self.bundle / "trace" / "positions.npy"
        if sources.is_file() and not np.array_equal(np.load(sources), self.tracks.positions):
            raise RuntimeError(
                "the source positions of the bundle are not those this code reads in the recipe"
            )

    # ---- what the bundle is checked against ------------------------------------------------

    def check_assets(self) -> list[str]:
        """The recipe's asset keys that are not this trace's; refused unless the bundle allows."""
        assert self.engine is not None
        found = found_assets(
            self.recipe,
            self.assets,
            voxel_low_key=self.engine.voxel_low_key,
            export_sha256=str(self.told.get("export_sha256", "")),
        )
        wrong = mismatched(self.recipe, found)
        self.report["assets"] = found
        self.report["assets_mismatched"] = wrong
        # A trace on another low grid than the recipe names says so in its bundle: that
        # key, and no other, then differs by intent. The pack's provenance keeps it.
        meant = {str(name) for name in self.told.get("allowed_mismatch", [])}
        refused = [name for name in wrong if name not in meant]
        if refused and not self.told.get("allow_asset_mismatch", False):
            raise RuntimeError(
                "the recipe was generated against other assets than this trace finds: "
                + ", ".join(refused)
            )
        if wrong:
            self.journal.say(f"assets: the recipe disagrees on {', '.join(wrong)}; allowed")
        return wrong

    # ---- stages ----------------------------------------------------------------------------

    def assign(self, centres: list[np.ndarray | None]) -> None:
        """The cell rule on the centres the arrays got, and the pairs it names."""
        scene = self.asked.shape[0]
        if len(centres) != scene + self.patch_cells.shape[0]:
            raise RuntimeError(f"{len(centres)} arrays for {scene} cells and the patch")
        self.placed = np.array([c is not None for c in centres[:scene]], dtype=bool)
        self.cells = np.array(
            [self.asked[i] if c is None else c for i, c in enumerate(centres[:scene])], dtype=float
        ).reshape(-1, 3)
        self.clearance = clearance_m(self.cells, self.assets.triangles)
        moved = np.linalg.norm(self.cells - self.asked, axis=1)
        self.low, refused = assign(self.tracks, self.cells, self.clearance, usable=self.placed)
        self.heard_at = pairs_of(self.tracks, self.low)
        if self.patch_cells.shape[0] and self.patch_source >= 0:
            stood = [scene + i for i, c in enumerate(centres[scene:]) if c is not None]
            self.heard_at[self.patch_source] = sorted(
                set(self.heard_at[self.patch_source]) | set(stood)
            )
        self.pairs = sorted(
            (position, cell)
            for position, cells in enumerate(self.heard_at)
            for cell in cells
            if cell < scene
        )
        self.report["cells"] = {
            "asked": scene,
            "without_an_array": int((~self.placed).sum()),
            "moved_m": {
                "median": round(float(np.median(moved)), 4) if scene else 0.0,
                "worst": round(float(moved.max()), 4) if scene else 0.0,
            },
            "patch": int(self.patch_cells.shape[0]),
            "patch_without_an_array": sum(1 for c in centres[scene:] if c is None),
        }
        self.report["fallback_steps"] = [[name, step] for name, step in refused]
        self.report["pairs"] = {
            "scene": len(self.pairs),
            "all": sum(len(cells) for cells in self.heard_at),
        }
        self.journal.say(
            f"assign: {scene} cells ({int((~self.placed).sum())} without an array),"
            f" {len(self.pairs)} pairs, {len(refused)} step(s) the rule refused"
        )
        assert self.engine is not None
        self.pair_key = [self.engine.key_of(position, cell) for position, cell in self.pairs]
        self.tail_rows = tail_cells(self.cells, self.kind)
        self._owner = None
        self._assigned = True
        if self.worker is None:
            # What a worker's process starts from: the centres, and nothing it can derive.
            state = self.out / "state"
            state.mkdir(parents=True, exist_ok=True)
            partial = state / "centres.partial.json"
            partial.write_text(
                json.dumps([None if c is None else [float(v) for v in c] for c in centres])
            )
            partial.replace(state / "centres.json")

    @property
    def owner(self) -> dict[tuple[int, int], str]:
        """The source a pair is levelled with: the first whose steps read it."""
        if self._owner is None:
            found: dict[tuple[int, int], str] = {}
            for name, track in self.tracks.sources.items():
                chosen = self.low[name]
                for step in np.flatnonzero(track.audible):
                    for position in track.read(int(step)):
                        for cell in chosen.cell[step]:
                            if position >= 0 and cell >= 0:
                                found.setdefault((int(position), int(cell)), name)
            self._owner = found
        return self._owner

    def _carried_pairs(self) -> int:
        """The pairs the bundle carried, into the cache; how many."""
        assert self.engine is not None
        carried_root = self.bundle / "pairs_cache"
        carried = 0
        if (carried_root / self.engine.voxel_low_key).is_dir():
            held = PairCache(carried_root, self.engine.voxel_low_key)
            for key, record in held.records().items():
                if held.has(key) and not self.engine.cache.has(key):
                    # The file as it is, in the form it was kept in: the cache reads both.
                    self.engine.cache.adopt(
                        key, held.path(key), {k: v for k, v in record.items() if k != "key"}
                    )
                    carried += 1
        return carried

    def solve(self) -> None:
        """The pairs the bundle carried into the cache, then those still to be solved."""
        assert self.engine is not None
        carried = self._carried_pairs()
        self.report["low_pairs"] = {**self.engine.solve(self.heard_at), "carried": carried}

    def solve_jobs(self, machine: Machine) -> list[Job]:
        """The solve as the queue's jobs: a launch a job where the engine plans launches.

        An engine that solves in one call of its own (the present engine, a
        source position a process) is one job in a thread, and what needs a
        card waits for it. ``self.made_by`` is the job each pair comes from.
        """
        assert self.engine is not None
        engine: Any = self.engine
        self.made_by: dict[tuple[int, int], str] = {}
        self.planned: dict[str, float] = {"node_updates": 0.0, "pairs": 0.0}
        # In one process a launch at a time would leave every card but one idle: the
        # engine's own solve, a thread a card, is then the whole of the stage, as it was
        # before the queue (``--host-workers 0`` on a machine of several cards).
        alone = self.pool.make is None and self.xp is not np and len(machine.cards) > 1
        if not hasattr(engine, "launches") or alone:
            for pair in self.pairs:
                self.made_by[pair] = "solve/all"
            return [Job("solve", "all", on=THREAD, work=self.solve)]
        carried = self._carried_pairs()
        launches = engine.launches(
            self.heard_at,
            free_bytes=[card.free_bytes for card in machine.cards],
            host_bytes=machine.ram_bytes,
        )
        on = CARD if engine.launches_on == "card" else HOST
        jobs = []
        for launch in launches:
            job = Job(
                "solve", launch["name"], launch["payload"], on=on, bytes=launch["bytes"], priority=1
            )
            jobs.append(job)
            self.planned["node_updates"] += float(launch.get("updates", 0.0))
            for position, cell in launch["pairs"]:
                self.made_by[(int(position), int(cell))] = job.key
                self.planned["pairs"] += 1

        def gathered() -> None:
            records = [
                record
                for key, record in self.pool.done.items()
                if key.startswith("solve/") and ("pairs" in record or "batch" in record)
            ]
            self.report["low_pairs"] = {
                **engine.solved_report(records, self.heard_at),
                "carried": carried,
                "launches": len(records),
            }

        jobs.append(
            Job("solve", "_report", on=PARENT, after=tuple(j.key for j in jobs), work=gathered)
        )
        return jobs

    def _spent(self, stage: str, name: str, since: float) -> None:
        held = self.seconds.setdefault(stage, {})
        held[name] = round(held.get(name, 0.0) + time.time() - since, 2)

    def _carried_early(self) -> None:
        """The early tables the bundle carries, where this run has none of that name yet.

        A table is read only if its digest is this run's (:func:`_load_early`),
        so a table of other positions, another scene or another region costs
        its copy and nothing else.
        """
        carried = self.bundle / "early_cache"
        if not carried.is_dir():
            return
        target = self.out / "early"
        target.mkdir(parents=True, exist_ok=True)
        taken = 0
        for table in sorted(carried.glob("*.npz")):
            if not (target / table.name).exists():
                shutil.copy2(table, target / table.name)
                taken += 1
        self.journal.say(f"paths: {taken} early table(s) taken from the bundle")

    # ---- the early trace: blocks of a source's steps, on the host's cores ----------------------

    def _prepared(self) -> None:
        """The mirror as the early trace reads it, the traced region, what names a table: once.

        On the host, whatever the machine holds. The early trace was bound
        by the interpreter and not by the device (25 ms a position on a
        laptop's core, 22 ms on an RTX 3090 with one core): its pairs are
        now sieved and validated by a compiled text, 2 ms a position
        (:mod:`reverberate.mirror.native`), a process a core, and its tables
        are the same bytes on every machine, with that text or with the
        ``numpy`` twin a machine without a compiler falls back to.
        """
        if getattr(self, "ms", None) is not None:
            return
        settings = self.assets.settings
        if self.worker is None:
            why = native.why_not()
            self.journal.say(
                "paths: sieved and validated by the compiled text"
                if why is None
                else f"paths: on numpy, about ten times the seconds: {why}"
            )
        t0 = time.time()
        # What every worker would build for itself is built once and mapped by the others
        # (:mod:`reverberate.mirror.shared`): the work no longer grows with the workers.
        self.store = Store(Path(os.environ.get(STORE_VARIABLE) or self.out / "mirror_store"))
        self.ms = prepare(self.assets.catalogue, settings, MovingSettings(), store=self.store)
        self._spent("paths", "prepare", t0)
        heads = np.concatenate([self.tracks.listener, self.cells])
        # Half a metre round the heads, out to the next quarter metre: the arrays' centres
        # are a grid's nodes, and a region that followed them to the millimetre would give
        # every low grid its own tables of the same sources (:data:`REGION_STEP_M`).
        self.region = (
            np.floor((heads.min(axis=0) - 0.5) / REGION_STEP_M) * REGION_STEP_M,
            np.ceil((heads.max(axis=0) + 0.5) / REGION_STEP_M) * REGION_STEP_M,
        )
        self.identity = [
            self.assets.catalogue.key,
            settings.record(),
            [list(r) for r in self.region],
        ]
        self._onset_field: Any = None

    def _onsets(self) -> Any:
        """The occupancy and its graph, when a table is to be traced and not before."""
        if self._onset_field is None:
            tracks = self.tracks
            every = [tracks.listener, self.cells, tracks.positions]
            every += [track.position for track in tracks.sources.values()]
            t0 = time.time()
            self._onset_field = onset_field(
                self.assets.catalogue,
                np.concatenate(every),
                sound_speed_m_s=self.assets.settings.sound_speed_m_s,
                store=self.store,
            )
            self._spent("paths", "onset_field", t0)
        return self._onset_field

    def _table_inputs(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A table's sources, heads and audible steps: a source's, or the pairs' at rest."""
        tracks = self.tracks
        if name == PAIRS_TABLE:
            rows = np.asarray(self.pairs, dtype=np.int64).reshape(-1, 2)
            return (
                tracks.positions[rows[:, 0]],
                self.cells[rows[:, 1]],
                np.ones(rows.shape[0], dtype=bool),
            )
        track = tracks.sources[name]
        return track.position, tracks.listener, np.asarray(track.audible, dtype=bool)

    def _block_path(self, name: str, block: int) -> Path:
        return self.out / "jobs" / "paths" / f"{name}.{block}.npz"

    def _trace_block(self, table: str, block: int, start: int, stop: int) -> dict[str, Any]:
        """One job of the early trace: steps ``start`` to ``stop`` of ``table``, to its file."""
        self._prepared()
        source, head, heard = (a[start:stop] for a in self._table_inputs(table))
        onsets = self._onsets()
        solved, read = int(onsets.solved), int(onsets.read)
        made, met = dict(self.store.made), dict(self.store.found)
        t0, cpu = time.time(), time.process_time()
        found = trace_early(
            self.ms,
            source,
            head,
            audible=heard,
            region=self.region,
            onsets=onsets,
            # A worker that holds a card sieves and validates on it where that is asked
            # for (:data:`reverberate.mirror.native.CARD_VARIABLE`): the same text as a
            # host's core runs, and the same table.
            xp=self.xp if (self.xp is not np and native.on_cards()) else np,
        )
        self._spent("paths", "trace", t0)
        _save_early(
            self._block_path(table, block), found, _digest(source, head, heard, self.identity)
        )
        return {
            "record": found.record,
            "fields": int(onsets.solved) - solved,
            "fields_read": int(onsets.read) - read,
            # This process's own seconds, without what it waited for another's entry.
            "cpu_s": round(time.process_time() - cpu, 4),
            "made": {k: v - made.get(k, 0) for k, v in self.store.made.items()},
            "found": {k: v - met.get(k, 0) for k, v in self.store.found.items()},
        }

    def _merge_early(self, name: str, digest: str, blocks: list[tuple[int, int]]) -> None:
        """A table from its blocks, in the blocks' order, to ``early/<name>.npz``."""
        source, head, heard = self._table_inputs(name)
        parts = []
        for block, (start, stop) in enumerate(blocks):
            part = _load_early(
                self._block_path(name, block),
                _digest(source[start:stop], head[start:stop], heard[start:stop], self.identity),
            )
            if part is None:
                raise RuntimeError(f"block {block} of the early table of {name} is not on disk")
            parts.append(part)
        found = _joined(parts, source, head)
        _save_early(self.out / "early" / f"{name}.npz", found, digest)
        self._paths_records[name] = found.record
        self._table_is(name, found)

    def _table_is(self, name: str, table: EarlyTable) -> None:
        if name == PAIRS_TABLE:
            self.pair_early = table
        else:
            self.early[name] = table

    def paths_jobs(self) -> list[Job]:
        """Every source's early table, and the pairs' at rest, in blocks; a merge a table."""
        self._prepared()
        self._carried_early()
        self.early: dict[str, EarlyTable] = {}
        self._paths_records: dict[str, Any] = {}
        jobs: list[Job] = []
        for name in [*self.tracks.sources, PAIRS_TABLE]:
            source, head, heard = self._table_inputs(name)
            digest = _digest(source, head, heard, self.identity)
            found = _load_early(self.out / "early" / f"{name}.npz", digest)
            if found is not None:
                self._table_is(name, found)
                self._paths_records[name] = {"cached": True}
                continue
            blocks = _blocks(heard, PAIRS_BLOCK if name == PAIRS_TABLE else PATHS_BLOCK)
            keys = []
            for block, (start, stop) in enumerate(blocks):
                wanted = _digest(
                    source[start:stop], head[start:stop], heard[start:stop], self.identity
                )
                if _early_is(self._block_path(name, block), wanted):
                    continue
                job = Job(
                    "paths",
                    f"{name}.{block}",
                    {"table": name, "block": block, "start": start, "stop": stop},
                    on=HOST,
                    # A table's blocks share their sources' image trees: a worker that
                    # grew them for one block is handed the next.
                    group=f"paths/{name}",
                )
                jobs.append(job)
                keys.append(job.key)
            jobs.append(
                Job(
                    "paths",
                    name,
                    on=PARENT,
                    after=tuple(keys),
                    work=functools.partial(self._merge_early, name, digest, blocks),
                )
            )
        if any(job.on == HOST for job in jobs):
            # The onsets' field too is made here, once, before a worker asks for it: every
            # worker's first block would otherwise make its own at the same moment.
            self._onsets()
        return jobs

    def paths(self) -> None:
        """Every source's early table, and the table of the pairs at rest: the stage alone."""
        self._alone(self.paths_jobs())
        self._paths_report()

    def _paths_report(self) -> None:
        self.report["paths"] = dict(self._paths_records)
        records = [r for key, r in self.pool.done.items() if key.startswith("paths/")]
        for name in ("fields", "fields_read"):
            self.report["distance_" + name] = int(sum(int(r.get(name, 0)) for r in records))
        # What the store was asked to hold and what was found in it, over every worker: a
        # tree, a list or a field made twice is two in ``made``.
        shared: dict[str, dict[str, int]] = {
            "made": dict(self.store.made),
            "found": dict(self.store.found),
        }
        if self.pool.make is not None:
            for record in records:
                for name in ("made", "found"):
                    for kind, count in dict(record.get(name, {})).items():
                        shared[name][kind] = shared[name].get(kind, 0) + int(count)
        self.report["mirror_store"] = {"root": str(self.store.root), **shared}
        self.report["paths_cpu_s"] = round(sum(float(r.get("cpu_s", 0.0)) for r in records), 2)

    # ---- the rays: a site a job on a card, a source's table a job on the host ------------------

    def rays_jobs(self) -> list[Job]:
        """The histograms not in their cache, a site a job; then each source's table."""
        self._prepared()
        settings = self.assets.settings
        tracks = self.tracks
        self._tails()
        shared = shared_scene(self.tail_cache, self.assets.catalogue, settings)
        rays = settings.traced_rays()
        cells = self.cells[self.tail_rows]
        self.tail: dict[str, dict[str, np.ndarray]] = {}
        positions: dict[str, np.ndarray] = {}
        absent: dict[str, int] = {}
        read_by: dict[str, list[str]] = {}
        at_rest: dict[str, list[int]] = {}
        for j, pair in enumerate(self.pairs):
            at_rest.setdefault(self.owner[pair], []).append(j)
        for name, track in tracks.sources.items():
            sites = tail_sites_of(self.recipe, track)
            rows = {int(r) for r in sites_read(track.position, sites, track.audible)}
            if at_rest.get(name):
                # The pairs' own tails, the source on its solved positions: the levelling's.
                solved = tracks.positions[[self.pairs[j][0] for j in at_rest[name]]]
                rows |= {int(r) for r in sites_read(solved, sites)}
            read_by[name] = []
            for row in sorted(rows):
                # A site is a job; what is kept of it is a (site, cell) at a time, so a
                # site is cast again only for the cells no recipe of the dwelling has read.
                key = site_key(shared["key"], sites.positions[row], rays)
                if key not in positions:
                    positions[key] = sites.positions[row]
                    absent[key] = len(
                        self.tail_cache.absent(
                            [tail_key(shared["key"], positions[key], cell, rays) for cell in cells]
                        )
                    )
                read_by[name].append(key)
        todo = [key for key in positions if absent[key]]
        self._ray_sites = (len(positions), len(todo))
        self._ray_cells = (len(positions) * len(cells), sum(absent.values()))
        jobs = [
            Job("rays", key, {"position": [float(v) for v in positions[key]]}, on=CARD)
            for key in todo
        ]
        traced = {job.name: job.key for job in jobs}
        self._ray_keys = tuple(traced.values())
        tables = [
            Job(
                "tails",
                name,
                {"source": name},
                on=HOST,
                after=tuple(traced[key] for key in read_by[name] if key in traced),
            )
            for name in tracks.sources
        ]

        def read() -> None:
            for name in tracks.sources:
                with np.load(self.out / "jobs" / "tails" / f"{name}.npz") as held:
                    self.tail[name] = {key: np.asarray(held[key]) for key in held.files}
            self.report["rays"] = {
                "tail_cells": int(self.tail_rows.size),
                "histograms_traced": int(self._ray_sites[1]),
                "histograms_read": int(self._ray_sites[0] - self._ray_sites[1]),
                # A (site, cell) at a time: what another recipe of the dwelling left.
                "site_cells_traced": int(self._ray_cells[1]),
                "site_cells_read": int(self._ray_cells[0] - self._ray_cells[1]),
                "structure": ray_structure(),
                "precision": rays.precision,
                "cache": str(self.tail_cache.directory),
            }

        merge = Job("tails", "_read", on=PARENT, after=tuple(t.key for t in tables), work=read)
        return [*jobs, *tables, merge]

    def _ray_site(self, position: list[float]) -> dict[str, Any]:
        """One job of the rays: a site's histograms over the tail's cells, into the cache."""
        from reverberate.mirror.tails import histograms

        self._tails()
        t0 = time.time()
        histograms(
            self.assets.catalogue,
            self.assets.settings,
            np.asarray(position, dtype=float)[None, :],
            self.cells[self.tail_rows],
            devices=self._ray_devices(),
            cache=self.tail_cache,
            store=self._ray_store(),
        )
        self._spent("rays", "sites", t0)
        return {"traced": 1}

    def _tails(self) -> TailCache:
        """The histograms' cache: the dwelling's where one is named, the run's own otherwise.

        Under :data:`STORE_VARIABLE` the histograms outlive the run beside
        the mirror's shared preparation: a second recipe of the dwelling
        reads every (site, cell) the first one cast.
        """
        if getattr(self, "tail_cache", None) is None:
            shared = os.environ.get(STORE_VARIABLE)
            root = Path(shared) / "tails" if shared else self.out / "tails"
            self.tail_cache = TailCache(root, keep=TAILS_KEPT)
        return self.tail_cache

    def _ray_store(self) -> Store:
        """Where the rays' tree is kept: the mirror's store, made here when it is not yet."""
        store = getattr(self, "store", None)
        if store is None:
            store = Store(Path(os.environ.get(STORE_VARIABLE) or self.out / "mirror_store"))
        return store

    def _ray_devices(self) -> Devices:
        """What one job's rays are cast on: this process's card, or one share on its host.

        The histograms are counts, so their sum does not depend on how the
        rays were shared; on the host a site is one share, which is what a
        trace without a card always cast.
        """
        if self.worker is None:
            assert self.devices is not None
            return self.devices
        return Devices((0,), 1) if self.xp is not np else Devices.host(1)

    def _tail_job(self, source: str) -> dict[str, Any]:
        """One job of the tables: a source's ``tail`` group in the pack's types, to its file."""
        self._prepared()
        self._tails()
        track = self.tracks.sources[source]
        t0 = time.time()
        held = tail_table(
            self.ms,
            self.assets.settings,
            track.position,
            self.tracks.listener,
            tail_sites_of(self.recipe, track),
            self.cells[self.tail_rows],
            audible=track.audible,
            devices=self._ray_devices(),
            cache=self.tail_cache,
            xp=np,
        ).pack()
        held["hist_cell"] = self.tail_rows[held["hist_cell"]].astype(np.int32)
        target = self.out / "jobs" / "tails" / f"{source}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.npz")
        arrays: dict[str, Any] = dict(held)
        np.savez(partial, **arrays)
        partial.replace(target)
        self._spent("rays", "tables", t0)
        return {"histograms": int(held["energy"].shape[0])}

    def rays(self) -> None:
        """Each source's tail table, in the pack's types; the histograms in their cache."""
        self._alone(self.rays_jobs())

    def pair_tails(self, name: str, mine: list[int]) -> TailTable:
        """The tails of the pairs in ``mine`` at rest, as the pack gives them to ``name``'s steps.

        A step with the source on the pair's solved position and the head on
        its cell: the sites and the cells either side, and their weights.
        """
        rows = np.asarray([self.pairs[j] for j in mine], dtype=np.int64)
        return tail_table(
            self.ms,
            self.assets.settings,
            self.tracks.positions[rows[:, 0]],
            self.cells[rows[:, 1]],
            tail_sites_of(self.recipe, self.tracks.sources[name]),
            self.cells[self.tail_rows],
            devices=self._ray_devices(),
            cache=self.tail_cache,
            xp=np,
        )

    def level_pair(
        self,
        j: int,
        tails: TailTable,
        local: int,
        atmosphere: Atmosphere,
        early: EarlyTable | None = None,
        row: int | None = None,
        noise: TailNoise | None = None,
    ) -> dict[str, Any]:
        """Pair ``j``: its seam, its onset, the mirror's first arrival, how its response ends.

        ``early`` and ``row`` are the table that holds the pair's paths and
        its row there: a block of the pairs' table, or the whole of it. On
        ``numpy`` on every machine: a pair is a dozen small transforms, and
        the host's cores do them side by side, a process each.
        """
        assert self.engine is not None
        settings = self.assets.settings
        early = self.pair_early if early is None else early
        row = j if row is None else row
        # The seam and the onset are read on channel 0 alone, and channel 0 alone is read
        # from the file: the others wait for the pack's rows.
        solved = np.asarray(_first_channel(self.engine.cache, self.pair_key[j]))
        onset, aired = pair_omni(
            at_pack_length(solved),
            atmosphere,
            sound_speed_m_s=settings.sound_speed_m_s,
            lead_s=self.assets.pack_lead_s,
        )
        mirror = mirror_omni(
            early,
            row,
            tails,
            local,
            self.assets,
            seed=settings.seed + self.pairs[j][1],
            xp=np,
            noise=noise,
        )
        first = first_arrival_s(early, row)
        # A diffracted onset has no reflection either: the direct path is a kind, not an order.
        direct = bool(np.any(early.kind[early.rows(row)] == KIND_DIRECT))
        # The clock and the scale are read where the mirror puts the direct sound, on
        # every pair that has one (:mod:`reverberate.trace.clock`).
        heard = (
            read_direct(
                aired,
                first,
                lead_s=self.assets.pack_lead_s,
                sound_speed_m_s=settings.sound_speed_m_s,
            )
            if direct
            else {}
        )
        return {
            "seam_db": pair_seam_db(aired, mirror, self.crossover),
            "onset_s": onset,
            # Where the pair's two bands are joined in pressure: its own direct sound,
            # and its loudest sample where it holds none (``level.pair_anchor_s``).
            "anchor_s": pair_anchor_s(
                aired,
                self._straight_s(j),
                lead_s=self.assets.pack_lead_s,
                sound_speed_m_s=settings.sound_speed_m_s,
            ),
            "first_s": first,
            "direct": direct,
            **heard,
            # The engine convolves a response as it stands: one that rings up to its
            # last 50 ms was cut, or wrapped, by whatever made it. One solved for fewer
            # seconds than the pack keeps is read where its solve ended, before its fade.
            "end_db": _end_db(aired if solved.shape[-1] == aired.shape[-1] else solved[0]),
        }

    # ---- the levelling: blocks of pairs, on the host's cores ---------------------------------

    def _levelled(self, read: bool = True) -> tuple[str, dict[str, dict[str, Any]]]:
        """What names a levelling, and the pairs ``level.jsonl`` already holds under it."""
        settings = self.assets.settings
        ledger = self.out / "level.jsonl"
        known: dict[str, dict[str, Any]] = {}
        identity = _digest(
            self.crossover.record(),
            self.recipe.atmosphere.to_dict(),
            settings.record(),
            # A ledger of before the direct sound was read is levelled again, and one of
            # before the join was anchored on it.
            {"lead_s": self.assets.pack_lead_s, "clock": "direct/1", "anchor": "direct/1"},
        )
        if read and ledger.is_file():
            for line in ledger.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("identity") == identity:
                        known[str(record["key"])] = record
        return identity, known

    def _level_path(self, block: int) -> Path:
        return self.out / "jobs" / "level" / f"{block}.json"

    def _level_groups(self) -> list[list[int]]:
        """The pairs in the levelling's jobs: whole cells, :data:`PAIRS_BLOCK` pairs or more.

        By cell, and not in the pairs' own order: the mirror's tail at a
        pair is noise drawn from the cell's seed, eleven million deviates
        and two thirds of what a pair's levelling cost, and the pairs of
        one cell draw the same. In the pairs' order a job of 64 held 49
        cells of the whole scene's 831; by cell a job draws once a cell. The
        groups depend on the pairs alone, never on the machine.
        """
        order = sorted(range(len(self.pairs)), key=lambda j: (self.pairs[j][1], self.pairs[j][0]))
        groups: list[list[int]] = []
        for j in order:
            fresh = not groups or (
                len(groups[-1]) >= PAIRS_BLOCK and self.pairs[groups[-1][-1]][1] != self.pairs[j][1]
            )
            if fresh:
                groups.append([])
            groups[-1].append(j)
        return groups

    def level_jobs(self, others: list[Job]) -> list[Job]:
        """The pairs not yet levelled, some cells' pairs a job, and the ledger's merge.

        A job waits for what it reads and for nothing else: the pairs'
        early table, the tail's sites, and the launches its pairs come from.
        A cell is heard from every source that speaks while the head is
        there, so its launches are many and its job starts late in the
        solves: what is left of the levelling when the last launch ends is a
        tenth of a second a pair over the host's workers.
        """
        identity, known = self._levelled()
        present = {job.key for job in others}
        jobs: list[Job] = []
        for block, group in enumerate(self._level_groups()):
            path = self._level_path(block)
            if path.is_file():
                # A job a run levelled and did not live to merge.
                for record in json.loads(path.read_text()):
                    if record.get("identity") == identity:
                        known.setdefault(str(record["key"]), record)
            todo = [j for j in group if self.pair_key[j] not in known]
            if not todo:
                continue
            waits = {f"paths/{PAIRS_TABLE}", *self._ray_keys}
            waits |= {self.made_by.get(self.pairs[j], "") for j in todo}
            jobs.append(
                Job(
                    "level",
                    str(block),
                    {"block": block, "pairs": todo},
                    on=HOST,
                    after=tuple(sorted(waits & present)),
                    priority=2,
                )
            )

        levelled = list(jobs)

        def merge() -> None:
            made = 0
            with (self.out / "level.jsonl").open("a") as handle:
                for job in levelled:
                    for record in json.loads(self._level_path(int(job.name)).read_text()):
                        if str(record["key"]) in known:
                            continue
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        known[str(record["key"])] = record
                        made += 1
            self._levels_report(known, made)

        jobs.append(Job("level", "_merge", on=PARENT, after=tuple(j.key for j in jobs), work=merge))
        return jobs

    def _level_block(self, block: int, pairs: list[int]) -> dict[str, Any]:
        """One job of the levelling: the pairs of ``pairs``, a record each, to the job's file."""
        self._prepared()
        self._tails()
        identity, _ = self._levelled(read=False)
        atmosphere = Atmosphere(**self.recipe.atmosphere.to_dict())
        table = self._pairs_early()
        # The tails a source at a time, as the pack gives them to that source's steps.
        t0 = time.time()
        todo: dict[str, list[int]] = {}
        for j in pairs:
            todo.setdefault(self.owner[self.pairs[j]], []).append(j)
        tails = {owner: self.pair_tails(owner, mine) for owner, mine in todo.items()}
        local = {j: (owner, k) for owner, mine in todo.items() for k, j in enumerate(mine)}
        self._spent("level", "tails", t0)
        # The pairs a cell at a time: a cell's noise is drawn once (:class:`TailNoise`),
        # and a pair's record is the same in whatever order the job's pairs are taken.
        t0 = time.time()
        noise = TailNoise()
        found: dict[int, dict[str, Any]] = {}
        for j in sorted(pairs, key=lambda j: (self.pairs[j][1], self.pairs[j][0])):
            owner, k = local[j]
            found[j] = self.level_pair(j, tails[owner], k, atmosphere, table, j, noise)
        self._spent("level", "pairs", t0)
        # A pair's record is its own whoever made it; the lines are in the pairs' order.
        records = [{"key": self.pair_key[j], "identity": identity, **found[j]} for j in pairs]
        target = self._level_path(block)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.json")
        partial.write_text(json.dumps(records, sort_keys=True))
        partial.replace(target)
        return {"pairs": len(records), "noise_drawn": noise.drawn}

    def _pairs_early(self) -> EarlyTable:
        """The early table of the pairs at rest, read from the run's file once a process."""
        if getattr(self, "pair_early", None) is None:
            whole = self._table_inputs(PAIRS_TABLE)
            held = _load_early(
                self.out / "early" / f"{PAIRS_TABLE}.npz", _digest(*whole, self.identity)
            )
            if held is None:
                raise RuntimeError("the early table of the pairs at rest is not on disk")
            self.pair_early = held
        return self.pair_early

    def level(self) -> None:
        """Every pair's seam and onset, a line each in ``level.jsonl``: the stage alone."""
        if not hasattr(self, "made_by"):
            self.made_by = {}
        if not hasattr(self, "_ray_keys"):
            self._ray_keys = ()
        self._alone(self.level_jobs([]))

    def _levels_report(self, known: dict[str, dict[str, Any]], made: int) -> None:
        self.levels = [known[key] for key in self.pair_key]
        seams = np.array([r["seam_db"] for r in self.levels], dtype=float)
        direct = [r for r in self.levels if r["direct"]]
        trails = np.array([r["onset_s"] - r["first_s"] for r in direct], dtype=float)
        away = np.array([r["first_s"] for r in direct]) * self.assets.settings.sound_speed_m_s
        # What the pairs' loudest samples trail their direct sound by: the report's, and no
        # longer the check's. A far pair's loudest sample is often a later arrival: at 4
        # to 9 m in hssd_0076 the direct sound is there, at its time and at its level, and
        # something 5 to 8 ms after it is half as loud again. Two whole scenes were
        # stopped by the median of this after all their solves (2026-10-05), and a window
        # of far pairs alone fails at its first decile too.
        early = float(np.percentile(trails, 10)) if trails.size else self.assets.pack_lead_s
        by_distance: dict[str, Any] = {}
        for low, high in ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))):
            held = trails[(away >= low) & (away < high)]
            if held.size:
                name = f"{low:g} to {high:g} m" if np.isfinite(high) else f"over {low:g} m"
                by_distance[name] = {"pairs": int(held.size), **(_spread(held, 6) or {})}
        self.report["level"] = {
            "pairs": len(self.levels),
            "made": made,
            "read": len(self.levels) - made,
            "seam_db": _spread(seams),
            "end_db": _spread(np.array([r["end_db"] for r in self.levels], dtype=float), 1),
            # What a pair's loudest sample trails its direct sound by: the pack's lead and
            # the solver's pulse where the direct sound is the loudest, more where a later
            # arrival is. The median and the spread by distance say how many pairs anchor
            # on something else than their direct sound.
            "trail_s_of_pairs_with_a_direct_path": _spread(trails, 6),
            "trail_s_at_the_first_decile": round(early, 6),
            "trail_s_by_distance": by_distance,
            "lead_s": self.assets.pack_lead_s,
        }
        # The check: each pair read where the mirror puts its direct sound, its time and
        # its level against 1/d on the field's scale.
        clock, stopped = verdict(
            self.levels,
            self.assets.pack_lead_s,
            sound_speed_m_s=self.assets.settings.sound_speed_m_s,
        )
        self.report["level"]["direct_sound"] = clock
        if stopped is not None:
            raise RuntimeError(stopped)
        # Pairs that carry no such reading are judged as they were, by their loudest samples.
        if not clock["pairs"] and abs(early - self.assets.pack_lead_s) > CLOCK_S:
            raise RuntimeError(
                "the low band and the mirror are not on one clock: the pairs' onsets trail their"
                f" direct sound by {early * 1e3:.2f} ms at the first decile, lead included,"
                f" and the pack's lead is {self.assets.pack_lead_s * 1e3:.2f} ms. The pair cache"
                " is on the geometric clock: a response in it starts when its source does"
            )

    # ---- the pack's rows: blocks of pairs, on the host's cores ---------------------------------

    def _rows_path(self, block: int, start: int, stop: int) -> Path:
        """Where a block of ``low/ir`` rows waits for the pack, named by what makes them."""
        digest = _digest(
            self.pair_key[start:stop],
            self.crossover.record(),
            self.recipe.atmosphere.to_dict(),
            {
                "c": self.assets.settings.sound_speed_m_s,
                "lead_s": self.assets.pack_lead_s,
                "unit": FIELD_UNIT_AT_1M,
                "anchor": "direct/1",
            },
        )
        return self.out / "jobs" / "rows" / f"{block}.{digest[:16]}.npy"

    def rows_jobs(self, others: list[Job]) -> list[Job]:
        """Every pair as the pack stores it, a block a job, as soon as its launches are home.

        The air and the masks of a pair's 64 channels are 45 ms of one core;
        made while the cards solve, a process a core, they cost the run
        nothing, and the pack's write is then a copy.
        """
        present = {job.key for job in others}
        jobs = []
        self._rows_files = {}
        for block, start in enumerate(range(0, len(self.pairs), PAIRS_BLOCK)):
            stop = min(start + PAIRS_BLOCK, len(self.pairs))
            path = self._rows_path(block, start, stop)
            self._rows_files[block] = path
            if path.is_file():
                continue
            waits = {self.made_by.get(pair, "") for pair in self.pairs[start:stop]} & present
            jobs.append(
                Job(
                    "rows",
                    str(block),
                    {"block": block, "start": start, "stop": stop},
                    on=HOST,
                    after=tuple(sorted(waits)),
                    priority=3,
                )
            )
        return jobs

    def _rows_block(self, block: int, start: int, stop: int) -> dict[str, Any]:
        """One job of the rows: pairs ``start`` to ``stop`` as ``low/ir`` keeps them, to a file."""
        atmosphere = Atmosphere(**self.recipe.atmosphere.to_dict())
        rows = np.stack([self._made_row(j, atmosphere, np) for j in range(start, stop)])
        target = self._rows_path(block, start, stop)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(f".{os.getpid()}.partial.npy")
        np.save(partial, rows)
        partial.replace(target)
        return {"pairs": int(rows.shape[0])}

    def _straight_s(self, j: int) -> float:
        """Pair ``j``'s source position to its cell over the sound speed: its direct sound's."""
        position, cell = self.pairs[j]
        apart = np.linalg.norm(self.tracks.positions[int(position)] - self.cells[int(cell)])
        return float(apart) / self.assets.settings.sound_speed_m_s

    def _made_row(self, j: int, atmosphere: Atmosphere, xp: Any) -> np.ndarray:
        assert self.engine is not None
        row = pair_low(
            at_pack_length(self.engine.cache.read(self.pair_key[j])),
            self.crossover,
            atmosphere,
            sound_speed_m_s=self.assets.settings.sound_speed_m_s,
            lead_s=self.assets.pack_lead_s,
            unit_at_1m=FIELD_UNIT_AT_1M,
            straight_s=self._straight_s(j),
            xp=xp,
        )[0]
        return np.asarray(row, dtype=np.float32)

    def _low_row(self, j: int, atmosphere: Atmosphere) -> np.ndarray:
        """Pair ``j`` as the pack stores it, ``[channel, sample]`` float32.

        Read from its block's file where the queue made one, the file
        mapped and one block open at a time; made here on ``numpy`` where
        there is none.
        """
        t0 = time.time()
        block = j // PAIRS_BLOCK
        path = self._rows_files.get(block)
        if path is not None and path.is_file():
            if self._rows_open is None or self._rows_open[0] != block:
                self._rows_open = (block, np.load(path, mmap_mode="r"))
            row = np.asarray(self._rows_open[1][j - block * PAIRS_BLOCK], dtype=np.float32)
        else:
            row = self._made_row(j, atmosphere, np)
        self._spent("write", "low_ir", t0)
        return row

    def write(self) -> Path:
        """``pack.h5``, a source at a time, and of a source's ``low/ir`` a pair at a time."""
        assert self.engine is not None
        recipe, tracks, settings = self.recipe, self.tracks, self.assets.settings
        atmosphere = Atmosphere(**recipe.atmosphere.to_dict())
        floor = recipe.dwelling.floor_y_m
        stations = np.asarray([s.position for s in recipe.stations], dtype=float).reshape(-1, 3)
        rooms = [s.room for s in recipe.stations]
        if stations.shape[0]:
            near = np.linalg.norm(self.cells[:, None, :] - stations[None, :, :], axis=2).argmin(
                axis=1
            )
            room = tuple(rooms[int(i)] for i in near)
        else:
            room = ("",) * self.cells.shape[0]
        rays = settings.traced_rays()
        models = directivity_models()
        named = sorted(
            {recipe.source(name).directivity.model for name in tracks.sources} | {"omni"}
        )
        provenance = {
            "recipe_sha256": hashlib.sha256(self.recipe_bytes).hexdigest(),
            "assets": self.report.get("assets", {}),
            "assets_mismatched": self.report.get("assets_mismatched", []),
            "code_version": str(self.told.get("code_version", "unknown")),
            "solver": self.engine.solver,
            "low_pairs": self.report.get("low_pairs", {}),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": self.profile.record(),
            "fallback_steps": len(self.report.get("fallback_steps", [])),
            "device": self.journal.status.get("device", {}),
            "timings_s": dict(self.journal.timings),
            # A cost without its rate is not written: the laptop, which rented, adds them.
            "cost": [],
        }
        # The level above the crossover a band (``render.seam``): the pair's own seam at
        # the join, the scene's one number above it. The scalar is written as it was.
        seam = np.array([r["seam_db"] for r in self.levels], dtype=float)
        constant = scene_constant_db([seam]) if SEAM_CONSTANT_DB is None else SEAM_CONSTANT_DB
        base_db = 20.0 * float(np.log10(self.assets.pack_gain))
        bank = band_centres(48000)
        shares = taper_of(bank, self.crossover.cutoff_hz, TAPER)
        provenance["seam"] = seam_record(
            bank, shares, base_db, constant, MEDIAN if SEAM_CONSTANT_DB is None else GIVEN
        )
        # What a variant changed, named only when it did: the reference's pack is as it was.
        low_seconds = dict(self.told.get("low") or {}).get("seconds")
        if low_seconds is not None:
            provenance["low_seconds"] = float(low_seconds)
        if settings.rays.rays != RAYS_MEASURED:
            provenance["rays"] = int(settings.rays.rays)
        # The form the responses are written in, and the form the cache they were read
        # from is kept in, as the bundle says them (``trace.low_levers``, ``pair_cache``);
        # a bundle that says neither writes ``low/ir`` as it always was.
        levers = Levers.parse(str(self.told.get("low_levers") or ""))
        if not levers.off:
            provenance["low_levers"] = levers.record()
        if getattr(self.engine.cache, "levers", None):
            provenance["pair_cache"] = str(self.engine.cache.levers)
        header = Header(
            profile="trace",
            recipe_sha256=provenance["recipe_sha256"],
            dwelling=recipe.dwelling.name,
            scene_id=recipe.dwelling.scene_id,
            duration_s=tracks.duration_s,
            steps=tracks.steps,
            sound_speed_m_s=settings.sound_speed_m_s,
            bank_bands_hz=band_centres(48000),
            has_low=True,
            has_tail=True,
            fusion=default_fusion(),
            provenance=provenance,
        )
        target = self.out / "pack.h5"
        partial = self.out / "pack.partial.h5"
        # The anchor of each pair's join, which its row was made with, and what it trails
        # the mirror's first arrival at the pair by: the engine's window follows it.
        onset = np.array([r["anchor_s"] for r in self.levels], dtype=float)
        trail = np.array([r["anchor_s"] - r["first_s"] for r in self.levels], dtype=float)
        row_of = {pair: j for j, pair in enumerate(self.pairs)}
        with PackWriter(
            partial,
            header,
            self.recipe_bytes,
            Listener(tracks.listener, tracks.orientation.astype(np.float32)),
            Cells(
                position=self.cells,
                kind=self.kind,
                lattice_index=np.full((self.cells.shape[0], 3), -1, dtype=np.int32),
                clearance_m=self.clearance.astype(np.float32),
                room=room,
                layers_y_m=(floor + recipe.heights.seated_m, floor + recipe.heights.standing_m),
            ),
            mirror=Mirror(
                signature=np.asarray(self.assets.signature, dtype=float),
                lead_s=self.assets.pack_lead_s,
                alignment_gain=self.assets.pack_gain,
                lowcut_hz=40.0,
                lowcut_order=8,
                tail_from_s=settings.render.tail_from_s,
                tail_bursts=settings.render.tail_bursts,
                tail_gain_db=tuple(float(v) for v in settings.parameters.tail_gain_db),
                histogram_bin_s=rays.bin_s,
                histogram_order=rays.order,
                receiver_radius_m=rays.receiver_radius_m,
                settings_json=json.dumps(settings.record(), sort_keys=True, default=str),
            ),
            crossover=self.crossover,
            air=Air(atmosphere, True),
            directivity={name: models[name] for name in named if name in models},
            low_levers=None if levers.off else levers,
        ) as writer:
            for number, (name, track) in enumerate(tracks.sources.items(), start=1):
                self.journal.set_status(source=name, job=f"{number}/{len(tracks.sources)}")
                chosen = self.low[name]
                steps = tracks.steps
                pair = np.full((steps, 2, 2), -1, dtype=np.int32)
                local: dict[int, int] = {}
                for step in np.flatnonzero(track.audible):
                    for a in range(2):
                        for b in range(2):
                            position, cell = int(track.slot[step, a]), int(chosen.cell[step, b])
                            if position >= 0 and cell >= 0:
                                j = row_of[(position, cell)]
                                pair[step, a, b] = local.setdefault(j, len(local))
                # More than two positions a step: the same rows, and those of the others.
                slot_pair = None
                if track.rail_slot is not None:
                    slot_pair = np.full((steps, track.rail_slot.shape[1], 2), -1, dtype=np.int32)
                    for step in np.flatnonzero(track.audible):
                        for a, position in enumerate(track.rail_slot[step]):
                            for b in range(2):
                                cell = int(chosen.cell[step, b])
                                if position >= 0 and cell >= 0:
                                    j = row_of[(int(position), cell)]
                                    slot_pair[step, a, b] = local.setdefault(j, len(local))
                mine = sorted(local, key=lambda j: local[j])
                # A pair at a time, from the cache to the file: the array of a source's
                # rows is never held (:class:`PairRows`).
                ir = PairRows(
                    (len(mine), header.channels, header.low_samples),
                    lambda row, mine=mine: self._low_row(mine[row], atmosphere),
                )
                high_gain_db, onset_s = step_levels(
                    audible=track.audible,
                    pair=pair,
                    position_weight=track.weight,
                    cell=chosen.cell,
                    mode=chosen.mode,
                    listener=tracks.listener,
                    cells=self.cells,
                    seam_db_of=seam[mine],
                    trail_s_of=trail[mine],
                    early=self.early[name],
                    alignment_gain=self.assets.pack_gain,
                )
                source = recipe.source(name)
                writer.add_source(
                    Source(
                        id=name,
                        kind=source.kind,
                        subtype=source.subtype or "",
                        position=track.position,
                        yaw_deg=track.yaw_deg.astype(np.float32),
                        audible=track.audible,
                        early=Early(**self.early[name].pack()),
                        tail=Tail(**self.tail[name]),
                        low=Low(
                            ir=ir,
                            pair_position=tracks.positions[
                                [self.pairs[j][0] for j in mine]
                            ].reshape(-1, 3),
                            pair_cell=np.array([self.pairs[j][1] for j in mine], dtype=np.int32),
                            pair_key=np.array([self.pair_key[j] for j in mine], dtype="S64"),
                            seam_db=seam[mine].astype(np.float32),
                            onset_s=onset[mine],
                            pair=pair,
                            position_weight=track.weight.astype(np.float32),
                            cell=chosen.cell,
                            mode=chosen.mode,
                            slot_pair=slot_pair,
                            slot_weight=track.rail_weight,
                            slot_knots_hz=tracks.rail_knots_hz,
                        ),
                        level=Level(
                            high_gain_db=high_gain_db,
                            onset_s=onset_s,
                            band_gain_db=band_levels(
                                high_gain_db, track.audible, base_db + constant, shares
                            ),
                        ),
                        directivity_model=source.directivity.model,
                        directivity_enabled=bool(source.directivity.enabled),
                        gain_db=float(source.gain_db),
                        tail_seed=tail_seed(recipe.seed, name),
                    )
                )
        partial.replace(target)
        self.report["pack"] = {"path": str(target), "bytes": target.stat().st_size}
        self.report["as_computed"] = write_as_computed(self)
        return target

    def check(self) -> None:
        """The pack read back; in full, the engine on the host against the engine on the card.

        :data:`CHECK_READ` reads the pack's structure and stops: the whole
        scene's check. :data:`CHECK_FULL` reads every dataset, renders the
        first minute on both array modules, traces a source's early part
        on both and brings some ``low/ir`` rows through both: a smoke run's.
        """
        full = self.check_mode == CHECK_FULL
        t0 = time.time()
        with read_pack(self.out / "pack.h5", deep=full) as pack:
            self.report["read_back"] = {
                "sources": len(pack.sources),
                "steps": pack.header.steps,
                "cells": int(pack.cells.position.shape[0]),
                "deep": full,
            }
        self._spent("check", "read_back", t0)
        self.report["check"] = self.check_mode
        if not full:
            self.journal.say("check: the pack's structure read; no render (check: read)")
            return
        t0 = time.time()
        self.report["render"] = render_check(
            self.out / "pack.h5",
            seconds=CHECK_SECONDS,
            card=self.xp is not np,
            sources=CHECK_SOURCES,
        )
        self._spent("check", "render", t0)
        if self.report["render"].get("passed") is False:
            # Said, not raised: the pack is written, and a failed run keeps its machine.
            self.journal.say("check: THE ENGINE'S TWO MODULES DO NOT PASS V4; see the report")
        if self.xp is not np and self.pairs:
            assert self.engine is not None
            atmosphere = Atmosphere(**self.recipe.atmosphere.to_dict())
            worst = 0.0
            some = list(range(0, len(self.pairs), max(1, len(self.pairs) // 8)))[:8]
            for j in some:
                cached = at_pack_length(self.engine.cache.read(self.pair_key[j]))
                c = self.assets.settings.sound_speed_m_s
                on_host, on_card = (
                    pair_low(cached, self.crossover, atmosphere, sound_speed_m_s=c, xp=xp)[0]
                    for xp in (np, self.xp)
                )
                peak = float(np.abs(on_host).max())
                if peak > 0.0:
                    worst = max(worst, float(np.abs(on_host - on_card).max()) / peak)
            self.report["low_host_against_card"] = {
                "pairs": len(some),
                "max_over_peak": worst,
                "tolerance": RENDER_TOLERANCE,
                "passed": bool(worst <= RENDER_TOLERANCE),
            }
        if self.xp is not np and self.tracks.sources:
            name, track = next(iter(self.tracks.sources.items()))
            heard = np.flatnonzero(track.audible)[:400]
            if heard.size:
                host = trace_early(self.ms, track.position[heard], self.tracks.listener[heard])
                card = trace_early(
                    self.ms, track.position[heard], self.tracks.listener[heard], xp=self.xp
                )
                a, b = host.pack(), card.pack()
                same = all(
                    np.array_equal(a[k], b[k]) for k in ("offsets", "path_id", "order", "kind")
                )
                self.report["early_host_against_card"] = {
                    "source": name,
                    "steps": int(heard.size),
                    "same_paths": bool(same),
                    "delay_s": float(np.abs(a["delay_s"] - b["delay_s"]).max(initial=0.0))
                    if same
                    else None,
                    "gain": float(np.abs(a["gain"] - b["gain"]).max(initial=0.0)) if same else None,
                }
        self.journal.say(f"check: render {self.report['render']}")

    # ---- the queue ---------------------------------------------------------------------------

    def run_job(self, stage: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """One job of the queue, in whichever process holds this trace."""
        self._restored()
        if stage == "solve":
            engine: Any = self.engine
            return dict(engine.run_launch(payload))
        if stage == "paths":
            return self._trace_block(**payload)
        if stage == "rays":
            return self._ray_site(payload["position"])
        if stage == "tails":
            return self._tail_job(payload["source"])
        if stage == "level":
            return self._level_block(**payload)
        if stage == "rows":
            return self._rows_block(**payload)
        raise ValueError(f"no stage of a trace is named {stage!r}")

    def free_bytes(self) -> float:
        """What this process's card has free now; 0 without one."""
        if self.xp is np:
            return 0.0
        self.xp.get_default_memory_pool().free_all_blocks()
        # With what the engine keeps there for the run: a launch's bytes count it too.
        kept = getattr(self.engine, "kept_bytes", None)
        return card_free_bytes(self.xp) + (float(kept()) if kept else 0.0)

    def _restored(self) -> None:
        """In a worker's process: the cell rule again, from the centres the run wrote."""
        if self._assigned:
            return
        centres = json.loads((self.out / "state" / "centres.json").read_text())
        self.assign([None if c is None else np.asarray(c, dtype=float) for c in centres])

    def _pool(self, machine: Machine) -> Pool:
        """The queue on ``machine``: processes where a worker can be built, this one otherwise."""
        say = self.journal.say
        in_processes = self.engine_told is not None and self.workers != 0
        specs = machine.workers(self.workers) if (in_processes or self.workers) else None
        if not in_processes:
            return Pool(specs or [WorkerSpec(0)], inline=self, say=say, faults=self.faults)
        assert specs is not None
        make = functools.partial(
            make_worker,
            bundle=str(self.bundle),
            out=str(self.out),
            engine_told=dict(self.engine_told or {}),
            pffdtd_dir=str(self.pffdtd_dir),
        )
        return Pool(specs, make, say=say, faults=self.faults, progress=self._progress)

    def _alone(self, jobs: list[Job]) -> None:
        """A stage by itself, in this process: what a test of one stage calls."""
        self.pool = Pool([WorkerSpec(0)], inline=self, say=self.journal.say, faults=self.faults)
        self.pool.add(*jobs)
        self.pool.run()

    def _progress(self, said: dict[str, Any]) -> None:
        now = time.time()
        if now - getattr(self, "_told", 0.0) < 5.0:
            return
        self._told = now
        self.journal.set_status(job=f"{said['done']}/{said['jobs']}", last=said["stage"])

    def work(self) -> None:
        """Every stage between the cell rule and the pack, as the jobs of one queue.

        The workers start first and measure what they hold while this
        process plans the launches; the prediction is said, and held
        against ``max_hours``, before the first long job is handed out.
        """
        journal = self.journal
        if self.xp is not np:
            # What this process took of its card to voxelise goes back before it is read.
            self.xp.get_default_memory_pool().free_all_blocks()
        machine = self.machine or Machine.detect(
            gpu=None if self.xp is not np else False, devices=self.card_devices
        )
        journal.set_status(stage="work", stage_started=time.time())
        self.pool = self._pool(machine)
        started = time.time()
        try:
            self.pool.start()
            jobs = self.solve_jobs(machine)
            jobs += self.paths_jobs()
            jobs += self.rays_jobs()
            jobs += self.level_jobs(jobs)
            jobs += self.rows_jobs(jobs)
            if machine.cards and any(job.on == THREAD for job in jobs):
                # An engine that solves in one call of its own takes every card while it
                # does: what else needs a card waits for it.
                for job in jobs:
                    if job.on == CARD:
                        job.after = (*job.after, "solve/all")
            machine = machine.measured(self.pool.said())
            self._predicted(machine, jobs)
            self.pool.add(*jobs)
            self.pool.run()
        finally:
            self.pool.close()
        self._paths_report()
        account = self.pool.account()
        self.report["machine"] = machine.record()
        self.report["pool"] = account
        # A line a job: which worker, when, how long. What a scaling is read from.
        (self.out / "pool.json").write_text(
            json.dumps({**account, "ledger": self.pool.ledger}, indent=1)
        )
        for stage, held in account["stages"].items():
            name = "rays" if stage == "tails" else stage
            journal.timings[name] = round(journal.timings.get(name, 0.0) + held["wall_s"], 1)
            self.seconds.setdefault(name, {})[f"{stage}_work"] = held["work_s"]
        for name in ("solve", "paths", "rays", "level"):
            journal.timings.setdefault(name, 0.0)
        journal.timings["work"] = round(time.time() - started, 1)
        journal.say(
            "work: done in "
            f"{(time.time() - started) / 60:.1f} min; "
            + ", ".join(
                f"{stage} {held['jobs']} job(s) {held['work_s']:.0f} s of work in"
                f" {held['wall_s']:.0f} s"
                for stage, held in account["stages"].items()
            )
        )

    def _predicted(self, machine: Machine, jobs: list[Job]) -> None:
        """The run's wall time and price on this machine, said and held against ``max_hours``."""
        from reverberate.wave.lowband.pairs import PAIR_S

        heard = sum(int(np.count_nonzero(t.audible)) for t in self.tracks.sources.values())
        count = {stage: sum(1 for j in jobs if j.stage == stage) for stage in ("rays", "level")}
        to_level = sum(len(j.payload.get("pairs", ())) for j in jobs if j.stage == "level")
        counts = {
            "node_updates": self.planned.get("node_updates", 0.0),
            "pairs": self.planned.get("pairs", 0.0),
            "launches": sum(1 for j in jobs if j.stage == "solve" and j.on != PARENT),
            "pair_s": PAIR_S if self.planned.get("node_updates") else 0.0,
            "sites": count["rays"],
            "site_s": (RAYS_SITE_S + int(self.tail_rows.size) * RAYS_SITE_CELL_S)
            * (self.assets.settings.rays.rays / RAYS_MEASURED),
            "paths_jobs": heard + len(self.pairs),
            "level_pairs": to_level,
            "row_pairs": len(self.pairs),
            "write_s": 0.010 * len(self.pairs),
        }
        rate = self.rate_usd_per_hour
        if rate is None:
            rate = dict(self.spec.get("estimate") or {}).get("billed_rate_usd_per_hour")
        host = sum(1 for w in self.pool.workers if w.spec.card is None) or 1
        said = predict(counts, machine, host_workers=host, rate_usd_per_hour=rate)
        self.report["predicted"] = {"counts": counts, **said}
        partial = self.out / "prediction.partial.json"
        partial.write_text(json.dumps({"machine": machine.record(), **self.report["predicted"]}))
        partial.replace(self.out / "prediction.json")
        self.journal.say(
            f"machine: {len(machine.cards)} card(s), {machine.cores} core(s),"
            f" {machine.ram_bytes / 1e9:.0f} GB; {json.dumps(machine.record()['cards'])}"
        )
        self.journal.say(
            f"predicted: {said['hours']:.2f} h of work on this machine"
            + (f", {said['usd']:.2f} USD at {rate:g} USD/h" if rate is not None else "")
            + ("" if said["calibrated"] else " (the solve's rate is the box's, not calibrated)")
            + f"; {json.dumps(said['seconds'])}"
        )
        # Held on the seconds, not on the hours as they are printed: a second and a half
        # of host's work is 0.000 h, and is still over an allowance it exceeds.
        over = self.max_hours is not None and said["wall_s"] > float(self.max_hours) * 3600.0
        if over and not said["calibrated"] and counts["node_updates"]:
            # A rate that is not the grid's own does not stop a rented machine.
            self.journal.say(
                f"predicted over the {float(self.max_hours or 0.0):g} h allowed, on a rate that"
                " is not calibrated: said, and the run goes on"
            )
        elif over:
            raise RuntimeError(
                f"the work is predicted to take {said['hours']:.2f} h on this machine, over the"
                f" {float(self.max_hours or 0.0):g} h it is allowed: stopped before it started"
            )

    # ---- the run ---------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Every stage; ``campaign.done`` or ``campaign.failed`` at the end."""
        journal = self.journal
        for marker in ("campaign.done", "campaign.failed"):
            (self.out / marker).unlink(missing_ok=True)
        try:
            journal.say(
                f"trace {self.recipe.dwelling.name}: {self.tracks.steps} steps,"
                f" {len(self.tracks.sources)} source(s), {self.asked.shape[0]} cell(s),"
                f" profile {self.profile.record()}"
            )
            self.check_assets()
            assert self.engine is not None
            centres = self.engine.place()
            journal.stage("assign", lambda: self.assign(centres))
            self.work()
            journal.stage("write", self.write)
            journal.stage("check", self.check)
            # The jobs' own files are in the pack and in the caches a rerun reads (the
            # early tables, the histograms, the ledger): not kept, and not sent home.
            self._rows_open = None
            shutil.rmtree(self.out / "jobs", ignore_errors=True)
            self.report.update(
                {
                    "dwelling": self.recipe.dwelling.name,
                    "recipe_sha256": hashlib.sha256(self.recipe_bytes).hexdigest(),
                    "device": journal.status.get("device", {}),
                    "timings_s": dict(journal.timings),
                    "seconds": {stage: dict(parts) for stage, parts in self.seconds.items()},
                    "total_s": round(time.time() - journal.started, 1),
                    "pairs_engine": getattr(self.engine, "report", {}),
                }
            )
            (self.out / "trace_report.json").write_text(
                json.dumps(self.report, indent=1, default=str)
            )
            journal.set_status(stage="done")
            (self.out / "campaign.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            journal.say(f"trace complete in {(time.time() - journal.started) / 3600:.2f} h")
            return self.report
        except BaseException as error:
            journal.set_status(stage="failed", error=repr(error)[:500])
            (self.out / "campaign.failed").write_text(repr(error)[:2000])
            journal.say(f"trace FAILED: {error!r}"[:600])
            raise


def _end_db(omni: np.ndarray) -> float:
    """The energy of a response's last 50 ms over that of the whole, dB; -200 when silent."""
    samples = int(round(0.05 * LOW_RATE_HZ))
    whole = float(np.sum(np.asarray(omni, dtype=float) ** 2))
    if whole <= 0.0:
        return -200.0
    last = float(np.sum(np.asarray(omni[-samples:], dtype=float) ** 2))
    return round(float(10.0 * np.log10(max(last / whole, 1e-20))), 1)


def _spread(values: np.ndarray, digits: int = 3) -> dict[str, float] | None:
    if not values.size:
        return None
    return {
        "p10": round(float(np.percentile(values, 10)), digits),
        "median": round(float(np.median(values)), digits),
        "p90": round(float(np.percentile(values, 90)), digits),
        "worst": round(float(values[np.argmax(np.abs(values))]), digits),
    }


def render_check(
    path: Path,
    *,
    seconds: float | None = None,
    card: bool | None = None,
    sources: int | None = None,
) -> dict[str, Any]:
    """The pack rendered by the signal engine on ``numpy`` and, with a card, on ``cupy``.

    Dry noise at every source, the first ``seconds`` of the scene, block by
    block; with ``sources``, at that many of them, those heard over most
    of those seconds. Without a card the render is only shown to be finite
    and not silent; with one the two must agree to
    :data:`RENDER_TOLERANCE` of the peak (V4 of the plan).

    **The record says which device computed**, so that an agreement is not
    one module rendered twice: the module each engine holds, the module of
    the arrays each engine's renderer returns before they are brought to
    the host, the card's name, and the bytes the card's pool held at its
    fullest during the render. Two transforms of different libraries do not
    agree to the last bit: a difference of exactly zero is reported
    (``identical``) and does not pass. ``card=True`` insists on a card and
    raises without one, as a trace that ran on a card does; ``None`` asks
    the machine.
    """
    from reverberate.compute import array_module_name, cuda_available, device_report, xp_for
    from reverberate.render.engine import Engine

    with_card = cuda_available() if card is None else bool(card)
    if with_card:
        xp_for(True)  # raises when the card that was promised is not there

    with read_pack(path) as pack:
        h = pack.header
        samples = h.samples
        if seconds is not None:
            samples = min(samples, int(round(seconds / h.step_s)) * h.step_samples)
        steps = samples // h.step_samples
        heard = {
            name: int(np.count_nonzero(np.asarray(source.audible)[:steps]))
            for name, source in pack.sources.items()
        }
        chosen = sorted(heard, key=lambda name: -heard[name])
        if sources is not None:
            chosen = chosen[:sources]
        dry = {
            name: np.random.default_rng(source.tail_seed % 2**32).standard_normal(samples)
            for name, source in pack.sources.items()
            if name in chosen
        }
        host = Engine(pack, dry, gpu=False)
        on_card = Engine(pack, dry, gpu=True) if with_card else None
        first = next(iter(dry), None)
        pool = None
        if on_card is not None:
            pool = on_card.xp.get_default_memory_pool()
        held_most = 0
        block = int(round(CHECK_BLOCK_S / h.step_s)) * h.step_samples
        peak = 0.0
        worst = 0.0
        finite = True
        seconds_host = seconds_card = 0.0
        for start in range(0, samples, block):
            stop = min(start + block, samples)
            t0 = time.time()
            a = host.render(start, stop, sources=list(dry))
            seconds_host += time.time() - t0
            finite = finite and bool(np.all(np.isfinite(a)))
            peak = max(peak, float(np.abs(a).max()))
            if on_card is not None:
                t0 = time.time()
                b = to_numpy(on_card.render(start, stop, sources=list(dry)))
                seconds_card += time.time() - t0
                worst = max(worst, float(np.abs(a - b).max()))
                if pool is not None:
                    held_most = max(held_most, int(pool.total_bytes()))
        record: dict[str, Any] = {
            "seconds": samples / h.sample_rate_hz,
            "sources": len(dry),
            "rendered": list(dry),
            "audible_s": round(sum(heard[name] for name in dry) * h.step_s, 2),
            "finite": finite,
            "peak": peak,
            "host_s": round(seconds_host, 2),
            "card": on_card is not None,
            "host_module": host.xp.__name__,
        }
        if first is not None and samples:
            record["host_arrays"] = array_module_name(host.source(first).render(0, h.step_samples))
        if on_card is not None:
            record["card_s"] = round(seconds_card, 2)
            record["card_module"] = on_card.xp.__name__
            if first is not None and samples:
                record["card_arrays"] = array_module_name(
                    on_card.source(first).render(0, h.step_samples)
                )
            record["device"] = device_report().get("gpu")
            record["card_pool_bytes"] = held_most
            record["max_over_peak"] = worst / peak if peak > 0.0 else None
            record["identical"] = bool(worst == 0.0)
            record["tolerance"] = RENDER_TOLERANCE
            on_two = (
                record["host_module"] == "numpy"
                and record["card_module"] == "cupy"
                and record.get("card_arrays", "cupy") == "cupy"
                and record.get("host_arrays", "numpy") == "numpy"
                and held_most > 0
            )
            record["computed_on_two_devices"] = bool(on_two)
            record["passed"] = bool(
                peak > 0.0 and on_two and 0.0 < worst <= RENDER_TOLERANCE * peak
            )
    return record


def run_trace(
    bundle: Path,
    out: Path,
    *,
    pffdtd_dir: Path = Path("/root/pffdtd"),
    devices: str | None = None,
    gpu: bool | None = None,
    solvers: int | None = None,
    engine: PairsEngine | None = None,
    check: str | None = None,
    workers: int | None = 0,
    engine_told: dict[str, Any] | None = None,
    max_hours: float | None = None,
) -> dict[str, Any]:
    return Trace(
        bundle=bundle,
        out=out,
        engine=engine,
        gpu=gpu,
        pffdtd_dir=pffdtd_dir,
        card_devices=devices,
        solvers=solvers,
        check_mode=check,
        workers=workers,
        engine_told=engine_told,
        max_hours=max_hours,
    ).run()
