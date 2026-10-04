"""The audit's stems: each source of a pack rendered by the signal engine, in chunks, and kept.

The owner validates a scene by ear, and what he hears has to be the output
of the engine training will use (:class:`reverberate.render.engine.Engine`),
at order 7 and 48 kHz. There is no second renderer anywhere: this module runs
that engine, one source at a time, and keeps what it gives.

**A stem is the engine's own output for one source**: ``Engine.stem`` cast to
float32, the frames ``docs/formats/scene-signal.md`` describes, in one file.
Nothing is decimated, truncated in order or compressed, so the page's stream
can be compared with the engine byte for byte. What keeps the disk sane is
that a stem is **sparse**: a chunk in which the engine returned nothing but
zeros is recorded and never written, and a source is silent about half of a
scene. (The file system keeps a hole only when it is large: on APFS a stem of
a few megabytes is allocated whole, one of a scene is not.)

**A chunk is one run of the engine** (``RenderSettings.chunk_steps`` steps,
half a second, 6.1 MB at order 7). The engine renders every run from the
pack and the dry signal alone, so runs are rendered in any order, by any
process, and a chunk of the cache is sample for sample the same range of a
one-shot render. The scheduler renders first the chunk under the play
cursor, for the sources that are heard, then ahead of it, then the sources
that are not heard (a solo is then instant), then the rest of the scene.

**Processes, one thread each.** The transforms of two engines in two
processes do not share cores, and ``workers = 1`` is part of what a stem is:
the engine's samples move in their last bits with the number of threads of
its transforms, so the cache fixes it. A one-shot ``Engine`` with the same
settings gives the same bytes.

**On disk**, under the cache folder::

    <pack sha256, 16 hex>/<source id>/<key, 16 hex>/
        key.json      what the key is the digest of
        stem.f32      the frames, sparse, at their place: frame n at byte 256 n
        chunks.jsonl  one line per chunk rendered: its SHA-256, peak and levels
        stem.json     the scene-signal header, once every chunk is there

The key is the digest of the pack's bytes, the source, the engine's code,
the settings that change samples, whether the source's directivity is
applied, and what the source was fed. Changing the directivity switch moves
the key of the sources it concerns and of no other. A chunk is in the cache
when its line is in ``chunks.jsonl``; the line is written after the bytes,
by the one process that owns the journal, so a render that stopped loses at
most the chunks in flight.

``python -m reverberate.viz.audit_stems <pack.h5>`` renders every stem of a
pack without a page, so that the first listen waits for nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import resource
import shutil
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.output import SCHEMA as SIGNAL_SCHEMA
from reverberate.render.output import SCHEMA_VERSION as SIGNAL_SCHEMA_VERSION
from reverberate.render.pack import ScenePack, read_pack
from reverberate.render.translate import SpatialTranslation
from reverberate.settings import data_root
from reverberate.viz.audit_dry import DryPlan, DrySources, dry_plan, dry_track

__all__ = [
    "ENGINE_SOURCES",
    "AuditSettings",
    "Session",
    "Stem",
    "StemService",
    "applies_directivity",
    "engine_digest",
    "file_sha256",
    "render_chunk",
    "stem_key",
]

#: The modules whose source is the engine's version: a change to any of them
#: is a change to what a stem holds.
ENGINE_SOURCES = (
    "render/engine.py",
    "render/early.py",
    "render/delay.py",
    "render/low.py",
    "render/tail.py",
    "render/noise.py",
    "render/dry.py",
    "render/translate.py",
    "render/pack.py",
    "spatial/sh.py",
    "spatial/translate.py",
    "spatial/lowband.py",
    "mirror/hybrid.py",
    "mirror/directivity.py",
    "audio.py",
    "viz/audit_dry.py",
)

#: Worker processes unless told otherwise. Measured on the laptop the audit runs on
#: (ten cores, four of them fast): a pack at rest renders 3.3 chunks a second with
#: one process, 7 with six and 6.3 with nine; a dense moving pack, whose tail fills
#: the memory bus from one core, renders 0.8 a second with any number.
DEFAULT_WORKERS = 6
#: Chunks ahead of the cursor rendered before anything else, for every source.
WINDOW_CHUNKS = 60
#: A session nobody asked about for this long is not rendered any further.
IDLE_S = 20.0
#: Rendering stops when the cache's disk has less than this free.
RESERVE_BYTES = 5 << 30
#: Engines a worker keeps, one per source: the rest are rebuilt on demand.
ENGINES_HELD = 6
#: Over how long the render rate is averaged, seconds.
RATE_WINDOW_S = 20.0
#: The level of a step in which the engine returned zeros.
SILENCE_DB = -200.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_digest() -> str:
    """A digest of the source of every module in :data:`ENGINE_SOURCES`."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ENGINE_SOURCES:
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AuditSettings:
    """What the audit chooses of a render: the owner's switch and nothing else."""

    #: ``None`` renders each source as the pack says, ``False`` every source
    #: omnidirectional, ``True`` every source with its model.
    directivity: bool | None = None

    def render(self) -> RenderSettings:
        return RenderSettings(directivity=self.directivity, workers=1, chunks_held=1)


def applies_directivity(pack: ScenePack, source_id: str, directivity: bool | None) -> bool:
    """Whether the engine applies ``source_id``'s model under the switch: its own rule."""
    source = pack.sources[source_id]
    wanted = source.directivity_enabled if directivity is None else directivity
    return bool(wanted and source.directivity_model in pack.directivity)


def stem_key(
    pack_sha256: str, pack: ScenePack, source_id: str, settings: AuditSettings, plan: DryPlan
) -> dict[str, Any]:
    """Everything a stem's bytes depend on; its digest names the stem's folder."""
    render = settings.render().record()
    # Neither changes a sample: one is replaced by what it does to this source,
    # the other is how many runs the engine holds in memory.
    del render["directivity"], render["chunks_held"]
    return {
        "pack_sha256": pack_sha256,
        "source": source_id,
        "engine": engine_digest(),
        "settings": render,
        "directivity": applies_directivity(pack, source_id, settings.directivity),
        "dry": plan.digest,
        "dtype": "<f4",
        "layout": "frames of channels",
    }


def _key_digest(key: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


class Stem:
    """One source's stem on disk: which chunks are there, and their bytes."""

    def __init__(
        self,
        folder: Path,
        key: Mapping[str, Any],
        *,
        channels: int,
        samples: int,
        chunk_samples: int,
        sample_rate_hz: float,
        order: int,
        recipe_sha256: str,
    ) -> None:
        self.folder = folder
        self.key = dict(key)
        self.channels, self.samples, self.chunk_samples = channels, samples, chunk_samples
        self.sample_rate_hz, self.order, self.recipe_sha256 = sample_rate_hz, order, recipe_sha256
        self.chunks = -(-samples // chunk_samples)
        self.done = np.zeros(self.chunks, dtype=bool)
        self.busy = np.zeros(self.chunks, dtype=bool)
        self.failed: dict[int, str] = {}
        self.records: dict[int, dict[str, Any]] = {}
        folder.mkdir(parents=True, exist_ok=True)
        key_path = folder / "key.json"
        if not key_path.is_file():
            key_path.write_text(json.dumps(self.key, indent=2, sort_keys=True) + "\n")
        self.data_path = folder / "stem.f32"
        self.journal_path = folder / "chunks.jsonl"
        size = samples * channels * 4
        if not self.data_path.is_file() or self.data_path.stat().st_size != size:
            # Its full length from the start, and nothing in it: the holes read as zeros.
            with open(self.data_path, "ab") as handle:
                handle.truncate(size)
            self.journal_path.unlink(missing_ok=True)
        if self.journal_path.is_file():
            for line in self.journal_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                    index = int(record["chunk"])
                except (ValueError, KeyError, TypeError):
                    continue  # a line cut short by a stop: the chunk is rendered again
                if 0 <= index < self.chunks and len(record.get("sha256", "")) == 64:
                    self.records[index] = record
                    self.done[index] = True

    @property
    def id(self) -> str:
        return self.folder.name

    @property
    def complete(self) -> bool:
        return bool(self.done.all())

    def span(self, index: int) -> tuple[int, int]:
        """The samples of chunk ``index``; the last chunk stops with the scene."""
        if not 0 <= index < self.chunks:
            raise IndexError(f"chunk {index} is not one of the stem's {self.chunks}")
        start = index * self.chunk_samples
        return start, min(start + self.chunk_samples, self.samples)

    def read(self, index: int) -> np.ndarray:
        """Chunk ``index`` as ``[frame, channel]`` float32, the bytes the engine gave."""
        start, stop = self.span(index)
        if not self.done[index]:
            raise KeyError(f"chunk {index} of {self.key['source']!r} is not rendered")
        if self.records[index].get("silent"):
            return np.zeros((stop - start, self.channels), dtype="<f4")
        with open(self.data_path, "rb") as handle:
            handle.seek(start * self.channels * 4)
            frames = np.fromfile(handle, dtype="<f4", count=(stop - start) * self.channels)
        return frames.reshape(stop - start, self.channels)

    def finish(self, record: Mapping[str, Any]) -> None:
        """A chunk a worker wrote: its line in the journal, and the header when it was the last."""
        index = int(record["chunk"])
        with open(self.journal_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records[index] = dict(record)
        self.done[index] = True
        self.busy[index] = False

    def seal(self) -> dict[str, Any]:
        """Write the scene-signal header of a complete stem: ``open_signal`` then reads it."""
        if not self.complete:
            raise ValueError("a stem is sealed once every chunk is rendered")
        header = {
            "schema": SIGNAL_SCHEMA,
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "data": self.data_path.name,
            "dtype": "<f4",
            "layout": "frames of channels",
            "sample_rate_hz": self.sample_rate_hz,
            "order": self.order,
            "channels": self.channels,
            "frames": self.samples,
            "ordering": "ACN",
            "normalisation": "N3D",
            "frame": "scene, fixed: the head's rotation is applied at decode",
            "recipe_sha256": self.recipe_sha256,
            "sources": [self.key["source"]],
            "peak": max((float(r["peak"]) for r in self.records.values()), default=0.0),
            "sha256": file_sha256(self.data_path),
            "complete": True,
            "stem_key": self.key,
        }
        scratch = self.folder / "stem.json.part"
        scratch.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, self.folder / "stem.json")
        return header

    def ranges(self, mask: np.ndarray | None = None) -> list[list[int]]:
        """The chunks rendered (or those of ``mask``) as ``[first, stop)`` runs."""
        flags = self.done if mask is None else mask
        edges = np.flatnonzero(np.diff(np.concatenate([[False], flags, [False]]).astype(np.int8)))
        return [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2], strict=True)]

    def disk_bytes(self) -> int:
        """What the stem really takes: a hole takes nothing."""
        return int(self.data_path.stat().st_blocks) * 512


# --------------------------------------------------------------------------
# the render of one chunk, in whichever process
# --------------------------------------------------------------------------


class _Engines:
    """What a worker keeps between chunks: the open packs, and an engine per source."""

    def __init__(self) -> None:
        self.packs: dict[str, tuple[ScenePack, SpatialTranslation]] = {}
        self.engines: OrderedDict[tuple[str, str, str], Engine] = OrderedDict()

    def engine(self, task: Mapping[str, Any]) -> Engine:
        path, source = str(task["pack"]), str(task["source"])
        name = (path, source, json.dumps(task["settings"], sort_keys=True))
        if name in self.engines:
            self.engines.move_to_end(name)
            return self.engines[name]
        if path not in self.packs:
            # Validated once by whoever opened the session, not by every worker.
            pack = read_pack(Path(path), check=False)
            h = pack.header
            self.packs[path] = (
                pack,
                SpatialTranslation.from_fusion(h.fusion, h.order, h.sound_speed_m_s),
            )
        pack, translation = self.packs[path]
        plan = dry_plan(pack, source, DrySources.from_record(task["dry"]))
        if plan.digest != task["dry_digest"]:
            raise ValueError(f"the dry audio of {source!r} changed since the stem was keyed")
        rate = pack.header.sample_rate_hz
        engine = Engine(
            pack,
            {source: dry_track(plan, rate)},
            settings=RenderSettings(**task["settings"]),
            translation=translation,
        )
        self.engines[name] = engine
        while len(self.engines) > ENGINES_HELD:
            self.engines.popitem(last=False)
        return engine

    def close(self) -> None:
        self.engines.clear()
        for pack, _ in self.packs.values():
            pack.close()
        self.packs.clear()


def render_chunk(task: Mapping[str, Any], engines: _Engines) -> dict[str, Any]:
    """Render one chunk with the engine, write its bytes at their place, and describe them.

    The SHA-256 is of the engine's samples as they left it, taken before
    anything is written: it is what the page's stream is held to.
    """
    started = time.perf_counter()
    engine = engines.engine(task)
    start, stop = int(task["start"]), int(task["stop"])
    stem = engine.stem(str(task["source"]), start, stop)
    frames = np.ascontiguousarray(stem.T.astype("<f4"))
    payload = frames.tobytes()
    peak = float(np.max(np.abs(frames))) if frames.size else 0.0
    silent = peak == 0.0
    if not silent:
        fd = os.open(str(task["data"]), os.O_WRONLY)
        try:
            os.pwrite(fd, payload, start * frames.shape[1] * 4)
        finally:
            os.close(fd)
    step = int(task["step_samples"])
    omni = frames[:, 0].astype(np.float64)
    levels = []
    for first in range(0, omni.size, step):
        power = float(np.mean(omni[first : first + step] ** 2))
        levels.append(round(10.0 * np.log10(power), 1) if power > 0.0 else SILENCE_DB)
    return {
        "chunk": int(task["chunk"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frames": int(frames.shape[0]),
        "peak": peak,
        "silent": silent,
        "levels_db": levels,
        "seconds": round(time.perf_counter() - started, 4),
        "rss_mb": round(_rss_bytes() / 1e6),
    }


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def _worker(connection: Any) -> None:
    """A worker process: chunks in, their descriptions out, until told to stop."""
    engines = _Engines()
    while True:
        try:
            task = connection.recv()
        except (EOFError, OSError):
            break
        if task is None:
            break
        try:
            connection.send(("done", render_chunk(task, engines)))
        except Exception as error:  # noqa: BLE001
            connection.send(("failed", f"{type(error).__name__}: {error}"))
    engines.close()


# --------------------------------------------------------------------------
# a pack being listened to
# --------------------------------------------------------------------------


class Session:
    """One pack under one choice of settings: its stems, and where the listener is."""

    def __init__(
        self,
        pack_path: Path,
        pack_sha256: str,
        pack: ScenePack,
        settings: AuditSettings,
        dry: DrySources,
        stems: dict[str, Stem],
        plans: dict[str, DryPlan],
    ) -> None:
        self.pack_path, self.pack_sha256, self.pack = pack_path, pack_sha256, pack
        self.settings, self.dry, self.stems, self.plans = settings, dry, stems, plans
        self.order = list(pack.sources)
        first = next(iter(stems.values()))
        self.chunks, self.chunk_samples = first.chunks, first.chunk_samples
        self.cursor = 0
        self.audible: list[str] = list(self.order)
        self.background = True
        # Asleep until somebody says where the listener is: nothing is rendered unasked.
        self.touched = float("-inf")

    def task(self, source_id: str, index: int) -> dict[str, Any]:
        stem = self.stems[source_id]
        start, stop = stem.span(index)
        return {
            "pack": str(self.pack_path),
            "source": source_id,
            "settings": self.settings.render().record(),
            "dry": self.dry.record(),
            "dry_digest": self.plans[source_id].digest,
            "chunk": index,
            "start": start,
            "stop": stop,
            "data": str(stem.data_path),
            "step_samples": self.pack.header.step_samples,
        }

    def pick(self, built: Iterable[str] = ()) -> tuple[str, int] | None:
        """The chunk to render next: see the module for the order."""
        cursor = min(max(self.cursor, 0), self.chunks - 1)
        window = min(self.chunks, cursor + WINDOW_CHUNKS)
        heard = [s for s in self.order if s in self.audible]
        others = [s for s in self.order if s not in self.audible]
        known = set(built)
        tiers = [(heard, cursor, window), (others, cursor, window)]
        if self.background:
            tiers += [(heard + others, window, self.chunks), (heard + others, 0, cursor)]
        for names, lo, hi in tiers:
            found: list[tuple[int, int, str]] = []
            for rank, name in enumerate(names):
                stem = self.stems[name]
                free = np.flatnonzero(~(stem.done[lo:hi] | stem.busy[lo:hi]))
                free = free[[i + lo not in stem.failed for i in free]] if stem.failed else free
                if free.size:
                    found.append((int(free[0]) + lo, rank, name))
            if found:
                earliest = min(found)[0]
                # Of the sources waiting at that chunk, one this worker has an engine for.
                tied = [f for f in found if f[0] == earliest]
                index, _, name = min(tied, key=lambda f: (f[2] not in known, f[1]))
                return name, index
        return None

    def ready(self, sources: Iterable[str], index: int) -> bool:
        return all(bool(self.stems[s].done[index]) for s in sources)

    def failure(self, sources: Iterable[str], index: int) -> str | None:
        for name in sources:
            if index in self.stems[name].failed:
                return f"{name}: {self.stems[name].failed[index]}"
        return None

    def mix(self, sources: Iterable[str], index: int) -> np.ndarray:
        """The chosen sources' stems summed, ``[frame, channel]`` float32.

        One source is its stem, untouched. Several are summed in float64 in
        the pack's order and brought back to float32, which is the engine's
        mix to the rounding of a stem to float32 (``scene-signal.md``).
        """
        chosen = [s for s in self.order if s in set(sources)]
        start, stop = next(iter(self.stems.values())).span(index)
        channels = self.pack.header.channels
        if not chosen:
            return np.zeros((stop - start, channels), dtype="<f4")
        if len(chosen) == 1:
            return self.stems[chosen[0]].read(index)
        total = np.zeros((stop - start, channels))
        for name in chosen:
            total += self.stems[name].read(index)
        return total.astype("<f4")

    def levels(self, sources: Iterable[str], index: int) -> dict[str, list[float]]:
        return {
            s: list(self.stems[s].records[index]["levels_db"])
            for s in sources
            if self.stems[s].done[index]
        }


class StemService:
    """The stems of every pack opened, and the workers that render them."""

    def __init__(
        self,
        cache_root: Path,
        *,
        workers: int | None = None,
        processes: bool = True,
        dry: DrySources | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.dry = dry or DrySources()
        count = workers if workers is not None else min(DEFAULT_WORKERS, (os.cpu_count() or 2) - 1)
        self.workers = max(1, count)
        self.processes = processes
        self._lock = threading.Condition()
        self._sessions: OrderedDict[tuple[str, bool | None], Session] = OrderedDict()
        self._stems: dict[Path, Stem] = {}
        self._digests: dict[str, str] = {}
        self._threads: list[threading.Thread] = []
        self._stopping = False
        self._rendered: deque[tuple[float, float]] = deque()
        self._costs: deque[float] = deque(maxlen=200)
        self._rss_mb = 0
        self.blocked: str | None = None

    # -- packs --------------------------------------------------------------

    def pack_sha256(self, path: Path) -> str:
        """The digest of a pack's bytes, remembered by path, size and time of change."""
        path = Path(path).resolve()
        stat = path.stat()
        name = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        if name in self._digests:
            return self._digests[name]
        memo = self.cache_root / "packs.json"
        known: dict[str, str] = {}
        if memo.is_file():
            try:
                known = json.loads(memo.read_text())
            except ValueError:
                known = {}
        if name not in known:
            known[name] = file_sha256(path)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            scratch = memo.with_suffix(".json.part")
            scratch.write_text(json.dumps(known, indent=1, sort_keys=True))
            os.replace(scratch, memo)
        self._digests[name] = known[name]
        return known[name]

    def open(self, pack_path: Path, settings: AuditSettings | None = None) -> Session:
        """The session of a pack under ``settings``, made on first use; rendering starts."""
        settings = settings or AuditSettings()
        pack_path = Path(pack_path).resolve()
        digest = self.pack_sha256(pack_path)
        name = (digest, settings.directivity)
        with self._lock:
            if name in self._sessions:
                session = self._sessions[name]
                session.touched = time.monotonic()
                self._sessions.move_to_end(name)
                return session
        pack = read_pack(pack_path)
        h = pack.header
        chunk_samples = settings.render().chunk_steps * h.step_samples
        stems: dict[str, Stem] = {}
        plans: dict[str, DryPlan] = {}
        with self._lock:
            for source_id in pack.sources:
                plans[source_id] = dry_plan(pack, source_id, self.dry)
                key = stem_key(digest, pack, source_id, settings, plans[source_id])
                folder = self.cache_root / digest[:16] / source_id / _key_digest(key)[:16]
                if folder not in self._stems:
                    self._stems[folder] = Stem(
                        folder,
                        key,
                        channels=h.channels,
                        samples=h.samples,
                        chunk_samples=chunk_samples,
                        sample_rate_hz=h.sample_rate_hz,
                        order=h.order,
                        recipe_sha256=h.recipe_sha256,
                    )
                stems[source_id] = self._stems[folder]
            session = Session(pack_path, digest, pack, settings, self.dry, stems, plans)
            self._sessions[name] = session
            self._start()
            self._lock.notify_all()
        return session

    def want(
        self,
        session: Session,
        *,
        cursor: int | None = None,
        audible: Iterable[str] | None = None,
        background: bool | None = None,
    ) -> None:
        """Where the listener is and what is heard: what is rendered next follows."""
        with self._lock:
            if cursor is not None:
                session.cursor = int(cursor)
            if audible is not None:
                session.audible = [s for s in session.order if s in set(audible)]
            if background is not None:
                session.background = bool(background)
            session.touched = time.monotonic()
            self._sessions.move_to_end((session.pack_sha256, session.settings.directivity))
            self._lock.notify_all()

    def wait(self, session: Session, sources: Iterable[str], index: int, timeout_s: float) -> bool:
        """Until chunk ``index`` of every one of ``sources`` is rendered; false on time out.

        Raises ``RuntimeError`` when the engine failed on one of them.
        """
        chosen = list(sources)
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while True:
                session.touched = time.monotonic()
                failure = session.failure(chosen, index)
                if failure:
                    raise RuntimeError(failure)
                if session.ready(chosen, index):
                    return True
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._lock.wait(min(left, 0.5))

    # -- the workers --------------------------------------------------------

    def start(self) -> None:
        """Start the workers before the first chunk is asked for: a process takes a second."""
        with self._lock:
            self._start()

    def _start(self) -> None:
        if self._threads or self._stopping:
            return
        for number in range(self.workers):
            thread = threading.Thread(target=self._drive, args=(number,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _next(self, built: set[str]) -> tuple[Session, str, int] | None:
        """Under the lock: the next chunk of the session last asked about, if it is awake."""
        if self._stopping or not self._sessions:
            return None
        session = next(reversed(self._sessions.values()))
        if time.monotonic() - session.touched > IDLE_S:
            return None
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.cache_root).free < RESERVE_BYTES:
            self.blocked = "the cache's disk has under 5 GB free"
            return None
        self.blocked = None
        found = session.pick(built)
        if found is None:
            return None
        name, index = found
        session.stems[name].busy[index] = True
        return session, name, index

    def _drive(self, number: int) -> None:
        """One worker's thread: hand it chunks, write their lines, until the service stops."""
        process: Any = None
        connection: Any = None
        local = _Engines()
        built: set[str] = set()

        def spawn() -> tuple[Any, Any]:
            context = multiprocessing.get_context("spawn")
            near, far = context.Pipe()
            made = context.Process(target=_worker, args=(far,), daemon=True)
            made.start()
            far.close()
            built.clear()
            return made, near

        try:
            if self.processes:
                process, connection = spawn()
            while True:
                with self._lock:
                    while True:
                        if self._stopping:
                            return
                        found = self._next(built)
                        if found is not None:
                            break
                        self._lock.wait(1.0)
                session, name, index = found
                task = session.task(name, index)
                stem = session.stems[name]
                try:
                    if not self.processes:
                        outcome: tuple[str, Any] = ("done", render_chunk(task, local))
                    else:
                        if process is None or not process.is_alive():
                            process, connection = spawn()
                        connection.send(task)
                        outcome = connection.recv()
                except Exception as error:  # noqa: BLE001
                    outcome = ("failed", f"{type(error).__name__}: {error}")
                    if process is not None:
                        process.kill()
                        process = None
                kind, result = outcome
                built.add(name)
                with self._lock:
                    if kind == "done":
                        stem.finish(result)
                        now = time.monotonic()
                        self._rendered.append((now, result["frames"] / stem.sample_rate_hz))
                        self._costs.append(
                            result["seconds"] / (result["frames"] / stem.sample_rate_hz)
                        )
                        self._rss_mb = max(self._rss_mb, int(result["rss_mb"]))
                        sealed = stem.complete
                    else:
                        stem.failed[index] = str(result)
                        stem.busy[index] = False
                        sealed = False
                    self._lock.notify_all()
                if sealed:
                    stem.seal()
        finally:
            local.close()
            if connection is not None:
                with contextlib.suppress(OSError, ValueError):
                    connection.send(None)
            if process is not None:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()

    def close(self) -> None:
        """Stop the workers; what is rendered stays on disk."""
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        for thread in self._threads:
            thread.join(timeout=10.0)
        self._threads.clear()
        with self._lock:
            for session in self._sessions.values():
                session.pack.close()
            self._sessions.clear()

    # -- what the page is told ------------------------------------------------

    def status(self, session: Session) -> dict[str, Any]:
        with self._lock:
            session.touched = time.monotonic()
            now = time.monotonic()
            while self._rendered and now - self._rendered[0][0] > RATE_WINDOW_S:
                self._rendered.popleft()
            span = min(RATE_WINDOW_S, now - self._rendered[0][0]) if self._rendered else 0.0
            rate = sum(s for _, s in self._rendered) / span if span > 1.0 else None
            sources = {}
            for name in session.order:
                stem = session.stems[name]
                sources[name] = {
                    "stem": stem.id,
                    "ready": stem.ranges(),
                    "busy": [int(i) for i in np.flatnonzero(stem.busy)],
                    "failed": {str(k): v for k, v in stem.failed.items()},
                    "complete": stem.complete,
                    "dry": session.plans[name].label,
                    "placeholder": session.plans[name].placeholder,
                    "directivity": bool(stem.key["directivity"]),
                    "disk_bytes": stem.disk_bytes(),
                }
            return {
                "pack_sha256": session.pack_sha256,
                "cache": str(self.cache_root / session.pack_sha256[:16]),
                "chunks": session.chunks,
                "chunk_samples": session.chunk_samples,
                "cursor": session.cursor,
                "audible": list(session.audible),
                "background": session.background,
                "workers": self.workers,
                "processes": self.processes,
                #: Seconds of stem rendered per second of wall clock, all workers together.
                "rate": None if rate is None else round(rate, 3),
                #: Seconds of one core per second of one source, lately.
                "cost": round(float(np.mean(self._costs)), 3) if self._costs else None,
                "worker_rss_mb": self._rss_mb,
                "blocked": self.blocked,
                "free_bytes": shutil.disk_usage(self.cache_root).free,
                "sources": sources,
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render every stem of a scene pack")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--cache", type=Path, default=None, help="<data root>/cache/audit_stems")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--clips", type=Path, default=None, help="default: <data root>/clips")
    parser.add_argument("--voices", type=Path, default=None, help="default: <data root>/voices")
    parser.add_argument(
        "--directivity", choices=("pack", "on", "off"), default="pack", help="the owner's switch"
    )
    arguments = parser.parse_args(argv)
    service = StemService(
        arguments.cache or data_root() / "cache" / "audit_stems",
        workers=arguments.workers,
        dry=DrySources(
            arguments.clips or data_root() / "clips", arguments.voices or data_root() / "voices"
        ),
    )
    switch = {"pack": None, "on": True, "off": False}[arguments.directivity]
    started = time.monotonic()
    try:
        session = service.open(arguments.pack, AuditSettings(switch))
        service.want(session, cursor=0, background=True)
        while True:
            status = service.status(session)
            done = sum(int(stem.done.sum()) for stem in session.stems.values())
            total = session.chunks * len(session.stems)
            disk = sum(source["disk_bytes"] for source in status["sources"].values())
            print(
                f"{done}/{total} chunks, {status['rate'] or 0:.2f} s of stem a second, "
                f"{disk / 1e9:.2f} GB, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            failed = [s for s in session.stems.values() if s.failed]
            if failed or status["blocked"]:
                print(status["blocked"] or next(iter(failed[0].failed.values())), file=sys.stderr)
                return 1
            if done == total:
                break
            time.sleep(5.0)
        print(f"stems of {arguments.pack.name}: {status['cache']}")
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    sys.exit(main())
