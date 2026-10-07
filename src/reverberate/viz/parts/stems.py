"""A whole scene played from its pack: each source's stem rendered as it is asked for.

A folder of the listening kit holds a window somebody rendered. A scene of
three minutes at order 7 is 12.3 MB a second a source sounds, and nobody
renders it whole to hear its first ten seconds. :class:`Streamed` stands
where a :class:`~reverberate.viz.parts.media.Library` stands, with one item,
the scene, whose stems are the pack's sources: a request for frames waits
for the chunks it covers, which the engine renders then, and what was
rendered once is read from the disk ever after.

Nothing is rendered here. The stems are the audit's
(:mod:`reverberate.viz.audit_stems`): the signal engine, one source at a
time, half a second a chunk, in processes of one thread each, kept under a
key that is everything a stem's bytes depend on. What this module chooses is
how much of a laptop a listen may take:

- at most :data:`MAX_WORKERS` processes, each made nicer by :data:`NICE`, so
  that the page and everything else come first;
- the chunks from the cursor to :data:`AHEAD_S` after it and no further, for
  the sources heard and then for the others (a solo is then instant); the
  rest of the scene is rendered when it is listened to;
- nothing at all once nobody has asked for twenty seconds.

What is kept is what was listened to, sparse where a source is silent, and
:func:`trim` forgets the scenes listened to longest ago once the folder
passes a budget: a stem is rendered again from its pack, nothing is lost.

A stem's level a step comes with its chunks (the omnidirectional channel's
mean square), which is what a light and a meter follow: :meth:`Streamed.levels`
gives it over a range, ``None`` where nothing is rendered yet.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.viz.audit_dry import DrySources
from reverberate.viz.audit_stems import SILENCE_DB, AuditSettings, StemService
from reverberate.viz.parts.media import PAGE_LEVEL_DB, Item, Stem
from reverberate.viz.parts.server import HttpError

__all__ = ["AHEAD_S", "MAX_WORKERS", "NICE", "SCENE", "Streamed", "trim"]

#: The one item of a streamed scene.
SCENE = "scene"
#: Worker processes at most, whatever is asked.
MAX_WORKERS = 4
#: Added to the niceness of every worker.
NICE = 10
#: How far past the cursor the stems are rendered, seconds.
AHEAD_S = 8.0
#: How long a request for frames waits for its chunks, seconds.
WAIT_S = 30.0


def _taken(folder: Path) -> int:
    """What a folder really takes on the disk: a hole in a sparse stem takes nothing."""
    return sum(f.stat().st_blocks * 512 for f in folder.rglob("*") if f.is_file())


def trim(cache: Path, *, budget_gb: float, keep: str | None = None) -> int:
    """Forget the stems of packs, the one opened longest ago first, until within budget.

    Called before a pack is opened: a scene listened to whole is some
    50 MB a second, and what was kept of it goes too once the folder is
    over, the last of all since it was opened last. ``keep`` names a
    folder that is never forgotten. Only folders named as a pack's stems
    are touched (sixteen hex digits); the bytes freed are returned.
    """
    cache = Path(cache)
    if not cache.is_dir():
        return 0
    folders = [f for f in cache.iterdir() if f.is_dir() and re.fullmatch(r"[0-9a-f]{16}", f.name)]
    sizes = {folder: _taken(folder) for folder in folders}
    total, freed = sum(sizes.values()), 0
    for folder in sorted(folders, key=lambda f: f.stat().st_mtime):
        if total <= budget_gb * 1e9:
            break
        if folder.name == keep:
            continue
        shutil.rmtree(folder, ignore_errors=True)
        total -= sizes[folder]
        freed += sizes[folder]
    return freed


class Streamed:
    """The sources of one pack as one item, rendered on demand and kept in ``cache``."""

    def __init__(
        self,
        pack: Path,
        cache: Path,
        *,
        order: Iterable[str] | None = None,
        workers: int = MAX_WORKERS,
        nice: int = NICE,
        ahead_s: float = AHEAD_S,
        processes: bool = True,
        dry: DrySources | None = None,
    ) -> None:
        self.pack_path = Path(pack).resolve()
        self.service = StemService(
            cache,
            workers=max(1, min(int(workers), MAX_WORKERS)),
            processes=processes,
            dry=dry,
            nice=nice,
        )
        self.session = self.service.open(self.pack_path, AuditSettings())
        # Opened now: :func:`trim` forgets the scenes opened longest ago first.
        with contextlib.suppress(OSError):
            os.utime(Path(cache) / self.session.pack_sha256[:16])
        header = self.session.pack.header
        chunk_s = self.session.chunk_samples / float(header.sample_rate_hz)
        self.service.window_chunks = max(1, math.ceil(ahead_s / chunk_s))
        self.session.window_chunks = self.service.window_chunks
        # Never the rest of the scene behind the listener's back: what is heard, and a little.
        self.session.background = False
        self.rate = float(header.sample_rate_hz)
        self.frames_total = int(header.samples)
        self.channels = int(header.channels)
        self.step_s = float(header.step_s)
        known = list(self.session.order)
        #: The sources in the order a track list holds them; the sum is in the pack's own.
        self.sources = [s for s in (order or known) if s in known]
        self.sources += [s for s in known if s not in self.sources]
        self._item = Item(
            SCENE,
            self.pack_path.name,
            tuple(
                Stem(
                    name,
                    self.session.stems[name].data_path,
                    "ambisonic",
                    self.channels,
                    self.frames_total,
                    self.rate,
                )
                for name in self.sources
            ),
            {"streamed": True, "duration_s": self.frames_total / self.rate},
        )

    # --- what a library answers -------------------------------------------------------------

    def item(self, name: str) -> Item:
        if name != SCENE:
            raise HttpError(404, f"no item named {name!r}")
        return self._item

    def describe(self) -> list[dict[str, Any]]:
        return [self._item.describe()]

    def _heard(self, gains: Mapping[str, float] | None) -> list[tuple[str, float]]:
        weights = [(name, float((gains or {}).get(name, 1.0))) for name in self.session.order]
        return [(name, weight) for name, weight in weights if weight != 0.0]

    def want(self, start: int, gains: Mapping[str, float] | None = None) -> None:
        """Say where the listener is: the chunks from there on are rendered first."""
        index = min(max(int(start), 0) // self.session.chunk_samples, self.session.chunks - 1)
        self.service.want(
            self.session,
            cursor=index,
            audible=[name for name, _ in self._heard(gains)],
            background=False,
        )

    def frames(
        self, name: str, start: int, count: int, gains: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """``count`` frames of the scene from ``start``, rendered if they were not.

        The sources heard are summed in float64 in the pack's order, each
        at its gain, and brought back to float32; one source at a gain of
        one is the engine's own samples. Past the scene's end are zeros.
        """
        self.item(name)
        start, count = int(start), int(count)
        if start < 0 or count < 0:
            raise HttpError(400, "a range starts at a frame and has a length")
        out = np.zeros((count, self.channels), dtype=np.float32)
        stop = min(start + count, self.frames_total)
        heard = self._heard(gains)
        if stop <= start or not heard:
            if stop > start:
                self.want(start, gains)
            return out
        self.want(start, gains)
        size = self.session.chunk_samples
        names = [source for source, _ in heard]
        alone = len(heard) == 1 and heard[0][1] == 1.0
        total = np.zeros((stop - start, self.channels))
        for index in range(start // size, (stop - 1) // size + 1):
            try:
                ready = self.service.wait(self.session, names, index, WAIT_S)
            except RuntimeError as error:
                raise HttpError(500, f"the engine failed: {error}") from error
            if not ready:
                raise HttpError(503, self.service.blocked or "the render is behind: ask again")
            first = index * size
            a, b = max(start, first), min(stop, first + size)
            for source, weight in heard:
                piece = self.session.stems[source].read(index)[a - first : b - first]
                if alone:
                    out[a - start : b - start] = piece
                else:
                    total[a - start : b - start] += weight * piece.astype(np.float64)
        if not alone:
            out[: stop - start] = total
        return out

    def mono(self, name: str, gains: Any = None, *, stem: str | None = None) -> np.ndarray:
        raise HttpError(409, "a scene that is streamed is not looked at whole")

    # --- what a light and a meter follow ----------------------------------------------------

    def levels(self, start_s: float, stop_s: float) -> dict[str, Any]:
        """Each source's level a step over ``[start_s, stop_s)``, at the page's level.

        ``None`` where the chunk is not rendered; :data:`SILENCE_DB` and
        under where the engine gave zeros.
        """
        per = self.session.chunk_samples // self.session.pack.header.step_samples
        steps = self.frames_total // self.session.pack.header.step_samples
        first = min(max(int(math.floor(start_s / self.step_s)), 0), steps)
        last = min(max(int(math.ceil(stop_s / self.step_s)), first), steps)
        stems: dict[str, list[float | None]] = {}
        for name in self.sources:
            stem = self.session.stems[name]
            row: list[float | None] = []
            for step in range(first, last):
                index = step // per
                record = stem.records.get(index) if stem.done[index] else None
                if record is None:
                    row.append(None)
                    continue
                values = record["levels_db"]
                value = (
                    float(values[step - index * per])
                    if step - index * per < len(values)
                    else SILENCE_DB
                )
                row.append(value if value <= SILENCE_DB else round(value + PAGE_LEVEL_DB, 1))
            stems[name] = row
        return {
            "item": SCENE,
            "hop_s": self.step_s,
            "first": first,
            "unit": "dB re full scale",
            "stems": stems,
        }

    def status(self) -> dict[str, Any]:
        """How the render goes: its rate, what a worker holds, what is kept."""
        told = self.service.status(self.session)
        seconds = self.session.chunk_samples / self.rate
        return {
            "workers": told["workers"],
            "nice": self.service.nice,
            "ahead_s": self.service.window_chunks * seconds,
            "rate": told["rate"],
            "cost": told["cost"],
            "worker_rss_mb": told["worker_rss_mb"],
            "blocked": told["blocked"],
            "cache": told["cache"],
            "rendered_s": {
                name: round(sum(b - a for a, b in source["ready"]) * seconds, 1)
                for name, source in told["sources"].items()
            },
            "failed": {
                name: source["failed"]
                for name, source in told["sources"].items()
                if source["failed"]
            },
        }

    def close(self) -> None:
        self.service.close()
