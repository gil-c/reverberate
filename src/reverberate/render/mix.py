"""A whole scene's mix on disk, rendered by several processes: what training reads.

:func:`reverberate.render.output.write_signal` writes what one engine
yields, block after block. A scene of twenty minutes is rendered here by
as many processes as the machine has cores to give, each an engine of its
own on a share of the scene's time: the engine renders any range from the
pack and the dry signals alone (``engine.py``), so the shares are
independent and the file is, sample for sample, what one engine with the
same settings writes from start to end.

**A share is a minute or so, in the scene's order.** A process renders its
share a run at a time and writes each run at its place in the file, so
memory holds a run and not the scene. The process that asked reads the
shares back in order as they are done, for the digest: the header's
SHA-256 is of the file's bytes, as ``scene-signal.md`` says.

**What the processes share.** A source's tail is noise drawn once from its
seed, 95 MB a source, and every process needs every source's: with
``REVERBERATE_CARRIERS`` naming a folder (set here, for the length of the
render) the first process to need one writes it there and the others map
the same pages (:meth:`reverberate.render.fast.FastTail._carrier`).

``python -m reverberate.render mix <pack.h5> <out>`` is this, fed the
pack's own recipe from the clip libraries.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import resource
import tempfile
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.fast import CARRIERS_VARIABLE
from reverberate.render.output import _paths, write_header

__all__ = ["CARRIERS_VARIABLE", "MIX_SETTINGS", "recipe_engine", "write_mix"]

#: What a mix is rendered with unless told: a thread a process, the tail two seconds at once.
MIX_SETTINGS = RenderSettings(workers=1, chunks_held=1, tail_steps=40)
#: Steps of the shortest share: ten seconds, a whole number of the tail's runs.
SHARE_STEPS = 200
#: Shares a process is handed over a scene, about: the fewer, the less is prepared twice.
SHARES_EACH = 3

_engine: Engine | None = None


def recipe_engine(
    pack_path: Path,
    clips_root: Path | None,
    manifest: Path | None,
    settings: RenderSettings = MIX_SETTINGS,
) -> Engine:
    """An engine on ``pack_path`` fed the pack's own recipe, as the sound check feeds it."""
    from reverberate.render.check.clips import ClipSource, feed_of
    from reverberate.render.pack import read_pack

    pack = read_pack(Path(pack_path), check=False)
    clips = ClipSource(clips_root, manifest)
    recipe = pack.recipe_json()
    dry = {name: feed_of(pack, name, recipe, clips).track for name in pack.sources}
    return Engine(pack, dry, settings=settings)


def _start(factory: Callable[[], Engine], carriers: str) -> None:
    global _engine
    os.environ[CARRIERS_VARIABLE] = carriers
    _engine = factory()


def _share(task: tuple[str, int, int, int]) -> dict[str, Any]:
    """Render the samples ``[start, stop)`` a run at a time, each written at its place."""
    path, start, stop, base = task
    engine = _engine
    assert engine is not None
    started = time.perf_counter()
    run = engine.settings.chunk_steps * engine.pack.header.step_samples
    peak = 0.0
    fd = os.open(path, os.O_WRONLY)
    try:
        lo = start
        while lo < stop:
            hi = min((lo // run + 1) * run, stop)
            frames = np.ascontiguousarray(engine.render(lo, hi).T, dtype="<f4")
            if frames.size:
                peak = max(peak, float(np.max(np.abs(frames))))
            os.pwrite(fd, frames, (lo - base) * frames.shape[1] * 4)
            lo = hi
    finally:
        os.close(fd)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "start": start,
        "stop": stop,
        "peak": peak,
        "seconds": time.perf_counter() - started,
        "cpu_s": usage.ru_utime + usage.ru_stime,
        "rss": int(usage.ru_maxrss) * (1 if os.uname().sysname == "Darwin" else 1024),
        "process": os.getpid(),
    }


def _shares(start: int, stop: int, size: int) -> Iterator[tuple[int, int]]:
    lo = start
    while lo < stop:
        hi = min((lo // size + 1) * size, stop)
        yield lo, hi
        lo = hi


def write_mix(
    target: Path,
    factory: Callable[[], Engine],
    *,
    processes: int | None = None,
    start: int = 0,
    stop: int | None = None,
    scratch: Path | None = None,
    say: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The mix of ``factory``'s engine over ``[start, stop)`` at ``target``; the header, returned.

    ``factory`` makes the engine, in every process: a function of a module
    and its arguments (:func:`functools.partial` of :func:`recipe_engine`),
    so that a process started afresh can call it. ``processes`` is how many
    render; one renders here, with no other process. ``scratch`` is where
    the carriers are kept while the render lasts, the system's temporary
    folder unless said. The header holds, beside what a signal's always
    does, how long the render took and what it cost.
    """
    engine = factory()
    h = engine.pack.header
    stop = h.samples if stop is None else stop
    if not 0 <= start <= stop <= h.samples:
        raise ValueError(f"[{start}, {stop}) is not inside the scene's {h.samples} samples")
    count = max(1, processes if processes is not None else (os.cpu_count() or 2) - 2)
    data_path, header_path = _paths(Path(target))
    data_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.unlink(missing_ok=True)
    frame_bytes = h.channels * 4
    with open(data_path, "wb") as handle:
        handle.truncate((stop - start) * frame_bytes)
    # A share is long: a process filters a source's clip twenty seconds at a time, and
    # one handed every sixth ten seconds would filter every piece again (measured: more
    # than the render). About three shares a process leave the last ones even.
    each = -(-(stop - start) // (count * SHARES_EACH * h.step_samples))
    steps = max(SHARE_STEPS, -(-each // SHARE_STEPS) * SHARE_STEPS)
    shares = list(_shares(start, stop, steps * h.step_samples))
    digest = hashlib.sha256()
    peak, cpu, rss = 0.0, {}, {}
    started = time.perf_counter()

    def take(done: dict[str, Any], handle: Any) -> None:
        nonlocal peak
        peak = max(peak, float(done["peak"]))
        cpu[done["process"]] = float(done["cpu_s"])
        rss[done["process"]] = int(done["rss"])
        left = (done["stop"] - done["start"]) * frame_bytes
        while left:
            block = handle.read(min(left, 1 << 24))
            if not block:
                raise OSError(f"{data_path} is shorter than what was rendered into it")
            digest.update(block)
            left -= len(block)
        if say is not None:
            at = (done["stop"] - start) / h.sample_rate_hz
            say(f"{at:.0f} s of scene in {time.perf_counter() - started:.0f} s")

    with tempfile.TemporaryDirectory(dir=scratch, prefix="carriers-") as carriers:
        # The file is written from ``start``: a share's place is its own less that.
        tasks = [(str(data_path), lo, hi, start) for lo, hi in shares]
        with open(data_path, "rb") as written:
            if count == 1:
                _start(lambda: engine, carriers)
                for task in tasks:
                    take(_share(task), written)
            else:
                context = multiprocessing.get_context("spawn")
                with context.Pool(count, initializer=_start, initargs=(factory, carriers)) as pool:
                    for done in pool.imap(_share, tasks):
                        take(done, written)
        os.environ.pop(CARRIERS_VARIABLE, None)
    wall = time.perf_counter() - started
    return write_header(
        Path(target),
        frames=stop - start,
        channels=h.channels,
        peak=peak,
        sha256=digest.hexdigest(),
        sample_rate_hz=h.sample_rate_hz,
        order=h.order,
        recipe_sha256=h.recipe_sha256,
        sources=list(engine.pack.sources),
        extra={
            "render": {
                "settings": engine.settings.record(),
                "processes": count,
                "wall_s": round(wall, 2),
                "cpu_s": round(sum(cpu.values()), 2),
                "process_rss_bytes": max(rss.values(), default=0),
            }
        },
    )


def main_factory(
    pack: Path, clips_root: Path | None, manifest: Path | None
) -> Callable[[], Engine]:
    """What the command line renders with."""
    return partial(recipe_engine, pack, clips_root, manifest)
