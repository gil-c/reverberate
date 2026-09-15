"""The engine on this machine: one solve per band, in receiver slices when RAM is short.

PFFDTD's CUDA engine reads four files from its working directory and writes
``sim_outs.h5`` beside them: ``Nr x Nt`` doubles, one row per receiver node,
held in host memory until the end and then written, which needs about 2.2
times the output in RAM (measured, ``oom_kill`` on a 125 GB output with 194
GB). A band whose output does not fit is solved as consecutive slices of
its receiver rows, each a full solve of the storey; the card's time
multiplies by the slices and the grid does not shrink. The first campaigns
did this over ssh from the laptop; here the same rules run on the machine
that holds the card, so the pressure is consumed by the encoder as each
slice lands and never crosses a network.

Every card the engine was built for is driven the same way: the binary is
compiled for the card it finds (``scripts/build_pffdtd.sh``), and
``CUDA_VISIBLE_DEVICES`` chooses how many of them a solve splits over.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.wave.comms import write_comms
from reverberate.wave.voxelise import CACHE_FILES

__all__ = [
    "RAM_PER_OUTPUT",
    "EngineResult",
    "host_memory_gb",
    "prepare_job",
    "output_sample_bytes",
    "run_engine",
    "slices_for",
    "solve_slices",
]

#: Measured on 50451826: the engine needs 2.2 x its output in host RAM when it writes
#: (two copies of every receiver's record, the second reordered for the file).
RAM_PER_OUTPUT = 2.2

#: Bytes a sample of a receiver's record takes in ``sim_outs.h5`` and in the
#: engine's host memory. Upstream keeps and writes float64 whatever the build;
#: ``scripts/pffdtd/0008-…patch`` keeps and writes the engine's own precision,
#: which is float32 for the single precision binaries the campaign runs. The
#: records were float32 numbers already (``u_out_buf`` is ``Real``), so the file
#: holds the same values as the float64 one rounded to float32, which is what
#: the encoder read from it; half the RAM, half the disk, half the writing.
OUTPUT_SAMPLE_BYTES_PATCHED = 4
OUTPUT_SAMPLE_BYTES_UPSTREAM = 8
PATCH_8_MARK = "REVERBERATE PATCH 8"

#: The engine keeps one sample per receiver on the card (``u_out_buf``) and
#: the whole record in host memory; what the card holds is the grid.
#: VRAM per grid node and the engine's fixed overhead, measured on an A100.
VRAM_PER_NODE_B = 9.027
VRAM_FIXED_B = 2.13e9
#: Kept free on every card, as a fraction of its memory.
VRAM_MARGIN = 0.1

#: The engine's progress line.
PROGRESS = re.compile(r"Running \[\s*([0-9.]+)%\]")

#: What a cache entry lends the engine; ``comms_out.h5`` is the run's own.
ENGINE_INPUTS = tuple(name for name in CACHE_FILES if name != "cart_grid.h5")


def output_sample_bytes(pffdtd_dir: Path | str, *, double_precision: bool = False) -> int:
    """How many bytes a record sample takes with the engine built in ``pffdtd_dir``."""
    if double_precision:
        return OUTPUT_SAMPLE_BYTES_UPSTREAM
    header = Path(pffdtd_dir) / "c_cuda" / "fdtd_data.h"
    try:
        patched = PATCH_8_MARK in header.read_text()
    except OSError:
        patched = False
    return OUTPUT_SAMPLE_BYTES_PATCHED if patched else OUTPUT_SAMPLE_BYTES_UPSTREAM


def host_memory_gb() -> float:
    """The memory this process may use: the cgroup's limit when there is one, else the host's."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return int(text) / 1e9
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return 16.0


def slices_for(output_bytes: float, ram_gb: float) -> int:
    """How many consecutive receiver slices a band needs to fit ``ram_gb``."""
    return max(1, int(math.ceil(RAM_PER_OUTPUT * output_bytes / 1e9 / max(ram_gb - 8.0, 1.0))))


def card_memory_bytes() -> list[int]:
    """Every visible card's memory, from cupy; empty without a card."""
    try:
        import cupy

        count = int(cupy.cuda.runtime.getDeviceCount())
        return [
            int(cupy.cuda.runtime.getDeviceProperties(i)["totalGlobalMem"]) for i in range(count)
        ]
    except Exception:  # noqa: BLE001 - no cupy or no card: the caller sizes by RAM alone
        return []


def grid_fits_cards(grid_points: float, card_bytes: list[int]) -> tuple[bool, str]:
    """Whether the grid, split over the visible cards, fits beside the engine's overhead.

    The receivers' records live in host memory (``gpu_engine.h`` copies one
    sample a receiver a step), so the cards hold the grid and nothing that
    grows with the solved window; slicing a band cannot make it fit.
    """
    if not card_bytes:
        return True, "no card counted"
    share = grid_points / len(card_bytes) * VRAM_PER_NODE_B + VRAM_FIXED_B
    smallest = min(card_bytes)
    fits = share <= (1.0 - VRAM_MARGIN) * smallest
    return fits, (
        f"{share / 1e9:.1f} GB a card for {grid_points / 1e9:.2f}e9 nodes over"
        f" {len(card_bytes)} card(s) of {smallest / 1e9:.0f} GB"
    )


def cuts_for(rows: list[list[int] | None], receivers: int, parts: int) -> list[int]:
    """Row cuts of ``parts`` consecutive slices, never through a point."""
    placed = [r for r in rows if r is not None]
    return [0] + [placed[len(placed) * k // parts][0] for k in range(1, parts)] + [int(receivers)]


def prepare_job(job_dir: Path, entry_path: Path, comms: Path) -> Path:
    """The engine's working directory: the entry's three files linked, the comms copied."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in ENGINE_INPUTS:
        target = job_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(Path(entry_path) / name, target)
        except OSError:
            target.symlink_to(Path(entry_path) / name)
    target = job_dir / "comms_out.h5"
    if Path(comms).resolve() != target.resolve():
        # A comms file written into the job directory already is the run's
        # own and stays; one from elsewhere is linked beside the grid.
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(comms, target)
        except OSError:
            target.symlink_to(Path(comms).resolve())
    if not target.exists():
        raise FileNotFoundError(f"{target} is missing")
    return job_dir


@dataclass(frozen=True)
class EngineResult:
    output: Path
    engine_s: float
    log: Path
    output_bytes: int


def run_engine(
    job_dir: Path,
    *,
    pffdtd_dir: Path,
    devices: str | None = None,
    double_precision: bool = False,
    timeout_s: float | None = None,
    on_progress: Any = None,
    poll_s: float = 30.0,
    stall_polls: int = 20,
) -> EngineResult:
    """Run the CUDA engine in ``job_dir`` and follow its log until it exits.

    ``devices`` is the ``CUDA_VISIBLE_DEVICES`` the engine splits over. A run
    whose percentage and output size both stop moving for ``stall_polls``
    polls is reported as stalled rather than waited on, and killed.
    """
    job_dir = Path(job_dir)
    precision = "double" if double_precision else "single"
    binary = Path(pffdtd_dir) / "c_cuda" / f"fdtd_main_gpu_{precision}.x"
    if not binary.is_file():
        raise FileNotFoundError(f"{binary} is not built; run scripts/build_pffdtd.sh")
    for name in (*ENGINE_INPUTS, "comms_out.h5"):
        if not (job_dir / name).exists():
            raise FileNotFoundError(f"{job_dir / name} is missing")
    output = job_dir / "sim_outs.h5"
    output.unlink(missing_ok=True)
    log = job_dir / "engine.log"
    environment = dict(os.environ)
    if devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = devices
    started = time.time()
    with log.open("wb") as handle:
        process = subprocess.Popen(
            [str(binary)],
            cwd=job_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
        )
        last_mark: tuple[float | None, int] = (None, -1)
        unchanged = 0
        while True:
            try:
                process.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            percent = _last_percent(log)
            written = output.stat().st_size if output.exists() else 0
            if on_progress is not None:
                on_progress(percent, time.time() - started)
            # At 100 per cent the engine is writing its output, which takes
            # minutes for a hundred gigabytes; the file growing is progress.
            mark = (percent, written)
            if mark == last_mark:
                unchanged += 1
                if unchanged >= stall_polls:
                    process.kill()
                    raise RuntimeError(
                        f"the engine stalled at {percent}% with {written} bytes written"
                        f" for {unchanged} polls of {poll_s:g} s"
                    )
            else:
                unchanged = 0
            last_mark = mark
            if timeout_s is not None and time.time() - started > timeout_s:
                process.kill()
                raise TimeoutError(f"the engine passed its {timeout_s:g} s budget")
    engine_s = time.time() - started
    if process.returncode != 0 or not output.is_file():
        tail = log.read_text(errors="replace")[-1500:]
        raise RuntimeError(
            f"the engine exited with {process.returncode} and "
            f"{'no' if not output.is_file() else 'an'} output; its log ends:\n{tail}"
        )
    return EngineResult(
        output=output, engine_s=round(engine_s, 1), log=log, output_bytes=output.stat().st_size
    )


#: What the engine prints last, after it has written its output and dumped
#: the final samples of every receiver to its log: tens of megabytes of
#: text after the last progress line, so the percentage is not in the tail.
ENGINE_DONE_MARK = b"sim data freed"


def pressure_complete(output: Path, log: Path, expected_bytes: float) -> bool:
    """Whether a slice's pressure on disk is a finished solve: the size and the log agree."""
    if not output.is_file() or output.stat().st_size < 0.95 * expected_bytes:
        return False
    try:
        with log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4000))
            tail = handle.read()
    except OSError:
        return False
    return ENGINE_DONE_MARK in tail or _last_percent(log) == 100.0


def _last_percent(log: Path) -> float | None:
    try:
        with log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 20000))
            text = handle.read().decode(errors="replace").replace("\r", "\n")
    except OSError:
        return None
    found = PROGRESS.findall(text)
    return float(found[-1]) if found else None


def solve_slices(
    *,
    job_root: Path,
    entry_path: Path,
    source_position: np.ndarray,
    positions: np.ndarray,
    rows: list[list[int] | None],
    duration_s: float,
    output_bytes: float,
    ram_gb: float,
    pffdtd_dir: Path,
    devices: str | None,
    consume: Any,
    say: Any = None,
    timeout_s: float | None = None,
    grid_points: float = 0.0,
    card_bytes: list[int] | None = None,
    already_done: Any = None,
) -> list[dict[str, Any]]:
    """Solve a band in slices and hand each slice to ``consume`` before the next.

    ``consume(slice_index, cut_start, cut_stop, sim_outs, comms)`` is called
    once per slice with the row range of the whole plan it covers; it is
    expected to read the pressure and may delete it. Returns one record per
    slice with its timings.

    The slices are sized by the host's RAM at the engine's write; the cards
    must hold the grid whole (``grid_points`` nodes over the visible cards),
    which is checked before anything is solved.
    """
    say = say or (lambda message: None)
    parts = slices_for(output_bytes, ram_gb)
    cuts = cuts_for(rows, int(positions.shape[0]), parts)
    say(f"{parts} slice(s) for {output_bytes / 1e9:.1f} GB of output with {ram_gb:.0f} GB of RAM")
    cards = card_bytes if card_bytes is not None else card_memory_bytes()
    if devices:
        cards = [cards[int(i)] for i in devices.split(",") if int(i) < len(cards)] or cards
    if cards and grid_points:
        fits, why = grid_fits_cards(grid_points, cards)
        say(f"cards: {why}")
        if not fits:
            raise RuntimeError(f"the grid does not fit the cards: {why}")
    records = []
    for k in range(parts):
        job_dir = Path(job_root) / (f"part{k}" if parts > 1 else "whole")
        job_dir.mkdir(parents=True, exist_ok=True)
        comms = job_dir / "comms_out.h5"
        if already_done is not None and already_done(k, cuts[k], cuts[k + 1]):
            say(f"slice {k + 1}/{parts}: already encoded, skipped")
            records.append({"slice": k, "rows": [cuts[k], cuts[k + 1]], "skipped": True})
            continue
        expected_bytes = output_bytes * (cuts[k + 1] - cuts[k]) / max(1, positions.shape[0])
        kept = job_dir / "sim_outs.h5"
        if comms.is_file() and pressure_complete(kept, job_dir / "engine.log", expected_bytes):
            # A solve that finished before a failure downstream: consumed, not redone.
            solved_gb = kept.stat().st_size / 1e9
            say(f"slice {k + 1}/{parts}: pressure already solved, {solved_gb:.1f} GB")
            t0 = time.time()
            consumed = consume(k, cuts[k], cuts[k + 1], kept, comms)
            records.append(
                {
                    "slice": k,
                    "rows": [cuts[k], cuts[k + 1]],
                    "reused": True,
                    "consume_s": round(time.time() - t0, 1),
                    "consumed": consumed,
                    "engine_s": 0.0,
                }
            )
            continue
        t0 = time.time()
        write_comms(
            entry_path,
            np.asarray(source_position, dtype=float),
            positions[cuts[k] : cuts[k + 1]],
            duration_s,
            diff_source=True,
            out_path=comms,
            interpolation="nearest",
        )
        prepare_job(job_dir, entry_path, comms)
        comms_s = time.time() - t0
        say(f"slice {k + 1}/{parts}: rows {cuts[k]}:{cuts[k + 1]}, comms in {comms_s:.0f} s")
        result = run_engine(
            job_dir,
            pffdtd_dir=pffdtd_dir,
            devices=devices,
            timeout_s=timeout_s,
            on_progress=lambda p, t: say(
                f"    engine {p if p is not None else '?'}% at {t / 60:.1f} min"
            ),
        )
        say(
            f"slice {k + 1}/{parts}: engine {result.engine_s / 60:.1f} min,"
            f" {result.output_bytes / 1e9:.1f} GB"
        )
        t0 = time.time()
        consumed = consume(k, cuts[k], cuts[k + 1], result.output, comms)
        records.append(
            {
                "slice": k,
                "rows": [cuts[k], cuts[k + 1]],
                "comms_s": round(comms_s, 1),
                "engine_s": result.engine_s,
                "output_bytes": result.output_bytes,
                "consume_s": round(time.time() - t0, 1),
                "consumed": consumed,
            }
        )
    return records
