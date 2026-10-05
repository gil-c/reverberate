"""How a trace's time goes with what it is given: the same bundle on 1, 2, 4, 8 workers or cards.

python -m reverberate.trace scaling --bundle B --out O [--free-field] [--cpu]
    [--workers 1,2,4,8] [--cards 1,2,4,8] [--seed-from RUN] [--keep]

Each count is a whole run of the bundle into a directory of its own, so
nothing is read from a cache another count filled; ``--seed-from`` hands
every run the same caches of an earlier one (``tails`` by default), which
is how a laptop measures the host's stages without casting the rays on its
cores. What is read is each run's own account
(:meth:`reverberate.trace.pool.Pool.account`):

- ``wall``: the seconds between the queue's first job and its last, and the
  seconds of the pack's write after it, which one process does;
- per stage, ``work``: the seconds its jobs took, summed over the workers.
  Work that grows with the workers is workers in each other's way (memory
  traffic, the page cache, the disk): the ``inflation`` column;
- ``ideal``: the first count's wall over the workers;
- the **serial fraction** of the whole (Karp and Flatt's:
  ``(1/speedup - 1/n) / (1 - 1/n)``), which is what bounds the curve where
  it is constant, and the workers' interference where it grows.

``scaling.json`` holds the runs, and the table is printed. With ``--cards``
the host's workers are the last of ``--workers`` at every count, and the
cards are the first ``n`` of the machine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

__all__ = ["main", "measure_run", "table"]

STAGES = ("solve", "paths", "rays", "tails", "level", "rows")


def measure_run(
    args: argparse.Namespace,
    told: dict[str, Any],
    out: Path,
    *,
    workers: int,
    cards: int | None,
) -> dict[str, Any]:
    """One whole run of the bundle on ``workers`` host processes and the first ``cards`` cards."""
    from reverberate.trace.engines import build_engine
    from reverberate.trace.run import Trace, pack_digest

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in args.seed or []:
        held = Path(args.seed_from) / name
        if held.is_dir():
            shutil.copytree(held, out / name)
    gpu = False if (args.cpu or cards == 0) else None
    devices = args.devices
    if cards:
        devices = ",".join(str(k) for k in range(cards))
    started = time.time()
    trace = Trace(
        bundle=args.bundle,
        out=out,
        engine=build_engine(told, args.bundle, out, gpu=gpu, pffdtd_dir=args.pffdtd),
        gpu=gpu,
        pffdtd_dir=args.pffdtd,
        card_devices=devices,
        check_mode="read",
        workers=workers,
        engine_told={**told, "devices": None},
        quiet=True,
    )
    report = trace.run()
    account = report["pool"]
    record = {
        "workers": workers,
        "cards": len(report["machine"]["cards"]),
        "processes": len(account["workers"]),
        "total_s": round(time.time() - started, 2),
        "work_wall_s": report["timings_s"]["work"],
        "queue_wall_s": account["wall_s"],
        "write_s": report["timings_s"].get("write", 0.0),
        "stages": {
            stage: {"jobs": held["jobs"], "work_s": held["work_s"], "wall_s": held["wall_s"]}
            for stage, held in account["stages"].items()
        },
        "busy_s": [w["busy_s"] for w in account["workers"]],
        "machine": report["machine"],
        "predicted": report.get("predicted", {}),
        # What the pack holds, but for its date and its seconds: one digest for every count.
        "pack": pack_digest(out / "pack.h5"),
    }
    if not args.keep:
        shutil.rmtree(out)
    return record


def table(runs: list[dict[str, Any]], by: str) -> str:
    """The runs as a table: wall against the ideal, the serial fraction, each stage's work."""
    first = runs[0]
    base = float(first["work_wall_s"])
    start = max(1, int(first[by]))
    stages = [s for s in STAGES if any(s in run["stages"] for run in runs)]
    lines = [
        f"{by:>8} {'wall s':>8} {'ideal s':>8} {'speed-up':>8} {'serial':>7} {'write s':>8}  "
        + "  ".join(f"{s + ' work s (x)':>20}" for s in stages)
    ]
    for run in runs:
        n = max(1, int(run[by])) / start
        wall = float(run["work_wall_s"])
        speed = base / max(wall, 1e-9)
        serial = "" if n <= 1 else f"{(1.0 / speed - 1.0 / n) / (1.0 - 1.0 / n):.3f}"
        cells = []
        for stage in stages:
            held = run["stages"].get(stage)
            origin = first["stages"].get(stage)
            if held is None:
                cells.append(f"{'':>20}")
                continue
            ratio = held["work_s"] / origin["work_s"] if origin and origin["work_s"] else 1.0
            cells.append(f"{held['work_s']:>13.1f} ({ratio:4.2f})")
        lines.append(
            f"{run[by]:>8} {wall:>8.1f} {base / n:>8.1f} {speed:>8.2f} {serial:>7}"
            f" {run['write_s']:>8.1f}  " + "  ".join(cells)
        )
    return "\n".join(lines)


def main(args: argparse.Namespace, told: dict[str, Any]) -> int:
    workers = [int(v) for v in str(args.workers).split(",") if v]
    cards = [int(v) for v in str(args.cards).split(",") if v] if args.cards else []
    args.seed = [v for v in str(args.seed).split(",") if v] if args.seed_from else []
    args.out.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    by = "cards" if cards else "workers"
    for count in cards or workers:
        name = f"{count}{'c' if cards else 'w'}"
        record = measure_run(
            args,
            told,
            args.out / name,
            workers=workers[-1] if cards else count,
            cards=count if cards else None,
        )
        runs.append(record)
        print(
            f"{name}: work {record['work_wall_s']} s, write {record['write_s']} s,"
            f" total {record['total_s']} s on {record['processes']} process(es)",
            flush=True,
        )
        (args.out / "scaling.json").write_text(json.dumps({"by": by, "runs": runs}, indent=1))
    print(table(runs, by))
    same = len({run["pack"] for run in runs}) == 1
    print(
        "the pack is the same at every count"
        if same
        else "THE PACKS DIFFER: " + ", ".join(f"{run[by]}: {run['pack'][:12]}" for run in runs)
    )
    return 0 if same else 1
