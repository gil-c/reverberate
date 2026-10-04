"""``python -m reverberate.render``: render a pack to a signal, or time the engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reverberate.render", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark", help="seconds of compute per second of scene")
    bench.add_argument("--duration", type=float, default=5.0, help="seconds of scene")
    bench.add_argument("--workers", type=int, default=-1, help="threads of the transforms")
    bench.add_argument("--gpu", action="store_true", help="on the card, with cupy")
    bench.add_argument(
        "--processes", default="", help="also render stems in these many processes: 1,4,8"
    )
    bench.add_argument("--save", type=Path, help="write the report here")
    bench.add_argument("--against", type=Path, help="an earlier report to print this one beside")
    error = commands.add_parser("interpolator", help="the delay line's measured error")
    error.set_defaults(command="interpolator")
    check = commands.add_parser("validate", help="check a pack against the format")
    check.add_argument("pack", type=Path)
    check.add_argument("--deep", action="store_true", help="read every low band response")
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        from reverberate.render.benchmark import measure, scaling, table

        report = measure(duration_s=args.duration, workers=args.workers, gpu=args.gpu)
        if args.processes:
            counts = [int(count) for count in args.processes.split(",")]
            report["scaling"] = scaling(duration_s=args.duration, processes=counts)
        print(json.dumps(report, indent=2))
        before = json.loads(args.against.read_text()) if args.against else None
        print(table(report, before))
        if args.save:
            args.save.write_text(json.dumps(report, indent=2))
    elif args.command == "interpolator":
        from reverberate.render.delay import worst_error

        bands = {"1 kHz to 20 kHz": (1000.0, 20000.0), "20 Hz to 1 kHz": (20.0, 1000.0)}
        print(json.dumps({name: worst_error(*band) for name, band in bands.items()}, indent=2))
    else:
        from reverberate.render.pack import read_pack

        with read_pack(args.pack, deep=args.deep) as pack:
            rows = {name: int(s.early.path_id.shape[0]) for name, s in pack.sources.items()}
            print(json.dumps({"steps": pack.header.steps, "rows": rows}, indent=2))
    np.seterr(all="warn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
