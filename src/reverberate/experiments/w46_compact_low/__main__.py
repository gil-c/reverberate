"""The command line of W46: one measurement, written to one directory."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import reverberate.experiments.w46_compact_low as package

#: The grid of the first scene's pairs: the wave grid of hssd_0076 to 1500 Hz.
SCENE_GRID = "a6422ab46225c2ea0bdb6518a99a9e13"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reverberate.experiments.w46_compact_low",
        description=package.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("pairs", "every format, cutoff and length on real pairs, at the head"),
        ("rails", "the rank of a rail's positions at one cell"),
        ("degrees", "a degree under a frequency, against the truth of the dense line"),
        ("pitch", "two fused cells a pitch apart, against the source's distance"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--out", type=Path, required=True, help="where the summary goes")
        if name in ("pairs", "rails"):
            command.add_argument(
                "--pairs",
                type=Path,
                action="append",
                help="a pair cache's root; the local one unless said",
            )
            command.add_argument("--grid", default=SCENE_GRID, help="the grid's key")
            command.add_argument("--count", type=int, default=120 if name == "pairs" else 6)
        else:
            command.add_argument("--field", type=Path, required=True, help="the line's field, HDF5")
            command.add_argument("--plan", type=Path, required=True, help="the line's plan.json")
            command.add_argument("--cases", type=int, default=40 if name == "degrees" else 16)
    args = parser.parse_args(argv)
    summary: dict[str, Any]
    say = functools.partial(print, flush=True)
    if args.command in ("pairs", "rails"):
        from reverberate.accel.pairs import PairCache
        from reverberate.experiments.w46_compact_low.pairs import measure_pairs, measure_rails

        roots = args.pairs or [PairCache.local(args.grid).root]
        if args.command == "pairs":
            summary = measure_pairs(roots, args.grid, count=args.count, say=say)
        else:
            summary = measure_rails(roots, args.grid, rails=args.count, say=say)
    else:
        from reverberate.experiments.w46_compact_low.line import (
            Line,
            measure_degrees,
            measure_pitch,
        )

        line = Line(args.field, args.plan, say=say)
        if args.command == "degrees":
            summary = measure_degrees(line, cases=args.cases, say=say)
        else:
            summary = measure_pitch(line, cases=args.cases, say=say)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"summary_{args.command}.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
