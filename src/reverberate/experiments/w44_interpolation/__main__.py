"""The command line of W44: one measurement of one field, written to one directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import reverberate.experiments.w44_interpolation as package
from reverberate.experiments.w44_interpolation.leave_one_out import leave_one_out
from reverberate.experiments.w44_interpolation.line_channels import line_channels
from reverberate.experiments.w44_interpolation.line_gaps import line_gaps
from reverberate.experiments.w44_interpolation.plane_wave import leave_one_out_fusion
from reverberate.experiments.w44_interpolation.translation_failures import translation_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reverberate.experiments.w44_interpolation",
        description=package.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("leave-one-out", "each point of a lattice from its neighbours, five predictors"),
        ("plane-wave", "each point of a lattice from one, two and four neighbours fused"),
        ("line-gaps", "the error against the spacing, on a line solved every 2 cm"),
        ("failures", "where a translation of one pitch fails round 500 Hz"),
        ("line-channels", "all 64 channels on the line, between the arrays' true centres"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--field", type=Path, required=True, help="one source's field, HDF5")
        command.add_argument("--out", type=Path, required=True, help="where the summary is written")
        if name == "leave-one-out":
            command.add_argument("--limit", type=int, default=None, help="the first points only")
        if name == "line-channels":
            command.add_argument("--plan", type=Path, default=None, help="the campaign's plan.json")
            command.add_argument("--scene", type=Path, default=None, help="mirror/scene.npz")
    args = parser.parse_args(argv)
    if args.command == "leave-one-out":
        leave_one_out(args.field, args.out, limit=args.limit)
    elif args.command == "plane-wave":
        leave_one_out_fusion(args.field, args.out)
    elif args.command == "line-gaps":
        line_gaps(args.field, args.out)
    elif args.command == "line-channels":
        line_channels(args.field, args.out, plan=args.plan, scene=args.scene)
    else:
        translation_failures(args.field, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
