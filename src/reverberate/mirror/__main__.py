"""Command line: the mirror field of a source, its calibration, and the join to the wave field.

python -m reverberate.mirror run --run DIR --models DIR --source S1 --position X Y Z
    [--parameters CALIBRATION.json] [--cpu]
python -m reverberate.mirror calibrate --run DIR --source S1 --position X Y Z
    [--parameters START.json] [--points 24] [--iterations 8] [--cpu]
python -m reverberate.mirror hybrid --run DIR --source S1 [--high field_mirror]
python -m reverberate.mirror judge --run DIR --source S1 --field field_hybrid [--every 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverberate.mirror", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", type=Path, required=True, help="run directory with field/<S>.h5")
        p.add_argument("--source", required=True)
        p.add_argument("--position", type=float, nargs=3, required=True, help="the source, m")
        p.add_argument("--parameters", type=Path, default=None, help="a calibration json")
        p.add_argument("--cpu", action="store_true", help="the host's cores even with a card")

    p = sub.add_parser("run", help="the mirror field of one source beside its wave field")
    common(p)
    p.add_argument("--models", type=Path, required=True, help="directory of apartment_full.json")
    p.add_argument("--out", default="field_mirror", help="folder of the field, in the run")

    p = sub.add_parser("calibrate", help="fit the parameters on a few points of a traced run")
    common(p)
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--workers", type=int, default=1, help="judges in parallel")

    p = sub.add_parser("hybrid", help="the wave field under the crossover, the mirror over it")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--low", default="field", help="folder of the wave field")
    p.add_argument("--high", default="field_mirror", help="folder of the mirror field")
    p.add_argument("--out", default="field_hybrid")

    p = sub.add_parser("judge", help="a field judged against the wave field, point by point")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--field", required=True, help="folder of the field judged, in the run")
    p.add_argument("--every", type=int, default=1, help="judge every n-th point")
    p.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hybrid":
        from reverberate.mirror.hybrid import Crossover, write_hybrid_field

        summary = write_hybrid_field(
            args.run / args.out / f"{args.source}.h5",
            args.run / args.low / f"{args.source}.h5",
            args.run / args.high / f"{args.source}.h5",
            crossover=Crossover(),
        )
        print(json.dumps(summary, indent=1))
        return 0

    if args.command == "judge":
        from reverberate.mirror.calibration.judge import judge_field

        summary = judge_field(
            args.run / "field" / f"{args.source}.h5",
            args.run / args.field / f"{args.source}.h5",
            every=args.every,
            workers=args.workers,
        )
        target = args.run / "mirror" / f"judged_{args.field}_{args.source}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=1))
        print(target, "criteria met", summary["criteria_met"])
        return 0

    from reverberate.compute import Devices
    from reverberate.mirror.parameters import Parameters, load_parameters
    from reverberate.mirror.pipeline import MirrorSettings

    parameters = load_parameters(args.parameters) if args.parameters else Parameters()
    settings = MirrorSettings(parameters=parameters)
    devices = Devices.host() if args.cpu else Devices.detect()
    if args.command == "run":
        from reverberate.mirror.pipeline import run

        report = run(
            args.run,
            args.source,
            args.position,
            models=args.models,
            settings=settings,
            devices=devices,
            out=args.out,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=1))
        return 0
    from reverberate.mirror.calibration.fit import calibrate

    _, target = calibrate(
        args.run,
        args.source,
        args.position,
        settings=settings.pinned(),
        points=args.points,
        iterations=args.iterations,
        devices=devices,
        workers=args.workers,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
