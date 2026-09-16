"""The mirror's command line.

python -m reverberate.mirror run --run DIR --models DIR --source S1 [--position x y z]
    [--rays N] [--order K] [--workers W] [--judge-every N] [--cpu]
python -m reverberate.mirror derive --model apartment_full.json --out DIR/scene
    [--manifest manifest.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverberate.mirror", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="the mirror of one source of a run: field, metrics, report")
    p.add_argument("--run", type=Path, required=True, help="run directory holding field/<S>.h5")
    p.add_argument("--models", type=Path, required=True, help="export with apartment_full.json")
    p.add_argument("--source", required=True, help="source name, as in walk.json")
    p.add_argument("--position", type=float, nargs=3, default=None, help="else from walk.json")
    p.add_argument("--rays", type=int, default=1_000_000)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--flutter", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--judge-every", type=int, default=1)
    p.add_argument("--sound-speed", type=float, default=343.2)
    p.add_argument("--cpu", action="store_true", help="the twins, no card")

    p = sub.add_parser("derive", help="the derived geometry of a solver model, with its census")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="path without suffix")
    p.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if args.cpu:
            os.environ["REVERBERATE_NO_GPU"] = "1"
        from reverberate.mirror.ism import IsmSettings
        from reverberate.mirror.rays import RaySettings
        from reverberate.mirror.stage import MirrorSettings, run_mirror

        position = args.position
        if position is None:
            manifest = json.loads((args.run / "walk.json").read_text())
            entry = next(
                s
                for s in manifest["sources"]
                if str(s.get("id")) == args.source or str(s.get("name")) == args.source
            )
            position = [float(v) for v in entry["position"]]
        settings = MirrorSettings(
            ism=IsmSettings(max_order=args.order, flutter_order=args.flutter),
            rays=RaySettings(rays=args.rays),
            workers=args.workers,
            judge_every=args.judge_every,
        )
        report = run_mirror(
            args.run,
            models=args.models,
            source={"name": args.source, "position": position},
            settings=settings,
            sound_speed_m_s=args.sound_speed,
        )
        print(
            json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=1, default=str)
        )
        return 0
    if args.command == "derive":
        from reverberate.mirror.geometry import derive, write_derived

        manifest = json.loads(args.manifest.read_text()) if args.manifest else None
        derived = derive(args.model, manifest=manifest, say=print)
        path = write_derived(derived, args.out)
        print(json.dumps(derived.census["totals"], indent=1))
        print("wrote", path)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
