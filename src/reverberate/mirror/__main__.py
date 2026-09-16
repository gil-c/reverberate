"""The mirror's command line.

python -m reverberate.mirror run --run DIR --models DIR --source S1 [--position x y z]
    [--rays N] [--order K] [--workers W] [--judge-every N] [--cpu] [--phase card|host|all]
    [--tag c] [--skip-specular 3] [--tail-from 0.005]
python -m reverberate.mirror derive --model apartment_full.json --out DIR/scene
    [--manifest manifest.json]
python -m reverberate.mirror calibrate --run DIR --source S1 [--points 24] [--iterations 40]
    [--tied] [--order 3] [--rays 100000] [--start FILE.json] [--skip-specular 3]
python -m reverberate.mirror subset --run DIR --source S1 --points 24 [--order 3]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


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
    p.add_argument("--max-images", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--judge-every", type=int, default=1)
    p.add_argument("--sound-speed", type=float, default=343.2)
    p.add_argument("--cpu", action="store_true", help="the twins, no card")
    p.add_argument(
        "--parameters",
        type=Path,
        default=None,
        help="a calibration json (mirror/calibration/<key>.json) to render with",
    )
    p.add_argument("--tag", default="", help="a second mirror beside the first, e.g. c")
    p.add_argument(
        "--skip-specular",
        type=int,
        default=0,
        help="rays whose bounces are all specular up to this order are left to the images",
    )
    p.add_argument("--tail-from", type=float, default=0.020, help="s after the first arrival")
    p.add_argument(
        "--window",
        type=float,
        default=0.080,
        help="s; the image tree's window, past which the rays count every arrival again",
    )
    p.add_argument("--no-signature", action="store_true", help="flat pulses, no source signature")
    p.add_argument("--no-air", action="store_true", help="no air absorption")
    p.add_argument("--band-limit", type=float, default=0.0, help="Hz; 0 keeps the whole band")
    p.add_argument(
        "--phase",
        choices=("card", "host", "all"),
        default="all",
        help="card: paths and rays where the card is; host: the rest where the field is",
    )

    p = sub.add_parser("calibrate", help="the calibration on a few points, where the card is")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--position", type=float, nargs=3, default=None, help="else from walk.json")
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--tied", action="store_true", help="three coordinates instead of fifteen")
    p.add_argument(
        "--method",
        choices=("nelder", "fixed"),
        default="nelder",
        help="fixed: band by band, T30 to absorption and colour to tail gain",
    )
    p.add_argument(
        "--scattering", type=float, nargs="*", default=(), help="fixed: scattering scales to try"
    )
    p.add_argument("--order", type=int, default=3, help="ambisonic order the cost is read at")
    p.add_argument("--rays", type=int, default=100_000)
    p.add_argument("--start", type=Path, default=None, help="a calibration json to start from")
    p.add_argument(
        "--skip-specular",
        type=int,
        default=0,
        help="rays whose bounces are all specular up to this order are left to the images",
    )
    p.add_argument("--tail-from", type=float, default=0.020, help="s after the first arrival")
    p.add_argument(
        "--window",
        type=float,
        default=0.080,
        help="s; the image tree's window, past which the rays count every arrival again",
    )
    p.add_argument("--no-signature", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="judges in parallel")
    p.add_argument("--sound-speed", type=float, default=343.2)
    p.add_argument("--cpu", action="store_true")

    p = sub.add_parser("subset", help="the references of the calibration points, to travel")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--points", type=int, default=24)
    p.add_argument("--order", type=int, default=3)

    p = sub.add_parser("derive", help="the derived geometry of a solver model, with its census")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="path without suffix")
    p.add_argument("--manifest", type=Path, default=None)
    return parser


def _position_of(args: argparse.Namespace) -> list[float]:
    if args.position is not None:
        return [float(v) for v in args.position]
    manifest = json.loads((args.run / "walk.json").read_text())
    entry = next(
        s
        for s in manifest["sources"]
        if str(s.get("id")) == args.source or str(s.get("name")) == args.source
    )
    return [float(v) for v in entry["position"]]


def _parameters_of(path: Path | None) -> Any:
    from reverberate.mirror.calibrate import Parameters

    if path is None:
        return Parameters()
    record = json.loads(path.read_text())
    return Parameters.from_record(record.get("parameters", record))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if args.cpu:
            os.environ["REVERBERATE_NO_GPU"] = "1"
        from reverberate.mirror.ism import IsmSettings
        from reverberate.mirror.rays import RaySettings
        from reverberate.mirror.render import RenderSettings
        from reverberate.mirror.stage import MirrorSettings, run_mirror

        position = _position_of(args)
        settings = MirrorSettings(
            ism=IsmSettings(
                max_order=args.order,
                flutter_order=args.flutter,
                window_s=args.window,
                max_images=args.max_images,
            ),
            rays=RaySettings(
                rays=args.rays,
                skip_specular_order=args.skip_specular,
                skip_window_s=args.window if args.skip_specular else 0.0,
            ),
            workers=args.workers,
            judge_every=args.judge_every,
            parameters=_parameters_of(args.parameters),
            signature=not args.no_signature,
            render=RenderSettings(
                band_limit_hz=args.band_limit,
                air_absorption=not args.no_air,
                tail_from_s=args.tail_from,
            ),
        )
        report = run_mirror(
            args.run,
            models=args.models,
            source={"name": args.source, "position": position},
            settings=settings,
            sound_speed_m_s=args.sound_speed,
            phase=args.phase,
            tag=args.tag,
        )
        print(
            json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=1, default=str)
        )
        return 0
    if args.command == "calibrate":
        if args.cpu:
            os.environ["REVERBERATE_NO_GPU"] = "1"
        from reverberate.mirror.rays import RaySettings
        from reverberate.mirror.render import RenderSettings
        from reverberate.mirror.stage import MirrorSettings
        from reverberate.mirror.tune import calibrate_run

        settings = MirrorSettings(
            rays=RaySettings(
                rays=args.rays,
                skip_specular_order=args.skip_specular,
                skip_window_s=args.window if args.skip_specular else 0.0,
            ),
            render=RenderSettings(tail_from_s=args.tail_from),
            parameters=_parameters_of(args.start),
            workers=args.workers,
            signature=not args.no_signature,
        )
        best, evaluations, target = calibrate_run(
            args.run,
            source={"name": args.source, "position": _position_of(args)},
            settings=settings,
            points=args.points,
            iterations=args.iterations,
            tied=args.tied,
            order=args.order,
            sound_speed_m_s=args.sound_speed,
            method=args.method,
            scattering=tuple(args.scattering),
        )
        print(json.dumps(best.record(), indent=1))
        print("wrote", target, "after", len(evaluations), "evaluations")
        return 0
    if args.command == "subset":
        from reverberate.mirror.stage import load_every
        from reverberate.mirror.tune import choose_points, write_reference_subset

        every = load_every(args.run / "mirror" / f"paths_{args.source}.npz")
        chosen = choose_points(every, args.points)
        target = write_reference_subset(
            args.run / "field" / f"{args.source}.h5",
            args.run / "mirror" / f"reference_subset_{args.source}.h5",
            chosen,
            args.order,
        )
        print("wrote", target, "points", chosen)
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
