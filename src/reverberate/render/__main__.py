"""``python -m reverberate.render``: time the engine, validate, compact or check a pack."""

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
    smaller = commands.add_parser(
        "compact", help="rewrite a pack with its low band responses in fewer bytes"
    )
    smaller.add_argument("pack", type=Path)
    smaller.add_argument("out", type=Path)
    smaller.add_argument(
        "--levers",
        default="",
        help="which, between commas: bins (the transform's bins under the crossover's top),"
        " int16 (those bins in 16 bits), degree=DB (each degree from where it holds that share"
        " of the energy within 0.30 m of the cell), decay=DB (each degree cut that far under"
        " the pressure's loudest moment)."
        " None: the pack's own bytes",
    )
    sealed = commands.add_parser("unseal", help="which variant each file of a blind set is")
    sealed.add_argument("key", type=Path, help="blind/key.sealed of a check of several packs")
    sound = commands.add_parser("check", help="check a rendered scene's sound by measurement")
    sound.add_argument("pack", type=Path)
    sound.add_argument(
        "--recipe", type=Path, help="a recipe to read the clips from, not the pack's"
    )
    sound.add_argument(
        "--clips", type=Path, help="the clip libraries; <data root>/clips unless said"
    )
    sound.add_argument("--manifest", type=Path, help="the library a placeholder is stood in from")
    sound.add_argument(
        "--out", type=Path, default=Path("sound-check"), help="where the report goes"
    )
    sound.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"), help="seconds")
    sound.add_argument("--sources", nargs="+", help="the sources checked; all of them unless said")
    sound.add_argument("--measured-head", type=Path, help="the SOFA head the page decodes with")
    sound.add_argument("--reference", type=Path, help="the validated hybrid field of the dwelling")
    sound.add_argument(
        "--reference-point",
        action="store_true",
        help="only this: a source that stands on the field's own, the head on a lattice point, "
        "against the field there, a third octave at a time",
    )
    sound.add_argument(
        "--against",
        type=Path,
        nargs="+",
        metavar="PACK",
        help="other packs of the same scene. One: the two side by side, as files to hear at"
        " one gain and their difference by third octave (ab.md). Several, or with --names:"
        " a file for each at one gain, each one's difference from the first, and a blind"
        " set (variants.md)",
    )
    sound.add_argument(
        "--names",
        nargs="+",
        help="with --against: the packs' names, the first pack's first; left out, each is"
        " named by the variant.json beside it",
    )
    sound.add_argument("--blind-seed", type=int, help="the blind set's order, for a test")
    sound.add_argument("--probe-seconds", type=float, default=5.0, help="of each steady probe")
    sound.add_argument("--workers", type=int, default=-1, help="threads of the transforms")
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
    elif args.command == "compact":
        from reverberate.render.compact import Levers, compact_pack

        print(json.dumps(compact_pack(args.pack, args.out, Levers.parse(args.levers), say=print)))
    elif args.command == "unseal":
        from reverberate.render.check.many import unseal

        for hidden, name in sorted(unseal(args.key).items()):
            print(f"{hidden}: {name}")
    elif args.command == "check":
        from reverberate.render.check.report import defaults, exit_code, run
        from reverberate.render.check.run import CheckSettings

        found = defaults(args.clips, args.manifest, args.measured_head, args.reference)
        if args.reference_point:
            from reverberate.render.check import reference
            from reverberate.render.pack import read_pack

            assert found["reference"] is not None
            with read_pack(args.pack) as pack:
                reference.run(
                    pack, found["reference"], args.out, sources=args.sources, workers=args.workers
                )
            return 0
        if args.against is not None and (len(args.against) > 1 or args.names):
            from reverberate.render.check import many

            many.run(
                args.pack,
                args.against,
                args.out,
                names=args.names,
                recipe_path=args.recipe,
                clips_root=found["clips_root"],
                manifest=found["manifest"],
                window_s=None if args.window is None else (args.window[0], args.window[1]),
                sources=args.sources,
                measured_head=found["measured_head"],
                settings=CheckSettings(probe_seconds=args.probe_seconds, workers=args.workers),
                blind_seed=args.blind_seed,
            )
            return 0
        if args.against is not None:
            from reverberate.render.check import against

            against.run(
                args.pack,
                args.against[0],
                args.out,
                recipe_path=args.recipe,
                clips_root=found["clips_root"],
                manifest=found["manifest"],
                window_s=None if args.window is None else (args.window[0], args.window[1]),
                sources=args.sources,
                measured_head=found["measured_head"],
                settings=CheckSettings(probe_seconds=args.probe_seconds, workers=args.workers),
            )
            return 0
        document = run(
            args.pack,
            args.out,
            recipe_path=args.recipe,
            window_s=None if args.window is None else (args.window[0], args.window[1]),
            sources=args.sources,
            settings=CheckSettings(probe_seconds=args.probe_seconds, workers=args.workers),
            clips_root=found["clips_root"],
            manifest=found["manifest"],
            measured_head=found["measured_head"],
            reference=found["reference"],
        )
        return exit_code(document)
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
