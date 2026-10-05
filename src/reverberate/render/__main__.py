"""``python -m reverberate.render``: time the engine; validate, compact, level or check a pack."""

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
    level = commands.add_parser(
        "relevel",
        help="steady a pack's levelling scalar along what moves, IN PLACE; the trace's own "
        "table is kept in the pack and --undo puts it back",
    )
    level.add_argument("pack", type=Path)
    level.add_argument(
        "--seconds", type=float, default=2.0, help="averaged over this long either side"
    )
    level.add_argument("--undo", action="store_true", help="put the trace's table back")
    whole = commands.add_parser(
        "mix", help="render a pack's whole mix to <out>.f32 and <out>.json, in several processes"
    )
    whole.add_argument("pack", type=Path)
    whole.add_argument("out", type=Path)
    whole.add_argument("--processes", type=int, help="how many render; the cores less two unless")
    whole.add_argument("--clips", type=Path, help="the clip libraries; <data root>/clips unless")
    whole.add_argument("--manifest", type=Path, help="the library a placeholder is stood in from")
    whole.add_argument("--scratch", type=Path, help="where the sources' noise is kept meanwhile")
    join = commands.add_parser(
        "seam",
        help="give a pack its level a band above the crossover, the tapered join, IN PLACE: "
        "the pair's own seam at the crossover, the scene's one number above; the trace's "
        "scalar is kept and --undo removes the table",
    )
    join.add_argument("pack", type=Path)
    join.add_argument(
        "--taper",
        type=float,
        nargs="+",
        default=None,
        metavar="SHARE",
        help="the share of a pair's own seam a band keeps, from the crossover's band up; the "
        "last holds above (1 0.5 0: 1 kHz whole, 2 kHz half, 4 kHz and over none)",
    )
    join.add_argument(
        "--constant-db",
        type=float,
        default=None,
        metavar="DB",
        help="the scene's one number over the alignment's gain; left out, the median of the "
        "pack's pairs' seams",
    )
    join.add_argument("--undo", action="store_true", help="remove the table")
    join.add_argument("--dry-run", action="store_true", help="write nothing: say what it does")
    loud = commands.add_parser(
        "gains",
        help="set sources' gains in a traced pack, IN PLACE; the trace's are kept and --undo "
        "puts them back; the provenance says which differ from the recipe's",
    )
    loud.add_argument("pack", type=Path)
    loud.add_argument(
        "--set", nargs="+", default=[], metavar="ID=DB", help="a source's gain, in dB"
    )
    loud.add_argument("--kind", help="with --add: the sources of this kind")
    loud.add_argument(
        "--add", type=float, metavar="DB", help="with --kind: added to the gain the trace wrote"
    )
    loud.add_argument("--undo", action="store_true", help="put the trace's gains back")
    loud.add_argument("--dry-run", action="store_true", help="write nothing: say what it does")
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
    elif args.command == "mix":
        from reverberate.render.check.report import defaults
        from reverberate.render.mix import main_factory, write_mix

        found = defaults(args.clips, args.manifest, None, None)
        header = write_mix(
            args.out,
            main_factory(args.pack, found["clips_root"], found["manifest"]),
            processes=args.processes,
            scratch=args.scratch,
            say=lambda text: print(text, flush=True),
        )
        print(json.dumps(header["render"], indent=1))
    elif args.command == "relevel":
        from reverberate.render.relevel import relevel_pack

        print(json.dumps(relevel_pack(args.pack, seconds=args.seconds, undo=args.undo), indent=1))
    elif args.command == "seam":
        from reverberate.render.seam import SEAM_CONSTANT_DB, TAPER, taper_pack

        said = taper_pack(
            args.pack,
            taper=TAPER if args.taper is None else tuple(args.taper),
            constant_db=SEAM_CONSTANT_DB if args.constant_db is None else args.constant_db,
            undo=args.undo,
            dry_run=args.dry_run,
        )
        print(json.dumps(said, indent=1))
    elif args.command == "gains":
        from reverberate.render.gains import set_gains

        if (args.kind is None) != (args.add is None):
            parser.error("--kind and --add go together")
        named = dict(item.split("=", 1) for item in args.set)
        said = set_gains(
            args.pack,
            gains_db={name: float(value) for name, value in named.items()},
            add_db={} if args.kind is None else {args.kind: args.add},
            undo=args.undo,
            dry_run=args.dry_run,
        )
        print(json.dumps(said, indent=1))
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
