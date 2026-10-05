"""The command line of the scene trace, on the laptop and on the machine.

On the laptop, the one command: plan, price, bundle, rent, run, fetch, destroy, verify:

python -m reverberate.trace rent --recipe R.json --home H
    (--mirror-from RUN/mirror [--mirror-source S1] [--calibration C.json] [--lead-s S --gain G]
     | --mirror DIR)
    [--models-from EXPORT/storey | --hssd-root DIR]
    [--dry-run] [--smoke SECONDS [--smoke-sources M] [--smoke-start T|auto]] [--patch [X Z]]
    [--low-engine lowband|pffdtd] [--low-ppw P] [--low-scheme cartesian|fcc]
    [--low-seconds S] [--rays N] [--rail-positions N]
    [--rate USD_PER_H] [--gpus N] [--max-hours H] [--hours H] [--max-dph D] [--gpu NAME]
    [--avoid ID ...] [--check full|read] [--no-fetch-pairs | --fetch-pairs]
    [--fetch-early] [--reuse-from HOME] [--publish-pairs] [--destroy-failed]
    [--allow-asset-mismatch] [--plan-offers] [--yes]

``--dry-run`` prints the plan and its cost and rents nothing. ``--plan-offers`` builds
the bundle, asks for the offers and prints what the run is predicted to take on each,
and rents nothing. The machine is the offer of the lowest predicted total within
``--max-hours`` of wall time, not the cheapest hour; ``--hours``, the watchdog, is taken
from that prediction unless given. Besides:

python -m reverberate.trace variants [--set FILE] --recipe R.json --window START|auto SECONDS
    --home DIR (--mirror-from RUN/mirror | --mirror DIR) [--sources M] [--pass "ARGS"] [--dry-run]
    a named set of cost variants (``trace/sets/listening_v1.json``): each one's predicted
    cost on the whole scene and on the excerpt, and the ``rent --smoke`` line that makes
    its pack under DIR/<name>, beside the ``variant.json`` the audit page reads
python -m reverberate.trace assets --mirror-from RUN/mirror ... --models-from EXPORT/storey
    the recipe's ``assets`` block as a trace of this dwelling finds it, for ``scenes generate``
python -m reverberate.trace bundle --recipe R.json --out B ...     the bundle alone
python -m reverberate.trace finish --home H [--publish-pairs]      the homecoming alone
python -m reverberate.trace run --bundle B --out O [--free-field] [--cpu] [--check full|read]
    the trace on this machine; ``--free-field`` replaces the solves by a monopole in free
    air, which is how the chain runs without a card, and on a card how every stage but
    the solve is timed without paying for one

``--check full`` reads the pack back whole and renders a minute of it on the host and on
the card (V4, with the proof of which device computed); ``read`` reads the pack's
structure. A smoke run checks in full and the whole scene reads, unless told. The pair
cache comes home with the whole scene and stays on the machine after a smoke run, unless
told: the pack holds every response a render reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mirror_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mirror", type=Path, default=None, help="a directory MirrorAssets.save wrote")
    p.add_argument("--mirror-from", type=Path, default=None, help="a run's mirror directory")
    p.add_argument("--mirror-source", default="S1", help="the source the mirror was aligned on")
    p.add_argument("--calibration", type=Path, default=None, help="a calibration json")
    p.add_argument("--lead-s", type=float, default=None, help="the alignment's lead, s")
    p.add_argument("--gain", type=float, default=None, help="the alignment's gain")


def _plan_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recipe", type=Path, required=True)
    _mirror_arguments(p)
    p.add_argument("--models-from", type=Path, default=None, help="an earlier export's storey")
    p.add_argument("--hssd-root", type=Path, default=None)
    p.add_argument("--smoke", type=float, default=None, metavar="SECONDS")
    p.add_argument("--smoke-sources", type=int, default=3, help="at most this many sources")
    p.add_argument(
        "--smoke-start", default="0", help="where the window starts, s, or auto: where most moves"
    )
    p.add_argument(
        "--patch",
        type=float,
        nargs="*",
        default=None,
        metavar="XZ",
        help="add the dense validation patch; its centre x z, or nothing for the nearest a surface",
    )
    p.add_argument(
        "--rail-positions",
        type=int,
        default=2,
        metavar="N",
        help="the solved positions a source on a rail reads: 2, weighted linearly (the "
        "default), or more, weighted per frequency, which a pitch over 0.08 m needs",
    )
    p.add_argument(
        "--low-engine",
        choices=("lowband", "pffdtd"),
        default="lowband",
        help="what solves the low band: the batched solver, or PFFDTD a source position",
    )
    p.add_argument(
        "--low-scheme",
        choices=("cartesian", "fcc"),
        default="cartesian",
        help="the batched solver's grid; another than the bundle's has its own keys",
    )
    p.add_argument(
        "--low-ppw",
        type=float,
        default=None,
        help="the batched solver's points per wavelength; left out, the validated grid's 10.5."
        " Another grid's pairs have their own keys, and its run its own --home",
    )
    p.add_argument(
        "--low-seconds",
        type=float,
        default=None,
        metavar="S",
        help="the seconds the batched solver simulates; left out, the pack's 1.2. A response"
        " solved for fewer is stored at the pack's length, faded to nothing over its last"
        " 20 ms; its pairs have their own keys, and its run its own --home",
    )
    p.add_argument(
        "--rays",
        type=int,
        default=None,
        metavar="N",
        help="the rays a tail site casts; left out, the mirror's own (100 000)",
    )
    p.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        metavar="HOME",
        help="an earlier run of the recipe: its early tables (--fetch-early) are not traced again",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help="USD an hour, for the estimate; left out, the rate of the card the low band's"
        " engine was measured on",
    )
    p.add_argument(
        "--check",
        choices=("full", "read"),
        default=None,
        help="the machine's check stage: both modules rendered, or the pack's structure read",
    )
    p.add_argument("--allow-asset-mismatch", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reverberate.trace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rent", help="a recipe to a pack on one rented machine")
    _plan_arguments(p)
    p.add_argument("--home", type=Path, required=True, help="where the run comes home")
    p.add_argument("--dry-run", action="store_true", help="the plan and its cost; nothing rented")
    p.add_argument(
        "--hours",
        type=float,
        default=None,
        help="the watchdog's cap, which cannot be extended; left out, the prediction for the"
        " offer taken, times 1.5 (2 for a card not measured), and half an hour",
    )
    p.add_argument("--max-dph", type=float, default=3.0)
    p.add_argument("--gpus", type=int, default=1, metavar="N", help="hosts of N cards or more")
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="wall time allowed: of the offers predicted within it, the lowest USD is rented",
    )
    p.add_argument(
        "--plan-offers",
        action="store_true",
        help="build the bundle, search the offers, say each one's predicted hours and USD;"
        " nothing rented",
    )
    p.add_argument(
        "--fetch-early",
        action="store_true",
        help="bring the early tables home, for a second run's --reuse-from",
    )
    p.add_argument(
        "--destroy-failed",
        action="store_true",
        help="a run that failed twice on its machine is fetched and destroyed, not kept",
    )
    p.add_argument("--gpu", default="", help="only cards whose name contains this")
    p.add_argument("--avoid", type=int, nargs="*", default=[], metavar="ID")
    p.add_argument("--instance", type=int, default=None, help="resume on a machine already rented")
    p.add_argument("--devices", default=None)
    p.add_argument("--campaign-args", default="", help="extra flags for the machine's command")
    p.add_argument("--fetch-grid", action="store_true", help="bring the low grid home too")
    p.add_argument("--publish-pairs", action="store_true", help="send the pair cache to the store")
    pairs = p.add_mutually_exclusive_group()
    pairs.add_argument(
        "--fetch-pairs",
        dest="fetch_pairs",
        action="store_true",
        default=None,
        help="bring the pair cache home (the whole scene's default)",
    )
    pairs.add_argument(
        "--no-fetch-pairs",
        dest="fetch_pairs",
        action="store_false",
        help="leave the pair cache on the machine (a smoke run's default)",
    )
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser(
        "variants", help="a set of cost variants: each one's price and its excerpt's command"
    )
    p.add_argument("--set", dest="variant_set", type=Path, default=None, help="a set, as JSON")
    p.add_argument("--recipe", type=Path, required=True, help="the reference recipe")
    _mirror_arguments(p)
    p.add_argument(
        "--window",
        nargs=2,
        required=True,
        metavar=("START", "SECONDS"),
        help="the excerpt every variant is traced on; START in seconds, or auto: where a"
        " source speaks on the move, passes the head, or is heard from another room",
    )
    p.add_argument("--home", type=Path, required=True, help="a directory a variant under it")
    p.add_argument("--sources", type=int, default=3, help="at most this many sources heard")
    p.add_argument("--rate", type=float, default=None, help="USD an hour, for the estimate")
    p.add_argument("--dry-run", action="store_true", help="print; write no file")
    p.add_argument(
        "--pass",
        dest="passed",
        default="",
        metavar="ARGS",
        help="words added to every command as they are: --models-from, --gpus, --max-hours",
    )

    p = sub.add_parser("bundle", help="the bundle of a trace, on the laptop")
    _plan_arguments(p)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("assets", help="the recipe's assets block as a trace finds it")
    _mirror_arguments(p)
    p.add_argument("--models-from", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=["voice_v1"], help="directivity models named")

    p = sub.add_parser("finish", help="what came home: cost stamped, pairs installed")
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--publish-pairs", action="store_true")

    p = sub.add_parser("run", help="the trace on this machine")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pffdtd", type=Path, default=Path("/root/pffdtd"))
    p.add_argument("--devices", default=None)
    p.add_argument("--cpu", action="store_true", help="numpy, even with a card")
    p.add_argument("--solvers", type=int, default=None)
    p.add_argument("--free-field", action="store_true", help="no solve: a monopole in free air")
    p.add_argument("--check", choices=("full", "read"), default=None)
    return parser


def _assets(args: argparse.Namespace) -> Any:
    from reverberate.trace.assets import MirrorAssets

    if (args.mirror is None) == (args.mirror_from is None):
        raise SystemExit("give the mirror as --mirror or as --mirror-from, not both")
    if args.mirror is not None:
        assets = MirrorAssets.load(args.mirror)
    else:
        assets = MirrorAssets.from_run(
            args.mirror_from,
            source=args.mirror_source,
            calibration=args.calibration,
            lead_s=args.lead_s,
            gain=args.gain,
        )
    return with_rays(assets, getattr(args, "rays", None))


def with_rays(assets: Any, rays: int | None) -> Any:
    """The mirror with another number of rays a tail site; as it is when none is given.

    A ray carries one part in ``rays`` of the source's energy, so a
    histogram of fewer rays is the same energy read with more noise, and
    nothing downstream is scaled.
    """
    from dataclasses import replace

    if rays is None or int(rays) == assets.settings.rays.rays:
        return assets
    if int(rays) < 1000:
        raise SystemExit(f"a tail site casts a thousand rays or more, not {rays}")
    settings = replace(assets.settings, rays=replace(assets.settings.rays, rays=int(rays)))
    return replace(assets, settings=settings)


def _rate(args: argparse.Namespace) -> float:
    """The hourly rate of the estimate: the one given, or the measured card's."""
    if args.rate is not None:
        return float(args.rate)
    if args.low_engine == "lowband":
        from reverberate.wave.lowband.pairs import MEASURED_RATE_USD_PER_HOUR

        return MEASURED_RATE_USD_PER_HOUR
    return 1.74


def _profile(args: argparse.Namespace, recipe: Any) -> tuple[Any, tuple[float, float] | None]:
    from reverberate.trace.plan import Profile, busiest_window

    centre = None
    if args.patch:
        if len(args.patch) != 2:
            raise SystemExit("--patch takes the centre's x and z, or nothing")
        centre = (float(args.patch[0]), float(args.patch[1]))
    if args.smoke is None:
        return Profile(patch=args.patch is not None, rail_positions=args.rail_positions), centre
    start = (
        busiest_window(recipe, args.smoke, args.smoke_sources)
        if args.smoke_start == "auto"
        else float(args.smoke_start)
    )
    return (
        Profile(
            seconds=args.smoke,
            sources=args.smoke_sources,
            patch=args.patch is not None,
            start_s=start,
            rail_positions=args.rail_positions,
        ),
        centre,
    )


def _variants(args: argparse.Namespace, recipe: Any, assets: Any) -> int:
    """The set priced on the whole scene and on the excerpt; a command a variant."""
    import shlex

    from reverberate.trace import variants
    from reverberate.wave.lowband.pairs import MEASURED_RATE_USD_PER_HOUR

    name, held = variants.load_set(args.variant_set)
    seconds = float(args.window[1])
    if args.window[0] == "auto":
        proposed = variants.listening_windows(recipe, seconds)
        print("windows proposed, the best first:")
        for window in proposed:
            print("  " + window.says())
        start = proposed[0].start_s
    else:
        start = float(args.window[0])
    # The mirror is said again in every command, as it was said here.
    mirror = []
    for flag in ("mirror", "mirror_from", "calibration", "lead_s", "gain"):
        value = getattr(args, flag)
        if value is not None:
            mirror += [f"--{flag.replace('_', '-')}", str(value)]
    if args.mirror_from is not None and args.mirror_source != "S1":
        mirror += ["--mirror-source", args.mirror_source]
    records = variants.price(
        held,
        recipe,
        args.recipe,
        assets.triangles,
        args.home,
        start_s=start,
        seconds=seconds,
        sources=args.sources,
        rate_usd_per_hour=(
            float(args.rate) if args.rate is not None else MEASURED_RATE_USD_PER_HOUR
        ),
        rays=int(assets.settings.rays.rays),
        passed=[*mirror, *shlex.split(args.passed)],
        write=not args.dry_run,
    )
    print(f"set {name}: {len(records)} variants on {seconds:g} s from {start:g} s")
    print(variants.table(records))
    if args.dry_run:
        print("dry run: no recipe and no variant.json written")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from reverberate.trace.run import Trace

        engine = None
        if args.free_field:
            import numpy as np

            from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
            from reverberate.trace.assets import MirrorAssets
            from reverberate.trace.engines import FreeFieldPairs

            held = args.bundle / "trace"
            with np.load(held / "plan.npz") as plan:
                cells = np.concatenate([plan["cells"], plan["patch_cells"].reshape(-1, 3)])
            mirror = MirrorAssets.load(held / "mirror")
            # In the cache form: on the geometric clock and the field's scale.
            engine = FreeFieldPairs(
                np.load(held / "positions.npy"),
                cells,
                args.out,
                sound_speed_m_s=mirror.settings.sound_speed_m_s,
                gain=FIELD_UNIT_AT_1M,
            )
        Trace(
            bundle=args.bundle,
            out=args.out,
            engine=engine,
            gpu=False if args.cpu else None,
            pffdtd_dir=args.pffdtd,
            card_devices=args.devices,
            solvers=args.solvers,
            check_mode=args.check,
        ).run()
        return 0
    if args.command == "finish":
        from reverberate.trace.driver import finish

        record = json.loads((args.home / "onebox.json").read_text())
        print(json.dumps(finish(args.home, record, publish_pairs=args.publish_pairs), indent=1))
        return 0
    if args.command == "assets":
        from reverberate.experiments.run import scene_spec
        from reverberate.spatial.lowband import solve_fmax_hz
        from reverberate.trace.assets import ROOMS_RULE, directivity_models
        from reverberate.trace.bundle import export_digest

        mirror = _assets(args)
        scene, _, _ = scene_spec(args.models_from, "apartment_full", solve_fmax_hz())
        models = directivity_models()
        print(
            json.dumps(
                {
                    "export_sha256": export_digest(args.models_from),
                    "voxel_low_key": scene.key,
                    "mirror_scene_key": mirror.catalogue.key,
                    "calibration_key": mirror.settings.parameters.key,
                    "directivity": {name: models[name].digest for name in args.models},
                    "rooms_rule": ROOMS_RULE,
                },
                indent=1,
            )
        )
        return 0

    from reverberate.scenes import load_recipe

    recipe = load_recipe(args.recipe)
    assets = _assets(args)
    if args.command == "variants":
        return _variants(args, recipe, assets)
    profile, centre = _profile(args, recipe)
    if args.command == "bundle":
        from reverberate.trace.bundle import build_bundle
        from reverberate.trace.driver import describe
        from reverberate.trace.plan import estimate, make_plan

        plan = make_plan(recipe, assets.triangles, profile, patch_centre_xz=centre)
        priced = estimate(
            plan,
            rate_usd_per_hour=_rate(args),
            low_engine=args.low_engine,
            check=args.check,
            low_ppw=args.low_ppw if args.low_engine == "lowband" else None,
            low_seconds=args.low_seconds,
            rays=args.rays,
        )
        print(describe(plan, priced))
        build_bundle(
            args.out,
            recipe,
            assets,
            plan,
            models_from=args.models_from,
            hssd_root=args.hssd_root,
            allow_asset_mismatch=args.allow_asset_mismatch,
            rate_usd_per_hour=_rate(args),
            low_engine=args.low_engine,
            low_scheme=args.low_scheme,
            low_ppw=args.low_ppw,
            low_seconds=args.low_seconds,
            reuse_from=args.reuse_from,
            check=args.check,
        )
        print(args.out)
        return 0
    from reverberate.trace.driver import launch

    result = launch(
        recipe,
        assets,
        args.home,
        profile=profile,
        patch_centre_xz=centre,
        models_from=args.models_from,
        hssd_root=args.hssd_root,
        rate_usd_per_hour=_rate(args),
        low_engine=args.low_engine,
        dry_run=args.dry_run,
        allow_asset_mismatch=args.allow_asset_mismatch,
        yes=args.yes,
        hours=args.hours,
        max_dph=args.max_dph,
        gpu=args.gpu,
        avoid=args.avoid,
        instance=args.instance,
        devices=args.devices,
        campaign_args=args.campaign_args,
        fetch_grid=args.fetch_grid,
        publish_pairs=args.publish_pairs,
        fetch_pairs=args.fetch_pairs,
        check=args.check,
        gpus=args.gpus,
        max_hours=args.max_hours,
        plan_offers=args.plan_offers,
        low_scheme=args.low_scheme,
        low_ppw=args.low_ppw,
        low_seconds=args.low_seconds,
        reuse_from=args.reuse_from,
        fetch_early=args.fetch_early,
        destroy_failed=args.destroy_failed,
    )
    # A rental that did not end in a pack is a failure of the command.
    outcome = dict(result.get("onebox", {})).get("outcome")
    return 0 if outcome in (None, "done") else 1
