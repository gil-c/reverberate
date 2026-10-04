"""The command line of the scene trace, on the laptop and on the machine.

On the laptop, the one command: plan, price, bundle, rent, run, fetch, destroy, verify:

python -m reverberate.trace rent --recipe R.json --home H
    (--mirror-from RUN/mirror [--mirror-source S1] [--calibration C.json] [--lead-s S --gain G]
     | --mirror DIR)
    [--models-from EXPORT/storey | --hssd-root DIR]
    [--dry-run] [--smoke SECONDS [--smoke-sources M] [--smoke-start T|auto]] [--patch [X Z]]
    [--low-engine lowband|pffdtd] [--rate USD_PER_H] [--hours H] [--max-dph D] [--gpu NAME]
    [--avoid ID ...]
    [--publish-pairs] [--allow-asset-mismatch] [--yes]

``--dry-run`` prints the plan and its cost and rents nothing. Besides:

python -m reverberate.trace assets --mirror-from RUN/mirror ... --models-from EXPORT/storey
    the recipe's ``assets`` block as a trace of this dwelling finds it, for ``scenes generate``
python -m reverberate.trace bundle --recipe R.json --out B ...     the bundle alone
python -m reverberate.trace finish --home H [--publish-pairs]      the homecoming alone
python -m reverberate.trace run --bundle B --out O [--free-field] [--cpu]
    the trace on this machine; ``--free-field`` replaces the solves by a monopole in free
    air, which is how the chain runs without a card
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
        "--low-engine",
        choices=("lowband", "pffdtd"),
        default="lowband",
        help="what solves the low band: the batched solver, or PFFDTD a source position",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=None,
        help="USD an hour, for the estimate; left out, the rate of the card the low band's"
        " engine was measured on",
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
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--max-dph", type=float, default=3.0)
    p.add_argument("--gpu", default="", help="only cards whose name contains this")
    p.add_argument("--avoid", type=int, nargs="*", default=[], metavar="ID")
    p.add_argument("--instance", type=int, default=None, help="resume on a machine already rented")
    p.add_argument("--devices", default=None)
    p.add_argument("--campaign-args", default="", help="extra flags for the machine's command")
    p.add_argument("--fetch-grid", action="store_true", help="bring the low grid home too")
    p.add_argument("--publish-pairs", action="store_true", help="send the pair cache to the store")
    p.add_argument("--yes", action="store_true")

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
    return parser


def _assets(args: argparse.Namespace) -> Any:
    from reverberate.trace.assets import MirrorAssets

    if (args.mirror is None) == (args.mirror_from is None):
        raise SystemExit("give the mirror as --mirror or as --mirror-from, not both")
    if args.mirror is not None:
        return MirrorAssets.load(args.mirror)
    return MirrorAssets.from_run(
        args.mirror_from,
        source=args.mirror_source,
        calibration=args.calibration,
        lead_s=args.lead_s,
        gain=args.gain,
    )


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
        return Profile(patch=args.patch is not None), centre
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
        ),
        centre,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from reverberate.trace.run import Trace

        engine = None
        if args.free_field:
            import numpy as np

            from reverberate.trace.assets import MirrorAssets
            from reverberate.trace.engines import FreeFieldPairs

            held = args.bundle / "trace"
            with np.load(held / "plan.npz") as plan:
                cells = np.concatenate([plan["cells"], plan["patch_cells"].reshape(-1, 3)])
            mirror = MirrorAssets.load(held / "mirror")
            engine = FreeFieldPairs(
                np.load(held / "positions.npy"),
                cells,
                args.out,
                sound_speed_m_s=mirror.settings.sound_speed_m_s,
                lead_s=mirror.pack_lead_s,
                gain=mirror.gain,
            )
        Trace(
            bundle=args.bundle,
            out=args.out,
            engine=engine,
            gpu=False if args.cpu else None,
            pffdtd_dir=args.pffdtd,
            card_devices=args.devices,
            solvers=args.solvers,
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
    profile, centre = _profile(args, recipe)
    if args.command == "bundle":
        from reverberate.trace.bundle import build_bundle
        from reverberate.trace.driver import describe
        from reverberate.trace.plan import estimate, make_plan

        plan = make_plan(recipe, assets.triangles, profile, patch_centre_xz=centre)
        priced = estimate(plan, rate_usd_per_hour=_rate(args), low_engine=args.low_engine)
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
        )
        print(args.out)
        return 0
    from reverberate.trace.driver import launch

    launch(
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
    )
    return 0
