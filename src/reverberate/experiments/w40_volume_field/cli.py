"""The command line: one campaign, or one of its stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reverberate.experiments.w40_volume_field.assemble import assemble_field
from reverberate.experiments.w40_volume_field.campaign import DURATIONS_S, campaign
from reverberate.experiments.w40_volume_field.encode import encode_sharded
from reverberate.experiments.w40_volume_field.plan import plan_field, prepare_field
from reverberate.experiments.w40_volume_field.solve import solve_on_card


def _meshes(text: str | None) -> dict[str, str] | None:
    return json.loads(text) if text else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="w40_volume_field", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def rental(p: argparse.ArgumentParser, prefix: str, *, hours: float, max_dph: float) -> None:
        p.add_argument(
            f"--{prefix}hours",
            type=float,
            default=hours,
            help="cap on the rental; the watchdog destroys the machine after it",
        )
        p.add_argument(
            f"--{prefix}max-dph",
            type=float,
            default=max_dph,
            help="USD per hour the search accepts",
        )

    p = sub.add_parser("plan", help="choose the grid, place every array, write plan.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hssd-root", type=Path, required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument(
        "--sources", type=Path, required=True, help="JSON list of {name, room, position}"
    )
    p.add_argument("--low", required=True, help="cache key of the storey at 1 kHz")
    p.add_argument("--mid", required=True, help="cache key of the storey at 4 kHz")
    p.add_argument(
        "--high",
        default=None,
        help="cache key of the storey at 8 kHz: one high band over every point",
    )
    p.add_argument(
        "--room",
        nargs=2,
        action="append",
        metavar=("ROOM", "KEY"),
        default=[],
        help="a room's own 8 kHz grid, when the storey does not fit a card",
    )
    p.add_argument("--height", type=float, default=1.70)
    p.add_argument(
        "--ram-gb",
        type=float,
        required=True,
        help="the points cap; a band beyond it is solved in slices",
    )
    p.add_argument("--pitch", type=float, default=None)
    p.add_argument("--order", type=int, default=7)
    p.add_argument("--fit-order", type=int, default=10)

    p = sub.add_parser("prepare", help="one comms file per source and band")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("solve", help="every solve on one card; the pressure stays on its host")
    p.add_argument("--out", type=Path, required=True)
    rental(p, "", hours=4.0, max_dph=2.1)
    p.add_argument("--instance", type=int, default=None, help="resume on this instance")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument(
        "--slice-ram-gb",
        type=float,
        default=180.0,
        help="host RAM one solve must fit; bigger outputs are sliced",
    )
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("encode", help="push the shards to fast-core boxes, encode, merge, assemble")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-instance", type=int, required=True)
    rental(p, "", hours=8.0, max_dph=0.45)
    p.add_argument("--min-cores", type=int, default=12)
    p.add_argument("--min-cpu-ghz", type=float, default=3.0)
    p.add_argument("--shards", type=int, default=4, help="boxes, each on a slice of the points")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser(
        "assemble", help="from the merged encodings to field/<source>.h5 and walk.json"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ceiling", type=float, default=None)
    p.add_argument(
        "--meshes",
        default=None,
        help='JSON {"1000": path, ...} of viewer meshes, when there are some',
    )

    p = sub.add_parser("campaign", help="all of the above, resuming from what exists")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hssd-root", type=Path, required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument(
        "--models", type=Path, required=True, help="the export directory, made if missing"
    )
    p.add_argument(
        "--low",
        default=None,
        help="cache keys of the grids when already cached; else they are voxelised",
    )
    p.add_argument("--mid", default=None)
    p.add_argument("--high", default=None)
    p.add_argument("--voxelise-hours", type=float, default=4.0)
    p.add_argument("--ram-gb", type=float, default=400.0)
    p.add_argument("--pitch", type=float, default=0.40)
    rental(p, "solve-", hours=4.0, max_dph=2.1)
    p.add_argument("--slice-ram-gb", type=float, default=180.0)
    rental(p, "encode-", hours=8.0, max_dph=0.45)
    p.add_argument("--min-cores", type=int, default=12)
    p.add_argument("--min-cpu-ghz", type=float, default=3.0)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan_field(
            args.out,
            hssd_root=args.hssd_root,
            scene_id=args.scene_id,
            sources=json.loads(args.sources.read_text()),
            storey_keys={"low": args.low, "mid": args.mid},
            room_keys=dict(args.room),
            durations_s=DURATIONS_S,
            height_m=args.height,
            ram_gb=args.ram_gb,
            pitch_m=args.pitch,
            order=args.order,
            fit_order=args.fit_order,
            high_key=args.high,
        )
    elif args.command == "prepare":
        prepare_field(args.out)
    elif args.command == "solve":
        solve_on_card(
            args.out,
            hours=args.hours,
            max_dph=args.max_dph,
            yes=args.yes,
            sources=args.sources,
            instance=args.instance,
            slice_ram_gb=args.slice_ram_gb,
        )
    elif args.command == "encode":
        encode_sharded(
            args.out,
            gpu_instance=args.gpu_instance,
            shards=args.shards,
            hours=args.hours,
            max_dph=args.max_dph,
            min_cores=args.min_cores,
            min_cpu_ghz=args.min_cpu_ghz,
            yes=args.yes,
        )
    elif args.command == "assemble":
        assemble_field(
            args.out,
            sources=args.sources,
            workers=args.workers,
            ceiling_hz=args.ceiling,
            meshes=_meshes(args.meshes),
        )
    elif args.command == "campaign":
        campaign(
            args.out,
            hssd_root=args.hssd_root,
            scene_id=args.scene_id,
            sources_file=args.sources,
            models=args.models,
            keys={"low": args.low, "mid": args.mid, "high": args.high}
            if args.low and args.mid and args.high
            else None,
            voxelise_hours=args.voxelise_hours,
            ram_gb=args.ram_gb,
            pitch_m=args.pitch,
            solve_hours=args.solve_hours,
            solve_max_dph=args.solve_max_dph,
            slice_ram_gb=args.slice_ram_gb,
            encode_hours=args.encode_hours,
            encode_max_dph=args.encode_max_dph,
            min_cores=args.min_cores,
            min_cpu_ghz=args.min_cpu_ghz,
            shards=args.shards,
            attempts=args.attempts,
        )
    return 0
