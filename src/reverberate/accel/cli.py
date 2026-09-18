"""The command line of the accelerated stages, on the machine and on the laptop.

python -m reverberate.accel campaign --bundle B --out O [--pffdtd DIR] [--devices 0,1] [--cpu]
python -m reverberate.accel bundle --out B --hssd-root H --scene-id S --sources sources.json
python -m reverberate.accel voxelise --model M --materials DIR --fmax F --out DIR [--cpu]
python -m reverberate.accel compare-entries A B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverberate.accel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("campaign", help="run every stage of a campaign on this machine")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pffdtd", type=Path, default=Path("/root/pffdtd"))
    p.add_argument("--devices", default=None, help="CUDA_VISIBLE_DEVICES for the engine")
    p.add_argument("--cpu", action="store_true", help="run the numpy twins, no card")
    p.add_argument("--workers", type=int, default=None, help="assembly workers")
    p.add_argument(
        "--solve-bands", default=None, help="comma separated bands to solve, for a flow test"
    )
    p.add_argument("--no-mirror", action="store_true", help="skip the geometric mirror stage")
    p.add_argument(
        "--mirror-parameters",
        type=Path,
        default=None,
        help="a calibration json to render the mirror with (mirror/calibration/<key>.json)",
    )

    p = sub.add_parser("bundle", help="prepare what the machine needs, on the laptop")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hssd-root", type=Path, required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--pitch", type=float, default=0.40)
    p.add_argument("--height", type=float, default=1.70)
    p.add_argument("--ram-gb", type=float, default=400.0)
    p.add_argument(
        "--models-from", type=Path, default=None, help="reuse an earlier export's directory"
    )

    p = sub.add_parser("voxelise", help="one grid on this machine, as a cache entry's files")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--materials", type=Path, required=True, help="directory of impedance files")
    p.add_argument("--fmax", type=float, required=True)
    p.add_argument("--nh", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-seal", action="store_true")

    p = sub.add_parser("selfcheck", help="encode a few points on the CPU path and compare")
    p.add_argument("--job", type=Path, required=True)

    p = sub.add_parser("compare-entries", help="two cache entries, dataset by dataset")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "campaign":
        from reverberate.accel.campaign import run_campaign

        run_campaign(
            args.bundle,
            args.out,
            pffdtd_dir=args.pffdtd,
            devices=args.devices,
            gpu=False if args.cpu else None,
            workers=args.workers,
            solve_bands=tuple(args.solve_bands.split(",")) if args.solve_bands else None,
            mirror_field=not args.no_mirror,
            mirror_parameters=args.mirror_parameters,
        )
        return 0
    if args.command == "bundle":
        from reverberate.accel.bundle import prepare_bundle

        record = prepare_bundle(
            args.out,
            hssd_root=args.hssd_root,
            scene_id=args.scene_id,
            sources=json.loads(args.sources.read_text()),
            pitch_m=args.pitch,
            height_m=args.height,
            ram_gb=args.ram_gb,
            models_from=args.models_from,
        )
        print(json.dumps({k: v for k, v in record.items() if k != "labels"}, indent=1))
        return 0
    if args.command == "voxelise":
        import numpy as np

        from reverberate.accel.backend import xp_for
        from reverberate.accel.voxelise import voxelise_scene
        from reverberate.wave.remote_voxelise import grid_shape_of
        from reverberate.wave.voxelise import nh_for

        labels = sorted(json.loads(args.model.read_text())["mats_hash"])
        mat_files = {label: f"{label}.h5" for label in labels if label != "_RIGID"}
        nh = args.nh or nh_for(grid_shape_of(args.model, args.fmax, 10.5))
        voxelised = voxelise_scene(
            args.model,
            args.out,
            mat_folder=args.materials,
            mat_files=mat_files,
            fmax=args.fmax,
            ppw=10.5,
            nh=nh,
            xp=np if args.cpu else xp_for(),
            say=print,
            seal_pockets=not args.no_seal,
        )
        print(json.dumps(voxelised.record(), indent=1))
        return 0
    if args.command == "selfcheck":
        import numpy as np

        from reverberate.accel.encode import encode_band
        from reverberate.accel.verify import compare_encoded

        job = json.loads(args.job.read_text())
        plan = json.loads(Path(job["plan"]).read_text())
        encode_band(
            plan,
            job["band"],
            job["source"],
            pressure=Path(job["pressure"]),
            comms=Path(job["comms"]),
            positions=np.load(job["positions"]),
            output=Path(job["cpu_encoded"]),
            xp=np,
            rows=job["rows"],
            say=print,
        )
        import time

        # The card's file: the band's merged encoding, or a slice's part while
        # the campaign still holds it. Written after the CPU path started, so
        # wait for it, up to an hour.
        gpu_encoded = Path(job["gpu_encoded"])
        merged = gpu_encoded.with_name(gpu_encoded.name.split(".part")[0] + ".h5")
        deadline = time.time() + 3600
        while not gpu_encoded.is_file() and not merged.is_file() and time.time() < deadline:
            time.sleep(15)
        found = gpu_encoded if gpu_encoded.is_file() else merged
        # The campaign may still be writing the merged file: HDF5 refuses to
        # open a file another process holds locked. Wait, up to an hour.
        while True:
            try:
                report = compare_encoded(found, Path(job["cpu_encoded"]))
                break
            except OSError as busy:
                if time.time() > deadline:
                    raise
                print(f"waiting for {found}: {busy}", flush=True)
                time.sleep(15)
        report["gpu_encoded"] = str(found)
        report["band"] = job["band"]
        Path(job["output"]).write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "per_point_max_over_peak"}))
        return 0
    if args.command == "compare-entries":
        from reverberate.accel.voxelise import entries_identical

        report = entries_identical(args.a, args.b)
        print(json.dumps(report, indent=1))
        return 0 if report["identical"] else 1
    return 2
