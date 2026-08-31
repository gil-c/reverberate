"""The split pipeline from a shell: voxelise here, solve there.

Three verbs, in the order section 11 puts them:

    python -m reverberate.wave voxelise --model scene.json --mat-folder mats \\
        --fmax 2000                          # CPU, local, cached, no GPU
    python -m reverberate.wave comms --key KEY --source X Y Z --receiver X Y Z \\
        --duration 0.1 --out comms_out.h5    # per pair, milliseconds
    python -m reverberate.wave solve --key KEY --comms comms_out.h5 \\
        --host H --port P --out sim_outs.h5  # the rented half

``solve`` deliberately takes a machine rather than renting one: section 12.1
requires the rate and the total to be stated and agreed before any instance
exists, and a CLI that rents in passing would route around that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reverberate.wave.comms import write_comms
from reverberate.wave.remote import DEFAULT_PFFDTD_DIR, DEFAULT_REMOTE_DIR, Machine, solve
from reverberate.wave.voxelise import (
    CacheEntry,
    SceneSpec,
    cache_root,
    engine_inputs,
    voxelise,
)


def _entry_from_key(key: str) -> CacheEntry:
    path = cache_root() / key
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"no cache entry {key} under {cache_root()}")
    return CacheEntry(path=path, key=key, manifest=json.loads(manifest.read_text()))


def _voxelise(args: argparse.Namespace) -> int:
    model = json.loads(Path(args.model).read_text())
    mat_files = {label: f"{label}.h5" for label in model["mats_hash"]}
    spec = SceneSpec(
        model_json=Path(args.model),
        mat_folder=Path(args.mat_folder),
        mat_files=mat_files,
        fmax=args.fmax,
        ppw=args.ppw,
        fcc=args.fcc,
    )
    entry = voxelise(spec, nprocs=args.nprocs, force=args.force)
    print(json.dumps({"key": entry.key, "path": str(entry.path), **entry.manifest}, indent=2))
    return 0


def _comms(args: argparse.Namespace) -> int:
    entry = _entry_from_key(args.key)
    receivers = np.array(args.receiver, dtype=float).reshape(-1, 3)
    out = write_comms(
        entry.path,
        np.array(args.source, dtype=float),
        receivers,
        args.duration,
        sig_type=args.signal,
        diff_source=not args.double,
        out_path=Path(args.out),
    )
    print(json.dumps({"comms_out": str(out), "bytes": out.stat().st_size}, indent=2))
    return 0


def _solve(args: argparse.Namespace) -> int:
    entry = _entry_from_key(args.key)
    files = engine_inputs(entry, Path(args.comms))
    machine = Machine(
        host=args.host,
        port=args.port,
        user=args.user,
        identity=Path(args.identity) if args.identity else None,
    )
    result = solve(
        machine,
        files,
        Path(args.out),
        remote_dir=args.remote_dir,
        pffdtd_dir=args.pffdtd_dir,
        double_precision=args.double,
    )
    print(
        json.dumps(
            {
                "output": str(result.output),
                "uploaded_bytes": result.uploaded_bytes,
                "upload_s": result.upload_s,
                "engine_s": result.engine_s,
                "fetch_s": result.fetch_s,
                "total_s": round(result.total_s, 3),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reverberate.wave", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    vox = sub.add_parser("voxelise", help="voxelise a scene locally and cache it")
    vox.add_argument("--model", required=True, help="PFFDTD model JSON")
    vox.add_argument("--mat-folder", required=True, help="folder of fitted impedance files")
    vox.add_argument("--fmax", type=float, required=True)
    vox.add_argument("--ppw", type=float, default=10.5)
    vox.add_argument("--fcc", action="store_true")
    vox.add_argument("--nprocs", type=int, default=None)
    vox.add_argument("--force", action="store_true", help="re-voxelise even if cached")
    vox.set_defaults(func=_voxelise)

    comms = sub.add_parser("comms", help="place one source and its receivers on a cached grid")
    comms.add_argument("--key", required=True, help="cache key from voxelise")
    comms.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    comms.add_argument(
        "--receiver",
        type=float,
        nargs="+",
        required=True,
        metavar="X Y Z",
        help="one or more receivers, as a flat list of coordinates",
    )
    comms.add_argument("--duration", type=float, required=True)
    comms.add_argument("--signal", default="impulse")
    comms.add_argument("--double", action="store_true", help="double precision engine")
    comms.add_argument("--out", required=True)
    comms.set_defaults(func=_comms)

    run = sub.add_parser("solve", help="ship the four files to a rented machine and run")
    run.add_argument("--key", required=True)
    run.add_argument("--comms", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--port", type=int, default=22)
    run.add_argument("--user", default="root")
    run.add_argument("--identity", default=None)
    run.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    run.add_argument("--pffdtd-dir", default=DEFAULT_PFFDTD_DIR)
    run.add_argument("--double", action="store_true")
    run.add_argument("--out", required=True)
    run.set_defaults(func=_solve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
