"""Voxelise one scene several ways and compare what the engine would read.

Driven by ``scripts/check_slabs.sh``; that file says what it is for and carries
the numbers it produced. Kept as a file rather than inlined as a heredoc so it
can be read, linted and edited like the rest of the project.

Datasets are compared, not file bytes: a slabbed run appends to resizable HDF5
datasets while a single pass writes contiguous ones, so the containers differ
while every number in them is the same.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.wave.voxelise import pffdtd_dir, pffdtd_python

DATASETS = ("bn_ixyz", "adj_bn", "mat_bn", "saf_bn", "xv", "yv", "zv", "h", "Nx", "Ny", "Nz", "Nb")


def run(model: str, mats: str, fmax: float, out: Path, tag: str, **extra: Any) -> Path:
    """One voxelisation into ``out/tag``, reusing it if it is already there."""
    directory = out / tag
    if (directory / "vox_out.h5").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    labels = sorted(json.loads(Path(model).read_text())["mats_hash"])
    job: dict[str, Any] = {
        "pffdtd_dir": str(pffdtd_dir()),
        "out_dir": str(directory),
        "model_json": model,
        "mat_folder": mats,
        "mat_files": {label: f"{label}.h5" for label in labels},
        "fmax": fmax,
        "ppw": 10.5,
        "Tc": 20.0,
        "rh": 50.0,
        "fcc": False,
        "bmin": None,
        "bmax": None,
        "rot_az_el": [0.0, 0.0],
        "nprocs": 8,
        "compress": None,
        "slabs": 1,
        "nh": None,
        "nvox_est": None,
    }
    job.update(extra)
    wave = Path(importlib.import_module("reverberate.wave.voxelise").__file__ or "").parent
    started = time.perf_counter()
    finished = subprocess.run(
        [pffdtd_python(), str(wave / "_child_voxelise.py")],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
    )
    if finished.returncode != 0:
        print(finished.stderr[-2500:])
        raise SystemExit(f"{tag} failed")
    marked = [line for line in finished.stdout.splitlines() if "@@REVERBERATE@@" in line]
    report = json.loads(marked[0].split("@@REVERBERATE@@")[1]) if marked else {}
    print(
        f"  {tag:22s} {time.perf_counter() - started:6.1f}s  "
        f"nodes {report.get('boundary_nodes', 0):>11,}  "
        f"adj {report.get('timings', {}).get('vox_scene_adj_s', 0):7.1f}s",
        flush=True,
    )
    return directory


def load(directory: Path) -> dict[str, Any]:
    with h5py.File(directory / "vox_out.h5", "r") as handle:
        return {name: handle[name][...] for name in DATASETS}


def main(argv: list[str]) -> int:
    model, mats, fmax, out = argv[0], argv[1], float(argv[2]), Path(argv[3])
    cases = [json.loads(case) for case in argv[4:]]
    base = load(run(model, mats, fmax, out, "single"))
    agreed = True
    for case in cases:
        tag = "_".join(f"{key}{value}" for key, value in case.items())
        other = load(run(model, mats, fmax, out, tag, **case))
        differ = [name for name in base if not np.array_equal(base[name], other[name])]
        print(f"  {tag:22s} -> {'IDENTICAL' if not differ else 'DIFFERS in ' + ','.join(differ)}")
        agreed &= not differ
    print("ALL IDENTICAL" if agreed else "MISMATCH")
    return 0 if agreed else 1


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main(sys.argv[1:]))
