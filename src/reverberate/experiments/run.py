"""Execute one run: a domain, a grid, a source and receiver, and a solver.

Promoted from ``data/runs/b0_truncation/b0_run.py`` and
``data/runs/w2_lateral_margin/w2_run.py``, which were the same program written
twice. Both chose a domain, put the four files the engine reads in a directory,
ran a binary there, timed it and wrote a JSON record. They differed only in
where the four files came from, so that is the only thing this module still
distinguishes: :func:`run_scene` starts from an exported scene and its
manifest, :func:`run_prepared` starts from a voxelisation cache key. Both end
in :func:`execute`.

**Bounds are chosen explicitly, never implicitly.** This is W1's result and it
is kept as a required argument rather than a default. The first B0 run snapped
each truncated domain independently and left the artificial absorbing box close
to the receiver, which produced a constant 2.1e-13 relative offset and a
departure at 11.5 ms that did not scale with the cut. Both defects were bounds
defects, so bounds are named:

- ``common``: every scene gets the reference scene's exact bounding box, so the
  truncated runs differ from the reference in nothing but which triangles are
  present. Interpolation weights are then bit identical by construction and the
  artificial boundary sits exactly where the reference's own outer boundary
  sits. This is the mode that makes bit exactness a meaningful question.
- ``padded``: the scene's own box grown by ``pad_m`` metres, defaulting to the
  cut, then clipped to the reference box and snapped outwards to whole cells
  from the reference origin. Cheaper than ``common``, and enough to push the
  artificial boundary further from the receiver than the cut, so it cannot be
  the first thing to arrive.

The chosen mode travels with the numbers in :class:`Bounds`, appears in the run
directory's name, and is recorded in the run's JSON. There is no code path that
picks one for you.

**PFFDTD is called, not absorbed.** ``b0_run.py`` imported ``sim_setup`` into
its own interpreter, which forced the whole experiment onto numpy below 2.
Voxelisation now goes through :func:`reverberate.wave.voxelise.voxelise`, which
already runs PFFDTD in its own interpreter and caches the result, and the
source and receiver through :func:`reverberate.wave.comms.write_comms`. Nothing
in :mod:`reverberate.wave` is modified or copied here.

    python -m reverberate.experiments.run scene --models MODELS --out OUT \\
        --scene apartment_cut10m --fmax 2000 --duration 0.1 --bounds common

    python -m reverberate.experiments.run prepared --key KEY --out OUT \\
        --comms comms_out.h5 --engine cpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from reverberate.experiments.engine import Engine, run_binary, sim_consts, write_record
from reverberate.wave import (
    ENGINE_FILES,
    CacheEntry,
    Machine,
    SceneSpec,
    engine_inputs,
    solve,
    voxelise,
    write_comms,
)
from reverberate.wave.voxelise import cache_root, pffdtd_dir, pffdtd_python

__all__ = [
    "Bounds",
    "BoundsMode",
    "RESULTS_FILE",
    "build_materials",
    "choose_bounds",
    "entry_from_key",
    "execute",
    "grid_step",
    "main",
    "run_prepared",
    "run_scene",
    "scene_bounds",
    "snap_bounds",
    "sound_speed",
]

#: The two modes W1 settled on. There is deliberately no default anywhere.
BoundsMode = Literal["common", "padded"]

#: One line per run, appended, read by :mod:`reverberate.experiments.compare`.
RESULTS_FILE = "runs.jsonl"


@dataclass(frozen=True)
class Bounds:
    """A domain and the reason it has those corners.

    ``mode`` is carried with ``bmin`` and ``bmax`` rather than left at the call
    site, so a pair of bounds can never be recorded without saying how it was
    arrived at. That is the whole point of W1.
    """

    mode: BoundsMode
    bmin: np.ndarray
    bmax: np.ndarray
    pad_m: float | None

    def as_spec(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The corners as :class:`reverberate.wave.SceneSpec` wants them."""
        lo = [float(v) for v in self.bmin]
        hi = [float(v) for v in self.bmax]
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError("bounds must be three dimensional")
        return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def sound_speed(temperature_c: float = 20.0) -> float:
    """PFFDTD's own sound speed, so ``h`` matches what ``SimConsts`` computes."""
    return float(343.2 * np.sqrt(temperature_c / 20.0))


def grid_step(fmax: float, ppw: float, temperature_c: float = 20.0) -> float:
    """``h = c / (fmax x PPW)``, copied from ``SimConsts``."""
    return float(sound_speed(temperature_c) / (fmax * ppw))


def snap_bounds(
    lo: np.ndarray, hi: np.ndarray, ref_lo: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grow ``lo``/``hi`` outwards to whole cells measured from ``ref_lo``.

    Only the low bound decides where the nodes fall: ``CartGrid`` puts its
    origin at ``bmin - 3.5h`` and steps from there, so aligning ``bmin`` to a
    whole number of cells is what makes two grids sample the same points. The
    high bound only fixes how many cells there are, and it is deliberately
    offset by half a cell: an extent that is an exact multiple of the step
    trips ``CartGrid``'s ``assert np.all(xyzmax>=xyzmax0)`` on a rounding hair.
    """
    lo_cells = np.floor((lo - ref_lo) / step) - 1.0
    hi_cells = np.ceil((hi - ref_lo) / step) + 1.5
    return ref_lo + lo_cells * step, ref_lo + hi_cells * step


def choose_bounds(
    mode: BoundsMode,
    scene_lo: np.ndarray,
    scene_hi: np.ndarray,
    ref_lo: np.ndarray,
    ref_hi: np.ndarray,
    step: float,
    pad_m: float,
) -> Bounds:
    """The domain a truncated scene is simulated on, per W1's two fixes.

    ``common`` hands back the reference box untouched, which is the strongest
    form of the experiment: identical grid, identical origin, identical outer
    boundary, and the only difference between the runs is the deleted
    triangles.

    ``padded`` grows the scene's own box by ``pad_m`` so the artificial
    boundary is further from the receiver than the cut, clips it to the
    reference box because there is nothing to simulate beyond it, and snaps it
    outwards to whole cells from the reference origin so shared nodes stay
    exactly where the reference put them.
    """
    if mode == "common":
        return Bounds(mode="common", bmin=ref_lo.copy(), bmax=ref_hi.copy(), pad_m=None)
    if mode != "padded":
        raise ValueError(f"unknown bounds mode {mode!r}")
    lo = np.maximum(scene_lo - pad_m, ref_lo)
    hi = np.minimum(scene_hi + pad_m, ref_hi)
    bmin, bmax = snap_bounds(lo, hi, ref_lo, step)
    # Snapping can push the box a cell outside the reference, which is harmless
    # for the physics but would leave the padded run with nodes the reference
    # never had. Clamping the low bound would break the whole cell alignment,
    # so it is left alone; RoomGeo widens bounds anyway, never narrows them.
    offset_cells = (bmin - ref_lo) / step
    if not np.allclose(offset_cells, np.round(offset_cells), atol=1e-9):
        raise AssertionError(f"padded bounds are off the reference grid: {offset_cells}")
    return Bounds(mode="padded", bmin=bmin, bmax=bmax, pad_m=float(pad_m))


def scene_bounds(model_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounding box of a scene, read from the model's own vertices.

    Deliberately not taken from the manifest, whose figures are rounded for
    human eyes. ``RoomGeo`` widens any supplied ``bmin`` to include every point,
    so a bound rounded a fraction of a millimetre the wrong way is silently
    replaced by the true one, the reference and the truncated runs then start
    their grids at different places, and the comparison measures that offset
    instead of the truncation.
    """
    model = json.loads(Path(model_json).read_text())
    points = np.vstack([np.asarray(v["pts"], dtype=float) for v in model["mats_hash"].values()])
    return points.min(axis=0), points.max(axis=0)


def build_materials(
    labels: set[str], table: dict[str, list[float]], out_dir: Path
) -> dict[str, str]:
    """One impedance file per label, fitted by PFFDTD in its own interpreter.

    PFFDTD needs numpy below 2 and this project does not, so the fit happens in
    a subprocess and only the ``.h5`` files cross over. Already fitted labels
    are left alone, which is what makes a sweep over one scene cheap.
    """
    missing_absorption = sorted(labels - set(table))
    if missing_absorption:
        raise RuntimeError(f"no absorption for {missing_absorption}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {label: f"{label}.h5" for label in sorted(labels)}
    todo = [label for label, name in mapping.items() if not (out_dir / name).is_file()]
    if todo:
        payload = json.dumps({label: table[label] for label in todo})
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(pffdtd_dir() / 'python')!r})\n"
            "import numpy as np\n"
            "from materials.adm_funcs import fit_to_Sabs_oct_11\n"
            f"for label, coeffs in json.loads({payload!r}).items():\n"
            "    fit_to_Sabs_oct_11(np.array(coeffs, dtype=float),\n"
            f"                       filename={str(out_dir)!r} + '/' + label + '.h5',\n"
            "                       plot=False)\n"
        )
        subprocess.run([pffdtd_python(), "-c", script], check=True)
    return mapping


def entry_from_key(key: str) -> CacheEntry:
    """A published voxelisation, by key. Absent is an error, never a rebuild."""
    path = cache_root() / key
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no cache entry {key} under {cache_root()}")
    return CacheEntry(path=path, key=key, manifest=json.loads(manifest.read_text()))


def execute(
    files: list[Path],
    run_dir: Path,
    *,
    engine: Engine,
    double_precision: bool,
    machine: Machine | None = None,
    remote_dir: str = "/root/solve",
    remote_pffdtd_dir: str = "/root/pffdtd",
) -> dict[str, Any]:
    """Run the solver on ``files``, here or on ``machine``, and time it.

    ``files`` is what :func:`reverberate.wave.voxelise.engine_inputs` returns,
    in :data:`reverberate.wave.ENGINE_FILES` order. The engine opens its inputs
    by name from its working directory, so they are copied in under the names
    it asks for; the comms file in particular is cached under its key.

    The copies are copies rather than links because the engine writes
    ``sim_outs.h5`` next to them and a cache entry is content addressed and has
    to stay exactly what it says it is. Afterwards every input but
    ``sim_consts.h5`` is deleted: it is a few kilobytes and carries the sample
    rate every later comparison needs, whereas fourteen copies of a multi
    gigabyte ``vox_out.h5`` would fill the disk for nothing.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for path, name in zip(files, ENGINE_FILES, strict=True):
        target = run_dir / name
        # The comms file may already be exactly where the engine wants it, in
        # which case copying it onto itself is an error rather than a no-op.
        if not (target.exists() and target.samefile(path)):
            shutil.copy2(path, target)
        staged.append(target)

    if machine is None:
        run = run_binary(run_dir, engine, double_precision=double_precision)
        record: dict[str, Any] = {
            "engine_s": run.engine_s,
            "where": f"local-{engine}",
        }
    else:
        result = solve(
            machine,
            staged,
            run_dir / "sim_outs.h5",
            remote_dir=remote_dir,
            pffdtd_dir=remote_pffdtd_dir,
            double_precision=double_precision,
        )
        (run_dir / "engine.log").write_text(result.log)
        record = {
            "engine_s": result.engine_s,
            "upload_s": result.upload_s,
            "fetch_s": result.fetch_s,
            "uploaded_bytes": result.uploaded_bytes,
            "where": f"{machine.user}@{machine.host}:{machine.port}",
        }

    for path in staged:
        if path.name != "sim_consts.h5":
            path.unlink(missing_ok=True)

    constants = sim_consts(run_dir)
    record |= {
        "engine": engine,
        "double_precision": double_precision,
        "h_m": constants.h,
        "sample_rate_hz": constants.sample_rate,
        "run_dir": str(run_dir),
        "sim_outs": str(run_dir / "sim_outs.h5"),
    }
    return record


def run_scene(
    models: Path,
    out: Path,
    scene_name: str,
    *,
    fmax: float,
    duration: float,
    bounds_mode: BoundsMode,
    ppw: float = 10.5,
    reference: str = "apartment_full",
    pad_m: float | None = None,
    double_precision: bool = False,
    engine: Engine = "cpu",
    machine: Machine | None = None,
    nprocs: int | None = None,
) -> dict[str, Any]:
    """Voxelise one exported scene on the chosen domain, then solve it.

    ``bounds_mode`` is required: see the module docstring. The reference scene
    is always simulated on its own exact box, which is what the two modes are
    defined relative to.
    """
    manifest = json.loads((models / "manifest.json").read_text())
    scenes = {s["name"]: s for s in manifest["scenes"]}
    scene = scenes[scene_name]

    step = grid_step(fmax, ppw)
    ref_lo, ref_hi = scene_bounds(models / scenes[reference]["file"])
    lo, hi = scene_bounds(models / scene["file"])
    cut_m = scene["cut_m"]
    if scene_name == reference:
        bounds = Bounds(mode=bounds_mode, bmin=ref_lo, bmax=ref_hi, pad_m=None)
    else:
        pad = float(pad_m) if pad_m is not None else float(cut_m or 0.0)
        bounds = choose_bounds(bounds_mode, lo, hi, ref_lo, ref_hi, step, pad)

    model_json = models / scene["file"]
    # PFFDTD asserts that the material mapping matches the scene's label list
    # exactly, so a truncated scene gets only the labels that survived the cut.
    labels = set(json.loads(model_json.read_text())["mats_hash"])
    mat_folder = out / "materials"
    mat_files = build_materials(labels, manifest["materials"], mat_folder)

    bmin, bmax = bounds.as_spec()
    spec = SceneSpec(
        model_json=model_json,
        mat_folder=mat_folder,
        mat_files=mat_files,
        fmax=fmax,
        ppw=ppw,
        bmin=bmin,
        bmax=bmax,
    )
    entry = voxelise(spec, nprocs=nprocs)

    run_dir = out / f"{scene_name}_f{int(fmax)}_p{ppw:g}_{bounds.mode}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    comms = write_comms(
        entry.path,
        np.asarray(manifest["source"], dtype=float),
        np.asarray(manifest["receiver"], dtype=float).reshape(1, 3),
        duration,
        diff_source=not double_precision,
        out_path=run_dir / "comms_out.h5",
    )

    record = execute(
        engine_inputs(entry, comms),
        run_dir,
        engine=engine,
        double_precision=double_precision,
        machine=machine,
    )
    record |= {
        "scene": scene_name,
        "cut_m": cut_m,
        "fmax": fmax,
        "ppw": ppw,
        "duration_s": duration,
        "bounds_mode": bounds.mode,
        "pad_m": bounds.pad_m,
        "bmin": [float(v) for v in bounds.bmin],
        "bmax": [float(v) for v in bounds.bmax],
        "triangles": scene["triangles"],
        "key": entry.key,
        "grid_points": entry.manifest.get("grid_points"),
        "boundary_nodes": entry.manifest.get("boundary_nodes"),
    }
    _publish(out, run_dir, record)
    return record


def run_prepared(
    key: str,
    out: Path,
    comms: Path,
    *,
    engine: Engine = "cpu",
    double_precision: bool = False,
    machine: Machine | None = None,
    remote_dir: str = "/root/solve",
    remote_pffdtd_dir: str = "/root/pffdtd",
) -> dict[str, Any]:
    """Solve an already voxelised domain with an already written comms file.

    This is W2's half of the split pipeline: the sweep voxelises every domain
    once on a laptop and only the solve is rented, so the run needs nothing but
    a cache key.
    """
    entry = entry_from_key(key)
    record = execute(
        engine_inputs(entry, comms),
        out,
        engine=engine,
        double_precision=double_precision,
        machine=machine,
        remote_dir=remote_dir,
        remote_pffdtd_dir=remote_pffdtd_dir,
    )
    record |= {
        "key": key,
        "grid_points": entry.manifest.get("grid_points"),
        "boundary_nodes": entry.manifest.get("boundary_nodes"),
    }
    _publish(out.parent, out, record)
    return record


def _publish(out: Path, run_dir: Path, record: dict[str, Any]) -> None:
    """The run's own record beside it, and one line in the sweep's log."""
    write_record(run_dir, "result.json", record)
    out.mkdir(parents=True, exist_ok=True)
    with (out / RESULTS_FILE).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _machine_from(args: argparse.Namespace) -> Machine | None:
    if not args.host:
        return None
    return Machine(
        host=args.host,
        port=args.port,
        user=args.user,
        identity=Path(args.identity) if args.identity else None,
    )


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--double", action="store_true", help="double precision engine")
    parser.add_argument("--host", default=None, help="solve over ssh instead of here")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity", default=None)
    parser.add_argument("--remote-dir", default="/root/solve")
    parser.add_argument("--remote-pffdtd-dir", default="/root/pffdtd")


def _scene(args: argparse.Namespace) -> int:
    run_scene(
        args.models,
        args.out,
        args.scene,
        fmax=args.fmax,
        duration=args.duration,
        bounds_mode=args.bounds,
        ppw=args.ppw,
        reference=args.reference,
        pad_m=args.pad,
        double_precision=args.double,
        engine=args.engine,
        machine=_machine_from(args),
        nprocs=args.nprocs,
    )
    return 0


def _prepared(args: argparse.Namespace) -> int:
    run_prepared(
        args.key,
        args.out,
        Path(args.comms),
        engine=args.engine,
        double_precision=args.double,
        machine=_machine_from(args),
        remote_dir=args.remote_dir,
        remote_pffdtd_dir=args.remote_pffdtd_dir,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reverberate.experiments.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scene = sub.add_parser("scene", help="voxelise an exported scene and solve it")
    scene.add_argument("--models", type=Path, required=True)
    scene.add_argument("--out", type=Path, required=True)
    scene.add_argument("--scene", required=True)
    scene.add_argument("--fmax", type=float, required=True)
    scene.add_argument("--duration", type=float, required=True)
    scene.add_argument("--ppw", type=float, default=10.5)
    scene.add_argument("--reference", default="apartment_full")
    scene.add_argument(
        "--bounds",
        required=True,
        choices=["common", "padded"],
        help="common: the reference box for every scene; padded: the scene's box grown by --pad",
    )
    scene.add_argument(
        "--pad",
        type=float,
        default=None,
        help="padding in metres for --bounds padded, defaulting to the scene's cut",
    )
    scene.add_argument("--nprocs", type=int, default=None)
    _add_engine_arguments(scene)
    scene.set_defaults(func=_scene)

    prepared = sub.add_parser("prepared", help="solve a cached voxelisation")
    prepared.add_argument("--key", required=True, help="voxelisation cache key")
    prepared.add_argument("--comms", required=True, help="comms_out.h5 for this pair")
    prepared.add_argument("--out", type=Path, required=True, help="run directory")
    _add_engine_arguments(prepared)
    prepared.set_defaults(func=_prepared)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
