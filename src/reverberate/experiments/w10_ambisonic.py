"""W10: an ambisonic response of one room at 16 kHz, and the rehearsal before it.

The chain this drives is new from end to end -- a volumetric array of grid
nodes, a per frequency regularised fit, a head, a decode -- and every one of
its conventions has a silent opposite. So nothing is rented until the whole
chain has been run on a scene whose answer is known.

**The rehearsal is a box, and its size is arithmetic rather than a guess.** The
array sits at the centre and the source is off to one side, and the box is
made just large enough that the whole array hears the direct sound alone for
long enough to encode it. In grid cells, with the array's outer radius ``a``,
the source at ``s`` from the centre and the box half width ``W``:

- the direct sound reaches the nearest node at ``s - a`` and the farthest at
  ``s + a``;
- the first wall reflection reaches the nearest node at ``2W - s - a``.

At the 16 kHz working point, ``W = 294``, ``s = 147`` and ``a = 78`` give a
clean window from 69 to 363 samples, which is 1.0 ms of direct sound over the
whole array, in a 1.2 m box of 2.03e8 points: about a minute of local processor
and no rental. Inside that window the field is an analytic monopole, so the
encoder's answer is known and the two questions the design leaves open can be
measured rather than argued:

- what the **direction of arrival** error is at the top of the band, which is
  the grid's own numerical dispersion and nothing else;
- whether fitting with the scheme's own wavenumber rather than ``omega / c``
  reduces it.

Only then is the room run worth renting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from reverberate.experiments.engine import Engine, sim_consts, write_record
from reverberate.experiments.run import build_materials, execute, grid_step, sound_speed
from reverberate.experiments.scene_export import material_table, to_model
from reverberate.experiments.w20_first_listen import (
    GPU_UPDATES_PER_SECOND,
    LOCAL_UPDATES_PER_SECOND,
)
from reverberate.geometry.orientation import orient_for_air
from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.materials.db import acoustic_classes
from reverberate.spatial.array import ArrayDesign, check_clearance, design_array
from reverberate.spatial.encode import EncoderSettings, conditioning
from reverberate.store import shared_store
from reverberate.wave import Machine, SceneSpec, engine_inputs, voxelise, write_comms
from reverberate.wave.comms import Grid, load_grid

__all__ = [
    "REHEARSAL",
    "Rehearsal",
    "array_at",
    "box_scene",
    "cost_record",
    "plan_record",
    "main",
]

#: The room the pipeline has already exported and voxelised at 16 kHz, on the
#: geometry W25 fixed and W33 carved. Named here rather than passed from a
#: shell history nobody kept.
SCENE_ID = "102344022"
ROOM = "bedroom.001"

#: W29's own source in that room, so the two runs are comparable where they
#: overlap. Its geometry differs, which the report has to say.
W29_SOURCE = (-1.3873, 0.8479, -0.9268)

#: W29's six receivers, carried along as ordinary extra nodes. They cost one
#: output row each and give the new run a per band comparison against a run
#: whose numbers are already published.
W29_RECEIVERS = (
    (-1.5872, 1.2, -0.1878),
    (0.2615, 1.2, -1.8548),
    (-0.8937, 1.2, -0.8321),
    (-0.4165, 1.6, 0.1596),
    (-0.3473, 1.6, -1.2007),
    (-0.0818, 1.6, -0.2312),
)


@dataclass(frozen=True)
class Rehearsal:
    """The box, in grid cells, so the timings are exact rather than rounded."""

    half_width_cells: int = 294
    source_cells: int = 147
    array_radius_m: float = 0.16
    fmax_hz: float = 16000.0
    ppw: float = 10.5
    #: Samples simulated. Past the first reflection on purpose, so the response
    #: shows where the clean window ends instead of asserting it.
    samples: int = 500

    @property
    def step_m(self) -> float:
        return grid_step(self.fmax_hz, self.ppw)

    @property
    def side_m(self) -> float:
        return 2.0 * self.half_width_cells * self.step_m

    @property
    def array_radius_cells(self) -> int:
        return int(round(self.array_radius_m / self.step_m))

    @property
    def first_reflection_sample(self) -> int:
        """When the earliest wall reflection reaches the nearest array node."""
        return 2 * self.half_width_cells - self.source_cells - self.array_radius_cells

    @property
    def direct_last_sample(self) -> int:
        """When the direct sound has reached every node of the array."""
        return self.source_cells + self.array_radius_cells

    def record(self) -> dict[str, Any]:
        return {
            "half_width_cells": self.half_width_cells,
            "side_m": round(self.side_m, 4),
            "step_m": self.step_m,
            "source_cells": self.source_cells,
            "source_distance_m": round(self.source_cells * self.step_m, 4),
            "array_radius_cells": self.array_radius_cells,
            "direct_first_sample": self.source_cells - self.array_radius_cells,
            "direct_last_sample": self.direct_last_sample,
            "first_reflection_sample": self.first_reflection_sample,
            "clean_window_samples": self.first_reflection_sample - self.direct_last_sample,
            "samples": self.samples,
            "note": (
                "inside the clean window the field at the array is an analytic "
                "monopole, so the encoder's answer is known"
            ),
        }


REHEARSAL = Rehearsal()


def box_scene(
    out_dir: Path, rehearsal: Rehearsal, *, material: str = "generic_hard"
) -> tuple[Path, dict[str, list[float]], np.ndarray, np.ndarray]:
    """A closed box, its material table, its centre and its source position.

    The box is oriented through the same code path a real room shell takes, so
    its faces reach the solver marked as facing the air on the inside. Building
    it by hand and writing ``2`` would be exactly the defect section 6.1 of the
    roadmap exists to record.
    """
    side = rehearsal.side_m
    mesh = trimesh.creation.box(extents=(side, side, side))
    mesh.apply_translation([side / 2.0, side / 2.0, side / 2.0])
    oriented = orient_for_air(mesh, "inside")
    if not oriented.authoritative:
        raise RuntimeError("a box should always be orientable; the geometry is wrong")
    assignment = MeshMaterialAssignment(
        mesh=oriented.mesh,
        material=acoustic_classes()[material].material(),
        name=f"{material}_shell",
        sides=oriented.sides,
    )
    centre = np.full(3, side / 2.0)
    source = centre + np.array([rehearsal.source_cells * rehearsal.step_m, 0.0, 0.0])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "box.json"
    path.write_text(json.dumps(to_model([assignment], source, centre)))
    return path, material_table([assignment]), centre, source


def array_at(
    centre: np.ndarray,
    entry_path: Path,
    *,
    outer_radius_m: float,
    settings: EncoderSettings,
    margin_m: float = 0.02,
) -> tuple[ArrayDesign, Grid, dict[str, Any]]:
    """Design the array on a voxelisation's own grid and prove it is in free air.

    The clearance check is the one that matters and it is not the one PFFDTD
    makes. PFFDTD refuses a receiver *on* a boundary node. The interior
    expansion needs more than that: it is a solution of the homogeneous
    Helmholtz equation on the whole ball, so a surface anywhere inside the ball
    makes the model wrong rather than noisy, and the fit would return a
    plausible field that is not the one in the room.
    """
    grid = load_grid(entry_path)
    design = design_array(centre, grid, fit_order=settings.fit_order, outer_radius_m=outer_radius_m)
    clearance = check_clearance(design, entry_path, grid, margin_m=margin_m)
    return design, grid, clearance


def cost_record(grid_points: int, samples: int, receivers: int) -> dict[str, Any]:
    """What the run will cost, stated before it is started rather than after."""
    updates = float(grid_points) * float(samples)
    return {
        "grid_points": int(grid_points),
        "steps": int(samples),
        "point_updates": updates,
        "estimated_local_s": round(updates / LOCAL_UPDATES_PER_SECOND, 1),
        "estimated_gpu_s": round(updates / GPU_UPDATES_PER_SECOND, 1),
        "receivers": int(receivers),
        "sim_outs_bytes": int(receivers * samples * 8),
    }


def plan_record(
    design: ArrayDesign,
    settings: EncoderSettings,
    *,
    sound_speed_m_s: float,
    frequencies: np.ndarray,
) -> dict[str, Any]:
    """The array, and what it is measured to support, before anything is solved."""
    report = conditioning(design, frequencies, sound_speed_m_s=sound_speed_m_s, settings=settings)
    return {
        "array": design.record(),
        "encoder": settings.record(),
        "conditioning": report.record(),
    }


def _voxelise_box(
    out_dir: Path, rehearsal: Rehearsal, model: Path, table: dict[str, list[float]]
) -> Any:
    labels = set(json.loads(model.read_text())["mats_hash"])
    materials = build_materials(labels, table, out_dir / "materials")
    spec = SceneSpec(
        model_json=model,
        mat_folder=out_dir / "materials",
        mat_files=materials,
        fmax=rehearsal.fmax_hz,
        ppw=rehearsal.ppw,
    )
    store = shared_store()
    if store is None:
        print("no store credentials: this voxelisation stays on this machine only")
    return voxelise(spec, store=store)


def run_rehearsal(
    out_dir: Path,
    rehearsal: Rehearsal,
    settings: EncoderSettings,
    *,
    engine: Engine = "cpu",
    double_precision: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Voxelise the box, place the array in it, solve it here, and record all of it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model, table, centre, source = box_scene(out_dir, rehearsal)
    entry = _voxelise_box(out_dir, rehearsal, model, table)
    design, grid, clearance = array_at(
        centre,
        entry.path,
        outer_radius_m=rehearsal.array_radius_m,
        settings=settings,
    )
    constants = sim_consts(entry.path)
    record: dict[str, Any] = {
        "run": out_dir.name,
        "what": "the whole spatial chain on a box whose direct sound is analytic",
        "rehearsal": rehearsal.record(),
        "cache_key": entry.key,
        "sample_rate_hz": constants.sample_rate,
        "sound_speed_m_s": sound_speed(),
        "source": [float(v) for v in source],
        "centre": [float(v) for v in centre],
        "clearance": clearance,
        "cost": cost_record(
            int(entry.manifest.get("grid_points", 0)), rehearsal.samples, design.count
        ),
    }
    record |= plan_record(
        design,
        settings,
        sound_speed_m_s=sound_speed(),
        frequencies=np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0]),
    )
    np.save(out_dir / "array_positions.npy", design.positions)
    write_record(out_dir, "plan.json", record)
    if plan_only:
        return record

    comms_dir = out_dir / "comms"
    comms_dir.mkdir(parents=True, exist_ok=True)
    comms = write_comms(
        entry.path,
        source,
        design.positions,
        rehearsal.samples / constants.sample_rate,
        diff_source=not double_precision,
        out_path=comms_dir / "source0.h5",
        interpolation="nearest",
    )
    solve = execute(
        engine_inputs(entry, comms),
        out_dir / "source0",
        engine=engine,
        double_precision=double_precision,
    )
    record["solve"] = solve
    write_record(out_dir, "solve.json", {"runs": [solve]})
    return record


def plan_room(
    out_dir: Path,
    entry_path: Path,
    settings: EncoderSettings,
    *,
    centre: np.ndarray,
    source: np.ndarray,
    duration_s: float,
    outer_radius_m: float = 0.16,
    extra_receivers: tuple[tuple[float, float, float], ...] = W29_RECEIVERS,
) -> dict[str, Any]:
    """Everything decided before a card is rented, written down and costed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    design, grid, clearance = array_at(
        centre, entry_path, outer_radius_m=outer_radius_m, settings=settings
    )
    constants = sim_consts(entry_path)
    samples = int(round(duration_s * constants.sample_rate))
    manifest = json.loads((entry_path / "manifest.json").read_text())
    extras = np.asarray(extra_receivers, dtype=float).reshape(-1, 3)
    record: dict[str, Any] = {
        "run": out_dir.name,
        "scene_id": SCENE_ID,
        "room": ROOM,
        "cache_key": manifest["key"],
        "geometry_sha256": manifest.get("geometry_sha256"),
        "reference_run": "w29_16k",
        "reference_note": (
            "same room and same source as w29_16k, but the carved geometry of "
            "W33 rather than W25's, so the comparison is across two geometries "
            "and is stated as such rather than reported as one number"
        ),
        "sample_rate_hz": constants.sample_rate,
        "sound_speed_m_s": sound_speed(),
        "duration_s": duration_s,
        "samples": samples,
        "source": [float(v) for v in source],
        "centre": [float(v) for v in centre],
        "extra_receivers": extras.tolist(),
        "clearance": clearance,
        "cost": cost_record(int(manifest["grid_points"]), samples, design.count + extras.shape[0]),
    }
    record |= plan_record(
        design,
        settings,
        sound_speed_m_s=sound_speed(),
        frequencies=np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]),
    )
    np.save(out_dir / "array_positions.npy", design.positions)
    write_record(out_dir, "plan.json", record)
    return record


def prepare_room(out_dir: Path, entry_path: Path, *, double_precision: bool = False) -> Path:
    """Write the one comms file the room run needs, from its own plan."""
    plan = json.loads((out_dir / "plan.json").read_text())
    positions = np.load(out_dir / "array_positions.npy")
    extras = np.asarray(plan["extra_receivers"], dtype=float).reshape(-1, 3)
    receivers = np.vstack([positions, extras])
    comms_dir = out_dir / "comms"
    comms_dir.mkdir(parents=True, exist_ok=True)
    return write_comms(
        entry_path,
        np.asarray(plan["source"], dtype=float),
        receivers,
        plan["samples"] / plan["sample_rate_hz"],
        diff_source=not double_precision,
        # Named for the engine, not for the source. ``execute`` renames on the
        # way in for a local run, but ``remote.upload`` refuses by name any file
        # that is not one of the four the engine reads, which is a rule worth
        # keeping: an unexpected file on a rented machine is a data policy
        # problem and not merely a slow upload. So the name is right here.
        out_path=comms_dir / "comms_out.h5",
        interpolation="nearest",
    )


def _settings(args: argparse.Namespace) -> EncoderSettings:
    return EncoderSettings(
        order=args.order,
        fit_order=args.fit_order,
        max_frequency_hz=args.fmax,
        dispersion=args.dispersion,
    )


def _rehearsal(args: argparse.Namespace) -> int:
    run_rehearsal(
        args.out,
        Rehearsal(
            half_width_cells=args.half_width,
            source_cells=args.source_cells,
            array_radius_m=args.radius,
            fmax_hz=args.fmax,
            samples=args.samples,
        ),
        _settings(args),
        engine=args.engine,
        double_precision=args.double,
        plan_only=args.plan_only,
    )
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan_room(
        args.out,
        args.entry,
        _settings(args),
        centre=np.asarray(args.centre, dtype=float),
        source=np.asarray(args.source, dtype=float),
        duration_s=args.duration,
        outer_radius_m=args.radius,
    )
    return 0


def _prepare(args: argparse.Namespace) -> int:
    print(prepare_room(args.out, args.entry, double_precision=args.double))
    return 0


def _solve(args: argparse.Namespace) -> int:
    from reverberate.experiments.run import entry_from_key

    plan = json.loads((args.out / "plan.json").read_text())
    entry = entry_from_key(plan["cache_key"])
    machine = (
        None
        if not args.host
        else Machine(
            host=args.host,
            port=args.port,
            user=args.user,
            identity=Path(args.identity) if args.identity else None,
        )
    )
    record = execute(
        engine_inputs(entry, args.out / "comms" / "source0.h5"),
        args.out / "source0",
        engine=args.engine,
        double_precision=args.double,
        machine=machine,
    )
    write_record(args.out, "solve.json", {"runs": [record]})
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order", type=int, default=7)
    parser.add_argument("--fit-order", type=int, default=10)
    parser.add_argument("--fmax", type=float, default=16000.0)
    parser.add_argument("--dispersion", choices=["ideal", "numerical"], default="numerical")
    parser.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--double", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rehearsal = sub.add_parser("rehearsal", help="the whole chain on a box, locally")
    _add_common(rehearsal)
    rehearsal.add_argument("--samples", type=int, default=REHEARSAL.samples)
    rehearsal.add_argument("--half-width", type=int, default=REHEARSAL.half_width_cells)
    rehearsal.add_argument("--source-cells", type=int, default=REHEARSAL.source_cells)
    rehearsal.add_argument("--radius", type=float, default=REHEARSAL.array_radius_m)
    rehearsal.add_argument("--plan-only", action="store_true")
    rehearsal.set_defaults(func=_rehearsal)

    plan = sub.add_parser("plan", help="design and cost the room run, solve nothing")
    _add_common(plan)
    plan.add_argument("--entry", type=Path, required=True, help="voxelisation cache entry")
    plan.add_argument("--centre", type=float, nargs=3, required=True)
    plan.add_argument("--source", type=float, nargs=3, default=list(W29_SOURCE))
    plan.add_argument("--duration", type=float, required=True)
    plan.add_argument("--radius", type=float, default=0.16)
    plan.set_defaults(func=_plan)

    prepare = sub.add_parser("prepare", help="write the comms file for a planned room run")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--entry", type=Path, required=True)
    prepare.add_argument("--double", action="store_true")
    prepare.set_defaults(func=_prepare)

    solve = sub.add_parser("solve", help="run the solver, here or over ssh")
    solve.add_argument("--out", type=Path, required=True)
    solve.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    solve.add_argument("--double", action="store_true")
    solve.add_argument("--host", default=None)
    solve.add_argument("--port", type=int, default=22)
    solve.add_argument("--user", default="root")
    solve.add_argument("--identity", default=None)
    solve.set_defaults(func=_solve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
