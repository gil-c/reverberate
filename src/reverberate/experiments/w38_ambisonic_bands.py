"""W38: one ambisonic response from three solves on three grids.

The economy of roadmap section 4.2 and the spatial output of section 7, run
together for the first time. Three solves, each on its own grid with its own
array snapped to its own nodes, are encoded separately and assembled in the
spherical harmonic domain by :mod:`reverberate.spatial.bands`, which says what
that costs. The mid and high bands are solved only for their computed windows
and continued with a diffuse tail per channel out to the low band's full
decay.

One run directory, one band directory inside it per solve, each of those a
complete W10 room run so that every W10 tool reads it unchanged::

    <run>/plan.json                 the three bands, the offsets, the clearances
    <run>/low/  mid/  high/         plan.json, array_positions.npy, comms/, source0/
    <run>/report.json               the assembled response, spatial shape
    <run>/responses/  audio/        what W10 writes, from the assembled response

The viewer opens ``<run>`` and never the band directories, which hold no
report of their own.

Usage::

    python -m reverberate.experiments.w38_ambisonic_bands plan \\
        --out data/runs/w38_living_16k --scene-id 102344022 --room living_room \\
        --centre 1.2 1.6 -0.4 --source -0.8 1.5 1.1 \\
        --band low KEY_1K 1.2 --band mid KEY_4K 0.4 --band high KEY_16K 0.06
    python -m reverberate.experiments.w38_ambisonic_bands prepare --out ...
    python scripts/remote_solve.py --run data/runs/w38_living_16k/high --key KEY_16K ...
    python -m reverberate.experiments.w38_ambisonic_bands assemble --out ... --audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.audio import Atmosphere
from reverberate.experiments.engine import write_record
from reverberate.experiments.run import entry_from_key, sound_speed
from reverberate.experiments.w10_ambisonic import COMMS_NAME, plan_room, prepare_room
from reverberate.experiments.w10_render import (
    EncodedRun,
    encode_run,
    finish_run,
    spatial_report,
)
from reverberate.experiments.w37_window import mean_absorption_of
from reverberate.metrics import band_centres
from reverberate.spatial.bands import BandSolve, assemble, centre_offsets, extend
from reverberate.spatial.encode import EncoderSettings
from reverberate.wave import Machine, engine_inputs

__all__ = ["BANDS", "assemble_run", "main", "outer_radius_for", "plan_bands", "prepare_bands"]

#: The three bands, lowest first. The order is what :func:`assemble` levels on.
BANDS = ("low", "mid", "high")

#: What an economy run must carry in its report, per roadmap W30.
TRICK = (
    "three solves on three grids, each band encoded on its own node-snapped array "
    "and assembled in the spherical harmonic domain; the mid and high bands are "
    "solved for their computed windows only and continued with a diffuse tail per "
    "channel to the low band's full decay"
)


def _settings(order: int, fit_order: int, fmax_hz: float) -> EncoderSettings:
    return EncoderSettings(order=order, fit_order=fit_order, max_frequency_hz=fmax_hz)


#: Shells per array, outermost over innermost. The innermost is six cells, so
#: this is what the outer radius has to be for the array to exist at all.
RADIUS_SPAN = 2.0


def outer_radius_for(grid_step_m: float, outer_radius_m: float) -> float:
    """The array's outer radius on this grid: the one asked for, or the one it needs.

    ``default_shells`` starts its innermost shell six cells out, and on the
    32.7 mm grid of the low band that is already 19.6 cm, past the 16 cm the
    high band uses. So the ball grows with the cell, and the array of a band
    is a different size on every grid. That is not a defect. ``k r`` is what
    an order needs and ``k`` falls with the band, so a bigger ball is exactly
    what the bottom of the band asks for; the cost is clearance, which the
    plan checks against the room, and it is reported per band.
    """
    return max(outer_radius_m, RADIUS_SPAN * 6.0 * grid_step_m)


def plan_bands(
    out: Path,
    bands: dict[str, tuple[str, float]],
    *,
    scene_id: str,
    room: str,
    centre: np.ndarray,
    source: np.ndarray,
    order: int = 7,
    fit_order: int = 10,
    outer_radius_m: float = 0.16,
) -> dict[str, Any]:
    """Plan one W10 room run per band, and write down what differs between them.

    ``bands`` maps a band name to its voxelisation cache key and its solve
    duration in seconds. Every band gets the same requested centre and source;
    what each actually expands about is its own nearest node, and the plan
    records the three offsets rather than pretending they are one point.
    """
    missing = [name for name in BANDS if name not in bands]
    if missing:
        raise ValueError(f"no plan for band(s) {missing}; all three are needed")
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for name in BANDS:
        key, duration_s = bands[name]
        entry = entry_from_key(key)
        fmax = float(str(entry.manifest["fmax"]))
        step = float(str(entry.manifest["h_m"]))
        records[name] = plan_room(
            out / name,
            entry.path,
            _settings(order, fit_order, fmax),
            centre=centre,
            source=source,
            duration_s=duration_s,
            outer_radius_m=outer_radius_for(step, outer_radius_m),
            extra_receivers=(),
            scene_id=scene_id,
            room=room,
            reference_run=None,
            reference_note=None,
        )
    steps = [float(records[name]["array"]["grid_step_m"]) for name in BANDS]
    fmaxes = [float(records[name]["fmax_hz"]) for name in BANDS]
    if not steps[0] > steps[1] > steps[2] or not fmaxes[0] < fmaxes[1] < fmaxes[2]:
        raise ValueError(
            f"the bands are not coarse to fine: steps {steps} m, fmax {fmaxes} Hz. "
            "The low band must be the coarsest grid and the high band the finest"
        )
    offsets = [
        {
            "band": name,
            "centre": records[name]["array"].get("centre"),
            "offset_m": round(
                float(
                    np.linalg.norm(
                        np.asarray(records[name]["array"].get("centre", centre), dtype=float)
                        - centre
                    )
                ),
                6,
            ),
            "grid_step_m": records[name]["array"]["grid_step_m"],
        }
        for name in BANDS
    ]
    plan = {
        "run": out.name,
        "scene_id": scene_id,
        "room": room,
        "kind": "three-band ambisonic",
        "trick": TRICK,
        "centre": [float(v) for v in centre],
        "source": [float(v) for v in source],
        "sound_speed_m_s": sound_speed(),
        "bands": {
            name: {
                "cache_key": records[name]["cache_key"],
                "fmax_hz": records[name]["fmax_hz"],
                "grid_step_m": records[name]["array"]["grid_step_m"],
                "duration_s": records[name]["duration_s"],
                "receivers": records[name]["array"]["nodes"],
                "outer_radius_m": records[name]["array"]["outer_radius_m"],
                "clearance": records[name]["clearance"],
                "cost": records[name]["cost"],
            }
            for name in BANDS
        },
        "centre_offsets": offsets,
        "encoder": _settings(order, fit_order, fmaxes[-1]).record(),
        # The keys the viewer reads from a spatial plan come from the band whose
        # grid the assembled response is centred on.
        "cache_key": records["high"]["cache_key"],
        "cache_root": records["high"].get("cache_root"),
        "model_json": records["high"].get("model_json"),
        "geometry_sha256": records["high"].get("geometry_sha256"),
        "array": records["high"]["array"],
        "conditioning": records["high"]["conditioning"],
    }
    write_record(out, "plan.json", plan)
    return plan


def prepare_bands(out: Path, *, double_precision: bool = False) -> list[Path]:
    """The comms file of every band, from each band's own plan."""
    plan = json.loads((out / "plan.json").read_text())
    written = []
    for name in BANDS:
        entry = entry_from_key(plan["bands"][name]["cache_key"])
        written.append(prepare_room(out / name, entry.path, double_precision=double_precision))
    return written


def _absorption(encoded: EncodedRun, bands: int) -> tuple[np.ndarray, str]:
    """Area weighted absorption per band from the room, or a stated assumption."""
    if encoded.room and encoded.room.get("per_class"):
        return mean_absorption_of({"room": encoded.room}, bands), "the solver's own boundary"
    return np.full(bands, 0.2), "assumed flat 0.2, because the voxelisation is not on this machine"


def assemble_run(
    out: Path,
    *,
    air: Atmosphere | None,
    order: int = 7,
    fit_order: int = 10,
    lowcut_hz: float | None = None,
    measured_path: Path | None = None,
    plain_decode: bool = False,
    seed: int = 20260908,
    rendered: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode each band on its own array, assemble, decode, measure. Nothing is written."""
    plan = json.loads((out / "plan.json").read_text())
    encoded: dict[str, EncodedRun] = {}
    for name in BANDS:
        band = plan["bands"][name]
        encoded[name] = encode_run(
            out / name,
            _settings(order, fit_order, float(band["fmax_hz"])),
            air=air,
            lowcut_hz=lowcut_hz,
        )
    solves = {
        name: BandSolve(
            name,
            encoded[name].ambisonic,
            float(plan["bands"][name]["fmax_hz"]),
            float(plan["bands"][name]["grid_step_m"]),
        )
        for name in BANDS
    }
    rate = int(round(encoded["high"].rate))
    total_s = solves["low"].ambisonic.duration_s
    absorption, absorption_source = _absorption(encoded["high"], len(band_centres(rate)))
    atmosphere = air or Atmosphere()
    tails: list[dict[str, Any]] = []
    for index, name in enumerate(("mid", "high")):
        solves[name], record = extend(
            solves[name],
            total_s,
            calibration=solves["mid"],
            mean_absorption=absorption,
            atmosphere=atmosphere,
            sound_speed_m_s=float(plan["sound_speed_m_s"]),
            seed=seed + 1000 * index,
        )
        tails.append(record)
    assembled, assembly = assemble(solves["low"], solves["mid"], solves["high"])

    extra = {
        "kind": plan["kind"],
        "trick": plan["trick"],
        "assembly": assembly,
        "tails": tails,
        "absorption_for_tail": {
            "per_band": [round(float(v), 4) for v in absorption],
            "source": absorption_source,
        },
        "centre_offsets": centre_offsets(list(solves.values()), np.asarray(plan["centre"])),
        "band_plans": {
            name: {
                "cache_key": plan["bands"][name]["cache_key"],
                "fmax_hz": plan["bands"][name]["fmax_hz"],
                "grid_step_m": plan["bands"][name]["grid_step_m"],
                "computed_s": round(encoded[name].ambisonic.duration_s, 4),
                "array": encoded[name].plan["array"],
                "conditioning": encoded[name].plan["conditioning"],
                "clearance": encoded[name].plan.get("clearance"),
            }
            for name in BANDS
        },
        "air_absorption_note": "applied to each band before it was windowed and extended",
    }
    report = spatial_report(
        encoded["high"],
        _settings(order, fit_order, solves["high"].fmax_hz),
        air=air,
        measured_path=measured_path,
        plain_decode=plain_decode,
        rendered=rendered,
        ambisonic=assembled,
        extra=extra,
    )
    report["run"] = out.name
    if rendered is not None:
        rendered["plan"] = {
            **encoded["high"].plan,
            **{k: plan[k] for k in ("run", "scene_id", "room")},
        }
    return report


def _parse_bands(rows: list[list[str]]) -> dict[str, tuple[str, float]]:
    bands: dict[str, tuple[str, float]] = {}
    for name, key, duration in rows:
        if name not in BANDS:
            raise SystemExit(f"band must be one of {BANDS}, got {name}")
        bands[name] = (key, float(duration))
    return bands


def _plan(args: argparse.Namespace) -> int:
    plan = plan_bands(
        args.out,
        _parse_bands(args.band),
        scene_id=args.scene_id,
        room=args.room,
        centre=np.asarray(args.centre, dtype=float),
        source=np.asarray(args.source, dtype=float),
        order=args.order,
        fit_order=args.fit_order,
        outer_radius_m=args.radius,
    )
    for name in BANDS:
        band = plan["bands"][name]
        print(
            f"{name:>4}: fmax {band['fmax_hz']:g} Hz, step {1000 * band['grid_step_m']:.2f} mm, "
            f"{band['duration_s']:g} s, {band['receivers']} receivers, "
            f"cost {json.dumps(band['cost'])[:200]}"
        )
    for row in plan["centre_offsets"]:
        print(f"      {row['band']} expands {1000 * row['offset_m']:.1f} mm from the request")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    for path in prepare_bands(args.out, double_precision=args.double):
        print(path)
    return 0


def _solve(args: argparse.Namespace) -> int:
    from reverberate.experiments.run import execute

    plan = json.loads((args.out / "plan.json").read_text())
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
    for name in args.bands:
        entry = entry_from_key(plan["bands"][name]["cache_key"])
        record = execute(
            engine_inputs(entry, args.out / name / "comms" / COMMS_NAME),
            args.out / name / "source0",
            engine=args.engine,
            double_precision=args.double,
            machine=machine,
        )
        write_record(args.out / name, "solve.json", {"runs": [record]})
    return 0


def _assemble(args: argparse.Namespace) -> int:
    air = (
        None
        if args.no_air
        else Atmosphere(temperature_c=args.temperature, humidity_percent=args.humidity)
    )
    rendered: dict[str, Any] = {}
    report = assemble_run(
        args.out,
        air=air,
        order=args.order,
        fit_order=args.fit_order,
        lowcut_hz=args.low_cut,
        measured_path=args.measured_head,
        plain_decode=args.plain_decode,
        seed=args.seed,
        rendered=rendered,
    )
    settings = _settings(args.order, args.fit_order, float(report["band_plans"]["high"]["fmax_hz"]))
    finish_run(
        args.out,
        report,
        rendered,
        settings,
        fmax_hz=float(report["band_plans"]["high"]["fmax_hz"]),
        band="three-band",
        audio_wanted=bool(args.audio),
        publish_wanted=bool(args.publish),
        seed=args.seed,
    )
    for row in report["assembly"]["bands"]:
        print(
            f"{row['band']:>4}: order {row['order']}, {row['computed_s']} s, "
            f"gain {row['gain_db']:+.2f} dB"
        )
    for row in report["tails"]:
        print(f"{row['band']:>4}: tail from {row['computed_s']} s to {row['total_s']} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="one W10 plan per band, and the differences between them")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--scene-id", required=True)
    plan.add_argument("--room", required=True)
    plan.add_argument("--centre", type=float, nargs=3, required=True)
    plan.add_argument("--source", type=float, nargs=3, required=True)
    plan.add_argument(
        "--band",
        nargs=3,
        action="append",
        metavar=("NAME", "KEY", "SECONDS"),
        required=True,
        help="a band, its cache key and its solve duration; give all three bands",
    )
    plan.add_argument("--order", type=int, default=7)
    plan.add_argument("--fit-order", type=int, default=10)
    plan.add_argument("--radius", type=float, default=0.16)
    plan.set_defaults(func=_plan)

    prepare = sub.add_parser("prepare", help="the comms file of every band")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--double", action="store_true")
    prepare.set_defaults(func=_prepare)

    solve = sub.add_parser("solve", help="solve bands here or over ssh; rent with remote_solve.py")
    solve.add_argument("--out", type=Path, required=True)
    solve.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS))
    solve.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    solve.add_argument("--double", action="store_true")
    solve.add_argument("--host")
    solve.add_argument("--port", type=int, default=22)
    solve.add_argument("--user", default="root")
    solve.add_argument("--identity")
    solve.set_defaults(func=_solve)

    build = sub.add_parser("assemble", help="encode, assemble, decode, write")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--order", type=int, default=7)
    build.add_argument("--fit-order", type=int, default=10)
    build.add_argument("--no-air", action="store_true")
    build.add_argument("--temperature", type=float, default=20.0)
    build.add_argument("--humidity", type=float, default=50.0)
    build.add_argument("--low-cut", type=float)
    build.add_argument("--measured-head", type=Path)
    build.add_argument("--plain-decode", action="store_true")
    build.add_argument("--audio", action="store_true")
    build.add_argument("--publish", action="store_true")
    build.add_argument("--seed", type=int, default=20260908)
    build.set_defaults(func=_assemble)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
