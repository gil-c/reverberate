"""Turn W20's two solver runs into twelve responses, some numbers, and audio.

This is the second half of the first listen. The solver has written two
``sim_outs.h5``; what remains is to reduce them to receivers, undo the source
differentiation, band limit, resample, measure, convolve a dry voice, and
publish. Each of those has a home already: :mod:`reverberate.audio` for the
signal path, :mod:`reverberate.metrics` for the measures,
:mod:`reverberate.clarify_library` for the voice, :mod:`reverberate.response`
for the artefacts and :mod:`reverberate.store` for the store. This module is
the wiring and the record, and deliberately holds no signal processing of its
own.

**The measures are put beside theory, not instead of it.** Sabine and Eyring
assume a diffuse field in a room whose absorption is spread evenly. A 13.6 m2
bedroom with a rug on one side and a bed against a wall is not that room, so
the comparison is reported as a difference to be explained rather than as a
pass or a fail. W3 measured that this pipeline realises about 0.89 of the
absorption it asks for, and that factor is applied to the theoretical figures
explicitly and named where it is applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio, metrics
from reverberate.acoustics import OCTAVE_BANDS
from reverberate.clarify_library import (
    EARS_ATTRIBUTION,
    EARS_LICENCE,
    catalogue,
    choose_clip,
    fetch_clip,
    read_wav,
)
from reverberate.geometry.pra_room import eyring_rt60, sabine_rt60
from reverberate.response import Provenance, ResponseSet, write_raw, write_sofa
from reverberate.store import ObjectStore

#: W3's measured shortfall between the absorption asked for and the absorption
#: the pipeline realises. Applied to Sabine and Eyring so the comparison is
#: against what this pipeline actually presents to the field, and named at
#: every use rather than folded silently into a constant.
REALISED_ABSORPTION_FACTOR = 0.89

#: Delivery rate. Physics happens at the grid rate; this is the last step.
DELIVERY_RATE_HZ = 48_000.0

#: Said on every artefact, because the shape of the data invites the opposite
#: assumption.
NOT_BINAURAL = (
    "Two bare omnidirectional pressure points in a room. No head, no torso, no "
    "pinna, no interaural time or level difference. This is not a binaural "
    "recording and must not be listened to as one."
)


@dataclass(frozen=True)
class Theory:
    """What the textbook says this room should do, and on what assumptions."""

    volume_m3: float
    surface_area_m2: float
    mean_absorption: float
    sabine_s: float
    eyring_s: float

    def record(self) -> dict[str, Any]:
        return {
            "volume_m3": round(self.volume_m3, 3),
            "surface_area_m2": round(self.surface_area_m2, 3),
            "mean_absorption_asked": round(self.mean_absorption, 4),
            "realised_absorption_factor": REALISED_ABSORPTION_FACTOR,
            "mean_absorption_realised": round(self.mean_absorption * REALISED_ABSORPTION_FACTOR, 4),
            "sabine_rt60_s": round(self.sabine_s, 3),
            "eyring_rt60_s": round(self.eyring_s, 3),
            "assumption": (
                "Sabine and Eyring assume a diffuse field and evenly spread "
                "absorption; this room has neither, so a difference is a thing "
                "to explain, not a failure"
            ),
        }


def theory(volume_m3: float, surface_area_m2: float, mean_absorption: float) -> Theory:
    """Sabine and Eyring with W3's realised absorption factor applied."""
    realised = mean_absorption * REALISED_ABSORPTION_FACTOR
    return Theory(
        volume_m3=volume_m3,
        surface_area_m2=surface_area_m2,
        mean_absorption=mean_absorption,
        sabine_s=sabine_rt60(volume_m3, surface_area_m2, realised),
        eyring_s=eyring_rt60(volume_m3, surface_area_m2, realised),
    )


def responses_of_run(
    run_dir: Path,
    *,
    fmax_hz: float,
    delivery_rate_hz: float = DELIVERY_RATE_HZ,
    comms_path: Path | None = None,
    lowcut_hz: float = 10.0,
) -> tuple[np.ndarray, float]:
    """One run's ``sim_outs.h5`` as ``[receiver, sample]`` at the delivery rate.

    The order is fixed and is not negotiable: reduce, integrate, band limit,
    then resample. Resampling first would alias the dispersive top of the grid's
    band down into the audible range, which is precisely the part that is known
    to be wrong.

    ``lowcut_hz`` is the high pass that travels with the integrator, and its
    default of 10 Hz is the reference implementation's. For a small room that
    default is too low to be useful and it was measured to be actively
    misleading: on the first W20 run, 99.7 per cent of the energy in the last
    half second of the response sat below 50 Hz, under the room's own first
    axial mode at about 40 Hz, where no standing wave can exist and where the
    fitted boundary filters absorb almost nothing. Left in, that residue is not
    reverberation but it dominates every decay measure taken from the tail.
    Callers with a room should pass its first axial mode frequency and say so.
    """
    reduced, differentiated = audio.read_engine_output(run_dir, comms_path)
    signals = audio.integrate_and_lowcut(
        reduced.signals,
        1.0 / reduced.sample_rate_hz,
        differentiated=differentiated,
        fcut=lowcut_hz,
    )
    signals = audio.lowpass(signals, reduced.sample_rate_hz, fmax_hz)
    return audio.resample_to(signals, reduced.sample_rate_hz, delivery_rate_hz), delivery_rate_hz


def _sealed_from_models(model_json: Path) -> dict[str, object] | None:
    """The sealed-volume census written beside the exported model, if any.

    Read rather than recomputed: the number the viewer shows has to be the one
    the solver was given, and recomputing it here would let the two drift.
    """
    manifest = Path(model_json).parent / "manifest.json"
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text()).get("sealed")
    return record if isinstance(record, dict) else None


def measure_all(
    ir: np.ndarray, sample_rate_hz: float, *, fmax_hz: float | None = None
) -> list[dict[str, Any]]:
    """Per band RT60, EDT, C50 and direct to reverberant, one row per receiver.

    ``bands_hz`` comes from the filter bank rather than from ``OCTAVE_BANDS``,
    because at 48 kHz the bank produces one more band than the material data is
    defined on. ``in_band`` marks which of those bands the solver actually
    simulated: anything centred above ``fmax_hz`` holds only the skirt of the
    low pass, so a decay time measured there describes the filter and not the
    room. The values are still reported, because deleting a measurement is worse
    than labelling it, but nothing above the cut should be read as acoustics.
    """
    rows = []
    for index in range(ir.shape[0]):
        band = metrics.measure(ir[index], int(sample_rate_hz))
        rows.append(
            {
                "receiver": index,
                "bands_hz": [int(value) for value in band.bands],
                "in_band": [
                    fmax_hz is None or float(centre) <= float(fmax_hz) for centre in band.bands
                ],
                "rt60_s": [_finite(value) for value in band.rt60],
                "edt_s": [_finite(value) for value in band.edt],
                "c50_db": [_finite(value) for value in band.c50],
                "drr_db": [_finite(value) for value in band.drr],
            }
        )
    return rows


def _finite(value: float) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else round(number, 4)


def build_response_set(
    ir: np.ndarray,
    sample_rate_hz: float,
    source: np.ndarray,
    receivers: np.ndarray,
    provenance: Provenance,
    volume_m3: float | None,
) -> ResponseSet:
    return ResponseSet(
        ir=ir,
        sample_rate_hz=sample_rate_hz,
        source_position=np.asarray(source, dtype=float),
        receiver_positions=np.asarray(receivers, dtype=float),
        provenance=provenance,
        room_volume_m3=volume_m3,
    )


def render_audio(
    ir: np.ndarray,
    sample_rate_hz: float,
    dry: np.ndarray,
    out_dir: Path,
    source_index: int,
) -> list[dict[str, Any]]:
    """One wet WAV per receiver, plus the dry voice once, and the gains used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(ir.shape[0]):
        wet = audio.convolve(dry, ir[index])
        path = out_dir / f"source{source_index}_receiver{index}_wet.wav"
        gain = audio.write_wav(path, wet, sample_rate_hz)
        written.append(
            {
                "path": path.name,
                "receiver": index,
                "seconds": round(wet.size / sample_rate_hz, 3),
                "write_gain": gain,
                "note": NOT_BINAURAL,
            }
        )
    return written


def dry_voice(
    store: ObjectStore, seed: int, *, dataset: str = "ears", shard: str = "p001-0000"
) -> tuple[np.ndarray, dict[str, Any]]:
    """One anechoic spoken passage, and the provenance that must travel with it."""
    clips = catalogue(store, dataset, shard)
    clip = choose_clip(clips, np.random.default_rng(seed), min_seconds=10.0)
    samples, rate = read_wav(fetch_clip(store, clip))
    if rate != int(DELIVERY_RATE_HZ):
        samples = audio.resample_to(samples[np.newaxis, :], float(rate), DELIVERY_RATE_HZ)[0]
    return samples, {
        "dataset": dataset,
        "shard": shard,
        "clip_id": clip.clip_id,
        "member_name": clip.member_name,
        "crc32": clip.crc32,
        "seconds": round(samples.size / DELIVERY_RATE_HZ, 3),
        "source_rate_hz": rate,
        "licence": EARS_LICENCE,
        "attribution": EARS_ATTRIBUTION,
        "anechoic": True,
    }


def publish(store: ObjectStore, run_id: str, files: list[Path]) -> dict[str, str]:
    """Push a run's artefacts under ``runs/<run_id>/`` and return their digests."""
    digests = {}
    for path in files:
        digests[path.name] = store.put_file(f"runs/{run_id}/{path.name}", path)
    return digests


def write_artefacts(
    response: ResponseSet, out_dir: Path, name: str, *, title: str, licence: str
) -> list[Path]:
    """Both files of the declared format, side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_raw(response, out_dir / f"{name}.h5"),
        write_sofa(response, out_dir / f"{name}.sofa", title=title, licence=licence),
    ]


def load_plan(run_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((run_dir / "plan.json").read_text())
    return record


@dataclass(frozen=True)
class RoomGeometry:
    """The room as the solver's own boundary sees it, not as the mesh drew it.

    Sabine and Eyring want a volume, an exposed surface and a mean absorption.
    Taking them from the exported mesh would describe a room the solver never
    simulated: the voxelisation staircases every surface, and it discards
    anything too thin to survive the grid. So the surface and the absorption are
    read back out of ``vox_out.h5`` and ``sim_mats.h5``, weighted by the
    solver's own per node surface area factor, and the absorption is the random
    incidence absorption of the fitted admittance filters themselves.

    Two things are worth saying about the numbers this produces.

    Boundary nodes whose material index is ``-1`` are perfectly rigid. They are
    the grid box outside the room's shell, not room surface, so they are
    excluded from the surface area and named in the record rather than quietly
    dropped.

    The remaining area is far larger than the shell's own area, because
    furniture is a two sided obstacle in a wave solver and a staircased surface
    is longer than the plane it approximates. That is the surface the field
    actually meets, so it is the surface the theory is given.
    """

    volume_m3: float
    surface_area_m2: float
    mean_absorption: float
    rigid_area_m2: float
    shell_area_m2: float
    #: c / 2L on the longest interior dimension. Nothing below this is a mode
    #: of this room, so nothing below it in the response is reverberation.
    first_axial_mode_hz: float
    per_class: list[dict[str, Any]]

    def record(self) -> dict[str, Any]:
        return {
            "volume_m3": round(self.volume_m3, 4),
            "surface_area_m2": round(self.surface_area_m2, 4),
            "mean_absorption": round(self.mean_absorption, 4),
            "rigid_area_m2": round(self.rigid_area_m2, 4),
            "shell_area_m2": round(self.shell_area_m2, 4),
            "first_axial_mode_hz": round(self.first_axial_mode_hz, 3),
            "per_class": self.per_class,
            "note": (
                "surface and absorption are measured on the solver's boundary "
                "nodes, weighted by its own surface area factor; rigid nodes "
                "are the grid box outside the shell and are excluded"
            ),
        }


def material_labels(model_json: Path) -> list[str]:
    """Material names in the solver's own index order.

    The reference implementation sorts the label list and drops ``_RIGID``
    before writing ``mat_00_DEF`` onwards, so index ``i`` is the ``i``-th name
    in sorted order. Read from its source rather than assumed, because the
    export writes them in mesh order and the two are not the same.
    """
    labels = sorted(json.loads(model_json.read_text())["mats_hash"])
    return [label for label in labels if label != "_RIGID"]


def room_geometry(
    entry_path: Path, model_json: Path, *, sound_speed_m_s: float = 343.0
) -> RoomGeometry:
    """Volume from the shell mesh, surface and absorption from the boundary.

    ``entry_path`` is the voxelisation cache entry, not a run directory: the
    engine deletes ``vox_out.h5`` and ``sim_mats.h5`` from a run once it has
    finished with them, and the cache entry is where they are kept.
    """
    import h5py
    import trimesh

    from reverberate.materials.extrapolation import random_incidence_absorption
    from reverberate.materials.impedance import admittance

    shell = json.loads(model_json.read_text())["mats_hash"]["shell"]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(shell["pts"], dtype=float),
        faces=np.asarray(shell["tris"], dtype=int),
        process=True,
    )
    if not mesh.is_watertight:
        raise ValueError(f"{model_json}: the shell is not watertight, so its volume is undefined")
    volume = abs(float(mesh.volume))
    shell_area = float(mesh.area)
    longest = float(np.max(mesh.bounds[1] - mesh.bounds[0]))

    with h5py.File(entry_path / "vox_out.h5", "r") as vox:
        step = float(vox["h"][()])
        material = np.asarray(vox["mat_bn"][:], dtype=int)
        factor = np.asarray(vox["saf_bn"][:], dtype=float)
    cell_area = factor * step * step

    with h5py.File(entry_path / "sim_mats.h5", "r") as mats:
        count = int(mats["Nmat"][()])
        filters = [
            np.asarray(mats[f"mat_{index:02d}_DEF"][:], dtype=float) for index in range(count)
        ]

    labels = material_labels(model_json)
    rows: list[dict[str, Any]] = []
    total = 0.0
    weighted = 0.0
    for index, triplets in enumerate(filters):
        area = float(cell_area[material == index].sum())
        if area <= 0.0:
            continue
        surface = 1.0 / admittance(triplets, np.asarray(OCTAVE_BANDS, dtype=float))
        absorption = random_incidence_absorption(surface)
        mean = float(np.mean(absorption))
        total += area
        weighted += area * mean
        rows.append(
            {
                "material_index": index,
                "label": labels[index] if index < len(labels) else None,
                "area_m2": round(area, 4),
                "bands_hz": [int(value) for value in OCTAVE_BANDS],
                "random_incidence_absorption": [round(float(v), 4) for v in absorption],
                "mean_absorption": round(mean, 4),
            }
        )
    if total <= 0.0:
        raise ValueError(
            f"{entry_path}: no boundary node carries a material, so absorption is undefined"
        )

    return RoomGeometry(
        volume_m3=volume,
        surface_area_m2=total,
        mean_absorption=weighted / total,
        rigid_area_m2=float(cell_area[material < 0].sum()),
        first_axial_mode_hz=sound_speed_m_s / (2.0 * longest),
        shell_area_m2=shell_area,
        per_class=rows,
    )


def _sound_speed(entry_path: Path) -> float:
    import h5py

    with h5py.File(entry_path / "sim_consts.h5", "r") as consts:
        return float(consts["c"][()])


def _provenance(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    run_dir: Path,
    solve_record: dict[str, Any] | None,
    scene_sha256: str,
) -> Provenance:
    return Provenance(
        scene_sha256=scene_sha256,
        mats_hash=str(manifest["key"]),
        engine=str((solve_record or {}).get("engine", "cpu")),
        band="wave",
        fmax_hz=float(manifest["fmax"]),
        grid_step_m=float(manifest["h_m"]),
        points_per_wavelength=float(manifest["ppw"]),
        sound_speed_m_s=_sound_speed(run_dir),
        seed=int(plan["placement"]["seed"]),
        run_id=run_dir.parent.name,
        solver_commit=str(manifest.get("pffdtd_commit", "")),
        wall_clock_s=(
            float(solve_record["engine_s"])
            if solve_record and solve_record.get("engine_s") is not None
            else None
        ),
        notes=NOT_BINAURAL,
    )


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear recipe
    """Render both runs: responses, measures, audio, artefacts, report."""
    import argparse
    import hashlib

    from reverberate import auth
    from reverberate.store import B2Store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="the w20 run directory")
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--seed", type=int, default=20250101)
    parser.add_argument("--publish", action="store_true", help="push artefacts to the store")
    args = parser.parse_args(argv)

    from reverberate.wave.voxelise import cache_root

    entry_path = cache_root() / args.cache_key
    manifest: dict[str, Any] = json.loads((entry_path / "manifest.json").read_text())
    plan = load_plan(args.run)
    model_json = Path(str(manifest["model_json"]))
    scene_sha256 = hashlib.sha256(model_json.read_bytes()).hexdigest()

    solve_path = args.run / "solve.json"
    solves = json.loads(solve_path.read_text())["runs"] if solve_path.exists() else []

    geometry = room_geometry(entry_path, model_json, sound_speed_m_s=_sound_speed(entry_path))
    prediction = theory(geometry.volume_m3, geometry.surface_area_m2, geometry.mean_absorption)
    # The same formula on the shell alone, as the bound at the other end. The
    # full boundary counts every staircased furniture face, including faces the
    # field never reaches because they are inside a closed object, so it
    # overstates the absorbing surface. The shell understates it by ignoring
    # furniture entirely. The measurement should fall between the two, and
    # saying which end it sits nearer is the useful statement.
    shell = next((row for row in geometry.per_class if row["label"] == "shell"), None)
    empty_room = (
        theory(geometry.volume_m3, float(shell["area_m2"]), float(shell["mean_absorption"]))
        if shell is not None
        else None
    )

    auth.inject()
    store = B2Store()
    dry, voice = dry_voice(store, args.seed)

    audio_dir = args.run / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio.write_wav(audio_dir / "dry_voice.wav", dry, DELIVERY_RATE_HZ)

    receivers = np.vstack(
        [np.asarray(r["position"], dtype=float) for r in plan["placement"]["receivers"]]
    )
    sources = plan["placement"]["sources"]

    written: list[Path] = [audio_dir / "dry_voice.wav"]
    per_source: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        run_dir = args.run / f"source{index}"
        comms = args.run / "comms" / f"source{index}.h5"
        ir, rate = responses_of_run(
            run_dir,
            fmax_hz=float(manifest["fmax"]),
            comms_path=comms if comms.exists() else None,
            lowcut_hz=geometry.first_axial_mode_hz,
        )
        record = next((r for r in solves if r.get("source_index") == index), None)
        response = build_response_set(
            ir,
            rate,
            np.asarray(source["position"], dtype=float),
            receivers,
            _provenance(manifest, plan, run_dir, record, scene_sha256),
            geometry.volume_m3,
        )
        written += write_artefacts(
            response,
            args.run / "responses",
            f"source{index}",
            title=f"reverberate W20 first listen, source {index}",
            licence="see report.json; the dry voice carries its own licence",
        )
        written += [
            audio_dir / entry["path"] for entry in render_audio(ir, rate, dry, audio_dir, index)
        ]
        per_source.append(
            {
                "source_index": index,
                "position": source["position"],
                "archetype": source.get("archetype"),
                "samples": int(ir.shape[1]),
                "sample_rate_hz": rate,
                "seconds": round(ir.shape[1] / rate, 4),
                "peak": round(float(np.max(np.abs(ir))), 6),
                "measures": measure_all(ir, rate, fmax_hz=float(manifest["fmax"])),
                "solve": record,
            }
        )

    report = {
        "run": args.run.name,
        "scene_sha256": scene_sha256,
        "model_json": str(model_json),
        "cache_key": args.cache_key,
        "responses": int(len(sources) * receivers.shape[0]),
        "binaural": False,
        "binaural_note": NOT_BINAURAL,
        "placement_seed": int(plan["placement"]["seed"]),
        "placement": plan["placement"],
        "cost": plan["cost"],
        "room": geometry.record(),
        # Carried through from the scene description so the run page can draw
        # what the solver sealed. Absent on runs exported before the census.
        "sealed": _sealed_from_models(model_json),
        "low_cut_hz": round(geometry.first_axial_mode_hz, 3),
        "low_cut_reason": (
            "the integrator's residue below the room's first axial mode is not "
            "reverberation; measured at 99.7 per cent of the tail energy below "
            "50 Hz before this cut was applied"
        ),
        "band_note": (
            "the filter bank runs to Nyquist, so it returns one more band than "
            "the material data is defined on. Bands with in_band false sit above "
            "the simulated fmax and contain only the low pass skirt; their decay "
            "times describe the filter, not the room"
        ),
        "measured_anomaly": {
            "what": (
                "RT60 rose with frequency, from about 0.7 s at 250 Hz to 2.0 to 2.9 s at "
                "2 and 4 kHz, on every receiver of both sources"
            ),
            "why_it_is_wrong": (
                "the fitted boundary absorption rises with frequency too, most strongly on "
                "the two largest absorbers. A more absorbing boundary cannot produce a "
                "longer decay, so at least one of the two numbers was not measuring the room"
            ),
            "cause": (
                "PFFDTD does not fill solids -- by design, so it can accept non-watertight "
                "scenes -- and for a one-sided surface it takes the material away from the "
                "back side while leaving that node adjacent to its neighbours. The inside of "
                "every closed object was therefore simulated as air behind a perfectly rigid "
                "boundary: absorption zero, Q enormous. KernelAirCart reads all six "
                "neighbours unconditionally, so a missed edge intersection drives the cavity "
                "from the room"
            ),
            "how_it_was_found": [
                "the decay has two slopes and the early one is sane: T20 of 0.85, 0.71, "
                "0.62, 0.67, 0.83, 0.85 s from 125 Hz to 4 kHz",
                "the late tail's spectral flatness is 0.021 against 0.27 early, so discrete "
                "lines and not numerical noise",
                "Schroeder is 286 Hz with 42.8 modes per Hz at 2 kHz, so isolated peaks "
                "there cannot be room modes",
                "the lines sit near 2091 Hz, implying 8.2 cm cavities, and a flood fill of "
                "the grid found 211 pockets in the 1 to 4 kHz band averaging 8.6 cm",
            ],
            "fix": (
                "two, and neither sufficient alone: obstacles reach the solver as the exact "
                "boolean union of their convex bodies, which removes the buried faces, and "
                "the non-air side of every surface is sealed rather than merely silenced. "
                "T30/T20 went from 1.68, 1.80, 1.70 at 1, 2 and 4 kHz to 1.06, 1.11, 1.11, "
                "and 125 Hz from 4.03 to 1.03 with T30 falling from 3.64 s to 0.45 s"
            ),
            "still_untested": [
                "air absorption is absent, which is the largest omission and acts hardest in "
                "the top octave; it was not the cause here",
                "locally reacting fitted impedance boundaries under-absorb at grazing "
                "incidence; not the cause here either",
                "10.5 points per wavelength is measured at fmax, so numerical dispersion is "
                "worst in the top octave",
            ],
            "status": (
                "closed by W25. Every band now decays straight, T30/T20 between 1.00 and "
                "1.11, and every band sits inside the theory bracket below rather than above "
                "it. Runs made before W25 still carry the defect and their top octaves "
                "should not be quoted"
            ),
        },
        "theory": prediction.record(),
        "theory_shell_only": empty_room.record() if empty_room is not None else None,
        "theory_note": (
            "two bounds, not one prediction: the full boundary counts furniture "
            "faces the field cannot reach and so is too absorbent, the shell "
            "alone ignores furniture and so is not absorbent enough"
        ),
        "dry_voice": voice,
        "sources": per_source,
        "omissions": [
            "no air absorption: the Stokes filter of the reference implementation "
            "is not applied, so the top of the band decays more slowly than in air",
            "no head, torso or pinna: see binaural_note",
        ],
    }
    (args.run / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    written.append(args.run / "report.json")

    if args.publish:
        report["published"] = publish(store, args.run.name, written)
        (args.run / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    _print_summary(report, written)
    return 0


def _print_summary(report: dict[str, Any], written: list[Path]) -> None:
    """Say on the terminal what was made and what it measures.

    A command that renders twelve responses and prints nothing gives the reader
    no way to tell success from a silent no-op.
    """
    room = report["room"]
    print(f"{report['responses']} responses, {len(written)} files written")
    print(
        f"room {room['volume_m3']:.3f} m3, {room['surface_area_m2']:.1f} m2 boundary, "
        f"mean absorption {room['mean_absorption']:.3f}, "
        f"first axial mode {report['low_cut_hz']:.1f} Hz"
    )
    full, shell = report["theory"], report["theory_shell_only"]
    print(
        f"theory RT60 {full['sabine_rt60_s']:.2f} s full boundary, "
        f"{shell['sabine_rt60_s']:.2f} s shell only (Sabine)"
        if shell
        else f"theory RT60 {full['sabine_rt60_s']:.2f} s (Sabine)"
    )
    for entry in report["sources"]:
        rows = entry["measures"]
        mid = [
            row["rt60_s"][index]
            for row in rows
            for index, (centre, ok) in enumerate(zip(row["bands_hz"], row["in_band"], strict=True))
            if ok and centre in (500, 1000) and row["rt60_s"][index] is not None
        ]
        span = f"{min(mid):.2f} to {max(mid):.2f} s" if mid else "not measurable"
        solve = entry["solve"] or {}
        engine = solve.get("engine", "unknown")
        seconds = solve.get("engine_s")
        where = f"{engine} in {seconds:.0f} s" if seconds is not None else engine
        print(
            f"source {entry['source_index']}: peak {entry['peak']:.4f}, "
            f"measured RT60 at 500 Hz and 1 kHz {span}, solved on {where}"
        )
    print(f"dry voice {report['dry_voice']['member_name']}, {report['dry_voice']['licence']}")
    anomaly = report.get("measured_anomaly")
    if anomaly:
        print(f"open: {anomaly['what']}")


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
