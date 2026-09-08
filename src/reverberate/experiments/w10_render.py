"""From W10's solved field to an ambisonic response, two ears, and the numbers.

The order of the chain is fixed and each step is here because leaving it out is
a mistake this project has already made or has written down:

1. **reduce**, one weight per receiver rather than eight, since W10's receivers
   are grid nodes;
2. **integrate and low cut**, because the single precision engine differentiates
   its source and reading the output without undoing that gives the derivative
   of the response;
3. **low pass at fmax**, because the stencil is dispersive at the top of its
   band and resampling first would fold exactly that part down into the audible
   range;
4. **resample to 48 kHz**, which is delivery and happens last of the four;
5. **air absorption**, ISO 9613-1 with its two parameters declared, which is
   new and which the roadmap's own table says is worth 38 per cent of T60 at
   16 kHz;
6. **encode**, per frequency, from the shells that frequency can use;
7. **decode**, through a head, with magnitude least squares above 2 kHz.

Steps 4 and 6 commute, which is why the encode may run at 48 kHz: resampling is
the same linear filter on every receiver, so it passes through a per frequency
linear fit untouched, and 24 000 bins is a third of the work of 145 638.

**What this module deliberately does not do is decide anything.** The encoder's
settings, the array and the head all arrive as arguments and all are recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio, metrics
from reverberate.experiments.engine import write_record
from reverberate.experiments.w20_render import dry_voice, publish
from reverberate.response import Provenance
from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.binaural import (
    BinauralDecoder,
    design_decoder,
    ild_db,
    interaural_coherence,
    itd_s,
    render,
)
from reverberate.spatial.encode import Ambisonic, EncoderSettings, encode
from reverberate.spatial.export import (
    AMBIX_NOTE,
    write_ambisonic_sofa,
    write_ambix_wav,
    write_brir_sofa,
)
from reverberate.spatial.hrtf import (
    HEAD_RADIUS_M,
    HrtfSet,
    ear_directions,
    sphere_hrtf,
    woodworth_itd_s,
)
from reverberate.spatial.sh import quadrature
from reverberate.spatial.validate import (
    direct_arrival_sample,
    direction_of_arrival,
    direction_to_scene_point,
    energy_per_order,
)
from reverberate.store import shared_store

__all__ = [
    "DELIVERY_RATE_HZ",
    "YAWS_DEG",
    "band_directions",
    "binaural_measures",
    "decoders",
    "pressures_of_run",
    "sphere_head",
    "write_audio",
    "main",
]

DELIVERY_RATE_HZ = 48000.0

#: What the artefacts are licensed as. HSSD is CC BY-NC, so anything derived
#: from its geometry inherits the non-commercial term, and so does the EARS
#: speech the audio is convolved with.
LICENCE = "CC BY-NC 4.0"

#: Head orientations rendered. Not a sweep: four angles a listener can check by
#: ear, and enough to show that the rotation is the exact one it claims to be.
YAWS_DEG = (0.0, 45.0, 90.0, -90.0)

#: The band the direction of arrival is measured in, per octave. Below 250 Hz a
#: 0.5 s response has too few cycles in a direct sound window to place one.
DOA_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def pressures_of_run(
    run_dir: Path,
    *,
    fmax_hz: float,
    comms_path: Path,
    lowcut_hz: float,
    delivery_rate_hz: float = DELIVERY_RATE_HZ,
    air: dict[str, float] | None = None,
    sound_speed_m_s: float = 343.2,
) -> tuple[np.ndarray, float]:
    """One row per receiver at the delivery rate, air absorption applied or not.

    ``air`` carries ``temperature_c`` and ``humidity_percent``; ``None`` skips
    the filter and is then an omission the report has to name.
    """
    reduced, differentiated = audio.read_engine_output(run_dir, comms_path)
    signals = audio.integrate_and_lowcut(
        reduced.signals,
        1.0 / reduced.sample_rate_hz,
        differentiated=differentiated,
        fcut=lowcut_hz,
    )
    signals = audio.lowpass(signals, reduced.sample_rate_hz, fmax_hz)
    signals = audio.resample_to(signals, reduced.sample_rate_hz, delivery_rate_hz)
    if air is not None:
        signals = audio.apply_air_absorption(
            signals,
            delivery_rate_hz,
            sound_speed_m_s=sound_speed_m_s,
            temperature_c=air["temperature_c"],
            humidity_percent=air["humidity_percent"],
        )
    return signals, delivery_rate_hz


def sphere_head(
    sample_rate_hz: float, filter_length: int, *, quadrature_degree: int = 60
) -> HrtfSet:
    """The analytic head, sampled on the decoder's own frequency grid."""
    grid, weights = quadrature(quadrature_degree)
    frequency = np.fft.rfftfreq(filter_length, 1.0 / sample_rate_hz)
    sampled = sphere_hrtf(grid, frequency)
    return HrtfSet(sampled.responses, grid, frequency, sampled.description, weights=weights)


def decoders(
    order: int,
    sample_rate_hz: float,
    *,
    filter_length: int = 512,
    magls_cut_on_hz: float = 2000.0,
) -> dict[str, BinauralDecoder]:
    """The decoders this run renders through, keyed by the name they are filed under.

    Both the plain least squares and the magnitude one, because the difference
    between them is a measurement this run reports rather than a setting it
    picks.
    """
    head = sphere_head(sample_rate_hz, filter_length)
    return {
        "sphere_magls": design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=magls_cut_on_hz,
        ),
        "sphere_plain": design_decoder(
            head,
            order=order,
            sample_rate_hz=sample_rate_hz,
            filter_length=filter_length,
            magls_cut_on_hz=float("inf"),
            covariance_constraint=False,
        ),
    }


def band_directions(
    ambisonic: Ambisonic,
    reference: np.ndarray,
    *,
    bands_hz: tuple[float, ...] = DOA_BANDS_HZ,
    window_s: float = 0.002,
    start: int | None = None,
) -> list[dict[str, Any]]:
    """Direction of arrival of the direct sound, per octave band, against the geometry.

    This is the check that catches every silent failure at once. A mirrored
    frame, an inverted odd order or the wrong kind of Hankel function all leave
    every level untouched and move this number by tens of degrees. At the top
    of the band it also measures something real rather than only guarding: the
    grid's own dispersion is direction dependent, so a residue here at 16 kHz
    that is absent at 1 kHz is the scheme and not the code.
    """
    rate = ambisonic.sample_rate_hz
    arrival = direct_arrival_sample(ambisonic) if start is None else start
    length = max(int(round(window_s * rate)), 16)
    rows = []
    for centre in bands_hz:
        low, high = centre / np.sqrt(2.0), min(centre * np.sqrt(2.0), 0.49 * rate)
        if low >= high:
            continue
        filtered = audio.lowpass(ambisonic.signals, rate, high)
        filtered = filtered - audio.lowpass(filtered, rate, low)
        banded = Ambisonic(filtered, rate, ambisonic.order, ambisonic.centre)
        measured = direction_of_arrival(banded, start=max(arrival - length // 4, 0), length=length)
        # A window shorter than a period cannot place a direction: the pressure
        # and velocity product integrates over less than a cycle and the answer
        # comes back reversed. A reader must not have to know that to read the
        # table, so each row says whether it may be read.
        #
        # The two thresholds are calibrated on W10's rehearsal rather than
        # chosen: there, half a cycle gave a 180 degree error at a
        # concentration of 0.53, one cycle gave 0.07 degrees at 0.88, and
        # everything above gave better than 0.02 degrees at 0.99 or more.
        cycles = length / rate * centre
        rows.append(
            {
                "band_hz": int(centre),
                "error_deg": round(measured.angle_to_deg(reference), 3),
                "cycles_in_window": round(float(cycles), 2),
                "usable": bool(cycles >= 1.0 and measured.concentration >= 0.6),
                **measured.record(),
            }
        )
    return rows


def binaural_measures(
    brir: np.ndarray, sample_rate_hz: float, source_azimuth_rad: float
) -> dict[str, Any]:
    """Interaural delay, level difference and coherence, beside what a sphere predicts."""
    times, coherence = interaural_coherence(brir, sample_rate_hz)
    late = times > (times[0] + 0.05) if times.size else np.zeros(0, dtype=bool)
    return {
        "itd_us": round(itd_s(brir, sample_rate_hz) * 1e6, 1),
        "woodworth_itd_us": round(float(woodworth_itd_s(np.array(source_azimuth_rad))) * 1e6, 1),
        "ild_db": round(ild_db(brir), 2),
        "late_coherence": round(float(np.mean(coherence[late])), 4) if late.any() else None,
        "late_coherence_note": (
            "roadmap 5.5 predicts a diffuse tail is nearly incoherent between "
            "the ears above 1 kHz, about 0.04 at 8 kHz"
        ),
    }


def _octave_rows(signal: np.ndarray, sample_rate_hz: float, fmax_hz: float) -> dict[str, Any]:
    band = metrics.measure(signal, int(sample_rate_hz))
    return {
        "bands_hz": [int(v) for v in band.bands],
        "in_band": [float(v) <= fmax_hz for v in band.bands],
        "rt60_s": [_finite(v) for v in band.rt60],
        "edt_s": [_finite(v) for v in band.edt],
        "c50_db": [_finite(v) for v in band.c50],
        "drr_db": [_finite(v) for v in band.drr],
    }


def _finite(value: float) -> float | None:
    number = float(value)
    return None if not np.isfinite(number) else round(number, 4)


def rehearsal_report(
    run_dir: Path, settings: EncoderSettings, *, window_pad: int = 0
) -> dict[str, Any]:
    """What the box was run for: does the chain point the right way, and how far off is 16 kHz.

    The response is cut at the first wall reflection, which the plan computed
    from the geometry before anything was solved, so what is encoded is a free
    field monopole and its answer is known. The encode is done twice, with the
    ideal wavenumber and with the scheme's own, and the difference between the
    two direction errors is the whole question.
    """
    plan = json.loads((run_dir / "plan.json").read_text())
    rehearsal = plan["rehearsal"]
    positions = np.load(run_dir / "array_positions.npy")
    design = ArrayDesign(
        centre=np.asarray(plan["centre"], dtype=float),
        positions=positions,
        offsets=positions - np.asarray(plan["centre"], dtype=float),
        radii=np.linalg.norm(positions - np.asarray(plan["centre"], dtype=float), axis=1),
        shell=np.zeros(positions.shape[0], dtype=int),
        nominal_radii=(rehearsal["array_radius_cells"] * rehearsal["step_m"],),
        grid_step_m=rehearsal["step_m"],
    )
    grid_rate = float(plan["sample_rate_hz"])
    cut = int(rehearsal["first_reflection_sample"]) + window_pad

    reduced, differentiated = audio.read_engine_output(
        run_dir / "source0", run_dir / "comms" / "source0.h5"
    )
    signals = audio.integrate_and_lowcut(
        reduced.signals, 1.0 / grid_rate, differentiated=differentiated, fcut=20.0
    )
    signals = audio.lowpass(signals, grid_rate, plan["encoder"]["max_frequency_hz"])
    direct = signals[:, :cut]

    reference = direction_to_scene_point(
        Ambisonic(np.zeros((4, 4)), grid_rate, 1, design.centre),
        np.asarray(plan["source"], dtype=float),
    )
    rows = {}
    for dispersion in ("ideal", "numerical"):
        variant = EncoderSettings(
            order=settings.order,
            fit_order=settings.fit_order,
            gain_cap_db=settings.gain_cap_db,
            gate_margin=settings.gate_margin,
            gate_taper=settings.gate_taper,
            max_frequency_hz=settings.max_frequency_hz,
            dispersion=dispersion,
        )
        ambisonic = encode(
            direct, grid_rate, design, sound_speed_m_s=plan["sound_speed_m_s"], settings=variant
        )
        rows[dispersion] = band_directions(
            ambisonic, reference, window_s=cut / grid_rate * 0.8, start=0
        )
    return {
        "run": run_dir.name,
        "what": (
            "the whole spatial chain on a box, over the window in which the "
            "array hears the direct sound alone"
        ),
        "rehearsal": rehearsal,
        "window_samples": cut,
        "reference_direction": [round(float(v), 5) for v in reference],
        "direction_of_arrival": rows,
        "reading": (
            "read only the rows marked usable: a direct sound window is short, "
            "and below a couple of periods an intensity vector can come back "
            "reversed. Where it is usable, the error is the chain's own at the "
            "bottom of the band and the grid's direction dependent dispersion "
            "at the top, which is the only thing that changes between the two "
            "encodes"
        ),
    }


def room_report(
    run_dir: Path,
    settings: EncoderSettings,
    *,
    air: dict[str, float] | None,
    lowcut_hz: float,
    yaws_deg: tuple[float, ...] = YAWS_DEG,
    filter_length: int = 512,
    rendered: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode the room run, decode it through the heads, and measure all of it.

    ``rendered`` is filled in with the encoded response and the decoded ear
    signals when a mapping is passed, so a caller that wants to write audio or
    artefacts does not have to encode a second time. Nothing is written here.
    """
    plan = json.loads((run_dir / "plan.json").read_text())
    positions = np.load(run_dir / "array_positions.npy")
    centre = np.asarray(plan["centre"], dtype=float)
    source = np.asarray(plan["source"], dtype=float)
    extras = np.asarray(plan["extra_receivers"], dtype=float).reshape(-1, 3)
    design = ArrayDesign(
        centre=centre,
        positions=positions,
        offsets=positions - centre,
        radii=np.linalg.norm(positions - centre, axis=1),
        shell=np.zeros(positions.shape[0], dtype=int),
        nominal_radii=(float(np.linalg.norm(positions - centre, axis=1).max()),),
        grid_step_m=float(plan["array"]["grid_step_m"]),
    )
    fmax = float(settings.max_frequency_hz or 16000.0)
    signals, rate = pressures_of_run(
        run_dir / "source0",
        fmax_hz=fmax,
        comms_path=run_dir / "comms" / "source0.h5",
        lowcut_hz=lowcut_hz,
        air=air,
        sound_speed_m_s=float(plan["sound_speed_m_s"]),
    )
    array_rows, extra_rows = signals[: design.count], signals[design.count :]

    ambisonic = encode(
        array_rows, rate, design, sound_speed_m_s=float(plan["sound_speed_m_s"]), settings=settings
    )
    reference = direction_to_scene_point(ambisonic, source)
    times, order_energy = energy_per_order(ambisonic)

    built = decoders(settings.order, rate, filter_length=filter_length)
    azimuth = float(np.arctan2(reference[1], reference[0]))
    binaural: dict[str, Any] = {}
    responses: dict[str, dict[float, np.ndarray]] = {}
    for name, decoder in built.items():
        rows = {}
        responses[name] = {}
        for yaw in yaws_deg:
            # The listener turns left by yaw, so the field turns right by it.
            brir = render(ambisonic, decoder, field_yaw_rad=np.radians(-yaw))
            responses[name][yaw] = brir
            rows[f"yaw_{int(yaw)}"] = binaural_measures(brir, rate, azimuth - np.radians(yaw))
        binaural[name] = {"decoder": decoder.record(), "measures": rows}

    if rendered is not None:
        rendered["ambisonic"] = ambisonic
        rendered["brirs"] = responses
        rendered["source"] = source
        rendered["centre"] = centre
        rendered["plan"] = plan

    return {
        "run": run_dir.name,
        "scene_id": plan.get("scene_id"),
        "room": plan.get("room"),
        "cache_key": plan.get("cache_key"),
        "geometry_sha256": plan.get("geometry_sha256"),
        "reference_run": plan.get("reference_run"),
        "reference_note": plan.get("reference_note"),
        "binaural": True,
        "sample_rate_hz": rate,
        "encoder": settings.record(),
        "array": plan["array"],
        "conditioning": plan["conditioning"],
        "air_absorption": air
        or {"applied": False, "why": "not requested; the treble of the tail is overstated"},
        "low_cut_hz": lowcut_hz,
        "direction_of_arrival": band_directions(ambisonic, reference),
        "reference_direction": [round(float(v), 5) for v in reference],
        "order_energy": {
            "times_s": [round(float(v), 5) for v in times[:: max(len(times) // 40, 1)]],
            "energy": [
                [round(float(v), 8) for v in row]
                for row in order_energy[:: max(len(times) // 40, 1)]
            ],
            "note": (
                "a diffuse tail carries the same energy per channel in every "
                "order, so these curves should converge as the tail develops"
            ),
        },
        "omnidirectional": _octave_rows(ambisonic.signals[0], rate, fmax),
        "extra_receivers": [
            {"receiver": index, "position": extras[index].tolist(), **_octave_rows(row, rate, fmax)}
            for index, row in enumerate(extra_rows)
        ],
        "binaural_decodes": binaural,
    }


def write_audio(
    ambisonic: Ambisonic,
    brirs: dict[str, dict[float, np.ndarray]],
    dry: np.ndarray,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """The ambisonic WAV, and the anechoic clip through every head and every yaw.

    **One gain over every file written here, and that is the point.** Level is a
    measurement. A gain per file would make a shadowed ear as loud as a near one
    and destroy the interaural level difference, which is the cue this dataset
    exists to carry; within a single ambisonic file it would destroy the
    direction itself, which is nothing but the ratios between the channels. So
    the peak is taken over everything and the same gain is handed to each write,
    recorded beside the realised peak of each file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wet = {
        f"{head}_yaw{int(yaw)}": np.stack([audio.convolve(dry, response[ear]) for ear in range(2)])
        for head, angles in brirs.items()
        for yaw, response in angles.items()
    }
    peak = max(
        [float(np.max(np.abs(block))) for block in wet.values()]
        + [float(np.max(np.abs(ambisonic.signals)))]
    )
    gain = 1.0 if peak == 0.0 else 10.0 ** (-1.0 / 20.0) / peak

    written: list[dict[str, Any]] = []
    ambix, _ = write_ambix_wav(
        Ambisonic(
            ambisonic.signals * gain,
            ambisonic.sample_rate_hz,
            ambisonic.order,
            ambisonic.centre,
        ),
        out_dir / "ambisonic_acn_sn3d.wav",
        headroom_db=0.0,
    )
    written.append({"path": ambix.name, "what": AMBIX_NOTE, "channels": ambisonic.signals.shape[0]})
    for name, block in wet.items():
        path = out_dir / f"binaural_{name}.wav"
        import soundfile

        soundfile.write(
            str(path),
            (block * gain).T,
            int(round(ambisonic.sample_rate_hz)),
            subtype="FLOAT",
            format="WAV",
        )
        written.append(
            {
                "path": path.name,
                "what": "two ears, the anechoic clip through this head at this yaw",
                "peak": round(float(np.max(np.abs(block * gain))), 5),
                "seconds": round(block.shape[1] / ambisonic.sample_rate_hz, 3),
            }
        )
    for entry in written:
        entry["write_gain"] = gain
    return written


def _rehearsal(args: argparse.Namespace) -> int:
    settings = EncoderSettings(
        order=args.order, fit_order=args.fit_order, max_frequency_hz=args.fmax
    )
    write_record(args.run, "report.json", rehearsal_report(args.run, settings))
    return 0


def _room(args: argparse.Namespace) -> int:
    settings = EncoderSettings(
        order=args.order, fit_order=args.fit_order, max_frequency_hz=args.fmax
    )
    air = (
        None
        if args.no_air
        else {"temperature_c": args.temperature, "humidity_percent": args.humidity}
    )
    rendered: dict[str, Any] = {}
    report = room_report(args.run, settings, air=air, lowcut_hz=args.low_cut, rendered=rendered)
    ambisonic = rendered["ambisonic"]
    provenance = Provenance(
        scene_sha256=str(rendered["plan"].get("geometry_sha256") or ""),
        mats_hash=str(rendered["plan"].get("cache_key") or ""),
        engine="cuda",
        band="high",
        fmax_hz=float(args.fmax),
        grid_step_m=float(rendered["plan"]["array"]["grid_step_m"]),
        points_per_wavelength=10.5,
        sound_speed_m_s=float(rendered["plan"]["sound_speed_m_s"]),
        seed=0,
        run_id=args.run.name,
        notes=(
            "ambisonic order "
            f"{settings.order}, N3D, ACN, about one point; the effective order "
            "per frequency is in report.json and is lower below 1 kHz"
        ),
    )
    artefacts = [
        write_ambisonic_sofa(
            ambisonic,
            args.run / "responses" / "ambisonic.sofa",
            source_position=rendered["source"],
            provenance=provenance,
            title=f"{args.run.name}: ambisonic room impulse response",
            licence=LICENCE,
        )
    ]
    ears = ear_directions()
    for head, angles in rendered["brirs"].items():
        yaws = sorted(angles)
        artefacts.append(
            write_brir_sofa(
                np.stack([angles[yaw] for yaw in yaws]),
                np.asarray(yaws, dtype=float),
                args.run / "responses" / f"binaural_{head}.sofa",
                sample_rate_hz=ambisonic.sample_rate_hz,
                listener_position=rendered["centre"],
                source_position=rendered["source"],
                ear_positions=HEAD_RADIUS_M * ears,
                provenance=provenance,
                title=f"{args.run.name}: binaural room impulse responses, {head}",
                licence=LICENCE,
            )
        )
    report["artefacts"] = [path.name for path in artefacts]

    if args.audio:
        store = shared_store()
        if store is None:
            print("no store credentials, so no anechoic clip and no audio")
        else:
            dry, dry_record = dry_voice(store, args.seed)
            report["dry_voice"] = dry_record
            report["audio"] = write_audio(ambisonic, rendered["brirs"], dry, args.run / "audio")
            audio.write_wav(args.run / "audio" / "dry_voice.wav", dry, ambisonic.sample_rate_hz)
    write_record(args.run, "report.json", report)

    if args.publish:
        store = shared_store()
        if store is None:
            raise SystemExit("no store credentials, so nothing can be published")
        files = [*artefacts, args.run / "report.json", args.run / "plan.json"]
        files += sorted((args.run / "audio").glob("*.wav"))
        report["published"] = publish(store, args.run.name, files)
        write_record(args.run, "report.json", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rehearsal = sub.add_parser("rehearsal", help="analyse the box run")
    rehearsal.add_argument("--run", type=Path, required=True)
    rehearsal.add_argument("--order", type=int, default=7)
    rehearsal.add_argument("--fit-order", type=int, default=10)
    rehearsal.add_argument("--fmax", type=float, default=16000.0)
    rehearsal.set_defaults(func=_rehearsal)

    room = sub.add_parser("room", help="encode, decode and measure the room run")
    room.add_argument("--run", type=Path, required=True)
    room.add_argument("--order", type=int, default=7)
    room.add_argument("--fit-order", type=int, default=10)
    room.add_argument("--fmax", type=float, default=16000.0)
    room.add_argument("--low-cut", type=float, required=True, help="the room's first axial mode")
    room.add_argument("--temperature", type=float, default=20.0)
    room.add_argument("--humidity", type=float, default=50.0)
    room.add_argument("--no-air", action="store_true", help="skip air absorption and say so")
    room.add_argument("--audio", action="store_true", help="render the anechoic clip through it")
    room.add_argument("--publish", action="store_true", help="push the artefacts to the store")
    room.add_argument("--seed", type=int, default=20250101)
    room.set_defaults(func=_room)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
