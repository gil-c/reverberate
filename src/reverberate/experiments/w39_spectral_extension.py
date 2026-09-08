"""W39: can the top of the spectrum be synthesised from a solve that stops at 8 kHz?

The question the campaign turns on. At 8 kHz one 80 GB card holds about 206 m2
of room against 26 m2 at 16 kHz, so if the octaves above 8 kHz can be written
from the ones below, the dataset can hold rooms of 200 m2 and the cost falls by
``2^4``. :func:`reverberate.spatial.bands.extend_spectrum` claims it can, for
what the downstream network reads: arrival times and directions of the early
reflections, the decay and level per band, and the interaural cues of the
decode. This experiment measures that claim on a response the project already
owns, and it rents nothing.

**The reference is a real 16 kHz ambisonic response**, ``w10_bedroom_16k``,
re-encoded here from its own stored pressures. It is low passed at the solve
frequency under test, extended back up, and the two are compared **only in the
octaves that were synthesised**, on:

- T30 per octave band on ``W``, against the 5 per cent just noticeable
  difference and the 3.1 per cent seed to seed floor;
- the energy envelope in 5 ms blocks over the first 100 ms, which is where
  the early reflections are, as an RMS error in decibels;
- the direction of arrival of the direct sound per band;
- interaural delay, level difference and late coherence after a rigid sphere
  decode, per band, which is what the separation network will see.

**What is not measured, and why.** The waveform itself: two grids of the same
room already differ by -6 dB sample by sample (W3), so a synthesised octave
cannot be expected to match a solved one in fine structure and the project
already compares statistics there and never waveforms.

Usage::

    python -m reverberate.experiments.w39_spectral_extension \\
        --run data/runs/w10_bedroom_16k --solved-to 8000 --out data/runs/w39_extension_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio, metrics
from reverberate.audio import Atmosphere
from reverberate.experiments.engine import write_record
from reverberate.experiments.w10_render import (
    band_directions,
    binaural_measures,
    decoders,
    encode_run,
)
from reverberate.experiments.w37_window import mean_absorption_of
from reverberate.spatial.bands import extend_spectrum
from reverberate.spatial.binaural import render
from reverberate.spatial.encode import Ambisonic, EncoderSettings
from reverberate.spatial.validate import direction_to_scene_point
from reverberate.tail import transpose

__all__ = ["compare", "main", "validate"]

#: Above this many per cent of T30 error the octave is not usable, per section 9.1.
T30_JND_PCT = 5.0


def _envelope_db(
    signal: np.ndarray, rate: int, *, block_s: float = 0.005, span_s: float = 0.1
) -> np.ndarray:
    """Energy per block over the early part, in decibels relative to the peak block."""
    block = int(round(block_s * rate))
    count = int(round(span_s / block_s))
    blocks = signal[: block * count].reshape(count, block)
    energy = (blocks**2).sum(axis=1)
    return np.asarray(10.0 * np.log10(np.maximum(energy, 1e-30) / max(energy.max(), 1e-30)))


def compare(
    truth: Ambisonic,
    extended: Ambisonic,
    *,
    solved_to_hz: float,
    source: np.ndarray,
    order: int,
) -> dict[str, Any]:
    """Every measure, on the synthesised octaves, truth beside synthesis."""
    rate = int(round(truth.sample_rate_hz))
    centres = np.asarray(metrics.band_centres(rate), dtype=float)
    synthesised = centres > 0.9 * solved_to_hz
    truth_t30 = metrics.rt60_per_band(truth.signals[0], rate)
    made_t30 = metrics.rt60_per_band(extended.signals[0], rate)
    t30_rows = []
    for band, centre in enumerate(centres):
        error = (
            None
            if not (np.isfinite(truth_t30[band]) and np.isfinite(made_t30[band]))
            else 100.0 * (made_t30[band] - truth_t30[band]) / truth_t30[band]
        )
        t30_rows.append(
            {
                "band_hz": int(centre),
                "synthesised": bool(synthesised[band]),
                "truth_s": None
                if not np.isfinite(truth_t30[band])
                else round(float(truth_t30[band]), 4),
                "synthesis_s": None
                if not np.isfinite(made_t30[band])
                else round(float(made_t30[band]), 4),
                "error_pct": None if error is None else round(float(error), 2),
                "within_jnd": None if error is None else bool(abs(error) <= T30_JND_PCT),
            }
        )

    truth_bands = metrics.octave_filter(truth.signals[0], rate)
    made_bands = metrics.octave_filter(extended.signals[0], rate)
    envelope_rows = []
    for band, centre in enumerate(centres):
        if not synthesised[band]:
            continue
        a = _envelope_db(truth_bands[band], rate)
        b = _envelope_db(made_bands[band], rate)
        level = 10.0 * np.log10(
            (made_bands[band] ** 2).sum() / max((truth_bands[band] ** 2).sum(), 1e-30)
        )
        envelope_rows.append(
            {
                "band_hz": int(centre),
                "early_envelope_rms_error_db": round(float(np.sqrt(np.mean((a - b) ** 2))), 3),
                "early_envelope_max_error_db": round(float(np.max(np.abs(a - b))), 3),
                "band_level_error_db": round(float(level), 3),
            }
        )

    reference = direction_to_scene_point(truth, source)
    doa_truth = {row["band_hz"]: row for row in band_directions(truth, reference)}
    doa_made = {row["band_hz"]: row for row in band_directions(extended, reference)}
    doa_rows = [
        {
            "band_hz": band,
            "synthesised": bool(band > 0.9 * solved_to_hz),
            "truth_error_deg": doa_truth[band].get("error_deg"),
            "synthesis_error_deg": doa_made[band].get("error_deg"),
            "usable_truth": doa_truth[band].get("usable"),
            "usable_synthesis": doa_made[band].get("usable"),
        }
        for band in doa_truth
    ]

    built, _ = decoders(order, float(rate), plain=True)
    decoder = next(iter(built.values()))
    azimuth = float(np.arctan2(reference[1], reference[0]))
    ears_truth = binaural_measures(render(truth, decoder), float(rate), azimuth)
    ears_made = binaural_measures(render(extended, decoder), float(rate), azimuth)
    binaural = {"truth": ears_truth, "synthesis": ears_made}

    return {
        "t30": t30_rows,
        "early_envelope": envelope_rows,
        "direction_of_arrival": doa_rows,
        "binaural": binaural,
    }


def validate(
    run_dir: Path,
    *,
    solved_to_hz: float,
    ceiling_hz: float | None = None,
    order: int = 7,
    fit_order: int = 10,
    atmosphere: Atmosphere | None = None,
) -> dict[str, Any]:
    """Low pass a real 16 kHz response, synthesise the top back, and measure it."""
    atmosphere = atmosphere or Atmosphere()
    settings = EncoderSettings(order=order, fit_order=fit_order, max_frequency_hz=16000.0)
    encoded = encode_run(run_dir, settings, air=atmosphere)
    truth = encoded.ambisonic
    rate = int(round(truth.sample_rate_hz))
    solved = Ambisonic(
        signals=audio.lowpass(truth.signals, float(rate), solved_to_hz),
        sample_rate_hz=truth.sample_rate_hz,
        order=truth.order,
        centre=truth.centre,
    )
    centres = metrics.band_centres(rate)
    if encoded.room and encoded.room.get("per_class"):
        absorption = mean_absorption_of({"room": encoded.room}, len(centres))
        absorption_source = "the solver's own boundary"
    else:
        absorption = np.full(len(centres), 0.25)
        absorption_source = "assumed flat 0.25"
    prediction = transpose(
        metrics.rt60_per_band(solved.signals[0], rate),
        absorption,
        rate,
        # The band that holds the solve's own edge is predicted, not copied: half
        # of it is the low pass skirt and its measured decay is not the room's.
        mid_fmax_hz=0.7 * solved_to_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=float(encoded.plan["sound_speed_m_s"]),
    )
    extended, record = extend_spectrum(
        solved,
        fmax_hz=solved_to_hz,
        t60_s=np.asarray(prediction.t60_s, dtype=float),
        atmosphere=atmosphere,
        sound_speed_m_s=float(encoded.plan["sound_speed_m_s"]),
        ceiling_hz=ceiling_hz,
    )
    measures = compare(
        truth, extended, solved_to_hz=solved_to_hz, source=encoded.source, order=order
    )
    return {
        "run": "w39_extension_check",
        "reference_run": run_dir.name,
        "trick": (
            f"the reference low passed at {solved_to_hz:g} Hz and the octaves above written "
            "back from the ones below by spatial.bands.extend_spectrum"
        ),
        "solved_to_hz": solved_to_hz,
        "sample_rate_hz": rate,
        "absorption_for_decay": {
            "per_band": [round(float(v), 4) for v in absorption],
            "source": absorption_source,
        },
        "transposition": prediction.record(),
        "extension": record,
        "air_absorption": atmosphere.record(),
        **measures,
        "not_measured": (
            "the waveform: two grids of one room differ by -6 dB sample by sample (W3), "
            "so a synthesised octave is compared on statistics only"
        ),
    }


def _print(report: dict[str, Any]) -> None:
    print(f"{report['reference_run']} low passed at {report['solved_to_hz']:g} Hz, then extended")
    print(f"{'band':>6} {'T30 truth':>10} {'T30 synth':>10} {'error':>8}  synthesised")
    for row in report["t30"]:
        error = "" if row["error_pct"] is None else f"{row['error_pct']:+6.1f} %"
        print(
            f"{row['band_hz']:>6} {row['truth_s'] or float('nan'):>10.3f} "
            f"{row['synthesis_s'] or float('nan'):>10.3f} {error:>8}  {row['synthesised']}"
        )
    for row in report["early_envelope"]:
        print(
            f"{row['band_hz']:>6} Hz: early envelope rms "
            f"{row['early_envelope_rms_error_db']:.2f} dB, max "
            f"{row['early_envelope_max_error_db']:.2f} dB, band level "
            f"{row['band_level_error_db']:+.2f} dB"
        )
    for row in report["direction_of_arrival"]:
        if row["synthesised"]:
            print(
                f"{row['band_hz']:>6} Hz: direction error truth {row['truth_error_deg']} deg, "
                f"synthesis {row['synthesis_error_deg']} deg"
            )
    for who in ("truth", "synthesis"):
        ears = report["binaural"][who]
        print(
            f"{who:>9}: ITD {ears.get('itd_us')} us, ILD {ears.get('ild_db')} dB, "
            f"late coherence {ears.get('late_coherence')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solved-to", type=float, default=8000.0)
    parser.add_argument("--ceiling", type=float, default=None)
    parser.add_argument("--order", type=int, default=7)
    parser.add_argument("--fit-order", type=int, default=10)
    args = parser.parse_args(argv)
    report = validate(
        args.run,
        solved_to_hz=args.solved_to,
        ceiling_hz=args.ceiling,
        order=args.order,
        fit_order=args.fit_order,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_record(args.out, "report.json", report)
    write_record(
        args.out,
        "plan.json",
        {"run": args.out.name, "scene_id": "n/a", "room": "n/a", "kind": "check"},
    )
    _print(report)
    print(json.dumps({k: report[k] for k in ("extension",)}, indent=1)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
