"""W37: the three-band response, assembled and made audible.

The architecture the owner settled on 2026-09-08, priced against the runs this
project already owns:

===== ================ ================================ ==================
band  window           why                              cost on this flat
===== ================ ================================ ==================
low   the whole decay  it is 0.2 per cent of the bill,  free in practice
                       and a synthetic tail below
                       1 kHz would be wrong anyway,
                       since interaural coherence is
                       not negligible there
mid   down to -30 dB   0.082 USD against 0.331 for a    4.0x saved
                       full 0.5 s, so there is no
                       reason to be aggressive
high  down to -20 dB   this is where the money is, and  4.0x saved
                       the error there is already
                       inside the noise floor
===== ================ ================================ ==================

**The window is measured, not typed.** :func:`reverberate.tail.window_for_level_s`
reads it off the energy decay curve of the slowest band a run contributes, so
the rule scales itself: air absorption steepens the decay towards the top, and
one level gives a long window low down and a short one high up.

**What this run can and cannot show.** Only two solves of this room exist, at
4 kHz and 16 kHz. The 4 kHz run therefore serves twice: untouched as the low
band, and truncated as the mid band. That is not the real architecture, where
the low band is its own coarse whole-apartment solve, and it is the right
substitute for the question being asked, because it holds the low band
identical between the two variants and lets the mid and high bands carry the
whole audible difference.

Usage::

    python -m reverberate.experiments.w37_three_band \\
        --mid data/runs/w27_sealed --high data/runs/w29_16k \\
        --out data/runs/w37_three_band
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import audio
from reverberate import bands as band_split
from reverberate import cost as cost_model
from reverberate.air import Atmosphere
from reverberate.air import apply as apply_air
from reverberate.experiments.w20_render import measure_all
from reverberate.experiments.w37_listen import shared_gain
from reverberate.experiments.w37_window import mean_absorption_of
from reverberate.metrics import band_centres, energy_decay_curve, octave_filter, rt60_per_band
from reverberate.response import Provenance, ResponseSet, read_raw, write_raw
from reverberate.tail import local_decay_s, splice, transpose, window_for_level_s

__all__ = ["VARIANTS", "build", "main"]

#: What a listener is asked to compare.
VARIANTS = (
    ("single band", "the 16 kHz solve alone, whole, air applied"),
    ("solved", "three bands, all solved to the end, nothing synthesised"),
    ("economy", "three bands, low whole, mid to -30 dB, high to -20 dB, tails synthesised"),
)

#: Octave bands both solves resolve cleanly, used to put them on one scale.
#: Below the mid run's own low end and well under its fmax, so neither is
#: calibrating on the skirt of its own low pass.
CALIBRATION_HZ = (500.0, 2000.0)

#: The levels the owner chose, per band.
MID_LEVEL_DB = -30.0
HIGH_LEVEL_DB = -20.0

TRICK = (
    "Three bands, three windows, each measured off the decay rather than typed. "
    "The low band is solved whole because it is 0.2 per cent of the bill and a "
    "noise tail below 1 kHz would be physically wrong, interaural coherence not "
    "being negligible there. The mid band stops at -30 dB and the high band at "
    "-20 dB, and a tail calibrated on the band below replaces the rest."
)


def _economy_band(
    wet: np.ndarray,
    sample_rate: int,
    absorption: np.ndarray,
    *,
    level_db: float,
    span_hz: tuple[float, float],
    calibration: np.ndarray,
    calibration_fmax_hz: float,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
    seed: int,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Truncate one band's run at its measured window and splice a tail on."""
    window_s = window_for_level_s(wet, sample_rate, level_db, bands_hz=span_hz)
    out = np.zeros_like(wet)
    predictions: list[dict[str, Any]] = []
    for receiver in range(wet.shape[0]):
        prediction = transpose(
            rt60_per_band(calibration[receiver], sample_rate),
            absorption,
            sample_rate,
            mid_fmax_hz=calibration_fmax_hz,
            atmosphere=atmosphere,
            sound_speed_m_s=sound_speed_m_s,
        )
        # The tail continues the slope it is joined to, not the average of the
        # whole response. Roadmap 5.5 asks for exactly this, and the difference
        # is 13 per cent at 2 kHz when the splice is as deep as -30 dB.
        transposed = np.array(prediction.t60_s)
        local = local_decay_s(wet[receiver : receiver + 1], sample_rate, window_s)
        decay = np.where(np.isfinite(local), local, transposed)
        predictions.append({**prediction.record(), "local_t60_s": local.tolist()})
        out[receiver] = splice(
            wet[receiver],
            sample_rate,
            window_s,
            decay,
            rng=np.random.default_rng(seed + receiver),
        )
    return out, window_s, predictions


def _decay_time(band: np.ndarray, sample_rate: int, end_db: float, factor: float) -> float:
    curve = energy_decay_curve(band)
    lo = np.flatnonzero(curve <= -5.0)
    hi = np.flatnonzero(curve <= end_db)
    if not len(lo) or not len(hi) or hi[0] <= lo[0]:
        return float("nan")
    return float((hi[0] - lo[0]) / sample_rate * factor)


def _decay_check(solved: np.ndarray, economy: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
    """T20 and T30 error per band per receiver, and why the two differ.

    **A metric cannot read past what was solved.** T20 spans -5 to -25 dB and
    T30 spans -5 to -35 dB, so a band solved to -30 dB has its T20 finish
    inside the solved part and its T30 finish 5 dB into the synthetic tail.
    Measured on the mid band as the level is deepened, T20 goes 8.5, 3.9, 0.9,
    0.4 per cent at -25, -30, -35 and -40 dB, while T30 goes 11.2, 10.5, 5.4,
    3.0. Both are reported so the reader can see which metric the window
    actually supports rather than reading one number.
    """
    centres = band_centres(sample_rate)
    rows: list[dict[str, Any]] = []
    for index, centre in enumerate(centres):
        entry: dict[str, Any] = {"centre_hz": int(centre), "receivers": []}
        for receiver in range(solved.shape[0]):
            reference = octave_filter(solved[receiver], sample_rate)[index]
            candidate = octave_filter(economy[receiver], sample_rate)[index]
            pairs = {}
            for name, end_db, factor in (("t20", -25.0, 3.0), ("t30", -35.0, 2.0)):
                a = _decay_time(reference, sample_rate, end_db, factor)
                b = _decay_time(candidate, sample_rate, end_db, factor)
                pairs[f"{name}_solved_s"] = None if not np.isfinite(a) else round(a, 4)
                pairs[f"{name}_economy_s"] = None if not np.isfinite(b) else round(b, 4)
                pairs[f"{name}_error_pct"] = (
                    None
                    if not (np.isfinite(a) and np.isfinite(b) and a > 0)
                    else round(100 * (b / a - 1), 2)
                )
            energy_a = float((reference**2).sum())
            energy_b = float((candidate**2).sum())
            pairs["energy_error_db"] = (
                None if energy_a <= 0 else round(10 * np.log10(energy_b / energy_a), 4)
            )
            entry["receivers"].append({"receiver": receiver, **pairs})
        rows.append(entry)
    return rows


def build(
    mid_dir: Path,
    high_dir: Path,
    out: Path,
    *,
    atmosphere: Atmosphere | None = None,
    crossovers_hz: tuple[float, float] = band_split.DEFAULT_CROSSOVERS_HZ,
    seed: int = 20250101,
    usd_per_hour_per_card: float = 0.917,
) -> dict[str, Any]:
    """Assemble both variants, render them, and write the run."""
    atmosphere = atmosphere or Atmosphere()
    out.mkdir(parents=True, exist_ok=True)

    mid_response = read_raw(mid_dir / "responses" / "source0.h5")
    high_response = read_raw(high_dir / "responses" / "source0.h5")
    if mid_response.provenance.scene_sha256 != high_response.provenance.scene_sha256:
        raise ValueError("the two solves were shot on different geometries")
    if mid_response.sample_rate_hz != high_response.sample_rate_hz:
        raise ValueError("the two solves are delivered at different sample rates")

    sample_rate = int(round(high_response.sample_rate_hz))
    sound_speed = high_response.provenance.sound_speed_m_s
    low_hz, high_hz = crossovers_hz

    mid_wet = np.atleast_2d(
        apply_air(mid_response.ir, float(sample_rate), atmosphere, sound_speed_m_s=sound_speed)
    )
    high_wet = np.atleast_2d(
        apply_air(high_response.ir, float(sample_rate), atmosphere, sound_speed_m_s=sound_speed)
    )

    report_source = json.loads((high_dir / "report.json").read_text())
    absorption = mean_absorption_of(report_source, len(band_centres(sample_rate)))
    split = band_split.split_filters(float(sample_rate), crossovers_hz)

    # Two solves are not on one scale. PFFDTD's source pulse has the run's own
    # fmax as its bandwidth, so its spectral density goes as 1/B and two runs
    # differ by 20 log10 of the ratio of their fmax in every shared band:
    # 12.04 dB between 4 and 16 kHz, predicted, 12.06 measured. Summing them
    # raw put the bottom two bands sixteen times too loud in energy.
    common = min(mid_wet.shape[1], high_wet.shape[1])
    gain_mid = band_split.level_ratio(
        mid_wet[:, :common], high_wet[:, :common], sample_rate, calibration_hz=CALIBRATION_HZ
    )
    predicted_gain = mid_response.provenance.fmax_hz / high_response.provenance.fmax_hz
    # The prediction is the cheapest test that two runs are comparable at all,
    # so it is checked rather than merely recorded. A pair that disagrees here
    # is a pair that must not be summed: the level defect this guards against
    # was 12 dB and showed only on a spectrogram.
    measured_db = 20.0 * float(np.log10(gain_mid))
    predicted_db = 20.0 * float(np.log10(predicted_gain))
    if abs(measured_db - predicted_db) > 3.0:
        raise ValueError(
            f"the two solves differ by {measured_db:+.2f} dB on the calibration bands "
            f"but their source bandwidths predict {predicted_db:+.2f} dB. They are not "
            "the same room under the same excitation, or one of them is not what its "
            "provenance says, and summing them would put a band at the wrong level."
        )
    gains = (gain_mid, gain_mid, 1.0)

    # Solved: every band taken from its run untouched, on one scale.
    solved = band_split.recombine(mid_wet, mid_wet, high_wet, split, gains=gains)

    # Economy: the low band is the same untouched run, the other two truncated.
    economy_mid, mid_window_s, mid_prediction = _economy_band(
        mid_wet,
        sample_rate,
        absorption,
        level_db=MID_LEVEL_DB,
        span_hz=(low_hz, high_hz),
        calibration=mid_wet,
        calibration_fmax_hz=mid_response.provenance.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed,
        seed=seed,
    )
    economy_high, high_window_s, high_prediction = _economy_band(
        high_wet,
        sample_rate,
        absorption,
        level_db=HIGH_LEVEL_DB,
        span_hz=(high_hz, float(band_centres(sample_rate)[-1])),
        calibration=mid_wet,
        calibration_fmax_hz=mid_response.provenance.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed,
        seed=seed + 1000,
    )
    economy = band_split.recombine(mid_wet, economy_mid, economy_high, split, gains=gains)

    # The single broadband solve, for the A/B the three-band page otherwise
    # cannot offer: switching runs reloads the page and loses the comparison.
    single = np.zeros_like(solved)
    single[:, : high_wet.shape[1]] = high_wet

    responses = [single, solved, economy]

    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dry_path = high_dir / "audio" / "dry_voice.wav"
    if not dry_path.is_file():
        raise ValueError(f"{dry_path} is missing; render the high band run's audio first")
    shutil.copy2(dry_path, audio_dir / "dry_voice.wav")

    import soundfile

    dry, dry_rate = soundfile.read(str(dry_path))
    dry = np.asarray(dry, dtype=float)
    if dry.ndim > 1:
        dry = dry[:, 0]

    wet_audio = [
        [audio.convolve(dry, response[receiver]) for receiver in range(response.shape[0])]
        for response in responses
    ]
    gain = shared_gain([block for group in wet_audio for block in group])

    sources: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for index, ((name, what), response) in enumerate(zip(VARIANTS, responses, strict=True)):
        write_raw(
            ResponseSet(
                ir=response,
                sample_rate_hz=float(sample_rate),
                source_position=high_response.source_position,
                receiver_positions=high_response.receiver_positions,
                provenance=Provenance(
                    **{
                        **vars(high_response.provenance),
                        "run_id": f"{out.name}_{name}",
                        "band": "three-band recombination",
                        "notes": f"{name}: {what}. {high_response.provenance.notes}",
                    }
                ),
                room_volume_m3=high_response.room_volume_m3,
            ),
            out / "responses" / f"source{index}.h5",
        )
        for receiver in range(response.shape[0]):
            path = audio_dir / f"source{index}_receiver{receiver}_wet.wav"
            block = np.atleast_2d(wet_audio[index][receiver] * gain)
            soundfile.write(str(path), block.T, int(dry_rate), subtype="FLOAT", format="WAV")
            written.append(
                {
                    "path": path.name,
                    "variant": name,
                    "receiver": receiver,
                    "write_gain": gain,
                    "peak_after_gain": round(float(np.max(np.abs(block))), 4),
                }
            )
        sources.append(
            {
                "source_index": index,
                "archetype": name,
                "label": f"{name}: {what}",
                "variant": name,
                "what": what,
                "measures": measure_all(
                    response, float(sample_rate), fmax_hz=high_response.provenance.fmax_hz
                ),
            }
        )

    plan = json.loads((high_dir / "plan.json").read_text())
    placement = dict(plan["placement"])
    placement["sources"] = [{**placement["sources"][0], "archetype": name} for name, _ in VARIANTS]
    (out / "plan.json").write_text(
        json.dumps({**plan, "placement": placement, "run": out.name}, indent=2) + "\n"
    )

    flat_span = (23.575, 2.885, 18.439)
    room_span = (4.354, 2.824, 3.145)
    priced = {
        "low": cost_model.estimate(
            cost_model.grid_points_for(flat_span, low_hz),
            mid_response.duration_s,
            cost_model.solver_rate_hz(low_hz),
            usd_per_hour_per_card=usd_per_hour_per_card,
        ),
        "mid": cost_model.estimate(
            cost_model.grid_points_for(flat_span, high_hz),
            mid_window_s,
            cost_model.solver_rate_hz(high_hz),
            usd_per_hour_per_card=usd_per_hour_per_card,
        ),
        "high": cost_model.estimate(
            cost_model.grid_points_for(room_span, 16000.0),
            high_window_s,
            cost_model.solver_rate_hz(16000.0),
            usd_per_hour_per_card=usd_per_hour_per_card,
        ),
    }

    report = {
        **report_source,
        "run": out.name,
        "reference_run": high_dir.name,
        "mid_band_run": mid_dir.name,
        "trick": TRICK,
        "atmosphere": atmosphere.record(),
        "split": split.record(),
        "level_calibration": {
            "gain_applied_to_mid_run": gain_mid,
            "gain_db": float(20.0 * np.log10(gain_mid)),
            "predicted_gain": predicted_gain,
            "predicted_gain_db": float(20.0 * np.log10(predicted_gain)),
            "calibration_bands_hz": list(CALIBRATION_HZ),
            "why": (
                "PFFDTD excites the grid with a band-limited impulse whose bandwidth is "
                "the run's own fmax, so its spectral density goes as 1/B and two solves "
                "of one room differ by 20 log10 of the ratio of their fmax in every band "
                "they share. Without this the bottom two bands are sixteen times too "
                "loud in energy, which a spectrogram shows at a glance."
            ),
        },
        "windows": {
            "low_s": mid_response.duration_s,
            "low_rule": "the whole decay, never truncated",
            "mid_s": mid_window_s,
            "mid_rule": f"measured, {MID_LEVEL_DB:g} dB on the slowest band it carries",
            "high_s": high_window_s,
            "high_rule": f"measured, {HIGH_LEVEL_DB:g} dB on the slowest band it carries",
        },
        "cost": {
            **{f"{band}_usd": value.usd for band, value in priced.items()},
            "total_usd": sum(value.usd for value in priced.values()),
            "usd_per_hour_per_card": usd_per_hour_per_card,
            "note": "per band, on this flat, with the measured windows above",
        },
        "transposition": {"mid": mid_prediction, "high": high_prediction},
        "decay_check": _decay_check(solved, economy, sample_rate),
        "single_band_check": _decay_check(single, solved, sample_rate),
        "audio": written,
        "write_gain": gain,
        "placement": placement,
        "sources": sources,
        "band_note": (
            f"Three renderings of one room. Crossovers at "
            f"{low_hz:.0f} and {high_hz:.0f} Hz built by subtraction so they sum to a "
            f"delayed impulse exactly. single band is the 16 kHz solve alone, for "
            f"comparison. solved takes every band from its run untouched, after the "
            f"4 kHz run is scaled by {20 * np.log10(gain_mid):+.2f} dB to put the two "
            f"solves on one scale. "
            f"economy keeps the low band whole, stops the mid band at {MID_LEVEL_DB:g} dB "
            f"({1000 * mid_window_s:.0f} ms) and the high band at {HIGH_LEVEL_DB:g} dB "
            f"({1000 * high_window_s:.0f} ms), and synthesises the rest. Both carry "
            f"{split.record()['group_delay_ms']:.1f} ms of filter delay, identically. "
            "The gain is shared, so a difference in loudness is a real difference."
        ),
        "omissions": [
            *report_source.get("omissions", []),
            "only two solves of this room exist, so the 4 kHz run serves twice: "
            "untouched as the low band and truncated as the mid band. In the real "
            "architecture the low band is its own coarse whole-apartment solve",
            "each band comes from a different grid, and W3 measured that two such "
            "runs decorrelate; inside a crossover the sum blends them, which is a "
            "property of a three-solve architecture and not of this assembly",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble and render the three-band response.")
    parser.add_argument("--mid", type=Path, required=True, help="the 4 kHz solve")
    parser.add_argument("--high", type=Path, required=True, help="the 16 kHz solve")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-humidity-pct", type=float, default=50.0)
    args = parser.parse_args(argv)

    report = build(
        args.mid,
        args.high,
        args.out,
        atmosphere=Atmosphere(relative_humidity_pct=args.relative_humidity_pct),
    )

    windows, cost = report["windows"], report["cost"]
    print(f"{report['run']}: {len(report['audio'])} files, one gain {report['write_gain']:.3f}\n")
    print(f"{'band':>6} {'window':>10} {'rule':<52} {'USD':>7}")
    for band in ("low", "mid", "high"):
        print(
            f"{band:>6} {1000 * windows[f'{band}_s']:>7.0f} ms "
            f"{windows[f'{band}_rule']:<52} {cost[f'{band}_usd']:7.3f}"
        )
    print(f"{'total':>6} {'':>10} {'':<52} {cost['total_usd']:7.3f}")
    print(f"\nat {cost['usd_per_hour_per_card']:.3f} USD/h per card, billed")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
