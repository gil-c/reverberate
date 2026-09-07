"""W37: how short the computed window can be, measured on responses already bought.

The high band's cost is linear in the window, and the window is a design choice
rather than a constraint: roadmap section 5.2 removed the artificial boundary
that used to bound it, so the question became "how much early response is
needed before the synthetic tail takes over".

**This costs no rental, and that is not a compromise.** An explicit
time-marching scheme cannot revise a sample it has already written, so
truncating a stored response is *exactly* what stopping the solve at that
window would have produced. ``w29_16k`` holds 1.0 s of real 16 kHz response and
``w27_sealed`` holds the same room, same source and same six receivers at
4 kHz. Splicing a mid-band-calibrated tail onto a truncated copy and comparing
against the untruncated original answers the whole duration lever offline.

**Air is applied to both legs or to neither.** The reference has no viscosity
term, so comparing a corrected prediction against an uncorrected truth reads as
a 39 per cent error that is entirely the missing air. Both legs are corrected
here, and ``atmosphere`` is recorded.

**Ten seeds, not one.** The tail is noise. At a 100 ms window one seed reads as
a worse result than at 60 and 150 ms, and the seed-to-seed standard deviation
says that is noise rather than a trend. W3 measured the response noise floor at
3.1 per cent, and an error below it is not a measurement.

Usage::

    python -m reverberate.experiments.w37_window \\
        --reference data/runs/w29_16k --mid data/runs/w27_sealed \\
        --out data/runs/w37_window --usd-per-hour 0.917
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import cost as cost_model
from reverberate.air import Atmosphere
from reverberate.air import apply as apply_air
from reverberate.metrics import band_centres, octave_filter, rt60_per_band
from reverberate.response import ResponseSet, read_raw
from reverberate.tail import Transposition, splice, transpose

__all__ = ["WindowError", "DEFAULT_WINDOWS_MS", "build", "main", "mean_absorption_of"]

#: Windows swept, in milliseconds. The short end is below the bedroom's 6.2 ms
#: mixing time, where a noise tail cannot be right because the field is not yet
#: diffuse; the long end is where the synthetic tail carries almost nothing.
DEFAULT_WINDOWS_MS = (15.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0)

#: Seeds. The tail is noise, so a single draw is a sample and not a result.
DEFAULT_SEEDS = 10

TRICK = (
    "The high band is computed only to a short window and the rest of the tail "
    "is synthesised, calibrated on the mid band run of the same room. Solver "
    "cost is linear in the window, so cutting 1.2 s to 60 ms is a factor of 20. "
    "It should be free because what transposes from mid to high is the averaged "
    "decay slope, which is what a noise tail needs, while what does not "
    "transpose is the waveform of an individual reflection, which lives in the "
    "early part the solver still computes."
)

VALIDITY = (
    "Truncating a stored response is exactly what stopping the solve early "
    "would have produced, because an explicit time-marching scheme cannot "
    "revise a sample it has already written. No rental was spent here and none "
    "was needed."
)


def mean_absorption_of(report: dict[str, Any], bands: int) -> np.ndarray:
    """Area-weighted absorption per band, from a run's own report.

    Extends the catalogue's top band to the analysis bank's top by the rule of
    roadmap 6.2: each class's own 2 to 4 kHz ratio applied per octave, clipped
    to at most 1 and at least 0.8. The catalogue stops at 8 kHz because that is
    where its measurements stop, and the extension is a judgement labelled as
    one rather than a measurement.
    """
    classes = report["room"]["per_class"]
    area = np.array([entry["area_m2"] for entry in classes], dtype=float)
    alpha = np.array([entry["random_incidence_absorption"] for entry in classes], dtype=float)
    if area.sum() <= 0.0:
        raise ValueError("the report has no surface area to weight absorption by")

    while alpha.shape[1] < bands:
        ratio = np.clip(
            np.where(alpha[:, -3] > 0.0, alpha[:, -2] / np.maximum(alpha[:, -3], 1e-9), 1.0),
            0.8,
            1.0,
        )
        alpha = np.column_stack([alpha, np.clip(alpha[:, -1] * ratio, 1e-4, 0.99)])
    return np.asarray((area[:, None] * alpha[:, :bands]).sum(axis=0) / area.sum())


@dataclass(frozen=True)
class WindowError:
    """What one window costs and what it costs in accuracy."""

    window_ms: float
    band_centres_hz: tuple[int, ...]
    in_band: tuple[bool, ...]
    #: Mean absolute T30 error against the untruncated reference, per band, per
    #: cent, averaged over seeds and receivers.
    t30_error_pct: tuple[float, ...]
    #: Standard deviation of the same, which says whether it is a measurement.
    t30_error_sd_pct: tuple[float, ...]
    #: Band energy error against the reference, in decibels.
    energy_error_db: tuple[float, ...]
    #: Energy arriving after this window in the reference, relative to the
    #: whole band. This is what the synthetic tail has to supply.
    late_energy_db: tuple[float, ...]
    gpu_hours: float
    usd: float
    usd_per_hour_per_card: float
    cards: int

    def record(self) -> dict[str, Any]:
        return {
            "window_ms": self.window_ms,
            "bands": [
                {
                    "centre_hz": centre,
                    "in_band": in_band,
                    "t30_error_pct": error,
                    "t30_error_sd_pct": spread,
                    "energy_error_db": energy,
                    "late_energy_db": late,
                }
                for centre, in_band, error, spread, energy, late in zip(
                    self.band_centres_hz,
                    self.in_band,
                    self.t30_error_pct,
                    self.t30_error_sd_pct,
                    self.energy_error_db,
                    self.late_energy_db,
                    strict=True,
                )
            ],
            "gpu_hours": self.gpu_hours,
            "usd": self.usd,
            "usd_per_hour_per_card": self.usd_per_hour_per_card,
            "cards": self.cards,
        }


def _corrected(response: ResponseSet, atmosphere: Atmosphere) -> np.ndarray:
    return np.atleast_2d(
        apply_air(
            response.ir,
            response.sample_rate_hz,
            atmosphere,
            sound_speed_m_s=response.provenance.sound_speed_m_s,
        )
    )


def _transpositions(
    mid: np.ndarray,
    sample_rate: int,
    absorption: np.ndarray,
    *,
    mid_fmax_hz: float,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
) -> list[Transposition]:
    return [
        transpose(
            rt60_per_band(mid[receiver], sample_rate),
            absorption,
            sample_rate,
            mid_fmax_hz=mid_fmax_hz,
            atmosphere=atmosphere,
            sound_speed_m_s=sound_speed_m_s,
        )
        for receiver in range(mid.shape[0])
    ]


def build(
    reference_dir: Path,
    mid_dir: Path,
    out: Path,
    *,
    usd_per_hour_per_card: float,
    atmosphere: Atmosphere | None = None,
    windows_ms: tuple[float, ...] = DEFAULT_WINDOWS_MS,
    seeds: int = DEFAULT_SEEDS,
    source: str = "source0",
    grid_points: int | None = None,
    solver_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Sweep the window, splice a tail at each, and price the result."""
    atmosphere = atmosphere or Atmosphere()
    out.mkdir(parents=True, exist_ok=True)

    reference = read_raw(reference_dir / "responses" / f"{source}.h5")
    mid = read_raw(mid_dir / "responses" / f"{source}.h5")
    if reference.provenance.scene_sha256 != mid.provenance.scene_sha256:
        raise ValueError(
            "the reference and the mid band run were shot on different geometries, "
            "so one cannot calibrate the other"
        )
    if reference.sample_rate_hz != mid.sample_rate_hz:
        raise ValueError("the two runs are delivered at different sample rates")

    sample_rate = int(round(reference.sample_rate_hz))
    centres = band_centres(sample_rate)
    truth = _corrected(reference, atmosphere)
    mid_corrected = _corrected(mid, atmosphere)

    report_path = reference_dir / "report.json"
    absorption = mean_absorption_of(json.loads(report_path.read_text()), len(centres))
    predictions = _transpositions(
        mid_corrected,
        sample_rate,
        absorption,
        mid_fmax_hz=mid.provenance.fmax_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=reference.provenance.sound_speed_m_s,
    )

    truth_decay = np.array([rt60_per_band(row, sample_rate) for row in truth])
    truth_energy = np.array([(octave_filter(row, sample_rate) ** 2).sum(axis=1) for row in truth])

    # The grid this window is priced on. Defaults to the reference's own, so
    # the price is of the run that produced the error beside it.
    points = grid_points or int(json.loads(report_path.read_text())["cost"]["grid_points"])
    rate = solver_rate_hz or _solver_rate(reference_dir)

    results: list[WindowError] = []
    for window_ms in windows_ms:
        errors, energies = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(20250101 + seed)
            for receiver in range(truth.shape[0]):
                spliced = splice(
                    truth[receiver],
                    sample_rate,
                    window_ms / 1000.0,
                    np.array(predictions[receiver].t60_s),
                    rng=rng,
                )
                errors.append(
                    100.0 * (rt60_per_band(spliced, sample_rate) / truth_decay[receiver] - 1.0)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    energies.append(
                        10.0
                        * np.log10(
                            (octave_filter(spliced, sample_rate) ** 2).sum(axis=1)
                            / truth_energy[receiver]
                        )
                    )
        error = np.abs(np.array(errors))
        energy = np.array(energies)

        cut = int(round(window_ms / 1000.0 * sample_rate))
        with np.errstate(divide="ignore", invalid="ignore"):
            late = np.array(
                [
                    10.0
                    * np.log10(
                        (octave_filter(row, sample_rate)[:, cut:] ** 2).sum(axis=1)
                        / (octave_filter(row, sample_rate) ** 2).sum(axis=1)
                    )
                    for row in truth
                ]
            )

        priced = cost_model.estimate(
            points,
            window_ms / 1000.0,
            rate,
            usd_per_hour_per_card=usd_per_hour_per_card,
        )
        results.append(
            WindowError(
                window_ms=window_ms,
                band_centres_hz=tuple(centres),
                in_band=tuple(float(c) <= reference.provenance.fmax_hz for c in centres),
                t30_error_pct=tuple(map(float, np.nanmean(error, axis=0))),
                t30_error_sd_pct=tuple(map(float, np.nanstd(error, axis=0))),
                energy_error_db=tuple(map(float, np.nanmean(energy, axis=0))),
                late_energy_db=tuple(map(float, np.nanmean(late, axis=0))),
                gpu_hours=priced.gpu_hours,
                usd=priced.usd,
                usd_per_hour_per_card=usd_per_hour_per_card,
                cards=priced.cards,
            )
        )

    full = cost_model.estimate(
        points,
        reference.duration_s,
        rate,
        usd_per_hour_per_card=usd_per_hour_per_card,
    )
    report: dict[str, Any] = {
        "run": out.name,
        "reference_run": reference_dir.name,
        "mid_band_run": mid_dir.name,
        "trick": TRICK,
        "validity": VALIDITY,
        "atmosphere": atmosphere.record(),
        "scene_sha256": reference.provenance.scene_sha256,
        "cache_key": json.loads(report_path.read_text()).get("cache_key"),
        "seeds": seeds,
        "receivers": int(truth.shape[0]),
        "grid_points": points,
        "solver_rate_hz": rate,
        "reference_duration_s": reference.duration_s,
        "reference_cost": full.record(),
        "transposition": [prediction.record() for prediction in predictions],
        "windows": [result.record() for result in results],
        "omissions": [
            "one room, one source: the window rule is measured where it can be "
            "measured for free and is not yet checked on a room with strong "
            "frequency contrast, which is what W30 chose the kitchen for",
            "the absorption above 8 kHz is the catalogue's extrapolation, not a measurement",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _solver_rate(run_dir: Path) -> float:
    """The grid's own step rate, from the run's provenance rather than derived."""
    record = json.loads((run_dir / "report.json").read_text())
    steps = record.get("cost", {}).get("steps")
    response = read_raw(run_dir / "responses" / "source0.h5")
    if steps:
        return float(steps) / response.duration_s
    provenance = response.provenance
    return float(provenance.sound_speed_m_s * np.sqrt(3.0) / provenance.grid_step_m)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep the computed window.")
    parser.add_argument("--reference", type=Path, required=True, help="the high band run")
    parser.add_argument("--mid", type=Path, required=True, help="the mid band run, same room")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--usd-per-hour",
        type=float,
        required=True,
        help="billed rate per card; required, because a cost without a rate is not a measurement",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--relative-humidity-pct", type=float, default=50.0)
    parser.add_argument(
        "--grid-points",
        type=int,
        default=None,
        help="price the window on another room's grid instead of the reference's",
    )
    args = parser.parse_args(argv)

    report = build(
        args.reference,
        args.mid,
        args.out,
        usd_per_hour_per_card=args.usd_per_hour,
        atmosphere=Atmosphere(relative_humidity_pct=args.relative_humidity_pct),
        seeds=args.seeds,
        grid_points=args.grid_points,
    )

    full = report["reference_cost"]
    print(
        f"reference {report['reference_run']}: {report['reference_duration_s']:.2f} s, "
        f"{full['gpu_hours']:.2f} card-h, {full['usd']:.2f} USD at "
        f"{full['usd_per_hour_per_card']:.3f} USD/h per card"
    )
    print(f"tail calibrated on {report['mid_band_run']}, {report['seeds']} seeds\n")
    top = [4000, 8000, 16000]
    header = "  ".join(f"{band:>5} Hz" for band in top)
    print(f"{'window':>8} {'cost':>8} {'saving':>8}   T30 error, mean absolute with sd")
    print(f"{'':>8} {'':>8} {'':>8}   {header}")
    for window in report["windows"]:
        bands = {band["centre_hz"]: band for band in window["bands"]}
        cells = "  ".join(
            f"{bands[b]['t30_error_pct']:4.1f}+-{bands[b]['t30_error_sd_pct']:.1f}%" for b in top
        )
        print(
            f"{window['window_ms']:>6.0f}ms {window['usd']:7.2f}$ "
            f"{full['usd'] / max(window['usd'], 1e-9):7.1f}x   {cells}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
