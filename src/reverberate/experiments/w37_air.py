"""W37: apply atmospheric absorption to responses that are already computed.

The term this repairs is the dominant one above 4 kHz and it is absent from
every response the project holds. Roadmap W30 measured what that costs on
``w29_16k``: T60 falls 4.5 per cent at 4 kHz, 14.3 at 8 kHz and 37.7 at 16 kHz,
so the omission is five times the size of the mid-to-high effect W29 was built
to measure.

This experiment costs no rental. It reads stored responses, applies
:mod:`reverberate.audio`, and reports the decay per octave band **and per
receiver**, which is the reporting rule W30 exists to enforce: W29's own lesson
is that one scalar hid a 29 per cent omission behind a 5 per cent agreement.

The comparison is legitimate on a pair of runs only if they share a geometry.
``w27_sealed`` and ``w29_16k`` both carry ``scene_sha256 = f880c7c8...``, one
at 4 kHz and one at 16 kHz, same source and same six receivers, so the pair is
internally consistent. That hash is superseded by the adaptive carve, and the
report says so rather than leaving a reader to discover it.

Usage::

    python -m reverberate.experiments.w37_air \\
        --run data/runs/w29_16k --run data/runs/w27_sealed \\
        --out data/runs/w37_air

Bands above a run's own ``fmax`` are reported with ``in_band`` false. Their
contents are the low-pass skirt, not the room, and quoting a decay from them is
how a 4 kHz run comes to claim a 16 kHz reverberation time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.audio import Atmosphere
from reverberate.audio import apply_air_absorption as apply
from reverberate.metrics import band_centres, edt_per_band, octave_filter, rt60_per_band
from reverberate.response import ResponseSet, read_raw

__all__ = [
    "BandDecay",
    "ReceiverDecay",
    "analytic_t60_s",
    "build",
    "main",
    "measure_run",
]

#: One paragraph in plain words, as the ledger rule requires: what was changed
#: and why it should be free.
TRICK = (
    "Atmospheric absorption applied after the solve. In an impulse response "
    "every sample arriving at time t has travelled exactly c t of path, so air "
    "absorption is exactly the per-sample gain exp(-m(f) c t) and not an "
    "approximation. It is free because it is arithmetic on a stored response, "
    "and it is exact because the solver's output is an impulse response rather "
    "than a field. See docs/adr/0009-air-absorption-as-post-processing.md."
)

#: The geometry this pair of runs was shot on, and what replaced it.
GEOMETRY_NOTE = (
    "scene_sha256 f880c7c8 predates the adaptive carve. The rules measured "
    "here are propagation rules and do not depend on the geometry, and the two "
    "runs compared share it, so the pair is internally consistent. The "
    "validation is to be replayed once a mid and a high band run exist on the "
    "current geometry."
)


@dataclass(frozen=True)
class BandDecay:
    """Decay in one octave band, before and after air is applied.

    Carries both the measured result and the analytic one the roadmap's own
    table is built from, because the two disagree and the disagreement has a
    mechanism. See :func:`analytic_t60_s`.
    """

    centre_hz: int
    in_band: bool
    t30_dry_s: float
    t30_air_s: float
    t30_analytic_s: float
    t60_air_alone_s: float
    edt_dry_s: float
    edt_air_s: float
    energy_lost_db: float

    @property
    def t30_change_pct(self) -> float:
        """Signed change in T30, per cent. NaN when the dry value is unusable."""
        if not np.isfinite(self.t30_dry_s) or self.t30_dry_s <= 0.0:
            return float("nan")
        return 100.0 * (self.t30_air_s / self.t30_dry_s - 1.0)

    @property
    def analytic_excess_pct(self) -> float:
        """How much longer the analytic prediction reads than the measurement."""
        if not np.isfinite(self.t30_air_s) or self.t30_air_s <= 0.0:
            return float("nan")
        return 100.0 * (self.t30_analytic_s / self.t30_air_s - 1.0)

    def record(self) -> dict[str, Any]:
        return {
            "centre_hz": self.centre_hz,
            "in_band": self.in_band,
            "t30_dry_s": _finite(self.t30_dry_s),
            "t30_air_s": _finite(self.t30_air_s),
            "t30_change_pct": _finite(self.t30_change_pct),
            "t30_analytic_s": _finite(self.t30_analytic_s),
            "analytic_excess_pct": _finite(self.analytic_excess_pct),
            "t60_air_alone_s": _finite(self.t60_air_alone_s),
            "edt_dry_s": _finite(self.edt_dry_s),
            "edt_air_s": _finite(self.edt_air_s),
            "energy_lost_db": _finite(self.energy_lost_db),
        }


def analytic_t60_s(
    t60_dry_s: float,
    centre_hz: float,
    atmosphere: Atmosphere,
    *,
    sound_speed_m_s: float,
) -> tuple[float, float]:
    """The band-centre prediction: ``(T60 from air alone, the two combined)``.

    Two decay mechanisms in parallel add their rates, so
    ``1/T = 1/T_surface + 1/T_air``. This is what roadmap W30's air-corrected
    column holds, and it reproduces that column to three decimals in all five
    bands it quotes.

    **It is a prediction, not the measurement, and it reads long.** The
    coefficient goes as the square of frequency, so within one octave the
    attenuation at the top is four times that at the bottom, and the band's
    upper half decays away faster than its centre. A T30 measured on the
    octave-filtered response is therefore shorter than the band-centre formula
    says. Measured on ``w29_16k``, the analytic value is 3.5 per cent long at
    4 kHz and 6.7 at 8 kHz. At 16 kHz the two agree, because the solver's own
    low-pass at ``fmax`` removes the top of that octave and leaves its energy
    below the nominal centre.

    Both are reported. The analytic one is the cheap estimate a plan can use
    before a response exists; the measured one is what the response says.
    """
    air_alone = 60.0 / (
        float(np.asarray(atmosphere.attenuation_db_per_m(centre_hz))) * sound_speed_m_s
    )
    if not np.isfinite(t60_dry_s) or t60_dry_s <= 0.0:
        return air_alone, float("nan")
    return air_alone, 1.0 / (1.0 / t60_dry_s + 1.0 / air_alone)


@dataclass(frozen=True)
class ReceiverDecay:
    """Every band of one receiver."""

    index: int
    position: tuple[float, float, float]
    distance_m: float
    bands: tuple[BandDecay, ...]

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "position": list(self.position),
            "distance_m": self.distance_m,
            "bands": [band.record() for band in self.bands],
        }


def _finite(value: float) -> float | None:
    """JSON has no NaN. An unmeasurable decay is null, never a plausible number."""
    return float(value) if np.isfinite(value) else None


def _band_decay(
    *,
    centre_hz: int,
    in_band: bool,
    t30_dry_s: float,
    t30_air_s: float,
    edt_dry_s: float,
    edt_air_s: float,
    energy_lost_db: float,
    atmosphere: Atmosphere,
    sound_speed_m_s: float,
) -> BandDecay:
    air_alone, combined = analytic_t60_s(
        t30_dry_s, float(centre_hz), atmosphere, sound_speed_m_s=sound_speed_m_s
    )
    return BandDecay(
        centre_hz=centre_hz,
        in_band=in_band,
        t30_dry_s=t30_dry_s,
        t30_air_s=t30_air_s,
        t30_analytic_s=combined,
        t60_air_alone_s=air_alone,
        edt_dry_s=edt_dry_s,
        edt_air_s=edt_air_s,
        energy_lost_db=energy_lost_db,
    )


def measure_run(
    response: ResponseSet,
    atmosphere: Atmosphere,
    *,
    fmax_hz: float,
    sound_speed_m_s: float,
) -> list[ReceiverDecay]:
    """Decay per band per receiver, dry and with air, for one stored run."""
    sample_rate = int(round(response.sample_rate_hz))
    centres = band_centres(sample_rate)
    absorbed = apply(
        response.ir,
        response.sample_rate_hz,
        atmosphere=atmosphere,
        sound_speed_m_s=sound_speed_m_s,
    )

    receivers: list[ReceiverDecay] = []
    for index in range(response.receiver_count):
        dry, wet = response.ir[index], absorbed[index]
        t30_dry, t30_air = rt60_per_band(dry, sample_rate), rt60_per_band(wet, sample_rate)
        edt_dry, edt_air = edt_per_band(dry, sample_rate), edt_per_band(wet, sample_rate)
        energy_dry = (octave_filter(dry, sample_rate) ** 2).sum(axis=1)
        energy_air = (octave_filter(wet, sample_rate) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lost = 10.0 * np.log10(np.where(energy_dry > 0.0, energy_air / energy_dry, np.nan))

        position = response.receiver_positions[index]
        receivers.append(
            ReceiverDecay(
                index=index,
                position=(float(position[0]), float(position[1]), float(position[2])),
                distance_m=float(np.linalg.norm(position - response.source_position)),
                bands=tuple(
                    _band_decay(
                        centre_hz=int(centre),
                        in_band=float(centre) <= fmax_hz,
                        t30_dry_s=float(t30_dry[band]),
                        t30_air_s=float(t30_air[band]),
                        edt_dry_s=float(edt_dry[band]),
                        edt_air_s=float(edt_air[band]),
                        energy_lost_db=float(lost[band]),
                        atmosphere=atmosphere,
                        sound_speed_m_s=sound_speed_m_s,
                    )
                    for band, centre in enumerate(centres)
                ),
            )
        )
    return receivers


def _summarise(receivers: list[ReceiverDecay]) -> list[dict[str, Any]]:
    """Mean and receiver spread per band, so a scalar never stands alone."""
    if not receivers:
        return []
    summary: list[dict[str, Any]] = []
    for band in range(len(receivers[0].bands)):
        column = [receiver.bands[band] for receiver in receivers]
        dry = np.array([value.t30_dry_s for value in column], dtype=float)
        air = np.array([value.t30_air_s for value in column], dtype=float)
        change = np.array([value.t30_change_pct for value in column], dtype=float)
        summary.append(
            {
                "centre_hz": column[0].centre_hz,
                "in_band": column[0].in_band,
                "t30_dry_mean_s": _finite(float(np.nanmean(dry))),
                "t30_air_mean_s": _finite(float(np.nanmean(air))),
                "t30_change_mean_pct": _finite(float(np.nanmean(change))),
                # The spread the change has to beat to mean anything. W29's
                # own receiver-to-receiver spread was 6 per cent against a
                # 5.3 per cent effect, and that is why it settled nothing.
                "t30_dry_spread_pct": _finite(
                    float(100.0 * np.nanstd(dry) / np.nanmean(dry)) if np.nanmean(dry) else np.nan
                ),
                "t30_analytic_mean_s": _finite(
                    float(np.nanmean([value.t30_analytic_s for value in column]))
                ),
                "analytic_excess_mean_pct": _finite(
                    float(np.nanmean([value.analytic_excess_pct for value in column]))
                ),
                "energy_lost_mean_db": _finite(
                    float(np.nanmean([value.energy_lost_db for value in column]))
                ),
            }
        )
    return summary


def build(
    runs: list[Path],
    out: Path,
    *,
    atmosphere: Atmosphere | None = None,
    source: str = "source0",
) -> dict[str, Any]:
    """Measure every run, write ``report.json``, return it."""
    atmosphere = atmosphere or Atmosphere()
    out.mkdir(parents=True, exist_ok=True)

    measured: list[dict[str, Any]] = []
    for run_dir in runs:
        response = read_raw(run_dir / "responses" / f"{source}.h5")
        provenance = response.provenance
        receivers = measure_run(
            response,
            atmosphere,
            fmax_hz=provenance.fmax_hz,
            sound_speed_m_s=provenance.sound_speed_m_s,
        )
        measured.append(
            {
                "run": run_dir.name,
                "scene_sha256": provenance.scene_sha256,
                "fmax_hz": provenance.fmax_hz,
                "grid_step_m": provenance.grid_step_m,
                "sample_rate_hz": response.sample_rate_hz,
                "duration_s": response.duration_s,
                "sound_speed_m_s": provenance.sound_speed_m_s,
                "source_position": list(map(float, response.source_position)),
                "receivers": [receiver.record() for receiver in receivers],
                "per_band": _summarise(receivers),
            }
        )

    hashes = {entry["scene_sha256"] for entry in measured}
    report: dict[str, Any] = {
        "run": out.name,
        # This is the reference for itself: nothing is compared against another
        # run here, the same responses are measured twice.
        "reference_run": None,
        "trick": TRICK,
        "atmosphere": atmosphere.record(),
        "scene_sha256": sorted(hashes),
        "scene_sha256_shared": len(hashes) == 1,
        "geometry_note": GEOMETRY_NOTE,
        "runs": measured,
        "omissions": [
            "no head, torso or pinna: these are bare pressure points",
            "the humidity is declared, not measured: no room this project "
            "simulates has a recorded atmosphere",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply air absorption to stored responses.")
    parser.add_argument("--run", type=Path, action="append", required=True, dest="runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default="source0")
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument("--humidity-percent", type=float, default=50.0)
    args = parser.parse_args(argv)

    atmosphere = Atmosphere(
        temperature_c=args.temperature_c,
        humidity_percent=args.humidity_percent,
    )
    report = build(args.runs, args.out, atmosphere=atmosphere, source=args.source)

    print(
        f"air at {atmosphere.temperature_c:g} C, "
        f"{atmosphere.humidity_percent:g} per cent relative humidity"
    )
    for entry in report["runs"]:
        print(f"\n{entry['run']}, fmax {entry['fmax_hz']:g} Hz, {entry['duration_s']:.2f} s")
        print(
            f"{'band':>7} {'T30 dry':>9} {'T30 air':>9} {'change':>8} "
            f"{'rx spread':>10} {'analytic':>9} {'excess':>8}"
        )
        for band in entry["per_band"]:
            if not band["in_band"] or band["t30_dry_mean_s"] is None:
                continue
            print(
                f"{band['centre_hz']:>7} {band['t30_dry_mean_s']:9.3f} "
                f"{band['t30_air_mean_s']:9.3f} {band['t30_change_mean_pct']:7.1f}% "
                f"{band['t30_dry_spread_pct']:9.1f}% {band['t30_analytic_mean_s']:9.3f} "
                f"{band['analytic_excess_mean_pct']:7.1f}%"
            )
    if not report["scene_sha256_shared"]:
        print("\nWARNING: these runs do not share a geometry; do not compare them.")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
