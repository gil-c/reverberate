"""W37 and W6: what the three-band split actually costs, and where the money is.

The roadmap's open question is blunt: "the mid band cost has never been
measured, and it is the dominant figure". This prices all three bands from
measured constants and sweeps the two crossovers, which is arithmetic once
:mod:`reverberate.cost` can turn a bounding box and a top frequency into grid
points.

**The architecture being priced**, from section 4.2. Three solves, each with its
own grid, its own domain and its own window:

===== ================= =============== ==================
band  domain            window          top frequency
===== ================= =============== ==================
low   whole apartment   full response   the low crossover
mid   whole apartment   long            the high crossover
high  one room          early part only 16 kHz, fixed
===== ================= =============== ==================

**The result is asymmetric, and it is worth stating plainly.** Each band's cost
depends on its *own* top frequency to the fourth power, so the low crossover
prices only the low band and the high crossover prices only the mid band. The
high band's cost does not move with either, because its top frequency is fixed
at 16 kHz by the deliverable.

So there is only one crossover with money behind it, and it is the upper one.
Moving it from 4 kHz to 2.5 kHz divides the mid band by ``(4 / 2.5)^4``, which
is 6.6. On this apartment that is most of the per-scene bill.

**And the saving is paid for entirely in validity, not in accuracy elsewhere.**
Lowering the upper crossover hands the 2.5 to 4 kHz octave from a whole-apartment
domain to a one-room domain. Nothing about the high band run changes. What
changes is whether one room is still a legitimate domain that low, which is a
question about doorways and about how much the rest of the flat contributes,
and it is **W23 and unmeasured**. This experiment prices the trade; it does not
settle it, and it must not be read as recommending a crossover.

Usage::

    python -m reverberate.experiments.w37_bands \\
        --scene-span 23.575 2.885 18.439 --room-span 4.354 2.824 3.145 \\
        --out data/runs/w37_bands --usd-per-hour 0.917
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate import cost as cost_model

__all__ = [
    "DEFAULT_HIGH_CROSSOVERS_HZ",
    "DEFAULT_LOW_CROSSOVERS_HZ",
    "BandPlan",
    "build",
    "main",
    "price",
]

#: Upper crossovers swept. 4 kHz is the roadmap's assumption and 2500 Hz its
#: stated alternative; the rest bracket them.
DEFAULT_HIGH_CROSSOVERS_HZ = (2000.0, 2500.0, 3000.0, 4000.0, 5000.0)

#: Lower crossovers swept. 1 kHz is the assumption, 800 Hz the alternative.
DEFAULT_LOW_CROSSOVERS_HZ = (700.0, 800.0, 1000.0, 1250.0)

#: The window each band computes, in seconds. The low and mid bands carry the
#: full decay because that is what the deliverable asks for; the high band's
#: 60 ms is the figure `w37_window` measured, where the tail error above 4 kHz
#: is inside the receiver spread.
DEFAULT_WINDOWS_S = {"low": 0.5, "mid": 0.5, "high": 0.060}

UNMEASURED = (
    "The upper crossover is the only one with money behind it, and lowering it "
    "buys that money entirely by widening the range the one-room high band "
    "domain has to cover. Whether one room is a legitimate domain below 4 kHz "
    "depends on doorway leakage and on what the rest of the flat contributes, "
    "which is W23 and is not measured. These figures price the trade; they do "
    "not choose a crossover."
)


@dataclass(frozen=True)
class BandPlan:
    """One band's domain, window and price."""

    name: str
    fmax_hz: float
    span_m: tuple[float, float, float]
    window_s: float
    domain: str
    cost: cost_model.SolveCost

    def record(self) -> dict[str, Any]:
        return {
            "band": self.name,
            "fmax_hz": self.fmax_hz,
            "span_m": list(self.span_m),
            "box_m3": float(self.span_m[0] * self.span_m[1] * self.span_m[2]),
            "window_s": self.window_s,
            "domain": self.domain,
            **self.cost.record(),
        }


def price(
    scene_span_m: tuple[float, float, float],
    room_span_m: tuple[float, float, float],
    *,
    low_crossover_hz: float,
    high_crossover_hz: float,
    usd_per_hour_per_card: float,
    top_hz: float = 16000.0,
    ppw: float = 10.5,
    windows_s: dict[str, float] | None = None,
) -> list[BandPlan]:
    """Price one three-band configuration.

    Each band is a separate solve on its own grid. The low and mid bands run on
    the whole apartment's bounding box; the high band runs on the box of the
    room holding the source and the receiver.
    """
    windows = {**DEFAULT_WINDOWS_S, **(windows_s or {})}
    if not 0.0 < low_crossover_hz < high_crossover_hz < top_hz:
        raise ValueError(
            f"crossovers must rise: got {low_crossover_hz}, {high_crossover_hz} under {top_hz} Hz"
        )

    layout = (
        ("low", low_crossover_hz, scene_span_m, "whole apartment"),
        ("mid", high_crossover_hz, scene_span_m, "whole apartment"),
        ("high", top_hz, room_span_m, "the room holding source and receiver"),
    )
    plans: list[BandPlan] = []
    for name, fmax, span, domain in layout:
        points = cost_model.grid_points_for(span, fmax, ppw=ppw)
        plans.append(
            BandPlan(
                name=name,
                fmax_hz=fmax,
                span_m=span,
                window_s=windows[name],
                domain=domain,
                cost=cost_model.estimate(
                    points,
                    windows[name],
                    cost_model.solver_rate_hz(fmax, ppw=ppw),
                    usd_per_hour_per_card=usd_per_hour_per_card,
                ),
            )
        )
    return plans


def build(
    scene_span_m: tuple[float, float, float],
    room_span_m: tuple[float, float, float],
    out: Path,
    *,
    usd_per_hour_per_card: float,
    low_crossovers_hz: tuple[float, ...] = DEFAULT_LOW_CROSSOVERS_HZ,
    high_crossovers_hz: tuple[float, ...] = DEFAULT_HIGH_CROSSOVERS_HZ,
    ppw: float = 10.5,
    windows_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Sweep both crossovers and write ``report.json``."""
    out.mkdir(parents=True, exist_ok=True)
    windows = {**DEFAULT_WINDOWS_S, **(windows_s or {})}

    sweep: list[dict[str, Any]] = []
    for low in low_crossovers_hz:
        for high in high_crossovers_hz:
            if low >= high:
                continue
            plans = price(
                scene_span_m,
                room_span_m,
                low_crossover_hz=low,
                high_crossover_hz=high,
                usd_per_hour_per_card=usd_per_hour_per_card,
                ppw=ppw,
                windows_s=windows,
            )
            sweep.append(
                {
                    "low_crossover_hz": low,
                    "high_crossover_hz": high,
                    "bands": [plan.record() for plan in plans],
                    "total_usd": sum(plan.cost.usd for plan in plans),
                    "total_gpu_hours": sum(plan.cost.gpu_hours for plan in plans),
                    "cards_max": max(plan.cost.cards for plan in plans),
                }
            )

    assumed = next(
        entry
        for entry in sweep
        if entry["low_crossover_hz"] == 1000.0 and entry["high_crossover_hz"] == 4000.0
    )
    report: dict[str, Any] = {
        "run": out.name,
        "reference_run": None,
        "trick": (
            "No trick: this prices the architecture rather than changing it. The "
            "three-band split of section 4.2, costed from the measured throughput "
            "and the exact grid-point count of each domain."
        ),
        "unmeasured": UNMEASURED,
        "scene_span_m": list(scene_span_m),
        "room_span_m": list(room_span_m),
        "points_per_wavelength": ppw,
        "windows_s": windows,
        "usd_per_hour_per_card": usd_per_hour_per_card,
        "assumed_split": assumed,
        "sweep": sweep,
        "omissions": [
            "one apartment and one room: the shares depend on how large the "
            "high band room is relative to the flat, and this is a single scene",
            "the low and mid bands assume a 0.5 s response, which suits a "
            "furnished flat and not a large or hard room",
            "voxelisation is not priced here: it is CPU time, it caches per "
            "geometry and grid step, and it is a separate budget line",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Price the three-band split.")
    parser.add_argument("--scene-span", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--room-span", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    parser.add_argument("--high-window-ms", type=float, default=60.0)
    args = parser.parse_args(argv)

    report = build(
        tuple(args.scene_span),
        tuple(args.room_span),
        args.out,
        usd_per_hour_per_card=args.usd_per_hour,
        windows_s={"high": args.high_window_ms / 1000.0},
    )

    rate = report["usd_per_hour_per_card"]
    assumed = report["assumed_split"]
    print(f"rate {rate:.3f} USD/h per card, windows {report['windows_s']}\n")
    print("The assumed split, 1 kHz and 4 kHz:")
    print(
        f"{'band':>5} {'fmax':>8} {'box':>10} {'points':>11} {'card-h':>8} {'USD':>7} {'share':>7}"
    )
    for band in assumed["bands"]:
        share = 100.0 * band["usd"] / assumed["total_usd"]
        print(
            f"{band['band']:>5} {band['fmax_hz']:>7.0f}H {band['box_m3']:>9.1f} "
            f"{band['grid_points']:>11.3e} {band['gpu_hours']:>8.2f} {band['usd']:>7.2f} "
            f"{share:>6.1f}%"
        )
    print(
        f"{'total':>5} {'':>8} {'':>10} {'':>11} "
        f"{assumed['total_gpu_hours']:>8.2f} {assumed['total_usd']:>7.2f}"
    )

    print("\nUpper crossover sweep, at a 1 kHz lower crossover:")
    print(f"{'high':>7} {'mid USD':>9} {'total USD':>10} {'saving':>8}")
    baseline = assumed["total_usd"]
    for entry in report["sweep"]:
        if entry["low_crossover_hz"] != 1000.0:
            continue
        mid = next(band for band in entry["bands"] if band["band"] == "mid")
        print(
            f"{entry['high_crossover_hz']:>6.0f}H {mid['usd']:>9.2f} "
            f"{entry['total_usd']:>10.2f} {baseline / entry['total_usd']:>7.2f}x"
        )

    print("\nLower crossover sweep, at a 4 kHz upper crossover:")
    print(f"{'low':>7} {'low USD':>9} {'total USD':>10} {'saving':>8}")
    for entry in report["sweep"]:
        if entry["high_crossover_hz"] != 4000.0:
            continue
        low = next(band for band in entry["bands"] if band["band"] == "low")
        print(
            f"{entry['low_crossover_hz']:>6.0f}H {low['usd']:>9.2f} "
            f"{entry['total_usd']:>10.2f} {baseline / entry['total_usd']:>7.2f}x"
        )
    print(f"\n{report['unmeasured']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
