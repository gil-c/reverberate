"""Compare responses: where, if anywhere, one run departs from another.

Promoted from ``data/runs/b0_truncation/b0_compare.py``. The claim it was
written to test is exactness, not similarity, so the primary number is not an
error norm but a **time**: the first sample at which a run stops being bit
identical to the reference, and whether that time scales with the cut divided
by the speed of sound.

Four things per cut, and the third is the whole experiment.

- ``first_inexact_ms``: the first sample that differs from the reference at all,
  bit for bit. With a common grid origin this is a meaningful question; without
  one it was answered by interpolation weights rather than by physics.
- Peak and RMS residual in dB relative to the reference peak, over the full
  compared window and over each candidate planning window.
- Whether the departure **scales with the cut length**. A departure caused by
  deleted triangles must; one caused by the artificial absorbing boundary does
  not, which is how the first B0 run was diagnosed.
- Where the departure falls relative to both ``cut / c`` and
  ``cut / (c sqrt(3))``. The discrete scheme propagates along its stencil at
  ``h / Ts``, which at the Courant limit is ``c sqrt(3)``, so the second is the
  conservative bound and the planning rule. The ratio measured here says how
  conservative it is.

Two departure thresholds, ``machine`` at single precision rounding and
``audible`` at -60 dB, so successive reports stay comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.engine import sim_consts
from reverberate.experiments.run import RESULTS_FILE

__all__ = [
    "C_AIR",
    "SQRT3",
    "compare",
    "departure_ms",
    "first_inexact_ms",
    "main",
    "residual_db",
    "response",
]

C_AIR = 343.2
SQRT3 = float(np.sqrt(3.0))


def response(run_dir: Path) -> tuple[np.ndarray, float]:
    """The raw receiver signal and its sample rate, before any filtering."""
    with h5py.File(Path(run_dir) / "sim_outs.h5", "r") as handle:
        signal = np.asarray(handle["u_out"], dtype=float)
    return signal[0], sim_consts(Path(run_dir)).sample_rate


def departure_ms(
    reference: np.ndarray, other: np.ndarray, sample_rate: float, threshold: float
) -> float | None:
    """When ``other`` first differs from ``reference`` by more than ``threshold``.

    ``None`` means it never did, over the samples the two runs share.
    """
    n = min(reference.size, other.size)
    difference = np.abs(reference[:n] - other[:n])
    exceeded = np.flatnonzero(difference > threshold)
    if exceeded.size == 0:
        return None
    return float(1000.0 * exceeded[0] / sample_rate)


def first_inexact_ms(reference: np.ndarray, other: np.ndarray, sample_rate: float) -> float | None:
    """The first sample that is not bit identical. ``None`` if none is."""
    return departure_ms(reference, other, sample_rate, 0.0)


def residual_db(
    reference: np.ndarray,
    other: np.ndarray,
    sample_rate: float,
    peak: float,
    until_ms: float | None,
) -> dict[str, float | int | None]:
    """Peak and RMS of ``other - reference`` in dB relative to the reference peak.

    ``until_ms`` truncates the comparison to a candidate planning window;
    ``None`` uses every sample the two runs share. A residual of exactly zero is
    reported as ``None`` rather than as minus infinity, because the honest
    statement is "bit identical", not a number.
    """
    n = min(reference.size, other.size)
    if until_ms is not None:
        n = min(n, int(np.floor(until_ms * sample_rate / 1000.0)))
    if n <= 0:
        return {"samples": 0, "peak_db": None, "rms_db": None}
    difference = reference[:n] - other[:n]
    peak_residual = float(np.max(np.abs(difference)))
    rms_residual = float(np.sqrt(np.mean(difference**2)))

    def to_db(value: float) -> float | None:
        return None if value <= 0.0 else round(20.0 * float(np.log10(value / peak)), 2)

    return {
        "samples": int(n),
        "peak_db": to_db(peak_residual),
        "rms_db": to_db(rms_residual),
    }


def _load_results(path: Path) -> dict[tuple[str, float, float, str], dict[str, Any]]:
    """The latest run per configuration, keyed by what makes it a configuration.

    A rerun of the same configuration supersedes the earlier one, and the bounds
    mode is part of the configuration: runs with common bounds must not be
    confused with runs made under some other choice.
    """
    latest: dict[tuple[str, float, float, str], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        mode = result.get("bounds_mode", "legacy")
        latest[(result["scene"], result["fmax"], result["ppw"], mode)] = result
    return latest


def compare(
    out: Path,
    reference_name: str = "apartment_full",
    results_file: str = RESULTS_FILE,
) -> list[dict[str, Any]]:
    """Compare every non-reference run in ``out`` against the reference run."""
    by_key = _load_results(out / results_file)

    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for (_scene, fmax, ppw, mode), result in by_key.items():
        groups.setdefault((fmax, ppw, mode), []).append(result)

    report: list[dict[str, Any]] = []
    for (fmax, ppw, mode), group in sorted(groups.items()):
        names = {r["scene"]: r for r in group}
        if reference_name not in names:
            continue
        ref, sample_rate = response(Path(names[reference_name]["run_dir"]))
        peak = float(np.max(np.abs(ref)))
        thresholds = {
            "machine": peak * float(np.finfo(np.float32).eps) * 8.0,
            "audible": peak * 10.0 ** (-60.0 / 20.0),
        }
        entries: list[dict[str, Any]] = []
        for name, result in sorted(names.items()):
            if name == reference_name or result["cut_m"] is None:
                continue
            other, other_rate = response(Path(result["run_dir"]))
            if abs(other_rate - sample_rate) >= 1e-9:
                raise ValueError(f"{name} and {reference_name} do not share a sample rate")
            cut_m = float(result["cut_m"])
            # Two candidate windows. ``cut / c`` is the physical one; the
            # discrete scheme's stencil carries information at ``c sqrt(3)`` at
            # the Courant limit, so ``cut / (c sqrt(3))`` is the conservative
            # one and the planning rule until a run says otherwise.
            window_c_ms = 1000.0 * cut_m / C_AIR
            window_stencil_ms = 1000.0 * cut_m / (C_AIR * SQRT3)
            n = min(ref.size, other.size)
            inexact = first_inexact_ms(ref, other, sample_rate)
            entry: dict[str, Any] = {
                "fmax": fmax,
                "ppw": ppw,
                "bounds_mode": mode,
                "scene": name,
                "cut_m": cut_m,
                "pad_m": result.get("pad_m"),
                "window_c_ms": round(window_c_ms, 3),
                "window_stencil_ms": round(window_stencil_ms, 3),
                "predicted_departure_ms": round(window_c_ms, 3),
                "samples_compared": int(n),
                "peak_reference": peak,
                "max_abs_diff": float(np.max(np.abs(ref[:n] - other[:n]))),
                "first_inexact_ms": None if inexact is None else round(inexact, 4),
                "bit_exact_throughout": inexact is None,
                "residual_full_window": residual_db(ref, other, sample_rate, peak, None),
                "residual_to_window_c": residual_db(ref, other, sample_rate, peak, window_c_ms),
                "residual_to_window_stencil": residual_db(
                    ref, other, sample_rate, peak, window_stencil_ms
                ),
                "grid_points_reference": names[reference_name]["grid_points"],
                "grid_points": result["grid_points"],
                "engine_s_reference": names[reference_name]["engine_s"],
                "engine_s": result["engine_s"],
            }
            for label, threshold in thresholds.items():
                departed = departure_ms(ref, other, sample_rate, threshold)
                entry[f"departure_{label}_ms"] = None if departed is None else round(departed, 3)
                entry[f"threshold_{label}"] = threshold
                entry[f"early_{label}"] = None if departed is None else bool(departed < window_c_ms)
                # How conservative each bound is: above 1 means the run stayed
                # clean past the bound, which is what a conservative bound looks
                # like. Below 1 means the bound was optimistic.
                for bound, value in (("c", window_c_ms), ("stencil", window_stencil_ms)):
                    entry[f"{label}_over_window_{bound}"] = (
                        None if departed is None else round(departed / value, 3)
                    )
            entries.append(entry)

        _annotate_scaling(entries)
        report.extend(entries)

    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    for entry in report:
        print(_describe(entry))
    return report


def _annotate_scaling(entries: list[dict[str, Any]]) -> None:
    """Whether the departure grows with the cut, which is the whole experiment.

    A departure caused by the deleted triangles scales with the cut, one caused
    by the artificial boundary does not. Two cuts differing by a factor of two
    give one honest test of that, and it is stated as a ratio rather than as a
    verdict word.
    """
    for label in ("first_inexact_ms", "departure_machine_ms", "departure_audible_ms"):
        usable = [e for e in entries if e[label] is not None]
        if len(usable) < 2:
            continue
        low = min(usable, key=lambda e: e["cut_m"])
        high = max(usable, key=lambda e: e["cut_m"])
        cut_ratio = high["cut_m"] / low["cut_m"]
        time_ratio = high[label] / low[label] if low[label] else None
        for entry in entries:
            entry[f"scaling_{label}"] = {
                "cut_ratio": round(cut_ratio, 3),
                "time_ratio": None if time_ratio is None else round(time_ratio, 3),
                "scales_with_cut": (
                    None
                    if time_ratio is None
                    else bool(abs(time_ratio - cut_ratio) < 0.25 * cut_ratio)
                ),
            }


def _describe(entry: dict[str, Any]) -> str:
    inexact = entry["first_inexact_ms"]
    full = entry["residual_full_window"]
    lines = [
        f"fmax {entry['fmax']:.0f} [{entry['bounds_mode']}] {entry['scene']}: "
        f"cut {entry['cut_m']} m, window c {entry['window_c_ms']:.2f} ms, "
        f"window c sqrt(3) {entry['window_stencil_ms']:.2f} ms, "
        f"first inexact {inexact if inexact is not None else 'never'} ms, "
        f"-60 dB departure {entry['departure_audible_ms']} ms, "
        f"peak residual {full['peak_db']} dB, RMS {full['rms_db']} dB, "
        f"grid {entry['grid_points'] / entry['grid_points_reference']:.3f} of reference"
    ]
    scaling = entry.get("scaling_first_inexact_ms")
    if scaling is not None:
        lines.append(
            f"    departure scales with cut: cut ratio {scaling['cut_ratio']}, "
            f"time ratio {scaling['time_ratio']}, verdict {scaling['scales_with_cut']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference", default="apartment_full")
    parser.add_argument(
        "--results",
        default=RESULTS_FILE,
        help="name of the JSONL log inside OUT, for reading an older sweep",
    )
    args = parser.parse_args(argv)
    compare(args.out, args.reference, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
