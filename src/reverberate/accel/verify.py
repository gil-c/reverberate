"""Did a stage reproduce its twin? The comparisons a campaign is judged by.

Grids are compared dataset by dataset and must be equal
(:func:`reverberate.accel.voxelise.entries_identical`). Signals are
compared point by point at a stated tolerance: the largest absolute
difference over the point's own peak, the relative energy of the
difference, how many points came back float32-identical, and, per octave
band, the level difference in decibels, because a listener hears levels
and a last-bit disagreement in a transform is not a level. The report is
numbers, never a verdict; the caller states the tolerance and reads them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = ["compare_encoded", "compare_fields", "compare_signals", "octave_levels_db"]

#: Octave band centres compared, in Hz.
OCTAVES_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


def octave_levels_db(signals: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Energy per octave band of ``[channel, sample]``, summed over channels, in dB."""
    block = np.asarray(signals, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(block, axis=-1)) ** 2
    frequency = np.fft.rfftfreq(block.shape[-1], 1.0 / sample_rate_hz)
    levels = []
    for centre in OCTAVES_HZ:
        low, high = centre / np.sqrt(2.0), centre * np.sqrt(2.0)
        band = (frequency >= low) & (frequency < high)
        energy = float(spectrum[:, band].sum())
        levels.append(10.0 * np.log10(energy) if energy > 0 else -np.inf)
    return np.asarray(levels)


def compare_signals(
    a: np.ndarray, b: np.ndarray, sample_rate_hz: float, *, tolerance: float = 1e-6
) -> dict[str, Any]:
    """``a`` against ``b``, both ``[point, channel, sample]``; ``b`` is the reference."""
    n = min(a.shape[-1], b.shape[-1])
    x = np.asarray(a[..., :n], dtype=np.float64)
    y = np.asarray(b[..., :n], dtype=np.float64)
    peak = np.abs(y).max(axis=(1, 2))
    peak = np.where(peak > 0, peak, 1.0)
    max_over_peak = np.abs(x - y).max(axis=(1, 2)) / peak
    energy_y = np.sum(y * y, axis=(1, 2))
    energy_y = np.where(energy_y > 0, energy_y, 1.0)
    relative_energy = np.sum((x - y) ** 2, axis=(1, 2)) / energy_y
    identical = np.asarray([np.array_equal(a[i, :, :n], b[i, :, :n]) for i in range(a.shape[0])])
    octave = np.asarray(
        [
            octave_levels_db(x[i], sample_rate_hz) - octave_levels_db(y[i], sample_rate_hz)
            for i in range(x.shape[0])
        ]
    )
    octave = np.where(np.isfinite(octave), octave, 0.0)
    return {
        "points": int(x.shape[0]),
        "samples_compared": int(n),
        "tolerance": tolerance,
        "max_over_peak": {
            "max": float(max_over_peak.max()),
            "median": float(np.median(max_over_peak)),
            "worst_point": int(np.argmax(max_over_peak)),
        },
        "relative_energy": {
            "max": float(relative_energy.max()),
            "median": float(np.median(relative_energy)),
        },
        "float32_identical": int(identical.sum()),
        "within_tolerance": int(np.count_nonzero(max_over_peak <= tolerance)),
        "octave_level_db": {
            "bands_hz": list(OCTAVES_HZ),
            "max_abs": [float(v) for v in np.abs(octave).max(axis=0)],
        },
        "per_point_max_over_peak": [float(v) for v in max_over_peak],
    }


def compare_encoded(
    a: Path, b: Path, *, tolerance: float = 1e-6, chunk: int = 32
) -> dict[str, Any]:
    """Two ``encoded.h5`` files, joined on ``point_index``, a chunk of points at a time."""
    return compare_fields(a, b, tolerance=tolerance, chunk=chunk, dataset="signals")


def compare_fields(
    a: Path, b: Path, *, tolerance: float = 1e-6, chunk: int = 32, dataset: str = "ir"
) -> dict[str, Any]:
    """Two ``field/<source>.h5`` files, point by point, without reading either whole."""
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        ia = np.asarray(fa["point_index"][...])
        ib = np.asarray(fb["point_index"][...])
        rate = float(fa.attrs["sample_rate_hz"])
        common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        flags: dict[str, Any] = {}
        for key in ("has_high", "solved_to_hz", "low_borrowed", "high_dropped"):
            if key in fa and key in fb:
                flags[key] = int(np.count_nonzero(fa[key][...][pa] != fb[key][...][pb]))
        attrs = {
            key: (float(fa.attrs[key]), float(fb.attrs[key]))
            for key in ("mid_on_high_gain", "high_dropped_count", "low_borrowed_count")
            if key in fa.attrs and key in fb.attrs
        }
        parts: list[dict[str, Any]] = []
        order = np.argsort(pa)
        pa, pb = pa[order], pb[order]
        for start in range(0, common.size, chunk):
            sa, sb = pa[start : start + chunk], pb[start : start + chunk]
            xa = np.asarray(fa[dataset][np.sort(sa)])[np.argsort(np.argsort(sa))]
            xb = np.asarray(fb[dataset][np.sort(sb)])[np.argsort(np.argsort(sb))]
            parts.append(compare_signals(xa, xb, rate, tolerance=tolerance))
    per_point = [v for part in parts for v in part["per_point_max_over_peak"]]
    octave_max = np.max([part["octave_level_db"]["max_abs"] for part in parts], axis=0)
    return {
        "points": int(common.size),
        "points_only_in_a": int(ia.size - common.size),
        "points_only_in_b": int(ib.size - common.size),
        "tolerance": tolerance,
        "max_over_peak": {
            "max": float(max(per_point)),
            "median": float(np.median(per_point)),
            "worst_point": int(common[int(np.argmax(per_point))]),
        },
        "relative_energy_max": float(max(part["relative_energy"]["max"] for part in parts)),
        "float32_identical": int(sum(part["float32_identical"] for part in parts)),
        "within_tolerance": int(sum(part["within_tolerance"] for part in parts)),
        "octave_level_db_max_abs": dict(
            zip([f"{int(f)}" for f in OCTAVES_HZ], [float(v) for v in octave_max], strict=True)
        ),
        "flags_differing": flags,
        "attrs": attrs,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["encoded", "field"])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path, help="the reference")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    compare = compare_encoded if args.kind == "encoded" else compare_fields
    report = compare(args.a, args.b, tolerance=args.tolerance)
    text = json.dumps({k: v for k, v in report.items() if k != "per_point_max_over_peak"}, indent=1)
    print(text)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
