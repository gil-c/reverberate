"""On the dense line: all 64 channels translated and fused, between the true centres.

:mod:`.line_gaps` read the pressure and took each point where the field says
it is. Two things are different here.

**The centres are the arrays' own.** An array stands on the node of its grid
nearest the point asked for, and a field records the point asked for. On the
line, 2 cm apart on a 32.7 mm grid, 341 points are 209 distinct expansions of
the low band, each up to 16 mm from where the field puts it. With the plan
that placed the arrays (``plan.json``), one point is kept per node of the low
grid and every offset is between the centres the plan records: the low
band's under :data:`SWITCH_HZ`, where a three band field takes its low
solve, the mid band's over it. Without a plan the offsets are the field's
positions, as before, and the difference between the two is the floor W44
could not explain.

**The error is read on the head's sphere.** Order 7 coefficients in N3D weigh
every degree alike, but at 250 Hz a head 10 cm in radius hears degree 5 at
``j_5(0.46)``, -90 dB. Each channel is therefore weighted by ``j_n(k a)``,
``a`` = :data:`HEAD_M`: the pressure the expansion gives on the sphere of the
head, whose mean square is the sum of the weighted channels' squares. The
error is that of all 64 weighted channels over the truth's, per octave, on
the early part; per degree it is that degree's error over the same whole, so
the degrees add to the total. Channel 0 alone is kept beside it.

Each case is set against how far the cells it read are from the source and
from the nearest surface (:func:`reverberate.spatial.translate.clearance_m`
on the occluders of ``mirror/scene.npz``), which is what the serving rule of
the translation library is drawn from.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import spherical_jn

from reverberate.experiments.w44_interpolation.scoring import EARLY_S, FADE_S, write_summary
from reverberate.spatial.sh import degrees_of
from reverberate.spatial.translate import (
    SOUND_SPEED_M_S,
    clearance_m,
    fusion_operator,
    translation_operator,
)

__all__ = ["BANDS", "FAR_M", "FMAX_HZ", "HEAD_M", "SWITCH_HZ", "line_channels"]

#: The octaves under the crossover, in Hz.
BANDS = [125, 250, 500, 1000]
#: The spectra are kept, and the errors read, up to here.
FMAX_HZ = 1500.0
#: The rate the kept band is brought back to time at.
RATE_HZ = 4000.0
#: Under this the offsets are the low band's centres, over it the mid band's:
#: the seam of a three band field, 0.8 of its low solve's 1 kHz.
SWITCH_HZ = 800.0
#: The radius of the sphere the error is read on, in metres.
HEAD_M = 0.10
#: A cell this far from the source or further is clear of its near field.
FAR_M = 1.2
#: The shares of the distance to the source the cases are grouped by.
SHARES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 1.00)
#: Distances up to this are pooled in the table by share, in metres.
POOLED_WITHIN_M = 0.20
#: At most this many targets a distance, evenly taken along the line.
CASES = 100


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    """Median, ninth decile and worst over cases of ``[case, band]``, per band."""
    return {
        "median": np.round(np.median(values, axis=0), 1).tolist(),
        "p90": np.round(np.percentile(values, 90, axis=0), 1).tolist(),
        "worst": np.round(values.max(axis=0), 1).tolist(),
    }


def _score(
    pred: np.ndarray, truth: np.ndarray, onset: np.ndarray, samples: int, rate: float, order: int
) -> np.ndarray:
    """Error over truth on the early part: ``[case, band, degree 0..order, all, channel 0]``, dB.

    ``pred`` and ``truth`` are ``[case, frequency, channel]``, the bins of
    the field's own transform up to :data:`FMAX_HZ`, already weighted.
    """
    low = round(samples * RATE_HZ / rate)
    freqs = np.fft.rfftfreq(low, 1 / RATE_HZ)
    masks = [(freqs >= b / np.sqrt(2)) & (freqs < b * np.sqrt(2)) for b in BANDS]
    degree = degrees_of(order)
    fade = int(FADE_S * RATE_HZ)
    out = np.zeros((pred.shape[0], len(BANDS), order + 3))

    def in_time(spectrum: np.ndarray) -> np.ndarray:
        full = np.zeros((low // 2 + 1, spectrum.shape[1]), dtype=np.complex64)
        full[: spectrum.shape[0]] = spectrum
        signal: np.ndarray = np.fft.irfft(full, n=low, axis=0).T
        return signal

    for case in range(pred.shape[0]):
        stop = min(low, round(onset[case] * RATE_HZ / rate) + int(EARLY_S * RATE_HZ))
        window = np.zeros(low)
        window[:stop] = 1.0
        window[stop - fade : stop] = 0.5 * (1 + np.cos(np.pi * np.arange(fade) / fade))
        true = in_time(truth[case])
        true_e = np.abs(np.fft.rfft(true * window, axis=-1)) ** 2
        error_e = np.abs(np.fft.rfft((in_time(pred[case]) - true) * window, axis=-1)) ** 2
        for b, mask in enumerate(masks):
            tb, eb = true_e[:, mask].sum(axis=1), error_e[:, mask].sum(axis=1)
            with np.errstate(divide="ignore"):
                for n in range(order + 1):
                    out[case, b, n] = 10 * np.log10(eb[degree == n].sum() / tb.sum())
                out[case, b, order + 1] = 10 * np.log10(eb.sum() / tb.sum())
                out[case, b, order + 2] = 10 * np.log10(eb[0] / tb[0])
    return out


def line_channels(
    field: Path,
    out: Path,
    *,
    plan: Path | None = None,
    scene: Path | None = None,
    steps: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12),
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Translation and fusion of all channels at each distance; ``summary_channels.json``.

    ``steps`` are distances in nodes of the low grid with ``plan``, in points
    of the line without. ``scene`` is the ``mirror/scene.npz`` whose occluders
    give each centre's clearance; without it the clearance is not known and
    is written as infinite.
    """
    started = time.time()
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        order = int(f.attrs["order"])
        source = np.asarray(f.attrs["source_position"], dtype=float)
        positions = f["positions"][:]
        points, channels, samples = f["ir"].shape
        freqs = np.fft.rfftfreq(samples, 1 / rate)
        keep = int(np.searchsorted(freqs, FMAX_HZ, side="right"))
        freqs = freqs[:keep]
        if abs(samples * RATE_HZ / rate - round(samples * RATE_HZ / rate)) > 1e-9:
            raise ValueError(f"{samples} samples at {rate:g} Hz are not whole at {RATE_HZ:g} Hz")
        spec = np.zeros((points, keep, channels), dtype=np.complex64)
        onset = np.zeros(points, dtype=int)
        for i in range(points):
            ir = f["ir"][i]
            onset[i] = int(np.argmax(np.abs(ir[0]) > 0.1 * np.abs(ir[0]).max()))
            spec[i] = np.fft.rfft(ir, axis=-1)[:, :keep].T
    say(f"{points} points read in {time.time() - started:.1f} s")
    weight = spherical_jn(
        degrees_of(order)[None, :], (2 * np.pi * freqs / SOUND_SPEED_M_S * HEAD_M)[:, None]
    ).astype(np.float32)

    low_centres = mid_centres = positions
    line_index = np.round(
        np.linalg.norm(positions - positions[0], axis=1)
        / np.linalg.norm(positions[1] - positions[0])
    ).astype(int)
    if plan is not None:
        placed = json.loads(Path(plan).read_text())["bands"]
        low_centres = np.asarray(placed["low"]["centres"], dtype=float)
        mid_centres = np.asarray(placed["mid"]["centres"], dtype=float)
        step = float(placed["low"]["grid_step_m"])
        along = np.linalg.norm(low_centres - low_centres[0], axis=1)
        line_index = np.round(along / step).astype(int)
    # One point per centre: the one the field puts nearest where its array stands.
    miss = np.linalg.norm(positions - low_centres, axis=1)
    of_node: dict[int, int] = {}
    for i in np.argsort(miss, kind="stable"):
        of_node.setdefault(int(line_index[i]), int(i))
    to_source = np.linalg.norm(low_centres - source[None, :], axis=1)
    clearance = np.full(points, np.inf)
    if scene is not None:
        with np.load(scene) as arrays:
            clearance = clearance_m(low_centres, np.asarray(arrays["occluder_vertices"]))
    under = freqs < SWITCH_HZ

    operators: dict[tuple[float, ...], np.ndarray] = {}

    def translation(low: np.ndarray, mid: np.ndarray) -> np.ndarray:
        key = (*np.round(low, 6), *np.round(mid, 6))
        if key not in operators:
            operator = translation_operator(mid, freqs, order)
            operator[under] = translation_operator(low, freqs[under], order)
            operators[key] = operator
        return operators[key]

    def fusion(low: np.ndarray, mid: np.ndarray) -> np.ndarray:
        key = (*np.round(low.ravel(), 6), *np.round(mid.ravel(), 6))
        if key not in operators:
            operator = fusion_operator(mid, freqs, order)
            operator[under] = fusion_operator(low, freqs[under], order)
            operators[key] = operator
        return operators[key]

    summary: dict[str, Any] = {
        "field": str(field),
        "plan": None if plan is None else str(plan),
        "bands_hz": BANDS,
        "head_m": HEAD_M,
        "far_m": FAR_M,
        "points": points,
        "centres": len(of_node),
        "columns": [f"degree {n}" for n in range(order + 1)] + ["all channels", "channel 0"],
        "distances": {},
    }
    pooled: dict[str, list[tuple[float, float, float]]] = {"translate": [], "fuse": []}
    for m in steps:
        cases = [
            (of_node[n], of_node[n - m], of_node[n + m])
            for n in sorted(of_node)
            if n - m in of_node and n + m in of_node
        ]
        cases = cases[:: max(1, len(cases) // CASES)]
        if not cases:
            continue
        operators.clear()
        truth = np.stack([spec[t] for t, _, _ in cases]) * weight[None]
        translated = np.zeros_like(truth)
        fused = np.zeros_like(truth)
        for j, (t, a, b) in enumerate(cases):
            offsets_low = np.stack(
                [low_centres[t] - low_centres[a], low_centres[t] - low_centres[b]]
            )
            offsets_mid = np.stack(
                [mid_centres[t] - mid_centres[a], mid_centres[t] - mid_centres[b]]
            )
            one = translation(offsets_low[0], offsets_mid[0])
            translated[j] = np.einsum("fce,fe->fc", one, spec[a]) * weight
            two = fusion(offsets_low, offsets_mid)
            fused[j] = np.einsum("fce,fe->fc", two, np.concatenate([spec[a], spec[b]], 1)) * weight
        targets = np.array([t for t, _, _ in cases])
        distance = float(
            np.median([np.linalg.norm(low_centres[t] - low_centres[a]) for t, a, _ in cases])
        )
        record: dict[str, Any] = {"distance_m": round(distance, 4), "cases": len(cases)}
        for name, pred, used in (("translate", translated, (1,)), ("fuse", fused, (1, 2))):
            errors = _score(pred, truth, onset[targets], samples, rate, order)
            cells = np.array([[case[u] for u in used] for case in cases])
            nearest = to_source[cells].min(axis=1)
            far = nearest >= FAR_M
            record[name] = {
                "far_cases": int(far.sum()),
                "least_clearance_m": round(float(clearance[cells].min()), 3),
            }
            if far.any():
                record[name]["far_all_channels"] = _stats(errors[far][:, :, order + 1])
                record[name]["far_channel_0"] = _stats(errors[far][:, :, order + 2])
                worst_band = errors[far][:, :, : order + 1].max(axis=1)
                record[name]["far_per_degree_worst_band"] = _stats(worst_band)
            if distance > POOLED_WITHIN_M:
                continue
            for share, all_db, omni_db in zip(
                distance / nearest,
                errors[:, :, order + 1].max(axis=1),
                errors[:, :, order + 2].max(axis=1),
                strict=True,
            ):
                pooled[name].append((float(share), float(all_db), float(omni_db)))
            say(
                f"{name:9s} {distance:.3f} m, {len(cases)} cases, {int(far.sum())} far:"
                f" all channels worst {record[name].get('far_all_channels', {}).get('worst')}"
            )
        summary["distances"][f"{distance:.3f}"] = record
    # Every case of every pooled distance, by the share of the source's distance it read.
    summary["by_share_of_source_distance"] = {}
    for name, rows in pooled.items():
        table = np.array(rows).reshape(-1, 3)
        found: dict[str, Any] = {}
        lower = 0.0
        for upper in SHARES:
            chosen = table[(table[:, 0] >= lower) & (table[:, 0] < upper)]
            if chosen.shape[0]:
                found[f"{lower:.2f}-{upper:.2f}"] = {
                    "cases": int(chosen.shape[0]),
                    "all_channels_p90": round(float(np.percentile(chosen[:, 1], 90)), 1),
                    "all_channels_worst": round(float(chosen[:, 1].max()), 1),
                    "channel_0_p90": round(float(np.percentile(chosen[:, 2], 90)), 1),
                    "channel_0_worst": round(float(chosen[:, 2].max()), 1),
                }
            lower = upper
        summary["by_share_of_source_distance"][name] = found
    summary["seconds"] = round(time.time() - started, 1)
    write_summary(out, "summary_channels.json", summary)
    return summary
