"""On the dense line: the error of each interpolation as a function of the spacing.

For a spacing ``g`` the two known points sit ``g / 2`` either side of a point of the line, whose
own solved response is the truth. Early part (onset to onset + 50 ms), W channel, per octave.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.w44_interpolation.scoring import (
    BANDS,
    EARLY_S,
    FADE_S,
    band_energy,
    delayed,
    early_window,
    error_db,
    onset_of,
    percentiles,
    write_summary,
)
from reverberate.spatial.translate import (
    SOUND_SPEED_M_S,
    fusion_weights,
    translation_weights,
)

__all__ = ["FMAX_HZ", "HALF_SPACINGS", "STEP_M", "line_gaps"]

#: The fields are compared up to here, where the line's solve stops.
FMAX_HZ = 11300.0
#: The line's pitch along z, in metres.
STEP_M = 0.02
#: Half the spacing between the two known points, in steps of the line:
#: spacings of 4, 8, 16, 24, 40 and 80 cm.
HALF_SPACINGS = (1, 2, 4, 6, 10, 20)


def line_gaps(field: Path, out: Path, *, say: Callable[[str], None] = print) -> dict[str, Any]:
    """Each interpolation's error at each spacing of the line; ``summary_gaps.json``, returned."""
    started = time.time()
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        order = int(f.attrs["order"])
        pos = f["positions"][:]
        dist = f["direct_path_m"][:]
        points, channels, samples = f["ir"].shape
        z = np.round((pos[:, 2] - pos[:, 2].min()) / STEP_M).astype(int)
        index = {int(k): i for i, k in enumerate(z)}
        freqs = np.fft.rfftfreq(samples, 1 / rate)
        keep = freqs <= FMAX_HZ
        spec = np.zeros((points, int(keep.sum()), channels), dtype=np.complex64)
        omni = np.zeros((points, samples), dtype=np.float32)
        for i in range(points):
            ir = f["ir"][i]
            omni[i] = ir[0]
            spec[i] = np.fft.rfft(ir, axis=-1)[:, keep].T
    say(f"{points} points read in {time.time() - started:.1f} s")
    fk = freqs[keep]
    lowpass = np.zeros(freqs.size)
    lowpass[keep] = 1.0
    fade = int(FADE_S * rate)

    def back(s: np.ndarray) -> np.ndarray:
        full = np.zeros(freqs.size, dtype=np.complex64)
        full[keep] = s
        signal: np.ndarray = np.fft.irfft(full, samples)
        return signal

    summary: dict[str, Any] = {"bands_hz": BANDS, "points": points, "step_m": STEP_M}
    for half in HALF_SPACINGS:
        off = np.array([0.0, 0.0, half * STEP_M])
        w_plus = translation_weights(off, fk, order)  # from the point below, a step up
        op = fusion_weights(np.stack([off, -off]), fk, order)  # the target seen from each
        rows: dict[str, list[list[float]]] = {
            k: [] for k in ("nearest", "linear", "aligned", "translate", "fusion")
        }
        for k in sorted(index):
            if k - half not in index or k + half not in index:
                continue
            t, a, b = index[k], index[k - half], index[k + half]
            truth = np.fft.irfft(np.fft.rfft(omni[t]) * lowpass, samples)
            win = early_window(samples, onset_of(omni[t]) + int(EARLY_S * rate), fade)
            ref = band_energy(truth * win, rate)
            wa, wb = (np.fft.irfft(np.fft.rfft(omni[n]) * lowpass, samples) for n in (a, b))
            al = [
                delayed(w, (dist[t] - dist[n]) / SOUND_SPEED_M_S, rate) * (dist[n] / dist[t])
                for w, n in ((wa, a), (wb, b))
            ]
            translated = back(np.einsum("fc,fc->f", w_plus, spec[a]))
            fused = back(
                np.einsum("fc,fc->f", op[:, 0], spec[a]) + np.einsum("fc,fc->f", op[:, 1], spec[b])
            )
            for name, pred in (
                ("nearest", wa),
                ("linear", 0.5 * (wa + wb)),
                ("aligned", 0.5 * (al[0] + al[1])),
                ("translate", translated),
                ("fusion", fused),
            ):
                rows[name].append(error_db(pred, truth, win, ref, rate))
        record = {}
        say(f"\nspacing {2 * half * STEP_M:.2f} m, {len(rows['linear'])} cases")
        for name, values in rows.items():
            record[name] = percentiles(np.array(values), worst=False)
            say(f"  {name:10s} median {record[name]['median']}  p90 {record[name]['p90']}")
        summary[f"spacing_{2 * half * STEP_M:.2f}"] = record
    summary["seconds"] = round(time.time() - started, 1)
    write_summary(out, "summary_gaps.json", summary)
    return summary
