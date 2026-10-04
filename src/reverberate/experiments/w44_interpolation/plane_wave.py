"""Leave-one-out with a plane wave fit: the neighbours' expansions fused per frequency.

Each point of the lattice is predicted by :func:`~.translate.fusion_operator`
from one neighbour, from the two opposite neighbours of an axis, and from all
four, and judged like :mod:`.leave_one_out` judges its predictors: the early
part of the W channel, per octave, in dB. The fields are low passed at
:data:`FMAX_HZ`, which the operators stop at.
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
    DIRS,
    EARLY_S,
    FADE_S,
    band_energy,
    early_window,
    error_db,
    onset_of,
    percentiles,
    write_summary,
)
from reverberate.experiments.w44_interpolation.translate import REGULARISATION, fusion_operator

__all__ = ["FMAX_HZ", "leave_one_out_fusion"]

#: The operators are computed, and the fields compared, up to here.
FMAX_HZ = 6000.0


def leave_one_out_fusion(
    field: Path, out: Path, *, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Every point of the lattice predicted by fusion; ``summary_planewave.json``, returned."""
    started = time.time()
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        order = int(f.attrs["order"])
        pitch = float(np.asarray(f.attrs["grid_step_m"])[0])
        cells = f["cell_index"][:]
        rooms = np.array([r.decode() for r in f["rooms"][:]])
        source_room = str(f.attrs["source_room"])
        borrowed = f["low_borrowed"][:]
        points, _, samples = f["ir"].shape
        freqs = np.fft.rfftfreq(samples, 1 / rate)
        keep = freqs <= FMAX_HZ
        fk = freqs[keep]
        names = list(DIRS)
        off = {d: np.array(DIRS[d], float) * pitch for d in names}
        # A neighbour on side d of the target sits at +off[d] from it.
        ops = {
            "pair_x": fusion_operator(np.stack([off["x-"], off["x+"]]), fk, order),
            "pair_z": fusion_operator(np.stack([off["z-"], off["z+"]]), fk, order),
            "four": fusion_operator(np.stack([off[d] for d in names]), fk, order),
            "one": fusion_operator(np.stack([off["x-"]]), fk, order),
        }
        say(f"operators in {time.time() - started:.1f} s")
        side = {"x-": ("pair_x", 0), "x+": ("pair_x", 1), "z-": ("pair_z", 0), "z+": ("pair_z", 1)}
        omni = np.zeros((points, samples), dtype=np.float32)
        part: dict[tuple[int, str, str], np.ndarray] = {}
        for i in range(points):
            ir = f["ir"][i]
            omni[i] = ir[0]
            spec = np.fft.rfft(ir, axis=-1)[:, keep].T.astype(np.complex64)  # [freq, channel]
            for j, d in enumerate(names):
                name, slot = side[d]
                part[i, d, "pair"] = np.einsum("fc,fc->f", ops[name][:, slot], spec)
                part[i, d, "four"] = np.einsum("fc,fc->f", ops["four"][:, j], spec)
            part[i, "x-", "one"] = np.einsum("fc,fc->f", ops["one"][:, 0], spec)
    index = {tuple(c): i for i, c in enumerate(cells)}

    def neighbour(i: int, d: str) -> int | None:
        j = index.get(tuple(cells[i] + np.array(DIRS[d])))
        return None if j is None or borrowed[j] else j

    def back(spec: np.ndarray) -> np.ndarray:
        full = np.zeros(freqs.size, dtype=np.complex64)
        full[keep] = spec
        signal: np.ndarray = np.fft.irfft(full, samples)
        return signal

    fade = int(FADE_S * rate)
    lowpass = np.zeros(freqs.size)
    lowpass[keep] = 1.0
    rows: list[dict[str, Any]] = []
    for t in range(points):
        if borrowed[t]:
            continue
        near = {d: neighbour(t, d) for d in names}
        truth = np.fft.irfft(np.fft.rfft(omni[t]) * lowpass, samples)
        win = early_window(samples, onset_of(omni[t]) + int(EARLY_S * rate), fade)
        ref = band_energy(truth * win, rate)
        row: dict[str, Any] = {"point": t, "room": rooms[t]}

        for axis in ("x", "z"):
            a, b = near[axis + "-"], near[axis + "+"]
            if a is not None and b is not None:
                fused = back(part[a, axis + "-", "pair"] + part[b, axis + "+", "pair"])
                row[f"pair_{axis}"] = error_db(fused, truth, win, ref, rate)
        one = near["x-"]
        if one is not None:
            row["one"] = error_db(back(part[one, "x-", "one"]), truth, win, ref, rate)
        four = [(n, d) for d, n in near.items() if n is not None]
        if len(four) == len(names):
            parts = [part[n, d, "four"] for n, d in four]
            fused = back(sum(parts[1:], start=parts[0]))
            row["four"] = error_db(fused, truth, win, ref, rate)
        rows.append(row)

    summary: dict[str, Any] = {"bands_hz": BANDS, "fmax_hz": FMAX_HZ, "lambda": REGULARISATION}
    for group, same in (("source_room", True), ("other_rooms", False)):
        chosen = [r for r in rows if (r["room"] == source_room) == same]
        summary[group] = {}
        for key in ("one", "pair", "four"):
            values = np.array(
                [r[k] for r in chosen for k in r if k == key or k.startswith(key + "_")]
            )
            if values.size:
                summary[group][key] = {"samples": int(values.shape[0]), **percentiles(values)}
                say(f"{group} {key} {summary[group][key]}")
    summary["seconds"] = round(time.time() - started, 1)
    write_summary(out, "summary_planewave.json", summary)
    say(f"total {summary['seconds']} s")
    return summary
