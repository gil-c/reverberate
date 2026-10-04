"""Leave-one-out on a field: how well a point's omni response is predicted from its neighbours.

Predictors, each judged on the W channel of the point left out:

- nearest: one neighbour a pitch away, as it is (what the web app plays);
- linear: the mean of the two opposite neighbours (a gap of two pitches);
- aligned: the same after each neighbour is put on the target's direct delay and 1/r level;
- translate: one neighbour's expansion evaluated a pitch away;
- translate2: the mean of the two opposite translations;
- midpoint: two neighbouring cells each translated half a pitch to their common midpoint,
  and how far the two disagree (no ground truth there: an agreement, not an error).

Error: energy of the difference over energy of the truth, per octave, in dB, on the early part
(onset to onset + 50 ms) and on the whole response. Late level: octave energy between
onset + 50 ms and onset + 150 ms, predicted as the mean of the neighbours' own.
"""

from __future__ import annotations

import json
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
    LATE_S,
    band_energy,
    delayed,
    early_window,
    onset_of,
    percentiles,
    write_summary,
)
from reverberate.experiments.w44_interpolation.translate import (
    SOUND_SPEED_M_S,
    translation_weights,
)

__all__ = ["leave_one_out"]


def leave_one_out(
    field: Path, out: Path, *, limit: int | None = None, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Every point of the lattice predicted from its neighbours; ``rows.json`` and ``summary.json``.

    ``field`` is one source's response field (``docs/formats/response-field.md``),
    ``limit`` keeps its first points only, for a quick look. Returns the summary.
    """
    started = time.time()
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        order = int(f.attrs["order"])
        step = np.asarray(f.attrs["grid_step_m"], dtype=float)
        cells = f["cell_index"][:]
        dist = f["direct_path_m"][:]
        rooms = np.array([r.decode() for r in f["rooms"][:]])
        source_room = str(f.attrs["source_room"])
        borrowed = f["low_borrowed"][:]
        points, _, samples = f["ir"].shape
        freqs = np.fft.rfftfreq(samples, 1 / rate)
        pitch = float(step[0])
        whole = {
            d: translation_weights(np.array(v, float) * pitch, freqs, order)
            for d, v in DIRS.items()
        }
        half = {
            d: translation_weights(np.array(v, float) * pitch / 2, freqs, order)
            for d, v in DIRS.items()
        }
        index = {tuple(c): i for i, c in enumerate(cells)}
        todo = range(points) if limit is None else range(limit)
        omni = np.zeros((points, samples), dtype=np.float32)
        pred_whole: dict[tuple[int, str], np.ndarray] = {}
        pred_half: dict[tuple[int, str], np.ndarray] = {}
        for i in todo:
            ir = f["ir"][i]
            omni[i] = ir[0]
            spec = np.fft.rfft(ir, axis=-1).astype(np.complex64).T  # [freq, channel]
            for d in DIRS:
                pred_whole[i, d] = np.einsum("fc,fc->f", whole[d], spec)
                pred_half[i, d] = np.einsum("fc,fc->f", half[d], spec)
        say(f"read and translated {len(todo)} points in {time.time() - started:.1f} s")

    def neighbour(i: int, d: str) -> int | None:
        c = cells[i] + np.array(DIRS[d])
        j = index.get(tuple(c))
        return j if j is not None and (limit is None or j < limit) else None

    fade = int(FADE_S * rate)
    rows: list[dict[str, Any]] = []
    for t in todo:
        if borrowed[t]:
            continue
        truth = omni[t]
        onset = onset_of(truth)
        early = early_window(samples, onset + int(EARLY_S * rate), fade)
        late_a, late_b = onset + int(EARLY_S * rate), onset + int(LATE_S * rate)
        for axis in ("x", "z"):
            a, b = neighbour(t, axis + "-"), neighbour(t, axis + "+")
            if a is None or b is None or borrowed[a] or borrowed[b]:
                continue
            # a sits on the minus side: its expansion reaches t by a "+" step.
            ta = np.fft.irfft(pred_whole[a, axis + "+"], samples)
            tb = np.fft.irfft(pred_whole[b, axis + "-"], samples)
            al = [
                delayed(omni[n], (dist[t] - dist[n]) / SOUND_SPEED_M_S, rate) * (dist[n] / dist[t])
                for n in (a, b)
            ]
            candidates = {
                "nearest": [omni[a], omni[b]],
                "linear": [0.5 * (omni[a] + omni[b])],
                "aligned": [0.5 * (al[0] + al[1])],
                "translate": [ta, tb],
                "translate2": [0.5 * (ta + tb)],
            }
            row: dict[str, Any] = {
                "point": int(t),
                "axis": axis,
                "room": rooms[t],
                "direct_m": float(dist[t]),
            }
            for label, win in (("early", early), ("whole", None)):
                ref = truth * win if win is not None else truth
                ref_e = band_energy(ref, rate)
                for name, preds in candidates.items():
                    err = np.mean(
                        [
                            band_energy((p * win if win is not None else p) - ref, rate)
                            for p in preds
                        ],
                        axis=0,
                    )
                    row[f"{name}_{label}"] = (10 * np.log10(err / ref_e)).tolist()
            late = []
            for n in (a, b):
                o = onset_of(omni[n])
                late.append(
                    band_energy(omni[n][o + int(EARLY_S * rate) : o + int(LATE_S * rate)], rate)
                )
            true_late = band_energy(truth[late_a:late_b], rate)
            row["late_level_db"] = (10 * np.log10(0.5 * (late[0] + late[1]) / true_late)).tolist()
            # The midpoint between t and b, reached from both sides by half a pitch.
            pa = np.fft.irfft(pred_half[t, axis + "+"], samples)
            pb = np.fft.irfft(pred_half[b, axis + "-"], samples)
            for label, win in (("early", early), ("whole", None)):
                qa, qb = (pa * win, pb * win) if win is not None else (pa, pb)
                row[f"midpoint_{label}"] = (
                    10 * np.log10(band_energy(qa - qb, rate) / band_energy(0.5 * (qa + qb), rate))
                ).tolist()
            rows.append(row)
    if not rows:
        raise ValueError(f"{field} has no point with two opposite neighbours")

    keys = [k for k in rows[0] if isinstance(rows[0][k], list)]
    summary: dict[str, Any] = {
        "field": str(field),
        "bands_hz": BANDS,
        "pitch_m": pitch,
        "order": order,
        "samples": len(rows),
        "source_room": source_room,
        "seconds": round(time.time() - started, 1),
    }
    for group, same in (("source_room", True), ("other_rooms", False)):
        chosen = [r for r in rows if (r["room"] == source_room) == same]
        if not chosen:
            continue
        summary[group] = {"samples": len(chosen)}
        for k in keys:
            v = np.array([r[k] for r in chosen])
            summary[group][k] = percentiles(np.abs(v) if k == "late_level_db" else v)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows))
    write_summary(out, "summary.json", summary)
    for group in ("source_room", "other_rooms"):
        if group not in summary:
            continue
        g = summary[group]
        say(f"\n{group}: {g['samples']} samples; bands {BANDS}")
        for k in keys:
            say(f"  {k:18s} median {g[k]['median']}  p90 {g[k]['p90']}")
    say(f"\ntotal {time.time() - started:.1f} s")
    return summary
