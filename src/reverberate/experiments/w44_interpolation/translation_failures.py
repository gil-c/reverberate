"""Where the translation of one pitch fails round 500 Hz, and what those cases share.

On the lattice the median translation holds at 500 Hz and its ninth decile
does not. Every translation of one neighbour to a point is read here in third
octaves, a case is called bad when its worst third octave between 400 and
630 Hz is over :data:`BAD_DB`, and the bad cases are set against the good on
what could explain them: the distance of the neighbour and of the target from
the source, the cells missing round each (a wall or a piece of furniture
within a pitch), the share of the neighbour's energy in its upper orders, and
the level of the target at 500 Hz.
"""

from __future__ import annotations

import collections
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.w44_interpolation.scoring import (
    DIRS,
    EARLY_S,
    FADE_S,
    early_window,
    onset_of,
    write_summary,
)
from reverberate.experiments.w44_interpolation.translate import translation_weights
from reverberate.spatial.sh import degrees_of

__all__ = ["BAD_DB", "THIRDS", "translation_failures"]

#: Third octave centres the error is read in, in Hz.
THIRDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250]
#: A translation is bad when its error in the worst of 400, 500 and 630 Hz is over this, in dB.
BAD_DB = -6.0
#: The octave of 500 Hz, in which the energy per order is read.
_OCTAVE_500 = (354.0, 707.0)
#: What is compared between the good and the bad cases.
_FACTORS = {
    "dn": "neighbour to source, m",
    "dt": "target to source, m",
    "mn": "cells missing round the neighbour",
    "mt": "cells missing round the target",
    "hi": "neighbour's energy per channel in orders 5 to 7 over orders 0 to 2, dB",
    "lvl": "target's early energy at 500 Hz, dB",
}


def _spread(values: np.ndarray) -> dict[str, float] | None:
    """Median, first and ninth decile of a factor over some cases; ``None`` over none."""
    if not values.size:
        return None
    return {
        "median": round(float(np.median(values)), 2),
        "p10": round(float(np.percentile(values, 10)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
    }


def _median_db(errors: np.ndarray) -> list[float] | None:
    """The median error per third octave over some cases; ``None`` over none."""
    if not errors.size:
        return None
    median: list[float] = np.round(np.median(errors, 0), 1).tolist()
    return median


def _third_energy(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    s = np.abs(np.fft.rfft(x)) ** 2
    return np.array(
        [s[(freqs >= c / 2 ** (1 / 6)) & (freqs < c * 2 ** (1 / 6))].sum() for c in THIRDS]
    )


def translation_failures(
    field: Path, out: Path, *, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Every one-pitch translation of the lattice; ``summary_failures.json`` and ``diag_rows.npy``.

    ``diag_rows.npy`` is ``[case, 3]``: the target, the neighbour, and the
    worst error of the three third octaves round 500 Hz, in dB.
    """
    started = time.time()
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        order = int(f.attrs["order"])
        pitch = float(np.asarray(f.attrs["grid_step_m"])[0])
        source = np.asarray(f.attrs["source_position"], dtype=float)
        cells = f["cell_index"][:]
        dist = f["direct_path_m"][:]
        pos = f["positions"][:]
        borrowed = f["low_borrowed"][:]
        rooms = np.array([r.decode() for r in f["rooms"][:]])
        points, _, samples = f["ir"].shape
        freqs = np.fft.rfftfreq(samples, 1 / rate)
        index = {tuple(c): i for i, c in enumerate(cells)}
        w = {
            d: translation_weights(np.array(v, float) * pitch, freqs, order)
            for d, v in DIRS.items()
        }
        omni = np.zeros((points, samples), np.float32)
        pred: dict[tuple[int, str], np.ndarray] = {}
        order_e = np.zeros((points, order + 1))
        deg = degrees_of(order)
        octave = (freqs >= _OCTAVE_500[0]) & (freqs < _OCTAVE_500[1])
        for i in range(points):
            ir = f["ir"][i]
            omni[i] = ir[0]
            spec = np.fft.rfft(ir, axis=-1).astype(np.complex64).T
            e = (np.abs(spec[octave]) ** 2).sum(0)
            order_e[i] = [e[deg == n].sum() / (2 * n + 1) for n in range(order + 1)]
            for d in DIRS:
                pred[i, d] = np.einsum("fc,fc->f", w[d], spec)

    def missing(i: int) -> int:
        """Cells of the three by three block round ``i`` that the field does not hold."""
        return sum(
            tuple(cells[i] + np.array([a, 0, b])) not in index
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
        )

    fade = int(FADE_S * rate)
    rows: list[dict[str, Any]] = []
    for t in range(points):
        if borrowed[t]:
            continue
        truth = omni[t]
        win = early_window(samples, onset_of(truth) + int(EARLY_S * rate), fade)
        ref = _third_energy(truth * win, freqs)
        for d, v in DIRS.items():
            # The neighbour on the minus side steps +d to t.
            n = index.get(tuple(cells[t] - np.array(v)))
            if n is None or borrowed[n]:
                continue
            p = np.fft.irfft(pred[n, d], samples)
            o = order_e[n]
            rows.append(
                {
                    "t": t,
                    "n": n,
                    "d": d,
                    "err": 10 * np.log10(_third_energy((p - truth) * win, freqs) / ref),
                    "dn": dist[n],
                    "dt": dist[t],
                    "mn": missing(n),
                    "mt": missing(t),
                    "room": rooms[t],
                    "hi": 10 * np.log10(o[5:].sum() / o[:3].sum()),
                    "lvl": 10 * np.log10(ref[THIRDS.index(500)]),
                    "toward": float(np.dot(np.array(v), pos[n] - source)),
                }
            )
    errors = np.array([r["err"] for r in rows])
    round_500 = [THIRDS.index(c) for c in (400, 500, 630)]
    worst = errors[:, round_500].max(1)
    bad = worst > BAD_DB
    summary: dict[str, Any] = {
        "field": str(field),
        "thirds_hz": THIRDS,
        "bad_db": BAD_DB,
        "cases": len(rows),
        "bad": int(bad.sum()),
        "bad_share": round(float(bad.mean()), 3),
        "median_good": _median_db(errors[~bad]),
        "median_bad": _median_db(errors[bad]),
        "factors": {},
    }
    say(f"cases {len(rows)} bad (worse than {BAD_DB:g} dB in 400-630 Hz) {bad.sum()}")
    say(f"thirds {THIRDS}")
    say(f"median good {summary['median_good']}")
    say(f"median bad  {summary['median_bad']}")
    for key, meaning in _FACTORS.items():
        factor = np.array([r[key] for r in rows], dtype=float)
        found = {"meaning": meaning, "good": _spread(factor[~bad]), "bad": _spread(factor[bad])}
        summary["factors"][key] = found
        say(f"{key:4s} good {found['good']} | bad {found['bad']}")
    dn = np.array([r["dn"] for r in rows])
    summary["by_neighbour_distance"] = {}
    for label, mask in (
        ("neighbour within 0.8 m of source", dn < 0.8),
        ("0.8-1.5 m", (dn >= 0.8) & (dn < 1.5)),
        ("beyond 1.5 m", dn >= 1.5),
    ):
        share = round(float(bad[mask].mean()), 3) if mask.any() else None
        summary["by_neighbour_distance"][label] = {"cases": int(mask.sum()), "bad_share": share}
        say(f"{label} {mask.sum()} bad share {share}")
    for key, who in (("mn", "neighbour"), ("mt", "target")):
        count = np.array([r[key] for r in rows])
        shares = summary[f"by_cells_missing_round_{who}"] = {}
        for k in range(7):
            mask = count == k
            if mask.sum():
                shares[str(k)] = {
                    "cases": int(mask.sum()),
                    "bad_share": round(float(bad[mask].mean()), 3),
                }
                say(
                    f"missing cells round {who} {k} {mask.sum()}"
                    f" bad share {shares[str(k)]['bad_share']}"
                )
    targets = np.array([r["t"] for r in rows])
    neighbours = np.array([r["n"] for r in rows])
    frequent = collections.Counter(neighbours[bad].tolist()).most_common(8)
    summary["bad_neighbours_most_frequent"] = [[int(n), int(count)] for n, count in frequent]
    summary["distinct_bad_neighbours"] = len(set(neighbours[bad].tolist()))
    summary["distinct_bad_targets"] = len(set(targets[bad].tolist()))
    say(f"bad neighbours most frequent {frequent}")
    say(
        f"distinct bad neighbours {summary['distinct_bad_neighbours']}"
        f" distinct bad targets {summary['distinct_bad_targets']}"
    )
    summary["seconds"] = round(time.time() - started, 1)
    write_summary(out, "summary_failures.json", summary)
    np.save(
        out / "diag_rows.npy",
        np.array([(r["t"], r["n"], e) for r, e in zip(rows, worst, strict=True)]),
    )
    return summary
