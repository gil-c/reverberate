"""A source's levelling scalar, steadied along what moves: ``level/high_gain_db`` over a walk.

``level/high_gain_db`` multiplies everything a source has above the
crossover. It is the mirror's gain plus the step's *seam*: how far the wave
response of the step's pairs stands over the mirror's in the crossover's
octave (``scene-pack.md``). Each pair is levelled on its own, as
``mirror.hybrid.blend`` levels a point, and for a source and a head at rest
that is one number and is right. Along a walk it is a new number every
8 cm of the source's rail and every 0.15 m of the head's path, and the
seams of neighbours are not neighbours: over the 18 219 pairs of the first
whole scene (2026-10-05) a seam is +0.4 dB at the first decile and +3.4 dB
at the ninth, and the scalar of a source that walked, or that the listener
walked past, went up and down by 1.8 to 2.5 dB within half a second in the
top tenth of those half seconds and by 5 to 8 dB within a second at the
worst (a steady street noise 5 m from a walking head: -7.6 to -1.0 dB
between 487.75 and 488.75 s). Everything over 1 kHz of a source breathing
by that much is heard, and no room does it: what a seam measures is how
far the two solvers disagree about a place, which is a property of the
dwelling's rooms and changes over metres, plus what one octave of one
response happens to hold at one point, which is the part that changes
over centimetres and belongs to neither band.

:func:`steadied` keeps the first and drops the second: within every run of
audible steps the scalar is averaged under a raised cosine two seconds
either side (:data:`STEADY_S`), in decibels. A source at rest in front of a
head at rest keeps its scalar to the bit. The pairs' own seams stay in the
pack (``low/seam_db``); only the per step table moves. On that scene's
fourteen sources the scalar then moves within half a second by 0.3 to
0.5 dB at the ninth decile, 0.5 to 1.5 dB at the 99th percentile and 0.6 to
1.9 dB at the worst, where it moved by 1.7 to 2.4, 2.6 to 4.9 and 3.7 to
6.8 dB; a just noticeable step of level is 1 dB. Averaged over one second
either side the 99th percentile was still 1.1 to 2.1 dB.

**An option, off.** The trace writes each step's own seam, as it did, and
nothing is decided before the owner has heard a pack both ways: a traced
pack is steadied by :func:`relevel_pack`, ``python -m reverberate.render
relevel PACK``, **in place**: the table is a few hundred kilobytes of a
file of gigabytes, what the trace wrote is kept beside it as
``level/high_gain_db_traced``, and ``--undo`` puts it back. No response is
solved or read again. If it is kept, the trace's own step is one call of
:func:`steadied` at the end of ``reverberate.trace.level.step_levels``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["STEADY_S", "relevel_pack", "steadied", "wander_db"]

#: Half the raised cosine the scalar is averaged under, seconds: a walk of two metres,
#: two dozen solved positions of a rail and a dozen cells of the head's path.
STEADY_S = 2.0
#: What the trace's own table is kept as in a pack that was relevelled.
TRACED = "high_gain_db_traced"


def steadied(high_gain_db: np.ndarray, audible: np.ndarray, half_steps: int) -> np.ndarray:
    """``high_gain_db`` averaged under a raised cosine ``half_steps`` either side, run by run.

    A run is a stretch of audible steps on end; nothing is averaged across
    a silence, on the other side of which the source may stand elsewhere.
    Near a run's end the window is what the run holds of it. Steps that are
    not audible keep what they had. Float32, as the pack keeps it.
    """
    values = np.asarray(high_gain_db, dtype=np.float64)
    heard = np.asarray(audible, dtype=bool)
    out = np.array(high_gain_db, dtype=np.float32)
    if half_steps < 1 or values.size == 0:
        return out
    window = 0.5 + 0.5 * np.cos(np.pi * np.arange(-half_steps, half_steps + 1) / (half_steps + 1))
    edges = np.flatnonzero(np.diff(np.concatenate([[False], heard, [False]]).astype(np.int8)))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        run = values[start:stop]
        if np.ptp(run) == 0.0:
            continue  # at rest: the same number, to the bit
        inside = slice(half_steps, half_steps + run.size)
        total = np.convolve(run, window, mode="full")[inside]
        weight = np.convolve(np.ones(run.size), window, mode="full")[inside]
        out[start:stop] = (total / weight).astype(np.float32)
    return out


def wander_db(high_gain_db: np.ndarray, audible: np.ndarray, steps: int) -> dict[str, float]:
    """How far the scalar moves within ``steps`` steps, over the runs where it moves at all.

    The range of every window of ``steps + 1`` audible steps on end in
    which the scalar is not constant: its ninth decile, its 99th
    percentile and its largest, in dB; zeros for a source that never moves.
    """
    values = np.asarray(high_gain_db, dtype=np.float64)
    heard = np.asarray(audible, dtype=bool)
    if values.size <= steps:
        return {"p90": 0.0, "p99": 0.0, "max": 0.0}
    views = np.lib.stride_tricks.sliding_window_view(values, steps + 1)
    whole = np.lib.stride_tricks.sliding_window_view(heard, steps + 1).all(axis=1)
    span = views.max(axis=1) - views.min(axis=1)
    moving = span[whole & (span > 0.0)]
    if moving.size == 0:
        return {"p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p90": round(float(np.percentile(moving, 90)), 2),
        "p99": round(float(np.percentile(moving, 99)), 2),
        "max": round(float(moving.max()), 2),
    }


def relevel_pack(path: Path, *, seconds: float = STEADY_S, undo: bool = False) -> dict[str, Any]:
    """Steady ``level/high_gain_db`` of every source of the pack at ``path``, in place.

    What the trace wrote is kept as ``level/high_gain_db_traced`` the first
    time and is what every later call starts from, so calling twice, or
    with another ``seconds``, does not steady a steadied table. ``undo``
    puts the trace's table back and removes the copy. Returns, per source,
    how far the scalar moved within half a second before and after, and
    the largest change of a step's value.
    """
    import h5py

    report: dict[str, Any] = {"pack": str(path), "seconds": None if undo else float(seconds)}
    with h5py.File(Path(path), "r+") as f:
        step_s = float(f.attrs["step_s"])
        half = int(round(float(seconds) / step_s))
        within = int(round(0.5 / step_s))
        sources: dict[str, Any] = {}
        for name, group in f["sources"].items():
            level = group["level"]
            audible = np.asarray(group["audible"][...], dtype=bool)
            traced = np.asarray(
                level[TRACED][...] if TRACED in level else level["high_gain_db"][...]
            )
            if undo:
                level["high_gain_db"][...] = traced
                if TRACED in level:
                    del level[TRACED]
                level.attrs.pop("steadied_s", None)
                sources[name] = {"restored": True}
                continue
            made = steadied(traced, audible, half)
            if TRACED not in level:
                level.create_dataset(TRACED, data=traced)
            level["high_gain_db"][...] = made
            level.attrs["steadied_s"] = float(seconds)
            sources[name] = {
                "within_half_a_second_before_db": wander_db(traced, audible, within),
                "within_half_a_second_after_db": wander_db(made, audible, within),
                "largest_change_db": round(
                    float(np.abs(made - traced)[audible].max(initial=0.0)), 2
                ),
            }
        report["sources"] = sources
    return report
