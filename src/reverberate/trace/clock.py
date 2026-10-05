"""Whether the low band and the mirror are on one clock and one scale, read on the direct sound.

A pack joins two bands that were computed apart: the wave solver's pairs,
on the geometric clock and the field's scale, and the mirror's arrivals.
The trace checks the join before it writes anything, and for a day it
checked it on the wrong thing. It read each pair's **loudest sample**
against the mirror's direct sound, and at a far pair the loudest sample
of the low band is not the direct sound: at 4 to 10 m in hssd_0076 the
direct sound is there, at its time and at its level, and an arrival 5 to
20 ms later is half as loud again. Two whole scenes were stopped after
five hours of solves by a clock that was right (2026-10-05), and a window
of far sources alone could not pass at any percentile.

So the response is read **where the mirror puts the direct sound**
(:func:`read_direct`), on every pair the mirror gives a direct path:

- *the time*: the first peak of the response within :data:`SEARCH_S` of
  the mirror's first arrival plus the pack's lead. The first, not the
  largest: a floor's reflection follows the direct sound by 1 to 2 ms at
  those distances and may be the larger;
- *the level*: the response's spectrum over :data:`BAND_HZ` in a window of
  :data:`WINDOW_S` about that peak, times the distance, over
  :data:`reverberate.spatial.lowband.FIELD_UNIT_AT_1M`: what the direct
  sound of a source of unit gain reads at 1 m, so 0 dB for a pair on the
  field's scale whatever the solver's own band limit does to its peak.

:func:`verdict` then says of a run's pairs that the two are on one clock
and one scale, or which of the two they are not and by how much. It reads
the **first quartile of the times and the third of the levels**: nothing
arrives before a direct sound and little makes it louder, while a pair's
own path may make it later or weaker (a floor's reflection taken for it,
a doorway that clips it at 500 Hz, a coarse grid's dispersion over ten
metres), so the early and the loud readings are the clock's and the
scale's and the others are the room's. On the first scene's pairs
(2026-10-05): -0.02 ms and +0.3 dB over every distance, +0.01 ms and
+0.3 dB at 4 m and more, +0.04 ms and -1.8 dB at 8 m and more, where the
loudest sample read 12.0 and 12.7 ms at its first decile against a lead of
10.67; at 7.2 points per wavelength, +0.17 ms and -1.7 dB at 8 m and more.
What the old statistic read stays in the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reverberate.spatial.lowband import FIELD_UNIT_AT_1M, from_stored

__all__ = ["BAND_HZ", "CLOCK_S", "SCALE_DB", "SEARCH_S", "WINDOW_S", "read_direct", "verdict"]

#: The direct sound must come within this of the mirror's, s: half a period of the
#: crossover's 1 kHz, beyond which the two bands cancel at the join.
CLOCK_S = 0.5e-3
#: And its level must be within this of ``1 / d`` on the field's scale, dB, in the median.
#: A pair's own reading moves 2 to 4 dB with what arrives inside its window; a cache on
#: another scale is off by a solve's ``fmax`` ratio or by the field's unit, 6 dB and more.
SCALE_DB = 3.0
#: How far either side of the mirror's time the direct sound is looked for, s.
SEARCH_S = 3.0e-3
#: A peak of the search counts from this share of the largest there: under it are the
#: ripples a band limited pulse has before it.
FIRST_PEAK_SHARE = 0.6
#: Half the window the level is read in, s, a raised cosine about the peak; and the band,
#: inside every solve's own (80 to 1500 Hz) and under the crossover.
WINDOW_S = 3.0e-3
BAND_HZ = (300.0, 800.0)
#: A pair holds its direct sound where the mirror puts it when the peak found there is
#: this share of the response's loudest sample or more, whatever its scale. The direct
#: sound of the first scene's far pairs is 0.53 of the loudest sample at the least; what a
#: response holds before anything has arrived is under 0.01 of it.
PRESENT_SHARE = 0.2
_RATE_HZ = 48000.0


def read_direct(
    omni: Any, first_s: float, *, lead_s: float, sound_speed_m_s: float
) -> dict[str, Any]:
    """A pair's response where the mirror puts its direct sound: when it peaks, how loud it is.

    ``omni`` is channel 0 on the pack's clock and the field's scale, at
    4 kHz (what :func:`reverberate.trace.level.pair_omni` returns);
    ``first_s`` the mirror's direct path, in seconds of travel. Returns
    ``direct_trail_s``, what the first peak near ``first_s + lead_s`` trails
    ``first_s`` by (the lead, on one clock; ``None`` where the response has
    no peak there and only rises or falls through the search),
    ``direct_level_db``, the level there against ``1 / d``, and
    ``direct_share``, the peak over the response's loudest sample.
    """
    signal = np.asarray(from_stored(np.asarray(omni, dtype=float)[None, :], _RATE_HZ)[0])
    size = np.abs(signal)
    centre = int(round((first_s + lead_s) * _RATE_HZ))
    reach = int(round(SEARCH_S * _RATE_HZ))
    low, high = max(centre - reach, 1), min(centre + reach, signal.size - 2)
    if high <= low:
        return {"direct_trail_s": None, "direct_level_db": -200.0, "direct_share": 0.0}
    held = size[low : high + 1]
    floor = FIRST_PEAK_SHARE * float(held.max())
    peaks = np.flatnonzero(
        (held >= floor) & (held >= size[low - 1 : high]) & (held >= size[low + 1 : high + 2])
    )
    at = low + int(peaks[0]) if peaks.size else min(max(centre, low), high)
    half = int(round(WINDOW_S * _RATE_HZ))
    span = np.arange(at - half, at + half + 1)
    inside = (span >= 0) & (span < signal.size)
    window = 0.5 + 0.5 * np.cos(np.pi * (span - at) / (half + 1.0))
    piece = np.zeros(span.size)
    piece[inside] = signal[span[inside]]
    points = int(_RATE_HZ / 10.0)
    spectrum = np.abs(np.fft.rfft(piece * window, n=points))
    freqs = np.fft.rfftfreq(points, 1.0 / _RATE_HZ)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    distance = max(float(first_s) * float(sound_speed_m_s), 1e-3)
    level = float(spectrum[band].mean()) * distance / FIELD_UNIT_AT_1M
    return {
        "direct_trail_s": at / _RATE_HZ - float(first_s) if peaks.size else None,
        "direct_level_db": round(float(20.0 * np.log10(max(level, 1e-10))), 3),
        "direct_share": round(float(size[at]) / max(float(size.max()), 1e-30), 4),
    }


def verdict(
    records: list[dict[str, Any]], lead_s: float, *, sound_speed_m_s: float
) -> tuple[dict[str, Any], str | None]:
    """What a run's pairs say of the clock and the scale: the report's part, and why to stop.

    ``records`` are the levelling's, one a pair; those the mirror gives a
    direct path and that carry a reading are read. The second answer is
    ``None`` when the two bands are on one clock and one scale, or when no
    pair has a direct path, and otherwise a message that says which of the
    two fails, by how much, and on how many pairs.
    """
    read = [r for r in records if r.get("direct") and "direct_level_db" in r]
    if not read:
        return {"pairs": 0}, None
    level = np.array([float(r["direct_level_db"]) for r in read])
    found = np.array([r.get("direct_trail_s") is not None for r in read])
    off = np.array(
        [float(r["direct_trail_s"]) if r.get("direct_trail_s") is not None else 0.0 for r in read]
    ) - float(lead_s)
    away = np.array([float(r["first_s"]) for r in read]) * float(sound_speed_m_s)
    share = np.array([float(r.get("direct_share", 0.0)) for r in read])
    present = found & (share >= PRESENT_SHARE)
    report: dict[str, Any] = {
        "pairs": len(read),
        "pairs_with_their_direct_sound_there": int(present.sum()),
        # The direct sound over the response's loudest sample: 1 where it is the loudest.
        "share_of_the_loudest": {
            "p10": round(float(np.percentile(share, 10)), 3),
            "median": round(float(np.median(share)), 3),
        },
        "allowed": {"off_s": CLOCK_S, "level_db": SCALE_DB},
    }
    loud = float(np.percentile(level, 75))
    if present.sum() * 2 < len(read):
        # Nothing arrives where the mirror puts it: the loudest samples say about how far.
        trails = np.array([float(r["onset_s"]) - float(r["first_s"]) for r in read])
        early = float(np.percentile(trails, 10))
        far = early - float(lead_s)
        report["level_db"] = {"third_quartile": round(loud, 2)}
        return report, (
            "the low band and the mirror are not on one clock: within"
            f" {SEARCH_S * 1e3:g} ms of where the mirror puts the direct sound the low band holds"
            f" none ({loud:+.1f} dB of 1/d at the third quartile of {len(read)} pairs with a"
            f" direct path, {int(present.sum())} of them with something there), and the pairs'"
            f" loudest samples trail the mirror's direct sound by {early * 1e3:.2f} ms at the"
            f" first decile where the pack's lead is {lead_s * 1e3:.2f} ms: the low band is"
            f" about {abs(far) * 1e3:.2f} ms {'late' if far > 0 else 'early'}. The pair cache is"
            " on the geometric clock: a response in it starts when its source does, and one"
            " that carries a lead of its own is late by it"
        )
    off, level, away = off[present], level[present], away[present]
    first = float(np.percentile(off, 25))
    loud = float(np.percentile(level, 75))
    by_distance: dict[str, Any] = {}
    for low, high in ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))):
        mine = (away >= low) & (away < high)
        if mine.any():
            name = f"{low:g} to {high:g} m" if np.isfinite(high) else f"over {low:g} m"
            by_distance[name] = {
                "pairs": int(mine.sum()),
                "off_ms": [round(float(v) * 1e3, 3) for v in np.percentile(off[mine], (25, 50))],
                "level_db": [round(float(v), 2) for v in np.percentile(level[mine], (50, 75))],
            }
    report.update(
        {
            "off_s": {
                "first_quartile": round(first, 6),
                "median": round(float(np.median(off)), 6),
                "p90": round(float(np.percentile(off, 90)), 6),
            },
            "level_db": {
                "p10": round(float(np.percentile(level, 10)), 2),
                "median": round(float(np.median(level)), 2),
                "third_quartile": round(loud, 2),
            },
            "by_distance": by_distance,
        }
    )
    if abs(first) > CLOCK_S:
        return report, (
            "the low band and the mirror are not on one clock: the low band's direct sound comes"
            f" {abs(first) * 1e3:.2f} ms {'after' if first > 0 else 'before'} the mirror's,"
            f" lead included, at the first quartile of {len(off)} pairs with a direct path"
            f" ({CLOCK_S * 1e3:g} ms are allowed; its level is {loud:+.1f} dB of 1/d). The pair"
            " cache is on the geometric clock: a response in it starts when its source does"
        )
    if abs(loud) > SCALE_DB:
        return report, _scale(loud, len(off), f"the clock is right, {first * 1e3:+.2f} ms")
    return report, None


def _scale(loud: float, pairs: int, clock: str) -> str:
    return (
        "the low band and the mirror are not on one scale: where the mirror puts the direct"
        f" sound the low band reads {abs(loud):.1f} dB {'over' if loud > 0 else 'under'} 1/d on"
        f" the field's scale, at the third quartile of {pairs} pairs with a direct path"
        f" ({SCALE_DB:g} dB are allowed; {clock}). The pair cache is not on the field's unit:"
        " its solve's fmax, or the unit itself"
    )
