"""On the dense line: what a degree is worth under a frequency, and how far apart cells may be.

The line of ``w44_clarify_interpolation`` is solved every 2 cm, so the
field at a head between two cells is known. Two things are read from it.

:func:`measure_degrees`: the cells' expansions are moved to a target as
the engine moves them, one cell translated or two fused
(:mod:`reverberate.spatial.translate`), **one input degree at a time**. The
prediction with degree ``n`` kept from a frequency up is then a sum of
those eight parts under eight weights, so every cutoff of the family
``alpha n c / (2 pi R)`` is scored against the solved truth from one pass,
with today's (every degree at every frequency) beside it. Two families are
tried: the rule in ``k R`` over ``n`` alone, and the rule of
:func:`reverberate.render.compact.cut_hz`, which follows ``j_n``.

:func:`measure_pitch`: the fusion of two cells a pitch apart, at every node
between them, against the source's distance.

As in :mod:`reverberate.experiments.w44_interpolation.line_channels`, one
point is kept per node of the low grid and every offset is between the
centres the arrays stood at; only the cases the serving rule allows are
read (:data:`reverberate.spatial.translate.SOURCE_SHARE`), unless said.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.w46_compact_low.scoring import (
    BANDS_HZ,
    RATE_HZ,
    head_weights,
    level_stats,
    scores,
    stats,
)
from reverberate.render.compact import REACH_M, cut_hz, degree_weights
from reverberate.spatial.sh import degrees_of
from reverberate.spatial.translate import (
    SOUND_SPEED_M_S,
    SOURCE_SHARE,
    apply_fusion,
    apply_translation,
    fusion_inverse,
)

__all__ = ["Line", "cutoffs", "measure_degrees", "measure_pitch"]

#: The spectra are kept up to here; a three band field takes its low solve under the seam.
FMAX_HZ = 1500.0
SWITCH_HZ = 800.0
#: The cutoffs tried: ``alpha`` of ``k R >= alpha n``, and the decibels of ``cut_hz``.
ALPHAS = (0.25, 0.35, 0.5, 0.7, 1.0)
FLOORS_DB = (60.0, 50.0, 40.0, 30.0, 20.0)
TODAY = "today"


def cutoffs(order: int, sound_speed_m_s: float = SOUND_SPEED_M_S) -> dict[str, np.ndarray]:
    """Every cutoff tried, by name: from where each degree is whole, ``[degree]``, in Hz."""
    found: dict[str, np.ndarray] = {TODAY: np.zeros(order + 1)}
    for floor in FLOORS_DB:
        found[f"{floor:g} dB"] = cut_hz(order, floor, sound_speed_m_s)
    per_degree = sound_speed_m_s / (2.0 * np.pi * REACH_M)
    for alpha in ALPHAS:
        found[f"k R = {alpha:g} n"] = alpha * np.arange(order + 1) * per_degree
    return found


class Line:
    """The line's spectra under :data:`FMAX_HZ`, one point a node, and where each array stood."""

    def __init__(self, field: Path, plan: Path, *, say: Callable[[str], None] = print) -> None:
        started = time.time()
        with h5py.File(field, "r") as f:
            self.rate = float(f.attrs["sample_rate_hz"])
            self.order = int(f.attrs["order"])
            self.source = np.asarray(f.attrs["source_position"], dtype=float)
            positions = f["positions"][:]
            points, channels, samples = f["ir"].shape
            self.samples = round(samples * RATE_HZ / self.rate)
            freqs = np.fft.rfftfreq(samples, 1.0 / self.rate)
            keep = int(np.searchsorted(freqs, FMAX_HZ, side="right"))
            self.freqs = freqs[:keep]
            placed = json.loads(Path(plan).read_text())["bands"]
            self.low = np.asarray(placed["low"]["centres"], dtype=float)
            self.mid = np.asarray(placed["mid"]["centres"], dtype=float)
            step = float(placed["low"]["grid_step_m"])
            along = np.linalg.norm(self.low - self.low[0], axis=1)
            index = np.round(along / step).astype(int)
            miss = np.linalg.norm(positions - self.low, axis=1)
            self.of_node: dict[int, int] = {}
            for i in np.argsort(miss, kind="stable"):
                self.of_node.setdefault(int(index[i]), int(i))
            self.step_m = step
            self.spec: dict[int, np.ndarray] = {}
            self.onset: dict[int, int] = {}
            scale = self.samples / samples
            for i in sorted(self.of_node.values()):
                ir = f["ir"][i]
                peak = np.abs(ir[0]) > 0.1 * np.abs(ir[0]).max()
                self.onset[i] = round(int(np.argmax(peak)) * RATE_HZ / self.rate)
                # On the scale of the response at 4 kHz: ``[channel, frequency]``.
                self.spec[i] = (np.fft.rfft(ir, axis=-1)[:, :keep] * scale).astype(np.complex64)
        self.to_source = np.linalg.norm(self.low - self.source[None, :], axis=1)
        self.under = self.freqs < SWITCH_HZ
        self.weight = head_weights(self.order, self.freqs)
        say(
            f"{len(self.of_node)} nodes of {points} points read in {time.time() - started:.0f} s;"
            f" the source is {self.to_source.min():.2f} to {self.to_source.max():.2f} m away"
        )

    def in_time(self, spectrum: np.ndarray) -> np.ndarray:
        """``[channel, frequency]`` under :data:`FMAX_HZ` as ``[channel, sample]`` at 4 kHz."""
        full = np.zeros((spectrum.shape[0], self.samples // 2 + 1), dtype=np.complex128)
        full[:, : spectrum.shape[1]] = spectrum
        made: np.ndarray = np.fft.irfft(full, n=self.samples, axis=-1)
        return made

    def mover(self, target: int, cells: list[int]) -> Callable[[list[np.ndarray]], np.ndarray]:
        """What brings fields at ``cells`` to ``target``: translated if one, fused if two.

        The cells' Gram matrix is inverted once here, for every field the
        returned function is then given.
        """
        parts = []
        for band, centres in ((self.under, self.low), (~self.under, self.mid)):
            offsets = np.stack([centres[target] - centres[c] for c in cells])
            freqs = self.freqs[band]
            inverse = None if len(cells) == 1 else fusion_inverse(-offsets, freqs, self.order)
            parts.append((band, offsets, freqs, inverse))

        def moved(fields: list[np.ndarray]) -> np.ndarray:
            out = np.zeros(fields[0].shape, dtype=np.complex128)
            for band, offsets, freqs, inverse in parts:
                if inverse is None:
                    out[:, band] = apply_translation(
                        fields[0][:, band], offsets[0], freqs, self.order
                    )
                else:
                    stacked = np.stack([one[:, band] for one in fields])
                    out[:, band] = apply_fusion(
                        stacked, offsets, freqs, self.order, inverse=inverse
                    )
            return out

        return moved

    def cases(
        self, before: int, after: int | None, *, every_share: bool = False
    ) -> list[tuple[int, ...]]:
        """``(target, cell[, cell])``: a cell ``before`` nodes before, one ``after`` after."""
        found = []
        for n in sorted(self.of_node):
            wanted = [n - before] + ([] if after is None else [n + after])
            if not all(w in self.of_node for w in wanted):
                continue
            case = (self.of_node[n], *(self.of_node[w] for w in wanted))
            reach = max(float(np.linalg.norm(self.low[case[0]] - self.low[c])) for c in case[1:])
            nearest = min(float(self.to_source[c]) for c in case[1:])
            if every_share or reach <= SOURCE_SHARE * nearest:
                found.append(case)
        return found


def _thin(cases: list[Any], count: int) -> list[Any]:
    if len(cases) <= count:
        return cases
    return [cases[i] for i in np.unique(np.linspace(0, len(cases) - 1, count).round().astype(int))]


def measure_degrees(
    line: Line, *, cases: int = 40, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Every cutoff of :func:`cutoffs` against the solved truth, one cell and two.

    The geometries are the engine's: one cell 0.10 m and 0.20 m away, two
    cells 0.16 m apart with the head 0.07 m from one, two cells 0.39 m
    apart with the head half way (0.20 m from each), and two 0.59 m apart
    with the head half way, the furthest a fusion may reach. For each: the
    error per third octave and per degree of the output, early and whole,
    ninth decile and worst case; and beside it what the cutoff alone
    changed, the prediction with it against the prediction without.
    """
    started = time.time()
    order = line.order
    degree = degrees_of(order)
    step = line.step_m
    geometries = {
        f"one cell, {3 * step:.3f} m": (3, None),
        f"one cell, {6 * step:.3f} m": (6, None),
        f"two cells {5 * step:.3f} m apart, head {2 * step:.3f} m from one": (2, 3),
        f"two cells {12 * step:.3f} m apart, head half way": (6, 6),
        f"two cells {18 * step:.3f} m apart, head half way": (9, 9),
    }
    rules = cutoffs(order)
    weights = {name: degree_weights(line.freqs, whole) for name, whole in rules.items()}
    kinds = ("early", "whole", "early_degree", "whole_degree", "early_level", "late_level")
    summary: dict[str, Any] = {
        "bands_hz": list(BANDS_HZ),
        "pass_hz": {name: np.round(whole).tolist() for name, whole in rules.items()},
        "geometries": {},
    }
    for name, (before, after) in geometries.items():
        chosen = _thin(line.cases(before, after), cases)
        if not chosen:
            continue
        found: dict[str, dict[str, list[np.ndarray]]] = {
            rule: {kind: [] for kind in (*kinds, "changed_early", "changed_whole")}
            for rule in rules
        }
        for case in chosen:
            target, cells = case[0], list(case[1:])
            # One input degree at a time: the cutoffs are then weights of these parts.
            parts = np.zeros((order + 1, degree.size, line.freqs.size), dtype=np.complex128)
            moved = line.mover(target, cells)
            for n in range(order + 1):
                only = (degree == n)[:, None]
                parts[n] = moved([line.spec[c] * only for c in cells])
            truth = line.in_time(line.spec[target] * line.weight)
            onset = line.onset[target]
            plain = None
            for rule in rules:
                predicted = line.in_time(
                    (parts * weights[rule][:, None, :]).sum(axis=0) * line.weight
                )
                if plain is None:
                    plain = predicted
                scored = scores(predicted - truth, truth, onset, order)
                for kind in kinds:
                    found[rule][kind].append(scored[kind])
                change = scores(predicted - plain, truth, onset, order)
                found[rule]["changed_early"].append(change["early"])
                found[rule]["changed_whole"].append(change["whole"])
        record: dict[str, Any] = {"cases": len(chosen), "rule": {}}
        for rule in rules:
            rows = found[rule]
            record["rule"][rule] = {
                "early": stats(np.stack(rows["early"])),
                "whole": stats(np.stack(rows["whole"])),
                "early_degree_worst_band": stats(np.stack(rows["early_degree"]).max(axis=1)),
                "whole_degree_worst_band": stats(np.stack(rows["whole_degree"]).max(axis=1)),
                "early_degree_worst_case": np.round(
                    np.stack(rows["early_degree"]).max(axis=0), 1
                ).tolist(),
                "whole_degree_worst_case": np.round(
                    np.stack(rows["whole_degree"]).max(axis=0), 1
                ).tolist(),
                "changed_early": stats(np.stack(rows["changed_early"])),
                "changed_whole": stats(np.stack(rows["changed_whole"])),
                "early_level": level_stats(np.stack(rows["early_level"])),
                "late_level": level_stats(np.stack(rows["late_level"])),
            }
        summary["geometries"][name] = record
        base = record["rule"][TODAY]
        say(f"{name}: {len(chosen)} cases, {time.time() - started:.0f} s")
        for rule in rules:
            row = record["rule"][rule]
            early, whole = max(row["early"]["worst"]), max(row["whole"]["worst"])
            say(
                f"  {rule:14s} early worst {early:6.1f} (today {max(base['early']['worst']):6.1f})"
                f" whole worst {whole:6.1f} (today {max(base['whole']['worst']):6.1f})"
                f" changed {max(row['changed_whole']['worst']):6.1f}"
            )
    summary["seconds"] = round(time.time() - started, 1)
    return summary


def measure_pitch(
    line: Line,
    *,
    pitches: tuple[int, ...] = (5, 6, 8, 9),
    classes_m: tuple[float, ...] = (0.6, 0.9, 1.3, 1.8, 2.5, 100.0),
    cases: int = 8,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The fusion of two cells a pitch apart, half way between them; by source distance.

    ``pitches`` are in nodes of the low grid (32.7 mm): 0.16, 0.20, 0.26
    and 0.29 m. The head is on the node half way, or on the two either
    side of half way, where it is furthest from both cells. Every pair of
    cells is read whatever the serving rule says, since the question is
    what the rule forbids, and the pairs the rule allows
    (:data:`reverberate.spatial.translate.SOURCE_SHARE`) are counted and
    read again alone. The classes are of the nearer cell's distance to
    the source.
    """
    started = time.time()
    order = line.order
    summary: dict[str, Any] = {
        "bands_hz": list(BANDS_HZ),
        "classes_m": list(classes_m),
        "pitches": {},
    }
    nodes = sorted(line.of_node)
    for pitch in pitches:
        rows: list[tuple[float, bool, np.ndarray, np.ndarray]] = []
        starts = [n for n in nodes if n + pitch in line.of_node]
        per_class: dict[int, list[int]] = {}
        for n in starts:
            near = min(line.to_source[line.of_node[n]], line.to_source[line.of_node[n + pitch]])
            per_class.setdefault(int(np.searchsorted(classes_m, near, side="right")), []).append(n)
        for members in per_class.values():
            for n in _thin(members, cases):
                a, b = line.of_node[n], line.of_node[n + pitch]
                near = float(min(line.to_source[a], line.to_source[b]))
                worst_early, worst_whole, allowed = None, None, True
                for inside in sorted({pitch // 2, (pitch + 1) // 2}):
                    if n + inside not in line.of_node:
                        continue
                    target = line.of_node[n + inside]
                    reach = max(
                        float(np.linalg.norm(line.low[target] - line.low[c])) for c in (a, b)
                    )
                    allowed = allowed and reach <= SOURCE_SHARE * near
                    predicted = line.in_time(
                        line.mover(target, [a, b])([line.spec[a], line.spec[b]]) * line.weight
                    )
                    truth = line.in_time(line.spec[target] * line.weight)
                    scored = scores(predicted - truth, truth, line.onset[target], order)
                    worst_early = (
                        scored["early"]
                        if worst_early is None
                        else np.maximum(worst_early, scored["early"])
                    )
                    worst_whole = (
                        scored["whole"]
                        if worst_whole is None
                        else np.maximum(worst_whole, scored["whole"])
                    )
                if worst_early is not None and worst_whole is not None:
                    rows.append((near, allowed, worst_early, worst_whole))
        record: dict[str, Any] = {"pitch_m": round(pitch * line.step_m, 3)}
        for lower, upper in zip(classes_m[:-1], classes_m[1:], strict=True):
            chosen = [row for row in rows if lower <= row[0] < upper]
            if not chosen:
                continue
            within = [row for row in chosen if row[1]]
            record[f"source {lower:g} to {upper:g} m"] = {
                "cell_pairs": len(chosen),
                "allowed_by_the_rule": len(within),
                "nearest_source_m": round(min(row[0] for row in chosen), 2),
                "early": stats(np.stack([row[2] for row in chosen])),
                "whole": stats(np.stack([row[3] for row in chosen])),
                **(
                    {"early_allowed": stats(np.stack([row[2] for row in within]))} if within else {}
                ),
            }
        summary["pitches"][str(pitch)] = record
        say(f"pitch {record['pitch_m']} m: {len(rows)} cell pairs, {time.time() - started:.0f} s")
        for name, value in record.items():
            if isinstance(value, dict):
                say(
                    f"  {name}: early worst {max(value['early']['worst']):.1f} p90"
                    f" {max(value['early']['p90']):.1f} ({value['cell_pairs']} pairs,"
                    f" {value['allowed_by_the_rule']} allowed)"
                )
    summary["seconds"] = round(time.time() - started, 1)
    return summary
