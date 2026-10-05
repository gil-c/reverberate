"""On the dense line: how far apart a moving source's solved positions may be.

By reciprocity the pressure a fixed source leaves along a line is the
pressure a source moving along that line leaves at a fixed omnidirectional
listener, so the line solved every 2 cm is a rail solved every 2 cm. Some of
its points are taken as the rail's solved positions, a pitch apart, and the
others are predicted from them and set against their own solve. Channel 0
alone: the other channels turn with the listener and are not a reciprocal
test.

**The interpolators.** ``linear`` is the pack's first rule, two positions.
``lagrange`` is the polynomial through 4 or 6. ``sinc`` is a sinc at the
pitch under a raised cosine over 4, 6 or 8. ``band`` is
:func:`reverberate.spatial.rail.band_limited_weights` over 4, 6 or 8, whose
weights depend on the frequency, held every
:data:`~reverberate.spatial.rail.KNOT_HZ` as a pack holds them; ``flat`` is
the same made once for the top of the crossover's ramp, one weight a
position.

**The centres are the arrays' own** (``plan.json``), as in
:mod:`.line_channels`: under :data:`SEAM_HZ` a three band field takes its
low solve, whose arrays stand on a 32.7 mm grid, 209 distinct centres for
341 points, so the pitches there are multiples of 32.7 mm; over it the mid
solve's, 2 cm apart to within 4 mm, and the pitches are multiples of 2 cm.
The third octaves from 100 to 630 Hz are read from the first and those from
1 kHz from the second. The 800 Hz one lies across the seam and is read from
the second: its figures carry the seam and are not the interpolation's.

**A rail has ends.** The whole line is one rail 6.8 m long; a ``short`` rail
is 1.2 m of it, the length of most rails of a recipe, sampled as
:func:`reverberate.scenes.rail_samples` does: every pitch from one end, then
the other end itself. A target in the first or last gap of a short rail is
an ``end`` case, with positions on one side only past the two round it.

The error is the energy of the difference over the energy of the solved
response per third octave, on the 50 ms after the onset and on the whole
response; ``masked`` is the same with the crossover's low mask on the error
alone, which is the error over what is heard in that band once the mirror
has given the rest. The level error is the energy of the prediction over the
truth's. Median, ninth decile and worst case over the targets; ``near`` are
the targets less than :data:`NEAR_M` from the source.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.w44_interpolation.scoring import EARLY_S, FADE_S, write_summary
from reverberate.mirror.hybrid import Crossover
from reverberate.spatial.rail import (
    KNOT_HZ,
    band_limited_weights,
    knots_hz,
    lagrange_weights,
    nearest_samples,
)

__all__ = [
    "METHODS",
    "NEAR_M",
    "SEAM_HZ",
    "THIRDS",
    "Regime",
    "line_regimes",
    "predict",
    "rail_interpolation",
    "score",
    "third_octave_masks",
]

#: Third octave centres the errors are read in, in Hz.
THIRDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250]
#: The spectra are kept up to here, and brought back to time at ``RATE_HZ``.
FMAX_HZ = 1500.0
RATE_HZ = 4000.0
#: Under this a three band field holds its low solve, over it its mid solve.
SEAM_HZ = 800.0
#: A target nearer the source than this is in its near field.
NEAR_M = 1.2
#: A short rail, in metres.
SHORT_M = 1.2
#: The frequency the ``flat`` weights are made for: the top of the crossover's ramp.
FLAT_HZ = 1414.0
#: Each interpolator and the positions it reads.
METHODS: tuple[tuple[str, int], ...] = (
    ("linear", 2),
    ("lagrange", 4),
    ("lagrange", 6),
    ("sinc", 4),
    ("sinc", 6),
    ("sinc", 8),
    ("flat", 4),
    ("flat", 6),
    ("flat", 8),
    ("band", 4),
    ("band", 6),
    ("band", 8),
)


@dataclass(frozen=True)
class Regime:
    """The line as one solve holds it: distinct centres, their spectra, the bands read there."""

    name: str
    #: ``[node]``: metres along the line, ascending.
    along: np.ndarray
    #: ``[node, 3]``: the centres, scene frame.
    centres: np.ndarray
    #: ``[node, bin]``: the pressure's spectrum up to :data:`FMAX_HZ`.
    spectra: np.ndarray
    #: ``[node]``: metres to the source.
    to_source: np.ndarray
    #: Indices into :data:`THIRDS` of the bands this solve is the truth of.
    bands: tuple[int, ...]
    #: The pitches measured, in nodes.
    pitches: tuple[int, ...]

    @property
    def node_m(self) -> float:
        return float((self.along[-1] - self.along[0]) / (self.along.size - 1))


def third_octave_masks(freqs: np.ndarray) -> np.ndarray:
    """``[band, bin]``: which bins of ``freqs`` each of :data:`THIRDS` holds."""
    edge = 2.0 ** (1.0 / 6.0)
    return np.array([(freqs >= b / edge) & (freqs < b * edge) for b in THIRDS])


def line_regimes(field: Path, plan: Path | None) -> tuple[list[Regime], np.ndarray, int]:
    """The line under and over the seam, the kept frequencies and the low rate's sample count."""
    with h5py.File(field, "r") as f:
        rate = float(f.attrs["sample_rate_hz"])
        positions = f["positions"][:]
        source = np.asarray(f.attrs["source_position"], dtype=float)
        points, _, samples = f["ir"].shape
        freqs = np.fft.rfftfreq(samples, 1.0 / rate)
        keep = freqs <= FMAX_HZ
        spectra = np.zeros((points, int(keep.sum())), dtype=np.complex128)
        for i in range(points):
            spectra[i] = np.fft.rfft(f["ir"][i, 0, :])[keep]
    low_centres = mid_centres = positions
    if plan is not None:
        placed = json.loads(Path(plan).read_text())["bands"]
        low_centres = np.asarray(placed["low"]["centres"], dtype=float)
        mid_centres = np.asarray(placed["mid"]["centres"], dtype=float)
    regimes = []
    for name, centres, bands, pitches in (
        ("low", low_centres, tuple(range(9)), (2, 3, 4, 5, 6, 7, 8, 10)),
        ("mid", mid_centres, (9, 10, 11), (2, 4, 5, 6, 7, 8, 10, 12, 16)),
    ):
        along = np.linalg.norm(centres - centres[0], axis=1)
        # One point per centre: the first the field puts on it.
        _, first = np.unique(np.round(along, 4), return_index=True)
        regimes.append(
            Regime(
                name=name,
                along=along[first],
                centres=centres[first],
                spectra=spectra[first],
                to_source=np.linalg.norm(centres[first] - source[None, :], axis=1),
                bands=bands,
                pitches=pitches,
            )
        )
    return regimes, freqs[keep], round(samples * RATE_HZ / rate)


def predict(
    method: str,
    count: int,
    arcs: np.ndarray,
    nodes: np.ndarray,
    at: np.ndarray,
    target: np.ndarray,
    freqs: np.ndarray,
    pitch_m: float,
    **band: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Which samples each target reads and their weights: ``[target, slot]``, ``[.., slot, bin]``.

    ``arcs`` are the samples' arc lengths along the rail and ``nodes`` their
    positions; ``at`` and ``target`` are the same of the targets.
    """
    chosen = nearest_samples(arcs, at, count)
    held = chosen >= 0
    safe = np.maximum(chosen, 0)
    offsets = arcs[safe] - at[:, None]
    if method == "linear" or method == "lagrange":
        flat = lagrange_weights(offsets, held)
    elif method == "sinc":
        reach = 0.5 * count * pitch_m
        window = np.where(np.abs(offsets) < reach, np.cos(0.5 * np.pi * offsets / reach) ** 2, 0.0)
        flat = np.where(held, np.sinc(offsets / pitch_m) * window, 0.0)
        flat = flat / flat.sum(axis=1, keepdims=True)
    elif method == "flat":
        flat = band_limited_weights(nodes[safe], target, np.array([FLAT_HZ]), valid=held, **band)[
            ..., 0
        ]
    elif method == "band":
        knots = knots_hz(float(freqs[-1]), KNOT_HZ)
        made = band_limited_weights(nodes[safe], target, knots, valid=held, **band)
        index = np.minimum((freqs / KNOT_HZ).astype(int), knots.size - 2)
        share = freqs / KNOT_HZ - index
        return chosen, made[..., index] * (1.0 - share) + made[..., index + 1] * share
    else:
        raise ValueError(f"no interpolator is called {method!r}")
    return chosen, np.repeat(flat[..., None], freqs.size, axis=-1)


def score(
    predicted: np.ndarray, truth: np.ndarray, freqs: np.ndarray, low_samples: int
) -> dict[str, np.ndarray]:
    """Per target and third octave, early and whole: the error, the same masked, the level.

    ``predicted`` and ``truth`` are ``[target, bin]`` on ``freqs``. Each
    entry is ``[target, band]`` in decibels.
    """
    cross = Crossover()
    bins = np.fft.rfftfreq(low_samples, 1.0 / RATE_HZ)
    bands = third_octave_masks(bins)
    fade = int(FADE_S * RATE_HZ)

    def in_time(spectrum: np.ndarray) -> np.ndarray:
        full = np.zeros((spectrum.shape[0], bins.size), dtype=np.complex128)
        full[:, : freqs.size] = spectrum
        signal: np.ndarray = np.fft.irfft(full, n=low_samples, axis=-1)
        return signal

    true = in_time(truth)
    wrong = in_time(predicted) - true
    onset = np.argmax(np.abs(true) > 0.1 * np.abs(true).max(axis=1, keepdims=True), axis=1)
    early = np.zeros_like(true)
    for case, start in enumerate(onset):
        stop = min(low_samples, int(start) + int(EARLY_S * RATE_HZ))
        early[case, :stop] = 1.0
        early[case, stop - fade : stop] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade) / fade))
    found: dict[str, np.ndarray] = {}
    for name, window, power in (("early", early, False), ("whole", 1.0, True)):
        mask = cross.masks(low_samples, RATE_HZ, power=power)[0]
        true_e = np.abs(np.fft.rfft(true * window, axis=-1)) ** 2
        wrong_e = np.abs(np.fft.rfft(wrong * window, axis=-1)) ** 2
        made_e = np.abs(np.fft.rfft((true + wrong) * window, axis=-1)) ** 2
        ref = true_e @ bands.T
        with np.errstate(divide="ignore"):
            found[f"{name}_error"] = 10.0 * np.log10(wrong_e @ bands.T / ref)
            found[f"{name}_masked"] = 10.0 * np.log10((wrong_e * mask**2) @ bands.T / ref)
            found[f"{name}_level"] = 10.0 * np.log10(made_e @ bands.T / ref)
    return found


def _rails(nodes: int, pitch: int, short: int) -> list[tuple[str, np.ndarray, int, int]]:
    """Rails cut from the line: the name of the kind, its samples' nodes, its first and last."""
    out = []
    for start in range(pitch):  # the whole line, at every phase of the pitch
        samples = np.arange(start, nodes, pitch)
        out.append(("line", samples, int(samples[0]), int(samples[-1])))
    for start in range(0, nodes - short, max(short // 4, 1)):
        samples = np.arange(start, start + short, pitch)
        if samples[-1] != start + short:
            samples = np.append(samples, start + short)
        out.append(("short", samples, start, start + short))
    return out


def _stats(values: np.ndarray, bands: tuple[int, ...], *, level: bool = False) -> dict[str, Any]:
    kept = np.abs(values[:, bands]) if level else values[:, bands]
    return {
        "median": np.round(np.median(kept, axis=0), 1).tolist(),
        "p90": np.round(np.percentile(kept, 90, axis=0), 1).tolist(),
        "worst": np.round(kept.max(axis=0), 1).tolist(),
    }


def rail_interpolation(
    field: Path,
    out: Path,
    *,
    plan: Path | None = None,
    methods: tuple[tuple[str, int], ...] = METHODS,
    pitches: tuple[int, ...] | None = None,
    say: Callable[[str], None] = print,
    **band: float,
) -> dict[str, Any]:
    """Every interpolator at every pitch of the line; ``summary_rail.json``, returned.

    ``pitches`` are in points of the line and replace each regime's own;
    ``band`` are the settings of the band limited weights, the library's own
    when none is given.
    """
    started = time.time()
    regimes, freqs, low_samples = line_regimes(field, plan)
    summary: dict[str, Any] = {
        "field": str(field),
        "plan": None if plan is None else str(plan),
        "thirds_hz": THIRDS,
        "near_m": NEAR_M,
        "short_rail_m": SHORT_M,
        "band_settings": band,
        "regimes": {},
    }
    for regime in regimes:
        step = regime.node_m
        short = round(SHORT_M / step)
        record: dict[str, Any] = {
            "nodes": int(regime.along.size),
            "node_m": round(step, 5),
            "bands_hz": [THIRDS[b] for b in regime.bands],
            "pitches": {},
        }
        for pitch in regime.pitches if pitches is None else pitches:
            rows: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
            for kind, samples, first, last in _rails(regime.along.size, pitch, short):
                targets = np.setdiff1d(np.arange(first, last + 1), samples)
                if targets.size == 0:
                    continue
                # Where a target stands among its rail's samples.
                before = np.searchsorted(samples, targets)
                after = samples.size - before
                if kind == "line":
                    targets = targets[(before >= 4) & (after >= 4)]
                    place = np.full(targets.size, "interior")
                else:
                    place = np.where((before == 1) | (after == 1), "end", "inner")
                if targets.size == 0:
                    continue
                reach = np.where(regime.to_source[targets] < NEAR_M, "near", "far")
                truth = regime.spectra[targets]
                for method, count in methods:
                    chosen, weights = predict(
                        method,
                        count,
                        regime.along[samples],
                        regime.centres[samples],
                        regime.along[targets],
                        regime.centres[targets],
                        freqs,
                        pitch * step,
                        **band,
                    )
                    read = regime.spectra[samples[np.maximum(chosen, 0)]]
                    scored = score(
                        np.einsum("tsf,tsf->tf", weights, read), truth, freqs, low_samples
                    )
                    for group in sorted(set(zip(place, reach, strict=True))):
                        mine = (place == group[0]) & (reach == group[1])
                        rows.setdefault((f"{method}{count}", f"{group[0]}_{group[1]}"), []).append(
                            {name: values[mine] for name, values in scored.items()}
                        )
            at_pitch: dict[str, Any] = {}
            for (method_name, group_name), parts in sorted(rows.items()):
                entry = at_pitch.setdefault(method_name, {}).setdefault(group_name, {})
                entry["cases"] = int(sum(part["early_error"].shape[0] for part in parts))
                for name in parts[0]:
                    values = np.concatenate([part[name] for part in parts])
                    entry[name] = _stats(values, regime.bands, level=name.endswith("level"))
            record["pitches"][f"{pitch * step:.3f}"] = at_pitch
            say(f"{regime.name} {pitch * step:.3f} m done at {time.time() - started:.0f} s")
        summary["regimes"][regime.name] = record
    summary["seconds"] = round(time.time() - started, 1)
    write_summary(out, "summary_rail.json", summary)
    return summary
