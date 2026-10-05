"""On the scene's own pairs: what each way of keeping a response in fewer bytes costs.

The pairs are read from a pair cache (:class:`reverberate.accel.pairs.PairCache`)
and put in the stored form a pack holds (:func:`reverberate.trace.level.pair_low`).
Each is then kept every way under trial, read back, and the difference
scored where a head hears it: the cell's expansion moved 0, 0.10 and 0.20 m
(:func:`reverberate.spatial.translate.translation_operator`) and read on
the head's sphere (:mod:`.scoring`). A pair is read at a distance only if
the serving rule lets the engine read it from there, 0.15 of the source's
distance.

:func:`measure_pairs` is the sample formats, the degrees against frequency
and the lengths; :func:`measure_rails` is what the positions of one rail
heard at one cell have in common.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.accel.pairs import PairCache
from reverberate.audio import Atmosphere
from reverberate.experiments.w46_compact_low.scoring import (
    BANDS_HZ,
    RATE_HZ,
    band_masks,
    decibels,
    head_weights,
    level_stats,
    scores,
    stats,
)
from reverberate.mirror.hybrid import Crossover
from reverberate.render.compact import Levers, decode, encode, first_hz
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M
from reverberate.spatial.sh import degrees_of
from reverberate.spatial.translate import SOUND_SPEED_M_S, SOURCE_SHARE, translation_operator
from reverberate.trace.level import pair_low

__all__ = ["DISTANCES_M", "Context", "find_pairs", "measure_pairs", "measure_rails", "variants"]

ORDER = 7
#: How far the head is from the cell when the error is read, in metres, along
#: one horizontal direction that is no axis of the harmonics.
DISTANCES_M = (0.0, 0.10, 0.20)
DIRECTION = np.array([0.6, 0.0, 0.8])
#: The window the noise a format leaves is set against the response's decay on.
LEVEL_S = 0.020
INT16 = 32767.0
#: The width of a block of bins that has its own scale, in the block scaled trial.
BLOCK_HZ = 100.0


@dataclass(frozen=True)
class Context:
    """What turns a cached pair into a pack's: the crossover, the air, the mirror's lead."""

    crossover: Crossover = field(default_factory=Crossover)
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    sound_speed_m_s: float = SOUND_SPEED_M_S
    lead_s: float = 512.0 / 48000.0

    @property
    def top_hz(self) -> float:
        return float(self.crossover.band_hz()[1])

    def stored(self, cached: np.ndarray) -> tuple[np.ndarray, int]:
        """``low/ir`` of a cached response, float64, and the sample its arrival falls on."""
        ir, onset, _ = pair_low(
            cached,
            self.crossover,
            self.atmosphere,
            sound_speed_m_s=self.sound_speed_m_s,
            lead_s=self.lead_s,
            unit_at_1m=FIELD_UNIT_AT_1M,
        )
        return np.asarray(ir, dtype=np.float64), round(onset * RATE_HZ)


def find_pairs(roots: list[Path], voxel_low_key: str) -> list[dict[str, Any]]:
    """Every pair that is in a cache under ``roots``, once, with its file and its distance."""
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        cache = PairCache(Path(root), voxel_low_key)
        for key, record in cache.records().items():
            if key in found or not cache.has(key):
                continue
            centre = np.asarray(record.get("centre_m", record["cell_m"]), dtype=float)
            source = np.asarray(record["source_m"], dtype=float)
            found[key] = {
                **record,
                "path": str(cache.path(key)),
                "distance_m": float(np.linalg.norm(source - centre)),
            }
    return sorted(found.values(), key=lambda record: (record["distance_m"], record["key"]))


def _spread(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """``count`` of the pairs, evenly over their order of distance: both ends are in."""
    if len(records) <= count:
        return records
    return [
        records[i] for i in np.unique(np.linspace(0, len(records) - 1, count).round().astype(int))
    ]


def _fixed(values: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """``values`` (``[channel, ...]``) as int16 and back, one scale a group of channels."""
    out = np.zeros_like(values)
    labels = np.zeros(values.shape[0], dtype=int) if groups is None else groups
    for label in np.unique(labels):
        rows = labels == label
        scale = float(np.abs(values[rows]).max()) / INT16
        if scale > 0.0:
            out[rows] = np.rint(values[rows] / scale) * scale
    return out


def variants(context: Context) -> dict[str, Callable[[np.ndarray], tuple[np.ndarray, int]]]:
    """Every way a response is kept here: its name, and what gives it back with its bytes."""
    degree = degrees_of(ORDER)
    channels = degree.size
    top = context.top_hz
    made: dict[str, Callable[[np.ndarray], tuple[np.ndarray, int]]] = {}

    def in_time(kind: str) -> Callable[[np.ndarray], tuple[np.ndarray, int]]:
        def keep(x: np.ndarray) -> tuple[np.ndarray, int]:
            if kind == "float16":
                # Under 6e-8 a float16 is nothing: the end of a tail is, which is the trial.
                with np.errstate(under="ignore"):
                    return x.astype(np.float16).astype(np.float64), 2 * x.size
            groups = {"pair": None, "channel": np.arange(channels), "degree": degree}[kind]
            return _fixed(x, groups), 2 * x.size

        return keep

    def in_bins(kind: str) -> Callable[[np.ndarray], tuple[np.ndarray, int]]:
        def keep(x: np.ndarray) -> tuple[np.ndarray, int]:
            samples = x.shape[-1]
            kept = int(np.floor(top * samples / RATE_HZ)) + 1
            spectrum = np.fft.rfft(x, axis=-1)[:, :kept]
            parts = np.stack([spectrum.real, spectrum.imag], axis=-1)
            if kind == "float32":
                parts = parts.astype(np.float32).astype(np.float64)
            elif kind == "float16":
                parts = parts.astype(np.float16).astype(np.float64)
            elif kind == "block":
                # One scale a channel and a block of BLOCK_HZ: the bins are [channel, bin, 2].
                width = max(1, round(BLOCK_HZ * samples / RATE_HZ))
                for start in range(0, kept, width):
                    parts[:, start : start + width] = _fixed(
                        parts[:, start : start + width], np.arange(channels)
                    )
            else:
                groups = {"pair": None, "channel": np.arange(channels), "degree": degree}[kind]
                parts = _fixed(parts, groups)
            full = np.zeros((channels, samples // 2 + 1), dtype=np.complex128)
            full[:, :kept] = parts[..., 0] + 1j * parts[..., 1]
            size = 4 if kind == "float32" else 2
            return np.fft.irfft(full, n=samples, axis=-1), size * parts.size

        return keep

    def with_levers(text: str) -> Callable[[np.ndarray], tuple[np.ndarray, int]]:
        levers = Levers.parse(text)
        starts = first_hz(ORDER, levers.degree_db, context.sound_speed_m_s)

        def keep(x: np.ndarray) -> tuple[np.ndarray, int]:
            kept = encode(
                x, levers, rate_hz=RATE_HZ, top_hz=top, sound_speed_m_s=context.sound_speed_m_s
            )
            back = decode(
                kept, first_hz=starts, top_hz=top, rate_hz=RATE_HZ, samples=x.shape[-1]
            ).astype(np.float64)
            return back, int(kept.data.nbytes) + (
                kept.scale.nbytes if levers.sample == "int16" else 0
            )

        return keep

    def cut_at(seconds: float) -> Callable[[np.ndarray], tuple[np.ndarray, int]]:
        def keep(x: np.ndarray) -> tuple[np.ndarray, int]:
            stop, fade = round(seconds * RATE_HZ), round(0.020 * RATE_HZ)
            out = x.copy()
            out[:, stop:] = 0.0
            out[:, stop - fade : stop] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(fade) + 1.0) / fade)
            return out, 4 * channels * stop

        return keep

    made["time float32, 4 kHz (today)"] = lambda x: (
        x.astype(np.float32).astype(np.float64),
        4 * x.size,
    )
    made["time float16"] = in_time("float16")
    for kind in ("pair", "channel", "degree"):
        made[f"time int16, a scale a {kind}"] = in_time(kind)
    made["bins float32"] = in_bins("float32")
    made["bins float16"] = in_bins("float16")
    for kind in ("pair", "channel", "degree"):
        made[f"bins int16, a scale a {kind}"] = in_bins(kind)
    made["bins int16, a scale a channel and 100 Hz"] = in_bins("block")
    for floor in (60, 50, 40, 30, 20):
        made[f"degree {floor}"] = with_levers(f"bins,degree={floor}")
    made["cut at 0.8 s"] = cut_at(0.8)
    for level in (50, 60, 70):
        made[f"decay {level}"] = with_levers(f"bins,decay={level}")
        for floor in (40, 50):
            made[f"degree {floor}, decay {level}"] = with_levers(
                f"bins,degree={floor},decay={level}"
            )
    for text in ("decay=60", "degree=40,decay=60", "degree=50,decay=60", "degree=50,decay=70"):
        made[f"bins int16, {text.replace('=', ' ').replace(',', ', ')}"] = with_levers(
            f"bins,int16,{text}"
        )
    return made


def _level(signal: np.ndarray, span: int) -> np.ndarray:
    """The mean square of all channels over ``span`` samples, at every sample."""
    power = (signal**2).sum(axis=0)
    summed = np.cumsum(np.pad(power, (span, 0)))
    found: np.ndarray = (summed[span:] - summed[:-span]) / span
    return found


def measure_pairs(
    roots: list[Path],
    voxel_low_key: str,
    *,
    count: int = 120,
    context: Context | None = None,
    only: list[str] | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """``count`` pairs kept every way of :func:`variants`; the summary.

    For each way: the bytes of a pair, the error at each of
    :data:`DISTANCES_M` per third octave (early and whole, all channels and
    per degree), the level it adds or takes, and how far under the
    response's own decay its noise lies.
    """
    started = time.time()
    context = context or Context()
    records = find_pairs(roots, voxel_low_key)
    chosen = _spread(records, count)
    say(
        f"{len(records)} pairs found, {len(chosen)} read, {chosen[0]['distance_m']:.2f} to"
        f" {chosen[-1]['distance_m']:.2f} m"
    )
    samples = 4800
    freqs = np.fft.rfftfreq(samples, 1.0 / RATE_HZ)
    kept = int(np.floor(context.top_hz * samples / RATE_HZ)) + 1
    weight = head_weights(ORDER, freqs[:kept])
    unit = DIRECTION / np.linalg.norm(DIRECTION)
    moves = [
        None
        if d == 0.0
        else translation_operator(
            d * unit, freqs[:kept], ORDER, sound_speed_m_s=context.sound_speed_m_s
        )
        for d in DISTANCES_M
    ]
    ways = variants(context)
    if only is not None:
        ways = {name: way for name, way in ways.items() if name in only}
    degree = degrees_of(ORDER)
    masks = band_masks(samples).astype(float)
    span = round(LEVEL_S * RATE_HZ)

    def at_head(signal: np.ndarray) -> list[np.ndarray]:
        spectrum = np.fft.rfft(signal, axis=-1)[:, :kept]
        heard = []
        for move in moves:
            moved = spectrum if move is None else np.einsum("fab,bf->af", move, spectrum)
            full = np.zeros((degree.size, samples // 2 + 1), dtype=np.complex128)
            full[:, :kept] = moved * weight
            heard.append(np.fft.irfft(full, n=samples, axis=-1))
        return heard

    kinds = ("early", "whole", "early_degree", "whole_degree", "early_level", "late_level")
    gathered: dict[str, Any] = {
        name: {
            "bytes": [],
            "margin": [],
            "floor": [],
            **{f"{kind}@{d:.2f}": [] for kind in kinds for d in DISTANCES_M},
        }
        for name in ways
    }
    spectrum_of_degrees: dict[str, list[np.ndarray]] = {"whole": [], "late": []}
    decays: list[np.ndarray] = []
    depth: list[float] = []
    slices = ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2))
    for index, record in enumerate(chosen):
        x, onset = context.stored(np.load(record["path"]))
        truth = at_head(x)
        loud = _level(truth[0], span)
        end = np.mean(loud[-round(0.1 * RATE_HZ) :])
        depth.append(float(decibels(loud.max() / end)))
        # A degree's channels against channel 0, as they are stored: N3D, unweighted.
        for name, start in (("whole", 0), ("late", round(0.4 * RATE_HZ))):
            energy = (np.abs(np.fft.rfft(x[:, start:], axis=-1)) ** 2) @ band_masks(
                samples - start
            ).T.astype(float)
            per = np.stack([energy[degree == n].mean(axis=0) for n in range(ORDER + 1)], axis=1)
            spectrum_of_degrees[name].append(decibels(per / np.maximum(energy[0][:, None], 1e-300)))
        # What is left of the response, band by band, in the fifths of a second after 0.4 s.
        whole = (np.abs(np.fft.rfft(truth[0], axis=-1)) ** 2).sum(axis=0) @ masks.T
        rows = []
        for start_s, stop_s in slices:
            piece = np.zeros_like(truth[0])
            a, b = round(start_s * RATE_HZ), round(stop_s * RATE_HZ)
            piece[:, a:b] = truth[0][:, a:b]
            rows.append(
                decibels(((np.abs(np.fft.rfft(piece, axis=-1)) ** 2).sum(axis=0) @ masks.T) / whole)
            )
        decays.append(np.stack(rows))
        for name, way in ways.items():
            back, size = way(x)
            wrong = at_head(back - x)
            into = gathered[name]
            into["bytes"].append(size)
            noise = _level(wrong[0], span)
            heard = slice(max(0, onset - span), samples)
            with np.errstate(divide="ignore"):
                into["margin"].append(
                    float(decibels(loud[heard] / np.maximum(noise[heard], 1e-300)).min())
                )
                into["floor"].append(float(decibels(loud.max() / max(float(noise.max()), 1e-300))))
            for d, true, error in zip(DISTANCES_M, truth, wrong, strict=True):
                if d > SOURCE_SHARE * record["distance_m"]:
                    continue  # the serving rule never reads this cell from so far
                found = scores(error, true, onset, ORDER)
                for kind in kinds:
                    into[f"{kind}@{d:.2f}"].append(found[kind])
        if (index + 1) % 20 == 0:
            say(f"{index + 1} pairs, {time.time() - started:.0f} s")
    summary: dict[str, Any] = {
        "pairs": len(chosen),
        "pairs_found": len(records),
        "distance_m": [round(chosen[0]["distance_m"], 2), round(chosen[-1]["distance_m"], 2)],
        "bands_hz": list(BANDS_HZ),
        "distances_m": list(DISTANCES_M),
        "top_hz": context.top_hz,
        "response_decay_db": stats(np.array(depth)[:, None], low_is_bad=True),
        "left_after_s": {
            f"{a:.1f}-{b:.1f}": stats(np.stack(decays)[:, i]) for i, (a, b) in enumerate(slices)
        },
        "degree_over_channel_0_db": {
            name: {
                "columns": [f"degree {n}" for n in range(ORDER + 1)],
                "median": np.round(np.median(np.stack(rows), axis=0), 1).tolist(),
                "p90": np.round(np.percentile(np.stack(rows), 90, axis=0), 1).tolist(),
            }
            for name, rows in spectrum_of_degrees.items()
        },
        "ways": {},
    }
    today = 4 * degree.size * samples
    for name, into in gathered.items():
        sizes = np.array(into["bytes"], dtype=float)
        entry: dict[str, Any] = {
            "bytes_mean": round(float(sizes.mean())),
            "bytes_worst": int(sizes.max()),
            "factor": round(today / float(sizes.mean()), 2),
            "margin_under_decay_db": stats(np.array(into["margin"])[:, None], low_is_bad=True),
            "noise_under_peak_db": stats(np.array(into["floor"])[:, None], low_is_bad=True),
        }
        for d in DISTANCES_M:
            at = f"{d:.2f}"
            if not into[f"early@{at}"]:
                continue
            entry[at] = {
                "pairs": len(into[f"early@{at}"]),
                "early": stats(np.stack(into[f"early@{at}"])),
                "whole": stats(np.stack(into[f"whole@{at}"])),
                "early_degree_worst_band": stats(np.stack(into[f"early_degree@{at}"]).max(axis=1)),
                "whole_degree_worst_band": stats(np.stack(into[f"whole_degree@{at}"]).max(axis=1)),
                "early_level": level_stats(np.stack(into[f"early_level@{at}"])),
                "late_level": level_stats(np.stack(into[f"late_level@{at}"])),
            }
        summary["ways"][name] = entry
    summary["seconds"] = round(time.time() - started, 1)
    return summary


def _chains(positions: np.ndarray, low_m: float = 0.05, high_m: float = 0.11) -> list[list[int]]:
    """Runs of positions each one rail sample from the next, longest first."""
    left = set(range(positions.shape[0]))
    chains = []
    while left:
        chain = [min(left)]
        left.discard(chain[0])
        for grow_at_end in (True, False):
            while True:
                tip = positions[chain[-1] if grow_at_end else chain[0]]
                near = [
                    i for i in left if low_m <= float(np.linalg.norm(positions[i] - tip)) <= high_m
                ]
                if not near:
                    break
                step = min(near, key=lambda i: float(np.linalg.norm(positions[i] - tip)))
                left.discard(step)
                chain = [*chain, step] if grow_at_end else [step, *chain]
        chains.append(chain)
    return sorted(chains, key=len, reverse=True)


def measure_rails(
    roots: list[Path],
    voxel_low_key: str,
    *,
    rails: int = 6,
    positions: int = 12,
    context: Context | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """What the positions of one rail heard at one cell share: the rank a rail needs.

    The longest runs of source positions a rail sample apart that one cell
    hears, ``positions`` of each. Their responses are stacked and reduced
    to their first singular vectors, as they are and after each is put on a
    common arrival time; the energy the first ``r`` hold, and the error per
    position at a quarter of the rank, which is the 4 times a low rank code
    would save.
    """
    started = time.time()
    context = context or Context()
    by_cell: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for pair in find_pairs(roots, voxel_low_key):
        by_cell.setdefault(tuple(np.round(pair["cell_m"], 3)), []).append(pair)
    runs = []
    for records in by_cell.values():
        if len(records) < positions:
            continue
        where = np.array([pair["source_m"] for pair in records], dtype=float)
        for chain in _chains(where):
            if len(chain) >= positions:
                runs.append([records[i] for i in chain[:positions]])
    runs.sort(key=lambda run: run[0]["distance_m"])
    chosen = (
        [runs[i] for i in np.unique(np.linspace(0, len(runs) - 1, rails).round().astype(int))]
        if runs
        else []
    )
    say(f"{len(runs)} runs of {positions} positions at one cell; {len(chosen)} read")
    samples = 4800
    freqs = np.fft.rfftfreq(samples, 1.0 / RATE_HZ)
    weight = head_weights(ORDER, freqs)
    summary: dict[str, Any] = {"rails": [], "positions": positions, "bands_hz": list(BANDS_HZ)}
    masks = band_masks(samples).astype(float)
    for run in chosen:
        stack = np.stack([context.stored(np.load(pair["path"]))[0] for pair in run])
        distance = np.array([pair["distance_m"] for pair in run])
        record: dict[str, Any] = {
            "cell_m": run[0]["cell_m"],
            "source_distance_m": [round(float(distance.min()), 2), round(float(distance.max()), 2)],
        }
        for name in ("as stored", "on one arrival time"):
            spectra = np.fft.rfft(stack, axis=-1)
            if name != "as stored":
                shift = (distance - distance.mean()) / context.sound_speed_m_s
                spectra = spectra * np.exp(2j * np.pi * freqs[None, None, :] * shift[:, None, None])
            # On the head's sphere, so that what is counted is what is heard.
            heard = np.fft.irfft(spectra * weight[None], n=samples, axis=-1).reshape(positions, -1)
            u, s, vt = np.linalg.svd(heard, full_matrices=False)
            energy = np.cumsum(s**2) / np.sum(s**2)
            rank = positions // 4
            low = (u[:, :rank] * s[:rank]) @ vt[:rank]
            wrong = (heard - low).reshape(stack.shape)
            true = heard.reshape(stack.shape)
            per_band = np.stack(
                [
                    decibels(
                        ((np.abs(np.fft.rfft(wrong[p], axis=-1)) ** 2).sum(axis=0) @ masks.T)
                        / ((np.abs(np.fft.rfft(true[p], axis=-1)) ** 2).sum(axis=0) @ masks.T)
                    )
                    for p in range(positions)
                ]
            )
            record[name] = {
                "left_after_rank_db": {
                    str(r): round(float(decibels(1.0 - energy[r - 1])), 1)
                    for r in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20)
                    if r < positions
                },
                "rank_for_db": {
                    str(level): int(np.argmax(decibels(1.0 - energy + 1e-300) <= level) + 1)
                    for level in (-20, -30, -40)
                },
                "at_a_quarter": {"rank": rank, **stats(per_band)},
            }
        summary["rails"].append(record)
        say(
            f"cell {record['cell_m']}: rank for -30 dB {record['as stored']['rank_for_db']['-30']}"
            f" of {positions}, aligned {record['on one arrival time']['rank_for_db']['-30']}"
        )
    summary["seconds"] = round(time.time() - started, 1)
    return summary
