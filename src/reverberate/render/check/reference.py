"""A pack against the validated field it should be: one source on the field's, the head on a point.

The sound check compares a scene with the validated field of its dwelling
at the nearest lattice point, whatever source the scene has: a room's
colour, read in bands. That leaves a difference without an owner, the
source's place or the pack. This settles it: a recipe whose one source
stands on the field's own source, omnidirectional, and whose listener rests
on a lattice point of the field (V1 of the plan). There the pack rendered at
rest and the field's response are the same thing made twice, and are held
to each other sample for sample where both are waves or paths, and in
energy where they are noise:

- the image paths, from 5 ms before the first arrival to 1 ms before the
  tail starts (``tail_from_s`` after it), a third octave at a time: the
  level of the pack over the field's, and the energy of their difference
  over the field's;
- the first 50 ms, which hold the tail's first 40 and are two draws of a
  noise: the level alone;
- the band under the crossover over the whole response, level and
  difference;
- the part after 50 ms per octave: its level, and T20.

What the difference can be at best: the head and the source stand where
the recipe's millimetres put them (0.2 mm from a lattice point), and the
low band's array on the node of its own grid, up to 19 mm from the one the
field's stood on, which is 20 degrees of phase at 1 kHz and an error of
-9 dB there.

Both are on the field's clock and scale: the pack's lead is the mirror's,
and a pack is physical, so it is multiplied by
:data:`reverberate.spatial.lowband.FIELD_UNIT_AT_1M`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.metrics import band_centres
from reverberate.render.check import measure
from reverberate.render.dry import DryTrack
from reverberate.render.engine import Engine, RenderSettings
from reverberate.render.pack import ScenePack, Source
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M

__all__ = ["ON_THE_POINT_M", "compare", "markdown", "rest_steps", "run"]

#: The source and the head must stand this near the field's source and a lattice point, m.
ON_THE_POINT_M = 0.05
#: The image paths are read from this before the field's first arrival, and to this
#: before the tail starts; the early part ends this after the first arrival, s.
BEFORE_S, MARGIN_S, EARLY_S = 0.005, 0.001, 0.050
#: Seconds of response compared: a field's window.
WINDOW_S = 1.2


def rest_steps(pack: ScenePack, source: Source, positions: np.ndarray) -> dict[int, int]:
    """Per lattice point the head rests on: the first step a whole response can be read from.

    A step whose source is audible for the next :data:`WINDOW_S`, with the
    head and the source still, the head within :data:`ON_THE_POINT_M` of a
    point of ``positions``.
    """
    h = pack.header
    need = int(np.ceil(WINDOW_S / h.step_s)) + 2
    heard = np.asarray(source.audible, dtype=bool)
    head = np.asarray(pack.listener.position, dtype=float)
    mouth = np.asarray(source.position, dtype=float)
    found: dict[int, int] = {}
    for k in range(max(heard.size - need, 0)):
        span = slice(k, k + need)
        if not heard[span].all():
            continue
        if np.ptp(head[span], axis=0).max() > 1e-6 or np.ptp(mouth[span], axis=0).max() > 1e-6:
            continue
        gaps = np.linalg.norm(positions - head[k][None, :], axis=1)
        point = int(np.argmin(gaps))
        if float(gaps[point]) <= ON_THE_POINT_M:
            found.setdefault(point, k)
    return found


def _bands_db(signal: np.ndarray, rate: float) -> np.ndarray:
    """:func:`measure.third_octave_db`, NaN where the signal is too short for a bin of the band."""
    levels = measure.third_octave_db(signal, rate)
    centres = np.asarray(measure.THIRD_OCTAVES_HZ)
    freqs = np.fft.rfftfreq(np.asarray(signal).size, 1.0 / rate)
    lowest = np.searchsorted(freqs, centres * 2.0 ** (-1.0 / 6.0), side="left")
    highest = np.searchsorted(freqs, centres * 2.0 ** (1.0 / 6.0), side="left")
    return np.where(highest > lowest, levels, np.nan)


def _point(
    pack: ScenePack,
    source: Source,
    step: int,
    field: np.ndarray,
    rate: float,
    workers: int,
) -> dict[str, Any]:
    """The table of one lattice point: ``field`` is the field's W response there."""
    h = pack.header
    sample = step * h.step_samples
    count = min(int(round(WINDOW_S * rate)), field.size, h.samples - sample)
    dry = DryTrack.from_array(np.array([1.0]), start_s=sample / rate, rate=rate)
    engine = Engine(
        pack, {source.id: dry}, settings=RenderSettings(directivity=False, workers=workers)
    )
    parts = {
        name: np.asarray(engine.stem(source.id, sample, sample + count, parts=[name])[0])
        * FIELD_UNIT_AT_1M
        for name in ("early", "low", "tail")
        if name in engine.source(source.id).parts
    }
    mine = np.sum(list(parts.values()), axis=0)
    theirs = np.asarray(field[:count], dtype=float)
    # The field's lead is a whole number of samples; the pack's is the same or says so.
    arrival = measure.envelope_arrival(theirs)
    centres = np.asarray(measure.THIRD_OCTAVES_HZ)
    a = max(arrival - int(BEFORE_S * rate), 0)
    images = max(pack.mirror.tail_from_s - MARGIN_S, MARGIN_S) if pack.header.has_tail else EARLY_S
    b = min(arrival + int(images * rate), count)
    split = min(arrival + int(EARLY_S * rate), count)

    def pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Level of ``x`` over ``y``, and energy of their difference over ``y``'s: third octaves."""
        want = _bands_db(y, rate)
        return _bands_db(x, rate) - want, _bands_db(x - y, rate) - want

    def cut(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
        out = np.array(x[lo:hi], dtype=float)
        edge = min(48, out.size // 2)
        ramp = np.sin(0.5 * np.pi * (np.arange(edge) + 0.5) / edge) ** 2
        out[:edge] *= ramp
        out[-edge:] *= ramp[::-1]
        return out

    image_level, image_error = pair(cut(mine, a, b), cut(theirs, a, b))
    early_level, _ = pair(cut(mine, a, split), cut(theirs, a, split))
    whole_level, whole_error = pair(mine, theirs)
    under = centres <= pack.crossover.band_hz()[0] * 2.0 ** (1.0 / 6.0)
    bank = np.asarray(band_centres(int(rate)))
    late_level = np.array(
        [
            float(
                measure.db(
                    np.sum(10.0 ** (_bands_db(mine[split:], rate)[near] / 10.0))
                    / np.sum(10.0 ** (_bands_db(theirs[split:], rate)[near] / 10.0)),
                    power=True,
                )
            )
            for near in ((centres > f * 0.7) & (centres < f * 1.42) for f in bank)
        ]
    )
    t20, _ = measure.decay_times_s(mine, rate, split)
    ref20, _ = measure.decay_times_s(theirs, rate, split)
    heard = centres <= 0.45 * rate / 2.0 ** (1.0 / 6.0)
    return {
        "step": int(step),
        "seconds": float(sample / rate),
        "distance_m": float(
            np.linalg.norm(
                np.asarray(source.position[step]) - np.asarray(pack.listener.position[step])
            )
        ),
        "arrival_ms": float(arrival / rate * 1e3),
        "pack_arrival_ms": float(measure.envelope_arrival(mine) / rate * 1e3),
        "third_octaves_hz": centres[heard],
        "images_ms": [float((a - arrival) / rate * 1e3), float((b - arrival) / rate * 1e3)],
        "images_level_db": image_level[heard],
        "images_error_db": image_error[heard],
        "early_level_db": early_level[heard],
        "low_third_octaves_hz": centres[under],
        "low_level_db": whole_level[under],
        "low_error_db": whole_error[under],
        "octaves_hz": bank,
        "late_level_db": late_level,
        "t20_s": t20,
        "reference_t20_s": ref20,
        "seam_db": float(source.low.seam_db[source.low.pair[step, 0, 0]])
        if source.low is not None
        else None,
        "direct": bool(np.any(np.asarray(source.early.kind[source.early.rows(step)]) == 0)),
    }


def compare(
    pack: ScenePack,
    reference: Path,
    *,
    sources: list[str] | None = None,
    workers: int = -1,
) -> dict[str, Any]:
    """Every source of ``pack`` that stands on the field's, at every point the head rests on.

    Refused, by a ``ValueError`` that says what is wrong, when the field is
    another dwelling's, when no source stands on the field's, or when the
    head rests on none of its points while that source sounds.
    """
    document: dict[str, Any] = {"reference": str(reference), "points": []}
    with h5py.File(reference, "r") as handle:
        if str(handle.attrs.get("scene_id", "")) != pack.header.scene_id:
            raise ValueError(
                f"the field is of scene {handle.attrs.get('scene_id')!r} and the pack of "
                f"{pack.header.scene_id!r}"
            )
        rate = float(handle.attrs["sample_rate_hz"])
        if rate != pack.header.sample_rate_hz:
            raise ValueError(f"the field is at {rate} Hz")
        at = np.asarray(handle.attrs["source_position"], dtype=float)
        positions = np.asarray(handle["positions"], dtype=float)
        gain = float(handle.attrs.get("gain", 1.0))
        named = sources if sources is not None else list(pack.sources)
        off = {
            name: float(
                np.linalg.norm(np.asarray(pack.sources[name].position, float) - at, axis=1).max()
            )
            for name in named
        }
        standing = [name for name in named if off[name] <= ON_THE_POINT_M]
        if not standing:
            raise ValueError(
                "no source of the pack stands on the field's source "
                f"({', '.join(f'{name} is {gap:.2f} m away' for name, gap in off.items())})"
            )
        for name in standing:
            source = pack.sources[name]
            steps = rest_steps(pack, source, positions)
            if not steps:
                raise ValueError(
                    f"while {name} sounds the head rests on no lattice point of the field for "
                    f"{WINDOW_S} s"
                )
            for point, step in steps.items():
                field = np.asarray(handle["ir"][point, 0, :], dtype=float) / gain
                document["points"].append(
                    {
                        "source": name,
                        "point": int(point),
                        "position": positions[point],
                        "head_off_the_point_m": float(
                            np.linalg.norm(positions[point] - pack.listener.position[step])
                        ),
                        "source_off_the_field_s_m": off[name],
                        **_point(pack, source, step, field, rate, workers),
                    }
                )
    document["lead_s"] = pack.mirror.lead_s
    return document


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 3) for v in value.ravel()]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _row(label: str, values: np.ndarray, digits: int = 1) -> str:
    cells = ["" if not np.isfinite(v) else f"{v:+.{digits}f}" for v in np.asarray(values, float)]
    return f"| {label} | " + " | ".join(cells) + " |"


def _head(label: str, hz: np.ndarray) -> list[str]:
    names = [f"{f / 1000:g}k" if f >= 1000 else f"{f:.0f}" for f in np.asarray(hz, float)]
    return [f"| {label} | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]


def markdown(document: dict[str, Any]) -> str:
    """The tables, a point after the other."""
    lines = [f"# The pack against {document['reference']}", ""]
    for found in document["points"]:
        lines += [
            f"## {found['source']} to lattice point {found['point']}, {found['distance_m']:.2f} m"
            + ("" if found["direct"] else ", no direct path"),
            "",
            f"Step {found['step']} ({found['seconds']:.2f} s); the head "
            f"{found['head_off_the_point_m'] * 1e3:.1f} mm off the point, the source "
            f"{found['source_off_the_field_s_m'] * 1e3:.1f} mm off the field's. First arrival "
            f"{found['pack_arrival_ms']:.2f} ms in the pack, {found['arrival_ms']:.2f} ms in "
            "the field."
            + ("" if found["seam_db"] is None else f" Seam {found['seam_db']:+.2f} dB."),
            "",
            f"The image paths, {found['images_ms'][0]:.0f} to {found['images_ms'][1]:+.0f} ms "
            "round the first arrival, per third octave, dB: the pack's level over the field's, "
            "the energy of their difference over the field's, and the level of the first 50 ms.",
            "",
            *_head("images", found["third_octaves_hz"]),
            _row("level", found["images_level_db"]),
            _row("error", found["images_error_db"]),
            _row("level to 50 ms", found["early_level_db"]),
            "",
            "Under the crossover, the whole response, dB.",
            "",
            *_head("low", found["low_third_octaves_hz"]),
            _row("level", found["low_level_db"]),
            _row("error", found["low_error_db"]),
            "",
            "After 50 ms, per octave: the pack's level over the field's in dB, and T20 in s.",
            "",
            *_head("late", found["octaves_hz"]),
            _row("level", found["late_level_db"]),
            _row("T20, pack", found["t20_s"], 2).replace("+", ""),
            _row("T20, field", found["reference_t20_s"], 2).replace("+", ""),
            "",
        ]
    return "\n".join(lines)


def run(
    pack: ScenePack,
    reference: Path,
    out: Path,
    *,
    sources: list[str] | None = None,
    workers: int = -1,
) -> dict[str, Any]:
    """:func:`compare`, printed, and written as ``reference_point.md`` and ``.json`` in ``out``."""
    document = compare(pack, reference, sources=sources, workers=workers)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    text = markdown(document)
    (out / "reference_point.md").write_text(text)
    (out / "reference_point.json").write_text(json.dumps(_plain(document), indent=1))
    print(text)
    return document
