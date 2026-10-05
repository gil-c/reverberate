"""What multiplies a source above the crossover, a step at a time: ``level/high_gain_db``.

``level/high_gain_db`` multiplies everything a source has above the
crossover. As the trace wrote it until 2026-10-06 it is the mirror's gain
plus the step's *seam*: how far the wave response of the step's pairs
stands over the mirror's in the crossover's octave (``scene-pack.md``).
Each pair is levelled on its own, as ``mirror.hybrid.blend`` levels a
point, and along a walk that is a new number every 8 cm of the source's
rail and every 0.4 m of the head's path: over the 18 219 pairs of the
first whole scene (2026-10-05) a seam is +0.4 dB at the first decile and
+3.4 dB at the ninth, and everything over 1 kHz of a source that walked,
or that the listener walked past, went up and down by 2 to 5 dB within
half a second and by up to 8 dB within a second.

**What a seam is made of** (``docs/open-questions/first-scene-defects.md``,
section 10, measured on those pairs):

- *a constant*, 1.9 dB in the median, the same within 0.3 dB whatever the
  distance, the room, the height, the source, with a direct path or
  without. It is one error of scale: the mirror's direct sound stands
  2.3 to 2.7 dB under ``1 / d`` from 300 Hz to 5.6 kHz where the wave
  field's is on it, because the alignment reads both pulses' energy in
  0.5 ms, which holds the mirror's pulse whole and three fifths of the
  wave solver's. It belongs in the calibration, once;
- *a scatter round it*, 1.1 to 1.4 dB, of which the room, the line of
  sight and the distance explain a twentieth. It is decided within 0.3 m,
  a wavelength at 1 kHz, and is flat from 1 m on: the two solvers'
  interference patterns in one octave, which do not coincide and are no
  property of the band above.

So a walk needs **one number**, not one a pair. Four ways of writing the
table are kept, so that they can be heard side by side (:data:`SEAMS`):

``pair``
    each step its pairs' own seam: what the trace wrote. Exact at the
    join, at rest; breathes along a walk.
``smooth``
    :func:`steadied`: within every run of audible steps the scalar is
    averaged under a raised cosine two seconds either side
    (:data:`STEADY_S`). It keeps what changes over metres, and the data
    hold nothing that does.
``constant``
    the mirror's gain plus one number for the scene
    (:func:`scene_constant_db`: the median of its pairs' seams, unless
    one is given). Nothing moves; what is left at the join is each
    pair's own distance from the median, under 1 kHz against over it.
``tapered``
    the constant from 4 kHz up, the step's own seam in the bands to
    1 kHz, half way between them, in decibels, at 2 kHz
    (:func:`taper`): the join stays exact and what breathes is the
    octave or two above it. It is a level **a band**, which the pack's
    table does not hold: it is made by :func:`pack_tables` and given to
    the engine (``Engine(high_gain_db=...)``), never written.

``python -m reverberate.render relevel PACK --seam ...`` rewrites a traced
pack's tables **in place** for the three that are scalars: the table is a
few hundred kilobytes of a file of gigabytes, what the trace wrote is kept
beside it as ``level/high_gain_db_traced``, and ``--undo`` (or ``--seam
pair``) puts it back to the bit. ``--dry-run`` writes nothing and says
what each would do. No response is solved or read again, and the pairs'
own seams stay in the pack (``low/seam_db``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SEAMS",
    "STEADY_S",
    "TAPER_HZ",
    "levelled",
    "pack_tables",
    "relevel_pack",
    "scene_constant_db",
    "steadied",
    "taper",
    "wander_db",
]

#: The ways a step's level above the crossover is written; the first is the trace's own.
SEAMS = ("pair", "smooth", "constant", "tapered")
#: Half the raised cosine the scalar is averaged under, seconds: a walk of two metres,
#: two dozen solved positions of a rail and a dozen cells of the head's path.
STEADY_S = 2.0
#: A tapered seam is the pair's own up to the first, Hz, and the scene's constant from the
#: second: the crossover's octave whole, the octave at 2 kHz half way, 4 kHz and over none.
TAPER_HZ = (1000.0, 4000.0)
#: What the trace's own table is kept as in a pack that was relevelled.
TRACED = "high_gain_db_traced"


def scene_constant_db(seam_db: np.ndarray) -> float:
    """The one number a scene's seams are replaced by: their median, in decibels.

    ``seam_db`` is one value a pair, each pair once. The median and not
    the mean: a pair behind two walls is the ratio of two small numbers.
    """
    seams = np.asarray(seam_db, dtype=float)
    return float(np.median(seams)) if seams.size else 0.0


def taper(bank_hz: Any, taper_hz: tuple[float, float] = TAPER_HZ) -> np.ndarray:
    """How much of a pair's own seam each band keeps: one to ``taper_hz[0]``, none from the other.

    Linear in octaves between the two, so that with the defaults the band
    at 2 kHz keeps half of it, in decibels.
    """
    full, none = float(taper_hz[0]), float(taper_hz[1])
    if not 0.0 < full < none:
        raise ValueError(f"a taper runs from a frequency up to a higher one, not {taper_hz}")
    centres = np.maximum(np.asarray(bank_hz, dtype=float), 1e-9)
    return np.asarray(1.0 - np.clip(np.log2(centres / full) / np.log2(none / full), 0.0, 1.0))


def levelled(
    traced: np.ndarray,
    audible: np.ndarray,
    *,
    seam: str,
    base_db: float,
    constant_db: float = 0.0,
    half_steps: int = 0,
    bank_hz: Any = None,
    taper_hz: tuple[float, float] = TAPER_HZ,
) -> np.ndarray:
    """The table of one source under ``seam``, from the one the trace wrote.

    ``traced`` is ``base_db`` (the mirror's gain) plus each step's own seam.
    ``[step]`` float32 for the scalar ways; ``[step, band]`` float32 over
    ``bank_hz`` for ``tapered``. A step that is not audible keeps what it
    had, and ``pair`` is ``traced`` itself.
    """
    if seam not in SEAMS:
        raise ValueError(f"a seam is one of {', '.join(SEAMS)}, not {seam!r}")
    held = np.array(traced, dtype=np.float32)
    heard = np.asarray(audible, dtype=bool)
    if seam == "pair":
        return held
    if seam == "smooth":
        return steadied(held, heard, half_steps)
    flat = np.float32(float(base_db) + float(constant_db))
    if seam == "constant":
        held[heard] = flat
        return held
    if bank_hz is None:
        raise ValueError("a tapered seam is a level a band: it needs the bank's centres")
    keep = taper(bank_hz, taper_hz)
    table = np.repeat(held[:, None], keep.size, axis=1).astype(np.float64)
    own = np.asarray(traced, dtype=np.float64)[heard]
    table[heard] = float(flat) + keep[None, :] * (own[:, None] - float(flat))
    return table.astype(np.float32)


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


def _spread(values: np.ndarray) -> dict[str, float]:
    """A distribution in four numbers, dB; zeros when there is nothing."""
    if values.size == 0:
        return {"p10": 0.0, "median": 0.0, "p90": 0.0, "largest": 0.0}
    return {
        "p10": round(float(np.percentile(values, 10)), 2),
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "largest": round(float(np.abs(values).max()), 2),
    }


def _measured(
    traced: np.ndarray, made: np.ndarray, audible: np.ndarray, step_s: float, bank: list[int]
) -> dict[str, Any]:
    """What a table does: how far it moves within half a second and one, and what it leaves.

    ``left_at_the_join_db`` is the step's own seam less what the table
    gives the crossover's band there: how far the two bands stand apart at
    the join, which a table that follows each pair hides and one that does
    not leaves.
    """
    half, one = int(round(0.5 / step_s)), int(round(1.0 / step_s))
    said: dict[str, Any] = {
        "within_half_a_second_before_db": wander_db(traced, audible, half),
        "within_a_second_before_db": wander_db(traced, audible, one),
    }
    if made.ndim == 1:
        said["within_half_a_second_after_db"] = wander_db(made, audible, half)
        said["within_a_second_after_db"] = wander_db(made, audible, one)
        at_join = made
    else:
        said["within_half_a_second_after_db"] = {
            str(c): wander_db(made[:, b], audible, half) for b, c in enumerate(bank)
        }
        said["within_a_second_after_db"] = {
            str(c): wander_db(made[:, b], audible, one) for b, c in enumerate(bank)
        }
        at_join = made[:, int(np.argmin(np.abs(np.asarray(bank, dtype=float) - 1000.0)))]
    left = (np.asarray(traced, dtype=np.float64) - np.asarray(at_join, dtype=np.float64))[audible]
    said["left_at_the_join_db"] = _spread(left)
    said["largest_change_db"] = round(float(np.abs(left).max(initial=0.0)), 2)
    return said


def pack_tables(
    path: Path,
    *,
    seam: str,
    seconds: float = STEADY_S,
    constant_db: float | None = None,
    taper_hz: tuple[float, float] = TAPER_HZ,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every source's table under ``seam``, and what it does; the pack at ``path`` is only read.

    The tables are what ``Engine(high_gain_db=...)`` takes, whatever the
    pack's own hold: they start from the trace's (``high_gain_db_traced``
    where the pack was relevelled). ``constant_db`` left out is the
    scene's (:func:`scene_constant_db` over the pack's pairs, each once).
    """
    import h5py

    if seam not in SEAMS:
        raise ValueError(f"a seam is one of {', '.join(SEAMS)}, not {seam!r}")
    tables: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"pack": str(path), "seam": seam}
    with h5py.File(Path(path), "r") as f:
        step_s = float(f.attrs["step_s"])
        base = 20.0 * float(np.log10(float(f["mirror"].attrs["alignment_gain"])))
        bank = [int(c) for c in f.attrs["bank_bands_hz"]]
        seams: dict[bytes, float] = {}
        for group in f["sources"].values():
            low = group.get("low")
            if low is not None and "seam_db" in low:
                seams.update(zip(low["pair_key"][...], low["seam_db"][...], strict=True))
        scene = scene_constant_db(np.array(list(seams.values()), dtype=float))
        constant = scene if constant_db is None else float(constant_db)
        report["pairs"] = len(seams)
        report["scene_constant_db"] = round(scene, 3)
        if seam == "smooth":
            report["seconds"] = float(seconds)
        if seam in ("constant", "tapered"):
            report["constant_db"] = round(constant, 3)
        if seam == "tapered":
            report["taper_hz"] = [float(v) for v in taper_hz]
            report["bank_bands_hz"] = bank
            report["kept_of_the_pair"] = [round(float(v), 3) for v in taper(bank, taper_hz)]
        sources: dict[str, Any] = {}
        for name, group in f["sources"].items():
            level = group["level"]
            audible = np.asarray(group["audible"][...], dtype=bool)
            traced = np.asarray(
                level[TRACED][...] if TRACED in level else level["high_gain_db"][...]
            )
            tables[name] = levelled(
                traced,
                audible,
                seam=seam,
                base_db=base,
                constant_db=constant,
                half_steps=int(round(float(seconds) / step_s)),
                bank_hz=bank,
                taper_hz=taper_hz,
            )
            sources[name] = _measured(traced, tables[name], audible, step_s, bank)
        report["sources"] = sources
    return tables, report


def relevel_pack(
    path: Path,
    *,
    seam: str = "smooth",
    seconds: float = STEADY_S,
    constant_db: float | None = None,
    taper_hz: tuple[float, float] = TAPER_HZ,
    undo: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write ``level/high_gain_db`` of every source of the pack at ``path`` another way, in place.

    What the trace wrote is kept as ``level/high_gain_db_traced`` the first
    time and is what every later call starts from, so calling twice, or
    with another way, does not relevel a relevelled table. ``undo``, or
    ``seam="pair"``, puts the trace's table back to the bit and removes the
    copy. ``dry_run`` writes nothing. A ``tapered`` seam is never written:
    the pack's table is a scalar a step (:func:`pack_tables`). The way is
    recorded on each ``level`` group and in the pack's provenance
    (``seam``). Returns what :func:`pack_tables` measured.
    """
    import h5py

    back = undo or seam == "pair"
    tables, report = pack_tables(
        path,
        seam="pair" if back else seam,
        seconds=seconds,
        constant_db=constant_db,
        taper_hz=taper_hz,
    )
    report["written"] = False
    if back:
        report["seconds"] = None
    if dry_run or (seam == "tapered" and not back):
        return report
    with h5py.File(Path(path), "r+") as f:
        for name, group in f["sources"].items():
            level = group["level"]
            for stale in ("steadied_s", "seam", "seam_constant_db"):
                level.attrs.pop(stale, None)
            if back:
                level["high_gain_db"][...] = tables[name]
                if TRACED in level:
                    del level[TRACED]
                report["sources"][name] = {"restored": True}
                continue
            if TRACED not in level:
                level.create_dataset(TRACED, data=np.asarray(level["high_gain_db"][...]))
            level["high_gain_db"][...] = tables[name]
            level.attrs["seam"] = seam
            if seam == "smooth":
                level.attrs["steadied_s"] = float(seconds)
            else:
                level.attrs["seam_constant_db"] = float(report["constant_db"])
        provenance = json.loads(f.attrs.get("provenance_json", "{}") or "{}")
        was = provenance.pop("seam", None)
        if not back:
            provenance["seam"] = {
                key: report[key] for key in ("seam", "seconds", "constant_db") if key in report
            }
            provenance["seam"]["relevelled"] = True
        if provenance.get("seam") != was:
            f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
    report["written"] = True
    return report
