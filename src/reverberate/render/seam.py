"""The tapered join: a pair's own seam at the crossover, the scene's one number above it.

``level/high_gain_db`` multiplies everything a source has above the
crossover: the mirror's gain plus the step's *seam*, how far the wave
response of the step's pairs stands over the mirror's in the crossover's
octave (``scene-pack.md``). A seam is two things. One is a constant, the
same within 0.3 dB whatever the room and the distance (1.9 dB in the
median of the first whole scene's 18 219 pairs): one error of scale
between the two solvers, which belongs in the calibration, once. The
other is each pair's own distance from it, 1.1 to 1.4 dB of scatter
decided within a wavelength at 1 kHz: what one octave of two interference
patterns happens to hold at one point. That part is what makes the two
bands meet at 1 kHz, and it says nothing of 4 kHz. Laid on the whole band
it made everything above 1 kHz of a source that walked, or that the head
walked past, go up and down by 2 to 5 dB within half a second
(``docs/open-questions/first-scene-defects.md``).

The owner's ruling (2026-10-05): the constant is one number for the scene
(:data:`SEAM_CONSTANT_DB`), and the pair's own part is **tapered**: whole
in the crossover's band, a share of it in the bands above, none from
4 kHz up (:data:`TAPER`). In decibels, for a step and a band of the bank::

    K = 20 log10(alignment_gain) + constant_db
    level[step, band] = K + taper[band] * (high_gain_db[step] - K)

A share of one is ``high_gain_db[step]`` itself, to the bit, and a share
of zero is ``K``. The table is ``level/band_gain_db``, ``[step, bank]``,
beside the scalar, which is never rewritten: a pack holds both, and
:attr:`reverberate.render.engine.RenderSettings.seam` says which the
engine applies, ``tapered`` (the table where the pack holds one) or
``broadband`` (the scalar, as every render before it, to the bit).

**No pair is solved again.** The trace writes the table; a pack traced
before it is given one **in place** by :func:`taper_pack`, ``python -m
reverberate.render seam PACK``: a few hundred kilobytes a source in a file
of gigabytes, said in the pack's provenance (``seam``), removed by
``--undo``. The table is made from what the trace wrote
(``level/high_gain_db_traced`` where ``relevel`` steadied the scalar).

**The engine's part** is :func:`level_table` and :class:`BandedTail`. The
arrivals carry a gain a band already and take the table as it is. The
late part has one gain a step: the bands of one level are rendered
together, on the dry signal through the sum of their filters of the
octave bank (:func:`group_kernels`), which add to the signal. Both parts
therefore lay the same gain on a frequency, :func:`applied_gain_db`: the
bank's bands cross in amplitude between two centres, so a level that
differs between 1 kHz and 2 kHz is a slope over that octave and not a
step.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.metrics import octave_bank
from reverberate.render.dry import DryTrack
from reverberate.render.pack import ScenePack, Source
from reverberate.render.tail import TailPart

__all__ = [
    "BAND_TABLE",
    "MODES",
    "SEAM_CONSTANT_DB",
    "TAPER",
    "BandedTail",
    "applied_gain_db",
    "band_levels",
    "group_kernels",
    "level_table",
    "motion_db",
    "scene_constant_db",
    "seam_record",
    "taper_of",
    "taper_pack",
]

#: What the engine applies above the crossover: the level a band where the pack holds
#: it, or the scalar whatever the pack holds.
MODES = ("tapered", "broadband")
#: The share of a pair's own seam a band keeps, from the crossover's band up; the last
#: holds for every band above, the first for those under. 1 kHz whole, 2 kHz half (in
#: decibels), 4 kHz and over none.
TAPER = (1.0, 0.5, 0.0)
#: THE normalisation between the two bands, in decibels over the alignment's gain: the
#: one number a scene's seams stand round. ``None`` is what the data give, the median
#: of the pack's pairs' seams (:func:`scene_constant_db`). The calibration that finds
#: the mirror's error of scale sets this, or gives it (``--constant-db``).
SEAM_CONSTANT_DB: float | None = None
#: Where a pack's constant came from, as its provenance says it.
GIVEN = "given"
MEDIAN = "the median of the pairs' seams"
#: The dataset of a source's ``level`` group.
BAND_TABLE = "band_gain_db"
#: The table the tapered join is made from where ``relevel`` rewrote the scalar.
TRACED = "high_gain_db_traced"


def taper_of(
    bank_hz: Iterable[float], cutoff_hz: float, taper: Sequence[float] = TAPER
) -> np.ndarray:
    """``[bank]``: the share of a pair's own seam each band of the bank keeps.

    ``taper`` starts at the band nearest the crossover, in octaves; its
    first value holds for the bands under it, its last for those above
    what it names.
    """
    shares = np.asarray(list(taper), dtype=float)
    if shares.size < 1 or np.any(shares < 0.0) or np.any(shares > 1.0):
        raise ValueError(f"a taper is shares between 0 and 1, from the crossover up: {taper}")
    bank = np.asarray(list(bank_hz), dtype=float)
    first = int(np.argmin(np.abs(np.log2(bank / float(cutoff_hz)))))
    return np.asarray(shares[np.clip(np.arange(bank.size) - first, 0, shares.size - 1)])


def scene_constant_db(seams_db: Iterable[np.ndarray]) -> float:
    """The scene's one number: the median of its pairs' seams, each pair once, in decibels.

    ``seams_db`` is every source's ``low/seam_db``, read in the pack's
    single precision so that a pack and its trace give the same number.
    The median and not the mean: a pair behind two walls is the ratio of
    two small numbers. On the first whole scene the pairs that hold a
    direct path and those that do not stand 0.3 dB apart, and the median
    of all is within 0.12 dB of the former's.
    """
    held = [np.asarray(s, dtype=np.float32).ravel() for s in seams_db]
    seams = np.concatenate(held) if held else np.zeros(0, dtype=np.float32)
    return float(np.median(seams)) if seams.size else 0.0


def band_levels(
    high_gain_db: np.ndarray, audible: np.ndarray, level_db: float, shares: np.ndarray
) -> np.ndarray:
    """``level/band_gain_db`` of one source: ``[step, bank]`` float32.

    ``high_gain_db`` is the trace's scalar, ``level_db`` the alignment's
    gain plus the scene's constant, ``shares`` one per band
    (:func:`taper_of`). A share of one is the scalar to the bit, a share
    of zero ``level_db``; a step that is not audible keeps the scalar.
    """
    traced = np.asarray(high_gain_db, dtype=np.float32)
    heard = np.asarray(audible, dtype=bool)
    made = np.repeat(traced[:, None], len(shares), axis=1)
    own = traced[heard].astype(np.float64) - float(level_db)
    for band, share in enumerate(np.asarray(shares, dtype=float)):
        if share == 1.0:
            continue
        made[heard, band] = np.float32(level_db) if share == 0.0 else level_db + share * own
    return made


def seam_record(
    bank_hz: Iterable[float], shares: np.ndarray, base_db: float, constant_db: float, constant: str
) -> dict[str, Any]:
    """What a pack's provenance says of its level a band, under ``seam``.

    ``constant`` says where the number came from; ``level_db`` is what a
    band whose share is zero is multiplied by, the alignment's gain plus
    the constant.
    """
    return {
        "mode": "tapered",
        "bank_hz": [int(v) for v in bank_hz],
        "shares": [float(v) for v in shares],
        "constant_db": round(float(constant_db), 4),
        "constant": constant,
        "level_db": round(float(base_db) + float(constant_db), 4),
    }


def motion_db(levels_db: np.ndarray, audible: np.ndarray, steps: int) -> dict[str, float]:
    """How far a level moves within ``steps`` steps, over the audible windows where it could.

    The range of every window of ``steps + 1`` audible steps on end: its
    median, ninth decile, 99th percentile and largest, in dB. Every such
    window is counted, also those in which the level stood still.
    """
    values = np.asarray(levels_db, dtype=np.float64)
    heard = np.asarray(audible, dtype=bool)
    nothing = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "windows": 0}
    if values.size <= steps:
        return nothing
    views = np.lib.stride_tricks.sliding_window_view(values, steps + 1)
    whole = np.lib.stride_tricks.sliding_window_view(heard, steps + 1).all(axis=1)
    span = (views.max(axis=1) - views.min(axis=1))[whole]
    if span.size == 0:
        return nothing
    p50, p90, p99 = np.percentile(span, [50, 90, 99])
    return {
        "p50": round(float(p50), 2),
        "p90": round(float(p90), 2),
        "p99": round(float(p99), 2),
        "max": round(float(span.max()), 2),
        "windows": int(span.size),
    }


# --------------------------------------------------------------------------
# the engine's part
# --------------------------------------------------------------------------


def level_table(source: Source, mode: str) -> np.ndarray | None:
    """What the engine multiplies ``source`` by above the crossover, a band: ``[step, bank]``.

    ``None`` is the pack's scalar, ``level/high_gain_db``: under
    ``broadband`` always, and under ``tapered`` for a pack that holds no
    ``level/band_gain_db``.
    """
    if mode not in MODES:
        raise ValueError(f"a seam is one of {', '.join(MODES)}, not {mode!r}")
    return None if mode == "broadband" else source.level.band_gain_db


def group_kernels(rate: float, groups: Sequence[Sequence[int]]) -> list[np.ndarray]:
    """Zero phase filters that share a signal among groups of the bank's bands, adding to it.

    ``groups`` names every band of the octave bank at ``rate`` once. A
    group's filter is the sum of its bands' filters of the bank, laid with
    their centre tap on the sample as the arrivals lay them; that of the
    group which holds the lowest band is the signal less the others, so
    that the filters add to the signal exactly and the bank's own lack
    under 180 Hz, which no part above the crossover holds, is nobody's.
    Odd, of the bank's length less one.
    """
    kernels = np.asarray(octave_bank(int(round(rate))).filters, dtype=float).T
    named = sorted(band for group in groups for band in group)
    if named != list(range(kernels.shape[0])):
        raise ValueError("the groups name every band of the bank once")
    centre = (kernels.shape[1] - 1) // 2
    whole = np.zeros(2 * centre + 1)
    whole[centre] = 1.0
    made = [kernels[list(group), : 2 * centre + 1].sum(axis=0) for group in groups]
    lowest = next(index for index, group in enumerate(groups) if 0 in group)
    made[lowest] = whole - sum(k for index, k in enumerate(made) if index != lowest)
    return made


def applied_gain_db(freqs_hz: np.ndarray, levels_db: np.ndarray, rate: float) -> np.ndarray:
    """The gain both parts lay on each frequency under a level a band, in decibels.

    The bank's filters are zero phase and cross in amplitude: the gain at a
    frequency is the bands' gains weighted by their filters there.
    """
    kernels = np.asarray(octave_bank(int(round(rate))).filters, dtype=float).T
    groups = [[band] for band in range(kernels.shape[0])]
    half = (kernels.shape[1] - 1) // 2
    turn = np.exp(
        -2j * np.pi * np.asarray(freqs_hz, float)[:, None] * (np.arange(2 * half + 1) - half) / rate
    )
    gains = 10.0 ** (np.asarray(levels_db, dtype=float) / 20.0)
    filters = group_kernels(rate, groups)
    total = sum(g * (turn @ k).real for g, k in zip(gains, filters, strict=True))
    return np.asarray(20.0 * np.log10(np.maximum(np.abs(total), 1e-12)))


class BandedTail:
    """The late part under a level a band: the bands of one level together, each set apart.

    The late part has one gain a step. The bands whose columns of the
    table are the same are one late part, on the dry signal through the
    filter that gives it those bands (:func:`group_kernels`), with their
    column as its table; the filters add to the signal, so a table whose
    columns are all one is the late part itself, to the bit. The sets
    share what does not depend on the dry signal: the carrier, the
    histograms' responses and their norms are made and held once.
    """

    def __init__(
        self,
        pack: ScenePack,
        source: Source,
        track: DryTrack,
        xp: Any,
        table: np.ndarray,
        *,
        workers: int,
        mask: np.ndarray | None,
        part: Any = TailPart,
        **more: Any,
    ) -> None:
        #: ``part`` is the late part the sets are made of: the reference's, or the fast
        #: engine's (:class:`reverberate.render.fast.FastTail`), with what it takes besides.
        make = part
        columns: dict[bytes, list[int]] = {}
        for band in range(table.shape[1]):
            columns.setdefault(np.ascontiguousarray(table[:, band]).tobytes(), []).append(band)
        groups = [tuple(bands) for bands in columns.values()]
        kernels = group_kernels(pack.header.sample_rate_hz, groups)
        self.parts: list[Any] = []
        for group, kernel in zip(groups, kernels, strict=True):
            level = replace(source.level, high_gain_db=np.ascontiguousarray(table[:, group[0]]))
            part = make(
                pack,
                replace(source, level=level),
                track if len(groups) == 1 else track.masked(kernel),
                xp,
                workers=workers,
                mask=mask,
                **more,
            )
            if self.parts:
                # One source, one seed, one mask: what the first made is the others' too.
                first = self.parts[0]
                part._token = first._token
                part._energies, part._amplitudes = first._energies, first._amplitudes
                part._norms, part._products = first._norms, first._products
            self.parts.append(part)

    def render(self, k0: int, k1: int) -> Any:
        made = self.parts[0].render(k0, k1)
        for part in self.parts[1:]:
            made = made + part.render(k0, k1)
        return made

    def waves(self, k0: int, k1: int, pool: Any = None) -> Any:
        """The sets' plane waves summed, for a mix that encodes every source's at once.

        Of the fast engine's parts only (:meth:`reverberate.render.fast.FastTail.waves`).
        """
        made = None
        for part in self.parts:
            got = part.waves(k0, k1, pool)
            if got is not None:
                made = got if made is None else made + got
        return made

    def encoded(self, waves: Any) -> Any:
        return self.parts[0].encoded(waves)


# --------------------------------------------------------------------------
# a traced pack, in place
# --------------------------------------------------------------------------


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def taper_pack(
    path: Path,
    *,
    taper: Sequence[float] = TAPER,
    constant_db: float | None = SEAM_CONSTANT_DB,
    undo: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Give the pack at ``path`` its level a band, ``level/band_gain_db``, in place.

    ``constant_db`` is the scene's one number; left out, the median of the
    pack's pairs' seams. The scalar the trace wrote is read and never
    written. ``undo`` removes the table and what the provenance says of
    it; ``dry_run`` writes nothing. Returns, per source and per band of
    the bank from the crossover up, how far the level moved within half a
    second under the scalar and how far it moves under the table.
    """
    import h5py

    report: dict[str, Any] = {"pack": str(path), "written": not dry_run}
    with h5py.File(Path(path), "r" if dry_run else "r+") as f:
        provenance = json.loads(_text(f.attrs["provenance_json"]))
        sources = f["sources"]
        if undo:
            for group in sources.values():
                if not dry_run and BAND_TABLE in group["level"]:
                    del group["level"][BAND_TABLE]
            provenance.pop("seam", None)
            if not dry_run:
                f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
            report["seam"] = None
            return report
        bank = [int(v) for v in f.attrs["bank_bands_hz"]]
        cutoff = float(f["crossover"].attrs["cutoff_hz"])
        shares = taper_of(bank, cutoff, taper)
        base_db = 20.0 * float(np.log10(float(f["mirror"].attrs["alignment_gain"])))
        given = constant_db is not None
        constant = (
            float(constant_db)
            if constant_db is not None
            else scene_constant_db(g["low"]["seam_db"][...] for g in sources.values() if "low" in g)
        )
        level_db = base_db + constant
        within = int(round(0.5 / float(f.attrs["step_s"])))
        first = int(np.argmin(np.abs(np.log2(np.asarray(bank, dtype=float) / cutoff))))
        said: dict[str, Any] = {}
        for name, group in sources.items():
            level = group["level"]
            audible = np.asarray(group["audible"][...], dtype=bool)
            traced = np.asarray(level[TRACED][...] if TRACED in level else level["high_gain_db"])
            made = band_levels(traced, audible, level_db, shares)
            if not dry_run:
                if BAND_TABLE in level:
                    del level[BAND_TABLE]
                level.create_dataset(BAND_TABLE, data=made)
            said[name] = {
                "within_half_a_second_scalar_db": motion_db(traced, audible, within),
                "within_half_a_second_db": {
                    str(bank[band]): motion_db(made[:, band], audible, within)
                    for band in range(first, len(bank))
                },
            }
        record = seam_record(bank, shares, base_db, constant, GIVEN if given else MEDIAN)
        if not dry_run:
            provenance["seam"] = record
            f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        report["seam"] = record
        report["sources"] = said
    return report
