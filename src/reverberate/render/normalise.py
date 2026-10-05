"""The two bands of a traced pack on one scale, once for all: in place, and back.

A pack is physical: a unit source reads ``1 / d`` in free air at every
frequency. The band under the crossover is (the solve over
``FIELD_UNIT_AT_1M``, within 0.1 dB), and the mirror is before anything is
measured (-0.06 dB). The audit of 2026-10-05
(``docs/open-questions/chain-audit.md``) found what stood between the two
in every pack traced until then, and none of it is a dwelling's:

- **D1**, ``/mirror`` ``alignment_gain``: fitted as a ratio of energies
  inside 0.5 ms round two pulses that are not the same pulse, it is 2.4 dB
  low with its signature at every frequency (-5.22 dB and +2.85 dB at
  1 kHz in the first scene's packs).
- **D2**, ``/mirror/signature``: the direct spectrum of a field solved to
  8 kHz, the band limit of that grid with it, laid on a band that is not
  that grid's: -1.8 dB at 8 kHz, -3.3 at 12 kHz, -5.0 at 16 kHz re 1 kHz.
- **D7**, ``low/seam_db``: a pair's seam read the band limit of its solve
  to 1500 Hz as a level, :data:`SEAM_READING_DB` low.
- **D3**, ``/directivity``: a table of unit mean power puts a voice's axis
  3.0 dB (1 kHz) to 6.2 dB (8 kHz) over its clip, and the band under the
  crossover stays at the clip's level.

:func:`normalise_pack`, ``python -m reverberate.render normalise PACK``,
writes what a pack traced today is born with, **in place** and with no
pair solved again. In decibels, ``a`` the old alignment gain, ``s`` the
old signature's level over the crossover's octave
(:func:`reverberate.mirror.direct.signature_level_db`), ``r`` the seam's
reading (:data:`SEAM_READING_DB`), and ``W`` the share of a step's weight
that reads a pair (one wherever a step has its pairs)::

    alignment_gain  = 1                        (0 dB)
    signature       = a unit pulse             (or the old one less s: --keep-signature)
    seam_db[pair]   = seam_db[pair] + a + s + r
    high_gain_db[k] = high_gain_db[k] - a + W[k] (a + s + r)
    gain_db[b, :]   = gain_db[b, :] - gain_db[b, 0]     (each directivity table)

The seam of a pair is how far the wave response stands over the mirror's
as the pack renders it: the mirror no longer carries ``a`` and ``s``, so
the same wave response stands ``a + s`` further over it, and ``r`` more
that the reading had lost. ``high_gain_db`` is the gain plus the step's
pairs' seams, both moved; with ``W = 1`` it moves by ``s + r``, and the
mirror, which lost ``s`` with its signature, is rendered ``r`` louder
under the scalar. Then ``level/band_gain_db`` is made again from these
(:func:`reverberate.render.seam.taper_pack`) with the code's one constant,
``K = 0`` dB: above 4 kHz the mirror's direct sound is ``1 / d``, and at
the crossover each step keeps how far its pairs stand from the scene's
median.

**Exact, and what is not.** ``a`` is exact. ``s`` is exact for a mirror
response that is flat over the octave: the signature moves by 0.9 dB
across it (+3.2 dB at 707 Hz, +2.3 at 1414 Hz), and a seam weighs each
frequency by what the pair holds there. ``r`` is a constant measured on
the direct sound of 92 pairs; a pair whose response is not flat over the
octave differs by its own tilt. Both are the scatter of a pair's seam,
which only the crossover's band keeps, tapered; neither is in the level.

**Nothing is lost.** What the trace wrote is kept in the pack beside it,
the group ``/as_traced``, the first time; ``--undo`` puts every value
back to the bit and removes the group. Asking twice changes nothing. The
provenance says every number changed and why, under ``normalisation``.
The recipe and its digest are not touched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.direct import signature_level_db, unit_signature
from reverberate.mirror.directivity import NORMALISATIONS, NORMALISED, radiated_db
from reverberate.render.pack import SCHEMA, SCHEMA_VERSION, ScenePack, Source
from reverberate.render.seam import BAND_TABLE, GIVEN, TAPER, TRACED, scene_constant_db, taper_pack
from reverberate.spatial.translate import MODE_FUSED

__all__ = [
    "KEPT",
    "SEAM_READING_DB",
    "TOOL",
    "normalise_pack",
    "radiating",
    "seam_weight",
]

#: What the octave's reading of a pair solved to 1500 Hz is short by, dB: the solve's
#: band limit read as a level. Measured on the direct sound of 92 pairs of the first
#: whole scene under 0.9 m (-0.45 dB at 1 kHz, -1.4 at 1189 Hz, -6.0 at 1414 Hz).
SEAM_READING_DB = 1.07
#: The group that keeps what the trace wrote, in a pack that was normalised.
KEPT = "as_traced"
#: What a pack's provenance names this tool.
TOOL = "reverberate.render.normalise/1"
#: A pack whose low band is no solve's has no band limit in the octave to give back.
_WHOLE = ("free-field", "synthetic")


def radiating(pack: ScenePack, source: Source, directivity: bool) -> ScenePack:
    """``pack`` as the late part of a directive ``source`` reads it: the tail at what it radiates.

    The late part is omnidirectional, the energy of rays that left the
    source alike in every direction. A table that is level with its axis
    radiates less than that, a band, by its directivity index
    (:func:`reverberate.mirror.directivity.radiated_db`): the pack is given
    back with that added to ``tail_gain_db``. The pack itself where the
    source's directivity is not applied, and for a table of unit mean
    power, which radiates what the omnidirectional source does: every
    render of such a pack is what it was, to the bit.
    """
    table = pack.directivity.get(source.directivity_model) if directivity else None
    if table is None or table.normalised != "axis":
        return pack
    down = radiated_db(table.gain_db, table.angles_deg)
    if not np.any(np.abs(down) > 1e-9):
        return pack
    gains = np.asarray(pack.mirror.tail_gain_db, dtype=float) + down
    return replace(pack, mirror=replace(pack.mirror, tail_gain_db=tuple(float(g) for g in gains)))


def seam_weight(
    audible: np.ndarray,
    pair: np.ndarray,
    position_weight: np.ndarray,
    cell: np.ndarray,
    mode: np.ndarray,
    listener: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    """``[step]``: the share of a step's weight that reads a pair, as the trace weighed them.

    The weights of :func:`reverberate.trace.level.step_levels`, summed
    over the pairs the step holds: one where it holds them all, zero for
    a step that is not audible or holds none.
    """
    weight = np.zeros(int(audible.shape[0]))
    for step in np.flatnonzero(audible):
        across = np.array([1.0 - float(position_weight[step]), float(position_weight[step])])
        between = np.array([1.0, 0.0])
        if int(mode[step]) == MODE_FUSED:
            away = np.linalg.norm(cells[cell[step]] - listener[step][None, :], axis=1)
            total = float(away.sum())
            if total > 0.0:
                between = np.array([away[1], away[0]]) / total
        held = pair[step] >= 0
        weight[step] = float(np.sum(np.outer(across, between)[held]))
    return weight


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _put(group: Any, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _refuse(why: str) -> None:
    raise ValueError(f"this pack is not one the normalisation understands: {why}")


def _restore(f: Any) -> None:
    """Every value the group ``/as_traced`` kept, back where the trace wrote it."""
    kept = f[KEPT]
    mirror = f["mirror"]
    mirror.attrs["alignment_gain"] = float(kept.attrs["alignment_gain"])
    _put(mirror, "signature", np.asarray(kept["signature"][...]))
    for model, held in kept["directivity"].items():
        group = f["directivity"][model]
        group["gain_db"][...] = held["gain_db"][...]
        for word in ("normalised", "pattern_sha256"):
            if word in held.attrs:
                group.attrs[word] = _text(held.attrs[word])
            else:
                group.attrs.pop(word, None)
    for name, held in kept["sources"].items():
        source = f["sources"][name]
        if "seam_db" in held:
            source["low"]["seam_db"][...] = held["seam_db"][...]
        for table in ("high_gain_db", TRACED):
            if table in held:
                source["level"][table][...] = held[table][...]
    del f[KEPT]


def _keep(f: Any) -> None:
    """What the trace wrote, copied into the group ``/as_traced``."""
    kept = f.create_group(KEPT)
    mirror = f["mirror"]
    kept.attrs["alignment_gain"] = float(mirror.attrs["alignment_gain"])
    kept.create_dataset("signature", data=np.asarray(mirror["signature"][...]))
    tables = kept.create_group("directivity")
    for model, group in f["directivity"].items():
        held = tables.create_group(model)
        held.create_dataset("gain_db", data=np.asarray(group["gain_db"][...]))
        for word in ("normalised", "pattern_sha256"):
            if word in group.attrs:
                held.attrs[word] = _text(group.attrs[word])
    sources = kept.create_group("sources")
    for name, source in f["sources"].items():
        held = sources.create_group(name)
        if "low" in source:
            held.create_dataset("seam_db", data=np.asarray(source["low"]["seam_db"][...]))
        for table in ("high_gain_db", TRACED):
            if table in source["level"]:
                held.create_dataset(table, data=np.asarray(source["level"][table][...]))


def _taper(seam: dict[str, Any] | None, cutoff_hz: float) -> tuple[float, ...]:
    """The taper a provenance's record of a level a band was made with; the code's where none."""
    if seam is None:
        return TAPER
    bank = np.asarray([float(v) for v in seam["bank_hz"]])
    first = int(np.argmin(np.abs(np.log2(bank / float(cutoff_hz)))))
    return tuple(float(v) for v in seam["shares"][first:])


def _again(path: Path, seam: dict[str, Any] | None, cutoff_hz: float) -> None:
    """The level a band as ``seam``, a provenance's record, says it was made; none where none."""
    if seam is None:
        taper_pack(path, undo=True)
        return
    given = seam.get("constant") == GIVEN
    taper_pack(
        path,
        taper=_taper(seam, cutoff_hz),
        constant_db=float(seam["constant_db"]) if given else None,
    )


def normalise_pack(
    path: Path,
    *,
    undo: bool = False,
    dry_run: bool = False,
    keep_signature: bool = False,
    directivity: str = NORMALISED,
    seam_reading_db: float | None = None,
) -> dict[str, Any]:
    """Put the pack at ``path`` on the physical scale in place; returns every number changed.

    ``keep_signature`` keeps the measured signature's colour, at 0 dB over
    the crossover's octave, and not a unit pulse: the treble every render
    before had, at the right level. ``directivity`` is what the tables are
    made level with, ``"axis"`` or ``"mean"``. ``seam_reading_db`` is what
    the pairs' seams were read short by; left out, :data:`SEAM_READING_DB`
    for a pack whose low band is a solve's and zero for a free field's.
    ``undo`` puts back what the trace wrote; ``dry_run`` writes nothing.
    """
    import h5py

    if directivity not in NORMALISATIONS:
        raise ValueError(f"a table is normalised on one of {NORMALISATIONS}, not {directivity!r}")
    path = Path(path)
    report: dict[str, Any] = {"pack": str(path), "written": not dry_run}
    with h5py.File(path, "r" if dry_run else "r+") as f:
        if _text(f.attrs.get("schema", "")) != SCHEMA:
            _refuse(f"its schema is {f.attrs.get('schema')!r}")
        if int(f.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            _refuse(f"its version is {f.attrs.get('schema_version')!r}")
        for name in ("mirror", "directivity", "sources", "listener", "cells"):
            if name not in f:
                _refuse(f"it holds no /{name}")
        cutoff = float(f["crossover"].attrs["cutoff_hz"]) if "crossover" in f else 1000.0
        if "provenance_json" not in f.attrs or "alignment_gain" not in f["mirror"].attrs:
            _refuse("it says nothing of its provenance or of its mirror's gain")
        provenance = json.loads(_text(f.attrs["provenance_json"]))
        said = dict(provenance.get("normalisation") or {})
        if said.get("born"):
            if undo:
                _refuse("it was traced on the physical scale and holds nothing to put back")
            report["normalisation"] = said
            report["changed"] = False
            return report
        if said and KEPT not in f:
            _refuse(
                "its provenance says it was normalised and it no longer holds what the trace "
                "wrote (/as_traced): it was rewritten since"
            )
        if KEPT in f and not said:
            _refuse("it holds /as_traced and its provenance does not say why")
        wanted = {
            "signature": "colour" if keep_signature else "unit",
            "directivity": directivity,
            "seam_reading_db": seam_reading_db,
        }
        if undo:
            if not said:
                report["normalisation"] = None
                report["changed"] = False
                return report
            before = said.get("seam_before")
            if not dry_run:
                _restore(f)
                provenance.pop("normalisation", None)
                provenance.pop("seam", None)
                f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
            report["normalisation"] = None
            report["changed"] = True
        elif said and said.get("asked") == wanted:
            report["normalisation"] = said
            report["changed"] = False
            return report
        else:
            before = said.get("seam_before") if said else provenance.get("seam")
            record = _normalise(f, provenance, said, wanted, before, dry_run)
            report["normalisation"] = record
            report["changed"] = True
    # The level a band, made again from what the pack now holds: the code's constant
    # for a normalised pack; as it was made, or not at all, for one put back.
    if undo and not dry_run:
        _again(path, before, cutoff)
    elif not undo and not dry_run:
        report["seam"] = taper_pack(path, taper=_taper(before, cutoff))["seam"]
    return report


def _normalise(
    f: Any,
    provenance: dict[str, Any],
    said: dict[str, Any],
    wanted: dict[str, Any],
    seam_before: dict[str, Any] | None,
    dry_run: bool,
) -> dict[str, Any]:
    """The pack open as ``f`` put on the physical scale; the provenance's record."""
    # From what the trace wrote, whatever an earlier normalisation made of it.
    kept = f.get(KEPT)
    mirror = f["mirror"]
    held_mirror = kept if kept is not None else mirror
    gain = float(held_mirror.attrs["alignment_gain"])
    signature = np.asarray(held_mirror["signature"][...], dtype=float)
    if not gain > 0.0 or signature.ndim != 1 or signature.size < 1:
        _refuse("its mirror has no gain or no signature")
    rate = float(f.attrs["sample_rate_hz"]) if "sample_rate_hz" in f.attrs else 48000.0
    has_low = "crossover" in f and any("low" in g for g in f["sources"].values())
    cutoff, width = 1000.0, 1.0
    if "crossover" in f:
        cutoff = float(f["crossover"].attrs["cutoff_hz"])
        width = float(f["crossover"].attrs["width_octaves"])
    octave = (cutoff * 2.0 ** (-0.5 * width), cutoff * 2.0 ** (0.5 * width))
    a_db = 20.0 * float(np.log10(gain))
    s_db = signature_level_db(signature, octave, rate)
    solver = str(provenance.get("solver", ""))
    reading = wanted["seam_reading_db"]
    if reading is None:
        reading = 0.0 if solver.startswith(_WHOLE) or not has_low else SEAM_READING_DB
    reading = float(reading)
    moved = a_db + s_db + reading
    made_signature = (
        np.ones(1) if wanted["signature"] == "unit" else unit_signature(signature, octave, rate)
    )
    freqs = np.fft.rfftfreq(4800, 1.0 / rate)
    spectrum = np.abs(np.fft.rfft(signature, 4800))
    after = np.abs(np.fft.rfft(made_signature, 4800))
    probes = [f_hz for f_hz in (1000, 2000, 4000, 8000, 12000, 16000) if f_hz < 0.5 * rate]

    def at(values: np.ndarray, hz: float) -> float:
        return round(float(20.0 * np.log10(values[int(np.argmin(np.abs(freqs - hz)))])), 3)

    if not dry_run:
        if kept is not None:
            _restore(f)
        _keep(f)
        mirror.attrs["alignment_gain"] = 1.0
        _put(mirror, "signature", np.asarray(made_signature, dtype="<f8"))
    changed: dict[str, Any] = {
        "mirror/alignment_gain": {
            "was": gain,
            "was_db": round(a_db, 4),
            "is": 1.0,
            "why": "D1: fitted as a ratio of energies inside 0.5 ms round two pulses that are "
            "not one pulse; the mirror renders a unit source as 1 / d with no gain",
        },
        "mirror/signature": {
            "was_taps": int(signature.size),
            "was_octave_db": round(s_db, 4),
            "mirror_direct_was_db": {str(hz): round(a_db + at(spectrum, hz), 3) for hz in probes},
            "mirror_direct_is_db": {str(hz): at(after, hz) for hz in probes},
            "is": "a unit pulse"
            if wanted["signature"] == "unit"
            else "the measured signature at 0 dB over the crossover's octave",
            "why": "D2: the direct spectrum of a field solved to 8 kHz, its grid's band limit "
            "with it, on a band that is not that grid's",
        },
    }
    # In a dry run of a pack already normalised, what the trace wrote is read where it
    # is kept; otherwise it was just put back where it was.
    aside = kept if (kept is not None and dry_run) else None
    # The tables of directivity, each from the pattern the trace wrote.
    tables: dict[str, Any] = {}
    for model, group in f["directivity"].items():
        traced = group if aside is None else aside["directivity"][model]
        table = np.asarray(traced["gain_db"][...], dtype=float)
        angles = np.asarray(group.attrs["angles_deg"], dtype=float)
        if table.ndim != 2 or table.shape[1] != angles.size or angles[0] != 0.0:
            _refuse(f"/directivity/{model} is not a table from 0 degrees")
        was = _text(traced.attrs["normalised"]) if "normalised" in traced.attrs else "mean"
        level = radiated_db(table, angles)[:, None]
        if wanted["directivity"] == "axis":
            level = table[:, :1]
        made = (table - level).astype(np.float32)
        if not dry_run:
            if was == "mean" and "pattern_sha256" not in group.attrs:
                # The recipe's key is that of the table the trace wrote: kept as a word.
                stored = np.ascontiguousarray(traced["gain_db"][...], dtype="<f4")
                group.attrs["pattern_sha256"] = hashlib.sha256(stored.tobytes()).hexdigest()
            group["gain_db"][...] = made
            group.attrs["normalised"] = wanted["directivity"]
        # A table already level as asked moves by its single precision, which is nothing.
        if np.any(np.abs(level) > 1e-5):
            tables[model] = {
                "was": was,
                "is": wanted["directivity"],
                "added_db": [round(float(-v), 4) for v in level[:, 0]],
                "radiates_db": [round(float(v), 4) for v in radiated_db(made, angles)],
            }
    if tables:
        changed["directivity"] = {
            "tables": tables,
            "why": "D3: a clip is its talker's axis; a table of unit mean power put the axis "
            "3.0 dB (1 kHz) to 6.2 dB (8 kHz) over the clip above the crossover and left "
            "the band under it at the clip's level. The late part of a source whose "
            "directivity is applied is rendered at what the table radiates",
        }
    # The seams and the scalar, a source at a time.
    listener = np.asarray(f["listener"]["position"][...], dtype=float)
    cells = np.asarray(f["cells"]["position"][...], dtype=float)
    seams_before, seams_after, partial = [], [], 0
    for name, source in f["sources"].items():
        levels = source["level"]
        audible = np.asarray(source["audible"][...], dtype=bool)
        weight = np.zeros(audible.shape[0])
        if "low" in source:
            low = source["low"]
            traced = low if aside is None else aside["sources"][name]
            seam = np.asarray(traced["seam_db"][...])
            seams_before.append(seam)
            made_seam = (seam.astype(np.float64) + moved).astype(np.float32)
            seams_after.append(made_seam)
            weight = seam_weight(
                audible,
                np.asarray(low["pair"][...]),
                np.asarray(low["position_weight"][...], dtype=float),
                np.asarray(low["cell"][...]),
                np.asarray(low["mode"][...]),
                listener,
                cells,
            )
            if not dry_run:
                low["seam_db"][...] = made_seam
        partial += int(np.sum(audible & (weight > 1e-6) & (weight < 1.0 - 1e-6)))
        for scalar in ("high_gain_db", TRACED):
            if scalar in levels and not dry_run:
                was_level = np.asarray(levels[scalar][...], dtype=np.float64)
                made_level = np.where(audible, was_level - a_db + weight * moved, was_level)
                levels[scalar][...] = made_level.astype(np.float32)
    changed["low/seam_db"] = {
        "added_db": round(moved, 4),
        "of_which": {
            "alignment_gain_db": round(a_db, 4),
            "signature_octave_db": round(s_db, 4),
            "seam_reading_db": round(reading, 4),
        },
        "median_was_db": round(scene_constant_db(seams_before), 4),
        "median_is_db": round(scene_constant_db(seams_after), 4),
        "why": "a seam is the wave response over the mirror's as the pack renders it: the "
        "mirror lost its gain and its signature's level; and D7: the seam read the band "
        "limit of a solve to 1500 Hz as a level",
    }
    changed["level/high_gain_db"] = {
        "added_db": round(s_db + reading, 4),
        "steps_that_read_a_part_of_their_pairs": partial,
        "why": "the mirror's gain plus the step's pairs' seams, both moved: "
        "- a + W (a + s + r), W the share of the step's weight that reads a pair",
    }
    record = {
        "alignment": "physical",
        "signature": wanted["signature"],
        "seam": "unbiased",
        "directivity": wanted["directivity"],
        "constant": "fixed",
        "born": False,
        "tool": TOOL,
        "asked": wanted,
        "alignment_gain_db": 0.0,
        "changed": changed,
        "kept": f"/{KEPT}",
        "seam_before": seam_before,
    }
    if not dry_run:
        provenance["normalisation"] = record
        f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        # The level a band is made again from the new values; never left on the old.
        for source in f["sources"].values():
            if BAND_TABLE in source["level"]:
                del source["level"][BAND_TABLE]
    return record
