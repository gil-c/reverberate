"""Two packs of one recipe, side by side: the files to hear and what differs, band by band.

The use is a decision taken by ear: the same scene traced with its low band
on two wave grids, the validated one and a coarser, cheaper one. Whether the
second sounds the same is the owner's to say; this module hands him the two
files and the numbers beside them, and judges nothing.

- **The files** (``listen/A_mix.wav``, ``listen/B_mix.wav`` and each
  source's): the same window, the same clips, the same head, through the
  page's decode (:mod:`reverberate.render.check.binaural`), **at one gain
  for both**. Nothing is normalised: a pack that is louder is louder in
  its file, which is part of what is being heard. Where the window is
  faint, one more set is written louder, by one figure for both.
- **The difference** (``ab.json``, ``ab.md``): for every source, the level
  of B less that of A in each third octave, read three ways. On the
  response to an impulse at a step where the source is heard, cut at 50 ms
  after its arrival: the *early* part, which is the direct sound and the
  first reflections, and the *late* part, which is the room. And on the
  recipe's clips over the whole window, which is what the files hold.

A difference is the two grids' and the levelling's that follows from them:
above the crossover the mirror is the same in both packs, and what remains
there is the gain each pack's seam gave its high side.

Both responses are cut at the same instant, the reference's arrival. The
arrival is read on the whole response, and where the listener does not see
the source it is the low band's, which the grid moves: on the first scene a
hidden voice arrived 16.6 ms earlier on the coarse grid, and a cut that
followed it moved 5 dB of the mirror, identical in both packs, from one side
of the cut to the other. Each side of the crossover stops at the ramp's
edge: the bands inside the ramp hold both the wave solve and the mirror.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.check import measure
from reverberate.render.check.clips import ClipSource
from reverberate.render.check.report import check_pack, write_ears
from reverberate.render.check.run import CheckSettings, _impulse, _steps_audible, _window
from reverberate.render.pack import ScenePack, read_pack, sources_of

__all__ = ["EARLY_S", "difference", "impulse_levels", "run"]

SCHEMA = "reverberate.sound-check.against"
#: Where the response to an impulse is cut, after its arrival: the bound of C50.
EARLY_S = 0.05
#: What is kept before the arrival in the early part: the bank's and the masks' ring.
BEFORE_S = 0.01
#: A band this far under the loudest band of the whole response holds nothing, dB.
FLOOR_DB = -60.0
#: A window whose loudest sample is under this, dB re full scale, is also written louder.
FAINT_DB = -26.0


def impulse_levels(
    pack: ScenePack,
    source_id: str,
    step: int,
    settings: CheckSettings,
    *,
    arrival: int | None = None,
) -> dict[str, Any] | None:
    """Third octave levels of the response to an impulse through a source at ``step``.

    The omnidirectional channel of the whole response, cut ``EARLY_S``
    after its arrival, or after ``arrival`` (a sample of the response, as
    ``arrival_sample`` gives it) when one is given: two packs are compared
    on one cut. ``None`` when the response is silent.
    """
    h = pack.header
    rate, n = h.sample_rate_hz, h.step_samples
    source = next(sources_of(pack, [source_id]))
    sample = step * n + n // 2
    lo = max(sample - int(0.1 * rate), 0)
    reach = max(h.low_samples / h.low_sample_rate_hz if h.has_low else 0.3, 1.2)
    hi = min(sample + int((reach + pack.mirror.lead_s + 0.15) * rate), h.samples)
    parts = _impulse(pack, source, sample, lo, hi, settings)
    full = sum(parts.values())
    if not isinstance(full, np.ndarray) or not np.any(full[0]):
        return None
    w = np.asarray(full[0], dtype=float)
    own = int(measure.envelope_arrival(w))
    arrival = own if arrival is None else int(arrival)
    cut = arrival + int(EARLY_S * rate)
    early = _window(w, arrival - int(BEFORE_S * rate), cut)
    late = _window(w, cut, w.size)
    return {
        "step": int(step),
        "time_s": round(sample / rate, 3),
        "arrival_s": round((own + lo - sample) / rate, 5),
        "arrival_sample": arrival,
        "early_db": measure.third_octave_db(early, rate),
        "late_db": measure.third_octave_db(late, rate),
        "whole_db": measure.third_octave_db(w, rate),
    }


def _less(
    b: np.ndarray,
    a: np.ndarray,
    *,
    top_b: float | None = None,
    top_a: float | None = None,
    floor_db: float = FLOOR_DB,
) -> list[float | None]:
    """``b - a`` band by band; ``None`` where either lies ``floor_db`` under its top.

    A band that holds nothing is not a band the two differ in. The top is
    the loudest band of the whole response when given (``top_a``,
    ``top_b``), so that a part of a response that holds nothing of it, the
    late part of a free field, says nothing at all; the part's own
    loudest band otherwise.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    over_a = float(a.max()) if top_a is None else float(top_a)
    over_b = float(b.max()) if top_b is None else float(top_b)
    held = (a > over_a + floor_db) & (b > over_b + floor_db)
    return [round(float(y - x), 2) if ok else None for x, y, ok in zip(a, b, held, strict=True)]


def _worst(values: list[float | None], low_hz: float, high_hz: float) -> float | None:
    inside = [
        abs(v)
        for v, f in zip(values, measure.THIRD_OCTAVES_HZ, strict=True)
        if v is not None and low_hz <= f <= high_hz
    ]
    return round(max(inside), 2) if inside else None


def difference(
    a: ScenePack,
    b: ScenePack,
    stems_a: dict[str, np.ndarray],
    stems_b: dict[str, np.ndarray],
    first: int,
    stop: int,
    settings: CheckSettings,
) -> dict[str, Any]:
    """For every source both packs hold: B less A per third octave, early, late and on the clips."""
    rate = a.header.sample_rate_hz
    crossover = float(a.crossover.cutoff_hz)
    edge = 2.0 ** (max(float(a.crossover.width_octaves), 1.0 / 3.0) / 2.0)
    found: dict[str, Any] = {}
    need = int(np.ceil(1.4 / a.header.step_s))
    for name in stems_a:
        if name not in stems_b:
            continue
        source = next(sources_of(a, [name]))
        record: dict[str, Any] = {}
        steps = _steps_audible(source, first, stop, min(need, max(stop - first - 1, 1)))
        if steps:
            step = steps[len(steps) // 2]
            settings.say(f"{name}: an impulse at {step * a.header.step_s:.2f} s through both packs")
            one = impulse_levels(a, name, step, settings)
            other = (
                None
                if one is None
                else impulse_levels(b, name, step, settings, arrival=one["arrival_sample"])
            )
            if one is not None and other is not None:
                tops = {
                    "top_a": float(np.max(one["whole_db"])),
                    "top_b": float(np.max(other["whole_db"])),
                }
                record["impulse"] = {
                    "time_s": one["time_s"],
                    "arrival_s": {"a": one["arrival_s"], "b": other["arrival_s"]},
                    "early_db": _less(other["early_db"], one["early_db"], **tops),
                    "late_db": _less(other["late_db"], one["late_db"], **tops),
                }
        heard_a, heard_b = (np.asarray(stems[name], dtype=float) for stems in (stems_a, stems_b))
        if np.any(heard_a) and np.any(heard_b):
            record["clips_db"] = _less(
                measure.third_octave_db(heard_b, rate), measure.third_octave_db(heard_a, rate)
            )
            record["clips_level_db"] = round(
                float(
                    measure.db(float(np.mean(heard_b**2)), power=True)
                    - measure.db(float(np.mean(heard_a**2)), power=True)
                ),
                2,
            )
        # The largest difference each side of the crossover's ramp: where the two packs can
        # differ by their grids, and where they share the mirror.
        record["worst_abs_db"] = {
            part: {
                "under_the_crossover": _worst(values, 0.0, crossover / edge),
                "over_the_crossover": _worst(values, crossover * edge, 1e9),
            }
            for part, values in (
                ("early", record.get("impulse", {}).get("early_db")),
                ("late", record.get("impulse", {}).get("late_db")),
                ("clips", record.get("clips_db")),
            )
            if values is not None
        }
        found[name] = record
    return found


def _table(found: dict[str, Any]) -> list[str]:
    bands = [f"{f:.0f}" if f < 1000 else f"{f / 1000:g}k" for f in measure.THIRD_OCTAVES_HZ]
    lines = []
    for name, record in found.items():
        lines += [f"### {name}", "", "| | " + " | ".join(bands) + " |"]
        lines.append("| --- |" + " --- |" * len(bands))
        rows = [
            ("early", record.get("impulse", {}).get("early_db")),
            ("late", record.get("impulse", {}).get("late_db")),
            ("clips", record.get("clips_db")),
        ]
        for label, values in rows:
            if values is not None:
                cells = ["" if v is None else f"{v:+.1f}" for v in values]
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if "clips_level_db" in record:
            lines += ["", f"The clips over the whole window: {record['clips_level_db']:+.2f} dB."]
        lines.append("")
    return lines


def run(
    pack_a: Path,
    pack_b: Path,
    out: Path,
    *,
    recipe_path: Path | None = None,
    clips_root: Path | None = None,
    manifest: Path | None = None,
    window_s: tuple[float, float] | None = None,
    sources: Iterable[str] | None = None,
    measured_head: Path | None = None,
    settings: CheckSettings | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """``python -m reverberate.render check A.h5 --against B.h5 --out DIR``: the document.

    The two packs must be of one recipe, and of the same steps: they are
    refused otherwise, since what would be heard is two scenes. The
    sources are those both hold, the window the one given.
    """
    started = time.time()
    settings = settings or CheckSettings()
    settings.say = say
    out = Path(out)
    notes: list[str] = []
    if measured_head is not None and not Path(measured_head).is_file():
        notes.append(f"no measured head at {measured_head}: the sphere stands in for the page's")
        measured_head = None
    with read_pack(Path(pack_a)) as a, read_pack(Path(pack_b)) as b:
        if a.header.recipe_sha256 != b.header.recipe_sha256:
            raise SystemExit(
                "the two packs are not of one recipe:"
                f" {a.header.recipe_sha256[:12]} and {b.header.recipe_sha256[:12]}"
            )
        if a.header.steps != b.header.steps:
            raise SystemExit(
                f"the two packs are not of the same window: {a.header.steps} steps and"
                f" {b.header.steps}"
            )
        recipe = None
        if recipe_path is not None:
            recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        both = [name for name in a.sources if name in b.sources]
        if sources is not None:
            both = [name for name in both if name in set(sources)]
        if not both:
            raise SystemExit("the two packs hold no source in common")
        clips = ClipSource(clips_root, manifest)
        outcomes = []
        for label, pack in (("A", a), ("B", b)):
            say(f"{label}: the mix of {len(both)} source(s)")
            outcomes.append(
                check_pack(
                    pack,
                    recipe=recipe,
                    clips=clips,
                    window_s=window_s,
                    sources=both,
                    measured_head=measured_head,
                    sphere_head=True,
                    settings=settings,
                    families=("mix",),
                )
            )
        one, other = outcomes
        rate = a.header.sample_rate_hz
        # One gain for both packs: the page's default, and for a faint window one more set,
        # louder by one figure taken on the louder of the two.
        peak = max(float(np.max(np.abs(o["mix"]["mix_ears"]))) for o in outcomes)
        gains: list[tuple[str, float]] = [("", settings.page_gain)]
        loud = peak * settings.page_gain
        if 0.0 < loud < 10.0 ** (FAINT_DB / 20.0):
            boost = 10.0 * float(np.floor(-(float(measure.db(loud)) + 6.0) / 10.0))
            gains.append((f"_plus{boost:.0f}dB", settings.page_gain * 10.0 ** (boost / 20.0)))
        folder = out / "listen"
        folder.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        clipped: dict[str, int] = {}
        for suffix, gain in gains:
            for label, outcome in (("A", one), ("B", other)):
                signals = {"mix": outcome["mix"]["mix_ears"], **outcome["mix"]["ears"]}
                for name, ears in signals.items():
                    if not np.any(ears):
                        continue
                    key = f"{label}_{name}{suffix}"
                    over = write_ears(folder / f"{key}.wav", ears, rate, gain)
                    files[key] = str(folder / f"{key}.wav")
                    if over:
                        clipped[key] = over
        n = a.header.step_samples
        first, stop = (int(round(t / a.header.step_s)) for t in one["window_s"])
        found = difference(a, b, one["mix"]["omni"], other["mix"]["omni"], first, stop, settings)
        level = {
            label: {
                "peak_db": round(float(measure.db(float(np.max(np.abs(o["mix"]["mix_ears"]))))), 2),
                "rms_db": round(
                    float(measure.db(float(np.mean(o["mix"]["mix_ears"] ** 2)), power=True)), 2
                ),
            }
            for label, o in (("A", one), ("B", other))
        }
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "a": {"pack": str(pack_a), "solver": a.header.provenance.get("solver")},
            "b": {"pack": str(pack_b), "solver": b.header.provenance.get("solver")},
            "recipe_sha256": a.header.recipe_sha256,
            "window_s": list(one["window_s"]),
            "samples": (stop - first) * n,
            "sources": both,
            "decoder": one["decoder"],
            "gain_db": float(measure.db(settings.page_gain)),
            "full_scale_spl_db": float(measure.FULL_SCALE_SPL_DB - measure.db(settings.page_gain)),
            "one_gain_for_both": True,
            "mix_before_gain": level,
            "files": files,
            "clipped_samples": clipped,
            "third_octaves_hz": [round(f, 1) for f in measure.THIRD_OCTAVES_HZ],
            "crossover_hz": float(a.crossover.cutoff_hz),
            "early_s": EARLY_S,
            "difference_b_less_a_db": found,
            "notes": notes,
        }
    out.mkdir(parents=True, exist_ok=True)
    document["seconds"] = round(time.time() - started, 1)
    (out / "ab.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    lines = [
        "# Two packs of one recipe",
        "",
        f"- A: `{pack_a}` ({document['a']['solver']})",
        f"- B: `{pack_b}` ({document['b']['solver']})",
        f"- window {document['window_s'][0]:g} s to {document['window_s'][1]:g} s,"
        f" sources {', '.join(both)}",
        f"- the files are at {document['gain_db']:g} dB for both packs: full scale stands for"
        f" {document['full_scale_spl_db']:g} dB SPL. Nothing is normalised.",
        f"- the mix before that gain, dB re full scale: A peak {level['A']['peak_db']}, rms"
        f" {level['A']['rms_db']}; B peak {level['B']['peak_db']}, rms {level['B']['rms_db']}",
    ]
    if clipped:
        lines.append(
            "- **clipped**, samples past full scale: "
            + ", ".join(f"{name} ({count})" for name, count in clipped.items())
        )
    lines += [f"- {note}" for note in notes]
    lines += [
        "",
        "## B less A, dB, by third octave",
        "",
        f"Early and late are the response to an impulse, cut {EARLY_S * 1e3:g} ms after its"
        " arrival; clips is the recipe's clips over the window. A band left empty holds"
        " nothing in one of the two.",
        "",
        *_table(found),
    ]
    (out / "ab.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"report: {out / 'ab.md'}")
    for name, path in files.items():
        say(f"{name}: {path}" + (f"  CLIPPED, {clipped[name]} samples" if name in clipped else ""))
    return document
