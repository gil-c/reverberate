"""The check from end to end, and what it leaves on disk.

``check.json`` (every result, machine readable), ``check.md`` (the same as
a table, the faults first), ``listen/*.wav`` (the two ears of each source
and of the mix, as the page would play them) and, where ``matplotlib`` is
installed, a few plots. Nothing is written beside the pack.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.check import measure
from reverberate.render.check.binaural import page_decoder
from reverberate.render.check.clips import ClipSource
from reverberate.render.check.run import (
    FAIL,
    INFO,
    LIMITS,
    PASS,
    SKIP,
    WARN,
    CheckSettings,
    Result,
    _moves,
    _plain,
    _steps_audible,
    continuity_of,
    duplicate_arrivals,
    mix_of,
    probe_source,
    reference_field,
    worst,
)
from reverberate.render.pack import ScenePack, read_pack, sources_of

__all__ = ["check_pack", "defaults", "exit_code", "run", "write_report"]

SCHEMA = "reverberate.sound-check"
#: The faults a listener meets first come first in the report.
ORDER = (
    "impulse",
    "audibility",
    "near_voice_seen",
    "level_at_listener",
    "direct_level",
    "direct_band_balance",
    "playback_level",
    "binaural_peak",
    "band_alignment",
    "pre_arrival_energy",
    "late_echo",
    "tone_click",
    "noise_click",
    "level_step",
    "interval_edge",
    "interval_join",
    "zipper",
    "binaural_left_right",
    "binaural_front_back",
    "direction",
    "duplicate_arrivals",
    "seam_third_octaves",
    "late_spectrum_reference",
    "reverberation_reference",
    "reverberation_range",
    "reverberation_plausible",
    "reverberation_spread",
    "c50",
    "arrival_time",
    "doppler",
    "level_distance",
    "silence",
    "dc",
    "speech_to_rest",
    "continuity",
)


def check_pack(
    pack: ScenePack,
    *,
    recipe: Mapping[str, Any] | None = None,
    clips: ClipSource | None = None,
    window_s: tuple[float, float] | None = None,
    sources: Iterable[str] | None = None,
    measured_head: Path | None = None,
    sphere_head: bool = False,
    settings: CheckSettings | None = None,
    families: Iterable[str] = ("pack", "impulse", "continuity", "mix"),
    keep: Callable[[str, int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Every check on ``pack``; the results, and what the files and plots are made from.

    ``measured_head`` is the SOFA file the page's decoder is designed on;
    without it the ears are checked on the analytic sphere when
    ``sphere_head`` is set, and not at all otherwise. ``keep`` is handed the
    mix's stems, order 7, a piece at a time (:func:`.run.mix_of`).
    """
    settings = settings or CheckSettings()
    h = pack.header
    n, rate = h.step_samples, h.sample_rate_hz
    last = (h.steps - 1) * h.step_s
    start_s, stop_s = window_s or (0.0, last)
    first, stop = int(round(max(start_s, 0.0) / h.step_s)), int(round(min(stop_s, last) / h.step_s))
    chosen = list(sources_of(pack, sources))
    decoder = None
    if measured_head is not None or sphere_head:
        settings.say("designing the page's decoder")
        decoder = page_decoder(measured_head, h.order, rate)
    families = tuple(families)
    results: list[Result] = []
    plots: dict[str, Any] = {"impulse": [], "continuity": []}
    need = int(np.ceil((1.2 + 0.2) / h.step_s))
    for source in chosen:
        audible = [k for k in range(first, stop) if source.audible[k]]
        if "pack" in families:
            results.append(duplicate_arrivals(pack, source, audible))
        if "impulse" in families:
            steps = _steps_audible(source, first, stop, min(need, max(stop - first - 1, 1)))
            if not steps:
                results.append(
                    Result("impulse", source.id, SKIP, None, "", "", "", "never audible for 1.4 s")
                )
            else:
                count = settings.probes_moving if _moves(pack, source, steps) else 1
                picks = sorted({steps[int(i)] for i in np.linspace(0, len(steps) - 1, count)})
                for step in picks:
                    settings.say(f"{source.id}: an impulse at {step * h.step_s:.2f} s")
                    made, drawn = probe_source(pack, source, step, settings, decoder)
                    results.extend(made)
                    if drawn:
                        plots["impulse"].append(drawn)
        if "continuity" in families:
            made, drawn = continuity_of(pack, source, first, stop, settings, decoder)
            results.extend(made)
            if drawn:
                plots["continuity"].append(drawn)
    spread = _spread(plots["impulse"])
    if spread is not None:
        results.append(spread)
    mix: dict[str, Any] = {}
    if "mix" in families:
        made, mix = mix_of(
            pack,
            chosen,
            recipe if recipe is not None else pack.recipe_json(),
            clips or ClipSource(),
            first * n,
            stop * n,
            settings,
            decoder,
            keep,
        )
        results.extend(made)
    return {
        "results": results,
        "plots": plots,
        "mix": mix,
        "window_s": (first * h.step_s, stop * h.step_s),
        "sources": [s.id for s in chosen],
        "decoder": None if decoder is None else decoder.record(),
    }


def _spread(probes: list[dict[str, Any]]) -> Result | None:
    """The dwelling's own range: T20 per octave over every probe of the pack, told.

    A source decays as the rooms between it and the head do, so the probes
    of one pack do not agree and are not held to; what they span is what a
    probe far from all the others is read against by whoever reads the
    report.
    """
    held = [probe for probe in probes if "t20_s" in probe and "bank_hz" in probe]
    if len(held) < 2:
        return None
    bank = np.asarray(held[0]["bank_hz"])
    table = np.array([np.asarray(probe["t20_s"], dtype=float) for probe in held])
    middle = (bank >= 250) & (bank <= 4000) & np.any(np.isfinite(table), axis=0)
    if not np.any(middle):
        return None
    least = np.nanmin(table[:, middle], axis=0)
    most = np.nanmax(table[:, middle], axis=0)
    centre = np.nanmedian(table[:, middle], axis=0)
    told = ", ".join(
        f"{int(f)} Hz {a:.2f} to {b:.2f} s (median {c:.2f})"
        for f, a, b, c in zip(bank[middle], least, most, centre, strict=True)
    )
    return Result(
        "reverberation_spread",
        "every probe",
        INFO,
        float(100.0 * np.max(most / least - 1.0)),
        "per cent",
        "told, not judged",
        "What the sources of one dwelling span between them, the longest T20 over the "
        "shortest in the octave where they differ the most: a source's decay is its rooms'.",
        f"T20 over the {len(held)} probes: {told}",
        {
            "bands_hz": bank[middle],
            "least_s": least,
            "most_s": most,
            "median_s": centre,
            "probes": [str(probe.get("name", "")) for probe in held],
        },
    )


def _key(result: Result) -> tuple[int, int, str, str]:
    rank = {FAIL: 0, WARN: 1, PASS: 2, INFO: 3, SKIP: 4}[result.status]
    name = result.test.split(":")[0]
    return rank, ORDER.index(name) if name in ORDER else len(ORDER), result.test, result.source


def _number(result: Result) -> str:
    if result.value is None:
        return ""
    return (
        f"{result.value:+.2f} {result.unit}" if np.isfinite(result.value) else f"none {result.unit}"
    )


def _table(results: list[Result]) -> list[str]:
    lines = [
        "| verdict | test | source | number | held to | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.status} | `{r.test}` | {r.source} | {_number(r)} | {r.threshold} | {r.note} |"
        )
    return lines


def write_report(target: Path, outcome: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """``check.json`` and ``check.md`` in ``target``; the JSON document, returned."""
    target.mkdir(parents=True, exist_ok=True)
    results: list[Result] = sorted(outcome["results"], key=_key)
    counts = {s: sum(r.status == s for r in results) for s in (FAIL, WARN, PASS, INFO, SKIP)}
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "verdict": worst(results),
        "counts": counts,
        **header,
        "window_s": list(outcome["window_s"]),
        "sources": outcome["sources"],
        "decoder": outcome["decoder"],
        "limits": {
            name: {
                "pass": lim.ok,
                "warn": lim.warn,
                "unit": lim.unit,
                "held": lim.text(),
                "why": lim.reason,
            }
            for name, lim in LIMITS.items()
        },
        "levels": {
            name: {k: v for k, v in row.items() if k != "fast_dba"}
            for name, row in outcome["mix"].get("levels", {}).items()
        },
        "files": header.get("files", {}),
        "results": [r.record() for r in results],
    }
    document = _plain(document)
    (target / "check.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    lines = [
        f"# Sound check: {header.get('pack', '')}",
        "",
        f"Verdict **{document['verdict']}**: "
        + ", ".join(f"{count} {status}" for status, count in counts.items())
        + ".",
        "",
        f"Window {outcome['window_s'][0]:.2f} to {outcome['window_s'][1]:.2f} s; sources "
        + ", ".join(outcome["sources"])
        + f"; decoder: {(outcome['decoder'] or {}).get('head', 'none, the ears are not checked')}.",
        "",
        "## What a listener would meet, the worst first",
        "",
        *_table([r for r in results if r.status in (FAIL, WARN)]),
        "",
        "## Levels at the listener",
        "",
        "| source | kind | distance | dB SPL | dBA | fed at 1 m | free field | loudest 125 ms "
        "| fed with |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in outcome["mix"].get("levels", {}).items():
        lines.append(
            f"| {name} | {row['kind']} | {row['distance_m']:.2f} m | {row['spl_db']:.1f} | "
            f"{row['spl_dba']:.1f} | {row['fed_at_1m_db']:.1f} | {row['free_field_db']:.1f} | "
            f"{row['fast_max_dba']:.1f} dBA | {row['fed']} |"
        )
    lines += ["", "## Files to hear", ""]
    for name, path in (header.get("files") or {}).items():
        lines.append(f"- {name}: `{path}`")
    lines += ["", "## Every result", "", *_table(results), "", "## The limits and why", ""]
    for name, lim in LIMITS.items():
        lines.append(f"- `{name}`: {lim.text()}. {lim.reason}")
    for r in results:
        if r.test not in LIMITS and r.reason and f"- `{r.test}`" not in "\n".join(lines):
            lines.append(f"- `{r.test}`: {r.threshold}. {r.reason}")
    (target / "check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(document)


def write_ears(path: Path, ears: np.ndarray, rate: float, scale: float) -> int:
    """Two ears (``[ear, sample]``) at ``scale`` as a 24 bit WAV; the samples past full scale.

    A file cannot hold more than full scale: what passes it is clipped, as
    the page's output is, and counted, so that a file that is not the
    scene's sound says so where it is listed.
    """
    import soundfile

    data = np.asarray(ears, dtype=float).T * float(scale)
    over = int(np.count_nonzero(np.abs(data) > 1.0))
    soundfile.write(
        str(path), np.clip(data, -1.0, 1.0), int(round(rate)), subtype="PCM_24", format="WAV"
    )
    return over


def _write_wavs(
    target: Path,
    mix: dict[str, Any],
    rate: float,
    gain: float,
    clipped: dict[str, int] | None = None,
) -> dict[str, str]:
    """The two ears of each source and of the mix, 24 bit, at ``gain``; and louder if faint.

    ``clipped`` receives, by file, the samples that passed full scale and
    were clipped; a file not named there holds the scene's sound as it is.
    """
    folder = target / "listen"
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    signals = {"mix": mix["mix_ears"], **mix["ears"]}
    peak = float(np.max(np.abs(mix["mix_ears"]))) * gain
    sets: list[tuple[str, float]] = [("", gain)]
    if 0.0 < peak < 10.0 ** (-26.0 / 20.0):
        # One gain for the whole set, so that the sources stay comparable.
        boost = 10.0 * float(np.floor(-(float(measure.db(peak)) + 6.0) / 10.0))
        sets.append((f"_plus{boost:.0f}dB", gain * 10.0 ** (boost / 20.0)))
    for suffix, scale in sets:
        for name, ears in signals.items():
            if not np.any(ears):
                continue
            path = folder / f"{name}{suffix}.wav"
            over = write_ears(path, ears, rate, scale)
            if over and clipped is not None:
                clipped[f"{name}{suffix}"] = over
            files[f"{name}{suffix}"] = str(path)
    return files


def _write_plots(target: Path, outcome: dict[str, Any]) -> dict[str, str]:
    """A few pictures, where matplotlib is installed; nothing otherwise."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt: Any = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return {}
    files: dict[str, str] = {}
    centres = np.asarray(measure.THIRD_OCTAVES_HZ)
    probes = outcome["plots"]["impulse"]
    if probes:
        figure, axes = plt.subplots(len(probes), 2, figsize=(12, 3.4 * len(probes)), squeeze=False)
        for row, probe in zip(axes, probes, strict=True):
            for label, key in (
                ("whole", "whole_db"),
                ("first 50 ms", "early_db"),
                ("after", "late_db"),
            ):
                row[0].semilogx(centres, probe[key], marker="o", ms=3, label=label)
            if "reference_late_db" in probe:
                shift = np.mean(probe["late_db"][8:20]) - np.mean(probe["reference_late_db"][8:20])
                row[0].semilogx(
                    centres,
                    probe["reference_late_db"] + shift,
                    "k--",
                    label="validated field, after",
                )
            row[0].axvspan(707, 1414, color="0.9")
            row[0].set(title=f"{probe['name']}: third octaves, W", xlabel="Hz", ylabel="dB")
            row[0].legend(fontsize=7)
            rate, start = probe["rate"], probe["arrival"] - int(0.02 * probe["rate"])
            w = probe["response"][max(start, 0) :]
            row[1].plot(
                np.arange(w.size) / rate * 1e3 - 20.0, measure.db(np.abs(w) + 1e-12), lw=0.4
            )
            if "reference_response" in probe:
                ref = probe["reference_response"][
                    max(probe["reference_arrival"] - int(0.02 * rate), 0) :
                ]
                scale = np.max(np.abs(w)) / np.max(np.abs(ref))
                row[1].plot(
                    np.arange(ref.size) / rate * 1e3 - 20.0,
                    measure.db(np.abs(ref) * scale + 1e-12),
                    lw=0.3,
                    alpha=0.5,
                    label="validated field (other source), peaks matched",
                )
                row[1].legend(fontsize=7)
            row[1].set(
                title="response, ms from the first arrival",
                xlim=(-20, 400),
                ylim=(measure.db(np.max(np.abs(w))) - 80, measure.db(np.max(np.abs(w))) + 3),
            )
        figure.tight_layout()
        files["impulse"] = str(target / "impulse.png")
        figure.savefig(files["impulse"], dpi=110)
        plt.close(figure)
    runs = outcome["plots"]["continuity"]
    if runs:
        figure, axes = plt.subplots(len(runs), 1, figsize=(12, 2.6 * len(runs)), squeeze=False)
        for axis, probe in zip(axes[:, 0], runs, strict=True):
            t = probe["start_s"] + np.arange(probe["residual"].size) / probe["rate"]
            amplitude = np.sqrt(2.0 * np.mean(probe["tone"] ** 2))
            axis.plot(t, measure.db(probe["residual"] / amplitude + 1e-15), lw=0.3)
            axis.axhline(LIMITS["tone_click"].ok, color="g", lw=0.6)
            axis.axhline(LIMITS["tone_click"].warn, color="r", lw=0.6)
            axis.set(title=f"{probe['name']}: what is not the 2.5 kHz tone, W", ylabel="dB re tone")
        figure.tight_layout()
        files["continuity"] = str(target / "continuity.png")
        figure.savefig(files["continuity"], dpi=110)
        plt.close(figure)
    levels = outcome["mix"].get("levels", {})
    if levels:
        figure, axis = plt.subplots(figsize=(12, 3.5))
        for name, row in levels.items():
            t = outcome["window_s"][0] + 0.125 * np.arange(row["fast_dba"].size)
            axis.plot(t, np.maximum(row["fast_dba"], 0.0), lw=0.8, label=name)
        axis.set(title="level at the listener, W, 125 ms", xlabel="s", ylabel="dBA")
        axis.legend(fontsize=7)
        figure.tight_layout()
        files["levels"] = str(target / "levels.png")
        figure.savefig(files["levels"], dpi=110)
        plt.close(figure)
    return files


def run(
    pack_path: Path,
    out: Path,
    *,
    recipe_path: Path | None = None,
    clips_root: Path | None = None,
    manifest: Path | None = None,
    window_s: tuple[float, float] | None = None,
    sources: Iterable[str] | None = None,
    measured_head: Path | None = None,
    reference: Path | None = None,
    settings: CheckSettings | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """``python -m reverberate.render check``: read, render, measure, write. The document."""
    started = time.time()
    settings = settings or CheckSettings()
    settings.say = say
    out = Path(out)
    with read_pack(Path(pack_path)) as pack:
        recipe = None
        notes = []
        if recipe_path is not None:
            recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        if reference is not None:
            settings.reference = reference_field(reference)
            if settings.reference is None:
                notes.append(f"no reference field at {reference}: the decay is not compared")
            elif settings.reference["scene_id"] != pack.header.scene_id:
                notes.append("the reference field is another scene's: the decay is not compared")
        if measured_head is not None and not Path(measured_head).is_file():
            notes.append(
                f"no measured head at {measured_head}: the sphere stands in for the page's"
            )
            measured_head = None
        outcome = check_pack(
            pack,
            recipe=recipe,
            clips=ClipSource(clips_root, manifest),
            window_s=window_s,
            sources=sources,
            measured_head=measured_head,
            sphere_head=True,
            settings=settings,
        )
        files: dict[str, str] = {}
        clipped: dict[str, int] = {}
        if outcome["mix"]:
            files.update(
                _write_wavs(
                    out, outcome["mix"], pack.header.sample_rate_hz, settings.page_gain, clipped
                )
            )
            if clipped:
                notes.append(
                    "files whose samples passed full scale and were clipped, with how many: "
                    + ", ".join(f"{name} ({count})" for name, count in clipped.items())
                )
        out.mkdir(parents=True, exist_ok=True)
        files.update({f"plot: {k}": v for k, v in _write_plots(out, outcome).items()})
        header = {
            "pack": str(pack_path),
            "recipe_sha256": pack.header.recipe_sha256,
            "profile": pack.header.profile,
            "recipe": None if recipe_path is None else str(recipe_path),
            "clips_root": None if clips_root is None else str(clips_root),
            "page_gain_db": float(measure.db(settings.page_gain)),
            # What the output's full scale stands for at that level, at 1 m.
            "full_scale_spl_db": float(measure.FULL_SCALE_SPL_DB - measure.db(settings.page_gain)),
            "clipped_samples": clipped,
            "notes": notes,
            "files": files,
        }
        document = write_report(out, outcome, header)
    document["seconds"] = round(time.time() - started, 1)
    say(f"verdict {document['verdict']}: {document['counts']}")
    say(f"report: {out / 'check.md'}")
    for name, path in files.items():
        say(f"{name}: {path}" + (f"  CLIPPED, {clipped[name]} samples" if name in clipped else ""))
    return document


def defaults(
    clips_root: Path | None,
    manifest: Path | None,
    measured_head: Path | None,
    reference: Path | None,
) -> dict[str, Path | None]:
    """What the command reads when it is not told: the data root's files, the package's manifest."""
    from reverberate.settings import data_root

    root = data_root()
    return {
        "clips_root": clips_root or root / "clips",
        "manifest": manifest
        or Path(__file__).parents[2] / "scenes" / "library" / "clarify_v1.json",
        "measured_head": measured_head or root / "raw" / "hrtf" / "HRIR_L2702.sofa",
        "reference": reference
        or root / "runs" / "w42_gpu_hssd_0076" / "field_hybrid_c10" / "S1.h5",
    }


def exit_code(document: Mapping[str, Any]) -> int:
    """0 unless something failed."""
    return 1 if document.get("verdict") == FAIL else 0
