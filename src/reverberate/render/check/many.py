"""Several packs of one scene: a file to hear for each, what each differs by, and a blind set.

:mod:`reverberate.render.check.against` puts two packs side by side. This
is the same for a reference and any number of variants of it (the
listening kit of ``python -m reverberate.trace variants``): the decision is
taken by ear, among all of them, and nothing here judges.

- **The files** (``listen/<name>_mix.wav`` and each source's): the same
  window, the same clips, the same head, through the page's decode, **at
  one gain for every pack**. A pack is named by the ``variant.json`` beside
  it (:mod:`reverberate.render.variant`), or as it is given.
- **The tables** (``variants.json``, ``variants.md``): for every variant,
  its level less the reference's in each third octave and for every
  source, on the early and the late part of the response to an impulse and
  on the clips, as :func:`reverberate.render.check.against.difference`
  reads them; and the largest of those each side of the crossover, which
  is the low band (the wave solve, where the variants of a grid, of a
  rail or of a response's length differ) and the high band (the mirror,
  where the variants of the rays do).
- **The blind set** (``blind/X1.wav`` ...): the mixes again under names
  that say nothing, in an order drawn at random, and ``blind/key.sealed``,
  which says which is which and is not readable at a glance: ``python -m
  reverberate.render unseal blind/key.sealed`` prints it once the owner
  has written down what he heard.

**Which packs may be compared.** Those of one recipe; and those of two
recipes that are one scene solved at another rail pitch, which a variant of
the rails is: the same steps, the same sources at the same places, the same
head. Anything else is refused, since what would be heard is two scenes.
"""

from __future__ import annotations

import base64
import json
import secrets
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.check import measure
from reverberate.render.check.against import EARLY_S, FAINT_DB, _table, difference
from reverberate.render.check.clips import ClipSource
from reverberate.render.check.report import check_pack, write_ears
from reverberate.render.check.run import CheckSettings
from reverberate.render.pack import ScenePack, read_pack
from reverberate.render.variant import label_of, summary

__all__ = ["SCHEMA", "blind_set", "run", "same_scene", "unseal"]

SCHEMA = "reverberate.sound-check.variants"
#: Two packs' sources and heads are at the same places to this, m.
SAME_PLACE_M = 1e-6


def same_scene(a: ScenePack, b: ScenePack) -> str | None:
    """Why ``b`` is not ``a``'s scene, or ``None``; one recipe, or one scene's movements."""
    if a.header.steps != b.header.steps or a.header.step_s != b.header.step_s:
        return f"not the same window: {a.header.steps} steps and {b.header.steps}"
    if a.header.recipe_sha256 == b.header.recipe_sha256:
        return None
    if sorted(a.sources) != sorted(b.sources):
        return "another recipe, and not the same sources"
    moved = float(np.abs(np.asarray(a.listener.position) - np.asarray(b.listener.position)).max())
    for name in a.sources:
        there = np.asarray(a.sources[name].position) - np.asarray(b.sources[name].position)
        moved = max(moved, float(np.abs(there).max()))
    if moved > SAME_PLACE_M:
        return f"another recipe, whose head or sources are elsewhere by {moved:.3g} m"
    return None


def blind_set(files: dict[str, Path], folder: Path, *, seed: int | None = None) -> dict[str, Any]:
    """Copies of ``files`` (by name) under names that say nothing, and the sealed key.

    The order is drawn from ``seed``, or from the system's randomness. The
    key is the JSON of ``{"X1": name, ...}`` in base64: not a secret, a
    thing that is not read by accident.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(files)
    rng = np.random.default_rng(secrets.randbits(63) if seed is None else int(seed))
    order = [names[int(i)] for i in rng.permutation(len(names))]
    key = {}
    for index, name in enumerate(order, start=1):
        hidden = f"X{index}"
        shutil.copyfile(files[name], folder / f"{hidden}.wav")
        key[hidden] = name
    sealed = base64.b64encode(json.dumps(key, sort_keys=True).encode()).decode()
    (folder / "key.sealed").write_text(sealed + "\n", encoding="utf-8")
    (folder / "README.txt").write_text(
        "The same window of the same scene, one file a variant, at one gain, in an order\n"
        "drawn at random. Write down what you hear, then read the key:\n"
        "    python -m reverberate.render unseal key.sealed\n",
        encoding="utf-8",
    )
    return {"files": [f"X{i}" for i in range(1, len(order) + 1)], "key": str(folder / "key.sealed")}


def unseal(path: Path) -> dict[str, str]:
    """Which variant each file of a blind set is."""
    return dict(json.loads(base64.b64decode(Path(path).read_text(encoding="utf-8").strip())))


def _worst_by_band(found: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Over the sources: the largest difference of each part, under and over the crossover."""
    out: dict[str, dict[str, float | None]] = {}
    for record in found.values():
        for part, sides in record.get("worst_abs_db", {}).items():
            held = out.setdefault(part, {})
            for side, value in sides.items():
                if value is not None and (held.get(side) is None or value > (held[side] or 0.0)):
                    held[side] = value
                held.setdefault(side, None)
    return out


def run(
    reference: Path,
    others: Iterable[Path],
    out: Path,
    *,
    names: list[str] | None = None,
    recipe_path: Path | None = None,
    clips_root: Path | None = None,
    manifest: Path | None = None,
    window_s: tuple[float, float] | None = None,
    sources: Iterable[str] | None = None,
    measured_head: Path | None = None,
    settings: CheckSettings | None = None,
    blind_seed: int | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """``python -m reverberate.render check REF.h5 --against V1.h5 V2.h5 ... --out DIR``.

    ``names`` names the packs, the reference first; left out, each is named
    by its ``variant.json`` or by where it lies.
    """
    started = time.time()
    settings = settings or CheckSettings()
    settings.say = say
    out = Path(out)
    paths = [Path(reference), *(Path(p) for p in others)]
    if names is not None and (len(names) != len(paths) or len(set(names)) != len(names)):
        raise SystemExit(f"{len(paths)} packs want {len(paths)} names, each once")
    taken: set[str] = set()
    labels = list(names) if names is not None else [label_of(path, taken) for path in paths]
    notes: list[str] = []
    if measured_head is not None and not Path(measured_head).is_file():
        notes.append(f"no measured head at {measured_head}: the sphere stands in for the page's")
        measured_head = None
    recipe = None
    if recipe_path is not None:
        recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    clips = ClipSource(clips_root, manifest)
    folder = out / "listen"
    folder.mkdir(parents=True, exist_ok=True)
    with read_pack(paths[0]) as ref:
        rate, n = ref.header.sample_rate_hz, ref.header.step_samples
        both = list(ref.sources)
        opened = []
        for label, path in zip(labels[1:], paths[1:], strict=True):
            with read_pack(path) as pack:
                why = same_scene(ref, pack)
                if why is not None:
                    raise SystemExit(f"{label} is not the reference's scene: {why}")
                if pack.header.recipe_sha256 != ref.header.recipe_sha256:
                    notes.append(
                        f"{label} is of another recipe ({pack.header.recipe_sha256[:12]}) with"
                        " the reference's movements: the same scene, its rails solved elsewhere"
                    )
                both = [name for name in both if name in pack.sources]
            opened.append(path)
        if sources is not None:
            both = [name for name in both if name in set(sources)]
        if not both:
            raise SystemExit("the packs hold no source in common")

        def mixed(label: str, pack: ScenePack) -> dict[str, Any]:
            say(f"{label}: the mix of {len(both)} source(s)")
            return check_pack(
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

        one = mixed(labels[0], ref)
        first, stop = (int(round(t / ref.header.step_s)) for t in one["window_s"])
        # A pack at a time: its ears are kept for the files, and nothing else of it.
        ears: dict[str, dict[str, np.ndarray]] = {
            labels[0]: {"mix": one["mix"]["mix_ears"], **one["mix"]["ears"]}
        }
        variants: dict[str, Any] = {}
        for label, path in zip(labels[1:], paths[1:], strict=True):
            with read_pack(path) as pack:
                other = mixed(label, pack)
                ears[label] = {"mix": other["mix"]["mix_ears"], **other["mix"]["ears"]}
                found = difference(
                    ref, pack, one["mix"]["omni"], other["mix"]["omni"], first, stop, settings
                )
                variants[label] = {
                    "pack": str(path),
                    "solver": pack.header.provenance.get("solver"),
                    "variant": summary(path, dict(pack.header.provenance)),
                    "less_the_reference_db": found,
                    "worst_abs_db": _worst_by_band(found),
                }
        # One gain for every pack: the page's, and for a faint window one more set, louder
        # by one figure taken on the loudest of them all.
        peak = max(float(np.max(np.abs(held["mix"]))) for held in ears.values())
        gains: list[tuple[str, float]] = [("", settings.page_gain)]
        loud = peak * settings.page_gain
        if 0.0 < loud < 10.0 ** (FAINT_DB / 20.0):
            boost = 10.0 * float(np.floor(-(float(measure.db(loud)) + 6.0) / 10.0))
            gains.append((f"_plus{boost:.0f}dB", settings.page_gain * 10.0 ** (boost / 20.0)))
        files: dict[str, str] = {}
        clipped: dict[str, int] = {}
        for suffix, gain in gains:
            for label, held in ears.items():
                for name, signal in held.items():
                    if not np.any(signal):
                        continue
                    key = f"{label}_{name}{suffix}"
                    over = write_ears(folder / f"{key}.wav", signal, rate, gain)
                    files[key] = str(folder / f"{key}.wav")
                    if over:
                        clipped[key] = over
        # The blind set is of the set heard at the gain a faint window needs, when there is one.
        suffix = gains[-1][0]
        blind = blind_set(
            {label: Path(files[f"{label}_mix{suffix}"]) for label in ears},
            out / "blind",
            seed=blind_seed,
        )
        level = {
            label: {
                "peak_db": round(float(measure.db(float(np.max(np.abs(held["mix"]))))), 2),
                "rms_db": round(float(measure.db(float(np.mean(held["mix"] ** 2)), power=True)), 2),
            }
            for label, held in ears.items()
        }
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "reference": {
                "name": labels[0],
                "pack": str(paths[0]),
                "solver": ref.header.provenance.get("solver"),
                "variant": summary(paths[0], dict(ref.header.provenance)),
            },
            "variants": variants,
            "recipe_sha256": ref.header.recipe_sha256,
            "window_s": list(one["window_s"]),
            "samples": (stop - first) * n,
            "sources": both,
            "decoder": one["decoder"],
            "gain_db": float(measure.db(settings.page_gain)),
            "full_scale_spl_db": float(measure.FULL_SCALE_SPL_DB - measure.db(settings.page_gain)),
            "one_gain_for_all": True,
            "mix_before_gain": level,
            "files": files,
            "clipped_samples": clipped,
            "blind": blind,
            "third_octaves_hz": [round(f, 1) for f in measure.THIRD_OCTAVES_HZ],
            "crossover_hz": float(ref.crossover.cutoff_hz),
            "early_s": EARLY_S,
            "notes": notes,
        }
    out.mkdir(parents=True, exist_ok=True)
    document["seconds"] = round(time.time() - started, 1)
    (out / "variants.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    (out / "variants.md").write_text("\n".join(_report(document)) + "\n", encoding="utf-8")
    say(f"report: {out / 'variants.md'}")
    say(f"blind set: {out / 'blind'} ({len(blind['files'])} files, key sealed)")
    for name, written in files.items():
        say(
            f"{name}: {written}"
            + (f"  CLIPPED, {clipped[name]} samples" if name in clipped else "")
        )
    return document


def _usd(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f}"


def _report(document: dict[str, Any]) -> list[str]:
    reference = document["reference"]
    lines = [
        "# Variants of one scene",
        "",
        f"- window {document['window_s'][0]:g} s to {document['window_s'][1]:g} s,"
        f" sources {', '.join(document['sources'])}",
        f"- the files are at {document['gain_db']:g} dB for every pack: full scale stands for"
        f" {document['full_scale_spl_db']:g} dB SPL. Nothing is normalised.",
        "- the blind set is `blind/`: the mixes in an order drawn at random; `python -m"
        " reverberate.render unseal blind/key.sealed` says which is which.",
    ]
    if document["clipped_samples"]:
        lines.append(
            "- **clipped**, samples past full scale: "
            + ", ".join(f"{name} ({count})" for name, count in document["clipped_samples"].items())
        )
    lines += [f"- {note}" for note in document["notes"]]
    lines += [
        "",
        "| pack | flags | whole scene, USD predicted | excerpt, USD predicted | measured, USD |"
        " mix peak, dB | mix rms, dB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [(reference["name"], reference), *document["variants"].items()]
    for name, record in rows:
        told = record["variant"]
        flags = ", ".join(f"{k}={v}" for k, v in told["flags"].items()) or (
            "the reference" if name == reference["name"] else ""
        )
        level = document["mix_before_gain"][name]
        measured = (told["measured"] or {}).get("usd")
        lines.append(
            f"| {name} | {flags} | {_usd(told['predicted_scene_usd'])} |"
            f" {_usd(told['predicted_excerpt_usd'])} | {_usd(measured)} |"
            f" {level['peak_db']} | {level['rms_db']} |"
        )
    lines += [
        "",
        "## The largest difference from the reference, dB",
        "",
        f"Over the sources and the third octaves, each side of the crossover"
        f" ({document['crossover_hz']:g} Hz): under it the wave solve, over it the mirror.",
        "",
        "| variant | early, low | early, high | late, low | late, high | clips, low |"
        " clips, high |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, record in document["variants"].items():
        worst = record["worst_abs_db"]
        cells = [
            _one(worst.get(part, {}).get(side))
            for part in ("early", "late", "clips")
            for side in ("under_the_crossover", "over_the_crossover")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name, record in document["variants"].items():
        lines += [
            "",
            f"## {name} less {reference['name']}, dB, by third octave",
            "",
            f"Early and late are the response to an impulse, cut {document['early_s'] * 1e3:g} ms"
            " after its arrival; clips is the recipe's clips over the window. A band left"
            " empty holds nothing in one of the two.",
            "",
            *_table(record["less_the_reference_db"]),
        ]
    return lines


def _one(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"
