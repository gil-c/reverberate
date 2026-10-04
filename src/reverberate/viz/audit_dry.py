"""The dry audio the audit feeds the signal engine, per source of a pack.

The engine reads a source's clips off the pack's recipe through a loader
(:class:`reverberate.render.dry.DryTrack`). The audit gives it that loader:

- **the recipe's own clip** where a clip library is on this disk:
  ``<clips root>/<library>/<name>.wav`` (or ``.flac``), refused when the
  recipe pins a digest and the file's is another;
- **a placeholder, said to be one**, wherever the clip is missing, the
  recipe names the ``placeholder`` library, or the recipe gives the source no
  activity at all (a synthetic pack): for a voice, one of the anechoic EARS
  passages the app already holds under ``<data root>/voices``, looped, and
  modulated noise where there are none; for anything else, shaped noise. So a
  scene can be listened to before a pinned library exists.

A recipe that gives no source any activity is a synthetic one: its sources
sound wherever the pack says they are audible. An interval longer than
:data:`PIECE_S` is cut into pieces, because the engine filters a whole
interval at once and holds it: a noise that runs twenty minutes would be
460 MB a filter. The cut is exact, the filters being linear and each piece
ringing out in full.

What a source is fed is part of its stem's identity: :func:`dry_plan` gives
the digest the cache keys on, which moves with the recipe's intervals, the
files' bytes and this module's version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.dry import ClipLoader, DryTrack
from reverberate.render.pack import ScenePack

__all__ = ["PIECE_S", "PLACEHOLDER", "DryPlan", "DrySources", "dry_plan", "dry_track"]

#: The library name of a clip this module made up.
PLACEHOLDER = "audit-placeholder"
#: Bumped whenever a placeholder's samples change: it is in every digest.
PLACEHOLDER_VERSION = 1
#: The longest piece of an interval handed to the engine at once, seconds.
PIECE_S = 20.0
#: Length and level of the noise loop, and of the stand-in for a voice.
NOISE_LOOP_S = 20.0
NOISE_RMS = 0.05
#: Faded at each end of a looped file, seconds, so the loop does not click.
LOOP_FADE_S = 0.01

RATE_HZ = 48000.0
_VOICE_KINDS = ("near_voice", "far_voice")


@dataclass(frozen=True)
class DrySources:
    """Where dry audio is looked for. Either may be ``None`` or missing."""

    clips_root: Path | None = None
    voices_root: Path | None = None

    def record(self) -> dict[str, str | None]:
        return {
            "clips_root": None if self.clips_root is None else str(self.clips_root),
            "voices_root": None if self.voices_root is None else str(self.voices_root),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, str | None]) -> DrySources:
        clips, voices = record.get("clips_root"), record.get("voices_root")
        return cls(Path(clips) if clips else None, Path(voices) if voices else None)


@dataclass(frozen=True)
class DryPlan:
    """What one source is fed: the intervals as the engine will read them."""

    source_id: str
    #: The source's entry of the recipe, its activity rewritten.
    source: Mapping[str, Any]
    #: What the page prints beside the source: ``"clips"`` or ``"placeholder: ..."``.
    label: str
    placeholder: bool
    digest: str


def _file_sha256(path: Path) -> str:
    return _file_sha256_at(str(path), path.stat().st_size, path.stat().st_mtime_ns)


@lru_cache(maxsize=256)
def _file_sha256_at(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_file(sources: DrySources, clip: Mapping[str, Any]) -> Path | None:
    if sources.clips_root is None:
        return None
    library, name = str(clip.get("library", "")), str(clip.get("name", ""))
    if not library or not name or "/" in library or ".." in name:
        return None
    for suffix in (".wav", ".flac"):
        path = sources.clips_root / library / f"{name}{suffix}"
        if path.is_file():
            return path
    return None


def _voice_files(sources: DrySources) -> list[Path]:
    if sources.voices_root is None or not sources.voices_root.is_dir():
        return []
    return sorted(sources.voices_root.glob("*.wav"))


def _pick(source_id: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:4], "little") % count


def _audible_intervals(pack: ScenePack, source_id: str) -> list[tuple[float, float]]:
    """The runs of steps the pack calls audible, in seconds, inside the scene."""
    h = pack.header
    heard = np.asarray(pack.sources[source_id].audible, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], heard, [False]]).astype(np.int8)))
    last = (h.steps - 1) * h.step_s
    runs = []
    for first, stop in zip(edges[::2], edges[1::2], strict=True):
        start, end = float(first * h.step_s), min(float(stop * h.step_s), last)
        if end > start:
            runs.append((start, end))
    return runs


def dry_plan(pack: ScenePack, source_id: str, sources: DrySources) -> DryPlan:
    """What ``source_id`` is fed, decided from the recipe and what is on this disk."""
    recipe = {s["id"]: s for s in pack.recipe_json().get("sources", [])}
    entry = dict(recipe.get(source_id, {"id": source_id}))
    kind = pack.sources[source_id].kind
    given = list(entry.get("activity", []))
    if not any(s.get("activity") for s in recipe.values()):
        given = [
            {"start_s": a, "end_s": b, "gain_db": 0.0}
            for a, b in _audible_intervals(pack, source_id)
        ]
    voices = _voice_files(sources)
    voice = voices[_pick(source_id, len(voices))] if voices and kind in _VOICE_KINDS else None
    if voice is not None:
        stand_in = {"library": PLACEHOLDER, "name": f"voice:{voice.stem}", "file": str(voice)}
        stand_in["sha256"] = _file_sha256(voice)
        what = f"placeholder: EARS {voice.stem}, looped"
    else:
        shape = "speech" if kind in _VOICE_KINDS else "noise"
        stand_in = {"library": PLACEHOLDER, "name": f"{shape}:{source_id}"}
        what = "placeholder: modulated noise" if shape == "speech" else "placeholder: shaped noise"

    pieces: list[dict[str, Any]] = []
    real = made = 0
    played = 0.0  # seconds of stand-in used so far: the loop goes on where it stopped
    for interval in given:
        start, end = float(interval["start_s"]), float(interval["end_s"])
        clip = interval.get("clip") or {}
        path = None if clip.get("library") in (None, "placeholder") else _clip_file(sources, clip)
        offset = float(interval.get("clip_offset_s", 0.0))
        if path is not None:
            wanted = clip.get("sha256")
            if wanted and len(str(wanted)) == 64 and _file_sha256(path) != wanted:
                raise ValueError(f"{path} is not the clip the recipe pins for {source_id!r}")
            chosen: dict[str, Any] = {**clip, "file": str(path), "sha256": _file_sha256(path)}
            real += 1
        else:
            chosen, offset = dict(stand_in), played
            played += end - start
            made += 1
        count = max(1, int(np.ceil((end - start) / PIECE_S - 1e-9)))
        for index in range(count):
            a = start + index * PIECE_S
            b = min(end, a + PIECE_S)
            at = offset + (a - start)
            piece_clip = dict(chosen)
            if chosen["library"] == PLACEHOLDER:
                # The loader cuts the piece out of the loop: it starts at the clip's zero.
                piece_clip["offset_s"] = at
                piece_clip["seconds"] = b - a
                at = 0.0
            pieces.append(
                {
                    "start_s": a,
                    "end_s": b,
                    "clip": piece_clip,
                    "clip_offset_s": at,
                    "gain_db": float(interval.get("gain_db", 0.0)),
                }
            )
    entry["activity"] = pieces
    label = "clips" if not made else (what if not real else f"clips, and {what}")
    text = json.dumps(
        {"version": PLACEHOLDER_VERSION, "piece_s": PIECE_S, "kind": kind, "activity": pieces},
        sort_keys=True,
    )
    # The paths are where the bytes were found, not what they are.
    identity = json.dumps(_without(json.loads(text), "file"), sort_keys=True)
    return DryPlan(
        source_id=source_id,
        source=entry,
        label=label,
        placeholder=bool(made),
        digest=hashlib.sha256(identity.encode()).hexdigest(),
    )


def _without(tree: Any, key: str) -> Any:
    if isinstance(tree, dict):
        return {k: _without(v, key) for k, v in tree.items() if k != key}
    if isinstance(tree, list):
        return [_without(v, key) for v in tree]
    return tree


def _read(path: str) -> tuple[np.ndarray, float]:
    import soundfile

    samples, rate = soundfile.read(path, dtype="float64", always_2d=True)
    return np.ascontiguousarray(samples.mean(axis=1)), float(rate)


@lru_cache(maxsize=4)
def _voice_loop(path: str) -> np.ndarray:
    """A voice file at the output rate, faded at its ends so that it loops."""
    samples, rate = _read(path)
    if rate != RATE_HZ:
        from reverberate.audio import resample_to

        samples = resample_to(samples, rate, RATE_HZ)
    fade = int(round(LOOP_FADE_S * RATE_HZ))
    ramp = np.sin(0.5 * np.pi * np.arange(fade) / fade) ** 2
    samples = samples.copy()
    samples[:fade] *= ramp
    samples[-fade:] *= ramp[::-1]
    return samples


@lru_cache(maxsize=16)
def _noise_loop(name: str) -> np.ndarray:
    """Twenty seconds of noise that loop without a seam, drawn from the clip's name.

    Shaped in frequency on the loop's own period, so its last sample runs
    into its first. ``noise``: 3 dB an octave down from 100 Hz, as a room's
    appliances are. ``speech``: the long term spectrum of speech in broad
    strokes, under a 4 Hz syllable envelope that is also periodic.
    """
    count = int(round(NOISE_LOOP_S * RATE_HZ))
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1.0 / RATE_HZ)
    if name.startswith("speech:"):
        shape = (freqs / 500.0) / (1.0 + (freqs / 500.0) ** 2) / (1.0 + (freqs / 4000.0) ** 2)
    else:
        shape = 1.0 / np.sqrt(np.maximum(freqs, 100.0) / 100.0)
        shape = shape * np.clip(freqs / 40.0, 0.0, 1.0)
    samples = np.fft.irfft(spectrum * shape, count)
    if name.startswith("speech:"):
        t = np.arange(count) / RATE_HZ
        samples = samples * (0.5 - 0.5 * np.cos(2.0 * np.pi * 4.0 * t)) ** 2
    return np.asarray(samples * (NOISE_RMS / np.sqrt(np.mean(samples**2))))


def _load(clip: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """A clip of a :class:`DryPlan` as samples and their rate: the engine's loader."""
    if clip.get("library") != PLACEHOLDER:
        return _read(str(clip["file"]))
    name = str(clip["name"])
    loop = _voice_loop(str(clip["file"])) if "file" in clip else _noise_loop(name)
    first = int(round(float(clip["offset_s"]) * RATE_HZ))
    count = int(np.ceil(float(clip["seconds"]) * RATE_HZ)) + 2
    return loop[(first + np.arange(count)) % loop.size], RATE_HZ


LOADER: ClipLoader = _load


def dry_track(plan: DryPlan, rate: float = RATE_HZ) -> DryTrack:
    """The track the engine reads for a plan's source."""
    return DryTrack.from_recipe(plan.source, LOADER, rate=rate)
