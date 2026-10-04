"""The dry audio the check feeds the engine: the recipe's clips, off the disk.

A recipe names a clip by library and name; the file is
``<clips root>/<library>/<name>.wav`` (``docs/formats/clip-library.md``).
A recipe generated before the library existed names the ``placeholder``
library, whose clips are no audio. The check then **stands a real clip in,
and says so**: ``voice_NN`` is the ``NN``-th speaker of the manifest, in the
order of the speakers' names, and ``noise_<subtype>`` the manifest's clips
of that subtype; the speaker's (or the subtype's) clips are put end to end
in the order of their names, looped, and the interval reads that playlist at
its ``clip_offset_s``. Such an interval is not cut on an utterance, as the
generator cuts a real library's: it starts and stops where the placeholder
said, in the middle of a word if need be.

Where there is neither a file nor a manifest, the stand-in is noise at the
level of a voice, and the report says that too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.dry import DryTrack
from reverberate.render.pack import ScenePack

__all__ = ["ClipSource", "Feed", "feed_of"]

_STAND_IN = "check-stand-in"
#: The active level of a stored voice, dB re full scale.
VOICE_DBFS = -26.0


def _read(path: Path) -> tuple[np.ndarray, float]:
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    return np.ascontiguousarray(samples.mean(axis=1)), float(rate)


@dataclass
class ClipSource:
    """Where clips are read: a root of libraries and, for stand-ins, a manifest."""

    root: Path | None = None
    manifest: Path | None = None
    _clips: list[dict[str, Any]] | None = None
    _library: str = ""
    _playlists: dict[str, np.ndarray] = field(default_factory=dict)

    def _entries(self) -> list[dict[str, Any]]:
        if self._clips is None:
            self._clips = []
            if self.manifest is not None and Path(self.manifest).is_file():
                document = json.loads(Path(self.manifest).read_text(encoding="utf-8"))
                self._clips = list(document.get("clips", []))
                self._library = str(document.get("library", ""))
        return self._clips

    def file(self, clip: Mapping[str, Any]) -> Path | None:
        library, name = str(clip.get("library", "")), str(clip.get("name", ""))
        if self.root is None or not library or not name or ".." in name or "/" in library:
            return None
        path = Path(self.root) / library / f"{name}.wav"
        return path if path.is_file() else None

    def stand_in(self, name: str) -> tuple[list[str], str]:
        """The manifest's clips a placeholder ``name`` is played from, and how that is said."""
        entries = self._entries()
        if name.startswith("voice_") and name[6:].isdigit():
            speakers = sorted({str(e.get("speaker", "")) for e in entries if e["kind"] == "voice"})
            if speakers:
                speaker = speakers[(int(name[6:]) - 1) % len(speakers)]
                names = sorted(e["name"] for e in entries if e.get("speaker") == speaker)
                return names, f"{name} -> speaker {speaker} of {self._library}, {len(names)} clips"
        elif name.startswith("noise_"):
            names = sorted(e["name"] for e in entries if e.get("subtype") == name[6:])
            if names:
                return names, f"{name} -> the {name[6:]} clips of {self._library}, {len(names)}"
        return [], f"{name} -> noise at {VOICE_DBFS:.0f} dB re full scale: no clip to stand in"

    def playlist(self, name: str) -> np.ndarray:
        """A placeholder's stand-in, its clips end to end at 48 kHz."""
        if name not in self._playlists:
            names, _ = self.stand_in(name)
            parts = []
            for clip in names:
                path = self.file({"library": self._library, "name": clip})
                if path is not None:
                    samples, rate = _read(path)
                    if rate != 48000.0:
                        from reverberate.audio import resample_to

                        samples = resample_to(samples, rate, 48000.0)
                    parts.append(samples)
            if not parts:
                seed = int.from_bytes(name.encode()[-4:].rjust(4, b"\0"), "little")
                noise = np.random.default_rng(seed).standard_normal(48000 * 10)
                parts = [noise * 10.0 ** (VOICE_DBFS / 20.0)]
            self._playlists[name] = np.concatenate(parts)
        return self._playlists[name]

    def load(self, clip: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        """The engine's loader for the intervals :func:`feed_of` wrote."""
        if clip.get("library") != _STAND_IN:
            return _read(Path(str(clip["file"])))
        loop = self.playlist(str(clip["name"]))
        first = int(round(float(clip["from_s"]) * 48000.0))
        count = int(np.ceil(float(clip["seconds"]) * 48000.0)) + 2
        return loop[(first + np.arange(count)) % loop.size], 48000.0


@dataclass(frozen=True)
class Feed:
    """What one source is fed over the pack's time."""

    source_id: str
    track: DryTrack
    #: ``(start_s, end_s)`` on the pack's clock, in order.
    intervals: tuple[tuple[float, float], ...]
    #: What the report prints: ``"clips"``, or what stands in.
    label: str
    stand_in: bool


def feed_of(pack: ScenePack, source_id: str, recipe: Mapping[str, Any], clips: ClipSource) -> Feed:
    """The dry track of ``source_id`` from ``recipe``'s activity, on the pack's clock.

    A pack that starts ``profile.start_s`` into its scene has its intervals
    moved by as much and cut to what it holds.
    """
    h = pack.header
    entry = next((s for s in recipe.get("sources", []) if s.get("id") == source_id), None)
    start_s = float((h.provenance.get("profile") or {}).get("start_s") or 0.0)
    last = (h.steps - 1) * h.step_s
    pieces: list[dict[str, Any]] = []
    labels: list[str] = []
    stood = False
    for interval in (entry or {}).get("activity", []):
        a, b = float(interval["start_s"]) - start_s, float(interval["end_s"]) - start_s
        cut = max(-a, 0.0)
        a, b = max(a, 0.0), min(b, last)
        if b <= a:
            continue
        clip = dict(interval.get("clip") or {})
        offset = float(interval.get("clip_offset_s", 0.0)) + cut
        path = None if clip.get("library") == "placeholder" else clips.file(clip)
        if path is not None:
            chosen: dict[str, Any] = {**clip, "file": str(path)}
            label = "clips"
        else:
            name = str(clip.get("name") or f"voice_{source_id}")
            chosen = {"library": _STAND_IN, "name": name, "from_s": offset, "seconds": b - a}
            label, offset, stood = clips.stand_in(name)[1], 0.0, True
        if label not in labels:
            labels.append(label)
        pieces.append(
            {
                "start_s": a,
                "end_s": b,
                "clip": chosen,
                "clip_offset_s": offset,
                "gain_db": float(interval.get("gain_db", 0.0)),
            }
        )
    track = DryTrack.from_recipe({"activity": pieces}, clips.load, rate=h.sample_rate_hz)
    return Feed(
        source_id,
        track,
        tuple((p["start_s"], p["end_s"]) for p in pieces),
        "; ".join(labels) if labels else "no activity in the pack's time",
        stood,
    )
