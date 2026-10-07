"""What there is to play: items made of stems, read by time range, mixed with a gain a stem.

An **item** is one thing a listener hears from its start to its end: a
variant's render of a window, a scene. It is made of one or several
**stems** that are summed, each a file:

- an order 7 signal of ``docs/formats/scene-signal.md`` (``.json`` beside
  ``.f32``), 64 channels in the scene's fixed frame, which a player decodes
  under the head its listener turns;
- or a WAV of two ears, decoded already for the scene's own head, which a
  player plays as written.

:meth:`Library.frames` is what a player streams: a time range of an item,
its stems summed in float64 in their order, each at its gain, and brought
back to float32; one stem at a gain of one is its own samples untouched.
:func:`kit_folder` reads the folder ``python -m reverberate.render check
--against`` writes, where the items are the variants and are held to one
length and one rate, since they are switched at the same instant.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.output import open_signal
from reverberate.viz.parts.server import HttpError

__all__ = ["Item", "Kit", "Library", "Stem", "ambisonic_stem", "colour_of", "kit_folder"]

#: The level a kit's WAV files are written at, and a player's default for order 7, dB.
PAGE_LEVEL_DB = -12.0
#: Colours a voice is drawn in, in the order the voices come; told apart by hue.
VOICE_COLOURS = (
    "#e4572e",
    "#2e86de",
    "#17a65b",
    "#c148c9",
    "#e0a100",
    "#0fb5ae",
    "#d6336c",
    "#7c5cff",
    "#8fbf26",
)
#: The colour of every noise source: what is not a voice is not told apart by hue.
NOISE_COLOUR = "#8a94a6"


def colour_of(index: int, kind: str) -> str:
    """The colour of the ``index``-th source of its kind, the same on every component."""
    return NOISE_COLOUR if kind == "noise" else VOICE_COLOURS[index % len(VOICE_COLOURS)]


def kind_of(source: str) -> str:
    """A source's kind by its name, as the generator names them: a noise, or a voice."""
    return "noise" if source.startswith("noise") else "voice"


@dataclass(frozen=True)
class Stem:
    id: str
    path: Path
    #: ``"ambisonic"`` (order 7, the scene's frame) or ``"binaural"`` (two ears, as written).
    kind: str
    channels: int
    frames: int
    rate: float
    peak: float = 0.0


@dataclass(frozen=True)
class Item:
    id: str
    label: str
    stems: tuple[Stem, ...]
    #: What the application shows beside it: the variant, the signal, the source.
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.stems[0].kind

    @property
    def channels(self) -> int:
        return self.stems[0].channels

    @property
    def frames(self) -> int:
        return self.stems[0].frames

    @property
    def rate(self) -> float:
        return self.stems[0].rate

    @property
    def level_db(self) -> float:
        """What a player adds for the page's default level: order 7 is physical, a WAV is not."""
        return PAGE_LEVEL_DB if self.kind == "ambisonic" else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "channels": self.channels,
            "frames": self.frames,
            "rate": self.rate,
            "level_db": self.level_db,
            "stems": [stem.id for stem in self.stems],
            **self.meta,
        }


def ambisonic_stem(header: Path, name: str | None = None) -> Stem:
    """The stem of a scene signal, by its header."""
    signal = open_signal(Path(header))
    h = signal.header
    return Stem(
        name or Path(header).stem,
        Path(header),
        "ambisonic",
        int(h["channels"]),
        int(h["frames"]),
        float(h["sample_rate_hz"]),
        float(h.get("peak", 0.0)),
    )


def binaural_stem(path: Path, name: str | None = None) -> Stem:
    """The stem of a WAV of two ears."""
    import soundfile

    info = soundfile.info(str(path))
    if info.channels != 2:
        raise ValueError(f"{path} has {info.channels} channel(s): two ears are two")
    return Stem(
        name or Path(path).stem, Path(path), "binaural", 2, int(info.frames), float(info.samplerate)
    )


class Library:
    """Items by name, and their samples."""

    def __init__(self, items: Iterable[Item] = ()) -> None:
        self._items: dict[str, Item] = {}
        self._mapped: dict[Path, np.ndarray] = {}
        self._lock = threading.Lock()
        for item in items:
            self.add(item)

    def add(self, item: Item) -> None:
        if not item.stems:
            raise ValueError(f"{item.id} has no stem")
        first = item.stems[0]
        for stem in item.stems[1:]:
            if (stem.kind, stem.channels, stem.frames, stem.rate) != (
                first.kind,
                first.channels,
                first.frames,
                first.rate,
            ):
                raise ValueError(f"the stems of {item.id} are not of one shape: {stem.id}")
        self._items[item.id] = item

    def item(self, name: str) -> Item:
        if name not in self._items:
            raise HttpError(404, f"no item named {name!r}")
        return self._items[name]

    def describe(self) -> list[dict[str, Any]]:
        return [item.describe() for item in self._items.values()]

    def _read(self, stem: Stem, start: int, stop: int) -> np.ndarray:
        """``[frame, channel]`` float32 of ``stem`` over ``[start, stop)``, inside the file."""
        if stem.kind == "ambisonic":
            with self._lock:
                if stem.path not in self._mapped:
                    self._mapped[stem.path] = open_signal(stem.path).frames
                frames = self._mapped[stem.path]
            return np.asarray(frames[start:stop])
        import soundfile

        data, _ = soundfile.read(
            str(stem.path), start=start, frames=stop - start, dtype="float32", always_2d=True
        )
        return np.asarray(data)

    def frames(
        self, name: str, start: int, count: int, gains: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """``count`` frames of an item from ``start``: ``[frame, channel]`` float32.

        What lies past the item's end is zeros, so that a player's last
        chunk has the length it asked for. ``gains`` is a factor a stem;
        a stem left out is at one.
        """
        item = self.item(name)
        start, count = int(start), int(count)
        if start < 0 or count < 0:
            raise HttpError(400, "a range starts at a frame and has a length")
        out = np.zeros((count, item.channels), dtype=np.float32)
        stop = min(start + count, item.frames)
        if stop <= start:
            return out
        weights = [float((gains or {}).get(stem.id, 1.0)) for stem in item.stems]
        heard = [(stem, w) for stem, w in zip(item.stems, weights, strict=True) if w != 0.0]
        if len(heard) == 1 and heard[0][1] == 1.0:
            out[: stop - start] = self._read(heard[0][0], start, stop)
        elif heard:
            total = np.zeros((stop - start, item.channels))
            for stem, weight in heard:
                total += weight * self._read(stem, start, stop).astype(np.float64)
            out[: stop - start] = total
        return out

    def mono(
        self, name: str, gains: Mapping[str, float] | None = None, *, stem: str | None = None
    ) -> np.ndarray:
        """One channel of a whole item to look at, at the page's level: float64.

        Of order 7 the omnidirectional channel, which is the pressure at
        the centre of the head and turns with nothing; of two ears their
        mean. ``stem`` reads one stem alone.
        """
        item = self.item(name)
        chosen = [s for s in item.stems if stem is None or s.id == stem]
        if not chosen:
            raise HttpError(404, f"{name} has no stem named {stem!r}")
        scale = 10.0 ** (item.level_db / 20.0)
        total = np.zeros(item.frames)
        for one in chosen:
            weight = float((gains or {}).get(one.id, 1.0)) if stem is None else 1.0
            if weight == 0.0:
                continue
            piece = 1 << 18
            for a in range(0, item.frames, piece):
                block = self._read(one, a, min(a + piece, item.frames))
                channel = block[:, 0] if item.kind == "ambisonic" else block.mean(axis=1)
                total[a : a + block.shape[0]] += weight * scale * channel
        return total


@dataclass(frozen=True)
class Kit:
    """A folder of the listening kit, read: what is compared, and what was measured of it."""

    folder: Path
    library: Library
    #: The variants, the reference first.
    variants: tuple[str, ...]
    #: ``signal -> track -> variant -> item id``; a track is ``"mix"`` or a source.
    sets: Mapping[str, Mapping[str, Mapping[str, str]]]
    #: ``variants.json`` or ``ab.json`` as written.
    document: Mapping[str, Any]
    #: The scene's own head over the window, or ``None`` where the folder does not say.
    head: Mapping[str, Any] | None
    notes: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        document = self.document
        return {
            "folder": str(self.folder),
            "variants": list(self.variants),
            "sets": self.sets,
            "signals": dict(document.get("signals") or {"clips": "the recipe's clips"}),
            "window_s": document.get("window_s"),
            "sources": document.get("sources"),
            "head": self.head,
            "decoder": document.get("decoder"),
            "gain_db": document.get("gain_db"),
            "notes": [*self.notes, *(document.get("notes") or [])],
            "items": self.library.describe(),
        }


def _listed(document: Mapping[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """The files of a folder written before it listed them: by the names it gave them."""
    rows = []
    files = dict(document.get("files") or {})
    for variant in names:
        for source in ["mix", *list(document.get("sources") or [])]:
            path = files.get(f"{variant}_{source}")
            if path is not None:
                rows.append(
                    {
                        "variant": variant,
                        "source": source,
                        "signal": "clips",
                        "wav": str(Path("listen") / Path(path).name),
                        "ambisonic": None,
                    }
                )
    return rows


def kit_folder(folder: Path) -> Kit:
    """The folder of a comparison (``variants.json``, or ``ab.json`` of two packs), as items.

    Where a source has its order 7 stem the item is that, and the mix is
    the sources' stems, which a player sums; elsewhere the item is the WAV
    of two ears. A variant that lacks what another has is left out of that
    set, with a note.
    """
    folder = Path(folder)
    for name in ("variants.json", "ab.json"):
        if (folder / name).is_file():
            document = json.loads((folder / name).read_text(encoding="utf-8"))
            break
    else:
        raise SystemExit(f"{folder} holds no variants.json and no ab.json: not a comparison")
    if "variants" in document:
        variants = [str(document["reference"]["name"]), *map(str, document["variants"])]
    else:
        variants = ["A", "B"]
    rows = list(document.get("listen") or _listed(document, variants))
    library = Library()
    notes: list[str] = []
    sets: dict[str, dict[str, dict[str, str]]] = {}
    by_set: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault((row["signal"], row["source"]), {})[row["variant"]] = row
    stems: dict[tuple[str, str, str], Stem] = {}
    for (signal, source), held in by_set.items():
        for variant, row in held.items():
            if row.get("ambisonic") and (folder / row["ambisonic"]).is_file():
                stems[(signal, source, variant)] = ambisonic_stem(folder / row["ambisonic"], source)

    def add(signal: str, track: str, variant: str, made: tuple[Stem, ...]) -> None:
        name = f"{variant}/{signal}/{track}"
        meta = {"variant": variant, "signal": signal, "track": track}
        library.add(Item(name, variant, made, meta))
        sets.setdefault(signal, {}).setdefault(track, {})[variant] = name

    for (signal, source), held in by_set.items():
        if set(held) != set(variants):
            notes.append(f"{signal}, {source}: not every variant has it, so it is not compared")
            continue
        if source == "mix":
            # The mix of order 7 is its sources' stems, where every variant has them all.
            names = [s for (g, s, v) in stems if g == signal and v == variants[0]]
            whole = all((signal, s, v) in stems for s in names for v in variants)
            every = {s for (g, s) in by_set if g == signal and s != "mix"}
            if names and whole and set(names) == every:
                for variant in variants:
                    add(signal, "mix", variant, tuple(stems[(signal, s, variant)] for s in names))
                continue
        elif all((signal, source, v) in stems for v in variants):
            for variant in variants:
                add(signal, source, variant, (stems[(signal, source, variant)],))
            continue
        paths = {v: folder / str(held[v]["wav"]) for v in variants}
        if not all(path.is_file() for path in paths.values()):
            notes.append(f"{signal}, {source}: a file is missing, so it is not compared")
            continue
        for variant in variants:
            add(signal, source, variant, (binaural_stem(paths[variant], source),))
    # One source alone is the mix: there is one thing to hear, and it is offered once.
    for signal, tracks in sets.items():
        alone = [name for name in tracks if name != "mix"]
        if len(alone) == 1:
            sets[signal] = {"mix": tracks.get("mix") or tracks[alone[0]]}
    for signal, tracks in sets.items():
        for track, chosen in tracks.items():
            shapes = {(library.item(i).frames, library.item(i).rate) for i in chosen.values()}
            if len(shapes) != 1:
                raise SystemExit(
                    f"{folder}: the variants of {signal}, {track} are not of one length and rate,"
                    " and could not be switched at the same instant"
                )
    if not sets:
        raise SystemExit(f"{folder} lists nothing that every variant has")
    head = document.get("head")
    return Kit(folder, library, tuple(variants), sets, document, head, tuple(notes))
