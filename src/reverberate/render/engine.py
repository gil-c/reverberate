"""The signal engine: a scene pack and dry audio in, the order 7 signal at the listener out.

One code on ``numpy`` and on ``cupy``: the array module is chosen when the
engine is made (:func:`reverberate.compute.xp_for`) and nothing here imports
``cupy``. The geometry, which is small, stays on the host in float64; the
signals are on the device.

**The output is a function of the sample, not of how it was asked for.** A
source is rendered in runs of :attr:`RenderSettings.chunk_steps` steps that
start at multiples of that count from the scene's start, each run from the
pack and the dry signal alone, with no state carried from the run before.
A block, a stem, a mix and a seek are slices of those runs, so blocks of
any size give the same samples to the last bit, and a window round a cursor
is rendered without what precedes it.

Three parts are summed per source, as ``scene-pack.md`` orders them:
:mod:`.early` (the arrivals above the crossover), :mod:`.tail` (the late
part) and :mod:`.low` (the band under it); ``parts`` renders any of them
alone, which is how the benchmark splits its time. The sources are summed
in the pack's order. The signal is in the scene's fixed frame: the head's
rotation is the decoder's.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from reverberate.compute import to_numpy, xp_for
from reverberate.render.dry import (
    ClipLoader,
    DryTrack,
    band_filter,
    group_kernels,
    mask_kernel,
)
from reverberate.render.early import EarlyPart
from reverberate.render.low import LowPart
from reverberate.render.pack import ScenePack, Source, sources_of
from reverberate.render.tail import TailPart
from reverberate.render.translate import SpatialTranslation, Translation

__all__ = ["PARTS", "Engine", "RenderSettings", "SourceRenderer"]

PARTS = ("early", "low", "tail")


@dataclass(frozen=True)
class RenderSettings:
    """What a render chooses. None of it changes with the block size."""

    #: Steps rendered at once; a run is half a second and 12 MB a source at 10. A
    #: setting of the render like any other: another value moves the samples by 2e-8
    #: of the peak, the early part's filters being laid on the run's own transform.
    chunk_steps: int = 10
    #: Instants a step at which a path's harmonics are evaluated.
    direction_nodes: int = 8
    #: The owner's switch: ``None`` renders each source as the pack says, ``False``
    #: every source omnidirectional, ``True`` every source with its model.
    directivity: bool | None = None
    #: Threads of the host's transforms; -1 is every core. A card ignores it.
    workers: int = -1
    #: Runs kept per source, so that overlapping reads do not render twice.
    chunks_held: int = 2

    def record(self) -> dict[str, Any]:
        return asdict(self)


class SourceRenderer:
    """One source of a pack: its stem, by sample range."""

    def __init__(
        self,
        pack: ScenePack,
        source: Source,
        dry: DryTrack | np.ndarray,
        xp: Any,
        settings: RenderSettings,
        translation: Translation,
        high_gain_db: np.ndarray | None = None,
    ) -> None:
        h = pack.header
        bands = None
        if high_gain_db is not None:
            # Another table than the pack's (``Engine``): a scalar a step is the pack's
            # own with that table in it; a level a band is given to each part below.
            table = np.asarray(high_gain_db, dtype=np.float32)
            if table.shape not in ((h.steps,), (h.steps, len(h.bank))):
                raise ValueError(
                    f"a table of {source.id!r} is [{h.steps}] or [{h.steps}, {len(h.bank)}],"
                    f" not {list(table.shape)}"
                )
            if table.ndim == 1:
                source = replace(source, level=replace(source.level, high_gain_db=table))
            else:
                bands = table
        self.pack, self.source, self.xp, self.settings = pack, source, xp, settings
        track = (
            dry if isinstance(dry, DryTrack) else DryTrack.from_array(dry, rate=h.sample_rate_hz)
        )
        if track.rate != h.sample_rate_hz:
            raise ValueError(f"the dry signal of {source.id!r} is not at the output rate")
        base = track.scaled(10.0 ** (source.gain_db / 20.0))
        m = pack.mirror
        high = base.high(m.signature, m.lowcut_hz, m.lowcut_order)
        if h.has_low:
            # Over the onset the two sides are joined in pressure, after it in power.
            press = high.masked(mask_kernel(pack.crossover, h.sample_rate_hz, False))
            tail_mask = mask_kernel(pack.crossover, h.sample_rate_hz, True)
            power = high.masked(tail_mask)
            early_tracks, tail_track = [press, power], power
        else:
            early_tracks, tail_track, tail_mask = [high], high, None
        directivity = (
            source.directivity_enabled if settings.directivity is None else settings.directivity
        ) and source.directivity_model in pack.directivity
        self.parts: dict[str, Any] = {
            "early": EarlyPart(
                pack,
                source,
                early_tracks,
                xp,
                directivity=directivity,
                direction_nodes=settings.direction_nodes,
                workers=settings.workers,
                band_gain_db=bands,
            )
        }
        if h.has_low:
            factor = int(round(h.sample_rate_hz / h.low_sample_rate_hz))
            taps = band_filter(h.sample_rate_hz, h.low_sample_rate_hz, pack.crossover.band_hz()[1])
            self.parts["low"] = LowPart(
                pack,
                source,
                base.decimated(factor, taps),
                xp,
                translation=translation,
                workers=settings.workers,
            )
        if h.has_tail and bands is None:
            self.parts["tail"] = TailPart(
                pack, source, tail_track, xp, workers=settings.workers, mask=tail_mask
            )
        elif h.has_tail and bands is not None:
            self.parts["tail"] = _BandedTail(
                pack, source, tail_track, xp, bands, workers=settings.workers, mask=tail_mask
            )
        self.step = h.step_samples
        self.chunk_samples = settings.chunk_steps * self.step
        self._chunks: OrderedDict[tuple[int, tuple[str, ...]], Any] = OrderedDict()

    def _chunk(self, index: int, parts: tuple[str, ...]) -> Any:
        key = (index, parts)
        if key in self._chunks:
            self._chunks.move_to_end(key)
            return self._chunks[key]
        k0 = index * self.settings.chunk_steps
        k1 = min(k0 + self.settings.chunk_steps, self.pack.header.steps - 1)
        made = self.xp.zeros((self.pack.header.channels, (k1 - k0) * self.step))
        # Always in this order, so the sum rounds the same whatever was asked.
        for name in PARTS:
            if name in parts and name in self.parts:
                made = made + self.parts[name].render(k0, k1)
        self._chunks[key] = made
        while len(self._chunks) > self.settings.chunks_held:
            self._chunks.popitem(last=False)
        return made

    def render(self, start: int, stop: int, parts: Iterable[str] = PARTS) -> Any:
        """Samples ``[start, stop)`` of the stem, ``[channel, sample]`` float64 on the device."""
        chosen = tuple(name for name in PARTS if name in set(parts))
        total = self.pack.header.samples
        if not 0 <= start <= stop <= total:
            raise ValueError(f"[{start}, {stop}) is not inside the scene's {total} samples")
        out = self.xp.zeros((self.pack.header.channels, stop - start))
        for index in range(start // self.chunk_samples, -(-stop // self.chunk_samples)):
            first = index * self.chunk_samples
            lo, hi = max(start, first), min(stop, first + self.chunk_samples)
            if hi > lo:
                out[:, lo - start : hi - start] = self._chunk(index, chosen)[
                    :, lo - first : hi - first
                ]
        return out


class _BandedTail:
    """The late part under a level a band: the bands of one level together, each set apart.

    The late part has one gain a step. The bands whose columns of the
    table are the same are rendered as one late part, on the dry signal
    through the filter that gives it those bands
    (:func:`reverberate.render.dry.group_kernels`), with their column as
    its table; the filters add to the signal, so a table whose columns are
    all one is the late part itself.
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
    ) -> None:
        h = pack.header
        columns: dict[bytes, list[int]] = {}
        for band in range(table.shape[1]):
            columns.setdefault(np.ascontiguousarray(table[:, band]).tobytes(), []).append(band)
        groups = tuple(tuple(bands) for bands in columns.values())
        kernels = group_kernels(h.sample_rate_hz, tuple(h.bank), groups)
        self.parts = [
            TailPart(
                pack,
                replace(
                    source,
                    level=replace(
                        source.level, high_gain_db=np.ascontiguousarray(table[:, group[0]])
                    ),
                ),
                track if len(groups) == 1 else track.masked(kernel),
                xp,
                workers=workers,
                mask=mask,
            )
            for group, kernel in zip(groups, kernels, strict=True)
        ]

    def render(self, k0: int, k1: int) -> Any:
        made = self.parts[0].render(k0, k1)
        for part in self.parts[1:]:
            made = made + part.render(k0, k1)
        return made


class Engine:
    """A pack and the dry audio of its sources; stems, mixes and streams of blocks.

    ``dry`` maps a source's id to its :class:`~reverberate.render.dry.DryTrack`
    or to an array at the output rate that starts with the scene. A source
    it does not name is read from the pack's recipe through ``clips``, which
    turns a clip of the recipe into samples and a rate.

    ``high_gain_db`` maps a source's id to a table that is rendered in
    place of its ``level/high_gain_db``: ``[step]``, which is the pack with
    that table written in it, to the bit; or ``[step, band]`` over the
    pack's bank, a level a band, which no pack holds
    (:mod:`reverberate.render.relevel`). A source it does not name, and
    every source without it, is rendered as the pack says. It is not a
    setting: a render with the pack's own tables has the settings it had.
    """

    def __init__(
        self,
        pack: ScenePack,
        dry: Mapping[str, DryTrack | np.ndarray] | None = None,
        *,
        clips: ClipLoader | None = None,
        settings: RenderSettings | None = None,
        gpu: bool | None = False,
        xp: Any = None,
        translation: Translation | None = None,
        high_gain_db: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.pack = pack
        self.high_gain_db = dict(high_gain_db or {})
        self.settings = settings or RenderSettings()
        self.xp = xp if xp is not None else xp_for(gpu)
        self.dry = dict(dry or {})
        self.clips = clips
        h = pack.header
        self.translation = translation or SpatialTranslation.from_fusion(
            h.fusion, h.order, h.sound_speed_m_s
        )
        self._renderers: dict[str, SourceRenderer] = {}

    @property
    def samples(self) -> int:
        return self.pack.header.samples

    @property
    def channels(self) -> int:
        return self.pack.header.channels

    @property
    def sample_rate_hz(self) -> float:
        return self.pack.header.sample_rate_hz

    def source(self, source_id: str) -> SourceRenderer:
        """The renderer of one source, made on first use and kept."""
        if source_id not in self._renderers:
            source = next(sources_of(self.pack, [source_id]))
            if source_id in self.dry:
                dry: DryTrack | np.ndarray = self.dry[source_id]
            else:
                if self.clips is None:
                    raise KeyError(f"no dry signal for the source {source_id!r} and no clip loader")
                recipe = {s["id"]: s for s in self.pack.recipe_json().get("sources", [])}
                dry = DryTrack.from_recipe(
                    recipe[source_id], self.clips, rate=self.pack.header.sample_rate_hz
                )
            self._renderers[source_id] = SourceRenderer(
                self.pack,
                source,
                dry,
                self.xp,
                self.settings,
                self.translation,
                self.high_gain_db.get(source_id),
            )
        return self._renderers[source_id]

    def _range(self, start: int, stop: int | None) -> tuple[int, int]:
        return start, self.samples if stop is None else stop

    def stem(
        self,
        source_id: str,
        start: int = 0,
        stop: int | None = None,
        *,
        parts: Iterable[str] = PARTS,
    ) -> np.ndarray:
        """One source alone over ``[start, stop)``: ``[channel, sample]``, float64, on the host."""
        start, stop = self._range(start, stop)
        return to_numpy(self.source(source_id).render(start, stop, parts))

    def render(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        sources: Iterable[str] | None = None,
        parts: Iterable[str] = PARTS,
    ) -> np.ndarray:
        """The chosen sources summed over ``[start, stop)``: ``[channel, sample]``, float64."""
        start, stop = self._range(start, stop)
        parts = tuple(parts)
        total = self.xp.zeros((self.channels, stop - start))
        for source in sources_of(self.pack, sources):
            total = total + self.source(source.id).render(start, stop, parts)
        return to_numpy(total)

    def blocks(
        self,
        block_samples: int,
        *,
        sources: Iterable[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[np.ndarray]:
        """The mix in blocks of ``block_samples``: ``[channel, sample]`` float32, the last shorter.

        What a server streams and what :func:`~reverberate.render.output.write_signal`
        writes. The samples do not depend on ``block_samples``.
        """
        chosen = None if sources is None else list(sources)
        for lo, hi in self._spans(block_samples, start, stop):
            yield self.render(lo, hi, sources=chosen).astype(np.float32)

    def stems(
        self,
        block_samples: int,
        *,
        sources: Iterable[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Every chosen source apart, block by block: its id to ``[channel, sample]`` float32.

        The audit's cache is made of these; their sum in the pack's order,
        in float64, is the mix.
        """
        chosen = [source.id for source in sources_of(self.pack, sources)]
        for lo, hi in self._spans(block_samples, start, stop):
            yield {name: self.stem(name, lo, hi).astype(np.float32) for name in chosen}

    def _spans(self, block_samples: int, start: int, stop: int | None) -> Iterator[tuple[int, int]]:
        if block_samples < 1:
            raise ValueError("a block holds at least one sample")
        start, stop = self._range(start, stop)
        for lo in range(start, stop, block_samples):
            yield lo, min(lo + block_samples, stop)
