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

**Two sets of parts.** On the host the engine renders with the parts of
:mod:`.fast` unless told (:attr:`RenderSettings.engine`): single precision,
and the same mathematics ordered so that nothing is computed that no sample
reads. The parts named below are the reference they are held against, in
double precision, and what a card runs. A mix by the fast parts is not
summed from stems: what its sources share is summed before it is raised,
transformed and encoded (:meth:`Engine._mixed`), so a mix is its stems' sum
to the rounding of single precision and not to the bit.
:mod:`.mix` renders a whole scene with several processes.

Three parts are summed per source, as ``scene-pack.md`` orders them:
:mod:`.early` (the arrivals above the crossover), :mod:`.tail` (the late
part) and :mod:`.low` (the band under it); ``parts`` renders any of them
alone, which is how the benchmark splits its time. A source the mirror
renders alone has the first two and no crossover: they are its whole band. The sources are summed
in the pack's order. The signal is in the scene's fixed frame: the head's
rotation is the decoder's.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from reverberate.compute import to_numpy, xp_for
from reverberate.render import tail
from reverberate.render.dry import ClipLoader, DryTrack, band_filter, mask_kernel
from reverberate.render.early import EarlyPart
from reverberate.render.fast import FastEarly, FastLow, FastTail, HeadOperators
from reverberate.render.low import LowPart
from reverberate.render.normalise import radiating
from reverberate.render.pack import ScenePack, Source, sources_of
from reverberate.render.seam import BandedTail, level_table
from reverberate.render.tail import TailPart
from reverberate.render.translate import SpatialTranslation, Translation

__all__ = ["ENGINES", "PARTS", "Engine", "RenderSettings", "SourceRenderer"]

PARTS = ("early", "low", "tail")
ENGINES = ("fast", "reference")
#: Names the engine of a render that does not choose one: ``fast`` unless set.
ENGINE_VARIABLE = "REVERBERATE_ENGINE"


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
    #: ``fast``: the parts of :mod:`reverberate.render.fast`, in single precision, on
    #: the host. ``reference``: the parts they are held against, in double precision,
    #: which is also what a card runs whatever is asked.
    #: ``REVERBERATE_ENGINE`` names the one a render takes when it does not say.
    engine: str = field(default_factory=lambda: os.environ.get(ENGINE_VARIABLE, "fast"))
    #: Steps the fast tail renders at once and keeps, a multiple of ``chunk_steps``; 0 is
    #: the run itself. A tail's response is 1.2 s: rendered half a second at a time its
    #: transform is 3.5 times what it gives, two seconds at a time 1.6 times. A setting
    #: like ``chunk_steps``: another value moves the samples by a transform's rounding.
    #: For a render that walks the scene in order; a worker handed scattered runs would
    #: render two seconds to give half of one.
    tail_steps: int = 0
    #: What is applied above the crossover (:mod:`reverberate.render.seam`): ``tapered``,
    #: the pack's level a band where it holds one; ``broadband``, its scalar always.
    seam: str = "tapered"

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"an engine is one of {ENGINES}, not {self.engine!r}")
        if self.tail_steps % self.chunk_steps:
            raise ValueError("the tail's steps are a multiple of a run's")

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
        operators: HeadOperators | None = None,
    ) -> None:
        h = pack.header
        self.pack, self.source, self.xp, self.settings = pack, source, xp, settings
        #: The fast parts are the host's; a card renders the reference.
        self.fast = settings.engine == "fast" and xp is np
        self.dtype = np.float32 if self.fast else np.float64
        early_part: Any = FastEarly if self.fast else EarlyPart
        low_part: Any = FastLow if self.fast else LowPart
        tail_part: Any = FastTail if self.fast else TailPart
        more: dict[str, Any] = {"operators": operators} if self.fast else {}
        track = (
            dry if isinstance(dry, DryTrack) else DryTrack.from_array(dry, rate=h.sample_rate_hz)
        )
        if track.rate != h.sample_rate_hz:
            raise ValueError(f"the dry signal of {source.id!r} is not at the output rate")
        base = track.scaled(10.0 ** (source.gain_db / 20.0))
        m = pack.mirror
        high = base.high(m.signature, m.lowcut_hz, m.lowcut_order)
        # A SOURCE THE MIRROR RENDERS ALONE (``scene-pack.md``): no ``low`` group in a pack
        # that has them. The crossover is not applied to it: its arrivals and its late
        # part are the whole band, as a pack without a low band renders every source.
        crossed = h.has_low and source.crossed
        if crossed:
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
        # THE SEAM (reverberate.render.seam): ``[step, bank]`` in dB, or ``None`` for the
        # pack's scalar. The arrivals take it as their gain a band; the late part is
        # rendered a set of bands of one level at a time.
        bands = level_table(source, settings.seam)
        self.parts: dict[str, Any] = {
            "early": early_part(
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
        if crossed:
            factor = int(round(h.sample_rate_hz / h.low_sample_rate_hz))
            taps = band_filter(h.sample_rate_hz, h.low_sample_rate_hz, pack.crossover.band_hz()[1])
            self.parts["low"] = low_part(
                pack,
                source,
                base.decimated(factor, taps),
                xp,
                translation=translation,
                workers=settings.workers,
                **more,
            )
        # L23 (D3, ``render.normalise.radiating``): the late part is omnidirectional. For a
        # source whose directivity is applied and whose table is level with its axis it is
        # rendered at what the table radiates, a band; any other pack is ``pack`` itself.
        tail_pack = radiating(pack, source, directivity)
        longer: dict[str, Any] = {"tail_steps": settings.tail_steps} if self.fast else {}
        if h.has_tail and bands is None:
            self.parts["tail"] = tail_part(
                tail_pack,
                source,
                tail_track,
                xp,
                workers=settings.workers,
                mask=tail_mask,
                **longer,
            )
        elif h.has_tail and bands is not None:
            self.parts["tail"] = BandedTail(
                tail_pack,
                source,
                tail_track,
                xp,
                bands,
                workers=settings.workers,
                mask=tail_mask,
                part=tail_part,
                **longer,
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
        made = self.xp.zeros((self.pack.header.channels, (k1 - k0) * self.step), dtype=self.dtype)
        # Always in this order, so the sum rounds the same whatever was asked.
        for name in PARTS:
            if name in parts and name in self.parts:
                made = made + self.parts[name].render(k0, k1)
        self._chunks[key] = made
        while len(self._chunks) > self.settings.chunks_held:
            self._chunks.popitem(last=False)
        return made

    def render(self, start: int, stop: int, parts: Iterable[str] = PARTS) -> Any:
        """Samples ``[start, stop)`` of the stem, ``[channel, sample]`` on the device.

        Float32 from the fast parts, float64 from the reference's.
        """
        chosen = tuple(name for name in PARTS if name in set(parts))
        total = self.pack.header.samples
        if not 0 <= start <= stop <= total:
            raise ValueError(f"[{start}, {stop}) is not inside the scene's {total} samples")
        out = self.xp.zeros((self.pack.header.channels, stop - start), dtype=self.dtype)
        for index in range(start // self.chunk_samples, -(-stop // self.chunk_samples)):
            first = index * self.chunk_samples
            lo, hi = max(start, first), min(stop, first + self.chunk_samples)
            if hi > lo:
                out[:, lo - start : hi - start] = self._chunk(index, chosen)[
                    :, lo - first : hi - first
                ]
        return out


class Engine:
    """A pack and the dry audio of its sources; stems, mixes and streams of blocks.

    ``dry`` maps a source's id to its :class:`~reverberate.render.dry.DryTrack`
    or to an array at the output rate that starts with the scene. A source
    it does not name is read from the pack's recipe through ``clips``, which
    turns a clip of the recipe into samples and a rate.
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
    ) -> None:
        self.pack = pack
        self.settings = settings or RenderSettings()
        self.xp = xp if xp is not None else xp_for(gpu)
        self.dry = dict(dry or {})
        self.clips = clips
        h = pack.header
        self.translation = translation or SpatialTranslation.from_fusion(
            h.fusion, h.order, h.sound_speed_m_s
        )
        #: What moves a head off its cell, formed once for all the sources; with another
        #: estimator than the library's the fast low band renders as the reference does.
        self.operators = (
            HeadOperators(self.translation)
            if isinstance(self.translation, SpatialTranslation)
            else None
        )
        self._renderers: dict[str, SourceRenderer] = {}
        self._mixes: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._tails: tuple[tuple[Any, ...], np.ndarray] | None = None

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
            if self.settings.engine == "fast" and self.pack.header.has_tail:
                # A mix reads every source's carrier in turn: room for them all, not for
                # three, or each run would draw the next source's noise again.
                tail.HELD.limit_bytes = max(
                    tail.HELD.limit_bytes,
                    tail.HELD_BYTES + (len(self._renderers) + 1) * tail.CARRIER_BYTES,
                )
            self._renderers[source_id] = SourceRenderer(
                self.pack, source, dry, self.xp, self.settings, self.translation, self.operators
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
        """One source alone over ``[start, stop)``: ``[channel, sample]``, on the host."""
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
        """The chosen sources summed over ``[start, stop)``: ``[channel, sample]``."""
        start, stop = self._range(start, stop)
        parts = tuple(parts)
        if self.settings.engine == "fast" and self.xp is np:
            names = tuple(source.id for source in sources_of(self.pack, sources))
            return self._mixed(start, stop, tuple(p for p in PARTS if p in set(parts)), names)
        total = None
        for source in sources_of(self.pack, sources):
            stem = self.source(source.id).render(start, stop, parts)
            total = stem if total is None else total + stem
        if total is None:
            total = self.xp.zeros((self.channels, stop - start))
        return to_numpy(total)

    # -- the fast engine's mix: what the sources share is paid once ---------------------

    def _mixed(
        self, start: int, stop: int, parts: tuple[str, ...], names: tuple[str, ...]
    ) -> np.ndarray:
        """The sources ``names`` summed over ``[start, stop)``, float32, by the fast parts.

        The same runs as a stem's, and the same samples as the stems' sum to
        the rounding of single precision. A mix is not summed from stems:
        the low bands are summed at their own rate and raised once, and
        the tails' plane waves are summed, as spectra wherever a response
        is heard at one level, and transformed and encoded once
        (:meth:`_mixed_tail`): the cost of fourteen sources' transforms is
        then one source's.
        """
        total = self.samples
        if not 0 <= start <= stop <= total:
            raise ValueError(f"[{start}, {stop}) is not inside the scene's {total} samples")
        run = self.settings.chunk_steps * self.pack.header.step_samples
        out = np.zeros((self.channels, stop - start), dtype=np.float32)
        for index in range(start // run, -(-stop // run)):
            first = index * run
            lo, hi = max(start, first), min(stop, first + run)
            if hi > lo:
                made = self._mixed_run(index, parts, names)
                out[:, lo - start : hi - start] = made[:, lo - first : hi - first]
        return out

    def _mixed_run(self, index: int, parts: tuple[str, ...], names: tuple[str, ...]) -> np.ndarray:
        key = (index, parts, names)
        if key in self._mixes:
            self._mixes.move_to_end(key)
            return self._mixes[key]
        h = self.pack.header
        k0 = index * self.settings.chunk_steps
        k1 = min(k0 + self.settings.chunk_steps, h.steps - 1)
        made = np.zeros((self.channels, (k1 - k0) * h.step_samples), dtype=np.float32)
        low: np.ndarray | None = None
        raiser: Any = None
        # The sources in the pack's order, so the sum rounds the same whatever was asked.
        for name in names:
            held = self.source(name).parts
            if "early" in parts:
                made += held["early"].render(k0, k1)
            if "low" in parts and "low" in held:
                part = held["low"]
                if not part.joins:
                    made += part.render(k0, k1)
                    continue
                band = part.low_band(k0, k1)
                if band is not None:
                    low = band if low is None else low + band
                    raiser = part
        if low is not None:
            made += raiser.raised(low, k1 - k0)
        if "tail" in parts and h.has_tail:
            made += self._mixed_tail(k0, k1, names)
        self._mixes[key] = made
        while len(self._mixes) > self.settings.chunks_held:
            self._mixes.popitem(last=False)
        return made

    def _mixed_tail(self, k0: int, k1: int, names: tuple[str, ...]) -> np.ndarray:
        """The tails of ``names`` over the run, encoded once: ``[channel, (k1 - k0) step]``."""
        h = self.pack.header
        span = self.settings.tail_steps or self.settings.chunk_steps
        index = k0 // span
        first = index * span
        last = min(first + span, h.steps - 1)
        key = (index, names)
        if self._tails is None or self._tails[0] != key:
            length = (last - first) * h.step_samples
            waves: np.ndarray | None = None
            pool: dict[tuple[int, int], np.ndarray] = {}
            encoder: Any = None
            for name in names:
                part = self.source(name).parts.get("tail")
                if part is None:
                    continue
                encoder = part
                got = part.waves(first, last, pool)
                if got is not None:
                    waves = got if waves is None else waves + got
            for (_, size), spectrum in pool.items():
                heard = np.fft.irfft(spectrum, size, axis=-1)[:, size - length :]
                waves = heard if waves is None else waves + heard
            if waves is None or encoder is None:
                made = np.zeros((self.channels, length), dtype=np.float32)
            else:
                made = encoder.encoded(waves)
            self._tails = (key, made)
        step = h.step_samples
        return self._tails[1][:, (k0 - first) * step : (k1 - first) * step]

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
