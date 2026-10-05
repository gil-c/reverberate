"""A source's dry signal over the scene, and what is filtered once before any path reads it.

A recipe gives a source intervals of activity, each a clip placed at a time
with a gain. A :class:`DryTrack` is those clips on the scene's clock: it is
read by sample range and is zero wherever the source is silent, so twenty
minutes of a source cost the memory of the clips being played, not of the
scene.

Everything of the pack that is one fixed filter is applied here, to the
clip, before the moving parts: the source's signature and the reference
chain's low cut, the crossover's two high masks, and the decimation to the
low band's rate. The format lists them after the delay line; a fixed filter
and a delay that moves commute to the order of the speed over the sound
speed, 0.4 per cent, the error the quasi-static rendering already has, and
at rest they commute exactly. Filtering the clip costs one pass per clip
and filtering the output would cost one per channel.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import butter, fftconvolve, firwin, kaiserord, sosfilt

from reverberate.mirror.hybrid import Crossover

__all__ = ["ClipLoader", "DryTrack", "band_filter", "group_kernels", "mask_kernel"]

#: The filters that share a signal among groups of bands: their length, and their grid.
GROUP_TAPS = 2047
GROUP_GRID = 8192

#: How long the low cut is let ring after a clip, in seconds: -140 dB by then.
RING_S = 1.0
#: Taps of a crossover mask as a filter, at 48 kHz: 85 ms, the mask smoothed by 23 Hz.
MASK_TAPS = 4097
#: The grid the masks are designed on: the 1.2 s response ``mirror.hybrid.blend`` joins.
MASK_GRID = 57600
#: What an interval of a recipe is faded in and out over, seconds: a raised cosine.
EDGE_FADE_S = 0.005
#: Stop band of the filter between the two rates, in dB; its pass band ripples by 1e-6.
RATE_FILTER_DB = 120.0

#: A clip of the recipe to its samples and their rate.
ClipLoader = Callable[[Mapping[str, Any]], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class _Segment:
    start: int
    length: int
    make: Callable[[], np.ndarray]


class DryTrack:
    """Signals placed on the scene's clock at one rate, read by range, zero elsewhere."""

    def __init__(self, segments: list[_Segment], rate: float, *, held: int = 4) -> None:
        self.segments = sorted(segments, key=lambda s: s.start)
        self.rate = float(rate)
        self._held = held
        self._made: OrderedDict[int, np.ndarray] = OrderedDict()

    @classmethod
    def from_array(
        cls, samples: np.ndarray, *, start_s: float = 0.0, rate: float = 48000.0
    ) -> DryTrack:
        """One signal starting at ``start_s``."""
        data = np.ascontiguousarray(np.asarray(samples, dtype=float))
        if data.ndim != 1:
            raise ValueError("a dry signal is mono")
        return cls([_Segment(int(round(start_s * rate)), data.size, lambda: data)], rate)

    @classmethod
    def from_recipe(
        cls, source: Mapping[str, Any], load: ClipLoader, *, rate: float = 48000.0
    ) -> DryTrack:
        """The intervals of a recipe's source: each clip from its offset, at its gain.

        The source's own ``gain_db`` is not applied here; the engine applies
        the pack's.

        **An interval is faded in and out** over :data:`EDGE_FADE_S`, inside
        its own length: a source that comes on or goes off away from zero is
        a step on 64 channels, which is a click (measured: a noise cut where
        it stands starts at its own rms). An end that another interval of
        the source starts on, to the sample, is not faded, nor is that
        start: they are one interval cut in pieces, as the audit cuts a long
        one, and their join stays exact.
        """
        spans = []
        for interval in source.get("activity", []):
            start = int(round(float(interval["start_s"]) * rate))
            length = int(round((float(interval["end_s"]) - float(interval["start_s"])) * rate))
            spans.append((start, length, interval))
        starts = {start for start, _, _ in spans}
        ends = {start + length for start, length, _ in spans}
        fade = int(round(EDGE_FADE_S * rate))
        segments = []
        for start, length, interval in spans:
            edges = (0 if start in ends else fade, 0 if start + length in starts else fade)
            segments.append(
                _Segment(start, length, _clip_maker(interval, load, rate, length, edges))
            )
        return cls(segments, rate)

    def _samples(self, index: int) -> np.ndarray:
        if index in self._made:
            self._made.move_to_end(index)
            return self._made[index]
        made = np.asarray(self.segments[index].make(), dtype=float)
        if made.shape != (self.segments[index].length,):
            raise ValueError("a dry segment is not the length it was placed with")
        self._made[index] = made
        while len(self._made) > self._held:
            self._made.popitem(last=False)
        return made

    def silent(self, start: int, stop: int) -> bool:
        """Whether nothing of the track lies in ``[start, stop)``."""
        return not any(s.start < stop and s.start + s.length > start for s in self.segments)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Samples ``[start, stop)`` of the scene's clock at the track's rate."""
        out = np.zeros(max(stop - start, 0))
        for index, segment in enumerate(self.segments):
            lo = max(start, segment.start)
            hi = min(stop, segment.start + segment.length)
            if hi > lo:
                samples = self._samples(index)
                out[lo - start : hi - start] += samples[lo - segment.start : hi - segment.start]
        return out

    def through(
        self, filt: Callable[[np.ndarray], np.ndarray], *, before: int, after: int
    ) -> DryTrack:
        """Every segment through ``filt``, which returns ``before + length + after`` samples."""

        def maker(index: int) -> Callable[[], np.ndarray]:
            return lambda: filt(self._samples(index))

        return DryTrack(
            [
                _Segment(s.start - before, s.length + before + after, maker(i))
                for i, s in enumerate(self.segments)
            ],
            self.rate,
        )

    def scaled(self, gain: float) -> DryTrack:
        return self.through(lambda x: x * gain, before=0, after=0)

    def high(self, signature: np.ndarray, lowcut_hz: float, lowcut_order: int) -> DryTrack:
        """Through the reference chain's low cut and the source's signature, both causal."""
        taps = np.asarray(signature, dtype=float)
        ring = int(round(RING_S * self.rate)) if lowcut_hz > 0.0 else 0
        sos = (
            butter(lowcut_order, lowcut_hz, btype="high", fs=self.rate, output="sos")
            if lowcut_hz > 0.0
            else None
        )

        def filt(x: np.ndarray) -> np.ndarray:
            y = np.concatenate([x, np.zeros(ring + taps.size - 1)])
            if sos is not None:
                y = np.asarray(sosfilt(sos, y))
            if taps.size > 1 or taps[0] != 1.0:
                y = np.asarray(fftconvolve(y, taps, mode="full"))[: y.size]
            return y

        return self.through(filt, before=0, after=ring + taps.size - 1)

    def masked(self, kernel: np.ndarray) -> DryTrack:
        """Through a zero phase filter of odd length, its centre on the sample."""
        half = kernel.size // 2
        return self.through(
            lambda x: np.asarray(fftconvolve(x, kernel, mode="full")), before=half, after=half
        )

    def decimated(self, factor: int, taps: np.ndarray) -> DryTrack:
        """At ``1 / factor`` of the rate through the zero phase ``taps``, on the scene's grid."""
        half = taps.size // 2
        segments = []
        for index, s in enumerate(self.segments):
            # Low sample m is the sample factor * m; the filter reaches half either side.
            first = -((half - s.start) // factor)
            last = (s.start + s.length - 1 + half) // factor

            def make(
                index: int = index, s: _Segment = s, first: int = first, last: int = last
            ) -> np.ndarray:
                full = fftconvolve(self._samples(index), taps, mode="full")
                at = factor * np.arange(first, last + 1) - s.start + half
                return np.asarray(full[at])

            segments.append(_Segment(first, last - first + 1, make))
        return DryTrack(segments, self.rate / factor)


def _clip_maker(
    interval: Mapping[str, Any],
    load: ClipLoader,
    rate: float,
    length: int,
    edges: tuple[int, int] = (0, 0),
) -> Callable[[], np.ndarray]:
    def make() -> np.ndarray:
        samples, clip_rate = load(interval["clip"])
        samples = np.asarray(samples, dtype=float)
        if clip_rate != rate:
            from reverberate.audio import resample_to

            samples = resample_to(samples, clip_rate, rate)
        first = int(round(float(interval.get("clip_offset_s", 0.0)) * rate))
        if first + length > samples.size:
            raise ValueError(
                f"the clip {interval['clip'].get('name')!r} is shorter than its interval"
            )
        gain = 10.0 ** (float(interval.get("gain_db", 0.0)) / 20.0)
        cut = np.asarray(samples[first : first + length] * gain)
        for count, side in zip(edges, (slice(None), slice(None, None, -1)), strict=True):
            count = min(count, length // 2)
            if count:
                # Zero on the first sample and on the last, one a fade's length in.
                cut[side][:count] *= np.sin(0.5 * np.pi * np.arange(count) / count) ** 2
        return cut

    return make


@lru_cache(maxsize=8)
def mask_kernel(crossover: Crossover, rate: float, power: bool) -> np.ndarray:
    """The crossover's high mask as a zero phase filter of :data:`MASK_TAPS` taps.

    The mask ``mirror.hybrid.blend`` multiplies a 1.2 s spectrum by, brought
    to the time domain on that grid and cut to 85 ms under a Hann window, so
    that it can be applied to a signal of any length.
    """
    grid = int(round(MASK_GRID * rate / 48000.0))
    _, high = crossover.masks(grid, rate, power=power)
    response = np.fft.irfft(high, n=grid)
    half = MASK_TAPS // 2
    kernel = np.concatenate([response[-half:], response[: half + 1]])
    kernel *= np.hanning(MASK_TAPS + 2)[1:-1]
    kernel.setflags(write=False)
    return kernel


def group_kernels(
    rate: float, bank_hz: tuple[int, ...], groups: tuple[tuple[int, ...], ...]
) -> list[np.ndarray]:
    """Zero phase filters that share a signal among groups of the bank's bands, adding to it.

    ``groups`` names every band of ``bank_hz`` once. A band holds its own
    centre whole and, between two centres, what is left of a straight line
    in octaves from the one to the other; a group's filter is its bands'
    shares, of :data:`GROUP_TAPS` taps. The last is the signal less the
    others, so that the filters add to the signal exactly: a late part
    rendered a group at a time with one gain is the late part rendered
    whole. What a level a band needs of a part that has one gain
    (:class:`reverberate.render.engine.SourceRenderer`).
    """
    named = sorted(band for group in groups for band in group)
    if named != list(range(len(bank_hz))):
        raise ValueError("the groups name every band of the bank once")
    grid = int(round(GROUP_GRID * rate / 48000.0))
    freqs = np.fft.rfftfreq(grid, 1.0 / rate)
    octaves = np.log2(np.maximum(freqs, 1e-9))
    centres = np.log2(np.asarray(bank_hz, dtype=float))
    half = GROUP_TAPS // 2
    window = np.hanning(GROUP_TAPS + 2)[1:-1]
    whole = np.zeros(GROUP_TAPS)
    whole[half] = 1.0
    kernels: list[np.ndarray] = []
    for group in groups[:-1]:
        owned = np.zeros(len(bank_hz))
        owned[list(group)] = 1.0
        response = np.fft.irfft(np.interp(octaves, centres, owned), n=grid)
        kernels.append(np.concatenate([response[-half:], response[: half + 1]]) * window)
    kernels.append(whole - np.sum(kernels, axis=0) if kernels else whole)
    return kernels


@lru_cache(maxsize=8)
def band_filter(rate: float, low_rate: float, pass_hz: float) -> np.ndarray:
    """The filter between the two rates: flat to ``pass_hz``, nothing from ``low_rate - pass_hz``.

    The low band's responses are zero above ``pass_hz``, so what the
    decimation folds between there and the low Nyquist meets a zero, and the
    interpolation's first image starts at ``low_rate - pass_hz``. Odd and
    symmetric: zero phase about its centre. Unit gain at the low rate; the
    interpolation multiplies it by the ratio.
    """
    width = low_rate - 2.0 * pass_hz
    if width <= 0.0:
        raise ValueError("the low rate leaves no room between the band and its image")
    taps, beta = kaiserord(RATE_FILTER_DB, width / (0.5 * rate))
    taps |= 1
    kernel = np.asarray(firwin(taps, 0.5 * low_rate, window=("kaiser", beta), fs=rate))
    kernel.setflags(write=False)
    return kernel
