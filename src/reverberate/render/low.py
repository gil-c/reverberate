"""The band under the crossover: the dry signal through the solved responses, moved to the head.

Everything here runs at the pack's low rate, 4 kHz, and is brought to the
output rate at the end by the filter of :func:`~reverberate.render.dry.band_filter`.

**A step's responses** are the pack's pairs: two source positions by one or
two cells, each on the scale of :func:`reverberate.spatial.lowband.to_stored`
(the 48 kHz response's samples, so ``sample_rate_hz / low_sample_rate_hz``
times them is what a convolution at 4 kHz needs). The dry signal is
convolved with each cell's response, the two source positions weighted by
``position_weight``, in one transform a step (the pairs' spectra are kept
while they are in use).

**The head** is reached in frames one step long, four to a step, under a
square root Hann window at both ends, so the frames add to one. A frame
takes the pairs, the weight, the cells and the mode of the step nearest its
centre, and the listener's offset from each cell *at* its centre: the
offset is followed every 12.5 ms, and the passage from one step's responses
to the next is the frames' own overlap, a raised cosine a step long. Mode 1
leaves the frame as it is; modes 2 and 3 hand its spectrum to the
:class:`~reverberate.render.translate.Translation`.

A frame is 50 ms and its transform has 20 Hz bins, which suits an operator
whose own response is a few milliseconds long; a translation of 4 samples
under these windows errs by 0.1 per cent, far under the truncation of an
order 7 expansion.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

from reverberate.render.dry import DryTrack, band_filter
from reverberate.render.early import fft_module, workers_of
from reverberate.render.pack import ScenePack, Source
from reverberate.render.translate import Translation

__all__ = ["FRAMES_PER_STEP", "LowPart"]

#: Frames of the head's translation in one step.
FRAMES_PER_STEP = 4


class LowPart:
    """One source's low band, rendered a run of steps at a time."""

    def __init__(
        self,
        pack: ScenePack,
        source: Source,
        track: DryTrack,
        xp: Any,
        *,
        translation: Translation,
        workers: int,
        pairs_held: int = 8,
    ) -> None:
        if source.low is None:
            raise ValueError(f"the source {source.id!r} has no low band")
        h = pack.header
        self.pack, self.low, self.track, self.xp = pack, source.low, track, xp
        self.translation = translation
        self.workers = workers
        self.rate = h.low_sample_rate_hz
        self.step = h.low_step_samples
        self.factor = int(round(h.sample_rate_hz / h.low_sample_rate_hz))
        if self.step % FRAMES_PER_STEP:
            raise ValueError("the frames do not divide a step of the low band")
        self.hop = self.step // FRAMES_PER_STEP
        self.frame = self.step
        #: A step's convolution covers the frames it owns: ``[step k - frame, step k + cover)``.
        self.cover = self.frame + (FRAMES_PER_STEP - 1) * self.hop
        self.n = next_fast_len(h.low_samples + self.cover, real=True)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.frame) / self.frame)
        # Frames a hop apart: FRAMES_PER_STEP / 2 Hann windows add at every sample.
        self.hann = xp.asarray(window / (FRAMES_PER_STEP / 2))
        self.root = xp.asarray(np.sqrt(window / (FRAMES_PER_STEP / 2)))
        freqs = np.fft.rfftfreq(self.frame, 1.0 / self.rate)
        limit = pack.crossover.band_hz()[1]
        self.kept = int(np.searchsorted(freqs, limit * 1.05, side="right"))
        self.freqs = freqs[: self.kept]
        taps = band_filter(h.sample_rate_hz, h.low_sample_rate_hz, limit)
        half = taps.size // 2
        self.reach = -(-half // self.factor)
        # out[factor q + r] = factor sum_i low[q - i] taps[half + factor i + r]
        poly = np.zeros((2 * self.reach + 1, self.factor))
        for j in range(2 * self.reach + 1):
            i = self.reach - j
            for r in range(self.factor):
                d = self.factor * i + r
                if abs(d) <= half:
                    poly[j, r] = self.factor * taps[half + d]
        self.poly = xp.asarray(poly)
        self._spectra: OrderedDict[int, Any] = OrderedDict()
        self._pairs_held = pairs_held
        self._frames: dict[int, Any] = {}
        self._fields: dict[int, Any] = {}
        self.steps = h.steps
        self.listener = np.asarray(pack.listener.position, dtype=float)
        self.cells = np.asarray(pack.cells.position, dtype=float)

    def _spectrum(self, row: int) -> Any:
        if row in self._spectra:
            self._spectra.move_to_end(row)
            return self._spectra[row]
        xp = self.xp
        # ``low/ir`` holds the 48 kHz response's own samples, one in ``factor``
        # (``spatial.lowband.to_stored``): a convolution at the low rate sums
        # ``factor`` times fewer of them, and is brought back to level here.
        response = xp.asarray(np.asarray(self.low.ir[row], dtype=float) * self.factor)
        made = fft_module(xp).rfft(response, self.n, axis=-1, **workers_of(xp, self.workers))
        self._spectra[row] = made
        while len(self._spectra) > self._pairs_held:
            self._spectra.popitem(last=False)
        return made

    def _field(self, k: int) -> Any:
        """The cells' fields round step ``k``: ``[cell, channel, cover]``, or ``None`` if silent."""
        if k in self._fields:
            return self._fields[k]
        xp = self.xp
        mode = int(self.low.mode[k])
        end = k * self.step - self.frame + self.cover
        made = None
        if mode != 0 and not self.track.silent(end - self.n, end):
            fft = fft_module(xp)
            dry = fft.rfft(xp.asarray(self.track.read(end - self.n, end)))
            weight = float(self.low.position_weight[k])
            fields = []
            for b in range(2 if mode == 3 else 1):
                spectrum = (1.0 - weight) * self._spectrum(int(self.low.pair[k, 0, b]))
                if weight > 0.0:
                    spectrum = spectrum + weight * self._spectrum(int(self.low.pair[k, 1, b]))
                fields.append(spectrum * dry[None, :])
            made = fft.irfft(xp.stack(fields), self.n, axis=-1, **workers_of(xp, self.workers))[
                ..., self.n - self.cover :
            ]
        self._fields[k] = made
        return made

    def _frame(self, m: int) -> Any:
        """Frame ``m``, centred on the low sample ``m hop``: ``[channel, frame]`` or ``None``."""
        if m in self._frames:
            return self._frames[m]
        xp = self.xp
        k = min(max((2 * m + FRAMES_PER_STEP) // (2 * FRAMES_PER_STEP), 0), self.steps - 1)
        field = self._field(k)
        made = None
        if field is not None:
            start = m * self.hop - self.frame // 2 - (k * self.step - self.frame)
            piece = field[:, :, start : start + self.frame]
            mode = int(self.low.mode[k])
            if mode == 1:
                made = piece[0] * self.hann[None, :]
            else:
                fft = fft_module(xp)
                spectrum = fft.rfft(piece * self.root[None, None, :], axis=-1)[..., : self.kept]
                at = min(max(m * self.hop / self.step, 0.0), self.steps - 1.0)
                lower = min(int(at), self.steps - 2)
                head = self.listener[lower] * (1.0 - (at - lower)) + self.listener[lower + 1] * (
                    at - lower
                )
                cells = self.low.cell[k, : piece.shape[0]]
                moved = self.translation.to_head(
                    spectrum, head[None, :] - self.cells[cells], self.freqs, xp
                )
                full = xp.zeros((moved.shape[0], self.frame // 2 + 1), dtype=xp.complex128)
                full[:, : self.kept] = moved
                made = fft.irfft(full, self.frame, axis=-1) * self.root[None, :]
        self._frames[m] = made
        return made

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1`` at the output rate: ``[channel, samples]``."""
        xp = self.xp
        channels = self.pack.header.channels
        first = k0 * self.step - self.reach
        last = k1 * self.step + self.reach
        low = xp.zeros((channels, last - first))
        half = self.frame // 2
        # Frames whose window meets [first, last), from the first that ends after 0.
        m0 = max(-((half - 1 - first) // self.hop), -(FRAMES_PER_STEP // 2) + 1)
        m1 = min((last + half - 1) // self.hop, (self.steps - 1) * FRAMES_PER_STEP + 1)
        heard = False
        for m in range(m0, m1 + 1):
            frame = self._frame(m)
            if frame is None:
                continue
            heard = True
            start = m * self.hop - half
            lo, hi = max(start, first), min(start + self.frame, last)
            low[:, lo - first : hi - first] += frame[:, lo - start : hi - start]
        # What the next run will not ask for again.
        for held, floor, ceiling in ((self._frames, m0, m1), (self._fields, k0 - 1, k1 + 1)):
            for key in [key for key in held if key < floor or key > ceiling]:
                del held[key]
        if not heard:
            return xp.zeros((channels, (k1 - k0) * self.step * self.factor))
        windows = xp.lib.stride_tricks.sliding_window_view(low, 2 * self.reach + 1, axis=1)
        return xp.matmul(windows, self.poly).reshape(channels, -1)
