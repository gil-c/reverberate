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

**More than two source positions** are read when the pack holds
``low/slot_pair``: every slot's response is weighted by ``low/slot_weight``,
which is given per frequency on a few knots and read between them in a
straight line. A weight that changes with frequency is a short filter with
no delay, as long before the arrival as after it, so the step's transform
then takes :data:`SLOT_REACH` samples more of the dry signal at both ends
and leaves them out of what it returns. A pack without those tables is
rendered as it always was, sample for sample.

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

__all__ = ["FRAMES_PER_STEP", "SLOT_REACH", "LowPart"]

#: Frames of the head's translation in one step.
FRAMES_PER_STEP = 4
#: Channels brought to the output rate at once.
CHANNELS_AT_ONCE = 8
#: What a slot's weights may reach either side of a response, in samples of the
#: low rate: 64 ms, three times the 20 ms period that knots 50 Hz apart give them.
SLOT_REACH = 256


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
        self.slots = self.low.slot_pair is not None
        self.pad = SLOT_REACH if self.slots else 0
        self.n = next_fast_len(h.low_samples + self.cover + 2 * self.pad, real=True)
        if self.low.slot_pair is not None and self.low.slot_knots_hz is not None:
            # Where each bin of a step's transform lies among the knots of the weights.
            knots = np.asarray(self.low.slot_knots_hz, dtype=float)
            at = np.fft.rfftfreq(self.n, 1.0 / self.rate) / (knots[1] - knots[0])
            at = np.minimum(at, knots.size - 1.0)
            self._knot = np.minimum(at.astype(np.int64), knots.size - 2)
            self._share = at - self._knot
            # Every slot's spectrum at both cells of the steps a run holds.
            pairs_held = max(pairs_held, 4 * int(self.low.slot_pair.shape[1]))
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
        last = end + self.pad
        if mode != 0 and not self.track.silent(last - self.n, last):
            fft = fft_module(xp)
            dry = fft.rfft(xp.asarray(self.track.read(last - self.n, last)))
            weight = float(self.low.position_weight[k])
            fields = []
            for b in range(2 if mode == 3 else 1):
                if self.slots:
                    spectrum = self._slotted(k, b)
                else:
                    spectrum = (1.0 - weight) * self._spectrum(int(self.low.pair[k, 0, b]))
                    if weight > 0.0:
                        spectrum = spectrum + weight * self._spectrum(int(self.low.pair[k, 1, b]))
                fields.append(spectrum * dry[None, :])
            made = fft.irfft(xp.stack(fields), self.n, axis=-1, **workers_of(xp, self.workers))[
                ..., self.n - self.pad - self.cover : self.n - self.pad
            ]
        self._fields[k] = made
        return made

    def _slotted(self, k: int, b: int) -> Any:
        """Step ``k``'s response at its cell ``b`` from every slot it reads, as a spectrum."""
        xp = self.xp
        assert self.low.slot_pair is not None and self.low.slot_weight is not None
        rows = np.asarray(self.low.slot_pair[k, :, b])
        if rows[1] < 0:
            return self._spectrum(int(rows[0]))  # one position, whose weight is one
        # A few numbers a slot, on the host: the weights at the knots, read at every bin.
        weights = np.asarray(self.low.slot_weight[k], dtype=float)
        lower, upper = weights[:, self._knot], weights[:, self._knot + 1]
        at_bins = xp.asarray(lower + (upper - lower) * self._share[None, :])
        spectrum = None
        for slot, row in enumerate(rows):
            if row < 0:
                continue
            term = at_bins[slot][None, :] * self._spectrum(int(row))
            spectrum = term if spectrum is None else spectrum + term
        return spectrum

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
        out = xp.zeros((channels, (k1 - k0) * self.step * self.factor))
        # A few channels at a time: the product lays its windows out, 1 MB a channel.
        for lo in range(0, channels, CHANNELS_AT_ONCE):
            hi = min(lo + CHANNELS_AT_ONCE, channels)
            out[lo:hi] = xp.matmul(windows[lo:hi], self.poly).reshape(hi - lo, -1)
        return out
