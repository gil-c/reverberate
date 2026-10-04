"""The late part: the dry signal through noise shaped by the step's histogram.

The model is ``mirror.render.tail_from_histogram``'s. Per 2 ms bin and per
band the histogram gives an energy and its directional moments; the energy
is drawn as ``tail_bursts`` bursts of noise, each from a direction sampled
with the density the moments give it, each normalised so the bank reads the
bin's energy back. What a scene adds is that the histogram changes: it is
the step's weighted sum of up to four of the pack's, in energy.

**One carrier a source.** The bursts' noise and the uniforms that pick
their directions are drawn once from the source's ``tail_seed`` by
:mod:`reverberate.render.noise`, the same bits on any device. A step shapes
that carrier; it does not draw another, or the tail would be a different
room twenty times a second.

**The response is kept as plane waves.** A burst's direction is a point of
the moments' quadrature, 45 of them at order 3, so a step's response is 45
signals and not 64 channels: the bursts are added onto their directions,
each band is filtered by the bank and the bands summed, in one transform a
direction and a band. The dry signal is convolved with those 45 and the
result encoded on the 64 harmonics at the end, which is the same sum in
another order. A response is kept while the step's histograms and weights
do not change: a source at a station heard from a seat computes one.

Between two steps the two responses' outputs are cross-faded, which is the
format's quasi-static rule: at time ``t`` the scene as it is at ``t``, for
the signal emitted so far.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

from reverberate.metrics import octave_bank
from reverberate.mirror.render import bank_reading
from reverberate.render import noise
from reverberate.render.dry import DryTrack
from reverberate.render.early import fft_module, run_all, workers_of
from reverberate.render.pack import ScenePack, Source, band_map
from reverberate.spatial.sh import quadrature, real_sh

__all__ = ["DIRECTION_STREAM", "TailPart"]

#: Streams of the generator: a burst's noise is ``band * bursts + burst``, and the
#: uniform that picks its direction in each bin the same plus this.
DIRECTION_STREAM = 1 << 31


class TailPart:
    """One source's tail, rendered a run of steps at a time."""

    def __init__(
        self,
        pack: ScenePack,
        source: Source,
        track: DryTrack,
        xp: Any,
        *,
        workers: int,
        held: int = 3,
    ) -> None:
        if source.tail is None:
            raise ValueError(f"the source {source.id!r} has no tail")
        h = pack.header
        m = pack.mirror
        self.pack, self.source, self.tail, self.track, self.xp = (
            pack,
            source,
            source.tail,
            track,
            xp,
        )
        self.workers = workers
        self.rate = h.sample_rate_hz
        self.step = h.step_samples
        self.bins = int(self.tail.energy.shape[1])
        self.bin_s = m.histogram_bin_s
        self.bin_samples = int(round(self.bin_s * self.rate))
        self.span = self.bins * self.bin_samples
        self.bursts = m.tail_bursts
        self.picks = band_map(h.bands_hz, h.bank)
        self.bands = len(h.bank)
        kernels = np.asarray(octave_bank(int(round(self.rate))).filters, dtype=float).T
        #: The bank delays by its centre tap; the dry signal is read that much ahead.
        self.centre = (kernels.shape[1] - 1) // 2
        self.n = next_fast_len(self.span + kernels.shape[1] + self.step, real=True)
        self.bank = xp.asarray(np.fft.rfft(kernels, self.n, axis=1))
        grid, weights = quadrature(2 * m.histogram_order + 2)
        self.weights = weights
        self.basis_low = real_sh(m.histogram_order, grid)
        self.encode = xp.asarray(np.ascontiguousarray(real_sh(h.order, grid).T))
        self.directions = grid.shape[0]
        self.reading = np.asarray(bank_reading(self.rate))
        self.band_power = (10.0 ** (np.asarray(m.tail_gain_db, dtype=float) / 10.0))[self.picks]
        self.lead = int(round(m.lead_s * self.rate))
        self.gain = 10.0 ** (np.asarray(source.level.high_gain_db, dtype=float) / 20.0)
        self.air = self._air_power() if pack.air.enabled else None
        self._carrier: Any = None
        self._have: Any = None
        self._drawn: np.ndarray | None = None
        self._responses: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._held = held

    def _air_power(self) -> np.ndarray:
        """``[bin, band]``: the share of a band's energy the air leaves at each bin's time."""
        freqs = np.fft.rfftfreq(self.n, 1.0 / self.rate)
        shape = np.abs(
            np.fft.rfft(np.asarray(octave_bank(int(round(self.rate))).filters).T, self.n)
        )
        power = shape**2  # [band, f]
        attenuation = self.pack.air.atmosphere.attenuation_np_per_m(freqs)
        times = (np.arange(self.bins) + 0.5) * self.bin_s
        loss = np.exp(
            -2.0 * attenuation[None, :] * (self.pack.header.sound_speed_m_s * times)[:, None]
        )
        return np.asarray((loss @ power.T) / power.sum(axis=1)[None, :])

    def _draw(self) -> None:
        """The source's carrier and direction uniforms, once."""
        xp = self.xp
        streams = np.arange(self.bands * self.bursts)
        carrier = noise.carrier(self.source.tail_seed, streams, 0, self.span, xp)
        # [band, bin, burst, sample]: a bin's bursts side by side, as they are laid.
        self._carrier = xp.ascontiguousarray(
            carrier.reshape(self.bands, self.bursts, self.bins, self.bin_samples).transpose(
                0, 2, 1, 3
            )
        )
        self._have = xp.sum(self._carrier**2, axis=-1)  # [band, bin, burst]
        drawn = noise.uniform(self.source.tail_seed, DIRECTION_STREAM + streams, 0, self.bins, np)
        self._drawn = drawn.reshape(self.bands, self.bursts, self.bins).transpose(2, 0, 1)

    def _key(self, k: int) -> tuple[Any, ...] | None:
        if not bool(self.source.audible[k]) or int(self.tail.hist[k, 0, 0]) < 0:
            return None
        rows = slice(int(self.source.early.offsets[k]), int(self.source.early.offsets[k + 1]))
        # Where the tail takes over: tail_from_s after the step's first arrival; -1
        # when the step has none, and the histogram's own first bin decides.
        from_bin = -1
        if rows.stop > rows.start:
            start = float(np.min(self.source.early.delay_s[rows]))
            from_bin = int(np.ceil((start + self.pack.mirror.tail_from_s) / self.bin_s))
        return (
            tuple(int(v) for v in np.asarray(self.tail.hist[k]).ravel()),
            float(self.tail.position_weight[k]),
            float(self.tail.cell_weight[k]),
            from_bin,
        )

    def _response(self, key: tuple[Any, ...]) -> Any:
        """A step's response as plane waves, in the frequency domain: ``[direction, f]``."""
        if key in self._responses:
            self._responses.move_to_end(key)
            return self._responses[key]
        if self._carrier is None:
            self._draw()
        assert self._drawn is not None
        xp = self.xp
        rows, w_source, w_cell, from_bin = key
        weights = np.outer([1.0 - w_source, w_source], [1.0 - w_cell, w_cell]).ravel()
        bands7 = len(self.pack.header.bands_hz)
        wanted = np.zeros((self.bins, self.bands))
        moments = np.zeros((self.bins, bands7, self.basis_low.shape[1]))
        raw = np.zeros((self.bins, bands7))
        for row, weight in zip(rows, weights, strict=True):
            if row < 0 or weight <= 0.0:
                continue
            energy = np.asarray(self.tail.energy[row], dtype=float)
            raw += weight * energy
            wanted += weight * energy[:, self.picks] * np.asarray(self.tail.scale[row])[None, :]
            moments += weight * np.asarray(self.tail.moments[row], dtype=float)
        if from_bin < 0:
            # No arrival at the step: the tail starts at the first bin that holds energy.
            holding = np.any(raw > 0.0, axis=1)
            from_bin = int(np.argmax(holding)) if holding.any() else self.bins
        # What to synthesise so the bank reads the energy wanted: its own
        # reading inverted, clipped at zero, as the mirror's tail does.
        wanted = wanted * self.band_power[None, :]
        wanted = np.maximum(np.linalg.solve(self.reading, wanted.T).T, 0.0)
        wanted[: min(from_bin, self.bins)] = 0.0
        if self.air is not None:
            wanted = wanted * self.air
        density = np.maximum(moments[:, self.picks, :] @ self.basis_low.T, 0.0) * self.weights
        empty = density.sum(axis=-1, keepdims=True) <= 0.0
        density = np.where(empty, self.weights, density)
        cumulative = np.cumsum(density / density.sum(axis=-1, keepdims=True), axis=-1)
        cumulative[..., -1] = np.inf
        # The first direction whose cumulative probability passes the draw: [bin, band, burst].
        chosen = np.argmax(self._drawn[..., None] < cumulative[:, :, None, :], axis=-1)
        share = xp.asarray(wanted.T / self.bursts)[:, :, None]  # [band, bin, 1]
        gain = xp.where(share > 0.0, xp.sqrt(share / xp.maximum(self._have, 1e-30)), 0.0)
        fft = fft_module(xp)
        every = xp.arange(self.bins)[:, None]
        each = xp.arange(self.bursts)[None, :]
        carrier = self._carrier

        def one(band: int) -> Any:
            """One band's bursts on their directions, through its filter: ``[direction, f]``."""
            # Each burst onto its direction at its gain: per bin, a matrix with one
            # entry a burst, times the bin's bursts.
            onto = xp.zeros((self.bins, self.directions, self.bursts))
            onto[every, xp.asarray(chosen[:, band, :]), each] = gain[band]
            laid = xp.matmul(onto, carrier[band])  # [bin, direction, sample]
            padded = xp.zeros((self.directions, self.n))
            padded[:, : self.span] = laid.transpose(1, 0, 2).reshape(self.directions, self.span)
            spectrum = fft.rfft(padded, axis=-1)
            # The filter keeps a share of a white burst's energy; restored on the
            # omnidirectional sum, which is the response's first channel.
            asked = float(xp.sum(laid.sum(axis=1) ** 2))
            omni = xp.abs(spectrum.sum(axis=0) * self.bank[band]) ** 2
            kept = float(2.0 * xp.sum(omni) - omni[0] - (omni[-1] if self.n % 2 == 0 else 0.0))
            kept /= self.n
            if asked <= 0.0 or kept <= 0.0:
                return None
            spectrum *= (self.bank[band] * float(np.sqrt(asked / kept)))[None, :]
            return spectrum

        # The bands are independent, a thread each on the host; they are summed in
        # their own order whatever finished first.
        held = [band for band in range(self.bands) if np.any(wanted[:, band] > 0.0)]
        response = xp.zeros((self.directions, self.n // 2 + 1), dtype=xp.complex128)
        for spectrum in run_all(one, held, xp, self.workers):
            if spectrum is not None:
                response += spectrum
        self._responses[key] = response
        while len(self._responses) > self._held:
            self._responses.popitem(last=False)
        return response

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1``: ``[channel, (k1 - k0) step]`` on ``xp``."""
        xp = self.xp
        step = self.step
        fft = fft_module(xp)
        kw = workers_of(xp, self.workers)
        out = xp.zeros((self.pack.header.channels, (k1 - k0) * step))
        u = xp.asarray(np.arange(step) / step)
        for k in range(k0, k1):
            keys = (self._key(k), self._key(k + 1))
            if keys[0] is None and keys[1] is None:
                continue
            end = (k + 1) * step + self.centre - self.lead
            if self.track.silent(end - self.n, end):
                continue
            dry = fft.rfft(xp.asarray(self.track.read(end - self.n, end)))
            fades = ((1.0 - u) * self.gain[k], u * self.gain[k + 1])
            if keys[0] == keys[1]:
                assert keys[0] is not None
                waves = fft.irfft(self._response(keys[0]) * dry[None, :], self.n, axis=-1, **kw)
                waves = waves[:, self.n - step :] * (fades[0] + fades[1])[None, :]
            else:
                waves = xp.zeros((self.directions, step))
                for key, fade in zip(keys, fades, strict=True):
                    if key is None:
                        continue
                    made = fft.irfft(self._response(key) * dry[None, :], self.n, axis=-1, **kw)
                    waves = waves + made[:, self.n - step :] * fade[None, :]
            out[:, (k - k0) * step : (k - k0 + 1) * step] = self.encode @ waves
        return out
