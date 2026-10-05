"""The late part: the dry signal through noise shaped by the step's histograms.

Per 2 ms bin and per band a histogram gives an energy and its directional
moments. The tail is noise that carries that energy from those directions,
as ``mirror.render.tail_from_histogram``'s is, and what a scene adds is that
the histogram changes: a step reads up to four of the pack's, with weights.

**One carrier a source, already through the bank.** For every bank band and
every direction of the moments' quadrature, 45 at order 3, the source has
one stream of noise, drawn from its ``tail_seed`` by
:mod:`reverberate.render.noise` (the same bits on any device) and passed
through the band's filter once. A histogram shapes that carrier: each
stream is multiplied by the square root of the energy its band and
direction hold in the bin, the energy being spread over the neighbouring
bins as the band's filter would spread it. The expected energy of every
bin, band and direction is what filtering shaped white noise gives, and the
filter is paid once a source instead of once a response. The bands are
then scaled so that the bank reads, on the first channel of the response as
it is rendered, what the histogram holds.

**A histogram has one response; a step mixes them, in energy.** A step's
response is the sum of its histograms' responses, each times the square
root of its weight (``scene-pack.md``). The histograms of a step read the
carrier from places further apart than the bank's filter is long, so their
noises are independent and the energy of the sum is the weighted sum of
their energies, in every bin, band and direction: the format's
interpolation in energy. The dry signal is convolved with each histogram's
response once a run and the steps are mixes of those outputs, so a walk
costs a convolution a histogram in use and not a response a step. A
histogram is the same noise wherever the scene reads it: coming back to a
place is coming back to its tail.

**The response is kept as plane waves**: 45 signals and not 64 channels,
encoded on the harmonics at the end, which is the same sum in another
order.

**Where the tail starts** is the step's (``tail_from_s`` after its first
arrival), not the histogram's. A histogram's response is made from the
earliest start any step of the scene reads it at; a step that starts later
takes the difference away, which is a response a few bins long.

Between two steps the two responses' outputs are cross-faded, which is the
format's quasi-static rule: at time ``t`` the scene as it is at ``t``, for
the signal emitted so far.

**What is kept** is counted in bytes for the whole process (:data:`HELD`):
the carriers in single precision and the spectra of the histograms in use.
The transforms and what leaves the part are in double precision.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import next_fast_len

from reverberate.compute import to_numpy
from reverberate.metrics import octave_bank
from reverberate.mirror.render import bank_reading
from reverberate.render import noise
from reverberate.render.dry import DryTrack
from reverberate.render.early import fft_module, workers_of
from reverberate.render.pack import ScenePack, Source, band_map
from reverberate.spatial.sh import quadrature, real_sh

__all__ = ["HELD", "Held", "TailPart"]

#: What the tails of one process keep between runs, in bytes: the carriers (95 MB a
#: source) and the histograms' spectra (30 MB each), the least recently used out first.
HELD_BYTES = 320 << 20
#: What one source's carrier and a few of its histograms' spectra take, in bytes: what
#: an engine that mixes many sources adds to :data:`HELD_BYTES` for each.
CARRIER_BYTES = 100 << 20
#: Places of the carrier a histogram's response may be read from. Two histograms of one
#: step read it further apart than the bank's filter is long, so their noises are
#: independent and their energies add.
PLACES = 16
#: The share of a band filter's energy left out at each end of the frequencies it is
#: read over, when a response is brought to what the bank reads.
SKIRT = 1e-9
#: The share of a band's energy the dry signal's mask must leave for the band to be
#: brought to what the bank reads of it: under it the band is not heard.
RENDERED = 1e-3
#: Directions transformed at once by a host thread: a block that stays in a core's
#: cache, so that processes side by side do not queue on the memory bus.
HOST_BLOCK = 4


class Held:
    """Values kept by key up to a number of bytes, the least recently used out first."""

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def held_bytes(self) -> int:
        return self._bytes

    def get(self, key: Hashable, make: Callable[[], Any]) -> Any:
        """The value of ``key``, made on first use. What is made is kept whatever its size."""
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
        value = make()
        size = sum(int(part.nbytes) for part in (value if isinstance(value, tuple) else (value,)))
        with self._lock:
            if key not in self._items:
                self._items[key] = (value, size)
                self._bytes += size
            while self._bytes > self.limit_bytes and len(self._items) > 1:
                oldest = next(iter(self._items))
                if oldest == key:
                    break
                self._bytes -= self._items.pop(oldest)[1]
        return value

    def forget(self, owner: object) -> None:
        """Drop everything whose key starts with ``owner``."""
        with self._lock:
            for key in [k for k in self._items if isinstance(k, tuple) and k[0] is owner]:
                self._bytes -= self._items.pop(key)[1]


#: The process's own: an engine a source and a worker that keeps several do not add up.
HELD = Held(HELD_BYTES)


@dataclass(frozen=True)
class _Step:
    """What a step reads: its histograms with their weights, and the bin it starts at."""

    rows: tuple[int, ...]
    weights: tuple[float, ...]
    start: int


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
        mask: np.ndarray | None = None,
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
        self.steps = h.steps
        self.bins = int(self.tail.energy.shape[1])
        self.bin_s = m.histogram_bin_s
        self.bin_samples = int(round(self.bin_s * self.rate))
        self.span = self.bins * self.bin_samples
        self.picks = band_map(h.bands_hz, h.bank)
        self.bands = len(h.bank)
        self.kernels = np.asarray(octave_bank(int(round(self.rate))).filters, dtype=float).T
        self.taps = int(self.kernels.shape[1])
        #: What a unit white noise holds after each band's filter, a sample.
        self.kept = np.sum(self.kernels**2, axis=1)
        self.spread = self._spread()
        #: The zero phase filter the dry signal came through, if any: the crossover's.
        self.mask = None if mask is None else np.asarray(mask, dtype=float)
        self.nominal, self.rendered = self._nominal()
        #: Bins between two places of the carrier: no shorter than the bank's filter.
        self.apart = -(-(self.taps - 1) // self.bin_samples)
        grid, weights = quadrature(2 * m.histogram_order + 2)
        self.weights = weights
        self.basis_low = real_sh(m.histogram_order, grid)
        self.encode = xp.asarray(np.ascontiguousarray(real_sh(h.order, grid).T))
        self.directions = grid.shape[0]
        self.block = HOST_BLOCK if xp is np and workers == 1 else self.directions
        self.reading = np.asarray(bank_reading(self.rate))
        self.band_power = (10.0 ** (np.asarray(m.tail_gain_db, dtype=float) / 10.0))[self.picks]
        self.lead = int(round(m.lead_s * self.rate))
        self.gain = 10.0 ** (np.asarray(source.level.high_gain_db, dtype=float) / 20.0)
        self.air = self._air_power() if pack.air.enabled else None
        self._read_steps()
        self._energies: OrderedDict[int, np.ndarray] = OrderedDict()
        self._amplitudes: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._norms: dict[int, np.ndarray] = {}
        self._bank: tuple[int, list[tuple[slice, Any]]] | None = None
        self._products: OrderedDict[tuple[int, int, int], float] = OrderedDict()
        self._token = object()
        weakref.finalize(self, HELD.forget, self._token)

    # ------------------------------------------------------------------
    # what is fixed for the source
    # ------------------------------------------------------------------

    def _spread(self) -> np.ndarray:
        """``[band, 2 r + 1]``: the share of a bin's energy each band's filter lays ``-r..r`` on.

        White noise of energy ``a`` a sample over one bin, through the band's
        filter with its centre tap on the sample, holds ``a kernel^2`` summed
        over the taps that reach each sample. Summed over a bin that is the
        bin's energy spread over its neighbours with these shares, which add
        to one.
        """
        centre = (self.taps - 1) // 2
        # A sample ``t`` of bin 0 takes tap ``m`` from the sample ``t - m + centre``.
        came_from = (
            np.arange(self.bin_samples)[:, None] - np.arange(self.taps)[None, :] + centre
        ) // self.bin_samples
        radius = int(np.abs(came_from).max())
        spread = np.zeros((self.bands, 2 * radius + 1))
        for band in range(self.bands):
            power = np.broadcast_to(self.kernels[band] ** 2, came_from.shape)
            # Energy of the bin ``j'`` reaches the bin ``j' + delta``, ``delta = -came_from``.
            np.add.at(spread[band], (radius - came_from).ravel(), power.ravel())
        return np.asarray(spread / spread.sum(axis=1, keepdims=True))

    def _air_power(self) -> np.ndarray:
        """``[bin, band]``: the share of a band's energy the air leaves at each bin's time."""
        # A filter of ``taps`` has a power spectrum of twice as many terms: a grid of
        # sixteen times as many frequencies holds its product with the air's smooth loss.
        n = next_fast_len(16 * self.taps, real=True)
        freqs = np.fft.rfftfreq(n, 1.0 / self.rate)
        power = np.abs(np.fft.rfft(self.kernels, n)) ** 2  # [band, f]
        if n % 2 == 0:
            power[:, -1] *= 0.5
        power[:, 0] *= 0.5
        attenuation = self.pack.air.atmosphere.attenuation_np_per_m(freqs)
        times = (np.arange(self.bins) + 0.5) * self.bin_s
        loss = np.exp(
            -2.0 * attenuation[None, :] * (self.pack.header.sound_speed_m_s * times)[:, None]
        )
        return np.asarray((loss @ power.T) / power.sum(axis=1)[None, :])

    def _read_steps(self) -> None:
        """Every step's histograms, weights and first bin, and each histogram's earliest start."""
        early = self.source.early
        offsets = np.asarray(early.offsets, dtype=np.int64)
        delay_s = np.asarray(early.delay_s, dtype=float)
        holds = np.diff(offsets)[: self.steps] > 0
        # Where the tail takes over: tail_from_s after the step's first arrival. A step
        # with no arrival starts at the histogram's own first bin: nothing is cut.
        start = np.zeros(self.steps, dtype=np.int64)
        if holds.any():
            first = np.minimum.reduceat(delay_s, offsets[:-1][: self.steps][holds])
            start[holds] = np.ceil((first + self.pack.mirror.tail_from_s) / self.bin_s).astype(
                np.int64
            )
        self.start = np.clip(start, 0, self.bins)
        rows = np.asarray(self.tail.hist, dtype=np.int64).reshape(self.steps, 4)
        w_source = np.asarray(self.tail.position_weight, dtype=float)
        w_cell = np.asarray(self.tail.cell_weight, dtype=float)
        weights = (
            np.stack([1.0 - w_source, w_source], axis=1)[:, :, None]
            * np.stack([1.0 - w_cell, w_cell], axis=1)[:, None, :]
        ).reshape(self.steps, 4)
        heard = np.asarray(self.source.audible, dtype=bool)[: self.steps] & (rows[:, 0] >= 0)
        self.rows = rows
        self.slot_weights = np.where((rows >= 0) & (weights > 0.0) & heard[:, None], weights, 0.0)
        count = int(self.tail.energy.shape[0])
        earliest = np.full(count, self.bins, dtype=np.int64)
        used = self.slot_weights > 0.0
        np.minimum.at(earliest, rows[used], np.broadcast_to(self.start[:, None], rows.shape)[used])
        self.earliest = earliest
        # The place of the carrier each histogram reads. Taken in the order of their
        # rows, each the lowest that no lower row it shares a step with holds.
        met: dict[int, set[int]] = {}
        for together in np.unique(np.where(used, rows, -1), axis=0):
            held = [int(row) for row in together if row >= 0]
            for row in held:
                met.setdefault(row, set()).update(held)
        place = np.arange(count) % PLACES
        for row in sorted(met):
            taken = {int(place[other]) for other in met[row] if other < row}
            free = [candidate for candidate in range(PLACES) if candidate not in taken]
            if free:
                place[row] = free[0]
        self.place = place

    def _carrier(self) -> Any:
        """``[band, direction, bin, sample]``, float32: the noise through the bank, unit variance.

        Stream ``band * directions + direction`` of the generator, from ``taps - 1``
        samples before the first bin so that the filter is full there, and
        :data:`PLACES` - 1 times ``apart`` bins longer than a histogram: a
        histogram reads it from its own place.
        """

        def make() -> Any:
            xp = self.xp
            fft = fft_module(xp)
            kw = workers_of(xp, self.workers)
            bins = self.bins + (PLACES - 1) * self.apart
            count = bins * self.bin_samples + self.taps - 1
            n = next_fast_len(count + self.taps - 1, real=True)
            made = xp.zeros((self.bands, self.directions, bins, self.bin_samples), dtype=xp.float32)
            for band in range(self.bands):
                streams = band * self.directions + np.arange(self.directions)
                white = noise.carrier(self.source.tail_seed, streams, 0, count, xp)
                kernel = xp.asarray(self.kernels[band] / np.sqrt(self.kept[band]))
                through = fft.irfft(
                    fft.rfft(white, n, axis=-1, **kw) * fft.rfft(kernel, n)[None, :],
                    n,
                    axis=-1,
                    **kw,
                )[:, self.taps - 1 : count]
                made[band] = through.reshape(self.directions, bins, self.bin_samples)
            return made

        return HELD.get((self._token, "carrier"), make)

    def _noise(self, row: int, lo: int = 0, hi: int | None = None) -> Any:
        """The carrier as the histogram ``row`` reads it, over its bins ``lo`` to ``hi``."""
        first = int(self.place[row]) * self.apart
        return self._carrier()[:, :, first + lo : first + (self.bins if hi is None else hi)]

    # ------------------------------------------------------------------
    # a histogram
    # ------------------------------------------------------------------

    def _energy(self, row: int) -> np.ndarray:
        """``[bin, band, direction]``: what the histogram asks of each direction, uncut, host."""
        if row in self._energies:
            self._energies.move_to_end(row)
            return self._energies[row]
        energy = np.asarray(self.tail.energy[row], dtype=float)
        wanted = energy[:, self.picks] * np.asarray(self.tail.scale[row], dtype=float)[None, :]
        # What to synthesise so the bank reads the energy wanted: its own
        # reading inverted, clipped at zero, as the mirror's tail does.
        wanted = wanted * self.band_power[None, :]
        wanted = np.maximum(np.linalg.solve(self.reading, wanted.T).T, 0.0)
        if self.air is not None:
            wanted = wanted * self.air
        moments = np.asarray(self.tail.moments[row], dtype=float)
        density = np.maximum(moments[:, self.picks, :] @ self.basis_low.T, 0.0) * self.weights
        empty = density.sum(axis=-1, keepdims=True) <= 0.0
        density = np.where(empty, self.weights, density)
        made = np.asarray(wanted[:, :, None] * (density / density.sum(axis=-1, keepdims=True)))
        self._energies[row] = made
        while len(self._energies) > 8:
            self._energies.popitem(last=False)
        return made

    def _raw(self, row: int, start: int) -> np.ndarray:
        """``[band, direction, bin]``: the carrier's gain for a tail from ``start``, host.

        The energy from ``start`` on, spread as the bank spreads it, a sample
        of unit variance noise.
        """
        energy = self._energy(row)
        if start > 0:
            energy = energy.copy()
            energy[:start] = 0.0
        radius = self.spread.shape[1] // 2
        spread = np.zeros_like(energy)
        for delta in range(-radius, radius + 1):
            share = self.spread[:, radius + delta][None, :, None]
            to = slice(max(delta, 0), self.bins + min(delta, 0))
            of = slice(max(-delta, 0), self.bins + min(-delta, 0))
            spread[to] += share * energy[of]
        return np.asarray(np.sqrt(spread / self.bin_samples).transpose(1, 2, 0))

    def _through(self, n: int) -> np.ndarray:
        """``[f]``: what is rendered of a spectrum of ``n`` points, as an energy.

        The dry signal's mask, with the weights that turn a sum over the
        frequencies of ``rfft`` into an energy: times a signal's
        ``abs(rfft) ** 2`` and summed, the energy of the signal once rendered.
        """
        weights = np.full(n // 2 + 1, 2.0 / n)
        weights[0] = 1.0 / n
        if n % 2 == 0:
            weights[-1] = 1.0 / n
        if self.mask is not None:
            weights = weights * np.abs(np.fft.rfft(self.mask, n)) ** 2
        return weights

    def _nominal(self) -> tuple[np.ndarray, np.ndarray]:
        """What the bank is to read, and what is rendered of each band.

        ``[b, k]``: the energy the bank's band ``b`` reads, once rendered, of
        unit energy of white noise through band ``k``'s filter, which is the
        reference renderer's tail in expectation; and ``[k]``, the share of
        that unit energy the dry signal's mask leaves.
        """
        n = next_fast_len(16 * max(self.taps, 0 if self.mask is None else self.mask.size))
        power = np.abs(np.fft.rfft(self.kernels, n, axis=1)) ** 2
        through = self._through(n)
        shape = power / self.kept[:, None]
        return np.asarray((power * through[None, :]) @ shape.T), np.asarray(shape @ through)

    def _bank_power(self, n: int) -> list[tuple[slice, Any]]:
        """Per band, what its filter keeps of what is rendered, and where.

        The band's filter times :meth:`_through`, over the frequencies that
        hold all of the filter's energy but :data:`SKIRT` at each end.
        """
        if self._bank is None or self._bank[0] != n:
            power = np.abs(np.fft.rfft(self.kernels, n, axis=1)) ** 2
            kept = power * self._through(n)[None, :]
            held = []
            for band in range(self.bands):
                share = np.cumsum(power[band]) / power[band].sum()
                lo = int(np.searchsorted(share, SKIRT))
                hi = int(np.searchsorted(share, 1.0 - SKIRT)) + 1
                held.append((slice(lo, hi), self.xp.asarray(kept[band, lo:hi])))
            self._bank = (n, held)
        return self._bank[1]

    def _norm(self, row: int) -> np.ndarray:
        """``[band]``: what brings the histogram's response to what the bank is to read of it.

        The energies were asked through the inverse of the bank's reading of
        noise shaped by each band's filter. The carrier is that noise times
        gains that move from bin to bin, which widens its band a little, and
        one draw of it reads off its expectation, the more so the shorter
        the tail and the lower the band. So the bank reads the response as
        it is rendered: the one from the histogram's earliest start, on the
        sum of its directions (the first channel), through the mask the dry
        signal came through; each band's share and what two shares have in
        common. The bands are then scaled, one gain each, so that every band
        of the bank reads what the reference renderer's tail reads in
        expectation. A band of which the mask leaves under :data:`RENDERED`
        is not heard and keeps its gain of one.
        """
        if row not in self._norms:
            xp = self.xp
            start = int(self.earliest[row])
            raw = xp.asarray(self._raw(row, start).astype(np.float32))
            omni = xp.einsum("bdj,bdjs->bjs", raw, self._noise(row)).astype(xp.float64)
            # Room for the response, the band's filter and the mask: nothing comes round.
            masked = 0 if self.mask is None else self.mask.size
            n = next_fast_len(self.span + self.taps + masked, real=True)
            spectrum = fft_module(xp).rfft(omni.reshape(self.bands, self.span), n, axis=-1)
            # read[b, j, k]: what the bank's band b reads of the response's bands j and k
            # together, over the frequencies its filter keeps.
            read = np.zeros((self.bands, self.bands, self.bands))
            for band, (where, weight) in enumerate(self._bank_power(n)):
                shares = spectrum[:, where]
                read[band] = np.asarray(
                    to_numpy(((shares * weight[None, :]) @ shares.conj().T).real)
                )
            asked = self._energy(row)[start:].sum(axis=(0, 2))
            heard = (asked > 0.0) & (np.einsum("bbb->b", read) > 0.0)
            free = heard & (self.rendered >= RENDERED)
            gains = np.where(heard & ~free, 1.0, 0.0)
            if free.any():
                wanted = self.nominal @ np.where(heard, asked, 0.0)
                gains = _gains(read, wanted, gains, free)
            self._norms[row] = gains
        return self._norms[row]

    def _amplitude(self, row: int, start: int) -> np.ndarray:
        """``[band, direction, bin]``, float32: the gains of the response from ``start``."""
        key = (row, start)
        if key in self._amplitudes:
            self._amplitudes.move_to_end(key)
            return self._amplitudes[key]
        made = np.asarray(
            (self._raw(row, start) * self._norm(row)[:, None, None]).astype(np.float32)
        )
        self._amplitudes[key] = made
        while len(self._amplitudes) > 24:
            self._amplitudes.popitem(last=False)
        return made

    def _reference(self, row: int) -> Any:
        """``[band, direction, bin]``, float32 on ``xp``: the gains of the histogram's own."""

        def make() -> Any:
            start = int(self.earliest[row])
            return self.xp.asarray(self._amplitude(row, start))

        return HELD.get((self._token, "reference", row), make)

    def _spectrum(self, row: int, n: int) -> Any:
        """The histogram's response as plane waves, in the frequency domain: ``[direction, f]``."""

        def make() -> Any:
            xp = self.xp
            padded = xp.zeros((self.directions, n))
            padded[:, : self.span] = xp.einsum(
                "bdj,bdjs->djs", self._reference(row), self._noise(row)
            ).reshape(self.directions, self.span)
            return fft_module(xp).rfft(padded, axis=-1, **workers_of(xp, self.workers))

        return HELD.get((self._token, "spectrum", row, n), make)

    def _late(self, row: int, start: int) -> tuple[int, Any]:
        """What a tail from ``start`` lacks of the histogram's own: its first sample, and it.

        ``[direction, sample]`` on ``xp``, over the bins the two differ in:
        those between the two starts and as far beyond as the bank spreads.
        """
        xp = self.xp
        radius = self.spread.shape[1] // 2
        first = int(self.earliest[row])
        lo = max(first - radius, 0)
        hi = min(start + radius + 1, self.bins)
        less = self._amplitude(row, first)[:, :, lo:hi] - self._amplitude(row, start)[:, :, lo:hi]
        made = xp.einsum("bdj,bdjs->djs", xp.asarray(less), self._noise(row, lo, hi))
        return lo * self.bin_samples, made.reshape(self.directions, -1).astype(xp.float64)

    # ------------------------------------------------------------------
    # a step
    # ------------------------------------------------------------------

    def _key(self, k: int) -> _Step | None:
        weights = self.slot_weights[k]
        if not np.any(weights > 0.0):
            return None
        summed: dict[int, float] = {}
        for row, weight in zip(self.rows[k], weights, strict=True):
            if weight > 0.0:
                summed[int(row)] = summed.get(int(row), 0.0) + float(weight)
        rows = tuple(sorted(summed))
        return _Step(rows, tuple(summed[row] for row in rows), int(self.start[k]))

    def _product(self, a: int, b: int, start: int) -> float:
        key = (min(a, b), max(a, b), start)
        if key in self._products:
            self._products.move_to_end(key)
            return self._products[key]
        made = float(
            np.vdot(
                self._amplitude(a, start).astype(np.float64),
                self._amplitude(b, start).astype(np.float64),
            )
        )
        self._products[key] = made
        while len(self._products) > 4096:
            self._products.popitem(last=False)
        return made

    def _gain(self, key: _Step) -> float:
        """What keeps the mix's energy the weighted sum of its histograms' energies.

        One when every histogram of the step reads the carrier from its own
        place, which is the rule. Two that share a place add in amplitude and
        not in energy: the energy expected of the sum then holds the products
        of their gains, and this brings it back to what was asked.
        """
        places = [int(self.place[row]) for row in key.rows]
        if len(set(places)) == len(places):
            return 1.0
        roots = np.sqrt(np.asarray(key.weights))
        products = np.array(
            [
                [
                    self._product(a, b, key.start) if places[i] == places[j] else 0.0
                    for j, b in enumerate(key.rows)
                ]
                for i, a in enumerate(key.rows)
            ]
        )
        wanted = float(roots**2 @ np.diag(products))
        mixed = float(roots @ products @ roots)
        return float(np.sqrt(wanted / mixed)) if wanted > 0.0 and mixed > 0.0 else 1.0

    def _mixed(self, key: _Step, n: int) -> Any:
        """A step's response from its histograms' earliest starts: ``[direction, f]``."""

        def make() -> Any:
            gain = self._gain(key)
            total = None
            for row, weight in zip(key.rows, key.weights, strict=True):
                part = self._spectrum(row, n) * (gain * float(np.sqrt(weight)))
                total = part if total is None else total + part
            return total

        if key.weights == (1.0,):
            return self._spectrum(key.rows[0], n)
        return HELD.get((self._token, "mixed", key.rows, key.weights, n), make)

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1``: ``[channel, (k1 - k0) step]`` on ``xp``."""
        xp = self.xp
        step = self.step
        length = (k1 - k0) * step
        channels = self.pack.header.channels
        end = k1 * step - self.lead
        n = next_fast_len(length + self.span, real=True)
        if self.track.silent(end - n, end):
            return xp.zeros((channels, length))
        # Every step's share of the run: its fade in, its fade out, its level.
        u = np.arange(step) / step
        shares: dict[_Step, np.ndarray] = {}
        for k in range(k0, k1 + 1):
            key = self._key(k)
            if key is None:
                continue
            share = shares.setdefault(key, np.zeros(length))
            if k > k0:
                share[(k - 1 - k0) * step : (k - k0) * step] += u * self.gain[k]
            if k < k1:
                share[(k - k0) * step : (k - k0 + 1) * step] += (1.0 - u) * self.gain[k]
        if not shares:
            return xp.zeros((channels, length))
        # The responses the run is made of, each with what multiplies its output.
        spectra: list[tuple[Any, np.ndarray]] = []
        lates: dict[tuple[int, int], np.ndarray] = {}
        own: dict[int, np.ndarray] = {}
        nothing = np.zeros(length)
        for key, share in shares.items():
            gain = self._gain(key)
            for row, weight in zip(key.rows, key.weights, strict=True):
                # In energy: the histograms' noises are independent, their weights' roots add.
                root = gain * float(np.sqrt(weight))
                if len(shares) > 1:
                    own[row] = own.get(row, nothing) + root * share
                if key.start > self.earliest[row]:
                    late = (row, key.start)
                    lates[late] = lates.get(late, nothing) + root * share
        if len(shares) == 1:
            # One response the whole run: its histograms are mixed once, as spectra.
            (key, share), *_ = shares.items()
            spectra.append((self._mixed(key, n), share))
        else:
            spectra.extend((self._spectrum(row, n), own[row]) for row in sorted(own))
        waves = xp.zeros((self.directions, length))
        fft = fft_module(xp)
        kw = workers_of(xp, self.workers)
        dry = fft.rfft(xp.asarray(self.track.read(end - n, end)))
        for spectrum, share in spectra:
            self._add(waves, spectrum, dry, n, xp.asarray(share))
        for row, start in sorted(lates):
            first, response = self._late(row, start)
            size = next_fast_len(length + response.shape[1], real=True)
            heard = fft.rfft(xp.asarray(self.track.read(end - first - size, end - first)))
            spectrum = fft.rfft(response, size, axis=-1, **kw)
            self._add(waves, spectrum, heard, size, xp.asarray(-lates[(row, start)]))
        return self.encode @ waves

    def _add(self, waves: Any, spectrum: Any, dry: Any, n: int, share: Any) -> None:
        """``waves`` plus ``share`` times the end of ``dry`` through ``spectrum``, in blocks."""
        fft = fft_module(self.xp)
        kw = workers_of(self.xp, self.workers)
        length = waves.shape[1]
        for lo in range(0, self.directions, self.block):
            hi = min(lo + self.block, self.directions)
            made = fft.irfft(spectrum[lo:hi] * dry[None, :], n, axis=-1, **kw)
            waves[lo:hi] += made[:, n - length :] * share[None, :]


def _gains(read: np.ndarray, wanted: np.ndarray, gains: np.ndarray, free: np.ndarray) -> np.ndarray:
    """``gains`` with those of the ``free`` bands set so that ``a read[b] a = wanted[b]`` there.

    From the gains that would do were the bands' shares to have nothing in
    common, by Newton's method: the shares in common are small beside a
    band's own. A band the others already fill past what is wanted of it
    would need a gain under zero: it is given none, its own reading is
    left as the others make it, and the rest are set again.
    """
    made = np.array(gains, dtype=float)
    free = np.array(free, dtype=bool)
    own = np.einsum("bjj->bj", read)
    while free.any():
        rest = wanted[free] - own[free][:, ~free] @ made[~free] ** 2
        first = np.linalg.solve(own[free][:, free], rest)
        if np.any(first < 0.0):
            lowest = np.flatnonzero(free)[int(np.argmin(first))]
            free[lowest], made[lowest] = False, 0.0
            continue
        made[free] = np.sqrt(first)
        start = made.copy()
        for _ in range(30):
            off = (np.einsum("j,bjk,k->b", made, read, made) - wanted)[free]
            if np.all(np.abs(off) <= 1e-12 * np.abs(wanted[free])):
                return made
            slope = 2.0 * np.einsum("bjk,k->bj", read, made)[free][:, free]
            made[free] = made[free] - np.linalg.solve(slope, off)
            if np.any(made[free] <= 0.0) or not np.all(np.isfinite(made)):
                break
        if not np.all(np.isfinite(made)) or np.all(made[free] > 0.0):
            # It did not settle: the gains it started from.
            return start
        lowest = np.flatnonzero(free)[int(np.argmin(made[free]))]
        made = start
        free[lowest], made[lowest] = False, 0.0
    return made
