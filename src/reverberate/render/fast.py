"""The signal engine's three parts as a convolution engine renders them, in single precision.

The same mathematics as :mod:`.early`, :mod:`.low` and :mod:`.tail`, which
stay as the reference (``RenderSettings.engine = "reference"``), ordered so
that nothing is computed that no sample reads
(``docs/open-questions/engine-speed.md`` has the cost model and what each
change was measured to give). What changes, part by part:

**Early.** The reference filters the dry signal into every combination of
band, mask and air distance, 96 signals at twice the rate over the whole
run, and a path is a mix of them: the bank is paid whether a step has three
arrivals or fifty. Here a path's row is *one* filter, the product of its
three factors (its band gains on the bank, its mask weights on the two
tracks, its air weights on the two distances either side of it), laid on
the transform of the two steps the row is read over. A row costs a
transform of six thousand points, and the read and the encoding are a loop
in C (:func:`reverberate.render.native.early_interval`).

**Low.** The reference convolves the dry signal with a step's responses on
a transform as long as a response, 1.2 s, to keep 87 ms of it, twenty times
a second. Here each response in use is convolved once over the whole run.
A head off its cell is then reached in the reference's own frames, with the
operator formed in closed form
(:func:`reverberate.spatial.translate.translation_matrices`) and kept: a
head that stands still applies one matrix to every frame of every source.

**Tail.** The reference's, with its transforms and its plane waves in
single precision.

Every part returns float32. The parts differ from the reference by the
rounding of single precision, by the grid a row's air loss is sampled on,
and, under the crossover and above 1 kHz, by the quadrature's own error in
the reference's translation; the tests and the document give the figures.
The parts run on the host only: on a card the engine keeps the reference.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.fft
from scipy.fft import next_fast_len

from reverberate.render import delay, native
from reverberate.render.early import MARGIN, TAPER_FROM, EarlyPart, run_all, workers_of
from reverberate.render.low import FRAMES_PER_STEP, LowPart
from reverberate.render.pack import JUMP_M
from reverberate.render.tail import HELD, PLACES, TailPart, _Step
from reverberate.render.translate import SpatialTranslation
from reverberate.spatial.sh import real_sh, scene_to_ambisonic
from reverberate.spatial.translate import pair_inverse, translation_matrices

__all__ = ["FastEarly", "FastLow", "FastTail", "HeadOperators"]

#: Names a folder the processes of one render keep their sources' carriers in, as files
#: they all map (:mod:`reverberate.render.mix` sets it): 95 MB a source once, not a process.
CARRIERS_VARIABLE = "REVERBERATE_CARRIERS"
#: Samples kept either side of what a row's reads were bounded to.
ROW_GUARD = 8
#: A row's transform is a fast length over a multiple of this: few lengths, few tables.
ROW_LADDER = 256
#: Responses of the low band kept as spectra, per source: 1.8 MB each.
PAIRS_HELD = 12
#: The tail convolves a histogram over whole multiples of this many steps of a run.
TAIL_LADDER_STEPS = 10


def _mixed_rows(weights: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``weights @ table``, a row of the table at a time: ``[row, k]`` by ``[k, f]``.

    The same sum in a fixed order whatever the count of rows. The library's
    product takes a single row through another kernel, whose rounding moves
    with where its arrays lie in memory: two processes would not give one
    output.
    """
    made = weights[:, 0, None] * table[0][None, :]
    for k in range(1, table.shape[0]):
        made += weights[:, k, None] * table[k][None, :]
    return np.asarray(made)


def _products(operators: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``operators @ columns`` for ``[f, a, b]`` by ``[f, b, m]``, never by a single column.

    One column is given a second, of zeros, and the product's first is
    kept: a matrix by a vector is the library's level 2, and its rounding
    is not the level 3's (see :func:`_mixed_rows`).
    """
    if columns.shape[-1] != 1:
        return np.asarray(np.matmul(operators, columns))
    doubled = np.concatenate([columns, np.zeros_like(columns)], axis=-1)
    return np.asarray(np.matmul(operators, doubled)[..., :1])


class FastEarly(EarlyPart):
    """One source's arrivals: a filter a row, a loop in C a path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.xp is not np:
            raise ValueError("the fast early part runs on the host")
        self._mask = np.ascontiguousarray(self.row_mask, dtype=np.float32)
        self._gain = np.ascontiguousarray(self.row_gain, dtype=np.float32)
        self._air = np.ascontiguousarray(self.row_air, dtype=np.float32)
        # The reference's own copies, in double precision, are not read again.
        self.row_mask = self.row_gain = self.row_air = np.zeros((0, 0))
        self._tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _table(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """The bank and the air's loss on a transform of ``n``: ``[band, f]``, ``[node, f]``."""
        if n not in self._tables:
            freqs = np.fft.rfftfreq(n, 1.0 / self.rate)
            advance = np.exp(2j * np.pi * freqs * self.centre / self.rate)
            bank = np.fft.rfft(self.kernels, n, axis=1) * advance[None, :]
            loss = np.exp(-self._attenuation(freqs)[None, :] * self.air_m[:, None])
            fade = np.clip(
                (freqs - TAPER_FROM * 0.5 * self.rate) / ((1 - TAPER_FROM) * 0.5 * self.rate), 0, 1
            )
            loss = loss * np.cos(0.5 * np.pi * fade)[None, :] ** 2
            self._tables[n] = (bank.astype(np.complex64), loss.astype(np.float32))
        return self._tables[n]

    def _rows(self, ids: np.ndarray, a: slice, b: slice) -> tuple[np.ndarray, ...]:
        """:meth:`EarlyPart._interval` by row: the apparent sources, and each end's row or -1."""
        ids_a, ids_b = ids[a], ids[b]
        _, in_a, in_b = np.intersect1d(ids_a, ids_b, assume_unique=True, return_indices=True)
        row_a, row_b = a.start + in_a, b.start + in_b
        moved = np.linalg.norm(self.apparent[row_b] - self.apparent[row_a], axis=1)
        same = moved <= JUMP_M
        kept_a, kept_b = row_a[same], row_b[same]
        only_a = np.setdiff1d(np.arange(a.start, a.stop), kept_a, assume_unique=True)
        only_b = np.setdiff1d(np.arange(b.start, b.stop), kept_b, assume_unique=True)
        q0 = np.concatenate([self.apparent[kept_a], self.apparent[only_a], self.apparent[only_b]])
        q1 = np.concatenate([self.apparent[kept_b], self.apparent[only_a], self.apparent[only_b]])
        none_a, none_b = np.full(only_b.size, -1), np.full(only_a.size, -1)
        return (
            q0,
            q1,
            np.concatenate([kept_a, only_a, none_a]).astype(np.int64),
            np.concatenate([kept_b, none_b, only_b]).astype(np.int64),
        )

    def _filtered(self, spans: dict[int, tuple[float, float]]) -> dict[int, tuple[np.ndarray, int]]:
        """Each row's signal at twice the rate over the dry samples it is read at.

        ``spans`` gives a row the first and last dry sample its reads may
        fall on. Returns the row's ``(signal, origin)``: float32, its sample
        0 the dry sample ``origin``. A row whose dry signal is silent there
        is left out.
        """
        made: dict[int, tuple[np.ndarray, int]] = {}
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for row, (lo, hi) in spans.items():
            first = int(np.floor(lo)) - ROW_GUARD
            length = int(np.ceil(hi)) + ROW_GUARD - first
            wanted = -(-(length + 2 * MARGIN) // ROW_LADDER) * ROW_LADDER
            n = next_fast_len(wanted, real=True)
            n += n % 2
            groups.setdefault(n, []).append((row, first, length))
        kw = workers_of(np, self.workers)
        for n, members in groups.items():
            starts = np.array([first - MARGIN for _, first, _ in members])
            rows = np.array([row for row, _, _ in members])
            low, high = int(starts.min()), int(starts.max()) + n
            at = (starts - low)[:, None] + np.arange(n)[None, :]
            spectrum: Any = None
            for m, track in enumerate(self.tracks):
                weight = self._mask[rows, m]
                if not weight.any() or track.silent(low, high):
                    continue
                dry = track.read(low, high).astype(np.float32)
                part = scipy.fft.rfft(dry[at], axis=-1, **kw) * weight[:, None]
                spectrum = part if spectrum is None else spectrum + part
            if spectrum is None:
                continue
            bank, loss = self._table(n)
            spectrum *= _mixed_rows(self._gain[rows], bank)
            spectrum *= _mixed_rows(self._air[rows], loss)
            # Twice the rate: the same spectrum in a transform twice as long, its own
            # Nyquist bin shared between the two it becomes.
            padded = np.zeros((rows.size, n + 1), dtype=np.complex64)
            padded[:, : n // 2 + 1] = spectrum
            padded[:, n // 2] *= 0.5
            doubled = scipy.fft.irfft(padded, 2 * n, axis=-1, **kw)
            doubled *= 2.0
            for index, (row, first, length) in enumerate(members):
                made[row] = (doubled[index, 2 * MARGIN : 2 * (MARGIN + length)], first)
        return made

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1``: ``[channel, (k1 - k0) step]`` float32."""
        step, rate = self.step, self.rate
        channels = self.pack.header.channels
        out = np.zeros((channels, (k1 - k0) * step), dtype=np.float32)
        offsets = np.asarray(self.source.early.offsets)
        lo_row, hi_row = int(offsets[k0]), int(offsets[k1 + 1])
        if hi_row == lo_row:
            return out
        reach = float(self.source.early.delay_s[lo_row:hi_row].max()) + self.lead
        first = k0 * step - int(np.ceil(reach * rate)) - step - delay.TAPS
        last = k1 * step + delay.TAPS
        if all(track.silent(first - MARGIN, last + MARGIN) for track in self.tracks):
            return out
        ids = np.asarray(self.source.early.path_id)
        plans = []
        spans: dict[int, tuple[float, float]] = {}
        for k in range(k0, k1):
            a = slice(int(offsets[k]), int(offsets[k + 1]))
            b = slice(int(offsets[k + 1]), int(offsets[k + 2]))
            if a.stop == a.start and b.stop == b.start:
                continue
            q0, q1, row_a, row_b = self._rows(ids, a, b)
            l0, l1 = self.listener[k], self.listener[k + 1]
            seen0, seen1 = q0 - l0[None, :], q1 - l1[None, :]
            d0, d1 = np.linalg.norm(seen0, axis=1), np.linalg.norm(seen1, axis=1)
            # A path's distance over the interval is between its two ends', less at most
            # what it moved: the dry samples its reads fall between.
            moved = np.linalg.norm(seen1 - seen0, axis=1)
            longest = np.maximum(d0, d1) / self.sound_speed + self.lead
            shortest = np.maximum(np.minimum(d0, d1) - moved, 0.0) / self.sound_speed + self.lead
            lo = k * step - longest * rate
            hi = (k + 1) * step - shortest * rate
            for rows in (row_a, row_b):
                for p in np.flatnonzero(rows >= 0):
                    row = int(rows[p])
                    held = spans.get(row)
                    spans[row] = (
                        (float(lo[p]), float(hi[p]))
                        if held is None
                        else (min(held[0], float(lo[p])), max(held[1], float(hi[p])))
                    )
            plans.append((k, q0, q1, l0, l1, row_a, row_b))
        signals = self._filtered(spans)
        node_u = np.arange(self.nodes + 1) / self.nodes
        # The harmonics of every path of the run at the nodes of its interval, at once.
        seen = [
            q0[:, None, :] * (1.0 - node_u)[None, :, None]
            + q1[:, None, :] * node_u[None, :, None]
            - (l0[None, :] * (1.0 - node_u)[:, None] + l1[None, :] * node_u[:, None])[None]
            for _, q0, q1, l0, l1, _, _ in plans
        ]
        if not seen:
            return out
        basis = real_sh(
            self.pack.header.order, scene_to_ambisonic(np.concatenate(seen).reshape(-1, 3))
        ).astype(np.float32)
        basis = basis.reshape(-1, self.nodes + 1, channels)
        starts = np.concatenate([[0], np.cumsum([len(at) for at in seen])])

        def one(index: int) -> tuple[int, np.ndarray] | None:
            k, q0, q1, l0, l1, row_a, row_b = plans[index]
            ends_a = [signals.get(int(row)) for row in row_a]
            ends_b = [signals.get(int(row)) for row in row_b]
            both = zip(ends_a, ends_b, strict=True)
            heard = [p for p, (a, b) in enumerate(both) if a is not None or b is not None]
            if not heard:
                return None
            q0, q1 = q0[heard], q1[heard]
            harmonics = basis[starts[index] : starts[index + 1]][heard]
            block = np.zeros((channels, step), dtype=np.float32)
            native.early_interval(
                block,
                start=k * step,
                nodes=self.nodes,
                rate=rate,
                lead=self.lead,
                speed=self.sound_speed,
                q0=q0,
                q1=q1,
                l0=l0,
                l1=l1,
                first=[ends_a[p] for p in heard],
                second=[ends_b[p] for p in heard],
                harmonics=harmonics,
            )
            return k, block

        for made in run_all(one, range(len(plans)), np, self.workers):
            if made is not None:
                k, block = made
                out[:, (k - k0) * step : (k - k0 + 1) * step] = block
        return out


class HeadOperators:
    """What moves a head off its cell, formed once and kept: one for all the sources.

    The operators depend on where the head is seen from its cells and on
    nothing of a source. ``matrices`` is the translation from one cell in
    closed form; two cells are the reference's fusion,
    ``[T(d_0) T(d_1)] (A W A^H + lambda)^-1``, whose inverse is the
    library's own, per vector between the cells.
    """

    def __init__(self, base: SpatialTranslation, *, held: int = 96, pairs: int = 8) -> None:
        self.base = base
        self._held = held
        self._kept: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        #: The inverses, apart: a walk forms a translation a frame and meets a pair of
        #: cells a few times a second. In double precision, 20 MB each: an inverse is
        #: five hundred times its matrix, and single precision would leave 3e-5 of the
        #: field's peak in what it is applied to.
        self._pairs = pairs
        self._inverses: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

    def _get(self, key: tuple[Any, ...], make: Any) -> np.ndarray:
        if key in self._kept:
            self._kept.move_to_end(key)
            return self._kept[key]
        made: np.ndarray = make()
        self._kept[key] = made
        while len(self._kept) > self._held:
            self._kept.popitem(last=False)
        return made

    @staticmethod
    def _name(offsets: np.ndarray, freqs_hz: np.ndarray) -> tuple[Any, ...]:
        return (offsets.tobytes(), int(freqs_hz.size), float(freqs_hz[-1]))

    def prepare(self, offsets: np.ndarray, freqs_hz: np.ndarray) -> None:
        """Form the translations of ``offsets``, ``[offset, 3]``, that are not kept: at once."""
        names = [("t", *self._name(np.ascontiguousarray(offset), freqs_hz)) for offset in offsets]
        missing = [i for i, name in enumerate(names) if name not in self._kept]
        if not missing:
            return
        # A few at a time: an operator is 2.4 MB and a walk asks for eighty a run.
        for first in range(0, len(missing), 16):
            some = missing[first : first + 16]
            made = translation_matrices(
                offsets[some], freqs_hz, self.base.order, sound_speed_m_s=self.base.sound_speed_m_s
            )
            for at, index in enumerate(some):
                self._get(names[index], lambda at=at, made=made: made[at].copy())

    def matrices(self, offset: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """``[f, channel, channel]`` complex64: the expansion about ``offset`` from the cell's."""
        return self._get(
            ("t", *self._name(offset, freqs_hz)),
            lambda: translation_matrices(
                offset, freqs_hz, self.base.order, sound_speed_m_s=self.base.sound_speed_m_s
            ),
        )

    def inverse(self, between: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """The two cells' Gram matrix inverted, ``[f, 2 channel, 2 channel]`` complex128."""
        key = self._name(np.round(between, 6) + 0.0, freqs_hz)
        if key in self._inverses:
            self._inverses.move_to_end(key)
            return self._inverses[key]
        made = pair_inverse(
            between,
            freqs_hz,
            self.base.order,
            regularisation=self.base.regularisation,
            sound_speed_m_s=self.base.sound_speed_m_s,
            quadrature_degree=self.base.quadrature_degree,
        )
        self._inverses[key] = made
        while len(self._inverses) > self._pairs:
            self._inverses.popitem(last=False)
        return made

    def fused(self, offsets: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """``[f, channel, 2 channel]`` complex64: the head from two cells' stacked fields."""

        def make() -> np.ndarray:
            both = np.concatenate(
                [self.matrices(offsets[0], freqs_hz), self.matrices(offsets[1], freqs_hz)], axis=2
            )
            product = np.matmul(both, self.inverse(offsets[0] - offsets[1], freqs_hz))
            return np.asarray(product.astype(np.complex64))

        return self._get(("g", *self._name(offsets, freqs_hz)), make)

    def apply(
        self, fields: np.ndarray, offsets: np.ndarray, freqs_hz: np.ndarray, still: bool
    ) -> np.ndarray:
        """The head from the cells: ``[frame, cell, channel, f]`` to ``[frame, channel, f]``.

        Every frame reads the head through the same ``offsets``, ``[cell, 3]``.
        ``still`` says the head stands there: the fusion's operator is then
        multiplied out and kept, one matrix for every frame of every source.
        A head that walks has another operator each frame, and its fields
        are solved against the cells' inverse and then moved, in two
        products, which leaves 3e-5 of the field's peak where the other
        leaves 1e-7: the inverse is five hundred times its matrix.
        """
        frames, cells, channels, count = fields.shape
        if cells == 1:
            operator = self.matrices(offsets[0], freqs_hz)
            columns = fields[:, 0].transpose(2, 1, 0)  # [f, channel, frame]
        elif still:
            operator = self.fused(offsets, freqs_hz)
            columns = fields.reshape(frames, cells * channels, count).transpose(2, 1, 0)
        else:
            stacked = fields.reshape(frames, cells * channels, count).transpose(2, 1, 0)
            solved = _products(
                self.inverse(offsets[0] - offsets[1], freqs_hz), stacked.astype(np.complex128)
            )
            moved = _products(self.matrices(offsets[0], freqs_hz), solved[:, :channels])
            moved += _products(self.matrices(offsets[1], freqs_hz), solved[:, channels:])
            return np.asarray(moved.transpose(2, 1, 0).astype(np.complex64))
        return np.asarray(_products(operator, np.ascontiguousarray(columns)).transpose(2, 1, 0))


class FastLow(LowPart):
    """One source's low band: a convolution a response in use, an operator a place of the head."""

    def __init__(self, *args: Any, operators: HeadOperators | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.xp is not np:
            raise ValueError("the fast low band runs on the host")
        self.operators = operators
        #: Whether the part renders the low band itself, and so gives it before it is raised.
        self.joins = not self.slots and operators is not None
        self._poly = np.ascontiguousarray(np.asarray(self.poly), dtype=np.float32)
        self._hann = np.asarray(self.hann, dtype=np.float32)
        self._root = np.asarray(self.root, dtype=np.float32)
        self._fast_spectra: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._samples = int(self.pack.header.low_samples)

    def _response(self, row: int, n: int) -> np.ndarray:
        key = (row, n)
        if key in self._fast_spectra:
            self._fast_spectra.move_to_end(key)
            return self._fast_spectra[key]
        response = np.asarray(self.low.ir[row], dtype=np.float32) * np.float32(self.factor)
        made: np.ndarray = scipy.fft.rfft(response, n, axis=-1, **workers_of(np, self.workers))
        self._fast_spectra[key] = made
        while len(self._fast_spectra) > PAIRS_HELD:
            self._fast_spectra.popitem(last=False)
        return made

    def _step_of(self, m: int) -> int:
        return min(max((2 * m + FRAMES_PER_STEP) // (2 * FRAMES_PER_STEP), 0), self.steps - 1)

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1`` at the output rate: ``[channel, samples]`` float32."""
        if not self.joins:
            # More than two source positions a step, or another estimator: the reference's.
            return np.asarray(super().render(k0, k1), dtype=np.float32)
        low = self.low_band(k0, k1)
        if low is None:
            channels = self.pack.header.channels
            return np.zeros((channels, (k1 - k0) * self.step * self.factor), dtype=np.float32)
        return self.raised(low, k1 - k0)

    def raised(self, low: np.ndarray, steps: int) -> np.ndarray:
        """What :meth:`low_band` gave, or the sum of several sources', at the output rate."""
        return native.upsample(low, self._poly, steps * self.step)

    def low_band(self, k0: int, k1: int) -> np.ndarray | None:
        """The intervals ``k0`` to ``k1 - 1`` at the low rate, or ``None`` when nothing sounds.

        ``[channel, samples]`` float32 from ``reach`` samples before the
        run to as many after it: what the filter between the rates reads.
        The sources of a mix are summed here, at 4 kHz, and raised once.
        """
        channels = self.pack.header.channels
        silence = None
        first = k0 * self.step - self.reach
        last = k1 * self.step + self.reach
        half = self.frame // 2
        m0 = max(-((half - 1 - first) // self.hop), -(FRAMES_PER_STEP // 2) + 1)
        m1 = min((last + half - 1) // self.hop, (self.steps - 1) * FRAMES_PER_STEP + 1)
        # The low samples the frames' steps are convolved over: [begin, end).
        begin = self._step_of(m0) * self.step - self.frame
        end = self._step_of(m1) * self.step - self.frame + self.cover
        n = next_fast_len(end - begin + self._samples, real=True)
        if self.track.silent(end - n, end):
            return silence
        kw = workers_of(np, self.workers)
        dry = scipy.fft.rfft(self.track.read(end - n, end).astype(np.float32))
        heard: dict[int, np.ndarray] = {}

        def through(row: int) -> np.ndarray:
            if row not in heard:
                heard[row] = scipy.fft.irfft(self._response(row, n) * dry[None, :], n, **kw)[
                    :, n - (end - begin) :
                ]
            return heard[row]

        fields: dict[int, np.ndarray | None] = {}

        def field(k: int) -> np.ndarray | None:
            """The cells' fields round step ``k``: ``[cell, channel, cover]``."""
            if k not in fields:
                mode = int(self.low.mode[k])
                made = None
                if mode != 0:
                    weight = np.float32(self.low.position_weight[k])
                    start = k * self.step - self.frame - begin
                    cells: list[np.ndarray] = []
                    for b in range(2 if mode == 3 else 1):
                        cell: np.ndarray = (np.float32(1.0) - weight) * through(
                            int(self.low.pair[k, 0, b])
                        )[:, start : start + self.cover]
                        if weight > 0.0:
                            cell = (
                                cell
                                + weight
                                * through(int(self.low.pair[k, 1, b]))[
                                    :, start : start + self.cover
                                ]
                            )
                        cells.append(cell)
                    made = np.stack(cells)
                fields[k] = made
            return fields[k]

        low = np.zeros((channels, last - first), dtype=np.float32)

        def lay(m: int, frame: np.ndarray) -> None:
            start = m * self.hop - half
            lo, hi = max(start, first), min(start + self.frame, last)
            low[:, lo - first : hi - first] += frame[:, lo - start : hi - start]

        # Frames that read the head through the same cells from the same place go together.
        groups: dict[tuple[Any, ...], list[tuple[int, np.ndarray]]] = {}
        places: dict[tuple[Any, ...], np.ndarray] = {}
        any_heard = False
        for m in range(m0, m1 + 1):
            k = self._step_of(m)
            made = field(k)
            if made is None:
                continue
            any_heard = True
            start = m * self.hop - half - (k * self.step - self.frame)
            piece = made[:, :, start : start + self.frame]
            if int(self.low.mode[k]) == 1:
                lay(m, piece[0] * self._hann[None, :])
                continue
            at = min(max(m * self.hop / self.step, 0.0), self.steps - 1.0)
            lower = min(int(at), self.steps - 2)
            head = self.listener[lower] * (1.0 - (at - lower)) + self.listener[lower + 1] * (
                at - lower
            )
            cells = self.low.cell[k, : piece.shape[0]]
            offsets = np.ascontiguousarray(head[None, :] - self.cells[cells])
            still = bool(np.array_equal(self.listener[lower], self.listener[lower + 1]))
            key = (piece.shape[0], offsets.tobytes(), still)
            groups.setdefault(key, []).append((m, piece))
            places[key] = offsets
        if not any_heard:
            return silence
        operators = self.operators
        assert operators is not None
        if places:
            operators.prepare(np.concatenate(list(places.values())), self.freqs)
        for key, members in groups.items():
            pieces = np.stack([piece for _, piece in members]) * self._root
            spectra = scipy.fft.rfft(pieces, axis=-1)[..., : self.kept]
            moved = operators.apply(spectra, places[key], self.freqs, key[2])
            full = np.zeros((len(members), channels, self.frame // 2 + 1), dtype=np.complex64)
            full[:, :, : self.kept] = moved
            frames = scipy.fft.irfft(full, self.frame, axis=-1) * self._root
            for index, (m, _) in enumerate(members):
                lay(m, frames[index])
        return low


class FastTail(TailPart):
    """One source's tail: the reference's, its transforms in single precision."""

    def __init__(self, *args: Any, tail_steps: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.xp is not np:
            raise ValueError("the fast tail runs on the host")
        self._encode = np.ascontiguousarray(np.asarray(self.encode), dtype=np.float32)
        self.block = self.directions
        self.tail_steps = tail_steps
        #: A response's output is convolved over whole multiples of this many samples.
        self._ladder = TAIL_LADDER_STEPS * self.step
        self._long: tuple[int, np.ndarray] | None = None

    def _carrier(self) -> Any:
        """:meth:`TailPart._carrier`, drawn in C and filtered in single precision."""

        def make() -> Any:
            kw = workers_of(np, self.workers)
            bins = self.bins + (PLACES - 1) * self.apart
            count = bins * self.bin_samples + self.taps - 1
            n = next_fast_len(count + self.taps - 1, real=True)
            made = np.zeros((self.bands, self.directions, bins, self.bin_samples), dtype=np.float32)
            for band in range(self.bands):
                streams = band * self.directions + np.arange(self.directions)
                white = native.carrier(self.source.tail_seed, streams, 0, count)
                kernel = (self.kernels[band] / np.sqrt(self.kept[band])).astype(np.float32)
                through = scipy.fft.irfft(
                    scipy.fft.rfft(white, n, axis=-1, **kw) * scipy.fft.rfft(kernel, n)[None, :],
                    n,
                    axis=-1,
                    **kw,
                )[:, self.taps - 1 : count]
                made[band] = through.reshape(self.directions, bins, self.bin_samples)
            return made

        def shared() -> Any:
            """The carrier as a file every process of a render maps: made by the first."""
            folder = Path(os.environ[CARRIERS_VARIABLE])
            bins = self.bins + (PLACES - 1) * self.apart
            shape = (self.bands, self.directions, bins, self.bin_samples)
            name = hashlib.sha256(
                repr((int(self.source.tail_seed), shape, self.taps, self.rate)).encode()
            ).hexdigest()[:24]
            path = folder / f"{name}.f32"
            if not path.is_file():
                made = make()
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
                    handle.write(made.tobytes())
                # Whole or not at all: another process may be drawing the same noise.
                os.replace(handle.name, path)
            return np.memmap(path, dtype=np.float32, mode="r", shape=shape)

        named = os.environ.get(CARRIERS_VARIABLE)
        wanted = shared if named and Path(named).is_dir() else make
        return HELD.get((self._token, "carrier"), wanted)

    def _spectrum(self, row: int, n: int) -> Any:
        def make() -> Any:
            padded = np.zeros((self.directions, n), dtype=np.float32)
            padded[:, : self.span] = np.einsum(
                "bdj,bdjs->djs", self._reference(row), self._noise(row)
            ).reshape(self.directions, self.span)
            return scipy.fft.rfft(padded, axis=-1, **workers_of(np, self.workers))

        return HELD.get((self._token, "spectrum32", row, n), make)

    def _mixed(self, key: _Step, n: int) -> Any:
        def make() -> Any:
            gain = self._gain(key)
            total = None
            for row, weight in zip(key.rows, key.weights, strict=True):
                part = self._spectrum(row, n) * np.float32(gain * float(np.sqrt(weight)))
                total = part if total is None else total + part
            return total

        if key.weights == (1.0,):
            return self._spectrum(key.rows[0], n)
        return HELD.get((self._token, "mixed32", key.rows, key.weights, n), make)

    def render(self, k0: int, k1: int) -> Any:
        """The intervals ``k0`` to ``k1 - 1``: ``[channel, (k1 - k0) step]`` float32.

        With ``tail_steps`` over the run's length the tail is rendered that
        many steps at once, from a multiple of it, and the run is cut from
        what is kept: a transform as long as a response is then paid once
        in two seconds and not once in half a second.
        """
        if self.tail_steps <= k1 - k0:
            return self._run(k0, k1)
        index = k0 // self.tail_steps
        first = index * self.tail_steps
        if self._long is None or self._long[0] != index:
            self._long = (index, self._run(first, min(first + self.tail_steps, self.steps - 1)))
        return self._long[1][:, (k0 - first) * self.step : (k1 - first) * self.step]

    def _run(self, k0: int, k1: int) -> np.ndarray:
        waves = self.waves(k0, k1)
        if waves is None:
            channels = self.pack.header.channels
            return np.zeros((channels, (k1 - k0) * self.step), dtype=np.float32)
        return np.asarray(self.encoded(waves))

    def encoded(self, waves: np.ndarray) -> np.ndarray:
        """Plane waves, ``[direction, sample]``, on the listener's harmonics."""
        return np.asarray(self._encode @ waves)

    def waves(
        self, k0: int, k1: int, pool: dict[tuple[int, int], np.ndarray] | None = None
    ) -> np.ndarray | None:
        """The intervals ``k0`` to ``k1 - 1`` as plane waves, ``None`` when nothing sounds.

        ``[direction, (k1 - k0) step]`` float32. With ``pool``, what a
        response gives at one level the whole run is not transformed here:
        its spectrum is added to ``pool`` under ``(stop, size)``, the dry
        window's end and the transform's length, and whoever holds the pool
        transforms the sum of every source's once
        (:meth:`reverberate.render.engine.Engine.render`). The rest, what
        fades between two responses, is returned.
        """
        step = self.step
        length = (k1 - k0) * step
        silence = None
        end = k1 * step - self.lead
        n = next_fast_len(length + self.span, real=True)
        if self.track.silent(end - n, end):
            return silence
        u = (np.arange(step) / step).astype(np.float32)
        shares: dict[_Step, np.ndarray] = {}
        for k in range(k0, k1 + 1):
            key = self._key(k)
            if key is None:
                continue
            share = shares.setdefault(key, np.zeros(length, dtype=np.float32))
            if k > k0:
                share[(k - 1 - k0) * step : (k - k0) * step] += u * np.float32(self.gain[k])
            if k < k1:
                share[(k - k0) * step : (k - k0 + 1) * step] += (1.0 - u) * np.float32(self.gain[k])
        if not shares:
            return silence
        lates: dict[tuple[int, int], np.ndarray] = {}
        own: dict[int, np.ndarray] = {}
        nothing = np.zeros(length, dtype=np.float32)
        for key, share in shares.items():
            gain = self._gain(key)
            for row, weight in zip(key.rows, key.weights, strict=True):
                root = np.float32(gain * float(np.sqrt(weight)))
                if len(shares) > 1:
                    own[row] = own.get(row, nothing) + root * share
                if key.start > self.earliest[row]:
                    late = (row, key.start)
                    lates[late] = lates.get(late, nothing) + root * share
        waves = np.zeros((self.directions, length), dtype=np.float32)
        kw = workers_of(np, self.workers)
        dries: dict[tuple[int, int], Any] = {}
        sounded = [pool is None]

        def lay(share: np.ndarray, reach: int, back: int, spectrum: Any) -> None:
            """Add a response's output where its share is not zero, and nowhere else.

            A response ``reach`` samples long whose first sample is ``back``
            after the arrival; ``spectrum`` gives its transform at a length.
            What a run convolves is then what its histograms are heard over:
            one that a walk reads for half a second does not cost the run.
            """
            sounding = np.flatnonzero(share)
            if not sounding.size:
                return
            lo = (int(sounding[0]) // self._ladder) * self._ladder
            hi = min(-(-(int(sounding[-1]) + 1) // self._ladder) * self._ladder, length)
            size = next_fast_len(hi - lo + reach, real=True)
            stop = end - (length - hi) - back
            if self.track.silent(stop - size, stop):
                return
            if (stop, size) not in dries:
                dries[(stop, size)] = scipy.fft.rfft(
                    self.track.read(stop - size, stop).astype(np.float32)
                )
            # One level the whole run, to the rounding of the two ramps that make it.
            level = share[lo]
            steady = float(np.ptp(share)) <= 1e-6 * abs(float(level))
            if pool is not None and lo == 0 and hi == length and steady:
                product = spectrum(size) * (dries[(stop, size)] * level)[None, :]
                if (stop, size) in pool:
                    pool[(stop, size)] += product
                else:
                    pool[(stop, size)] = product
                return
            sounded[0] = True
            self._add(waves[:, lo:hi], spectrum(size), dries[(stop, size)], size, share[lo:hi])

        if len(shares) == 1:
            # One response the whole run: its histograms are mixed once, as spectra.
            (key, share), *_ = shares.items()
            lay(share, self.span, 0, lambda size: self._mixed(key, size))
        else:
            for row in sorted(own):
                lay(own[row], self.span, 0, lambda size, row=row: self._spectrum(row, size))
        for row, start in sorted(lates):
            first, response = self._late(row, start)
            short = response.astype(np.float32)
            lay(
                -lates[(row, start)],
                int(short.shape[1]),
                first,
                lambda size, short=short: scipy.fft.rfft(short, size, axis=-1, **kw),
            )
        return waves if sounded[0] else None
