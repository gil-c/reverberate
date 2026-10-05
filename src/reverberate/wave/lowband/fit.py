"""From the records on the card to a cell's response in the cache form, the fit prepared once.

The filters are those of :class:`reverberate.accel.pairs.PairEncoder`, in
its order and with its kernels: the integration with the 40 Hz low cut, the
zero phase low pass at ``fmax``, the resampling to 4 kHz, the window of the
cache form. They run where the records are, at the grid's rate, and nothing
is kept at any other.

**The fit is an operator, not a solve.** :func:`reverberate.accel.encode.
encode_point` solves the regularised normal equations of every bin for every
cell: a batched factorisation of 3600 matrices of 121 by 121, the same
matrices each time, because every cell's array is the same shells about a
node. Here they are solved once per geometry against the weighted array
matrix, which leaves ``E[bin] = (G'WG + lambda I)^-1 G'W`` cut to the 64
channels kept, 1.8 GB on the card for the grid to 1500 Hz; a cell is then
two transforms and one batched product. The two agree to the conditioning
of the normal matrices times the rounding of float64, and the tests measure
it.

**Many cells a launch.** The filters are one thread a record, each walking
its record alone, so a thousand records leave a card idle: the cells of a
solve are filtered and fitted several at a time, as many as the card's free
memory holds (:meth:`CellEncoder.cells`). The two forward filters are one
pass of their sections in order, which is the same arithmetic. The
resampler's weights depend on the output sample alone, not on the record:
they are computed once (:class:`Resampler`), by resampy's own expression,
and a record's sample is then a sum over them in resampy's order, the same
numbers as :func:`reverberate.accel.dsp.resample` without a table read a tap.
Measured on an RTX 3080 at the grid to 1500 Hz: 0.45 s a pair one at a time
with the table, of which 0.27 s the resampler.

The wavenumber the fit reads the array with is the scheme's own
(:func:`reverberate.wave.lowband.scheme.wavenumber`): on the Cartesian grid
it is the encoder's, to the bit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reverberate.accel import dsp
from reverberate.accel.encode import BandEncoder, geometry_key, prepare_band
from reverberate.accel.lattice import sim_constants
from reverberate.accel.pairs import LOWCUT_HZ, LOWCUT_ORDER, REFERENCE_FMAX_HZ
from reverberate.compute import raw_kernel, to_numpy
from reverberate.spatial.encode import EncoderSettings
from reverberate.spatial.lowband import LOW_RATE_HZ
from reverberate.wave.lowband.scheme import CARTESIAN, Scheme, wavenumber

__all__ = ["FITS", "CellEncoder", "FitOperator", "Resampler", "SpectralChain", "level_scale"]

#: How a cell's records reach the fit: through the filters and the resampler, a sample at a
#: time, or as products in the fit's own spectra (:class:`SpectralChain`).
FITS = ("time", "spectra")


def level_scale(scheme: Scheme, fmax_hz: float, ts: float, ppw: float) -> float:
    """What puts a solve on the scale of a field's 8 kHz solve, whatever its grid.

    A source is one unit of volume velocity over one step, on either grid
    (``write_comms`` scales it by the node's volume), so a response's level
    goes as the time step. On the Cartesian grid at 10.5 points per
    wavelength that is ``fmax / 8000``, :mod:`reverberate.accel.pairs`'s own
    figure, and is given as such; any other grid is given the ratio of the
    steps, of which that figure is the special case. Either way the cache's
    contract holds: on the geometric clock, a unit source in free air reads
    :data:`reverberate.spatial.lowband.FIELD_UNIT_AT_1M` over its distance.
    """
    if not scheme.fcc and ppw == CARTESIAN.ppw:
        return float(fmax_hz / REFERENCE_FMAX_HZ)
    reference = sim_constants(20.0, 50.0, REFERENCE_FMAX_HZ, CARTESIAN.ppw).ts
    return float(reference / ts)


_WEIGHTS_KERNEL = r"""
extern "C" __global__ void lowband_resample_weights(
    const double* __restrict__ interp_win, const double* __restrict__ interp_delta, int nwin,
    int num_table, double scale, int index_step, double time_increment, int n_orig, int n_out,
    int taps, double* __restrict__ left, double* __restrict__ right,
    int* __restrict__ first, int* __restrict__ left_count, int* __restrict__ right_count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_out) return;
    double time_register = (double)t * time_increment;
    int n = (int)time_register;
    double frac = scale * (time_register - (double)n);
    double index_frac = frac * (double)num_table;
    int offset = (int)index_frac;
    double eta = index_frac - (double)offset;
    int i_max = min(n + 1, (nwin - offset) / index_step);
    for (int i = 0; i < i_max; ++i) {
        int tap = offset + i * index_step;
        left[(long long)t * taps + i] = interp_win[tap] + eta * interp_delta[tap];
    }
    frac = scale - frac;
    index_frac = frac * (double)num_table;
    offset = (int)index_frac;
    eta = index_frac - (double)offset;
    int k_max = min(n_orig - n - 1, (nwin - offset) / index_step);
    for (int k = 0; k < k_max; ++k) {
        int tap = offset + k * index_step;
        right[(long long)t * taps + k] = interp_win[tap] + eta * interp_delta[tap];
    }
    first[t] = n; left_count[t] = i_max; right_count[t] = k_max;
}
"""

_RESAMPLE_KERNEL = r"""
extern "C" __global__ void lowband_resample(
    const double* __restrict__ x, int rows, int n_orig, double* __restrict__ y, int n_out,
    int taps, const double* __restrict__ left, const double* __restrict__ right,
    const int* __restrict__ first, const int* __restrict__ left_count,
    const int* __restrict__ right_count)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)rows * n_out) return;
    int row = (int)(idx / n_out);
    int t = (int)(idx % n_out);
    const double* xr = x + (long long)row * n_orig;
    const double* wl = left + (long long)t * taps;
    const double* wr = right + (long long)t * taps;
    int n = first[t];
    double acc = 0.0;
    int i_max = left_count[t];
    for (int i = 0; i < i_max; ++i) acc += wl[i] * xr[n - i];
    int k_max = right_count[t];
    for (int k = 0; k < k_max; ++k) acc += wr[k] * xr[n + k + 1];
    y[idx] = acc;
}
"""


@dataclass
class Resampler:
    """resampy's ``kaiser_best`` between two rates for one record length, its weights kept."""

    n_orig: int
    n_out: int
    taps: int
    arrays: tuple[Any, ...]

    @classmethod
    def prepare(cls, n_orig: int, sample_rate_hz: float, target_hz: float, xp: Any) -> Resampler:
        table = dsp.resampler_table(sample_rate_hz, target_hz)
        n_out = int(n_orig * float(target_hz) / float(sample_rate_hz))
        nwin = int(table["interp_win"].size)
        taps = nwin // int(table["index_step"]) + 1
        left = xp.zeros((n_out, taps), dtype=xp.float64)
        right = xp.zeros((n_out, taps), dtype=xp.float64)
        first = xp.zeros(n_out, dtype=xp.int32)
        left_count = xp.zeros(n_out, dtype=xp.int32)
        right_count = xp.zeros(n_out, dtype=xp.int32)
        threads = 128
        raw_kernel(_WEIGHTS_KERNEL, "lowband_resample_weights")(
            ((n_out + threads - 1) // threads,),
            (threads,),
            (
                xp.asarray(table["interp_win"]),
                xp.asarray(table["interp_delta"]),
                np.int32(nwin),
                np.int32(table["num_table"]),
                np.float64(table["scale"]),
                np.int32(table["index_step"]),
                np.float64(table["time_increment"]),
                np.int32(n_orig),
                np.int32(n_out),
                np.int32(taps),
                left,
                right,
                first,
                left_count,
                right_count,
            ),
        )
        return cls(n_orig, n_out, taps, (left, right, first, left_count, right_count))

    def apply(self, signals: Any, xp: Any) -> Any:
        """:func:`reverberate.accel.dsp.resample` of ``[row, n_orig]``, the same sums."""
        block = xp.ascontiguousarray(xp.asarray(signals, dtype=xp.float64))
        rows, n_orig = block.shape
        if n_orig != self.n_orig:
            raise ValueError(f"prepared for records of {self.n_orig} samples, not {n_orig}")
        out = xp.zeros((rows, self.n_out), dtype=xp.float64)
        total = rows * self.n_out
        threads = 256
        raw_kernel(_RESAMPLE_KERNEL, "lowband_resample")(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                block,
                np.int32(rows),
                np.int32(n_orig),
                out,
                np.int32(self.n_out),
                np.int32(self.taps),
                *self.arrays,
            ),
        )
        return out


@dataclass
class FitOperator:
    """One geometry's fit as a product: ``[chunk][bin, channel, receiver]`` on the device."""

    encoder: BandEncoder
    parts: list[Any]
    bytes_on_device: int

    @classmethod
    def prepare(cls, encoder: BandEncoder, xp: Any) -> FitOperator:
        """Solve the band's normal equations against the weighted matrix, once; free both."""
        keep = encoder.keep
        parts = []
        for weighted, normal in zip(encoder.weighted, encoder.normal, strict=True):
            solved = xp.linalg.solve(normal, xp.transpose(weighted, (0, 2, 1)))
            parts.append(xp.ascontiguousarray(solved[:, :keep, :]))
        encoder.weighted.clear()
        encoder.normal.clear()
        return cls(encoder=encoder, parts=parts, bytes_on_device=int(sum(p.nbytes for p in parts)))

    def apply(self, signals: Any, xp: Any) -> Any:
        """:func:`reverberate.accel.encode.encode_point` by the operator: ``[channel, sample]``."""
        encoder = self.encoder
        spectrum = xp.fft.rfft(xp.asarray(signals, dtype=xp.float64), n=encoder.length, axis=-1)
        fitted_bins = xp.asarray(np.flatnonzero(encoder.fitted))
        return self.apply_bins(spectrum[:, fitted_bins], xp)

    def apply_bins(self, bins: Any, xp: Any) -> Any:
        """The fit of spectra already cut to the fitted bins, ``[receiver, bin]`` complex."""
        encoder = self.encoder
        fitted_bins = xp.asarray(np.flatnonzero(encoder.fitted))
        block = bins.T
        solved_parts = []
        start = 0
        for part in self.parts:
            count = int(part.shape[0])
            rows = block[start : start + count]
            solved = xp.matmul(part, xp.stack([rows.real, rows.imag], axis=2))
            solved_parts.append(solved[:, :, 0] + 1j * solved[:, :, 1])
            start += count
        coefficients = xp.zeros((encoder.length // 2 + 1, encoder.keep), dtype=xp.complex128)
        coefficients[fitted_bins] = xp.concatenate(solved_parts, axis=0)
        coefficients = coefficients / xp.asarray(1j**encoder.degree_keep)[None, :]
        return xp.fft.irfft(coefficients.T, n=encoder.length, axis=-1)[..., : encoder.samples]

    def apply_many(self, signals: Any, xp: Any) -> Any:
        """:meth:`apply` for several cells of this geometry: ``[cell, receiver, sample]`` in."""
        encoder = self.encoder
        cells = int(signals.shape[0])
        spectrum = xp.fft.rfft(xp.asarray(signals, dtype=xp.float64), n=encoder.length, axis=-1)
        bins = int(spectrum.shape[-1])
        fitted_bins = xp.asarray(np.flatnonzero(encoder.fitted))
        # [bin, receiver, cell], then real and imaginary parts as columns.
        block = xp.transpose(spectrum[:, :, fitted_bins], (2, 1, 0))
        del spectrum
        solved_parts = []
        start = 0
        for part in self.parts:
            count = int(part.shape[0])
            rows = block[start : start + count]
            solved = xp.matmul(part, xp.concatenate([rows.real, rows.imag], axis=2))
            solved_parts.append(solved[:, :, :cells] + 1j * solved[:, :, cells:])
            start += count
        coefficients = xp.zeros((bins, encoder.keep, cells), dtype=xp.complex128)
        coefficients[fitted_bins] = xp.concatenate(solved_parts, axis=0)
        coefficients = coefficients / xp.asarray(1j**encoder.degree_keep)[None, :, None]
        return xp.fft.irfft(xp.transpose(coefficients, (2, 1, 0)), n=encoder.length, axis=-1)[
            ..., : encoder.samples
        ]


#: Of the record's length, the least the transform is padded to: the spectrum is then read
#: between its bins by a short kernel (:class:`SpectralChain`).
SPECTRAL_PADDING = 3.0
#: Half the taps of that kernel, and its Kaiser window's shape: an error of a millionth.
SPECTRAL_TAPS = 8
SPECTRAL_BETA = 13.0


@dataclass
class SpectralChain:
    """The filters and the resampling as products in the fit's spectra, for one record length.

    Every step between the records and the fit is linear and does not
    depend on time: the integration with the low cut, the zero phase low
    pass, the resampling to 4 kHz. The fit itself begins with a transform
    and reads only the bins it fits. So the records need never be filtered
    or resampled: their own spectrum, read at the fit's frequencies and
    multiplied by the filters' exact responses there, is what the fit
    wants.

    The fit's frequencies (``k`` times 4 kHz over its length) are not bins
    of a transform at the grid's rate, whose step is in no ratio to them. A
    record is ``N`` samples and nothing else, so its spectrum between the
    bins of a transform padded to ``M`` is given exactly by those bins; with
    ``M`` at least three times ``N`` a Kaiser windowed sinc of sixteen taps
    reads it to a millionth. The transform is the records' own single
    precision; what is read is kept in double.

    **Where it is not the time chain's.** That chain filters ``N`` samples
    and stops: what the filters would still give after the record's last
    sample, in answer to its last milliseconds, and what the backward low
    pass would give before its first, are dropped, and the resampler has a
    passband ripple and a stopband of its own. Here the filters' whole
    response is kept and the band is cut exactly. The two differ by the
    response to a record's two ends, where a room's response is silent at
    the start and 60 dB down at the end; the tests measure it.
    """

    n_orig: int
    padded: int
    first: Any
    weights: Any
    response: Any

    @classmethod
    def prepare(
        cls,
        n_orig: int,
        grid_rate_hz: float,
        encoder: BandEncoder,
        sections: tuple[np.ndarray, np.ndarray],
        xp: Any,
    ) -> SpectralChain:
        """For records of ``n_orig`` samples; ``sections`` are the low cut's and the low pass's."""
        from scipy.signal import sosfreqz

        padded = 1 << int(np.ceil(np.log2(SPECTRAL_PADDING * n_orig)))
        fitted = np.flatnonzero(encoder.fitted)
        frequency = fitted * encoder.sample_rate_hz / encoder.length
        at = frequency * padded / grid_rate_hz
        first = np.floor(at).astype(np.int64) - SPECTRAL_TAPS + 1
        if first.size and first.max() + 2 * SPECTRAL_TAPS > padded // 2 + 1:
            raise ValueError("a fitted bin lies too near the grid's own limit to be read")
        away = (first[:, None] + np.arange(2 * SPECTRAL_TAPS)[None, :]) - at[:, None]
        window = np.i0(SPECTRAL_BETA * np.sqrt(np.clip(1.0 - (away / SPECTRAL_TAPS) ** 2, 0, 1)))
        kernel = np.sinc(away) * window / np.i0(SPECTRAL_BETA)
        # The record lies after time zero, not about it: read about its middle, where the
        # spectrum is as smooth as a record of its length allows.
        weights = kernel * np.exp(1j * np.pi * away * (n_orig - 1) / padded)
        lowcut, lowpass = sections
        omega = 2.0 * np.pi * frequency / grid_rate_hz
        low_pass = sosfreqz(lowpass, worN=omega)[1]
        response = sosfreqz(lowcut, worN=omega)[1] * np.abs(low_pass) ** 2
        response = response * (encoder.sample_rate_hz / grid_rate_hz)
        return cls(
            n_orig=int(n_orig),
            padded=int(padded),
            first=xp.asarray(first),
            weights=xp.asarray(weights),
            response=xp.asarray(response),
        )

    def apply(self, records: Any, xp: Any) -> Any:
        """Records ``[receiver, n_orig]`` as the fit's bins of their response, a row each."""
        if int(records.shape[1]) != self.n_orig:
            raise ValueError(
                f"prepared for records of {self.n_orig} samples, not {records.shape[1]}"
            )
        spectrum = xp.fft.rfft(xp.asarray(records, dtype=xp.float32), n=self.padded, axis=-1)
        read = xp.zeros((spectrum.shape[0], int(self.first.shape[0])), dtype=xp.complex128)
        for tap in range(2 * SPECTRAL_TAPS):
            index = self.first + tap
            held = spectrum[:, abs(index)]
            if bool((index < 0).any()):
                # Under zero a real record's spectrum is the conjugate of the one above it.
                held = xp.where((index < 0)[None, :], xp.conj(held), held)
            read += held * self.weights[None, :, tap]
        return read * self.response[None, :]


@dataclass
class CellEncoder:
    """A cell's node records to its order 7 response at 4 kHz, on one device."""

    scheme: Scheme
    grid_rate_hz: float
    grid_step_m: float
    sound_speed_m_s: float
    fmax_hz: float
    settings: EncoderSettings
    xp: Any
    samples: int
    scale: float
    operators: dict[str, FitOperator] = field(default_factory=dict)
    prepare_s: float = 0.0
    resampler: Resampler | None = None
    #: One of :data:`FITS`. The default is the chain every cached pair was made by.
    fit: str = "time"
    chains: dict[tuple[str, int], SpectralChain] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.fit not in FITS:
            raise ValueError(f"a fit is one of {FITS}, not {self.fit!r}")
        self.lowcut = dsp.lowcut_sos(
            self.grid_rate_hz, LOWCUT_HZ, LOWCUT_ORDER, differentiated=True
        )
        self.lowpass = dsp.lowpass_sos(self.grid_rate_hz, self.fmax_hz)

    def operator_for(self, offsets: np.ndarray) -> FitOperator:
        """The fit of this geometry: prepared once for the campaign."""
        key = geometry_key(offsets)
        with self._lock:
            if key not in self.operators:
                scheme = self.scheme
                xp = self.xp
                keep = (int(self.settings.order) + 1) ** 2
                parts: list[Any] = []

                def reduce(weighted: Any, normal: Any) -> None:
                    # A chunk solved as it is made, as :meth:`FitOperator.prepare` solves
                    # it, and the chunk let go: the same numbers, a chunk held at a time.
                    # What is kept waits on the host: cut on the card from a block the
                    # chunk had just freed, each part held the whole of that block, and
                    # the operator's 1.8 GB held 3.9 GB of a card (measured, 2026-10-05).
                    solved = xp.linalg.solve(normal, xp.transpose(weighted, (0, 2, 1)))
                    parts.append(to_numpy(solved[:, :keep, :]))
                    del solved
                    if xp is not np:
                        xp.get_default_memory_pool().free_all_blocks()

                encoder = prepare_band(
                    offsets,
                    grid_step_m=self.grid_step_m,
                    sound_speed_m_s=self.sound_speed_m_s,
                    settings=self.settings,
                    samples=self.samples,
                    sample_rate_hz=LOW_RATE_HZ,
                    xp=xp,
                    wavenumber_of=(
                        (lambda f: wavenumber(scheme, f, self.grid_step_m, self.sound_speed_m_s))
                        if scheme.fcc
                        else None
                    ),
                    reduce=reduce,
                )
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()
                parts = [xp.ascontiguousarray(xp.asarray(part)) for part in parts]
                self.operators[key] = FitOperator(
                    encoder=encoder, parts=parts, bytes_on_device=int(sum(p.nbytes for p in parts))
                )
                self.prepare_s += encoder.prepare_s
            return self.operators[key]

    def prepare_for(self, steps: int) -> None:
        """The resampler of records of ``steps`` samples, made before any record exists.

        It stays on the card for the campaign. Made inside the first cell's
        filters it was cut from a block those filters had just freed, and
        the pool could then never give that block back.
        """
        if self.xp is np or self.fit == "spectra":
            return
        if self.resampler is None or self.resampler.n_orig != int(steps):
            self.resampler = Resampler.prepare(int(steps), self.grid_rate_hz, LOW_RATE_HZ, self.xp)
        self.xp.get_default_memory_pool().free_all_blocks()

    def signals(self, u_out: Any) -> Any:
        """Node records at the grid's rate as responses at 4 kHz over the window, float64."""
        xp = self.xp
        signals = xp.asarray(u_out, dtype=xp.float64)
        # The low cut then the low pass, one pass: the same sections in the same order.
        signals = dsp.sosfilt(np.vstack([self.lowcut, self.lowpass]), signals, xp)
        signals = dsp.sosfilt(self.lowpass, signals, xp, reverse=True)
        if xp is np:
            signals = dsp.resample(signals, self.grid_rate_hz, LOW_RATE_HZ, xp)
        else:
            length = int(signals.shape[1])
            if self.resampler is None or self.resampler.n_orig != length:
                self.resampler = Resampler.prepare(length, self.grid_rate_hz, LOW_RATE_HZ, xp)
            signals = self.resampler.apply(signals, xp)
        block = xp.zeros((signals.shape[0], self.samples), dtype=xp.float64)
        kept = min(self.samples, int(signals.shape[1]))
        block[:, :kept] = signals[:, :kept]
        return block

    def bins(self, u_out: Any, offsets: np.ndarray) -> Any:
        """Node records as the fit's bins of their response at 4 kHz: :class:`SpectralChain`."""
        operator = self.operator_for(offsets)
        key = (geometry_key(offsets), int(u_out.shape[1]))
        with self._lock:
            if key not in self.chains:
                self.chains[key] = SpectralChain.prepare(
                    key[1],
                    self.grid_rate_hz,
                    operator.encoder,
                    (self.lowcut, self.lowpass),
                    self.xp,
                )
        return self.chains[key].apply(u_out, self.xp)

    def cell(self, u_out: Any, offsets: np.ndarray) -> np.ndarray:
        """One cell's response in the cache form, ``[channel, sample]`` float32 on the host."""
        operator = self.operator_for(offsets)
        if self.fit == "spectra":
            encoded = operator.apply_bins(self.bins(u_out, offsets), self.xp)
        else:
            encoded = operator.apply(self.signals(u_out), self.xp)
        return np.asarray(to_numpy(encoded * self.scale), dtype=np.float32)

    def at_once(self, nodes: int, steps: int) -> int:
        """Cells of ``nodes`` records filtered together: what half the free memory holds."""
        if self.xp is np:
            return 4
        self.xp.get_default_memory_pool().free_all_blocks()
        from reverberate.compute import card_free_bytes

        free = card_free_bytes(self.xp)
        # Two copies of the records in double precision at the widest point, and the
        # fit's spectra after them: three complex arrays of a cell's transform.
        filters = 2.0 * 8.0 * nodes * steps
        fit = 3.0 * 16.0 * nodes * (self.samples + 1)
        return int(max(1, min(32, 0.5 * float(free) // (filters + fit))))

    def cells(self, records: Any, offsets: list[np.ndarray]) -> list[np.ndarray]:
        """Consecutive cells' records, ``[node, step]``, to each cell's response in the cache form.

        The cells' nodes follow each other in ``records`` in the order of
        ``offsets``. Cells of one geometry are filtered and fitted several
        at a time; each response is what :meth:`cell` gives, to the rounding
        of the fit's product.
        """
        xp = self.xp
        out: list[np.ndarray] = []
        row = 0
        if self.fit == "spectra":
            # A cell at a time: a transform and a product, with nothing to share between cells.
            for held in offsets:
                count = int(held.shape[0])
                out.append(self.cell(records[row : row + count], held))
                row += count
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()
            return out
        index = 0
        keys = [geometry_key(o) for o in offsets]
        while index < len(offsets):
            count = int(offsets[index].shape[0])
            group = self.at_once(count, int(records.shape[1]))
            last = index
            while last < len(offsets) and last - index < group and keys[last] == keys[index]:
                last += 1
            cells = last - index
            operator = self.operator_for(offsets[index])
            block = self.signals(records[row : row + cells * count])
            if xp is not np:
                xp.get_default_memory_pool().free_all_blocks()
            for first in range(cells):
                # Fitted a cell at a time: the product is small, and its spectra are not.
                encoded = operator.apply(block[first * count : (first + 1) * count], xp)
                out.append(np.asarray(to_numpy(encoded * self.scale), dtype=np.float32))
            del block
            row += cells * count
            index = last
        return out
