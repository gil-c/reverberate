"""From paths and a ray histogram to one ambisonic response.

The early part: every path a band limited pulse at its arrival time, from
its arrival direction, at its per band gain, laid through the project's own
octave bank so a band here is the band the reference is analysed in. The
tail: noise bursts per histogram bin, drawn from the directions the bin's
moments say, at the energy the bin holds, scaled on the direct sound. Then
air absorption, the reference chain's low cut, and the source's signature.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import butter, sosfreqz

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import to_numpy
from reverberate.metrics import band_centres, octave_filter_rows
from reverberate.mirror.direct import apply_signature
from reverberate.mirror.ism import Paths
from reverberate.mirror.rays import Histogram
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count, real_sh, scene_to_ambisonic

__all__ = [
    "RenderSettings",
    "TailNoise",
    "band_pulse_energy",
    "bank_reading",
    "early_signals",
    "render_point",
    "tail_from_histogram",
]

#: The reference chain's own low cut (``w40_volume_field.plan``): 40 Hz, order 8.
LOWCUT_HZ = 40.0
LOWCUT_ORDER = 8
#: Half length of the windowed sinc that places a pulse between samples.
DELAY_HALF_TAPS = 16


@dataclass(frozen=True)
class RenderSettings:
    """What a render chooses: the reference's format, and where and how the tail is drawn."""

    sample_rate_hz: float = 48000.0
    order: int = 7
    duration_s: float = 1.2
    #: Where the rays' tail takes over from the pulses, after the direct sound.
    tail_from_s: float = 0.010
    #: Noise bursts per histogram bin, each from its own sampled direction.
    tail_bursts: int = 24

    def record(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=8)
def _band_picks(rate: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    centres = band_centres(int(round(rate)))
    material = np.asarray(OCTAVE_BANDS, dtype=float)
    return centres, tuple(int(np.argmin(np.abs(material - c))) for c in centres)


def _band_map(rate: float) -> tuple[tuple[int, ...], np.ndarray]:
    """The bank's bands at this rate, and which material band each reads its gain from."""
    centres, picks = _band_picks(float(rate))
    return centres, np.array(picks, dtype=int)


def _fractional_pulses(
    delays_samples: np.ndarray, amplitudes: np.ndarray, half: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed sinc kernels placing each pulse between samples: taps and their sample indices.

    Both ``[pulse, 2 half + 1]``; the caller accumulates the taps at the indices.
    """
    offsets = np.arange(-half, half + 1)
    base = np.floor(delays_samples).astype(int)
    fraction = delays_samples - base
    taps = offsets[None, :] - fraction[:, None]
    kernel = np.sinc(taps) * np.hanning(2 * half + 3)[1:-1][None, :]
    return np.asarray(kernel * amplitudes[:, None]), np.asarray(base[:, None] + offsets[None, :])


@lru_cache(maxsize=4)
def _bank_kernels(rate: float) -> np.ndarray:
    """The bank's FIR filters, ``[band, taps]``, as :func:`octave_filter_rows` reads them."""
    from reverberate.metrics import octave_bank

    kernels = np.ascontiguousarray(np.asarray(octave_bank(int(round(rate))).filters, float).T)
    kernels.setflags(write=False)
    return kernels


@lru_cache(maxsize=8)
def _bank_spectra(rate: float, bands: tuple[int, ...], transform: int) -> np.ndarray:
    """The transforms of the bank's filters ``bands`` at a length: what a convolution reads."""
    from scipy import fft as sp_fft

    kernels = np.asarray(_bank_kernels(rate))[np.asarray(bands, dtype=int)]
    spectra = np.asarray(sp_fft.rfftn(kernels, [transform], axes=[1]))
    spectra.setflags(write=False)
    return spectra


def _band_rows_host(rows: np.ndarray, rate: float, bands: np.ndarray) -> np.ndarray:
    """:func:`octave_filter_rows` with the filters' transforms read once, to the bit.

    ``scipy.signal.fftconvolve`` in ``same`` mode written out: the rows'
    transform at the next fast length, times the filters', back, and the
    middle kept. The filters' transform is the same call on the same array
    every time, a third of a tail's filtering, and is kept by rate and bands.
    """
    from scipy import fft as sp_fft

    block = np.atleast_2d(np.asarray(rows, dtype=float))
    held = int(block.shape[1])
    taps = int(_bank_kernels(rate).shape[1])
    full = held + taps - 1
    transform = int(sp_fft.next_fast_len(full, True))
    spectra = _bank_spectra(rate, tuple(int(b) for b in np.asarray(bands).ravel()), transform)
    out = sp_fft.irfftn(sp_fft.rfftn(block, [transform], axes=[1]) * spectra, [transform], axes=[1])
    first = (full - held) // 2
    return np.asarray(out[:, :full][:, first : first + held].copy())


def band_rows(rows: Any, rate: float, bands: np.ndarray, xp: Any = np) -> Any:
    """:func:`octave_filter_rows` on ``xp``: row k through band ``bands[k]``, same length."""
    if xp is np:
        return _band_rows_host(np.asarray(rows, dtype=float), rate, bands)
    from cupyx.scipy.signal import fftconvolve  # type: ignore[import-not-found]

    kernels = xp.asarray(_bank_kernels(rate))[xp.asarray(np.asarray(bands, dtype=int))]
    return fftconvolve(xp.asarray(rows, dtype=xp.float64), kernels, mode="same", axes=1)


@lru_cache(maxsize=8)
def band_pulse_energy(rate: float) -> np.ndarray:
    """The energy per band of a unit pulse through the bank, over its whole response.

    A 1 ms window around the direct pulse keeps only part of it where the
    band filter rings longer than the window: -4.5, -3.9, -3.7 and -0.7 dB at
    125, 250, 500 and 1000 Hz at 48 kHz. The tail's scale reads this instead.
    """
    centres, _ = _band_map(rate)
    length = int(round(0.2 * rate))
    impulse = np.zeros((len(centres), length))
    impulse[:, length // 2] = 1.0
    rows = octave_filter_rows(impulse, int(round(rate)), np.arange(len(centres)))
    energy = np.asarray(np.sum(rows**2, axis=1), dtype=float)
    energy.setflags(write=False)
    return energy


@lru_cache(maxsize=8)
def bank_reading(rate: float) -> np.ndarray:
    """``M[b, k]``: what the bank's band b reads of unit energy synthesised in band k.

    The tail is noise shaped by band k's filter, so a reading through the
    same bank applies that filter twice and leaks into the neighbours: a flat
    synthesis reads 1 dB low in every band and 4.6 dB low at 125 Hz. The
    tail's energies go through the inverse of this matrix first.
    """
    centres, _ = _band_map(rate)
    count = len(centres)
    fs = int(round(rate))
    noise = np.random.default_rng(12345).standard_normal(2 * fs)
    shaped = octave_filter_rows(np.repeat(noise[None, :], count, 0), fs, np.arange(count))
    shaped /= np.sqrt(np.sum(shaped**2, axis=1, keepdims=True))
    reading = np.zeros((count, count))
    for k in range(count):
        read = octave_filter_rows(np.repeat(shaped[k][None, :], count, 0), fs, np.arange(count))
        reading[:, k] = np.sum(read**2, axis=1)
    reading.setflags(write=False)
    return reading


def early_signals(
    paths: Paths, settings: RenderSettings, sound_speed_m_s: float, xp: Any = np
) -> Any:
    """The discrete part as ``[channel, sample]`` on ``xp``: every pulse, every band at once."""
    rate = settings.sample_rate_hz
    length = int(round(settings.duration_s * rate))
    channels = channel_count(settings.order)
    centres, picks = _band_map(rate)
    if paths.count == 0:
        return xp.zeros((channels, length))
    directions = scene_to_ambisonic(paths.direction)
    harmonics = real_sh(settings.order, directions)  # [path, channel]
    delays = paths.length_m / sound_speed_m_s * rate
    half = DELAY_HALF_TAPS
    # The early part lives in a short buffer, filtered there, then placed.
    latest = int(np.ceil(delays.max())) + half + 2
    span = latest + 1024
    kernels, indices = _fractional_pulses(delays, np.ones(paths.count), half)
    bands = len(centres)
    # value[band, path, tap, channel] = gain[path, band] kernel[path, tap] Y[path, channel],
    # added path after path as slices: the same order of additions on every device.
    weighted = paths.gain[:, picks].T[:, :, None] * kernels[None, :, :]  # [band, path, tap]
    # Laid on the host, where a slice costs nothing, then moved once.
    values = weighted[:, :, :, None] * harmonics[None, :, None, :]
    laid = np.zeros((bands, span, channels))
    for path in range(paths.count):
        first = int(indices[path, 0])
        lo = max(first, 0)
        hi = min(first + indices.shape[1], span)
        if hi <= lo:
            continue
        laid[:, lo:hi, :] += values[:, path, lo - first : hi - first, :]
    rows = xp.asarray(np.ascontiguousarray(laid.transpose(0, 2, 1)).reshape(bands * channels, span))
    filtered = band_rows(rows, rate, np.repeat(np.arange(bands), channels), xp)
    early = filtered.reshape(bands, channels, span).sum(axis=0)
    signals = xp.zeros((channels, length))
    take = min(span, length)
    signals[:, :take] = early[:, :take]
    return signals


@dataclass
class TailNoise:
    """The draws of a tail's seed, kept for the next tail of that seed.

    A tail's noise is a function of its seed and of its sizes alone: eleven
    million normal deviates for 1.2 s at 48 kHz, 40 ms of one core, and
    their sum of squares bin by bin. The levelling renders the mirror at
    every pair, and a pair's seed is its cell's, so the pairs of one cell
    draw the same noise: read here, they draw it once. The arrays are the
    generator's own, in its own order, and are not written to afterwards.
    """

    #: Seeds kept: each is 8 bytes a deviate, 88 MB for the levelling's tail.
    keep: int = 2
    drawn: int = 0
    read: int = 0
    _held: OrderedDict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=OrderedDict, repr=False
    )

    def of(
        self, seed: int, held: int, count: int, bursts: int, bin_samples: int, length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The uniform draws ``[bin, band, burst]``, the bursts and their energies a bin."""
        name = (int(seed), held, count, bursts, bin_samples, length)
        found = self._held.get(name)
        if found is None:
            found = _tail_draws(seed, held, count, bursts, bin_samples, length)
            self._held[name] = found
            self.drawn += 1
            while len(self._held) > max(1, self.keep):
                self._held.popitem(last=False)
        else:
            self._held.move_to_end(name)
            self.read += 1
        return found


def _tail_draws(
    seed: int, held: int, count: int, bursts: int, bin_samples: int, length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a tail takes of its generator, in the generator's order, on the host."""
    rng = np.random.default_rng(seed)
    drawn = rng.random((held, count, bursts))
    span = held * bin_samples
    draws = rng.standard_normal((count, bursts, span))
    if span > length:
        draws[:, :, length:] = 0.0
    segments = draws.reshape(count, bursts, held, bin_samples)
    return drawn, segments, np.sum(segments**2, axis=-1)


def tail_from_histogram(
    histogram: Histogram,
    receiver: int,
    settings: RenderSettings,
    *,
    sound_speed_m_s: float,
    start_s: float,
    seed: int,
    bursts: int,
    scale_per_band: np.ndarray,
    band_gain_db: np.ndarray | None = None,
    xp: Any = np,
    noise: TailNoise | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The tail as noise shaped by a receiver's histogram, band by band, with its anisotropy.

    Per band and per time bin the histogram gives the energy and its
    directional moments to the histogram's order. The energy is drawn as
    ``bursts`` independent noise bursts per bin, each from a direction
    sampled with the probability the moments give it (the moments'
    expansion, clipped at zero, on a quadrature of the sphere), each encoded
    on the harmonics of its direction at the output order. Channels are
    therefore decorrelated as a diffuse field is, and where the histogram
    is anisotropic the tail is too.

    ``scale_per_band`` turns the histogram's energy into the response's, per
    bank band; ``band_gain_db``, per octave band, is the calibration's tail
    gain on top. ``noise`` keeps a seed's draws for the next tail of that
    seed (:class:`TailNoise`), on the host.
    """
    from reverberate.spatial.sh import quadrature

    rate = settings.sample_rate_hz
    length = int(round(settings.duration_s * rate))
    channels = channel_count(settings.order)
    centres, picks = _band_map(rate)
    energy = histogram.energy[receiver]  # [bin, band]
    moments = histogram.moments[receiver]  # [bin, band, channel]
    bins = energy.shape[0]
    bin_samples = int(round(histogram.bin_s * rate))
    grid, weights = quadrature(2 * histogram.order + 2)
    basis_low = real_sh(histogram.order, grid)  # [direction, low channel]
    basis_out = real_sh(settings.order, grid)  # [direction, out channel]
    from_bin = int(np.ceil((start_s + settings.tail_from_s) / histogram.bin_s))
    scale = np.asarray(scale_per_band, dtype=float)[: len(picks)]
    band_power = np.ones(len(OCTAVE_BANDS))
    if band_gain_db is not None:
        band_power = 10.0 ** (np.asarray(band_gain_db, dtype=float) / 10.0)
    # The energy each bin should read per band, then what to synthesise so the
    # bank reads it: the inverse of the bank's own reading, clipped at zero.
    wanted = energy[:, picks] * (scale * band_power[picks])[None, :]  # [bin, band]
    wanted = np.maximum(np.linalg.solve(bank_reading(rate), wanted.T).T, 0.0)
    # Every bin and band at once: the directions drawn from the moments'
    # density on the quadrature, the bursts laid bin by bin, each burst an
    # equal share of its bin's energy, encoded on its direction's harmonics.
    count = len(picks)
    held = min(bins, -(-length // bin_samples))
    energy_in = np.array(wanted[:held], dtype=float)
    energy_in[: min(from_bin, held)] = 0.0
    span = held * bin_samples
    if xp is not np:
        rng = np.random.default_rng(seed)
        drawn = rng.random((held, count, bursts))
        draws = xp.random.default_rng(seed).standard_normal((count, bursts, span))
        if span > length:
            draws[:, :, length:] = 0.0
        segments = draws.reshape(count, bursts, held, bin_samples)
        have = xp.sum(segments**2, axis=-1)  # [band, burst, bin]
    elif noise is not None:
        drawn, segments, have = noise.of(seed, held, count, bursts, bin_samples, length)
    else:
        drawn, segments, have = _tail_draws(seed, held, count, bursts, bin_samples, length)
    if bool(np.all(basis_out == basis_out[0])):
        # Every direction encodes alike (order 0): which one a burst drew changes nothing,
        # and is not looked for. The draws are taken all the same, above.
        coefficients = xp.asarray(
            np.broadcast_to(basis_out[0], (held, count, bursts, basis_out.shape[1]))
        )
    else:
        density = np.maximum(moments[:held][:, picks, :] @ basis_low.T, 0.0) * weights
        empty = density.sum(axis=-1, keepdims=True) <= 0.0
        density = np.where(empty, weights, density)
        cumulative = np.cumsum(density / density.sum(axis=-1, keepdims=True), axis=-1)
        cumulative[..., -1] = np.inf
        # The same comparisons on either device: the first direction whose
        # cumulative probability passes the draw, [bin, band, burst].
        chosen = xp.argmax(
            xp.asarray(drawn)[..., None] < xp.asarray(cumulative)[:, :, None, :], axis=-1
        )
        coefficients = xp.asarray(basis_out)[chosen]  # [bin, band, burst, channel]
    share = xp.asarray(energy_in.T / bursts)[:, None, :]  # [band, 1, bin]
    gain = xp.where(share > 0.0, xp.sqrt(share / xp.maximum(have, 1e-30)), 0.0)
    coefficients = coefficients * gain.transpose(2, 0, 1)[..., None]
    laid = xp.matmul(
        coefficients.transpose(1, 0, 3, 2),  # [band, bin, channel, burst]
        segments.transpose(0, 2, 1, 3),  # [band, bin, burst, sample]
    )  # [band, bin, channel, sample]
    per_band = xp.zeros((count, channels, length))
    stop = min(span, length)
    per_band[:, :, :stop] = laid.transpose(0, 2, 1, 3).reshape(count, channels, span)[:, :, :stop]
    # Band limit each band's noise to its octave: the bursts were white, so
    # each band's rows go through their own filter and the bands are summed;
    # the filter keeps a fraction of a white burst's energy, restored here
    # band by band on the omni channel so the histogram's energy is kept.
    rows = per_band.reshape(count * channels, length)
    row_bands = np.repeat(np.arange(count), channels)
    shaped = band_rows(rows, rate, row_bands, xp).reshape(count, channels, length)
    wanted_omni = xp.sum(per_band[:, 0] ** 2, axis=1)
    kept_omni = xp.sum(shaped[:, 0] ** 2, axis=1)
    ratio = xp.where(
        (kept_omni > 0.0) & (wanted_omni > 0.0),
        xp.sqrt(wanted_omni / xp.maximum(kept_omni, 1e-300)),
        1.0,
    )
    shaped *= ratio[:, None, None]
    tail = shaped.sum(axis=0)
    record = {
        "kind": "histogram of the rays, noise bursts per bin from sampled directions",
        "bursts_per_bin": bursts,
        "scale_per_band": [round(float(v), 6) for v in scale],
        "seed": seed,
    }
    return tail, record


@lru_cache(maxsize=8)
def _lowcut_sections(rate: float) -> np.ndarray:
    """The reference chain's low cut as second order sections: designed once a rate."""
    sos = np.asarray(butter(LOWCUT_ORDER, LOWCUT_HZ, btype="high", fs=rate, output="sos"))
    sos.setflags(write=False)
    return sos


@lru_cache(maxsize=8)
def _lowcut_response(n_fft: int, rate: float) -> np.ndarray:
    """The reference chain's low cut at the bins of a transform of ``n_fft``: read once."""
    sos = butter(LOWCUT_ORDER, LOWCUT_HZ, btype="high", fs=rate, output="sos")
    _, lowcut = sosfreqz(sos, worN=np.fft.rfftfreq(n_fft, 1.0 / rate), fs=rate)
    lowcut = np.asarray(lowcut)
    lowcut.setflags(write=False)
    return lowcut


def _through_spectrum(
    signals: Any, rate: float, taps: np.ndarray | None, sos: np.ndarray, xp: Any
) -> Any:
    """The signature's FIR and the low cut's IIR as one spectrum, on ``xp``.

    The transform is at least twice the response, so what the recursion
    would ring past the end (about -140 dB after a second) is what folds
    back: 1e-13 of the peak against the recursion.
    """
    n = signals.shape[-1]
    extra = 0 if taps is None else taps.size
    n_fft = 1 << int(np.ceil(np.log2(2 * n + extra)))
    spectrum = xp.fft.rfft(signals, n_fft, axis=-1)
    response = np.ones(n_fft // 2 + 1, dtype=complex)
    if taps is not None and taps.size:
        response *= np.fft.rfft(taps, n_fft)
    lowcut = _lowcut_response(n_fft, float(rate))
    spectrum *= xp.asarray(response * lowcut)[None, :]
    return xp.fft.irfft(spectrum, n_fft, axis=-1)[..., :n]


def render_point(
    index: int,
    paths: Paths,
    histogram: Histogram,
    settings: RenderSettings,
    *,
    tail_gain_db: np.ndarray,
    receiver_radius_m: float,
    sound_speed_m_s: float,
    seed: int,
    fallback: tuple[Any, ...] | None = None,
    signature: np.ndarray | None = None,
    xp: Any = np,
    noise: TailNoise | None = None,
) -> tuple[Ambisonic, dict[str, Any]]:
    """Point ``index``: its paths, then its histogram's tail, then air, low cut and signature.

    With a direct path the tail is scaled on it: the energy a sphere of
    radius ``receiver_radius_m`` at that distance catches of rays of energy
    1/N, against the direct pulse's whole energy per band. Without one,
    ``fallback`` is ``(scale_per_band, straight_line_s, onset)``: the scale the
    other points read (``None`` when no point had one: the onset alone), the
    straight line time, and the point's diffracted onset (a :class:`Paths`, or
    ``None``), whose arrival starts the tail.
    """
    from reverberate.accel.dsp import air_absorption, sosfilt

    rate = settings.sample_rate_hz
    signals = early_signals(paths, settings, sound_speed_m_s, xp)
    record: dict[str, Any] = {"paths": int(paths.count)}
    direct = paths.order == 0
    heard = histogram.hits[index].sum() > 0
    tail_args = {
        "sound_speed_m_s": sound_speed_m_s,
        "seed": seed,
        "bursts": settings.tail_bursts,
        "band_gain_db": np.asarray(tail_gain_db, dtype=float),
        "xp": xp,
        "noise": noise,
    }
    if direct.any() and heard:
        distance = float(paths.length_m[direct][0])
        _, picks = _band_map(rate)
        expected = receiver_radius_m**2 / (4.0 * max(distance, 1.05 * receiver_radius_m) ** 2)
        amplitude = paths.gain[np.flatnonzero(direct)[0]][picks]
        scale = np.asarray(amplitude**2 * band_pulse_energy(rate) / expected, dtype=float)
        tail, record["tail"] = tail_from_histogram(
            histogram,
            index,
            settings,
            start_s=distance / sound_speed_m_s,
            scale_per_band=scale,
            **tail_args,
        )
        signals = signals + tail
    elif fallback is not None and not direct.any():
        scale, straight_s = fallback[0], fallback[1]
        onset = fallback[2]
        # The rays' own first arrival round the doorway starts the tail; the
        # straight line through the wall is not a clock here.
        arrived = np.any(histogram.energy[index] > 0.0, axis=1)
        start_s = max(
            float(np.argmax(arrived)) * histogram.bin_s - settings.tail_from_s, straight_s
        )
        if onset is not None:
            start_s = float(onset.length_m[0]) / sound_speed_m_s
            signals = signals + early_signals(onset, settings, sound_speed_m_s, xp)
            record["diffracted"] = {
                "length_m": round(float(onset.length_m[0]), 4),
                "gain_db": [round(float(v), 2) for v in 20.0 * np.log10(onset.gain[0])],
            }
        record["tail"] = None
        if heard and scale is not None:
            tail, tail_record = tail_from_histogram(
                histogram,
                index,
                settings,
                start_s=start_s,
                scale_per_band=scale,
                **tail_args,
            )
            signals = signals + tail
            record["tail"] = {**tail_record, "starts_s": round(start_s, 4)}
    else:
        record["tail"] = None if not direct.any() else "no ray reached this receiver"
    signals = air_absorption(signals, rate, xp, sound_speed_m_s=sound_speed_m_s)
    # A copy: the recursion's own code asks for an array it may write to.
    sos = _lowcut_sections(float(rate)).copy()
    if xp is np:
        if signature is not None and signature.size:
            signals = apply_signature(signals, signature)
        signals = sosfilt(sos, signals, xp)
    else:
        # On a card the recursion is slow and a transform is not.
        signals = _through_spectrum(signals, rate, signature, sos, xp)
    return Ambisonic(to_numpy(signals), rate, settings.order, paths.receiver), record
