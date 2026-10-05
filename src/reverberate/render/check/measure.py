"""What the check measures: each function a number from a signal, with no pack and no engine.

Kept apart from :mod:`.run` so that every detector is tested on a signal
whose answer is known, and on a seeded fault that must make it fire.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin, hilbert

from reverberate.metrics import energy_decay_curve, octave_filter

__all__ = [
    "FULL_SCALE_SPL_DB",
    "PAGE_DEFAULT_GAIN",
    "PAGE_DEFAULT_LEVEL_DB",
    "THIRD_OCTAVES_HZ",
    "Sidebands",
    "a_weighted",
    "annihilate",
    "arrival_near",
    "band_split",
    "c50_db",
    "comb",
    "decay_times_s",
    "difference_outlier",
    "direction_of",
    "envelope_arrival",
    "frame_levels_db",
    "instantaneous_hz",
    "join_step",
    "level_step_db",
    "pink_noise",
    "seam_deviation_db",
    "sidebands_db",
    "spl_db",
    "third_octave_db",
    "tone_residual_db",
]

#: What 0 dB re full scale stands for, at 1 m: ``docs/formats/clip-library.md``.
FULL_SCALE_SPL_DB = 86.0
#: The level the audit page plays at unless told, dB, and so the level of the files written
#: for listening: ``DEFAULT_LEVEL_DB`` of ``viz/app/scene/sound-plan.js``, the same number.
#: A pack is physical, a voice at 1 m about 60 dB SPL; at 0 dB three of a scene's fourteen
#: sources peaked at -3.8 dB re full scale in the two ears and fourteen pass it. At -12 dB
#: the output's full scale stands for 98 dB SPL: the levels between sources are the
#: scene's own, and the listener makes up the 12 dB on the volume knob.
PAGE_DEFAULT_LEVEL_DB = -12.0
PAGE_DEFAULT_GAIN = float(10.0 ** (PAGE_DEFAULT_LEVEL_DB / 20.0))
#: Third octave centres the spectra are read on, base two from 1 kHz.
THIRD_OCTAVES_HZ = tuple(float(1000.0 * 2.0 ** (k / 3.0)) for k in range(-10, 13))
#: The third octaves round the crossover, and the three either side they are read against.
SEAM_HZ = (630.0, 1600.0)
SEAM_BELOW_HZ = (315.0, 500.0)
SEAM_ABOVE_HZ = (2000.0, 3150.0)


def db(ratio: float | np.ndarray, *, power: bool = False) -> np.ndarray:
    """Decibels of a ratio, ``-inf`` at zero and no warning."""
    value = np.maximum(np.asarray(ratio, dtype=float), 1e-300)
    return np.asarray((10.0 if power else 20.0) * np.log10(value))


def third_octave_db(signal: np.ndarray, rate: float) -> np.ndarray:
    """Energy of ``signal`` in each band of :data:`THIRD_OCTAVES_HZ`, in dB.

    The bins of one transform summed between ``f 2^(-1/6)`` and ``f 2^(1/6)``:
    a response is short and is read whole, so no filter rings into it.
    """
    x = np.asarray(signal, dtype=float)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    out = np.empty(len(THIRD_OCTAVES_HZ))
    for i, centre in enumerate(THIRD_OCTAVES_HZ):
        inside = (freqs >= centre * 2.0 ** (-1.0 / 6.0)) & (freqs < centre * 2.0 ** (1.0 / 6.0))
        out[i] = float(db(np.sum(power[inside]) / max(x.size, 1), power=True))
    return out


def _centres_between(low: float, high: float) -> np.ndarray:
    centres = np.asarray(THIRD_OCTAVES_HZ)
    return np.flatnonzero((centres > low * 0.97) & (centres < high * 1.03))


def seam_deviation_db(levels_db: np.ndarray) -> tuple[float, np.ndarray]:
    """How far the third octaves round the crossover lie from their neighbours' line.

    The line is straight in dB against the logarithm of the frequency, through
    the mean of the three bands below (315 to 500 Hz) and of the three above
    (2 to 3.15 kHz): a tilt of the source or of the room is not a hole.
    Returns the worst departure, signed, and all five.
    """
    centres = np.log2(np.asarray(THIRD_OCTAVES_HZ))
    below, above = _centres_between(*SEAM_BELOW_HZ), _centres_between(*SEAM_ABOVE_HZ)
    seam = _centres_between(*SEAM_HZ)
    x0, x1 = float(np.mean(centres[below])), float(np.mean(centres[above]))
    y0, y1 = float(np.mean(levels_db[below])), float(np.mean(levels_db[above]))
    line = y0 + (centres[seam] - x0) * (y1 - y0) / (x1 - x0)
    departure = np.asarray(levels_db)[seam] - line
    return float(departure[int(np.argmax(np.abs(departure)))]), departure


def a_weighted(signal: np.ndarray, rate: float) -> np.ndarray:
    """``signal`` through the A weighting of IEC 61672, applied on its transform."""
    x = np.asarray(signal, dtype=float)
    f2 = np.fft.rfftfreq(x.shape[-1], 1.0 / rate) ** 2
    weight = (
        12194.0**2
        * f2**2
        / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    )
    return np.asarray(np.fft.irfft(np.fft.rfft(x) * weight * 10.0 ** (2.0 / 20.0), n=x.shape[-1]))


def spl_db(signal: np.ndarray) -> float:
    """The level of a pressure signal in dB SPL, full scale being :data:`FULL_SCALE_SPL_DB`."""
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        return float("-inf")
    return float(FULL_SCALE_SPL_DB + db(np.mean(x * x), power=True))


def frame_levels_db(signal: np.ndarray, frame: int) -> np.ndarray:
    """The level of each whole frame of ``frame`` samples, dB re full scale."""
    x = np.asarray(signal, dtype=float)
    count = x.size // frame
    frames = x[: count * frame].reshape(count, frame)
    return db(np.mean(frames * frames, axis=1), power=True)


def envelope_arrival(signal: np.ndarray, *, share: float = 0.5) -> int:
    """The sample of the first peak of the envelope that reaches ``share`` of its largest.

    The first arrival of a response whose later arrivals may be louder: the
    envelope is the analytic signal's, so a band limited pulse peaks where it
    arrives and not a quarter period away.
    """
    envelope = np.abs(hilbert(np.asarray(signal, dtype=float)))
    top = float(envelope.max())
    if top <= 0.0:
        return -1
    over = envelope >= share * top
    start = int(np.argmax(over))
    stop = start
    while stop + 1 < envelope.size and over[stop + 1]:
        stop += 1
    return start + int(np.argmax(envelope[start : stop + 1]))


def arrival_near(
    signal: np.ndarray, expected: float, reach: int, *, share: float = 0.6
) -> tuple[int, float]:
    """The first peak of the envelope within ``reach`` samples of ``expected``, and its size.

    Where a response's direct sound is, when something says where to look:
    at a far or shadowed place a band's direct sound is weaker than what
    follows it by 5 to 20 ms, and a first arrival read against the band's
    largest peak (:func:`envelope_arrival`) is then a reflection. The first
    peak, not the largest: a floor's reflection follows the direct sound by
    1 to 2 ms and may be the larger. A peak counts from ``share`` of the
    largest envelope inside the search; under it are the ripples a band
    limited pulse has before it. The size is the peak over the envelope's
    largest anywhere. ``(-1, 0.0)`` where the envelope has no peak there and
    only rises or falls through the search.
    """
    envelope = np.abs(hilbert(np.asarray(signal, dtype=float)))
    centre = int(round(expected))
    low, high = max(centre - reach, 1), min(centre + reach, envelope.size - 2)
    top = float(envelope.max())
    if high <= low or top <= 0.0:
        return -1, 0.0
    held = envelope[low : high + 1]
    peaks = np.flatnonzero(
        (held >= share * float(held.max()))
        & (held >= envelope[low - 1 : high])
        & (held >= envelope[low + 1 : high + 2])
    )
    if peaks.size == 0:
        return -1, 0.0
    at = low + int(peaks[0])
    return at, float(envelope[at]) / top


def direction_of(first_order: np.ndarray) -> np.ndarray:
    """The direction a sound comes from, from the first four channels (ACN, N3D).

    The intensity vector: the omnidirectional channel times the three
    dipoles, summed over the window. A unit vector in the ambisonic frame;
    exact for one plane wave.
    """
    block = np.asarray(first_order, dtype=float)
    vector = np.array(
        [np.sum(block[0] * block[3]), np.sum(block[0] * block[1]), np.sum(block[0] * block[2])]
    )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def decay_times_s(response: np.ndarray, rate: float, arrival: int) -> tuple[np.ndarray, np.ndarray]:
    """T20 and T30 per octave band of the bank, from the Schroeder curve after ``arrival``.

    NaN where the curve does not fall far enough to be read.
    """
    bands = octave_filter(np.asarray(response, dtype=float)[max(arrival, 0) :], int(rate))
    t20, t30 = np.full(bands.shape[0], np.nan), np.full(bands.shape[0], np.nan)
    for i, band in enumerate(bands):
        curve = energy_decay_curve(band)
        time = np.arange(curve.size) / rate
        for out, end in ((t20, -25.0), (t30, -35.0)):
            span = (curve <= -5.0) & (curve >= end)
            if curve[-1] <= end and np.count_nonzero(span) > 8:
                slope = float(np.polyfit(time[span], curve[span], 1)[0])
                if slope < 0.0:
                    out[i] = -60.0 / slope
    return t20, t30


def c50_db(response: np.ndarray, rate: float, arrival: int) -> np.ndarray:
    """Early to late ratio per octave band: the 50 ms after ``arrival`` over the rest."""
    bands = octave_filter(np.asarray(response, dtype=float), int(rate))
    split = arrival + int(round(0.050 * rate))
    early = np.sum(bands[:, max(arrival - int(0.002 * rate), 0) : split] ** 2, axis=1)
    late = np.sum(bands[:, split:] ** 2, axis=1)
    return db(early / np.maximum(late, 1e-300), power=True)


# --- continuity ---------------------------------------------------------------


def band_split(signal: np.ndarray, rate: float, cut_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """``signal`` under and over ``cut_hz``, zero phase; the two add back to it."""
    taps = firwin(1201, cut_hz, fs=rate)
    low = np.asarray(fftconvolve(np.asarray(signal, dtype=float), taps, mode="same"))
    return low, np.asarray(signal, dtype=float) - low


def annihilate(signal: np.ndarray, tone_hz: float, rate: float) -> np.ndarray:
    """``x[n] - 2 cos(w) x[n-1] + x[n-2]``: zero for a steady tone at ``tone_hz``.

    Whatever is left is what is not that tone: a click, a step of gain, a
    jump of phase. A gain that moves smoothly leaves twice its change a
    sample.
    """
    x = np.asarray(signal, dtype=float)
    return np.asarray(x[2:] - 2.0 * np.cos(2.0 * np.pi * tone_hz / rate) * x[1:-1] + x[:-2])


def tone_residual_db(signal: np.ndarray, tone_hz: float, rate: float) -> tuple[float, int]:
    """The largest sample the annihilator leaves, in dB re the tone's amplitude, and where.

    The amplitude is the root of twice the mean square. A steady tone reads
    the float's rounding, -250 dB or so; a click of one sample reads its own
    size; a 50 ms raised cosine from nothing to full reads -58 dB.
    """
    x = np.asarray(signal, dtype=float)
    amplitude = float(np.sqrt(2.0 * np.mean(x * x)))
    if amplitude <= 0.0 or x.size < 3:
        return float("-inf"), 0
    rest = np.abs(annihilate(x, tone_hz, rate))
    at = int(np.argmax(rest))
    return float(db(rest[at] / amplitude)), at + 1


def difference_outlier(
    signal: np.ndarray, rate: float, *, window_s: float = 0.02
) -> tuple[float, int]:
    """The largest sample to sample difference over the local rms of the differences.

    The local rms is taken over ``window_s`` round the sample, the sample
    left out. For Gaussian noise through any linear system the ratio is a
    standard normal's: 6.5 is passed once in 1.2e10 samples, 8 once in 8e14.
    """
    step = np.diff(np.asarray(signal, dtype=float))
    half = max(int(round(0.5 * window_s * rate)), 8)
    square = step * step
    total = np.concatenate([[0.0], np.cumsum(square)])
    lo = np.maximum(np.arange(step.size) - half, 0)
    hi = np.minimum(np.arange(step.size) + half + 1, step.size)
    local = (total[hi] - total[lo] - square) / np.maximum(hi - lo - 1, 1)
    floor = 1e-12 * float(np.mean(square)) + 1e-300
    ratio = np.abs(step) / np.sqrt(np.maximum(local, floor))
    at = int(np.argmax(ratio))
    return float(ratio[at]), at + 1


def comb(samples: int, rate: float, tones_hz: Iterable[float], *, seed: int = 0) -> np.ndarray:
    """Tones of one size at ``tones_hz``, their phases drawn from ``seed``, at an rms of 0.1.

    With every tone a multiple of 100 Hz, a frame of 10 ms holds a whole
    number of periods of each and of every difference between two: the
    power of the comb is the same in every such frame, to rounding, which
    no noise is. It is what a band's level is read on
    (:func:`level_step_db`): through a room each tone fades on its own as
    things move, and the sum of many keeps the band's level.
    """
    tones = np.asarray(list(tones_hz), dtype=float)
    phases = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, tones.size)
    time = np.arange(samples) / rate
    made = np.zeros(samples)
    for tone, phase in zip(tones, phases, strict=True):
        made += np.sin(2.0 * np.pi * tone * time + phase)
    return np.asarray(made * (0.1 / np.sqrt(0.5 * max(tones.size, 1))))


def join_step(signal: np.ndarray, at: int, rate: float, *, window_s: float = 0.02) -> float:
    """The step from sample ``at - 1`` to ``at`` over the rms of the steps round it.

    What two pieces put end to end are held to where they meet: a signal
    that goes on through the join steps there as it does everywhere, about
    once its rms step; a cut reads the signal's size over that, tens of
    times it. The statistic of :func:`difference_outlier`, at one place.
    """
    x = np.asarray(signal, dtype=float)
    half = max(int(round(0.5 * window_s * rate)), 8)
    lo, hi = max(at - half, 0), min(at + half, x.size)
    if not lo < at < hi:
        return 0.0
    steps = np.diff(x[lo:hi])
    here = at - 1 - lo
    others = np.delete(steps, here)
    typical = float(np.sqrt(np.mean(others * others))) if others.size else 0.0
    if typical <= 0.0:
        return 0.0 if steps[here] == 0.0 else float("inf")
    return float(abs(steps[here]) / typical)


def level_step_db(
    signal: np.ndarray,
    rate: float,
    *,
    frame_s: float = 0.01,
    within_db: float = 15.0,
    envelope: bool = True,
) -> tuple[float, int]:
    """The largest change of level between a frame and the next but one, in dB, and where.

    Frames of ``frame_s`` of the envelope's power, or of the signal's own
    without ``envelope`` (a :func:`comb`, whose frames hold whole periods).
    The frame between the two is skipped so that a step in the middle of a
    frame is read whole. Only pairs within ``within_db`` of the median level
    count: a tone carried through a null of the room moves fast and is not a
    step.
    """
    held = np.asarray(signal, dtype=float)
    read = np.abs(hilbert(held)) if envelope else held
    frame = max(int(round(frame_s * rate)), 1)
    levels = frame_levels_db(read, frame)
    if levels.size < 3:
        return 0.0, 0
    floor = float(np.median(levels)) - within_db
    change = np.abs(levels[2:] - levels[:-2])
    change[(levels[2:] < floor) | (levels[:-2] < floor)] = 0.0
    at = int(np.argmax(change))
    return float(change[at]), (at + 1) * frame


@dataclass(frozen=True)
class Sidebands:
    """Lines at multiples of a rate either side of a tone."""

    #: The loudest line, dB re the carrier.
    level_db: float
    #: That line over the louder of what lies halfway to its two neighbours, dB.
    prominence_db: float
    #: Which multiple it is.
    harmonic: int


def sidebands_db(
    signal: np.ndarray,
    rate: float,
    tone_hz: float,
    spacing_hz: float,
    *,
    harmonics: int = 5,
    beyond_hz: float = 0.0,
) -> Sidebands:
    """The lines ``spacing_hz`` apart round ``tone_hz``: a gain that moves in steps.

    A Blackman-Harris window over the whole signal, whose side lobes are
    92 dB down; the carrier is the largest bin within one per cent of the
    tone (a Doppler shift moves it), a line the largest within two bins of
    its place, and its floor the louder of the two places halfway to the
    next lines: a carrier spread by a movement has a skirt, which is not a
    line. The line returned is the loudest that stands 6 dB over its floor,
    or failing one the loudest. Lines nearer the carrier than the window
    resolves (a signal shorter than eight periods of the spacing) read
    nothing.

    ``beyond_hz`` is the Doppler spread of the carrier: how far a path of
    the scene may shift the tone, each way. A moving source or listener
    turns one tone into as many as there are paths, each at its own shift,
    and those are lines too: a place within ``beyond_hz`` (and the window's
    two bins) of the carrier is not read, and where no place is left the
    answer is no line.
    """
    x = np.asarray(signal, dtype=float)
    n = np.arange(x.size)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    window = sum(
        ((-1) ** k) * a[k] * np.cos(2.0 * np.pi * k * n / max(x.size - 1, 1)) for k in range(4)
    )
    spectrum = np.abs(np.fft.rfft(x * window))
    resolution = rate / x.size
    near = slice(int(0.99 * tone_hz / resolution), int(1.01 * tone_hz / resolution) + 2)
    carrier_bin = near.start + int(np.argmax(spectrum[near]))
    carrier = float(spectrum[carrier_bin])
    if carrier <= 0.0:
        return Sidebands(float("-inf"), 0.0, 0)
    if spacing_hz < 8.0 * resolution:
        return Sidebands(float("-inf"), 0.0, 0)

    def around(offset_hz: float) -> float:
        at = carrier_bin + int(round(offset_hz / resolution))
        return (
            float(spectrum[max(at - 2, 1) : at + 3].max()) if 3 <= at < spectrum.size - 3 else 0.0
        )

    found = []
    for k in range(1, harmonics + 1):
        if beyond_hz > 0.0 and k * spacing_hz <= beyond_hz + 2.0 * resolution:
            continue
        for sign in (-1.0, 1.0):
            line = around(sign * k * spacing_hz)
            floor = max(
                around(sign * (k - 0.5) * spacing_hz), around(sign * (k + 0.5) * spacing_hz)
            )
            prominence = float(db(line / max(floor, 1e-15 * carrier)))
            found.append((prominence >= 6.0, float(db(line / carrier)), prominence, k))
    if not found:
        return Sidebands(float("-inf"), 0.0, 0)
    _, level, prominence, k = max(found)
    return Sidebands(level, prominence, k)


def instantaneous_hz(signal: np.ndarray, rate: float, *, frame_s: float = 0.1) -> np.ndarray:
    """The frequency of a tone per frame: the slope of the analytic signal's phase."""
    phase = np.unwrap(np.angle(hilbert(np.asarray(signal, dtype=float))))
    frame = int(round(frame_s * rate))
    count = phase.size // frame
    out = np.empty(count)
    time = np.arange(frame) / rate
    for i in range(count):
        out[i] = float(np.polyfit(time, phase[i * frame : (i + 1) * frame], 1)[0]) / (2.0 * np.pi)
    return out


def pink_noise(samples: int, rate: float, *, seed: int = 0) -> np.ndarray:
    """Gaussian noise 3 dB an octave down, 100 Hz to 12 kHz, at an rms of 0.1."""
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(samples))
    freqs = np.fft.rfftfreq(samples, 1.0 / rate)
    shape = np.where(
        (freqs >= 100.0) & (freqs <= 12000.0), 1.0 / np.sqrt(np.maximum(freqs, 1.0)), 0.0
    )
    noise = np.fft.irfft(spectrum * shape, n=samples)
    return np.asarray(noise * (0.1 / np.sqrt(np.mean(noise * noise))))
