"""The direct sound: where it arrives, how much energy it carries, and the source's signature.

The wave solver drives its source with a pulse of its own, and its grid
colours what it carries; its direct sound is therefore not flat, while the
mirror renders every path as a flat pulse. The mirror is given the
reference's direct spectrum as a short minimum phase filter, convolved into
every response. It is a property of the source, not of the room.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.signal import butter, minimum_phase, sosfiltfilt

__all__ = [
    "apply_signature",
    "bandpass",
    "direct_arrival",
    "direct_energy",
    "measure_signature",
]

#: The direct sound is looked for in this band, Hz.
DETECTION_BAND_HZ = (500.0, 8000.0)
#: Energy of the direct sound is read over this window around its arrival, s.
DIRECT_WINDOW_S = 0.0005


def bandpass(signals: np.ndarray, rate: float, low_hz: float | None, high_hz: float) -> np.ndarray:
    """Zero phase band limiting, so a pulse does not move in time."""
    nyquist = 0.5 * rate
    high = min(high_hz, 0.98 * nyquist)
    if low_hz is not None and low_hz > 0.0:
        sos = butter(4, [low_hz / nyquist, high / nyquist], btype="band", output="sos")
    else:
        sos = butter(4, high / nyquist, btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, signals, axis=-1), dtype=float)


def direct_arrival(
    signals: np.ndarray,
    rate: float,
    *,
    band_hz: tuple[float, float] = DETECTION_BAND_HZ,
    window_s: float = DIRECT_WINDOW_S,
) -> int:
    """The sample of the direct sound: the earliest strong peak of the band limited omni channel.

    The first sample within 6 dB of the strongest, then the local maximum
    within ``window_s`` after it: in a corner the strongest arrival can be a
    reflection. Raises ``ValueError`` on a silent response.
    """
    omni = bandpass(signals[0:1], rate, band_hz[0], band_hz[1])[0]
    envelope = omni**2
    peak = float(envelope.max())
    if peak <= 0.0:
        raise ValueError("a silent response has no direct sound")
    first = int(np.flatnonzero(envelope >= peak * 10 ** (-6.0 / 10.0))[0])
    stop = min(first + int(round(window_s * rate)), envelope.size)
    return first + int(np.argmax(envelope[first:stop]))


def direct_energy(signals: np.ndarray, rate: float, window_s: float = DIRECT_WINDOW_S) -> float:
    """The omni channel's energy over ``window_s`` around the direct sound; zero when silent."""
    try:
        start = direct_arrival(signals, rate, window_s=window_s)
    except ValueError:
        return 0.0
    half = int(round(window_s * rate)) // 2
    omni = signals[0, max(start - half, 0) : start + half + 1]
    return float(np.sum(omni**2))


def measure_signature(
    omni: Iterable[np.ndarray], rate: float, *, window_s: float = 0.002, taps: int = 128
) -> tuple[np.ndarray, dict[str, Any]]:
    """The minimum phase filter with the median direct spectrum of ``omni``, and its record.

    Each direct pulse is cut over ``window_s`` around its arrival and its
    magnitude spectrum normalised at its peak; the filter of ``taps`` fits
    the median over the responses and has unit energy, so the alignment
    gain keeps its meaning.
    """
    n_fft = 4096
    half = int(round(window_s * rate / 2))
    spectra = []
    for signal in omni:
        try:
            start = direct_arrival(signal[None, :], rate)
        except ValueError:
            continue
        cut = signal[max(start - half // 2, 0) : start + half + half // 2]
        if cut.size < 8 or float(np.max(np.abs(cut))) == 0.0:
            continue
        magnitude = np.abs(np.fft.rfft(cut * np.hanning(cut.size), n_fft))
        spectra.append(magnitude / magnitude.max())
    if not spectra:
        raise ValueError("no response with a clean direct sound to read the signature from")
    median = np.median(np.stack(spectra), axis=0)
    # A linear phase prototype with that magnitude, then its minimum phase version.
    prototype = np.roll(np.fft.irfft(median, n_fft), n_fft // 2)
    centre = n_fft // 2
    linear = prototype[centre - taps : centre + taps + 1] * np.hanning(taps * 2 + 1)
    filt = minimum_phase(linear / np.sum(np.abs(linear)), method="homomorphic", n_fft=n_fft * 4)
    filt = filt[:taps] / np.sqrt(np.sum(filt[:taps] ** 2))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / rate)
    checkpoints = [500, 1000, 2000, 4000, 8000, 16000]
    record = {
        "responses": len(spectra),
        "window_s": window_s,
        "taps": int(taps),
        "sample_rate_hz": rate,
        "median_db": {
            str(f): round(
                float(20.0 * np.log10(max(median[np.argmin(np.abs(frequencies - f))], 1e-9))), 2
            )
            for f in checkpoints
        },
    }
    return filt, record


def apply_signature(signals: Any, taps: np.ndarray, xp: Any = np) -> Any:
    """Every channel convolved with the signature, same length, on ``xp``."""
    if taps.size == 0:
        return signals
    n = signals.shape[-1]
    n_fft = 1 << int(np.ceil(np.log2(n + taps.size)))
    spectrum = xp.fft.rfft(signals, n_fft, axis=-1) * xp.fft.rfft(xp.asarray(taps), n_fft)[None, :]
    return xp.fft.irfft(spectrum, n_fft, axis=-1)[..., :n]
