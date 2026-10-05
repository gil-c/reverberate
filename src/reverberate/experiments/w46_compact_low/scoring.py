"""What the measurements of W46 share: third octaves, the head's sphere, the two windows.

An error is the energy of a difference over the energy of the truth, read
as :mod:`reverberate.experiments.w44_interpolation.line_channels` reads it:
on a sphere of the head's radius, each channel weighted by ``j_n(k a)``,
and per third octave from 100 Hz to 1250 Hz. It is read twice, on the 50 ms
after the arrival and on the whole response, and the worst case stands
beside the ninth decile: never a median alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import spherical_jn

from reverberate.spatial.sh import degrees_of
from reverberate.spatial.translate import SOUND_SPEED_M_S

__all__ = [
    "BANDS_HZ",
    "BEFORE_S",
    "EARLY_S",
    "HEAD_M",
    "RATE_HZ",
    "band_masks",
    "decibels",
    "early_window",
    "head_weights",
    "scores",
    "stats",
]

#: The third octaves an error is read in: their centres, in Hz.
BANDS_HZ = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250)
#: The rate everything here is at, and the radius of the sphere an error is read on.
RATE_HZ = 4000.0
HEAD_M = 0.10
#: The early part: from this long before the arrival to this long after it, and its fade.
BEFORE_S, EARLY_S, FADE_S = 0.010, 0.050, 0.005


def band_masks(samples: int, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Which bins of ``rfftfreq(samples)`` lie in each third octave: ``[band, bin]``.

    A sixth of an octave either side of the nominal centres, which are not
    quite a third of an octave apart: two neighbours share a hertz at one
    edge and leave two at another.
    """
    freqs = np.fft.rfftfreq(samples, 1.0 / rate_hz)
    edge = 2.0 ** (1.0 / 6.0)
    return np.stack([(freqs >= b / edge) & (freqs < b * edge) for b in BANDS_HZ])


def head_weights(order: int, freqs_hz: np.ndarray, radius_m: float = HEAD_M) -> np.ndarray:
    """``j_n(k a)`` of each channel at each frequency: ``[channel, frequency]``."""
    k = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) / SOUND_SPEED_M_S
    found: np.ndarray = spherical_jn(degrees_of(order)[:, None], (k * radius_m)[None, :])
    return found


def early_window(samples: int, onset: int, rate_hz: float = RATE_HZ) -> np.ndarray:
    """One from :data:`BEFORE_S` before ``onset`` to :data:`EARLY_S` after, faded at its end."""
    start = max(0, onset - round(BEFORE_S * rate_hz))
    stop = min(samples, onset + round(EARLY_S * rate_hz))
    fade = min(round(FADE_S * rate_hz), stop - start)
    window = np.zeros(samples)
    window[start:stop] = 1.0
    window[stop - fade : stop] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade) / fade))
    return window


def decibels(ratio: Any) -> Any:
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(ratio, 1e-30))


def scores(
    error: np.ndarray, truth: np.ndarray, onset: int, order: int, rate_hz: float = RATE_HZ
) -> dict[str, np.ndarray]:
    """The error over the truth, and the level the error leaves, per third octave.

    ``error`` and ``truth`` are ``[channel, sample]`` on the head's sphere
    (already weighted). Returns, in dB, for the early window and for the
    whole response: ``early`` and ``whole`` (``[band]``, all channels),
    ``early_degree`` and ``whole_degree`` (``[band, degree]``, a degree's
    error over the whole truth, so the degrees add to the total), and
    ``early_level`` and ``late_level`` (``[band]``, the level of truth plus
    error over the truth's, before and after the early window's end).
    """
    samples = int(truth.shape[-1])
    masks = band_masks(samples, rate_hz)
    degree = degrees_of(order)
    early = early_window(samples, onset, rate_hz)
    late = 1.0 - early
    late[: max(0, onset - round(BEFORE_S * rate_hz))] = 0.0
    found: dict[str, np.ndarray] = {}

    def energies(signal: np.ndarray, window: np.ndarray | None) -> np.ndarray:
        cut = signal if window is None else signal * window[None, :]
        spectrum = np.abs(np.fft.rfft(cut, axis=-1)) ** 2
        made: np.ndarray = spectrum @ masks.T.astype(float)  # [channel, band]
        return made

    for name, window in (("early", early), ("whole", None)):
        true_e, error_e = energies(truth, window), energies(error, window)
        total = true_e.sum(axis=0)
        found[name] = decibels(error_e.sum(axis=0) / total)
        by_degree = np.stack([error_e[degree == n].sum(axis=0) for n in range(order + 1)], axis=1)
        found[f"{name}_degree"] = decibels(by_degree / total[:, None])
    for name, window in (("early_level", early), ("late_level", late)):
        true_e = energies(truth, window).sum(axis=0)
        found[name] = decibels(energies(truth + error, window).sum(axis=0) / true_e)
    return found


def stats(values: np.ndarray, *, low_is_bad: bool = False) -> dict[str, list[float]]:
    """The ninth decile and the worst case over the first axis; the median beside them.

    ``low_is_bad`` reads a margin, where the worst case is the least.
    """
    table = np.asarray(values, dtype=float)
    if low_is_bad:
        return {
            "median": np.round(np.median(table, axis=0), 1).tolist(),
            "p10": np.round(np.percentile(table, 10, axis=0), 1).tolist(),
            "worst": np.round(table.min(axis=0), 1).tolist(),
        }
    return {
        "median": np.round(np.median(table, axis=0), 1).tolist(),
        "p90": np.round(np.percentile(table, 90, axis=0), 1).tolist(),
        "worst": np.round(table.max(axis=0), 1).tolist(),
    }


def level_stats(values: np.ndarray) -> dict[str, list[float]]:
    """Of level differences in dB: the largest either way, and the ninth decile of its size."""
    table = np.asarray(values, dtype=float)
    return {
        "p90_abs": np.round(np.percentile(np.abs(table), 90, axis=0), 2).tolist(),
        "worst_abs": np.round(np.abs(table).max(axis=0), 2).tolist(),
        "least": np.round(table.min(axis=0), 2).tolist(),
        "most": np.round(table.max(axis=0), 2).tolist(),
    }
