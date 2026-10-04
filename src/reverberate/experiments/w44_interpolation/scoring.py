"""What the measurements of W44 share: the bands, the early window, the error in decibels.

An error here is the energy of the difference over the energy of the truth,
per octave, on the omnidirectional channel: 0 dB is a prediction no better
than silence, -20 dB one whose error is a tenth of the truth's amplitude.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BANDS",
    "DIRS",
    "EARLY_S",
    "FADE_S",
    "LATE_S",
    "band_energy",
    "delayed",
    "early_window",
    "error_db",
    "onset_of",
    "percentiles",
    "write_summary",
]

#: Octave band centres the errors are read in, in Hz.
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]

#: The early part runs from the onset to this long after it; the late level is
#: read from there to ``LATE_S`` after the onset. The window ends on a half
#: cosine ``FADE_S`` long. Seconds.
EARLY_S, LATE_S, FADE_S = 0.050, 0.150, 0.005

#: The four neighbours of a cell of the lattice, as steps of ``cell_index``.
DIRS: dict[str, tuple[int, int, int]] = {
    "x+": (1, 0, 0),
    "x-": (-1, 0, 0),
    "z+": (0, 0, 1),
    "z-": (0, 0, -1),
}


def onset_of(w: np.ndarray) -> int:
    """The first sample over a tenth of the response's peak."""
    return int(np.argmax(np.abs(w) > 0.1 * np.abs(w).max()))


def early_window(samples: int, stop: int, fade: int) -> np.ndarray:
    """One up to ``stop``, the last ``fade`` samples of it a half cosine down to zero."""
    win = np.zeros(samples, dtype=np.float32)
    stop = min(stop, samples)
    win[:stop] = 1.0
    f = min(fade, stop)
    win[stop - f : stop] = 0.5 * (1 + np.cos(np.pi * np.arange(f) / f))
    return win


def band_energy(x: np.ndarray, rate: float, bands: list[int] | None = None) -> np.ndarray:
    """Energy of ``x`` in each octave of ``bands`` (:data:`BANDS` by default)."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / rate)
    return np.array(
        [
            spec[(freqs >= b / np.sqrt(2)) & (freqs < b * np.sqrt(2))].sum()
            for b in (BANDS if bands is None else bands)
        ]
    )


def error_db(
    pred: np.ndarray, truth: np.ndarray, win: np.ndarray, ref: np.ndarray, rate: float
) -> list[float]:
    """The windowed error of ``pred`` against ``truth`` per octave, in dB re ``ref``.

    ``ref`` is the band energy of the windowed truth, computed once by the caller.
    """
    error: list[float] = (10 * np.log10(band_energy((pred - truth) * win, rate) / ref)).tolist()
    return error


def delayed(w: np.ndarray, delay_s: float, rate: float) -> np.ndarray:
    """``w`` later by ``delay_s``, a fraction of a sample included."""
    freqs = np.fft.rfftfreq(w.shape[-1], 1 / rate)
    shifted: np.ndarray = np.fft.irfft(
        np.fft.rfft(w) * np.exp(-2j * np.pi * freqs * delay_s), w.shape[-1]
    )
    return shifted


def percentiles(values: np.ndarray, *, worst: bool = True) -> dict[str, list[float]]:
    """Median, ninth decile and, when asked, the worst case of ``[case, band]``, per band."""
    found: dict[str, list[float]] = {
        "median": np.round(np.median(values, axis=0), 1).tolist(),
        "p90": np.round(np.percentile(values, 90, axis=0), 1).tolist(),
    }
    if worst:
        found["worst"] = np.round(values.max(axis=0), 1).tolist()
    return found


def write_summary(out: Path, name: str, summary: dict[str, Any]) -> Path:
    """The summary as indented JSON in ``out``, created when it is not there."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(summary, indent=1))
    return path
