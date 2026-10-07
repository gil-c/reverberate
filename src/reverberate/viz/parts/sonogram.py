"""A signal over time and frequency, on a scale that does not move; and a source's level.

A sonogram is read against another: the same instant, the same band, the
same colour for the same level. So nothing here is normalised to what a
signal holds. A cell is the mean square of the signal in its band over a
window of 43 ms, in dB re a full scale of one: a sine of full scale reads
-3 dB in the band it lies in, whatever else sounds.

The bands are spaced by equal ratios from 50 Hz to 16 kHz, twelve an
octave. Each takes the transform's bins that lie in it, a bin that
straddles an edge shared by the part of it on each side, so the bands sum to
the signal's power between the two ends and a band narrower than a bin holds
its share of that bin.

:func:`difference` is one sonogram less another, cell by cell, in dB, and
empty (``nan``) where neither holds anything: 60 dB under the loudest cell
of the two is not a level two renders differ in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "BANDS_AN_OCTAVE",
    "FLOOR_DB",
    "Sonogram",
    "band_edges",
    "difference",
    "levels_db",
    "sonogram",
]

LOW_HZ = 50.0
HIGH_HZ = 16000.0
BANDS_AN_OCTAVE = 12
#: Samples of the transform's window, a Hann: 43 ms at 48 kHz, bins 11.7 Hz apart.
WINDOW = 2048
#: Seconds between two columns.
HOP_S = 0.01
#: A cell this far under the loudest of the two sonograms holds nothing to differ in, dB.
FLOOR_DB = -60.0
#: What stands for silence in a sonogram, dB re full scale.
SILENCE_DB = -160.0


@dataclass(frozen=True)
class Sonogram:
    #: ``[column, band]`` float32, dB re a full scale of one.
    levels_db: np.ndarray
    #: The bands' edges, one more than the bands, Hz.
    edges_hz: np.ndarray
    hop_s: float

    @property
    def centres_hz(self) -> np.ndarray:
        return np.asarray(np.sqrt(self.edges_hz[:-1] * self.edges_hz[1:]))

    def describe(self) -> dict[str, float | int]:
        return {
            "columns": int(self.levels_db.shape[0]),
            "bands": int(self.levels_db.shape[1]),
            "hop_s": self.hop_s,
            "low_hz": float(self.edges_hz[0]),
            "high_hz": float(self.edges_hz[-1]),
            "bands_an_octave": BANDS_AN_OCTAVE,
        }


def band_edges(low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> np.ndarray:
    """The edges of the bands from ``low_hz``, :data:`BANDS_AN_OCTAVE` an octave, to ``high_hz``."""
    count = int(round(BANDS_AN_OCTAVE * np.log2(high_hz / low_hz)))
    return np.asarray(low_hz * 2.0 ** (np.arange(count + 1) / BANDS_AN_OCTAVE))


@lru_cache(maxsize=4)
def _shares(rate: float, window: int, low_hz: float, high_hz: float) -> np.ndarray:
    """``[band, bin]``: the share of each bin's power that lies in each band."""
    edges = band_edges(low_hz, high_hz)
    spacing = rate / window
    centre = np.arange(window // 2 + 1) * spacing
    lower, upper = centre - spacing / 2.0, centre + spacing / 2.0
    overlap = np.minimum(upper[None, :], edges[1:, None]) - np.maximum(
        lower[None, :], edges[:-1, None]
    )
    return np.asarray(np.maximum(overlap, 0.0) / spacing)


def sonogram(
    signal: np.ndarray,
    rate: float,
    *,
    hop_s: float = HOP_S,
    window: int = WINDOW,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
) -> Sonogram:
    """The sonogram of a mono ``signal``: a column every ``hop_s``, centred on its instant."""
    x = np.asarray(signal, dtype=float)
    hop = int(round(hop_s * rate))
    columns = max(int(np.ceil(x.size / hop)), 1)
    taper = np.hanning(window + 1)[:-1]
    shares = _shares(float(rate), window, low_hz, high_hz)
    padded = np.concatenate([np.zeros(window // 2), x, np.zeros(window + hop)])
    out = np.empty((columns, shares.shape[0]), dtype=np.float32)
    # Parseval: twice the one-sided bins over the window's own energy is the mean square.
    scale = 2.0 / (window * float(np.sum(taper**2)))
    piece = 2048
    for a in range(0, columns, piece):
        b = min(a + piece, columns)
        index = (np.arange(a, b) * hop)[:, None] + np.arange(window)[None, :]
        power = np.abs(np.fft.rfft(padded[index] * taper, axis=1)) ** 2 * scale
        out[a:b] = 10.0 * np.log10(np.maximum(power @ shares.T, 10.0 ** (SILENCE_DB / 10.0)))
    return Sonogram(out, band_edges(low_hz, high_hz), hop / rate)


def difference(b: Sonogram, a: Sonogram, *, floor_db: float = FLOOR_DB) -> np.ndarray:
    """``b`` less ``a``, dB, ``[column, band]``; ``nan`` where neither holds anything."""
    if b.levels_db.shape != a.levels_db.shape:
        raise ValueError("two sonograms of two shapes are not of one scene")
    top = max(float(b.levels_db.max()), float(a.levels_db.max()))
    held = (b.levels_db > top + floor_db) | (a.levels_db > top + floor_db)
    out = (b.levels_db - a.levels_db).astype(np.float32)
    out[~held] = np.nan
    return np.asarray(out)


def levels_db(signal: np.ndarray, rate: float, *, hop_s: float = 0.05) -> np.ndarray:
    """The level of a mono ``signal`` every ``hop_s``: the rms of that stretch, dB re full scale.

    What a source's light and its meter follow: a stretch of silence reads
    :data:`SILENCE_DB`.
    """
    x = np.asarray(signal, dtype=float)
    hop = int(round(hop_s * rate))
    count = max(int(np.ceil(x.size / hop)), 1)
    padded = np.zeros(count * hop)
    padded[: x.size] = x
    power = np.mean(padded.reshape(count, hop) ** 2, axis=1)
    return np.asarray(10.0 * np.log10(np.maximum(power, 10.0 ** (SILENCE_DB / 10.0))))
