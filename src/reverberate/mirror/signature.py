"""The source signature: the reference's own direct pulse, given to the mirror.

The wave solver drives its source with a pulse of its own and its grid
disperses what it carries above the band it is trusted to; the reference
field's direct sound is therefore not flat. The mirror renders every path
as a flat band limited pulse. What the ear hears first as a difference is
that spectrum, so the mirror is given the reference's: the median magnitude
spectrum of the direct pulse over a few dozen points with a clean direct
sound, turned into a short minimum phase filter and convolved into every
response. It is a property of the source, not of the room, and is written
beside the field it was read from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import minimum_phase

from reverberate.mirror.criteria import CriteriaSettings, direct_arrival

__all__ = ["apply_signature", "load_signature", "measure_signature", "write_signature"]


def measure_signature(
    reference: Path,
    indices: list[int],
    *,
    window_s: float = 0.002,
    taps: int = 128,
    settings: CriteriaSettings | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The minimum phase filter with the reference's median direct spectrum, and its record.

    The direct pulse is cut over ``window_s`` around its arrival on the omni
    channel, its magnitude spectrum is normalised at the band's peak, the
    median over the points is taken, and a minimum phase filter of ``taps``
    is fitted to it. The filter has unit energy at its peak band so the
    mirror's alignment gain still means what it meant.
    """
    settings = settings or CriteriaSettings()
    spectra = []
    used = []
    with h5py.File(reference, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        half = int(round(window_s * rate / 2))
        n_fft = 4096
        for index in indices:
            omni = np.asarray(handle["ir"][index][0], dtype=float)
            try:
                start = direct_arrival(omni[None, :], rate, settings)
            except ValueError:
                continue
            cut = omni[max(start - half // 2, 0) : start + half + half // 2]
            if cut.size < 8 or float(np.max(np.abs(cut))) == 0.0:
                continue
            magnitude = np.abs(np.fft.rfft(cut * np.hanning(cut.size), n_fft))
            spectra.append(magnitude / magnitude.max())
            used.append(int(index))
    if not spectra:
        raise ValueError("no point with a clean direct sound to read the signature from")
    median = np.median(np.stack(spectra), axis=0)
    # A linear phase prototype with that magnitude, then its minimum phase version.
    prototype = np.fft.irfft(median, n_fft)
    prototype = np.roll(prototype, n_fft // 2)
    window = np.hanning(taps * 2 + 1)
    centre = n_fft // 2
    linear = prototype[centre - taps : centre + taps + 1] * window
    filt = minimum_phase(linear / np.sum(np.abs(linear)), method="homomorphic", n_fft=n_fft * 4)
    filt = filt[:taps]
    filt = filt / np.sqrt(np.sum(filt**2))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / rate)
    response = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(filt, n_fft)), 1e-9))
    checkpoints = [500, 1000, 2000, 4000, 8000, 12000, 16000, 20000]
    record = {
        "kind": "minimum phase filter with the reference's median direct spectrum",
        "points": used,
        "window_s": window_s,
        "taps": int(taps),
        "sample_rate_hz": rate,
        "median_db": {
            str(f): round(
                float(20.0 * np.log10(max(median[np.argmin(np.abs(frequencies - f))], 1e-9))), 2
            )
            for f in checkpoints
        },
        "filter_db": {
            str(f): round(float(response[np.argmin(np.abs(frequencies - f))]), 2)
            for f in checkpoints
        },
    }
    return filt, record


def apply_signature(signals: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Every channel convolved with the signature, same length."""
    if taps.size == 0:
        return signals
    n = signals.shape[-1]
    n_fft = 1 << int(np.ceil(np.log2(n + taps.size)))
    spectrum = np.fft.rfft(signals, n_fft, axis=-1) * np.fft.rfft(taps, n_fft)[None, :]
    return np.asarray(np.fft.irfft(spectrum, n_fft, axis=-1)[..., :n])


def write_signature(target: Path, taps: np.ndarray, record: dict[str, Any]) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target.with_suffix(".npy"), taps)
    target.with_suffix(".json").write_text(json.dumps(record, indent=1))
    return target.with_suffix(".npy")


def load_signature(target: Path) -> np.ndarray:
    return np.asarray(np.load(Path(target).with_suffix(".npy")), dtype=float)
