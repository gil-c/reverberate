"""The moving delay line's interpolator: a Kaiser windowed sinc on a signal held at twice the rate.

A moving path reads the dry signal at a delay that is never a whole number
of samples, and its error is heard as a colouration that changes with the
movement. At 48 kHz a band to 20 kHz leaves a transition of 4 kHz, and a
windowed sinc that holds it needs some sixty taps; a linear interpolator is
9 dB down at 20 kHz half way between two samples.

The engine therefore holds the filtered dry signal at **twice the rate**,
which its filter bank's transform gives for nothing, and interpolates that.
The band then ends at 0.42 of the new Nyquist and its first image starts at
1.58: the transition is 56 kHz wide and :data:`TAPS` taps hold it. The
kernel is tabulated at :data:`PHASES` fractions of a sample and read
linearly between two of them.

:func:`worst_error` measures what the module does, tone by tone and
fraction by fraction; the figure is in the tests and in the format's
document.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

__all__ = ["OVERSAMPLE", "PHASES", "TAPS", "kernel_table", "read", "worst_error"]

#: The rate of the signal the delay line reads, over the output's.
OVERSAMPLE = 2
#: Taps of the kernel, at the oversampled rate: 0.125 ms.
TAPS = 12
#: Fractions of a sample the kernel is tabulated at.
PHASES = 1024
#: The Kaiser window's shape, the one that errs least over 1 to 20 kHz at this length:
#: -94 dB, against -86 at 10 and -75 at 12.
BETA = 11.0


@lru_cache(maxsize=1)
def kernel_table() -> np.ndarray:
    """``[PHASES + 1, TAPS]``: the kernel for a read ``phase / PHASES`` past a sample.

    Tap ``j`` weighs the sample ``j - TAPS / 2 + 1`` after the one read
    from. At phase 0 the kernel is one tap of one: a whole delay is exact.
    """
    fraction = np.arange(PHASES + 1)[:, None] / PHASES
    offsets = np.arange(TAPS)[None, :] - (TAPS // 2 - 1)
    x = offsets - fraction
    window = np.i0(BETA * np.sqrt(np.clip(1.0 - (x / (TAPS / 2)) ** 2, 0.0, None))) / np.i0(BETA)
    table = np.sinc(x) * window
    # Each row sums to one, so a constant is read as itself at every fraction.
    table /= table.sum(axis=1, keepdims=True)
    table.setflags(write=False)
    return np.asarray(table)


def read(signals: Any, position: np.ndarray, xp: Any = np) -> Any:
    """``signals[p, :, v]`` read at ``position[p, n]``, in samples of ``signals``: ``[p, n, v]``.

    ``signals`` is ``[path, sample, value]`` on ``xp``; ``position`` is on
    the host, where the geometry is. Every read must lie ``TAPS / 2`` inside
    the signal.
    """
    base = np.floor(position)
    scaled = (position - base) * PHASES
    phase = np.minimum(scaled.astype(np.int64), PHASES - 1)
    blend = xp.asarray(scaled - phase)[..., None]
    table = xp.asarray(kernel_table())
    phase_x = xp.asarray(phase)
    kernel = table[phase_x] * (1.0 - blend) + table[phase_x + 1] * blend  # [p, n, tap]
    paths, width, values = signals.shape
    first = base.astype(np.int64) - (TAPS // 2 - 1)
    if first.size and (first.min() < 0 or first.max() + TAPS > width):
        raise ValueError("a delay line was read outside the signal it holds")
    index = (np.arange(paths)[:, None] * width + first)[..., None] + np.arange(TAPS)
    taken = signals.reshape(paths * width, values)[xp.asarray(index)]  # [p, n, tap, v]
    return xp.einsum("pnt,pntv->pnv", kernel, taken)


def worst_error(
    low_hz: float = 1000.0, high_hz: float = 20000.0, rate: float = 48000.0
) -> dict[str, float]:
    """The interpolator's worst error over a band, against the exact delayed tone.

    A complex tone at the oversampled rate read at every sixteenth of a
    sample: the error's modulus over the tone's, worst over the fractions
    and over 96 frequencies of the band, as a ratio and in dB, with the
    worst change of level beside it.
    """
    freqs = np.geomspace(low_hz, high_hz, 96)
    fractions = (np.arange(16) + 0.5) / 16.0
    n = np.arange(256)
    position = (128.0 + fractions)[None, :]
    worst, level = 0.0, 0.0
    for f in freqs:
        omega = 2.0 * np.pi * f / (rate * OVERSAMPLE)
        tone = np.stack([np.cos(omega * n), np.sin(omega * n)], axis=-1)[None]
        got = read(tone, position)[0]
        got_c = got[:, 0] + 1j * got[:, 1]
        want = np.exp(1j * omega * position[0])
        worst = max(worst, float(np.max(np.abs(got_c - want))))
        level = max(level, float(np.max(np.abs(20.0 * np.log10(np.abs(got_c))))))
    return {"error": worst, "error_db": float(20.0 * np.log10(worst)), "level_db": level}
