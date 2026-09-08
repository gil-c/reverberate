"""Split a response into the roadmap's three bands, and put them back together.

Section 4.2. Three solves, three grids, three domains: the whole apartment at
the bottom with the full decay, the whole apartment in the middle with a long
window, one room at the top with a short one. What reaches a listener is the
sum, and how that sum is formed is not a free choice.

**The filters are built by subtraction, never as three independent
Butterworths.** Low is a low pass; mid is the higher low pass minus the lower;
high is the complement of the higher. Written that way the three responses add
to a delayed unit impulse **exactly**, by construction rather than by tuning,
so recombination cannot put a bump or a notch at a crossover. Three separately
designed filters cannot promise that at any order.

**Linear phase, one length, one group delay.** Every filter here is the same
number of taps, so all three arrive with the same delay and the sum is
sample-accurate. The complement is that delayed impulse minus the low pass,
which is why it needs no design of its own.

**One caveat that belongs to the architecture rather than to this module.**
Each band is taken from a different solve, on a different grid. W3 measured a
response level floor of -6 dB between two runs differing only by a sub-cell
grid offset, so two bands of the same room do not correlate sample by sample.
Away from a crossover that does not matter, because each frequency comes from
exactly one run. **Inside a crossover it does**: there the sum blends two
decorrelated versions of the same physical content, and the result is neither
run's waveform. It is a real property of a three-solve architecture, it is why
section 9 says cross-band checks compare statistics and never waveforms, and
narrow crossovers keep it confined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

__all__ = [
    "DEFAULT_CROSSOVERS_HZ",
    "BandSplit",
    "recombine",
    "split_filters",
]

#: The roadmap's assumed crossovers, section 4.2, stated as an assumption
#: there and not optimised. ``w37_bands`` prices moving the upper one.
DEFAULT_CROSSOVERS_HZ = (1000.0, 4000.0)

#: Taps in every filter of the bank. Odd so the group delay is a whole number
#: of samples and the complement is an exact impulse minus a low pass. 1023 at
#: 48 kHz is 21 ms of delay and a transition band of roughly 140 Hz, which is
#: narrow enough that the decorrelated crossover of the module docstring covers
#: a small fraction of an octave.
TAPS = 1023


@dataclass(frozen=True)
class BandSplit:
    """The three band-limiting responses, and what they were built from."""

    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    crossovers_hz: tuple[float, float]
    sample_rate_hz: float
    taps: int

    @property
    def group_delay(self) -> int:
        """Samples of delay every band carries, identically."""
        return (self.taps - 1) // 2

    def record(self) -> dict[str, Any]:
        return {
            "crossovers_hz": list(self.crossovers_hz),
            "taps": self.taps,
            "group_delay_samples": self.group_delay,
            "group_delay_ms": 1000.0 * self.group_delay / self.sample_rate_hz,
            "construction": (
                "low is a low pass, mid is the higher low pass minus the lower, "
                "high is a delayed impulse minus the higher low pass, so the three "
                "sum to that impulse exactly"
            ),
        }


def split_filters(
    sample_rate_hz: float,
    crossovers_hz: tuple[float, float] = DEFAULT_CROSSOVERS_HZ,
    *,
    taps: int = TAPS,
) -> BandSplit:
    """Build the three filters, by subtraction from two low passes."""
    low_hz, high_hz = crossovers_hz
    if not 0.0 < low_hz < high_hz < sample_rate_hz / 2.0:
        raise ValueError(
            f"crossovers must rise and stay under Nyquist: got {crossovers_hz} "
            f"at {sample_rate_hz} Hz"
        )
    if taps % 2 == 0:
        raise ValueError(f"taps must be odd so the group delay is whole, got {taps}")

    lower = signal.firwin(taps, low_hz, fs=sample_rate_hz)
    upper = signal.firwin(taps, high_hz, fs=sample_rate_hz)
    impulse = np.zeros(taps)
    impulse[(taps - 1) // 2] = 1.0

    return BandSplit(
        low=lower,
        mid=upper - lower,
        high=impulse - upper,
        crossovers_hz=(low_hz, high_hz),
        sample_rate_hz=sample_rate_hz,
        taps=taps,
    )


def recombine(
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    split: BandSplit,
) -> np.ndarray:
    """Band-limit each solve's response and add them.

    ``low``, ``mid`` and ``high`` are ``(receivers, samples)`` or
    ``(samples,)`` and come from three different solves of the same room. They
    are padded to a common length first, because three runs of different
    windows have different lengths and the sum must not be cut to the shortest.

    The output carries the bank's group delay, identically in all three bands,
    so it is not removed: an impulse response that has been band-split has a
    known offset and silently shifting it would break the arrival times the
    early part exists to carry. :attr:`BandSplit.group_delay` reports it.
    """
    blocks = [np.atleast_2d(np.asarray(part, dtype=float)) for part in (low, mid, high)]
    receivers = {block.shape[0] for block in blocks}
    if len(receivers) != 1:
        raise ValueError(f"the three bands hold different receiver counts: {sorted(receivers)}")

    length = max(block.shape[1] for block in blocks)
    padded = [np.pad(block, ((0, 0), (0, length - block.shape[1]))) for block in blocks]

    out = np.zeros((padded[0].shape[0], length + split.taps - 1))
    for block, kernel in zip(padded, (split.low, split.mid, split.high), strict=True):
        for receiver in range(block.shape[0]):
            out[receiver] += np.convolve(block[receiver], kernel)

    trimmed = out[:, :length]
    return trimmed[0] if np.ndim(low) == 1 and np.ndim(mid) == 1 else trimmed
