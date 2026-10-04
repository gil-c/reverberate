"""One counter based generator, the same bits on the host and on a card.

The tail of ``mirror.render`` draws its noise from ``numpy``'s generator on
the host and from ``cupy``'s on a card, and the two do not agree: its render
is equal between devices "in law only" (ADR 0014). The signal engine draws
from this module instead.

The generator is Threefry 2x32 with twenty rounds (Salmon et al., *Parallel
random numbers: as easy as 1, 2, 3*, 2011), the block cipher behind
Random123 and JAX: additions, rotations and exclusive ors of 32 bit words,
which every array library does alike. A value is a function of
``(seed, stream, index)`` and of nothing else, so any slice of any stream is
drawn directly, on any device, in any order.

What turns the words into numbers is exact too. A uniform is the word plus
a half over ``2^32``. A carrier sample is the sum of the eight bytes of the
two words, centred and scaled to unit variance: an Irwin-Hall variate,
normal to an excess kurtosis of -0.15 and bounded at 4.9 standard
deviations. A logarithm and a cosine would make it exactly normal and would
round differently on a card; the tail normalises every burst to the energy
its bin holds, so the carrier's law is not heard and its bits are what
matter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["carrier", "threefry2x32", "uniform", "words"]

_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
_PARITY = 0x1BD11BDA
_MASK = 0xFFFFFFFF
#: Mean and standard deviation of the sum of eight bytes drawn uniformly.
_BYTES_MEAN = 8 * 127.5
_BYTES_SCALE = 1.0 / float(np.sqrt(8.0 * (256.0**2 - 1.0) / 12.0))


def _rotate(x: Any, bits: int, xp: Any) -> Any:
    # The shifts are 32 bit words too: a bare integer would widen the result on a card.
    u32 = xp.uint32
    return xp.bitwise_or(xp.left_shift(x, u32(bits)), xp.right_shift(x, u32(32 - bits)))


def threefry2x32(key: tuple[int, int], x0: Any, x1: Any, xp: Any = np) -> tuple[Any, Any]:
    """Threefry 2x32, twenty rounds: the counter ``(x0, x1)`` under ``key``, as two words."""
    u32 = xp.uint32
    ks = (key[0] & _MASK, key[1] & _MASK, (key[0] ^ key[1] ^ _PARITY) & _MASK)
    x0 = xp.asarray(x0, dtype=u32) + u32(ks[0])
    x1 = xp.asarray(x1, dtype=u32) + u32(ks[1])
    for group in range(5):
        for bits in _ROTATIONS[group % 2]:
            x0 = x0 + x1
            x1 = xp.bitwise_xor(_rotate(x1, bits, xp), x0)
        x0 = x0 + u32(ks[(group + 1) % 3])
        x1 = x1 + u32((ks[(group + 2) % 3] + group + 1) & _MASK)
    return x0, x1


def words(seed: int, stream: Any, start: int, count: int, xp: Any = np) -> tuple[Any, Any]:
    """Two words for each of ``count`` indices from ``start``, per stream: ``[stream, count]``.

    The key is the seed's two halves; the counter is the index and the
    stream. An index runs to ``2^32``: twenty four hours at 48 kHz.
    """
    if start < 0 or start + count > 1 << 32:
        raise ValueError("a stream holds 2^32 values")
    key = (int(seed) & _MASK, (int(seed) >> 32) & _MASK)
    streams = xp.asarray(np.atleast_1d(np.asarray(stream, dtype=np.uint32)))
    index = xp.arange(start, start + count, dtype=xp.uint32)
    x0 = xp.broadcast_to(index[None, :], (streams.shape[0], count))
    x1 = xp.broadcast_to(streams[:, None], (streams.shape[0], count))
    with np.errstate(over="ignore"):
        return threefry2x32(key, x0, x1, xp)


def uniform(seed: int, stream: Any, start: int, count: int, xp: Any = np) -> Any:
    """Uniforms strictly inside ``(0, 1)``, float64, ``[stream, count]``."""
    w0, _ = words(seed, stream, start, count, xp)
    return (w0.astype(xp.float64) + 0.5) * (1.0 / 4294967296.0)


def carrier(seed: int, stream: Any, start: int, count: int, xp: Any = np) -> Any:
    """Zero mean, unit variance noise, float64, ``[stream, count]``; see the module."""
    w0, w1 = words(seed, stream, start, count, xp)
    total = xp.zeros(w0.shape, dtype=xp.uint32)
    for word in (w0, w1):
        for shift in (0, 8, 16, 24):
            byte = xp.right_shift(word, xp.uint32(shift))
            total = total + xp.bitwise_and(byte, xp.uint32(0xFF))
    return (total.astype(xp.float64) - _BYTES_MEAN) * _BYTES_SCALE
