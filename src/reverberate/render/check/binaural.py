"""The page's binaural decode, offline: the same filters, the same head, the same blocks.

The audit page decodes the engine's order 7 stream itself
(``viz/app/scene/sound-decode.js``): the decoder's filters
(:func:`reverberate.spatial.binaural.design_decoder` on the measured head,
512 taps, as :mod:`reverberate.viz.decoders` exports them) are turned by the
head's matrix, and the stream goes through them by overlap-save in blocks of
512, a new head being a linear crossfade over one block between the two
decodes of the same samples. :class:`PageDecoder` is that, in ``numpy``, so
that a file written here is what the page plays for the scene's own head.

Two things are not the page's. The head is read at the first sample of each
block from the pack's ``listener/orientation``, linear between steps; the
page reads the recipe's keyframes on its animation frame, a few milliseconds
either way. And the page fades a start in over one block, which a file does
not need.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from reverberate.spatial.binaural import BinauralDecoder, design_decoder
from reverberate.spatial.sh import channel_count, quadrature, real_sh

__all__ = ["BLOCK", "PageDecoder", "head_matrix", "page_decoder", "rotation"]

#: Samples per block of the page's decode.
BLOCK = 512
#: The decoder's length on the page (``viz.decoders.FILTER_LENGTH``).
FILTER_LENGTH = 512


def head_matrix(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """The head's orientation in the ambisonic frame: ``Rz(yaw) Ry(-pitch) Rx(roll)``.

    The recipe's convention (``docs/formats/scene-recipe.md``) and
    ``headMatrix3`` of the page. Its columns are the head's front, left and
    up.
    """
    psi, theta, rho = np.radians(yaw_deg), -np.radians(pitch_deg), np.radians(roll_deg)
    cz, sz, cy, sy, cx, sx = (
        np.cos(psi),
        np.sin(psi),
        np.cos(theta),
        np.sin(theta),
        np.cos(rho),
        np.sin(rho),
    )
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def rotation(order: int, matrix: np.ndarray) -> np.ndarray:
    """The harmonics' rotation for a head ``matrix``: ``R[i, j] = <Y_i(x), Y_j(M x)>``.

    ``R c`` is the field ``c`` in the head's frame: a plane wave from ``d``
    becomes one from ``M^T d``. By quadrature, exact, as ``rotationBlocks``
    of ``viz/app/audio/sh.js``.
    """
    points, weights = quadrature(2 * order)
    at = real_sh(order, points)
    turned = real_sh(order, points @ np.asarray(matrix, dtype=float).T)
    return np.asarray((at * weights[:, None]).T @ turned / (4.0 * np.pi))


@lru_cache(maxsize=2)
def _designed(path: str | None, order: int, rate: float) -> BinauralDecoder:
    if path is None:
        from reverberate.spatial.hrtf import sphere_head

        head = sphere_head(rate, FILTER_LENGTH)
    else:
        from reverberate.spatial.hrtf import measured_head

        head, _ = measured_head(path, rate, FILTER_LENGTH)
    return design_decoder(head, order=order, sample_rate_hz=rate, filter_length=FILTER_LENGTH)


def page_decoder(measured: Path | None, order: int = 7, rate: float = 48000.0) -> BinauralDecoder:
    """The decoder the page listens with: the measured head at ``measured``.

    With ``None`` the analytic rigid sphere stands in, which is not the
    page's head and is said so by the decoder's ``head``.
    """
    return _designed(None if measured is None else str(measured), order, float(rate))


class PageDecoder:
    """Order 7 in, two ears out, a block at a time, the head turning the filters."""

    def __init__(self, decoder: BinauralDecoder) -> None:
        if decoder.length > BLOCK + 1:
            raise ValueError(f"a decoder of {decoder.length} taps does not fit blocks of {BLOCK}")
        self.decoder = decoder
        self.order = decoder.order
        self.channels = channel_count(decoder.order)
        # As the page holds them: float32 taps, transformed on two blocks.
        taps = np.asarray(decoder.filters, dtype=np.float32).astype(float)
        self.spectra = np.fft.rfft(taps, n=2 * BLOCK, axis=-1)
        self._turned: dict[tuple[float, float, float], np.ndarray] = {}

    def filters_for(self, pose: tuple[float, float, float]) -> np.ndarray:
        """The filters turned by the head at ``pose`` (yaw, pitch, roll, degrees)."""
        key = (round(pose[0], 3), round(pose[1], 3), round(pose[2], 3))
        if key not in self._turned:
            if len(self._turned) > 256:
                self._turned.clear()
            turn = rotation(self.order, head_matrix(*key))
            # G_j = sum_i R_ij F_i: the ear is sum_i F_i (R c)_i.
            self._turned[key] = np.einsum("ij,eif->ejf", turn, self.spectra)
        return self._turned[key]

    def decode(
        self,
        signal: np.ndarray,
        pose_at: Callable[[int], tuple[float, float, float]],
        *,
        start: int = 0,
        before: np.ndarray | None = None,
    ) -> np.ndarray:
        """``signal`` (``[channel, sample]``) to two ears, ``[ear, sample]``, the same length.

        ``pose_at(n)`` is the head at sample ``n`` of the scene; ``start``
        is the scene's sample of ``signal``'s first, and ``before`` the 512
        samples that precede it (zeros when not given), so that a signal
        decoded in pieces is the signal decoded whole. The output is on the
        page's clock: late by the decoder's modelling delay.
        """
        x = np.asarray(signal, dtype=float)[: self.channels]
        count = x.shape[1]
        blocks = -(-count // BLOCK)
        padded = np.zeros((self.channels, (blocks + 1) * BLOCK))
        if before is not None:
            padded[:, :BLOCK] = np.asarray(before, dtype=float)[: self.channels, -BLOCK:]
        padded[:, BLOCK : BLOCK + count] = x
        out = np.zeros((2, blocks * BLOCK))
        ramp = (np.arange(BLOCK) + 1.0) / BLOCK
        previous = self.filters_for(pose_at(start))
        for b in range(blocks):
            window = np.fft.rfft(padded[:, b * BLOCK : (b + 2) * BLOCK], axis=-1)
            filters = self.filters_for(pose_at(start + b * BLOCK))
            made = np.fft.irfft(np.einsum("ecf,cf->ef", filters, window), n=2 * BLOCK)[:, BLOCK:]
            if filters is not previous:
                old = np.fft.irfft(np.einsum("ecf,cf->ef", previous, window), n=2 * BLOCK)
                made = old[:, BLOCK:] + (made - old[:, BLOCK:]) * ramp
                previous = filters
            out[:, b * BLOCK : (b + 1) * BLOCK] = made
        return out[:, :count]
