"""An ambisonic expansion evaluated away from its centre, and several fused into one.

Both functions are pure: offsets, frequencies and an order in, a complex
filter per frequency and channel out. Coefficients are N3D in ACN order, as
:mod:`reverberate.spatial.sh` writes them, and offsets are in scene
coordinates, in metres. The time convention is numpy's ``rfft``, so a plane
wave arriving from the unit direction ``s`` is ``exp(+i k s . x)`` at ``x``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

from reverberate.spatial.sh import degrees_of, quadrature, real_sh, scene_to_ambisonic

__all__ = ["REGULARISATION", "SOUND_SPEED_M_S", "fusion_operator", "translation_weights"]

#: The speed of sound the fields of this experiment were solved at, in m/s.
SOUND_SPEED_M_S = 343.2

#: Tikhonov term of the fusion, relative to the mean diagonal of its Gram matrix.
REGULARISATION = 1e-3

#: Frequencies solved together in :func:`fusion_operator`; it bounds the memory.
_CHUNK = 256


def translation_weights(
    offset_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
) -> np.ndarray:
    """What multiplies an expansion's coefficients to give the pressure at an offset.

    ``p(d) = sum_c i^n j_n(k |d|) Y_c(d / |d|) B_c``: the interior expansion of
    a field free of sources within ``|d|`` of the centre, truncated at
    ``order``, so it holds to about ``k |d| = order``. ``offset_scene`` is
    ``d``, from the expansion's centre to the point wanted. Returns
    ``[frequency, channel]``, complex64.
    """
    amb = scene_to_ambisonic(offset_scene[None, :])[0]
    r = float(np.linalg.norm(amb))
    n = degrees_of(order)
    basis = real_sh(order, (amb / r)[None, :])[0]
    radial = spherical_jn(n[None, :], (2 * np.pi * freqs_hz / sound_speed_m_s * r)[:, None])
    weights: np.ndarray = ((1j**n)[None, :] * radial * basis[None, :]).astype(np.complex64)
    return weights


def fusion_operator(
    offsets_scene: np.ndarray,
    freqs_hz: np.ndarray,
    order: int,
    *,
    regularisation: float = REGULARISATION,
    sound_speed_m_s: float = SOUND_SPEED_M_S,
) -> np.ndarray:
    """What multiplies each neighbour's coefficients to give the pressure at the target.

    At one frequency a field free of sources is a sum of plane waves whose
    wave vectors lie on the sphere ``|k| = omega / c``. A plane wave from
    ``s`` of amplitude ``a_s`` at the target gives the neighbour at ``r_p``
    the coefficients ``Y_c(s) exp(i k s . r_p) a_s``. The estimator is the
    minimum norm one under a uniform prior over directions,
    ``g = 1' W A^H (A W A^H + l I)^-1``, one fixed filter per geometry; the
    target's pressure is ``g . b``.

    ``offsets_scene`` is ``[neighbour, 3]``, each neighbour's position seen
    from the target (the opposite of :func:`translation_weights`' offset).
    Returns ``[frequency, neighbour, channel]``, complex64.
    """
    dirs, w = quadrature(2 * order + 12)
    w = w / w.sum()
    basis = real_sh(order, dirs)  # [direction, channel]
    r = scene_to_ambisonic(offsets_scene)  # [neighbour, 3]
    proj = dirs @ r.T  # [direction, neighbour]
    channels = basis.shape[1]
    out = np.zeros((freqs_hz.size, r.shape[0], channels), dtype=np.complex64)
    for start in range(0, freqs_hz.size, _CHUNK):
        k = 2 * np.pi * freqs_hz[start : start + _CHUNK] / sound_speed_m_s
        phase = np.exp(1j * k[:, None, None] * proj[None, :, :])  # [f, direction, neighbour]
        a = np.einsum("fdp,dc->fpcd", phase, basis).reshape(k.size, -1, dirs.shape[0])
        aw = a * w[None, None, :]
        gram = aw @ a.conj().transpose(0, 2, 1)
        scale = np.trace(gram, axis1=1, axis2=2).real / gram.shape[1]
        gram += regularisation * scale[:, None, None] * np.eye(gram.shape[1])[None]
        rhs = aw.sum(axis=2)  # A W 1; the filter is g with g^T = 1' W A^H (gram)^-1
        g = np.linalg.solve(gram.transpose(0, 2, 1), rhs.conj()[..., None])[..., 0]
        out[start : start + _CHUNK] = g.reshape(k.size, r.shape[0], channels)
    return out
