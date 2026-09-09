"""Analytic free fields, in the conventions the encoder is tested against.

Everything is in the ``numpy.fft.rfft`` convention: analysis ``e^{-i omega t}``,
synthesis ``e^{+i omega t}``. A delay is therefore ``e^{-i omega tau}``, an
outgoing spherical wave is ``e^{-ikr} / r``, and the outgoing spherical Hankel
function is the second kind, ``h_n = j_n - i y_n``. The first kind would keep
every magnitude and flip every odd order, which is a point reflection of every
direction through the listener. The direction of arrival test is the catch.

A plane wave of unit amplitude arriving *from* ``s`` is ``e^{+ik s . r}``, and
its interior expansion in N3D harmonics is ``a_nm = i^n Y_nm(s)``; the
ambisonic signal is ``b_nm = a_nm / i^n = Y_nm(s)``, so the first channel is 1.
A monopole ``e^{-ikR} / (4 pi R)`` at ``r_s`` has ``a_nm = -(i k / 4 pi) h_n(k r_s)
Y_nm(s)`` inside the sphere ``r < r_s``, and far away it tends to the plane
wave scaled by ``e^{-i k r_s} / (4 pi r_s)``, which is the unit test.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from reverberate.spatial.sh import channel_count, degrees_of, real_sh

__all__ = [
    "interior_pressure",
    "monopole_coefficients",
    "monopole_pressure",
    "plane_wave_coefficients",
    "plane_wave_pressure",
    "spherical_hankel2",
]


def spherical_hankel2(n: np.ndarray | int, x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """``h_n^(2)(x) = j_n(x) - i y_n(x)``, the outgoing wave of this convention."""
    j = np.asarray(spherical_jn(n, x, derivative=derivative), dtype=float)
    y = np.asarray(spherical_yn(n, x, derivative=derivative), dtype=float)
    return np.asarray(j - 1j * y, dtype=complex)


def plane_wave_pressure(direction: np.ndarray, points: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``e^{+ik s . r}`` at ``points``, ``[k, point]``: unit plane wave arriving from ``s``."""
    s = np.asarray(direction, dtype=float)
    s = s / np.linalg.norm(s)
    projection = np.atleast_2d(np.asarray(points, dtype=float)) @ s
    kk = np.atleast_1d(np.asarray(k, dtype=float))[:, None]
    return np.asarray(np.exp(1j * kk * projection[None, :]), dtype=complex)


def monopole_pressure(source: np.ndarray, points: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``e^{-ikR} / (4 pi R)`` at ``points``, ``[k, point]``."""
    distance = np.linalg.norm(
        np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(source), axis=1
    )
    if np.any(distance == 0.0):
        raise ValueError("a point coincides with the source")
    kk = np.atleast_1d(np.asarray(k, dtype=float))[:, None]
    return np.asarray(
        np.exp(-1j * kk * distance[None, :]) / (4.0 * np.pi * distance[None, :]), dtype=complex
    )


def plane_wave_coefficients(direction: np.ndarray, order: int) -> np.ndarray:
    """Interior coefficients ``a_nm = i^n Y_nm(s)`` of a unit plane wave from ``s``."""
    basis = real_sh(order, np.asarray(direction, dtype=float)[None, :])[0]
    return np.asarray(1j ** degrees_of(order) * basis, dtype=complex)


def monopole_coefficients(source: np.ndarray, k: np.ndarray, order: int) -> np.ndarray:
    """Interior coefficients of a monopole at ``source``, ``[k, channel]``.

    Valid inside the sphere ``r < |source|``.
    """
    source = np.asarray(source, dtype=float)
    r_s = float(np.linalg.norm(source))
    if r_s == 0.0:
        raise ValueError("the source cannot sit at the expansion centre")
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    n = degrees_of(order)
    hankel = spherical_hankel2(n[None, :], kk[:, None] * r_s)
    basis = real_sh(order, source[None, :])[0]
    return np.asarray(-(1j * kk[:, None] / (4.0 * np.pi)) * hankel * basis[None, :], dtype=complex)


def interior_pressure(
    coefficients: np.ndarray, points: np.ndarray, k: np.ndarray, order: int
) -> np.ndarray:
    """Synthesise ``sum a_nm j_n(kr) Y_nm`` at ``points``, ``[k, point]``."""
    coefficients = np.atleast_2d(np.asarray(coefficients))
    if coefficients.shape[1] != channel_count(order):
        raise ValueError(f"{coefficients.shape[1]} channels do not match order {order}")
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    radius = np.linalg.norm(pts, axis=1)
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    n = degrees_of(order)
    # At the centre the direction is undefined; only n = 0 survives there, so
    # any direction serves.
    safe = np.where(radius[:, None] > 0.0, pts, np.array([[1.0, 0.0, 0.0]]))
    basis = real_sh(order, safe)  # [point, channel]
    radial = np.asarray(
        spherical_jn(n[None, None, :], kk[:, None, None] * radius[None, :, None]), dtype=float
    )
    return np.asarray(np.einsum("kc,kpc,pc->kp", coefficients, radial, basis), dtype=complex)
