"""Real spherical harmonics in the ambisonic convention, and the frame they live in.

Every sign here is a convention, and a wrong one is silent: the magnitudes of
everything downstream stay the same while every direction is mirrored or every
odd order is flipped. So the choices are written down once and each has a test
that fails on the alternative.

**Frame.** Ambisonics and SOFA agree: ``x`` front, ``y`` left, ``z`` up. The
scene is Y up with positions ``(x, height, z)``. The single rotation between
them is :func:`reverberate.response.to_sofa_coordinates`, ``(x, y, z)`` to
``(x, -z, y)``, and :func:`scene_to_ambisonic` is that function and nothing
else. Two rotations is how a dataset acquires a mirrored axis.

**Ordering and normalisation.** ACN, ``acn = n^2 + n + m``, and N3D: ``Y_00 = 1``
and ``integral of Y_nm^2 over the sphere = 4 pi``. That is the ambisonic
convention rather than the orthonormal one of physics, chosen so that the
first channel of a plane wave of unit amplitude is exactly 1. The synthesis
``f = sum c_nm Y_nm`` therefore pairs with the analysis
``c_nm = (1 / 4 pi) integral f Y_nm``, which :func:`analyse` performs on a
quadrature whose weights sum to ``4 pi``. SN3D, the ambiX convention, is
``N3D / sqrt(2 n + 1)`` and is applied on export only.

**No Condon-Shortley phase.** ``scipy.special.lpmv`` includes ``(-1)^m``; it is
multiplied back out. The check: ACN 1, 2, 3 are ``sqrt(3)`` times ``y``, ``z``,
``x``.

**Rotation.** Only yaw, about ``z``. For each ``m > 0`` the pair
``(b_{n,m}, b_{n,-m})`` rotates by the angle ``m psi``; ``m = 0`` is untouched.
The argument is the rotation of the *field*: a head turned left by ``psi``
hears the field rotated by ``-psi``.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import lpmv, roots_legendre

from reverberate.response import to_sofa_coordinates

__all__ = [
    "acn",
    "analyse",
    "channel_count",
    "degrees_of",
    "directions",
    "orders_of",
    "quadrature",
    "real_sh",
    "rotate_yaw",
    "scene_to_ambisonic",
    "sn3d_scale",
]


def channel_count(order: int) -> int:
    """``(N + 1)^2`` channels for order ``N``."""
    if order < 0:
        raise ValueError("order must be non negative")
    return (order + 1) ** 2


def acn(n: int, m: int) -> int:
    """The Ambisonic Channel Number of degree ``n`` and order ``m``."""
    if abs(m) > n:
        raise ValueError(f"|m| = {abs(m)} exceeds n = {n}")
    return n * n + n + m


def degrees_of(order: int) -> np.ndarray:
    """The degree ``n`` of every channel, in ACN order."""
    return np.array([n for n in range(order + 1) for _ in range(-n, n + 1)], dtype=int)


def orders_of(order: int) -> np.ndarray:
    """The order ``m`` of every channel, in ACN order."""
    return np.array([m for n in range(order + 1) for m in range(-n, n + 1)], dtype=int)


def sn3d_scale(order: int) -> np.ndarray:
    """Per channel factor taking N3D coefficients to SN3D, ``1 / sqrt(2n + 1)``."""
    return np.asarray(1.0 / np.sqrt(2.0 * degrees_of(order) + 1.0), dtype=float)


def scene_to_ambisonic(points: np.ndarray) -> np.ndarray:
    """Scene coordinates, Y up, to the ambisonic frame, x front, y left, z up.

    The one rotation of the project, shared with the SOFA writer.
    """
    return to_sofa_coordinates(points)


def directions(azimuth_rad: np.ndarray, elevation_rad: np.ndarray) -> np.ndarray:
    """Unit vectors from azimuth, counter clockwise from front, and elevation from horizontal."""
    az = np.asarray(azimuth_rad, dtype=float)
    el = np.asarray(elevation_rad, dtype=float)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def real_sh(order: int, unit_vectors: np.ndarray) -> np.ndarray:
    """N3D real spherical harmonics, ACN order, ``[direction, channel]``.

    ``unit_vectors`` is ``[direction, 3]`` in the ambisonic frame. They are
    normalised here, so a point on a shell can be passed directly.
    """
    vectors = np.atleast_2d(np.asarray(unit_vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("a direction cannot be the zero vector")
    vectors = vectors / norms
    cos_theta = np.clip(vectors[:, 2], -1.0, 1.0)
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])

    out = np.empty((vectors.shape[0], channel_count(order)), dtype=float)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            k = abs(m)
            # lpmv carries the Condon-Shortley phase; (-1)^k removes it.
            legendre = np.asarray(lpmv(k, n, cos_theta), dtype=float) * (-1.0) ** k
            norm = np.sqrt((2.0 * n + 1.0) * factorial(n - k) / factorial(n + k))
            if m > 0:
                angular = np.sqrt(2.0) * np.cos(k * phi)
            elif m < 0:
                angular = np.sqrt(2.0) * np.sin(k * phi)
            else:
                angular = np.ones_like(phi)
            out[:, acn(n, m)] = norm * legendre * angular
    return out


def quadrature(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Directions and weights integrating polynomials of ``degree`` exactly on the sphere.

    Gauss-Legendre in ``cos(theta)`` and a uniform ring in azimuth, so the
    product of two harmonics of order ``N`` integrates exactly when
    ``degree >= 2N``. Weights sum to ``4 pi``. No table, no download.
    """
    if degree < 0:
        raise ValueError("degree must be non negative")
    n_theta = degree // 2 + 1
    n_phi = degree + 1
    nodes, weights = roots_legendre(n_theta)
    cos_theta = np.asarray(nodes, dtype=float)
    w_theta = np.asarray(weights, dtype=float)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    x = np.outer(sin_theta, np.cos(phi)).ravel()
    y = np.outer(sin_theta, np.sin(phi)).ravel()
    z = np.repeat(cos_theta, n_phi)
    w = np.repeat(w_theta, n_phi) * (2.0 * np.pi / n_phi)
    return np.stack([x, y, z], axis=1), w


def analyse(
    values: np.ndarray, unit_vectors: np.ndarray, weights: np.ndarray, order: int
) -> np.ndarray:
    """N3D coefficients of a function sampled on a quadrature, ``[..., channel]``.

    ``values`` is ``[..., direction]``; the last axis is integrated against
    the harmonics with the ``4 pi`` of the convention divided out, so that
    ``sum c_nm Y_nm`` gives the function back.
    """
    basis = real_sh(order, unit_vectors)
    weighted = np.asarray(values) * np.asarray(weights, dtype=float)
    return np.asarray(weighted @ basis / (4.0 * np.pi))


def rotate_yaw(coefficients: np.ndarray, field_yaw_rad: float, order: int) -> np.ndarray:
    """Rotate a field about ``z`` by ``field_yaw_rad``, counter clockwise from above.

    ``coefficients`` is ``[..., channel]``. Exact for every order: real
    harmonics of order ``+m`` and ``-m`` are ``cos`` and ``sin`` of ``m phi``,
    so a rotation is a plane rotation of each pair by ``m psi``.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.shape[-1] != channel_count(order):
        raise ValueError(f"{coefficients.shape[-1]} channels do not match order {order}")
    out = np.array(coefficients, copy=True)
    for n in range(1, order + 1):
        for m in range(1, n + 1):
            c, s = np.cos(m * field_yaw_rad), np.sin(m * field_yaw_rad)
            plus, minus = acn(n, m), acn(n, -m)
            cos_part = coefficients[..., plus]
            sin_part = coefficients[..., minus]
            out[..., plus] = c * cos_part - s * sin_part
            out[..., minus] = s * cos_part + c * sin_part
    return out
