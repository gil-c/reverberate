"""The analytic fields the encoder is validated against, in the rfft convention.

The point of each test is the sign: the second kind Hankel function, the
``e^{-ikR}`` of an outgoing wave and ``i^n`` in the plane wave are all things
whose opposite gives the same magnitudes and mirrored directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.spatial.field import (
    interior_pressure,
    monopole_coefficients,
    monopole_pressure,
    plane_wave_coefficients,
    plane_wave_pressure,
    spherical_hankel2,
)
from reverberate.spatial.sh import quadrature


def test_the_outgoing_hankel_function_satisfies_the_wronskian() -> None:
    """``j_n h_n' - j_n' h_n = -i / x^2`` for the second kind; ``+i`` for the first."""
    from scipy.special import spherical_jn

    x = np.array([0.7, 2.3, 9.1])
    for n in range(0, 6):
        j, jd = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
        h, hd = spherical_hankel2(n, x), spherical_hankel2(n, x, derivative=True)
        assert np.allclose(j * hd - jd * h, -1j / x**2)


def test_the_plane_wave_coefficients_reproduce_the_plane_wave_inside_a_sphere() -> None:
    order = 12
    k = np.array([20.0, 60.0])
    s = np.array([0.2, -0.9, 0.38])
    grid, _ = quadrature(20)
    points = 0.05 * grid
    synthesised = interior_pressure(
        np.tile(plane_wave_coefficients(s, order), (2, 1)), points, k, order
    )
    assert np.allclose(synthesised, plane_wave_pressure(s, points, k), atol=1e-6)


def test_a_plane_wave_arriving_from_the_front_leads_in_front_of_the_listener() -> None:
    """``e^{+ik s.r}``: a point towards the source sees the wave earlier, i.e. positive phase."""
    ahead = plane_wave_pressure(
        np.array([1.0, 0.0, 0.0]), np.array([[0.1, 0.0, 0.0]]), np.array([10.0])
    )
    assert np.angle(ahead[0, 0]) == pytest.approx(1.0)


def test_the_monopole_coefficients_reproduce_the_monopole_inside_the_source_radius() -> None:
    order = 24
    k = np.array([15.0, 45.0, 90.0])
    source = np.array([0.5, -0.3, 0.2])
    grid, _ = quadrature(24)
    # kr reaches 7.2, and the series needs a margin of about ten orders above it.
    points = 0.08 * grid
    synthesised = interior_pressure(monopole_coefficients(source, k, order), points, k, order)
    reference = monopole_pressure(source, points, k)
    assert np.allclose(synthesised, reference, rtol=1e-5, atol=1e-7)


def test_far_from_the_source_the_monopole_becomes_a_plane_wave() -> None:
    order = 6
    k = np.array([100.0])
    direction = np.array([0.3, 0.4, 0.866])
    direction = direction / np.linalg.norm(direction)
    distance = 400.0
    near = monopole_coefficients(distance * direction, k, order)[0]
    plane = (
        plane_wave_coefficients(direction, order)
        * np.exp(-1j * k[0] * distance)
        / (4.0 * np.pi * distance)
    )
    assert np.allclose(near, plane, rtol=1e-3, atol=1e-9)
