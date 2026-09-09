"""The spherical harmonic conventions, each locked against its silent alternative.

A wrong sign or a stray Condon-Shortley phase leaves every magnitude intact and
mirrors every direction, so the tests here check directions and orders, never
levels alone. Synthetic, offline, well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from reverberate.response import to_sofa_coordinates
from reverberate.spatial.sh import (
    acn,
    analyse,
    channel_count,
    degrees_of,
    directions,
    orders_of,
    quadrature,
    real_sh,
    rotate_yaw,
    scene_to_ambisonic,
    sn3d_scale,
)


def test_the_first_order_channels_are_root_three_times_the_direction_cosines() -> None:
    """ACN 1, 2, 3 are y, z, x, which is where a Condon-Shortley phase would show."""
    v = np.array([[0.3, -0.5, 0.812403840463596]])
    basis = real_sh(1, v)[0]
    assert basis[0] == pytest.approx(1.0)
    assert basis[acn(1, -1)] == pytest.approx(np.sqrt(3.0) * v[0, 1])
    assert basis[acn(1, 0)] == pytest.approx(np.sqrt(3.0) * v[0, 2])
    assert basis[acn(1, 1)] == pytest.approx(np.sqrt(3.0) * v[0, 0])


def test_the_harmonics_are_orthogonal_with_norm_four_pi_on_the_quadrature() -> None:
    """N3D: ``Y_00 = 1`` and every channel integrates its square to ``4 pi``."""
    order = 9
    grid, weights = quadrature(2 * order)
    basis = real_sh(order, grid)
    gram = basis.T @ (basis * weights[:, None])
    assert np.allclose(gram, 4.0 * np.pi * np.eye(channel_count(order)), atol=1e-9)
    assert weights.sum() == pytest.approx(4.0 * np.pi)


def test_analysis_inverts_synthesis() -> None:
    order = 5
    grid, weights = quadrature(2 * order)
    rng = np.random.default_rng(3)
    coefficients = rng.standard_normal(channel_count(order))
    values = real_sh(order, grid) @ coefficients
    assert np.allclose(analyse(values, grid, weights, order), coefficients)


def test_the_second_order_channel_indices_follow_acn() -> None:
    assert acn(2, -2) == 4
    assert acn(2, 0) == 6
    assert acn(2, 2) == 8
    assert list(degrees_of(2)) == [0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert list(orders_of(2)) == [0, -1, 0, 1, -2, -1, 0, 1, 2]
    with pytest.raises(ValueError):
        acn(1, 2)


def test_sn3d_divides_each_degree_by_root_two_n_plus_one() -> None:
    scale = sn3d_scale(2)
    assert scale[0] == pytest.approx(1.0)
    assert scale[acn(1, 0)] == pytest.approx(1.0 / np.sqrt(3.0))
    assert scale[acn(2, 1)] == pytest.approx(1.0 / np.sqrt(5.0))


def test_a_point_above_the_listener_in_the_scene_lands_in_the_z_channel() -> None:
    """The scene is Y up. The frame change is the SOFA one and nothing else."""
    above = np.array([[0.0, 1.0, 0.0]])
    assert np.allclose(scene_to_ambisonic(above), to_sofa_coordinates(above))
    basis = real_sh(1, scene_to_ambisonic(above))[0]
    assert basis[acn(1, 0)] == pytest.approx(np.sqrt(3.0))
    assert basis[acn(1, 1)] == pytest.approx(0.0, abs=1e-12)
    # Scene -z is the ambisonic left, +y.
    left = real_sh(1, scene_to_ambisonic(np.array([[0.0, 0.0, -1.0]])))[0]
    assert left[acn(1, -1)] == pytest.approx(np.sqrt(3.0))


def test_azimuth_runs_counter_clockwise_from_the_front() -> None:
    front = directions(np.array(0.0), np.array(0.0))
    left = directions(np.array(np.pi / 2), np.array(0.0))
    up = directions(np.array(0.0), np.array(np.pi / 2))
    assert np.allclose(front, [1.0, 0.0, 0.0])
    assert np.allclose(left, [0.0, 1.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])


def test_rotating_the_field_moves_a_source_by_the_same_yaw() -> None:
    """A source at azimuth phi rotated by psi is a source at phi + psi, every order."""
    order = 4
    phi, psi = 0.4, 1.1
    source = real_sh(order, directions(np.array(phi), np.array(0.2)))[0]
    moved = real_sh(order, directions(np.array(phi + psi), np.array(0.2)))[0]
    assert np.allclose(rotate_yaw(source, psi, order), moved)


def test_yaw_rotation_is_orthogonal_and_undone_by_the_opposite_angle() -> None:
    order = 3
    rng = np.random.default_rng(1)
    c = rng.standard_normal((5, channel_count(order)))
    turned = rotate_yaw(c, 0.7, order)
    assert np.allclose(np.linalg.norm(turned, axis=1), np.linalg.norm(c, axis=1))
    assert np.allclose(rotate_yaw(turned, -0.7, order), c)
    with pytest.raises(ValueError, match="channels"):
        rotate_yaw(c, 0.1, order + 1)


def test_a_zero_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="zero"):
        real_sh(1, np.zeros((1, 3)))
