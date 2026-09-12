"""The frame conventions the app's JavaScript mirrors, pinned in Python.

The page cannot be run here, so what is pinned is the arithmetic it copies:
a camera yaw in the y-up scene becomes a head yaw in the ambisonic frame by a
quarter turn, and the field is rotated by minus that. Tested the only way a
convention can be: a source on the listener's left must be louder in the left
ear through the library's own decoder.
"""

from __future__ import annotations

import numpy as np

from reverberate.spatial.binaural import design_decoder, render
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.hrtf import sphere_head
from reverberate.spatial.sh import real_sh, scene_to_ambisonic


def head_yaw_of_camera(camera_yaw: float) -> float:
    """`headYawOfCamera` of `viz/app/audio/sh.js`."""
    return camera_yaw + np.pi / 2


def camera_forward(camera_yaw: float) -> np.ndarray:
    """`viewport.js`: forward is (-sin yaw, 0, -cos yaw) in the scene."""
    return np.array([-np.sin(camera_yaw), 0.0, -np.cos(camera_yaw)])


def test_a_source_on_the_cameras_left_is_louder_in_the_left_ear() -> None:
    rate, taps, order = 48_000.0, 256, 3
    decoder = design_decoder(
        sphere_head(rate, taps), order=order, sample_rate_hz=rate, filter_length=taps
    )
    rng = np.random.default_rng(0)
    for camera_yaw in rng.uniform(-np.pi, np.pi, 5):
        forward = camera_forward(camera_yaw)
        # Left of the camera: up (y) cross forward, in a right-handed y-up frame.
        left = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        direction = scene_to_ambisonic(left[None, :])[0]
        signals = np.zeros((16, 64))
        signals[:, 8] = real_sh(order, direction[None, :])[0]
        field = Ambisonic(signals=signals, sample_rate_hz=rate, order=order, centre=np.zeros(3))
        ears = render(field, decoder, field_yaw_rad=-head_yaw_of_camera(camera_yaw))
        assert (ears[0] ** 2).sum() > 1.5 * (ears[1] ** 2).sum(), camera_yaw


def test_a_source_straight_ahead_is_heard_alike_in_both_ears() -> None:
    rate, taps, order = 48_000.0, 256, 3
    decoder = design_decoder(
        sphere_head(rate, taps), order=order, sample_rate_hz=rate, filter_length=taps
    )
    camera_yaw = 0.7
    direction = scene_to_ambisonic(camera_forward(camera_yaw)[None, :])[0]
    signals = np.zeros((16, 64))
    signals[:, 8] = real_sh(order, direction[None, :])[0]
    field = Ambisonic(signals=signals, sample_rate_hz=rate, order=order, centre=np.zeros(3))
    ears = render(field, decoder, field_yaw_rad=-head_yaw_of_camera(camera_yaw))
    assert np.isclose((ears[0] ** 2).sum(), (ears[1] ** 2).sum(), rtol=0.05)
