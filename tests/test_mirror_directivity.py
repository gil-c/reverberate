"""The directivity tables: unit mean power, the pack's shape, and the gain a path takes.

A model radiates what the omnidirectional source does, in every band; the
voice is loudest ahead and loses more behind as the band rises; and a path's
gain is read at the angle between its departure and where the source faces.
"""

from __future__ import annotations

import hashlib

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.directivity import (
    ANGLES_DEG,
    VOICE_V1_BACK_DB,
    directivity_gain,
    facing,
    mean_power,
    model,
    omni,
    voice_v1,
)


def test_a_model_is_the_pack_s_table_and_radiates_unit_mean_power() -> None:
    voice = voice_v1()
    assert voice.gain_db.shape == (len(OCTAVE_BANDS), 37)
    np.testing.assert_array_equal(voice.angles_deg, np.arange(0, 181, 5))
    np.testing.assert_allclose(mean_power(voice.gain_db), 1.0, atol=1e-12)
    np.testing.assert_array_equal(omni().gain_db, 0.0)
    packed = voice.pack()
    assert packed["gain_db"].dtype == np.float32
    assert voice.digest == hashlib.sha256(packed["gain_db"].tobytes()).hexdigest()
    assert voice.digest != omni().digest
    # The same power from points on the sphere as from the table's own sum.
    rng = np.random.default_rng(0)
    points = rng.normal(size=(200_000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    power = (directivity_gain(voice, points, 30.0) ** 2).mean(axis=0)
    np.testing.assert_allclose(power, 1.0, atol=0.02)
    # A pattern given at any level comes out at the same one.
    shifted = model("shifted", voice.gain_db + 7.0)
    np.testing.assert_allclose(shifted.gain_db, voice.gain_db, atol=1e-12)


def test_the_voice_is_loudest_ahead_and_duller_behind() -> None:
    voice = voice_v1()
    assert np.all(np.diff(voice.gain_db, axis=1) <= 1e-12)
    back = voice.gain_db[:, 0] - voice.gain_db[:, -1]
    np.testing.assert_allclose(back, VOICE_V1_BACK_DB, atol=1e-9)
    assert np.all(np.diff(back) > 0.0)
    # On the axis a directional source is louder than the omni it has the power of.
    assert np.all(voice.gain_db[:, 0] > 0.0) and np.all(np.diff(voice.gain_db[:, 0]) > 0.0)


def test_a_path_s_gain_is_read_at_its_angle_from_the_facing() -> None:
    voice = voice_v1()
    np.testing.assert_allclose(facing(0.0), [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(facing(90.0), [0.0, 0.0, -1.0], atol=1e-15)
    ahead = directivity_gain(voice, np.array([[0.0, 0.0, -1.0]]), 90.0)[0]
    np.testing.assert_allclose(20.0 * np.log10(ahead), voice.gain_db[:, 0], atol=1e-9)
    behind = directivity_gain(voice, np.array([[0.0, 0.0, 2.0]]), 90.0)[0]
    np.testing.assert_allclose(20.0 * np.log10(behind), voice.gain_db[:, -1], atol=1e-9)
    # Overhead is at 90 degrees whatever the yaw: a figure of revolution.
    up = directivity_gain(voice, np.array([[0.0, 1.0, 0.0]] * 2), np.array([10.0, 250.0]))
    np.testing.assert_allclose(
        20.0 * np.log10(up), np.tile(voice.gain_db[:, 18], (2, 1)), atol=1e-9
    )
    # Between two angles of the table, linear in decibels.
    yaw = 0.0
    at = np.radians(12.0)
    between = directivity_gain(voice, np.array([[np.cos(at), 0.0, np.sin(at)]]), yaw)[0]
    want = 0.6 * voice.gain_db[:, 2] + 0.4 * voice.gain_db[:, 3]
    np.testing.assert_allclose(20.0 * np.log10(between), want, atol=1e-9)
    np.testing.assert_array_equal(directivity_gain(omni(), np.eye(3), 45.0), 1.0)
    assert ANGLES_DEG.size == 37
