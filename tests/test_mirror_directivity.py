"""The directivity tables: level with their axis, the pack's shape, and the gain a path takes.

A model is its clip ahead, 0 dB in every band, and radiates less than the
omnidirectional source by its directivity index; normalised the old way it
radiates what that source does. The voice is loudest ahead and loses more
behind as the band rises; and a path's gain is read at the angle between
its departure and where the source faces.
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
    radiated_db,
    voice_v1,
)


def test_a_model_is_level_with_its_axis_and_radiates_its_index_less() -> None:
    """A clip is its talker's axis: 0 dB ahead in every band, and less everywhere else."""
    voice = voice_v1()
    assert voice.normalised == "axis"
    assert voice.gain_db.shape == (len(OCTAVE_BANDS), 37)
    np.testing.assert_array_equal(voice.gain_db[:, 0], 0.0)
    assert np.all(voice.gain_db <= 0.0)
    # What it radiates, against the omnidirectional source: the audit's seven figures.
    np.testing.assert_allclose(
        radiated_db(voice.gain_db), [-0.96, -1.41, -2.26, -3.04, -4.08, -5.26, -6.25], atol=0.01
    )
    # The same pattern either way, and the same key: a recipe drawn before names it still.
    mean = voice_v1("mean")
    np.testing.assert_allclose(mean.gain_db - mean.gain_db[:, :1], voice.gain_db, atol=1e-12)
    assert mean.digest == voice.digest
    assert mean.digest == hashlib.sha256(mean.pack()["gain_db"].tobytes()).hexdigest()
    assert voice.pack()["normalised"] == "axis" and voice.pack()["pattern_digest"] == voice.digest
    np.testing.assert_array_equal(omni().gain_db, 0.0)
    np.testing.assert_allclose(radiated_db(omni().gain_db), 0.0, atol=1e-12)


def test_a_model_of_unit_mean_power_is_the_pack_s_table_as_it_was() -> None:
    voice = voice_v1("mean")
    assert voice.normalised == "mean"
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
    shifted = model("shifted", voice.gain_db + 7.0, normalised="mean")
    np.testing.assert_allclose(shifted.gain_db, voice.gain_db, atol=1e-12)
    np.testing.assert_allclose(
        model("shifted", voice.gain_db + 7.0).gain_db, voice_v1().gain_db, atol=1e-12
    )


def test_the_voice_is_loudest_ahead_and_duller_behind() -> None:
    voice = voice_v1()
    assert np.all(np.diff(voice.gain_db, axis=1) <= 1e-12)
    back = voice.gain_db[:, 0] - voice.gain_db[:, -1]
    np.testing.assert_allclose(back, VOICE_V1_BACK_DB, atol=1e-9)
    assert np.all(np.diff(back) > 0.0)
    # Of unit mean power, a directional source is louder ahead than the omni it has the
    # power of: the 3 dB step at the crossover the audit found (D3).
    mean = voice_v1("mean")
    assert np.all(mean.gain_db[:, 0] > 0.0) and np.all(np.diff(mean.gain_db[:, 0]) > 0.0)


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
