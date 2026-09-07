"""What comes out of the spatial chain: the measurements, and the files.

The measurements are checked against fields whose direction is known, because
a mirrored frame is invisible in any spectrum. The files are checked by being
read back, because a convention that nobody exercises is a convention nobody
checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reverberate.response import Provenance, from_sofa_coordinates
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.export import (
    to_ambix,
    write_ambisonic_sofa,
    write_ambix_wav,
    write_brir_sofa,
)
from reverberate.spatial.sh import acn, channel_count, directions, real_sh
from reverberate.spatial.validate import (
    direct_arrival_sample,
    direction_of_arrival,
    direction_to_scene_point,
    energy_per_order,
)

RATE = 48000.0


def a_plane_wave(azimuth_deg: float, elevation_deg: float, order: int = 3) -> Ambisonic:
    direction = directions(np.array(np.radians(azimuth_deg)), np.array(np.radians(elevation_deg)))
    signals = np.zeros((channel_count(order), 512))
    signals[:, 100] = real_sh(order, direction[None, :])[0]
    return Ambisonic(signals, RATE, order, np.zeros(3))


@pytest.mark.parametrize(
    ("azimuth", "elevation"), [(0.0, 0.0), (90.0, 0.0), (-45.0, 30.0), (180.0, 0.0)]
)
def test_the_direction_of_arrival_is_the_direction_the_wave_came_from(
    azimuth: float, elevation: float
) -> None:
    """Exact for a single plane wave, which is what makes it a frame check."""
    measured = direction_of_arrival(a_plane_wave(azimuth, elevation))
    assert measured.azimuth_deg == pytest.approx(azimuth, abs=0.01)
    assert measured.elevation_deg == pytest.approx(elevation, abs=0.01)
    assert measured.concentration == pytest.approx(1.0, abs=0.01)


def test_a_diffuse_field_has_no_direction_and_says_so() -> None:
    rng = np.random.default_rng(0)
    diffuse = Ambisonic(rng.standard_normal((16, 4096)), RATE, 3, np.zeros(3))
    assert direction_of_arrival(diffuse, start=0, length=4096).concentration < 0.1


def test_the_direct_arrival_is_found_on_the_omnidirectional_channel() -> None:
    assert direct_arrival_sample(a_plane_wave(0.0, 0.0)) == 100


def test_a_scene_point_becomes_a_direction_in_the_one_frame_conversion() -> None:
    """The scene is Y up, so its ``-z`` is the listener's left."""
    ambisonic = Ambisonic(np.zeros((16, 8)), RATE, 3, np.array([0.0, 1.2, 0.0]))
    left = direction_to_scene_point(ambisonic, np.array([0.0, 1.2, -1.0]))
    assert np.allclose(left, [0.0, 1.0, 0.0])
    above = direction_to_scene_point(ambisonic, np.array([0.0, 2.2, 0.0]))
    assert np.allclose(above, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="centre"):
        direction_to_scene_point(ambisonic, np.array([0.0, 1.2, 0.0]))


def test_a_measured_direction_can_be_compared_with_the_geometry() -> None:
    source = np.array([1.0, 1.2, -1.0])
    ambisonic = a_plane_wave(45.0, 0.0)
    reference = direction_to_scene_point(
        Ambisonic(ambisonic.signals, RATE, 3, np.array([0.0, 1.2, 0.0])), source
    )
    assert direction_of_arrival(ambisonic).angle_to_deg(reference) < 0.01


def test_a_plane_wave_puts_no_energy_where_it_should_not() -> None:
    """A wave from straight ahead has nothing in the left or the up channel."""
    signals = a_plane_wave(0.0, 0.0).signals
    assert abs(signals[acn(1, 1), 100]) == pytest.approx(np.sqrt(3.0))
    assert abs(signals[acn(1, -1), 100]) < 1e-12
    assert abs(signals[acn(1, 0), 100]) < 1e-12


def test_a_diffuse_field_carries_the_same_energy_in_every_order() -> None:
    """The property that makes ``energy_per_order`` a diffuseness measure."""
    rng = np.random.default_rng(1)
    ambisonic = Ambisonic(rng.standard_normal((16, 24000)), RATE, 3, np.zeros(3))
    _, energy = energy_per_order(ambisonic, frame_s=0.05, hop_s=0.05)
    per_channel = energy / np.array([1, 3, 5, 7])
    assert np.allclose(per_channel.mean(axis=0) / per_channel.mean(), 1.0, atol=0.1)


def test_ambix_divides_each_degree_by_root_two_n_plus_one() -> None:
    ambisonic = Ambisonic(np.ones((16, 4)), RATE, 3, np.zeros(3))
    converted = to_ambix(ambisonic)
    assert converted[0, 0] == pytest.approx(1.0)
    assert converted[acn(1, 0), 0] == pytest.approx(1.0 / np.sqrt(3.0))
    assert converted[acn(3, 2), 0] == pytest.approx(1.0 / np.sqrt(7.0))


def test_the_ambix_wav_has_every_channel_and_one_shared_gain(tmp_path: Path) -> None:
    """Scaling channels separately would destroy the direction, which is the ratios."""
    soundfile = pytest.importorskip("soundfile")
    rng = np.random.default_rng(2)
    ambisonic = Ambisonic(rng.standard_normal((64, 256)), RATE, 7, np.zeros(3))
    path, gain = write_ambix_wav(ambisonic, tmp_path / "a.wav")
    samples, rate = soundfile.read(str(path))
    assert rate == int(RATE)
    assert samples.shape == (256, 64)
    assert np.allclose(samples.T, to_ambix(ambisonic) * gain, atol=1e-6)
    assert np.max(np.abs(samples)) == pytest.approx(10.0 ** (-1.0 / 20.0), abs=1e-6)


def a_provenance() -> Provenance:
    return Provenance(
        scene_sha256="a" * 64,
        mats_hash="b" * 32,
        engine="cpu",
        band="high",
        fmax_hz=16000.0,
        grid_step_m=0.002,
        points_per_wavelength=10.5,
        sound_speed_m_s=343.2,
        seed=1,
        run_id="test",
    )


def test_the_ambisonic_sofa_declares_its_receivers_as_harmonics(tmp_path: Path) -> None:
    sofar = pytest.importorskip("sofar")
    rng = np.random.default_rng(3)
    ambisonic = Ambisonic(rng.standard_normal((16, 128)), RATE, 3, np.array([0.1, 1.2, -0.3]))
    path = write_ambisonic_sofa(
        ambisonic,
        tmp_path / "a.sofa",
        source_position=np.array([1.0, 1.2, 0.0]),
        provenance=a_provenance(),
        title="test",
        licence="CC BY-NC 4.0",
    )
    read = sofar.read_sofa(str(path))
    assert read.ReceiverPosition_Type == "spherical harmonics"
    assert read.Data_IR.shape == (1, 16, 128)
    assert np.allclose(read.Data_IR[0], ambisonic.signals)
    assert "ACN 3, n=1, m=1" in str(np.asarray(read.ReceiverDescriptions).ravel()[3])
    assert np.allclose(from_sofa_coordinates(read.ListenerPosition)[0], ambisonic.centre)
    assert a_provenance().to_json() == read.GLOBAL_Comment


def test_the_binaural_sofa_carries_one_measurement_per_head_orientation(
    tmp_path: Path,
) -> None:
    sofar = pytest.importorskip("sofar")
    rng = np.random.default_rng(4)
    responses = rng.standard_normal((3, 2, 64))
    path = write_brir_sofa(
        responses,
        np.array([0.0, 45.0, -90.0]),
        tmp_path / "b.sofa",
        sample_rate_hz=RATE,
        listener_position=np.array([0.0, 1.2, 0.0]),
        source_position=np.array([1.0, 1.2, 0.0]),
        ear_positions=np.array([[0.0, 0.0875, 0.0], [0.0, -0.0875, 0.0]]),
        provenance=a_provenance(),
        title="test",
        licence="CC BY-NC 4.0",
    )
    read = sofar.read_sofa(str(path))
    assert read.Data_IR.shape == (3, 2, 64)
    assert np.allclose(read.ListenerView[1], [np.sqrt(0.5), np.sqrt(0.5), 0.0])
    assert list(np.asarray(read.ReceiverDescriptions).ravel()) == ["left ear", "right ear"]


def test_a_binaural_file_with_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("sofar")
    ears = np.array([[0.0, 0.0875, 0.0], [0.0, -0.0875, 0.0]])
    with pytest.raises(ValueError, match=r"\[orientation, 2, sample\]"):
        write_brir_sofa(
            np.zeros((3, 3, 8)),
            np.zeros(3),
            tmp_path / "x.sofa",
            sample_rate_hz=RATE,
            listener_position=np.zeros(3),
            source_position=np.ones(3),
            ear_positions=ears,
            provenance=a_provenance(),
            title="test",
            licence="l",
        )
    with pytest.raises(ValueError, match="orientations"):
        write_brir_sofa(
            np.zeros((3, 2, 8)),
            np.zeros(2),
            tmp_path / "y.sofa",
            sample_rate_hz=RATE,
            listener_position=np.zeros(3),
            source_position=np.ones(3),
            ear_positions=ears,
            provenance=a_provenance(),
            title="test",
            licence="l",
        )
