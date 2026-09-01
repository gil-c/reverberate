"""Tests for what an impulse response is on disk.

The properties worth constraining are the ones a dataset silently loses:
**the samples must survive a round trip bit for bit**, **the coordinate
conversion must be its own inverse**, and **the provenance must arrive in the
portable file too**, since a SOFA reader outside this project has nothing else.

Offline, synthetic, well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reverberate.response import (
    SOFA_CONVENTION,
    Provenance,
    ResponseSet,
    from_sofa_coordinates,
    read_raw,
    to_sofa_coordinates,
    write_raw,
    write_sofa,
)


def a_provenance() -> Provenance:
    return Provenance(
        scene_sha256="a" * 64,
        mats_hash="b" * 32,
        engine="cpu",
        band="mid",
        fmax_hz=4000.0,
        grid_step_m=0.00816,
        points_per_wavelength=10.5,
        sound_speed_m_s=343.0,
        seed=20,
        run_id="w20_first_listen",
        notes="two bare omnidirectional points, this is not a binaural response",
    )


def a_response(receivers: int = 6, samples: int = 512) -> ResponseSet:
    rng = np.random.default_rng(0)
    return ResponseSet(
        ir=rng.standard_normal((receivers, samples)),
        sample_rate_hz=72_400.0,
        source_position=np.array([1.0, 1.5, 2.0]),
        receiver_positions=rng.uniform(0.5, 3.0, size=(receivers, 3)),
        provenance=a_provenance(),
        room_volume_m3=38.1,
    )


def test_raw_round_trip_is_bit_exact(tmp_path: Path) -> None:
    original = a_response()
    write_raw(original, tmp_path / "response.h5")
    back = read_raw(tmp_path / "response.h5")
    assert np.array_equal(back.ir, original.ir)
    assert np.array_equal(back.receiver_positions, original.receiver_positions)
    assert np.array_equal(back.source_position, original.source_position)
    assert back.sample_rate_hz == original.sample_rate_hz
    assert back.provenance == original.provenance
    assert back.room_volume_m3 == original.room_volume_m3


def test_duration_is_samples_over_rate() -> None:
    response = a_response(samples=72_400)
    assert response.duration_s == pytest.approx(1.0)


def test_a_mismatched_receiver_count_is_refused() -> None:
    with pytest.raises(ValueError, match="receiver positions"):
        ResponseSet(
            ir=np.zeros((6, 10)),
            sample_rate_hz=48_000.0,
            source_position=np.zeros(3),
            receiver_positions=np.zeros((5, 3)),
            provenance=a_provenance(),
        )


def test_a_one_dimensional_response_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[receiver, sample\]"):
        ResponseSet(
            ir=np.zeros(10),
            sample_rate_hz=48_000.0,
            source_position=np.zeros(3),
            receiver_positions=np.zeros((1, 3)),
            provenance=a_provenance(),
        )


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(-50, 50, allow_nan=False),
            st.floats(-50, 50, allow_nan=False),
            st.floats(-50, 50, allow_nan=False),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_the_coordinate_conversion_is_its_own_inverse(points: list[tuple[float, ...]]) -> None:
    array = np.array(points, dtype=float)
    assert np.allclose(from_sofa_coordinates(to_sofa_coordinates(array)), array)


def test_the_conversion_puts_scene_height_on_the_sofa_vertical() -> None:
    """Y up in the scene, Z up in SOFA. A mirrored axis here is invisible later."""
    converted = to_sofa_coordinates(np.array([[1.0, 1.7, 2.0]]))
    assert converted[0, 2] == pytest.approx(1.7)
    assert converted[0, 0] == pytest.approx(1.0)
    assert converted[0, 1] == pytest.approx(-2.0)


def test_the_conversion_preserves_handedness() -> None:
    basis = np.eye(3)
    converted = to_sofa_coordinates(basis)
    assert np.linalg.det(converted) == pytest.approx(1.0)


def test_sofa_carries_the_samples_the_positions_and_the_provenance(tmp_path: Path) -> None:
    sofar = pytest.importorskip("sofar")
    original = a_response()
    write_sofa(original, tmp_path / "response.sofa", title="w20", licence="CC BY-NC 4.0")
    read = sofar.read_sofa(str(tmp_path / "response.sofa"))

    assert read.GLOBAL_SOFAConventions == SOFA_CONVENTION
    assert np.array_equal(read.Data_IR[:, 0, :], original.ir)
    assert read.Data_SamplingRate == original.sample_rate_hz
    assert np.allclose(from_sofa_coordinates(read.ListenerPosition), original.receiver_positions)
    assert np.allclose(from_sofa_coordinates(read.SourcePosition)[0], original.source_position)
    assert Provenance.from_json(read.GLOBAL_Comment) == original.provenance
    assert read.GLOBAL_License == "CC BY-NC 4.0"


def test_provenance_json_round_trip() -> None:
    provenance = a_provenance()
    assert Provenance.from_json(provenance.to_json()) == provenance
