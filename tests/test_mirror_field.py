"""The mirror field: the reference's lattice, the mirror's responses, the app's checker satisfied.

A mock reference field carries the lattice; the mirror field written from
it must pass the app's own checker, keep every per point dataset, hold the
responses given and silence for the rest, and the alignment must read back
a lead and a gain planted in the reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.mirror.field import align_to_reference, read_lattice, write_mirror_field
from reverberate.spatial.encode import Ambisonic
from reverberate.viz.field_payload import check, mock_field, read_header


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    return mock_field(
        tmp_path / "S1.h5",
        source_id="S1",
        source_position=[0.3, 0.8, 0.4],
        box_lo=[-1.0, 0.0, -1.0],
        box_hi=[1.0, 1.5, 1.0],
        step_m=1.0,
        margin_m=0.0,
        heights_m=(1.0,),
        order=1,
        total_s=0.2,
    )


def test_the_mirror_field_passes_the_checker_and_keeps_the_lattice(
    reference: Path, tmp_path: Path
) -> None:
    header = read_header(reference)
    rate = header.sample_rate_hz
    signals = np.zeros((header.channels, header.samples))
    signals[0, 100] = 1.0
    responses = {0: Ambisonic(signals, rate, header.order, np.zeros(3))}
    target = write_mirror_field(
        tmp_path / "field_mirror" / "S1.h5",
        reference,
        responses,
        provenance={"engine": "test"},
        gain=0.5,
    )
    assert check(target) == []
    lattice = read_lattice(reference)
    with h5py.File(target, "r") as handle:
        assert handle["ir"].shape == lattice["ir_shape"]
        assert float(handle["ir"][0, 0, 100]) == pytest.approx(0.5)
        assert np.all(np.asarray(handle["ir"][1]) == 0.0)
        assert list(handle["mirror_silent"][...]) == list(range(1, lattice["ir_shape"][0]))
        np.testing.assert_array_equal(handle["positions"][...], lattice["datasets"]["positions"])
        assert json.loads(handle.attrs["provenance_json"])["applied_gain"] == 0.5
        assert handle.attrs["source_id"] == "S1"


def test_the_alignment_reads_the_reference_s_lead_and_gain(reference: Path) -> None:
    # Plant a direct pulse in the reference at the path length plus 5 ms, at amplitude 3.
    c = 343.2
    with h5py.File(reference, "a") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        direct = np.asarray(handle["direct_path_m"][...])
        for point in range(handle["ir"].shape[0]):
            block = np.zeros(handle["ir"].shape[1:], dtype=np.float32)
            at = int(round((direct[point] / c + 0.005) * rate))
            block[0, at] = 3.0
            block[1:, at] = 0.3
            handle["ir"][point] = block
    alignment = align_to_reference(reference, {point: 1.0 for point in range(9)}, sound_speed_m_s=c)
    assert alignment.lead_s == pytest.approx(0.005, abs=2e-4)
    # The reference's window energy is 9 against the mirror's 1: gain 3.
    assert alignment.gain == pytest.approx(3.0, rel=0.02)
    assert alignment.points_used == 9
    assert alignment.lead_spread_s < 2e-4
