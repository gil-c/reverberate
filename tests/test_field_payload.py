"""The impulse response field: the contract with the producer, and its mock.

The checker must refuse what the app cannot read, the mock must pass it, and
the index must name every cell's bytes where they really are.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate.viz.field_payload import build_site, check, mock_field, read_header


@pytest.fixture
def field(tmp_path: Path) -> Path:
    """A small mock: a 2 x 1.5 x 2 m box at 1 m steps, one height, order 1, 0.1 s."""
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


def test_the_mock_passes_the_checker_and_reads_back(field: Path) -> None:
    assert check(field) == []
    header = read_header(field)
    assert (header.order, header.channels, header.positions) == (1, 4, 9)
    assert header.grid_shape == (3, 1, 3) and header.grid_step_m[1] == 0.0
    assert header.samples == 9600


@pytest.mark.parametrize(
    ("spoil", "wording"),
    [
        (lambda h: h.attrs.__setitem__("normalisation", "SN3D"), "SN3D"),
        (lambda h: h["cell_index"].__setitem__(0, [99, 0, 0]), "grid_shape"),
    ],
)
def test_what_the_app_cannot_read_is_refused_by_name(
    field: Path, spoil: Callable[[h5py.File], None], wording: str
) -> None:
    """SN3D would decode a few decibels wrong per order; a cell off the
    lattice would be found by nobody. Both are named, not guessed at."""
    with h5py.File(field, "a") as handle:
        spoil(handle)
    assert any(wording in problem for problem in check(field))


def test_a_compressed_field_is_refused_because_it_cannot_be_range_read(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "z.h5", "w") as handle:
        handle.create_dataset("ir", data=np.zeros((2, 4, 8), np.float32), compression="gzip")
        handle["positions"] = np.zeros((2, 3))
        handle["cell_index"] = np.zeros((2, 3), np.int32)
        handle["direct_path_m"] = np.zeros(2)
        for name in ("order", "sample_rate_hz", "gain"):
            handle.attrs[name] = 1
        for name in (
            "ordering",
            "normalisation",
            "source_id",
            "directivity",
            "scene_id",
            "dwelling",
        ):
            handle.attrs[name] = {"ordering": "ACN", "normalisation": "N3D"}.get(name, "x")
        handle.attrs["source_position"] = [0.0, 0.0, 0.0]
        handle.attrs["grid_origin_m"] = [0.0, 0.0, 0.0]
        handle.attrs["grid_step_m"] = [1.0, 0.0, 1.0]
        handle.attrs["grid_shape"] = [2, 1, 1]
    assert any("uncompressed" in problem for problem in check(tmp_path / "z.h5"))


def test_the_direct_arrival_is_louder_nearer_the_source(field: Path) -> None:
    """The mock's one claim on physics: level falls with distance."""
    with h5py.File(field, "r") as handle:
        peak = np.abs(handle["ir"][:, 0, :]).max(axis=1)
        direct = handle["direct_path_m"][:]
    assert peak[np.argmin(direct)] > peak[np.argmax(direct)]


def test_the_index_names_every_cells_bytes_and_survives_an_unchanged_file(
    field: Path, tmp_path: Path
) -> None:
    """The page reads a cell straight out of the file by range; the bytes at
    the named offset must be that cell, in the index's order. The index is
    rebuilt only when the file changes."""
    record = build_site(field, tmp_path / "web")
    raw = field.read_bytes()
    with h5py.File(field, "r") as handle:
        ir = handle["ir"][:]
    assert (tmp_path / "web" / "field.h5").is_symlink()
    assert record["cell_bytes"] == ir.shape[1] * ir.shape[2] * 4
    for position, offset in enumerate(record["offsets"]):
        block = np.frombuffer(raw[offset : offset + record["cell_bytes"]], dtype="<f4")
        assert np.array_equal(block, ir[position].ravel())
    assert record["rooms"] == ["box"] * 9 and record["solved_to_hz"] == [8000.0] * 9

    marker = (tmp_path / "web" / "index.json").stat().st_mtime_ns
    assert build_site(field, tmp_path / "web") == record
    assert (tmp_path / "web" / "index.json").stat().st_mtime_ns == marker
    with h5py.File(field, "a") as handle:
        handle.attrs["gain"] = 0.5
    assert build_site(field, tmp_path / "web")["gain"] == 0.5
