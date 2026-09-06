"""Tests for voxelising a slab of the grid at a time.

The claim slabbing rests on is that *how* a voxelisation is cut up cannot change
*what* it produces. That claim is checked for real by
``scripts/check_slabs.sh``, which voxelises one bedroom four ways and compares
the datasets; a real voxelisation needs the PFFDTD checkout and half a minute
per case, so what is asserted here is the reasoning that makes it true and the
plumbing that carries it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverberate.wave.voxelise import SceneSpec


def make_spec(tmp_path: Path, **extra: object) -> SceneSpec:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "mats_hash": {
                    "wall": {
                        "tris": [[0, 1, 2]],
                        "pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "sides": [2],
                        "color": [128, 128, 128],
                    }
                },
                "sources": [],
                "receivers": [],
            }
        )
    )
    (tmp_path / "wall.h5").write_bytes(b"not really an impedance filter")
    return SceneSpec(
        model_json=model,
        mat_folder=tmp_path,
        mat_files={"wall": "wall.h5"},
        fmax=1000.0,
        ppw=10.5,
        **extra,  # type: ignore[arg-type]
    )


class TestTheLeversAreOutsideTheKey:
    """Slabbing and ``nh`` change the route, not the destination.

    A boundary node's row is decided by the triangles crossing its own six legs,
    and every voxel is handed every triangle overlapping it plus a one-cell
    halo. Which voxel a node lands in, and how many nodes are consolidated at a
    time, therefore cannot reach the answer. Keying on them would split a cache
    the roadmap sizes in terabytes across choices that produce identical bytes.
    """

    def test_slabbing_does_not_change_the_key(self, tmp_path: Path) -> None:
        one = make_spec(tmp_path / "a")
        many = make_spec(tmp_path / "b", slabs=8)
        assert one.key == many.key

    def test_the_voxel_side_does_not_change_the_key(self, tmp_path: Path) -> None:
        assert make_spec(tmp_path / "a").key == make_spec(tmp_path / "b", nh=32).key

    def test_a_changed_grid_still_does(self, tmp_path: Path) -> None:
        """The guard on the guard: if nothing changed the key, the test above
        would pass by saying nothing."""
        from dataclasses import replace

        spec = make_spec(tmp_path / "a")
        assert replace(spec, fmax=2000.0).key != spec.key


class TestGuards:
    def test_slabs_must_be_at_least_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            make_spec(tmp_path / "a", slabs=0)
