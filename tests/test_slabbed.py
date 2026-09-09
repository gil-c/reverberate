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
from collections.abc import Callable
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


def _slab_groups() -> Callable[[list[int], list[int], int], list[list[int]]]:
    """``slab_groups`` without importing the child, which needs PFFDTD."""
    import importlib.util

    path = Path("src/reverberate/wave/_child_voxelise.py")
    spec = importlib.util.spec_from_file_location("_child_voxelise_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found: Callable[[list[int], list[int], int], list[list[int]]] = module.slab_groups
    return found


class TestSlabGroups:
    """Slabs are cut by what they hold, not by how wide they are.

    Equal spatial bands make "peak memory is one slab" false on a scene that is
    not uniform along the cut axis. Measured on this flat at 16 kHz with sixteen
    equal bands: killed by the kernel at the fifteenth slab, 20.4 GB of 22
    written, on a box whose resident set had never passed 12 of 31 GB.
    """

    def test_a_scene_dense_at_one_end_is_still_split_evenly(self) -> None:
        """The failure case, in miniature: most of the surface in a tenth of
        the width. Equal bands give one slab nine tenths of the work."""
        groups = _slab_groups()
        voxels = list(range(400))
        # Coordinates 0..99, but 300 of the 400 voxels sit in 90..99.
        starts = sorted([i % 90 for i in range(100)] + [90 + (i % 10) for i in range(300)])
        sizes = [len(g) for g in groups(voxels, starts, 4)]
        assert sum(sizes) == 400
        assert max(sizes) <= 130, sizes

    def test_the_last_slab_is_not_the_remainder(self) -> None:
        """Filling one group until it is full pushes what it did not take to
        the end, and the end is where the run was already dying: measured on
        this flat at 16 kHz with 32 slabs, a greedy cut left the last one
        32 147 non-empty voxels against about 7 500 for its neighbours."""
        import random

        groups = _slab_groups()
        rng = random.Random(1)
        voxels = list(range(20_000))
        starts = sorted(rng.randrange(289) for _ in voxels)
        sizes = [len(g) for g in groups(voxels, starts, 32)]
        assert sum(sizes) == 20_000
        # Nothing is more than half again the mean, last one included.
        assert max(sizes) < 1.5 * (20_000 / 32), sizes
        assert sizes[-1] < 1.5 * (20_000 / 32), sizes[-1]

    def test_a_cut_never_falls_inside_one_coordinate(self) -> None:
        """A slab has to be a contiguous range of engine indices, so two
        voxels at the same coordinate on the cut axis cannot be separated."""
        groups = _slab_groups()
        voxels = list(range(12))
        starts = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        found = groups(voxels, starts, 4)
        for group in found:
            coords = {starts[v] for v in group}
            assert len(coords) <= 1 or group == sorted(group)
        # Every voxel of a coordinate lands in the same slab.
        owner = {v: n for n, g in enumerate(found) for v in g}
        for coordinate in set(starts):
            same = {owner[v] for v in voxels if starts[v] == coordinate}
            assert len(same) == 1

    def test_it_returns_the_slab_count_it_was_asked_for(self) -> None:
        """The caller loops over the list, so a short one changes the loop."""
        groups = _slab_groups()
        found = groups([1, 2], [0, 0], 5)
        assert len(found) == 5
        assert sum(len(g) for g in found) == 2

    def test_one_slab_is_everything_in_order(self) -> None:
        groups = _slab_groups()
        assert groups([3, 1, 2], [9, 4, 7], 1) == [[3, 1, 2]]

    def test_no_voxel_is_lost_or_duplicated(self) -> None:
        """The whole exactness argument rests on the cores tiling the grid."""
        import random

        groups = _slab_groups()
        rng = random.Random(0)
        voxels = list(range(500))
        starts = [rng.randrange(60) for _ in voxels]
        flat = [v for g in groups(voxels, starts, 7) for v in g]
        assert sorted(flat) == voxels
