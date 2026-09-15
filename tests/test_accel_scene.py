"""The scene port: PFFDTD's arithmetic, not an approximation of it.

The fast layer checks the algebra on shapes whose answer is known and the
one identity the whole port rests on, that a three term dot product summed
left to right is what ``numpy.sum`` computes. The slow layer runs PFFDTD's
own ``RoomGeo`` in its own interpreter on a real export and demands every
array back bit for bit.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reverberate import settings
from reverberate.accel.scene import dotv, load_scene, normalise, tris_precompute, vecnorm

pffdtd_available = pytest.mark.skipif(
    not (os.environ.get("PFFDTD_DIR") and os.environ.get("PFFDTD_PYTHON")),
    reason="needs PFFDTD_DIR and PFFDTD_PYTHON",
)


def write_scene(path: Path, extra_material: bool = False) -> Path:
    """A closed unit box with inward normals, twelve triangles, air inside."""
    pts = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    # Wound so that the area normal points into the box.
    tris = [
        [0, 1, 2],
        [0, 2, 3],  # z = 0, normal +z
        [4, 6, 5],
        [4, 7, 6],  # z = 1, normal -z
        [0, 5, 1],
        [0, 4, 5],  # y = 0, normal +y
        [3, 6, 7],
        [3, 2, 6],  # y = 1, normal -y
        [0, 7, 4],
        [0, 3, 7],  # x = 0, normal +x
        [1, 6, 2],
        [1, 5, 6],  # x = 1, normal -x
    ]
    mats = {
        "wall": {"pts": pts, "tris": tris, "sides": [2] * 12, "color": [1, 2, 3]},
    }
    if extra_material:
        mats["_RIGID"] = {
            "pts": [[0.4, 0.4, 0.4], [0.6, 0.4, 0.4], [0.5, 0.6, 0.4], [0.5, 0.5, 0.4]],
            "tris": [[0, 1, 2], [0, 1, 3]],
            "sides": [0, 0],
            "color": [0, 0, 0],
        }
        mats["carpet"] = {
            "pts": [[0.2, 0.0, 0.2], [0.8, 0.0, 0.2], [0.8, 0.0, 0.8], [0.2, 0.0, 0.8]],
            "tris": [[0, 2, 1], [0, 3, 2]],
            "sides": [2, 2],
            "color": [4, 5, 6],
        }
    path.write_text(
        json.dumps(
            {
                "mats_hash": mats,
                "sources": [{"xyz": [0.3, 0.3, 0.3], "name": "S1"}],
                "receivers": [{"xyz": [0.7, 0.6, 0.5], "name": "R1"}],
            }
        )
    )
    return path


class TestTheAlgebra:
    def test_dotv_is_numpy_s_sum_over_three_products(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((5000, 3)) * 10.0 ** rng.integers(-8, 8, (5000, 1))
        b = rng.standard_normal((5000, 3)) * 10.0 ** rng.integers(-8, 8, (5000, 1))
        assert np.array_equal(dotv(a, b), np.sum(a * b, axis=-1))

    def test_normalise_carries_the_epsilon_upstream_adds(self) -> None:
        v = np.array([[1.0, 0.0, 0.0]])
        assert normalise(v)[0, 0] == 1.0 / (1.0 + np.finfo(float).eps)
        assert normalise(v)[0, 0] != 1.0

    def test_precompute_on_a_right_triangle(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pre = tris_precompute(pts, np.array([[0, 1, 2]]))
        assert pre.area[0] == pytest.approx(2.0)
        assert np.allclose(pre.unor[0], [0.0, 0.0, 1.0], atol=1e-15)
        assert np.allclose(pre.cent[0], [2 / 3, 2 / 3, 0.0])
        assert np.array_equal(pre.bmin[0], [0.0, 0.0, 0.0])
        assert np.array_equal(pre.bmax[0], [2.0, 2.0, 0.0])
        # Outward edge normals point away from the third vertex.
        assert pre.eab_unor[0] @ (pts[2] - pts[0]) < 0
        assert pre.ebc_unor[0] @ (pts[0] - pts[1]) < 0
        assert pre.eca_unor[0] @ (pts[1] - pts[2]) < 0
        assert vecnorm(pre.eab_unor)[0] == pytest.approx(1.0, abs=1e-15)


class TestLoadScene:
    def test_materials_are_alphabetical_with_rigid_last_and_negative(self, tmp_path: Path) -> None:
        scene = load_scene(write_scene(tmp_path / "box.json", extra_material=True))
        assert scene.mat_str == ["carpet", "wall", "_RIGID"]
        assert scene.nmat == 2
        assert set(np.unique(scene.mat_ind).tolist()) == {-1, 0, 1}
        assert scene.triangles == 12 + 2 + 2
        assert np.all(scene.mat_side[scene.mat_ind == -1] == 0)
        assert np.array_equal(scene.bmin, [0.0, 0.0, 0.0])
        assert np.array_equal(scene.bmax, [1.0, 1.0, 1.0])

    def test_degenerate_triangles_are_pruned_as_upstream_prunes_them(self, tmp_path: Path) -> None:
        path = write_scene(tmp_path / "box.json")
        model = json.loads(path.read_text())
        model["mats_hash"]["wall"]["pts"].append([0.5, 0.5, 0.5])
        model["mats_hash"]["wall"]["pts"].append([0.5, 0.5, 0.5 + 1e-9])
        model["mats_hash"]["wall"]["tris"].append([8, 9, 8])
        model["mats_hash"]["wall"]["sides"].append(2)
        path.write_text(json.dumps(model))
        scene = load_scene(path)
        assert scene.triangles == 12
        assert scene.pts.shape[0] == 10

    def test_a_rigid_triangle_with_a_side_is_refused(self, tmp_path: Path) -> None:
        path = write_scene(tmp_path / "box.json", extra_material=True)
        model = json.loads(path.read_text())
        model["mats_hash"]["_RIGID"]["sides"] = [1, 0]
        path.write_text(json.dumps(model))
        with pytest.raises(ValueError, match="_RIGID"):
            load_scene(path)


@pytest.mark.slow
@pffdtd_available
def test_the_port_reproduces_room_geo_bit_for_bit(tmp_path: Path) -> None:
    """PFFDTD's own ``RoomGeo`` on the B0 bedroom, every array equal."""
    models = settings.data_root() / "runs" / "b0_truncation" / "models"
    scene_file = models / "bedroom_only.json"
    if not scene_file.is_file():
        pytest.skip("the B0 models are not on this machine")
    dump = tmp_path / "room_geo.npz"
    subprocess.run(
        [
            os.environ["PFFDTD_PYTHON"],
            "-c",
            "import sys, numpy as np;"
            f"sys.path.insert(0, {str(Path(os.environ['PFFDTD_DIR']) / 'python')!r});"
            "from common.room_geo import RoomGeo;"
            f"g = RoomGeo({str(scene_file)!r});"
            f"np.savez({str(dump)!r}, pts=g.pts, tris=g.tris, mat_ind=g.mat_ind,"
            " mat_side=g.mat_side, scene_bmin=g.bmin, scene_bmax=g.bmax,"
            " **{k: g.tris_pre[k] for k in ('v', 'nor', 'unor', 'eab_unor', 'ebc_unor',"
            " 'eca_unor', 'cent', 'bmin', 'bmax', 'area')})",
        ],
        check=True,
        capture_output=True,
    )
    theirs = np.load(dump)
    ours = load_scene(scene_file)
    for name in ("pts", "tris", "mat_ind", "mat_side"):
        assert np.array_equal(theirs[name], getattr(ours, name)), name
    for name in ("v", "nor", "unor", "eab_unor", "ebc_unor", "eca_unor", "cent", "area"):
        assert np.array_equal(theirs[name], getattr(ours.pre, name)), name
    assert np.array_equal(theirs["bmin"], ours.pre.bmin)
    assert np.array_equal(theirs["bmax"], ours.pre.bmax)
    assert np.array_equal(theirs["scene_bmin"], ours.bmin)
    assert np.array_equal(theirs["scene_bmax"], ours.bmax)
