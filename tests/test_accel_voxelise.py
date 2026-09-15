"""The voxeliser's twin against what a closed box must give, and against PFFDTD itself.

The fast layer voxelises a unit box with the numpy twin and checks what can
be known without the upstream code: boundary nodes lie one cell inside the
walls and nowhere else, their legs point into the wall, the floor is the
wall's material, the engine space is sorted and unique, and the files a
cache entry holds are written. The slow layer voxelises the B0 bedroom
with PFFDTD's own interpreter and demands every dataset back bit for bit,
which is the only test that says the port is right. The card's kernel is
compared with the twin on the same box when a card is present.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate import settings
from reverberate.accel.backend import cuda_available
from reverberate.accel.lattice import cart_grid, lattice_for, sim_constants, voxel_triangles
from reverberate.accel.scene import load_scene
from reverberate.accel.voxelise import (
    adjacency_numpy,
    engine_space,
    entries_identical,
    finish,
    voxelise_scene,
)
from test_accel_scene import pffdtd_available, write_scene

gpu = pytest.mark.skipif(not cuda_available(), reason="needs a CUDA device and cupy")


def write_material(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("DEF", data=np.array([[1.0, 2.0, 3.0], [0.5, 0.1, 0.2]]))


def box_setup(
    tmp_path: Path, h: float = 0.05, nh: int = 5
) -> tuple[Any, Any, Any, tuple[np.ndarray, np.ndarray]]:
    scene = load_scene(write_scene(tmp_path / "box.json"))
    constants = sim_constants(20.0, 50.0, 343.2 / (h * 10.5), 10.5)
    grid = cart_grid(constants.h, 3.5, scene.bmin, scene.bmax)
    lattice = lattice_for(grid, nh)
    return scene, grid, lattice, voxel_triangles(lattice, scene.pre, np)


class TestTheTwinOnABox:
    def test_boundary_nodes_hug_the_walls_and_point_into_them(self, tmp_path: Path) -> None:
        scene, grid, lattice, (offsets, tri_ids) = box_setup(tmp_path)
        nodes = adjacency_numpy(scene, grid, lattice, offsets, tri_ids)
        assert nodes.count > 0
        bn_ixyz, adj_bn, mat_bn, saf_bn = finish(nodes, scene, grid, np)
        ny, nz = grid.shape[1], grid.shape[2]
        iz = bn_ixyz % nz
        iy = (bn_ixyz // nz) % ny
        ix = bn_ixyz // (ny * nz)
        xyz = np.stack([grid.xv[ix], grid.yv[iy], grid.zv[iz]], axis=1)
        h = grid.h
        # Every boundary node is within one cell of some wall plane...
        distance = np.minimum(np.abs(xyz), np.abs(xyz - 1.0)).min(axis=1)
        assert np.all(distance <= h * (1 + 1e-6))
        # ...and every interior node that far from a wall is one.
        inside = xyz[np.all((xyz > 0.0) & (xyz < 1.0), axis=1)]
        assert inside.shape[0] > 0
        assert np.all(mat_bn[np.all((xyz > 0.0) & (xyz < 1.0), axis=1)] == 0)
        # A node just above the floor is not adjacent downwards and is adjacent upwards.
        floor = np.all((xyz > 0.0) & (xyz < 1.0), axis=1) & (xyz[:, 2] < h)
        assert floor.any()
        assert np.all(~adj_bn[floor, 5]) and np.all(adj_bn[floor, 4])
        assert np.all(saf_bn[floor] >= 1.0 - 1e-12)
        # Outside the box, on the back side of every wall, nodes are sealed.
        outside = ~np.all((xyz >= -1e-9) & (xyz <= 1.0 + 1e-9), axis=1)
        assert np.all(mat_bn[outside] == -1)
        assert np.all(~adj_bn[outside].any(axis=1))

    def test_the_engine_space_is_sorted_unique_and_rotated_biggest_first(
        self, tmp_path: Path
    ) -> None:
        scene = load_scene(write_scene(tmp_path / "box.json"))
        constants = sim_constants(20.0, 50.0, 343.2 / (0.05 * 10.5), 10.5)
        grid = cart_grid(constants.h, 3.5, scene.bmin, np.array([1.5, 1.0, 2.0]))
        lattice = lattice_for(grid, 5)
        offsets, tri_ids = voxel_triangles(lattice, scene.pre, np)
        nodes = adjacency_numpy(scene, grid, lattice, offsets, tri_ids)
        space = engine_space(grid, *finish(nodes, scene, grid, np), np)
        assert list(space.order) == [2, 0, 1]
        assert space.shape == (grid.shape[2], grid.shape[0], grid.shape[1])
        assert np.all(np.diff(space.bn_ixyz) > 0)
        assert space.xv is grid.zv and space.zv is grid.yv
        assert space.adj_bn.shape == (nodes.count, 6)
        assert int(space.bn_ixyz.max()) < int(np.prod(space.shape))

    def test_the_four_files_are_written_and_the_entry_equals_itself(self, tmp_path: Path) -> None:
        model = write_scene(tmp_path / "box.json")
        write_material(tmp_path / "wall.h5")
        report = voxelise_scene(
            model,
            tmp_path / "entry",
            mat_folder=tmp_path,
            mat_files={"wall": "wall.h5"},
            fmax=343.2 / (0.05 * 10.5),
            ppw=10.5,
            nh=5,
            xp=np,
        )
        record = report.record()
        assert record["boundary_nodes"] > 0
        assert record["voxeliser"].startswith("reverberate.accel")
        for name in ("vox_out.h5", "cart_grid.h5", "sim_consts.h5", "sim_mats.h5"):
            assert (tmp_path / "entry" / name).is_file()
        with h5py.File(tmp_path / "entry" / "vox_out.h5") as handle:
            assert int(handle["Nb"][()]) == record["boundary_nodes"]
            assert handle["adj_bn"].dtype == bool
            assert handle["mat_bn"].dtype == np.int8
        with h5py.File(tmp_path / "entry" / "sim_mats.h5") as handle:
            assert int(handle["Nmat"][()]) == 1
            assert list(handle["Mb"][()]) == [2]
        assert entries_identical(tmp_path / "entry", tmp_path / "entry")["identical"]

    def test_a_material_file_missing_is_refused(self, tmp_path: Path) -> None:
        model = write_scene(tmp_path / "box.json")
        write_material(tmp_path / "wall.h5")
        with pytest.raises(ValueError, match="materials"):
            voxelise_scene(
                model,
                tmp_path / "entry",
                mat_folder=tmp_path,
                mat_files={"wall": "wall.h5", "extra": "extra.h5"},
                fmax=343.2 / (0.05 * 10.5),
                ppw=10.5,
                nh=5,
                xp=np,
            )


@gpu
def test_the_kernel_agrees_with_the_twin_on_the_box(tmp_path: Path) -> None:
    import cupy

    from reverberate.accel.voxelise import adjacency_cuda

    scene, grid, lattice, (offsets, tri_ids) = box_setup(tmp_path, h=0.02, nh=6)
    twin = adjacency_numpy(scene, grid, lattice, offsets, tri_ids)
    card = adjacency_cuda(scene, grid, lattice, offsets, tri_ids, nodes_per_slab=5000)
    order_t, order_c = np.argsort(twin.flat), np.argsort(card.flat)
    assert np.array_equal(twin.flat[order_t], card.flat[order_c])
    assert np.array_equal(twin.adj[order_t], card.adj[order_c])
    assert np.array_equal(twin.tidx[order_t], card.tidx[order_c])
    a = finish(twin, scene, grid, np)
    b = finish(card, scene, grid, cupy)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x[np.argsort(a[0])], y[np.argsort(b[0])])


@pytest.mark.slow
@pffdtd_available
def test_the_twin_reproduces_pffdtd_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PFFDTD's own voxeliser on the B0 bedroom at 500 Hz; every dataset equal."""
    from reverberate.wave.remote_voxelise import grid_shape_of
    from reverberate.wave.voxelise import SceneSpec, nh_for, voxelise

    models = settings.data_root() / "runs" / "b0_truncation" / "models"
    scene_file = models / "bedroom_only.json"
    if not scene_file.is_file():
        pytest.skip("the B0 models are not on this machine")
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
    model = json.loads(scene_file.read_text())
    manifest = json.loads((models / "manifest.json").read_text())
    labels = sorted(model["mats_hash"])
    mat_folder = tmp_path / "materials"
    mat_folder.mkdir()
    payload = json.dumps({label: manifest["materials"][label] for label in labels})
    subprocess.run(
        [
            os.environ["PFFDTD_PYTHON"],
            "-c",
            "import sys, json; sys.path.insert(0, "
            f"{str(Path(os.environ['PFFDTD_DIR']) / 'python')!r});"
            "import numpy as np;"
            "from materials.adm_funcs import fit_to_Sabs_oct_11;"
            f"[fit_to_Sabs_oct_11(np.array(c, dtype=float), filename={str(mat_folder)!r}"
            '+ "/" + label + ".h5", plot=False)'
            f" for label, c in json.loads({payload!r}).items()]",
        ],
        check=True,
        capture_output=True,
    )
    mat_files = {label: f"{label}.h5" for label in labels}
    nh = nh_for(grid_shape_of(scene_file, 500.0, 10.5))
    reference = voxelise(
        SceneSpec(
            model_json=scene_file,
            mat_folder=mat_folder,
            mat_files=mat_files,
            fmax=500.0,
            ppw=10.5,
            nh=nh,
        ),
        nprocs=4,
    )
    ours = tmp_path / "accel"
    voxelise_scene(
        scene_file,
        ours,
        mat_folder=mat_folder,
        mat_files=mat_files,
        fmax=500.0,
        ppw=10.5,
        nh=nh,
        xp=np,
    )
    report = entries_identical(reference.path, ours)
    assert report["identical"], report
