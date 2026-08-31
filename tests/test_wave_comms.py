"""Tests for the split pipeline, section 11 and task W8.

The expensive failure here is silent: a ``comms_out.h5`` whose indices are in
the wrong space runs perfectly well on a rented GPU and returns an impulse
response measured somewhere else in the room. Nothing crashes, and the bill is
paid either way. So the tests come in two layers.

The fast layer builds a small grid by hand and checks the pieces that can be
checked without PFFDTD: interpolation weights, the axis permutation, the FCC
fold, the sort, the clash refusal, the cache key. The slow layer, skipped unless
``PFFDTD_DIR`` and ``PFFDTD_PYTHON`` point at a working install, runs a real
``sim_setup`` and demands that the split pipeline reproduce its ``comms_out.h5``
**dataset for dataset, bit for bit**. That is the only check that means
anything, and it is the one the module was developed against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from reverberate import settings
from reverberate.wave import comms as comms_module
from reverberate.wave.comms import (
    ENGINE_FILES,
    Grid,
    fold_fcc,
    interp_weights,
    source_signal,
    transpose_order,
    write_comms,
)
from reverberate.wave.remote import Machine, upload
from reverberate.wave.voxelise import (
    CacheEntry,
    SceneSpec,
    cache_root,
    engine_inputs,
    entry_for,
)


def make_grid(h: float = 0.1, fcc_flag: int = 0) -> Grid:
    """A small, deliberately lopsided grid, so a wrong axis order shows."""
    return Grid(
        h=h,
        Ts=h / 343.2 * 0.577,
        l2=1 / 3,
        fcc_flag=fcc_flag,
        xv=np.arange(6) * h,
        yv=np.arange(9) * h,
        zv=np.arange(4) * h,
    )


def write_grid(directory: Path, grid: Grid) -> Path:
    """The two files :func:`load_grid` reads, and nothing else."""
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory / "sim_consts.h5", "w") as handle:
        handle.create_dataset("h", data=np.float64(grid.h))
        handle.create_dataset("Ts", data=np.float64(grid.Ts))
        handle.create_dataset("l2", data=np.float64(grid.l2))
        handle.create_dataset("fcc_flag", data=np.int8(grid.fcc_flag))
    with h5py.File(directory / "cart_grid.h5", "w") as handle:
        handle.create_dataset("xv", data=grid.xv)
        handle.create_dataset("yv", data=grid.yv)
        handle.create_dataset("zv", data=grid.zv)
    return directory


class TestInterpolation:
    def test_a_node_gets_all_the_weight(self) -> None:
        grid = make_grid()
        alpha, ixyz = interp_weights(np.array([0.2, 0.3, 0.1]), grid)
        assert alpha[0] == pytest.approx(1.0)
        assert alpha[1:] == pytest.approx(np.zeros(7))
        nx, ny, nz = grid.shape
        assert ixyz[0] == 2 * nz * ny + 3 * nz + 1

    def test_the_weights_reproduce_the_point(self) -> None:
        grid = make_grid()
        position = np.array([0.24, 0.37, 0.13])
        alpha, ixyz = interp_weights(position, grid)
        nx, ny, nz = grid.shape
        iz = ixyz % nz
        iy = (ixyz - iz) // nz % ny
        ix = ((ixyz - iz) // nz - iy) // ny
        corners = np.c_[grid.xv[ix], grid.yv[iy], grid.zv[iz]]
        assert np.sum(alpha) == pytest.approx(1.0)
        assert alpha @ corners == pytest.approx(position)

    def test_a_point_outside_the_grid_is_refused(self) -> None:
        grid = make_grid()
        with pytest.raises(ValueError, match="outside the grid"):
            interp_weights(np.array([99.0, 0.3, 0.1]), grid)

    def test_one_point_at_a_time(self) -> None:
        with pytest.raises(ValueError, match="one xyz point"):
            interp_weights(np.zeros((2, 3)), make_grid())


class TestSignals:
    def test_an_impulse_is_one_sample(self) -> None:
        signal = source_signal(0.01, 1e-4, "impulse")
        assert signal.size == 100
        assert signal[0] == 1.0
        assert not signal[1:].any()

    def test_a_hann_window_starts_and_ends_at_zero(self) -> None:
        signal = source_signal(0.01, 1e-4, "hann20")
        assert signal[0] == pytest.approx(0.0)
        assert signal[19] == pytest.approx(signal[1])
        assert not signal[20:].any()

    def test_an_unknown_signal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown signal type"):
            source_signal(0.01, 1e-4, "sawtooth")  # type: ignore[arg-type]


class TestIndexSpace:
    def test_the_longest_axis_goes_first(self) -> None:
        assert list(transpose_order((6, 9, 4))) == [1, 0, 2]
        assert list(transpose_order((9, 6, 4))) == [0, 1, 2]

    def test_folding_mirrors_the_upper_half(self) -> None:
        # A grid of Ny = 8: y = 5 folds onto 8 - 5 - 1 = 2.
        shape = (3, 8, 2)
        ny_half = 8 // 2 + 1
        upper = 1 * 8 * 2 + 5 * 2 + 1
        lower = 1 * 8 * 2 + 2 * 2 + 1
        folded = fold_fcc(np.array([upper, lower]), shape)
        expected = 1 * 2 * ny_half + 2 * 2 + 1
        assert folded[0] == expected
        assert folded[1] == expected

    def test_folding_needs_an_even_axis(self) -> None:
        with pytest.raises(ValueError, match="even Ny"):
            fold_fcc(np.array([0]), (3, 7, 2))


class TestWriteComms:
    def test_it_writes_what_the_engine_reads(self, tmp_path: Path) -> None:
        directory = write_grid(tmp_path / "entry", make_grid())
        out = write_comms(
            directory,
            np.array([0.2, 0.3, 0.1]),
            np.array([[0.3, 0.4, 0.2]]),
            0.01,
            out_path=tmp_path / "comms_out.h5",
        )
        with h5py.File(out, "r") as handle:
            assert set(handle.keys()) == {
                "in_ixyz",
                "out_ixyz",
                "out_alpha",
                "out_reorder",
                "in_sigs",
                "Ns",
                "Nr",
                "Nt",
                "diff",
            }
            assert handle["Ns"][()] == 8
            assert handle["Nr"][()] == 8
            assert handle["diff"][()] == 1
            # sort_sim_data's contract: both index arrays are sorted, and
            # out_reorder puts the engine's output back in receiver order.
            assert np.all(np.diff(handle["in_ixyz"][...]) > 0)
            assert np.all(np.diff(handle["out_ixyz"][...]) > 0)
            assert np.array_equal(np.sort(handle["out_reorder"][...]), np.arange(8))

    def test_many_receivers_keep_their_own_weights(self, tmp_path: Path) -> None:
        directory = write_grid(tmp_path / "entry", make_grid())
        receivers = np.array([[0.3, 0.4, 0.2], [0.35, 0.45, 0.15], [0.2, 0.6, 0.1]])
        out = write_comms(
            directory,
            np.array([0.2, 0.3, 0.1]),
            receivers,
            0.005,
            out_path=tmp_path / "comms_out.h5",
        )
        with h5py.File(out, "r") as handle:
            assert handle["out_alpha"].shape == (3, 8)
            assert handle["Nr"][()] == 24
            # out_alpha is in the caller's receiver order, not the sorted one:
            # the engine applies out_reorder to the signals, not to the weights.
            assert np.allclose(handle["out_alpha"][...].sum(axis=1), 1.0)

    def test_it_needs_a_receiver(self, tmp_path: Path) -> None:
        directory = write_grid(tmp_path / "entry", make_grid())
        with pytest.raises(ValueError, match="at least one receiver"):
            write_comms(directory, np.array([0.2, 0.3, 0.1]), np.empty((0, 3)), 0.01)

    def test_a_receiver_on_a_boundary_node_is_refused(self, tmp_path: Path) -> None:
        """The scheme only supports air nodes, and a clash is silent otherwise."""
        directory = write_grid(tmp_path / "entry", make_grid())
        grid = make_grid()
        _, ixyz = interp_weights(np.array([0.3, 0.4, 0.2]), grid)
        with h5py.File(directory / "vox_out.h5", "w") as handle:
            handle.create_dataset("bn_ixyz", data=np.sort(ixyz))
        with pytest.raises(ValueError, match="receiver interpolation touches"):
            write_comms(
                directory,
                np.array([0.2, 0.3, 0.1]),
                np.array([[0.3, 0.4, 0.2]]),
                0.01,
                out_path=directory / "comms_out.h5",
            )

    def test_double_precision_does_not_differentiate(self, tmp_path: Path) -> None:
        directory = write_grid(tmp_path / "entry", make_grid())
        out = write_comms(
            directory,
            np.array([0.2, 0.3, 0.1]),
            np.array([[0.3, 0.4, 0.2]]),
            0.01,
            diff_source=False,
            out_path=tmp_path / "comms_out.h5",
        )
        with h5py.File(out, "r") as handle:
            assert handle["diff"][()] == 0
            signals = handle["in_sigs"][...]
        # An undifferentiated impulse stays one sample long, whatever weights
        # the interpolation gives it.
        assert not signals[:, 1:].any()
        assert signals[:, 0].sum() == pytest.approx(1 / 3 / 0.1)


class TestDifferentiator:
    def test_it_matches_the_bilinear_filter(self) -> None:
        """Checked against the recurrence scipy's lfilter would run."""
        ts = 1e-4
        signals = np.random.default_rng(0).normal(size=(3, 32))
        got = comms_module._differentiate(signals.copy(), ts)
        expected = np.zeros_like(signals)
        for row in range(signals.shape[0]):
            previous_in = previous_out = 0.0
            for n in range(signals.shape[1]):
                current = 2 / ts * (signals[row, n] - previous_in) - previous_out
                expected[row, n] = current
                previous_in, previous_out = signals[row, n], current
        assert np.allclose(got, expected)


class TestCacheKey:
    def _spec(self, tmp_path: Path, fmax: float = 2000.0, body: str = "{}") -> SceneSpec:
        model = tmp_path / "scene.json"
        model.write_text(body)
        mats = tmp_path / "mats"
        mats.mkdir(exist_ok=True)
        (mats / "wall.h5").write_bytes(b"impedance")
        return SceneSpec(
            model_json=model,
            mat_folder=mats,
            mat_files={"wall": "wall.h5"},
            fmax=fmax,
            ppw=10.5,
        )

    def test_the_same_scene_and_grid_share_an_entry(self, tmp_path: Path) -> None:
        assert self._spec(tmp_path).key == self._spec(tmp_path).key

    def test_a_finer_grid_is_a_different_entry(self, tmp_path: Path) -> None:
        assert self._spec(tmp_path).key != self._spec(tmp_path, fmax=4000.0).key

    def test_changed_geometry_is_a_different_entry(self, tmp_path: Path) -> None:
        first = self._spec(tmp_path).key
        assert first != self._spec(tmp_path, body='{"moved": true}').key

    def test_changed_materials_are_a_different_entry(self, tmp_path: Path) -> None:
        """The roadmap says 'scene and grid step'; sim_mats.h5 says otherwise."""
        spec = self._spec(tmp_path)
        first = spec.key
        (Path(spec.mat_folder) / "wall.h5").write_bytes(b"other impedance")
        assert first != spec.key

    def test_an_uncomputed_entry_is_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        entry = entry_for(self._spec(tmp_path))
        assert entry.path.parent == cache_root()
        assert not entry.complete
        assert entry.manifest == {}


class TestUpload:
    def test_it_ships_only_what_the_engine_reads(self, tmp_path: Path) -> None:
        """cart_grid.h5 is pure bandwidth, and a scene file is a policy problem."""
        stray = tmp_path / "cart_grid.h5"
        stray.write_bytes(b"not for the engine")
        machine = Machine(host="example.invalid")
        with pytest.raises(ValueError, match="refusing"):
            upload(machine, [stray])

    def test_the_engine_reads_four_files(self) -> None:
        assert ENGINE_FILES == (
            "sim_consts.h5",
            "vox_out.h5",
            "comms_out.h5",
            "sim_mats.h5",
        )

    def test_the_ssh_and_scp_argv_carry_the_port(self) -> None:
        machine = Machine(host="ssh5.vast.ai", port=41234, identity=Path("/tmp/key"))
        assert machine.ssh_command("true")[-2:] == ["root@ssh5.vast.ai", "true"]
        assert "-p" in machine.ssh_command("true")
        argv = machine.scp_command([Path("/tmp/a.h5")], "/root/run", download=False)
        assert argv[-1] == "root@ssh5.vast.ai:/root/run"
        assert "-P" in argv


class TestEngineInputs:
    def _entry(self, tmp_path: Path) -> CacheEntry:
        for name in ("sim_consts.h5", "vox_out.h5", "sim_mats.h5", "cart_grid.h5"):
            (tmp_path / name).write_bytes(b"x")
        return CacheEntry(path=tmp_path, key="k", manifest={})

    def test_it_is_the_four_files_in_the_engine_s_order(self, tmp_path: Path) -> None:
        entry = self._entry(tmp_path)
        comms = tmp_path / "elsewhere" / "comms_out.h5"
        comms.parent.mkdir()
        comms.write_bytes(b"x")
        paths = engine_inputs(entry, comms)
        assert [p.name for p in paths] == list(ENGINE_FILES)
        # cart_grid.h5 stays home: the engine has never read it.
        assert all(p.name != "cart_grid.h5" for p in paths)
        assert paths[2] == comms

    def test_a_missing_input_is_caught_before_the_meter_starts(self, tmp_path: Path) -> None:
        entry = self._entry(tmp_path)
        with pytest.raises(FileNotFoundError, match="missing engine inputs"):
            engine_inputs(entry, tmp_path / "no_such_comms.h5")


pffdtd_available = pytest.mark.skipif(
    not (os.environ.get("PFFDTD_DIR") and os.environ.get("PFFDTD_PYTHON")),
    reason="needs a PFFDTD install; set PFFDTD_DIR and PFFDTD_PYTHON",
)


@pytest.mark.slow
@pffdtd_available
@pytest.mark.parametrize("fcc", [False, True])
def test_the_split_reproduces_sim_setup_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcc: bool
) -> None:
    """The only test that proves the split is safe to spend money on.

    Runs PFFDTD's own ``sim_setup``, then rebuilds the same run out of a cached
    voxelisation plus a regenerated ``comms_out.h5``, and demands equality on
    every dataset in both ``comms_out.h5`` and ``vox_out.h5``. Bit for bit,
    because anything less means the engine is being fed a different simulation.
    """
    import subprocess

    # Resolved before the data root is redirected, because the B0 models live
    # wherever REVERBERATE_DATA points and the cache is going to tmp_path.
    models = settings.data_root() / "runs" / "b0_truncation" / "models"
    if not models.is_dir():
        pytest.skip("the B0 models are not on this machine")
    monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))

    scene_file = models / "bedroom_only.json"
    model = json.loads(scene_file.read_text())
    manifest = json.loads((models / "manifest.json").read_text())
    labels = sorted(model["mats_hash"])
    mat_folder = tmp_path / "materials"
    mat_folder.mkdir()
    pffdtd_python = os.environ["PFFDTD_PYTHON"]
    payload = json.dumps({label: manifest["materials"][label] for label in labels})
    subprocess.run(
        [
            pffdtd_python,
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
    )
    mat_files = {label: f"{label}.h5" for label in labels}

    reference = tmp_path / "reference"
    subprocess.run(
        [
            pffdtd_python,
            "-c",
            "import sys, json, multiprocessing;"
            'multiprocessing.set_start_method("fork", force=True);'
            f"sys.path.insert(0, {str(Path(os.environ['PFFDTD_DIR']) / 'python')!r});"
            "from sim_setup import sim_setup;"
            f"sim_setup(model_json_file={str(scene_file)!r}, mat_folder={str(mat_folder)!r},"
            f" mat_files_dict={mat_files!r}, source_num=1, insig_type='impulse',"
            f" diff_source=True, duration=0.01, Tc=20, rh=50, fcc_flag={fcc},"
            f" PPW=10.5, fmax=500, save_folder={str(reference)!r},"
            f" save_folder_gpu={str(reference)!r}, compress=0)",
        ],
        check=True,
    )

    from reverberate.wave.voxelise import voxelise

    entry = voxelise(
        SceneSpec(
            model_json=scene_file,
            mat_folder=mat_folder,
            mat_files=mat_files,
            fmax=500,
            ppw=10.5,
            fcc=fcc,
        )
    )
    out = write_comms(
        entry.path,
        np.array(model["sources"][0]["xyz"], dtype=float),
        np.array([r["xyz"] for r in model["receivers"]], dtype=float),
        0.01,
        out_path=tmp_path / "comms_out.h5",
    )

    for name, produced in (("comms_out.h5", out), ("vox_out.h5", entry.path / "vox_out.h5")):
        with h5py.File(reference / name, "r") as expected, h5py.File(produced, "r") as got:
            assert set(expected.keys()) == set(got.keys()), name
            for key in expected:
                assert np.array_equal(expected[key][...], got[key][...]), f"{name}:{key}"
