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
from reverberate.wave import remote as comms_remote
from reverberate.wave.comms import (
    ENGINE_FILES,
    Grid,
    fold_fcc,
    interp_weights,
    nearest_node,
    source_signal,
    transpose_order,
    write_comms,
)
from reverberate.wave.remote import Machine, upload
from reverberate.wave.voxelise import (
    CacheEntry,
    SceneSpec,
    engine_inputs,
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


class TestSignals:
    def test_an_impulse_is_one_sample(self) -> None:
        signal = source_signal(0.01, 1e-4, "impulse")
        assert signal.size == 100
        assert signal[0] == 1.0
        assert not signal[1:].any()


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

    def test_a_nearest_node_receiver_has_one_weight_of_one(self, tmp_path: Path) -> None:
        """Snapped to a node, so no interpolation error and one row per receiver."""
        directory = write_grid(tmp_path / "entry", make_grid())
        receivers = np.array([[0.31, 0.42, 0.19], [0.2, 0.6, 0.1]])
        out = write_comms(
            directory,
            np.array([0.2, 0.3, 0.1]),
            receivers,
            0.005,
            out_path=tmp_path / "comms_out.h5",
            interpolation="nearest",
        )
        with h5py.File(out, "r") as handle:
            assert handle["out_alpha"].shape == (2, 1)
            assert np.array_equal(handle["out_alpha"][...], np.ones((2, 1)))
            assert handle["Nr"][()] == 2

    def test_the_nearest_node_is_the_nearest_node(self) -> None:
        grid = make_grid()
        coordinate, index = nearest_node(np.array([0.31, 0.42, 0.19]), grid)
        assert np.allclose(coordinate, [0.3, 0.4, 0.2])
        nx, ny, nz = grid.shape
        assert index == 3 * nz * ny + 4 * nz + 2
        # A point already on a node keeps it.
        assert np.allclose(nearest_node(np.array([0.2, 0.5, 0.1]), grid)[0], [0.2, 0.5, 0.1])

    def test_the_nearest_node_stays_on_the_fcc_subgrid(self) -> None:
        """Only nodes of even index sum exist on FCC, so an odd corner is not a node."""
        grid = make_grid(fcc_flag=1)
        nx, ny, nz = grid.shape
        for point in ([0.31, 0.42, 0.19], [0.11, 0.13, 0.09], [0.25, 0.25, 0.25]):
            _, index = nearest_node(np.array(point), grid)
            iz = index % nz
            iy = (index - iz) // nz % ny
            ix = ((index - iz) // nz - iy) // ny
            assert (ix + iy + iz) % 2 == 0

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
        """The roadmap says 'scene and grid step'; sim_mats.h5 says otherwise.

        Two specs, not two reads of one: ``key`` is cached per instance, so a
        file changed after the first read of an already-read spec would not
        be reflected -- the mutation has to happen before ``key`` is read at
        all, which a fresh second spec guarantees.
        """
        first_dir, second_dir = tmp_path / "first", tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first = self._spec(first_dir).key
        spec = self._spec(second_dir)
        (Path(spec.mat_folder) / "wall.h5").write_bytes(b"other impedance")
        assert first != spec.key

    def _model(
        self,
        *,
        tris: list[list[int]] | None = None,
        sides: list[int] | None = None,
        src: float = 1.0,
        stamp: str = "first export",
    ) -> str:
        """A model file shaped like the one the exporter actually writes."""
        return json.dumps(
            {
                "mats_hash": {
                    "wall": {
                        "pts": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                        "tris": tris if tris is not None else [[0, 1, 2]],
                        "sides": sides if sides is not None else [2],
                        "color": [128, 128, 128],
                    }
                },
                "sources": [{"xyz": [src, 1.0, 1.0], "name": "S1"}],
                "receivers": [{"xyz": [2.0, 1.0, 1.0], "name": "R1"}],
                "export_datetime": stamp,
            }
        )

    def test_the_key_follows_the_exported_mesh(self, tmp_path: Path) -> None:
        """Not the scene description: a moved triangle must miss the cache.

        The cache holds ``vox_out.h5``, which is a statement about where the
        boundary nodes are. If geometry could change under a fixed key, a run
        would be handed the previous scene's boundary and nothing would say so.
        """
        first = self._spec(tmp_path, body=self._model()).key
        moved = self._spec(tmp_path, body=self._model(tris=[[0, 2, 1]])).key
        assert first != moved

    def test_a_changed_sidedness_is_a_different_entry(self, tmp_path: Path) -> None:
        """Sidedness decides which nodes carry the material, so it is geometry."""
        first = self._spec(tmp_path, body=self._model()).key
        assert first != self._spec(tmp_path, body=self._model(sides=[3])).key

    def test_moving_the_source_reuses_the_voxelisation(self, tmp_path: Path) -> None:
        """The whole economics of W8: one grid, many source and receiver pairs.

        Sources and receivers live in the same file as the mesh but reach the
        solver through ``comms_out.h5``, which the cache deliberately does not
        hold. Keying on them would voxelise the same room again for every pair.
        """
        first = self._spec(tmp_path, body=self._model(src=1.0)).key
        assert first == self._spec(tmp_path, body=self._model(src=3.5)).key

    def test_re_exporting_the_same_geometry_keeps_the_entry(self, tmp_path: Path) -> None:
        """``export_datetime`` changes on every export and means nothing here."""
        first = self._spec(tmp_path, body=self._model(stamp="first export")).key
        assert first == self._spec(tmp_path, body=self._model(stamp="second export")).key

    def test_an_unparsable_model_is_hashed_whole(self, tmp_path: Path) -> None:
        """An unrecognised layout over-keys rather than guessing at its shape."""
        first = self._spec(tmp_path, body="not json at all").key
        assert first != self._spec(tmp_path, body="also not json").key


class TestUpload:
    def test_it_ships_only_what_the_engine_reads(self, tmp_path: Path) -> None:
        """cart_grid.h5 is pure bandwidth, and a scene file is a policy problem."""
        stray = tmp_path / "cart_grid.h5"
        stray.write_bytes(b"not for the engine")
        machine = Machine(host="example.invalid")
        with pytest.raises(ValueError, match="refusing"):
            upload(machine, [stray])


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

    scene_file = models / "room_only.json"
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


class TestDetachedEngine:
    """The engine is launched detached and polled, never held on one ssh channel.

    Three failures this shape exists to prevent, and all three have happened:
    an A100 destroyed mid-solve because ``nohup &`` over ssh never returns; a
    five hour run whose progress was only returned at the end; and an engine
    that exits with its own message on stdout while the exception carries
    stderr, so the caller is told the login banner and nothing else.
    """

    def test_the_launcher_detaches_every_descriptor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[str] = []

        def fake_run(argv: list[str], *, what: str, timeout: float | None = None) -> str:
            sent.append(argv[-1])
            return "1"

        monkeypatch.setattr(comms_remote, "_run", fake_run)
        comms_remote.start_engine(Machine(host="h", identity=None))
        launched = " ".join(sent)
        assert "setsid" in launched
        assert "< /dev/null" in launched
        assert "> " in launched and "2>&1" in launched
        assert "fdtd_main_gpu_single.x" in launched

    def test_progress_reads_the_percentage_the_engine_prints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: (
                "PROCS=1\nBYTES=0\nLAST=Running [42.3%] [02:51:07<06:44:12]\n"
            ),
        )
        progress = comms_remote.engine_progress(Machine(host="h", identity=None))
        assert progress.running is True
        assert progress.percent == pytest.approx(42.3)
        assert progress.finished is False

    def test_a_login_banner_on_stdout_does_not_shift_the_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vast's hosts print "Welcome to vast.ai ... Have fun!" on stdout.

        Read by position, the banner became the process count and the file
        size became the engine's last words, so a finished 67 minute solve was
        reported as an engine that had exited without writing anything.
        """
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: (
                "Welcome to vast.ai. If authentication fails, try again.\nHave fun!\n"
                "PROCS=0\nBYTES=237919552\nLAST=Running [100.0%] [01:07:00<00:00:00]\n"
            ),
        )
        progress = comms_remote.engine_progress(Machine(host="h", identity=None))
        assert progress.finished is True
        assert progress.output_bytes == 237919552

    def test_a_finished_run_is_no_process_and_an_output_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: "PROCS=0\nBYTES=1200000000\nLAST=done\n",
        )
        progress = comms_remote.engine_progress(Machine(host="h", identity=None))
        assert progress.finished is True
        assert progress.output_bytes == 1_200_000_000

    def test_an_engine_that_dies_without_output_is_reported_with_its_own_last_words(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that cost an hour of A100: CUDA error 209, said silently."""
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: (
                "PROCS=0\nBYTES=0\nLAST=Global memory allocation done\n"
            ),
        )
        with pytest.raises(RuntimeError, match="Global memory allocation done"):
            comms_remote.watch_engine(Machine(host="h", identity=None), poll_s=0.0)

    def test_the_latest_percentage_is_taken_and_not_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine overwrites one terminal line, so its log is carriage returns.

        Deleting them joins the whole log into a single line, and a search then
        returns its oldest percentage for ever, which reads exactly like a run
        that has stalled at 2.9 per cent.
        """
        blob = "Running [2.9%][00:04:44<02:46:12] Running [61.4%][01:44:00<01:05:00]"
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: f"PROCS=1\nBYTES=0\nLAST={blob}\n",
        )
        progress = comms_remote.engine_progress(Machine(host="h", identity=None))
        assert progress.percent == pytest.approx(61.4)

    def test_the_probe_translates_carriage_returns_rather_than_deleting_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[str] = []

        def record(argv: list[str], *, what: str, timeout: float | None = None) -> str:
            sent.append(argv[-1])
            return "PROCS=0\nBYTES=1\nLAST=x\n"

        monkeypatch.setattr(comms_remote, "_run", record)
        comms_remote.engine_progress(Machine(host="h", identity=None))
        assert "tr " in sent[0]
        assert "tr -d" not in sent[0]

    def test_a_percentage_that_stops_moving_is_stalled_and_not_merely_slow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            comms_remote,
            "_run",
            lambda argv, *, what, timeout=None: (
                "PROCS=1\nBYTES=0\nLAST=Running [11.0%] [00:10:00<01:00:00]\n"
            ),
        )
        with pytest.raises(RuntimeError, match="STALLED at 11.0"):
            comms_remote.watch_engine(Machine(host="h", identity=None), poll_s=0.0, stall_polls=3)
