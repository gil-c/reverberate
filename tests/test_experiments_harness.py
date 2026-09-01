"""Offline tests for the promoted experiment harness.

No HSSD, no PFFDTD, no GPU: the solver is a stub script and the scenes are
three triangles. What is under test is the part that made W1's first attempt
wrong, which is bookkeeping rather than physics -- which domain a run is given,
whether that choice is recorded, and whether the comparison calls a bit
identical pair bit identical.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.experiments import compare as compare_mod
from reverberate.experiments import engine as engine_mod
from reverberate.experiments import run as run_mod
from reverberate.experiments import scene_export


def _model(points: np.ndarray) -> dict[str, Any]:
    return {
        "mats_hash": {
            "Wall": {
                "pts": points.tolist(),
                "tris": [[0, 1, 2]],
                "sides": [3],
                "color": [128, 128, 128],
            }
        }
    }


def test_scene_bounds_reads_vertices_not_the_manifest(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(_model(points)))
    lo, hi = run_mod.scene_bounds(path)
    assert lo.tolist() == [0.0, 0.0, 0.0]
    assert hi.tolist() == [2.0, 1.0, 3.0]


def test_common_bounds_are_the_reference_box_exactly() -> None:
    ref_lo, ref_hi = np.zeros(3), np.array([10.0, 3.0, 8.0])
    bounds = run_mod.choose_bounds(
        "common", np.array([4.0, 0.5, 4.0]), np.array([5.0, 1.0, 5.0]), ref_lo, ref_hi, 0.1, 5.0
    )
    assert bounds.mode == "common"
    assert bounds.pad_m is None
    assert bounds.bmin.tolist() == ref_lo.tolist()
    assert bounds.bmax.tolist() == ref_hi.tolist()


def test_padded_bounds_stay_on_the_reference_grid_and_inside_it() -> None:
    ref_lo, ref_hi = np.zeros(3), np.array([10.0, 3.0, 8.0])
    step = run_mod.grid_step(2000.0, 10.5)
    bounds = run_mod.choose_bounds(
        "padded", np.array([4.0, 0.5, 4.0]), np.array([5.0, 1.0, 5.0]), ref_lo, ref_hi, step, 1.0
    )
    assert bounds.mode == "padded"
    assert bounds.pad_m == 1.0
    offset = (bounds.bmin - ref_lo) / step
    assert np.allclose(offset, np.round(offset))
    assert np.all(bounds.bmin >= ref_lo - step)
    assert np.all(bounds.bmax <= ref_hi + 2.0 * step)


def test_bounds_mode_must_be_one_of_the_two() -> None:
    with pytest.raises(ValueError, match="unknown bounds mode"):
        run_mod.choose_bounds(
            "snapped",  # type: ignore[arg-type]
            np.zeros(3),
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            0.1,
            1.0,
        )


def test_bounds_mode_has_no_default_on_the_command_line() -> None:
    parser_error = SystemExit
    with pytest.raises(parser_error):
        run_mod.main(
            [
                "scene",
                "--models",
                "m",
                "--out",
                "o",
                "--scene",
                "s",
                "--fmax",
                "2000",
                "--duration",
                "0.1",
            ]
        )


def test_grid_step_matches_pffdtd_sim_consts() -> None:
    assert run_mod.grid_step(2000.0, 10.5) == pytest.approx(343.2 / (2000.0 * 10.5))


def _stub_engine(root: Path, *, exit_code: int = 0) -> Path:
    """A PFFDTD checkout that is only a directory layout and one shell script."""
    (root / "python").mkdir(parents=True, exist_ok=True)
    (root / "python" / "sim_setup.py").write_text("")
    binaries = root / "c_cuda"
    binaries.mkdir(parents=True, exist_ok=True)
    binary = binaries / "fdtd_main_cpu_single.x"
    binary.write_text(f"#!/bin/sh\necho ran\nexit {exit_code}\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return root


def test_run_binary_logs_and_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFFDTD_DIR", str(_stub_engine(tmp_path / "pffdtd")))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = engine_mod.run_binary(run_dir, "cpu", double_precision=False)
    assert result.engine_s >= 0.0
    assert "ran" in result.log.read_text()


def test_run_binary_raises_and_still_leaves_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PFFDTD_DIR", str(_stub_engine(tmp_path / "pffdtd", exit_code=3)))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="engine failed"):
        engine_mod.run_binary(run_dir, "cpu", double_precision=False)
    assert (run_dir / "engine.log").is_file()


def test_engine_binary_says_how_to_build_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PFFDTD_DIR", str(_stub_engine(tmp_path / "pffdtd")))
    with pytest.raises(FileNotFoundError, match="is not built"):
        engine_mod.engine_binary("gpu", double_precision=True)


def _write_run(run_dir: Path, signal: np.ndarray, sample_rate: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(run_dir / "sim_outs.h5", "w") as handle:
        handle["u_out"] = signal.reshape(1, -1)
    with h5py.File(run_dir / "sim_consts.h5", "w") as handle:
        handle["SR"] = sample_rate
        handle["h"] = 0.016


def test_sim_consts_reads_both_constants(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", np.zeros(4), 48000.0)
    constants = engine_mod.sim_consts(tmp_path / "run")
    assert constants.sample_rate == 48000.0
    assert constants.h == pytest.approx(0.016)


def test_write_record_round_trips(tmp_path: Path) -> None:
    path = engine_mod.write_record(tmp_path, "result.json", {"scene": "a", "engine_s": 1.0})
    assert json.loads(path.read_text()) == {"scene": "a", "engine_s": 1.0}


def _result(out: Path, scene: str, cut_m: float | None, run_dir: Path) -> dict[str, Any]:
    return {
        "scene": scene,
        "fmax": 500.0,
        "ppw": 10.5,
        "bounds_mode": "common",
        "cut_m": cut_m,
        "pad_m": None,
        "grid_points": 1000,
        "engine_s": 1.0,
        "run_dir": str(run_dir),
    }


def test_compare_calls_a_bit_identical_pair_bit_identical(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(400)
    out = tmp_path / "out"
    out.mkdir()
    lines = []
    for name, cut in (("apartment_full", None), ("apartment_cut10m", 10.0)):
        run_dir = out / name
        _write_run(run_dir, signal, 48000.0)
        lines.append(json.dumps(_result(out, name, cut, run_dir)))
    (out / run_mod.RESULTS_FILE).write_text("\n".join(lines) + "\n")

    report = compare_mod.compare(out)
    assert len(report) == 1
    assert report[0]["bit_exact_throughout"] is True
    assert report[0]["first_inexact_ms"] is None
    assert report[0]["residual_full_window"]["peak_db"] is None
    assert json.loads((out / "comparison.json").read_text()) == report


def test_compare_finds_the_first_differing_sample(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(400)
    other = reference.copy()
    other[100] += 1e-9
    out = tmp_path / "out"
    out.mkdir()
    _write_run(out / "apartment_full", reference, 48000.0)
    _write_run(out / "apartment_cut10m", other, 48000.0)
    (out / run_mod.RESULTS_FILE).write_text(
        json.dumps(_result(out, "apartment_full", None, out / "apartment_full"))
        + "\n"
        + json.dumps(_result(out, "apartment_cut10m", 10.0, out / "apartment_cut10m"))
        + "\n"
    )
    report = compare_mod.compare(out)
    assert report[0]["first_inexact_ms"] == pytest.approx(1000.0 * 100 / 48000.0, abs=1e-4)
    assert report[0]["bit_exact_throughout"] is False


def test_compare_refuses_runs_that_disagree_on_the_sample_rate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _write_run(out / "apartment_full", np.zeros(10), 48000.0)
    _write_run(out / "apartment_cut10m", np.zeros(10), 44100.0)
    (out / run_mod.RESULTS_FILE).write_text(
        json.dumps(_result(out, "apartment_full", None, out / "apartment_full"))
        + "\n"
        + json.dumps(_result(out, "apartment_cut10m", 10.0, out / "apartment_cut10m"))
        + "\n"
    )
    with pytest.raises(ValueError, match="sample rate"):
        compare_mod.compare(out)


def test_a_rerun_supersedes_the_earlier_run_of_the_same_configuration(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    stale = _result(out, "apartment_cut10m", 10.0, out / "stale")
    fresh = _result(out, "apartment_cut10m", 10.0, out / "fresh")
    (out / run_mod.RESULTS_FILE).write_text(json.dumps(stale) + "\n\n" + json.dumps(fresh) + "\n")
    latest = compare_mod._load_results(out / run_mod.RESULTS_FILE)
    assert latest[("apartment_cut10m", 500.0, 10.5, "common")]["run_dir"] == str(out / "fresh")


def test_path_length_bound_keeps_a_triangle_the_direct_path_touches() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    near = scene_export.path_length_bound(mesh, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    far = scene_export.path_length_bound(
        mesh, np.array([50.0, 0.0, 0.0]), np.array([50.0, 0.0, 1.0])
    )
    assert near[0] < far[0]
    assert near[0] <= 1.0


def test_material_table_covers_pffdtds_eleven_bands() -> None:
    assert scene_export.BANDS.shape == (11,)
    assert scene_export.BANDS[0] == 16.0
    assert scene_export.BANDS[-1] == 16000.0


def test_the_harness_never_imports_pffdtd_into_this_interpreter() -> None:
    """PFFDTD needs numpy below 2; crossing that line is a subprocess, always.

    ``b0_run.py`` did the opposite: it put PFFDTD's ``python`` directory on
    ``sys.path`` at import time and called ``sim_setup`` in process, which
    pinned the whole experiment to that interpreter's dependencies.
    """
    for module in (run_mod, compare_mod, engine_mod, scene_export):
        assert module.__name__ in sys.modules
    for pffdtd_module in ("sim_setup", "materials.adm_funcs", "voxelizer", "fdtd"):
        assert pffdtd_module not in sys.modules
    assert int(np.__version__.split(".")[0]) >= 2
