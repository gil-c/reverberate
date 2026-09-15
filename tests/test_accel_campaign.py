"""The machine-side driver and the laptop-side bundle: order, resumption, what comes home.

The stages are faked with functions that leave on disk what the real ones
leave, so what runs is the driver's own logic: reading the bundle, skipping
what exists, writing the status the watcher reads, and marking the end.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.accel.bundle import take_home
from reverberate.accel.campaign import Campaign, slice_rows


class TestSliceRows:
    def test_rows_inside_the_slice_are_shifted_and_the_rest_dropped(self) -> None:
        rows = [[0, 5], [5, 12], None, [12, 20], [20, 30]]
        assert slice_rows(rows, 5, 20) == [None, [0, 7], None, [7, 15], None]

    def test_a_row_cut_by_the_slice_is_not_taken(self) -> None:
        assert slice_rows([[0, 5], [5, 12]], 0, 10) == [[0, 5], None]


def write_bundle(bundle: Path, bands: tuple[str, ...] = ("low", "mid", "high")) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "models" / "storey").mkdir(parents=True)
    (bundle / "models" / "materials").mkdir(parents=True)
    spec = {
        "scene_id": "104862621_172226772",
        "dwelling": "hssd_0076",
        "models": "models/storey",
        "model_json": "models/storey/apartment_full.json",
        "storey_scene": "apartment_full",
        "materials": "models/materials",
        "bands": {
            band: {"fmax_hz": f, "duration_s": d, "cache_key": f"k{band}", "nh": 8}
            for band, f, d in (("low", 1000.0, 1.2), ("mid", 4000.0, 0.4), ("high", 8000.0, 0.15))
            if band in bands
        },
        "ppw": 10.5,
        "tc": 20.0,
        "rh": 50.0,
        "pitch_m": 0.4,
        "height_m": 1.7,
        "ram_gb": 400.0,
        "order": 7,
        "fit_order": 10,
        "points": 2,
        "labels": ["living room", "living room"],
        "free_floor_m2": 10.0,
        "sources": [{"name": "S1", "room": "living room", "position": [0.0, 1.7, 0.0]}],
    }
    (bundle / "campaign.json").write_text(json.dumps(spec))
    np.save(bundle / "points.npy", np.array([[0.0, 1.7, 0.0], [0.4, 1.7, 0.0]]))
    (bundle / "rooms.json").write_text("[]")


class Driven(Campaign):
    """The real driver with every stage faked, leaving what the stage leaves."""

    calls: list[str]

    def __post_init__(self) -> None:
        self.calls = []
        self.gpu = False
        super().__post_init__()

    def voxelise(self) -> dict[str, Any]:
        self.calls.append("voxelise")
        return {"low": {"cached": False}}

    def audit(self) -> dict[str, str]:
        self.calls.append("audit")
        return {"1000": "audit/1000/voxels"}

    def plan(self) -> dict[str, Any]:
        self.calls.append("plan")
        plan = {"points": [[0, 1.7, 0], [0.4, 1.7, 0]], "bands": {"low": {}}, "sources": []}
        (self.out / "plan.json").write_text(json.dumps(plan))
        return plan

    def solve_and_encode(self, plan: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("solve")
        return {"S1__low": {"engine_s": 1.0}}

    def assemble(self, meshes: dict[str, str]) -> list[Path]:
        self.calls.append("assemble")
        (self.out / "field").mkdir(exist_ok=True)
        (self.out / "field" / "S1.h5").write_bytes(b"field")
        return [self.out / "field" / "S1.h5"]


class TestTheDriver:
    def test_every_stage_runs_in_order_and_the_run_is_marked_done(self, tmp_path: Path) -> None:
        write_bundle(tmp_path / "bundle")
        run = Driven(bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path)
        report = run.run()
        assert run.calls == ["voxelise", "audit", "plan", "solve", "assemble"]
        assert (tmp_path / "out" / "campaign.done").is_file()
        assert not (tmp_path / "out" / "campaign.failed").exists()
        status = json.loads((tmp_path / "out" / "status.json").read_text())
        assert status["stage"] == "done"
        assert set(report["timings_s"]) == {"voxelise", "audit", "plan", "solve", "assemble"}
        assert "voxelise" in (tmp_path / "out" / "campaign.log").read_text()
        assert json.loads((tmp_path / "out" / "report.json").read_text())["dwelling"] == "hssd_0076"

    def test_a_failing_stage_marks_the_run_failed_with_the_error(self, tmp_path: Path) -> None:
        write_bundle(tmp_path / "bundle")

        class Broken(Driven):
            def plan(self) -> dict[str, Any]:
                raise RuntimeError("no grid for the arrays")

        run = Broken(bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path)
        with pytest.raises(RuntimeError):
            run.run()
        assert "no grid" in (tmp_path / "out" / "campaign.failed").read_text()
        assert json.loads((tmp_path / "out" / "status.json").read_text())["stage"] == "failed"

    def test_the_bundle_s_keys_and_bands_are_read(self, tmp_path: Path) -> None:
        write_bundle(tmp_path / "bundle", bands=("low", "mid"))
        run = Driven(bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path)
        assert run.keys == {"low": "klow", "mid": "kmid"}
        assert run.fmax_hz == {"low": 1000.0, "mid": 4000.0}
        assert run.durations_s["mid"] == 0.4
        assert run.model_json.name == "apartment_full.json"


class TestTakeHome:
    def test_the_run_comes_home_and_new_grids_are_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        pulled = tmp_path / "pulled"
        (pulled / "field").mkdir(parents=True)
        (pulled / "field" / "S1.h5").write_bytes(b"x")
        (pulled / "walk.json").write_text("{}")
        (pulled / "scratch.bin").write_bytes(b"not taken")
        entries = tmp_path / "cache"
        for key in ("k1", "k2"):
            (entries / key).mkdir(parents=True)
            for name in ("sim_consts.h5", "vox_out.h5", "sim_mats.h5", "cart_grid.h5"):
                with h5py.File(entries / key / name, "w") as handle:
                    handle.create_dataset("x", data=1)
            (entries / key / "manifest.json").write_text(json.dumps({"key": key}))
        (entries / "broken").mkdir()
        (entries / "broken" / "manifest.json").write_text("{}")
        from reverberate.wave.voxelise import cache_root

        (cache_root() / "k2").mkdir(parents=True)
        (cache_root() / "k2" / "manifest.json").write_text("{}")
        report = take_home(pulled, tmp_path / "run", cache_entries=entries)
        assert report["copied"] == ["field", "walk.json"]
        assert (tmp_path / "run" / "field" / "S1.h5").read_bytes() == b"x"
        assert not (tmp_path / "run" / "scratch.bin").exists()
        assert report["installed"] == ["k1"]
        assert (cache_root() / "k1" / "vox_out.h5").is_file()


class TestTheStagesWithMocks:
    """Each real stage against fakes of what it calls: the driver's own logic."""

    def test_voxelise_skips_an_installed_grid_and_refuses_a_key_the_laptop_did_not_compute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reverberate.experiments.run as run_module
        from reverberate.accel import campaign as module

        write_bundle(tmp_path / "bundle", bands=("low", "mid"))
        run = Campaign(
            bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path, gpu=False
        )
        keys = {"low": "klow", "mid": "kmid"}

        class Spec:
            def __init__(self, key: str) -> None:
                self.key = key

        monkeypatch.setattr(
            run_module,
            "scene_spec",
            lambda models, scene, fmax: (Spec(keys["low"] if fmax == 1000.0 else "other"), None, 0),
        )

        class Entry:
            complete = True

        vox_module = importlib.import_module("reverberate.wave.voxelise")
        monkeypatch.setattr(vox_module, "entry_for", lambda scene: Entry())
        with pytest.raises(RuntimeError, match="disagree"):
            run.voxelise()
        (tmp_path / "out" / "campaign.log").unlink()
        monkeypatch.setattr(
            run_module,
            "scene_spec",
            lambda models, scene, fmax: (Spec(keys["low" if fmax == 1000.0 else "mid"]), None, 0),
        )
        assert run.voxelise() == {"low": {"cached": True}, "mid": {"cached": True}}
        assert "already installed" in (tmp_path / "out" / "campaign.log").read_text()
        assert module.slice_rows([[0, 1]], 0, 1) == [[0, 1]]

    def test_audit_hands_the_bundle_s_rooms_to_the_view_and_needs_no_hssd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reverberate.experiments.audit_view as view

        write_bundle(tmp_path / "bundle", bands=("low",))
        (tmp_path / "bundle" / "rooms.json").write_text(
            json.dumps(
                [
                    {
                        "name": "living room",
                        "label": "living room",
                        "regions": ["living room"],
                        "outdoor": False,
                        "polygon": "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
                    }
                ]
            )
        )
        run = Campaign(
            bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path, gpu=False
        )
        calls: list[dict[str, Any]] = []

        def fake_build(
            key: str, hssd_root: Path | None, scene_id: str, target: Path, **kw: Any
        ) -> dict[str, Any]:
            calls.append({"key": key, "hssd_root": hssd_root, "scene_id": scene_id, **kw})
            (target / "voxels").mkdir(parents=True)
            (target / "voxels" / "rooms.json").write_text("{}")
            return {"rooms": ["living room"]}

        monkeypatch.setattr(view, "build", fake_build)
        meshes = run.audit()
        assert meshes == {"1000": "audit/1000/voxels"}
        assert calls[0]["key"] == "klow" and calls[0]["hssd_root"] is None
        assert calls[0]["coarse_span"] == 1
        assert [r.name for r in calls[0]["partition_rooms"]] == ["living room"]
        # A second call finds the payload and builds nothing.
        assert run.audit() == meshes and len(calls) == 1

    def test_plan_is_read_back_when_it_exists_and_placed_from_the_bundle_otherwise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import reverberate.experiments.w40_volume_field.plan as plan_module

        write_bundle(tmp_path / "bundle")
        run = Campaign(
            bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path, gpu=False
        )
        seen: dict[str, Any] = {}

        def fake_plan_arrays(out: Path, **kw: Any) -> dict[str, Any]:
            seen.update(kw)
            return {"points": kw["points_and_labels"][0].tolist(), "bands": {}}

        monkeypatch.setattr(plan_module, "plan_arrays", fake_plan_arrays)
        plan = run.plan()
        assert len(plan["points"]) == 2
        assert seen["storey_keys"] == {"low": "klow", "mid": "kmid"} and seen["high_key"] == "khigh"
        assert seen["area"] == 10.0 and seen["rooms"] is None and seen["pitch_m"] == 0.4
        assert seen["points_and_labels"][1] == ["living room", "living room"]
        (tmp_path / "out" / "plan.json").write_text(
            json.dumps({"points": [[0, 0, 0]], "bands": {}})
        )
        assert run.plan() == {"points": [[0, 0, 0]], "bands": {}}

    def test_solve_sizes_the_output_by_the_engine_and_merges_the_slices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The solve stage with the engine and the encoder faked: slices, parts, the merge."""
        import reverberate.experiments.run as run_module
        from reverberate.accel import campaign as module

        write_bundle(tmp_path / "bundle", bands=("low",))
        (tmp_path / "c_cuda").mkdir()
        (tmp_path / "c_cuda" / "fdtd_data.h").write_text("Real *u_out; //REVERBERATE PATCH 8\n")
        run = Campaign(
            bundle=tmp_path / "bundle",
            out=tmp_path / "out",
            pffdtd_dir=tmp_path,
            gpu=False,
            selfcheck_points=0,
        )
        rows: list[list[int] | None] = [[0, 10], [10, 20]]
        band_spec = {
            "cache_key": "klow",
            "rows": rows,
            "samples": 249_999_999,
            "sample_rate_hz": 1000.0,
            "cost": {"receivers": 20, "grid_points": 100.0, "estimated_gpu_s": 1.0},
        }
        plan = {
            "sources": [{"name": "S1", "room": "living room", "position": [0.0, 1.7, 0.0]}],
            "bands": {"low": band_spec, "mid": band_spec, "high": band_spec},
        }
        run.solve_bands = ("low",)
        np.save(tmp_path / "out" / "array_positions_low.npy", np.zeros((20, 3)))

        class Entry:
            path = tmp_path / "entry"

        monkeypatch.setattr(run_module, "entry_from_key", lambda key: Entry())
        # 20 receivers of 250 million float32 samples: 20 GB; a host with 60 % of 2.2 x that.
        monkeypatch.setattr(module, "host_memory_gb", lambda: 8.0 + 2.2 * 20.0 * 0.6)
        seen: dict[str, Any] = {}

        def fake_solve_slices(**kw: Any) -> list[dict[str, Any]]:
            seen.update(kw)
            records = []
            for k in range(2):
                sim_outs = tmp_path / f"sim{k}.h5"
                sim_outs.write_bytes(b"p")
                consumed = kw["consume"](k, k * 10, (k + 1) * 10, sim_outs, tmp_path / "comms.h5")
                records.append(
                    {"slice": k, "engine_s": 5.0, "consume_s": 1.0, "consumed": consumed}
                )
            return records

        monkeypatch.setattr(module, "solve_slices", fake_solve_slices)

        def fake_encode_band(
            plan: Any,
            band: str,
            source: Any,
            *,
            output: Path,
            rows: Any,
            row_offset: int,
            **kw: Any,
        ) -> dict[str, Any]:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"rows": rows, "offset": row_offset}))
            return {"points": sum(r is not None for r in rows)}

        monkeypatch.setattr("reverberate.accel.encode.encode_band", fake_encode_band)
        merged_from: list[str] = []

        def fake_merge(parts: list[Path], merged: Path) -> None:
            merged_from.extend(p.name for p in parts)
            merged.write_text("merged")

        monkeypatch.setattr("reverberate.accel.encode.merge_encoded", fake_merge)
        records = run.solve_and_encode(plan)
        # The engine writes 4 bytes a sample and one step more than the plan.
        assert seen["output_bytes"] == 20 * 250_000_000 * 4
        assert seen["duration_s"] == 249_999.999 and seen["grid_points"] == 100.0
        # Two slices: RAM held 60 % of 2.2 x the output.
        assert run.status["slices"] == 2
        assert merged_from == ["S1__low.part0.h5", "S1__low.part1.h5"]
        assert not list((tmp_path / "out" / "encoded").glob("*.part*"))
        assert (tmp_path / "out" / "encoded" / "S1__low.h5").read_text() == "merged"
        assert records["S1__low"]["engine_s"] == 10.0 and records["S1__low"]["encode_s"] == 2.0
        assert not (tmp_path / "sim0.h5").exists(), "the pressure is deleted once it is signals"
        assert run.status["stage_detail"] == "engine"
        # A merged band is not solved again.
        assert run.solve_and_encode(plan) == {}


class TestSelfcheckLaunch:
    def test_the_sample_the_comms_and_the_job_are_written_and_the_checker_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_bundle(tmp_path / "bundle", bands=("low",))
        run = Campaign(
            bundle=tmp_path / "bundle", out=tmp_path / "out", pffdtd_dir=tmp_path, gpu=False
        )
        rows: list[list[int] | None] = [[0, 3], None, [3, 7], [7, 9]]
        plan = {"bands": {"low": {"rows": rows}}}
        (tmp_path / "out" / "plan.json").write_text(json.dumps(plan))
        np.save(tmp_path / "out" / "array_positions_low.npy", np.arange(27.0).reshape(9, 3))
        with h5py.File(tmp_path / "sim_outs.h5", "w") as handle:
            handle.create_dataset("u_out", data=np.arange(9 * 5, dtype=np.float32).reshape(9, 5))
        with h5py.File(tmp_path / "comms.h5", "w") as handle:
            handle.create_dataset("out_alpha", data=np.arange(9.0).reshape(9, 1))
            handle.create_dataset("diff", data=np.int8(1))
        started: list[list[str]] = []

        class Process:
            pid = 4242
            returncode = 0

            def __init__(self, args: list[str], **kwargs: Any) -> None:
                started.append(args)
                assert kwargs["env"]["REVERBERATE_NO_GPU"] == "1"

            def wait(self, timeout: float | None = None) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", Process)
        run.launch_selfcheck(
            "S1__low",
            "low",
            {"name": "S1"},
            tmp_path / "sim_outs.h5",
            tmp_path / "comms.h5",
            0,
            9,
            tmp_path,
        )
        folder = tmp_path / "out" / "selfcheck"
        job = json.loads((folder / "S1__low.job.json").read_text())
        # The first two placed points: rows 0:3 and 3:7 of the slice, re-based in the sample.
        assert job["rows"] == [[0, 3], None, [3, 7], None]
        with h5py.File(job["pressure"]) as handle:
            assert handle["u_out"].shape == (7, 5)
            assert np.array_equal(handle["u_out"][3], np.arange(15, 20))
        with h5py.File(job["comms"]) as handle:
            assert list(handle["out_alpha"][:, 0]) == [0, 1, 2, 3, 4, 5, 6]
        assert np.load(job["positions"]).shape == (7, 3)
        assert job["gpu_encoded"].endswith("encoded/S1__low.h5")
        assert started[0][-2:] == ["--job", str(folder / "S1__low.job.json")]
        (folder / "S1__low.json").write_text(json.dumps({"max_over_peak": {"max": 1e-10}}))
        assert run.wait_selfchecks()["S1__low"]["max_over_peak"]["max"] == 1e-10
