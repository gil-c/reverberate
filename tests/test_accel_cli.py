"""The command line, end to end where a command is cheap and against fakes where it is not."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from reverberate.accel.cli import main
from test_accel_encode import a_grid, a_plan, pressure, small_array
from test_accel_scene import write_scene
from test_accel_voxelise import write_material


def an_entry(tmp_path: Path, name: str) -> Path:
    from reverberate.accel.voxelise import voxelise_scene

    model = write_scene(tmp_path / f"{name}.json")
    write_material(tmp_path / "wall.h5")
    voxelise_scene(
        model,
        tmp_path / name,
        mat_folder=tmp_path,
        mat_files={"wall": "wall.h5"},
        fmax=343.2 / (0.1 * 10.5),
        ppw=10.5,
        nh=4,
        xp=np,
    )
    return tmp_path / name


class TestCompareEntries:
    def test_an_entry_equals_itself_and_differs_from_a_changed_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = an_entry(tmp_path, "a")
        b = an_entry(tmp_path, "b")
        assert main(["compare-entries", str(a), str(b)]) == 0
        with h5py.File(b / "sim_consts.h5", "a") as handle:
            handle["Tc"][()] = 21.0
        assert main(["compare-entries", str(a), str(b)]) == 1
        assert "sim_consts.h5/Tc" in capsys.readouterr().out


class TestSelfcheck:
    def test_the_cpu_path_is_encoded_and_compared_with_the_card_s_file(
        self, tmp_path: Path
    ) -> None:
        """The self-check job as the campaign writes it, with the 'card' file made on numpy too."""
        from reverberate.accel.encode import encode_band

        grid = a_grid()
        design = small_array(grid, 50)
        rows: list[list[int] | None] = [[0, design.count]]
        plan = a_plan()
        plan["bands"]["mid"]["rows"] = rows
        plan["bands"]["mid"]["centres"] = [[float(v) for v in design.centre]]
        (tmp_path / "plan.json").write_text(json.dumps(plan))
        u = pressure(design.count, seed=3)
        with h5py.File(tmp_path / "sample.h5", "w") as handle:
            handle.create_dataset("u_out", data=u)
        with h5py.File(tmp_path / "comms.h5", "w") as handle:
            handle.create_dataset("out_alpha", data=np.ones((design.count, 1)))
            handle.create_dataset("diff", data=np.int8(1))
        np.save(tmp_path / "positions.npy", design.positions)
        encode_band(
            plan,
            "mid",
            {"name": "S1"},
            pressure=tmp_path / "sample.h5",
            comms=tmp_path / "comms.h5",
            positions=design.positions,
            output=tmp_path / "gpu.h5",
            xp=np,
        )
        job = {
            "plan": str(tmp_path / "plan.json"),
            "band": "mid",
            "source": {"name": "S1"},
            "pressure": str(tmp_path / "sample.h5"),
            "comms": str(tmp_path / "comms.h5"),
            "positions": str(tmp_path / "positions.npy"),
            "rows": rows,
            "gpu_encoded": str(tmp_path / "gpu.h5"),
            "output": str(tmp_path / "report.json"),
            "cpu_encoded": str(tmp_path / "cpu.h5"),
        }
        (tmp_path / "job.json").write_text(json.dumps(job))
        assert main(["selfcheck", "--job", str(tmp_path / "job.json")]) == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["points"] == 1 and report["float32_identical"] == 1
        assert report["max_over_peak"]["max"] == 0.0 and report["band"] == "mid"


class TestDispatch:
    def test_campaign_and_bundle_reach_their_functions_with_the_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            "reverberate.accel.campaign.run_campaign",
            lambda bundle, out, **kw: seen.update(kw, bundle=bundle),
        )
        assert (
            main(["campaign", "--bundle", "B", "--out", "O", "--cpu", "--solve-bands", "low,mid"])
            == 0
        )
        assert seen["gpu"] is False and seen["solve_bands"] == ("low", "mid")
        assert seen["bundle"] == Path("B") and seen["pffdtd_dir"] == Path("/root/pffdtd")
        (tmp_path / "sources.json").write_text("[]")
        monkeypatch.setattr(
            "reverberate.accel.bundle.prepare_bundle",
            lambda out, **kw: {
                "labels": ["x"],
                "points": 0,
                "pitch": kw["pitch_m"],
                "from": str(kw["models_from"]),
            },
        )
        assert (
            main(
                [
                    "bundle",
                    "--out",
                    "B",
                    "--hssd-root",
                    "H",
                    "--scene-id",
                    "S",
                    "--sources",
                    str(tmp_path / "sources.json"),
                    "--pitch",
                    "0.2",
                    "--models-from",
                    "M",
                ]
            )
            == 0
        )

    def test_voxelise_writes_an_entry(self, tmp_path: Path) -> None:
        model = write_scene(tmp_path / "box.json")
        write_material(tmp_path / "wall.h5")
        assert (
            main(
                [
                    "voxelise",
                    "--model",
                    str(model),
                    "--materials",
                    str(tmp_path),
                    "--fmax",
                    str(343.2 / (0.1 * 10.5)),
                    "--nh",
                    "4",
                    "--out",
                    str(tmp_path / "entry"),
                    "--cpu",
                    "--no-seal",
                ]
            )
            == 0
        )
        assert (tmp_path / "entry" / "vox_out.h5").is_file()
