"""The campaign: what it skips, what it retries, where it resumes, and the order it runs in.

The stages are faked with functions that leave on disk what the real ones
leave, so the driver's own logic is what runs: reading the state, choosing
the next stage, renting the encode boxes while the card is on its last band.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from reverberate.experiments.w40_volume_field.campaign import Campaign, audit_meshes
from reverberate.experiments.w40_volume_field.plan import COMMS_NAME

KEYS = {"low": "k1", "mid": "k4", "high": "k8"}
# The package re-exports the ``campaign`` function under the submodule's name,
# so the module itself is reached by import path, not by attribute.
driver = importlib.import_module("reverberate.experiments.w40_volume_field.campaign")


class FakeClient:
    def __init__(self, alive: set[int]) -> None:
        self.alive = alive

    def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
        return {"credit": 4.0}

    def instance(self, instance_id: int) -> object | None:
        return object() if instance_id in self.alive else None


def write_plan(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "scene_id": "104862621_172226772",
        "sources": [{"name": "S1"}],
        "bands": {"low": {}, "mid": {}, "high": {}},
        "points": [[0, 0, 0]],
        "pitch_m": 0.4,
    }
    (out / "plan.json").write_text(json.dumps(plan))


def write_comms(out: Path) -> None:
    for band in ("low", "mid", "high"):
        comms = out / "S1" / band / "comms" / COMMS_NAME
        comms.parent.mkdir(parents=True, exist_ok=True)
        comms.write_bytes(b"")


def write_encoded(out: Path) -> None:
    (out / "encoded").mkdir(exist_ok=True)
    for band in ("low", "mid", "high"):
        (out / "encoded" / f"S1__{band}.h5").write_bytes(b"")


def write_field(out: Path) -> None:
    (out / "field").mkdir(exist_ok=True)
    (out / "field" / "S1.h5").write_bytes(b"")


def write_audit(out: Path) -> None:
    for fmax in ("1000", "4000", "8000"):
        (out / "audit" / fmax / "voxels").mkdir(parents=True, exist_ok=True)
        (out / "audit" / fmax / "voxels" / "rooms.json").write_text("{}")


@pytest.fixture
def run(tmp_path: Path) -> Campaign:
    write_plan(tmp_path)
    return Campaign(tmp_path, FakeClient(alive={50811926}), attempts=3, pause_s=0.0)


class TestState:
    def test_the_state_is_read_off_the_disk_and_nothing_else(self, run: Campaign) -> None:
        assert not run.comms_ready() and not run.encoded_ready() and not run.field_ready()
        assert run.solve_state() == {}
        write_comms(run.out)
        write_encoded(run.out)
        assert run.comms_ready() and run.encoded_ready() and not run.field_ready()
        write_field(run.out)
        assert run.field_ready()

    def test_a_solve_whose_instance_is_gone_starts_over(self, run: Campaign) -> None:
        """Its pressure went with it; resuming on it would wait forever."""
        (run.out / "solve.json").write_text(json.dumps({"instance": 50822274, "solves": [1]}))
        assert run.solve_state() == {}
        (run.out / "solve.json").write_text(json.dumps({"instance": 50811926, "solves": [1]}))
        assert run.solve_state()["instance"] == 50811926


class TestStage:
    def test_a_stage_is_retried_and_the_log_names_every_attempt(self, run: Campaign) -> None:
        calls: list[int] = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("proxy closed")
            return "ok"

        assert run.stage("solve", flaky) == "ok"
        log = run.log_path.read_text()
        assert log.count("solve: start") == 3 and log.count("FAILED proxy closed") == 2
        assert "credit 4.00 USD" in log

    def test_the_last_failure_is_raised_not_swallowed(self, run: Campaign) -> None:
        def broken() -> None:
            raise SystemExit("no offer produced a machine that answered")

        with pytest.raises(SystemExit):
            run.stage("encode", broken)
        assert run.log_path.read_text().count("encode: start") == 3


class TestAuditMeshes:
    def test_only_payloads_that_exist_are_named_relative_to_the_run(self, tmp_path: Path) -> None:
        """The app refuses a mesh of another scene, and the first campaign's
        hard-coded paths drew hssd_0002's walls around hssd_0076: a mesh is
        named only when this campaign built it."""
        audit = tmp_path / "audit"
        (audit / "1000" / "voxels").mkdir(parents=True)
        (audit / "1000" / "voxels" / "rooms.json").write_text("{}")
        (audit / "4000" / "voxels").mkdir(parents=True)  # built nothing
        assert audit_meshes(tmp_path, audit) == {"1000": "audit/1000/voxels"}
        assert audit_meshes(tmp_path, tmp_path / "elsewhere") == {}


class FakeStages:
    """Every stage of a campaign as a fake that leaves the real stage's state on disk."""

    def __init__(self, out: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool) -> None:
        self.out = out
        self.calls: list[str] = []
        self.cached = cached
        out.mkdir(parents=True, exist_ok=True)
        (out / "sources.json").write_text("[]")
        if cached:
            (out / "models").mkdir(exist_ok=True)
            (out / "models" / "manifest.json").write_text("{}")

        def export(hssd_root: Path, scene_id: str, models: Path) -> Path:
            self.calls.append("export")
            models.mkdir(exist_ok=True)
            (models / "manifest.json").write_text("{}")
            return models

        def voxelise(models: Path, scratch: Path, *, hours: float, say: Any) -> None:
            self.calls.append("voxelise")
            self.cached = True

        def audit(
            keys: dict[str, str], hssd_root: Path, scene_id: str, audit: Path, say: Any
        ) -> None:
            self.calls.append("audit")
            write_audit(out)

        def plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append("plan")
            write_plan(out)
            return {}

        def prepare(out_: Path) -> list[Path]:
            self.calls.append("prepare")
            write_comms(out)
            return []

        def solve(out_: Path, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(f"solve(instance={kwargs['instance']})")
            (out / "solve.json").write_text(
                json.dumps({"instance": 50811926, "solves": [], "complete": True})
            )
            return {"instance": 50811926}

        def rent(out_: Path, **kwargs: Any) -> list[tuple[str, int]]:
            self.calls.append("rent encode boxes")
            return [("box", 1), ("box", 2)]

        def encode(out_: Path, **kwargs: Any) -> list[Path]:
            self.calls.append(f"encode(boxes={len(kwargs['boxes'] or [])})")
            write_encoded(out)
            return []

        def assemble(out_: Path, **kwargs: Any) -> list[Path]:
            self.calls.append(f"assemble(meshes={sorted(kwargs['meshes'] or {})})")
            write_field(out)
            return []

        monkeypatch.setattr(driver, "export_scene", export)
        monkeypatch.setattr(driver, "storey_keys", lambda models: KEYS)
        monkeypatch.setattr(driver, "grids_cached", lambda keys: self.cached)
        monkeypatch.setattr(driver, "voxelise_storey", voxelise)
        monkeypatch.setattr(driver, "build_audit", audit)
        monkeypatch.setattr(driver, "plan_field", plan)
        monkeypatch.setattr(driver, "prepare_field", prepare)
        monkeypatch.setattr(driver, "solve_on_card", solve)
        monkeypatch.setattr(driver, "rent_encode_boxes", rent)
        monkeypatch.setattr(driver, "encode_sharded", encode)
        monkeypatch.setattr(driver, "assemble_field", assemble)
        monkeypatch.setattr(driver, "POLL_S", 0.01)

        from reverberate import auth
        from reverberate.gpu import vast

        monkeypatch.setattr(auth, "inject", lambda names: None)
        monkeypatch.setattr(vast, "VastClient", lambda timeout: FakeClient(alive={50811926}))

    def go(self) -> list[str]:
        driver.campaign(
            self.out,
            hssd_root=self.out / "hssd",
            scene_id="104862621_172226772",
            sources_file=self.out / "sources.json",
            models=self.out / "models",
            ram_gb=400.0,
            pitch_m=0.4,
            solve_hours=4.0,
            solve_max_dph=2.1,
            slice_ram_gb=180.0,
            encode_hours=8.0,
            encode_max_dph=0.45,
            min_cores=12,
            min_cpu_ghz=3.0,
            shards=2,
            attempts=1,
        )
        return self.calls


class TestCampaignOrder:
    def test_from_nothing_every_stage_runs_once_in_order_and_the_boxes_are_rented_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stages = FakeStages(tmp_path, monkeypatch, cached=False)
        calls = stages.go()
        assert calls[:6] == [
            "export",
            "voxelise",
            "audit",
            "plan",
            "prepare",
            "solve(instance=None)",
        ]
        assert "rent encode boxes" in calls, "rented while the card was still up"
        assert calls[-2:] == ["encode(boxes=2)", "assemble(meshes=['1000', '4000', '8000'])"]
        assert len(calls) == 9, "every stage exactly once"

    def test_a_finished_run_touches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_plan(tmp_path)
        write_comms(tmp_path)
        write_encoded(tmp_path)
        write_field(tmp_path)
        write_audit(tmp_path)
        stages = FakeStages(tmp_path, monkeypatch, cached=True)
        assert stages.go() == []

    def test_a_run_solved_on_a_live_card_skips_the_solve_and_encodes_from_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The morning case: the card is up with the pressure, the encode is what is left."""
        write_plan(tmp_path)
        write_comms(tmp_path)
        write_audit(tmp_path)
        (tmp_path / "solve.json").write_text(
            json.dumps({"instance": 50811926, "solves": [], "complete": True})
        )
        stages = FakeStages(tmp_path, monkeypatch, cached=True)
        calls = stages.go()
        assert calls == ["encode(boxes=0)", "assemble(meshes=['1000', '4000', '8000'])"], calls
