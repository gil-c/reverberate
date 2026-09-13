"""The campaign's state: what it skips, what it retries, and where it resumes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverberate.experiments.w40_volume_field.campaign import Campaign
from reverberate.experiments.w40_volume_field.plan import COMMS_NAME


class FakeClient:
    def __init__(self, alive: set[int]) -> None:
        self.alive = alive

    def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
        return {"credit": 4.0}

    def instance(self, instance_id: int) -> object | None:
        return object() if instance_id in self.alive else None


@pytest.fixture
def run(tmp_path: Path) -> Campaign:
    plan = {
        "sources": [{"name": "S1"}],
        "bands": {"low": {}, "mid": {}, "high": {}},
        "points": [[0, 0, 0]],
        "pitch_m": 0.4,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    return Campaign(tmp_path, FakeClient(alive={50811926}), attempts=3, pause_s=0.0)


class TestState:
    def test_nothing_is_ready_on_an_empty_run(self, run: Campaign) -> None:
        assert not run.comms_ready() and not run.encoded_ready() and not run.field_ready()
        assert run.solve_state() == {}

    def test_comms_encoded_and_field_are_read_off_the_disk(self, run: Campaign) -> None:
        for band in ("low", "mid", "high"):
            comms = run.out / "S1" / band / "comms" / COMMS_NAME
            comms.parent.mkdir(parents=True)
            comms.write_bytes(b"")
            (run.out / "encoded").mkdir(exist_ok=True)
            (run.out / "encoded" / f"S1__{band}.h5").write_bytes(b"")
        assert run.comms_ready() and run.encoded_ready() and not run.field_ready()
        (run.out / "field").mkdir()
        (run.out / "field" / "S1.h5").write_bytes(b"")
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
        from reverberate.experiments.w40_volume_field.campaign import audit_meshes

        audit = tmp_path / "audit"
        (audit / "1000" / "voxels").mkdir(parents=True)
        (audit / "1000" / "voxels" / "rooms.json").write_text("{}")
        (audit / "4000" / "voxels").mkdir(parents=True)  # built nothing
        assert audit_meshes(tmp_path, audit) == {"1000": "audit/1000/voxels"}
        assert audit_meshes(tmp_path, tmp_path / "elsewhere") == {}
