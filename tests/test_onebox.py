"""The watcher's judgement and the machine's sizing, against fakes.

Nothing here talks to Vast or to a machine: the client and the ssh call are
functions that answer what a real one would, so what is tested is what the
watcher concludes from an answer and what it asks for from an offer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reverberate.gpu.onebox import MachineNeed, Watch, choose_offers, monitor_once


@dataclass
class FakeInstance:
    id: int
    dph_total: float
    start_date: float

    def uptime_hours(self, now: float | None = None) -> float:
        return ((now if now is not None else time.time()) - self.start_date) / 3600.0


class FakeClient:
    def __init__(self, alive: bool = True, credit: float = 40.0) -> None:
        self.alive = alive
        self.credit = credit

    def instance(self, instance_id: int) -> FakeInstance | None:
        return FakeInstance(instance_id, 1.5, time.time() - 7200) if self.alive else None

    def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
        return {"credit": self.credit}


def answer(
    status: dict[str, Any],
    *,
    gpu: str = "97 %, 20000 MiB, 24576 MiB",
    disk: str = "150G",
    marks: str = "",
    log: str = "line one\nline two",
) -> str:
    return (
        json.dumps(status)
        + "\n@@GPU\n"
        + gpu
        + ";\n@@DISK\n"
        + disk
        + "\n@@MARK\n"
        + marks
        + "\n@@LOG\n"
        + log
    )


class TestMonitorOnce:
    def test_a_healthy_solve_is_read_with_its_bill(self) -> None:
        status = {"stage": "solve", "band": "mid", "updated": time.time() - 60}
        watch = monitor_once(
            FakeClient(),
            7,
            object(),
            run_on=lambda m, c, what, timeout: answer(status),
        )
        assert watch.instance_alive and not watch.stalled and not watch.done
        assert watch.uptime_h == pytest_approx(2.0)
        assert watch.cost_usd == pytest_approx(3.0)
        assert watch.credit_usd == 40.0
        assert watch.disk_free_gb == 150.0
        assert watch.status["band"] == "mid"
        assert "line two" in watch.log_tail
        assert watch.notes == []
        assert "solve band=mid" in watch.line()

    def test_a_status_that_stopped_moving_is_a_stall(self) -> None:
        status = {"stage": "plan", "updated": time.time() - 4000}
        watch = monitor_once(
            FakeClient(), 7, object(), run_on=lambda m, c, what, timeout: answer(status)
        )
        assert watch.stalled
        assert any("not updated" in note for note in watch.notes)

    def test_the_markers_are_read(self) -> None:
        status = {"stage": "done", "updated": time.time()}
        done = monitor_once(
            FakeClient(),
            7,
            object(),
            run_on=lambda m, c, what, timeout: answer(status, marks="/root/out/campaign.done"),
        )
        assert done.done and not done.failed
        failed = monitor_once(
            FakeClient(),
            7,
            object(),
            run_on=lambda m, c, what, timeout: answer(status, marks="/root/out/campaign.failed"),
        )
        assert failed.failed

    def test_an_idle_card_during_a_solve_and_a_full_disk_are_noted(self) -> None:
        status = {"stage": "solve", "updated": time.time()}
        watch = monitor_once(
            FakeClient(),
            7,
            object(),
            run_on=lambda m, c, what, timeout: answer(
                status, gpu="0 %, 1 MiB, 24576 MiB", disk="12G"
            ),
        )
        assert "card idle during a solve" in watch.notes
        assert "under 20 GB of disk free" in watch.notes

    def test_a_vanished_instance_is_reported_not_raised(self) -> None:
        watch = monitor_once(FakeClient(alive=False), 7, object(), run_on=lambda *a, **k: "")
        assert not watch.instance_alive
        assert "no longer exists" in watch.notes[0]

    def test_a_bad_ssh_minute_keeps_the_last_status(self) -> None:
        previous = Watch(at=0.0, instance_alive=True, uptime_h=1.0, cost_usd=1.0, credit_usd=None)
        previous.status = {"stage": "voxelise"}

        def broken(*args: object, **kwargs: object) -> str:
            raise RuntimeError("Connection closed")

        watch = monitor_once(FakeClient(), 7, object(), previous=previous, run_on=broken)
        assert watch.status == {"stage": "voxelise"}
        assert watch.notes and "ssh failed" in watch.notes[0]


@dataclass
class Offer:
    id: int
    num_gpus: int
    gpu_ram_gb: float
    ram_gb: float
    cpu_cores: float
    disk_gb: float
    dph_total: float
    cuda_max: float
    reliability: float


class TestChooseOffers:
    need = MachineNeed(
        vram_gb=83.0, ram_gb=237.0, disk_gb=300, largest_output_gb=104.0, nodes_by_band={}
    )

    def test_cards_are_summed_and_the_host_is_sized(self) -> None:
        offers = [
            Offer(1, 1, 80, 250, 32, 500, 1.2, 12.8, 0.99),  # one card short of the grid
            Offer(2, 2, 80, 128, 32, 500, 1.9, 12.8, 0.99),  # fits, slices the mid band
            Offer(3, 1, 141, 250, 32, 500, 2.6, 12.8, 0.99),  # fits in one slice
            Offer(4, 2, 80, 250, 32, 200, 1.5, 12.8, 0.99),  # too little disk
            Offer(5, 2, 80, 250, 32, 500, 3.5, 12.8, 0.99),  # too dear
            Offer(6, 2, 80, 250, 32, 500, 1.7, 12.2, 0.99),  # CUDA too old for the image
        ]
        chosen = choose_offers(offers, self.need, max_dph=3.0)
        assert [o.id for o in chosen] == [2, 3]

    def test_a_host_that_holds_the_band_whole_wins_a_small_premium(self) -> None:
        offers = [
            Offer(1, 2, 80, 128, 32, 500, 2.0, 12.8, 0.99),
            Offer(2, 2, 80, 260, 32, 500, 2.2, 12.8, 0.99),
        ]
        assert [o.id for o in choose_offers(offers, self.need, max_dph=3.0)] == [2, 1]


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, abs=0.01)


class TestLaunch:
    def test_the_launcher_is_a_script_started_detached_in_a_subshell(self) -> None:
        from reverberate.gpu.onebox import _launch_command

        command = _launch_command(
            "/root/campaign/bundle",
            "/root/campaign/out",
            "0,1",
            "/root/pffdtd",
            "--solve-bands low",
        )
        assert "(setsid nohup ./launch.sh > /dev/null 2>&1 < /dev/null &)" in command
        assert "export CUDA_VISIBLE_DEVICES=0,1" in command
        assert "--solve-bands low" in command
        assert "REVERBERATE_DATA=/root/data" in command
        assert (
            "rm -f /root/campaign/out/campaign.done /root/campaign/out/campaign.failed" in command
        )


class TestFetch:
    def test_only_what_the_laptop_keeps_comes_home(self) -> None:
        from reverberate.gpu.onebox import fetch_items

        present = [
            "field",
            "audit",
            "encoded",
            "jobs",
            "vox",
            "selfcheck",
            "walk.json",
            "plan.json",
            "campaign.log",
            "report.json",
            "status.json",
            "driver.log",
            "campaign.done",
        ]
        items = fetch_items(present)
        assert "field" in items and "audit" in items and "walk.json" in items
        assert "campaign.done" in items and "driver.log" in items
        for heavy in ("encoded", "jobs", "vox", "selfcheck"):
            assert heavy not in items
        assert fetch_items(["field"]) == ["field"]

    def test_only_the_grids_the_laptop_lacks_are_fetched(self, tmp_path: Path) -> None:
        from reverberate.gpu.onebox import missing_grids

        (tmp_path / "k1").mkdir()
        (tmp_path / "k1" / "manifest.json").write_text("{}")
        (tmp_path / "k2").mkdir()  # no manifest: not a complete entry
        assert missing_grids(["k1", "k2", "k3"], tmp_path) == ["k2", "k3"]


class TestRunEndToEnd:
    """The renter's loop against fakes: rent, provision, launch, watch, relaunch, fetch, destroy."""

    def test_a_campaign_that_stalls_once_is_relaunched_then_fetched_and_the_host_destroyed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reverberate.gpu import onebox

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "campaign.json").write_text(
            json.dumps(
                {
                    "model_json": "model.json",
                    "ppw": 10.5,
                    "tc": 20.0,
                    "rh": 50.0,
                    "points": 4,
                    "bands": {"low": {"fmax_hz": 1000.0, "duration_s": 0.1, "cache_key": "k1"}},
                }
            )
        )
        (bundle / "model.json").write_text(
            json.dumps({"mats_hash": {"wall": {"pts": [[0, 0, 0], [3, 3, 3]]}}})
        )
        calls: list[str] = []
        machine = object()

        class Offer:
            id = 99
            dph_total = 1.0
            num_gpus = 2
            gpu_ram_gb = 80.0
            gpu_name = "A100"
            ram_gb = 200.0
            cpu_cores = 32.0
            disk_gb = 500.0
            cuda_max = 12.8
            reliability = 0.99

            def describe(self) -> str:
                return "an offer"

        class Client:
            def search(self, query: str, limit: int = 20) -> list[Offer]:
                return [Offer()]

            def instance(self, instance_id: int) -> FakeInstance | None:
                return FakeInstance(instance_id, 1.0, time.time() - 3600)

            def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
                return {"credit": 10.0}

        # What the machine answers, look after look: solving, stalled, solving, done.
        answers = iter(
            [
                answer({"stage": "solve", "updated": time.time()}),
                answer({"stage": "solve", "updated": time.time() - 9000}),
                answer({"stage": "solve", "updated": time.time()}),
                answer(
                    {"stage": "done", "updated": time.time()},
                    marks="/root/campaign/out/campaign.done",
                ),
            ]
        )

        def fake_run_on(m: Any, command: str, *, what: str, timeout: Any = None) -> str:
            calls.append(what)
            if what == "watch":
                return next(answers)
            if what == "list run":
                return "field\naudit\nencoded\nselfcheck\nwalk.json\ncampaign.done\n"
            return "started"

        pulled: list[list[str]] = []

        def fake_rsync(
            m: Any, sources: list[str], target: str, *, download: bool, **kw: Any
        ) -> None:
            calls.append("rsync down" if download else "rsync up")
            if download:
                pulled.append(sources)

        monkeypatch.setattr(onebox.auth, "inject", lambda names: None)
        monkeypatch.setattr(onebox.vast, "VastClient", lambda timeout: Client())
        monkeypatch.setattr(onebox.vast, "account_identity", lambda client: tmp_path / "id")
        monkeypatch.setattr(
            onebox.vast, "teardown", lambda client, instance: calls.append("teardown") or True
        )
        monkeypatch.setattr(
            onebox.vast, "rent_one", lambda *a, **k: calls.append("rent") or (machine, 7)
        )
        monkeypatch.setattr(
            onebox, "provision", lambda m, script, beside=(): calls.append("build") or 60.0
        )
        monkeypatch.setattr(onebox, "run_on", fake_run_on)
        monkeypatch.setattr(onebox, "rsync", fake_rsync)
        monkeypatch.setattr(onebox, "cache_root", lambda: tmp_path / "cache")
        monkeypatch.setattr(onebox.time, "sleep", lambda s: None)
        record = onebox.run(
            bundle,
            tmp_path / "home",
            hours=2.0,
            max_dph=3.0,
            yes=True,
            repo=Path(__file__).parents[1],
            poll_s=0.0,
            say=lambda m: None,
        )
        assert record["outcome"] == "done" and record["destroyed"] is True
        assert record["instance"] == 7 and record["offer"] == 99
        assert calls.count("launch") == 2, "relaunched once after the stall"
        assert calls.index("kill") < calls.index("launch", calls.index("launch") + 1)
        assert calls.index("build") < calls.index("provision") < calls.index("launch")
        assert calls[-1] == "teardown" and "rsync down" in calls
        # What came home: the kept items, the self-check reports, the grid the laptop lacks.
        assert pulled[0] == [
            "/root/campaign/out/field",
            "/root/campaign/out/audit",
            "/root/campaign/out/walk.json",
            "/root/campaign/out/campaign.done",
        ]
        assert pulled[1] == [
            "/root/campaign/out/selfcheck/*.json",
            "/root/campaign/out/selfcheck/*.log",
        ]
        assert pulled[2] == ["/root/data/cache/vox/k1"]
        assert len(record["watches"]) == 4
        saved = json.loads((tmp_path / "home" / "onebox.json").read_text())
        assert saved["outcome"] == "done"

    def test_without_yes_nothing_is_rented(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reverberate.gpu import onebox

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "campaign.json").write_text(
            json.dumps(
                {
                    "model_json": "model.json",
                    "ppw": 10.5,
                    "tc": 20.0,
                    "rh": 50.0,
                    "points": 4,
                    "bands": {"low": {"fmax_hz": 1000.0, "duration_s": 0.1, "cache_key": "k1"}},
                }
            )
        )
        (bundle / "model.json").write_text(
            json.dumps({"mats_hash": {"wall": {"pts": [[0, 0, 0], [3, 3, 3]]}}})
        )
        monkeypatch.setattr(onebox.auth, "inject", lambda names: None)
        monkeypatch.setattr(onebox.vast, "VastClient", lambda timeout: object())
        monkeypatch.setattr(onebox.vast, "account_identity", lambda client: tmp_path)
        monkeypatch.setattr(onebox.vast, "rent_one", lambda *a, **k: pytest.fail("rented"))
        record = onebox.run(
            bundle,
            tmp_path / "home",
            hours=1,
            max_dph=1,
            yes=False,
            repo=tmp_path,
            say=lambda m: None,
        )
        assert "instance" not in record
