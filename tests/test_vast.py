"""Tests for the Vast.ai rental module.

The expensive failure this module guards against is an instance nobody
destroys, so the tests that matter are the ones about refusal: no deadline, no
rental; no watchdog, no rental; over the ceiling, no rental. Those are checked
against a fake client so they cost nothing and need no credential.

Nothing here touches the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from reverberate import settings
from reverberate.gpu import vast
from reverberate.gpu.vast import (
    Instance,
    Offer,
    VastError,
    cheapest,
    estimate_cost_usd,
    ledger_total_usd,
    parse_query,
    search_query,
)


def make_offer(offer_id: int = 1, dph: float = 0.30, gpu_ram_gb: float = 24.0) -> Offer:
    return Offer(
        id=offer_id,
        gpu_name="RTX 4090",
        num_gpus=1,
        dph_total=dph,
        gpu_ram_gb=gpu_ram_gb,
        cpu_cores=32.0,
        ram_gb=128.0,
        disk_gb=60.0,
        cuda_max=12.6,
        reliability=0.995,
        inet_down_mbps=800.0,
        location="Canada",
    )


class FakeClient:
    """Stands in for VastClient. Records what it was asked to do."""

    def __init__(self, *, create_fails: bool = False) -> None:
        self.created: list[int] = []
        self.destroyed: list[int] = []
        self.live: dict[int, Instance] = {}
        self._create_fails = create_fails
        self._next_id = 1000

    def create(self, offer_id: int, image: str, disk_gb: int = 60) -> int:
        if self._create_fails:
            raise VastError("refused")
        self._next_id += 1
        self.created.append(offer_id)
        self.live[self._next_id] = Instance(
            id=self._next_id,
            status="running",
            dph_total=0.30,
            ssh_host="ssh.example",
            ssh_port=22,
            gpu_name="RTX 4090",
            start_date=None,
        )
        return self._next_id

    def instance(self, instance_id: int) -> Instance | None:
        return self.live.get(instance_id)

    def instances(self) -> list[Instance]:
        return list(self.live.values())

    def destroy_and_verify(self, instance_id: int, attempts: int = 5, pause: float = 0.0) -> bool:
        self.destroyed.append(instance_id)
        self.live.pop(instance_id, None)
        return True


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the ledger at a temporary tree so tests never see real spend."""
    monkeypatch.setenv(settings.DATA_ROOT_ENV, str(tmp_path))
    return tmp_path


class TestOfferSelection:
    def test_query_names_every_filter_that_picked_the_machine(self) -> None:
        query = search_query(gpu_name="RTX_4090", min_disk_gb=60)
        assert "gpu_name=RTX_4090" in query
        assert "disk_space>60" in query
        assert "rentable=true" in query

    def test_the_default_card_name_carries_a_space_not_an_underscore(self) -> None:
        """The API matches ``RTX 4090``, so the underscore form finds nothing.

        This returned zero offers with no error for as long as the default said
        ``RTX_4090``, which is how Vast.ai's own documentation writes it.
        """
        assert "gpu_name=RTX 4090" in search_query()
        assert "RTX_4090" not in search_query()

    def test_query_parses_into_the_operators_the_api_expects(self) -> None:
        parsed = parse_query("gpu_name=RTX_4090 disk_space>60 cuda_vers>=12.0")
        assert parsed["gpu_name"] == {"eq": "RTX_4090"}
        assert parsed["disk_space"] == {"gt": 60}
        # >= must not be read as > followed by a stray =
        assert parsed["cuda_vers"] == {"gte": 12.0}
        assert parsed["type"] == "on-demand"

    def test_cheapest_respects_the_memory_floor(self) -> None:
        small = make_offer(1, dph=0.10, gpu_ram_gb=12.0)
        big = make_offer(2, dph=0.40, gpu_ram_gb=24.0)
        assert cheapest([small, big]) is small
        # B1 hit an out-of-memory at 8 kHz on 24 GB, so the floor is the point.
        assert cheapest([small, big], min_gpu_ram_gb=24.0) is big

    def test_no_offer_meets_the_floor(self) -> None:
        assert cheapest([make_offer(gpu_ram_gb=12.0)], min_gpu_ram_gb=80.0) is None


class TestCost:
    def test_cost_is_rate_times_hours(self) -> None:
        assert estimate_cost_usd(0.327, 3.0) == pytest.approx(0.981)

    def test_negative_inputs_cannot_manufacture_headroom(self) -> None:
        assert estimate_cost_usd(0.30, -5.0) == 0.0
        assert estimate_cost_usd(-0.30, 5.0) == 0.0

    def test_uptime_is_zero_before_the_contract_starts(self) -> None:
        pending = Instance(1, "loading", 0.3, "h", 22, "RTX 4090", start_date=None)
        assert pending.uptime_hours() == 0.0

    def test_uptime_counts_from_the_contract_start(self) -> None:
        running = Instance(1, "running", 0.3, "h", 22, "RTX 4090", start_date=1000.0)
        assert running.uptime_hours(now=1000.0 + 7200.0) == pytest.approx(2.0)


class TestLedger:
    def test_a_live_rental_counts_at_its_full_budget(self) -> None:
        # Nothing has been torn down, so the worst case is the only honest number.
        entries = [{"event": "rent", "instance_id": 1, "budget_usd": 1.0}]
        assert ledger_total_usd(entries) == pytest.approx(1.0)

    def test_teardown_replaces_the_budget_with_what_it_cost(self) -> None:
        entries = [
            {"event": "rent", "instance_id": 1, "budget_usd": 1.0},
            {"event": "teardown", "instance_id": 1, "cost_usd": 0.42},
        ]
        assert ledger_total_usd(entries) == pytest.approx(0.42)

    def test_totals_accumulate_across_rentals(self) -> None:
        entries = [
            {"event": "rent", "instance_id": 1, "budget_usd": 1.0},
            {"event": "teardown", "instance_id": 1, "cost_usd": 0.42},
            {"event": "rent", "instance_id": 2, "budget_usd": 2.0},
        ]
        assert ledger_total_usd(entries) == pytest.approx(2.42)

    def test_empty_ledger_is_zero_not_an_error(self, data_root: Path) -> None:
        assert vast.read_ledger() == []
        assert ledger_total_usd(vast.read_ledger()) == 0.0

    def test_entries_round_trip_through_the_file(self, data_root: Path) -> None:
        vast.append_ledger({"event": "rent", "instance_id": 7, "budget_usd": 1.5})
        entries = vast.read_ledger()
        assert entries[0]["instance_id"] == 7
        assert "at" in entries[0]


class TestRentRefuses:
    def test_a_rental_without_a_deadline(self, data_root: Path) -> None:
        client = FakeClient()
        with pytest.raises(VastError, match="deadline"):
            vast.rent(client, make_offer(), hours=0, image="img")  # type: ignore[arg-type]
        assert client.created == []

    def test_a_rental_that_would_pass_the_ceiling(self, data_root: Path) -> None:
        vast.append_ledger({"event": "rent", "instance_id": 1, "budget_usd": 999.0})
        client = FakeClient()
        with pytest.raises(VastError, match="ceiling"):
            vast.rent(client, make_offer(dph=1.0), hours=3, image="img", ceiling_usd=1000.0)  # type: ignore[arg-type]
        assert client.created == []

    def test_and_destroys_the_instance_when_the_watchdog_cannot_be_armed(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An instance with no hard stop is exactly the failure mode this module
        # exists to prevent, so it must not survive.
        def boom(instance_id: int, deadline: float) -> int:
            raise OSError("no fork")

        monkeypatch.setattr(vast, "arm_hard_stop", boom)
        client = FakeClient()
        with pytest.raises(VastError, match="hard stop"):
            vast.rent(client, make_offer(), hours=3, image="img")  # type: ignore[arg-type]
        assert client.destroyed and client.live == {}


class TestRentSucceeds:
    def test_arms_the_watchdog_and_ledgers_the_budget(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        armed: list[tuple[int, float]] = []

        def record(instance_id: int, deadline: float) -> int:
            armed.append((instance_id, deadline))
            return 4242

        monkeypatch.setattr(vast, "arm_hard_stop", record)
        client = FakeClient()
        before = time.time()
        rental = vast.rent(client, make_offer(dph=0.327), hours=3.0, image="cuda")  # type: ignore[arg-type]

        # An absolute timestamp, not a duration: see arm_hard_stop on W22.
        assert len(armed) == 1
        assert armed[0][0] == rental.instance_id
        assert before + 3 * 3600 <= armed[0][1] <= time.time() + 3 * 3600
        assert rental.watchdog_pid == 4242
        entry = vast.read_ledger()[0]
        assert entry["event"] == "rent"
        assert entry["budget_usd"] == pytest.approx(0.981)
        assert entry["watchdog_pid"] == 4242

    def test_the_credential_never_reaches_the_ledger(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vast, "arm_hard_stop", lambda instance_id, deadline: 1)
        monkeypatch.setenv(vast.API_KEY_ENV, "secret-key-value")
        vast.rent(FakeClient(), make_offer(), hours=1.0, image="img")  # type: ignore[arg-type]
        assert "secret-key-value" not in vast.ledger_path().read_text()


class TestTeardown:
    def test_records_what_the_instance_actually_cost(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeClient()
        instance_id = client.create(offer_id=1, image="img")
        client.live[instance_id] = Instance(
            id=instance_id,
            status="running",
            dph_total=0.30,
            ssh_host="h",
            ssh_port=22,
            gpu_name="RTX 4090",
            start_date=1000.0,
        )
        monkeypatch.setattr("reverberate.gpu.vast.time.time", lambda: 1000.0 + 3600.0)

        assert vast.teardown(client, instance_id) is True  # type: ignore[arg-type]
        entry = vast.read_ledger()[-1]
        assert entry["event"] == "teardown"
        assert entry["cost_usd"] == pytest.approx(0.30)
        assert entry["verified_gone"] is True

    def test_an_already_gone_instance_is_free_and_still_recorded(self, data_root: Path) -> None:
        client = FakeClient()
        assert vast.teardown(client, 999) is True  # type: ignore[arg-type]
        entry = vast.read_ledger()[-1]
        assert entry["cost_usd"] == 0.0


class TestClient:
    def test_refuses_to_exist_without_a_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(vast.API_KEY_ENV, raising=False)
        with pytest.raises(VastError, match=vast.API_KEY_ENV):
            vast.VastClient()

    def test_parses_the_fields_the_rental_decision_depends_on(self) -> None:
        raw: dict[str, Any] = {
            "id": 45449275,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.327,
            "gpu_ram": 24576,
            "cpu_ram": 515000,
            "disk_space": 60.0,
            "cuda_max_good": 12.8,
            "reliability2": 0.997,
            "geolocation": "Canada",
        }
        offer = Offer.from_api(raw)
        assert offer.gpu_ram_gb == pytest.approx(24.0)
        assert offer.dph_total == pytest.approx(0.327)
        assert "45449275" in offer.describe()

    def test_a_missing_geolocation_does_not_crash_the_summary(self) -> None:
        offer = Offer.from_api({"id": 1, "geolocation": None})
        assert offer.location == "unknown"


def test_build_script_pins_the_commit_b1_measured() -> None:
    """The cost table only describes one PFFDTD tree; the script must pin it."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pffdtd.sh"
    text = script.read_text()
    assert "aa319f6c86517cb95aabfae8656277da62c3ead5" in text
    for patch in ("-arch=sm_", "np.finfo(float)", "numpy==1.26.4", "pffdtd_compat"):
        assert patch in text, f"build script lost the {patch!r} patch"


def test_ledger_file_is_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    vast.append_ledger({"event": "rent", "instance_id": 1}, path=path)
    vast.append_ledger({"event": "teardown", "instance_id": 1}, path=path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["instance_id"] == 1 for line in lines)


class TestHardStopSurvivesSuspension:
    """W22: the watchdog must not lose time while the machine is asleep."""

    @staticmethod
    def _client_holding(instance_id: int) -> FakeClient:
        client = FakeClient()
        client.live[instance_id] = Instance(
            id=instance_id,
            status="running",
            dph_total=0.30,
            ssh_host="ssh.example",
            ssh_port=22,
            gpu_name="RTX 4090",
            start_date=None,
        )
        return client

    def test_a_deadline_already_past_kills_at_once(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Waking up late is the suspend case, and it must still fire.

        A watchdog spending its whole wait in one ``time.sleep`` cannot tell
        that the machine suspended underneath it. One polling the wall clock
        finds the deadline gone and acts immediately, which is the fix.
        """
        client = self._client_holding(77)
        monkeypatch.setattr(vast, "VastClient", lambda *a, **k: client)
        monkeypatch.setattr("reverberate.auth.inject", lambda names: None)

        assert vast._hard_stop(77, deadline=time.time() - 3600.0) == 0
        assert client.destroyed == [77]

    def test_no_single_sleep_outlasts_the_deadline(
        self, data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each nap is capped, so a suspend can cost one poll interval, not the run."""
        slept: list[float] = []
        clock = [1000.0]

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += seconds

        client = self._client_holding(77)
        monkeypatch.setattr("reverberate.gpu.vast.time.sleep", fake_sleep)
        monkeypatch.setattr("reverberate.gpu.vast.time.time", lambda: clock[0])
        monkeypatch.setattr(vast, "VastClient", lambda *a, **k: client)
        monkeypatch.setattr("reverberate.auth.inject", lambda names: None)

        vast._hard_stop(77, deadline=1070.0, poll_s=30.0)

        assert slept == [30.0, 30.0, 10.0]
        assert client.destroyed == [77]


class TestInstanceDiagnostics:
    """status_msg is what separates a slow pull from a dead host."""

    def test_the_status_message_is_carried_and_collapsed(self) -> None:
        instance = Instance.from_api(
            {
                "id": 7,
                "actual_status": "loading",
                "dph_total": 1.0,
                "ssh_host": "ssh9.vast.ai",
                "ssh_port": 2,
                "gpu_name": "A100 SXM4",
                "status_msg": "  pulling\n  image   layer 3/9 \n",
            }
        )
        assert instance.status == "loading"
        assert instance.status_msg == "pulling image layer 3/9"

    def test_a_missing_status_message_is_empty_not_none(self) -> None:
        """Callers format it into logs, so it must never be the string "None"."""
        instance = Instance.from_api(
            {"id": 7, "actual_status": "running", "dph_total": 1.0, "gpu_name": "x"}
        )
        assert instance.status_msg == ""
