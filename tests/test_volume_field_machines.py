"""The rental policies, against the offers and failures of the two campaigns.

The client is a fake: what is tested is which offer is picked, what is
refused, and what a silent host costs the caller. No account is touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from reverberate.experiments.w40_volume_field import machines
from reverberate.gpu import vast


@dataclass
class Offer:
    id: int
    gpu_name: str = "RTX 4090"
    num_gpus: int = 1
    gpu_ram_gb: float = 24.0
    dph_total: float = 0.2
    ram_gb: float = 126.0
    disk_gb: float = 500.0
    cpu_cores: float = 32.0
    cpu_ghz: float = 3.6
    cpu_name: str = "AMD EPYC 7C13"
    location: str = "somewhere"

    def describe(self) -> str:
        return f"offer {self.id}"


class TestCardHosts:
    def test_two_40_gb_cards_do_not_hold_an_83_gb_grid_but_one_h200_does(self) -> None:
        """The market of 00:20 on 2026-09-13: the last two-card A100 host had
        40 GB cards; the H200 at 1.98 USD/h held the storey on one card."""
        offers = [
            Offer(1, "A100 SXM4", 2, 40.0, 1.34, 157.0),
            Offer(2, "H200", 1, 140.0, 1.975, 236.0),
            Offer(3, "A100 SXM4", 2, 80.0, 1.90, 197.0),
        ]
        picked = machines.card_hosts(offers, vram_gb=83.2, ram_gb=180.0, max_dph=2.1)
        assert [o.id for o in picked] == [3, 2]

    def test_blackwell_and_cheap_but_small_ram_hosts_are_out(self) -> None:
        offers = [
            Offer(1, "RTX PRO 6000 WS", 1, 96.0, 1.0, 126.0),
            Offer(2, "H100 SXM", 2, 80.0, 3.9, 120.0),
        ]
        assert machines.card_hosts(offers, vram_gb=83.0, ram_gb=180.0, max_dph=5.0) == []


class TestEncodeBoxes:
    def test_ranked_by_price_per_core_ghz_and_ivy_bridge_out(self) -> None:
        offers = [
            Offer(1, dph_total=0.122, cpu_cores=16, cpu_ghz=3.4, cpu_name="Xeon E5-2650 v2"),
            Offer(2, dph_total=0.268, cpu_cores=88, cpu_ghz=3.6, cpu_name="Xeon E5-2699 v4"),
            Offer(3, dph_total=0.255, cpu_cores=12, cpu_ghz=7.2, cpu_name="Core Ultra 9"),
            Offer(4, dph_total=0.60, cpu_cores=64, cpu_ghz=3.1, cpu_name="EPYC 9754"),
        ]
        picked = machines.encode_boxes(offers, max_dph=0.45)
        assert [o.id for o in picked] == [2, 3]

    def test_widening_relaxes_cores_then_clock_then_price(self) -> None:
        steps = machines.widening_steps(16, 3.5, 0.3)
        assert (steps[0].min_cores, steps[0].min_cpu_ghz, steps[0].max_dph) == (16, 3.5, 0.3)
        assert steps[-1].min_cores < 16 and steps[-1].min_cpu_ghz < 3.5
        assert steps[-1].max_dph > 0.3


class TestWorkers:
    def test_seven_gb_a_worker_caps_a_36_core_box_in_a_91_gb_container(self) -> None:
        """36 cores would have run 35 workers and did: BrokenProcessPool."""
        assert machines.workers_for(36, 91.0) == 13
        assert machines.workers_for(64, 183.0) == 26
        assert machines.workers_for(12, 91.0) == 11
        assert machines.workers_for(2, 1.0) == 1, "never zero"


class TestCredit:
    def test_the_whole_remaining_plan_must_be_payable(self) -> None:
        assert not machines.enough_credit(2.2, remaining_usd=4.0)
        assert machines.enough_credit(5.0, remaining_usd=4.0)

    def test_an_unanswered_api_does_not_end_a_campaign(self) -> None:
        assert machines.enough_credit(math.nan, remaining_usd=100.0)


class FakeClient:
    """Answers the calls :func:`machines.rent_one` makes; scripts the outcome per offer."""

    def __init__(
        self,
        credit: float,
        silent: frozenset[int] = frozenset(),
        refused: frozenset[int] = frozenset(),
    ) -> None:
        self.credit = credit
        self.silent = silent
        self.refused = refused
        self.destroyed: list[int] = []

    def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
        assert (method, path) == ("GET", "/users/current/")
        return {"credit": self.credit}


@dataclass
class Rented:
    instance_id: int
    offer: Offer
    deadline: float = 0.0
    log: list[str] = field(default_factory=list)


class TestRentOne:
    @pytest.fixture
    def fake_vast(self, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
        client = FakeClient(credit=10.0, silent=frozenset({2}), refused=frozenset({1}))

        def rent(c: FakeClient, offer: Offer, **_: object) -> Rented:
            if offer.id in c.refused:
                raise vast.VastError("PUT /asks failed: HTTP 400")
            return Rented(instance_id=1000 + offer.id, offer=offer)

        def wait_for_ssh(
            c: FakeClient, instance_id: int, identity: object, timeout: float = 0.0
        ) -> str:
            if instance_id - 1000 in c.silent:
                raise TimeoutError("never answered on ssh")
            return f"machine-{instance_id}"

        monkeypatch.setattr(vast, "rent", rent)
        monkeypatch.setattr(vast, "wait_for_ssh", wait_for_ssh)
        monkeypatch.setattr(vast, "teardown", lambda c, i: c.destroyed.append(i))
        return client

    def test_a_refused_then_a_silent_offer_are_consumed_and_the_third_rents(
        self, fake_vast: FakeClient
    ) -> None:
        offers = [Offer(1), Offer(2), Offer(3), Offer(4)]
        said: list[str] = []
        machine, instance = machines.rent_one(
            fake_vast, None, offers, hours=1.0, disk_gb=10, image="img", say=said.append
        )
        assert (machine, instance) == ("machine-1003", 1003)
        assert [o.id for o in offers] == [4], "tried offers leave the list, rented or not"
        assert fake_vast.destroyed == [1002], "the silent host is destroyed, not left billing"

    def test_too_little_credit_for_the_rest_of_the_plan_stops_before_renting(
        self, fake_vast: FakeClient
    ) -> None:
        fake_vast.credit = 3.0
        with pytest.raises(SystemExit, match="rest of the plan"):
            machines.rent_one(
                fake_vast, None, [Offer(3)], hours=1.0, disk_gb=10, image="img", remaining_usd=8.0
            )


class TestResumableTransfer:
    def test_the_loop_resumes_and_keeps_partial_files(self) -> None:
        command = machines.resumable_transfer(
            "/root/pressure/a.h5",
            "/root/pressure/",
            port=22,
            host="ssh1.vast.ai",
            key="/k",
            name="push 0",
        )
        assert command.startswith("until rsync -a --partial")
        assert "ServerAliveInterval=15" in command and "sleep 20; done" in command
