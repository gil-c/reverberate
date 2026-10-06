"""What a rental must survive, against fakes: every case here cost a rented machine once.

A connection that drops while a host is provisioned, a host that refuses
every connection, a driver that fails after the machine is rented, a host
that dies with hours of solves on it, a machine taken by its hourly price
for a run of two thousand solves. Nothing here talks to Vast or to a
machine: ``ssh`` is a function that answers as one did, the client is a
class that counts what it destroyed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reverberate import auth
from reverberate.accel.pairs import PairCache, install_pairs
from reverberate.gpu import onebox, vast
from reverberate.trace import machines
from reverberate.trace.cli import main
from reverberate.trace.plan import estimate
from reverberate.wave import remote, remote_voxelise
from reverberate.wave.lowband import pairs as batched
from reverberate.wave.remote import ConnectionLost, Machine, RemoteError, connection_level
from test_onebox import FakeInstance, answer

BANNER = "Welcome to vast.ai. If authentication fails, try again after a few seconds.\nHave fun!"
CLOSED = BANNER + "\nConnection to ssh6.vast.ai closed by remote host."
REFUSED = "ssh: connect to host ssh3.vast.ai port 13832: Connection refused"
SSH = ["ssh", "-p", "22", "root@host", "true"]


# --------------------------------------------------------------------------
# the connection and the command
# --------------------------------------------------------------------------


class TestTheConnection:
    def test_a_lost_connection_is_told_from_a_command_that_failed(self) -> None:
        # The two seen on 2026-10-05, and the banner with nothing after it.
        assert connection_level(SSH, 255, CLOSED)
        assert connection_level(SSH, 255, REFUSED)
        assert connection_level(SSH, 255, BANNER)
        # A remote command that exits non-zero is the command's, whatever it printed.
        assert not connection_level(SSH, 1, BANNER + "\ncurl: (7) Connection refused")
        assert not connection_level(SSH, 2, BANNER + "\nmake: *** [all] Error 2")
        assert not connection_level(SSH, 255, BANNER + "\nbash: line 1: exit: bad")
        # rsync says it by its code: a broken stream, not a missing file.
        assert connection_level(["rsync", "-a"], 12, "error in rsync protocol data stream")
        assert connection_level(["rsync", "-a"], 255, "")
        assert not connection_level(["rsync", "-a"], 23, "No such file or directory")
        assert not connection_level(["python", "x"], 255, REFUSED)

    @pytest.fixture
    def scripted(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """``subprocess.run`` answering from a list; the pauses taken are kept."""
        script: dict[str, Any] = {"answers": [], "ran": 0, "slept": []}

        def run(argv: list[str], **given: Any) -> Any:
            script["ran"] += 1
            code, err = script["answers"].pop(0)
            return subprocess.CompletedProcess(argv, code, stdout="out", stderr=err)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(time, "sleep", script["slept"].append)
        return script

    def test_a_dropped_connection_is_tried_again_with_a_growing_pause(
        self, scripted: dict[str, Any]
    ) -> None:
        scripted["answers"] = [(255, CLOSED), (255, REFUSED), (0, "")]
        assert remote._run(SSH, what="build pffdtd") == "out"
        assert scripted["ran"] == 3 and scripted["slept"] == [15.0, 30.0]

    def test_a_command_that_failed_is_not_tried_again(self, scripted: dict[str, Any]) -> None:
        scripted["answers"] = [(2, BANNER + "\nmake: *** Error 2")]
        with pytest.raises(RemoteError, match="Error 2") as raised:
            remote._run(SSH, what="build pffdtd")
        assert not isinstance(raised.value, ConnectionLost) and raised.value.returncode == 2
        assert scripted["ran"] == 1 and scripted["slept"] == []

    def test_a_host_that_refuses_every_connection_ends_the_attempts(
        self, scripted: dict[str, Any]
    ) -> None:
        scripted["answers"] = [(255, REFUSED)] * remote.ATTEMPTS
        with pytest.raises(ConnectionLost, match="Connection refused"):
            remote._run(SSH, what="ssh probe")
        assert scripted["ran"] == remote.ATTEMPTS
        # Bounded: five pauses, none over two minutes.
        assert scripted["slept"] == [15.0, 30.0, 60.0, 120.0, 120.0]

    def test_a_transfer_is_resumed_on_a_broken_stream_and_not_on_a_missing_file(
        self, scripted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        machine = Machine(host="h", port=22, user="root", identity=None)
        scripted["answers"] = [
            (12, "rsync: connection unexpectedly closed"),
            (255, CLOSED),
            (0, ""),
        ]
        remote_voxelise.rsync(machine, ["a"], "b", download=True, exclude=("*.partial.npy",))
        assert scripted["ran"] == 3
        scripted["answers"] = [(23, "rsync: link_stat failed: No such file or directory")]
        with pytest.raises(RemoteError, match="No such file"):
            remote_voxelise.rsync(machine, ["a"], "b", download=True)
        assert scripted["ran"] == 4
        scripted["answers"] = [(12, "broken")] * 2
        with pytest.raises(ConnectionLost, match="after 2 attempts"):
            remote_voxelise.rsync(machine, ["a"], "b", download=True, attempts=2)

    def test_a_build_tried_again_waits_for_itself(self, tmp_path: Path) -> None:
        """The command under a real shell: behind ``flock`` where there is one, plain where not."""
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("no bash")
        ran = tmp_path / "ran"
        command = remote.one_at_a_time(f"bash -c 'echo built >> {ran}'", str(tmp_path / "lock"))
        shim = tmp_path / "bin"
        shim.mkdir()
        (shim / "flock").write_text(f'#!{bash}\necho "$1" >> {tmp_path}/locked\nshift\nexec "$@"\n')
        (shim / "flock").chmod(0o755)
        for path in (f"{shim}:/usr/bin:/bin", "/nowhere"):
            found = shutil.which("flock", path=path)
            done = subprocess.run(
                [bash, "-c", command], env={"PATH": path}, capture_output=True, text=True
            )
            if found is None and path == "/nowhere":
                # No flock, and here no bash on the path either: the command is run as it is.
                assert "flock" not in done.stderr
                continue
            assert done.returncode == 0, done.stderr
        assert ran.read_text() == "built\n"
        assert (tmp_path / "locked").read_text() == f"{tmp_path / 'lock'}\n"

    def test_a_launch_made_twice_runs_one_campaign(self) -> None:
        command = onebox._launch_command("/root/campaign/bundle", "/root/campaign/out", None, "/p")
        pid = "/root/campaign/campaign.pid"
        # The launcher writes its id, which ``exec`` keeps for the campaign ...
        assert f"echo $$ > {pid}\n" in command and "\nexec " in command
        # ... and a launch first ends the campaign that id names, before it starts another.
        assert command.index(f'kill "$(cat {pid})"') < command.index("setsid nohup ./launch.sh")
        assert command.startswith(f'if [ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then')


# --------------------------------------------------------------------------
# the rental, from the first offer to the last line
# --------------------------------------------------------------------------


class Offer:
    """An offer the driver can choose, price and rent."""

    cpu_cores = 32.0
    ram_gb = 200.0
    disk_gb = 500.0
    cuda_max = 12.8
    reliability = 0.99

    def __init__(
        self,
        id: int,
        gpu_name: str = "RTX 3080",
        num_gpus: int = 1,
        dph_total: float = 0.14,
        gpu_ram_gb: float = 20.0,
        machine_id: int = 0,
    ) -> None:
        self.id, self.gpu_name, self.num_gpus = id, gpu_name, num_gpus
        self.dph_total, self.gpu_ram_gb = dph_total, gpu_ram_gb
        self.machine_id = machine_id or 9000 + id

    def describe(self) -> str:
        return f"offer {self.id}: {self.num_gpus}x {self.gpu_name}, {self.dph_total} USD/h"


class Client:
    """Vast, as far as a run sees it: the offers, the instances, the credit."""

    def __init__(self, offers: list[Offer]) -> None:
        self.offers = offers
        self.alive: set[int] = set()
        self.searched: list[str] = []

    def search(self, query: str, limit: int = 20) -> list[Offer]:
        self.searched.append(query)
        count = int(query.split("num_gpus=")[1].split()[0])
        return [offer for offer in self.offers if offer.num_gpus == count]

    def instance(self, instance_id: int) -> FakeInstance | None:
        if instance_id not in self.alive:
            return None
        return FakeInstance(instance_id, 0.5, time.time() - 3600)

    def request(self, method: str, path: str, payload: object = None) -> dict[str, float]:
        return {"credit": 28.0}


def a_bundle(tmp_path: Path) -> Path:
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
    return bundle


DONE = "/root/campaign/out/campaign.done"


@pytest.fixture
def rental(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A scripted night: which hosts fail to provision, what the machine answers, what is said.

    ``rent_one`` takes the first offer not avoided and names its instance
    ``1000 + offer``; ``teardown`` is verified unless the script says not.
    """
    script: dict[str, Any] = {
        "client": Client([Offer(1), Offer(2, dph_total=0.2), Offer(3, dph_total=0.3)]),
        "bad": set(),
        "calls": [],
        "pulled": [],
        "said": [],
        "watch": [answer({"stage": "done", "updated": time.time()}, marks=DONE)],
        "launch": None,
        "unverified": set(),
        "rented": [],
        "hours": [],
        "disk": [],
        "home": tmp_path / "home",
        "bundle": a_bundle(tmp_path),
    }
    client = script["client"]

    def rent_one(
        _: Any, identity: Any, offers: list[Offer], *, hours: float, avoid: set[int], **kw: Any
    ) -> tuple[Any, int]:
        offer = offers.pop(0)
        script["rented"].append(offer.id)
        script["hours"].append(hours)
        script["disk"].append(kw.get("disk_gb"))
        client.alive.add(1000 + offer.id)
        return f"machine-{offer.id}", 1000 + offer.id

    def teardown(_: Any, instance: int) -> bool:
        script["calls"].append(f"teardown {instance}")
        if instance in script["unverified"]:
            return False
        client.alive.discard(instance)
        return True

    def provision(machine: str, script_path: Path, beside: Any = ()) -> float:
        script["calls"].append(f"build {machine}")
        if int(machine.split("-")[1]) in script["bad"]:
            raise ConnectionLost(f"build pffdtd failed, the connection lost 6 times: {CLOSED}")
        return 60.0

    def run_on(machine: Any, command: str, *, what: str, timeout: Any = None) -> str:
        script["calls"].append(what)
        if what == "launch" and script["launch"] is not None:
            raise script["launch"]
        if what == "watch":
            return str(script["watch"].pop(0)) if len(script["watch"]) > 1 else script["watch"][0]
        if what == "list run":
            return "pack.h5\npairs\ntrace_report.json\ncampaign.done\nearly\n"
        return "started"

    def rsync(machine: Any, sources: list[str], target: str, *, download: bool, **kw: Any) -> None:
        script["calls"].append("rsync down" if download else "rsync up")
        if download:
            script["pulled"].append((sources, kw))

    def fetch_tree(machine: Any, remote_dir: str, local_dir: Path, **kw: Any) -> dict[str, Any]:
        # The pair cache: batches of whole files (``gpu.homecoming``), not an rsync.
        script["calls"].append(f"tree {Path(remote_dir).name}")
        script["pulled"].append(([remote_dir], kw))
        return {"files": 0, "bytes": 0, "left": 0, "complete": True, "there": 1, "seconds": 0.0}

    def fetch_file(machine: Any, remote: str, local: Path, **kw: Any) -> dict[str, Any]:
        script["calls"].append(f"chunks {Path(remote).name}")
        return {"bytes": 9_500_000_000, "seconds": 2160.0, "bytes_per_s": 4.4e6, "failures": 0}

    monkeypatch.setattr(auth, "inject", lambda names: None)
    monkeypatch.setattr(vast, "VastClient", lambda timeout: client)
    monkeypatch.setattr(vast, "account_identity", lambda client: tmp_path / "id")
    monkeypatch.setattr(vast, "teardown", teardown)
    monkeypatch.setattr(vast, "rent_one", rent_one)
    monkeypatch.setattr(onebox, "provision", provision)
    monkeypatch.setattr(onebox, "run_on", run_on)
    monkeypatch.setattr(onebox, "rsync", rsync)
    monkeypatch.setattr(onebox, "fetch_tree", fetch_tree)
    monkeypatch.setattr(onebox, "fetch_file", fetch_file)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return script


def run(rental: dict[str, Any], **given: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "hours": 2.0,
        "max_dph": 3.0,
        "yes": True,
        "repo": Path(__file__).parents[1],
        "poll_s": 0.0,
        "fetch_cache": False,
        "say": rental["said"].append,
    }
    return onebox.run(rental["bundle"], rental["home"], **{**options, **given})


class TestTheRental:
    def test_a_host_whose_provisioning_drops_is_destroyed_avoided_and_the_next_taken(
        self, rental: dict[str, Any]
    ) -> None:
        rental["bad"] = {1, 2}
        record = run(rental)
        # Three rentals, the first two destroyed and verified before the next was taken.
        assert rental["rented"] == [1, 2, 3] and record["instance"] == 1003
        calls = rental["calls"]
        assert calls.index("teardown 1001") < calls.index("build machine-2")
        assert calls.index("teardown 1002") < calls.index("build machine-3")
        assert record["abandoned"] == [
            {"instance": 1001, "offer": 1, "destroyed": True},
            {"instance": 1002, "offer": 2, "destroyed": True},
        ]
        # Both their ids and their hosts' are in the list the next run is given.
        assert {1, 9001, 2, 9002} <= set(record["avoided"])
        assert record["outcome"] == "done" and record["destroyed"] is True
        assert rental["client"].alive == set()
        assert any("provisioning of 1001 failed" in line for line in rental["said"])

    def test_no_host_that_provisions_ends_with_nothing_rented(self, rental: dict[str, Any]) -> None:
        rental["bad"] = {1, 2, 3}
        with pytest.raises(SystemExit, match="no offer|no host could be provisioned"):
            run(rental)
        assert rental["client"].alive == set() and rental["rented"] == [1, 2, 3]
        saved = json.loads((rental["home"] / "onebox.json").read_text())
        assert len(saved["abandoned"]) == 3 and "instance" not in saved

    def test_a_host_not_verified_destroyed_stops_the_run_and_is_named(
        self, rental: dict[str, Any]
    ) -> None:
        rental["bad"], rental["unverified"] = {1}, {1001}
        with pytest.raises(SystemExit, match="INSTANCE 1001 IS STILL RENTED"):
            run(rental)
        # No second machine beside one that may still be billing.
        assert rental["rented"] == [1]
        assert "0.500 USD/h until its watchdog" in rental["said"][-1]

    def test_a_driver_that_fails_after_the_rental_destroys_and_verifies(
        self, rental: dict[str, Any]
    ) -> None:
        rental["launch"] = RuntimeError("something nobody foresaw")
        record = run(rental, sync=("pairs",))
        assert record["outcome"] == "error" and "nobody foresaw" in record["error"]
        assert "RuntimeError" in record["traceback"]
        assert record["destroyed"] is True and rental["client"].alive == set()
        # What can be saved is asked for once before the machine goes.
        assert rental["pulled"][0][0] == ["/root/campaign/out/pairs"]
        assert rental["calls"][-1] == "teardown 1001" and "left_alive" not in record
        assert json.loads((rental["home"] / "onebox.json").read_text())["outcome"] == "error"

    def test_a_failure_that_cannot_destroy_says_so_in_its_last_line(
        self, rental: dict[str, Any]
    ) -> None:
        rental["launch"], rental["unverified"] = RuntimeError("no"), {1001}
        record = run(rental)
        assert record["destroyed"] is False
        last = rental["said"][-1]
        assert last == record["left_alive"] and last.startswith("INSTANCE 1001 IS STILL RENTED")
        assert "the driver failed and the destruction is not verified" in last
        assert "Resume with --instance 1001" in last

    def test_a_person_s_interruption_leaves_the_campaign_and_names_the_instance(
        self, rental: dict[str, Any]
    ) -> None:
        rental["launch"] = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            run(rental)
        assert rental["client"].alive == {1001}
        assert "INSTANCE 1001 IS STILL RENTED: the driver was interrupted" in rental["said"][-1]

    def test_a_campaign_that_failed_twice_is_kept_and_said_or_destroyed_when_asked(
        self, rental: dict[str, Any]
    ) -> None:
        failed = answer({"stage": "failed"}, marks="/root/campaign/out/campaign.failed")
        rental["watch"] = [failed]
        record = run(rental)
        assert record["outcome"] == "failed" and rental["client"].alive == {1001}
        assert "failed twice and is kept for inspection" in rental["said"][-1]
        # The last lines say what to do: the command that resumes it, on what it holds,
        # what an hour of leaving it costs, and the command that ends it.
        rental["client"].alive = {1001}
        record = run(
            rental,
            resume_command="python -m reverberate.trace rent --recipe R.json --home H"
            " --instance {instance}",
            inventory=lambda machine: [
                "on the machine: 16887 pairs solved, 15 early tables, no pack",
                "a resume keeps all of it and makes again: the pack's write and its check",
            ],
        )
        last = rental["said"][-1].splitlines()
        assert last[0].startswith("INSTANCE 1001 IS STILL RENTED: the campaign failed twice")
        assert "It bills 0.500 USD/h until its watchdog" in last[0]
        assert last[1] == "  on the machine: 16887 pairs solved, 15 early tables, no pack"
        assert last[2].startswith("  a resume keeps all of it and makes again")
        assert last[3] == (
            "  resume:  python -m reverberate.trace rent --recipe R.json --home H --instance 1001"
        )
        assert last[4] == "  destroy: python -m reverberate.gpu.vast destroy 1001"
        assert record["left_alive"] == rental["said"][-1]
        # A machine that cannot be asked is said so, and the command is still given.
        rental["client"].alive = {1001}

        def silent(machine: Any) -> list[str]:
            raise ConnectionLost("Connection refused")

        run(rental, resume_command="resume --instance {instance}", inventory=silent)
        assert "could not be asked" in rental["said"][-1]
        assert "  resume:  resume --instance 1001" in rental["said"][-1]
        rental["client"].alive.clear()
        record = run(rental, destroy_failed=True)
        assert record["outcome"] == "failed" and record["destroyed"] is True
        assert rental["client"].alive == set()

    def test_a_fetch_that_fails_keeps_the_finished_machine_and_says_what_it_bills(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*a: Any, **k: Any) -> Path:
            raise ConnectionLost("transfer failed after 6 attempts")

        monkeypatch.setattr(onebox, "fetch", broken)
        record = run(rental)
        assert record["outcome"] == "done" and "destroyed" not in record
        assert "its fetch failed; what it made is on the machine" in record["left_alive"]
        assert "USD more at most" in record["left_alive"]
        assert "a resume launches nothing" in record["left_alive"]

    def test_a_resume_on_a_machine_whose_campaign_is_done_fetches_and_launches_nothing(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rental["client"].alive = {1001}
        monkeypatch.setattr(vast, "wait_for_ssh", lambda client, instance, identity: "machine-1")
        monkeypatch.setattr(vast, "deadline_of", lambda instance: time.time() + 7200.0)
        asked: Any = vars(onebox)["run_on"]

        def run_on(machine: Any, command: str, *, what: str, timeout: Any = None) -> str:
            if what == "done":
                return "/root/campaign/out/campaign.done\n"
            return str(asked(machine, command, what=what, timeout=timeout))

        monkeypatch.setattr(onebox, "run_on", run_on)
        record = run(rental, instance=1001, hours=None, leave=("pairs",))
        assert record["outcome"] == "done" and record["destroyed"] is True
        assert "launch" not in rental["calls"] and "watch" not in rental["calls"]
        assert "chunks pack.h5" in rental["calls"]
        assert any("nothing is launched" in line for line in rental["said"])
        # Told to, or on a machine whose campaign is not done, it launches as it did.
        rental["calls"].clear()
        rental["client"].alive = {1001}
        run(rental, instance=1001, hours=None, relaunch=True)
        assert "launch" in rental["calls"]
        rental["calls"].clear()
        rental["client"].alive = {1001}
        monkeypatch.setattr(onebox, "run_on", asked)
        run(rental, instance=1001, hours=None)
        assert "launch" in rental["calls"]

    def test_the_pack_comes_first_in_chunks_then_the_reports_then_the_pairs_if_asked(
        self, rental: dict[str, Any]
    ) -> None:
        record = run(rental)
        calls = rental["calls"]
        # What the run is for, then its reports, then what the next run would not solve again.
        assert calls.index("chunks pack.h5") < calls.index("rsync down") < calls.index("tree pairs")
        assert calls.index("tree pairs") < calls.index("teardown 1001")
        assert record["fetched"]["pack.h5"]["bytes_per_s"] == 4.4e6
        assert onebox.fetch_order(["pairs", "status.json", "pack.h5", "level.jsonl"]) == [
            "pack.h5",
            "status.json",
            "level.jsonl",
            "pairs",
        ]
        # Left on the machine: not asked for at all.
        rental["calls"].clear()
        rental["client"].alive.clear()
        run(rental, leave=("pairs",))
        assert "tree pairs" not in rental["calls"] and "chunks pack.h5" in rental["calls"]

    def test_the_instance_s_own_address_is_tried_first_and_the_proxy_is_fallen_back_on(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        through: list[Any] = []

        def fetch_file(machine: Any, remote: str, local: Path, **kw: Any) -> dict[str, Any]:
            through.append(machine)
            if machine == "its own address":
                raise ConnectionLost("Connection refused")
            return {"bytes": 1, "seconds": 1.0, "bytes_per_s": 1.0, "failures": 0}

        monkeypatch.setattr(onebox, "fastest", lambda machine, say=None: "its own address")
        monkeypatch.setattr(onebox, "fetch_file", fetch_file)
        record = run(rental, leave=("pairs",))
        assert through == ["its own address", "machine-1"]
        assert record["outcome"] == "done" and record["fetched"]["pack.h5"]["direct"] is False
        assert any("the direct way failed" in line for line in rental["said"])
        # Where it answers, the pack comes by it and the record says so.
        through.clear()
        rental["client"].alive.clear()
        monkeypatch.setattr(onebox, "fastest", lambda machine, say=None: "another address")
        assert run(rental, leave=("pairs",))["fetched"]["pack.h5"]["direct"] is True
        assert through == ["another address"]

    def test_a_pack_that_does_not_come_in_chunks_comes_as_it_did(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_dd(*a: Any, **k: Any) -> dict[str, Any]:
            raise RemoteError("chunk failed: dd: not found", 127)

        monkeypatch.setattr(onebox, "fetch_file", no_dd)
        record = run(rental, leave=("pairs",))
        assert record["outcome"] == "done" and record["fetched"]["pack.h5"]["by"] == "rsync"
        assert any("not home in chunks" in line for line in rental["said"])
        assert (["/root/campaign/out/pack.h5"]) in [sources for sources, _ in rental["pulled"]]

    def test_a_look_that_fails_is_taken_again(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real, looks = onebox.monitor_once, {"n": 0}

        def flaky(*a: Any, **k: Any) -> Any:
            looks["n"] += 1
            if looks["n"] <= 2:
                raise vast.VastError("GET /instances failed: the laptop's line")
            return real(*a, **k)

        monkeypatch.setattr(onebox, "monitor_once", flaky)
        record = run(rental)
        assert record["outcome"] == "done" and looks["n"] == 3
        assert sum("look failed" in line for line in rental["said"]) == 2

    def test_hosts_known_bad_are_never_rented_whatever_is_given(
        self, rental: dict[str, Any]
    ) -> None:
        assert vast.KNOWN_BAD_HOSTS[35928][0] == "2026-10-05"
        rental["client"].offers = [Offer(1, machine_id=35928), Offer(2, dph_total=0.5)]
        record = run(rental, avoid=[77])
        assert rental["rented"] == [2]
        assert {77, 35928} <= set(record["avoided"])
        assert any("hosts known bad" in line and "35928" in line for line in rental["said"])


# --------------------------------------------------------------------------
# the pairs come home while the run lasts
# --------------------------------------------------------------------------


class TestTheHomecoming:
    def test_the_pairs_come_home_at_every_interval_and_a_dead_host_keeps_them(
        self, rental: dict[str, Any]
    ) -> None:
        solving = answer({"stage": "solve", "updated": time.time()})
        rental["watch"] = [solving, solving, solving, "gone"]
        looks = {"n": 0}
        client = rental["client"]
        alive = client.instance

        def instance(instance_id: int) -> Any:
            # The host dies after the third look.
            looks["n"] += 1
            return alive(instance_id) if looks["n"] <= 3 else None

        client.instance = instance
        record = run(rental, sync=("pairs",), sync_s=0.0)
        assert record["outcome"] == "instance vanished"
        # Three looks, three homecomings: incremental, as they are, the files being written left.
        assert len(rental["pulled"]) == 3
        for sources, given in rental["pulled"]:
            assert sources == ["/root/campaign/out/pairs"]
            assert "*.partial.npy" in given["exclude"] and "*.partial.npz" in given["exclude"]
            assert given["seconds"] == onebox.SYNC_TIMEOUT_S
        # Nothing is fetched from a host that is gone, and nothing is left to bill.
        assert "list run" not in rental["calls"] and "left_alive" not in record
        assert record["synced"]["items"] == ["pairs"]

    def test_a_homecoming_that_fails_is_a_note_and_the_watch_goes_on(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def cut(*a: Any, **k: Any) -> dict[str, Any]:
            return {"complete": False, "left": 5, "there": 9, "error": "files failed: broken pipe"}

        monkeypatch.setattr(onebox, "fetch_tree", cut)
        said: list[str] = []
        assert onebox.sync_home("m", rental["home"], ("pairs",), said.append) is False
        assert "homecoming of pairs not complete" in said[0] and "broken pipe" in said[0]
        # A directory the campaign has not made yet is not worth a line.
        monkeypatch.setattr(
            onebox,
            "fetch_tree",
            lambda *a, **k: {"complete": False, "there": 0, "error": "No such file or directory"},
        )
        assert onebox.sync_home("m", rental["home"], ("pairs",), said.append) is False
        assert len(said) == 2
        # Another entry than the pair cache still goes by rsync, as it is.
        monkeypatch.setattr(
            onebox,
            "rsync",
            lambda *a, **k: (_ for _ in ()).throw(RemoteError("rsync down failed: reset", 12)),
        )
        assert onebox.sync_home("m", rental["home"], ("early",), said.append) is False
        assert "homecoming of early not complete" in said[2]

    def test_a_pass_that_reaches_its_limit_keeps_what_came_and_says_how_much(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = rental["home"]
        grid = home / "pulled" / "pairs" / "grid" / "aa"
        on_machine = {"n": 7}

        def bounded(machine: Any, remote_dir: str, local_dir: Path, **given: Any) -> dict[str, Any]:
            # Two pairs arrive whole, one in each form; a file an rsync of before had cut
            # is still there under its hidden name.
            grid.mkdir(parents=True, exist_ok=True)
            held = len(list(grid.glob("[!.]*.np[yz]")))
            (grid / f"aa{held}.npy").write_bytes(b"whole")
            (grid / f"aa{held + 1}.npz").write_bytes(b"whole")
            (grid / f".aa{held + 2}.npy.Xy12Zw").write_bytes(b"cu")
            assert given["seconds"] == 5.0
            return {"files": 2, "left": 3, "complete": False, "there": 8}

        monkeypatch.setattr(onebox, "fetch_tree", bounded)
        monkeypatch.setattr(onebox, "run_on", lambda *a, **k: f"{on_machine['n']}\n")
        said: list[str] = []
        assert onebox.sync_home("m", home, ("pairs",), said.append, timeout=5.0) is False
        assert onebox.sync_home("m", home, ("pairs",), said.append, timeout=5.0) is False
        assert "pairs home: 2 of 7 on the machine (+2 in" in said[0]
        assert "the pass ended at its 5 s and the next goes on" in said[0]
        assert "pairs home: 4 of 7 on the machine (+2 in" in said[1]
        assert "not complete" not in "".join(said)
        # What a pass left of its last file is not kept, and is not counted as a pair.
        assert not list(grid.glob(".*")) and onebox.pairs_home(home / "pulled") == 4
        # A machine that does not answer the count is a question mark, not a failure.
        monkeypatch.setattr(onebox, "run_on", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
        monkeypatch.setattr(
            onebox, "fetch_tree", lambda *a, **k: {"files": 0, "complete": True, "there": 5}
        )
        assert onebox.sync_home("m", home, ("pairs",), said.append) is True
        assert "pairs home: 4 of ? on the machine (+0 in" in said[2]

    def test_a_pass_is_taken_out_of_the_pause_between_two_looks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clock = {"t": 1000.0}
        slept: list[float] = []
        limits: list[float] = []
        looks = {"n": 0}

        def look(*a: Any, **k: Any) -> Any:
            looks["n"] += 1
            return SimpleNamespace(
                line=lambda: "look",
                instance_alive=looks["n"] < 4,
                done=False,
                failed=False,
                stalled=False,
            )

        def home_pass(
            machine: Any, home: Path, items: Any, say: Any, *, timeout: float, way: Any = None
        ) -> bool:
            # The transfers' way is asked once: a machine with no address of its own is itself.
            assert way == "m"
            limits.append(timeout)
            clock["t"] += timeout
            return False

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            clock["t"] += seconds

        monkeypatch.setattr(onebox, "monitor_once", look)
        monkeypatch.setattr(onebox, "sync_home", home_pass)
        monkeypatch.setattr(time, "time", lambda: clock["t"])
        monkeypatch.setattr(time, "sleep", sleep)
        onebox.watch(
            None,
            1,
            "m",
            lambda: None,
            deadline=1e9,
            poll_s=300.0,
            record={"watches": []},
            home=tmp_path,
            say=lambda m: None,
            sync=("pairs",),
        )
        # The first look syncs nothing; then every look does, and the looks stay 300 s apart.
        assert limits == [270.0, 270.0] and slept == [300.0, 30.0, 30.0]

    def test_a_pair_cut_in_its_transfer_is_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REVERBERATE_DATA", str(tmp_path / "data"))
        arrived = PairCache(tmp_path / "pulled" / "pairs", "grid")
        whole = np.ones((4, 16), dtype=np.float32)
        for key in ("aa11", "bb22", "cc33"):
            arrived.write(key, whole, {"solver": "s"})
        # ``--partial`` leaves what arrived of the last file under the file's own name.
        cut = arrived.path("bb22")
        cut.write_bytes(cut.read_bytes()[:150])
        found = install_pairs(tmp_path / "pulled" / "pairs")
        assert sorted(found["installed"]) == ["aa11", "cc33"] and found["damaged"] == ["bb22"]
        local = PairCache.local("grid")
        assert local.has("aa11") and not local.has("bb22")
        assert np.array_equal(local.read("cc33"), whole)


# --------------------------------------------------------------------------
# the machine is chosen by what the run costs on it
# --------------------------------------------------------------------------

#: A plan like the realistic recipe's: 1646 positions, ten cells each, five heard widely.
COUNTS = [770] * 5 + [8] * 1641
RECORD = {
    "source_positions": len(COUNTS),
    "pairs": sum(COUNTS),
    "cells_a_position": COUNTS,
    "tail_sites": 202,
    "tail_cells": 53,
    "step_pairs": 45864,
}


class TestTheChoice:
    def test_the_solves_count_the_records_a_card_holds(self) -> None:
        # The measured point: a 20 GiB card holds the records of 57 cells a solve.
        assert batched.cells_a_solve(20.0) == batched.CELLS_A_SOLVE_MEASURED == 57
        assert batched.cells_a_solve(16.0) < 57 < batched.cells_a_solve(24.0)
        assert batched.cells_a_solve(80.0) > 400
        # A coarser grid has smaller records and a smaller grid beside them.
        assert batched.cells_a_solve(20.0, ppw=7.2) > 2 * 57
        # A card that holds the grid and no cell's records cannot run the solve.
        assert batched.cells_a_solve(8.0) == 0 and batched.solves_needed([3], 8.0) is None
        assert batched.solves_needed(COUNTS, 20.0) == 1641 + 5 * 14
        assert batched.solves_needed(COUNTS, 80.0) == 1641 + 5 * 2
        assert batched.solves_needed([0, 57, 58], 20.0) == 3

    def test_the_estimate_counts_the_extra_solves_and_the_way_home(self) -> None:
        priced = estimate(RECORD, rate_usd_per_hour=0.136)
        assert priced["solves"] == 1711 and priced["extra_solves"] == 65
        assert priced["low"]["solves"] == 1711
        low = batched.ONCE_S + 1711 * batched.SOLVE_S_AT_1500 + sum(COUNTS) * batched.PAIR_S
        assert priced["seconds"]["low"] == pytest.approx(low, abs=0.1)
        # The pack's bytes through the proxy: gigabytes at 4.4 MB/s are over an hour.
        assert priced["seconds"]["transfer_pack"] == pytest.approx(
            priced["pack_gb"] * 1e9 / 4.4e6, rel=1e-3
        )
        assert priced["pack_gb"] > 9 and priced["seconds"]["transfer_pack"] > 2400
        # Written as the bins in 16 bits the pack is 0.357 of its low band, and the pair
        # cache half; the decay's lever is a listening variant's 0.14.
        compact = estimate(RECORD, rate_usd_per_hour=0.136, low_levers="bins,int16")
        pairs = sum(COUNTS)
        assert priced["pack_gb"] - compact["pack_gb"] == pytest.approx(
            pairs * (1_228_800 - 439_080) / 1e9, abs=0.01
        )
        assert compact["pair_cache_gb"] == pytest.approx(pairs * 621_446 / 1e9, abs=0.01)
        assert compact["low_levers"] == "bins,int16" and "low_levers" not in priced
        assert estimate(RECORD, rate_usd_per_hour=0.136, low_levers="none") == priced
        decay = estimate(RECORD, rate_usd_per_hour=0.136, low_levers="bins,int16,decay=60")
        assert decay["pack_gb"] < 0.6 * compact["pack_gb"]
        assert decay["pair_cache_gb"] == compact["pair_cache_gb"]
        assert compact["seconds"]["write"] > priced["seconds"]["write"]
        # A record without the cells of each position prices the positions alone.
        bare = {k: v for k, v in RECORD.items() if k != "cells_a_position"}
        assert estimate(bare, rate_usd_per_hour=0.136)["extra_solves"] == 0

    def test_the_coarser_grid_is_priced_from_its_own_measurement(self) -> None:
        coarse = batched.estimate(100, 0, fmax_hz=1500.0, ppw=7.2)
        # 36 s a source position, measured on the same card, against 110.
        assert coarse["stencil_s_per_source"] == pytest.approx(36.0, abs=0.5)
        assert batched.estimate(100, 0, fmax_hz=1500.0)["stencil_s_per_source"] == 110.0
        priced = estimate(RECORD, rate_usd_per_hour=0.136, low_ppw=7.2)
        assert priced["low_ppw"] == 7.2 and priced["low"]["ppw"] == 7.2
        assert (
            priced["seconds"]["low"]
            < 0.4 * estimate(RECORD, rate_usd_per_hour=0.136)["seconds"]["low"]
        )
        with pytest.raises(ValueError, match="only the batched solver"):
            estimate(RECORD, rate_usd_per_hour=1.74, low_engine="pffdtd", low_ppw=7.2)

    def test_the_table_says_which_cards_were_measured(self) -> None:
        measured = {card.name for card in machines.CARDS if card.measured}
        assert measured == {"RTX 3080", "RTX 3090", "A100", "Tesla P100"}
        assert machines.card_of("RTX 3080").throughput == 1.0  # type: ignore[union-attr]
        assert machines.card_of("RTX 3090").throughput == 1.25  # type: ignore[union-attr]
        # An offer's name, as Vast writes it, finds its row and not a shorter one's.
        named = {
            "RTX 3080 Ti": "RTX 3080 Ti",
            "RTX 3090": "RTX 3090",
            "A100 SXM4": "A100",
            "A100_PCIE": "A100",
            "Tesla P100": "Tesla P100",
            "P100": "Tesla P100",
            "RTX A4000": "RTX A4000",
            "A40": "A40",
        }
        for offered, row in named.items():
            assert machines.card_of(offered).name == row, offered  # type: ignore[union-attr]
        assert machines.card_of("GTX 1080") is None and machines.card_of("L4") is None
        table = "\n".join(machines.throughput_table())
        assert "ESTIMATED" in table and "measured" in table and "183 s" in table

    def test_a_prediction_is_the_queue_over_the_cards_with_the_fetch_in_it(self) -> None:
        def on(**offer: Any) -> dict[str, Any]:
            told = machines.predict(RECORD, **{"dph_total": 0.14, "gpu_ram_gb": 20.0, **offer})
            assert told is not None
            return told

        one, four = on(gpu_name="RTX 3080", num_gpus=1), on(gpu_name="RTX 3080", num_gpus=4)
        # A launch's records are on the host: a position is solved once whatever a card holds.
        assert one["solves"] == len(COUNTS) and one["extra_solves"] == 0 and one["measured"]
        # Today's code is measured on the RTX 3090: another card by the two throughputs.
        unit = 70.5 * 1.25
        assert one["work"] == four["work"] and one["source_s"] == round(unit, 1)
        assert one["launches"] == -(-len(COUNTS) // batched.LAUNCH_SOURCES)
        # The cards divide the solves, the fits and the rays; several idle half a launch
        # each at the end, where one card has no end to wait for.
        tail = 0.5 * batched.LAUNCH_SOURCES * unit
        assert four["seconds"]["low"] == pytest.approx(
            (one["seconds"]["low"] - 55.0) / 4 + 55.0 + tail, abs=0.5
        )
        assert four["seconds"]["rays"] == pytest.approx(one["seconds"]["rays"] / 4, abs=0.1)
        # The host's stages are under the solves, and the rest is the same on both.
        assert (
            one["seconds"]["host_beyond_the_cards"] == four["seconds"]["host_beyond_the_cards"] == 0
        )
        for stage in ("start", "prepare", "write", "check", "transfer_pack"):
            assert four["seconds"][stage] == one["seconds"][stage]
        assert one["hours"] == pytest.approx(sum(one["seconds"].values()) / 3600.0, abs=1e-3)
        assert one["hours"] > 40 and four["hours"] < 0.3 * one["hours"]
        assert one["usd"] == pytest.approx(one["hours"] * 0.14)
        # The code as it solves today, on one RTX 3090: 70.5 s a position with the walls'
        # seven branches and 0.29 s a pair's fit (solver-boundary.md), 0.54 s a tail site
        # through the tree (ray-tracer.md), 1.9 ms a position of the early trace and 25 ms
        # a pair levelled on a core (the appendix of every card and every core).
        big = on(gpu_name="RTX 3090", num_gpus=8, gpu_ram_gb=24.0, dph_total=1.38)
        assert big["source_s"] == 70.5 and big["measured"] and big["card"] == "RTX 3090"
        pairs, launches = sum(COUNTS), -(-len(COUNTS) // batched.LAUNCH_SOURCES)
        assert big["work"]["solve_card_s"] == pytest.approx(
            len(COUNTS) * 70.5 + launches * 8.6, abs=0.1
        )
        assert big["work"]["fit_card_s"] == pytest.approx(pairs * 0.29, abs=0.1)
        assert big["work"]["rays_card_s"] == pytest.approx(202 * 0.54, abs=0.1)
        assert on(gpu_name="RTX 3090", num_gpus=8, rays=50_000)["work"][
            "rays_card_s"
        ] == pytest.approx(101 * 0.54, abs=0.1)
        assert big["work"]["host_core_s"] == pytest.approx(
            2.0 * ((45864 + pairs) * 0.0019 + pairs * (0.025 + 0.010)), abs=0.1
        )
        assert on(gpu_name="RTX 3090", num_gpus=4, low_ppw=7.2)["source_s"] == 34.6
        # A card and a grid no run measured: from the RTX 3090's, the throughputs, the points.
        assert on(gpu_name="RTX 4090", num_gpus=4)["source_s"] == pytest.approx(unit / 1.3, abs=0.1)
        assert not on(gpu_name="RTX 4090", num_gpus=4)["measured"]
        assert on(gpu_name="RTX 3080", num_gpus=4, low_ppw=7.2)["source_s"] == pytest.approx(
            34.6 * 1.25, abs=0.1
        )
        assert on(gpu_name="RTX 3090", num_gpus=4, low_ppw=9.0)["source_s"] == pytest.approx(
            70.5 * (9.0 / 10.5) ** batched.PPW_EXPONENT, abs=0.1
        )
        # The fetch is in the total, at the machine's rate: the form the pack is written in
        # is minutes of the rental, and the same pack costs a dearer machine more.
        plain = on(gpu_name="RTX 3090", num_gpus=8, dph_total=1.38, low_levers="none")
        compact = on(gpu_name="RTX 3090", num_gpus=8, dph_total=1.38, low_levers="bins,int16")
        assert compact["seconds"]["transfer_pack"] < 0.5 * plain["seconds"]["transfer_pack"]
        assert compact["fetch_usd"] == pytest.approx(
            compact["seconds"]["transfer_pack"] / 3600.0 * 1.38, abs=1e-3
        )
        assert plain["usd"] - compact["usd"] > 0.4
        cheap = on(gpu_name="RTX 3090", num_gpus=8, dph_total=0.69, low_levers="bins,int16")
        assert cheap["fetch_usd"] == pytest.approx(compact["fetch_usd"] / 2, abs=1e-3)
        assert "the fetch" in compact["note"]
        # The rental's hour is the offer's and its disk's: the total is of what is billed.
        told = machines.predictor(RECORD, disk_gb=219.0)(
            SimpleNamespace(
                gpu_name="RTX 3090",
                num_gpus=4,
                gpu_ram_gb=24.0,
                dph_total=0.543,
                billed_dph=lambda disk_gb: 0.543 + 0.85 * (disk_gb - 5.0) / 730.0,
            )
        )
        assert told["rate_usd_per_hour"] == pytest.approx(0.792, abs=0.001)
        assert told["usd"] == pytest.approx(told["hours"] * told["rate_usd_per_hour"], abs=0.01)
        assert "billed 0.792 USD/h with its disk" in told["note"]
        # A direct line is the laptop's to where the host is, and is not a measurement.
        there = {"gpu_name": "RTX 3090", "num_gpus": 4, "line": "direct"}
        europe = on(**there, location="Czechia, CZ", inet_up_mbps=900.0)
        states = on(**there, location="California, US", inet_up_mbps=900.0)
        slow = on(**there, location="Czechia, CZ", inet_up_mbps=40.0)
        proxy = on(gpu_name="RTX 3090", num_gpus=4, location="Czechia, CZ")
        assert europe["line_bytes_per_s"] == 70e6 and states["line_bytes_per_s"] == 17e6
        assert slow["line_bytes_per_s"] == pytest.approx(0.8 * 40e6 / 8)
        assert proxy["line_bytes_per_s"] == 4.4e6 and proxy["line_measured"]
        assert not europe["line_measured"]
        assert europe["usd"] < states["usd"] < proxy["usd"]
        # A host of two cores takes longer over its stages than eight fast cards over theirs.
        fast: dict[str, Any] = {
            "gpu_name": "A100",
            "num_gpus": 8,
            "gpu_ram_gb": 80.0,
            "dph_total": 2.0,
        }
        walked: dict[str, Any] = {**RECORD, "step_pairs": 4_586_400}
        few = machines.predict(walked, **fast, cpu_cores=2.0)
        many = machines.predict(walked, **fast, cpu_cores=64.0)
        assert few is not None and many is not None
        assert (
            few["seconds"]["host_beyond_the_cards"] > 0 == many["seconds"]["host_beyond_the_cards"]
        )
        # The pairs come home while the machine works: no time of their own unless left over.
        late = machines.predict(
            RECORD,
            gpu_name="RTX 3080",
            num_gpus=1,
            gpu_ram_gb=20.0,
            dph_total=0.14,
            synced=False,
        )
        assert late is not None and one["seconds"]["transfer_pairs"] == 0.0
        assert late["seconds"]["transfer_pairs"] > 2000
        # The trace of before the queue: records on the card, one process after the solves.
        old = on(gpu_name="RTX 3080", num_gpus=1, queue=False)
        assert old["solves"] == 1711 and old["extra_solves"] == 65 and not old["queue"]
        # And the code of those runs: eleven branches, the rays through the grid.
        assert old["source_s"] == 110.0
        ran = on(gpu_name="RTX 3090", num_gpus=8, gpu_ram_gb=24.0, queue=False)
        assert ran["source_s"] == 88.0
        assert ran["work"]["fit_card_s"] == pytest.approx(pairs * 0.55, abs=0.1)
        assert ran["work"]["rays_card_s"] == pytest.approx(202 * (0.53 + 53 * 0.001), abs=0.1)
        assert old["seconds"]["paths"] > 0 and old["seconds"]["level"] > 0
        assert "host_beyond_the_cards" not in old["seconds"]
        # No figure, no price: an unknown card, a card too small, the present engine.
        assert (
            on.__name__
            and machines.predict(
                RECORD, gpu_name="GTX 1080", num_gpus=4, gpu_ram_gb=20.0, dph_total=0.1
            )
            is None
        )
        assert (
            machines.predict(RECORD, gpu_name="RTX 3080", num_gpus=4, gpu_ram_gb=8.0, dph_total=0.1)
            is None
        )
        assert (
            machines.predict(
                RECORD,
                gpu_name="RTX 3080",
                num_gpus=1,
                gpu_ram_gb=20.0,
                dph_total=0.1,
                low_engine="pffdtd",
            )
            is None
        )

    def test_a_run_s_own_log_is_read_against_its_prediction(self, tmp_path: Path) -> None:
        """Two attempts as the first whole scene's log wrote them, and a watch's bill."""
        home = tmp_path / "A"
        (home / "pulled").mkdir(parents=True)
        (home / "bundle" / "trace").mkdir(parents=True)
        (home / "bundle" / "campaign.json").write_text(
            json.dumps({"trace": {"low": {"ppw": None}, "low_levers": "none"}})
        )
        (home / "bundle" / "trace" / "plan.json").write_text(json.dumps(RECORD))
        day = "2026-10-05 "
        first = "trace hssd_0076: 24001 steps, 14 source(s), 831 cell(s), profile {}"
        planned = "solve(s) in {} batch(es) on 8 card(s), up to 14 at once"
        log = [
            f"05:43:27 +0.00 h | {first}",
            "05:43:27 +0.00 h | voxelise: start",
            "05:44:18 +0.02 h | voxelise: done in 0.8 min",
            "05:45:15 +0.03 h | solve: start",
            f"05:46:12 +0.05 h | solve: 1600 {planned.format(304)}",
            "05:49:02 +0.09 h | batch of 1: 89 pair(s), solve 91.868 s at 1.7e+10 updates/s,"
            " encode 75.666 s",
            "06:19:15 +0.60 h | trace FAILED: OutOfMemoryError('Out of memory')",
            f"06:29:20 +0.00 h | {first}",
            "06:30:30 +0.02 h | solve: start",
            f"06:31:27 +0.04 h | solve: 1505 {planned.format(210)}",
            "11:24:20 +4.92 h | batch of 14: 14 pair(s), solve 1229.698 s at 1.77e+10 updates/s,"
            " encode 3.791 s",
            "11:26:13 +4.95 h | solve: done in 295.7 min",
            "11:26:13 +4.95 h | paths: start",
            "11:44:05 +5.25 h | paths: done in 17.9 min",
            "11:44:05 +5.25 h | level: start",
            "12:25:55 +5.94 h | trace FAILED: RuntimeError('not on one clock')",
        ]
        (home / "pulled" / "campaign.log").write_text("\n".join(day + line for line in log))
        (home / "driver.log").write_text(
            "08:13:44 up 0.64 h cost 0.89 USD credit 28.04 | solve\n"
            "14:28:00 up 6.88 h cost 9.50 USD credit 12.74 | failed\n"
        )
        (home / "onebox.json").write_text(
            json.dumps({"watches": ["14:39:44 up 7.07 h cost 9.77 USD credit 12.15 | done"]})
        )
        took = machines.actual(home)
        assert len(took["attempts"]) == 2 and took["attempts"][0]["failed"].startswith("OutOf")
        # A stage that did not end, ended with its attempt: 34 minutes of solves, then 295.7.
        assert took["stages"]["solve"] == pytest.approx(34 * 60 + 295.7 * 60, abs=45)
        assert took["stages"]["paths"] == 1072.0
        assert took["stages"]["level"] == pytest.approx(41 * 60 + 50, abs=1)
        assert took["idle_s"] == 605.0 and took["launches"] == 2 and took["sources"] == 15
        assert took["solve_card_s"] == pytest.approx(91.868 + 1229.698, abs=0.1)
        assert took["longest_launch_s"] == pytest.approx(1233.5, abs=0.1)
        assert took["billed"] == {"hours": 7.07, "usd": 9.77, "rate_usd_per_hour": 1.382}
        assert took["attempts"][0]["planned"]["launch_sources"] == 14
        told = machines.against(home, gpu_name="RTX 3090", num_gpus=8, gpu_ram_gb=24.0)
        # A log with a paths stage of its own is of before the queue, and is priced so:
        # as that run planned its launches, at the rate its watches say.
        assert not told["predicted"]["queue"] and told["rate_usd_per_hour"] == 1.382
        assert told["predicted"]["launches"] == 304
        text = "\n".join(told["lines"])
        assert "8 x RTX 3090 at 1.382 USD/h, the trace of before the queue" in text
        assert "paths (one process)" in text and "between attempts" in text
        assert "billed at the last watch: 7.07 h, 9.77 USD" in text
        queued = machines.against(
            home, gpu_name="RTX 3090", num_gpus=8, gpu_ram_gb=24.0, queue=True, fetch_s=2160.0
        )
        assert queued["predicted"]["queue"]
        assert "host stages beyond the cards" in "\n".join(queued["lines"])
        asked = ["ledger", "--home", str(home), "--gpu", "RTX 3090", "--gpus", "8"]
        assert main([*asked, "--gpu-ram", "24"]) == 0

    def offers(self) -> list[Offer]:
        return [
            Offer(1, "RTX 3080", 1, 0.14),  # the cheapest hour, 55 h of it
            Offer(2, "RTX 3080", 4, 0.60),
            Offer(3, "RTX 3090", 4, 0.80, 24.0),
            Offer(4, "A100 SXM4", 2, 1.80, 80.0),
            Offer(5, "GTX 1080", 8, 0.10, 20.0),  # no figure
            Offer(6, "RTX 3090", 2, 0.34, 24.0),
        ]

    def test_the_offer_taken_is_the_lowest_total_within_the_wall_time(
        self, rental: dict[str, Any]
    ) -> None:
        rental["client"].offers = self.offers()
        predict = machines.predictor(RECORD)
        said: list[str] = []
        need = onebox.campaign_need(rental["bundle"])
        common: dict[str, Any] = {"max_dph": 3.0, "min_ram_gb": 60.0, "gpu": "", "say": said.append}
        priced = {o.id: predict(o) for o in self.offers()}
        assert priced[5] is None
        # Without a limit of wall time, the lowest total; here that is not the cheapest hour.
        free = onebox.plan_rental(rental["client"], need, hours=None, predict=predict, **common)
        totals = [p.usd for p in free.priced]
        assert totals == sorted(totals) and 5 not in [p.offer.id for p in free.priced]
        assert free.offers[0].id == min(
            (i for i in priced if priced[i] is not None),
            key=lambda i: priced[i]["usd"],
        )
        # Within fourteen hours the single card and the pair of cards are out, and so says the log.
        said.clear()
        plan = onebox.plan_rental(
            rental["client"], need, hours=None, predict=predict, max_hours=14.0, **common
        )
        kept = [p.offer.id for p in plan.priced]
        assert kept == [2, 3, 4]
        assert all(priced[i]["hours"] <= 14.0 for i in kept)
        first = plan.priced[0]
        assert first.usd == min(priced[i]["usd"] for i in kept)
        text = "\n".join(said)
        assert "1 offer(s) left out: a card the prediction has no figure for" in text
        assert "2 offer(s) left out: predicted over the 14 h of wall time allowed" in text
        assert f"chosen: offer {first.offer.id}" in text and "measured" in text
        assert "the fetch" in text
        # The watchdog is the largest cap of the offers that may be tried, margin stated.
        caps = [onebox.cap_hours(p) for p in plan.priced]
        assert plan.hours == max(caps) and "watchdog:" in text and "times 1.5" in text
        assert onebox.cap_hours(first) == pytest.approx(
            first.hours * (1.5 if first.measured else 2.0) + 0.5, abs=0.01
        )
        # Two cards or more: the single card is not even asked for.
        rental["client"].searched.clear()
        two = onebox.plan_rental(
            rental["client"], need, hours=None, predict=predict, min_gpus=2, **common
        )
        assert all(p.offer.num_gpus >= 2 for p in two.priced)
        assert not any("num_gpus=1 " in query for query in rental["client"].searched)
        # A cap given under the prediction is said aloud before anything is rented.
        said.clear()
        onebox.plan_rental(rental["client"], need, hours=1.0, predict=predict, **common)
        assert any("THE CAP GIVEN IS UNDER THE PREDICTION" in line for line in said)

    def test_the_run_rents_the_offer_chosen_with_the_cap_predicted(
        self, rental: dict[str, Any]
    ) -> None:
        rental["client"].offers = self.offers()
        record = run(rental, hours=None, predict=machines.predictor(RECORD), max_hours=12.0, gpus=2)
        chosen = record["offers"][0]
        assert rental["rented"] == [chosen["offer"]] and record["offer"] == chosen["offer"]
        assert rental["hours"] == [record["cap_hours"]] and record["cap_hours"] > chosen["hours"]
        assert record["outcome"] == "done"

    def test_the_disk_asked_is_the_plan_s_with_its_margin_said(
        self, rental: dict[str, Any]
    ) -> None:
        """The first scenes asked 219 GB, a field campaign's formula, and used 53."""
        scene = {**RECORD, "source_positions": 1529, "pairs": 18_364, "cells_a_position": []}
        need = machines.disk_need(scene, low_levers="bins,int16")
        parts = need["parts_gb"]
        # While the pack is written: the pair cache as it is kept, every pair's row as
        # samples, the pack in its form, a histogram a (site, cell), the mirror's store.
        assert parts["pairs"] == pytest.approx(18_364 * 621_446 / 1e9, abs=0.01)
        assert parts["rows"] == pytest.approx(18_364 * 1_228_800 / 1e9, abs=0.01)
        assert parts["pack"] == pytest.approx(
            (18_364 * 439_080 + 202 * 53 * 286_000) / 1e9, abs=0.01
        )
        assert parts["tails"] == pytest.approx(202 * 53 * 660_000 / 1e9, abs=0.01)
        assert parts["mirror_store"] == 8.0 and parts["rest"] == 3.0 and parts["bundle"] == 0.0
        counted = sum(parts.values())
        assert need["counted_gb"] == pytest.approx(counted, abs=0.01)
        assert counted == pytest.approx(63.2, abs=0.1)
        # The margin is stated: a quarter of what is counted, and ten for the machine's own.
        assert need["margin_gb"] == pytest.approx(0.25 * counted, abs=0.01)
        assert need["disk_gb"] == 89 and need["system_gb"] == 10.0
        assert need["disk_gb"] >= counted + need["margin_gb"] + 10.0 > need["disk_gb"] - 1
        line = need["line"]
        assert line.startswith("disk: 89 GB asked of the rental: the pair cache 11.41,")
        assert "the pack's rows until it is checked 22.57" in line
        assert "the pack (bins,int16) 11.13" in line and "the bundle" not in line
        assert "63.2 GB counted, 25% more (15.8) and 10 for the machine's own" in line
        # The samples are more of both; a bundle that carries pairs is counted as pushed.
        plain = machines.disk_need(scene, low_levers="none", bundle_gb=4.0)
        assert plain["parts_gb"]["pairs"] == plain["parts_gb"]["rows"] > 22
        assert plain["parts_gb"]["bundle"] == 4.0 and "the bundle 4" in plain["line"]
        assert plain["disk_gb"] > need["disk_gb"] + 20
        # A plan that counts its pairs on the grid is sized by that count.
        counted_so = {**scene, "pairs": 16_887, "on_the_grid": {**scene, "ppw": 10.5}}
        assert machines.disk_need(counted_so, low_levers="bins,int16") == need
        assert machines.disk_need({**scene, "pairs": 16_887})["disk_gb"] < plain["disk_gb"]
        # The rental is searched, asked and billed with it; left out, the field's formula.
        field = onebox.campaign_need(rental["bundle"]).disk_gb
        record = run(rental, disk_gb=need["disk_gb"])
        assert record["disk_gb"] == 89 != field and rental["disk"] == [89]
        assert any("89 GB of disk" in line for line in rental["said"])
        assert all("disk_space>89 " in query for query in rental["client"].searched)

    def test_the_engine_is_built_only_for_a_campaign_that_opens_it(
        self, rental: dict[str, Any]
    ) -> None:
        """PFFDTD was built on every rental, 216 and 498 s, for a solver that never ran it."""
        built = run(rental)
        assert "build machine-1" in rental["calls"] and "compiler" not in rental["calls"]
        assert built["engine_built"] is True and built["outcome"] == "done"
        assert any(line.startswith("engine built in") for line in rental["said"])
        rental["calls"].clear()
        rental["said"].clear()
        rental["home"] = rental["home"].with_name("home_without")
        plain = run(rental, engine_build=False)
        calls = rental["calls"]
        assert not any(call.startswith("build") for call in calls)
        # What is left of the build: a C compiler asked for, then the interpreter as before.
        assert calls.index("compiler") < calls.index("provision") < calls.index("launch")
        assert plain["engine_built"] is False and plain["outcome"] == "done"
        assert "engine not built: this campaign's solver does not open it" in rental["said"]
        assert "command -v cc" in onebox.COMPILER_COMMAND
        assert "build-essential" in onebox.COMPILER_COMMAND
        # The prediction follows: the start without the build, and with it where it is asked.
        on: dict[str, Any] = {"gpu_name": "RTX 3090", "num_gpus": 8, "gpu_ram_gb": 24.0}
        on["dph_total"] = 1.38
        without = machines.predict(RECORD, **on)
        with_it = machines.predict(RECORD, **on, engine_build=True)
        before = machines.predict(RECORD, **on, queue=False)
        assert without is not None and with_it is not None and before is not None
        assert without["seconds"]["start"] == 290.0 and not without["engine_built"]
        assert with_it["seconds"]["start"] == before["seconds"]["start"] == 290.0 + 357.0
        assert with_it["hours"] - without["hours"] == pytest.approx(357.0 / 3600.0, abs=1e-3)
        assert machines.predictor(RECORD, engine_build=True)(SimpleNamespace(**on))["engine_built"]

    def test_a_preferred_region_orders_the_offers_and_leaves_none_out(
        self, rental: dict[str, Any]
    ) -> None:
        """A pack came home at 14 to 57 MB/s from France, and at 5 to 14 through the proxy."""
        offers = self.offers()
        places = {2: "California, US", 3: "France, FR", 4: "Quebec, CA", 6: "France, FR"}
        for offer in offers:
            offer.location = places.get(offer.id, "Texas, US")  # type: ignore[attr-defined]
        rental["client"].offers = offers
        predict = machines.predictor(RECORD)
        need = onebox.campaign_need(rental["bundle"])
        said: list[str] = []
        common: dict[str, Any] = {"max_dph": 3.0, "min_ram_gb": 60.0, "gpu": "", "say": said.append}
        common |= {"hours": None, "predict": predict}
        free = onebox.plan_rental(rental["client"], need, **common)
        plain = [p.offer.id for p in free.priced]
        assert not any("preferred" in line for line in said)
        # France first, each region's offers by their totals; then the others, by theirs.
        there = onebox.plan_rental(rental["client"], need, prefer_regions=["FR"], **common)
        ranked = [p.offer.id for p in there.priced]
        french = [i for i in plain if i in (3, 6)]
        assert ranked == french + [i for i in plain if i not in (3, 6)]
        assert sorted(ranked) == sorted(plain) and there.offers[0].id == french[0]
        text = "\n".join(said)
        assert f"preferred region(s) FR: offer {french[0]} (France, FR)," in text
        assert f"the lowest total anywhere is offer {plain[0]} (" in text
        assert "the lowest total first within a preferred region" in text
        assert f"chosen: offer {french[0]}" in text
        # The watchdog's cap is still the largest of the offers that may be tried.
        assert there.hours == max(onebox.cap_hours(p) for p in there.priced)
        # Several regions are tried in the order given, by a code or by a place's name.
        two = onebox.plan_rental(rental["client"], need, prefer_regions=["quebec", "FR"], **common)
        assert [p.offer.id for p in two.priced][:3] == [4, *french]
        # A region no offer is in changes nothing, and says so; neither does the wall time.
        said.clear()
        none = onebox.plan_rental(rental["client"], need, prefer_regions=["JP"], **common)
        assert [p.offer.id for p in none.priced] == plain
        assert "  no offer in the preferred region(s) JP: the order is the totals' own" in said
        within = onebox.plan_rental(
            rental["client"], need, prefer_regions=["FR"], max_hours=14.0, **common
        )
        assert [p.offer.id for p in within.priced] == [3, 2, 4]
        # A host that says it has no open port is reached through the proxy: not preferred.
        offers[5].direct_ports = 0  # type: ignore[attr-defined]
        closed = onebox.plan_rental(rental["client"], need, prefer_regions=["FR"], **common)
        assert [p.offer.id for p in closed.priced] == [3, *[i for i in plain if i != 3]]
        # Where the lowest total is in the region already, that is said and nothing moves.
        said.clear()
        same = onebox.plan_rental(
            rental["client"], need, prefer_regions=[places.get(plain[0], "Texas")], **common
        )
        assert same.priced[0].offer.id == plain[0]
        assert any("the lowest total is there already" in line for line in said)
        # Without a prediction the cheapest hour of the region is first, and the run rents it.
        common.pop("predict")
        hourly = onebox.plan_rental(rental["client"], need, prefer_regions=["CA"], **common)
        assert hourly.offers[0].id == 4 and len(hourly.offers) == len(free.offers) + 1
        rental["home"] = rental["home"].with_name("home_fr")
        record = run(rental, hours=None, predict=predict, prefer_regions=("FR",), gpus=2)
        assert rental["rented"] == [3] and record["offer"] == 3 and record["outcome"] == "done"

    def test_the_offers_are_said_and_nothing_is_rented(self, rental: dict[str, Any]) -> None:
        rental["client"].offers = self.offers()
        record = run(
            rental, yes=False, hours=None, predict=machines.predictor(RECORD), plan_only=True
        )
        assert rental["rented"] == [] and "instance" not in record and "outcome" not in record
        assert len(record["offers"]) == 5 and record["cap_hours"] > 0
        assert rental["said"][-1] == "offers planned: nothing rented"
        assert any("predicted for this run, the lowest total first" in s for s in rental["said"])
