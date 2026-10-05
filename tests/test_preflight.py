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

    monkeypatch.setattr(auth, "inject", lambda names: None)
    monkeypatch.setattr(vast, "VastClient", lambda timeout: client)
    monkeypatch.setattr(vast, "account_identity", lambda client: tmp_path / "id")
    monkeypatch.setattr(vast, "teardown", teardown)
    monkeypatch.setattr(vast, "rent_one", rent_one)
    monkeypatch.setattr(onebox, "provision", provision)
    monkeypatch.setattr(onebox, "run_on", run_on)
    monkeypatch.setattr(onebox, "rsync", rsync)
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
            assert given["compress"] is False and "*.partial.npy" in given["exclude"]
            assert given["attempts"] == 1 and given["timeout"] == onebox.SYNC_TIMEOUT_S
        # Nothing is fetched from a host that is gone, and nothing is left to bill.
        assert "list run" not in rental["calls"] and "left_alive" not in record
        assert record["synced"]["items"] == ["pairs"]

    def test_a_homecoming_that_fails_is_a_note_and_the_watch_goes_on(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def cut(*a: Any, **k: Any) -> None:
            raise RemoteError("rsync down failed: broken pipe", 12)

        monkeypatch.setattr(onebox, "rsync", cut)
        said: list[str] = []
        assert onebox.sync_home("m", rental["home"], ("pairs",), said.append) is False
        assert "homecoming of pairs not complete" in said[0]
        # A directory the campaign has not made yet is not worth a line.
        monkeypatch.setattr(
            onebox,
            "rsync",
            lambda *a, **k: (_ for _ in ()).throw(RemoteError("No such file or directory", 23)),
        )
        assert onebox.sync_home("m", rental["home"], ("pairs",), said.append) is False
        assert len(said) == 2

    def test_a_pass_that_reaches_its_limit_keeps_what_came_and_says_how_much(
        self, rental: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = rental["home"]
        grid = home / "pulled" / "pairs" / "grid" / "aa"
        on_machine = {"n": 7}

        def bounded(machine: Any, sources: Any, target: str, **given: Any) -> None:
            # Two files arrive whole; the third is cut under rsync's hidden name.
            grid.mkdir(parents=True, exist_ok=True)
            held = len(list(grid.glob("[!.]*.npy")))
            for index in range(held, held + 2):
                (grid / f"aa{index}.npy").write_bytes(b"whole")
            (grid / f".aa{held + 2}.npy.Xy12Zw").write_bytes(b"cu")
            raise RemoteError(f"rsync down did not end in {given['timeout']:g} s")

        monkeypatch.setattr(onebox, "rsync", bounded)
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
        monkeypatch.setattr(onebox, "rsync", lambda *a, **k: None)
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

        def home_pass(machine: Any, home: Path, items: Any, say: Any, *, timeout: float) -> bool:
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
        # The pack's bytes at the laptop's line: gigabytes at 8 MB/s are most of an hour.
        assert priced["seconds"]["transfer_pack"] == pytest.approx(
            priced["pack_gb"] * 1e9 / 8.1e6, rel=1e-3
        )
        assert priced["pack_gb"] > 9 and priced["seconds"]["transfer_pack"] > 1200
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
        assert measured == {"RTX 3080", "A100", "Tesla P100"}
        assert machines.card_of("RTX 3080").throughput == 1.0  # type: ignore[union-attr]
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

    def test_a_prediction_divides_the_low_band_and_the_rays_over_the_cards(self) -> None:
        def on(**offer: Any) -> dict[str, Any]:
            told = machines.predict(RECORD, **{"dph_total": 0.14, "gpu_ram_gb": 20.0, **offer})
            assert told is not None
            return told

        one, four = on(gpu_name="RTX 3080", num_gpus=1), on(gpu_name="RTX 3080", num_gpus=4)
        assert one["solves"] == 1711 and one["extra_solves"] == 65 and one["measured"]
        once = batched.ONCE_S
        assert four["seconds"]["low"] - once == pytest.approx(
            (one["seconds"]["low"] - once) / 4, abs=0.2
        )
        assert four["seconds"]["rays"] == pytest.approx(one["seconds"]["rays"] / 4, abs=0.1)
        # The host's stages and the pack's way home are the same on both.
        for stage in ("paths", "level", "write", "transfer_pack", "fixed"):
            assert four["seconds"][stage] == one["seconds"][stage]
        assert one["hours"] > 50 and four["hours"] < 0.3 * one["hours"]
        assert one["usd"] == pytest.approx(one["hours"] * 0.14)
        # A larger card makes fewer solves; a faster one makes them sooner; neither is measured.
        big = on(gpu_name="RTX 3090", num_gpus=1, gpu_ram_gb=24.0)
        assert big["solves"] < one["solves"] and not big["measured"]
        assert big["seconds"]["low"] < one["seconds"]["low"]
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
        # Within twenty hours the single card and the pair of cards are out, and so says the log.
        said.clear()
        plan = onebox.plan_rental(
            rental["client"], need, hours=None, predict=predict, max_hours=20.0, **common
        )
        kept = [p.offer.id for p in plan.priced]
        assert kept == [2, 3, 4]
        assert all(priced[i]["hours"] <= 20.0 for i in kept)
        first = plan.priced[0]
        assert first.usd == min(priced[i]["usd"] for i in kept)
        text = "\n".join(said)
        assert "1 offer(s) left out: a card the prediction has no figure for" in text
        assert "2 offer(s) left out: predicted over the 20 h of wall time allowed" in text
        assert f"chosen: offer {first.offer.id}" in text and "ESTIMATED" in text
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

    def test_the_offers_are_said_and_nothing_is_rented(self, rental: dict[str, Any]) -> None:
        rental["client"].offers = self.offers()
        record = run(
            rental, yes=False, hours=None, predict=machines.predictor(RECORD), plan_only=True
        )
        assert rental["rented"] == [] and "instance" not in record and "outcome" not in record
        assert len(record["offers"]) == 5 and record["cap_hours"] > 0
        assert rental["said"][-1] == "offers planned: nothing rented"
        assert any("predicted for this run, the lowest total first" in s for s in rental["said"])
