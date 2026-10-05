"""Tests for the direct route to a rented machine and the pinning of its host keys.

What these guard is a connection that believes the wrong machine, or lends
it something: so the tests that matter are that the keys are read through
the route already trusted, that a direct command names the pinned file and
no other, that nothing is forwarded, and that every way the pinning can fail
leaves the caller a machine through the proxy with no unchecked address.

Fake clients and a fake ssh. Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import base64
import stat
from pathlib import Path
from typing import Any

import pytest

from reverberate import settings
from reverberate.gpu import direct, hostkeys, vast
from reverberate.gpu.vast import Instance, Offer
from reverberate.wave.remote import Machine, RemoteError


def blob(kind: str, body: bytes = b"\x01" * 32) -> str:
    """A key blob that names its kind, as a real one does."""
    name = kind.encode()
    raw = len(name).to_bytes(4, "big") + name + len(body).to_bytes(4, "big") + body
    return base64.b64encode(raw).decode()


ED = f"ssh-ed25519 {blob('ssh-ed25519')} root@box"
RSA = f"ssh-rsa {blob('ssh-rsa', b'r' * 64)} root@box"
ECDSA = f"ecdsa-sha2-nistp256 {blob('ecdsa-sha2-nistp256', b'e' * 40)}"
BANNER = "Welcome to vast.ai. If authentication fails, try again after a few seconds\nHave fun!"

ADDRESS = ("203.0.113.7", 41022)
#: The machine as ``wait_for_ssh`` builds it: through the proxy, its own address beside.
PROXY = Machine(host="ssh9.vast.ai", port=12345, identity=Path("/keys/account"), direct=ADDRESS)
RAW: dict[str, Any] = {
    "id": 77,
    "actual_status": "running",
    "ssh_host": "ssh9.vast.ai",
    "ssh_port": 12345,
    "public_ipaddr": "203.0.113.7",
    "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]},
}


def make_offer(offer_id: int = 1, *, ports: int = 0, location: str = "France, FR") -> Offer:
    return Offer(
        id=offer_id,
        gpu_name="GTX 1080",
        num_gpus=1,
        dph_total=0.05 + offer_id / 1000.0,
        gpu_ram_gb=8.0,
        cpu_cores=8.0,
        cpu_ghz=3.5,
        cpu_name="Core i7",
        ram_gb=32.0,
        disk_gb=60.0,
        cuda_max=12.6,
        reliability=0.99,
        inet_down_mbps=900.0,
        location=location,
        machine_id=100 + offer_id,
        direct_ports=ports,
    )


class FakeSsh:
    """Stands in for ssh: what the machine prints, and every argv it was run with."""

    def __init__(self, printed: str = ED, *, direct_fails: str | None = None) -> None:
        self.printed = printed
        self.direct_fails = direct_fails
        self.through: list[Any] = []
        self.probed: list[list[str]] = []

    def run(self, machine: Any, command: str, *, what: str, timeout: float | None = None) -> str:
        self.through.append((machine, command))
        return self.printed

    def probe(self, argv: list[str], **_: Any) -> str:
        self.probed.append(argv)
        if self.direct_fails is not None:
            raise RemoteError(f"direct ssh probe failed: {self.direct_fails}", 255)
        return ""


def upgraded(ssh: FakeSsh, tmp_path: Path, said: list[str], **more: Any) -> Any:
    machine = more.pop("machine", PROXY)
    return direct.upgrade(
        machine, 77, directory=tmp_path, say=said.append, run=ssh.run, probe=ssh.probe, **more
    )


def options_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, word in enumerate(argv) if word == "-o"]


class TestHostKeys:
    def test_a_banner_is_not_a_key_and_the_strongest_kind_comes_first(self) -> None:
        keys = hostkeys.parse(f"{BANNER}\n{RSA}\n{ECDSA}\n{ED}\n{ED}\n")
        assert [key.kind for key in keys] == ["ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa"]

    def test_a_blob_that_names_another_kind_is_dropped(self) -> None:
        assert hostkeys.parse(f"ssh-ed25519 {blob('ssh-rsa')}") == []
        assert hostkeys.parse("ssh-ed25519 not/base64!") == []
        assert hostkeys.parse(f"ssh-dss {blob('ssh-dss')}") == []

    def test_the_fingerprint_has_the_shape_ssh_prints(self) -> None:
        (key,) = hostkeys.parse(ED)
        assert key.fingerprint.startswith("SHA256:")
        assert len(key.fingerprint) == len("SHA256:") + 43 and "=" not in key.fingerprint

    def test_an_address_is_named_as_known_hosts_names_it(self) -> None:
        assert hostkeys.host_pattern("203.0.113.7", 22) == "203.0.113.7"
        assert hostkeys.host_pattern("203.0.113.7", 41022) == "[203.0.113.7]:41022"
        for bad in ("", "a b", "*", "host,other"):
            with pytest.raises(ValueError):
                hostkeys.host_pattern(bad, 22)

    def test_the_file_pins_one_address_and_is_the_users_alone(self, tmp_path: Path) -> None:
        keys = hostkeys.parse(f"{ED}\n{RSA}")
        path = hostkeys.pin(keys, "203.0.113.7", 41022, tmp_path / "deep" / "instance_77")
        lines = path.read_text().splitlines()
        assert lines == [f"[203.0.113.7]:41022 {key.kind} {key.blob}" for key in keys]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(path.parent.iterdir()) == [path]

    def test_a_second_pin_replaces_the_first(self, tmp_path: Path) -> None:
        path = tmp_path / "instance_77"
        hostkeys.pin(hostkeys.parse(RSA), "203.0.113.7", 41022, path)
        hostkeys.pin(hostkeys.parse(ED), "203.0.113.9", 500, path)
        assert path.read_text().splitlines() == [f"[203.0.113.9]:500 {ED.rsplit(' ', 1)[0]}"]

    def test_no_key_is_never_written_as_a_file_that_trusts_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            hostkeys.pin([], "203.0.113.7", 41022, tmp_path / "instance_77")
        assert not (tmp_path / "instance_77").exists()

    def test_the_keys_are_asked_of_the_machine_given(self) -> None:
        ssh = FakeSsh(f"{BANNER}\n{ED}\n")
        (key,) = hostkeys.read_through(PROXY, run=ssh.run)
        assert key.kind == "ssh-ed25519"
        assert ssh.through == [(PROXY, hostkeys.READ_COMMAND)]

    def test_each_instance_has_its_own_file_and_it_can_be_forgotten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(settings.DATA_ROOT_ENV, str(tmp_path))
        one, two = hostkeys.pinned_path(1), hostkeys.pinned_path(2)
        assert one != two and one.parent == tmp_path / "runs" / "known_hosts"
        hostkeys.pin(hostkeys.parse(ED), "203.0.113.7", 22, one)
        hostkeys.forget(1)
        hostkeys.forget(1)
        assert not one.exists()


class TestUpgrade:
    def test_the_address_is_kept_with_the_keys_read_through_the_proxy(self, tmp_path: Path) -> None:
        ssh, said = FakeSsh(f"{BANNER}\n{ED}\n"), list[str]()
        machine = upgraded(ssh, tmp_path, said)
        # Still the proxy's machine, for every command.
        assert isinstance(machine, direct.PinnedMachine)
        assert (machine.host, machine.port, machine.direct) == ("ssh9.vast.ai", 12345, ADDRESS)
        assert machine.identity == PROXY.identity and direct.is_pinned(machine)
        # The keys were read on the machine through the proxy, never from the address.
        assert ssh.through == [(PROXY, hostkeys.READ_COMMAND)]
        assert machine.known_hosts == tmp_path / "instance_77"
        assert machine.known_hosts.read_text().startswith("[203.0.113.7]:41022 ssh-ed25519 ")
        # Its own address is a machine that checks them, and the probe went there.
        there = machine.directly()
        assert isinstance(there, direct.DirectMachine)
        assert (there.host, there.port, there.direct) == ("203.0.113.7", 41022, None)
        assert there.known_hosts == machine.known_hosts and there.directly() is None
        (probe,) = ssh.probed
        assert probe == there.ssh_command("true")
        assert f"UserKnownHostsFile={tmp_path / 'instance_77'}" in options_of(probe)
        assert "pinned ssh-ed25519 SHA256:" in said[-1]

    def test_a_machine_without_an_address_comes_back_as_it_is(self, tmp_path: Path) -> None:
        plain = Machine(host="ssh9.vast.ai", port=12345)
        ssh, said = FakeSsh(), list[str]()
        assert upgraded(ssh, tmp_path, said, machine=plain) is plain
        assert ssh.through == [] and ssh.probed == [] and said == []
        assert upgraded(ssh, tmp_path, said, machine="machine-1003") == "machine-1003"

    def test_keys_that_cannot_be_read_cost_the_address_and_not_the_machine(
        self, tmp_path: Path
    ) -> None:
        def unreadable(*_: Any, **__: Any) -> str:
            raise RemoteError("host keys failed: Connection reset", 255)

        said = list[str]()
        machine = direct.upgrade(PROXY, 77, directory=tmp_path, say=said.append, run=unreadable)
        assert type(machine) is Machine and (machine.host, machine.port) == (PROXY.host, 12345)
        # Nothing later may connect to an address whose keys were never checked.
        assert machine.direct is None and machine.directly() is None
        assert "host keys not read" in said[-1]

    def test_a_file_that_cannot_be_written_costs_the_address_and_not_the_run(
        self, tmp_path: Path
    ) -> None:
        in_the_way = tmp_path / "a file where the directory should be"
        in_the_way.write_text("")
        ssh, said = FakeSsh(ED), list[str]()
        machine = direct.upgrade(
            PROXY, 77, directory=in_the_way, say=said.append, run=ssh.run, probe=ssh.probe
        )
        assert machine.directly() is None and "could not be pinned" in said[-1]
        assert ssh.probed == []

    def test_a_machine_that_shows_no_key_is_not_pinned(self, tmp_path: Path) -> None:
        ssh, said = FakeSsh(BANNER), list[str]()
        machine = upgraded(ssh, tmp_path, said)
        assert machine.directly() is None and not direct.is_pinned(machine)
        assert "no host key" in said[-1]
        assert ssh.probed == [] and list(tmp_path.iterdir()) == []

    def test_a_key_that_does_not_match_is_said_and_the_address_dropped(
        self, tmp_path: Path
    ) -> None:
        words = (
            "@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @\nHost key verification failed."
        )
        ssh, said = FakeSsh(direct_fails=words), list[str]()
        machine = upgraded(ssh, tmp_path, said)
        assert machine.directly() is None and type(machine) is Machine
        assert "DIRECT ROUTE REFUSED" in said[-1]
        assert list(tmp_path.iterdir()) == []

    def test_an_address_that_does_not_answer_is_dropped(self, tmp_path: Path) -> None:
        ssh, said = FakeSsh(direct_fails="Connection timed out"), list[str]()
        machine = upgraded(ssh, tmp_path, said)
        assert machine.directly() is None
        assert "did not answer" in said[-1] and "REFUSED" not in said[-1]

    def test_a_mismatch_is_the_commands_failure_and_is_never_tried_again(self) -> None:
        """``_run`` retries a lost connection; a refused key must not be one."""
        from reverberate.wave.remote import connection_level

        words = (
            "@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @\nHost key verification failed."
        )
        assert not connection_level(["ssh"], 255, words)


class TestOptions:
    @pytest.fixture
    def pinned(self, tmp_path: Path) -> direct.PinnedMachine:
        return direct.PinnedMachine(
            host="ssh9.vast.ai",
            port=12345,
            identity=Path("/keys/account"),
            direct=ADDRESS,
            known_hosts=tmp_path / "instance_77",
        )

    @pytest.fixture
    def machine(self, pinned: direct.PinnedMachine) -> Machine:
        there = pinned.directly()
        assert there is not None
        return there

    def test_the_pinned_file_is_the_only_word_on_the_host(self, machine: Machine) -> None:
        options = options_of(machine.ssh_command("true"))
        assert "StrictHostKeyChecking=yes" in options
        assert f"UserKnownHostsFile={machine.known_hosts}" in options  # type: ignore[attr-defined]
        assert "GlobalKnownHostsFile=/dev/null" in options
        assert "UpdateHostKeys=no" in options
        assert not any("accept-new" in option for option in options)

    def test_nothing_is_lent_and_only_the_named_key_is_offered_on_either_route(
        self, machine: Machine, pinned: direct.PinnedMachine
    ) -> None:
        for route in (machine, pinned):
            for argv in (
                route.ssh_command("true"),
                route.scp_command([Path("a")], "/root/a", download=False),
                direct.rsync_shell(route).split(),
                route.sharing("fetch.0").ssh_command("true"),
            ):
                options = options_of(argv)
                for word in (
                    "ForwardAgent=no",
                    "ForwardX11=no",
                    "ClearAllForwardings=yes",
                    "IdentitiesOnly=yes",
                    "PasswordAuthentication=no",
                    "KbdInteractiveAuthentication=no",
                    "PreferredAuthentications=publickey",
                    "BatchMode=yes",
                ):
                    assert word in options
                assert not {"-A", "-R", "-L", "-D", "-X", "-Y"} & set(argv)
                assert argv[argv.index("-i") + 1] == "/keys/account"

    def test_the_proxy_is_still_addressed_as_the_proxy(self, pinned: direct.PinnedMachine) -> None:
        argv = pinned.ssh_command("uname")
        assert argv[-2:] == ["root@ssh9.vast.ai", "uname"] and argv[argv.index("-p") + 1] == "12345"
        assert "StrictHostKeyChecking=accept-new" in options_of(argv)
        assert direct.ssh_options(pinned) == pinned._options()

    def test_the_commands_address_the_host_on_its_port(self, machine: Machine) -> None:
        ssh = machine.ssh_command("uname")
        assert ssh[0] == "ssh" and ssh[-2:] == ["root@203.0.113.7", "uname"]
        assert ssh[ssh.index("-p") + 1] == "41022"
        down = machine.scp_command([Path("/root/a b")], "here", download=True)
        assert down[0] == "scp" and down[down.index("-P") + 1] == "41022"
        assert down[-2:] == ["root@203.0.113.7:'/root/a b'", "here"]

    def test_a_route_without_pinned_keys_is_not_connected_to(self) -> None:
        with pytest.raises(ValueError):
            direct.DirectMachine(host="203.0.113.7", port=41022).ssh_command("true")
        unpinned = direct.PinnedMachine(host="ssh9.vast.ai", port=1, direct=ADDRESS)
        assert unpinned.directly() is None and not direct.is_pinned(unpinned)

    def test_a_kept_connection_stays_on_its_route_with_its_keys(self, machine: Machine) -> None:
        assert not any("Control" in option for option in options_of(machine.ssh_command("true")))
        kept = machine.sharing("fetch.0")
        assert isinstance(kept, direct.DirectMachine) and kept.known_hosts is not None
        options = options_of(kept.ssh_command("true"))
        assert "ControlMaster=auto" in options and "StrictHostKeyChecking=yes" in options
        assert any(option.startswith("ControlPath=") and "%C" in option for option in options)
        assert kept.close_command() is not None

    def test_a_plain_machine_is_addressed_as_before(self) -> None:
        assert direct.rsync_shell(PROXY) == (
            "ssh -p 12345 -o StrictHostKeyChecking=accept-new -i /keys/account"
        )
        assert direct.rsync_shell(Machine(host="h")) == (
            "ssh -p 22 -o StrictHostKeyChecking=accept-new"
        )
        assert not direct.is_pinned(PROXY)

    def test_a_path_with_a_space_is_refused_rather_than_split(self) -> None:
        spaced = direct.DirectMachine(
            host="203.0.113.7", port=41022, known_hosts=Path("/my runs/instance_77")
        )
        with pytest.raises(ValueError, match="splits"):
            direct.rsync_shell(spaced)

    def test_rsync_is_given_the_shell_of_the_route(
        self, machine: Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reverberate.wave import remote_voxelise

        ran: list[list[str]] = []

        def ran_instead(argv: list[str], **_: Any) -> str:
            ran.append(argv)
            return ""

        monkeypatch.setattr(remote_voxelise, "_run", ran_instead)
        remote_voxelise.rsync(machine, ["/root/out/"], "/tmp/home", download=True)
        remote_voxelise.rsync(PROXY, ["/root/out/"], "/tmp/home", download=True)
        direct_shell, proxy_shell = (argv[argv.index("-e") + 1] for argv in ran)
        assert "StrictHostKeyChecking=yes" in direct_shell and "-p 41022" in direct_shell
        assert "root@203.0.113.7:/root/out/" in ran[0]
        assert proxy_shell == "ssh -p 12345 -o StrictHostKeyChecking=accept-new -i /keys/account"

    def test_the_homecoming_takes_the_checked_route(
        self, pinned: direct.PinnedMachine, machine: Machine
    ) -> None:
        from reverberate.gpu.homecoming import fastest

        assert fastest(pinned, probe=lambda route: "") == machine


class FakeClient:
    """Stands in for VastClient: one instance, its log, and what creation was asked."""

    def __init__(self, url: str = "https://logs.example/77.log") -> None:
        self.url = url
        self.asked: list[tuple[str, str]] = []

    def instance(self, instance_id: int) -> Instance | None:
        return Instance.from_api(RAW)

    def destroy_and_verify(self, instance_id: int) -> bool:
        return True

    def request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        self.asked.append((method, path))
        return {"success": True, "result_url": self.url}


class TestRentals:
    """Every rented machine passes through the pinning; a destroyed one leaves nothing."""

    def test_the_machine_that_answers_comes_back_pinned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(settings.DATA_ROOT_ENV, str(tmp_path))
        ran: list[list[str]] = []

        def answers(argv: list[str], **_: Any) -> str:
            ran.append(argv)
            return ED

        monkeypatch.setattr("reverberate.wave.remote._run", answers)
        monkeypatch.setattr(direct, "_run", answers)
        client: Any = FakeClient()
        machine = vast.wait_for_ssh(client, 77, Path("/keys/account"))
        assert isinstance(machine, direct.PinnedMachine) and direct.is_pinned(machine)
        assert machine.known_hosts == hostkeys.pinned_path(77)
        # Asked of the proxy twice, the probe and the keys; then of the address, once.
        assert [argv[-2] for argv in ran] == ["root@ssh9.vast.ai"] * 2 + ["root@203.0.113.7"]
        plain = vast.wait_for_ssh(client, 77, Path("/keys/account"), pin=False)
        assert type(plain) is Machine and plain.direct == ADDRESS

    def test_a_destroyed_instance_takes_its_pinned_keys_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(settings.DATA_ROOT_ENV, str(tmp_path))
        path = hostkeys.pin(hostkeys.parse(ED), *ADDRESS, hostkeys.pinned_path(77))
        client: Any = FakeClient()
        assert vast.teardown(client, 77)
        assert not path.exists()

    def test_an_offer_reads_its_open_ports(self) -> None:
        raw = {"id": 1, "direct_port_count": 49, "geolocation": "Maine, US"}
        assert Offer.from_api(raw).direct_ports == 49
        assert Offer.from_api({"id": 1, "direct_port_count": None}).direct_ports == 0

    def test_a_port_of_the_instance_is_asked_as_a_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(settings.DATA_ROOT_ENV, str(tmp_path))
        monkeypatch.setattr(vast, "arm_hard_stop", lambda instance_id, deadline: 4242)
        sent: list[Any] = []
        client = vast.VastClient(api_key="placeholder")

        def request(method: str, path: str, body: Any = None, **_: Any) -> Any:
            sent.append(body)
            return {"success": True, "new_contract": 77}

        monkeypatch.setattr(client, "request", request)
        vast.rent(client, make_offer(1, ports=12), hours=0.5, image="img")
        vast.rent(client, make_offer(2, ports=12), hours=0.5, image="img", ports=(8443,))
        assert "env" not in sent[0]
        assert sent[1]["env"] == {"-p 8443:8443": "1"}
        assert sent[1]["runtype"] == vast.RUNTYPE_DIRECT


class TestConfirmedByApi:
    """The second channel: the instance's own log, from the API, shows the pinned keys."""

    keys = hostkeys.parse(f"{ED}\n{RSA}")

    def confirmed(self, client: FakeClient, log: str, ssh: FakeSsh | None = None) -> bool:
        return hostkeys.confirmed_by_api(
            client,
            77,
            PROXY,
            self.keys,
            run=(ssh or FakeSsh()).run,
            fetch=lambda _client, _url: log,
            tries=2,
            pause_s=0.0,
        )

    def test_every_fingerprint_in_the_log_confirms(self) -> None:
        client, ssh = FakeClient(), FakeSsh()
        log = "\n".join(f"256 {key.fingerprint} root@box ({key.kind})" for key in self.keys)
        assert self.confirmed(client, log, ssh)
        assert client.asked == [("PUT", "/instances/request_logs/77/")]
        assert ssh.through == [(PROXY, hostkeys.LOG_COMMAND)]

    def test_one_key_missing_from_the_log_does_not(self) -> None:
        assert not self.confirmed(FakeClient(), f"256 {self.keys[0].fingerprint} root@box")
        assert not self.confirmed(FakeClient(), "")

    def test_a_log_that_is_not_served_over_tls_is_not_read(self) -> None:
        log = " ".join(key.fingerprint for key in self.keys)
        assert not self.confirmed(FakeClient(url="http://logs.example/77.log"), log)
        assert not self.confirmed(FakeClient(url=""), log)

    def test_an_api_that_fails_is_unconfirmed_and_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = " ".join(key.fingerprint for key in self.keys)
        client = FakeClient()

        def refused(method: str, path: str, body: Any = None) -> dict[str, Any]:
            raise vast.VastError("PUT failed: HTTP 429", 429)

        monkeypatch.setattr(client, "request", refused)
        assert not self.confirmed(client, log)
        assert not hostkeys.confirmed_by_api(FakeClient(), 77, PROXY, [], run=FakeSsh().run)

    def test_upgrade_says_what_the_api_showed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (key,) = hostkeys.parse(ED)
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        for log, words in ((key.fingerprint, "the instance's own"), ("", "not confirmed")):
            said: list[str] = []
            monkeypatch.setattr(hostkeys, "_fetch", lambda _client, _url, log=log: log)
            machine = upgraded(FakeSsh(ED), tmp_path, said, confirm=True, client=FakeClient())
            # Unconfirmed is not refuted: the route stands, and what was found is said.
            assert direct.is_pinned(machine)
            assert words in said[-1]


class TestRank:
    def test_open_ports_first_then_the_preferred_regions_then_the_order_given(self) -> None:
        offers = [
            make_offer(1, ports=0, location="France, FR"),
            make_offer(2, ports=9, location="California, US"),
            make_offer(3, ports=9, location="Quebec, CA"),
            make_offer(4, ports=9, location="France, FR"),
            make_offer(5, ports=9, location="United Kingdom, GB"),
            make_offer(6, ports=9, location="Maine, US"),
        ]
        ranked = direct.rank(offers, ["FR", "gb", "quebec"])
        assert [offer.id for offer in ranked] == [4, 5, 3, 2, 6, 1]
        assert len(ranked) == len(offers)

    def test_without_a_preference_only_the_ports_reorder(self) -> None:
        offers = [make_offer(1, ports=0), make_offer(2, ports=4), make_offer(3, ports=0)]
        assert [offer.id for offer in direct.rank(offers)] == [2, 1, 3]

    def test_two_letters_are_a_country_and_never_a_piece_of_a_name(self) -> None:
        # "CA" is Canada, not California; "US" is not found in "Belarus, BY".
        assert direct.region_rank("California, US", ["CA"]) == 1
        assert direct.region_rank("Quebec, CA", ["CA"]) == 0
        assert direct.region_rank("Belarus, BY", ["US"]) == 1
        assert direct.region_rank("California, US", ["california"]) == 0
        assert direct.region_rank("unknown", ["FR", ""]) == 2

    def test_the_argument_is_a_list_however_it_is_written(self) -> None:
        parser = argparse.ArgumentParser()
        direct.add_arguments(parser)
        args = parser.parse_args(["--prefer-region", "FR, GB", "--prefer-region", "Quebec"])
        assert direct.regions(args.prefer_region) == ("FR", "GB", "Quebec")
        assert direct.regions(None) == () and direct.regions("FR") == ("FR",)
        assert parser.parse_args([]).prefer_region == []
