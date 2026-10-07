"""The fastest way home: the range server, the workers' count, the choice of a way.

What these guard is, first, what the server on a rented machine gives and to
whom: one folder, to the bearer of the run's token, under the certificate
the laptop read over ssh and no other. Then that a file fetched in ranges
ends as every other, verified and resumed; that the count of workers grows
only while growing pays; and that a way which fails hands over to the next.

The server runs here, on 127.0.0.1, with a certificate ``openssl`` makes for
the test, and its line is slowed or cut by its own flags. No machine, no
ssh, nothing beyond this host.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import ssl
import subprocess
import threading
import time
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from reverberate.gpu import direct, onebox, rangeserver, transfer, transfer_bench, vast
from reverberate.gpu.homecoming import BLOCK_BYTES, Pace, fetch_file
from reverberate.gpu.transfer import HTTPS, SSH_DIRECT, SSH_PROXY, Early, Lanes, Served, Way
from reverberate.wave.remote import ConnectionLost, Machine, RemoteError

CHUNK = BLOCK_BYTES
TOKEN = "t" * 43

needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="no openssl")


def _bytes(count: int, seed: int = 0) -> bytes:
    block = hashlib.sha256(str(seed).encode()).digest()
    return (block * (count // len(block) + 1))[:count]


def _pace(give_up_after: int = 30) -> Pace:
    """A pace that does not wait: its clock moves when it sleeps."""
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    return Pace(clock=lambda: now[0], sleep=sleep, give_up_after=give_up_after)


class Control:
    """The ssh side of a fetch, answered from the folder: sizes and digests, never bytes."""

    def __init__(self) -> None:
        self.closed = 0

    def size(self, remote: str) -> int:
        return Path(remote).stat().st_size

    def sha256(self, remote: str) -> str:
        return hashlib.sha256(Path(remote).read_bytes()).hexdigest()

    def chunk_digests(self, remote: str, chunk_bytes: int, chunks: Sequence[int]) -> list[str]:
        held = Path(remote).read_bytes()
        return [
            hashlib.sha256(held[c * chunk_bytes : (c + 1) * chunk_bytes]).hexdigest()
            for c in chunks
        ]

    def read(self, remote: str, offset: int, count: int, target: Path, worker: int) -> None:
        raise AssertionError("the bytes come in ranges, not over ssh")

    def listing(self, remote_dir: str, exclude: Collection[str]) -> list[tuple[str, int]]:
        return []

    def files(self, *given: Any) -> None:
        return

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A certificate and its key, as the machine makes them for a run."""
    where = tmp_path_factory.mktemp("tls")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2"]
        + ["-keyout", str(where / "key.pem"), "-out", str(where / "cert.pem")]
        + ["-subj", "/CN=rv-homecoming"],
        capture_output=True,
        check=True,
    )
    return where / "cert.pem", where / "key.pem"


class Running:
    """A range server on 127.0.0.1 for one test, and what a client knows of it."""

    def __init__(self, root: Path, certificate: tuple[Path, Path], **options: Any) -> None:
        self.server = rangeserver.Server(
            ("127.0.0.1", 0),
            root=str(root),
            token=TOKEN,
            certificate=str(certificate[0]),
            key=str(certificate[1]),
            quiet=True,
            **options,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()
        self.served = Served(
            "127.0.0.1", self.server.server_address[1], TOKEN, certificate[0].read_text(), str(root)
        )

    def ask(self, method: str, path: str, token: str | None = TOKEN, **headers: str) -> Any:
        connection = transfer._Pinned(self.served, 10.0)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        try:
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def machine_dir(tmp_path: Path) -> Path:
    """The run's output folder on the machine, with a secret beside it and not inside."""
    out = tmp_path / "campaign" / "out"
    (out / "pairs").mkdir(parents=True)
    (out / "pack.h5").write_bytes(_bytes(3 * CHUNK + 12345))
    (out / "pairs" / "a.npy").write_bytes(b"pair")
    (tmp_path / "campaign" / "secret").write_text("not of the run")
    (out / "way_out").symlink_to(tmp_path / "campaign" / "secret")
    return out


@pytest.fixture
def running(machine_dir: Path, certificate: tuple[Path, Path]) -> Iterator[Running]:
    server = Running(machine_dir, certificate)
    yield server
    server.close()


class TestWhatIsServed:
    def test_a_path_is_served_only_where_it_is_a_file_under_the_folder(
        self, machine_dir: Path
    ) -> None:
        root = os.path.realpath(machine_dir)
        assert rangeserver.resolve(root, "/pack.h5") == os.path.join(root, "pack.h5")
        assert rangeserver.resolve(root, "/pairs/a.npy") is not None
        # A folder is not listed; a path that climbs out is nothing, however it is written.
        assert rangeserver.resolve(root, "/") is None
        assert rangeserver.resolve(root, "/pairs") is None
        assert rangeserver.resolve(root, "/../secret") is None
        assert rangeserver.resolve(root, "/%2e%2e/secret") is None
        assert rangeserver.resolve(root, "/pairs/../../secret") is None
        assert rangeserver.resolve(root, "//etc/hosts") is None
        assert rangeserver.resolve(root, "/pack.h5%00") is None
        # A link in the folder that points out of it is followed, and refused there.
        assert rangeserver.resolve(root, "/way_out") is None

    def test_one_range_is_read_and_anything_else_is_refused(self) -> None:
        assert rangeserver.byte_range(None, 100) == (0, 100)
        assert rangeserver.byte_range("bytes=10-19", 100) == (10, 10)
        assert rangeserver.byte_range("bytes=90-", 100) == (90, 10)
        assert rangeserver.byte_range("bytes=90-500", 100) == (90, 10)
        for refused in ("bytes=100-", "bytes=-5", "bytes=5-2", "bytes=0-1,5-6", "lines=0-1"):
            assert rangeserver.byte_range(refused, 100) is None

    def test_a_token_is_long(self, machine_dir: Path, certificate: tuple[Path, Path]) -> None:
        with pytest.raises(ValueError, match="32 characters"):
            rangeserver.Server(
                ("127.0.0.1", 0),
                root=str(machine_dir),
                token="short",
                certificate=str(certificate[0]),
                key=str(certificate[1]),
            )

    @needs_openssl
    def test_without_the_token_nothing_is_said_and_with_it_the_folder_alone(
        self, running: Running, machine_dir: Path
    ) -> None:
        held = (machine_dir / "pack.h5").read_bytes()
        # No token, a wrong one, one that only starts right: 401, before the path is read.
        for token in (None, "x" * 43, TOKEN[:-1]):
            for path in ("/pack.h5", "/nothing", rangeserver.PING):
                status, body, _ = running.ask("GET", path, token)
                assert (status, body) == (401, b"")
        assert running.ask("GET", rangeserver.PING)[0] == 204
        status, body, headers = running.ask("GET", "/pack.h5", Range="bytes=4194304-4194403")
        assert status == 206 and body == held[4194304:4194404]
        assert headers["Content-Range"] == f"bytes 4194304-4194403/{len(held)}"
        assert running.ask("HEAD", "/pack.h5")[2]["Content-Length"] == str(len(held))
        assert running.ask("GET", "/pairs/a.npy")[1] == b"pair"
        # Out of the folder, a folder, a link out of it, a range past the end.
        for path in ("/../secret", "/%2e%2e/secret", "/pairs", "/", "/way_out"):
            assert running.ask("GET", path)[0] == 404
        assert running.ask("GET", "/pack.h5", Range=f"bytes={len(held)}-")[0] == 416
        # Reading is all there is.
        for method in ("PUT", "POST", "DELETE"):
            assert running.ask(method, "/pack.h5")[0] == 501

    @needs_openssl
    def test_another_certificate_is_refused_and_plain_http_is_not_spoken(
        self, running: Running, tmp_path: Path
    ) -> None:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2"]
            + ["-keyout", str(tmp_path / "k.pem"), "-out", str(tmp_path / "c.pem")]
            + ["-subj", "/CN=rv-homecoming"],
            capture_output=True,
            check=True,
        )
        other = Served(
            "127.0.0.1", running.served.port, TOKEN, (tmp_path / "c.pem").read_text(), "/"
        )
        with pytest.raises(ssl.SSLCertVerificationError):
            transfer._Pinned(other, 5.0).connect()
        # The transport says it as the command's failure, which is never tried again.
        with pytest.raises(RemoteError, match="certificate is not the machine's"):
            transfer.HttpsTransport(other, Control()).read("/pack.h5", 0, 10, tmp_path / "x", 0)
        with pytest.raises(RemoteError, match="certificate is not the machine's"):
            transfer._ping(other)
        plain = http.client.HTTPConnection("127.0.0.1", running.served.port, timeout=5)
        with pytest.raises((OSError, http.client.HTTPException)):
            plain.request("GET", "/pack.h5", headers={"Authorization": f"Bearer {TOKEN}"})
            plain.getresponse()
        # The system's authorities do not vouch for it either.
        with pytest.raises(ssl.SSLError):
            ssl.create_default_context().wrap_socket(
                __import__("socket").create_connection(("127.0.0.1", running.served.port)),
                server_hostname="127.0.0.1",
            )


@needs_openssl
class TestFetchInRanges:
    def test_a_pack_comes_home_in_ranges_and_its_digest_is_the_machines(
        self, running: Running, machine_dir: Path, tmp_path: Path
    ) -> None:
        control = Control()
        lanes = Lanes(2, 4)
        record = fetch_file(
            None,
            str(machine_dir / "pack.h5"),
            tmp_path / "home" / "pack.h5",
            chunk_bytes=CHUNK,
            transport=transfer.HttpsTransport(running.served, control),
            lanes=lanes,
            pace=_pace(),
        )
        held = (machine_dir / "pack.h5").read_bytes()
        assert (tmp_path / "home" / "pack.h5").read_bytes() == held
        assert record["sha256"] == hashlib.sha256(held).hexdigest()
        assert record["chunks"] == 4 and record["failures"] == 0 and record["workers"] <= 4
        assert control.closed == 1
        # The same file asked for again brings nothing, whatever was written into it since.
        (tmp_path / "home" / "pack.h5").write_bytes(b"stamped")
        again = fetch_file(
            None,
            str(machine_dir / "pack.h5"),
            tmp_path / "home" / "pack.h5",
            chunk_bytes=CHUNK,
            transport=transfer.HttpsTransport(running.served, Control()),
            pace=_pace(),
        )
        assert again["already_home"] is True and again["sha256"] == record["sha256"]
        # A file of the machine that changed is brought.
        (machine_dir / "pack.h5").write_bytes(_bytes(2 * CHUNK, seed=1))
        new = fetch_file(
            None,
            str(machine_dir / "pack.h5"),
            tmp_path / "home" / "pack.h5",
            chunk_bytes=CHUNK,
            transport=transfer.HttpsTransport(running.served, Control()),
            pace=_pace(),
        )
        assert "already_home" not in new
        assert (tmp_path / "home" / "pack.h5").read_bytes() == _bytes(2 * CHUNK, seed=1)

    def test_connections_that_are_cut_cost_their_chunks_and_fewer_workers(
        self, machine_dir: Path, certificate: tuple[Path, Path], tmp_path: Path
    ) -> None:
        server = Running(machine_dir, certificate, drop_after=CHUNK // 2, drops=3)
        try:
            lanes = Lanes(4, 4)
            record = fetch_file(
                None,
                str(machine_dir / "pack.h5"),
                tmp_path / "home" / "pack.h5",
                chunk_bytes=CHUNK,
                transport=transfer.HttpsTransport(server.served, Control()),
                lanes=lanes,
                pace=_pace(),
            )
        finally:
            server.close()
        assert record["failures"] == 3 and lanes.allowed == 3
        assert (tmp_path / "home" / "pack.h5").read_bytes() == (
            machine_dir / "pack.h5"
        ).read_bytes()

    def test_a_fetch_that_gives_up_is_resumed_from_its_chunks(
        self, machine_dir: Path, certificate: tuple[Path, Path], tmp_path: Path
    ) -> None:
        target = tmp_path / "home" / "pack.h5"
        # Every body longer than a quarter of a chunk is cut: only the short last chunk comes.
        server = Running(machine_dir, certificate, drop_after=CHUNK // 4, drops=10**6)
        try:
            with pytest.raises(ConnectionLost):
                fetch_file(
                    None,
                    str(machine_dir / "pack.h5"),
                    target,
                    chunk_bytes=CHUNK,
                    transport=transfer.HttpsTransport(server.served, Control()),
                    pace=_pace(give_up_after=4),
                )
        finally:
            server.close()
        assert not target.exists() and target.with_name("pack.h5.partial").exists()
        said: list[str] = []
        server = Running(machine_dir, certificate)
        try:
            record = fetch_file(
                None,
                str(machine_dir / "pack.h5"),
                target,
                chunk_bytes=CHUNK,
                transport=transfer.HttpsTransport(server.served, Control()),
                pace=_pace(),
                say=said.append,
            )
        finally:
            server.close()
        assert record["chunks_resumed"] == 1 and "1 chunks already home" in said[0]
        assert target.read_bytes() == (machine_dir / "pack.h5").read_bytes()

    def test_a_file_out_of_the_folder_is_not_asked_for(self, running: Running) -> None:
        transport = transfer.HttpsTransport(running.served, Control())
        with pytest.raises(RemoteError, match="not under the folder"):
            transport.read("/root/.ssh/id_ed25519", 0, 10, Path("unused"), 0)
        with pytest.raises(RemoteError, match="answered 404"):
            transport.read(running.served.root + "/absent", 0, 10, Path(os.devnull), 0)


class TestLanes:
    @staticmethod
    def _lanes(start: int, most: int) -> tuple[Lanes, list[float]]:
        now = [0.0]
        return Lanes(start, most, window_s=4.0, clock=lambda: now[0]), now

    @staticmethod
    def _window(lanes: Lanes, now: list[float], megabytes_per_s: float) -> None:
        """Four seconds at so many megabytes a second, a chunk a worker."""
        chunks = lanes.allowed
        now[0] += 4.0
        for _ in range(chunks):
            lanes.brought(int(megabytes_per_s * 4.0 * 1e6 / chunks))

    def test_workers_are_added_while_they_bring_more_and_no_longer(self) -> None:
        lanes, now = self._lanes(4, 32)
        assert [lanes.may(w) for w in (0, 3, 4)] == [True, True, False]
        self._window(lanes, now, 20.0)
        assert lanes.allowed == 8
        self._window(lanes, now, 39.0)
        assert lanes.allowed == 16
        # Sixteen bring hardly more than eight: a line, not a window. They are kept, and
        # the climb is over.
        self._window(lanes, now, 42.0)
        assert lanes.allowed == 16 and lanes.settled
        self._window(lanes, now, 90.0)
        assert lanes.allowed == 16
        assert [count for count, _ in lanes.history] == [4, 8, 16]

    def test_a_count_that_brings_less_is_gone_back_from(self) -> None:
        lanes, now = self._lanes(8, 32)
        self._window(lanes, now, 50.0)
        self._window(lanes, now, 31.0)
        assert lanes.allowed == 8 and lanes.settled

    def test_a_window_is_not_judged_before_a_chunk_a_worker(self) -> None:
        lanes, now = self._lanes(4, 8)
        now[0] += 60.0
        for _ in range(3):
            lanes.brought(10**6)
        assert lanes.allowed == 4 and not lanes.history
        lanes.brought(10**6)
        # At the most it may use, and judged once more: eight may bring less than four did.
        assert lanes.allowed == 8 and not lanes.settled
        now[0] += 4.0
        for _ in range(8):
            lanes.brought(10**3)
        assert lanes.allowed == 4 and lanes.settled

    def test_a_refused_connection_takes_workers_back_and_ends_the_climb(self) -> None:
        lanes, now = self._lanes(8, 32)
        # Three streams cut in the same second are one event.
        for _ in range(3):
            lanes.failed()
        assert lanes.allowed == 6 and lanes.settled
        self._window(lanes, now, 500.0)
        assert lanes.allowed == 6
        for _ in range(20):
            now[0] += 5.0
            lanes.failed()
        assert lanes.allowed == 1 and lanes.may(0)
        # A way's workers never go under the four a fetch always had.
        kept = Way(SSH_DIRECT, "m", 4, 8).options()["lanes"]
        kept.failed()
        assert kept.allowed == 4 and kept.settled
        assert Way(HTTPS, "m", 16, 32).options()["lanes"].least == 4


PINNED = direct.DirectMachine(
    host="203.0.113.7", port=41022, known_hosts=Path("/runs/known_hosts/instance_77")
)
CERTIFICATE = "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"


class TestServe:
    def test_a_server_is_started_over_pinned_keys_or_not_at_all(self) -> None:
        for machine in (Machine(host="ssh9.vast.ai", port=1), "a name", None):
            with pytest.raises(RemoteError, match="pinned host keys"):
                transfer.serve(machine, "/root/campaign/out", ask=lambda *a: pytest.fail("asked"))
        unpinned = direct.DirectMachine(host="203.0.113.7", port=41022)
        with pytest.raises(RemoteError, match="pinned host keys"):
            transfer.serve(unpinned, "/root/campaign/out", ask=lambda *a: pytest.fail("asked"))

    def test_the_token_goes_down_stdin_and_the_port_is_the_hosts(self) -> None:
        asked: list[tuple[str, bytes]] = []

        def ask(machine: Any, command: str, data: bytes) -> str:
            asked.append((command, data))
            return f"Have fun!\nmapped 50000\n{CERTIFICATE}"

        served = transfer.serve(PINNED, "/root/campaign/out", ask=ask, ping=False)
        command, data = asked[0]
        token, _, source = data.partition(b"\n")
        assert served.token == token.decode() and len(served.token) >= 43
        # The command line, which the machine's every process can read, does not hold it.
        assert served.token not in command and "--forget" in command
        assert source == Path(rangeserver.__file__).read_bytes()
        assert "--root /root/campaign/out --port 8443" in command and "umask 077" in command
        assert (served.host, served.port, served.root) == (
            "203.0.113.7",
            50000,
            "/root/campaign/out",
        )
        assert served.certificate == CERTIFICATE and served.token not in repr(served)
        # The record of the rental, where it names the port, is believed before the machine.
        mapped = direct.DirectMachine(
            host="203.0.113.7", port=41022, known_hosts=PINNED.known_hosts, mapped=((8443, 50007),)
        )
        assert transfer.serve(mapped, "/out", ask=ask, ping=False).port == 50007
        assert transfer.serve(PINNED, "/out", ask=ask, ping=False).token != served.token

    def test_a_host_that_maps_no_port_is_not_this_way_and_nothing_is_left_listening(self) -> None:
        ended: list[Any] = []
        with pytest.raises(RemoteError, match="maps no port"):
            transfer.serve(
                PINNED,
                "/out",
                ask=lambda *a: f"mapped \n{CERTIFICATE}",
                ending=lambda machine, where: ended.append(where),
            )
        with pytest.raises(RemoteError, match="no certificate"):
            transfer.serve(
                PINNED,
                "/out",
                ask=lambda *a: "mapped 50000\n",
                ending=lambda machine, where: ended.append(where),
            )
        assert ended == [transfer.REMOTE_SERVE, transfer.REMOTE_SERVE]

    @needs_openssl
    @pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
    def test_the_script_itself_starts_a_server_that_answers_and_forgets_its_secrets(
        self, machine_dir: Path, tmp_path: Path
    ) -> None:
        def ask(machine: Any, command: str, data: bytes) -> str:
            return subprocess.run(
                ["bash", "-c", command], input=data, capture_output=True, check=True
            ).stdout.decode()

        def end(machine: Any, where: str) -> None:
            pid = Path(where) / "pid"
            if pid.is_file():
                os.kill(int(pid.read_text()), 15)

        with __import__("socket").socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        local = direct.DirectMachine(host="127.0.0.1", port=22, known_hosts=tmp_path / "pinned")
        where = tmp_path / "serve"
        served = transfer.serve(
            local,
            str(machine_dir),
            host_port=port,
            port=port,
            where=str(where),
            ask=ask,
            ending=end,
        )
        try:
            # ``serve`` asked once with the token and once without: 204, then 401.
            assert not (where / "key.pem").exists() and not (where / "token").exists()
            assert oct(where.stat().st_mode & 0o777) == "0o700"
            target = tmp_path / "range"
            transfer.HttpsTransport(served, Control()).read(
                str(machine_dir / "pack.h5"), CHUNK, 1000, target, 0
            )
            assert (
                target.read_bytes() == (machine_dir / "pack.h5").read_bytes()[CHUNK : CHUNK + 1000]
            )
        finally:
            end(local, str(where))


class TestWays:
    @staticmethod
    def _serving(machine: Any, root: str) -> Served:
        return Served(machine.host, 50000, TOKEN, CERTIFICATE, root)

    def test_ranges_first_then_the_machines_address_then_the_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stopped: list[Any] = []
        monkeypatch.setattr(transfer, "stop", lambda machine, **k: stopped.append(machine))
        said: list[str] = []
        found, end = transfer.ways(
            "proxy", "/out", direct=PINNED, say=said.append, verdict={}, serving=self._serving
        )
        assert [(w.name, w.through, w.start, w.most, w.direct) for w in found] == [
            (HTTPS, PINNED, 16, 32, True),
            (SSH_DIRECT, PINNED, 4, 8, True),
            (SSH_PROXY, "proxy", 4, 4, False),
        ]
        assert isinstance(found[0].options()["transport"], transfer.HttpsTransport)
        assert (
            isinstance(found[1].options()["lanes"], Lanes) and "transport" not in found[1].options()
        )
        # The proxy is fetched from as it always was.
        assert found[2].options() == {"workers": 4}
        assert any("certificate" in line for line in said) and TOKEN not in " ".join(said)
        end()
        assert stopped == [PINNED]

    def test_a_host_without_the_port_or_without_an_address_keeps_ssh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transfer, "stop", lambda machine, **k: pytest.fail("nothing started"))

        def closed(machine: Any, root: str) -> Served:
            raise RemoteError("the host maps no port to the instance's 8443")

        said: list[str] = []
        found, end = transfer.ways(
            "proxy", "/out", direct=PINNED, say=said.append, verdict={}, serving=closed
        )
        assert [w.name for w in found] == [SSH_DIRECT, SSH_PROXY]
        assert any("no range server" in line for line in said)
        end()
        found, _ = transfer.ways("proxy", "/out", direct="proxy", verdict={}, serving=closed)
        assert [w.name for w in found] == [SSH_PROXY]

    def test_what_the_bench_measured_on_the_host_orders_the_ways(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transfer, "stop", lambda machine, **k: None)
        pinned = direct.DirectMachine(
            host="203.0.113.7", port=41022, known_hosts=tmp_path / "instance_77"
        )
        assert transfer.verdict_of(pinned) == {}
        path = transfer.verdict_path(pinned)
        assert path == tmp_path / "instance_77.transfer.json"
        path.write_text(
            '{"order": ["direct ssh", "https"], "streams": {"direct ssh": 6, "https": 16},'
            ' "cipher": "aes128-gcm@openssh.com"}'
        )
        found, _ = transfer.ways("proxy", "/out", direct=pinned, serving=self._serving)
        assert [(w.name, w.start, w.most) for w in found] == [
            (SSH_DIRECT, 6, 8),
            (HTTPS, 16, 32),
            (SSH_PROXY, 4, 4),
        ]
        assert found[0].options()["transport"].cipher == "aes128-gcm@openssh.com"
        # A file that is not a verdict is no verdict.
        path.write_text("[1, 2")
        assert transfer.verdict_of(pinned) == {}

    def test_a_cipher_is_put_before_sshs_own_options(self) -> None:
        from reverberate.gpu.homecoming import SshTransport

        ran: list[list[str]] = []

        def fake_run(argv: list[str], **given: Any) -> Any:
            ran.append(argv)
            return subprocess.CompletedProcess(argv, 0)

        transport = SshTransport(PINNED, share=False, cipher="aes128-gcm@openssh.com")
        original = subprocess.run
        try:
            subprocess.run = fake_run  # type: ignore[assignment]
            transport.read("/out/pack.h5", 0, CHUNK, Path(os.devnull), 0)
        finally:
            subprocess.run = original
        assert ran[0][:3] == ["ssh", "-c", "aes128-gcm@openssh.com"]

    def test_minutes_home(self) -> None:
        assert transfer.minutes_home(9.5, 20.0) == pytest.approx(7.92, abs=0.01)
        assert transfer.minutes_home(9.5, 50.0) == pytest.approx(3.17, abs=0.01)
        assert transfer.minutes_home(9.5, 100.0, overhead_s=30.0) == pytest.approx(2.08, abs=0.01)


class TestEarly:
    def test_the_pack_is_brought_when_it_is_written_and_the_end_is_said_once(self) -> None:
        there = [""]
        brought: list[str] = []
        said: list[str] = []
        early = Early(
            "m",
            "/root/campaign/out",
            brought.append,
            say=said.append,
            ask=lambda machine, command, **k: there[0],
        )
        early.look()
        assert not brought and not early.ended.is_set()
        there[0] = "pack.h5\n"
        early.look()
        early.look()
        assert brought == ["pack.h5"] and "pack.h5" in early.home
        assert not early.ended.is_set() and "brought now" in said[0]
        there[0] = "pack.h5\ncampaign.done\n"
        early.look()
        assert early.ended.is_set()
        # Whoever waited looks at once, and is not woken again for the same end.
        early.wait(30.0)
        early.look()
        assert not early.ended.is_set() and brought == ["pack.h5"]
        # A campaign that is launched again and ends again is said again.
        there[0] = "pack.h5\n"
        early.look()
        there[0] = "pack.h5\ncampaign.failed\n"
        early.look()
        assert early.ended.is_set()

    def test_a_pack_that_does_not_come_is_tried_three_times_and_left_to_the_fetch(self) -> None:
        tries: list[str] = []

        def bring(item: str) -> None:
            tries.append(item)
            raise RemoteError("pack.h5 changed on the machine while it was fetched")

        said: list[str] = []
        early = Early("m", "/out", bring, say=said.append, ask=lambda *a, **k: "pack.h5\n")
        for _ in range(6):
            early.look()
        assert len(tries) == 3 and not early.home
        assert sum("not home early" in line for line in said) == 3

    def test_the_looks_start_after_a_pause_and_end_when_told(self) -> None:
        looked: list[str] = []

        def ask(*a: Any, **k: Any) -> str:
            looked.append("x")
            return ""

        early = Early("m", "/out", lambda item: None, poll_s=30.0, ask=ask).start()
        early.finish()
        assert not looked

    def test_the_watch_looks_at_once_when_the_end_appears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        waited: list[float] = []
        looks = {"n": 0}

        class Seen:
            def wait(self, seconds: float) -> None:
                waited.append(seconds)

        def look(*a: Any, **k: Any) -> Any:
            looks["n"] += 1
            return SimpleNamespace(
                line=lambda: "look",
                instance_alive=True,
                done=looks["n"] == 2,
                failed=False,
                stalled=False,
            )

        monkeypatch.setattr(onebox, "monitor_once", look)
        monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("slept its whole pause"))
        outcome = onebox.watch(
            None,
            1,
            "m",
            lambda: None,
            deadline=1e12,
            poll_s=300.0,
            record={"watches": []},
            home=tmp_path,
            say=lambda m: None,
            early=Seen(),
        )
        assert outcome == "done" and waited == [300.0]

    def test_a_pack_brought_early_is_in_the_record_and_what_is_left_is_not_looked_for(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        brought: list[str] = []

        def bring_large(machine: Any, item: str, pulled: Path, say: Any, fetched: Any) -> None:
            brought.append(item)
            fetched[item] = {"by": HTTPS}

        monkeypatch.setattr(onebox, "bring_large", bring_large)
        monkeypatch.setattr(onebox, "run_on", lambda *a, **k: "pack.h5\n")
        record: dict[str, Any] = {}
        early = onebox.early_homecoming("m", tmp_path, lambda m: None, record, poll_s=3600.0)
        early.look()
        early.finish()
        assert brought == ["pack.h5"] and record["fetched"]["pack.h5"]["by"] == HTTPS
        assert "seconds" in record["early"]["pack.h5"] and (tmp_path / "pulled").is_dir()
        left = onebox.early_homecoming("m", tmp_path, lambda m: None, {}, leave=("pack.h5",))
        left.finish()
        assert left.items == ()


class TestBringLarge:
    @staticmethod
    def _ways(*names: str) -> list[Way]:
        return [Way(name, f"through {name}", 4, 4, direct=name != SSH_PROXY) for name in names]

    def test_a_way_that_fails_hands_over_to_the_next_and_the_record_says_which(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tried: list[Any] = []

        def fetch_file(machine: Any, remote: str, local: Path, **given: Any) -> dict[str, Any]:
            tried.append(machine)
            if machine == f"through {HTTPS}":
                raise ConnectionLost("range of pack.h5: timed out")
            return {"bytes": 5, "seconds": 1.0, "bytes_per_s": 5, "failures": 0, "workers": 4}

        monkeypatch.setattr(onebox, "fetch_file", fetch_file)
        said: list[str] = []
        fetched: dict[str, Any] = {}
        onebox.bring_large(
            "m",
            "pack.h5",
            tmp_path,
            said.append,
            fetched,
            ways=self._ways(HTTPS, SSH_DIRECT, SSH_PROXY),
        )
        assert tried == [f"through {HTTPS}", f"through {SSH_DIRECT}"]
        assert fetched["pack.h5"]["by"] == SSH_DIRECT and fetched["pack.h5"]["direct"] is True
        assert any("https failed" in line for line in said)

    def test_the_pair_cache_never_comes_in_ranges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        through: list[Any] = []

        def fetch_tree(
            machine: Any, remote_dir: str, local_dir: Path, **given: Any
        ) -> dict[str, Any]:
            through.append(machine)
            return {"files": 3, "bytes": 9, "left": 0, "complete": True, "seconds": 1.0}

        monkeypatch.setattr(onebox, "fetch_tree", fetch_tree)
        fetched: dict[str, Any] = {}
        onebox.bring_large(
            "m", "pairs", tmp_path, lambda m: None, fetched, ways=self._ways(HTTPS, SSH_DIRECT)
        )
        assert through == [f"through {SSH_DIRECT}"] and fetched["pairs"]["by"] == SSH_DIRECT


class TestTheRental:
    def test_the_hosts_port_for_each_port_of_the_instance_is_read(self) -> None:
        raw = {
            "id": 77,
            "ports": {
                "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}],
                "8443/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41000"}],
                "8443/udp": [{"HostPort": "41001"}],
                "9000/tcp": [],
                "x/tcp": [{"HostPort": "5"}],
            },
        }
        assert vast.mapped_ports(raw) == ((22, 41022), (8443, 41000))
        assert vast.mapped_ports({"ports": None}) == ()
        assert vast.Instance.from_api(raw).ports == ((22, 41022), (8443, 41000))

    def test_a_floor_on_what_a_host_sends_is_asked_only_when_given(self) -> None:
        assert "inet_up" not in vast.search_query()
        assert "inet_up>200" in vast.search_query(min_inet_up_mbps=200).split()
        assert onebox.MIN_INET_UP_MBPS == 200

    def test_the_pinned_machine_carries_the_ports_to_its_direct_route(self, tmp_path: Path) -> None:
        import base64

        name = b"ssh-ed25519"
        raw = len(name).to_bytes(4, "big") + name + (32).to_bytes(4, "big") + b"\x01" * 32
        proxy = Machine(host="ssh9.vast.ai", port=12345, direct=("203.0.113.7", 41022))
        pinned = direct.upgrade(
            proxy,
            77,
            directory=tmp_path,
            say=lambda m: None,
            run=lambda *a, **k: f"ssh-ed25519 {base64.b64encode(raw).decode()} root@box",
            probe=lambda *a, **k: "",
            mapped=((22, 41022), (8443, 41000)),
        )
        way = pinned.directly()
        assert way.mapped == ((22, 41022), (8443, 41000))
        assert transfer.verdict_path(way) == tmp_path / "instance_77.transfer.json"
        # The verdict goes with the instance, as its keys do.
        (tmp_path / "instance_77.transfer.json").write_text("{}")
        from reverberate.gpu import hostkeys

        hostkeys.forget(77, tmp_path)
        assert not list(tmp_path.iterdir())


class TestBench:
    @staticmethod
    def _row(way: str, streams: int, late: float, note: str = "", **more: Any) -> Any:
        return transfer_bench.Row(way, streams, note, late_megabytes_per_s=late, **more)

    def test_the_verdict_is_the_fastest_way_at_its_best_count(self) -> None:
        row = self._row
        found = transfer_bench.verdict(
            [
                row(SSH_PROXY, 4, 12.0),
                row(SSH_DIRECT, 1, 9.0),
                row(SSH_DIRECT, 4, 41.0),
                row(SSH_DIRECT, 8, 44.0),
                row(SSH_DIRECT, 4, 43.0, "aes128-gcm@openssh.com"),
                row(SSH_DIRECT, 4, 30.0, "one shared connection"),
                row("rsync", 1, 15.0, "direct"),
                row(HTTPS, 8, 52.0),
                row(HTTPS, 16, 61.0),
                row(HTTPS, 32, 60.0),
                row(SSH_DIRECT, 4, 900.0, direction="up"),
            ]
        )
        assert found["order"] == [HTTPS, SSH_DIRECT, SSH_PROXY]
        assert found["streams"] == {HTTPS: 16, SSH_DIRECT: 8, SSH_PROXY: 4}
        # A cipher within a tenth of ssh's own choice is not worth naming.
        assert found["cipher"] is None and found["megabytes_per_s"][HTTPS] == 61.0

    def test_a_way_must_be_worth_its_moving_parts(self) -> None:
        row = self._row
        found = transfer_bench.verdict(
            [
                row(SSH_PROXY, 4, 12.0),
                row(SSH_DIRECT, 4, 40.0),
                row(SSH_DIRECT, 4, 55.0, "aes128-gcm@openssh.com"),
                row(SSH_DIRECT, 4, 47.0, "chacha20-poly1305@openssh.com"),
                row(HTTPS, 8, 42.0),
                row(HTTPS, 16, 0.0, error="status 401"),
            ]
        )
        # Ranges that bring one twentieth more than ssh do not go before it.
        assert found["order"] == [SSH_DIRECT, HTTPS, SSH_PROXY]
        assert found["cipher"] == "aes128-gcm@openssh.com"
        assert transfer_bench.verdict([row(HTTPS, 0, 0.0, error="no port")])["order"] == []

    def test_streams_are_measured_together_and_the_late_half_apart(self) -> None:
        class Steady(transfer_bench.Stream):
            def run(self, meter: Any, stop: threading.Event) -> None:
                while not stop.wait(0.01):
                    meter.add(1000)

        class Silent(transfer_bench.Stream):
            said = "Connection refused"

            def run(self, meter: Any, stop: threading.Event) -> None:
                stop.wait(5.0)

        mean, late, first, error = transfer_bench.measure([Steady(), Steady()], 0.3)
        assert mean > 0 and late > 0 and first is not None and error == ""
        mean, late, first, error = transfer_bench.measure([Silent()], 0.05)
        assert (mean, late, first, error) == (0.0, 0.0, None, "Connection refused")
        now = [0.0]
        meter = transfer_bench.Meter(clock=lambda: now[0])
        for at, count in ((1.0, 10), (4.0, 10), (6.0, 30), (9.0, 30)):
            now[0] = at
            meter.add(count)
        assert meter.rates(10.0) == (8.0, 12.0)

    def test_the_table_says_each_row_and_which_way_it_went(self) -> None:
        text = transfer_bench.table(
            [
                self._row(HTTPS, 16, 61.0, megabytes_per_s=55.5, first_byte_s=0.2),
                self._row(SSH_DIRECT, 4, 9.0, direction="up"),
                self._row(HTTPS, 0, 0.0, error="the host maps no port"),
            ]
        )
        lines = text.splitlines()
        assert lines[0].split() == ["way", "streams", "MB/s", "late", "first", "byte"]
        assert "61.0" in lines[1] and "55.5" in lines[1] and "0.2 s" in lines[1]
        assert "the host maps no port" in lines[2] and "(to the machine)" in lines[3]
