"""W47: what a rented machine's routes carry, measured from the place the result comes home to.

``docs/open-questions/direct-connection.md`` is what this found. One rented
machine is reached three ways and each is timed in both directions:

- Vast's ssh proxy, as every run has used it;
- ssh at the host's own address, host keys pinned
  (:mod:`reverberate.gpu.direct`), with separate connections and with one
  shared connection (``ControlMaster``);
- an HTTPS server on the instance, on a port the host maps, behind a token
  made for the one run and a certificate made on the machine and read back
  through ssh, fetched as parallel byte ranges.

What is timed is the channel and not a tool: bytes that do not compress,
read from the machine's ``/dev/urandom`` into a file once and streamed out of
``ssh`` into nothing, or sent into ``cat > /dev/null``. A megabyte is 1e6
bytes. Every measurement is capped in bytes and in seconds.

``line`` times the laptop's own line against public files in each region.
``offers`` lists what could be rented, read only. ``measure`` **rents**: one
machine, the watchdog armed by :func:`reverberate.gpu.vast.rent`, destroyed
and verified destroyed whatever happens.

    python -m reverberate.experiments.w47_direct_connection offers --prefer-region FR,GB
    python -m reverberate.experiments.w47_direct_connection line
    python -m reverberate.experiments.w47_direct_connection measure --machine 149401 --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reverberate.gpu import direct, hostkeys, vast
from reverberate.wave.remote import RemoteError, run_on

#: A small image: the card is not used, and a large image is minutes of rental.
IMAGE = "ubuntu:22.04"
#: The ceilings of one rental of this experiment.
MAX_DPH = 0.25
HOURS = 0.67
DISK_GB = 20
#: The port of the instance the HTTPS server listens on.
HTTPS_PORT = 8443
#: Where everything of the experiment lives on the machine.
REMOTE = "/root/w47"
#: The file streamed, in MiB; every measurement reads a part of it.
BLOB_MIB = 128
MB = 1e6
#: What one measurement may move and how long it may take.
DOWN_ONE, DOWN_FOUR = 100 * MB, 128 * MB
UP_ONE, UP_FOUR = 50 * MB, 80 * MB
CAP_S = 40.0

#: Public files to time the laptop's own line against, by region.
REFERENCE_FILES = {
    "France (OVH)": "https://proof.ovh.net/files/100Mb.dat",
    "Germany (Hetzner)": "https://fsn1-speed.hetzner.com/100MB.bin",
    "US east (Hetzner Ashburn)": "https://ash-speed.hetzner.com/100MB.bin",
    "US west (Hetzner Hillsboro)": "https://hil-speed.hetzner.com/100MB.bin",
}


@dataclass
class Rate:
    """One measurement: a route, a direction, a count of streams, and what it carried."""

    route: str
    direction: str
    streams: int
    megabytes: float
    seconds: float
    mb_per_s: float
    #: Seconds from the first command to the first byte, the handshake in it.
    first_byte_s: float
    #: The rate over the second half of the time, the start left out.
    late_mb_per_s: float
    complete: bool
    note: str = ""


class _Clock:
    """Bytes against time, across the streams of one measurement."""

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.lock = threading.Lock()
        self.total = 0
        self.first: float | None = None
        self.samples: list[tuple[float, int]] = []

    def add(self, count: int) -> None:
        now = time.monotonic() - self.start
        with self.lock:
            if self.first is None:
                self.first = now
            self.total += count
            self.samples.append((now, self.total))

    def late_rate(self, end: float) -> float:
        half = end / 2.0
        before = max((total for at, total in self.samples if at <= half), default=0)
        return (self.total - before) / MB / max(end - half, 1e-6)


def _down_stream(argv: list[str], clock: _Clock, deadline: float, done: list[bool]) -> None:
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    try:
        while time.monotonic() < deadline:
            chunk = process.stdout.read1(1 << 20)  # type: ignore[attr-defined]
            if not chunk:
                done.append(process.wait(timeout=20) == 0)
                return
            clock.add(len(chunk))
        done.append(False)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _up_stream(
    argv: list[str], share: int, clock: _Clock, deadline: float, done: list[bool]
) -> None:
    process = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert process.stdin is not None
    block = os.urandom(1 << 18)
    sent = 0
    try:
        while sent < share and time.monotonic() < deadline:
            process.stdin.write(block)
            sent += len(block)
            clock.add(len(block))
        process.stdin.close()
        # What ssh still holds is on its way: the stream ends when ssh does.
        done.append(process.wait(timeout=max(5.0, deadline - time.monotonic() + 20)) == 0)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        done.append(False)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def ssh_rate(
    machine: Any, route: str, direction: str, streams: int, megabytes: float, cap_s: float = CAP_S
) -> Rate:
    """Time ``streams`` ssh sessions carrying ``megabytes`` between them, one way."""
    share_mib = max(1, int(megabytes * MB / streams) >> 20)
    clock = _Clock()
    deadline = time.monotonic() + cap_s
    done: list[bool] = []
    threads = []
    for index in range(streams):
        if direction == "down":
            remote = (
                f"dd if={REMOTE}/www/blob bs=1M skip={index * share_mib} count={share_mib}"
                " status=none"
            )
            target: Callable[..., None] = _down_stream
            arguments: tuple[Any, ...] = (machine.ssh_command(remote), clock, deadline, done)
        else:
            target = _up_stream
            arguments = (
                machine.ssh_command("cat > /dev/null"),
                share_mib << 20,
                clock,
                deadline,
                done,
            )
        threads.append(threading.Thread(target=target, args=arguments, daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seconds = time.monotonic() - clock.start
    return Rate(
        route=route,
        direction=direction,
        streams=streams,
        megabytes=round(clock.total / MB, 1),
        seconds=round(seconds, 2),
        mb_per_s=round(clock.total / MB / max(seconds, 1e-6), 2),
        first_byte_s=round(clock.first if clock.first is not None else float("nan"), 2),
        late_mb_per_s=round(clock.late_rate(seconds), 2),
        complete=len(done) == streams and all(done),
    )


def https_rate(
    url: str, certificate: Path, token: str, direction: str, streams: int, megabytes: float
) -> Rate:
    """Time ``streams`` curl transfers of byte ranges, or of files put, over pinned TLS."""
    share = int(megabytes * MB / streams)
    common = [
        "curl",
        "-sS",
        "--cacert",
        str(certificate),
        "-H",
        f"Authorization: Bearer {token}",
        "--max-time",
        str(int(CAP_S)),
        "-o",
        "/dev/null",
        "-w",
        "%{time_starttransfer} %{size_download} %{size_upload} %{http_code}",
    ]
    started = time.monotonic()
    processes = []
    with tempfile.TemporaryDirectory() as scratch:
        payload = Path(scratch) / "payload"
        if direction == "up":
            payload.write_bytes(os.urandom(share))
        for index in range(streams):
            if direction == "down":
                span = f"{index * share}-{(index + 1) * share - 1}"
                argv = [*common, "-r", span, f"{url}/blob"]
            else:
                argv = [*common, "-T", str(payload), f"{url}/up/{index}.bin"]
            processes.append(
                subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            )
        answers = [process.communicate() for process in processes]
    seconds = time.monotonic() - started
    moved, first, codes = 0.0, [], []
    for out, _ in answers:
        words = out.split()
        if len(words) == 4:
            first.append(float(words[0]))
            moved += float(words[1]) + float(words[2])
            codes.append(words[3])
    good = {"206", "200"} if direction == "down" else {"201", "204"}
    return Rate(
        route="https",
        direction=direction,
        streams=streams,
        megabytes=round(moved / MB, 1),
        seconds=round(seconds, 2),
        mb_per_s=round(moved / MB / max(seconds, 1e-6), 2),
        first_byte_s=round(min(first), 2) if first else float("nan"),
        late_mb_per_s=float("nan"),
        complete=len(codes) == streams and all(code in good for code in codes),
        note=",".join(sorted(set(codes))),
    )


def handshake_s(machine: Any, repeats: int = 3) -> list[float]:
    """Seconds for ``ssh true``, a few times: what every poll of a run pays."""
    times = []
    for _ in range(repeats):
        started = time.monotonic()
        subprocess.run(machine.ssh_command("true"), capture_output=True, timeout=60, check=False)
        times.append(round(time.monotonic() - started, 2))
    return times


# --------------------------------------------------------------------------
# the machine's side
# --------------------------------------------------------------------------

_NGINX = """user root;
worker_processes 2;
pid {remote}/nginx.pid;
error_log {remote}/nginx.log;
events {{ worker_connections 64; }}
http {{
  access_log off;
  sendfile on;
  client_max_body_size 0;
  client_body_temp_path {remote}/tmp;
  server {{
    listen {port} ssl;
    ssl_certificate {remote}/cert.pem;
    ssl_certificate_key {remote}/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    root {remote}/www;
    if ($http_authorization != "Bearer {token}") {{ return 403; }}
    location / {{ dav_methods PUT; create_full_put_path on; }}
  }}
}}
"""


def prepare(machine: Any, say: Callable[[str], None]) -> None:
    """The file to stream, and the tools: rsync, and nginx for the HTTPS route."""
    started = time.monotonic()
    run_on(
        machine,
        f"mkdir -p {REMOTE}/www {REMOTE}/tmp && "
        f"head -c {BLOB_MIB}M /dev/urandom > {REMOTE}/www/blob && "
        "export DEBIAN_FRONTEND=noninteractive && "
        "(apt-get update -qq && apt-get install -y -qq rsync nginx-core openssl) "
        f"> {REMOTE}/apt.log 2>&1; command -v rsync nginx openssl | wc -l",
        what="prepare",
        timeout=420,
    )
    say(f"  machine prepared in {time.monotonic() - started:.0f} s")


def serve_https(machine: Any, address: str, token: str, out: Path) -> Path:
    """Start the HTTPS server on the machine; the certificate it made, read back through ssh.

    The key is made on the machine and stays there. The certificate names
    the host's address and nothing else, lives two days, and is the only
    one ``curl`` is told to believe (``--cacert``): a server that shows
    another is refused.
    """
    conf = _NGINX.format(remote=REMOTE, port=HTTPS_PORT, token=token)
    run_on(
        machine,
        f"cd {REMOTE} && openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1"
        " -nodes -days 2 -keyout key.pem -out cert.pem -subj /CN=w47"
        f" -addext subjectAltName=IP:{address} 2>/dev/null && chmod 600 key.pem && "
        f"cat > nginx.conf <<'W47_EOF'\n{conf}W47_EOF\n"
        f"nginx -c {REMOTE}/nginx.conf && sleep 1 && test -s nginx.pid",
        what="https server",
        timeout=120,
    )
    certificate = out / "https_cert.pem"
    certificate.write_text(run_on(machine, f"cat {REMOTE}/cert.pem", what="certificate"))
    return certificate


def host_ports(client: Any, instance_id: int) -> dict[str, int]:
    """The host's port for each port of the instance, from the API's record."""
    raw = client.request("GET", f"/instances/{instance_id}/").get("instances") or {}
    mapped = {}
    for inside, outside in (raw.get("ports") or {}).items():
        with contextlib.suppress(LookupError, TypeError, ValueError):
            mapped[str(inside)] = int(outside[0]["HostPort"])
    return mapped


def security_checks(proxy: Any, machine: Any, out: Path) -> dict[str, Any]:
    """What the machine and the routes show about who is believed and what is lent."""
    found: dict[str, Any] = {}
    sshd = run_on(
        proxy,
        "sshd -T 2>/dev/null | grep -Ei "
        "'^(passwordauthentication|kbdinteractiveauthentication|permitrootlogin"
        "|pubkeyauthentication|allowagentforwarding|maxstartups|maxsessions) '",
        what="sshd settings",
        timeout=60,
    )
    found["sshd"] = dict(
        line.split(None, 1) for line in sshd.splitlines() if len(line.split(None, 1)) == 2
    )
    held = {key.fingerprint for key in hostkeys.read_through(proxy)}
    found["machine_keys"] = sorted(held)

    def shown(host: str, port: int) -> list[str]:
        scan = subprocess.run(
            ["ssh-keyscan", "-T", "15", "-p", str(port), host],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        keys = hostkeys.parse(
            "\n".join(line.split(None, 1)[-1] for line in scan.stdout.splitlines() if line)
        )
        return sorted(key.fingerprint for key in keys)

    found["proxy_shows"] = shown(proxy.host, proxy.port)
    found["proxy_shows_the_machine_keys"] = bool(set(found["proxy_shows"]) & held)
    if machine is not None:
        found["direct_shows"] = shown(machine.host, machine.port)
        found["direct_shows_the_machine_keys"] = set(found["direct_shows"]) <= held and bool(
            found["direct_shows"]
        )
        found["agent_on_machine"] = run_on(
            machine, 'echo "sock=[${SSH_AUTH_SOCK:-}]"', what="agent", timeout=60
        ).strip()
        # A pinned file that names another key: the connection must be refused.
        wrong = out / "known_hosts_wrong"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(out / "decoy")],
            check=True,
            capture_output=True,
        )
        decoy = hostkeys.parse((out / "decoy.pub").read_text())
        hostkeys.pin(decoy, machine.host, machine.port, wrong)
        (out / "decoy").unlink()
        impostor = direct.DirectMachine(
            host=machine.host,
            port=machine.port,
            identity=machine.identity,
            known_hosts=wrong,
        )
        refused = subprocess.run(
            impostor.ssh_command("true"), capture_output=True, text=True, timeout=60, check=False
        )
        found["wrong_key_refused"] = refused.returncode == 255 and (
            "Host key verification failed" in refused.stderr
        )
        found["wrong_key_words"] = refused.stderr.strip().splitlines()[-1:]
    return found


# --------------------------------------------------------------------------
# one machine, rented, measured, destroyed
# --------------------------------------------------------------------------


def find_offer(client: Any, machine_id: int) -> Any:
    """The cheapest offer of one host, under the experiment's ceiling."""
    offers = client.search(f"machine_id={machine_id} rentable=true disk_space>{DISK_GB}", limit=20)
    good = [offer for offer in offers if offer.dph_total <= MAX_DPH]
    if not good:
        raise SystemExit(f"host {machine_id} has no offer under {MAX_DPH} USD/h now")
    return good[0]


def measure_machine(
    proxy: Any, machine: Any, port: int, out: Path, say: Callable[[str], None]
) -> dict[str, Any]:
    """Every route of one machine, both ways, the route a run would use twice.

    ``proxy`` is the machine as it answered, ``machine`` its direct route
    with its keys pinned or ``None``, ``port`` the host's port for the
    HTTPS server or 0.
    """
    record: dict[str, Any] = {"rates": []}
    rates: list[Rate] = []

    def keep(rate: Rate) -> None:
        rates.append(rate)
        record["rates"] = [asdict(r) for r in rates]
        (out / "result.json").write_text(json.dumps(record, indent=2))
        say(
            f"  {rate.route:9s} {rate.direction:4s} x{rate.streams}  {rate.mb_per_s:6.2f} MB/s"
            f"  (late {rate.late_mb_per_s:6.2f})  {rate.megabytes:6.1f} MB in {rate.seconds:5.1f} s"
            f"  first byte {rate.first_byte_s:.2f} s{'' if rate.complete else '  CUT'}"
            f"{' ' + rate.note if rate.note else ''}"
        )
        time.sleep(2)

    prepare(proxy, say)
    record["security"] = security_checks(proxy, machine, out)
    say(f"  security: {json.dumps(record['security'])}")
    record["handshake_s"] = {"proxy": handshake_s(proxy)}
    routes: list[tuple[str, Any]] = [("proxy", proxy)]
    shared = None
    if machine is not None:
        routes.append(("direct", machine))
        record["handshake_s"]["direct"] = handshake_s(machine)
        shared = machine.sharing(f"{os.getpid()}.w47")
        record["handshake_s"]["direct_shared"] = handshake_s(shared, repeats=4)
    say(f"  handshake, s: {record['handshake_s']}")

    # The route a run would use is measured twice: the direct one where there
    # is one, the proxy where there is not. The bytes go to it, the other
    # route is given less.
    main = routes[-1][0]
    for turn in (1, 2):
        for direction, sizes in (("down", (DOWN_ONE, DOWN_FOUR)), ("up", (UP_ONE, UP_FOUR))):
            for streams, megabytes in zip((1, 4), sizes, strict=True):
                for name, route in routes:
                    if name != main and turn == 2:
                        continue
                    part = 1.0 if name == main else 0.6
                    keep(ssh_rate(route, name, direction, streams, part * megabytes / MB))
            if turn == 1 and shared is not None:
                keep(ssh_rate(shared, "shared", direction, 4, 0.6 * sizes[1] / MB))

    if machine is not None and port:
        token = secrets.token_urlsafe(24)
        try:
            certificate = serve_https(machine, machine.host, token, out)
            url = f"https://{machine.host}:{port}"
            unsigned = subprocess.run(
                ["curl", "-sS", "--cacert", str(certificate), "-o", "/dev/null"]
                + ["-w", "%{http_code}", "--max-time", "20", f"{url}/blob"],
                capture_output=True,
                text=True,
                check=False,
            )
            record["https_without_token"] = unsigned.stdout.strip()
            unpinned = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "--max-time", "20", f"{url}/blob"],
                capture_output=True,
                text=True,
                check=False,
            )
            record["https_without_the_certificate_refused"] = unpinned.returncode != 0
            say(
                f"  https: no token -> {record['https_without_token']}, system store alone"
                f" refused={record['https_without_the_certificate_refused']}"
            )
            keep(https_rate(url, certificate, token, "down", 1, DOWN_ONE / MB))
            keep(https_rate(url, certificate, token, "down", 4, DOWN_FOUR / MB))
            keep(https_rate(url, certificate, token, "up", 4, UP_FOUR / MB))
        except (RemoteError, OSError) as error:
            record["https_error"] = str(error)[-300:]
            say(f"  https route not measured: {record['https_error']}")
        finally:
            with_stop = f"nginx -c {REMOTE}/nginx.conf -s stop; rm -f {REMOTE}/key.pem"
            subprocess.run(
                proxy.ssh_command(with_stop), capture_output=True, timeout=60, check=False
            )

    # The repository's own transfer, on each route: that rsync takes the shell it is given.
    from reverberate.wave.remote_voxelise import rsync

    run_on(proxy, f"head -c 20M {REMOTE}/www/blob > {REMOTE}/part", what="part", timeout=60)
    record["rsync_s"] = {}
    for name, route in routes:
        target = out / f"rsync_{name}"
        target.mkdir(exist_ok=True)
        started = time.monotonic()
        rsync(route, [f"{REMOTE}/part"], str(target) + "/", download=True, compress=False)
        record["rsync_s"][name] = round(time.monotonic() - started, 2)
        size = (target / "part").stat().st_size
        (target / "part").unlink()
        say(f"  rsync of {size / MB:.0f} MB down, {name}: {record['rsync_s'][name]} s")
    if shared is not None:
        subprocess.run(shared.close_command(), capture_output=True, timeout=30, check=False)
    (out / "result.json").write_text(json.dumps(record, indent=2))
    return record


def _terminated(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def measure(machine_id: int, out: Path, say: Callable[[str], None] = print) -> int:
    """Rent the host's cheapest offer, measure, destroy, verify. The spend is in the ledger."""
    from reverberate import auth

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    identity = vast.account_identity(client)
    offer = find_offer(client, machine_id)
    say(f"renting {offer.describe()}, open ports {offer.direct_ports}")
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _terminated)
    rental = vast.rent(
        client, offer, hours=HOURS, image=IMAGE, disk_gb=DISK_GB, ports=(HTTPS_PORT,)
    )
    say(f"rented {rental.instance_id}, watchdog {rental.watchdog_pid}, {HOURS} h at most")
    summary: dict[str, Any] = {
        "instance": rental.instance_id,
        "offer": offer.id,
        "machine_id": machine_id,
        "location": offer.location,
        "dph_total": offer.dph_total,
        "open_ports": offer.direct_ports,
    }
    try:
        started = time.monotonic()
        proxy = vast.wait_for_ssh(client, rental.instance_id, identity, timeout=600.0)
        summary["answered_after_s"] = round(time.monotonic() - started)
        say(f"  answered through {proxy.host}:{proxy.port} after {summary['answered_after_s']} s")
        # ``wait_for_ssh`` pinned the host keys; the direct route is the machine's to give.
        machine = proxy.directly() if direct.is_pinned(proxy) else None
        mapped = host_ports(client, rental.instance_id)
        summary["proxy"] = f"{proxy.host}:{proxy.port}"
        summary["direct"] = f"{machine.host}:{machine.port}" if machine is not None else ""
        summary["host_ports"] = mapped
        summary.update(
            measure_machine(proxy, machine, mapped.get(f"{HTTPS_PORT}/tcp", 0), out, say)
        )
        summary["keys_confirmed_by_api"] = hostkeys.confirmed_by_api(
            client, rental.instance_id, proxy, hostkeys.read_through(proxy)
        )
        say(f"  host keys in the instance's log, by the API: {summary['keys_confirmed_by_api']}")
    except BaseException as error:  # noqa: BLE001 - the machine is destroyed whatever ended this
        summary["ended_by"] = f"{type(error).__name__}: {str(error)[-300:]}"
        say(f"ENDED BY {summary['ended_by']}")
        with contextlib.suppress(Exception):
            # What Vast says of a host that did not answer, before it is gone.
            found = client.instance(rental.instance_id)
            summary["status_at_the_end"] = f"{found.status}: {found.status_msg}" if found else ""
            say(f"  the instance said: {summary['status_at_the_end']}")
    finally:
        gone = vast.teardown(client, rental.instance_id)
        absent = client.instance(rental.instance_id) is None
        entries = [
            entry
            for entry in vast.read_ledger()
            if entry.get("instance_id") == rental.instance_id and entry.get("event") == "teardown"
        ]
        summary["destroyed"] = gone and absent
        summary["billed_minutes"] = round(60 * entries[-1]["hours"], 1) if entries else None
        summary["cost_usd"] = entries[-1]["cost_usd"] if entries else None
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        say(
            f"instance {rental.instance_id} destroyed={gone} absent={absent}"
            f" billed {summary['billed_minutes']} min, {summary['cost_usd']} USD"
        )
    return 0 if summary["destroyed"] and "ended_by" not in summary else 1


def line(say: Callable[[str], None] = print) -> int:
    """The laptop's own line to a public file in each region, one stream, 20 s at most."""
    for name, url in REFERENCE_FILES.items():
        answer = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "--max-time", "20"]
            + ["-w", "%{time_starttransfer} %{size_download} %{time_total}", url],
            capture_output=True,
            text=True,
            check=False,
        )
        words = answer.stdout.split()
        if len(words) == 3 and float(words[2]) > 0:
            say(
                f"{name}: {float(words[1]) / MB / float(words[2]):.1f} MB/s,"
                f" {float(words[1]) / MB:.0f} MB, first byte {float(words[0]):.2f} s"
            )
        else:
            say(f"{name}: no answer ({answer.stderr.strip()[-80:]})")
    return 0


def offers(prefer: Sequence[str], say: Callable[[str], None] = print) -> int:
    """What could be rented under the ceiling, in the order :func:`direct.rank` gives. Read only."""
    from reverberate import auth

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    found = client.search(
        f"rentable=true dph_total<={MAX_DPH} reliability2>0.97 inet_down>200 inet_up>200"
        f" disk_space>{DISK_GB}",
        limit=300,
    )
    hosts: dict[int, Any] = {}
    for offer in found:
        hosts.setdefault(offer.machine_id, offer)
    without = [offer for offer in hosts.values() if offer.direct_ports <= 0]
    say(f"{len(hosts)} hosts under {MAX_DPH} USD/h, {len(without)} without an open port")
    for offer in direct.rank(hosts.values(), prefer)[:25]:
        say(f"  ports {offer.direct_ports:3d}  {offer.describe()}")
    for offer in without:
        say(f"  no open port: {offer.describe()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("offers", help="what could be rented, read only")
    direct.add_arguments(listing)
    commands.add_parser("line", help="the laptop's own line to each region")
    measured = commands.add_parser("measure", help="RENTS one machine and measures its routes")
    measured.add_argument("--machine", type=int, required=True, help="the host's machine id")
    measured.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "offers":
        return offers(direct.regions(args.prefer_region))
    if args.command == "line":
        return line()
    print(shlex.join(sys.argv), flush=True)
    return measure(args.machine, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
