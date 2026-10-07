"""What each way of bringing files home carries on one rented machine, in two minutes.

    python -m reverberate.gpu.transfer_bench --instance 12345678

Run at the start of a rental, on the instance the driver just rented (it
says its number), beside the driver and without disturbing it. A file of
random bytes is written on the machine, and each way reads it for a few
seconds: the proxy, ssh at the machine's own address on one, four and
eight connections, with each cipher, on one shared connection, ``rsync``,
and ranges over HTTPS on one to thirty-two connections; then the way up.
The table says megabytes a second over the whole of each measurement and
over its second half ("late"), which is nearer what gigabytes see: the
first half holds the handshakes and TCP's slow start.

**The verdict is kept beside the instance's pinned keys**
(:func:`reverberate.gpu.transfer.verdict_path`), and the driver's fetch
reads it: the ways in the order of what they brought here, each starting
on the count of streams it was the fastest on, ssh with the cipher that
was the fastest. Without a verdict the driver takes HTTPS, then direct
ssh, then the proxy. The file goes with the instance.

What limits the rate is read off the table. One stream far under four:
the path's window, not the line. A rate that stops growing with the
streams: a line, the machine's or the laptop's, and
``python -m reverberate.experiments.w47_direct_connection line`` says the
laptop's own. Ciphers that differ: a processor. HTTPS over ssh at the same
count: ``sshd``'s window.

It costs two minutes of the instance and moves some hundreds of megabytes.
Nothing here rents or destroys.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reverberate.gpu import transfer
from reverberate.gpu.homecoming import fastest

__all__ = ["Meter", "Row", "bench", "main", "measure", "table", "verdict"]

MB = 1e6
#: Where the bench's file is on the machine, and the folder a range server gives of it.
REMOTE = "/root/.rv-bench"
REMOTE_FILE = f"{REMOTE}/out/bench.bin"
#: The ciphers ssh is measured with beside its own choice: the two with authentication
#: built in, of which one leans on the processor's AES and the other does not, and the
#: plain counter mode.
CIPHERS = ("aes128-gcm@openssh.com", "chacha20-poly1305@openssh.com", "aes128-ctr")
#: How much faster a way must be to be put before one with fewer moving parts.
WORTH = 1.10


class Meter:
    """Bytes against time, across the streams of one measurement."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.start = clock()
        self._lock = threading.Lock()
        self.total = 0
        self.first: float | None = None
        self.samples: list[tuple[float, int]] = []

    def add(self, count: int) -> None:
        now = self._clock() - self.start
        with self._lock:
            if self.first is None:
                self.first = now
            self.total += int(count)
            self.samples.append((now, self.total))

    def rates(self, seconds: float) -> tuple[float, float]:
        """Bytes a second over ``seconds``, and over their second half."""
        half = seconds / 2.0
        with self._lock:
            before = max((total for at, total in self.samples if at <= half), default=0)
            return self.total / max(seconds, 1e-9), (self.total - before) / max(half, 1e-9)


class Stream:
    """One stream of a measurement: ``run`` moves bytes until told to stop; ``kill`` ends it."""

    def run(self, meter: Meter, stop: threading.Event) -> None:
        raise NotImplementedError

    def kill(self) -> None:
        return


class Command(Stream):
    """A command whose output is the stream (``up`` false) or which is fed one (``up`` true)."""

    def __init__(self, argv: list[str], *, up: bool = False) -> None:
        self.argv, self.up = argv, up
        self._process: subprocess.Popen[bytes] | None = None
        self.said = ""

    def run(self, meter: Meter, stop: threading.Event) -> None:
        with tempfile.TemporaryFile() as said:
            self._process = process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE if self.up else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL if self.up else subprocess.PIPE,
                stderr=said,
            )
            try:
                if self.up:
                    assert process.stdin is not None
                    block = bytes(1 << 18)
                    with contextlib.suppress(OSError):
                        while not stop.is_set():
                            process.stdin.write(block)
                            meter.add(len(block))
                else:
                    assert process.stdout is not None
                    while not stop.is_set():
                        block = process.stdout.read1(1 << 20)  # type: ignore[attr-defined]
                        if not block:
                            break
                        meter.add(len(block))
            finally:
                self.kill()
                process.wait()
                said.seek(0)
                self.said = said.read().decode(errors="replace").strip()[-200:]

    def kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            with contextlib.suppress(OSError):
                self._process.kill()


class Ranges(Stream):
    """One HTTPS connection reading a file again and again."""

    def __init__(self, served: transfer.Served, remote: str) -> None:
        self.served, self.remote = served, remote
        self._connection: http.client.HTTPSConnection | None = None
        self.said = ""

    def run(self, meter: Meter, stop: threading.Event) -> None:
        path = self.served.path(self.remote)
        self._connection = connection = transfer._Pinned(self.served, 30.0)
        buffer = memoryview(bytearray(1 << 20))
        try:
            while not stop.is_set():
                connection.request(
                    "GET", path, headers={"Authorization": f"Bearer {self.served.token}"}
                )
                response = connection.getresponse()
                if response.status != 200:
                    self.said = f"status {response.status}"
                    return
                while not stop.is_set():
                    got = response.readinto(buffer)
                    if not got:
                        break
                    meter.add(got)
        except (OSError, http.client.HTTPException, ValueError, AttributeError) as error:
            if not stop.is_set():
                self.said = f"{type(error).__name__}: {error}"[:200]
        finally:
            self.kill()

    def kill(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(Exception):
                self._connection.close()


class Growing(Stream):
    """A command that fills a folder (``rsync``): what the folder holds is the stream."""

    def __init__(self, argv: list[str], folder: Path) -> None:
        self.command, self.folder = Command(argv), folder
        self.said = ""

    def run(self, meter: Meter, stop: threading.Event) -> None:
        worker = threading.Thread(target=self.command.run, args=(Meter(), stop), daemon=True)
        worker.start()
        held = 0
        while worker.is_alive() and not stop.is_set():
            time.sleep(0.2)
            now = sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())
            if now > held:
                meter.add(now - held)
                held = now
        self.kill()
        worker.join(timeout=10)
        self.said = self.command.said

    def kill(self) -> None:
        self.command.kill()


@dataclass
class Row:
    """One measurement: a way, a count of streams, a direction; what it carried."""

    way: str
    streams: int
    note: str = ""
    direction: str = "home"
    megabytes_per_s: float = 0.0
    late_megabytes_per_s: float = 0.0
    first_byte_s: float | None = None
    error: str = ""


def measure(streams: Sequence[Stream], seconds: float) -> tuple[float, float, float | None, str]:
    """Run ``streams`` together for ``seconds``: bytes a second, late, first byte, what failed."""
    meter = Meter()
    stop = threading.Event()
    threads = [
        threading.Thread(target=stream.run, args=(meter, stop), daemon=True) for stream in streams
    ]
    for thread in threads:
        thread.start()
    stop.wait(seconds)
    mean, late = meter.rates(seconds)
    stop.set()
    for stream in streams:
        stream.kill()
    for thread in threads:
        thread.join(timeout=15)
    failed = sorted({str(getattr(stream, "said", "")) for stream in streams} - {""})
    return mean, late, meter.first, ("; ".join(failed)[:200] if meter.total == 0 else "")


def table(rows: Sequence[Row]) -> str:
    """The rows as the lines a person reads."""
    lines = [
        f"{'way':<12} {'streams':>7}  {'':<30} {'MB/s':>7} {'late':>7} {'first byte':>10}",
    ]
    for direction in ("home", "up"):
        for row in rows:
            if row.direction != direction:
                continue
            first = "" if row.first_byte_s is None else f"{row.first_byte_s:.1f} s"
            what = row.note + (" (to the machine)" if direction == "up" else "")
            lines.append(
                f"{row.way:<12} {row.streams:>7}  {what:<30} {row.megabytes_per_s:>7.1f}"
                f" {row.late_megabytes_per_s:>7.1f} {first:>10}"
                + (f"  {row.error}" if row.error else "")
            )
    return "\n".join(lines)


def verdict(rows: Sequence[Row]) -> dict[str, Any]:
    """What a fetch on this host does, from what each way brought home late.

    Each way is taken at the count of streams it was the fastest on; a
    cipher is named for direct ssh where it beat the client's own choice by
    :data:`WORTH`. The order is the rates', except that a way must beat
    one with fewer parts by :data:`WORTH` to go before it: the proxy is
    what always worked, direct ssh opens no port, HTTPS runs a server.
    """
    home = [row for row in rows if row.direction == "home" and not row.error]
    best: dict[str, Row] = {}
    for row in home:
        if row.way not in (transfer.HTTPS, transfer.SSH_DIRECT, transfer.SSH_PROXY):
            continue
        if row.way == transfer.SSH_DIRECT and row.note:
            # A cipher's, or the shared connection's: judged below, not a count of streams.
            continue
        if row.way not in best or row.late_megabytes_per_s > best[row.way].late_megabytes_per_s:
            best[row.way] = row
    cipher = None
    named = [row for row in home if row.way == transfer.SSH_DIRECT and row.note in CIPHERS]
    if named:
        ahead = max(named, key=lambda row: row.late_megabytes_per_s)
        own = [
            row.late_megabytes_per_s
            for row in home
            if row.way == transfer.SSH_DIRECT and not row.note and row.streams == ahead.streams
        ]
        if own and ahead.late_megabytes_per_s > WORTH * own[0]:
            cipher = ahead.note
    simplest = [transfer.SSH_PROXY, transfer.SSH_DIRECT, transfer.HTTPS]
    order: list[str] = []
    for way in simplest:
        if way not in best:
            continue
        rate = best[way].late_megabytes_per_s
        at = len(order)
        for index, other in enumerate(order):
            if rate > WORTH * best[other].late_megabytes_per_s:
                at = index
                break
        order.insert(at, way)
    return {
        "order": order,
        "streams": {way: best[way].streams for way in order},
        "megabytes_per_s": {way: round(best[way].late_megabytes_per_s, 1) for way in order},
        "cipher": cipher,
    }


def _with(argv: list[str], *words: str) -> list[str]:
    """``argv``, an ``ssh`` command, with ``words`` put before its own options."""
    return [argv[0], *words, *argv[1:]]


def bench(
    machine: Any,
    *,
    seconds: float = 5.0,
    megabytes: int = 512,
    say: Callable[[str], None] = print,
    run: Callable[..., str] | None = None,
    serving: Callable[..., transfer.Served] | None = None,
    up: bool = True,
) -> dict[str, Any]:
    """Measure every way on ``machine``; the rows, the verdict, and where it was written.

    ``machine`` is the instance as :func:`reverberate.gpu.vast.wait_for_ssh`
    hands it: on the proxy, its own address pinned where it has one.
    """
    from reverberate.gpu.direct import rsync_shell
    from reverberate.wave.remote import run_on

    run = run or run_on
    direct = fastest(machine, say=say)
    here = direct if direct is not machine else machine
    told = run(
        here,
        f"mkdir -p {REMOTE}/out && cd {REMOTE}/out &&"
        f' {{ [ "$(stat -c %s bench.bin 2>/dev/null)" = {megabytes * 2**20} ] ||'
        f" head -c {megabytes}M /dev/urandom > bench.bin; }};"
        " t=$(date +%s.%N); sha256sum bench.bin >/dev/null; echo sha256 $t $(date +%s.%N);"
        " echo cores $(nproc);"
        " echo sshd $(sshd -T 2>/dev/null | grep -i -E '^(maxstartups|maxsessions)'"
        " | tr '\\n' ' ')",
        what="bench file",
        timeout=300,
    )
    facts: dict[str, Any] = {}
    for line in str(told).splitlines():
        name, _, rest = line.strip().partition(" ")
        if name == "sha256" and len(rest.split()) == 2:
            with contextlib.suppress(ValueError):
                began, ended = (float(word) for word in rest.split())
                facts["machine_sha256_megabytes_per_s"] = round(
                    megabytes * 2**20 / MB / max(ended - began, 1e-6)
                )
        elif name in ("cores", "sshd"):
            facts[name] = rest
    say(f"on the machine: {json.dumps(facts)}")
    rows: list[Row] = []
    # Again and again: a fast line reads the file through before the measurement ends.
    read = f"while cat {shlex.quote(REMOTE_FILE)}; do :; done"
    sink = "cat > /dev/null"
    plain = ("-o", "Compression=no")

    def take(row: Row, streams: Sequence[Stream]) -> None:
        mean, late, first, error = measure(streams, seconds)
        row.megabytes_per_s, row.late_megabytes_per_s = round(mean / MB, 1), round(late / MB, 1)
        row.first_byte_s = None if first is None else round(first, 2)
        row.error = error
        rows.append(row)
        say(table([row]).splitlines()[-1])

    def ssh(route: Any, count: int, command: str, *words: str, up: bool = False) -> list[Stream]:
        return [
            Command(_with(route.ssh_command(command), *plain, *words), up=up) for _ in range(count)
        ]

    say(table([]))
    take(Row(transfer.SSH_PROXY, 4), ssh(machine, 4, read))
    if direct is not machine:
        for count in (1, 4, 8):
            take(Row(transfer.SSH_DIRECT, count), ssh(direct, count, read))
        for cipher in CIPHERS:
            take(Row(transfer.SSH_DIRECT, 4, cipher), ssh(direct, 4, read, "-c", cipher))
        shared = direct.sharing(f"bench.{os.getpid()}")
        try:
            run(shared, "true", what="shared connection", timeout=60)
            take(Row(transfer.SSH_DIRECT, 4, "one shared connection"), ssh(shared, 4, read))
        except Exception as error:  # noqa: BLE001 - a row that could not be measured
            rows.append(
                Row(transfer.SSH_DIRECT, 4, "one shared connection", error=str(error)[:120])
            )
        finally:
            closing = shared.close_command()
            if closing is not None:
                subprocess.run(closing, capture_output=True, timeout=20, check=False)
        with tempfile.TemporaryDirectory(prefix="rv-bench-") as folder:
            argv = ["rsync", "-a", "--partial", "-e", rsync_shell(direct)]
            argv += [f"{direct.user}@{direct.host}:{REMOTE_FILE}", folder + "/"]
            take(Row("rsync", 1, "direct"), [Growing(argv, Path(folder))])
        try:
            served = (serving or transfer.serve)(direct, f"{REMOTE}/out")
        except Exception as error:  # noqa: BLE001 - said in the table, and ssh stands
            rows.append(Row(transfer.HTTPS, 0, error=str(error)[:160]))
            say(f"{transfer.HTTPS:<12} not open: {str(error)[:160]}")
        else:
            try:
                for count in (1, 4, 8, 16, 32):
                    take(
                        Row(transfer.HTTPS, count),
                        [Ranges(served, REMOTE_FILE) for _ in range(count)],
                    )
            finally:
                transfer.stop(direct, run=run)
    if up:
        if direct is not machine:
            for count in (1, 4):
                take(
                    Row(transfer.SSH_DIRECT, count, direction="up"),
                    ssh(direct, count, sink, up=True),
                )
        take(Row(transfer.SSH_PROXY, 4, direction="up"), ssh(machine, 4, sink, up=True))
    with contextlib.suppress(Exception):
        run(here, f"rm -rf {REMOTE}", what="bench file removed", timeout=60)
    found = verdict(rows)
    record = {**found, "measured": time.time(), "machine": facts, "rows": [asdict(r) for r in rows]}
    where = transfer.verdict_path(direct)
    if where is not None and found["order"]:
        where.write_text(json.dumps(record, indent=1))
        record["written"] = str(where)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--instance", type=int, required=True, help="an instance already rented")
    parser.add_argument("--seconds", type=float, default=5.0, help="what each measurement lasts")
    parser.add_argument("--megabytes", type=int, default=512, help="the file read on the machine")
    parser.add_argument("--no-up", action="store_true", help="leave the way up out")
    parser.add_argument("--json", type=Path, default=None, help="the record, written here too")
    args = parser.parse_args(argv)
    from reverberate import auth
    from reverberate.gpu import vast

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    machine = vast.wait_for_ssh(client, args.instance, vast.account_identity(client), timeout=180)
    began = time.time()
    record = bench(machine, seconds=args.seconds, megabytes=args.megabytes, up=not args.no_up)
    print()
    print(table([Row(**row) for row in record["rows"]]))
    print()
    if record["order"]:
        first = record["order"][0]
        rate = record["megabytes_per_s"][first]
        print(
            f"verdict: {first} on {record['streams'][first]} streams, {rate:g} MB/s late"
            + (f"; ssh with {record['cipher']}" if record["cipher"] else "")
            + f"; then {', '.join(record['order'][1:]) or 'nothing'}"
        )
        if rate > 0:
            print(f"a pack of 9.5 GB at that rate: {transfer.minutes_home(9.5, rate):.1f} min")
        print(
            f"kept for the driver's fetch: {record.get('written', 'nowhere (no pinned address)')}"
        )
    else:
        print("verdict: nothing was measured")
    print(f"the bench took {time.time() - began:.0f} s")
    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=1))
    return 0 if record["order"] else 1


if __name__ == "__main__":
    sys.exit(main())
