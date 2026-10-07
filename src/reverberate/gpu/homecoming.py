"""Bringing a run's files home from a rented machine: in chunks, resumed, verified.

The first whole scene (2026-10-05) came home through Vast's ssh proxy, and
what that measured is what this module is built on:

- **one stream is 2.0 to 2.3 MB/s**, four are 4.4 and eight are 5.9, while
  the laptop's own line carries 70 MB/s to Europe and 17 to the United
  States: the proxy is the limit, and a few streams are worth having;
- **twelve streams that reconnect at once are refused together**. The
  instance's ``sshd`` admits ten connections that have not yet
  authenticated (``MaxStartups``): "Connection timed out during banner
  exchange", then every stream dropped in the same second;
- **a stream that drops must not cost what it had brought**. A pack in
  twelve parts of 2 GB lost a part's whole progress at each drop;
- **the machine bills while the laptop downloads**: 1.38 USD an hour for
  the forty minutes of a pack.

So a large file (:func:`fetch_file`) is read as ranges of
:data:`CHUNK_BYTES` by :data:`WORKERS` workers. A chunk that arrived whole
is written at its place in the file and never asked for again, by this run
or by the next; a chunk is checked by its size and the file by the SHA-256
the machine computes of it, and where the two differ the chunks are
compared one by one and only the wrong ones fetched again. A directory of
many files, a pair cache (:func:`fetch_tree`), goes as batches of whole
files of about that size, each a ``tar`` stream, each file checked by its
size; what is home is not sent again, and a pass may be given a time to end
by.

Every worker keeps **one connection** and reads its chunks on it
(``ControlMaster``, :meth:`reverberate.wave.remote.Machine.sharing`), so a
transfer of three hundred chunks opens four connections and not three
hundred. A failure makes **every** worker wait (:class:`Pace`): a refused
connection is the host saying it has too many, and three other workers
trying again at once are what it was refusing.

**A direct connection is tried first** (:func:`fastest`) where the
instance says it has one, and the proxy is what is fallen back on.

Nothing here knows Vast or a trace: a machine is addressed over ssh, and
:class:`Transport` is the four things asked of it, which the tests answer
from a directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from reverberate.wave.remote import (
    ConnectionLost,
    Machine,
    RemoteError,
    _run,
    connection_level,
)

__all__ = [
    "BLOCK_BYTES",
    "CHUNK_BYTES",
    "WORKERS",
    "Pace",
    "SshTransport",
    "Transport",
    "fastest",
    "fetch_file",
    "fetch_tree",
]

#: What ``dd`` reads at once on the machine; a chunk is a whole number of them.
BLOCK_BYTES = 4 * 2**20
#: A chunk of a large file, and a batch of small ones: what a dropped stream costs at most.
CHUNK_BYTES = 8 * BLOCK_BYTES
#: Streams at once. Four brought 4.4 MB/s through the proxy and eight 5.9; twelve were
#: refused together. Four leaves the host's ``sshd`` room for the watcher's own commands.
WORKERS = 4
#: A chunk that brings less than this, in bytes a second, has stalled and is ended: a
#: stream of the first scene brought 2 MB/s, and one of them none for ten minutes while
#: its connection stayed open (2026-10-05).
STALLED_BYTES_PER_S = 150e3
#: The pause after a failure, doubling with each one that follows it, and its cap, s.
PAUSE_S = 3.0
PAUSE_CAP_S = 120.0
#: Failures in a row, over all workers, after which the homecoming gives up.
GIVE_UP_AFTER = 30
#: How long a worker that is not among those working waits before it looks again, s.
LANE_WAIT_S = 0.2
#: Streams a file of a directory is asked for in, in one pass, before it is left to the next.
FILE_TRIES = 3


class Pace:
    """One pause for every worker of a homecoming.

    A worker asks :meth:`wait` before each chunk. A failure pushes the
    moment everyone may go on, twice as far with each failure that follows
    another; a success halves the count. After :data:`GIVE_UP_AFTER`
    failures with no success between, :meth:`failed` raises.
    """

    def __init__(
        self,
        *,
        pause_s: float = PAUSE_S,
        cap_s: float = PAUSE_CAP_S,
        give_up_after: int = GIVE_UP_AFTER,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pause_s, self.cap_s, self.give_up_after = pause_s, cap_s, give_up_after
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._streak = 0
        self._row = 0
        self._not_before = 0.0
        #: Every failure, and the longest pause it asked for.
        self.failures = 0
        self.longest_s = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                left = self._not_before - self._clock()
            if left <= 0.0:
                return
            self._sleep(min(left, 5.0))

    def failed(self, why: str = "") -> float:
        with self._lock:
            self.failures += 1
            self._streak += 1
            self._row += 1
            pause = float(min(self.pause_s * 2.0 ** (self._streak - 1), self.cap_s))
            self.longest_s = max(self.longest_s, pause)
            self._not_before = max(self._not_before, self._clock() + pause)
            if self._row > self.give_up_after:
                raise ConnectionLost(
                    f"the homecoming failed {self._row} times in a row: {why[-300:]}", 255
                )
            return pause

    def succeeded(self) -> None:
        with self._lock:
            self._streak //= 2
            self._row = 0


class Transport(Protocol):
    """What a homecoming asks of the machine."""

    def size(self, remote: str) -> int: ...

    def sha256(self, remote: str) -> str: ...

    def read(self, remote: str, offset: int, count: int, target: Path, worker: int) -> None:
        """Bytes ``offset`` to ``offset + count`` of ``remote`` into ``target``."""
        ...

    def chunk_digests(self, remote: str, chunk_bytes: int, chunks: Sequence[int]) -> list[str]: ...

    def listing(self, remote_dir: str, exclude: Collection[str]) -> list[tuple[str, int]]:
        """Every file under ``remote_dir``: its path there and its bytes."""
        ...

    def files(
        self,
        remote_dir: str,
        names: Sequence[str],
        target: Path,
        worker: int,
        timeout: float | None,
    ) -> None:
        """The files ``names`` of ``remote_dir`` under ``target``, as far as the time allows."""
        ...

    def close(self) -> None: ...


class SshTransport:
    """The machine over ssh: a kept connection a worker, ``dd`` for a range, ``tar`` for files."""

    def __init__(self, machine: Machine, *, share: bool = True, cipher: str | None = None) -> None:
        self.machine = machine
        self.share = share
        #: The cipher the ranges are read with (``ssh -c``); the client's own choice when
        #: none is named. A host's bench names one where it was the faster there.
        self.cipher = cipher
        self._kept: dict[int, Machine] = {}

    def _machine(self, worker: int) -> Machine:
        if not self.share:
            return self.machine
        if worker not in self._kept:
            self._kept[worker] = self.machine.sharing(f"{os.getpid()}.{worker}")
        return self._kept[worker]

    def size(self, remote: str) -> int:
        said = _run(
            self.machine.ssh_command(f"stat -c %s {shlex.quote(remote)}"), what="size", timeout=90
        )
        return int(said.strip().split()[-1])

    def sha256(self, remote: str) -> str:
        said = _run(
            self.machine.ssh_command(f"sha256sum {shlex.quote(remote)}"),
            what="sha256",
            timeout=3600,
        )
        return _digest_of(said)

    def chunk_digests(self, remote: str, chunk_bytes: int, chunks: Sequence[int]) -> list[str]:
        per = chunk_bytes // BLOCK_BYTES
        script = (
            f"for i in {' '.join(str(int(c)) for c in chunks)}; do"
            f" dd if={shlex.quote(remote)} bs={BLOCK_BYTES} skip=$((i * {per})) count={per}"
            " 2>/dev/null | sha256sum; done"
        )
        said = _run(self.machine.ssh_command(script), what="chunk digests", timeout=3600)
        found = [line.split()[0] for line in said.splitlines() if _is_digest(line.split()[:1])]
        if len(found) != len(chunks):
            raise RemoteError(f"{len(found)} digests came for {len(chunks)} chunks")
        return found

    def read(self, remote: str, offset: int, count: int, target: Path, worker: int) -> None:
        if offset % BLOCK_BYTES:
            raise ValueError("a range starts on a block")
        blocks = -(-count // BLOCK_BYTES)
        argv = self._machine(worker).ssh_command(
            f"dd if={shlex.quote(remote)} bs={BLOCK_BYTES} skip={offset // BLOCK_BYTES}"
            f" count={blocks} 2>/dev/null"
        )
        if self.cipher:
            argv = [argv[0], "-c", self.cipher, *argv[1:]]
        limit = max(120.0, count / STALLED_BYTES_PER_S)
        # What ssh says goes to a file and not to a pipe: the connection that is kept
        # outlives the command, and a pipe it held open would be waited on with it.
        with target.open("wb") as handle, tempfile.TemporaryFile() as said:
            try:
                done = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=said,
                    timeout=limit,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise ConnectionLost(f"a chunk did not come in {limit:g} s", None) from None
            said.seek(0)
            stderr = said.read().decode(errors="replace")
        _judge(argv, done.returncode, stderr, "chunk")

    def listing(self, remote_dir: str, exclude: Collection[str]) -> list[tuple[str, int]]:
        skip = " ".join(f"! -name {shlex.quote(pattern)}" for pattern in exclude)
        said = _run(
            self.machine.ssh_command(
                f"cd {shlex.quote(remote_dir)} 2>/dev/null &&"
                f" find . -type f {skip} -printf '%s\\t%P\\n'; true"
            ),
            what="list",
            timeout=120,
            attempts=2,
        )
        found = []
        for line in said.splitlines():
            size, _, name = line.partition("\t")
            if name and size.isdigit():
                found.append((name, int(size)))
        return found

    def files(
        self,
        remote_dir: str,
        names: Sequence[str],
        target: Path,
        worker: int,
        timeout: float | None,
    ) -> None:
        argv = self._machine(worker).ssh_command(
            f"tar -C {shlex.quote(remote_dir)} -cf - --null -T -"
        )
        # What ssh says goes to a file and not to a pipe, as for a chunk.
        with tempfile.TemporaryFile() as said:
            process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=said
            )
            assert process.stdin is not None and process.stdout is not None
            timer = threading.Timer(timeout, process.kill) if timeout is not None else None
            if timer is not None:
                timer.start()
            try:
                with contextlib.suppress(OSError):
                    process.stdin.write("\0".join(names).encode() + b"\0")
                    process.stdin.close()
                # A stream that is cut ends the archive in the middle of a file: what was
                # whole before it is kept, and the caller reads each file's size.
                with (
                    contextlib.suppress(tarfile.TarError, OSError, EOFError),
                    tarfile.open(fileobj=process.stdout, mode="r|") as archive,
                ):
                    archive.extractall(target, filter="data")
                process.stdout.close()
                code = process.wait()
                said.seek(0)
                stderr = said.read().decode(errors="replace")
            finally:
                if timer is not None:
                    timer.cancel()
                if process.poll() is None:
                    process.kill()
        if code < 0:
            raise ConnectionLost("a batch of files was ended at its time", None)
        if code == 1:
            # ``tar``'s word for a file that grew while it was read: a campaign's index
            # does. What came is judged by the caller, file by file.
            return
        _judge(argv, code, stderr, "files")

    def close(self) -> None:
        for kept in self._kept.values():
            argv = kept.close_command()
            if argv is not None:
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(argv, capture_output=True, timeout=20, check=False)
        self._kept.clear()


def _judge(argv: list[str], code: int, stderr: str, what: str) -> None:
    """Raise what a finished ``ssh`` says of itself: the connection's failure, or the command's."""
    if code == 0:
        return
    if connection_level(argv, code, stderr.strip()):
        raise ConnectionLost(f"{what}: the connection was lost: {stderr.strip()[-300:]}", 255)
    raise RemoteError(f"{what} failed: {stderr.strip()[-600:]}", code)


def _is_digest(words: list[str]) -> bool:
    return bool(words) and len(words[0]) == 64 and all(c in "0123456789abcdef" for c in words[0])


def _digest_of(said: str) -> str:
    for line in said.splitlines():
        if _is_digest(line.split()[:1]):
            return line.split()[0]
    raise RemoteError(f"no digest in {said[-200:]!r}")


def fastest(machine: Machine, *, say: Any = None, probe: Any = None) -> Machine:
    """``machine`` by its own address where that answers; as it is, through the proxy, otherwise.

    One command, once, with no retry: a direct port that does not answer in
    :data:`reverberate.wave.remote.CONNECT_TIMEOUT_S` is not waited for.
    """
    say = say or (lambda message: None)
    directly = getattr(machine, "directly", None)
    direct = directly() if directly is not None else None
    if direct is None:
        return machine
    probe = probe or (
        lambda m: _run(m.ssh_command("true"), what="direct ssh", timeout=45, attempts=1)
    )
    try:
        probe(direct)
    except Exception as error:  # noqa: BLE001 - the proxy is the way that is known to work
        say(
            f"  direct ssh to {direct.host}:{direct.port} did not answer"
            f" ({str(error)[:120]}): the proxy"
        )
        return machine
    say(f"  direct ssh to {direct.host}:{direct.port} answers: transfers go past the proxy")
    found: Machine = direct
    return found


def _sha256(path: Path, start: int = 0, count: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        left = count
        while left is None or left > 0:
            block = handle.read(BLOCK_BYTES if left is None else min(BLOCK_BYTES, left))
            if not block:
                break
            digest.update(block)
            if left is not None:
                left -= len(block)
    return digest.hexdigest()


def _arrived(note: Path, remote: str, size: int) -> str | None:
    """The digest ``note`` says came home for ``remote`` of ``size`` bytes, or ``None``."""
    try:
        held = json.loads(note.read_text())
        if (held["remote"], int(held["size"])) == (remote, size):
            return str(held["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def fetch_file(
    machine: Machine | None,
    remote: str,
    local: Path,
    *,
    workers: int = WORKERS,
    chunk_bytes: int = CHUNK_BYTES,
    say: Any = None,
    transport: Transport | None = None,
    pace: Pace | None = None,
    lanes: Any = None,
) -> dict[str, Any]:
    """``remote`` on the machine as ``local``, verified; what the transfer was.

    The file is built in ``<local>.partial``, each chunk written at its
    place as it arrives whole, and ``<local>.chunks.json`` says which have:
    a run that is interrupted, or fails, leaves both, and the next call
    asks only for what is missing. A file whose size on the machine is not
    the one the chunks were of starts again. ``local`` appears, whole, when
    its SHA-256 is the machine's, and ``<local>.home.json`` then says which
    file of the machine it is: a later call for the same file, by its size
    and its digest there, brings nothing (``already_home``), whatever was
    written into the local copy since.

    ``lanes`` (:class:`reverberate.gpu.transfer.Lanes`) says how many of
    the workers work at a time, and is told of every chunk and every
    failure; its ``most`` is then the number of workers. Without it
    ``workers`` all work, as they always did.

    Raises :class:`reverberate.wave.remote.ConnectionLost` when the
    connection fails :data:`GIVE_UP_AFTER` times in a row, and
    :class:`reverberate.wave.remote.RemoteError` for a file that still
    differs from the machine's after its wrong chunks were fetched again.
    """
    if chunk_bytes % BLOCK_BYTES:
        raise ValueError(f"a chunk is a whole number of blocks of {BLOCK_BYTES} bytes")
    say = say or (lambda message: None)
    if transport is None:
        if machine is None:
            raise ValueError("a machine or a transport")
        transport = SshTransport(machine)
    pace = pace or Pace()
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(local.name + ".partial")
    ledger = local.with_name(local.name + ".chunks.json")
    arrived = local.with_name(local.name + ".home.json")
    if lanes is not None:
        workers = int(lanes.most)
    started = time.time()
    try:
        size = transport.size(remote)
        count = -(-size // chunk_bytes)
        was = _arrived(arrived, remote, size) if local.is_file() else None
        if was is not None and transport.sha256(remote) == was:
            say(f"  {Path(remote).name}: home already, the machine's file is the one that came")
            return {
                "file": str(local),
                "bytes": size,
                "sha256": was,
                "chunks": count,
                "chunks_resumed": count,
                "chunks_refetched": 0,
                "failures": 0,
                "seconds": round(time.time() - started, 1),
                "bytes_per_s": 0,
                "already_home": True,
            }
        done: set[int] = set()
        if ledger.is_file() and partial.is_file():
            with contextlib.suppress(ValueError, KeyError, TypeError):
                held = json.loads(ledger.read_text())
                if (held["size"], held["chunk_bytes"], held["remote"]) == (
                    size,
                    chunk_bytes,
                    remote,
                ):
                    done = {int(i) for i in held["done"] if 0 <= int(i) < count}
        if not done:
            with partial.open("wb") as handle:
                handle.truncate(size)
        resumed = len(done)
        say(
            f"  {Path(remote).name}: {size / 1e9:.2f} GB in {count} chunks of"
            f" {chunk_bytes / 2**20:g} MB,"
            + (
                f" {workers} streams"
                if lanes is None or lanes.allowed >= workers
                else f" {lanes.allowed} streams, {workers} at most"
            )
            + (f"; {resumed} chunks already home" if resumed else "")
        )
        # The machine reads its file once for the digest while the chunks come.
        digest: dict[str, Any] = {}

        def remote_digest() -> None:
            try:
                digest["sha256"] = transport.sha256(remote)
            except Exception as error:  # noqa: BLE001 - asked again below
                digest["error"] = error

        digesting = threading.Thread(target=remote_digest, daemon=True)
        digesting.start()
        lock = threading.Lock()
        scratch = local.with_name(local.name + ".chunks")
        scratch.mkdir(exist_ok=True)

        def note() -> None:
            ledger.write_text(
                json.dumps(
                    {
                        "remote": remote,
                        "size": size,
                        "chunk_bytes": chunk_bytes,
                        "done": sorted(done),
                    }
                )
            )

        def bring(todo: list[int]) -> None:
            queue = list(todo)
            failure: list[BaseException] = []

            def work(worker: int) -> None:
                while not failure:
                    if lanes is not None and not lanes.may(worker):
                        # Not one of those that work now: it looks again, until all is taken.
                        with lock:
                            if not queue:
                                return
                        time.sleep(LANE_WAIT_S)
                        continue
                    with lock:
                        if not queue:
                            return
                        chunk = queue.pop(0)
                    want = min(chunk_bytes, size - chunk * chunk_bytes)
                    piece = scratch / f"{chunk:06d}.{worker}"
                    while not failure:
                        pace.wait()
                        try:
                            transport.read(remote, chunk * chunk_bytes, want, piece, worker)
                            if piece.stat().st_size < want:
                                raise ConnectionLost(
                                    f"chunk {chunk} came short: {piece.stat().st_size} of {want}",
                                    None,
                                )
                        except ConnectionLost as error:
                            if lanes is not None:
                                lanes.failed()
                            try:
                                pace.failed(str(error))
                            except ConnectionLost as last:
                                failure.append(last)
                            continue
                        except BaseException as error:  # noqa: BLE001 - the others stop too
                            failure.append(error)
                            continue
                        pace.succeeded()
                        if lanes is not None:
                            lanes.brought(want)
                        with lock:
                            with partial.open("r+b") as whole, piece.open("rb") as part:
                                whole.seek(chunk * chunk_bytes)
                                whole.write(part.read(want))
                            done.add(chunk)
                            note()
                            if len(done) % 25 == 0 or len(done) == count:
                                rate = (
                                    (len(done) - resumed)
                                    * chunk_bytes
                                    / max(time.time() - started, 1e-9)
                                )
                                say(
                                    f"  {Path(remote).name}: {len(done)} of {count} chunks home,"
                                    f" {rate / 1e6:.1f} MB/s"
                                )
                        piece.unlink(missing_ok=True)
                        break

            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for future in [pool.submit(work, w) for w in range(max(1, workers))]:
                    future.result()
            if failure:
                raise failure[0]

        note()
        bring([chunk for chunk in range(count) if chunk not in done])
        digesting.join()
        wanted = digest.get("sha256") or transport.sha256(remote)
        refetched = 0
        if _sha256(partial) != wanted:
            # Which chunks: the machine's digest of each against this file's.
            chunks = list(range(count))
            theirs = transport.chunk_digests(remote, chunk_bytes, chunks)
            wrong = [
                chunk
                for chunk in chunks
                if _sha256(partial, chunk * chunk_bytes, chunk_bytes) != theirs[chunk]
            ]
            say(f"  {Path(remote).name}: {len(wrong)} chunk(s) differ from the machine's; again")
            if not wrong:
                raise RemoteError(
                    f"{remote} changed on the machine while it was fetched: its digest is not"
                    " the one of its chunks"
                )
            done -= set(wrong)
            note()
            bring(wrong)
            refetched = len(wrong)
            if _sha256(partial) != wanted:
                raise RemoteError(f"{remote} still differs from the machine's after {wrong}")
        partial.replace(local)
        arrived.write_text(json.dumps({"remote": remote, "size": size, "sha256": wanted}))
        ledger.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
    finally:
        transport.close()
    seconds = time.time() - started
    brought = (count - resumed + refetched) * chunk_bytes
    rate = round(min(brought, size) / max(seconds, 1e-9))
    record: dict[str, Any] = {
        "file": str(local),
        "bytes": size,
        "sha256": wanted,
        "chunks": count,
        "chunks_resumed": resumed,
        "chunks_refetched": refetched,
        "failures": pace.failures,
        "seconds": round(seconds, 1),
        "bytes_per_s": rate,
    }
    if lanes is not None:
        record["workers"] = int(lanes.allowed)
    say(
        f"  {Path(remote).name}: home and verified in {seconds / 60:.1f} min,"
        f" {rate / 1e6:.1f} MB/s, {pace.failures} failure(s) on the way"
    )
    return record


def fetch_tree(
    machine: Machine | None,
    remote_dir: str,
    local_dir: Path,
    *,
    exclude: Collection[str] = (),
    workers: int = WORKERS,
    batch_bytes: int = CHUNK_BYTES,
    seconds: float | None = None,
    say: Any = None,
    transport: Transport | None = None,
    pace: Pace | None = None,
) -> dict[str, Any]:
    """What ``local_dir`` lacks of ``remote_dir``, in batches of whole files; what came.

    A file is home when it is there with the machine's size; the others
    are asked for, about ``batch_bytes`` a batch, a batch a ``tar`` stream.
    A batch lands beside its place and each of its files is moved in when
    its size is right, so a stream that drops costs the file it was in and
    nothing that came before. A file that came longer than it was listed
    grew meanwhile, as a running campaign's index does: it is kept, and
    the next pass brings it again. A file that does not come whole in
    :data:`FILE_TRIES` streams is left to the next pass. ``seconds`` bounds
    the pass: no batch starts
    after it and those running are ended there, which is a pass and not a
    failure (``complete`` is then false and the next goes on). Never raises
    for a connection: what it could not bring is in ``left``.
    """
    say = say or (lambda message: None)
    if transport is None:
        if machine is None:
            raise ValueError("a machine or a transport")
        transport = SshTransport(machine)
    pace = pace or Pace()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ends = None if seconds is None else started + float(seconds)
    record: dict[str, Any] = {"files": 0, "bytes": 0, "left": 0, "complete": False, "there": 0}
    try:
        try:
            listed = transport.listing(remote_dir, exclude)
        except RemoteError as error:
            record["error"] = str(error)[:300]
            return record
        record["there"] = len(listed)
        missing = [
            (name, size)
            for name, size in sorted(listed)
            if not (local_dir / name).is_file() or (local_dir / name).stat().st_size != size
        ]
        batches: list[list[tuple[str, int]]] = [[]]
        held = 0
        for name, size in missing:
            if batches[-1] and held + size > batch_bytes:
                batches.append([])
                held = 0
            batches[-1].append((name, size))
            held += size
        batches = [batch for batch in batches if batch]
        lock = threading.Lock()
        stop: list[str] = []
        tried: dict[str, int] = {}
        given_up: list[str] = []

        def work(worker: int) -> None:
            incoming = local_dir / f".incoming.{os.getpid()}.{worker}"
            while not stop:
                with lock:
                    if not batches:
                        return
                    batch = batches.pop(0)
                left = None if ends is None else ends - time.time()
                if left is not None and left <= 1.0:
                    with lock:
                        batches.insert(0, batch)
                    return
                pace.wait()
                shutil.rmtree(incoming, ignore_errors=True)
                incoming.mkdir(parents=True)
                failed = ""
                try:
                    transport.files(remote_dir, [n for n, _ in batch], incoming, worker, left)
                except ConnectionLost as error:
                    failed = str(error)
                except RemoteError as error:
                    stop.append(str(error))
                    failed = str(error)
                again = []
                for name, size in batch:
                    came = incoming / name
                    if came.is_file() and came.stat().st_size >= size:
                        (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
                        came.replace(local_dir / name)
                        with lock:
                            record["files"] += 1
                            record["bytes"] += size
                        continue
                    with lock:
                        tried[name] = tried.get(name, 0) + 1
                        if tried[name] >= FILE_TRIES:
                            given_up.append(name)
                            continue
                    again.append((name, size))
                shutil.rmtree(incoming, ignore_errors=True)
                if again:
                    with lock:
                        batches.insert(0, again)
                if failed and not stop and ends is not None and time.time() >= ends - 1.0:
                    # Ended at the pass's own time: a pass, not a failure.
                    return
                if failed and not stop:
                    try:
                        pace.failed(failed)
                    except ConnectionLost as last:
                        stop.append(str(last))
                elif not failed:
                    pace.succeeded()

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for future in [pool.submit(work, w) for w in range(max(1, workers))]:
                future.result()
        record["left"] = sum(len(batch) for batch in batches) + len(given_up)
        record["complete"] = record["left"] == 0
        if stop:
            record["error"] = stop[0][:300]
    finally:
        transport.close()
        record["seconds"] = round(time.time() - started, 1)
        record["failures"] = pace.failures
    return record
