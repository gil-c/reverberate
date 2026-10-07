"""A run's files brought home in chunks: resumed, verified, and paced as one.

No machine and no ssh: :class:`FakeTransport` answers from a directory and
fails when it is told to, which is what the first whole scene's proxy did.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reverberate.accel.pairs import PairCache
from reverberate.gpu.homecoming import (
    BLOCK_BYTES,
    Pace,
    SshTransport,
    fastest,
    fetch_file,
    fetch_tree,
)
from reverberate.wave.remote import ConnectionLost, Machine, RemoteError

#: A chunk of the tests: the smallest a range may be.
CHUNK = BLOCK_BYTES


def _bytes(count: int, seed: int = 0) -> bytes:
    block = hashlib.sha256(str(seed).encode()).digest()
    return (block * (count // len(block) + 1))[:count]


class FakeTransport:
    """A directory standing in for the machine; ``fail`` says which reads are lost."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reads: list[tuple[int, int]] = []
        self.fail: dict[int, int] = {}
        self.corrupt: set[int] = set()
        self.short: set[int] = set()
        self.batches: list[list[str]] = []
        self.drop_after: int | None = None
        self.closed = 0
        self._lock = threading.Lock()

    def size(self, remote: str) -> int:
        return (self.root / remote).stat().st_size

    def sha256(self, remote: str) -> str:
        return hashlib.sha256((self.root / remote).read_bytes()).hexdigest()

    def chunk_digests(self, remote: str, chunk_bytes: int, chunks: Sequence[int]) -> list[str]:
        held = (self.root / remote).read_bytes()
        return [
            hashlib.sha256(held[c * chunk_bytes : (c + 1) * chunk_bytes]).hexdigest()
            for c in chunks
        ]

    def read(self, remote: str, offset: int, count: int, target: Path, worker: int) -> None:
        chunk = offset // CHUNK
        with self._lock:
            self.reads.append((chunk, worker))
            left = self.fail.get(chunk, 0)
            if left:
                self.fail[chunk] = left - 1
        if left:
            raise ConnectionLost("Connection timed out during banner exchange", 255)
        held = (self.root / remote).read_bytes()[offset : offset + count]
        if chunk in self.short:
            self.short.discard(chunk)
            held = held[: count // 2]
        if chunk in self.corrupt:
            self.corrupt.discard(chunk)
            held = bytes(count)
        target.write_bytes(held)

    def listing(self, remote_dir: str, exclude: Collection[str]) -> list[tuple[str, int]]:
        base = self.root / remote_dir
        if not base.is_dir():
            raise RemoteError("list failed: No such file or directory", 1)
        return [
            (str(path.relative_to(base)), path.stat().st_size)
            for path in sorted(base.rglob("*"))
            if path.is_file() and not any(path.match(pattern) for pattern in exclude)
        ]

    def files(
        self,
        remote_dir: str,
        names: Sequence[str],
        target: Path,
        worker: int,
        timeout: float | None,
    ) -> None:
        with self._lock:
            self.batches.append(list(names))
            drop = self.drop_after
            self.drop_after = None
        for k, name in enumerate(names):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            if drop is not None and k == drop:
                # The stream is cut in the middle of this file.
                (target / name).write_bytes((self.root / remote_dir / name).read_bytes()[:7])
                raise ConnectionLost("Connection closed by remote host", 255)
            shutil.copyfile(self.root / remote_dir / name, target / name)

    def close(self) -> None:
        self.closed += 1


def _pace(pauses: list[float] | None = None) -> Pace:
    """A pace that does not wait: its clock moves when it sleeps."""
    now = [0.0]

    def sleep(seconds: float) -> None:
        if pauses is not None:
            pauses.append(seconds)
        now[0] += seconds

    return Pace(clock=lambda: now[0], sleep=sleep)


class TestFetchFile:
    def test_a_file_comes_home_whole_and_its_digest_is_the_machines(self, tmp_path: Path) -> None:
        machine = tmp_path / "machine"
        machine.mkdir()
        held = _bytes(3 * CHUNK + 12345)
        (machine / "pack.h5").write_bytes(held)
        transport = FakeTransport(machine)
        said: list[str] = []
        record = fetch_file(
            None,
            "pack.h5",
            tmp_path / "home" / "pack.h5",
            chunk_bytes=CHUNK,
            transport=transport,
            pace=_pace(),
            say=said.append,
        )
        assert (tmp_path / "home" / "pack.h5").read_bytes() == held
        assert record["sha256"] == hashlib.sha256(held).hexdigest()
        assert record["chunks"] == 4 and record["failures"] == 0
        assert sorted(chunk for chunk, _ in transport.reads) == [0, 1, 2, 3]
        # Nothing of the transfer is left beside the file but the note of which file of the
        # machine it is, and the connections are closed.
        assert sorted(p.name for p in (tmp_path / "home").iterdir()) == [
            "pack.h5",
            "pack.h5.home.json",
        ]
        assert transport.closed == 1
        assert "4 chunks" in said[0] and "verified" in said[-1]

    def test_a_stream_that_drops_costs_its_chunk_and_slows_every_worker(
        self, tmp_path: Path
    ) -> None:
        machine = tmp_path / "machine"
        machine.mkdir()
        held = _bytes(4 * CHUNK)
        (machine / "pack.h5").write_bytes(held)
        transport = FakeTransport(machine)
        transport.fail = {1: 3}
        transport.short = {2}
        pauses: list[float] = []
        pace = _pace(pauses)
        record = fetch_file(
            None,
            "pack.h5",
            tmp_path / "pack.h5",
            chunk_bytes=CHUNK,
            workers=2,
            transport=transport,
            pace=pace,
        )
        assert (tmp_path / "pack.h5").read_bytes() == held
        # Chunk 1 was asked for four times and chunk 2 twice; the others once.
        asked = [chunk for chunk, _ in transport.reads]
        assert asked.count(1) == 4 and asked.count(2) == 2
        assert asked.count(0) == 1 and asked.count(3) == 1
        assert record["failures"] == 4
        # The pause grew with the failures that followed one another.
        assert pace.longest_s >= 2 * pace.pause_s
        assert pauses

    def test_a_run_that_was_cut_goes_on_from_the_chunks_it_had(self, tmp_path: Path) -> None:
        machine = tmp_path / "machine"
        machine.mkdir()
        held = _bytes(5 * CHUNK + 99)
        (machine / "pack.h5").write_bytes(held)
        first = FakeTransport(machine)
        first.fail = {3: 1000}
        with pytest.raises(ConnectionLost, match="in a row"):
            fetch_file(
                None,
                "pack.h5",
                tmp_path / "pack.h5",
                chunk_bytes=CHUNK,
                workers=1,
                transport=first,
                pace=_pace(),
            )
        assert not (tmp_path / "pack.h5").exists()
        ledger = json.loads((tmp_path / "pack.h5.chunks.json").read_text())
        assert ledger["done"] == [0, 1, 2]
        second = FakeTransport(machine)
        record = fetch_file(
            None,
            "pack.h5",
            tmp_path / "pack.h5",
            chunk_bytes=CHUNK,
            workers=1,
            transport=second,
            pace=_pace(),
        )
        assert sorted(chunk for chunk, _ in second.reads) == [3, 4, 5]
        assert record["chunks_resumed"] == 3
        assert (tmp_path / "pack.h5").read_bytes() == held

    def test_a_chunk_that_arrived_wrong_is_found_by_the_digest_and_fetched_again(
        self, tmp_path: Path
    ) -> None:
        machine = tmp_path / "machine"
        machine.mkdir()
        held = _bytes(3 * CHUNK, seed=4)
        (machine / "pack.h5").write_bytes(held)
        transport = FakeTransport(machine)
        transport.corrupt = {1}
        said: list[str] = []
        record = fetch_file(
            None,
            "pack.h5",
            tmp_path / "pack.h5",
            chunk_bytes=CHUNK,
            transport=transport,
            pace=_pace(),
            say=said.append,
        )
        assert (tmp_path / "pack.h5").read_bytes() == held
        assert record["chunks_refetched"] == 1
        assert [chunk for chunk, _ in transport.reads].count(1) == 2
        assert any("1 chunk(s) differ" in line for line in said)

    def test_a_file_of_another_size_starts_again(self, tmp_path: Path) -> None:
        machine = tmp_path / "machine"
        machine.mkdir()
        (machine / "pack.h5").write_bytes(_bytes(2 * CHUNK))
        target = tmp_path / "pack.h5"
        target.with_name("pack.h5.partial").write_bytes(b"old")
        target.with_name("pack.h5.chunks.json").write_text(
            json.dumps({"remote": "pack.h5", "size": 7, "chunk_bytes": CHUNK, "done": [0]})
        )
        transport = FakeTransport(machine)
        fetch_file(None, "pack.h5", target, chunk_bytes=CHUNK, transport=transport, pace=_pace())
        assert sorted(chunk for chunk, _ in transport.reads) == [0, 1]
        assert target.read_bytes() == _bytes(2 * CHUNK)

    def test_a_chunk_is_a_whole_number_of_blocks(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="blocks"):
            fetch_file(
                None, "x", tmp_path / "x", chunk_bytes=1000, transport=FakeTransport(tmp_path)
            )


class TestFetchTree:
    def _machine(self, tmp_path: Path, files: int = 7) -> Path:
        machine = tmp_path / "machine"
        for k in range(files):
            path = machine / "pairs" / "grid" / f"{k % 3:02d}" / f"{k:04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_bytes(1000 + k, seed=k))
        (machine / "pairs" / "grid" / "index.jsonl").write_text("{}\n" * files)
        (machine / "pairs" / "grid" / "00" / "x.partial.npz").write_bytes(b"half")
        return machine

    def test_what_is_missing_comes_in_batches_and_what_is_home_is_not_asked_again(
        self, tmp_path: Path
    ) -> None:
        machine = self._machine(tmp_path)
        home = tmp_path / "home" / "pairs"
        transport = FakeTransport(machine)
        record = fetch_tree(
            None,
            "pairs",
            home,
            exclude=("*.partial.*",),
            batch_bytes=2500,
            transport=transport,
            pace=_pace(),
        )
        assert record["complete"] and record["files"] == 8 and record["left"] == 0
        assert record["there"] == 8
        assert len(transport.batches) >= 3
        assert not list(home.rglob("*.partial.*")) and not list(home.glob(".incoming*"))
        for path in (machine / "pairs").rglob("*.npz"):
            if "partial" not in path.name:
                assert (
                    home / path.relative_to(machine / "pairs")
                ).read_bytes() == path.read_bytes()
        # A file that grew on the machine (the index) is asked for again, and nothing else.
        (machine / "pairs" / "grid" / "index.jsonl").write_text("{}\n" * 9)
        again = FakeTransport(machine)
        record = fetch_tree(
            None, "pairs", home, exclude=("*.partial.*",), transport=again, pace=_pace()
        )
        assert again.batches == [["grid/index.jsonl"]] and record["files"] == 1

    def test_a_stream_that_is_cut_keeps_the_files_it_had_brought(self, tmp_path: Path) -> None:
        machine = self._machine(tmp_path)
        home = tmp_path / "home" / "pairs"
        transport = FakeTransport(machine)
        transport.drop_after = 2
        pace = _pace()
        record = fetch_tree(
            None,
            "pairs",
            home,
            exclude=("*.partial.*",),
            workers=1,
            transport=transport,
            pace=pace,
        )
        assert record["complete"] and record["failures"] == 1
        # The second stream asked for what the first had not brought whole, and no more.
        assert len(transport.batches) == 2
        assert len(transport.batches[1]) == len(transport.batches[0]) - 2

    def test_a_file_that_grew_is_kept_and_one_that_never_comes_whole_is_left(
        self, tmp_path: Path
    ) -> None:
        machine = self._machine(tmp_path, files=2)
        home = tmp_path / "home" / "pairs"
        index = machine / "pairs" / "grid" / "index.jsonl"
        index.write_text('{"key": "old"}\n')

        class Growing(FakeTransport):
            def files(self, remote_dir: str, names: Any, target: Path, *a: Any) -> None:
                # The campaign writes its index while the batch is read.
                index.write_text(index.read_text() + '{"key": "new"}\n{"key": "ha')
                super().files(remote_dir, names, target, *a)

        record = fetch_tree(
            None, "pairs", home, exclude=("*.partial.*",), transport=Growing(machine), pace=_pace()
        )
        assert record["complete"] and (home / "grid" / "index.jsonl").stat().st_size > 16
        # Half a line at its end is not a pair's record.
        assert set(PairCache(home, "grid").records()) == {"old", "new"}

        class Cut(FakeTransport):
            def files(self, remote_dir: str, names: Any, target: Path, *a: Any) -> None:
                self.batches.append(list(names))
                for name in names:
                    (target / name).parent.mkdir(parents=True, exist_ok=True)
                    (target / name).write_bytes(b"cut")

        cut = Cut(machine)
        record = fetch_tree(None, "pairs", tmp_path / "other", transport=cut, pace=_pace())
        # Three streams a file, then the pass ends without it: the next pass asks again.
        assert not record["complete"] and record["left"] == 4 and record["files"] == 0
        assert len(cut.batches) == 3

    def test_a_pass_ends_at_its_time_and_is_not_a_failure(self, tmp_path: Path) -> None:
        machine = self._machine(tmp_path)
        home = tmp_path / "home" / "pairs"
        record = fetch_tree(
            None,
            "pairs",
            home,
            exclude=("*.partial.*",),
            seconds=0.0,
            transport=FakeTransport(machine),
            pace=_pace(),
        )
        assert not record["complete"] and record["left"] == 8 and record["files"] == 0
        assert "error" not in record

    def test_a_directory_that_is_not_there_is_said_and_not_raised(self, tmp_path: Path) -> None:
        record = fetch_tree(
            None, "pairs", tmp_path / "home", transport=FakeTransport(tmp_path), pace=_pace()
        )
        assert not record["complete"] and "No such file" in record["error"]


class TestPace:
    def test_the_pause_doubles_is_capped_and_gives_up(self) -> None:
        pace = Pace(
            pause_s=1.0, cap_s=5.0, give_up_after=6, clock=lambda: 0.0, sleep=lambda s: None
        )
        assert [pace.failed() for _ in range(5)] == [1.0, 2.0, 4.0, 5.0, 5.0]
        pace.succeeded()
        # A success halves what the next failure waits, and the row starts again.
        assert pace.failed() == 4.0
        for _ in range(5):
            pace.failed()
        with pytest.raises(ConnectionLost, match="7 times in a row"):
            pace.failed("Connection refused")


class TestTheConnection:
    def test_a_worker_keeps_one_connection_for_all_its_chunks(self) -> None:
        machine = Machine("ssh5.vast.ai", 12345, identity=Path("/keys/id"))
        transport: Any = SshTransport(machine)
        first, again, other = (transport._machine(w) for w in (0, 0, 1))
        assert first is again and first.control != other.control
        argv = first.ssh_command("true")
        assert "ControlMaster=auto" in argv and f"ControlPath={first.control}" in argv
        assert "ConnectTimeout=30" in argv
        # The path of a socket is bounded near a hundred characters.
        assert len(str(first.control).replace("%C", "c" * 40)) < 100
        assert machine.ssh_command("true").count("ControlMaster=auto") == 0
        assert first.close_command() is not None and machine.close_command() is None

    def test_the_direct_address_is_taken_where_it_answers_and_the_proxy_otherwise(self) -> None:
        proxied = Machine("ssh5.vast.ai", 12345)
        assert fastest(proxied) is proxied
        both = Machine("ssh5.vast.ai", 12345, direct=("203.0.113.7", 40022))
        said: list[str] = []
        direct = fastest(both, say=said.append, probe=lambda machine: "")
        assert (direct.host, direct.port, direct.direct) == ("203.0.113.7", 40022, None)
        assert "past the proxy" in said[0]

        def refused(machine: Machine) -> str:
            raise ConnectionLost("Connection refused", 255)

        assert fastest(both, say=said.append, probe=refused) is both
        assert "the proxy" in said[1]


@dataclass(frozen=True)
class Here(Machine):
    """This machine standing in for a rented one: the command, in a shell, with no ssh."""

    def ssh_command(self, remote: str) -> list[str]:
        return ["sh", "-c", remote]

    def close_command(self) -> list[str] | None:
        return None


@pytest.mark.skipif(
    shutil.which("dd") is None or shutil.which("tar") is None, reason="no dd or no tar"
)
class TestTheCommandsThemselves:
    """``dd`` for a range and ``tar`` for a batch, as the machine runs them, in a real shell."""

    def test_a_range_is_read_by_dd_and_a_batch_by_tar(self, tmp_path: Path) -> None:
        there = tmp_path / "machine"
        (there / "pairs" / "grid" / "aa").mkdir(parents=True)
        held = _bytes(2 * BLOCK_BYTES + 4321, seed=9)
        (there / "pack.h5").write_bytes(held)
        transport = SshTransport(Here("here"))
        # The last range of a file is short: ``dd`` stops where the file does.
        transport.read(str(there / "pack.h5"), BLOCK_BYTES, BLOCK_BYTES, tmp_path / "one", 0)
        assert (tmp_path / "one").read_bytes() == held[BLOCK_BYTES : 2 * BLOCK_BYTES]
        transport.read(str(there / "pack.h5"), 2 * BLOCK_BYTES, 4321, tmp_path / "last", 1)
        assert (tmp_path / "last").read_bytes() == held[2 * BLOCK_BYTES :]
        with pytest.raises(ValueError, match="starts on a block"):
            transport.read(str(there / "pack.h5"), 7, 10, tmp_path / "x", 0)
        names = ["grid/aa/one.npz", "grid/aa/two with a space.npz", "grid/index.jsonl"]
        for k, name in enumerate(names):
            (there / "pairs" / name).write_bytes(_bytes(3000 + k, seed=k))
        transport.files(str(there / "pairs"), names, tmp_path / "in", 0, 30.0)
        for name in names:
            assert (tmp_path / "in" / name).read_bytes() == (there / "pairs" / name).read_bytes()
        # A whole fetch over the same shell: the chunks, the machine's digest left to a fake.
        whole = SshTransport(Here("here"))
        whole.size = lambda remote: len(held)  # type: ignore[method-assign]
        whole.sha256 = lambda remote: hashlib.sha256(held).hexdigest()  # type: ignore[method-assign]
        record = fetch_file(
            None,
            str(there / "pack.h5"),
            tmp_path / "home" / "pack.h5",
            chunk_bytes=BLOCK_BYTES,
            workers=2,
            transport=whole,
            pace=_pace(),
        )
        assert (tmp_path / "home" / "pack.h5").read_bytes() == held and record["chunks"] == 3
        transport.close()
