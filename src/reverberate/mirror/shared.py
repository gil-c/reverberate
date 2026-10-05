"""What the processes of a machine build once between them: arrays on disk, named by their making.

Every worker of a trace prepared the mirror for itself and grew again the
image trees, the short lists and the distance fields another worker
already had: the early trace's work grew with its workers (2.5 times one
worker's at sixteen, ``docs/adr/0016-appendix-every-card-every-core.md``).
A :class:`Store` is a directory in which each of those is an entry named by
a digest of what it was made from, so that whoever needs one finds it or
makes it, and nobody makes it twice:

- an entry is a directory of ``.npy`` files, written under another name and
  renamed, so it is there whole or not at all, and two processes that make
  the same entry at once leave one of them;
- an entry that costs is made under a lock of its own name (:meth:`Store.make`):
  a process that asks while another makes it waits, then maps what was made;
- an entry is read by mapping its files: the page cache holds one copy for
  every process of the machine, and a tree of thirty megabytes costs a
  worker nothing it does not touch;
- a name holds everything the arrays depend on (the scene's key, the
  settings, the anchor, the region), so a resumed run finds its entries and
  a second scene of the same dwelling finds those that do not depend on the
  scene's positions. Nothing is ever invalidated: what would be stale has
  another name.

The store holds no result of a trace, only what a trace derives on its way:
deleting it costs time and changes nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Not on every system; without it two processes may make one entry at once, and one stays.
    import fcntl
except ImportError:  # pragma: no cover - a system without POSIX locks
    fcntl = None  # type: ignore[assignment]

import numpy as np

__all__ = ["MAPPED_FROM_BYTES", "Store", "name_of"]

#: A file is mapped from this size and read whole under it: a map costs a system call
#: or two more than a read, which is all a small array costs.
MAPPED_FROM_BYTES = 1 << 16


def name_of(*parts: Any) -> str:
    """An entry's name: a digest of what it was made from, arrays by their bytes."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str(part.dtype).encode())
            digest.update(str(part.shape).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
        digest.update(b"|")
    return digest.hexdigest()[:32]


@dataclass
class Store:
    """A directory of entries, each a set of arrays under a kind and a name."""

    root: Path
    #: Entries found and entries made by this process, by kind: what a run reports.
    found: dict[str, int] = field(default_factory=dict)
    made: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def path(self, kind: str, name: str) -> Path:
        return self.root / kind / name[:2] / name

    def has(self, kind: str, name: str) -> bool:
        return self.path(kind, name).is_dir()

    def load(self, kind: str, name: str) -> dict[str, np.ndarray] | None:
        """The entry's arrays, read only, the large ones mapped; ``None`` when it is not there."""
        held = self.path(kind, name)
        if not held.is_dir():
            return None
        arrays: dict[str, np.ndarray] = {}
        try:
            for file in sorted(held.glob("*.npy")):
                mapped = file.stat().st_size >= MAPPED_FROM_BYTES
                arrays[file.stem] = np.load(file, mmap_mode="r" if mapped else None)
        except (OSError, ValueError):
            # An entry is renamed into place whole; one that does not read was damaged
            # afterwards, and is made again under another process's eyes or ours.
            return None
        self.found[kind] = self.found.get(kind, 0) + 1
        return arrays

    def save(self, kind: str, name: str, arrays: dict[str, Any]) -> None:
        """The entry, whole or not at all. One that is already there is left as it is."""
        target = self.path(kind, name)
        if target.is_dir():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.parent / f".{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        scratch.mkdir()
        try:
            for key, array in arrays.items():
                np.save(scratch / f"{key}.npy", np.asarray(array), allow_pickle=False)
            try:
                scratch.rename(target)
            except OSError:
                # Another process made the same entry first: the same arrays, by its name.
                if not target.is_dir():
                    raise
            else:
                self.made[kind] = self.made.get(kind, 0) + 1
        finally:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)

    def claim(self, kind: str, name: str, wait: bool = False) -> Any:
        """The entry's making, taken for this process: a handle to ``close``, or ``None``.

        ``None`` says another process is making the entry now; with ``wait``
        the call returns when that process is done, or is gone. A claim is
        let go when its handle is closed and when its process dies.
        """
        target = self.path(kind, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = (target.parent / f".{name}.lock").open("w")
        if fcntl is None:
            return handle
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    @contextlib.contextmanager
    def _alone(self, kind: str, name: str) -> Iterator[None]:
        """This process alone among those that would make the entry; let go when it dies."""
        handle = self.claim(kind, name, wait=True)
        try:
            yield
        finally:
            handle.close()

    def make(
        self, kind: str, name: str, build: Callable[[], dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        """The entry, made by ``build`` in one process of the machine and read by the others.

        Returns the arrays and whether this process built them: what it
        built it has in memory, and hands back as it made it.
        """
        found = self.load(kind, name)
        if found is not None:
            return found, False
        with self._alone(kind, name):
            found = self.load(kind, name)
            if found is not None:
                return found, False
            arrays = build()
            self.save(kind, name, arrays)
            return arrays, True

    def record(self) -> dict[str, Any]:
        return {"root": str(self.root), "found": dict(self.found), "made": dict(self.made)}
