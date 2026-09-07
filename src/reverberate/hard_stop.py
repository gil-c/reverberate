"""A cap on a local job that the job cannot talk its way out of.

`signal.alarm` is not one, and this module exists because that was learned the
expensive way. A `SIGALRM` handler runs between bytecodes, so a single long
NumPy call -- a sort of a billion keys, a merge over a room -- holds the
interpreter for its whole duration and the alarm waits behind it. An audit build
wrapped in `signal.alarm(1800)` ran for **7 h 40 on a laptop overnight**, at a
hundred per cent of one core, and the alarm never fired.

So the cap lives in another process, it sends `SIGKILL`, and it waits against an
absolute wall-clock deadline rather than sleeping for a duration. That last part
is W22's lesson, recorded in :func:`reverberate.gpu.vast.arm_hard_stop` for
rented instances: a `sleep` does not advance while the machine is suspended, so
a laptop that closes its lid postpones the deadline by exactly as long as it
sleeps. Polled against `time.time`, a suspend costs one interval and no more.

Usage::

    with hard_stop.capped(minutes=15):
        build_the_thing()

The process is killed outright at the deadline. That is the point: a cap that
can be caught, handled or delayed is a request, and this is not a request.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = ["POLL_S", "Watchdog", "arm", "capped", "disarm", "main"]

#: How often the watchdog compares the clock against the deadline. Short enough
#: that a suspend costs little, long enough to be free.
POLL_S = 5.0


@dataclass(frozen=True)
class Watchdog:
    """A running watchdog, and the handle that lets it be reaped.

    The handle is the point. A watchdog is a child process, and a child that
    has exited stays a zombie until someone waits on it -- so ``os.kill(pid, 0)``
    still succeeds and a caller asking "is it still watching" is told yes
    forever. Keeping the ``Popen`` is what makes :meth:`alive` mean what it
    says, and what stops a long-lived process leaving one zombie per cap.
    """

    pid: int
    process: subprocess.Popen[bytes]

    def alive(self) -> bool:
        """True while the watchdog is still watching, reaping it if it is not."""
        return self.process.poll() is None


def arm(pid: int, deadline: float, poll_s: float = POLL_S) -> Watchdog:
    """Spawn a detached watchdog that kills ``pid`` at ``deadline``.

    ``deadline`` is an absolute unix timestamp, never a duration.

    Detached with ``start_new_session`` so it outlives the shell that started
    the job: a cap that dies with its terminal is not a cap.
    """
    command = [
        sys.executable,
        "-m",
        "reverberate.hard_stop",
        str(pid),
        repr(float(deadline)),
        repr(float(poll_s)),
    ]
    child = subprocess.Popen(  # noqa: S603 - our own module, no shell
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    return Watchdog(pid=int(child.pid), process=child)


def disarm(watchdog: Watchdog) -> None:
    """Stop a watchdog whose job finished in time, and reap it. Never raises."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        watchdog.process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        watchdog.process.wait(timeout=5)


@contextmanager
def capped(seconds: float = 0.0, minutes: float = 0.0, poll_s: float = POLL_S) -> Iterator[float]:
    """Run the block under a wall-clock cap enforced from outside this process.

    Yields the deadline. On leaving the block the watchdog is disarmed, so a
    job that finishes early costs nothing.
    """
    total = float(seconds) + 60.0 * float(minutes)
    if total <= 0:
        raise ValueError("a cap needs a positive duration")
    deadline = time.time() + total
    watchdog = arm(os.getpid(), deadline, poll_s)
    try:
        yield deadline
    finally:
        disarm(watchdog)


def _watch(pid: int, deadline: float, poll_s: float) -> int:
    """Watchdog body: poll the wall clock, then kill.

    Exits early and quietly if the job finishes first, which is the common
    case: ``os.kill(pid, 0)`` raises once the process is gone.
    """
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 0  # the job finished on its own
        except PermissionError:
            pass  # alive, and not ours to signal; keep waiting anyway
        time.sleep(min(poll_s, max(remaining, 0.01)))
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        raise SystemExit("usage: python -m reverberate.hard_stop <pid> <deadline> [poll_s]")
    poll = float(args[2]) if len(args) == 3 else POLL_S
    return _watch(int(args[0]), float(args[1]), poll)


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
