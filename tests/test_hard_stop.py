"""Tests for the cap a local job cannot talk its way out of.

The test that matters is the last one. `signal.alarm` was the cap before this
module, and it failed in the one way that costs something: a long NumPy call
holds the interpreter, the handler waits behind it, and a build meant to stop
after thirty minutes ran for 7 h 40 overnight. So the cap is asserted against a
process that ignores every signal it can, and does so inside a C call.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reverberate import hard_stop

SRC = str(Path(__file__).resolve().parents[1] / "src")


def spawn(body: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.Popen([sys.executable, "-c", body], env=env)  # noqa: S603


def wait_gone(process: subprocess.Popen[bytes], timeout: float = 20.0) -> bool:
    limit = time.time() + timeout
    while time.time() < limit:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.slow
def test_it_kills_a_job_that_ignores_every_signal_it_can() -> None:
    """The defect this module exists for.

    The job traps SIGTERM and SIGALRM and then spends its life inside one call
    that does not return, so nothing delivered *to* it can stop it. Only a
    kill from outside can, and only SIGKILL, which cannot be trapped.
    """
    job = spawn(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: None)\n"
        "signal.signal(signal.SIGALRM, lambda *a: None)\n"
        "signal.alarm(1)\n"
        "time.sleep(300)\n"
    )
    watchdog = hard_stop.arm(job.pid, time.time() + 1.0, poll_s=0.1)
    try:
        assert wait_gone(job), "the watchdog did not kill a job that ignores signals"
        assert job.returncode == -signal.SIGKILL
    finally:
        job.kill()
        hard_stop.disarm(watchdog)


@pytest.mark.slow
def test_a_job_that_finishes_in_time_is_left_alone() -> None:
    """A cap that killed a job which finished early would be worse than none."""
    with hard_stop.capped(seconds=30, poll_s=0.1) as deadline:
        assert deadline > time.time()
        result = 2 + 2
    assert result == 4


@pytest.mark.slow
def test_the_watchdog_stops_when_its_job_does() -> None:
    """It must not linger and kill a pid the operating system has reused."""
    job = spawn("import time; time.sleep(0.2)")
    watchdog = hard_stop.arm(job.pid, time.time() + 60.0, poll_s=0.05)
    job.wait(timeout=10)
    limit = time.time() + 10
    while time.time() < limit and watchdog.alive():
        time.sleep(0.05)
    alive = watchdog.alive()
    hard_stop.disarm(watchdog)
    assert not alive, "the watchdog outlived the job it was watching"


def test_a_cap_needs_a_duration() -> None:
    with pytest.raises(ValueError, match="positive duration"), hard_stop.capped(seconds=0):
        pass


def test_disarming_a_watchdog_twice_is_quiet() -> None:
    watchdog = hard_stop.arm(os.getpid(), time.time() + 600.0, poll_s=0.05)
    hard_stop.disarm(watchdog)
    hard_stop.disarm(watchdog)  # must not raise
    assert not watchdog.alive()


def test_liveness_is_asked_of_the_handle_and_not_of_the_pid() -> None:
    """A watchdog that has exited stays a zombie until someone waits on it, and
    ``os.kill(pid, 0)`` answers yes for a zombie. Asked that way, a finished
    watchdog reads as still watching forever -- which is what the first version
    of the test above got wrong. The handle polls, which both answers correctly
    and reaps."""
    job = spawn("import time; time.sleep(0.1)")
    watchdog = hard_stop.arm(job.pid, time.time() + 600.0, poll_s=0.05)
    job.wait(timeout=10)
    limit = time.time() + 10
    while time.time() < limit and watchdog.alive():
        time.sleep(0.05)

    assert not watchdog.alive()
    assert watchdog.process.returncode == 0, "it should exit quietly, not be killed"


def test_the_deadline_is_a_timestamp_not_a_duration() -> None:
    """W22's defect: a duration slept through is a duration the machine can
    postpone by suspending. Against a wall-clock deadline already in the past,
    the watchdog must act at once rather than wait out an interval."""
    job = spawn("import time; time.sleep(300)")
    watchdog = hard_stop.arm(job.pid, time.time() - 5.0, poll_s=30.0)
    try:
        assert wait_gone(job, timeout=15.0)
    finally:
        job.kill()
        hard_stop.disarm(watchdog)
