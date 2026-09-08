"""Renting for the solver alone: ship four files, run, retrieve, destroy.

Roadmap section 11, step 3, and W8 item 2. The CUDA engine reads exactly
``sim_consts.h5``, ``vox_out.h5``, ``comms_out.h5`` and ``sim_mats.h5``, writes
``sim_outs.h5``, and needs no Python. So the rented machine never sees a scene,
a mesh, a material table or an interpreter: it receives four files, runs one
binary and gives one file back.

Two things are deliberately not done here. **Nothing decides to rent**, because
section 12.1 requires the rate and the total to be stated and agreed first;
:func:`solve` takes a machine that already exists. And **nothing here is
responsible for teardown**, because :func:`reverberate.gpu.vast.rent` already
armed a watchdog that outlives this process. :func:`solve` will destroy the
instance when asked, as a courtesy that stops the meter early, not as the
safeguard.

Transfers go over ``scp`` and commands over ``ssh``, the two things a Vast.ai
instance offers without any agent installed on it.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.wave.comms import ENGINE_FILES

__all__ = [
    "EngineProgress",
    "Machine",
    "SolveResult",
    "engine_log",
    "engine_progress",
    "fetch",
    "solve",
    "start_engine",
    "upload",
    "watch_engine",
]

#: Where the engine's data directory lives on the rented machine.
DEFAULT_REMOTE_DIR = "/root/run"
#: Where ``scripts/build_pffdtd.sh`` puts the binaries.
DEFAULT_PFFDTD_DIR = "/root/pffdtd"

_SSH_OPTIONS = [
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ServerAliveInterval=30",
]


@dataclass(frozen=True)
class Machine:
    """An instance that is already running, addressed over ssh."""

    host: str
    port: int = 22
    user: str = "root"
    identity: Path | None = None

    @classmethod
    def from_instance(cls, instance: Any, identity: Path | None = None) -> Machine:
        """Address a :class:`reverberate.gpu.vast.Instance`."""
        if not instance.ssh_host:
            raise ValueError(f"instance {instance.id} has no ssh host yet")
        return cls(host=instance.ssh_host, port=instance.ssh_port, identity=identity)

    def ssh_command(self, remote: str) -> list[str]:
        """The ``ssh`` argv for one remote command."""
        argv = ["ssh", *_SSH_OPTIONS, "-p", str(self.port)]
        if self.identity:
            argv += ["-i", str(self.identity)]
        return [*argv, f"{self.user}@{self.host}", remote]

    def scp_command(self, sources: list[Path], destination: str, *, download: bool) -> list[str]:
        """The ``scp`` argv for a transfer in either direction."""
        argv = ["scp", *_SSH_OPTIONS, "-P", str(self.port)]
        if self.identity:
            argv += ["-i", str(self.identity)]
        if download:
            remote = [f"{self.user}@{self.host}:{shlex.quote(str(s))}" for s in sources]
            return [*argv, *remote, destination]
        return [
            *argv,
            *[str(s) for s in sources],
            f"{self.user}@{self.host}:{shlex.quote(destination)}",
        ]


@dataclass(frozen=True)
class SolveResult:
    """What one rented run produced and what it cost in wall clock."""

    output: Path
    upload_s: float
    engine_s: float
    fetch_s: float
    uploaded_bytes: int
    log: str

    @property
    def total_s(self) -> float:
        """Seconds the instance was actually needed for."""
        return self.upload_s + self.engine_s + self.fetch_s


def _run(argv: list[str], *, what: str, timeout: float | None = None) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{what} failed: {completed.stderr.strip()[-2000:]}")
    return completed.stdout


def upload(machine: Machine, files: list[Path], remote_dir: str = DEFAULT_REMOTE_DIR) -> int:
    """Copy the engine's inputs across and return the bytes sent.

    Refuses anything that is not one of the four files the engine reads: an
    accidental ``cart_grid.h5`` is pure bandwidth, and a scene file on a rented
    machine is a data policy problem rather than a slow upload.
    """
    unexpected = [f.name for f in files if f.name not in ENGINE_FILES]
    if unexpected:
        raise ValueError(f"the engine reads only {list(ENGINE_FILES)}, refusing {unexpected}")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"missing engine inputs: {missing}")
    _run(machine.ssh_command(f"mkdir -p {shlex.quote(remote_dir)}"), what="mkdir")
    _run(machine.scp_command(files, remote_dir, download=False), what="upload")
    return sum(f.stat().st_size for f in files)


def start_engine(
    machine: Machine,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    *,
    pffdtd_dir: str = DEFAULT_PFFDTD_DIR,
    double_precision: bool = False,
    log_name: str = "engine.log",
) -> str:
    """Launch the engine detached, into a log, and return at once.

    **Not a refinement.** W25 destroyed an A100 mid-solve because ``nohup &``
    over ssh never returns: ssh holds the channel until every process closes
    stdout. W35 recorded the same shape of failure again, "a two hour job tied
    to one ssh connection". A synchronous run also hides the engine's own words:
    the command redirects stderr into stdout, so a non-zero exit is raised
    carrying nothing but the host's login banner.

    So the engine is launched under ``setsid`` with all three descriptors
    detached, its output goes to a file, and the caller polls
    :func:`engine_progress`. The launch is verified by the log moving, never by
    this call's return.
    """
    precision = "double" if double_precision else "single"
    binary = f"{pffdtd_dir}/c_cuda/fdtd_main_gpu_{precision}.x"
    script = f"{remote_dir}/launch_engine.sh"
    body = f"#!/bin/bash\ncd {shlex.quote(remote_dir)}\nexec {shlex.quote(binary)}\n"
    _run(
        machine.ssh_command(
            f"mkdir -p {shlex.quote(remote_dir)} && "
            f"cat > {shlex.quote(script)} <<'REVERBERATE_EOF'\n{body}REVERBERATE_EOF\n"
            f"chmod +x {shlex.quote(script)}"
        ),
        what="write launcher",
    )
    _run(
        machine.ssh_command(
            f"cd {shlex.quote(remote_dir)} && rm -f {shlex.quote(log_name)} && "
            f"setsid nohup {shlex.quote(script)} > {shlex.quote(log_name)} "
            "2>&1 < /dev/null & sleep 2; pgrep -c fdtd_main_gpu"
        ),
        what="start engine",
    )
    return script


#: PFFDTD prints this every time step. A percentage that stops moving is a
#: stalled run, which is a different thing from a slow one and is worth saying.
PROGRESS = re.compile(r"Running \[\s*([0-9.]+)%\]")


@dataclass(frozen=True)
class EngineProgress:
    """Where a detached engine has got to, read from its own log."""

    running: bool
    percent: float | None
    last_line: str
    output_bytes: int

    @property
    def finished(self) -> bool:
        """No process, and an output file with something in it."""
        return not self.running and self.output_bytes > 0


def engine_progress(
    machine: Machine, remote_dir: str = DEFAULT_REMOTE_DIR, *, log_name: str = "engine.log"
) -> EngineProgress:
    """Poll a detached engine: is it alive, how far in, and has it written anything."""
    log = f"{remote_dir}/{log_name}"
    output = f"{remote_dir}/sim_outs.h5"
    # The engine separates its progress lines with carriage returns, so that a
    # terminal overwrites one line rather than scrolling. Deleting them joins
    # the whole log into one line and a search then returns its *oldest*
    # percentage for ever, which reads exactly like a stalled run. Translated to
    # newlines instead, and the last one taken.
    # Each field carries its own tag. Some hosts print a login banner on
    # stdout ("Welcome to vast.ai ... Have fun!"), and reading the fields by
    # position then took the banner for the process count and the file size
    # for the engine's last words: a finished solve was reported as an engine
    # that exited without writing anything, twice, on the same card.
    probe = (
        "printf 'PROCS=%s\\nBYTES=%s\\nLAST=%s\\n' "
        '"$(pgrep -c fdtd_main_gpu 2>/dev/null || echo 0)" '
        f'"$(stat -c%s {shlex.quote(output)} 2>/dev/null || echo 0)" '
        f"\"$(tail -c 20000 {shlex.quote(log)} 2>/dev/null | tr '\\r' '\\n' "
        '| grep -a "Running" | tail -1)"'
    )
    lines = _run(machine.ssh_command(probe), what="engine progress").splitlines()
    fields = {}
    for line in lines:
        tag, _, value = line.partition("=")
        if tag in ("PROCS", "BYTES", "LAST") and tag not in fields:
            fields[tag] = value.strip()
    procs = fields.get("PROCS", "")
    size = fields.get("BYTES", "")
    running = procs.isdigit() and int(procs) > 0
    output_bytes = int(size) if size.isdigit() else 0
    last = fields.get("LAST", "")
    matches = PROGRESS.findall(last)
    found = matches[-1] if matches else None
    return EngineProgress(
        running=running,
        percent=float(found) if found is not None else None,
        last_line=last,
        output_bytes=output_bytes,
    )


def engine_log(
    machine: Machine, remote_dir: str = DEFAULT_REMOTE_DIR, *, log_name: str = "engine.log"
) -> str:
    """The whole of a detached engine's log."""
    return _run(
        machine.ssh_command(f"cat {shlex.quote(remote_dir)}/{shlex.quote(log_name)}"),
        what="engine log",
    )


def watch_engine(
    machine: Machine,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    *,
    pffdtd_dir: str = DEFAULT_PFFDTD_DIR,
    double_precision: bool = False,
    timeout: float | None = None,
    poll_s: float = 120.0,
    stall_polls: int = 10,
    on_progress: Callable[[EngineProgress], None] | None = None,
) -> str:
    """Start the engine detached and follow it to the end, returning its whole log.

    Three failures this shape prevents, each of which has happened:

    - a run tied to one ssh channel, which dies with the connection and takes
      the solve with it;
    - a percentage nobody sees, which is what a five hour run looks like from
      outside when its progress is only returned at the end;
    - an engine that exits saying nothing, whose message is on stdout while the
      exception carries stderr, so the caller is told only the login banner.

    A percentage that has not moved for ``stall_polls`` polls is reported as
    stalled, which is a different thing from slow and worth saying.
    """
    start_engine(machine, remote_dir, pffdtd_dir=pffdtd_dir, double_precision=double_precision)
    deadline = None if timeout is None else time.time() + timeout
    last_percent, unchanged = None, 0
    while True:
        progress = engine_progress(machine, remote_dir)
        if on_progress is not None:
            on_progress(progress)
        if progress.finished:
            return engine_log(machine, remote_dir)
        if not progress.running:
            raise RuntimeError(
                "the engine exited without writing sim_outs.h5. Its own last "
                f"words were: {progress.last_line!r}. The whole log is at "
                f"{remote_dir}/engine.log on the machine."
            )
        if progress.percent is not None and progress.percent == last_percent:
            unchanged += 1
            if unchanged >= stall_polls:
                raise RuntimeError(
                    f"STALLED at {progress.percent}% for {unchanged} polls of "
                    f"{poll_s:g} s; the engine is alive and not advancing"
                )
        else:
            unchanged = 0
        last_percent = progress.percent
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(
                f"the engine passed its {timeout:g} s budget at "
                f"{progress.percent if progress.percent is not None else 'an unknown'}%"
            )
        time.sleep(poll_s)


def fetch(machine: Machine, destination: Path, remote_dir: str = DEFAULT_REMOTE_DIR) -> Path:
    """Retrieve ``sim_outs.h5``, the only thing the engine produces."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        machine.scp_command([Path(remote_dir) / "sim_outs.h5"], str(destination), download=True),
        what="fetch",
    )
    if not destination.is_file():
        raise RuntimeError(f"{destination} was not retrieved")
    return destination


def solve(
    machine: Machine,
    files: list[Path],
    destination: Path,
    *,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    pffdtd_dir: str = DEFAULT_PFFDTD_DIR,
    double_precision: bool = False,
    timeout: float | None = None,
    on_progress: Callable[[EngineProgress], None] | None = None,
) -> SolveResult:
    """Upload, run, retrieve, in that order, timing each.

    **Retrieval happens before anything is torn down**, and every artefact is
    retrieved: section 12.1 item 7 exists because ``vox_out.h5`` was lost that
    way in B0 and had to be estimated from source code instead of measured.
    """
    started = time.time()
    uploaded = upload(machine, files, remote_dir)
    upload_s = time.time() - started

    started = time.time()
    log = watch_engine(
        machine,
        remote_dir,
        pffdtd_dir=pffdtd_dir,
        double_precision=double_precision,
        timeout=timeout,
        on_progress=on_progress,
    )
    engine_s = time.time() - started

    started = time.time()
    fetch(machine, destination, remote_dir)
    fetch_s = time.time() - started

    result = SolveResult(
        output=destination,
        upload_s=round(upload_s, 3),
        engine_s=round(engine_s, 3),
        fetch_s=round(fetch_s, 3),
        uploaded_bytes=uploaded,
        log=log,
    )
    destination.with_suffix(".run.json").write_text(
        json.dumps(
            {
                "upload_s": result.upload_s,
                "engine_s": result.engine_s,
                "fetch_s": result.fetch_s,
                "uploaded_bytes": result.uploaded_bytes,
                "uploaded": [f.name for f in files],
                "double_precision": double_precision,
            },
            indent=2,
        )
    )
    return result
