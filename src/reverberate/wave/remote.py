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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reverberate.wave.comms import ENGINE_FILES

__all__ = [
    "ConnectionLost",
    "EngineProgress",
    "Machine",
    "RemoteError",
    "SolveResult",
    "connection_level",
    "engine_log",
    "engine_progress",
    "fetch",
    "one_at_a_time",
    "solve",
    "start_engine",
    "run_on",
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
#: A connection that is kept and used again (``ControlMaster``) closes itself this long
#: after its last command, s.
CONTROL_PERSIST_S = 120
#: A connection that is not answered in this long is a failure and not a wait, s: Vast's
#: proxy let twelve streams wait for its banner and dropped them together (2026-10-05).
CONNECT_TIMEOUT_S = 30


@dataclass(frozen=True)
class Machine:
    """An instance that is already running, addressed over ssh."""

    host: str
    port: int = 22
    user: str = "root"
    identity: Path | None = None
    #: The instance's own address and the port its 22 is mapped to, where it was created
    #: with direct ssh and says them: a way round the proxy, tried first for a transfer.
    direct: tuple[str, int] | None = None
    #: A socket's path (``ControlPath``): commands that name the same one share one
    #: connection, opened by the first and kept :data:`CONTROL_PERSIST_S` after the last.
    control: str | None = None

    @classmethod
    def from_instance(cls, instance: Any, identity: Path | None = None) -> Machine:
        """Address a :class:`reverberate.gpu.vast.Instance`."""
        if not instance.ssh_host:
            raise ValueError(f"instance {instance.id} has no ssh host yet")
        return cls(
            host=instance.ssh_host,
            port=instance.ssh_port,
            identity=identity,
            direct=getattr(instance, "direct", None),
        )

    def directly(self) -> Machine | None:
        """This machine by its own address, past the proxy; ``None`` where it gave none."""
        if self.direct is None:
            return None
        return replace(self, host=self.direct[0], port=int(self.direct[1]), direct=None)

    def sharing(self, name: str, directory: Path | None = None) -> Machine:
        """This machine with every command on one kept connection, the one ``name`` names.

        Each new ``ssh`` is a new connection to the host's ``sshd``, which
        admits ten at once before it starts refusing (``MaxStartups``); a
        transfer in chunks that opens one a chunk is a storm of them. With
        a socket the first command opens the connection and the next ones
        are sessions on it. The path is short on purpose: a socket's is
        bounded near a hundred characters.
        """
        where = Path(directory) if directory is not None else Path.home() / ".ssh"
        return replace(self, control=str(where / f"rv-%C.{name}"))

    def _options(self) -> list[str]:
        options = [*_SSH_OPTIONS, "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}"]
        if self.control:
            options += [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={self.control}",
                "-o",
                f"ControlPersist={CONTROL_PERSIST_S}",
            ]
        return options

    def ssh_command(self, remote: str) -> list[str]:
        """The ``ssh`` argv for one remote command."""
        argv = ["ssh", *self._options(), "-p", str(self.port)]
        if self.identity:
            argv += ["-i", str(self.identity)]
        return [*argv, f"{self.user}@{self.host}", remote]

    def close_command(self) -> list[str] | None:
        """The argv that ends this machine's kept connection; ``None`` where it keeps none."""
        if not self.control:
            return None
        return [
            "ssh",
            "-o",
            f"ControlPath={self.control}",
            "-O",
            "exit",
            "-p",
            str(self.port),
            f"{self.user}@{self.host}",
        ]

    def scp_command(self, sources: list[Path], destination: str, *, download: bool) -> list[str]:
        """The ``scp`` argv for a transfer in either direction."""
        argv = ["scp", *self._options(), "-P", str(self.port)]
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


#: What a failure of the connection looks like in ``ssh``'s, ``scp``'s and ``rsync``'s own
#: words. Vast's proxy refused a connection for a few seconds in the middle of a twenty
#: minute solve and the whole orchestration died on it; the engine kept running, unwatched.
#: On 2026-10-05 three rentals were lost the same way while provisioning ("Connection to
#: ssh6.vast.ai closed by remote host"), which the list did not name.
_TRANSIENT = (
    "Connection refused",
    "Connection closed",
    "closed by remote host",
    "Connection reset",
    "Connection timed out",
    "Operation timed out",
    "mux_client_request_session",
    "Control socket connect",
    "Broken pipe",
    "kex_exchange",
    "ssh_exchange_identification",
    "client_loop: send disconnect",
    "Timeout, server",
    "No route to host",
    "Network is unreachable",
    "Could not resolve hostname",
)
#: What a host prints on every login, which is no word of the command's: a failure that
#: says nothing else is the connection's.
_BANNER = ("Welcome to vast.ai", "Have fun!", "Warning: Permanently added")
#: ``ssh`` exits with this when it, and not the remote command, failed.
SSH_FAILED = 255
#: ``rsync``'s exit codes for a stream that broke: socket, protocol, the two timeouts, ssh.
RSYNC_CONNECTION_CODES = (10, 12, 30, 35, SSH_FAILED)
#: A command is tried this many times when the connection fails, this long apart at most.
ATTEMPTS = 6
BACKOFF_S = 15.0
BACKOFF_CAP_S = 120.0


class RemoteError(RuntimeError):
    """A command on a machine failed; ``returncode`` is its own, or ``None`` for a timeout."""

    def __init__(self, message: str, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class ConnectionLost(RemoteError):
    """The connection failed, every attempt: the command itself may never have run."""


def connection_level(argv: list[str], returncode: int, stderr: str) -> bool:
    """Whether a failure is the connection's and not the command's own.

    ``ssh`` returns 255 for its own failures and the remote command's code
    otherwise, so a remote command that exits non-zero is never taken for
    a lost connection, whatever it printed. A 255 that names a connection
    failure, or says nothing but the host's banner, is one. ``scp`` and
    ``rsync`` are judged by their words and, for ``rsync``, its codes.
    """
    tool = Path(argv[0]).name if argv else ""
    named = any(text in stderr for text in _TRANSIENT)
    if tool == "ssh":
        said = [
            line
            for line in stderr.splitlines()
            if line.strip() and not any(text in line for text in _BANNER)
        ]
        return returncode == SSH_FAILED and (named or not said)
    if tool == "scp":
        return named or returncode == SSH_FAILED
    if tool == "rsync":
        return named or returncode in RSYNC_CONNECTION_CODES
    return False


def backoff_s(attempt: int) -> float:
    """Seconds before the attempt after ``attempt``: doubling, capped."""
    return float(min(BACKOFF_S * 2.0 ** (attempt - 1), BACKOFF_CAP_S))


def _run(
    argv: list[str], *, what: str, timeout: float | None = None, attempts: int = ATTEMPTS
) -> str:
    """Run ``argv`` and return its stdout; a lost connection is retried, a failed command is not.

    ``ssh`` and ``scp`` are tried again, with a growing pause and at most
    ``attempts`` times, when :func:`connection_level` says the connection
    failed: then :class:`ConnectionLost`. Any other failure is the
    command's and is raised at once as :class:`RemoteError`. A command
    that passes ``timeout`` is not tried again: it may still be running.
    """
    retried = bool(argv) and Path(argv[0]).name in ("ssh", "scp")
    stderr = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            raise RemoteError(f"{what} did not end in {timeout:g} s") from None
        if completed.returncode == 0:
            return completed.stdout
        stderr = completed.stderr.strip()
        if not connection_level(argv, completed.returncode, stderr):
            raise RemoteError(f"{what} failed: {stderr[-2000:]}", completed.returncode)
        if not retried or attempt >= attempts:
            break
        time.sleep(backoff_s(attempt))
    raise ConnectionLost(
        f"{what} failed, the connection lost"
        f"{f' {attempts} times' if retried else ''}: {stderr[-600:]}",
        SSH_FAILED,
    )


def run_on(machine: Machine, command: str, *, what: str, timeout: float | None = None) -> str:
    """Run one shell command on ``machine`` over ssh and return its stdout.

    Retries the failures of the connection like every other ssh call here;
    the command must therefore be one that may run twice.
    """
    return _run(machine.ssh_command(command), what=what, timeout=timeout)


def one_at_a_time(command: str, lock: str = "/root/provision.lock") -> str:
    """``command`` behind a lock on the machine, so that tried again it waits for itself.

    A build whose connection drops goes on without it: its output is a
    pipe to a reader that is still there. Tried again at once, a second
    build would run in the same tree beside the first. Behind ``flock``
    the second waits for the first, then finds everything made. A machine
    without ``flock`` runs the command as it is.
    """
    return f'LOCK="$(command -v flock >/dev/null 2>&1 && echo "flock {lock}")"; $LOCK {command}'


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
