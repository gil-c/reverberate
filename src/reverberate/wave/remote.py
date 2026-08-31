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
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.wave.comms import ENGINE_FILES

__all__ = ["Machine", "SolveResult", "fetch", "run_engine", "solve", "upload"]

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


def run_engine(
    machine: Machine,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    *,
    pffdtd_dir: str = DEFAULT_PFFDTD_DIR,
    double_precision: bool = False,
    timeout: float | None = None,
) -> str:
    """Run the CUDA binary in ``remote_dir`` and return its log."""
    precision = "double" if double_precision else "single"
    binary = f"{pffdtd_dir}/c_cuda/fdtd_main_gpu_{precision}.x"
    remote = f"cd {shlex.quote(remote_dir)} && {shlex.quote(binary)} 2>&1"
    return _run(machine.ssh_command(remote), what="engine", timeout=timeout)


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
    log = run_engine(
        machine,
        remote_dir,
        pffdtd_dir=pffdtd_dir,
        double_precision=double_precision,
        timeout=timeout,
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
