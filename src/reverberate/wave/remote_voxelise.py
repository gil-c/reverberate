"""Voxelising on a rented CPU, which is the half of the split the roadmap left open.

Section 11 says voxelisation "happens here, on whatever CPU is cheapest", and
until now "here" could only mean this laptop: :mod:`reverberate.wave.remote`
ships the four HDF5 files the CUDA engine reads and runs one binary, which is
the *solve* half. The voxeliser needs the other kind of machine entirely -- a
Python interpreter, numpy below 2, PFFDTD's own tree and this project's patches
on top of it -- and none of that was ever put on a rented box.

**It is a CPU job, and that is the point.** Nothing here touches the GPU. Vast
prices a machine by its card, so the cheapest way to buy 40 cores is to rent a
box whose GPU will sit idle, which inverts B0's mistake rather than repeating
it: B0 paid for an idle *GPU* while the CPU voxelised, and this pays for an
idle GPU on purpose because the CPU beside it is what is being bought.

**What crosses, and what does not.** Up goes the exported model, the impedance
files, the child that drives PFFDTD and the three vendored replacements. Down
comes the cache entry. The scene mesh does go onto a rented machine here, which
:mod:`reverberate.wave.remote` deliberately refuses for the solve; that is a
real difference and the reason is simply that a voxeliser cannot work without
it. It is worth knowing rather than discovering.

**Amdahl decides the machine, not the core count.** Measured on the flat at
4 kHz, 92 per cent of the work is parallel and 8 per cent is not, and the serial
part is ``consolidate``, which runs once per slab. Past about forty cores the
serial floor dominates, so a fast core beats a numerous one and paying for 192
of them buys almost nothing.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from reverberate.wave import vendored
from reverberate.wave.remote import Machine, _run
from reverberate.wave.voxelise import CACHE_FILES, SceneSpec

__all__ = [
    "REMOTE_PFFDTD",
    "RetrievalFailed",
    "REMOTE_WORK",
    "RemoteVoxelisation",
    "install_patches",
    "provision",
    "run_child",
    "voxelise_remote",
]

#: Where ``scripts/build_pffdtd.sh`` puts the checkout and its interpreter.
REMOTE_PFFDTD = "/root/pffdtd"
REMOTE_VENV = "/root/pffdtd-venv/bin/python"
#: Where the model, the materials and the result live on the rented machine.
REMOTE_WORK = "/root/vox"


def _rsync(
    machine: Machine,
    sources: list[str],
    destination: str,
    *,
    download: bool,
    attempts: int = 4,
) -> None:
    """Transfer with resume and retries, because one connection is one thing to lose.

    Not hypothetical: the first whole-flat run voxelised correctly on a rented
    machine and then died on ``Connection closed by remote host`` while fetching
    25 GB, having already spent the compute.

    **Only flags openrsync accepts.** macOS ships openrsync, not GNU rsync, and
    it rejects ``--append-verify`` outright -- which cost a rental to discover,
    because the run died on the *upload* before it computed anything. ``-P``
    (``--partial --progress``) and ``--timeout`` are common to both, and
    ``--partial`` is the one that matters: a broken transfer leaves what
    arrived, and the retry resumes against it rather than starting over.
    """
    identity = ["-i", str(machine.identity)] if machine.identity else []
    shell = " ".join(
        ["ssh", "-p", str(machine.port), "-o", "StrictHostKeyChecking=accept-new", *identity]
    )
    remote = f"{machine.user}@{machine.host}"
    argv = ["rsync", "-az", "--partial", "--timeout=120", "-e", shell]
    if download:
        argv += [f"{remote}:{source}" for source in sources]
        argv.append(destination)
    else:
        argv += [*sources, f"{remote}:{destination}"]

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _run(argv, what=f"rsync {'down' if download else 'up'}")
            return
        except Exception as error:  # noqa: BLE001 - a dropped transfer is the common case
            last = error
            if attempt < attempts:
                print(f"  transfer attempt {attempt} failed, resuming: {error}", flush=True)
                time.sleep(10 * attempt)
    raise RuntimeError(f"transfer failed after {attempts} attempts: {last}")


class RetrievalFailed(RuntimeError):
    """The voxelisation finished and getting it back did not.

    Carried as its own type so a caller can tell the two apart, because the
    right response differs completely: a compute that failed leaves nothing
    worth paying for, and a *retrieval* that failed leaves a finished grid on a
    machine that is still running. Destroying on both is what lost the first
    whole-flat run.
    """

    def __init__(self, report: dict[str, object], log: str, cause: Exception) -> None:
        super().__init__(f"the grid was computed but could not be fetched: {cause}")
        self.report = report
        self.log = log


@dataclass(frozen=True)
class RemoteVoxelisation:
    """What one rented voxelisation produced, and where the time went."""

    entry: Path
    report: dict[str, object]
    provision_s: float
    upload_s: float
    voxelise_s: float
    fetch_s: float
    uploaded_bytes: int
    log: str

    @property
    def total_s(self) -> float:
        """Seconds the instance was actually needed for, which is what it bills."""
        return self.provision_s + self.upload_s + self.voxelise_s + self.fetch_s

    def summary(self) -> str:
        return (
            f"provision {self.provision_s / 60:.1f} min, upload {self.upload_s / 60:.1f} min "
            f"({self.uploaded_bytes / 1e6:.0f} MB), voxelise {self.voxelise_s / 60:.1f} min, "
            f"fetch {self.fetch_s / 60:.1f} min -- {self.total_s / 3600:.2f} h billed"
        )


def provision(machine: Machine, script: Path, timeout: float = 3600.0) -> float:
    """Build PFFDTD and its interpreter on ``machine``. Returns seconds taken.

    ``scripts/build_pffdtd.sh`` unchanged and idempotent, so a machine that is
    already built costs one ssh round trip. It also compiles the CUDA binaries,
    which this path does not need; that is left alone rather than special-cased,
    because a second build script that drifts from the first is worse than a few
    wasted minutes of nvcc.
    """
    started = time.time()
    _run(machine.scp_command([script], "/root/build_pffdtd.sh", download=False), what="send build")
    _run(
        machine.ssh_command("bash /root/build_pffdtd.sh 2>&1 | tail -40"),
        what="build pffdtd",
        timeout=timeout,
    )
    return time.time() - started


def install_patches(machine: Machine) -> list[str]:
    """Put this project's replacement files into the rented checkout.

    The same three files ``ensure_patched`` installs locally, and for the same
    reason: without patch 5 the rented machine produces geometry with unsealed
    interiors, without patch 6 it produces the right geometry a hundred times
    too slowly, and the cache key -- which hashes all three -- would then name a
    grid that this project's code would never produce.

    The build script's own ``sed`` for ``np.float`` is overwritten here by patch
    7, which is the declared version of the same repair.
    """
    written = []
    for relative in sorted(vendored.PATCHED_FILES):
        target = f"{REMOTE_PFFDTD}/{relative}"
        _run(
            machine.ssh_command(f"mkdir -p {shlex.quote(str(Path(target).parent))}"),
            what="mkdir patches",
        )
        _run(
            machine.scp_command([vendored.patched_path(relative)], target, download=False),
            what=f"install {relative}",
        )
        written.append(relative)
    return written


def run_child(
    machine: Machine,
    spec: SceneSpec,
    *,
    nprocs: int | None = None,
    remote_dir: str = REMOTE_WORK,
    timeout: float | None = None,
) -> tuple[dict[str, object], str]:
    """Run ``_child_voxelise.py`` on the rented machine. Returns its report and log.

    The job dict is built here rather than imported from
    :func:`reverberate.wave.voxelise.voxelise`, because every path in it names a
    location on the *other* machine. Keeping the two in step is what
    ``tests/test_remote_voxelise.py`` checks.
    """
    labels = sorted(spec.mat_files)
    job = {
        "pffdtd_dir": REMOTE_PFFDTD,
        "out_dir": remote_dir,
        "model_json": f"{remote_dir}/{Path(spec.model_json).name}",
        "mat_folder": f"{remote_dir}/materials",
        "mat_files": {label: spec.mat_files[label] for label in labels},
        "fmax": spec.fmax,
        "ppw": spec.ppw,
        "Tc": spec.tc,
        "rh": spec.rh,
        "fcc": spec.fcc,
        "bmin": list(spec.bmin) if spec.bmin else None,
        "bmax": list(spec.bmax) if spec.bmax else None,
        "rot_az_el": list(spec.rot_az_el),
        "nprocs": nprocs,
        "compress": None,
        "slabs": spec.slabs,
        "nh": spec.nh,
        "nvox_est": spec.nvox_est,
    }
    remote = (
        f"cd {shlex.quote(remote_dir)} && "
        f"echo {shlex.quote(json.dumps(job))} | {REMOTE_VENV} {remote_dir}/_child_voxelise.py 2>&1"
    )
    log = _run(machine.ssh_command(remote), what="voxelise", timeout=timeout)
    marked = [line for line in log.splitlines() if "@@REVERBERATE@@" in line]
    if not marked:
        raise RuntimeError(f"the child printed no report:\n{log[-3000:]}")
    report: dict[str, object] = json.loads(marked[0].split("@@REVERBERATE@@")[1])
    return report, log


def voxelise_remote(
    machine: Machine,
    spec: SceneSpec,
    destination: Path,
    *,
    build_script: Path,
    nprocs: int | None = None,
    remote_dir: str = REMOTE_WORK,
    timeout: float | None = None,
) -> RemoteVoxelisation:
    """Provision, upload, voxelise, retrieve. Does not rent and does not destroy.

    Deliberately, and for the same reason :func:`reverberate.wave.remote.solve`
    does neither: section 12.1 wants the rate and the total agreed before an
    instance exists, and :func:`reverberate.gpu.vast.rent` has already armed a
    watchdog that outlives this process. Handing this function a machine that is
    already running keeps the decision to spend money in one place.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    child = Path(__file__).with_name("_child_voxelise.py")

    provision_s = provision(machine, build_script)
    install_patches(machine)

    started = time.time()
    materials = [Path(spec.mat_folder) / name for name in sorted(spec.mat_files.values())]
    payload = [Path(spec.model_json), child]
    _run(machine.ssh_command(f"mkdir -p {shlex.quote(remote_dir)}/materials"), what="mkdir work")
    _rsync(machine, [str(f) for f in payload], remote_dir, download=False)
    _rsync(machine, [str(f) for f in materials], f"{remote_dir}/materials", download=False)
    uploaded = sum(f.stat().st_size for f in [*payload, *materials])
    upload_s = time.time() - started

    started = time.time()
    report, log = run_child(machine, spec, nprocs=nprocs, remote_dir=remote_dir, timeout=timeout)
    voxelise_s = time.time() - started

    started = time.time()
    try:
        _rsync(
            machine,
            [f"{remote_dir}/{name}" for name in CACHE_FILES],
            str(destination),
            download=True,
        )
    except Exception as error:  # noqa: BLE001 - the type is what the caller needs
        (destination / "voxelise.log").write_text(log)
        raise RetrievalFailed(report, log, error) from error
    fetch_s = time.time() - started

    (destination / "voxelise.log").write_text(log)
    return RemoteVoxelisation(
        entry=destination,
        report=report,
        provision_s=provision_s,
        upload_s=upload_s,
        voxelise_s=voxelise_s,
        fetch_s=fetch_s,
        uploaded_bytes=uploaded,
        log=log,
    )


def remote_disk_free_gb(machine: Machine) -> float:
    """Free space on the rented machine, in GB.

    Checked before uploading rather than discovered when the voxeliser's own
    disk guard prompts on a stdin that has already been consumed -- which does
    not read as "out of space", it reads as a hang.
    """
    out = _run(machine.ssh_command("df -Pk / | tail -1"), what="df")
    return float(out.split()[3]) / 1e6
