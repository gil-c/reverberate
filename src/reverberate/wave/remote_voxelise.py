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
from typing import Any

from reverberate.wave import vendored
from reverberate.wave.remote import Machine, _run
from reverberate.wave.voxelise import CACHE_FILES, SceneSpec

__all__ = [
    "MachineNeed",
    "REMOTE_PFFDTD",
    "RetrievalFailed",
    "build_payload_remote",
    "payload_need",
    "payload_need_for",
    "voxelise_need",
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


@dataclass(frozen=True)
class MachineNeed:
    """What a stage needs from a machine, computed from the job it was given.

    Every rental this project has lost was lost to a number nobody worked out
    beforehand: an ssh key that was never going to match, a disk guard wanting
    302 GB on a 120 GB box, an estimate that ignored the cost of reserved disk.
    So a stage states its requirement as arithmetic over the grid it is about to
    build, and :meth:`unmet` refuses **before** an instance exists.

    The three stages want different machines and must not share one:

    ======== ============================ ==========================
    stage    wants                        does not want
    ======== ============================ ==========================
    voxelise many cores, large disk       a GPU
    payload  memory, one fast core        a GPU, many cores
    solve    VRAM                         cores
    ======== ============================ ==========================
    """

    cores: int
    ram_gb: float
    disk_gb: float
    why: str
    needs_gpu: bool = False
    vram_gb: float = 0.0

    def merge(self, other: MachineNeed) -> MachineNeed:
        """The machine that satisfies both stages, for a rental that runs both."""
        return MachineNeed(
            cores=max(self.cores, other.cores),
            ram_gb=max(self.ram_gb, other.ram_gb),
            disk_gb=max(self.disk_gb, other.disk_gb),
            why=f"{self.why}; then {other.why}",
            needs_gpu=self.needs_gpu or other.needs_gpu,
            vram_gb=max(self.vram_gb, other.vram_gb),
        )

    def unmet(self, offer: Any) -> list[str]:
        """Every requirement ``offer`` fails, as sentences. Empty means rentable."""
        problems = []
        if offer.cpu_cores < self.cores:
            problems.append(f"{offer.cpu_cores:.0f} vCPU, needs {self.cores}")
        if offer.ram_gb < self.ram_gb:
            problems.append(f"{offer.ram_gb:.0f} GB RAM, needs {self.ram_gb:.0f}")
        if offer.disk_gb < self.disk_gb:
            problems.append(f"{offer.disk_gb:.0f} GB disk, needs {self.disk_gb:.0f}")
        if self.needs_gpu and offer.gpu_ram_gb < self.vram_gb:
            problems.append(f"{offer.gpu_ram_gb:.0f} GB VRAM, needs {self.vram_gb:.0f}")
        return problems


def grid_shape_of(model_json: Path, fmax: float, ppw: float) -> tuple[int, int, int]:
    """The grid a voxelisation will build, without building it.

    ``CartGrid`` is ``ceil((bmax - bmin + 2 * offset * h) / h) + 1`` per axis
    with ``offset`` 3.5, and the bounds come from the model's own points. Worked
    out here so a machine can be sized before one is rented, which is the whole
    difference between a requirement and a hope.
    """
    import numpy as np

    from reverberate.experiments.run import grid_step

    model = json.loads(Path(model_json).read_text())
    points = np.concatenate(
        [np.asarray(group["pts"], dtype=float) for group in model["mats_hash"].values()]
    )
    step = grid_step(fmax, ppw)
    span = points.max(axis=0) - points.min(axis=0) + 2 * 3.5 * step
    shape = np.ceil(span / step).astype(np.int64) + 1
    return int(shape[0]), int(shape[1]), int(shape[2])


#: Bytes a boundary node occupies in ``vox_out.h5``, measured at 23.0 across
#: four entries spanning three orders of magnitude.
BYTES_PER_NODE = 23.0


def nodes_from_shape(shape: tuple[int, int, int]) -> float:
    """How many boundary nodes a grid of this shape will hold, near enough to size a box.

    Boundary nodes cover a *surface*, so they scale as the grid to the two
    thirds. The constant is read off the flat at 4 kHz -- 66 159 665 nodes on a
    2894 x 360 x 2265 grid -- and reproduces the same flat at 16 kHz to within a
    few per cent, which is all a machine requirement needs.
    """
    points = float(shape[0]) * shape[1] * shape[2]
    return float(66_159_665 * (points / 2.3598e9) ** (2 / 3))


def voxelise_need(model_json: Path, fmax: float, ppw: float = 10.5, slabs: int = 1) -> MachineNeed:
    """Sized from the grid the job will actually build.

    Disk is the binding constraint and it is not the output file. PFFDTD's own
    guard compares ``Nx*Ny*Nz`` against **half** the free space and, when it is
    unhappy, asks a question on a stdin the child has already consumed -- which
    presents as a hang, not as a full disk. So the requirement is twice the grid
    in bytes, plus the entry, plus the per-voxel spill.

    That guard is checking for space a slabbed run never uses, since it never
    runs ``check_adj_full``. Until upstream is told so, the space has to be
    rented anyway.
    """
    shape = grid_shape_of(model_json, fmax, ppw)
    grid_bytes = float(shape[0]) * shape[1] * shape[2]
    nodes = nodes_from_shape(shape)
    entry_gb = nodes * BYTES_PER_NODE / 1e9
    disk = 2 * grid_bytes / 1e9 + entry_gb + 30.0
    ram = max(8.0, 60.0 * nodes / slabs / 1e9 + 4.0)
    return MachineNeed(
        cores=16,
        ram_gb=ram,
        disk_gb=disk,
        why=(
            f"grid {shape[0]}x{shape[1]}x{shape[2]}, {grid_bytes / 1e9:.0f} GB for PFFDTD's "
            f"disk guard, ~{entry_gb:.1f} GB entry, {slabs} slab(s)"
        ),
    )


def payload_need_for(nodes: float, shape: tuple[int, int, int], target_cubes: int) -> MachineNeed:
    """The payload stage's requirement from a node count and a grid shape.

    Taken separately from :func:`payload_need` so a rental that will voxelise
    *and then* build the payload can be sized for both before either exists.
    """
    span = 1
    while nodes / span**2 > target_cubes:
        span *= 2
    lattice = 1.0
    for size in shape:
        lattice *= -(-size // span)
    # `_bin_voxels` holds the node arrays; `surface_of` holds the block lattice
    # twice, as int16 labels and as a bool. Fitted to the one measurement there
    # is -- 16.9 GB for the flat at 4 kHz, 66 159 665 nodes at one block each --
    # and then rounded **up**, because a requirement that under-asks is a
    # machine rented to swap for seven hours, which is how this number came to
    # be measured at all.
    ram = 30.0 * nodes / 1e9 + 5.0 * lattice / 1e9 + 4.0
    return MachineNeed(
        cores=4,
        ram_gb=ram,
        disk_gb=nodes * BYTES_PER_NODE / 1e9 + 20.0,
        why=(
            f"payload: {nodes:,.0f} nodes at one block per {span}, lattice "
            f"{lattice / 1e6:.0f}M cells; memory-bound and single-threaded, "
            "so cores are wasted on it"
        ),
    )


def payload_need(cache_dir: Path, target_cubes: int) -> MachineNeed:
    """Sized from the grid on disk, which by now is a fact rather than a forecast."""
    import h5py

    with h5py.File(Path(cache_dir) / "vox_out.h5", "r") as handle:
        nodes = int(handle["bn_ixyz"].shape[0])
        shape = (
            int(handle["Nx"][()]),
            int(handle["Ny"][()]),
            int(handle["Nz"][()]),
        )
    return payload_need_for(nodes, shape, target_cubes)


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
    fetch_entry: bool = True,
) -> RemoteVoxelisation:
    """Provision, upload, voxelise, retrieve. Does not rent and does not destroy.

    ``fetch_entry=False`` leaves the grid on the machine, which is what the
    caller wants when the next thing it does is build the viewer payload there:
    25 GB stays put and 185 MB comes home instead.

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
    if fetch_entry:
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


def build_payload_remote(
    machine: Machine,
    destination: Path,
    *,
    labels: list[str],
    target_cubes: int,
    remote_dir: str = REMOTE_WORK,
    timeout: float | None = None,
) -> tuple[dict[str, object], str]:
    """Build the viewer's payload where the grid already is, and fetch only that.

    The point of doing it here rather than at home: ``vox_out.h5`` is 25 GB for
    the flat at 16 kHz and the payload the browser fetches is about 185 MB. One
    of those transfers has already failed and cost a finished grid; the other
    takes seconds. It also keeps the work off the laptop, which is the rule.

    No second machine and no second provisioning: the build needs numpy and
    h5py, PFFDTD's own interpreter has both, so this runs on the box that has
    just voxelised. ``vox_view`` and ``comms`` are copied across as files.
    """
    modules = f"{remote_dir}/modules"
    here = Path(__file__).parent
    _run(machine.ssh_command(f"mkdir -p {shlex.quote(modules)}"), what="mkdir modules")
    _rsync(
        machine,
        [
            str(here.parent / "viz" / "vox_view.py"),
            str(here / "comms.py"),
            str(here / "_child_payload.py"),
        ],
        modules,
        download=False,
    )
    job = {
        "module_dir": modules,
        "cache_dir": remote_dir,
        "out_dir": f"{remote_dir}/payload",
        "target_cubes": target_cubes,
        "labels": labels,
    }
    remote = (
        f"cd {shlex.quote(remote_dir)} && echo {shlex.quote(json.dumps(job))} | "
        f"{REMOTE_VENV} {modules}/_child_payload.py 2>&1"
    )
    log = _run(machine.ssh_command(remote), what="payload", timeout=timeout)
    marked = [line for line in log.splitlines() if "@@REVERBERATE@@" in line]
    if not marked:
        raise RuntimeError(f"the payload child printed no report:\n{log[-3000:]}")
    report: dict[str, object] = json.loads(marked[0].split("@@REVERBERATE@@")[1])

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _rsync(machine, [f"{remote_dir}/payload/"], str(destination), download=True)
    return report, log


def remote_disk_free_gb(machine: Machine) -> float:
    """Free space on the rented machine, in GB.

    Checked before uploading rather than discovered when the voxeliser's own
    disk guard prompts on a stdin that has already been consumed -- which does
    not read as "out of space", it reads as a hang.
    """
    out = _run(machine.ssh_command("df -Pk / | tail -1"), what="df")
    return float(out.split()[3]) / 1e6
