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

import hashlib
import json
import shlex
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.wave import vendored
from reverberate.wave.remote import Machine, _run
from reverberate.wave.voxelise import CACHE_FILES, CacheEntry, SceneSpec

__all__ = [
    "MachineLost",
    "MachineNeed",
    "install_entry",
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
    "POLL_S",
    "launch_command",
    "run_child",
    "start_child",
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

#: Bytes one triangle's precomputed record occupies. ``tris_precompute``
#: returns a structured dtype of sixteen fields -- vertices, three edges, two
#: normals, three edge normals, centroid, two bounds, three squared lengths and
#: the area -- and ``.dtype.itemsize`` is **368**, read off the vendored
#: checkout rather than counted by hand.
BYTES_PER_TRIANGLE = 368.0

#: How many voxels a triangle ends up in, as a multiple of the triangle count.
#: ``VoxGrid.fill`` gives every non-empty voxel **its own copy** of its
#: triangles' precomputed records -- ``vox.tris_pre = self.tris_pre[tri_idxs]``
#: -- so the scene's precompute is held once per voxel it touches, and that is
#: the term the requirement used to ignore.
#:
#: Measured on this flat at an 8.17 cm voxel, which is what
#: :func:`reverberate.wave.voxelise.nh_for` gives at both 4 and 16 kHz:
#: ``tris redundant=12 528 926`` against 3 983 792 triangles, so **3.15**. At
#: 1 kHz's 13.07 cm voxel it is 4.64, higher because a coarser lattice cannot
#: be the reason -- the halo is, and it is proportionally larger. The larger of
#: the two is used, because under-asking here is an out-of-memory kill three
#: minutes into a rental and it does not say so: 2026-09-07, a 15 GB box,
#: ``1461 Killed`` and nothing else.
TRIANGLE_REDUNDANCY = 4.7

#: Bytes one ``VoxBase`` costs: a Python object, four attributes and a list.
BYTES_PER_VOXEL = 250.0

#: Bytes a node in the slab being consolidated costs at the peak, counted off
#: the arrays ``_child_voxelise._slabbed_adj`` holds at once rather than
#: estimated: ``bn_ixyz`` 8, the three ``subs`` 24, the rotated index 8, the
#: argsort permutation 8, ``adj_bn`` and its two fancy-indexed copies 18,
#: ``mat_bn`` twice 2, ``saf_bn`` twice 16. W32's figure was 60, which is the
#: sum without the copies the rotate and sort passes make.
BYTES_PER_SLAB_NODE = 90.0


def nodes_from_shape(shape: tuple[int, int, int]) -> float:
    """How many boundary nodes a grid of this shape will hold, near enough to size a box.

    Boundary nodes cover a *surface*, so they scale as the grid to the two
    thirds. The constant is read off the flat at 4 kHz -- 66 159 665 nodes on a
    2894 x 360 x 2265 grid -- and reproduces the same flat at 16 kHz to within a
    few per cent, which is all a machine requirement needs.
    """
    points = float(shape[0]) * shape[1] * shape[2]
    return float(66_159_665 * (points / 2.3598e9) ** (2 / 3))


def voxelise_need(
    model_json: Path,
    fmax: float,
    ppw: float = 10.5,
    slabs: int = 1,
    triangles: int | None = None,
    voxels: int | None = None,
) -> MachineNeed:
    """Sized from the grid the job will actually build.

    Disk is the binding constraint and it is not the output file. PFFDTD's own
    guard compares ``Nx*Ny*Nz`` against **half** the free space and, when it is
    unhappy, asks a question on a stdin the child has already consumed -- which
    presents as a hang, not as a full disk. So the requirement is twice the grid
    in bytes, plus the entry, plus the per-voxel spill.

    That guard is checking for space a slabbed run never uses, since it never
    runs ``check_adj_full``. Until upstream is told so, the space has to be
    rented anyway.

    **Memory has four terms and it used to have one.** Consolidate was the only
    one modelled, at 60 bytes a node, because that is the term slabbing exists
    to cap. The other three are the scene, not the grid, and on a flat carved
    at 2 mm they are the larger half:

    ``consolidate``
        :data:`BYTES_PER_SLAB_NODE` a node, divided by the slab count -- which
        is only the peak if the slabs are balanced, and
        ``_child_voxelise.slab_groups`` is what makes them so.
    ``the scene's precompute``
        368 bytes a triangle, once. :data:`BYTES_PER_TRIANGLE`.
    ``the per-voxel copies``
        368 bytes again for every (triangle, voxel) pair the fill keeps, and
        there are :data:`TRIANGLE_REDUNDANCY` of them per triangle. This is the
        term whose absence cost a rental: a 15 GB box, ``1461 Killed`` three
        minutes in, and a requirement that had said 8 GB.
    ``the voxel objects``
        250 bytes each, 2.35 million of them for this flat at 16 kHz.

    ``triangles`` and ``voxels`` are passed in rather than parsed here: the
    caller already has the model open and the lattice chosen. Left unset they
    fall back to bounds that cannot under-ask -- a scene of four million
    triangles and the budget's own voxel ceiling -- because a requirement that
    guesses low is the failure this whole function exists to prevent.
    """
    from reverberate.wave.voxelise import VOXEL_BUDGET

    shape = grid_shape_of(model_json, fmax, ppw)
    grid_bytes = float(shape[0]) * shape[1] * shape[2]
    nodes = nodes_from_shape(shape)
    entry_gb = nodes * BYTES_PER_NODE / 1e9
    disk = 2 * grid_bytes / 1e9 + entry_gb + 30.0

    tris = float(triangles if triangles else 4_000_000)
    nvox = float(voxels if voxels else VOXEL_BUDGET)
    consolidate_gb = BYTES_PER_SLAB_NODE * nodes / slabs / 1e9
    scene_gb = tris * BYTES_PER_TRIANGLE / 1e9
    copies_gb = tris * TRIANGLE_REDUNDANCY * BYTES_PER_TRIANGLE / 1e9
    voxels_gb = nvox * BYTES_PER_VOXEL / 1e9
    # Four gigabytes over the sum, for the worker pool's own per-voxel arrays
    # and the interpreter. The one machine this band has ever completed on had
    # 34 GB, and the arithmetic above puts the flat at 16 kHz at 16, so the
    # margin is not what was missing -- the terms were.
    ram = max(8.0, consolidate_gb + scene_gb + copies_gb + voxels_gb + 4.0)
    return MachineNeed(
        cores=16,
        ram_gb=ram,
        disk_gb=disk,
        why=(
            f"grid {shape[0]}x{shape[1]}x{shape[2]}, {grid_bytes / 1e9:.0f} GB for PFFDTD's "
            f"disk guard, ~{entry_gb:.1f} GB entry, {slabs} slab(s); memory is "
            f"{consolidate_gb:.1f} consolidate + {scene_gb:.1f} scene + {copies_gb:.1f} "
            f"per-voxel copies + {voxels_gb:.1f} voxel objects GB"
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


class MachineLost(RuntimeError):
    """The polls stopped answering, and the job may well still be running.

    Its own type for the reason :class:`RetrievalFailed` has one: the child is
    detached now, so losing the connection is not losing the work, and
    destroying the instance on this would throw away a grid that is still being
    written. W29's lesson, in the one place the detaching moved it to.
    """


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
    ``tests/test_remote_chain.py`` checks.

    **Detached, and polled.** This used to be one foreground ``ssh`` whose
    stdout was the job's: the voxelisation lived and died with a single TCP
    connection. A whole flat at 16 kHz is two hours on that connection, and it
    does not survive them -- ``Connection closed by remote host``, at 17.7 GB
    of a 22 GB grid, with the instance then destroyed because the driver could
    not tell a dead job from a dead socket. W35 had already recorded the same
    sentence killing a *fetch*; it kills a compute the same way.

    So the child is started under ``setsid nohup`` with its output going to a
    file on the rented machine, and this function polls a fresh, short-lived
    connection for the pid. A dropped poll is retried; the job never notices.
    The log is fetched once, at the end, from the file -- which also means an
    interrupted driver can be pointed at a machine that is still working.
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
    pid = start_child(machine, job, remote_dir)
    return _await_child(machine, pid, remote_dir, timeout)


def launch_command(job: dict[str, Any], remote_dir: str) -> str:
    """The one-liner that starts the child detached and prints its pid.

    Written as its own function because it is a shell quoting problem with a
    trap in it, and a trap that costs a rental deserves a test rather than a
    reading. ``cmd1 && cmd2 & cmd3`` backgrounds **the whole ``&&`` chain**, so
    the first version of this backgrounded the job file's own creation, took
    the chain's pid for the child's, and left ``ssh`` waiting until its five
    minute timeout -- on a machine where the child was in fact already running.

    So the setup runs to completion first, ended with ``;``, and only the
    launcher is backgrounded. The launcher is a script rather than an inline
    command so the child's exit status can be kept beside its log; and every
    one of its three descriptors is redirected to a file, because ``ssh``
    returns when nothing still holds the channel and a background process
    holding stdout holds it.
    """
    job_file = f"{remote_dir}/job.json"
    log_file = f"{remote_dir}/voxelise.out"
    rc_file = f"{remote_dir}/voxelise.rc"
    runner = f"{remote_dir}/voxelise.sh"
    script = (
        f"{REMOTE_VENV} {remote_dir}/_child_voxelise.py "
        f"< {job_file} > {log_file} 2>&1; echo $? > {rc_file}"
    )
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"printf %s {shlex.quote(json.dumps(job))} > {shlex.quote(job_file)} && "
        f"printf %s {shlex.quote(script)} > {shlex.quote(runner)} && "
        f"rm -f {shlex.quote(log_file)} {shlex.quote(rc_file)}; "
        f"setsid sh {shlex.quote(runner)} < /dev/null > /dev/null 2>&1 & "
        f"echo $!"
    )


def start_child(machine: Machine, job: dict[str, Any], remote_dir: str) -> str:
    """Start the detached child and return its pid."""
    printed = _run(
        machine.ssh_command(launch_command(job, remote_dir)), what="start voxelise", timeout=300
    )
    pid = printed.strip().split()[-1]
    if not pid.isdigit():
        raise RuntimeError(f"the launcher printed no pid: {printed[-500:]}")
    return pid


#: How often :func:`_await_child` asks whether the job is still alive. Long
#: enough that a two hour run costs a couple of hundred short connections,
#: short enough that the driver is not sitting on a finished machine.
POLL_S = 60.0


def _await_child(
    machine: Machine, pid: str, remote_dir: str, timeout: float | None
) -> tuple[dict[str, object], str]:
    """Wait for a detached child, over connections that may each fail.

    A poll that cannot connect is not a failed job -- it is a failed poll, and
    the two were the same thing until the child was detached. Only a run of
    them is treated as the machine being gone.
    """
    log_file = f"{remote_dir}/voxelise.out"
    deadline = time.time() + timeout if timeout else None
    misses = 0
    while True:
        if deadline and time.time() > deadline:
            raise TimeoutError(f"the voxelisation passed its {timeout:.0f} s deadline")
        time.sleep(POLL_S)
        try:
            alive = _run(
                machine.ssh_command(
                    f"kill -0 {shlex.quote(pid)} 2>/dev/null && echo yes || echo no"
                ),
                what="poll voxelise",
                timeout=120,
            ).strip()
            misses = 0
        except Exception as error:  # noqa: BLE001 - a dropped poll is the common case
            misses += 1
            if misses >= 10:
                raise MachineLost(
                    f"ten polls running went unanswered, and the child is detached, "
                    f"so it is probably still working: {error}"
                ) from error
            continue
        if alive.endswith("yes"):
            continue
        break

    log = _run(
        machine.ssh_command(f"cat {shlex.quote(log_file)}"), what="voxelise log", timeout=600
    )
    marked = [line for line in log.splitlines() if "@@REVERBERATE@@" in line]
    if not marked:
        code = _run(
            machine.ssh_command(f"cat {remote_dir}/voxelise.rc 2>/dev/null || echo ?"),
            what="voxelise status",
            timeout=120,
        ).strip()
        raise RuntimeError(f"the child printed no report (exit {code}):\n{log[-3000:]}")
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


def install_entry(
    spec: SceneSpec, fetched: Path, report: dict[str, object], wall_s: float
) -> CacheEntry:
    """Move a fetched grid into this machine's cache and write its manifest.

    Without this a rented voxelisation was not a cache entry. It arrived as
    four HDF5 files in whatever directory the caller named, and everything
    downstream -- :func:`reverberate.experiments.run.entry_from_key`, the run
    page, the audit view, :func:`reverberate.wave.vox_store.publish_entry` --
    addresses a grid by its key and reads ``manifest.json`` for the material
    labels. So the rented half of the split ended at the transfer, and the
    grid had to be recomputed locally to be usable, which is the opposite of
    the point.

    The manifest is the same one :func:`reverberate.wave.voxelise.voxelise`
    writes, with two fields that can only differ: the checkout is the rented
    one, and the commit is the pin the patches were derived from rather than
    the output of ``git`` on a machine that no longer exists.
    """
    from reverberate.wave import vendored
    from reverberate.wave.voxelise import _geometry_bytes, cache_root, entry_for

    fetched = Path(fetched)
    missing = [name for name in CACHE_FILES if not (fetched / name).is_file()]
    if missing:
        raise RuntimeError(f"the fetched grid has no {missing}, refusing to install it")

    model_json = Path(spec.model_json)
    manifest = dict(report)
    manifest.update(
        {
            "key": spec.key,
            "model_json": str(model_json.resolve()),
            "model_sha256": hashlib.sha256(model_json.read_bytes()).hexdigest(),
            "geometry_sha256": hashlib.sha256(_geometry_bytes(model_json)).hexdigest(),
            "fmax": spec.fmax,
            "ppw": spec.ppw,
            "nh": spec.nh,
            "slabs": spec.slabs,
            "Tc": spec.tc,
            "rh": spec.rh,
            "fcc": spec.fcc,
            "materials": dict(spec.mat_files),
            "wall_s": round(wall_s, 3),
            "computed_on": "rented",
            "pffdtd_dir": REMOTE_PFFDTD,
            "pffdtd_commit": vendored.UPSTREAM_COMMIT,
            "file_bytes": {
                name: (fetched / name).stat().st_size
                for name in CACHE_FILES
                if (fetched / name).is_file()
            },
        }
    )
    (fetched / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    destination = cache_root() / spec.key
    if destination.resolve() != fetched.resolve():
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fetched), str(destination))
    return entry_for(spec)


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
