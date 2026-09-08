"""Run one stage of the pipeline on a rented machine, sized for that stage.

**The laptop is not a compute node.** Its ten cores have been pinned at 100 per
cent for hours more than once through a script error or an estimate that was
wrong, and the dataset will be rented anyway, so the chain is proved rented from
the start. Nothing here runs locally except argument parsing.

Three stages, three machines, because their needs do not overlap:

========= ============================== =========================
stage     wants                          wasted on it
========= ============================== =========================
voxelise  many cores, a very large disk  a GPU
payload   memory, one fast core          a GPU, many cores
solve     VRAM                           cores
========= ============================== =========================

``voxelise`` and ``payload`` can share one rental, and by default they do:
``vox_out.h5`` is 25 GB for the flat at 16 kHz while the payload the browser
fetches is about 185 MB, so building the payload on the machine that has just
made the grid turns a transfer that has already failed once into one that takes
seconds.

**Every requirement is arithmetic over the job, not a guess.** ``MachineNeed``
is computed from the grid the run will build, offers that cannot meet it are
rejected before an instance exists, and the rate and the cap are printed before
anything is created. Four rentals were lost this way before it was written; see
``MachineNeed`` for what each one bought.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/remote_chain.py \\
        --models data/runs/w32_carved/models --scene apartment_full \\
        --fmax 16000 --slabs 16 --nh 40 --hours 4 --out /tmp/flat16k \\
        --payload --viewer-cubes 2000000000 --yes

Add ``--payload`` and the viewer's files are built on that same machine and
fetched instead of the grid, which is the whole point: 185 MB comes home and
25 GB stays where it was made.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from reverberate import auth
from reverberate.experiments.run import build_materials
from reverberate.gpu import vast
from reverberate.wave.remote import Machine, _run
from reverberate.wave.remote_voxelise import (
    REMOTE_WORK,
    MachineLost,
    MachineNeed,
    RetrievalFailed,
    build_payload_remote,
    grid_shape_of,
    install_entry,
    nodes_from_shape,
    payload_need_for,
    remote_disk_free_gb,
    voxelise_need,
    voxelise_remote,
)
from reverberate.wave.voxelise import SceneSpec, nh_for

#: A CUDA image because Vast's cheap boxes are GPU boxes and the build script
#: compiles the engine too. The voxelise and payload stages never use the card.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

#: Gigabytes of the reserved disk that are gone before the job sees any of it:
#: the image, the layers the build unpacks, PFFDTD's checkout and its venv.
#: Measured on the 2026-09-07 rental -- 402 GB advertised, 367 reserved, **385
#: free** -- so an offer advertising exactly what the guard needs does not have
#: it. Named because the alternative is a three hour band that dies on
#: PFFDTD's disk prompt, which does not read as "out of space", it reads as a
#: hang.
IMAGE_DISK_GB = 25.0


def account_identity(client: vast.VastClient) -> Path:
    """The local private key whose public half Vast will install, or refuse.

    Checked before renting. The alternative is what it cost to learn: an
    instance comes up, ssh answers ``Permission denied (publickey)`` on every
    poll for the full timeout, and the run tears down having done nothing.
    """
    registered = {
        (key.get("public_key") or "").split()[1]
        for key in client._request("GET", "/ssh/")
        if len((key.get("public_key") or "").split()) > 1
    }
    if not registered:
        raise SystemExit("the Vast account has no ssh key registered; add one in the console")
    for public in sorted(Path.home().joinpath(".ssh").glob("*.pub")):
        blob = public.read_text().split()
        if len(blob) > 1 and blob[1] in registered:
            private = public.with_suffix("")
            if private.is_file():
                return private
    raise SystemExit(
        "no private key here matches a key registered on the Vast account, so ssh into "
        "the instance would be refused; nothing was rented"
    )


def pick_offers(
    client: vast.VastClient, need: MachineNeed, max_dph: float, min_cpu_ghz: float = 0.0
) -> list[vast.Offer]:
    """Every offer that meets ``need``, cheapest first, or an explanation and none.

    A *list*, because an offer is an advertisement and not a promise: the
    cheapest one that fits answered ``HTTP 400`` to two rentals in a row while
    its advertised disk quietly fell from 632 GB to 609, and a function that
    returns one offer sends the caller back to the same dead host every time.

    ``min_cpu_ghz`` is the one filter a CPU-bound stage wants and the one that
    was missing. W35 measured 192 vCPU voxelising at the same speed as ten, and
    on 2026-09-07 a 48 vCPU box at 3.0 GHz took about twice the laptop's time on
    this flat at 4 kHz -- while a 16 core part at 4.8 GHz was on offer for the
    same money. Cheapest-that-fits is the right rule only once the fit includes
    the thing being bought.
    """
    query = vast.search_query(
        gpu_name="",
        min_disk_gb=int(need.disk_gb + IMAGE_DISK_GB),
        min_cpu_cores=need.cores,
        min_reliability=0.99,
        min_cpu_ghz=min_cpu_ghz,
    )
    offers = client.search(query, limit=200)
    affordable = [offer for offer in offers if offer.dph_total <= max_dph]
    eligible = [
        offer
        for offer in affordable
        if not need.unmet(offer) and offer.disk_gb >= need.disk_gb + IMAGE_DISK_GB
    ]
    if not eligible:
        print(f"{len(offers)} offers matched the query, {len(affordable)} under {max_dph} USD/h")
        for offer in affordable[:5]:
            problems = need.unmet(offer)
            if offer.disk_gb < need.disk_gb + IMAGE_DISK_GB:
                problems.append(
                    f"{offer.disk_gb:.0f} GB disk advertised, needs "
                    f"{need.disk_gb + IMAGE_DISK_GB:.0f} with the image's share"
                )
            print(f"  {offer.id}: {', '.join(problems)}")
        raise SystemExit(f"nothing meets: {need.why}")
    return sorted(eligible, key=lambda offer: offer.dph_total)


def wait_for_ssh(
    client: vast.VastClient, instance_id: int, identity: Path, timeout: float = 900.0
) -> Machine:
    """Block until the instance answers a command, not merely until it exists."""
    from reverberate.wave.remote import _run

    deadline = time.time() + timeout
    while time.time() < deadline:
        instance = client.instance(instance_id)
        if instance is None:
            raise RuntimeError(f"instance {instance_id} vanished while starting")
        if instance.ssh_host and instance.status == "running":
            machine = Machine(host=instance.ssh_host, port=instance.ssh_port, identity=identity)
            try:
                _run(machine.ssh_command("true"), what="ssh probe", timeout=30)
                return machine
            except Exception:  # noqa: BLE001 - not up yet is the common case
                pass
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} never answered on ssh")


def spec_from(args: argparse.Namespace, fmax: float) -> tuple[SceneSpec, Path, int]:
    """The scene at one band, the model file it names, and its triangle count.

    The triangle count comes back because the memory requirement needs it and
    the manifest already carries it: the alternative is parsing a 233 MB model
    a second time to learn something the exporter wrote down.
    """
    manifest = json.loads((args.models / "manifest.json").read_text())
    scene = {entry["name"]: entry for entry in manifest["scenes"]}[args.scene]
    model_json = (args.models / scene["file"]).resolve()
    labels = set(json.loads(model_json.read_text())["mats_hash"])
    mat_folder = args.models.parent / "materials"
    mat_files = build_materials(labels, manifest["materials"], mat_folder)
    return (
        SceneSpec(
            model_json=model_json,
            mat_folder=mat_folder,
            mat_files=mat_files,
            fmax=fmax,
            ppw=10.5,
            slabs=args.slabs,
            # A cell count only holds at one band, so it is derived from the
            # grid unless the caller insists. 40 cells is 8.2 cm at 16 kHz and
            # 1.31 m at 1 kHz, and the second of those does not finish; see
            # reverberate.wave.voxelise.VOXEL_BUDGET.
            nh=args.nh if args.nh else nh_for(grid_shape_of(model_json, fmax, 10.5)),
        ),
        model_json,
        int(scene["triangles"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--fmax",
        type=float,
        nargs="+",
        required=True,
        help="one or more bands, computed in ascending order on ONE rental. The machine "
        "is sized for the largest, so the cheap bands ride along on a box that had to "
        "be rented for the expensive one anyway, and each grid is installed into the "
        "local cache under its own key",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True, help="hard deadline")
    parser.add_argument("--slabs", type=int, default=1)
    parser.add_argument(
        "--nh",
        type=int,
        default=None,
        help="voxel side in grid steps. Left unset it is derived per band from "
        "reverberate.wave.voxelise.VOXEL_BUDGET, which is the only form that "
        "survives changing fmax",
    )
    parser.add_argument(
        "--viewer-cubes",
        type=int,
        default=20_000_000,
        help="block budget; large enough forces one block per node, which is lossless",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="after voxelising, build the viewer payload on the same machine and fetch "
        "only that, leaving the grid where it was made",
    )
    parser.add_argument("--max-dph", type=float, default=0.40)
    parser.add_argument(
        "--min-cpu-ghz",
        type=float,
        default=4.0,
        help="floor on the advertised core clock. Voxelisation is CPU bound and "
        "92 per cent parallel, so past about forty cores the serial floor "
        "dominates and only the clock is left to buy",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="keep each fetched grid local instead of pushing it to the object store",
    )
    parser.add_argument("--yes", action="store_true", help="required to spend money")
    args = parser.parse_args(argv)

    # Ascending, so a mistake in the arithmetic shows up on a one minute grid
    # rather than after three hours of the expensive one.
    bands = sorted(dict.fromkeys(args.fmax))
    specs = {fmax: spec_from(args, fmax) for fmax in bands}
    model_json = specs[bands[0]][1]

    triangles = specs[bands[0]][2]
    need = None
    for fmax in bands:
        nh = args.nh if args.nh else nh_for(grid_shape_of(model_json, fmax, 10.5))
        shape = grid_shape_of(model_json, fmax, 10.5)
        lattice = 1
        for size in shape:
            lattice *= -(-size // nh)
        one = voxelise_need(model_json, fmax, slabs=args.slabs, triangles=triangles, voxels=lattice)
        if args.payload:
            # One rental does both, so it has to satisfy both: the voxeliser
            # wants cores and disk, the payload build wants memory, and neither
            # is the other's constraint.
            one = one.merge(payload_need_for(nodes_from_shape(shape), shape, args.viewer_cubes))
        need = one if need is None else need.merge(one)
    assert need is not None

    for fmax in bands:
        spec = specs[fmax][0]
        print(f"{args.scene} at {fmax:g} Hz, key {spec.key}, nh {spec.nh}")
    print(f"  needs {need.cores} vCPU, {need.ram_gb:.0f} GB RAM, {need.disk_gb:.0f} GB disk")
    print(f"  because {need.why}")

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    identity = account_identity(client)
    print(f"  ssh identity {identity}")

    candidates = pick_offers(client, need, args.max_dph, args.min_cpu_ghz)
    offer = candidates[0]
    budget = vast.estimate_cost_usd(offer.dph_total, args.hours)
    print(f"\n{offer.describe()}")
    print(f"  {len(candidates)} offers meet the requirement; this is the cheapest")
    print(f"  cap {args.hours:g} h -> at most {budget:.2f} USD at the offer's rate")
    print("  the bill will be higher: Vast adds the disk you reserve, about 0.05 USD/h per")
    print(f"  200 GB, so {int(need.disk_gb)} GB is roughly +{need.disk_gb * 0.000265:.3f} USD/h")
    print(f"  already committed: {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")
    if not args.yes:
        print("\nnothing rented: pass --yes to spend")
        return 0

    store = None
    if not args.no_publish:
        from reverberate.store import shared_store

        store = shared_store()
        if store is None:
            print("no store credentials here, so the grids stay on this machine")

    # Down the list until one actually rents. Nothing has been created yet, so
    # a refusal here costs nothing but the round trip.
    # Down the list until one rents *and answers*. An offer is an
    # advertisement: 45591732 refused two rentals with HTTP 400 and then, when
    # it finally accepted one, never answered ssh at all -- three quarters of
    # an hour of wall clock for a host that was never going to work. Renting
    # and reaching are one step here because failing either means the same
    # thing: try the next machine, not give up on the band.
    rental = None
    machine = None
    for candidate in candidates[:8]:
        try:
            attempt = vast.rent(
                client, candidate, hours=args.hours, image=IMAGE, disk_gb=int(need.disk_gb) + 20
            )
        except vast.VastError as refusal:
            print(f"  {candidate.id} would not rent ({refusal}); trying the next")
            continue
        print(f"\nrented {attempt.instance_id} on {candidate.describe()}")
        print(f"watchdog pid {attempt.watchdog_pid}")
        try:
            machine = wait_for_ssh(client, attempt.instance_id, identity, timeout=420.0)
        except (TimeoutError, RuntimeError) as silence:
            print(f"  {attempt.instance_id} never answered ({silence}); destroying it")
            vast.teardown(client, attempt.instance_id)
            continue
        rental, offer = attempt, candidate
        break
    if rental is None or machine is None:
        raise SystemExit("no offer that met the requirement produced a machine that answered")

    computed = False
    try:
        print(f"ssh up at {machine.host}:{machine.port}")
        free = remote_disk_free_gb(machine)
        print(f"free disk {free:.0f} GB, need {need.disk_gb:.0f}")
        if free < need.disk_gb:
            raise SystemExit(
                f"{free:.0f} GB free against {need.disk_gb:.0f} needed; stopping before the "
                "upload rather than paying for a run that stalls on PFFDTD's disk prompt"
            )
        spent = 0.0
        for fmax in bands:
            spec = specs[fmax][0]
            # A directory per band, because the child writes its result under
            # fixed names and spills its per-voxel scratch beside them; two
            # bands sharing one would delete each other's and die on an
            # assertion about triangle counts that reads like a bad scene.
            remote_dir = f"{REMOTE_WORK}/{fmax:g}"
            out = args.out / f"{fmax:g}"
            print(f"\n=== {fmax:g} Hz, key {spec.key}", flush=True)
            result = voxelise_remote(
                machine,
                spec,
                out,
                build_script=Path(__file__).with_name("build_pffdtd.sh"),
                remote_dir=remote_dir,
                timeout=args.hours * 3600,
                fetch_entry=not args.payload,
            )
            computed = True
            spent += result.total_s
            print(result.summary())
            print(json.dumps(result.report, indent=2)[:900])

            if args.payload:
                print("building the payload on the machine that just made the grid")
                report, _ = build_payload_remote(
                    machine,
                    out,
                    labels=sorted(spec.mat_files),
                    target_cubes=args.viewer_cubes,
                    remote_dir=remote_dir,
                    timeout=args.hours * 3600,
                )
                print(json.dumps(report, indent=2)[:600])
            else:
                # Into the cache under its key, with the manifest everything
                # downstream reads. A fetched grid that is not an entry is a
                # grid that has to be computed again to be used.
                entry = install_entry(spec, out, result.report, result.voxelise_s)
                print(f"installed {entry.key} -> {entry.path} ({entry.complete=})")
                if store is not None:
                    from reverberate.wave.vox_store import publish_entry

                    started = time.time()
                    digests = publish_entry(store, entry)
                    print(
                        f"published {len(digests)} files to the store in "
                        f"{(time.time() - started) / 60:.1f} min"
                    )
            # The grid is off the machine; its scratch is not, and 300 GB of
            # spill would refuse the next band on the same disk.
            _run(machine.ssh_command(f"rm -rf {shlex.quote(remote_dir)}"), what="clear band")
        billed = vast.estimate_cost_usd(offer.dph_total, spent / 3600)
        print(f"\nabout {spent / 3600:.2f} h at the offer's rate = {billed:.2f} USD")
    except (RetrievalFailed, MachineLost):
        # Both mean the work is not the thing that failed, so the instance is
        # left running for a human to fetch from. The watchdog still ends it.
        computed = True
        raise
    finally:
        if computed and machine is not None and sys.exc_info()[0] is not None:
            print(f"\nthe grid was computed; instance {rental.instance_id} is LEFT RUNNING:")
            print(f"  ssh -i {identity} -p {machine.port} root@{machine.host}")
            print(f"  the watchdog destroys it at the {args.hours:g} h deadline regardless")
        else:
            print(f"destroying {rental.instance_id}...")
            try:
                client.destroy_and_verify(rental.instance_id)
                print("destroyed and verified")
            except Exception as failure:  # noqa: BLE001 - the watchdog is the backstop
                print(f"destroy failed ({failure}); the watchdog fires", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
