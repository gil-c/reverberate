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
import sys
import time
from pathlib import Path

from reverberate import auth
from reverberate.experiments.run import build_materials
from reverberate.gpu import vast
from reverberate.wave.remote import Machine
from reverberate.wave.remote_voxelise import (
    MachineNeed,
    RetrievalFailed,
    build_payload_remote,
    grid_shape_of,
    nodes_from_shape,
    payload_need_for,
    remote_disk_free_gb,
    voxelise_need,
    voxelise_remote,
)
from reverberate.wave.voxelise import SceneSpec

#: A CUDA image because Vast's cheap boxes are GPU boxes and the build script
#: compiles the engine too. The voxelise and payload stages never use the card.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"


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


def pick_offer(client: vast.VastClient, need: MachineNeed, max_dph: float) -> vast.Offer:
    """The cheapest offer that meets ``need``, or an explanation and no rental."""
    query = vast.search_query(
        gpu_name="",
        min_disk_gb=int(need.disk_gb),
        min_cpu_cores=need.cores,
        min_reliability=0.99,
    )
    offers = client.search(query, limit=200)
    affordable = [offer for offer in offers if offer.dph_total <= max_dph]
    eligible = [offer for offer in affordable if not need.unmet(offer)]
    if not eligible:
        print(f"{len(offers)} offers matched the query, {len(affordable)} under {max_dph} USD/h")
        for offer in affordable[:5]:
            print(f"  {offer.id}: {', '.join(need.unmet(offer))}")
        raise SystemExit(f"nothing meets: {need.why}")
    return min(eligible, key=lambda offer: offer.dph_total)


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


def spec_from(args: argparse.Namespace) -> tuple[SceneSpec, Path]:
    """The scene, and the model file it names."""
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
            fmax=args.fmax,
            ppw=10.5,
            slabs=args.slabs,
            nh=args.nh,
        ),
        model_json,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--fmax", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True, help="hard deadline")
    parser.add_argument("--slabs", type=int, default=1)
    parser.add_argument("--nh", type=int, default=None)
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
    parser.add_argument("--yes", action="store_true", help="required to spend money")
    args = parser.parse_args(argv)

    spec, model_json = spec_from(args)
    need = voxelise_need(model_json, args.fmax, slabs=args.slabs)
    if args.payload:
        # One rental does both, so it has to satisfy both: the voxeliser wants
        # cores and disk, the payload build wants memory, and neither is the
        # other's constraint.
        shape = grid_shape_of(model_json, args.fmax, 10.5)
        need = need.merge(payload_need_for(nodes_from_shape(shape), shape, args.viewer_cubes))

    print(f"{args.scene} at {args.fmax:g} Hz, key {spec.key}")
    print(f"  needs {need.cores} vCPU, {need.ram_gb:.0f} GB RAM, {need.disk_gb:.0f} GB disk")
    print(f"  because {need.why}")

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    identity = account_identity(client)
    print(f"  ssh identity {identity}")

    offer = pick_offer(client, need, args.max_dph)
    budget = vast.estimate_cost_usd(offer.dph_total, args.hours)
    print(f"\n{offer.describe()}")
    print(f"  cap {args.hours:g} h -> at most {budget:.2f} USD at the offer's rate")
    print("  the bill will be higher: Vast adds the disk you reserve, about 0.05 USD/h per")
    print(f"  200 GB, so {int(need.disk_gb)} GB is roughly +{need.disk_gb * 0.000265:.3f} USD/h")
    print(f"  already committed: {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")
    if not args.yes:
        print("\nnothing rented: pass --yes to spend")
        return 0

    rental = vast.rent(client, offer, hours=args.hours, image=IMAGE, disk_gb=int(need.disk_gb) + 20)
    print(f"\nrented {rental.instance_id}, watchdog pid {rental.watchdog_pid}")
    computed = False
    machine = None
    try:
        machine = wait_for_ssh(client, rental.instance_id, identity)
        print(f"ssh up at {machine.host}:{machine.port}")
        free = remote_disk_free_gb(machine)
        print(f"free disk {free:.0f} GB, need {need.disk_gb:.0f}")
        if free < need.disk_gb:
            raise SystemExit(
                f"{free:.0f} GB free against {need.disk_gb:.0f} needed; stopping before the "
                "upload rather than paying for a run that stalls on PFFDTD's disk prompt"
            )
        result = voxelise_remote(
            machine,
            spec,
            args.out,
            build_script=Path(__file__).with_name("build_pffdtd.sh"),
            timeout=args.hours * 3600,
            fetch_entry=not args.payload,
        )
        computed = True
        print(f"\n{result.summary()}")
        print(json.dumps(result.report, indent=2)[:1200])

        if args.payload:
            labels = sorted(spec.mat_files)
            print("\nbuilding the payload on the machine that just made the grid")
            report, _ = build_payload_remote(
                machine,
                args.out,
                labels=labels,
                target_cubes=args.viewer_cubes,
                timeout=args.hours * 3600,
            )
            print(json.dumps(report, indent=2)[:900])
        billed = vast.estimate_cost_usd(offer.dph_total, result.total_s / 3600)
        print(f"\nabout {result.total_s / 3600:.2f} h at the offer's rate = {billed:.2f} USD")
    except RetrievalFailed:
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
