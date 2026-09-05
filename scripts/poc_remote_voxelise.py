"""Rent a CPU, voxelise on it, bring the grid back, destroy the instance.

The proof of concept for the half of roadmap section 11 that was never built:
voxelisation "on whatever CPU is cheapest" could only ever mean this laptop,
because nothing put PFFDTD's Python side on a rented box.

**Every exit path tears down.** W25 lost 1.69 USD to a ``SystemExit`` raised
outside a ``try``, so the destroy here is in a ``finally`` and runs whatever
happens, on top of the detached watchdog that ``vast.rent`` arms before it
returns. The watchdog is the safeguard; this is the courtesy that stops the
meter early.

**The rate and the total are printed before anything is created**, which is
section 12.1 and W12's rule, and ``--yes`` is required to actually spend.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/poc_remote_voxelise.py \\
        --models data/runs/w32_carved/models --scene bedroom_only --fmax 4000 \\
        --hours 1 --out /tmp/remote_bedroom --yes
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
    RetrievalFailed,
    remote_disk_free_gb,
    voxelise_remote,
)
from reverberate.wave.voxelise import SceneSpec

#: A CUDA image, because the build script compiles the engine too and Vast's
#: cheap boxes are GPU boxes. Nothing here runs on the card.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"


def account_identity(client: vast.VastClient) -> Path:
    """The private key on this machine whose public half Vast will install.

    Checked **before renting**, because the alternative is what it cost to
    learn: an instance came up, ssh answered ``Permission denied (publickey)``
    on every poll for the full fifteen minute timeout, and the run tore down
    having done nothing. Three cents that time, and the whole cap every time
    the cap is larger.

    Vast installs the account's registered keys into the instance, so the
    question is not "is there a key here" but "is the key here one the account
    knows". Matching on the key blob answers exactly that.
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
        "no private key on this machine matches a key registered on the Vast account, "
        "so ssh into the instance would be refused; nothing was rented"
    )


def wait_for_ssh(
    client: vast.VastClient, instance_id: int, identity: Path, timeout: float = 900.0
) -> Machine:
    """Block until the instance answers on ssh, or give up.

    The instance existing is not the instance being reachable: W25 destroyed an
    A100 by treating a launch timeout as proof of failure, and lost another to a
    host stuck in ``loading``. So this polls for the address and then for a
    command that actually returns.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        instance = client.instance(instance_id)
        if instance is None:
            raise RuntimeError(f"instance {instance_id} vanished while starting")
        if instance.status_msg:
            print(f"  {instance.status}: {instance.status_msg}")
        if instance.ssh_host and instance.status == "running":
            machine = Machine(host=instance.ssh_host, port=instance.ssh_port, identity=identity)
            try:
                from reverberate.wave.remote import _run

                _run(machine.ssh_command("true"), what="ssh probe", timeout=30)
                return machine
            except Exception:  # noqa: BLE001 - not up yet is the common case
                pass
        time.sleep(15)
    raise TimeoutError(f"instance {instance_id} never answered on ssh")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--fmax", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True, help="hard deadline for the rental")
    parser.add_argument("--slabs", type=int, default=1)
    parser.add_argument("--nh", type=int, default=None)
    parser.add_argument("--min-cores", type=int, default=24)
    parser.add_argument("--min-disk", type=int, default=200)
    parser.add_argument("--max-dph", type=float, default=0.25)
    parser.add_argument(
        "--offer", type=int, default=None, help="restrict the search to this offer id"
    )
    parser.add_argument(
        "--identity", default=None, help="ssh private key; found from the account when omitted"
    )
    parser.add_argument(
        "--need-free-gb",
        type=float,
        default=0.0,
        help="abort before uploading unless the machine has this much free; PFFDTD's own "
        "disk guard prompts on a stdin the child has already consumed, so too little space "
        "is not an error message, it is a hang",
    )
    parser.add_argument("--yes", action="store_true", help="required to spend money")
    args = parser.parse_args(argv)

    manifest = json.loads((args.models / "manifest.json").read_text())
    scene = {entry["name"]: entry for entry in manifest["scenes"]}[args.scene]
    model_json = (args.models / scene["file"]).resolve()
    labels = set(json.loads(model_json.read_text())["mats_hash"])
    mat_folder = args.models.parent / "materials"
    mat_files = build_materials(labels, manifest["materials"], mat_folder)
    spec = SceneSpec(
        model_json=model_json,
        mat_folder=mat_folder,
        mat_files=mat_files,
        fmax=args.fmax,
        ppw=10.5,
        slabs=args.slabs,
        nh=args.nh,
    )
    print(f"scene {args.scene} at {args.fmax:g} Hz, key {spec.key}")
    print(f"  {len(labels)} materials, model {model_json.stat().st_size / 1e6:.0f} MB")

    auth.inject([vast.API_KEY_ENV])
    client = vast.VastClient()
    identity = Path(args.identity) if args.identity else account_identity(client)
    print(f"  ssh identity: {identity}")

    query = vast.search_query(
        gpu_name="",
        min_disk_gb=args.min_disk,
        min_cpu_cores=args.min_cores,
        min_reliability=0.99,
    )
    candidates = [o for o in client.search(query, limit=200) if o.dph_total <= args.max_dph]
    if args.offer:
        candidates = [o for o in candidates if o.id == args.offer]
    if not candidates:
        raise SystemExit(f"no offer under {args.max_dph} USD/h matched {query!r}")
    # Most cores per dollar among the ones that clear the floor: the parallel
    # part is 92 per cent of the work, and the serial floor is the same
    # whichever of these is picked.
    offer = max(candidates, key=lambda o: o.cpu_cores / max(o.dph_total, 1e-6))

    budget = vast.estimate_cost_usd(offer.dph_total, args.hours)
    print(f"\n{offer.describe()}")
    print(f"  cap {args.hours:g} h -> at most {budget:.2f} USD")
    print(f"  already committed: {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")
    if not args.yes:
        print("\nnothing rented: pass --yes to spend")
        return 0

    rental = vast.rent(client, offer, hours=args.hours, image=IMAGE, disk_gb=args.min_disk)
    print(f"\nrented instance {rental.instance_id}, watchdog pid {rental.watchdog_pid}")
    computed = False
    machine = None
    try:
        machine = wait_for_ssh(client, rental.instance_id, identity)
        print(f"ssh up at {machine.host}:{machine.port}")
        free = remote_disk_free_gb(machine)
        print(f"free disk: {free:.0f} GB")
        if args.need_free_gb and free < args.need_free_gb:
            raise SystemExit(
                f"the machine has {free:.0f} GB free and this run needs "
                f"{args.need_free_gb:.0f} GB; stopping before the upload rather than "
                "paying for a run that will stall on PFFDTD's disk prompt"
            )
        result = voxelise_remote(
            machine,
            spec,
            args.out,
            build_script=Path(__file__).with_name("build_pffdtd.sh"),
            timeout=args.hours * 3600,
        )
        print(f"\n{result.summary()}")
        print(json.dumps(result.report, indent=2)[:1500])
        billed = vast.estimate_cost_usd(offer.dph_total, result.total_s / 3600)
        print(f"\nactually billed for about {result.total_s / 3600:.2f} h = {billed:.2f} USD")
    except RetrievalFailed:
        computed = True
        raise
    except Exception as error:  # noqa: BLE001 - what failed decides whether to destroy
        # Destroying on *any* failure is what lost the first whole-flat run: the
        # voxelisation had succeeded and only the retrieval failed, so tearing
        # down deleted a good result and its timings. That is W29's own lesson --
        # a grid dying with the machine that made it -- committed in a file that
        # cites W29. A failure after the compute leaves the instance up, and the
        # watchdog still ends it at the deadline.
        print(f"failed before computing anything: {error}", file=sys.stderr)
        raise
    finally:
        if computed and machine is not None:
            print("\nthe grid was computed; only fetching it failed.")
            print(f"instance {rental.instance_id} is LEFT RUNNING so it can be retrieved:")
            print(f"  ssh -i {identity} -p {machine.port} root@{machine.host}")
            print(f"  the watchdog destroys it at the {args.hours:g} h deadline regardless")
        else:
            # Nothing worth keeping, so stop the meter now. W25 paid 1.69 USD
            # for 1.62 h of nothing because this was not in a finally at all.
            print(f"destroying instance {rental.instance_id}...")
            try:
                client.destroy_and_verify(rental.instance_id)
                print("destroyed and verified")
            except Exception as failure:  # noqa: BLE001 - the watchdog is the backstop
                print(f"destroy failed ({failure}); the watchdog fires", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
