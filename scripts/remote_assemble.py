"""Rent cores, build the whole dataset's collider pool on them, bring it home.

Assembling an apartment is dominated by one thing per template: the boolean
union of its collision proxy and the flood fill behind it, of the order of a
minute on a cold cache. HSSD places 15 684 distinct templates, so warming the
pool for the whole dataset is about 266 hours of one core -- days on the laptop,
and the laptop is not what this project runs long jobs on.

**Only the pool is built here.** The instance runs
:mod:`reverberate.viz.assemble_dataset` with nothing published, which fills
``cache/colliders`` as a side effect of assembling every storey. That directory
is what comes home; the apartments are then re-assembled locally in seconds and
published from the machine that already holds the credentials.

**No credential leaves this machine.** The instance needs to read one object,
the dataset tarball, so it is handed a presigned GET URL for that object alone,
valid for the length of the rental. It never sees the bucket's keys and cannot
write to the bucket at all.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/remote_assemble.py \\
        --hours 8 --max-dph 0.35 --min-cores 64 --yes
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from reverberate import auth
from reverberate.geometry import collider_cache
from reverberate.gpu import vast
from reverberate.store import BUCKET_ENV, shared_store
from reverberate.wave.remote import Machine, _run

#: A plain CPU image. Nothing here compiles CUDA, and a four gigabyte devel
#: image would be four gigabytes of pull time before the first template. Its
#: own Python is 3.10 and too old for the pinned ``scikit-image``, so the
#: bootstrap installs 3.12 beside it.
IMAGE = "ubuntu:22.04"

#: Where the dataset tarball sits in the store, as ``remote_assemble`` put it.
DATASET_KEY = "datasets/hssd-hab.tar"

#: Bytes of resident memory one worker peaks at, measured on the laptop. Used
#: to cap the worker count against the machine's RAM rather than its core
#: count: a box with 96 vCPU and 63 GB cannot run 96 flood fills.
WORKER_RAM_GB = 1.5

REMOTE = "/root"

#: What the instance installs, named rather than taken from
#: ``requirements.txt``. That file pulls a private git checkout for the vault
#: and three visualisation packages, and this machine needs neither: it has no
#: credentials by design, and it draws nothing. ``manifold3d`` is the one that
#: matters -- without it every boolean union quietly fails and the pool fills
#: with meshes that kept their buried interior faces.
REMOTE_PACKAGES = " ".join(
    (
        "'trimesh>=4.4'",
        "'manifold3d>=3.5'",
        "'mapbox_earcut>=2.0'",
        "'rtree>=1.4'",
        "'scikit-image>=0.26'",
        "'fast_simplification>=0.1'",
        "'shapely>=2.1'",
        "'pyroomacoustics>=0.10'",
        "'scipy>=1.13'",
        "'h5py>=3.11'",
    )
)


def say(message: str) -> None:
    print(message, flush=True)


def _ssh_shell(machine: Machine) -> str:
    """The ``-e`` argument rsync needs to reach this machine."""
    identity = f" -i {machine.identity}" if machine.identity else ""
    return (
        "ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "
        f"-p {machine.port}{identity}"
    )


def _rsync_up(machine: Machine, source: str, destination: str) -> None:
    subprocess.run(
        [
            "rsync",
            "-az",
            "--partial",
            "-e",
            _ssh_shell(machine),
            source,
            f"{machine.user}@{machine.host}:{destination}",
        ],
        check=True,
    )


def _rsync_down(machine: Machine, source: str, destination: str) -> None:
    subprocess.run(
        [
            "rsync",
            "-az",
            # Partial files wait outside the pool: --partial kept an interrupted
            # file under its final name, and a zero-byte mesh was read as an entry.
            "--partial-dir=.rsync-partial",
            "--timeout=120",
            "-e",
            _ssh_shell(machine),
            f"{machine.user}@{machine.host}:{source}",
            destination,
        ],
        check=True,
    )


def presigned_dataset_url(hours: float) -> str:
    """A time limited, read only URL for the one object the instance may read."""
    store = shared_store()
    if store is None:
        raise SystemExit("no store on this machine; the instance has nothing to read")
    return store.presigned_get(DATASET_KEY, int(hours * 3600) + 1800)


def bootstrap(url: str, workers: int, shard: str) -> str:
    """The one script the instance runs, detached, writing its own log."""
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq software-properties-common build-essential curl rsync >/dev/null
# The image ships Python 3.10 and scikit-image needs 3.11 or later, so the
# interpreter is installed rather than assumed. Named here rather than solved
# by relaxing a pin: the pin is what the laptop runs, and a pool built against
# a different one is a pool that answers for code this project does not run.
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >/dev/null
# The path carries the version. ``python3.12 -m venv`` over a venv built by
# 3.10 leaves 3.10's own interpreter symlink in place and installs into 3.12's
# site-packages, so the check below failed to import a package pip had just
# reported installing.
python3.12 -m venv {REMOTE}/venv312
{REMOTE}/venv312/bin/pip install -q --upgrade pip
{REMOTE}/venv312/bin/pip install -q {REMOTE_PACKAGES}
# The union is silent when it fails: outer_surface hands back the unmerged
# mesh and reports False, which is the right behaviour for one bad object and
# the wrong thing to discover after eight hours of it. So the backend is
# checked before a single template is carved.
{REMOTE}/venv312/bin/python -c "import manifold3d, trimesh; \\
from trimesh.boolean import boolean_manifold; print('boolean engine ok')"
mkdir -p {REMOTE}/hssd-hab {REMOTE}/data
if [ ! -f {REMOTE}/hssd-hab/.unpacked ]; then
  curl -fsSL {shlex.quote(url)} -o {REMOTE}/hssd-hab.tar
  tar -xf {REMOTE}/hssd-hab.tar -C {REMOTE}/hssd-hab
  rm -f {REMOTE}/hssd-hab.tar
  touch {REMOTE}/hssd-hab/.unpacked
fi
cd {REMOTE}/reverberate
REVERBERATE_DATA={REMOTE}/data PYTHONPATH={REMOTE}/reverberate/src \\
  {REMOTE}/venv312/bin/python -u -m reverberate.viz.assemble_dataset \\
  {REMOTE}/hssd-hab --workers {workers} --warm {shard}
touch {REMOTE}/DONE
"""


def rank(
    offers: list[vast.Offer],
    max_dph: float,
    min_cores: int,
    exclude: frozenset[int] = frozenset(),
) -> list[vast.Offer]:
    """Cheapest per unit of work first: dollars over workers times clock.

    Two corrections to ranking on cores, each paid for. **Workers, not cores**:
    a 192 vCPU box with 63 GB of RAM runs 41 flood fills, not 192. **Clock, not
    only count**: 96 cores at 2.2 GHz managed 0.7 templates a minute for 0.351
    USD/h, against 24 a minute on 40 cores at 3.6 GHz for 0.203 -- a factor of
    thirty the wrong way at a higher price. Clock is not the whole story, and
    two identical offers measured 68 and 21 a minute, but it is the part an
    offer states.
    """
    eligible = [
        offer
        for offer in offers
        if offer.id not in exclude
        and offer.dph_total <= max_dph
        and offer.cpu_cores >= min_cores
        and offer.ram_gb >= min_cores * WORKER_RAM_GB * 0.5
    ]
    return sorted(
        eligible,
        key=lambda offer: offer.dph_total / (workers_for(offer) * max(offer.cpu_ghz, 1.0)),
    )


def workers_for(offer: vast.Offer) -> int:
    """As many workers as the cores allow and the memory can hold."""
    return max(1, min(int(offer.cpu_cores), int(offer.ram_gb / WORKER_RAM_GB)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0, help="rental deadline")
    parser.add_argument("--max-dph", type=float, default=0.35)
    parser.add_argument("--min-cores", type=int, default=48)
    parser.add_argument(
        "--min-ghz",
        type=float,
        default=3.0,
        help="refuse slow clocks outright; 2.2 GHz cost thirty times the work per dollar",
    )
    parser.add_argument("--disk-gb", type=int, default=60)
    parser.add_argument(
        "--warm",
        default="0/1",
        help="which shard of the templates this machine takes, as i/n",
    )
    parser.add_argument(
        "--exclude-offer",
        default="",
        help="comma separated offer ids to skip, for a host that will not build",
    )
    parser.add_argument("--yes", action="store_true")
    arguments = parser.parse_args(argv)

    started = time.time()
    auth.inject([vast.API_KEY_ENV, BUCKET_ENV])
    client = vast.VastClient()
    identity = vast.account_identity(client)
    say(f"ledger stands at {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")

    machine: Machine | None = None
    instance_id = 0
    offers = client.search(
        vast.search_query(
            gpu_name="",
            min_disk_gb=arguments.disk_gb,
            min_reliability=0.98,
            # Pushed into the query rather than filtered afterwards: the
            # search returns a page, and a page ordered by anything but
            # cores holds almost no machine this job wants.
            min_cpu_cores=arguments.min_cores,
            min_cpu_ghz=arguments.min_ghz,
        ),
        limit=200,
    )
    # An offer is an advertisement, and a host can be advertising a
    # machine it cannot actually build a container on. Two rentals from
    # 46624748 in a row came up 'docker_build() error writing dockerfile'
    # and never left `loading`, so naming a bad host is worth an option
    # rather than a fourth rental on it.
    excluded = frozenset(int(part) for part in arguments.exclude_offer.split(",") if part.strip())
    ranked = rank(offers, arguments.max_dph, arguments.min_cores, excluded)
    if not ranked:
        raise SystemExit(f"nothing under {arguments.max_dph} USD/h has {arguments.min_cores} vCPU")
    say(f"{len(ranked)} offers meet it, cheapest per worker and per GHz first:")
    for offer in ranked[:6]:
        say(
            f"  {offer.id} {offer.dph_total:.3f} USD/h  {offer.cpu_cores:.0f} vCPU "
            f"@ {offer.cpu_ghz:.1f} GHz, {offer.ram_gb:.0f} GB RAM "
            f"-> {workers_for(offer)} workers, "
            f"{vast.estimate_cost_usd(offer.dph_total, arguments.hours):.2f} USD "
            f"for {arguments.hours:g} h"
        )
    best = ranked[0]
    say(f"\nchosen: {best.describe()}")
    say(
        f"  cap {arguments.hours:g} h -> at most "
        f"{vast.estimate_cost_usd(best.dph_total, arguments.hours):.2f} USD"
    )
    if not arguments.yes:
        say("\nnothing rented: pass --yes once the figures above are approved")
        return 0
    for candidate in ranked[:5]:
        # The ids live before the attempt, so an instance the attempt
        # created can be told apart. vast answered HTTP 410 to a rental
        # that had in fact created its instance: the loop moved on, and
        # the orphan billed for 5.6 hours with no job on it.
        before = {instance.id for instance in client.instances()}
        try:
            rental = vast.rent(
                client,
                candidate,
                hours=arguments.hours,
                image=IMAGE,
                disk_gb=arguments.disk_gb,
            )
        except vast.VastError as refusal:
            say(f"  {candidate.id} would not rent ({refusal}); trying the next")
            for orphan in {i.id for i in client.instances()} - before:
                say(f"  ...but instance {orphan} exists anyway; destroying it")
                vast.teardown(client, orphan)
            continue
        instance_id = rental.instance_id
        say(f"rented {instance_id}, watchdog pid {rental.watchdog_pid}")
        try:
            machine = vast.wait_for_ssh(client, instance_id, identity, timeout=600.0)
        except (TimeoutError, vast.VastError) as silence:
            say(f"  {instance_id} never answered ({silence}); destroying it")
            vast.teardown(client, instance_id)
            continue
        best = candidate
        break
    if machine is None:
        raise SystemExit("no offer that met the requirement produced a machine that answered")
    chosen_workers = workers_for(best)

    if not arguments.yes:
        say("\nnothing started: pass --yes")
        return 0

    say(f"sending the source tree, then starting {chosen_workers} workers")
    _run(machine.ssh_command(f"mkdir -p {REMOTE}/reverberate"), what="mkdir")
    _rsync_up(machine, "src/", f"{REMOTE}/reverberate/src/")
    script = bootstrap(presigned_dataset_url(arguments.hours), chosen_workers, arguments.warm)
    local_script = Path(tempfile.mkdtemp()) / "bootstrap.sh"
    local_script.write_text(script)
    _run(
        machine.scp_command([local_script], f"{REMOTE}/bootstrap.sh", download=False),
        what="send bootstrap",
    )
    _run(
        machine.ssh_command(
            f"rm -f {REMOTE}/DONE; nohup bash {REMOTE}/bootstrap.sh "
            f"> {REMOTE}/assemble.log 2>&1 & echo started $!"
        ),
        what="start",
    )
    say(f"started on {instance_id}")

    deadline = time.time() + arguments.hours * 3600
    finished = False
    while time.time() < deadline:
        time.sleep(300)
        try:
            probe = subprocess.run(
                machine.ssh_command(
                    f"test -f {REMOTE}/DONE && echo DONE; tail -1 {REMOTE}/assemble.log 2>/dev/null"
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as silence:  # noqa: BLE001 - a lost poll is not a lost job
            say(f"  poll failed ({silence}); the job is detached and still running")
            continue
        line = probe.stdout.strip().replace("\n", " | ")
        say(f"  [{(time.time() - started) / 60:.0f} min] {line[:200]}")
        if "DONE" in probe.stdout:
            finished = True
            break

    if not finished:
        say(
            f"the deadline passed with no DONE on {instance_id}. Nothing is destroyed: "
            "fetch what the pool holds by hand before the watchdog fires."
        )
        return 1

    # Teardown only once the pool is home: rsync raises on anything short of a
    # clean transfer, including a stall, and the instance is then left up for
    # its watchdog. A `finally` that cannot tell a failed compute from a failed
    # retrieval throws away work that finished (W35).
    pool = collider_cache.cache_root()
    say(f"fetching the collider pool into {pool}")
    _rsync_down(machine, f"{REMOTE}/data/cache/colliders/", f"{pool}/")
    vast.teardown(client, instance_id)
    say(f"destroyed {instance_id}; ledger now {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
