"""One machine for the whole campaign: rent it, provision it, watch it, bring the field home.

Everything about Vast lives here and nothing about acoustics does. The
library in :mod:`reverberate.accel` runs on the machine; this module puts it
there, starts it detached, and looks at it every five minutes -- through the
API for the instance and its bill, through ssh for the campaign's own
``status.json``, the card's utilisation and the disk -- and acts on what it
sees rather than waiting for a marker that may never appear: a campaign
whose status has not moved for longer than its stage should take is
restarted once from its state on disk, a machine that vanished is reported
with what it had done, and a machine whose work is home is destroyed and
verified destroyed.

Two campaigns cost five to ten times their compute to idle cards, transfers
on the card's clock and boxes rented after the solves. Here there is one
rental, the transfers are the bundle up (a few hundred megabytes) and the
field down, and the card is released the moment the field is verified home.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import auth
from reverberate.accel.bundle import HOME_ITEMS
from reverberate.accel.lattice import sim_constants
from reverberate.accel.solve import OUTPUT_SAMPLE_BYTES_PATCHED
from reverberate.gpu import vast
from reverberate.wave.remote import run_on
from reverberate.wave.remote_voxelise import grid_shape_of, provision, rsync
from reverberate.wave.voxelise import cache_root

__all__ = [
    "MachineNeed",
    "Watch",
    "campaign_need",
    "choose_offers",
    "fetch",
    "monitor_once",
    "provision_machine",
    "rent",
    "run",
    "watch",
]

#: The image the engine is built in; sm_80 to sm_90 cards, CUDA 12.4.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

#: VRAM per grid node and the engine's fixed overhead, measured on an A100 80 GB.
VRAM_PER_NODE_B = 9.027
VRAM_FIXED_GB = 2.13

#: Where the campaign lives on the machine.
REMOTE_ROOT = "/root/campaign"
REMOTE_BUNDLE = f"{REMOTE_ROOT}/bundle"
REMOTE_OUT = f"{REMOTE_ROOT}/out"
REMOTE_SRC = "/root/reverberate"


#: What the fetch brings home from the run directory: what ``take_home`` keeps,
#: the self-check reports and the driver's own log. Not the encodings (the
#: field holds them), not the self-check samples, not the jobs.
FETCH_ITEMS = (*HOME_ITEMS, "driver.log", "campaign.done", "campaign.failed")
#: Of the self-check directory, the reports and logs; not the pressure samples.
SELFCHECK_PATTERNS = ("*.json", "*.log")


def fetch_items(present: list[str]) -> list[str]:
    """The entries of the run directory worth the transfer, among those present."""
    return [item for item in FETCH_ITEMS if item in present]


def missing_grids(keys: list[str], local_cache: Path) -> list[str]:
    """The bundle's cache keys the laptop does not hold as a complete entry."""
    return [key for key in keys if not (Path(local_cache) / key / "manifest.json").is_file()]


def engine_patches(repo: Path) -> list[Path]:
    """The engine patches the build script applies, pushed beside it."""
    return sorted((Path(repo) / "scripts" / "pffdtd").glob("*.patch"))


REMOTE_DATA = "/root/data"
ACCEL_PYTHON = "/root/accel-venv/bin/python"

#: How often the laptop looks.
POLL_S = 300.0

#: A stage's status that has not moved for this long is a stall.
STALL_S = {
    "start": 1800.0,
    "voxelise": 3600.0,
    "audit": 3600.0,
    "plan": 1800.0,
    "solve": 5400.0,
    "assemble": 3600.0,
}


@dataclass(frozen=True)
class MachineNeed:
    """What the campaign needs from a machine, from the bundle alone."""

    vram_gb: float
    ram_gb: float
    disk_gb: int
    largest_output_gb: float
    nodes_by_band: dict[str, float]

    def describe(self) -> str:
        return (
            f"{self.vram_gb:.0f} GB of VRAM in total, {self.ram_gb:.0f} GB of RAM wanted,"
            f" {self.disk_gb} GB of disk; grids {self.nodes_by_band}"
        )


def campaign_need(bundle: Path) -> MachineNeed:
    """Size the machine from the bundle: the largest grid's VRAM, the largest band's output."""
    spec = json.loads((Path(bundle) / "campaign.json").read_text())
    model_json = Path(bundle) / spec["model_json"]
    nodes: dict[str, float] = {}
    outputs: dict[str, float] = {}
    for band, record in spec["bands"].items():
        shape = grid_shape_of(model_json, float(record["fmax_hz"]), float(spec["ppw"]))
        nodes[band] = float(np.prod(np.asarray(shape, dtype=np.float64)))
        constants = sim_constants(
            float(spec["tc"]), float(spec["rh"]), float(record["fmax_hz"]), float(spec["ppw"])
        )
        steps = int(round(float(record["duration_s"]) * constants.sr))
        # About 1021 nodes a point; the plan will say exactly. The engine our
        # build script makes writes its own precision (patch 8), 4 bytes a sample.
        outputs[band] = float(spec["points"]) * 1021.0 * steps * OUTPUT_SAMPLE_BYTES_PATCHED / 1e9
    largest = max(outputs.values())
    vram = max(nodes.values()) * VRAM_PER_NODE_B / 1e9 + VRAM_FIXED_GB
    return MachineNeed(
        vram_gb=vram,
        ram_gb=min(2.2 * largest + 8.0, 512.0),
        # One slice of pressure at a time on disk, the encodings (about a
        # twentieth of the pressure), the grids, the audit view and a margin.
        disk_gb=int(1.2 * largest + 0.05 * sum(outputs.values()) + 80.0),
        largest_output_gb=largest,
        nodes_by_band=nodes,
    )


def choose_offers(
    offers: list[Any],
    need: MachineNeed,
    *,
    max_dph: float,
    min_ram_gb: float = 60.0,
    min_cores: int = 8,
    gpu: str = "",
) -> list[Any]:
    """Offers whose cards together hold the grid, cheapest first, sized by the whole host.

    ``gpu`` restricts the card's name to those containing it (``A100``).
    """
    good = [
        o
        for o in offers
        if o.num_gpus * o.gpu_ram_gb >= need.vram_gb
        and (not gpu or gpu.lower() in o.gpu_name.lower())
        and o.ram_gb >= min_ram_gb
        and o.cpu_cores >= min_cores
        and o.disk_gb >= need.disk_gb
        and o.dph_total <= max_dph
        and o.cuda_max >= 12.4
    ]

    # A host whose RAM holds the largest band in one slice saves a whole
    # solve of the storey; worth a little money, not a lot.
    def rank(o: Any) -> tuple[float, float]:
        whole = o.ram_gb >= need.ram_gb
        return (o.dph_total * (0.85 if whole else 1.0), -o.reliability)

    return sorted(good, key=rank)


@dataclass
class Watch:
    """What one look at the machine found, and what the watcher did about it."""

    at: float
    instance_alive: bool
    uptime_h: float
    cost_usd: float
    credit_usd: float | None
    status: dict[str, Any] = field(default_factory=dict)
    gpu: str = ""
    disk_free_gb: float | None = None
    log_tail: str = ""
    done: bool = False
    failed: bool = False
    stalled: bool = False
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        stage = self.status.get("stage", "?")
        detail = " ".join(
            f"{k}={self.status[k]}"
            for k in ("band", "source", "job", "slice", "stage_detail")
            if k in self.status
        )
        return (
            f"{time.strftime('%H:%M:%S', time.localtime(self.at))} up {self.uptime_h:.2f} h"
            f" cost {self.cost_usd:.2f} USD"
            f"{f' credit {self.credit_usd:.2f}' if self.credit_usd is not None else ''}"
            f" | {stage} {detail} | gpu {self.gpu} | disk {self.disk_free_gb} GB free"
            f"{' | DONE' if self.done else ''}{' | FAILED' if self.failed else ''}"
            f"{' | STALLED' if self.stalled else ''}"
            f"{(' | ' + '; '.join(self.notes)) if self.notes else ''}"
        )


def monitor_once(
    client: Any,
    instance_id: int,
    machine: Any,
    *,
    previous: Watch | None = None,
    remote_out: str = REMOTE_OUT,
    run_on: Any = None,
) -> Watch:
    """One look: the API for the bill, ssh for the campaign, and a judgement."""
    run_on = run_on or globals()["run_on"]
    now = time.time()
    instance = client.instance(instance_id)
    try:
        credit: float | None = vast.credit_of(client)
    except Exception:  # noqa: BLE001 - the bill is worth a note, not a stop
        credit = None
    if instance is None:
        return Watch(
            at=now,
            instance_alive=False,
            uptime_h=previous.uptime_h if previous else 0.0,
            cost_usd=previous.cost_usd if previous else 0.0,
            credit_usd=credit,
            notes=["the instance no longer exists"],
        )
    uptime = instance.uptime_hours(now)
    watch = Watch(
        at=now,
        instance_alive=True,
        uptime_h=uptime,
        cost_usd=uptime * instance.dph_total,
        credit_usd=credit,
    )
    probe = (
        f"cat {remote_out}/status.json 2>/dev/null; echo; echo @@GPU;"
        " nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total"
        " --format=csv,noheader 2>/dev/null | tr '\\n' ';'; echo; echo @@DISK;"
        f" df -BG {REMOTE_ROOT} 2>/dev/null | awk 'NR==2{{print $4}}'; echo @@MARK;"
        f" ls {remote_out}/campaign.done {remote_out}/campaign.failed 2>/dev/null; echo @@LOG;"
        f" tail -n 4 {remote_out}/campaign.log 2>/dev/null"
    )
    try:
        text = run_on(machine, probe, what="watch", timeout=90)
    except Exception as error:  # noqa: BLE001 - a bad minute of the proxy is a note
        watch.notes.append(f"ssh failed: {str(error)[:120]}")
        if previous is not None:
            watch.status = previous.status
        return watch
    head, _, rest = text.partition("@@GPU")
    gpu_text, _, rest = rest.partition("@@DISK")
    disk_text, _, rest = rest.partition("@@MARK")
    mark_text, _, log_text = rest.partition("@@LOG")
    try:
        watch.status = json.loads(head.strip()) if head.strip() else {}
    except json.JSONDecodeError:
        watch.status = previous.status if previous else {}
        watch.notes.append("status.json unreadable")
    watch.gpu = gpu_text.strip().rstrip(";")
    digits = "".join(ch for ch in disk_text.strip() if ch.isdigit())
    watch.disk_free_gb = float(digits) if digits else None
    watch.done = "campaign.done" in mark_text
    watch.failed = "campaign.failed" in mark_text
    watch.log_tail = log_text.strip()
    stage = str(watch.status.get("stage", "start"))
    updated = float(watch.status.get("updated", now))
    if not watch.done and not watch.failed and now - updated > STALL_S.get(stage, 3600.0):
        watch.stalled = True
        watch.notes.append(f"status not updated for {(now - updated) / 60:.0f} min in {stage}")
    if watch.disk_free_gb is not None and watch.disk_free_gb < 20 and not watch.done:
        watch.notes.append("under 20 GB of disk free")
    if stage == "solve" and watch.gpu:
        utilisation = watch.gpu.split(",")[0].replace("%", "").strip()
        encoding = "encode" in str(watch.status.get("stage_detail", ""))
        if utilisation.isdigit() and int(utilisation) == 0 and not encoding:
            watch.notes.append("card idle during a solve")
    return watch


def _launch_command(
    bundle: str, out: str, devices: str | None, pffdtd: str, extra: str = ""
) -> str:
    """A launcher script written and started detached, the way every long job here is.

    ``setsid nohup script &`` with every descriptor detached, inside a
    subshell so the ssh session has nothing left to wait for: without the
    subshell the session hung until the harness killed it, and the job died
    with it.
    """
    devices_line = f"export CUDA_VISIBLE_DEVICES={shlex.quote(devices)}\n" if devices else ""
    script = (
        "#!/bin/bash\n"
        f"cd {REMOTE_ROOT}\n"
        f"export PYTHONPATH={REMOTE_SRC}/src REVERBERATE_DATA={REMOTE_DATA}"
        " OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore\n"
        f"{devices_line}"
        f"exec {ACCEL_PYTHON} -m reverberate.accel campaign --bundle {bundle} --out {out}"
        f" --pffdtd {pffdtd}{(' ' + extra) if extra else ''} > {out}/driver.log 2>&1\n"
    )
    return (
        f"mkdir -p {out} && rm -f {out}/campaign.done {out}/campaign.failed && "
        f"cat > {REMOTE_ROOT}/launch.sh <<'REVERBERATE_EOF'\n{script}REVERBERATE_EOF\n"
        f"chmod +x {REMOTE_ROOT}/launch.sh && cd {REMOTE_ROOT} && "
        "(setsid nohup ./launch.sh > /dev/null 2>&1 < /dev/null &); sleep 2; echo started"
    )


def rent(
    client: Any,
    identity: Any,
    need: MachineNeed,
    *,
    hours: float,
    max_dph: float,
    min_ram_gb: float,
    gpu: str,
    say: Any,
) -> tuple[Any, int, Any]:
    """The cheapest host that holds the campaign: the machine, its id and the offer taken."""
    offers: list[Any] = []
    for count in (1, 2, 4):
        offers += client.search(
            vast.search_query(
                gpu_name="",
                num_gpus=count,
                min_disk_gb=need.disk_gb,
                min_gpu_ram_gb=16,
                min_reliability=0.97,
                min_inet_down_mbps=300,
                min_cpu_cores=8,
            ),
            limit=400,
        )
    good = choose_offers(offers, need, max_dph=max_dph, min_ram_gb=min_ram_gb, gpu=gpu)
    if not good:
        raise SystemExit(f"no offer under {max_dph} USD/h fits: {need.describe()}")
    for offer in good[:6]:
        say("  " + offer.describe())
    say(f"  cap {hours:g} h -> at most {vast.estimate_cost_usd(good[0].dph_total, hours):.2f} USD")
    machine, instance = vast.rent_one(
        client, identity, good, hours=hours, disk_gb=need.disk_gb, image=IMAGE, say=say
    )
    return machine, instance, good[0]


def provision_machine(machine: Any, repo: Path, bundle: Path, say: Any) -> dict[str, float]:
    """The engine built with its patches, the interpreter with cupy, the code, the bundle."""
    t0 = time.time()
    build_s = provision(machine, repo / "scripts" / "build_pffdtd.sh", beside=engine_patches(repo))
    say(f"engine built in {build_s / 60:.1f} min")
    run_on(
        machine, f"mkdir -p {REMOTE_SRC} {REMOTE_BUNDLE} {REMOTE_OUT} {REMOTE_DATA}", what="mkdir"
    )
    rsync(
        machine,
        [str(repo / "requirements-remote.txt"), str(repo / "scripts" / "provision_accel.sh")],
        "/root/",
        download=False,
    )
    run_on(machine, "bash /root/provision_accel.sh 2>&1 | tail -6", what="provision", timeout=3600)
    rsync(machine, [str(repo / "src"), str(repo / "scripts")], REMOTE_SRC + "/", download=False)
    provision_s = round(time.time() - t0, 1)
    say(f"provisioned in {provision_s / 60:.1f} min")
    t0 = time.time()
    rsync(machine, [str(bundle) + "/"], REMOTE_BUNDLE + "/", download=False)
    push_s = round(time.time() - t0, 1)
    say(f"bundle pushed in {push_s / 60:.1f} min")
    return {"provision_s": provision_s, "push_s": push_s}


def watch(
    client: Any,
    instance: int,
    machine: Any,
    launch: Any,
    *,
    deadline: float,
    poll_s: float,
    record: dict[str, Any],
    home: Path,
    say: Any,
) -> str:
    """Look every ``poll_s`` until the campaign ends; the outcome, and the watches in ``record``.

    A failed or stalled campaign is relaunched once from its state on disk;
    the second time it is the outcome.
    """
    watches: list[Watch] = []
    relaunched = 0
    while True:
        look = monitor_once(client, instance, machine, previous=watches[-1] if watches else None)
        watches.append(look)
        record["watches"].append(look.line())
        say(look.line())
        (home / "onebox.json").write_text(json.dumps(record, indent=1, default=str))
        if not look.instance_alive:
            return "instance vanished"
        if look.done:
            return "done"
        if look.failed or look.stalled:
            say(f"campaign {'failed' if look.failed else 'stalled'}: {look.log_tail[-400:]}")
            if relaunched >= 1:
                return "failed"
            relaunched += 1
            say("relaunching once from its state on disk")
            # The bracket keeps pkill from matching the shell that runs it.
            run_on(machine, "pkill -f '[r]everberate.accel campaign'; true", what="kill")
            time.sleep(10)
            launch()
        if time.time() > deadline - 1200:
            say("within 20 minutes of the rental's deadline; fetching what exists")
            return "deadline"
        time.sleep(poll_s)


def fetch(machine: Any, bundle: Path, home: Path, *, fetch_cache: bool, say: Any) -> Path:
    """What the laptop keeps, into ``home/pulled``, and the grids it lacks into ``home/cache``.

    The first fetch of a whole run pulled 21 GB in 45 min, of which the
    field and the audit view were 6.6 GB; the encodings, the self-check
    samples and grids the laptop already held were the rest.
    """
    pulled = home / "pulled"
    pulled.mkdir(exist_ok=True)
    listing = run_on(machine, f"ls -1 {REMOTE_OUT}", what="list run", timeout=90).split()
    items = fetch_items(listing)
    rsync(machine, [f"{REMOTE_OUT}/{item}" for item in items], str(pulled) + "/", download=True)
    if "selfcheck" in listing:
        (pulled / "selfcheck").mkdir(exist_ok=True)
        try:
            rsync(
                machine,
                [f"{REMOTE_OUT}/selfcheck/{pattern}" for pattern in SELFCHECK_PATTERNS],
                str(pulled / "selfcheck") + "/",
                download=True,
            )
        except Exception as error:  # noqa: BLE001 - reports, not the field
            say(f"self-check reports not fetched: {str(error)[:200]}")
    if fetch_cache:
        spec = json.loads((bundle / "campaign.json").read_text())
        keys = [str(band["cache_key"]) for band in spec["bands"].values()]
        missing = missing_grids(keys, cache_root())
        if missing:
            (home / "cache").mkdir(exist_ok=True)
            try:
                rsync(
                    machine,
                    [f"{REMOTE_DATA}/cache/vox/{key}" for key in missing],
                    str(home / "cache") + "/",
                    download=True,
                )
            except Exception as error:  # noqa: BLE001 - the field matters more than the cache
                say(f"cache not fetched: {str(error)[:200]}")
        else:
            say("grids: every key of the bundle is installed here already, none fetched")
    return pulled


def run(
    bundle: Path,
    home: Path,
    *,
    hours: float,
    max_dph: float,
    yes: bool,
    repo: Path,
    instance: int | None = None,
    devices: str | None = None,
    poll_s: float = POLL_S,
    fetch_cache: bool = True,
    min_ram_gb: float = 60.0,
    gpu: str = "",
    campaign_args: str = "",
    say: Any = print,
) -> dict[str, Any]:
    """Rent, provision, push, launch, watch every five minutes, fetch, destroy.

    ``instance`` resumes on a machine already rented (the campaign resumes
    from its state on disk). Returns the record written to ``home/onebox.json``.
    """
    bundle, home, repo = Path(bundle), Path(home), Path(repo)
    home.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"bundle": str(bundle), "started": time.time(), "watches": []}
    need = campaign_need(bundle)
    say(f"need: {need.describe()}")
    auth.inject(["VASTAI_API_KEY"])
    client = vast.VastClient(timeout=60)
    identity = vast.account_identity(client)
    started = time.time()
    if instance is None:
        if not yes:
            say("nothing rented: pass --yes")
            return record
        machine, instance, offer = rent(
            client,
            identity,
            need,
            hours=hours,
            max_dph=max_dph,
            min_ram_gb=min_ram_gb,
            gpu=gpu,
            say=say,
        )
        record["instance"] = instance
        record["offer"] = offer.id
        (home / "onebox.json").write_text(json.dumps(record, indent=1, default=str))
        record.update(provision_machine(machine, repo, bundle, say))
    else:
        machine = vast.wait_for_ssh(client, instance, identity)
        record["instance"] = instance
        rsync(machine, [str(repo / "src")], REMOTE_SRC + "/", download=False)

    def launch() -> None:
        started_text = run_on(
            machine,
            _launch_command(REMOTE_BUNDLE, REMOTE_OUT, devices, "/root/pffdtd", campaign_args),
            what="launch",
        )
        say(f"campaign launched ({started_text.strip()[-40:]})")

    launch()
    record["outcome"] = watch(
        client,
        instance,
        machine,
        launch,
        deadline=started + hours * 3600.0,
        poll_s=poll_s,
        record=record,
        home=home,
        say=say,
    )
    t0 = time.time()
    pulled = fetch(machine, bundle, home, fetch_cache=fetch_cache, say=say)
    record["fetch_s"] = round(time.time() - t0, 1)
    say(f"fetched in {(time.time() - t0) / 60:.1f} min -> {pulled}")
    if record["outcome"] == "done":
        found = client.instance(instance)
        record["hours_billed"] = round(found.uptime_hours(), 3) if found else None
        record["cost_usd"] = round(found.uptime_hours() * found.dph_total, 3) if found else None
        gone = vast.teardown(client, instance)
        record["destroyed"] = gone
        say(f"instance {instance} destroyed={gone}, cost about {record['cost_usd']} USD")
    else:
        say(f"instance {instance} KEPT for inspection; outcome {record['outcome']}")
    record["total_s"] = round(time.time() - started, 1)
    (home / "onebox.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True, help="where the run comes home")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--max-dph", type=float, default=3.0)
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--poll", type=float, default=POLL_S)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--min-ram-gb", type=float, default=60.0, help="hosts under this much RAM are skipped"
    )
    parser.add_argument("--gpu", default="", help="only cards whose name contains this")
    parser.add_argument(
        "--campaign-args", default="", help="extra flags for the campaign, e.g. a flow test's bands"
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.bundle,
        args.home,
        hours=args.hours,
        max_dph=args.max_dph,
        yes=args.yes,
        repo=args.repo,
        instance=args.instance,
        devices=args.devices,
        poll_s=args.poll,
        fetch_cache=not args.no_cache,
        min_ram_gb=args.min_ram_gb,
        gpu=args.gpu,
        campaign_args=args.campaign_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
