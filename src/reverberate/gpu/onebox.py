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

A host is looked at before it is used. Its cards are queried the moment ssh
answers, and a host on which another tenant holds memory is destroyed and the
next offer taken: one such host carried about 70 GB in use on each card, and
the campaign died at the encode, out of memory, after the solves were paid
for. Hosts refused or silent in a run are not offered to it again, and a list
of offers and machines to avoid can be given from outside.

Two campaigns cost five to ten times their compute to idle cards, transfers
on the card's clock and boxes rented after the solves. Here there is one
rental, the transfers are the bundle up (a few hundred megabytes) and the
field down, and the card is released the moment the field is verified home.

**A rental never ends unaccounted for.** A host that cannot be provisioned
(its connection dropped three times on 2026-10-05, and one host refused
every connection) is destroyed, verified, avoided, and the next offer is
taken. After the launch, a failure of the driver itself destroys the
instance and verifies it; where the instance is left on purpose (a campaign
that failed twice, kept for inspection; a finished run whose fetch failed),
the last line names it, says why, and what it bills until its watchdog.

**The machine is chosen by what the run will cost, not by its hourly
price**, when the caller can say (``predict``): each offer's wall hours and
USD for this run, the lowest USD within ``max_hours`` taken, and the
watchdog's cap set from the prediction with a margin, since it can never be
extended once the machine is rented. What can be brought home while the
campaign runs (``sync``) is, every few minutes, so that a host that dies
takes only its last minutes with it.

**What comes home comes in chunks, the pack first.** A machine bills while
the laptop downloads: the first whole scene's pack was forty minutes of an
eight card host at 1.38 USD an hour. The large files go through
:mod:`reverberate.gpu.homecoming` (ranges of 32 MB on four kept
connections, resumed and verified, past the proxy where the instance has
an address of its own), in the order of what the run is for: the pack,
then the reports, then the pair cache if it was asked for.

**The fastest way that works on the host is taken, and the pack does not
wait for the campaign's end** (:mod:`reverberate.gpu.transfer`). A rental
asks its host to map one port more; where the host does, a server on the
machine gives the run's output folder alone, over TLS, to the bearer of a
token made for the run, and the pack comes in as many ranges at once as
still bring it faster. Where it does not, ssh at the machine's own address,
then the proxy. While the campaign runs the machine is looked at every
:data:`EARLY_POLL_S`: the pack is brought the moment it is written, while
the campaign checks it, and the campaign's end is seen then and not at the
next look of five minutes.

**A campaign that failed says what to do.** The last lines of a run that
leaves its machine rented are what is on it, what a resume keeps and makes
again, the command that resumes it, and what an hour of leaving it costs.
"""

from __future__ import annotations

import json
import shlex
import time
import traceback
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import auth
from reverberate.accel.bundle import HOME_ITEMS
from reverberate.accel.lattice import sim_constants
from reverberate.accel.solve import OUTPUT_SAMPLE_BYTES_PATCHED
from reverberate.gpu import direct, transfer, vast
from reverberate.gpu.homecoming import WORKERS, fastest, fetch_file, fetch_tree
from reverberate.wave.remote import one_at_a_time, run_on
from reverberate.wave.remote_voxelise import grid_shape_of, provision, rsync
from reverberate.wave.voxelise import cache_root

__all__ = [
    "MachineNeed",
    "Priced",
    "Watch",
    "campaign_need",
    "cap_hours",
    "cards_in_use",
    "choose_offers",
    "fetch",
    "monitor_once",
    "occupied_cards",
    "price_offers",
    "provision_machine",
    "rent",
    "run",
    "search_offers",
    "sync_home",
    "watch",
]

#: The image the engine is built in; sm_80 to sm_90 cards, CUDA 12.4.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

#: VRAM per grid node and the engine's fixed overhead, measured on an A100 80 GB.
VRAM_PER_NODE_B = 9.027
VRAM_FIXED_GB = 2.13

#: What a card may hold before anything of ours runs on it, in MiB. An idle
#: card reports a few MiB, a desktop session some hundreds; another tenant's
#: job reports gigabytes.
CARD_USED_LIMIT_MIB = 1024.0
#: One line a card: memory in use, memory in total, in MiB; then whether CUDA starts at
#: all on the host, ``cuInit`` and the driver's answer, 0 when it does. ``nvidia-smi`` lists
#: the cards of a host whose ``/dev/nvidia-uvm`` cannot be opened: machine 152135 answered
#: 999 on 2026-10-05, and a campaign there would have had no card.
CARD_QUERY = (
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
    " && python3 -c \"import ctypes; print('cuInit', ctypes.CDLL('libcuda.so.1').cuInit(0))\""
)

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


#: Of what comes home, what is single precision samples: sent as it is, not deflated.
FETCH_AS_IS = ("pack.h5", "pairs")
#: The order of a fetch: what the run is for, then its reports, then what the next run
#: would not solve again. A fetch that is cut has brought the first of these it could.
FETCH_FIRST = ("pack.h5", "field")
FETCH_LAST = ("pairs",)


def fetch_items(
    present: list[str], leave: Collection[str] = (), also: Collection[str] = ()
) -> list[str]:
    """The entries of the run directory worth the transfer, among those present.

    ``leave`` names entries that stay on the machine: a trace's pair cache
    is as large as its pack, and the pack holds every response a render reads.
    ``also`` names entries brought besides: a trace's early tables, which a
    second trace of the recipe does not make again.
    """
    wanted = (*FETCH_ITEMS, *(item for item in also if item not in FETCH_ITEMS))
    return [item for item in wanted if item in present and item not in leave]


def fetch_order(items: list[str]) -> list[str]:
    """``items`` in the order they come: :data:`FETCH_FIRST`, the rest, :data:`FETCH_LAST`."""
    first = [item for item in FETCH_FIRST if item in items]
    last = [item for item in FETCH_LAST if item in items]
    return [*first, *(item for item in items if item not in first and item not in last), *last]


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
#: How often the machine is asked, on a kept connection, whether the pack is written and
#: whether the campaign has ended (:class:`reverberate.gpu.transfer.Early`), s. A look of
#: :data:`POLL_S` found a finished campaign two and a half minutes late in the mean, on a
#: machine billed meanwhile.
EARLY_POLL_S = 20.0
#: What is brought home as soon as it is written, before the campaign's end.
EARLY_ITEMS = ("pack.h5",)
#: The floor on what a host says its line sends, megabits a second: 25 MB/s were it true,
#: under which a pack of 9.5 GB keeps its machine six minutes and more. Of 248 hosts
#: offered under 0.25 USD/h on 2026-10-05, every one said more.
MIN_INET_UP_MBPS = 200
#: How often what ``sync`` names is brought home while the campaign runs, and how long one
#: such pass may take: every look, and less than the time between two looks, so that the
#: watch is never late for a transfer. A pass that reaches its limit is ended and keeps
#: every file that arrived whole; the next goes on from there. The whole scene's first run
#: (2026-10-05) gave each pass ten minutes every ten: no pass ended, 846 pairs of 1.9 MB
#: came in 1.5 h, and the looks were fifteen minutes apart instead of five.
SYNC_S = 300.0
SYNC_TIMEOUT_S = 270.0
#: Of the time between two looks, what a pass may take at most.
SYNC_SHARE = 0.9
#: Files a campaign is still writing: they appear whole under their own name.
PARTIAL_PATTERNS = ("*.partial.npy", "*.partial.h5", "*.partial.npz")
#: A pair of the cache, in either form it is kept in (:class:`reverberate.accel.pairs.PairCache`).
PAIR_SUFFIXES = (".npy", ".npz")
#: Hosts tried, at most, when provisioning fails on one after the other.
PROVISION_HOSTS = 4
#: The watchdog's cap from a predicted run: the prediction times a factor, and half an
#: hour. A cap can never be extended once the machine is rented: too small destroys the
#: run, too large only bounds what a forgotten machine can bill. The factor is larger
#: where the prediction rests on a card nobody measured.
CAP_FACTOR_MEASURED = 1.5
CAP_FACTOR_ESTIMATED = 2.0
CAP_MARGIN_H = 0.5
#: The cap when nobody predicts.
DEFAULT_HOURS = 8.0
#: The offers shown, and handed to the renter, best first.
OFFERS_SHOWN = 6

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
    avoid: Collection[int] = (),
    min_gpus: int = 1,
) -> list[Any]:
    """Offers whose cards together hold the grid, cheapest first, sized by the whole host.

    ``gpu`` restricts the card's name to those containing it (``A100``).
    ``avoid`` drops the offers named there, by their own id or their host's.
    ``min_gpus`` is the fewest cards a host may have.
    """
    avoided = set(avoid)
    good = [
        o
        for o in offers
        if not vast.offer_ids(o) & avoided
        and o.num_gpus >= min_gpus
        and o.num_gpus * o.gpu_ram_gb >= need.vram_gb
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


@dataclass(frozen=True)
class Priced:
    """An offer with what the caller predicts the run costs on it."""

    offer: Any
    hours: float
    usd: float
    #: Whether every constant of the prediction was measured on this kind of card.
    measured: bool = True
    note: str = ""

    def line(self) -> str:
        return (
            f"{self.hours:6.2f} h  {self.usd:6.2f} USD"
            f"  {'measured ' if self.measured else 'ESTIMATED'}  {self.offer.describe()}"
            f"{('  [' + self.note + ']') if self.note else ''}"
        )


def cap_hours(priced: Priced) -> float:
    """The watchdog's cap for a predicted run, in hours: see :data:`CAP_FACTOR_MEASURED`."""
    factor = CAP_FACTOR_MEASURED if priced.measured else CAP_FACTOR_ESTIMATED
    return round(priced.hours * factor + CAP_MARGIN_H, 2)


def price_offers(
    offers: list[Any],
    predict: Callable[[Any], dict[str, Any] | None],
    *,
    max_hours: float | None = None,
    say: Any = None,
) -> list[Priced]:
    """The offers by what the run is predicted to cost on each, the lowest USD first.

    ``predict(offer)`` answers ``hours`` and ``usd``, with ``measured`` and
    a ``note`` if it likes, or ``None`` for a card it cannot price or the
    run does not fit. Offers over ``max_hours`` of wall time are dropped;
    what was dropped, and why, is said.
    """
    say = say or (lambda message: None)
    priced: list[Priced] = []
    unpriced = slow = 0
    for offer in offers:
        told = predict(offer)
        if told is None:
            unpriced += 1
            continue
        found = Priced(
            offer=offer,
            hours=float(told["hours"]),
            usd=float(told["usd"]),
            measured=bool(told.get("measured", True)),
            note=str(told.get("note", "")),
        )
        if max_hours is not None and found.hours > max_hours:
            slow += 1
            continue
        priced.append(found)
    if unpriced:
        say(
            f"  {unpriced} offer(s) left out: a card the prediction has no figure for, or too small"
        )
    if slow:
        say(f"  {slow} offer(s) left out: predicted over the {max_hours:g} h of wall time allowed")
    return sorted(priced, key=lambda p: (p.usd, p.hours, -p.offer.reliability))


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

    **Launched twice, it runs once.** The launcher writes its process id,
    which the campaign keeps, and a launch first ends the campaign that id
    names: a launch whose connection dropped after it started is tried
    again, and a run resumed on its machine may find its campaign alive.
    Two campaigns on one card halve each other and write the same files.
    """
    devices_line = f"export CUDA_VISIBLE_DEVICES={shlex.quote(devices)}\n" if devices else ""
    pid = f"{REMOTE_ROOT}/campaign.pid"
    script = (
        "#!/bin/bash\n"
        f"cd {REMOTE_ROOT}\n"
        f"echo $$ > {pid}\n"
        f"export PYTHONPATH={REMOTE_SRC}/src REVERBERATE_DATA={REMOTE_DATA}"
        " OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore\n"
        f"{devices_line}"
        f"exec {ACCEL_PYTHON} -m reverberate.accel campaign --bundle {bundle} --out {out}"
        f" --pffdtd {pffdtd}{(' ' + extra) if extra else ''} > {out}/driver.log 2>&1\n"
    )
    return (
        f'if [ -f {pid} ] && kill -0 "$(cat {pid})" 2>/dev/null; then'
        f' kill "$(cat {pid})"; sleep 5; fi; '
        f"mkdir -p {out} && rm -f {out}/campaign.done {out}/campaign.failed && "
        f"cat > {REMOTE_ROOT}/launch.sh <<'REVERBERATE_EOF'\n{script}REVERBERATE_EOF\n"
        f"chmod +x {REMOTE_ROOT}/launch.sh && cd {REMOTE_ROOT} && "
        "(setsid nohup ./launch.sh > /dev/null 2>&1 < /dev/null &); sleep 2; echo started"
    )


def cards_in_use(text: str) -> list[tuple[float, float]]:
    """Memory in use and in total on each card, in MiB, from :data:`CARD_QUERY`'s answer."""
    cards = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("cuInit"):
            continue
        used, total = (float(value) for value in line.split(","))
        cards.append((used, total))
    return cards


def cuda_refused(text: str) -> str | None:
    """Why CUDA does not start on the host, from :data:`CARD_QUERY`; ``None`` when it does."""
    for line in text.splitlines():
        if line.startswith("cuInit") and line.split()[-1] != "0":
            return (
                f"CUDA cannot be initialised on this host ({line.strip()}): nvidia-smi lists"
                " its cards and no program can open them"
            )
    return None


def occupied_cards(machine: Any, say: Any, *, limit_mib: float = CARD_USED_LIMIT_MIB) -> str | None:
    """Why the host's cards cannot be used, or ``None`` when every one of them is empty.

    Asked once, before anything is built or pushed. A host whose cards do not
    answer the query is refused like one whose cards are held.
    """
    try:
        answer = run_on(machine, CARD_QUERY, what="cards", timeout=90)
        cards = cards_in_use(answer)
    except Exception as error:  # noqa: BLE001 - a host that cannot be asked is not used
        return f"the cards did not answer: {str(error)[:120]}"
    if not cards:
        return "the host reports no card"
    dead = cuda_refused(answer)
    if dead:
        return dead
    say(
        "card memory in use before anything runs: "
        + ", ".join(f"{used:.0f} of {total:.0f} MiB" for used, total in cards)
    )
    held = [f"card {i} {used:.0f} MiB" for i, (used, _) in enumerate(cards) if used > limit_mib]
    if held:
        return f"cards held by someone else, over {limit_mib:.0f} MiB in use: {', '.join(held)}"
    return None


#: The card counts a search asks for; an offer is one count of a host's cards.
CARD_COUNTS = (1, 2, 3, 4, 6, 8)


def search_offers(client: Any, need: MachineNeed, *, min_gpus: int = 1) -> list[Any]:
    """Every offer that could hold the campaign, asked a card count at a time. Read only."""
    offers: list[Any] = []
    for count in CARD_COUNTS:
        if count < min_gpus:
            continue
        offers += client.search(
            vast.search_query(
                gpu_name="",
                num_gpus=count,
                min_disk_gb=need.disk_gb,
                min_gpu_ram_gb=16,
                min_reliability=0.97,
                min_inet_down_mbps=300,
                min_cpu_cores=8,
                min_inet_up_mbps=MIN_INET_UP_MBPS,
            ),
            limit=400,
        )
    return offers


def prefer(found: list[Any], regions: Sequence[str], say: Any, *, what: Any) -> list[Any]:
    """``found`` with the offers of the preferred regions first; their own order within.

    A preference and not a filter: no offer is dropped, and the order
    given, the lowest total first, is kept among the offers of one region
    and among those of none. A host is preferred for its line to the
    laptop, so only where the transfer can go to it directly: an offer
    that says it has no open port is ranked as any other. ``what`` says an
    entry in a line; what the preference changes of the first choice is
    said.
    """
    wanted = [word for word in regions if word.strip()]
    if not wanted or not found:
        return found

    def offer_of(entry: Any) -> Any:
        return getattr(entry, "offer", entry)

    def rank(entry: Any) -> int:
        offer = offer_of(entry)
        if int(getattr(offer, "direct_ports", 1) or 0) <= 0:
            return len(wanted)
        return direct.region_rank(str(getattr(offer, "location", "") or ""), wanted)

    ranked = sorted(found, key=rank)
    named = ", ".join(wanted)
    if rank(ranked[0]) == len(wanted):
        say(f"  no offer in the preferred region(s) {named}: the order is the totals' own")
    elif ranked[0] is not found[0]:
        say(
            f"  preferred region(s) {named}: {what(ranked[0])} goes first;"
            f" the lowest total anywhere is {what(found[0])}"
        )
    else:
        say(f"  preferred region(s) {named}: the lowest total is there already")
    return ranked


@dataclass
class RentalPlan:
    """The offers to try, best first, the watchdog's cap, and why."""

    offers: list[Any]
    hours: float
    priced: list[Priced] = field(default_factory=list)

    def record(self) -> list[dict[str, Any]]:
        return [
            {
                "offer": int(p.offer.id),
                "hours": round(p.hours, 3),
                "usd": round(p.usd, 3),
                "measured": p.measured,
                "note": p.note,
            }
            for p in self.priced
        ]


def plan_rental(
    client: Any,
    need: MachineNeed,
    *,
    hours: float | None,
    max_dph: float,
    min_ram_gb: float,
    gpu: str,
    say: Any,
    avoid: Collection[int] = (),
    min_gpus: int = 1,
    predict: Callable[[Any], dict[str, Any] | None] | None = None,
    max_hours: float | None = None,
    prefer_regions: Sequence[str] = (),
) -> RentalPlan:
    """Search, choose, and say the choice and its reasons; nothing is rented.

    ``prefer_regions`` puts the offers of those regions first, in the
    order given (:func:`prefer`: country codes or place names, as
    :func:`reverberate.gpu.direct.region_rank` reads them); nothing is
    left out for it.

    Without ``predict`` the cheapest hour that fits is first, as before.
    With it, each offer is priced for this run and the lowest predicted
    USD within ``max_hours`` is first; the table is said before anything
    is rented. ``hours`` is the watchdog's cap; ``None`` takes it from the
    prediction (:func:`cap_hours`), the largest among the offers that may
    be tried, since the cap is set before it is known which one answers.
    """
    offers = search_offers(client, need, min_gpus=min_gpus)
    good = choose_offers(
        offers,
        need,
        max_dph=max_dph,
        min_ram_gb=min_ram_gb,
        gpu=gpu,
        avoid=avoid,
        min_gpus=min_gpus,
    )
    if not good:
        raise SystemExit(
            f"no offer under {max_dph} USD/h"
            f"{f' with {min_gpus} card(s) or more' if min_gpus > 1 else ''} fits: {need.describe()}"
        )
    if predict is None:
        cap = DEFAULT_HOURS if hours is None else float(hours)
        good = prefer(
            good,
            prefer_regions,
            say,
            what=lambda o: f"offer {o.id} ({getattr(o, 'location', '')}, {o.dph_total:.3f} USD/h)",
        )
        for offer in good[:OFFERS_SHOWN]:
            say("  " + offer.describe())
        say(f"  chosen by hourly price: no prediction of this run was given; cap {cap:g} h")
        say(f"  cap {cap:g} h -> at most {vast.estimate_cost_usd(good[0].dph_total, cap):.2f} USD")
        return RentalPlan(offers=list(good), hours=cap)
    priced = price_offers(good, predict, max_hours=max_hours, say=say)
    if not priced:
        raise SystemExit(
            f"no offer is predicted to end within {max_hours} h of wall time"
            if max_hours is not None
            else "no offer could be priced for this run"
        )
    priced = prefer(
        priced,
        prefer_regions,
        say,
        what=lambda p: (
            f"offer {p.offer.id} ({getattr(p.offer, 'location', '')}),"
            f" {p.hours:.2f} h and {p.usd:.2f} USD"
        ),
    )
    shown = priced[:OFFERS_SHOWN]
    say(
        "  predicted for this run, the lowest total first"
        + (" within a preferred region" if any(w.strip() for w in prefer_regions) else "")
        + " (wall hours, USD, offer):"
    )
    for found in shown:
        say("  " + found.line())
    first = shown[0]
    by_hour = min(shown, key=lambda p: p.offer.dph_total)
    reason = (
        f"  chosen: offer {first.offer.id}, {first.hours:.2f} h and {first.usd:.2f} USD predicted"
    )
    if by_hour is not first:
        reason += (
            f"; the cheapest hour shown ({by_hour.offer.id}, {by_hour.offer.dph_total:.3f} USD/h)"
            f" would take {by_hour.hours:.2f} h and {by_hour.usd:.2f} USD"
        )
    say(reason)
    if hours is None:
        cap = max(cap_hours(found) for found in shown)
        say(
            f"  watchdog: {cap:g} h, the prediction times {CAP_FACTOR_MEASURED:g}"
            f" ({CAP_FACTOR_ESTIMATED:g} for a card not measured) and {CAP_MARGIN_H:g} h, the"
            f" largest of the offers shown; at most"
            f" {vast.estimate_cost_usd(first.offer.dph_total, cap):.2f} USD on the first"
        )
    else:
        cap = float(hours)
        say(f"  watchdog: {cap:g} h as given, against {first.hours:.2f} h predicted")
        if cap < first.hours:
            say("  THE CAP GIVEN IS UNDER THE PREDICTION: the watchdog would destroy the run")
    return RentalPlan(offers=[p.offer for p in shown], hours=cap, priced=shown)


def rent(
    client: Any,
    identity: Any,
    need: MachineNeed,
    *,
    hours: float | None,
    max_dph: float,
    min_ram_gb: float,
    gpu: str,
    say: Any,
    avoid: set[int] | None = None,
    plan: RentalPlan | None = None,
    remaining_usd: float = 0.0,
) -> tuple[Any, int, Any]:
    """The best host that holds the campaign: the machine, its id and the offer taken.

    Offers named in ``avoid``, by their own id or their host's, are not
    rented. A host whose cards are not empty when ssh answers is destroyed and
    the next offer tried; it and any host that stayed silent are added to
    ``avoid``, which the caller keeps. ``plan`` is :func:`plan_rental`'s,
    made here when not given.
    """
    avoid = set() if avoid is None else avoid
    if plan is None:
        plan = plan_rental(
            client,
            need,
            hours=hours,
            max_dph=max_dph,
            min_ram_gb=min_ram_gb,
            gpu=gpu,
            say=say,
            avoid=avoid,
        )
    good = [offer for offer in plan.offers if not vast.offer_ids(offer) & avoid]
    if not good:
        raise SystemExit("every offer of the plan is on a host to avoid")
    # ``rent_one`` removes every offer it tries from the list, so the one it
    # kept is the last it removed, and the list is empty when that was the last.
    ranked = list(good)
    machine, instance = vast.rent_one(
        client,
        identity,
        good,
        hours=plan.hours,
        disk_gb=need.disk_gb,
        image=IMAGE,
        remaining_usd=remaining_usd,
        avoid=avoid,
        refuse=lambda machine: occupied_cards(machine, say),
        say=say,
        # One port more than ssh's on the host: the run's files come home by it, in ranges.
        ports=(transfer.HOMECOMING_PORT,),
    )
    return machine, instance, ranked[len(ranked) - len(good) - 1]


#: Where the engine is not built, the one thing of its build a campaign still wants: a C
#: compiler, for the text the mirror's paths are sieved by. The image is expected to hold
#: one; where it does not, the packages are installed, and a machine that cannot install
#: them runs those paths on ``numpy`` and says so.
COMPILER_COMMAND = (
    "command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1"
    " || (apt-get update -qq && apt-get install -y -qq build-essential >/dev/null)"
    " || echo 'no C compiler could be installed'"
)


def provision_machine(
    machine: Any, repo: Path, bundle: Path, say: Any, *, engine_build: bool = True
) -> dict[str, Any]:
    """The engine built with its patches, the interpreter with cupy, the code, the bundle.

    ``engine_build`` false leaves PFFDTD out: packages, a clone, eight
    patches, an interpreter of its own and four CUDA binaries, 216 and 498 s
    on the two hosts of 2026-10-05
    (``docs/open-questions/performance-audit.md``, section 7). A campaign
    whose solver is ``wave.lowband`` on its Cartesian grid opens nothing of
    it; the present engine and the face centred grid's voxeliser do.
    """
    t0 = time.time()
    if engine_build:
        build_s = provision(
            machine, repo / "scripts" / "build_pffdtd.sh", beside=engine_patches(repo)
        )
        say(f"engine built in {build_s / 60:.1f} min")
    else:
        say("engine not built: this campaign's solver does not open it")
        run_on(machine, COMPILER_COMMAND, what="compiler", timeout=900)
    run_on(
        machine, f"mkdir -p {REMOTE_SRC} {REMOTE_BUNDLE} {REMOTE_OUT} {REMOTE_DATA}", what="mkdir"
    )
    rsync(
        machine,
        [str(repo / "requirements-remote.txt"), str(repo / "scripts" / "provision_accel.sh")],
        "/root/",
        download=False,
    )
    run_on(
        machine,
        # Idempotent, and behind a lock: a dropped connection tries it again.
        one_at_a_time("bash /root/provision_accel.sh") + " 2>&1 | tail -6",
        what="provision",
        timeout=3600,
    )
    # Past the proxy where the machine has an address of its own: the proxy carried 3 to 5
    # MB/s from the laptop where the direct route carried the laptop's own 10.
    way = fastest(machine, say=say)
    push(way, machine, [str(repo / "src"), str(repo / "scripts")], REMOTE_SRC + "/", say)
    provision_s = round(time.time() - t0, 1)
    say(f"provisioned in {provision_s / 60:.1f} min")
    t0 = time.time()
    push(way, machine, [str(bundle) + "/"], REMOTE_BUNDLE + "/", say)
    push_s = round(time.time() - t0, 1)
    say(f"bundle pushed in {push_s / 60:.1f} min")
    return {"provision_s": provision_s, "push_s": push_s, "engine_built": bool(engine_build)}


def push(way: Any, machine: Any, sources: list[str], destination: str, say: Any) -> None:
    """``sources`` up to the machine by ``way``, and through the proxy where that fails."""
    if way is not machine:
        try:
            rsync(way, sources, destination, download=False, attempts=2)
            return
        except Exception as error:  # noqa: BLE001 - the proxy is the way that is known to work
            say(f"  the direct way up failed ({str(error)[:120]}); through the proxy")
    rsync(machine, sources, destination, download=False)


def pairs_home(pulled: Path) -> int:
    """The pairs that arrived whole under ``pulled/pairs``."""
    return sum(1 for path in (Path(pulled) / "pairs").rglob("*.np[yz]") if _whole(path))


def _whole(path: Path) -> bool:
    return (
        path.suffix in PAIR_SUFFIXES
        and not path.name.startswith(".")
        and ".partial." not in path.name
        and ".incoming." not in str(path.parent)
    )


def clear_cut_files(pulled: Path) -> int:
    """Remove what a pass ended at its limit left of its last file; how many.

    rsync receives a file under a hidden name beside its place and renames
    it when it is whole. Ended from outside it leaves that name, which no
    later pass looks for.
    """
    removed = 0
    for path in (Path(pulled) / "pairs").rglob(".*.np[yz].*"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def pairs_on_machine(machine: Any) -> int | None:
    """The pairs the campaign has written so far; ``None`` when the machine does not say."""
    try:
        printed = run_on(
            machine,
            f"find {REMOTE_OUT}/pairs -name '*.np[yz]' ! -name '*.partial.*' 2>/dev/null | wc -l",
            what="count pairs",
            timeout=60,
        )
        return int(printed.strip().split()[-1])
    except Exception:  # noqa: BLE001 - a count is a line of the log, not the campaign
        return None


def sync_home(
    machine: Any,
    home: Path,
    items: Collection[str],
    say: Any,
    *,
    timeout: float | None = SYNC_TIMEOUT_S,
    way: Any = None,
) -> bool:
    """What is new of ``items`` in the run directory, into ``home/pulled``; whether all came.

    Incremental and bounded: a file already home is not sent again, and a
    pass that reaches ``timeout`` is ended there with every whole file it
    brought kept, which is a pass and not a failure: the next goes on.
    Files the campaign is still writing are left. The pairs home are said
    against those on the machine. Never raises: the campaign matters more
    than its copy, and the next look tries again.

    The pair cache, thousands of files of half a megabyte, comes as
    batches of whole files on a few kept connections
    (:func:`reverberate.gpu.homecoming.fetch_tree`); one ``rsync`` stream
    brought the first scene's at 0.3 to 0.5 MB/s. ``way`` is the machine as
    the transfers reach it, its own address where it has one.
    """
    if not items:
        return True
    pulled = Path(home) / "pulled"
    pulled.mkdir(parents=True, exist_ok=True)
    counted = "pairs" in items
    before = pairs_home(pulled) if counted else 0
    started = time.time()
    came, note = True, ""
    try:
        others = [item for item in items if item not in FETCH_LAST]
        if others:
            rsync(
                machine,
                [f"{REMOTE_OUT}/{item}" for item in others],
                str(pulled) + "/",
                download=True,
                compress=False,
                exclude=PARTIAL_PATTERNS,
                attempts=1,
                timeout=timeout,
            )
        for item in (item for item in items if item in FETCH_LAST):
            told = fetch_tree(
                way if way is not None else machine,
                f"{REMOTE_OUT}/{item}",
                pulled / item,
                exclude=PARTIAL_PATTERNS,
                seconds=timeout,
            )
            if "No such file" in str(told.get("error", "")) or not told.get("there"):
                # A directory the campaign has not made yet is not worth a line.
                return False
            if told.get("error"):
                raise RuntimeError(str(told["error"]))
            if not told["complete"]:
                came = False
                note = f"; the pass ended at its {timeout:g} s and the next goes on"
    except Exception as error:  # noqa: BLE001 - a copy that failed is tried again
        came = False
        if "No such file" in str(error):
            return False
        if "did not end in" in str(error):
            note = f"; the pass ended at its {timeout:g} s and the next goes on"
        else:
            say(f"  homecoming of {', '.join(items)} not complete: {str(error)[:160]}")
    if counted:
        clear_cut_files(pulled)
        now = pairs_home(pulled)
        there = pairs_on_machine(machine)
        say(
            f"  pairs home: {now} of {'?' if there is None else there} on the machine"
            f" (+{now - before} in {time.time() - started:.0f} s){note}"
        )
    elif note:
        say(f"  homecoming of {', '.join(items)}{note}")
    return came


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
    sync: Collection[str] = (),
    sync_s: float = SYNC_S,
    early: Any = None,
) -> str:
    """Look every ``poll_s`` until the campaign ends; the outcome, and the watches in ``record``.

    ``early`` (:class:`reverberate.gpu.transfer.Early`) ends the pause between two looks
    the moment the campaign's end appears on the machine, so that the look that follows
    finds it; without it the pause is always whole.

    A failed or stalled campaign is relaunched once from its state on disk;
    the second time it is the outcome. Every ``sync_s`` what ``sync`` names
    is brought home (:func:`sync_home`), within the pause between two looks.
    A look that cannot be taken (the
    API or the laptop's own line failing) is said and taken again: only
    the deadline, the campaign's end or the instance's ends the watch.
    """
    watches: list[Watch] = []
    relaunched = 0
    synced = time.time()
    # Asked once, at the first pass: the instance's own address where it answers.
    way: list[Any] = []
    while True:
        try:
            look = monitor_once(
                client, instance, machine, previous=watches[-1] if watches else None
            )
        except Exception as error:  # noqa: BLE001 - a look that failed is not the campaign's end
            say(f"look failed ({str(error)[:160]}); looking again")
            look = None
        if look is not None:
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
                try:
                    # The bracket keeps pkill from matching the shell that runs it.
                    run_on(machine, "pkill -f '[r]everberate.accel campaign'; true", what="kill")
                    time.sleep(10)
                    launch()
                except Exception as error:  # noqa: BLE001 - said; the next look judges
                    say(f"relaunch failed ({str(error)[:160]})")
                    relaunched -= 1
        if time.time() > deadline - 1200:
            say("within 20 minutes of the rental's deadline; fetching what exists")
            return "deadline"
        spent = 0.0
        if sync and time.time() - synced >= sync_s:
            # Out of the pause between two looks, never added to it.
            limit = SYNC_TIMEOUT_S if poll_s <= 0 else min(SYNC_TIMEOUT_S, SYNC_SHARE * poll_s)
            began = time.time()
            if not way:
                way.append(fastest(machine, say=say))
            came = sync_home(machine, home, sync, say, timeout=limit, way=way[0])
            spent = time.time() - began
            # Counted from the pass's start: one that took its whole limit is not skipped next.
            synced = began
            record["synced"] = {"at": began + spent, "complete": came, "items": list(sync)}
        (time.sleep if early is None else early.wait)(max(0.0, poll_s - spent))


def bring_large(
    machine: Any,
    item: str,
    pulled: Path,
    say: Any,
    fetched: dict[str, Any],
    *,
    workers: int = WORKERS,
    ways: list[Any] | None = None,
) -> None:
    """One large entry of the run home, by the first way that brings it; ``fetched`` says how.

    ``ways`` are :func:`reverberate.gpu.transfer.ways`'s, opened here and
    ended here when not given: HTTPS ranges where the host maps the port,
    the instance's own address, the proxy. A way that fails, whatever
    failed, hands over to the next with the chunks it brought, and after
    the last the entry comes by ``rsync``, the way that was. The pair
    cache is thousands of small files: it comes as batches of whole files
    over ssh, never in ranges.
    """
    end = None
    if ways is None:
        ways, end = transfer.ways(machine, REMOTE_OUT, direct=fastest(machine, say=say), say=say)
    began = time.time()
    why = ""
    try:
        for way in ways:
            tree = item in FETCH_LAST
            if tree and way.name == transfer.HTTPS:
                continue
            try:
                if tree:
                    told: dict[str, Any] = fetch_tree(
                        way.through,
                        f"{REMOTE_OUT}/{item}",
                        pulled / item,
                        exclude=PARTIAL_PATTERNS,
                        workers=workers,
                    )
                    if not told["complete"]:
                        raise RuntimeError(
                            f"{told['left']} file(s) of {item} did not come:"
                            f" {told.get('error', '')}"
                        )
                else:
                    told = fetch_file(
                        way.through, f"{REMOTE_OUT}/{item}", pulled / item, say=say, **way.options()
                    )
            except Exception as error:  # noqa: BLE001 - the next way is tried
                why = str(error)[:160]
                if way is not ways[-1]:
                    kind = "the direct way" if way.name == transfer.SSH_DIRECT else way.name
                    say(f"  {item}: {kind} failed ({why}); the next way")
                continue
            fetched[item] = {
                k: told[k]
                for k in ("bytes", "seconds", "bytes_per_s", "failures", "workers", "already_home")
                if k in told
            }
            fetched[item]["direct"] = bool(way.direct)
            fetched[item]["by"] = way.name
            return
        say(f"  {item}: not home in chunks ({why}); by rsync")
        rsync(
            machine,
            [f"{REMOTE_OUT}/{item}"],
            str(pulled) + "/",
            download=True,
            compress=False,
            exclude=PARTIAL_PATTERNS,
        )
        fetched[item] = {"seconds": round(time.time() - began, 1), "by": "rsync"}
    finally:
        if end is not None:
            end()


def fetch(
    machine: Any,
    bundle: Path,
    home: Path,
    *,
    fetch_cache: bool,
    say: Any,
    leave: Collection[str] = (),
    also: Collection[str] = (),
    record: dict[str, Any] | None = None,
    workers: int = WORKERS,
) -> Path:
    """What the laptop keeps, into ``home/pulled``, and the grids it lacks into ``home/cache``.

    The first fetch of a whole run pulled 21 GB in 45 min, of which the
    field and the audit view were 6.6 GB; the encodings, the self-check
    samples and grids the laptop already held were the rest.

    **In the order of what the run is for** (:func:`fetch_order`): the
    pack, the reports, then the pair cache where it is not left. The pack
    comes in chunks, resumed and verified against the machine's own digest
    (:func:`reverberate.gpu.homecoming.fetch_file`), and the pair cache as
    batches of whole files; each by the first way that brings it
    (:func:`bring_large`): ranges over HTTPS, ssh at the instance's own
    address, ssh through the proxy, then ``rsync``, the way that was. A
    pack that came while the campaign still ran is not brought again.
    ``record`` receives what each large transfer was (``fetched``): bytes,
    seconds, bytes a second, and the way it came by.
    """
    pulled = home / "pulled"
    pulled.mkdir(exist_ok=True)
    listing = run_on(machine, f"ls -1 {REMOTE_OUT}", what="list run", timeout=90).split()
    items = fetch_order(fetch_items(listing, leave, also))
    left = [item for item in leave if item in listing]
    if left:
        say(f"left on the machine: {', '.join(left)}")
    small = [item for item in items if item not in FETCH_AS_IS]
    large = [item for item in items if item in FETCH_AS_IS]
    fetched: dict[str, Any] = {} if record is None else record.setdefault("fetched", {})
    # The ways are opened once for all that is large, and ended when it is home.
    opened, end = (
        transfer.ways(machine, REMOTE_OUT, direct=fastest(machine, say=say), say=say)
        if large
        else ([], lambda: None)
    )

    def in_chunks(item: str) -> None:
        bring_large(machine, item, pulled, say, fetched, workers=workers, ways=opened)

    try:
        for item in (item for item in large if item not in FETCH_LAST):
            in_chunks(item)
        if small:
            rsync(
                machine,
                [f"{REMOTE_OUT}/{item}" for item in small],
                str(pulled) + "/",
                download=True,
            )
        for item in (item for item in large if item in FETCH_LAST):
            in_chunks(item)
    finally:
        end()
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


def early_homecoming(
    machine: Any,
    home: Path,
    say: Any,
    record: dict[str, Any],
    *,
    leave: Collection[str] = (),
    poll_s: float = EARLY_POLL_S,
) -> transfer.Early:
    """The looks that go on beside the watch, started: see :class:`reverberate.gpu.transfer.Early`.

    What :data:`EARLY_ITEMS` names and ``leave`` does not is brought into
    ``home/pulled`` as soon as the machine holds it whole, by
    :func:`bring_large`, and ``record`` says so (``fetched``, and
    ``early`` for when). The looks go at the machine's own address where
    it has one, on a connection that is kept.
    """
    pulled = Path(home) / "pulled"
    fetched: dict[str, Any] = record.setdefault("fetched", {})

    def bring(item: str) -> None:
        pulled.mkdir(parents=True, exist_ok=True)
        began = time.time()
        bring_large(machine, item, pulled, say, fetched)
        record.setdefault("early", {})[item] = {
            "at": time.time(),
            "seconds": round(time.time() - began, 1),
        }

    direct = getattr(machine, "directly", None)
    way = (direct() if direct is not None else None) or machine
    sharing = getattr(way, "sharing", None)
    return transfer.Early(
        way if sharing is None else sharing("early"),
        REMOTE_OUT,
        bring,
        items=[item for item in EARLY_ITEMS if item not in leave],
        poll_s=poll_s,
        say=say,
        ask=lambda *a, **k: run_on(*a, **k),
    ).start()


def campaign_done(machine: Any) -> bool:
    """Whether the campaign on ``machine`` has ended well; false where it cannot be asked."""
    try:
        said = run_on(
            machine, f"ls {REMOTE_OUT}/campaign.done 2>/dev/null; true", what="done", timeout=60
        )
    except Exception:  # noqa: BLE001 - a machine that does not answer is launched, as before
        return False
    return "campaign.done" in str(said)


def still_rented(
    client: Any,
    instance: int,
    why: str,
    *,
    deadline: float | None,
    say: Any,
    resume: str = "",
    held: Collection[str] = (),
    command: str = "",
) -> str:
    """The last lines of a run that leaves its instance alive: which, why, and what to do.

    The first line is the instance, the reason and its bill: what an hour
    of leaving it costs, and what it costs at most until its watchdog.
    ``held`` are lines of what is on the machine and what a resume makes
    again; ``command`` is the exact command that resumes the run, in which
    ``{instance}`` is the instance. Then the command that destroys it.
    """
    rate = None
    try:
        found = client.instance(instance)
        rate = None if found is None else float(found.dph_total)
    except Exception:  # noqa: BLE001 - the line is said whatever the API answers
        pass
    bill = ""
    if rate is not None:
        bill = f" It bills {rate:.3f} USD/h"
        if deadline is not None:
            left = max(deadline - time.time(), 0.0) / 3600.0
            bill += (
                f" until its watchdog at {time.strftime('%H:%M', time.localtime(deadline))}:"
                f" {left * rate:.2f} USD more at most"
            )
        bill += "."
    lines = [
        f"INSTANCE {instance} IS STILL RENTED: {why}.{bill}"
        f" Resume with --instance {instance}{(' ' + resume) if resume else ''}, or destroy it.",
        *(f"  {line}" for line in held),
    ]
    if command:
        lines.append("  resume:  " + command.replace("{instance}", str(instance)))
    lines.append(f"  destroy: python -m reverberate.gpu.vast destroy {instance}")
    line = "\n".join(lines)
    say(line)
    return line


def release(client: Any, instance: int, record: dict[str, Any], say: Any) -> bool:
    """Destroy the instance, verify it, and record its bill; whether it is verified gone."""
    try:
        found = client.instance(instance)
    except Exception:  # noqa: BLE001 - the bill is worth a note, not the teardown
        found = None
    record["hours_billed"] = round(found.uptime_hours(), 3) if found else None
    record["cost_usd"] = round(found.uptime_hours() * found.dph_total, 3) if found else None
    gone = bool(vast.teardown(client, instance))
    record["destroyed"] = gone
    say(f"instance {instance} destroyed={gone}, cost about {record['cost_usd']} USD")
    return gone


def run(
    bundle: Path,
    home: Path,
    *,
    hours: float | None = None,
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
    avoid: Collection[int] = (),
    leave: Collection[str] = (),
    also: Collection[str] = (),
    say: Any = print,
    gpus: int = 1,
    max_hours: float | None = None,
    predict: Callable[[Any], dict[str, Any] | None] | None = None,
    sync: Collection[str] = (),
    sync_s: float = SYNC_S,
    plan_only: bool = False,
    destroy_failed: bool = False,
    cap_flag: str = "",
    resume_command: str = "",
    inventory: Callable[[Any], list[str]] | None = None,
    relaunch: bool = False,
    disk_gb: int | None = None,
    engine_build: bool = True,
    prefer_regions: Sequence[str] = (),
) -> dict[str, Any]:
    """Rent, check the cards are empty, provision, push, launch, watch, fetch, destroy.

    ``prefer_regions`` orders the offers, those of the regions named first
    (:func:`prefer`): a preference for the hosts whose line to this laptop
    is the fastest, and not a filter.

    ``engine_build`` false provisions the machine without PFFDTD
    (:func:`provision_machine`): a campaign that does not run it.

    ``disk_gb`` is the disk asked of the rental where the caller sizes it
    from its own plan (a trace: :func:`reverberate.trace.machines.disk_need`);
    left out, a field campaign's (:func:`campaign_need`). It is what the
    offers are searched with, what the rental asks for and what it is
    billed for.

    ``resume_command`` is the command that resumes this run on its
    machine, ``{instance}`` standing for the instance; ``inventory`` asks a
    machine what of the run it holds and answers in lines. Both are said
    in the last lines of a run that leaves its machine rented
    (:func:`still_rented`), a campaign's failure first among them.

    **A run resumed on a machine whose campaign is done fetches and
    launches nothing**: what failed there was the fetch, and a campaign
    launched again would write its pack again under the chunks that are
    home. ``relaunch`` launches it all the same.

    ``cap_flag`` names a flag of the campaign's command that is told the
    hours the watchdog leaves it (``--max-hours`` for a trace, which
    measures its machine and stops before its long work where it predicts
    more).

    ``leave`` names entries of the run directory that are not fetched,
    ``also`` entries fetched besides those a campaign always brings home;
    ``sync`` those brought home every ``sync_s`` while the campaign runs.

    ``gpus`` is the fewest cards of the host. ``predict`` prices the run on
    an offer (:func:`price_offers`): the lowest predicted USD within
    ``max_hours`` of wall time is rented, and ``hours``, the watchdog's
    cap, is taken from the prediction when left out. ``plan_only`` says
    the offers and their predictions and rents nothing.

    ``instance`` resumes on a machine already rented (the campaign resumes
    from its state on disk). ``avoid`` names offers and machines not to rent,
    to which :data:`reverberate.gpu.vast.KNOWN_BAD_HOSTS` are added;
    the record's ``avoided`` is that list and the hosts this run refused,
    ready to be given to the next. Returns the record written to
    ``home/onebox.json``.

    **How a rental ends.** ``outcome`` is ``done`` (fetched, destroyed,
    verified), ``deadline`` (what exists fetched, destroyed), ``instance
    vanished``, ``failed`` (a campaign that failed twice: fetched, and kept
    for inspection unless ``destroy_failed``) or ``error`` (the driver
    itself failed: what ``sync`` names brought home once more, destroyed,
    verified). An instance left alive is named in the last line said, with
    why and what it bills (:func:`still_rented`), and in ``left_alive``.
    """
    bundle, home, repo = Path(bundle), Path(home), Path(repo)
    home.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"bundle": str(bundle), "started": time.time(), "watches": []}
    need = campaign_need(bundle)
    if disk_gb is not None:
        need = replace(need, disk_gb=int(disk_gb))
    record["disk_gb"] = need.disk_gb
    say(f"need: {need.describe()}")
    if instance is None and not yes and not plan_only:
        say("nothing rented: pass --yes")
        return record
    auth.inject(["VASTAI_API_KEY"])
    client = vast.VastClient(timeout=60)
    identity = vast.account_identity(client)
    started = time.time()
    deadline: float | None = None
    resumed = instance is not None

    def save() -> None:
        (home / "onebox.json").write_text(json.dumps(record, indent=1, default=str))

    if instance is None:
        avoided = vast.hosts_to_avoid(avoid)
        known = sorted(set(vast.KNOWN_BAD_HOSTS) - {int(i) for i in avoid})
        if known:
            say(
                "hosts known bad, never rented: "
                + "; ".join(f"{i} ({vast.KNOWN_BAD_HOSTS[i][0]})" for i in known)
            )
        machine = None
        try:
            for _ in range(PROVISION_HOSTS):
                plan = plan_rental(
                    client,
                    need,
                    hours=hours,
                    max_dph=max_dph,
                    min_ram_gb=min_ram_gb,
                    gpu=gpu,
                    say=say,
                    avoid=avoided,
                    min_gpus=gpus,
                    predict=predict,
                    max_hours=max_hours,
                    prefer_regions=prefer_regions,
                )
                record["offers"] = plan.record()
                record["cap_hours"] = plan.hours
                if plan_only:
                    say("offers planned: nothing rented")
                    return record
                machine, instance, offer = rent(
                    client,
                    identity,
                    need,
                    hours=plan.hours,
                    max_dph=max_dph,
                    min_ram_gb=min_ram_gb,
                    gpu=gpu,
                    say=say,
                    avoid=avoided,
                    plan=plan,
                    remaining_usd=plan.priced[0].usd if plan.priced else 0.0,
                )
                started = time.time()
                deadline = started + plan.hours * 3600.0
                record["instance"] = instance
                record["offer"] = offer.id
                record["avoided"] = sorted(avoided)
                save()
                try:
                    record.update(
                        provision_machine(machine, repo, bundle, say, engine_build=engine_build)
                    )
                    break
                except BaseException as error:
                    # A host that cannot be provisioned is not waited for: three rentals
                    # sat idle to their watchdogs on a connection that had dropped.
                    say(
                        f"provisioning of {instance} failed ({str(error)[:300]});"
                        " destroying it and taking the next offer"
                    )
                    avoided |= vast.offer_ids(offer)
                    gone = bool(vast.teardown(client, instance))
                    record.setdefault("abandoned", []).append(
                        {"instance": instance, "offer": offer.id, "destroyed": gone}
                    )
                    if not gone:
                        record["left_alive"] = still_rented(
                            client,
                            instance,
                            "its provisioning failed and its destruction is not verified",
                            deadline=deadline,
                            say=say,
                        )
                        raise SystemExit(record["left_alive"]) from error
                    record.pop("instance", None)
                    record.pop("offer", None)
                    machine, instance = None, None
                    if not isinstance(error, Exception):
                        raise
            if machine is None or instance is None:
                raise SystemExit(
                    f"no host could be provisioned in {PROVISION_HOSTS} rentals;"
                    " each was destroyed and verified"
                )
        finally:
            # Kept even when no offer produced a machine: the next run needs it most then.
            record["avoided"] = sorted(avoided)
            save()
    else:
        machine = vast.wait_for_ssh(client, instance, identity)
        record["instance"] = instance
        rsync(machine, [str(repo / "src")], REMOTE_SRC + "/", download=False)
        # Left unsaid: the watchdog armed when it was rented, as this machine's ledger has it.
        deadline = started + hours * 3600.0 if hours is not None else vast.deadline_of(instance)
    if getattr(client, "direct_refused", False):
        say("direct ssh was refused at the rental: every transfer goes through the proxy")

    def launch() -> None:
        told = campaign_args
        if cap_flag and deadline is not None:
            # What is left of the watchdog, less the fetch it keeps for itself: a campaign
            # that predicts more of its own work on the machine it finds stops at once.
            left = (deadline - 1200.0 - time.time()) / 3600.0
            told = f"{told} {cap_flag} {max(left, 0.0):.2f}".strip()
        started_text = run_on(
            machine,
            _launch_command(REMOTE_BUNDLE, REMOTE_OUT, devices, "/root/pffdtd", told),
            what="launch",
        )
        say(f"campaign launched ({started_text.strip()[-40:]})")

    try:
        if resumed and not relaunch and campaign_done(machine):
            say("the campaign is done on the machine: nothing is launched, what it made is fetched")
            record["outcome"] = "done"
        else:
            launch()
            early = early_homecoming(machine, home, say, record, leave=leave)
            try:
                record["outcome"] = watch(
                    client,
                    instance,
                    machine,
                    launch,
                    deadline=deadline if deadline is not None else started + DEFAULT_HOURS * 3600.0,
                    poll_s=poll_s,
                    record=record,
                    home=home,
                    say=say,
                    sync=sync,
                    sync_s=sync_s,
                    early=early,
                )
            except BaseException:
                early.finish(wait=False)
                raise
            # A pack on its way is waited for: the fetch below then finds it home.
            early.finish()
    except BaseException as error:
        record["outcome"] = "error"
        record["error"] = repr(error)[:2000]
        record["traceback"] = traceback.format_exc()[-6000:]
        say(f"the driver failed after the rental: {error!r}"[:600])
        if isinstance(error, Exception):
            # What can be saved is, once; then the machine is not left to its watchdog.
            sync_home(machine, home, sync, say)
            if not release(client, instance, record, say):
                record["left_alive"] = still_rented(
                    client,
                    instance,
                    "the driver failed and the destruction is not verified",
                    deadline=deadline,
                    say=say,
                )
            record["total_s"] = round(time.time() - started, 1)
            save()
            return record
        # Interrupted by a person: the campaign runs on, detached, and is theirs to end.
        record["left_alive"] = still_rented(
            client,
            instance,
            "the driver was interrupted",
            deadline=deadline,
            say=say,
            command=resume_command,
        )
        save()
        raise
    outcome = str(record["outcome"])
    fetched = True
    t0 = time.time()
    if outcome != "instance vanished":
        try:
            pulled = fetch(
                machine,
                bundle,
                home,
                fetch_cache=fetch_cache,
                say=say,
                leave=leave,
                also=also,
                record=record,
            )
            say(f"fetched in {(time.time() - t0) / 60:.1f} min -> {pulled}")
        except Exception as error:  # noqa: BLE001 - said, and the instance accounted for below
            fetched = False
            record["fetch_error"] = repr(error)[:2000]
            say(f"the fetch failed: {error!r}"[:600])
    record["fetch_s"] = round(time.time() - t0, 1)

    def held() -> list[str]:
        if inventory is None:
            return []
        try:
            return list(inventory(machine))
        except Exception as error:  # noqa: BLE001 - the last lines are said whatever answers
            return [f"what the machine holds could not be asked ({str(error)[:120]})"]

    if outcome == "done" and not fetched:
        record["left_alive"] = still_rented(
            client,
            instance,
            "the campaign is done and its fetch failed; what it made is on the machine",
            deadline=deadline,
            say=say,
            held=[
                "a resume launches nothing: it finds the campaign done and fetches, from the"
                " chunks that are home",
            ],
            command=resume_command,
        )
    elif outcome == "failed" and not destroy_failed:
        record["left_alive"] = still_rented(
            client,
            instance,
            "the campaign failed twice and is kept for inspection",
            deadline=deadline,
            say=say,
            held=held(),
            command=resume_command,
        )
    elif not release(client, instance, record, say):
        record["left_alive"] = still_rented(
            client, instance, "its destruction is not verified", deadline=deadline, say=say
        )
    record["total_s"] = round(time.time() - started, 1)
    save()
    return record


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True, help="where the run comes home")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help=f"the watchdog's cap; {DEFAULT_HOURS:g} unless said",
    )
    parser.add_argument("--max-dph", type=float, default=3.0)
    parser.add_argument("--gpus", type=int, default=1, help="the fewest cards of the host")
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--poll", type=float, default=POLL_S)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--min-ram-gb", type=float, default=60.0, help="hosts under this much RAM are skipped"
    )
    parser.add_argument("--gpu", default="", help="only cards whose name contains this")
    parser.add_argument(
        "--avoid",
        type=int,
        nargs="*",
        default=[],
        metavar="ID",
        help="offer or machine ids never to rent; hosts refused in the run are added",
    )
    parser.add_argument(
        "--campaign-args", default="", help="extra flags for the campaign, e.g. a flow test's bands"
    )
    parser.add_argument(
        "--sync", nargs="*", default=[], help="entries of the run brought home as it runs"
    )
    parser.add_argument("--plan-offers", action="store_true", help="the offers; nothing rented")
    parser.add_argument(
        "--destroy-failed", action="store_true", help="a campaign that failed twice is not kept"
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    record = run(
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
        avoid=args.avoid,
        gpus=args.gpus,
        sync=args.sync,
        plan_only=args.plan_offers,
        destroy_failed=args.destroy_failed,
    )
    return 0 if record.get("outcome", "done") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
