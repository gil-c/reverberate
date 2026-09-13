"""Renting and driving Vast machines for a campaign.

Everything here was learnt on a rented machine at night, and each rule names
the night it cost. The policies are pure functions over offers so they can be
tested without an account; the actions take the client and are thin.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Bytes per worker the child encoder was measured at (fit order 10, 1021
#: nodes, a whole band of samples): about 7 GB on every box of 2026-09-13.
#: ``workers = cores - 1`` overran a 91 GB container with 36 cores.
ENCODE_WORKER_GB = 7.0

#: Cards the CUDA 12.4 image builds PFFDTD for. Blackwell (RTX PRO 6000,
#: B200) wants 12.8 and is untested.
CARD_NAMES = ("A100", "H100", "H200")

#: Ivy Bridge Xeons encoded at 125 s a point, four times slower than the
#: next box; they are never worth their price.
SLOW_CPU_MARKS = (" v2",)


# --------------------------------------------------------------------------
# policies: pure, tested
# --------------------------------------------------------------------------


def workers_for(cores: int, cgroup_gb: float, per_worker_gb: float = ENCODE_WORKER_GB) -> int:
    """How many encode workers a container can hold: cores, capped by memory."""
    return max(1, min(cores - 1, int(cgroup_gb / per_worker_gb)))


def card_hosts(offers: list[Any], *, vram_gb: float, ram_gb: float, max_dph: float) -> list[Any]:
    """Offers whose cards together hold ``vram_gb``, cheapest first.

    PFFDTD splits the grid over every card it sees, so the host's VRAM in
    total is what must hold the grid: one 80 GB card, or two, or a 140 GB
    H200. A search is per card count, so the caller searches 1, 2 and 4.
    """
    good = [
        o
        for o in offers
        if o.dph_total <= max_dph
        and o.ram_gb >= ram_gb
        and any(name in o.gpu_name for name in CARD_NAMES)
        and o.num_gpus * o.gpu_ram_gb >= vram_gb
    ]
    return sorted(good, key=lambda o: o.dph_total)


def encode_boxes(offers: list[Any], *, max_dph: float, min_ram_gb: float = 48.0) -> list[Any]:
    """CPU boxes for the encode, best price per core-GHz first, slow Xeons out."""
    good = [
        o
        for o in offers
        if o.dph_total <= max_dph
        and o.ram_gb >= min_ram_gb
        and not any(mark in o.cpu_name for mark in SLOW_CPU_MARKS)
    ]
    return sorted(good, key=lambda o: o.dph_total / max(o.cpu_cores * o.cpu_ghz, 1.0))


@dataclass(frozen=True)
class Widening:
    """One step of relaxation of an offer search: cores, clock, price."""

    min_cores: int
    min_cpu_ghz: float
    max_dph: float


def widening_steps(min_cores: int, min_cpu_ghz: float, max_dph: float) -> list[Widening]:
    """The filters to try in turn when the market is thin.

    At 01:49 on 2026-09-13 the first query found zero boxes and the driver
    failed three times on the same query while the card billed idle. Widen
    instead: fewer cores, then a slower clock, then a higher price.
    """
    return [
        Widening(min_cores, min_cpu_ghz, max_dph),
        Widening(max(8, min_cores // 2), min_cpu_ghz, max_dph),
        Widening(max(8, min_cores // 2), max(3.0, min_cpu_ghz - 0.5), max_dph),
        Widening(max(8, min_cores // 2), max(3.0, min_cpu_ghz - 0.5), max_dph * 1.5),
    ]


def enough_credit(credit: float, *, remaining_usd: float) -> bool:
    """Whether the account can pay the rest of the plan, not just one hour.

    The night of 2026-09-13 started with 12.5 USD, spent 10 before the encode
    and ran out under it. ``credit`` is NaN when the API did not answer; then
    the rental goes ahead, since a status read must not end a campaign.
    """
    return credit != credit or credit >= remaining_usd


# --------------------------------------------------------------------------
# actions: thin over the client
# --------------------------------------------------------------------------


def credit_of(client: Any) -> float:
    """The account's credit in USD, or NaN when the API does not answer."""
    try:
        return float(client.request("GET", "/users/current/").get("credit") or 0.0)
    except Exception:  # noqa: BLE001 - a status read must never end a campaign
        return float("nan")


def cgroup_memory_gb(machine: Any) -> float:
    """The container's memory limit, v2 or v1; the host's RAM when neither answers."""
    from reverberate.wave.remote import run_on

    text = run_on(
        machine,
        "cat /sys/fs/cgroup/memory.max 2>/dev/null"
        " || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null"
        " || free -b | awk 'NR==2{print $2}'",
        what="cgroup memory",
    ).strip()
    return float(text) / 1e9 if text.isdigit() else 1e6


def rent_one(
    client: Any,
    identity: Any,
    offers: list[Any],
    *,
    hours: float,
    disk_gb: int,
    image: str,
    remaining_usd: float = 0.0,
    say: Callable[[str], None] = print,
) -> tuple[Any, int]:
    """Down the list until one rents and answers; the machine and its id.

    Every offer tried is removed from ``offers`` in place, rented or not, so a
    caller renting several boxes from one list never returns to a host that
    stayed silent (two did on 2026-09-12). The credit is read before each
    rental against ``remaining_usd``.
    """
    from reverberate.gpu import vast

    for candidate in list(offers[:8]):
        offers.remove(candidate)
        credit = credit_of(client)
        if not enough_credit(credit, remaining_usd=max(remaining_usd, candidate.dph_total + 0.5)):
            raise SystemExit(
                f"credit {credit:.2f} USD is under the {remaining_usd:.2f} USD"
                " the rest of the plan needs"
            )
        say(f"  credit {credit:.2f} USD before renting {candidate.id}")
        try:
            rental = vast.rent(client, candidate, hours=hours, image=image, disk_gb=disk_gb)
        except vast.VastError as refusal:
            say(f"  {candidate.id} would not rent ({refusal})")
            continue
        say(f"rented {rental.instance_id} on {candidate.describe()}")
        try:
            machine = vast.wait_for_ssh(client, rental.instance_id, identity, timeout=420.0)
        except (TimeoutError, vast.VastError) as silence:
            say(f"  {rental.instance_id} never answered ({silence}); destroying, next")
            vast.teardown(client, rental.instance_id)
            continue
        return machine, rental.instance_id
    raise SystemExit("no offer produced a machine that answered")


def rearm_watchdog(instance_id: int, hours: float, say: Callable[[str], None] = print) -> None:
    """Replace the instance's hard stop with one ``hours`` from now.

    The deadline was set at rental time for the stage that rented; a later
    stage that still needs the data on that host must push it back, or the
    watchdog destroys the only copy of the pressure (nearly happened twice).
    """
    from reverberate.gpu import vast

    subprocess.run(["pkill", "-f", f"hard-stop {instance_id} "], check=False)
    vast.arm_hard_stop(instance_id, time.time() + hours * 3600.0)
    say(f"  watchdog on {instance_id} re-armed for {hours:g} h")


def launch(machine: Any, workdir: str, name: str, command: str) -> None:
    """Start ``command`` detached; ``name.done`` or ``name.failed`` says how it ended.

    Never hold a long job in one ssh session: a 2.5 h pull died on
    "Connection to ssh3.vast.ai closed by remote host" and took its box with
    it. Everything long on a rented machine goes through here.
    """
    from reverberate.wave.remote import run_on

    script = (
        "#!/bin/bash\n"
        f"cd {shlex.quote(workdir)}\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
        "ulimit -n 65536 2>/dev/null\n"
        f"( {command} ) > {name}.log 2>&1 && touch {name}.done || touch {name}.failed\n"
    )
    run_on(
        machine,
        f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}"
        f" && rm -f {name}.done {name}.failed && "
        f"cat > {name}.sh <<'REVERBERATE_EOF'\n{script}REVERBERATE_EOF\n"
        f"chmod +x {name}.sh && setsid nohup ./{name}.sh > /dev/null 2>&1 < /dev/null &"
        " sleep 1; echo started",
        what=f"launch {name}",
    )


def wait(machine: Any, workdir: str, name: str, *, timeout: float, poll_s: float = 60.0) -> str:
    """Poll a detached job until its marker appears; tolerate the proxy's bad minutes."""
    from reverberate.wave.remote import run_on

    started = time.time()
    misses = 0
    while True:
        try:
            state = run_on(
                machine,
                f"cd {shlex.quote(workdir)} && (test -f {name}.done && echo done"
                f" || (test -f {name}.failed && echo failed || echo running));"
                f" tail -c 2000 {name}.log 2>/dev/null",
                what=f"poll {name}",
            )
            misses = 0
        except RuntimeError:
            misses += 1
            if misses > 15:
                raise
            time.sleep(poll_s)
            continue
        status, _, log = state.partition("\n")
        if status.strip() == "done":
            return log
        if status.strip() == "failed":
            raise RuntimeError(f"{name} failed on the machine:\n{log[-2000:]}")
        if time.time() - started > timeout:
            raise TimeoutError(f"{name} still running after {timeout / 3600:.1f} h")
        time.sleep(poll_s)


def resumable_transfer(
    files: str, target: str, *, port: int, host: str, key: str, name: str
) -> str:
    """A shell loop that pushes ``files`` to ``root@host:target`` until rsync succeeds.

    Vast's ssh proxies refuse or drop sessions for minutes at a time; on
    2026-09-13 four pulls into the card's proxy stalled at once and two were
    marked failed by a one-shot rsync. ``--partial`` keeps what arrived,
    keepalives make a dead session die, and the loop resumes it.
    """
    ssh = (
        f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no"
        " -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=30"
    )
    return (
        f"until rsync -a --partial --timeout=60 -e {shlex.quote(ssh)}"
        f" {files} root@{host}:{target}; "
        f'do echo "$(date +%T) {name}: rsync failed, resuming"; sleep 20; done'
    )
