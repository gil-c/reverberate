"""Rent a card, build the engine on it, solve one prepared run, bring it home.

`scripts/remote_chain.py` drives the voxelise and payload stages. This is the
third one, and it is separate because it wants a different machine: VRAM and
almost nothing else, where voxelisation wants cores and disk.

**Every rule here was bought.** Each is section 12.1 of the roadmap turned into
code rather than discipline:

- the requirement is arithmetic over the grid, and an offer that cannot meet it
  is refused **before** an instance exists;
- the rate, the cap and the worst case total are printed and ``--yes`` is
  required, so no run starts without the figure having been shown;
- the ssh identity is checked against the account's registered keys before
  renting, because an instance that answers ``Permission denied`` on every poll
  costs a full timeout and does nothing;
- **teardown is conditional on the artefact being home.** W35 lost a finished
  grid to a ``finally`` that could not tell a failed compute from a failed
  retrieval, in a file that cited the earlier time the same thing happened. If
  the solve finished and the fetch did not, the instance is left up, its
  deadline is named, and the operator is told how to fetch by hand.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/remote_solve.py \\
        --run data/runs/w10_bedroom_16k --key <cache key> \\
        --hours 4 --max-dph 0.70 --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reverberate import auth
from reverberate.experiments.engine import write_record
from reverberate.experiments.run import entry_from_key
from reverberate.gpu import vast
from reverberate.wave import engine_inputs
from reverberate.wave.remote import solve
from reverberate.wave.remote_voxelise import MachineNeed, provision

#: A CUDA image, because the build script compiles the engine on the box.
IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

#: Memory bandwidth in GB/s, for the cards this project is likely to be offered.
#: The solver is bandwidth bound: the roadmap calls its 67 G point updates per
#: second "the A100's HBM2e roofline at about 2.0 TB/s, a hardware ceiling not a
#: code limit". So **the hourly rate is the wrong thing to rank offers by.** On
#: this run an A100 PCIE at 0.669 USD/h finishes in 2.8 h for 1.86 USD while an
#: RTX A6000 at 0.469 takes 7 h for 3.30, and only the first fits a four hour
#: cap. Ranking by rate would have chosen the second.
#:
#: A card that is not in this table is not guessed at. It is offered last with
#: its bandwidth named as unknown, so renting one is a decision rather than an
#: accident.
BANDWIDTH_GB_S = {
    "A100 SXM4": 1555.0,
    "A100 PCIE": 1935.0,
    "A100X": 1935.0,
    "H100 PCIE": 2000.0,
    "H100 SXM": 3350.0,
    "H200": 4800.0,
    "L40S": 864.0,
    "RTX 6000Ada": 960.0,
    "RTX A6000": 768.0,
    "Q RTX 8000": 672.0,
    "RTX 4090": 1008.0,
    "RTX 5090": 1792.0,
    "RTX 3090": 936.0,
}

#: The card the anchor of section 10 was measured on.
ANCHOR_GPU = "A100 PCIE"

#: Bytes of VRAM per grid point, measured. Roadmap section 5.4: 8.125 for the
#: two fields and the mask, plus a surface term, and 2.13 GB of fixed context.
VRAM_BYTES_PER_POINT = 9.027
VRAM_CONTEXT_GB = 2.13


def solve_need(grid_points: int, receivers: int, samples: int) -> MachineNeed:
    """What the solve stage needs, from the grid and the output it will write.

    The host side matters as well as the card: the engine holds the whole
    output in host memory as ``Nr`` by ``Nt`` doubles before it writes it, so a
    run with a thousand receivers wants a gigabyte of RAM that a run with six
    does not.
    """
    vram = grid_points * VRAM_BYTES_PER_POINT / 1e9 + VRAM_CONTEXT_GB
    output_gb = receivers * samples * 8 / 1e9
    return MachineNeed(
        cores=4,
        ram_gb=max(16.0, 4.0 * output_gb + 8.0),
        disk_gb=max(60.0, 3.0 * (grid_points * 23.0 / 1e9) + 20.0),
        why=(
            f"{grid_points:.3g} grid points at {VRAM_BYTES_PER_POINT} bytes plus "
            f"{VRAM_CONTEXT_GB} GB of context is {vram:.0f} GB of VRAM, and "
            f"{receivers} receivers by {samples} samples is {output_gb:.2f} GB of output "
            "held in host memory before it is written"
        ),
        needs_gpu=True,
        vram_gb=vram,
    )


def hours_on(offer: Any, anchor_hours: float) -> tuple[float | None, str]:
    """How long the anchor's job takes on this card, and where the figure comes from.

    Scaled by memory bandwidth alone, because the solver is bandwidth bound.
    Returns ``None`` for a card whose bandwidth is not in the table rather than
    inventing one.
    """
    bandwidth = BANDWIDTH_GB_S.get(offer.gpu_name)
    if bandwidth is None:
        return None, f"{offer.gpu_name} bandwidth unknown"
    scale = BANDWIDTH_GB_S[ANCHOR_GPU] / bandwidth
    return anchor_hours * scale, f"{bandwidth:.0f} GB/s, {scale:.2f}x the anchor"


def rank_offers(
    offers: Sequence[Any], need: MachineNeed, anchor_hours: float, cap_hours: float
) -> list[tuple[Any, float | None, float | None, str]]:
    """Offers by what the whole job costs, not by the hourly rate.

    A card that cannot finish inside the cap stays in the list and is marked,
    rather than being filtered away: "cheapest per hour, and it did not finish"
    is a real way to spend money for nothing and it should be visible.
    """
    rows: list[tuple[Any, float | None, float | None, str]] = []
    for offer in offers:
        if need.unmet(offer):
            continue
        hours, why = hours_on(offer, anchor_hours)
        rows.append((offer, hours, None if hours is None else offer.dph_total * hours, why))
    known = [row for row in rows if row[1] is not None and row[2] is not None]
    known.sort(key=lambda row: (float(row[1] or 0.0) > cap_hours, float(row[2] or 0.0)))
    unknown = [row for row in rows if row[1] is None]
    unknown.sort(key=lambda row: float(row[0].dph_total))
    return known + unknown


def say(message: str) -> None:
    """Print and flush.

    Every line this driver prints is progress on a job of hours, and Python
    buffers stdout when it is redirected to a file. A run whose log stays empty
    for three hours cannot be watched, cannot be diagnosed while it is alive,
    and looks identical to a hang.
    """
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a run directory holding plan.json")
    parser.add_argument("--key", required=True, help="voxelisation cache key")
    parser.add_argument("--hours", type=float, required=True, help="hard deadline")
    parser.add_argument("--max-dph", type=float, default=0.70)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--yes", action="store_true", help="required to spend money")
    parser.add_argument(
        "--instance",
        type=int,
        default=None,
        help=(
            "attach to an instance already rented rather than renting another. "
            "For the case this exists for: a run that failed after the rental "
            "and before the solve, where renting again pays twice for the same "
            "boot and the same build"
        ),
    )
    args = parser.parse_args(argv)

    plan = json.loads((args.run / "plan.json").read_text())
    comms = args.run / "comms" / "comms_out.h5"
    if not comms.is_file():
        raise SystemExit(f"{comms} is missing: run the prepare step first")
    entry = entry_from_key(args.key)
    need = solve_need(
        int(plan["cost"]["grid_points"]), int(plan["cost"]["receivers"]), int(plan["samples"])
    )

    auth.inject()
    client = vast.VastClient()
    identity = vast.account_identity(client)
    anchor_hours = float(plan["cost"]["estimated_gpu_s"]) / 3600.0
    files = engine_inputs(entry, comms)
    upload_gb = sum(f.stat().st_size for f in files) / 1e9
    say(f"run {args.run.name}, cache {args.key}")
    say(f"  needs {need.vram_gb:.0f} GB VRAM, {need.ram_gb:.0f} GB RAM, {need.disk_gb:.0f} GB disk")
    say(f"  because {need.why}")
    say(f"  the solver takes {anchor_hours:.2f} h on the anchor, an {ANCHOR_GPU}")
    say(f"  upload {upload_gb:.2f} GB, fetch {plan['cost']['sim_outs_bytes'] / 1e9:.2f} GB")
    say(f"  ledger stands at {vast.ledger_total_usd(vast.read_ledger()):.2f} USD")

    offer = None
    hours: float | None = None
    cost: float | None = None
    why = ""
    if args.instance is None:
        # Ranked by what the whole job costs rather than by the hourly rate. The
        # search is skipped entirely when attaching, because an instance already
        # rented is not a choice between offers and a market that has moved on
        # is no reason to refuse to use it.
        offers = client.search(
            vast.search_query(
                gpu_name="",
                min_disk_gb=int(need.disk_gb),
                min_gpu_ram_gb=need.vram_gb,
                min_reliability=0.99,
            ),
            limit=200,
        )
        ranked = rank_offers(
            [candidate for candidate in offers if candidate.dph_total <= args.max_dph],
            need,
            anchor_hours,
            args.hours,
        )
        if not ranked:
            raise SystemExit(f"nothing under {args.max_dph} USD/h meets: {need.why}")
        say(f"\n  {len(ranked)} offers meet it, ranked by what the whole job costs:")
        for candidate, candidate_hours, candidate_cost, candidate_why in ranked[:6]:
            if candidate_hours is None or candidate_cost is None:
                say(
                    f"    {candidate.id} {candidate.gpu_name} "
                    f"{candidate.dph_total:.3f} USD/h: {candidate_why}"
                )
                continue
            flag = "" if candidate_hours <= args.hours else f"  OVER THE {args.hours:g} h CAP"
            say(
                f"    {candidate.id} {candidate.gpu_name} {candidate.dph_total:.3f} USD/h -> "
                f"{candidate_hours:.2f} h, {candidate_cost:.2f} USD ({candidate_why}){flag}"
            )
        offer, hours, cost, why = ranked[0]
        if hours is not None and hours > args.hours:
            raise SystemExit(
                f"the best offer needs {hours:.2f} h against a {args.hours:g} h cap; "
                "raise the cap or the rate rather than starting a run that cannot finish"
            )
        say(f"\n  chosen: {offer.describe()}")
        say(
            f"  cap {args.hours:g} h -> at most "
            f"{vast.estimate_cost_usd(offer.dph_total, args.hours):.2f} USD at the offer's rate"
        )
    if not args.yes:
        say("\nnothing rented: pass --yes once the figures above are approved")
        return 0

    if args.instance is None:
        assert offer is not None
        rental = vast.rent(
            client, offer, hours=args.hours, image=IMAGE, disk_gb=int(need.disk_gb) + 20
        )
        instance_id = rental.instance_id
        say(f"rented {instance_id}, watchdog pid {rental.watchdog_pid}")
    else:
        instance_id = int(args.instance)
        existing = client.instance(instance_id)
        if existing is None:
            raise SystemExit(f"instance {instance_id} does not exist; nothing to attach to")
        attached_gpu, attached_dph = existing.gpu_name, existing.dph_total
        say(
            f"attached to {instance_id}, {existing.gpu_name} at {existing.dph_total:.3f} USD/h, "
            f"up {existing.uptime_hours():.2f} h. Its own watchdog still holds the deadline; "
            "no second rental and no second ledger row"
        )
    retrieved = False
    try:
        machine = vast.wait_for_ssh(client, instance_id, identity)
        say(f"ssh up at {machine.host}:{machine.port}")
        built = provision(machine, Path("scripts/build_pffdtd.sh"))
        say(f"engine built in {built / 60:.1f} min")
        result = solve(
            machine,
            files,
            args.run / "source0" / "sim_outs.h5",
            double_precision=args.double,
            timeout=args.hours * 3600,
        )
        retrieved = True
        (args.run / "source0" / "engine.log").write_text(result.log)
        write_record(
            args.run,
            "solve.json",
            {
                "runs": [
                    {
                        "source_index": 0,
                        "engine": "gpu",
                        "engine_s": result.engine_s,
                        "upload_s": result.upload_s,
                        "fetch_s": result.fetch_s,
                        "uploaded_bytes": result.uploaded_bytes,
                        "where": f"{machine.user}@{machine.host}:{machine.port}",
                        "instance_id": instance_id,
                        "gpu": offer.gpu_name if offer else attached_gpu,
                        "quoted_dph": offer.dph_total if offer else attached_dph,
                        "predicted_hours": hours,
                        "predicted_cost_usd": cost,
                        "bandwidth_note": why,
                        "image": IMAGE,
                        "hard_deadline_hours": args.hours,
                    }
                ]
            },
        )
        say(f"solved in {result.engine_s / 3600:.2f} h, fetched in {result.fetch_s:.0f} s")
    finally:
        if retrieved:
            say("tearing down, the artefact is home")
            vast.teardown(client, instance_id)
        else:
            say(
                f"NOT tearing down instance {instance_id}: the artefact is not home. "
                f"The watchdog destroys it at the {args.hours:g} h deadline regardless. "
                f"Fetch by hand from /root/run/sim_outs.h5 before then if the solve finished."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
