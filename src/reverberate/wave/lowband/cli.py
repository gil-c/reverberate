"""The low band solver's commands: its numbers, its proof on a card, its error and its cost.

python -m reverberate.wave.lowband numbers [--entry DIR --source X Y Z]
    each scheme's solve on paper: step, nodes, steps, memory a source, sources a card
python -m reverberate.wave.lowband verify --out O [--pffdtd DIR] [--cpu]
    a small lossy room on both grids: the card against numpy bit for bit, a batch against its
    sources alone, and the present engine's binary on the same files
python -m reverberate.wave.lowband line --field S1.h5 --plan plan.json --out DIR [--every N]
    on the laptop: a field's low side in the stored form, and the positions of a pairs bundle
python -m reverberate.wave.lowband compare --bundle B --out O [--scheme S] [--ppw P]
        (--reference O_REF [--pffdtd DIR] | --line DIR/line.npz) [--rate USD_PER_H]
    the pairs of the bundle on the batched solver, then their error against the reference
python -m reverberate.wave.lowband cost --bundle B --out O [--scheme S] [--ppw P]
        [--batches 1,4,16,64] [--steps 400] [--cells 10] --rate USD_PER_H
    node updates a second, seconds and USD per source position and per pair, per batch size
python -m reverberate.wave.lowband counts --heard heard_at.json [--channels 64]
    solves needed from the source side, from the cells' side, and by the best mix

``compare`` and ``cost`` take the bundle of ``python -m reverberate.accel
pairs-bundle``; with ``--reference`` naming a directory that holds no
finished campaign and ``--pffdtd`` given, the present engine is run there
first, so that one command gives the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverberate.wave.lowband", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("numbers", help="each scheme's solve on paper")
    p.add_argument("--fmax", type=float, default=1500.0)
    p.add_argument("--duration", type=float, default=1.2)
    p.add_argument("--box-nodes", type=float, default=65.9e6, help="Cartesian box at 10.5")
    p.add_argument("--reached-share", type=float, default=0.724)
    p.add_argument("--lossy-share", type=float, default=0.0403)
    p.add_argument("--branches", type=int, default=11)
    p.add_argument("--rows", type=int, default=9840, help="nodes a source is read at")
    p.add_argument("--entry", type=Path, default=None, help="read the shares on this grid")
    p.add_argument("--source", type=float, nargs=3, default=None)

    p = sub.add_parser("verify", help="the solver against itself and the present engine")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pffdtd", type=Path, default=None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--cpu", action="store_true")

    p = sub.add_parser("line", help="a field's low side as a reference, on the laptop")
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="a directory")
    p.add_argument("--every", type=int, default=1)

    for name in ("compare", "cost"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--scheme", choices=("cartesian", "fcc"), default="cartesian")
        p.add_argument("--ppw", type=float, default=None)
        p.add_argument("--devices", default=None)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--rate", type=float, default=None, help="USD an hour of the machine")
        p.add_argument("--pffdtd", type=Path, default=None)
        if name == "compare":
            p.add_argument("--reference", type=Path, default=None, help="a campaign's directory")
            p.add_argument("--line", type=Path, default=None, help="line.npz of the line command")
            p.add_argument("--batch", type=int, default=None)
        else:
            p.add_argument("--batches", default="1,2,4,8,16,32,64")
            p.add_argument("--steps", type=int, default=400)
            p.add_argument("--cells", type=int, default=10)

    p = sub.add_parser("counts", help="forward, reciprocal and mixed solve counts")
    p.add_argument("--heard", type=Path, required=True)
    p.add_argument("--channels", type=int, default=64)
    return parser


def _campaign(args: argparse.Namespace) -> Any:
    from reverberate.wave.lowband.pairs import LowbandPairs

    return LowbandPairs(
        bundle=args.bundle,
        out=args.out,
        pffdtd_dir=args.pffdtd or Path("/root/pffdtd"),
        devices=args.devices,
        gpu=False if args.cpu else None,
        scheme=args.scheme,
        ppw=args.ppw,
        batch=getattr(args, "batch", None),
    )


def _numbers(args: argparse.Namespace) -> int:
    from reverberate.wave.lowband.harness import paper_numbers
    from reverberate.wave.lowband.scheme import CARTESIAN, FCC, points_per_wavelength

    reached, lossy, branches, nodes = (
        args.reached_share,
        args.lossy_share,
        args.branches,
        args.box_nodes,
    )
    if args.entry is not None:
        from reverberate.wave.comms import engine_indices, interp_weights, load_grid
        from reverberate.wave.lowband.problem import load_problem

        grid = load_grid(args.entry)
        seeds = None
        if args.source is not None:
            seeds = engine_indices(interp_weights(np.asarray(args.source), grid)[1], grid)
        problem = load_problem(args.entry, seeds)
        print(json.dumps({**problem.record, "updated_nodes": problem.updated}, indent=1))
        reached = problem.reached / problem.box_nodes
        branches = problem.max_branches
    rows = paper_numbers(
        fmax_hz=args.fmax,
        duration_s=args.duration,
        reference_nodes=nodes,
        reached_share=reached,
        lossy_share=lossy,
        branches=branches,
        rows=args.rows,
    )
    budget = {
        f"{100 * b:g} %": {
            "cartesian": round(points_per_wavelength(CARTESIAN, b), 2),
            "fcc": round(points_per_wavelength(FCC, b), 2),
        }
        for b in (0.005, 0.01, 0.02)
    }
    print(json.dumps({"points_per_wavelength_for_a_velocity_error": budget}, indent=1))
    print(json.dumps(rows, indent=1))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from reverberate.accel.pairs import PairsCampaign
    from reverberate.wave.lowband.harness import compare_responses, format_table, stored_of_cache

    if (args.reference is None) == (args.line is None):
        raise SystemExit("give the reference as --reference or as --line, not both")
    campaign = _campaign(args)
    report = campaign.run()
    source = 0
    cells = [c for c in campaign.heard_at[source] if campaign.designs[c] is not None]
    records = campaign.cache.records()
    candidate = np.stack(
        [stored_of_cache(campaign.cache, campaign.key_of(source, c)) for c in cells]
    )
    centres = np.asarray(
        [records[campaign.key_of(source, c)]["centre_m"] for c in cells], dtype=float
    )
    if args.line is not None:
        with np.load(args.line) as held:
            if not np.allclose(held["cells"], campaign.cells, atol=1e-6):
                raise SystemExit("the bundle's cells are not the line's")
            reference = np.asarray(held["stored"][cells], dtype=np.float64)
            reference_centres = [
                (0.0, np.asarray(held["centres_low"])[cells]),
                (float(held["seam_hz"]), np.asarray(held["centres_mid"])[cells]),
            ]
        named = str(args.line)
    else:
        present = PairsCampaign(
            bundle=args.bundle,
            out=args.reference,
            pffdtd_dir=args.pffdtd or Path("/root/pffdtd"),
            devices=args.devices,
            gpu=False if args.cpu else None,
        )
        if not (args.reference / "campaign.done").is_file():
            if args.pffdtd is None:
                raise SystemExit(f"{args.reference} holds no finished campaign; give --pffdtd")
            present.run()
        held_records = present.cache.records()
        keys = [present.key_of(source, c) for c in cells]
        reference = np.stack([stored_of_cache(present.cache, key) for key in keys])
        reference_centres = [
            (0.0, np.asarray([held_records[key]["centre_m"] for key in keys], dtype=float))
        ]
        named = str(args.reference)
    table = compare_responses(
        reference,
        candidate,
        reference_centres=reference_centres,
        candidate_centres=centres,
        sound_speed_m_s=campaign.sound_speed_m_s(),
    )
    table["candidate"] = str(campaign.spec["solver"])
    table["reference"] = named
    table["grid"] = getattr(campaign, "problem_record", {})
    table["solves"] = report.get("solves")
    if args.rate is not None:
        table["ledger"] = campaign.ledger(args.rate)
    (args.out / "compare.json").write_text(json.dumps(table, indent=1))
    print(f"{table['candidate']} against {named}")
    print(format_table(table))
    if "ledger" in table:
        print(json.dumps(table["ledger"], indent=1))
    return 0


def _cost(args: argparse.Namespace) -> int:
    from reverberate.wave.lowband.harness import cost_table

    if args.rate is None:
        raise SystemExit("a cost needs the machine's hourly rate: --rate")
    campaign = _campaign(args)
    campaign.stage("voxelise", campaign.voxelise)
    campaign.stage("plan", campaign.place)
    placed = [c for c, design in enumerate(campaign.designs) if design is not None]
    if not placed:
        raise SystemExit("no cell of the bundle holds an array")
    cells = placed[: args.cells]
    xp = campaign.xp
    encoder = campaign.encoder_on(xp)
    design = campaign.designs[cells[0]]
    table = cost_table(
        campaign.entry_path,
        campaign.sources,
        [campaign.cell_nodes(c) for c in cells],
        xp=xp,
        duration_s=campaign.durations_s["low"],
        batches=tuple(int(b) for b in args.batches.split(",")),
        measured_steps=args.steps,
        rate_usd_per_hour=args.rate,
        cards=max(1, len(campaign.card_indices())),
        encoder=encoder,
        offsets=design.positions - design.centre,
        say=campaign.say,
    )
    table["solver"] = str(campaign.spec["solver"])
    table["device"] = campaign.status["device"]
    (args.out / "cost.json").write_text(json.dumps(table, indent=1, default=str))
    print(json.dumps(table, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "numbers":
        return _numbers(args)
    if args.command == "verify":
        from reverberate.compute import xp_for
        from reverberate.wave.lowband.harness import verify

        verify(
            args.out,
            xp=xp_for(False if args.cpu else None),
            pffdtd_dir=args.pffdtd,
            steps=args.steps,
        )
        return 0
    if args.command == "line":
        from reverberate.wave.lowband.harness import extract_line

        args.out.mkdir(parents=True, exist_ok=True)
        record = extract_line(args.field, args.plan, args.out / "line.npz", every=args.every)
        with np.load(args.out / "line.npz") as held:
            np.save(args.out / "cells.npy", held["cells"])
            np.save(args.out / "sources.npy", held["source"][None, :])
        print(json.dumps(record, indent=1))
        return 0
    if args.command == "compare":
        return _compare(args)
    if args.command == "cost":
        return _cost(args)
    if args.command == "counts":
        from reverberate.wave.lowband.reciprocity import solve_counts

        heard = json.loads(args.heard.read_text())
        cells = 1 + max((c for listed in heard for c in listed), default=-1)
        print(json.dumps(solve_counts(heard, cells, channels=args.channels), indent=1))
        return 0
    raise SystemExit(f"unknown command {args.command}")
