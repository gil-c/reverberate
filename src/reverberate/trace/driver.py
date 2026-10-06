"""A recipe to a pack on one rented machine, from the laptop, in one command.

The plan and its cost are made and printed first, and with ``dry_run``
nothing else happens: no key is read and no offer is asked for. Otherwise
the bundle is built (:mod:`reverberate.trace.bundle`) and handed to
:func:`reverberate.gpu.onebox.run`, which rents (the offers to avoid
honoured, a host whose cards are not empty refused), provisions, launches
``python -m reverberate.accel campaign``, which reads that the bundle is a
trace, watches, fetches, destroys and verifies. :func:`finish` is the
homecoming: the pack is read, its provenance receives the cost records with
the rate the rental was billed at, and the pairs go into this machine's
cache, and into the shared store when asked.

**A run that fails after its solves says what to do.** Two whole scenes
ended ``campaign.failed`` after five hours of solves (2026-10-05) and the
driver said "kept for inspection". Its last lines are now what the machine
holds (:func:`machine_holds`), what a resume keeps of it and makes again,
the exact command that resumes (:func:`resume_command`), and what an hour
of leaving the machine costs.

Nothing here knows acoustics, and nothing in :mod:`reverberate.trace.run`
knows Vast.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import h5py

from reverberate.scenes import Recipe
from reverberate.trace.assets import MirrorAssets, found_assets, mismatched
from reverberate.trace.bundle import build_bundle, levers_text
from reverberate.trace.plan import (
    RAYS_MEASURED,
    Plan,
    Profile,
    as_made,
    estimate,
    make_plan,
)

__all__ = [
    "cost_records",
    "describe",
    "finish",
    "launch",
    "machine_holds",
    "resume_command",
    "stamp_cost",
]

#: A stage of the machine's report to the stage a cost record names.
_STAGE_OF = {
    "voxelise": "low",
    "plan": "low",
    "solve": "low",
    "assign": "level",
    "paths": "paths",
    "rays": "rays",
    "level": "level",
    "write": "write",
    "check": "check",
}


def describe(plan: Plan, priced: dict[str, Any]) -> str:
    """The plan and its cost in the lines a person reads before renting."""
    r = plan.record
    cells = r["cells"]
    lines = [
        f"recipe {r['recipe_sha256']}  {r['dwelling']}  {r['duration_s']:g} s, {r['steps']} steps"
        + (f"  profile {r['profile']}" if plan.profile != Profile() else ""),
        f"sources: {len(r['sources'])} ({', '.join(r['sources'])})",
        f"audible steps: {r['audible_steps_total']}, of which {r['step_pairs']} distinct"
        " (source, head) positions to trace",
        f"listening cells: {cells['cells']} ({cells['rest_places']} rest places of which"
        f" {cells['seats']} seats, {cells['path_cells']} on {cells['path_m']} m of path,"
        f" {cells['dense_cells']} by the 0.10 m rule, {cells['head_cells']} on the head)",
        f"  steps refused at 0.15 m: {cells['steps_refused_at_0.15_m']}; at 0.10 m:"
        f" {cells['steps_refused_at_0.10_m']}; served by the nearest cell alone, flagged:"
        f" {len(r['fallback_steps'])}",
        f"  cells whose ball of 0.26 m is not free in the mirror's scene:"
        f" {cells['cells_whose_ball_is_not_free']}",
        f"modes: exact {r['modes']['exact']}, translated {r['modes']['translated']},"
        f" fused {r['modes']['fused']}",
        f"low band: {r['source_positions']} source positions, {r['pairs']} pairs"
        + (f" ({r['pairs_of_the_patch']} of the patch)" if r["pairs_of_the_patch"] else ""),
        *_on_the_grid(r),
        f"tail: {r['tail_sites']} sites over {r['tail_cells']} cells",
    ]
    if r["patch"] is not None:
        lines.append(f"patch: {r['patch']}")
    rate = priced["billed_rate_usd_per_hour"]
    lines.append(f"cost at {rate:g} USD/h, machine-seconds and USD by stage:")
    for name, seconds in priced["seconds"].items():
        where = priced.get("measured_on_by_stage", {}).get(name)
        kind = f"measured on {where}" if where else "PROJECTED, no machine has run it"
        lines.append(f"  {name:<14} {seconds:>9.1f} s  {priced['usd'][name]:>7.3f} USD  ({kind})")
    lines.append(
        f"  {'total':<14} {priced['total_s']:>9.1f} s  {priced['total_usd']:>7.2f} USD"
        f"   pack {priced['pack_gb']} GB, pair cache {priced['pair_cache_gb']} GB"
        + ("" if priced.get("pairs_fetched", True) else " (left on the machine)")
    )
    lines.append(
        f"  the pack's low band is written as {priced.get('low_levers') or 'low/ir, the samples'}"
        " (--low-levers); the fetch is priced at the machine's rate, which bills until the"
        " pack is home"
    )
    if "non_solve_s" in priced:
        lines.append(
            f"  all but the low band's solves: {priced['non_solve_s']:.1f} s,"
            f" {priced['non_solve_usd']:.3f} USD; check: {priced.get('check')}"
        )
    return "\n".join(lines)


def _on_the_grid(record: dict[str, Any]) -> list[str]:
    """The line that says how many pairs a machine makes of a plan's, and why they differ."""
    grid = record.get("on_the_grid")
    if not grid:
        return []
    without = int(grid["cells_without_an_array"])
    return [
        f"  on the grid's nodes ({grid['ppw']:g} points, {1000.0 * grid['step_m']:.1f} mm):"
        f" {grid['pairs']} pairs, the count this run is priced by. No array stands on a cell"
        f" asked: {grid['modes']['exact']} steps stay exact of {record['modes']['exact']},"
        f" {grid['modes']['fused']} are fused from two cells for {record['modes']['fused']}"
        + (f"; {without} cell(s) taken to get no array" if without else "")
    ]


#: Flags of ``trace rent`` that a resume does not repeat, and how many words follow each.
_NOT_RESUMED = {"--yes": 0, "--plan-offers": 0, "--dry-run": 0, "--instance": 1}


def resume_command(argv: Sequence[str] | None = None) -> str:
    """The command that resumes a run on its machine, ``{instance}`` standing for the instance.

    ``argv`` is the command's own words (left out: this process's): the
    same recipe, home, mirror and options, without what rents, and
    ``--instance``. The home is what makes it a resume: its bundle is the
    run's, and what came home is carried.
    """
    words = list(sys.argv[1:] if argv is None else argv)
    kept: list[str] = []
    skip = 0
    for word in words:
        if skip:
            skip -= 1
            continue
        if word in _NOT_RESUMED:
            skip = _NOT_RESUMED[word]
            continue
        kept.append(word)
    return f"python -m reverberate.trace {shlex.join(kept)} --instance {{instance}}"


def machine_holds(
    machine: Any, record: dict[str, Any] | None = None, *, run_on: Any = None
) -> list[str]:
    """What a trace's machine holds of its run, and what a resume makes again: lines to say.

    One command on the machine counts what the trace keeps as it goes:
    the pairs in their cache, the early tables, the tails' histograms, the
    pairs levelled, the pack's rows, the pack. A resume reads each of them
    and computes what is missing, so the lines say both. ``record`` is the
    plan's, for the pairs wanted.
    """
    from reverberate.gpu.onebox import REMOTE_OUT
    from reverberate.wave import remote

    ask = run_on or remote.run_on
    said = ask(
        machine,
        f"cd {REMOTE_OUT} 2>/dev/null || exit 0;"
        " echo pairs $(find pairs -name '*.np[yz]' ! -name '*.partial.*' 2>/dev/null | wc -l);"
        " echo early $(find early -name '*.npz' ! -name '*.partial.*' 2>/dev/null | wc -l);"
        " echo tails $(find tails -type f 2>/dev/null | wc -l);"
        " echo level $(cat level.jsonl 2>/dev/null | wc -l);"
        " echo rows $(find jobs/rows -name '*.npy' ! -name '*.partial.*' 2>/dev/null | wc -l);"
        " echo pack $(stat -c %s pack.h5 2>/dev/null || echo 0);"
        " echo failed $(head -c 400 campaign.failed 2>/dev/null | tr '\n' ' ')",
        what="what the machine holds",
        timeout=120,
    )
    found: dict[str, str] = {}
    for line in str(said).splitlines():
        name, _, value = line.strip().partition(" ")
        if name in ("pairs", "early", "tails", "level", "rows", "pack", "failed"):
            found[name] = value.strip()

    def count(name: str) -> int:
        value = found.get(name, "0").split()[:1]
        return int(value[0]) if value and value[0].isdigit() else 0

    wanted = int((record or {}).get("pairs", 0))
    pairs, pack = count("pairs"), count("pack")
    of = f" (the plan counts {wanted})" if wanted else ""
    lines = [
        f"on the machine: {pairs} pairs solved{of}, {count('early')} early tables,"
        f" {count('tails')} histograms, {count('level')} pairs levelled, {count('rows')} blocks"
        f" of the pack's rows, {'no pack' if not pack else f'a pack of {pack / 1e9:.2f} GB'}",
    ]
    again = []
    if wanted and pairs < wanted:
        again.append(f"the solves of about {wanted - pairs} pairs")
    if not count("early"):
        again.append("the early trace")
    if not count("tails"):
        again.append("the rays")
    if not count("level") or (wanted and count("level") < min(pairs, wanted)):
        again.append("the levelling of the pairs level.jsonl lacks")
    if not pack:
        again.append("the pack's write and its check")
    lines.append(
        "a resume keeps all of it and makes again: "
        + ("; ".join(again) if again else "nothing but the fetch")
        + ". A stage's check runs again on what is kept"
    )
    if found.get("failed"):
        lines.append(f"it stopped with: {found['failed'][:300]}")
    return lines


def cost_records(
    report: dict[str, Any],
    *,
    billed_rate_usd_per_hour: float,
    instance: int | str,
    fetch_s: float = 0.0,
    billed_s: float | None = None,
) -> list[dict[str, Any]]:
    """A pack's cost records, one a stage, from the machine's report and the rental's rate.

    ``billed_s`` is what the rental was billed for in all; what the stages
    and the fetch do not account for (the provisioning, the bundle's push,
    the watcher's polls) is the record ``rental``. A cost without its rate
    is not written: the machine does not know what it is billed.
    """
    device = report.get("device", {})
    rate = float(billed_rate_usd_per_hour)
    seconds: dict[str, float] = {}
    for name, value in dict(report.get("timings_s", {})).items():
        stage = _STAGE_OF.get(str(name), str(name))
        seconds[stage] = seconds.get(stage, 0.0) + float(value)
    if fetch_s:
        seconds["transfer"] = float(fetch_s)
    if billed_s is not None:
        seconds["rental"] = max(float(billed_s) - sum(seconds.values()), 0.0)
    return [
        {
            "stage": stage,
            "seconds": round(value, 1),
            "card": device.get("gpu"),
            "cards": device.get("devices"),
            "billed_rate_usd_per_hour": rate,
            "usd": round(value / 3600.0 * rate, 4),
            "instance": instance,
        }
        for stage, value in seconds.items()
    ]


def stamp_cost(pack: Path, records: list[dict[str, Any]]) -> None:
    """The cost records into the pack's ``provenance_json``; nothing else of the file moves."""
    with h5py.File(pack, "r+") as handle:
        text = handle.attrs["provenance_json"]
        held = json.loads(text.decode() if isinstance(text, bytes) else str(text))
        held["cost"] = records
        handle.attrs["provenance_json"] = json.dumps(held, sort_keys=True)


def finish(home: Path, record: dict[str, Any], *, publish_pairs: bool = False) -> dict[str, Any]:
    """What came home: the pack checked, its cost stamped, the pairs installed."""
    from reverberate.render.pack import read_pack

    pulled = Path(home) / "pulled"
    found: dict[str, Any] = {"pack": None, "cost": [], "pairs": None}
    report_file = pulled / "trace_report.json"
    report = json.loads(report_file.read_text()) if report_file.is_file() else {}
    hours, cost = record.get("hours_billed"), record.get("cost_usd")
    if hours and cost is not None:
        found["cost"] = cost_records(
            report,
            billed_rate_usd_per_hour=float(cost) / float(hours),
            instance=record.get("instance", ""),
            fetch_s=float(record.get("fetch_s", 0.0)),
            billed_s=float(hours) * 3600.0,
        )
    # The pairs first, and whatever the rest is worth: after a run that did not end they
    # are what came home, and the next run of the recipe carries them.
    if (pulled / "pairs").is_dir():
        from reverberate.accel.pairs import install_pairs

        installed = install_pairs(pulled / "pairs", publish=publish_pairs)
        found["pairs"] = {
            "installed": len(installed["installed"]),
            "published": len(installed["published"]),
        }
        if installed["damaged"]:
            # Cut in their transfer: left out of the cache, and solved again by the next run.
            found["pairs_damaged"] = len(installed["damaged"])
    pack = pulled / "pack.h5"
    if pack.is_file():
        try:
            if found["cost"]:
                stamp_cost(pack, found["cost"])
            with read_pack(pack) as held:
                found["pack"] = {
                    "path": str(pack),
                    "bytes": pack.stat().st_size,
                    "steps": held.header.steps,
                    "sources": list(held.sources),
                }
        except Exception as error:  # noqa: BLE001 - a transfer that was cut leaves part of a file
            if record.get("outcome") == "done" and not record.get("fetch_error"):
                raise
            found["pack_unreadable"] = repr(error)[:300]
    found["render"] = report.get("render")
    found["usd"] = round(sum(float(r["usd"]) for r in found["cost"]), 4)
    (Path(home) / "cost.json").write_text(json.dumps(found, indent=1))
    return found


def launch(
    recipe: Recipe,
    assets: MirrorAssets,
    home: Path,
    *,
    profile: Profile | None = None,
    patch_centre_xz: tuple[float, float] | None = None,
    models_from: Path | None = None,
    hssd_root: Path | None = None,
    rate_usd_per_hour: float = 1.74,
    dry_run: bool = False,
    allow_asset_mismatch: bool = False,
    yes: bool = False,
    hours: float | None = None,
    max_dph: float = 3.0,
    gpus: int = 1,
    max_hours: float | None = None,
    plan_offers: bool = False,
    low_scheme: str = "cartesian",
    low_ppw: float | None = None,
    low_seconds: float | None = None,
    reuse_from: Path | None = None,
    fetch_early: bool = False,
    destroy_failed: bool = False,
    sync_s: float | None = None,
    gpu: str = "",
    avoid: Collection[int] = (),
    instance: int | None = None,
    devices: str | None = None,
    campaign_args: str = "",
    fetch_grid: bool = False,
    publish_pairs: bool = False,
    fetch_pairs: bool | None = None,
    check: str | None = None,
    repo: Path | None = None,
    say: Any = print,
    low_engine: str = "lowband",
    low_levers: str | None = None,
    line: str = "proxy",
    resume: str = "",
    relaunch: bool = False,
) -> dict[str, Any]:
    """Plan, price, and unless ``dry_run``: bundle, rent, run, fetch, destroy, verify, finish.

    ``low_levers`` is the form the machine writes the pack's low band in
    (:data:`reverberate.trace.bundle.LOW_LEVERS` when left out, ``none``
    for the samples); the estimate's pack and its way home follow, and so
    does the pair cache's form. ``line`` prices the offers' way home
    through the proxy or by the laptop's line to each host
    (:func:`reverberate.trace.machines.line_bytes_per_s`). ``resume`` is
    the command that resumes this run on its machine
    (:func:`resume_command`): the last lines of a run that leaves its
    machine rented say it, after what the machine holds. A resume on a
    machine whose trace is done fetches and launches nothing, unless
    ``relaunch``.

    ``fetch_pairs`` brings the pair cache home beside the pack. The pack
    holds every response a render reads; the cache holds them before their
    air and their masks, which is what a trace of another crossover or
    another scene of the dwelling would not solve again. ``None`` brings it
    for the whole scene and leaves it for a smoke run; publishing needs it
    home. When it is brought, it is brought while the run lasts, every
    ``sync_s``, and installed in this machine's cache whatever the outcome:
    a host that dies takes its last minutes and no more, and the next run
    of the recipe carries what came home and solves the rest.

    The machine is chosen by what this plan is predicted to cost on each
    offer (:mod:`reverberate.trace.machines`), the lowest USD within
    ``max_hours`` of wall time, among hosts of ``gpus`` cards or more; the
    watchdog's ``hours`` is taken from the prediction unless given.
    ``plan_offers`` builds the bundle, says the offers and their
    predictions, and rents nothing.

    ``low_scheme`` and ``low_ppw`` put the low band on another grid, whose
    pairs have their own keys; such a run has its own ``home``.
    ``reuse_from`` is the home of an earlier run of the recipe, whose early
    tables (brought home with ``fetch_early``) are not traced again.
    """
    home = Path(home)
    plan = make_plan(
        recipe,
        assets.triangles,
        profile,
        patch_centre_xz=patch_centre_xz,
        low_ppw=low_ppw if low_engine == "lowband" else None,
    )
    cast = int(assets.settings.rays.rays)
    rays = None if cast == RAYS_MEASURED else cast
    brought = (
        plan.profile.seconds is None if fetch_pairs is None else bool(fetch_pairs)
    ) or publish_pairs
    levers = levers_text(low_levers)
    priced = estimate(
        plan,
        rate_usd_per_hour=rate_usd_per_hour,
        fetch_pairs=brought,
        check=check,
        low_engine=low_engine,
        low_ppw=low_ppw if low_engine == "lowband" else None,
        low_seconds=low_seconds,
        rays=rays,
        low_levers=levers,
    )
    say(describe(plan, priced))
    result: dict[str, Any] = {"plan": plan.record, "estimate": priced}
    if dry_run:
        # The grid's key and the export's digest are known once the bundle holds the export.
        unknown = {"voxel_low_key"} | ({"export_sha256"} if not assets.export_sha256 else set())
        wrong = [
            name
            for name in mismatched(recipe, found_assets(recipe, assets, voxel_low_key=""))
            if name not in unknown
        ]
        if wrong:
            say("the recipe's asset keys that are not this dwelling's: " + ", ".join(wrong))
        say("dry run: nothing built, nothing rented")
        return result
    if models_from is None and hssd_root is None:
        raise SystemExit("a rental needs the storey's export: --models-from or --hssd-root")
    repo = repo or Path(__file__).resolve().parents[3]
    bundle = home / "bundle"
    asked = {"engine": low_engine, "scheme": low_scheme, "ppw": low_ppw}
    if low_seconds is not None:
        asked["seconds"] = float(low_seconds)
    earlier = bundle / "campaign.json"
    if earlier.is_file():
        # Two grids' runs in one home would leave one's pack beside the other's pairs.
        held = dict(json.loads(earlier.read_text()).get("trace", {})).get("low")
        named = [*asked, *(["seconds"] if held and "seconds" in held else [])]
        was = None if held is None else {name: held.get(name) for name in dict.fromkeys(named)}
        if was is not None and was != asked:
            raise SystemExit(
                f"{home} is the home of a run whose low band is {was}, not {asked}:"
                " another grid, engine or duration has its own home"
            )
    campaign = build_bundle(
        bundle,
        recipe,
        assets,
        plan,
        models_from=models_from,
        hssd_root=hssd_root,
        allow_asset_mismatch=allow_asset_mismatch,
        rate_usd_per_hour=rate_usd_per_hour,
        repo=repo,
        check=check,
        low_engine=low_engine,
        low_scheme=low_scheme,
        low_ppw=low_ppw,
        low_seconds=low_seconds,
        reuse_from=reuse_from,
        low_levers=levers,
    )
    found = found_assets(
        recipe,
        assets,
        voxel_low_key=str(campaign["bands"]["low"]["cache_key"]),
        export_sha256=str(campaign["trace"]["export_sha256"]),
    )
    wrong = mismatched(recipe, found)
    if wrong and not allow_asset_mismatch:
        raise SystemExit(
            "the recipe was generated against other assets: "
            + ", ".join(wrong)
            + f"\nthis trace finds {json.dumps(found)}"
        )
    low = dict(campaign["trace"]["low"])
    say(
        f"bundle: {bundle}  grid {campaign['bands']['low']['cache_key']}"
        + ("" if low.get("bundle_grid", True) else f"; the low band on {low['voxel_low_key']}")
        + f"; {campaign.get('pairs_carried', 0)} pair(s) carried from this machine's cache"
        + (f", {campaign['early_carried']} early table(s)" if "early_carried" in campaign else "")
    )
    from reverberate.gpu import onebox
    from reverberate.trace.machines import predictor

    leave = () if brought else ("pairs",)
    if leave:
        say("the pair cache stays on the machine and goes with it: --fetch-pairs brings it home")
    # The machine's command reads the engine and the grid in the bundle; said again here
    # so that the command line of the run shows them.
    flags = f"--low-engine {low_engine}"
    if low_engine == "lowband" and (low_scheme != "cartesian" or low_ppw is not None):
        flags += f" --low-scheme {low_scheme}" + (f" --low-ppw {low_ppw:g}" if low_ppw else "")
    record = onebox.run(
        bundle,
        home,
        hours=hours,
        max_dph=max_dph,
        yes=yes,
        repo=repo,
        instance=instance,
        devices=devices,
        fetch_cache=fetch_grid,
        gpu=gpu,
        campaign_args=f"{flags} {campaign_args}".strip(),
        avoid=avoid,
        leave=leave,
        also=("early",) if fetch_early else (),
        say=say,
        gpus=gpus,
        max_hours=max_hours,
        # The offers are priced for the pairs still to solve: what is carried is not paid.
        predict=predictor(
            _unsolved(as_made(plan.record), int(campaign.get("pairs_carried", 0))),
            fetch_pairs=brought,
            check=check,
            low_engine=low_engine,
            low_ppw=low_ppw,
            low_seconds=low_seconds,
            rays=rays,
            low_levers=levers,
            line=line,
            # The disk the rental asks for is billed by the hour with the cards.
            disk_gb=_disk_gb(bundle),
        )
        if low_engine == "lowband"
        else None,
        resume_command=resume,
        relaunch=relaunch,
        inventory=lambda machine: machine_holds(machine, as_made(plan.record)),
        sync=("pairs",) if brought else (),
        **({} if sync_s is None else {"sync_s": sync_s}),
        plan_only=plan_offers,
        destroy_failed=destroy_failed,
        # The machine's own prediction, from what it measures of itself, against the cap.
        cap_flag="--max-hours",
    )
    result["onebox"] = record
    if "instance" not in record and "outcome" not in record:
        return result
    # Whatever the outcome: what came home is installed, so the next run carries it.
    result["home"] = finish(home, record, publish_pairs=publish_pairs)
    say(f"home: {json.dumps(result['home'])}")
    if record.get("left_alive"):
        say(str(record["left_alive"]))
    return result


def _disk_gb(bundle: Path) -> float:
    """The disk the rental of ``bundle`` asks for, GB; 0 where the bundle does not size one."""
    from reverberate.gpu import onebox

    try:
        return float(onebox.campaign_need(bundle).disk_gb)
    except (KeyError, OSError, ValueError):
        return 0.0


def _unsolved(record: dict[str, Any], carried: int) -> dict[str, Any]:
    """A plan's record with the pairs the bundle carries taken off its low band.

    Which positions the carried pairs belong to is not looked up: every
    position's cells are thinned in the pairs' proportion, which is exact
    when nothing or everything is carried and counts too many solves
    between, the side a watchdog's cap should err on.
    """
    pairs = int(record["pairs"])
    if carried <= 0 or pairs <= 0:
        return record
    left = max(0.0, 1.0 - carried / pairs)
    counts = [round(int(c) * left) for c in record.get("cells_a_position", [])]
    kept = [c for c in counts if c > 0]
    return {
        **record,
        "pairs": max(0, pairs - carried),
        "source_positions": len(kept) if kept else round(int(record["source_positions"]) * left),
        "cells_a_position": kept,
    }
