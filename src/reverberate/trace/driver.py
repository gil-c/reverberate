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

Nothing here knows acoustics, and nothing in :mod:`reverberate.trace.run`
knows Vast.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import h5py

from reverberate.scenes import Recipe
from reverberate.trace.assets import MirrorAssets, found_assets, mismatched
from reverberate.trace.bundle import build_bundle
from reverberate.trace.plan import RAYS_MEASURED, Plan, Profile, estimate, make_plan

__all__ = ["cost_records", "describe", "finish", "launch", "stamp_cost"]

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
    if "non_solve_s" in priced:
        lines.append(
            f"  all but the low band's solves: {priced['non_solve_s']:.1f} s,"
            f" {priced['non_solve_usd']:.3f} USD; check: {priced.get('check')}"
        )
    return "\n".join(lines)


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
) -> dict[str, Any]:
    """Plan, price, and unless ``dry_run``: bundle, rent, run, fetch, destroy, verify, finish.

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
    plan = make_plan(recipe, assets.triangles, profile, patch_centre_xz=patch_centre_xz)
    cast = int(assets.settings.rays.rays)
    rays = None if cast == RAYS_MEASURED else cast
    brought = (
        plan.profile.seconds is None if fetch_pairs is None else bool(fetch_pairs)
    ) or publish_pairs
    priced = estimate(
        plan,
        rate_usd_per_hour=rate_usd_per_hour,
        fetch_pairs=brought,
        check=check,
        low_engine=low_engine,
        low_ppw=low_ppw if low_engine == "lowband" else None,
        low_seconds=low_seconds,
        rays=rays,
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
            _unsolved(plan.record, int(campaign.get("pairs_carried", 0))),
            fetch_pairs=brought,
            check=check,
            low_engine=low_engine,
            low_ppw=low_ppw,
            low_seconds=low_seconds,
            rays=rays,
        )
        if low_engine == "lowband"
        else None,
        sync=("pairs",) if brought else (),
        **({} if sync_s is None else {"sync_s": sync_s}),
        plan_only=plan_offers,
        destroy_failed=destroy_failed,
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
