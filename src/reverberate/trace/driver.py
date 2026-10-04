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
from reverberate.trace.plan import Plan, Profile, estimate, make_plan

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
        kind = "measured on 2 x A100" if name in priced["measured"] else "projected"
        lines.append(f"  {name:<12} {seconds:>9.1f} s  {priced['usd'][name]:>7.3f} USD  ({kind})")
    lines.append(
        f"  {'total':<12} {priced['total_s']:>9.1f} s  {priced['total_usd']:>7.2f} USD"
        f"   pack {priced['pack_gb']} GB, pair cache {priced['pair_cache_gb']} GB"
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
    pack = pulled / "pack.h5"
    if pack.is_file():
        if found["cost"]:
            stamp_cost(pack, found["cost"])
        with read_pack(pack) as held:
            found["pack"] = {
                "path": str(pack),
                "bytes": pack.stat().st_size,
                "steps": held.header.steps,
                "sources": list(held.sources),
            }
    if (pulled / "pairs").is_dir():
        from reverberate.accel.pairs import install_pairs

        installed = install_pairs(pulled / "pairs", publish=publish_pairs)
        found["pairs"] = {
            "installed": len(installed["installed"]),
            "published": len(installed["published"]),
        }
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
    hours: float = 8.0,
    max_dph: float = 3.0,
    gpu: str = "",
    avoid: Collection[int] = (),
    instance: int | None = None,
    devices: str | None = None,
    campaign_args: str = "",
    fetch_grid: bool = False,
    publish_pairs: bool = False,
    repo: Path | None = None,
    say: Any = print,
) -> dict[str, Any]:
    """Plan, price, and unless ``dry_run``: bundle, rent, run, fetch, destroy, verify, finish."""
    home = Path(home)
    plan = make_plan(recipe, assets.triangles, profile, patch_centre_xz=patch_centre_xz)
    priced = estimate(plan, rate_usd_per_hour=rate_usd_per_hour)
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
    say(f"bundle: {bundle}  grid {campaign['bands']['low']['cache_key']}")
    from reverberate.gpu import onebox

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
        campaign_args=campaign_args,
        avoid=avoid,
        say=say,
    )
    result["onebox"] = record
    if "outcome" not in record:
        return result
    result["home"] = finish(home, record, publish_pairs=publish_pairs)
    say(f"home: {json.dumps(result['home'])}")
    if record.get("outcome") != "done":
        say(f"the instance {record.get('instance')} was KEPT: outcome {record.get('outcome')}")
    return result
