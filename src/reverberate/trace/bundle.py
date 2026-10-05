"""What the laptop prepares for a trace: one directory the machine needs nothing beside.

```
<bundle>/campaign.json      what it is (``scene-trace``), the grid's band, the trace's settings
<bundle>/pairs/             the low band pairs' bundle (``accel.pairs``): export, positions, cells
<bundle>/trace/recipe.json  the recipe, canonical
<bundle>/trace/plan.json    the plan's record, with its cost
<bundle>/trace/plan.npz     the cells asked for, and the patch
<bundle>/trace/positions.npy the source positions, as the laptop read them in the recipe
<bundle>/trace/mirror/      the mirror's scene, calibration, signature and alignment
<bundle>/pairs_cache/       pairs this machine already holds, so that they are not solved again
<bundle>/early_cache/       early tables of an earlier run of the recipe (``reuse_from``)
```

``campaign.json`` carries what :func:`reverberate.gpu.onebox.campaign_need`
sizes a machine by, so the rental is a field's. Without an export
(``models_from`` and ``hssd_root`` both left out) there is no ``pairs``
directory and the bundle runs with the free field engine alone, on a laptop.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.hybrid import Crossover
from reverberate.scenes import Recipe, save_recipe
from reverberate.spatial.lowband import pair_key
from reverberate.trace.assets import MirrorAssets
from reverberate.trace.plan import Plan, estimate
from reverberate.trace.run import KIND

__all__ = ["build_bundle", "carry_early", "carry_pairs", "code_version", "export_digest"]


def code_version(repo: Path) -> str:
    """The commit of ``reverberate`` that prepared the bundle; ``unknown`` outside a checkout."""
    try:
        found = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return found.stdout.strip() or "unknown"


def export_digest(models: Path) -> str:
    """The export's digest as ``mirror.pipeline.derive_scene`` computes it: the recipe's key."""
    digest = hashlib.sha256((Path(models) / "apartment_full.json").read_bytes())
    manifest = Path(models) / "manifest.json"
    if manifest.is_file():
        digest.update(manifest.read_bytes())
    return digest.hexdigest()


def carry_pairs(carried_root: Path, local: Any, keys: list[str]) -> int:
    """The pairs of ``keys`` this machine's cache holds, into the bundle; how many it carries.

    ``local`` is the cache of the grid the machine will solve on, and the
    keys are the machine's own (grid, positions, encoder, solver, window),
    so what is carried is found there and not solved again.
    """
    from reverberate.accel.pairs import PairCache

    carried = PairCache(Path(carried_root), local.voxel_low_key)
    records = local.records()
    for name in keys:
        if local.has(name) and not carried.has(name):
            record = {k: v for k, v in records.get(name, {}).items() if k != "key"}
            carried.path(name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local.path(name), carried.path(name))
            with (carried.directory / "index.jsonl").open("a") as handle:
                handle.write(json.dumps({"key": name, **record}, sort_keys=True) + "\n")
    return len(carried.records())


def carry_early(carried: Path, home: Path) -> int:
    """The early tables an earlier run brought home, into the bundle; how many.

    ``home`` is that run's home, or the directory of its tables. A table
    is used by the machine only if its digest is the new run's.
    """
    found = next((d for d in (home / "pulled" / "early", home / "early", home) if d.is_dir()), None)
    tables = sorted(found.glob("*.npz")) if found is not None else []
    if not tables:
        raise SystemExit(f"{home} holds no early table: was the run fetched with --fetch-early?")
    Path(carried).mkdir(parents=True, exist_ok=True)
    for table in tables:
        shutil.copy2(table, Path(carried) / table.name)
    return len(tables)


def build_bundle(
    bundle: Path,
    recipe: Recipe,
    assets: MirrorAssets,
    plan: Plan,
    *,
    crossover: Crossover | None = None,
    models_from: Path | None = None,
    hssd_root: Path | None = None,
    allow_asset_mismatch: bool = False,
    rate_usd_per_hour: float = 1.74,
    repo: Path | None = None,
    with_cache: bool = True,
    check: str | None = None,
    low_engine: str = "lowband",
    low_scheme: str = "cartesian",
    low_ppw: float | None = None,
    reuse_from: Path | None = None,
) -> dict[str, Any]:
    """Everything a trace of ``plan`` reads, into ``bundle``; returns ``campaign.json``.

    ``low_engine``, ``low_scheme`` and ``low_ppw`` say what solves the low
    band and on which grid; the machine's command reads them in
    ``trace.low``, and the pairs carried are named as that engine on that
    grid will ask for them. A grid other than the bundle's is another key:
    its pairs never meet the validated grid's, in this machine's cache or
    in the store, and the recipe's ``voxel_low_key`` is then allowed to
    differ, that key alone.

    ``reuse_from`` is the home of an earlier run of the recipe: the early
    tables it brought home are carried (``early_cache``), and a table whose
    sources, heads and scene are this run's is not traced again.
    """
    bundle = Path(bundle)
    held = bundle / "trace"
    held.mkdir(parents=True, exist_ok=True)
    save_recipe(recipe, held / "recipe.json")
    priced = estimate(
        plan,
        rate_usd_per_hour=rate_usd_per_hour,
        check=check,
        low_engine=low_engine,
        low_ppw=low_ppw if low_engine == "lowband" else None,
    )
    plan.save(held)
    (held / "plan.json").write_text(
        json.dumps({**plan.record, "estimate": priced}, indent=1, sort_keys=True)
    )
    np.save(held / "positions.npy", plan.tracks.positions)
    assets.save(held / "mirror")
    campaign: dict[str, Any] = {
        "kind": KIND,
        "scene_id": recipe.dwelling.scene_id,
        "dwelling": recipe.dwelling.name,
    }
    export = ""
    if models_from is not None or hssd_root is not None:
        from reverberate.accel.pairs import PairCache, prepare_pairs_bundle

        pairs = prepare_pairs_bundle(
            bundle / "pairs",
            scene_id=recipe.dwelling.scene_id,
            sources=plan.tracks.positions,
            cells=plan.all_cells,
            heard_at=plan.heard_at,
            models_from=models_from,
            hssd_root=hssd_root,
        )
        export = export_digest(bundle / "pairs" / pairs["models"])
        # What the rental is sized by, read from this directory and not from ``pairs``.
        for name in ("bands", "ppw", "tc", "rh", "ram_gb", "points", "storey_scene", "solver"):
            campaign[name] = pairs[name]
        for name in ("models", "model_json", "materials"):
            campaign[name] = str(Path("pairs") / pairs[name])
        # The grid and the solver's name as the machine's engine will key its pairs.
        key, solver = str(pairs["bands"]["low"]["cache_key"]), str(pairs["solver"])
        if low_engine == "lowband":
            from reverberate.wave.lowband.pairs import low_grid

            scene, solver = low_grid(
                bundle / "pairs" / str(pairs["models"]),
                str(pairs["storey_scene"]),
                float(pairs["bands"]["low"]["fmax_hz"]),
                scheme=low_scheme,
                ppw=low_ppw,
                bundle_ppw=float(pairs["ppw"]),
            )
            key = str(scene.key)
        elif low_scheme != "cartesian" or low_ppw is not None:
            raise ValueError("another grid than the bundle's is the batched solver's alone")
        low = {
            "engine": low_engine,
            "scheme": low_scheme,
            "ppw": low_ppw,
            "voxel_low_key": key,
            "solver": solver,
            "bundle_grid": key == str(pairs["bands"]["low"]["cache_key"]),
        }
        if with_cache:
            campaign["pairs_carried"] = carry_pairs(
                bundle / "pairs_cache",
                PairCache.local(key),
                [
                    pair_key(
                        key,
                        plan.tracks.positions[position],
                        plan.all_cells[cell],
                        encoder=dict(pairs["encoder"]),
                        solver=solver,
                        window_s=float(pairs["bands"]["low"]["duration_s"]),
                    )
                    for position, cells in enumerate(plan.heard_at)
                    for cell in cells
                ],
            )
    else:
        low = {"engine": low_engine, "scheme": low_scheme, "ppw": low_ppw}
    if reuse_from is not None:
        campaign["early_carried"] = carry_early(bundle / "early_cache", Path(reuse_from))
    campaign["trace"] = {
        "low": low,
        # The one key that differs by intent when the low band is on another grid.
        "allowed_mismatch": [] if low.get("bundle_grid", True) else ["voxel_low_key"],
        "profile": plan.profile.record(),
        "crossover": (crossover or Crossover()).record(),
        "allow_asset_mismatch": bool(allow_asset_mismatch),
        "code_version": code_version(repo or Path(__file__).resolve().parents[3]),
        "export_sha256": export or assets.export_sha256,
        "recipe_sha256": plan.recipe_sha256,
    }
    if check is not None:
        # What the machine's ``check`` stage does; left out, the profile decides
        # (:data:`reverberate.trace.run.CHECK_FULL` for a smoke run).
        campaign["trace"]["check"] = str(check)
    campaign["estimate"] = priced
    (bundle / "campaign.json").write_text(json.dumps(campaign, indent=1))
    return campaign
