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

__all__ = ["build_bundle", "code_version", "export_digest"]


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
) -> dict[str, Any]:
    """Everything a trace of ``plan`` reads, into ``bundle``; returns ``campaign.json``."""
    bundle = Path(bundle)
    held = bundle / "trace"
    held.mkdir(parents=True, exist_ok=True)
    save_recipe(recipe, held / "recipe.json")
    priced = estimate(plan, rate_usd_per_hour=rate_usd_per_hour, check=check, low_engine=low_engine)
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
        if with_cache:
            key = str(pairs["bands"]["low"]["cache_key"])
            local = PairCache.local(key)
            carried = PairCache(bundle / "pairs_cache", key)
            records = local.records()
            for position, cells in enumerate(plan.heard_at):
                for cell in cells:
                    name = pair_key(
                        key,
                        plan.tracks.positions[position],
                        plan.all_cells[cell],
                        encoder=dict(pairs["encoder"]),
                        solver=str(pairs["solver"]),
                        window_s=float(pairs["bands"]["low"]["duration_s"]),
                    )
                    if local.has(name) and not carried.has(name):
                        record = {k: v for k, v in records.get(name, {}).items() if k != "key"}
                        carried.path(name).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(local.path(name), carried.path(name))
                        with (carried.directory / "index.jsonl").open("a") as handle:
                            handle.write(json.dumps({"key": name, **record}, sort_keys=True) + "\n")
            campaign["pairs_carried"] = len(carried.records())
    campaign["trace"] = {
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
