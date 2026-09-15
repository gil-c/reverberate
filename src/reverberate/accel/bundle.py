"""What the laptop prepares for the machine, and what it takes home.

The machine has no HSSD download and no store key. Everything a campaign
reads that comes from the dataset is therefore prepared here, once, into a
bundle of a few hundred megabytes: the storey's mesh and materials
(:mod:`reverberate.experiments.scene_export`), the listening grid and its
room labels (:func:`plan_points`), the rooms themselves as polygons for the
audit view's partition, the sources, and the campaign's parameters with the
cache key each band's grid will have. The keys are computed here because
they hash the mesh, the materials and the voxeliser, none of which the
machine can change, so what it installs is what the laptop already knows.

Coming home is the reverse: the run directory the machine wrote (fields,
audit view, walk.json, plan, logs) is copied into the run on this machine,
and the grids it voxelised are installed in the local cache and published
to the store when asked, so the next campaign on the same dwelling finds
them.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DEFAULT_BANDS", "STOREY_SCENE", "prepare_bundle", "take_home"]

#: The storey's export, the scene name ``scene_export`` gives it.
STOREY_SCENE = "apartment_full"

#: The three bands of a campaign: the top frequency and the solved window.
DEFAULT_BANDS: dict[str, dict[str, float]] = {
    "low": {"fmax_hz": 1000.0, "duration_s": 1.2},
    "mid": {"fmax_hz": 4000.0, "duration_s": 0.4},
    "high": {"fmax_hz": 8000.0, "duration_s": 0.15},
}

#: What comes home from the run directory on the machine.
HOME_ITEMS = (
    "field",
    "audit",
    "walk.json",
    "plan.json",
    "campaign.log",
    "report.json",
    "status.json",
)


def prepare_bundle(
    bundle: Path,
    *,
    hssd_root: Path,
    scene_id: str,
    sources: list[dict[str, Any]],
    pitch_m: float = 0.40,
    height_m: float = 1.70,
    ram_gb: float = 400.0,
    bands: dict[str, dict[str, float]] | None = None,
    order: int = 7,
    fit_order: int = 10,
    ppw: float = 10.5,
    models_from: Path | None = None,
) -> dict[str, Any]:
    """Export, grid, rooms, sources and parameters into ``bundle``; returns ``campaign.json``.

    ``models_from`` reuses an earlier export (the directory holding
    ``manifest.json`` and the storey's model, with ``materials`` beside it)
    instead of exporting again: the export changes as the geometry code
    does, and a campaign that must reproduce an earlier field must start
    from the same mesh.
    """
    from reverberate.experiments.run import scene_spec
    from reverberate.experiments.w40_volume_field.plan import dwelling_of, free_floor, plan_points
    from reverberate.experiments.w40_volume_field.storey import export_scene
    from reverberate.geometry.rooms import rooms_record

    bands = bands or DEFAULT_BANDS
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    models = bundle / "models" / "storey"
    started = time.time()
    if models_from is not None:
        source_models = Path(models_from)
        if not (source_models / "manifest.json").is_file():
            raise FileNotFoundError(f"{source_models} holds no export manifest")
        if models.exists():
            shutil.rmtree(models)
        models.mkdir(parents=True)
        for item in (source_models / "manifest.json", source_models / f"{STOREY_SCENE}.json"):
            shutil.copy2(item, models / item.name)
        materials_dir = bundle / "models" / "materials"
        if materials_dir.exists():
            shutil.rmtree(materials_dir)
        shutil.copytree(source_models.parent / "materials", materials_dir)
    else:
        export_scene(Path(hssd_root), scene_id, models)
    # The export also writes the room alone and two truncations, which the
    # campaign never reads and would only lengthen the upload.
    for extra in models.glob("*.json"):
        if extra.name not in (f"{STOREY_SCENE}.json", "manifest.json"):
            extra.unlink()
    keys: dict[str, str] = {}
    nh: dict[str, int] = {}
    manifest = json.loads((models / "manifest.json").read_text())
    for band, spec in bands.items():
        scene, _, _ = scene_spec(models, STOREY_SCENE, float(spec["fmax_hz"]))
        if scene.ppw != ppw:
            raise ValueError(f"the export's spec uses {scene.ppw} points per wavelength, not {ppw}")
        keys[band] = scene.key
        nh[band] = int(scene.nh or 0)
    points, labels, area_m2 = plan_points(
        Path(hssd_root), scene_id, height_m=height_m, pitch_m=pitch_m
    )
    _, _, rooms = free_floor(Path(hssd_root), scene_id)
    np.save(bundle / "points.npy", points)
    (bundle / "rooms.json").write_text(json.dumps(rooms_record(list(rooms)), indent=1))
    (bundle / "sources.json").write_text(json.dumps(sources, indent=1))
    campaign = {
        "scene_id": scene_id,
        "dwelling": dwelling_of(scene_id),
        "models": str(models.relative_to(bundle)),
        "model_json": str((models / f"{STOREY_SCENE}.json").relative_to(bundle)),
        "storey_scene": STOREY_SCENE,
        "materials": str((bundle / "models" / "materials").relative_to(bundle)),
        "bands": {
            band: {**spec, "cache_key": keys[band], "nh": nh[band]} for band, spec in bands.items()
        },
        "ppw": ppw,
        "tc": 20.0,
        "rh": 50.0,
        "pitch_m": pitch_m,
        "height_m": height_m,
        "ram_gb": ram_gb,
        "order": order,
        "fit_order": fit_order,
        "points": int(points.shape[0]),
        "labels": labels,
        "free_floor_m2": area_m2,
        "sources": sources,
        "export_manifest": {
            k: manifest.get(k) for k in ("dwelling", "room", "room_regions", "max_edge_m", "seed")
        },
        "prepared_s": round(time.time() - started, 1),
    }
    (bundle / "campaign.json").write_text(json.dumps(campaign, indent=1))
    return campaign


def take_home(
    pulled: Path,
    run_dir: Path,
    *,
    cache_entries: Path | None = None,
    publish: bool = False,
) -> dict[str, Any]:
    """Copy the machine's run into ``run_dir``; install and publish the grids it made.

    ``cache_entries`` is a directory of ``<key>/`` entries fetched from the
    machine's cache; each is moved under this machine's cache root when it
    is not there yet, and published to the store with ``publish``.
    """
    from reverberate.wave.voxelise import CACHE_FILES, cache_root

    pulled, run_dir = Path(pulled), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in HOME_ITEMS:
        source = pulled / name
        if not source.exists():
            continue
        target = run_dir / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        copied.append(name)
    installed: list[str] = []
    published: list[str] = []
    if cache_entries is not None and Path(cache_entries).is_dir():
        root = cache_root()
        for entry in sorted(Path(cache_entries).iterdir()):
            if not (entry / "manifest.json").is_file():
                continue
            if any(not (entry / name).is_file() for name in CACHE_FILES):
                continue
            target = root / entry.name
            if not (target / "manifest.json").is_file():
                shutil.copytree(entry, target)
                installed.append(entry.name)
            if publish:
                from reverberate.store import shared_store
                from reverberate.wave.vox_store import publish_entry
                from reverberate.wave.voxelise import CacheEntry

                store = shared_store()
                if store is not None:
                    manifest = json.loads((target / "manifest.json").read_text())
                    publish_entry(store, CacheEntry(path=target, key=entry.name, manifest=manifest))
                    published.append(entry.name)
    return {"copied": copied, "installed": installed, "published": published, "run": str(run_dir)}
