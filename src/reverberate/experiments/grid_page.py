"""Publish a voxelisation as a run page, with no solve behind it.

The acoustic view answers the question the mesh view cannot: the mesh says what
left the exporter, the grid says what the solver received, and a surface whose
material landed on the wrong side looks identical in triangles. Until now the
only way to see it was to render a *run* -- which means a plan, a placement,
comms files, a solve, responses and audio. Looking at the geometry cost a GPU.

So: the same page, from the voxelisation alone. It writes the ``plan.json`` and
``report.json`` that :mod:`reverberate.viz.run_view` reads, filled from the
grid, the exported model and the room's own geometry, and nothing else. The
sections that need responses come out empty rather than invented, and
``omissions`` says so on the page.

Usage::

    python -m reverberate.experiments.grid_page \\
        --models data/runs/w32_carved/models --scene bedroom_only \\
        --cache-key 78bca376... --out data/runs/w32_bedroom_16k

Then start the viewer as usual and pick the run:

    python -m reverberate.viz.serve_room data/raw/hssd-hab --runs data/runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reverberate.experiments.run import entry_from_key
from reverberate.experiments.w20_render import room_geometry, theory
from reverberate.wave.voxelise import cache_root

__all__ = ["build", "main"]

#: What this page does not have, said on the page rather than left as an empty
#: panel a reader has to interpret. The run view already renders ``omissions``.
NO_SOLVE = (
    "No solve stands behind this page. It is the geometry and the grid the "
    "solver would read, published so the scene can be checked before any GPU "
    "time is spent on it, so there are no impulse responses, no decay curves "
    "and no audio."
)


def _sound_speed(entry_path: Path) -> float:
    import h5py

    with h5py.File(entry_path / "sim_consts.h5", "r") as handle:
        return float(handle["c"][()])


def build(
    models: Path, scene_name: str, cache_key: str, out: Path, viewer_cubes: int | None = None
) -> dict[str, Any]:
    """Write ``plan.json`` and ``report.json`` for one voxelisation.

    Everything is read back from the artefacts themselves -- the cache entry's
    manifest, its ``vox_out.h5``, and the exported model -- rather than passed
    in, so a page cannot describe a grid it was not built from.
    """
    models, out = Path(models), Path(out)
    manifest = json.loads((models / "manifest.json").read_text())
    scene = {entry["name"]: entry for entry in manifest["scenes"]}[scene_name]
    model_json = (models / scene["file"]).resolve()

    entry = entry_from_key(cache_key)
    geometry = room_geometry(entry.path, model_json, sound_speed_m_s=_sound_speed(entry.path))
    prediction = theory(geometry.volume_m3, geometry.surface_area_m2, geometry.mean_absorption)

    scene_id = str(manifest.get("scene_id", ""))
    room = str(manifest.get("room", scene_name))
    out.mkdir(parents=True, exist_ok=True)

    # An empty placement, not a fabricated one. The page draws a marker per
    # source and per receiver, and a plausible-looking pair nobody placed would
    # be exactly the kind of invented detail this whole module exists to avoid.
    placement: dict[str, Any] = {"sources": [], "receivers": []}

    plan = {
        "scene_id": scene_id,
        "room": room,
        "cache_key": cache_key,
        "cache_root": str(cache_root()),
        "placement": placement,
        "note": NO_SOLVE,
    }
    report = {
        "run": out.name,
        "scene_sha256": hashlib.sha256(model_json.read_bytes()).hexdigest(),
        "cache_key": cache_key,
        "cache_root": str(cache_root()),
        "model_json": str(model_json),
        "room": geometry.record(),
        "theory": prediction.record(),
        "sealed": manifest.get("sealed" if scene_name.startswith("bedroom") else "sealed_full"),
        "placement": placement,
        # The measured sources, as opposed to the placed ones. Empty for the
        # same reason: there is nothing measured on this grid.
        "sources": [],
        "binaural_note": "no receivers: nothing was solved on this grid",
        "dry_voice": None,
        "omissions": [NO_SOLVE],
        # Optional, and only meaningful for a page whose purpose is to be
        # looked at: how many blocks the viewer may spend on it. See
        # ``reverberate.viz.vox_view.TARGET_CUBES``.
        **({"viewer_cubes": viewer_cubes} if viewer_cubes else {}),
        "grid": {
            key: entry.manifest.get(key)
            for key in ("fmax", "h_m", "grid_shape", "grid_points", "boundary_nodes", "triangles")
        },
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="a scene name from the models manifest")
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--viewer-cubes",
        type=int,
        default=None,
        help="blocks the viewer may spend, overriding TARGET_CUBES; a finer picture "
        "and a longer first build",
    )
    args = parser.parse_args(argv)

    report = build(args.models, args.scene, args.cache_key, args.out, args.viewer_cubes)
    grid = report["grid"]
    print(
        f"{args.out.name}: {grid['boundary_nodes']:,} boundary nodes at "
        f"{grid['h_m'] * 1000:.2f} mm, {grid['triangles']:,} triangles, "
        f"fmax {grid['fmax']:g} Hz"
    )
    room = report["room"]
    print(f"volume {room['volume_m3']:.1f} m3, surface {room['surface_area_m2']:.1f} m2")
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
