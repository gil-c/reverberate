"""The mirror solver's audit for the app: what the mirror read, drawable.

The reflecting facets and the decimated occluders as their own triangles,
coloured by material label, and per listening point the paths the image
sources validated. Everything is written from ``mirror/scene`` and
``mirror/paths_<source>.npz``, the very files the mirror read and wrote, so
the picture is the solver's input by construction.

Triangles go out in the quad payload of :mod:`reverberate.viz.vox_view`
(each triangle a quad whose fourth corner repeats the third), which the
app's mesh loader draws with the grid's shader and palette.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.files import load_paths
from reverberate.mirror.geometry import DerivedScene, load_derived
from reverberate.mirror.ism import Paths

__all__ = ["mirror_record", "write_layers", "write_paths"]

SOUND_SPEED_M_S = 343.2


def _quads(vertices: np.ndarray, labels: np.ndarray, target: Path, stem: str) -> dict[str, Any]:
    """``[n, 3, 3]`` triangles as quads: corners f32, index u32, label i16 per corner."""
    count = int(vertices.shape[0])
    corners = np.concatenate([vertices, vertices[:, 2:3, :]], axis=1)
    index = (
        (np.arange(count, dtype=np.uint32) * 4)[:, None]
        + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    ).ravel()
    (target / f"{stem}.f32").write_bytes(corners.reshape(-1, 3).astype(np.float32).tobytes())
    (target / f"{stem}_index.u32").write_bytes(index.tobytes())
    (target / f"{stem}_label.i16").write_bytes(np.repeat(labels.astype(np.int16), 4).tobytes())
    lo = vertices.reshape(-1, 3).min(axis=0) if count else np.zeros(3)
    hi = vertices.reshape(-1, 3).max(axis=0) if count else np.zeros(3)
    return {
        "quads": count,
        "triangles": count,
        "bytes": count * 80,
        "corners_url": f"{stem}.f32",
        "index_url": f"{stem}_index.u32",
        "label_url": f"{stem}_label.i16",
        "bounds": [lo.tolist(), hi.tolist()],
    }


def write_layers(scene: DerivedScene, target: Path) -> Path:
    """``layers.json`` and the reflector and occluder payloads under ``target``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    facet_labels = np.asarray([f.label for f in scene.facets], dtype=np.int16)
    reflector_labels = (
        facet_labels[scene.reflector_facet] if facet_labels.size else np.zeros(0, dtype=np.int16)
    )
    record = {
        "key": scene.key,
        "labels": list(scene.labels),
        "layers": {
            "reflectors": _quads(scene.reflector_vertices, reflector_labels, target, "reflectors"),
            "occluders": _quads(scene.occluder_vertices, scene.occluder_label, target, "occluders"),
        },
        "facets": [
            {
                "label": scene.labels[f.label],
                "kind": f.kind,
                "area_m2": round(float(f.area), 3),
                "normal": [round(float(v), 4) for v in f.normal],
                "triangles": int(f.triangles.size),
                "sides": int(f.sides),
            }
            for f in scene.facets
        ],
        "census": scene.census,
        "rules": scene.rules.record(),
        "materials": {
            "bands_hz": [int(v) for v in scene.materials.bands_hz],
            "absorption": np.round(scene.materials.absorption, 4).tolist(),
            "scattering": np.round(scene.materials.scattering, 4).tolist(),
            "source": [str(v) for v in scene.materials.source],
        },
    }
    path = target / "layers.json"
    path.write_text(json.dumps(record))
    return path


def write_paths(every: list[Paths], scene: DerivedScene, target: Path) -> Path:
    """Per point, ``[order, time_ms, facets bounced on, flat vertices]`` for every path."""
    out = []
    for paths in every:
        rows = []
        for k in np.argsort(paths.length_m):
            order = int(paths.order[k])
            rows.append(
                [
                    order,
                    round(float(paths.length_m[k] / SOUND_SPEED_M_S * 1000.0), 3),
                    [
                        f"{scene.facets[int(f)].kind}/{scene.labels[scene.facets[int(f)].label]}"
                        for f in paths.sequence[k]
                        if f >= 0
                    ],
                    [round(float(v), 3) for v in paths.points[k, : order + 2].ravel()],
                ]
            )
        out.append(rows)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"points": out}))
    return target


def mirror_record(scene_path: Path, paths: dict[str, Path], site: Path) -> dict[str, Any] | None:
    """The audit of the derived scene at ``scene_path`` and each source's paths file, in ``site``.

    Returns the record ``run.json`` carries, or ``None`` without a scene.
    The layers are rewritten only when the scene's key changed.
    """
    if not Path(scene_path).with_suffix(".json").is_file():
        return None
    scene = load_derived(scene_path)
    audit = Path(site) / "mirror" / "audit"
    layers = audit / "layers.json"
    if not layers.is_file() or json.loads(layers.read_text()).get("key") != scene.key:
        write_layers(scene, audit)
    urls: dict[str, str] = {}
    for source, found in paths.items():
        if Path(found).is_file():
            write_paths(
                load_paths(found), scene, Path(site) / "mirror" / "paths" / f"{source}.json"
            )
            urls[source] = f"mirror/paths/{source}.json"
    return {"audit": "mirror/audit", "paths": urls, "key": scene.key}
