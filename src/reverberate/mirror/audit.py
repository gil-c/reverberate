"""What the mirror was handed, drawable: the reflectors, the occluders, the paths.

The audit view of the wave solver draws the grid the solver read (ADR 0007).
The mirror's audit draws what the mirror read: every reflecting facet as its
own triangles coloured by material label, the decimated occluders the same
way, and, per listening point, the paths the image source model validated,
from the source through every hit point to the point. The census of
:mod:`reverberate.mirror.geometry` rides beside them, so what was left out
is counted where the picture is.

The triangles are written in the quad payload of :mod:`reverberate.viz.vox_view`
(each triangle a quad whose fourth corner repeats the third), so the app's
mesh loader draws them with the grid's own shader and palette.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import Paths

__all__ = ["write_geometry_layers", "write_paths"]


def _quads(vertices: np.ndarray, labels: np.ndarray, target: Path, stem: str) -> dict[str, Any]:
    """``[n, 3, 3]`` triangles as quads: corners f32, index u32, label i16 per corner."""
    target.mkdir(parents=True, exist_ok=True)
    count = int(vertices.shape[0])
    corners = np.concatenate([vertices, vertices[:, 2:3, :]], axis=1)  # [n, 4, 3]
    base = (np.arange(count, dtype=np.uint32) * 4)[:, None]
    index = (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).ravel()
    (target / f"{stem}.f32").write_bytes(corners.reshape(-1, 3).astype(np.float32).tobytes())
    (target / f"{stem}_index.u32").write_bytes(index.tobytes())
    (target / f"{stem}_label.i16").write_bytes(
        np.repeat(labels.astype(np.int16), 4).astype(np.int16).tobytes()
    )
    lo = vertices.reshape(-1, 3).min(axis=0) if count else np.zeros(3)
    hi = vertices.reshape(-1, 3).max(axis=0) if count else np.zeros(3)
    return {
        "quads": count,
        "triangles": count,
        "bytes": count * 80,
        "corners_url": f"{stem}.f32",
        "index_url": f"{stem}_index.u32",
        "label_url": f"{stem}_label.i16",
        "bounds": [[float(v) for v in lo], [float(v) for v in hi]],
    }


def write_geometry_layers(scene: DerivedScene, target: Path) -> Path:
    """``layers.json`` and the two quad payloads under ``target``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    facet_labels = np.asarray([f.label for f in scene.facets], dtype=np.int16)
    reflector_labels = (
        facet_labels[scene.reflector_facet] if facet_labels.size else np.zeros(0, dtype=np.int16)
    )
    reflectors = _quads(scene.reflector_vertices, reflector_labels, target, "reflectors")
    occluders = _quads(scene.occluder_vertices, scene.occluder_label, target, "occluders")
    facets = [
        {
            "index": i,
            "label": scene.labels[f.label],
            "kind": f.kind,
            "area_m2": round(float(f.area), 3),
            "normal": [round(float(v), 4) for v in f.normal],
            "triangles": int(f.triangles.size),
            "sides": int(f.sides),
        }
        for i, f in enumerate(scene.facets)
    ]
    record = {
        "key": scene.key,
        "labels": list(scene.labels),
        "layers": {"reflectors": reflectors, "occluders": occluders},
        "facets": facets,
        "census": scene.census,
        "rules": scene.rules.record(),
        "note": (
            "What the mirror read: reflectors are the planar facets above the area rule, "
            "drawn as their own triangles; occluders are the decimated outer surfaces "
            "(open meshes kept whole). Colours are material labels, as in the grid view. "
            "The census counts what reflects, what is diffuse and what was dropped."
        ),
    }
    path = target / "layers.json"
    path.write_text(json.dumps(record))
    return path


def write_paths(
    every: list[Paths], scene: DerivedScene, target: Path, *, sound_speed_m_s: float
) -> Path:
    """The validated paths of every point, for the app to draw at the listener's cell.

    One JSON: per point, a list of ``[order, time_ms, kinds, points]`` where
    ``points`` is the flat list of the path's vertices (source, hits, point)
    in scene coordinates and ``kinds`` names the facets bounced on.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    out: list[list[Any]] = []
    for paths in every:
        rows = []
        for k in np.argsort(paths.length_m):
            order = int(paths.order[k])
            vertices = paths.points[k, : order + 2]
            kinds = [
                f"{scene.facets[int(f)].kind}/{scene.labels[scene.facets[int(f)].label]}"
                for f in paths.sequence[k]
                if f >= 0
            ]
            rows.append(
                [
                    order,
                    round(float(paths.length_m[k] / sound_speed_m_s * 1000.0), 3),
                    kinds,
                    [round(float(v), 3) for v in vertices.ravel()],
                ]
            )
        out.append(rows)
    target.write_text(json.dumps({"points": out, "sound_speed_m_s": sound_speed_m_s}))
    return target
