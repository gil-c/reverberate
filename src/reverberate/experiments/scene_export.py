"""Export one apartment storey, and truncated copies of it, for the solver.

Promoted from ``data/runs/b0_truncation/b0_export.py``. The question it was
written for is B0's -- does deleting every piece of geometry whose shortest
source to surface to receiver path exceeds ``c x T`` leave the first ``T`` of
the response unchanged? -- but the function it performs is general enough to be
named for itself: it writes a PFFDTD model JSON for a scene, and for that scene
cut back to a path length budget.

The truncation criterion is **total path length, not room membership**. For a
triangle, the quantity that matters is

    min over p in the triangle of   |p - S| + |p - R|

which is the shortest single bounce path that can touch it. Anything above the
cut cannot influence the receiver before ``cut / c``. That minimum is estimated
from seven samples per triangle (three vertices, three edge midpoints, the
centroid) and then relaxed by the triangle's longest edge, so the estimate errs
towards keeping a triangle rather than deleting one. Erring the other way would
manufacture exactly the failure the experiment is looking for.

A truncated scene is open, which PFFDTD supports: the voxeliser has no interior
flood fill, ``sim_setup`` takes explicit ``bmin`` and ``bmax`` "for open
scenes", and the engine carries absorbing boundary nodes on the exterior of the
grid. The wave therefore leaves through the cut instead of reflecting off it,
which is what the domain of dependence argument assumes.

**This exporter does not invent sidedness.** It used to write ``2`` for every
triangle believing that meant "two sided". Read from PFFDTD's own source, ``2``
means *front side only*: ``vox_scene.py`` marks every boundary node on the
normal's negative side as rigid. Every surface whose HSSD normal happened to
point away from the air therefore contributed no absorption while keeping its
area in every report. The sidedness comes from
:func:`reverberate.geometry.sim_geometry.simulation_geometry`, which derives it
once from the geometry, and ``3`` (both sides) is written only where the mesh
genuinely cannot answer the question.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from reverberate.experiments.small_objects import isolated_storey
from reverberate.geometry.apartment import build_apartment, instances_on_storey
from reverberate.geometry.hssd_room import load_object_instances
from reverberate.geometry.orientation import BOTH
from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.geometry.sim_geometry import instances_in_room, simulation_geometry

__all__ = [
    "C_AIR",
    "Scene",
    "export",
    "main",
    "material_table",
    "path_length_bound",
    "refine",
    "source_receiver",
    "to_model",
    "truncate",
    "write",
]

#: Speed of sound at 20 C, matching PFFDTD's default ``Tc=20``.
C_AIR = 343.0

#: PFFDTD's eleven octave bands, 16 Hz to 16 kHz.
BANDS = np.array([16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])


@dataclass(frozen=True)
class Scene:
    """One exported model, with the numbers the report has to quote."""

    name: str
    path: Path
    cut_m: float | None
    labels: int
    triangles: int
    bbox_lo: list[float]
    bbox_hi: list[float]
    exact_until_ms: float | None

    def describe(self) -> str:
        cut = "none" if self.cut_m is None else f"{self.cut_m:.1f} m"
        until = "" if self.exact_until_ms is None else f", exact to {self.exact_until_ms:.1f} ms"
        return f"{self.name}: cut {cut}, {self.labels} labels, {self.triangles} triangles{until}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.path.name,
            "cut_m": self.cut_m,
            "labels": self.labels,
            "triangles": self.triangles,
            "bbox_lo": self.bbox_lo,
            "bbox_hi": self.bbox_hi,
            "exact_until_ms": self.exact_until_ms,
        }


def _label_of(assignment: MeshMaterialAssignment) -> str:
    """The material label a mesh contributes to, sanitised for PFFDTD."""
    label = assignment.name.rsplit("_", 1)[0] or "Unlabelled"
    return "".join(c for c in label if c.isalnum() or c in "_-") or "Unlabelled"


def path_length_bound(mesh: trimesh.Trimesh, src: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Per triangle lower bound on ``|p - S| + |p - R|``, one value per face.

    Seven samples per triangle, then the longest edge subtracted. The result is
    a genuine lower bound up to the sampling, and biased towards keeping.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    corners = verts[faces]  # (F, 3, 3)
    midpoints = (corners + corners[:, [1, 2, 0], :]) / 2.0
    centroid = corners.mean(axis=1, keepdims=True)
    samples = np.concatenate([corners, midpoints, centroid], axis=1)  # (F, 7, 3)

    to_src = np.linalg.norm(samples - src, axis=2)
    to_rec = np.linalg.norm(samples - rec, axis=2)
    shortest = (to_src + to_rec).min(axis=1)

    edges = np.linalg.norm(corners - corners[:, [1, 2, 0], :], axis=2)
    bound: np.ndarray = shortest - edges.max(axis=1)
    return bound


def refine(
    assignments: list[MeshMaterialAssignment], max_edge_m: float
) -> list[MeshMaterialAssignment]:
    """Split every triangle until no edge exceeds ``max_edge_m``.

    The apartment shell arrives as a handful of very large triangles, one of
    which can span the whole storey. Culling whole triangles would then keep a
    floor that reaches the far wall merely because one of its corners is near
    the receiver, and the truncated domain would be no smaller than the full
    one. Subdivision does not move the surface, so the reference scene is
    refined too and both scenes share one triangulation.
    """
    refined: list[MeshMaterialAssignment] = []
    for assignment in assignments:
        mesh = cast(trimesh.Trimesh, assignment.mesh.subdivide_to_size(max_edge=max_edge_m))
        # Subdivision splits triangles without turning any of them over, so a
        # face inherits its parent's side. There is no face map to slice, but
        # sidedness is uniform per assignment by construction, so the whole
        # refined mesh takes the value the original carried.
        refined.append(
            MeshMaterialAssignment(
                name=assignment.name,
                mesh=mesh,
                material=assignment.material,
                sides=_uniform_sides(assignment, len(mesh.faces)),
            )
        )
    return refined


def _uniform_sides(assignment: MeshMaterialAssignment, count: int) -> np.ndarray:
    """The assignment's single sidedness, restated over ``count`` faces."""
    if assignment.sides is None or len(assignment.sides) == 0:
        return np.full(count, BOTH, dtype=int)
    values = np.unique(np.asarray(assignment.sides, dtype=int))
    side = int(values[0]) if len(values) == 1 else BOTH
    return np.full(count, side, dtype=int)


def truncate(
    assignments: list[MeshMaterialAssignment],
    src: np.ndarray,
    rec: np.ndarray,
    cut_m: float,
) -> list[MeshMaterialAssignment]:
    """Drop every triangle no single bounce path shorter than ``cut_m`` reaches."""
    kept: list[MeshMaterialAssignment] = []
    for assignment in assignments:
        mesh = assignment.mesh
        keep = path_length_bound(mesh, src, rec) <= cut_m
        if not keep.any():
            continue
        faces = np.flatnonzero(keep)
        trimmed = cast(trimesh.Trimesh, mesh.submesh([faces], append=True, repair=False))
        # Truncation opens the mesh, so the sidedness cannot be re-derived from
        # the result. It is sliced from the intact scene instead, which is the
        # point of deriving it once: the truncated scene must differ from the
        # reference only in which triangles are present.
        kept.append(
            MeshMaterialAssignment(
                name=assignment.name,
                mesh=trimmed,
                material=assignment.material,
                sides=(
                    np.asarray(assignment.sides, dtype=int)[faces]
                    if assignment.sides is not None
                    else None
                ),
            )
        )
    return kept


def to_model(
    assignments: list[MeshMaterialAssignment],
    src: np.ndarray,
    rec: np.ndarray,
) -> dict[str, Any]:
    """PFFDTD's model JSON: triangles grouped by material label."""
    mats_hash: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        entry = mats_hash.setdefault(
            _label_of(assignment), {"tris": [], "pts": [], "color": [128, 128, 128], "sides": []}
        )
        offset = len(entry["pts"])
        entry["pts"].extend(np.asarray(assignment.mesh.vertices, dtype=float).tolist())
        faces = np.asarray(assignment.mesh.faces, dtype=int)
        # Derived in the scene description, never here. A mesh whose orientation
        # could not be established carries BOTH, which is wasteful but can never
        # be silently rigid; anything else would be this exporter guessing.
        sides = (
            np.asarray(assignment.sides, dtype=int)
            if assignment.sides is not None
            else np.full(len(faces), BOTH, dtype=int)
        )
        if len(sides) != len(faces):
            raise ValueError(f"{assignment.name}: {len(sides)} sides for {len(faces)} faces")
        for face, side in zip(faces, sides, strict=True):
            entry["tris"].append([int(v) + offset for v in face])
            entry["sides"].append(int(side))
    return {
        "mats_hash": mats_hash,
        "sources": [{"xyz": [float(v) for v in src], "name": "S1"}],
        "receivers": [{"xyz": [float(v) for v in rec], "name": "R1"}],
        "export_datetime": "reverberate experiments.scene_export",
    }


def material_table(assignments: list[MeshMaterialAssignment]) -> dict[str, list[float]]:
    """Per label Sabine absorption on PFFDTD's 11 octave bands, 16 Hz to 16 kHz.

    The project's materials carry pyroomacoustics' bands, which start at 125 Hz
    and stop at 4 or 8 kHz. The bands below the first and above the last are
    filled by holding the end value, which is what the absorption tables
    themselves do. Labels shared by several meshes are averaged by area, so a
    label's coefficient is the one its surface actually presents.
    """
    weighted: dict[str, list[tuple[float, np.ndarray]]] = {}
    for assignment in assignments:
        coeffs = np.asarray(assignment.material.energy_absorption["coeffs"], dtype=float)
        centres = np.asarray(assignment.material.energy_absorption["center_freqs"], dtype=float)
        resampled = np.interp(BANDS, centres, coeffs)
        weighted.setdefault(_label_of(assignment), []).append(
            (float(assignment.mesh.area), resampled)
        )

    table: dict[str, list[float]] = {}
    for label, parts in weighted.items():
        areas = np.array([a for a, _ in parts])
        stack = np.vstack([c for _, c in parts])
        total = areas.sum()
        mean = stack.mean(axis=0) if total <= 0 else (areas[:, None] * stack).sum(axis=0) / total
        table[label] = [round(float(v), 5) for v in np.clip(mean, 0.001, 0.999)]
    return table


def write(
    name: str,
    assignments: list[MeshMaterialAssignment],
    src: np.ndarray,
    rec: np.ndarray,
    out_dir: Path,
    cut_m: float | None,
) -> Scene:
    """Write one model JSON and return what the manifest has to say about it."""
    model = to_model(assignments, src, rec)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(model))
    verts = np.vstack([np.asarray(a.mesh.vertices) for a in assignments])
    return Scene(
        name=name,
        path=path,
        cut_m=cut_m,
        labels=len(model["mats_hash"]),
        triangles=sum(len(v["tris"]) for v in model["mats_hash"].values()),
        bbox_lo=[round(float(v), 3) for v in verts.min(axis=0)],
        bbox_hi=[round(float(v), 3) for v in verts.max(axis=0)],
        exact_until_ms=None if cut_m is None else 1000.0 * cut_m / C_AIR,
    )


def source_receiver(
    room_assignments: list[MeshMaterialAssignment],
) -> tuple[np.ndarray, np.ndarray]:
    """One source and one receiver inside the listener's room, as B1 placed them."""
    verts = np.vstack([np.asarray(a.mesh.vertices) for a in room_assignments])
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mid = (lo + hi) / 2.0
    src = np.array([mid[0] - (hi[0] - lo[0]) * 0.2, lo[1] + 1.7, mid[2]])
    rec = np.array([mid[0] + (hi[0] - lo[0]) * 0.2, lo[1] + 1.2, mid[2]])
    return src, rec


def export(
    hssd_root: Path,
    scene_id: str,
    room_name: str,
    out_dir: Path,
    cuts_m: tuple[float, ...] = (10.0, 5.0),
    max_edge_m: float = 0.25,
    seed: int = 0,
) -> list[Scene]:
    """Write the reference scene, the listener's room, and one cut per entry.

    The manifest beside them is what :mod:`reverberate.experiments.run` reads:
    it carries the absorption table, the source and receiver, and each scene's
    cut, which is the only place the cut is recorded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    storeys = build_apartment(hssd_root, scene_id)
    storey = storeys[0]
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    instances = instances_on_storey(instances, storey, storeys)

    # The listener's room alone, which fixes the source and receiver and is also
    # the domain of the 16 kHz cost probe.
    #
    # Scoped by footprint against the same polygon the shell is extruded from,
    # not by the instance's origin point: see instances_in_room. The origin test
    # this replaces dropped every picture on the bedroom's own walls, because an
    # asset's origin sits in the wall band where no room polygon contains it.
    room_storey = isolated_storey(storey, room_name)
    in_room = instances_in_room(hssd_root, room_storey, instances)
    room_assignments, room_summary = simulation_geometry(hssd_root, room_storey, in_room, seed=seed)
    src, rec = source_receiver(room_assignments)
    print(f"{scene_id}/{room_name} (room only): {room_summary.summary()}")

    # The whole storey, which is the reference the truncations are judged against.
    full_assignments, full_summary = simulation_geometry(hssd_root, storey, instances, seed=seed)
    print(f"{scene_id} (whole storey): {full_summary.summary()}")
    full_assignments = refine(full_assignments, max_edge_m)
    print(
        f"refined to {max_edge_m} m edges: "
        f"{sum(len(a.mesh.faces) for a in full_assignments)} triangles"
    )

    direct = float(np.linalg.norm(src - rec))
    print(f"source={src.round(3).tolist()} receiver={rec.round(3).tolist()}")
    print(f"direct path {direct:.3f} m, {1000.0 * direct / C_AIR:.2f} ms")

    scenes = [
        write("apartment_full", full_assignments, src, rec, out_dir, None),
        write("bedroom_only", room_assignments, src, rec, out_dir, None),
    ]
    for cut in cuts_m:
        trimmed = truncate(full_assignments, src, rec, cut)
        scenes.append(write(f"apartment_cut{cut:g}m", trimmed, src, rec, out_dir, cut))

    manifest = {
        "scene_id": scene_id,
        "room": room_name,
        "seed": seed,
        "c_air": C_AIR,
        "max_edge_m": max_edge_m,
        "materials": material_table([*full_assignments, *room_assignments]),
        "source": [float(v) for v in src],
        "receiver": [float(v) for v in rec],
        "direct_path_m": direct,
        # What the solver will seal, decided from the meshes rather than
        # recovered from a voxel grid, so the viewer can draw it and a grid
        # census can be checked against it. See reverberate.geometry.sealed.
        "sealed": room_summary.sealed.record(),
        "sealed_full": full_summary.sealed.record(),
        # What the carve did to HSSD's collision proxies. Carried here for the
        # same reason as the sealing census: the difference between the mesh a
        # viewer sees and the solid the solver received is 2.02x in volume on
        # this room, and a number in the manifest is the only place that can be
        # checked. See reverberate.geometry.carve.
        "carve": {
            "carved": room_summary.carve.carved,
            "skipped": room_summary.carve.skipped,
        },
        "carve_full": {
            "carved": full_summary.carve.carved,
            "skipped": full_summary.carve.skipped,
        },
        "scenes": [s.as_dict() for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for scene in scenes:
        print(scene.describe())
    return scenes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--scene-id", default="102344022")
    parser.add_argument("--room", default="bedroom.001")
    parser.add_argument("--cuts", type=float, nargs="*", default=[10.0, 5.0])
    parser.add_argument("--max-edge", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    export(
        args.hssd_root,
        args.scene_id,
        args.room,
        args.out_dir,
        tuple(args.cuts),
        args.max_edge,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
