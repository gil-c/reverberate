"""The geometry the mirror is handed, derived from the solver's own model.

The wave solver reads a model of 1.6 million triangles for one storey and is
indifferent to their number. A geometric engine is not: its images are made
on planes and its rays are tested against every triangle, so the model is
**derived** here into three things, each by a rule tied to the wavelength and
each recorded in the census so nothing is dropped in silence:

1. **Facets, the reflectors.** Coplanar adjacent triangles of one label are
   merged into one planar facet (an exact operation, nothing moves) and a
   facet is a reflector when its area reaches ``reflector_area_m2``. A wall,
   a floor, a table top, a wardrobe front. The facet keeps its triangles, so
   "the reflection point lies on the facet" is a ray-triangle test, the one
   primitive the card runs.
2. **Occluders and scatterers.** The outer surface of every label, decimated
   by quadric edge collapse to a triangle budget set by its area and
   ``edge_m``, and measured afterwards: the largest distance from the
   original vertices to the decimated surface is reported per label, and a
   label whose distance exceeds ``distance_m`` is decimated less. These
   triangles block a path and scatter a ray; they make no image.
3. **Materials.** Per label, the solver's own absorption on the octave
   bands and the class's scattering coefficient. The shell's three surfaces
   are told apart by normal so a later fix can give them their own
   materials; by default the mirror reproduces what the solver received.

The model is the solver's, loaded by :func:`reverberate.accel.scene.load_scene`,
so the mirror and the solver start from the same triangles, which is what
roadmap constraint 9 asks. The derived scene is serialised as one ``.npz`` and
one ``.json`` beside it, keyed by a digest of the arrays and the thresholds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from reverberate.accel.scene import Scene, load_scene
from reverberate.acoustics import OCTAVE_BANDS
from reverberate.materials import class_for_category

__all__ = [
    "DerivedScene",
    "Facet",
    "GeometryRules",
    "MaterialTable",
    "derive",
    "load_derived",
    "shell_surface_of",
    "write_derived",
]

#: The label the export gives the extruded storey.
SHELL_LABEL = "shell"

#: A shell face whose normal is this close to vertical is floor or ceiling.
VERTICAL = 0.9


@dataclass(frozen=True)
class GeometryRules:
    """The three thresholds, each in the provenance."""

    #: A planar facet reflects specularly when its area reaches this. 0.4 m2
    #: is the side of a 63 cm square, about two wavelengths at 1 kHz.
    reflector_area_m2: float = 0.4
    #: Target edge of a decimated triangle: the triangle budget of a label is
    #: its area over the area of such a triangle. 10 cm is a wavelength at
    #: 3.4 kHz.
    edge_m: float = 0.10
    #: The decimated surface may not stray farther than this from the
    #: original vertices; 2 cm is an eighth of a wavelength at 2 kHz.
    distance_m: float = 0.02
    #: Fewest and most triangles a label keeps.
    min_triangles: int = 12
    max_triangles: int = 20000
    #: trimesh's flatness ratio for the facet merge: two adjacent faces are
    #: one facet when the radius of the cylinder through them, over the span
    #: of their shared edge, exceeds this. trimesh's own default of five merges
    #: a curved wall into one "plane" whose normal matches none of its
    #: triangles; a thousand keeps the merge to faces coplanar to rounding.
    facet_threshold: float = 1000.0
    #: A merged group whose vertices stray farther than this from its mean
    #: plane is not a facet and is split back into single triangles.
    planar_m: float = 0.002

    def record(self) -> dict[str, Any]:
        return {
            "reflector_area_m2": self.reflector_area_m2,
            "edge_m": self.edge_m,
            "distance_m": self.distance_m,
            "min_triangles": self.min_triangles,
            "max_triangles": self.max_triangles,
            "facet_threshold": self.facet_threshold,
            "planar_m": self.planar_m,
        }


@dataclass(frozen=True)
class MaterialTable:
    """Absorption per octave band and scattering, per label, and where they came from."""

    labels: tuple[str, ...]
    #: ``[label, band]`` on :data:`OCTAVE_BANDS`.
    absorption: np.ndarray
    #: ``[label]``.
    scattering: np.ndarray
    bands_hz: tuple[int, ...] = OCTAVE_BANDS
    #: Per label, the class the numbers came from, or the note that they are
    #: the solver's own averaged figure.
    source: tuple[str, ...] = ()

    def index(self, label: str) -> int:
        return self.labels.index(label)

    def record(self) -> dict[str, Any]:
        return {
            "bands_hz": list(self.bands_hz),
            "labels": {
                label: {
                    "absorption": [round(float(v), 5) for v in self.absorption[i]],
                    "scattering": round(float(self.scattering[i]), 4),
                    "source": self.source[i] if self.source else "",
                }
                for i, label in enumerate(self.labels)
            },
        }


@dataclass(frozen=True)
class Facet:
    """One planar reflector: its plane, its label, its triangles."""

    label: int
    normal: np.ndarray
    #: ``normal . x = offset`` on the plane.
    offset: float
    area: float
    #: Indices into the derived scene's reflector triangle arrays.
    triangles: np.ndarray
    #: ``shell_floor``, ``shell_wall``, ``shell_ceiling`` or ``furniture``.
    kind: str
    #: PFFDTD's sidedness of the facet's triangles: 2 reflects on the
    #: normal's side only, 3 on both (an unoriented mesh).
    sides: int = 2


@dataclass(frozen=True)
class DerivedScene:
    """What the mirror computes on: reflectors, occluders, materials, census."""

    labels: tuple[str, ...]
    materials: MaterialTable
    facets: tuple[Facet, ...]
    #: Reflector triangles, ``[triangle, 3, 3]``, and their facet index.
    reflector_vertices: np.ndarray
    reflector_facet: np.ndarray
    #: Occluder and scatterer triangles, ``[triangle, 3, 3]``, their label and
    #: their sidedness (2 front only, 3 both) carried from the model.
    occluder_vertices: np.ndarray
    occluder_label: np.ndarray
    occluder_sides: np.ndarray
    rules: GeometryRules
    census: dict[str, Any] = field(default_factory=dict)
    bmin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bmax: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def key(self) -> str:
        """A digest of the arrays and the rules: the derived scene's identity."""
        digest = hashlib.sha256()
        for array in (
            self.reflector_vertices,
            self.reflector_facet,
            self.occluder_vertices,
            self.occluder_label,
            self.occluder_sides,
            self.materials.absorption,
            self.materials.scattering,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(json.dumps(self.rules.record(), sort_keys=True).encode())
        digest.update("|".join(self.labels).encode())
        return digest.hexdigest()[:32]

    def summary(self) -> str:
        return (
            f"{len(self.facets)} facets over {self.reflector_vertices.shape[0]} triangles, "
            f"{self.occluder_vertices.shape[0]} occluder triangles, {len(self.labels)} labels"
        )


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------


def _solver_absorption(manifest: dict[str, Any] | None, label: str) -> np.ndarray | None:
    """The solver's own per label absorption, on the octave bands, from the export manifest."""
    if not manifest:
        return None
    table = manifest.get("materials") or {}
    row = table.get(label)
    if row is None:
        return None
    # The manifest holds PFFDTD's eleven bands, 15.6 Hz to 16 kHz; the octave
    # bands of the project are its seventh to thirteenth... its bands 3 to 9.
    solver_bands = 1000.0 * 2.0 ** np.arange(-6, 5)
    return np.asarray(np.interp(OCTAVE_BANDS, solver_bands, np.asarray(row, dtype=float)))


def material_table(
    labels: tuple[str, ...], manifest: dict[str, Any] | None = None
) -> MaterialTable:
    """Absorption and scattering per label.

    The absorption is the solver's own figure for the label when the export's
    manifest is given, which is what the reference field heard, including the
    shell averaged over its three surfaces (see the plan, finding 7.2); the
    class's measured absorption otherwise. The scattering comes from the
    class either way, since the solver has no such number.
    """
    absorption = np.zeros((len(labels), len(OCTAVE_BANDS)))
    scattering = np.zeros(len(labels))
    sources = []
    for i, label in enumerate(labels):
        category = "wall" if label == SHELL_LABEL else label
        try:
            cls = class_for_category(category)
            scattering[i] = cls.scattering
            class_absorption: np.ndarray | None = np.asarray(cls.absorption, dtype=float)
            source = f"class {cls.name}"
        except KeyError:
            class_absorption = None
            scattering[i] = 0.1
            source = "unknown category, scattering assumed 0.1"
        own = _solver_absorption(manifest, label)
        if own is not None:
            absorption[i] = own
            sources.append(f"solver manifest ({source})")
        elif class_absorption is not None:
            absorption[i] = class_absorption
            sources.append(source)
        else:
            absorption[i] = 0.1
            sources.append(source + ", absorption assumed 0.1")
    return MaterialTable(labels, absorption, scattering, OCTAVE_BANDS, tuple(sources))


# --------------------------------------------------------------------------
# facets and occluders
# --------------------------------------------------------------------------


def shell_surface_of(normals: np.ndarray) -> np.ndarray:
    """``floor``, ``ceiling`` or ``wall`` per face of the shell, from its air facing normal.

    The shell's normals point into the air, so the floor's point up.
    """
    kinds = np.full(normals.shape[0], "wall", dtype=object)
    kinds[normals[:, 1] > VERTICAL] = "floor"
    kinds[normals[:, 1] < -VERTICAL] = "ceiling"
    return kinds


def _mesh_of(scene: Scene, label_index: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    """The triangles of one label as a mesh, and their indices in the scene."""
    indices = np.flatnonzero(scene.mat_ind == label_index)
    tris = scene.tris[indices]
    used, inverse = np.unique(tris.ravel(), return_inverse=True)
    mesh = trimesh.Trimesh(vertices=scene.pts[used], faces=inverse.reshape(-1, 3), process=False)
    return mesh, indices


def _facets_of(
    mesh: trimesh.Trimesh, threshold: float, planar_m: float
) -> tuple[list[np.ndarray], int]:
    """Groups of coplanar adjacent faces, plus every face left alone as its own group.

    A group whose vertices stray from its area weighted plane by more than
    ``planar_m`` is not planar whatever the grouping said, and is split back
    into single faces; the count of such groups is returned beside the list.
    """
    grouped_faces = trimesh.graph.facets(mesh, facet_threshold=threshold)
    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    groups: list[np.ndarray] = []
    split = 0
    for raw in grouped_faces:
        group = np.asarray(raw, dtype=int)
        normal = (normals[group] * areas[group][:, None]).sum(axis=0)
        norm = float(np.linalg.norm(normal))
        vertices = mesh.vertices[mesh.faces[group]].reshape(-1, 3)
        if norm > 0.0:
            normal = normal / norm
            heights = vertices @ normal
            if float(heights.max() - heights.min()) <= planar_m:
                groups.append(group)
                continue
        split += 1
        groups.extend(np.array([i]) for i in group)
    grouped = np.zeros(len(mesh.faces), dtype=bool)
    for g in groups:
        grouped[g] = True
    groups.extend(np.array([i]) for i in np.flatnonzero(~grouped))
    return groups, split


def _decimate(
    mesh: trimesh.Trimesh, rules: GeometryRules, rng: np.random.Generator
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """The label's outer surface under a triangle budget, and the distance it strayed.

    Only a closed mesh is decimated. Edge collapse on a surface with open
    edges throws triangles across the openings, such as a doorway, and every
    reflection behind them dies. An open mesh keeps its triangles and says so.
    """
    import fast_simplification

    from reverberate.geometry.orientation import is_closed

    area = float(mesh.area)
    before = len(mesh.faces)
    if not is_closed(mesh):
        return mesh, {
            "triangles_before": before,
            "triangles_after": before,
            "distance_m": 0.0,
            "max_distance_m": 0.0,
            "kept_open": True,
        }
    budget = int(np.clip(area / (0.5 * rules.edge_m**2), rules.min_triangles, rules.max_triangles))
    if before <= budget:
        return mesh, {
            "triangles_before": before,
            "triangles_after": before,
            "distance_m": 0.0,
            "max_distance_m": 0.0,
        }
    sample = mesh.vertices[rng.choice(len(mesh.vertices), min(2000, len(mesh.vertices)), False)]
    target = budget
    cap = min(rules.max_triangles, before - 1)
    for _ in range(16):
        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            target_count=min(target, cap),
        )
        decimated = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(decimated.faces) == 0:
            break
        _, distances, _ = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
            decimated, sample
        )
        # The bound is on the surface as a whole, the 95th percentile of the
        # sampled distances: a knob that vanishes is one vertex far away and
        # must not keep a wardrobe at its full count. The largest distance is
        # reported beside it.
        strayed = float(np.percentile(distances, 95))
        if strayed <= rules.distance_m or target >= before:
            return decimated, {
                "triangles_before": before,
                "triangles_after": len(decimated.faces),
                "distance_m": round(strayed, 5),
                "max_distance_m": round(float(np.max(distances)), 5),
            }
        if target >= cap:
            # The cap is the answer: an object of filaments never meets the
            # bound and must not reach the card at its full count. Reported
            # as over the bound rather than quietly kept.
            return decimated, {
                "triangles_before": before,
                "triangles_after": len(decimated.faces),
                "distance_m": round(strayed, 5),
                "max_distance_m": round(float(np.max(distances)), 5),
                "over_bound": True,
            }
        target = min(target * 2, cap)
    return mesh, {
        "triangles_before": before,
        "triangles_after": before,
        "distance_m": 0.0,
        "max_distance_m": 0.0,
    }


def derive(
    model_json: Path | str,
    *,
    rules: GeometryRules | None = None,
    manifest: dict[str, Any] | None = None,
    seed: int = 0,
    say: Any = None,
) -> DerivedScene:
    """The derived scene of one solver model.

    ``manifest`` is the export's ``manifest.json`` when the materials must
    be the solver's own; ``seed`` fixes the vertex sample the decimation
    distance is measured on, so the derivation is deterministic.
    """
    rules = rules or GeometryRules()
    scene = load_scene(model_json)
    rng = np.random.default_rng(seed)
    labels = tuple(scene.mat_str[: scene.nmat])
    materials = material_table(labels, manifest)
    facets: list[Facet] = []
    reflector_vertices: list[np.ndarray] = []
    reflector_facet: list[np.ndarray] = []
    occluder_vertices: list[np.ndarray] = []
    occluder_label: list[np.ndarray] = []
    occluder_sides: list[np.ndarray] = []
    census: dict[str, Any] = {"labels": {}, "rules": rules.record()}

    for index, label in enumerate(labels):
        mesh, scene_indices = _mesh_of(scene, index)
        area = float(mesh.area)
        row: dict[str, Any] = {"area_m2": round(area, 4), "facets": 0, "reflector_area_m2": 0.0}
        # Reflectors: facets of enough area. A label whose whole area is under
        # the threshold has none and is not even grouped.
        if area >= rules.reflector_area_m2:
            normals = np.asarray(mesh.face_normals)
            areas = np.asarray(mesh.area_faces)
            groups, split = _facets_of(mesh, rules.facet_threshold, rules.planar_m)
            row["groups_split_as_not_planar"] = split
            for group in groups:
                facet_area = float(areas[group].sum())
                if facet_area < rules.reflector_area_m2:
                    continue
                normal = (normals[group] * areas[group][:, None]).sum(axis=0)
                norm = float(np.linalg.norm(normal))
                if norm == 0.0:
                    continue
                normal = normal / norm
                origin = mesh.vertices[mesh.faces[group[0]]].mean(axis=0)
                kind = "furniture"
                if label == SHELL_LABEL:
                    kind = f"shell_{shell_surface_of(normal[None, :])[0]}"
                start = sum(len(v) for v in reflector_vertices)
                triangles = mesh.vertices[mesh.faces[group]]
                group_sides = scene.mat_side[scene_indices[group]].astype(int)
                sides = int(np.bincount(group_sides).argmax()) if group_sides.size else 2
                reflector_vertices.append(np.asarray(triangles, dtype=np.float64))
                reflector_facet.append(np.full(len(group), len(facets), dtype=np.int32))
                facets.append(
                    Facet(
                        label=index,
                        normal=normal,
                        offset=float(normal @ origin),
                        area=facet_area,
                        triangles=np.arange(start, start + len(group), dtype=np.int32),
                        kind=kind,
                        sides=sides,
                    )
                )
                row["facets"] += 1
                row["reflector_area_m2"] = round(row["reflector_area_m2"] + facet_area, 4)
        # Occluders and scatterers: the whole label, decimated.
        decimated, report = _decimate(mesh, rules, rng)
        row.update(report)
        row["diffuse_area_m2"] = round(area - row["reflector_area_m2"], 4)
        sides = scene.mat_side[scene_indices]
        side = int(np.bincount(sides.astype(int)).argmax()) if sides.size else 2
        occluder_vertices.append(np.asarray(decimated.vertices[decimated.faces], dtype=np.float64))
        occluder_label.append(np.full(len(decimated.faces), index, dtype=np.int16))
        occluder_sides.append(np.full(len(decimated.faces), side, dtype=np.int8))
        census["labels"][label] = row
        if say is not None:
            say(
                f"{label}: {row['facets']} facets, {row['reflector_area_m2']} of"
                f" {row['area_m2']} m2 reflecting, {report['triangles_before']} ->"
                f" {report['triangles_after']} triangles, strayed {report['distance_m']} m"
            )

    reflector_v = np.concatenate(reflector_vertices) if reflector_vertices else np.zeros((0, 3, 3))
    reflector_f = (
        np.concatenate(reflector_facet) if reflector_facet else np.zeros(0, dtype=np.int32)
    )
    occluder_v = np.concatenate(occluder_vertices) if occluder_vertices else np.zeros((0, 3, 3))
    census["totals"] = {
        "labels": len(labels),
        "facets": len(facets),
        "reflector_triangles": int(reflector_v.shape[0]),
        "occluder_triangles": int(occluder_v.shape[0]),
        "model_triangles": int(scene.triangles),
        "model_area_m2": round(float(sum(r["area_m2"] for r in census["labels"].values())), 3),
        "reflector_area_m2": round(
            float(sum(r["reflector_area_m2"] for r in census["labels"].values())), 3
        ),
        "diffuse_area_m2": round(
            float(sum(r["diffuse_area_m2"] for r in census["labels"].values())), 3
        ),
        "max_distance_m": max(
            (float(r["distance_m"]) for r in census["labels"].values()), default=0.0
        ),
        "labels_over_bound": sorted(
            label for label, r in census["labels"].items() if r.get("over_bound")
        ),
        "labels_kept_open": sorted(
            label for label, r in census["labels"].items() if r.get("kept_open")
        ),
        "groups_split_as_not_planar": int(
            sum(int(r.get("groups_split_as_not_planar", 0)) for r in census["labels"].values())
        ),
        "facets_by_kind": {
            kind: sum(1 for f in facets if f.kind == kind)
            for kind in ("shell_floor", "shell_wall", "shell_ceiling", "furniture")
        },
    }
    return DerivedScene(
        labels=labels,
        materials=materials,
        facets=tuple(facets),
        reflector_vertices=reflector_v,
        reflector_facet=reflector_f,
        occluder_vertices=occluder_v,
        occluder_label=np.concatenate(occluder_label)
        if occluder_label
        else np.zeros(0, dtype=np.int16),
        occluder_sides=np.concatenate(occluder_sides)
        if occluder_sides
        else np.zeros(0, dtype=np.int8),
        rules=rules,
        census=census,
        bmin=np.asarray(scene.bmin, dtype=float),
        bmax=np.asarray(scene.bmax, dtype=float),
    )


# --------------------------------------------------------------------------
# on disk
# --------------------------------------------------------------------------


def write_derived(derived: DerivedScene, target: Path) -> Path:
    """``<target>.npz`` and ``<target>.json``; returns the JSON path."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target.with_suffix(".npz"),
        reflector_vertices=derived.reflector_vertices,
        reflector_facet=derived.reflector_facet,
        occluder_vertices=derived.occluder_vertices,
        occluder_label=derived.occluder_label,
        occluder_sides=derived.occluder_sides,
        facet_label=np.asarray([f.label for f in derived.facets], dtype=np.int32),
        facet_normal=np.asarray([f.normal for f in derived.facets], dtype=float).reshape(-1, 3),
        facet_offset=np.asarray([f.offset for f in derived.facets], dtype=float),
        facet_area=np.asarray([f.area for f in derived.facets], dtype=float),
        facet_start=np.asarray(
            [int(f.triangles[0]) if f.triangles.size else 0 for f in derived.facets], dtype=np.int32
        ),
        facet_count=np.asarray([f.triangles.size for f in derived.facets], dtype=np.int32),
        facet_sides=np.asarray([f.sides for f in derived.facets], dtype=np.int8),
        absorption=derived.materials.absorption,
        scattering=derived.materials.scattering,
        bmin=derived.bmin,
        bmax=derived.bmax,
    )
    record = {
        "key": derived.key,
        "labels": list(derived.labels),
        "facet_kinds": [f.kind for f in derived.facets],
        "materials": derived.materials.record(),
        "rules": derived.rules.record(),
        "census": derived.census,
        "summary": derived.summary(),
    }
    path = target.with_suffix(".json")
    path.write_text(json.dumps(record, indent=1))
    return path


def load_derived(target: Path) -> DerivedScene:
    """The inverse of :func:`write_derived`."""
    target = Path(target)
    record = json.loads(target.with_suffix(".json").read_text())
    with np.load(target.with_suffix(".npz")) as arrays:
        data = {name: np.asarray(arrays[name]) for name in arrays.files}
    labels = tuple(str(v) for v in record["labels"])
    materials = MaterialTable(
        labels,
        data["absorption"],
        data["scattering"],
        tuple(int(v) for v in record["materials"]["bands_hz"]),
        tuple(str(record["materials"]["labels"][label]["source"]) for label in labels),
    )
    facets = tuple(
        Facet(
            label=int(data["facet_label"][i]),
            normal=data["facet_normal"][i],
            offset=float(data["facet_offset"][i]),
            area=float(data["facet_area"][i]),
            triangles=np.arange(
                int(data["facet_start"][i]),
                int(data["facet_start"][i]) + int(data["facet_count"][i]),
                dtype=np.int32,
            ),
            kind=str(record["facet_kinds"][i]),
            sides=int(data["facet_sides"][i]),
        )
        for i in range(data["facet_label"].shape[0])
    )
    rules = GeometryRules(**record["rules"])
    derived = DerivedScene(
        labels=labels,
        materials=materials,
        facets=facets,
        reflector_vertices=data["reflector_vertices"],
        reflector_facet=data["reflector_facet"],
        occluder_vertices=data["occluder_vertices"],
        occluder_label=data["occluder_label"],
        occluder_sides=data["occluder_sides"],
        rules=rules,
        census=dict(record["census"]),
        bmin=data["bmin"],
        bmax=data["bmax"],
    )
    if derived.key != record["key"]:
        raise ValueError(f"{target}: the arrays do not match the key {record['key']}")
    return derived
