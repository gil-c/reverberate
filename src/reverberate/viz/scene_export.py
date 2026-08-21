"""Export a reconstructed HSSD room as a glTF scene, for inspection in a browser.

The point of this module is to show *our interpretation* of the dataset, not
the dataset as the authors' own tools would render it. Rendering the raw scene
with the official viewer would confirm nothing about the reconstruction: it
would bypass every hypothesis this project makes. So both modes below are
built from exactly the objects the rest of the pipeline uses.

Two modes:

``render``
    The textured render assets, placed by *our* instance matrices, inside
    *our* extruded room shell (drawn as a translucent box). If our transforms
    or our shell were wrong, furniture would visibly float, sink, interpenetrate
    or sit outside the walls.
``acoustic``
    The collider meshes actually handed to ``pyroomacoustics``, flat-coloured
    by the absorption coefficient assigned to them by
    ``reverberate.geometry.materials``. This is the simulator's view of the
    room: coarse shapes, no textures, and colour carrying the acoustic
    interpretation rather than appearance.

Both export a single self contained ``.glb``, which any standard glTF viewer
renders, so the visualisation is a plain web component with no simulator,
renderer or native dependency of its own.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import (
    FurnitureInstance,
    RoomRegion,
    load_object_instances,
    load_regions,
    match_instances_to_regions,
)
from reverberate.geometry.materials import material_for_label

#: Opacity of the room shell in render mode, so furniture stays visible while
#: the walls we reconstructed are still verifiable against it.
SHELL_ALPHA = 60

#: Face budget per furniture piece in render mode. HSSD render assets are
#: dense (over 14k faces for a small object is common), which makes a whole
#: room too heavy to open in a browser. Decimating for display only is safe:
#: nothing here feeds the simulator, which uses the collider meshes instead.
RENDER_FACE_BUDGET = 4000


def decimate_for_display(geometry: trimesh.Trimesh, face_budget: int) -> trimesh.Trimesh:
    """Reduce a display mesh's face count, keeping its appearance and materials."""
    if len(geometry.faces) <= face_budget:
        return geometry
    try:
        simplified = geometry.simplify_quadric_decimation(face_count=face_budget)
    except Exception:
        # Decimation is a display optimisation, never a correctness
        # requirement, so a mesh it cannot handle is passed through as is.
        return geometry
    if geometry.visual is not None:
        simplified.visual = geometry.visual.copy()
    return simplified


def load_display_geometry(path: Path, face_budget: int) -> trimesh.Scene:
    """Load a render asset and decimate each of its parts to the face budget."""
    loaded = trimesh.load(path)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    for name, geometry in list(scene.geometry.items()):
        if isinstance(geometry, trimesh.Trimesh):
            scene.geometry[name] = decimate_for_display(geometry, face_budget)
    return scene


@dataclass
class ExportReport:
    """What actually made it into the scene, so gaps are visible not silent."""

    scene_id: str
    region_name: str
    placed: int = 0
    unresolved: list[str] = field(default_factory=list)
    missing_collider: list[str] = field(default_factory=list)
    layouts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        layout_text = ", ".join(f"{name}: {count}" for name, count in sorted(self.layouts.items()))
        return (
            f"{self.scene_id}/{self.region_name}: {self.placed} pieces placed "
            f"({layout_text}); {len(self.unresolved)} unresolved, "
            f"{len(self.missing_collider)} without a collider"
        )


def absorption_colour(mean_absorption: float) -> np.ndarray:
    """Blue (reflective) to red (absorptive), as an RGBA byte colour.

    A deliberately simple two-colour ramp rather than a perceptual colormap:
    the question this view answers is "did this surface get a plausible
    absorption", which only needs an ordered scale.
    """
    fraction = float(np.clip(mean_absorption, 0.0, 1.0))
    return np.array(
        [int(255 * fraction), 40, int(255 * (1.0 - fraction)), 255],
        dtype=np.uint8,
    )


def shell_mesh(region: RoomRegion, translucent: bool) -> trimesh.Trimesh:
    shell = region.extrude()
    colour = np.array([200, 200, 200, SHELL_ALPHA if translucent else 255], dtype=np.uint8)
    shell.visual = trimesh.visual.ColorVisuals(shell, face_colors=colour)
    return shell


#: A face whose outward normal is this close to vertical is floor or ceiling
#: rather than wall. The shell is a prism, so its faces are either exactly
#: vertical or exactly horizontal and the threshold is not delicate.
VERTICAL_NORMAL_THRESHOLD = 0.9


def shell_surface_labels(shell: trimesh.Trimesh) -> np.ndarray:
    """Label each shell face floor, ceiling or wall from its outward normal.

    The room shell is one extruded prism, so without this split the whole
    enclosure would take a single material. That matters acoustically: carpet
    underfoot and plasterboard overhead are the two ends of the absorption
    range, and averaging them away would flatten the very signal the model is
    meant to learn.
    """
    vertical = shell.face_normals[:, 1]
    labels = np.full(len(shell.faces), "wall", dtype=object)
    labels[vertical > VERTICAL_NORMAL_THRESHOLD] = "ceiling"
    labels[vertical < -VERTICAL_NORMAL_THRESHOLD] = "floor"
    return labels


def build_render_scene(
    hssd_root: Path,
    scene_id: str,
    region: RoomRegion,
    instances: list[FurnitureInstance],
) -> tuple[trimesh.Scene, ExportReport]:
    """Textured render assets placed by our instance matrices."""
    scene = trimesh.Scene()
    report = ExportReport(scene_id=scene_id, region_name=region.name)
    scene.add_geometry(shell_mesh(region, translucent=True), node_name="room_shell")
    objects_dir = hssd_root / "objects"
    cache: dict[str, trimesh.Scene] = {}
    for index, instance in enumerate(instances):
        asset = resolve_asset(objects_dir, instance.template_name)
        if asset is None:
            report.unresolved.append(instance.template_name)
            continue
        if instance.template_name not in cache:
            cache[instance.template_name] = load_display_geometry(asset.render, RENDER_FACE_BUDGET)
        asset_scene = cache[instance.template_name]
        placement = instance.transform_matrix()
        for part_name, geometry in asset_scene.geometry.items():
            # A shared geom_name makes repeated templates (a room usually has
            # several identical chairs) reference one copy of the geometry in
            # the exported glTF instead of duplicating it per instance.
            for node_transform in _node_transforms(asset_scene, part_name):
                scene.add_geometry(
                    geometry,
                    geom_name=f"{instance.template_name}::{part_name}",
                    node_name=f"piece_{index}_{part_name}",
                    transform=placement @ node_transform,
                )
        report.placed += 1
        report.layouts[asset.layout] = report.layouts.get(asset.layout, 0) + 1
        if not asset.has_collider:
            report.missing_collider.append(instance.template_name)
    return scene, report


def _node_transforms(asset_scene: trimesh.Scene, geometry_name: str) -> list[np.ndarray]:
    """Every placement of one geometry inside its own asset's node graph.

    A render asset is itself a small scene whose parts carry transforms; those
    must be composed with the instance matrix or parts land at the origin.
    """
    transforms = [
        asset_scene.graph[node][0]
        for node in asset_scene.graph.nodes_geometry
        if asset_scene.graph[node][1] == geometry_name
    ]
    return transforms or [np.eye(4)]


def build_acoustic_scene(
    hssd_root: Path,
    scene_id: str,
    region: RoomRegion,
    instances: list[FurnitureInstance],
    seed: int = 0,
) -> tuple[trimesh.Scene, ExportReport]:
    """The collider geometry the simulator sees, coloured by assigned absorption."""
    scene = trimesh.Scene()
    report = ExportReport(scene_id=scene_id, region_name=region.name)
    rng = np.random.default_rng(seed)
    objects_dir = hssd_root / "objects"

    shell = region.extrude()
    labels = shell_surface_labels(shell)
    face_colours = np.zeros((len(shell.faces), 4), dtype=np.uint8)
    for surface in ("floor", "wall", "ceiling"):
        material = material_for_label(surface, rng)
        mean = float(np.mean(material.energy_absorption["coeffs"]))
        colour = absorption_colour(mean)
        colour[3] = SHELL_ALPHA  # translucent, or it hides the furniture it contains
        face_colours[labels == surface] = colour
    shell.visual = trimesh.visual.ColorVisuals(shell, face_colors=face_colours)
    scene.add_geometry(shell, node_name="room_shell")

    for index, instance in enumerate(instances):
        asset = resolve_asset(objects_dir, instance.template_name)
        if asset is None:
            report.unresolved.append(instance.template_name)
            continue
        if asset.collider is None:
            report.missing_collider.append(instance.template_name)
            continue
        mesh = trimesh.load(asset.collider, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            report.unresolved.append(instance.template_name)
            continue
        mesh = mesh.copy()
        mesh.apply_transform(instance.transform_matrix())
        category = category_for_template(hssd_root, instance.template_name)
        material = material_for_label(category, rng)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh,
            face_colors=absorption_colour(float(np.mean(material.energy_absorption["coeffs"]))),
        )
        scene.add_geometry(mesh, node_name=f"piece_{index}")
        report.placed += 1
        report.layouts[asset.layout] = report.layouts.get(asset.layout, 0) + 1
    return scene, report


def export_region(
    hssd_root: Path,
    scene_id: str,
    region_name: str | None,
    mode: str,
    output_path: Path,
) -> ExportReport:
    regions = load_regions(hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json")
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    assignment = match_instances_to_regions(regions, instances)

    if region_name is None:
        index = max(range(len(regions)), key=lambda i: len(assignment[i]))
    else:
        matching = [i for i, region in enumerate(regions) if region.name == region_name]
        if not matching:
            available = ", ".join(region.name for region in regions)
            raise SystemExit(f"no region named {region_name!r}; available: {available}")
        index = matching[0]

    builder = build_render_scene if mode == "render" else build_acoustic_scene
    scene, report = builder(hssd_root, scene_id, regions[index], assignment[index])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported: bytes = scene.export(file_type="glb")  # type: ignore[no-untyped-call]
    output_path.write_bytes(exported)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--region", default=None, help="defaults to the busiest region")
    parser.add_argument("--mode", choices=("render", "acoustic"), default="render")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    report = export_region(
        arguments.hssd_root,
        arguments.scene_id,
        arguments.region,
        arguments.mode,
        arguments.output,
    )
    print(report.summary())
    if report.unresolved:
        print(f"unresolved templates: {', '.join(sorted(set(report.unresolved))[:10])}")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
