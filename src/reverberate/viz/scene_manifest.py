"""Describe a reconstructed room to the browser, without re-encoding its assets.

HSSD's render assets carry their textures as KTX2/Basis images
(``KHR_texture_basisu``). ``trimesh`` cannot decode that format and drops
those textures silently, so composing the room into a single glTF in Python
produces a grey, untextured room: the merge itself destroys the appearance we
are trying to show. three.js decodes KTX2 natively.

So the split is: Python owns every *interpretation* (which asset a template
resolves to, where each instance sits, what semantic category it has, what
acoustic material that implies, and the room shell we extrude ourselves), and
the browser owns only decoding and drawing. The manifest below is that
contract. Assets are referenced in place rather than copied or rewritten, so
what the browser draws is the dataset's own geometry under our transforms.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.apartment import (
    Storey,
    build_apartment,
    extrude_storey,
    instances_on_storey,
)
from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import FurnitureInstance, load_object_instances
from reverberate.geometry.materials import material_for_label
from reverberate.geometry.rooms import RoomPartition
from reverberate.geometry.scene_ids import local_name, scene_names
from reverberate.geometry.sim_geometry import obstacle_collider
from reverberate.viz.label_palette import (
    SHELL_LABEL_COLOURS,
    SHELL_RENDER_COLOURS,
    category_colour,
    rgba,
)
from reverberate.viz.room_surfaces import shell_surface_labels
from reverberate.viz.scene_pool import ASSET_DIR, SIM_DIR, link_into, render_path, sim_path

__all__ = [
    "ASSET_DIR",
    "SIM_DIR",
    "InstanceEntry",
    "ManifestReport",
    "build_instances",
    "export_simulation_collider",
    "link_asset",
    "outline_json",
    "rooms_json",
    "shell_meshes",
    "write_manifest",
]


@dataclass
class InstanceEntry:
    """One placed piece of furniture, as the browser needs to see it."""

    template: str
    category: str
    render_url: str
    collider_url: str
    #: True when this piece has no dedicated collider and its render mesh is
    #: what gets simulated, so the viewer can say so rather than imply a
    #: precision the data does not have.
    collider_is_render: bool
    #: Column-major 4x4, the layout ``THREE.Matrix4.fromArray`` expects.
    matrix: list[float]
    label_colour: list[int]
    #: The tabulated coefficient for this category, as handed to the solver.
    #: There is no longer a "before" and an "after": nothing rescales it.
    absorption: float
    #: The obstacle's surface under its instance matrix, in m².
    area: float


@dataclass
class ManifestReport:
    """Counts worth printing, so what was dropped is never silent."""

    placed: int = 0
    unresolved: list[str] = field(default_factory=list)
    render_as_collider: list[str] = field(default_factory=list)
    layouts: dict[str, int] = field(default_factory=dict)
    storey: str = ""

    def summary(self) -> str:
        layouts = ", ".join(f"{name}: {count}" for name, count in sorted(self.layouts.items()))
        return (
            f"{self.placed} pieces placed ({layouts}); "
            f"{len(self.unresolved)} unresolved, "
            f"{len(self.render_as_collider)} simulated from their render mesh"
        )


def column_major(matrix: np.ndarray) -> list[float]:
    """Flatten a numpy 4x4 into the column-major order glTF and three.js use."""
    return [float(value) for value in matrix.T.reshape(-1)]


def link_asset(scene: Path, hssd_root: Path, template: str) -> str | None:
    """Expose this template's render mesh under ``scene``, through the pool.

    The dataset's own glTF, referenced rather than copied or re-encoded: its
    KTX2 textures do not survive a Python side merge, and one copy on disk
    answers for every apartment that places the piece. See
    :mod:`reverberate.viz.scene_pool`.
    """
    pooled = render_path(hssd_root, template)
    if pooled is None:
        return None
    return link_into(scene, ASSET_DIR, f"{template}.glb", pooled)


def export_simulation_collider(hssd_root: Path, template: str, scene: Path) -> str | None:
    """Expose the exact mesh the solver will use for this template.

    It comes from ``obstacle_collider``, the single place that mesh is chosen,
    so the viewer cannot show one thing while the solver receives another.
    Calling it is what fills the collider pool, and the file linked here is the
    pool entry itself rather than a second export of the same mesh.
    """
    if obstacle_collider(hssd_root, template) is None:
        return None
    pooled = sim_path(template)
    if pooled is None:  # pragma: no cover - the call above has just built it
        return None
    return link_into(scene, SIM_DIR, f"{template}.glb", pooled)


def build_instances(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    scene: Path,
    seed: int = 0,
) -> tuple[list[InstanceEntry], ManifestReport]:
    rng = np.random.default_rng(seed)
    report = ManifestReport()
    entries: list[InstanceEntry] = []
    objects_dir = hssd_root / "objects"
    for instance in instances:
        asset = resolve_asset(objects_dir, instance.template_name)
        if asset is None:
            report.unresolved.append(instance.template_name)
            continue
        category = category_for_template(hssd_root, instance.template_name) or "unknown"
        material = material_for_label(category, rng)
        base_absorption = float(np.mean(material.energy_absorption["coeffs"]))
        if asset.collider_is_render:
            report.render_as_collider.append(instance.template_name)

        collider = obstacle_collider(hssd_root, instance.template_name)
        render_url = link_asset(scene, hssd_root, instance.template_name)
        if collider is None or render_url is None:
            report.unresolved.append(instance.template_name)
            continue
        placed = collider.copy()
        placed.apply_transform(instance.transform_matrix())

        entries.append(
            InstanceEntry(
                template=instance.template_name,
                category=category,
                render_url=render_url,
                collider_url=export_simulation_collider(hssd_root, instance.template_name, scene)
                or render_url,
                collider_is_render=asset.collider_is_render,
                matrix=column_major(instance.transform_matrix()),
                label_colour=list(category_colour(category)),
                absorption=base_absorption,
                area=float(placed.area),
            )
        )
        report.placed += 1
        report.layouts[asset.layout] = report.layouts.get(asset.layout, 0) + 1
    return entries, report


def shell_meshes(
    storey: Storey, seed: int = 0
) -> tuple[dict[str, trimesh.Trimesh], dict[str, float]]:
    """The apartment shell in each view, plus the absorption assigned per surface.

    Both are the *same* mesh with different colours, and that mesh is the one
    ``simulation_geometry`` hands to the simulator, so what is drawn is the
    simulator's input rather than a lookalike.
    """
    rng = np.random.default_rng(seed)
    absorptions = {
        surface: float(np.mean(material_for_label(surface, rng).energy_absorption["coeffs"]))
        for surface in ("floor", "wall", "ceiling")
    }
    base = extrude_storey(storey)
    labels = shell_surface_labels(base)

    def coloured(colour_of: dict[str, tuple[int, int, int]]) -> trimesh.Trimesh:
        mesh = base.copy()
        face_colours = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for surface, colour in colour_of.items():
            face_colours[labels == surface] = rgba(colour)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=face_colours)
        return mesh

    return (
        {
            "colour": coloured(SHELL_RENDER_COLOURS),
            "label": coloured(SHELL_LABEL_COLOURS),
        },
        absorptions,
    )


def outline_json(storey: Storey) -> list[dict[str, object]]:
    """The walkable outline, exteriors and holes, as the browser needs it."""
    return [
        {
            "exterior": [[float(x), float(z)] for x, z in polygon.exterior.coords],
            "holes": [[[float(x), float(z)] for x, z in hole.coords] for hole in polygon.interiors],
        }
        for polygon in storey.polygons
    ]


def rooms_json(rooms: Sequence[RoomPartition]) -> list[dict[str, object]]:
    """The everyday rooms as the plan draws them: name, footprint, where to write it.

    The footprint is the exterior ring of each piece, two decimals, the same
    shape the audit payload's ``rooms.json`` carries, so one point-in-polygon
    in the browser serves both. ``label_at`` is shapely's representative point,
    which is inside the polygon where a centroid of an L-shaped room is not.
    """
    entries = []
    for room in rooms:
        polygon = room.polygon
        pieces = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        point = polygon.representative_point()
        entries.append(
            {
                "name": room.name,
                "label": room.label,
                "regions": list(room.regions),
                "area_m2": round(room.area_m2, 3),
                "outline": [
                    [[round(float(x), 2), round(float(z), 2)] for x, z in piece.exterior.coords]
                    for piece in pieces
                    if not piece.is_empty
                ],
                "label_at": [round(float(point.x), 2), round(float(point.y), 2)],
            }
        )
    return entries


def write_manifest(
    hssd_root: Path, scene_id: str, target: Path, storey_index: int = 0
) -> ManifestReport:
    """Describe one apartment storey: its shell, its furniture and where you may walk.

    ``storey_index`` counts from the ground, and the default is the ground
    floor, which is what every caller wanted while only one storey was ever
    assembled. Seven of HSSD's 168 scenes have a second one; naming them is
    :mod:`reverberate.geometry.scene_ids`, and assembling them is this.
    """
    target.mkdir(parents=True, exist_ok=True)
    storeys = build_apartment(hssd_root, scene_id)
    if not 0 <= storey_index < len(storeys):
        raise ValueError(f"{scene_id} has {len(storeys)} storeys, asked for index {storey_index}")
    storey = storeys[storey_index]
    all_instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    instances = instances_on_storey(all_instances, storey)
    entries, report = build_instances(hssd_root, instances, target)
    report.storey = storey.summary()

    meshes, absorptions = shell_meshes(storey)
    for name, mesh in meshes.items():
        exported = mesh.export(file_type="glb")
        assert isinstance(exported, bytes)
        (target / f"shell_{name}.glb").write_bytes(exported)

    categories = sorted({entry.category for entry in entries})
    # The rooms a person would name, by the four rules of ADR 0010, as the
    # plan draws them. Not the dataset's region labels: `living room` on
    # 102344403 is 126 m2 of open space, not the 81 m2 slice so labelled.
    #
    # The storey's own rooms, not the scene's. `rooms_of_scene` returns every
    # room of every storey and keeps the outdoor ones, so a manifest built
    # from it would draw a garden on the plan and give the first floor the
    # ground floor's rooms as well as its own.
    rooms = list(storey.everyday)
    # This project's own name for the storey, which is what a report, a run
    # directory and the app's selector all use. The HSSD id travels beside it
    # rather than instead of it: 135 of the 168 are compound and say nothing.
    # The frozen table is the authority on how many storeys a scene has, not
    # this call's grouping: a name printed in a report has to keep meaning the
    # same thing, so a disagreement raises here rather than inventing a name.
    named = scene_names()[scene_id]
    local = local_name(scene_id, storey_index + 1 if named.storeys > 1 else None)
    manifest = {
        "local": local,
        "scene_id": scene_id,
        "storey_index": storey_index,
        "storeys": named.storeys,
        "title": f"{local}: {len(rooms)} rooms, {storey.doorways} doorways",
        "hint": f"{report.summary()}; {storey.summary()}",
        "rooms": rooms_json(rooms),
        # The walkable outline: rooms joined through the doorways found in the
        # stage's own walls. The viewer walks on exactly this, and the shell it
        # draws is this outline extruded, which is what gets simulated.
        "outline": outline_json(storey),
        "floorHeight": storey.floor_height,
        "ceilingHeight": storey.ceiling_height,
        "instances": [asdict(entry) for entry in entries],
        "legends": {
            "colour": [],
            "label": [
                {"label": category, "colour": list(category_colour(category))}
                for category in categories
            ]
            + [
                {"label": surface, "colour": list(colour)}
                for surface, colour in SHELL_LABEL_COLOURS.items()
            ],
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storey", type=int, default=0, help="storey index from the ground")
    arguments = parser.parse_args(argv)
    report = write_manifest(
        arguments.hssd_root, arguments.scene_id, arguments.output, arguments.storey
    )
    print(report.summary())
    print(report.storey)
    return 0


if __name__ == "__main__":
    sys.exit(main())
