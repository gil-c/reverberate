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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import FurnitureInstance, RoomRegion
from reverberate.geometry.materials import material_for_label
from reverberate.viz.label_palette import SHELL_LABEL_COLOURS, SHELL_RENDER_COLOURS, category_colour
from reverberate.viz.room_surfaces import (
    absorption_colour,
    select_region,
    shell_mesh,
    shell_surface_labels,
)

#: Subdirectory of the served site where asset symlinks are created.
ASSET_DIR = "assets"


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
    acoustic_colour: list[int]
    absorption: float


@dataclass
class ManifestReport:
    """Counts worth printing, so what was dropped is never silent."""

    placed: int = 0
    unresolved: list[str] = field(default_factory=list)
    render_as_collider: list[str] = field(default_factory=list)
    layouts: dict[str, int] = field(default_factory=dict)

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


def link_asset(source: Path, target_dir: Path) -> str:
    """Expose one asset file under the served directory, without copying it.

    A room's unique render assets run to well over a hundred megabytes, and
    copying them per export would be both slow and pointless: they are read
    only. A symlink keeps the served site self describing while leaving one
    copy on disk.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    link = target_dir / source.name
    if not link.exists():
        link.symlink_to(source.resolve())
    return f"{ASSET_DIR}/{source.name}"


def build_instances(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    asset_target: Path,
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
        absorption = float(np.mean(material.energy_absorption["coeffs"]))
        if asset.collider_is_render:
            report.render_as_collider.append(instance.template_name)
        entries.append(
            InstanceEntry(
                template=instance.template_name,
                category=category,
                render_url=link_asset(asset.render, asset_target),
                collider_url=link_asset(asset.collider, asset_target),
                collider_is_render=asset.collider_is_render,
                matrix=column_major(instance.transform_matrix()),
                label_colour=list(category_colour(category)),
                acoustic_colour=absorption_colour(absorption)[:3].tolist(),
                absorption=absorption,
            )
        )
        report.placed += 1
        report.layouts[asset.layout] = report.layouts.get(asset.layout, 0) + 1
    return entries, report


def export_shells(region: RoomRegion, target: Path, seed: int = 0) -> dict[str, float]:
    """Write the room shell once per view, and report its acoustic assignment.

    The shell is our own extrusion rather than dataset geometry, so it is
    exported here (small, vertex coloured, no textures involved) instead of
    being rebuilt in the browser: the room the viewer draws must be the room
    the reconstruction produced.
    """
    rng = np.random.default_rng(seed)
    absorptions = {
        surface: float(np.mean(material_for_label(surface, rng).energy_absorption["coeffs"]))
        for surface in ("floor", "wall", "ceiling")
    }

    acoustic = region.extrude()
    labels = shell_surface_labels(acoustic)
    face_colours = np.zeros((len(acoustic.faces), 4), dtype=np.uint8)
    for surface, absorption in absorptions.items():
        face_colours[labels == surface] = absorption_colour(absorption)
    acoustic.visual = trimesh.visual.ColorVisuals(acoustic, face_colors=face_colours)

    for name, mesh in (
        ("colour", shell_mesh(region, SHELL_RENDER_COLOURS)),
        ("label", shell_mesh(region, SHELL_LABEL_COLOURS)),
        ("acoustic", acoustic),
    ):
        exported = mesh.export(file_type="glb")
        assert isinstance(exported, bytes)
        (target / f"shell_{name}.glb").write_bytes(exported)
    return absorptions


def write_manifest(
    hssd_root: Path, scene_id: str, region_name: str | None, target: Path
) -> ManifestReport:
    target.mkdir(parents=True, exist_ok=True)
    region, instances = select_region(hssd_root, scene_id, region_name)
    entries, report = build_instances(hssd_root, instances, target / ASSET_DIR)
    absorptions = export_shells(region, target)

    categories = sorted({entry.category for entry in entries})
    manifest = {
        "title": f"{scene_id} / {region.name} ({region.label})",
        "hint": report.summary(),
        # The authored floor polygon, so the walk camera refuses to leave the
        # room using the very same notion of "inside" as the reconstruction.
        "polygon": [[float(x), float(z)] for x, _, z in region.poly_loop],
        "floorHeight": float(region.floor_height),
        "ceilingHeight": float(region.floor_height + region.extrusion_height),
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
            "acoustic": [
                {
                    "label": f"{surface} ({absorption:.2f})",
                    "colour": absorption_colour(absorption)[:3].tolist(),
                }
                for surface, absorption in absorptions.items()
            ],
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--region", default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = write_manifest(
        arguments.hssd_root, arguments.scene_id, arguments.region, arguments.output
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
