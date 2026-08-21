"""Offline audit of HSSD furniture placement against its room shell.

Answers, on real data and without any rendering stack, the question the
rejected interior viewer could not: does placed furniture actually sit inside
the room shell it was matched to?

For every region of every audited scene, each matched furniture piece is
transformed by its instance matrix and its footprint compared to the region
polygon. The report is a falsifiable count (pieces whose footprint escapes the
shell, pieces sunk below or floating above the floor) rather than a visual
impression, plus an optional 2D plan PNG per region.

Run as ``python -m reverberate.geometry.audit_placement <hssd_root> [scene_id ...]``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

from reverberate.geometry.hssd_room import (
    FurnitureInstance,
    RoomRegion,
    load_collider_mesh,
    load_object_instances,
    load_regions,
    match_instances_to_regions,
)

#: A piece is only reported as escaping when a clear majority of its footprint
#: is outside, so that furniture legitimately flush against (or slightly
#: clipping through) a wall is not counted as a placement failure.
ESCAPE_AREA_FRACTION = 0.5

#: Vertical tolerance, in metres, for a piece resting on the floor.
FLOOR_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class PiecePlacement:
    """Where one transformed furniture piece ended up relative to its room."""

    template_name: str
    outside_area_fraction: float
    lowest_y: float
    highest_y: float
    footprint: Polygon

    @property
    def escapes_shell(self) -> bool:
        return self.outside_area_fraction > ESCAPE_AREA_FRACTION


@dataclass(frozen=True)
class RegionAudit:
    """Placement outcome for one room."""

    scene_id: str
    region_name: str
    region_label: str
    placements: list[PiecePlacement]
    floor_height: float
    ceiling_height: float

    @property
    def escaped(self) -> list[PiecePlacement]:
        return [p for p in self.placements if p.escapes_shell]

    @property
    def sunk(self) -> list[PiecePlacement]:
        return [p for p in self.placements if p.lowest_y < self.floor_height - FLOOR_TOLERANCE_M]

    @property
    def floating(self) -> list[PiecePlacement]:
        return [p for p in self.placements if p.lowest_y > self.floor_height + FLOOR_TOLERANCE_M]

    @property
    def above_ceiling(self) -> list[PiecePlacement]:
        return [p for p in self.placements if p.highest_y > self.ceiling_height]


def placed_mesh(instance: FurnitureInstance, objects_dir: Path) -> trimesh.Trimesh:
    """The furniture's collider mesh, in world coordinates."""
    mesh = load_collider_mesh(objects_dir, instance.template_name).copy()
    mesh.apply_transform(instance.transform_matrix())
    return mesh


def footprint_xz(mesh: trimesh.Trimesh) -> Polygon:
    """The mesh's ground footprint, as the convex hull of its vertices in (x, z).

    The convex hull is deliberate: it over-estimates the footprint, so a piece
    reported as inside the room really is inside, and it is robust to colliders
    made of several disconnected parts.
    """
    points = np.column_stack((mesh.vertices[:, 0], mesh.vertices[:, 2]))
    hull = Polygon(points).convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        raise ValueError("collider footprint degenerated to a point or a line")
    return hull


def audit_region(
    scene_id: str,
    region: RoomRegion,
    instances: list[FurnitureInstance],
    objects_dir: Path,
) -> RegionAudit:
    room_polygon = region.polygon_xz
    if not room_polygon.is_valid:
        room_polygon = room_polygon.buffer(0)
    placements = []
    for instance in instances:
        try:
            mesh = placed_mesh(instance, objects_dir)
        except (FileNotFoundError, TypeError, ValueError):
            continue
        footprint = footprint_xz(mesh)
        outside = footprint.difference(room_polygon).area
        placements.append(
            PiecePlacement(
                template_name=instance.template_name,
                outside_area_fraction=outside / footprint.area if footprint.area else 0.0,
                lowest_y=float(mesh.vertices[:, 1].min()),
                highest_y=float(mesh.vertices[:, 1].max()),
                footprint=footprint,
            )
        )
    return RegionAudit(
        scene_id=scene_id,
        region_name=region.name,
        region_label=region.label,
        placements=placements,
        floor_height=region.floor_height,
        ceiling_height=region.floor_height + region.extrusion_height,
    )


def audit_scene(hssd_root: Path, scene_id: str) -> list[RegionAudit]:
    regions = load_regions(hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json")
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    assignment = match_instances_to_regions(regions, instances)
    objects_dir = hssd_root / "objects"
    return [
        audit_region(scene_id, region, assignment[index], objects_dir)
        for index, region in enumerate(regions)
    ]


def plot_region(audit: RegionAudit, region: RoomRegion, output_path: Path) -> None:
    """Dump a 2D plan of the room polygon with every furniture footprint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 8))
    room_x, room_z = region.polygon_xz.exterior.xy
    axes.plot(room_x, room_z, color="black", linewidth=2, label="room polygon")
    for placement in audit.placements:
        hull_x, hull_z = placement.footprint.exterior.xy
        colour = "red" if placement.escapes_shell else "tab:blue"
        axes.fill(hull_x, hull_z, alpha=0.35, color=colour)
    axes.set_aspect("equal")
    axes.set_title(f"{audit.scene_id} / {audit.region_name} ({audit.region_label})")
    axes.set_xlabel("world x (m)")
    axes.set_ylabel("world z (m)")
    figure.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def format_report(audits: list[RegionAudit]) -> str:
    lines = []
    total = escaped = sunk = floating = above = 0
    for audit in audits:
        if not audit.placements:
            continue
        total += len(audit.placements)
        escaped += len(audit.escaped)
        sunk += len(audit.sunk)
        floating += len(audit.floating)
        above += len(audit.above_ceiling)
        lines.append(
            f"{audit.scene_id}/{audit.region_name} ({audit.region_label}): "
            f"{len(audit.placements)} pieces, {len(audit.escaped)} outside shell, "
            f"{len(audit.sunk)} below floor, {len(audit.floating)} floating, "
            f"{len(audit.above_ceiling)} above ceiling"
        )
        for placement in audit.escaped:
            lines.append(
                f"    OUTSIDE {placement.template_name}: "
                f"{placement.outside_area_fraction:.0%} of footprint out"
            )
    lines.append(
        f"TOTAL: {total} pieces, {escaped} outside shell, {sunk} below floor, "
        f"{floating} floating, {above} above ceiling"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("scene_ids", nargs="*", help="defaults to the first few scenes found")
    parser.add_argument("--limit", type=int, default=5, help="scenes to audit when none named")
    parser.add_argument("--plot-dir", type=Path, default=None, help="write 2D plan PNGs here")
    arguments = parser.parse_args(argv)

    scene_ids = arguments.scene_ids or [
        path.name.split(".")[0]
        for path in sorted((arguments.hssd_root / "scenes").glob("*.scene_instance.json"))[
            : arguments.limit
        ]
    ]

    all_audits = []
    for scene_id in scene_ids:
        audits = audit_scene(arguments.hssd_root, scene_id)
        all_audits.extend(audits)
        if arguments.plot_dir is not None:
            arguments.plot_dir.mkdir(parents=True, exist_ok=True)
            regions = load_regions(
                arguments.hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
            )
            for audit, region in zip(audits, regions, strict=True):
                if audit.placements:
                    safe_name = audit.region_name.replace("/", "_")
                    plot_region(audit, region, arguments.plot_dir / f"{scene_id}_{safe_name}.png")

    print(format_report(all_audits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
