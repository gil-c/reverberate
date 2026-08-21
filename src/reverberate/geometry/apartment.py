"""Assemble a whole apartment: rooms joined through the doorways in its walls.

A scene is an apartment and its regions are the rooms, but HSSD stores no
connectivity: the semantic config has only per-room floor polygons, and a
search of `SemanticRegion` finds no adjacency property either. Habitat itself
never needs one, because it hands the stage mesh to Recast and a doorway is
simply a gap in the wall geometry.

This module reads those gaps rather than inventing them. The stage mesh is
sectioned horizontally near the floor, which gives the walls as they really
are; the space between neighbouring rooms is then whatever that section does
*not* cover. A gap that touches two or more rooms is a doorway, and the rooms
it joins become one walkable volume.

The polygon produced here is used for three things at once, which is the point:
the viewer walks on it, the room shell is extruded from it, and the same
extrusion is what pyroomacoustics receives. There is no second geometry that
could quietly disagree with what is on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from reverberate.geometry.hssd_room import RoomRegion, load_regions

#: Height above a floor at which the stage is sectioned to find its walls.
#: Doors reach the floor, so a doorway is a gap at this height, while window
#: sills sit around 0.7 m and stay solid: sectioning higher would let the
#: walkable area leak outdoors through the windows.
WALL_SECTION_HEIGHT = 0.3

#: Half thickness given to a wall cross-section line. The stage's walls are
#: modelled as thin shells, so a section returns their outlines; widening
#: those lines slightly turns them into barriers that can be subtracted.
WALL_LINE_HALF_WIDTH = 0.05

#: How far a room is dilated when looking for doorways. Must exceed the
#: thickest wall between two rooms (measured: gaps start at 0.15 m).
DOORWAY_SEARCH_DISTANCE = 0.6

#: A doorway candidate smaller than this is noise from the section rather than
#: a passage a person could use.
MIN_DOORWAY_AREA = 0.05

#: Regions within this distance in height belong to the same storey.
STOREY_TOLERANCE = 0.1

#: Tolerance used to clean the walkable outline before it is extruded. The
#: union of rooms and doorways leaves slivers and near-duplicate vertices
#: where buffered edges meet, and those make the triangulated prism
#: non-watertight, which pyroomacoustics cannot use. Cleaning is applied to
#: the walkable polygon itself, not to a copy, so the surface the viewer walks
#: on stays identical to the volume that gets simulated.
OUTLINE_CLEAN_TOLERANCE = 0.005


@dataclass(frozen=True)
class Storey:
    """One walkable level of an apartment, and the rooms that make it up."""

    floor_height: float
    ceiling_height: float
    walkable: Polygon | MultiPolygon
    rooms: list[RoomRegion]
    doorways: int

    @property
    def polygons(self) -> list[Polygon]:
        if isinstance(self.walkable, MultiPolygon):
            return list(self.walkable.geoms)
        return [self.walkable]

    def summary(self) -> str:
        return (
            f"storey at {self.floor_height:.1f} m: {len(self.rooms)} rooms, "
            f"{self.doorways} doorways, {len(self.polygons)} connected part(s), "
            f"{self.walkable.area:.0f} m2"
        )


def group_by_storey(regions: list[RoomRegion]) -> dict[float, list[RoomRegion]]:
    """Bucket regions by floor height, since apartments can have storeys."""
    storeys: dict[float, list[RoomRegion]] = {}
    for region in regions:
        for height in storeys:
            if abs(height - region.floor_height) <= STOREY_TOLERANCE:
                storeys[height].append(region)
                break
        else:
            storeys[region.floor_height] = [region]
    return storeys


def wall_footprint(stage: trimesh.Trimesh, height: float) -> Polygon | MultiPolygon:
    """The walls of the architecture, as seen in a horizontal slice at ``height``.

    The section returns the wall *surfaces* as polylines, which are widened
    into thin barriers. Filling the closed loops instead would be wrong in
    both directions: those loops run around the inside of a room, so filling
    them turns whole rooms into solid wall, while walls modelled as open
    shells produce no loop to fill at all.
    """
    section = stage.section(plane_origin=[0, height, 0], plane_normal=[0, 1, 0])
    if section is None:
        return Polygon()
    lines = [
        LineString(
            np.column_stack(
                (section.vertices[entity.points][:, 0], section.vertices[entity.points][:, 2])
            )
        )
        for entity in section.entities
        if len(entity.points) > 1
    ]
    if not lines:
        return Polygon()
    return unary_union(lines).buffer(WALL_LINE_HALF_WIDTH)


def find_doorways(rooms: list[Polygon], walls: Polygon | MultiPolygon) -> list[Polygon]:
    """The pieces of wall-gap that join two or more rooms.

    Everything between the rooms that is not wall is a candidate; a candidate
    is only a doorway if it actually reaches into at least two rooms, which is
    what rules out the space outside the front door, alcoves and section noise.
    """
    room_area = unary_union(rooms)
    between = room_area.buffer(DOORWAY_SEARCH_DISTANCE).difference(room_area).difference(walls)
    candidates = list(between.geoms) if isinstance(between, MultiPolygon) else [between]
    doorways = []
    for candidate in candidates:
        if candidate.is_empty or candidate.area < MIN_DOORWAY_AREA:
            continue
        touched = sum(1 for room in rooms if candidate.intersects(room))
        if touched >= 2:
            doorways.append(candidate)
    return doorways


def clean_outline(walkable: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Remove slivers and redundant vertices from the walkable outline."""
    tolerance = OUTLINE_CLEAN_TOLERANCE
    cleaned = walkable.buffer(tolerance / 5).buffer(-tolerance / 5).simplify(tolerance)
    if cleaned.is_empty:
        return walkable
    return cleaned


def build_storey(regions: list[RoomRegion], stage: trimesh.Trimesh) -> Storey:
    rooms = [region.polygon_xz.buffer(0) for region in regions]
    floor_height = float(np.mean([region.floor_height for region in regions]))
    walls = wall_footprint(stage, floor_height + WALL_SECTION_HEIGHT)
    doorways = find_doorways(rooms, walls)
    # The outer boundary stays exactly the authored room polygons: only the
    # doorway pieces are added, so the apartment never grows a dilated,
    # rounded-off silhouette that no wall corresponds to.
    walkable = unary_union([*rooms, *doorways])
    walkable = clean_outline(walkable)
    ceiling = floor_height + float(np.mean([r.extrusion_height for r in regions]))
    return Storey(
        floor_height=floor_height,
        ceiling_height=ceiling,
        walkable=walkable,
        rooms=regions,
        doorways=len(doorways),
    )


def load_stage(hssd_root: Path, scene_id: str) -> trimesh.Trimesh:
    """The scene's architecture mesh, welded so its walls section cleanly.

    HSSD ships the stage with unwelded vertices (11205 vertices for 4254
    faces), which leaves a horizontal section as disconnected fragments.
    """
    stage = trimesh.load(hssd_root / "stages" / f"{scene_id}.glb", force="mesh")
    if not isinstance(stage, trimesh.Trimesh):
        raise TypeError(f"expected a Trimesh stage for {scene_id}")
    stage.merge_vertices()
    return stage


def build_apartment(hssd_root: Path, scene_id: str) -> list[Storey]:
    """Every walkable storey of one apartment, largest floor area first."""
    regions = load_regions(hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json")
    stage = load_stage(hssd_root, scene_id)
    storeys = [build_storey(group, stage) for group in group_by_storey(regions).values()]
    return sorted(storeys, key=lambda storey: -storey.walkable.area)


def extrude_storey(storey: Storey) -> trimesh.Trimesh:
    """The storey's air volume, as a watertight mesh for the simulator.

    This is the mesh pyroomacoustics receives *and* the mesh the viewer draws
    as the room shell, so what is on screen cannot drift from what is
    simulated.
    """
    parts = []
    for polygon in storey.polygons:
        if polygon.is_empty or polygon.area <= 0:
            continue
        # extrude_polygon works in the polygon's own XY frame and extrudes
        # along +Z; the same -z convention as RoomRegion.extrude maps that
        # frame onto the world with a rotation rather than a mirror.
        flipped = Polygon(
            [(x, -z) for x, z in polygon.exterior.coords],
            [[(x, -z) for x, z in interior.coords] for interior in polygon.interiors],
        )
        if not flipped.is_valid:
            flipped = flipped.buffer(0)
        prism = trimesh.creation.extrude_polygon(
            flipped, height=storey.ceiling_height - storey.floor_height
        )
        prism.apply_transform(
            trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
                angle=-np.pi / 2, direction=[1, 0, 0]
            )
        )
        prism.apply_translation([0.0, storey.floor_height, 0.0])
        parts.append(prism)
    if not parts:
        raise ValueError("storey has no extrudable walkable area")
    combined = trimesh.util.concatenate(parts)
    assert isinstance(combined, trimesh.Trimesh)
    return combined
