"""Which room each grid node belongs to, in the everyday sense of the word.

The audit view has to draw one room at the grid's own step and its neighbours
coarser, so it needs to say which node is in which room. HSSD does not: its
`SemanticRegion` list splits a bedroom from the wardrobe it opens into, and the
scene this project uses has **seven** such closets among nineteen interior
regions. A wardrobe is part of the room a person stands in, so the everyday
room is a region plus the closets that open into it.

**The closet is assigned by the doorway it opens through, not by the wall it
shares.** W34 measured both criteria on this scene and they disagree on five of
seven, because a closet typically shares almost exactly as much boundary with
the room behind it as with the room it opens into: 2.10 m against 2.10 m. The
gap in the wall is the thing that decides it physically, and
:func:`~reverberate.geometry.apartment.find_doorways` already reads those gaps
out of the stage mesh rather than inventing them.

**The partition is two dimensional, because rooms are prisms.** A region is a
floor polygon extruded to the ceiling, so a node's room depends on its x and z
and not on its height. That turns a billion point-in-polygon tests into one
raster of the grid's own x-z lattice, 104 MB for the flat at 16 kHz, and a
lookup per node.

**Every node gets an owner, including the ones inside walls.** A boundary node
sits *on* a surface, so a great many of them are in the wall band between two
rooms, or under furniture that overhangs the authored outline, and no polygon
contains them. Those take the nearest room by a distance transform of the
raster, which is the only rule that keeps the partition total: a node with no
owner would silently vanish from every tier of the picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reverberate.geometry.apartment import (
    OUTDOOR_LABELS,
    WALL_SECTION_HEIGHT,
    find_doorways,
    load_stage,
)
from reverberate.geometry.hssd_room import RoomRegion, load_regions

__all__ = [
    "MIN_ROOM_AREA",
    "Partition",
    "RoomPartition",
    "merge_closets",
    "partition_of_scene",
]

#: A region smaller than this is an alcove of a room rather than a room. The
#: seven closets of scene 102344022 run from 0.56 to 2.92 m2 and the smallest
#: thing anyone would call a room is the 4.67 m2 toilet, so the boundary is
#: wide rather than delicate.
MIN_ROOM_AREA = 4.0


@dataclass(frozen=True)
class RoomPartition:
    """One everyday room: a region, plus the alcoves that open into it."""

    name: str
    label: str
    polygon: Polygon | MultiPolygon
    #: The region names folded in, the room's own first.
    regions: tuple[str, ...]

    @property
    def area_m2(self) -> float:
        return float(self.polygon.area)


@dataclass(frozen=True)
class Partition:
    """The rooms of a scene and the raster that assigns a node to one of them."""

    rooms: tuple[RoomPartition, ...]
    #: (nx, nz) int8, one room index per cell of the grid's x-z lattice. Total
    #: by construction: every cell names a room.
    raster: np.ndarray
    #: The axis samples the raster was built on, so a caller cannot pair it
    #: with a grid of a different step.
    xv: np.ndarray
    zv: np.ndarray

    @property
    def names(self) -> list[str]:
        return [room.name for room in self.rooms]

    def of_nodes(self, sub_x: np.ndarray, sub_z: np.ndarray) -> np.ndarray:
        """The room index of every node, from its x and z subscripts."""
        return np.asarray(self.raster[sub_x, sub_z])

    def summary(self) -> str:
        rows = ", ".join(
            f"{room.name} {room.area_m2:.1f} m2"
            + (f" (+{len(room.regions) - 1})" if len(room.regions) > 1 else "")
            for room in self.rooms
        )
        return f"{len(self.rooms)} rooms: {rows}"


def merge_closets(regions: list[RoomRegion], doorways: list[Polygon]) -> list[RoomPartition]:
    """Fold every region under :data:`MIN_ROOM_AREA` into the room it opens into.

    A closet is joined to the largest room reachable through a doorway that
    touches it. When no doorway touches it -- an alcove with no gap in the
    section, which happens where the opening is above the section height -- it
    falls back to the room it shares the most boundary with, and that fallback
    is the criterion W34 measured as ambiguous, so it is used only when the
    reliable one has nothing to say.
    """
    shapes = {region.name: region.polygon_xz.buffer(0) for region in regions}
    rooms = [region for region in regions if shapes[region.name].area >= MIN_ROOM_AREA]
    closets = [region for region in regions if shapes[region.name].area < MIN_ROOM_AREA]
    joined: dict[str, list[str]] = {region.name: [] for region in rooms}

    for closet in closets:
        shape = shapes[closet.name]
        reachable: set[str] = set()
        for doorway in doorways:
            if not doorway.intersects(shape):
                continue
            reachable |= {room.name for room in rooms if doorway.intersects(shapes[room.name])}
        if reachable:
            host = max(reachable, key=lambda name: shapes[name].area)
        else:
            # No gap in the section reaches this one. Shared boundary is the
            # only thing left, and it is the criterion that disagrees.
            host = max(
                (room.name for room in rooms),
                key=lambda name: shape.buffer(0.15).intersection(shapes[name]).area,
            )
        joined[host].append(closet.name)

    return [
        RoomPartition(
            name=room.name,
            label=room.label,
            polygon=unary_union([shapes[room.name], *(shapes[n] for n in joined[room.name])]),
            regions=(room.name, *joined[room.name]),
        )
        for room in rooms
    ]


def _rasterise(rooms: list[RoomPartition], xv: np.ndarray, zv: np.ndarray) -> np.ndarray:
    """One room index per cell of the x-z lattice, unclaimed cells left at -1.

    Rasterised polygon by polygon over its own bounding box rather than by
    testing the whole lattice against every room: the flat at 16 kHz is 104 M
    cells and thirteen rooms, and a room covers a few per cent of the plane.
    """
    raster = np.full((xv.size, zv.size), -1, dtype=np.int8)
    for index, room in enumerate(rooms):
        x0, z0, x1, z1 = room.polygon.bounds
        lo_x, hi_x = np.searchsorted(xv, (x0, x1), side="left")
        lo_z, hi_z = np.searchsorted(zv, (z0, z1), side="left")
        hi_x, hi_z = min(hi_x + 1, xv.size), min(hi_z + 1, zv.size)
        if hi_x <= lo_x or hi_z <= lo_z:
            continue
        gx, gz = np.meshgrid(xv[lo_x:hi_x], zv[lo_z:hi_z], indexing="ij")
        inside = shapely.contains_xy(room.polygon, gx, gz)
        window = raster[lo_x:hi_x, lo_z:hi_z]
        # First room wins an overlap. Regions do not overlap by construction,
        # and where a cleaned boundary makes them touch by a cell the choice
        # is arbitrary either way.
        np.copyto(window, np.int8(index), where=inside & (window < 0))
    return raster


def _fill_unclaimed(raster: np.ndarray) -> np.ndarray:
    """Give every cell an owner: the nearest room, by Euclidean distance.

    Wall interiors, doorway bands and furniture overhanging the authored
    outline are all unclaimed, and they hold a large share of the boundary
    nodes -- a boundary node sits on a surface, and the surfaces are the walls.
    Leaving them out would drop them from every tier of the picture, which is
    exactly the silent loss an audit view exists to prevent.
    """
    from scipy import ndimage

    missing = raster < 0
    if not missing.any():
        return raster
    _, nearest = ndimage.distance_transform_edt(missing, return_indices=True)
    filled = raster.copy()
    filled[missing] = raster[nearest[0][missing], nearest[1][missing]]
    return filled


def partition_of_scene(hssd_root: Path, scene_id: str, xv: np.ndarray, zv: np.ndarray) -> Partition:
    """The everyday rooms of a scene, rasterised onto one grid's x-z lattice.

    ``xv`` and ``zv`` are ``cart_grid.h5``'s own axis samples, so the raster
    lines up with the voxelisation cell for cell and a node's room is a lookup
    rather than a search.
    """
    regions = [
        region
        for region in load_regions(
            Path(hssd_root) / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
        )
        if region.label not in OUTDOOR_LABELS
    ]
    if not regions:
        raise ValueError(f"{scene_id} has no interior regions to partition")
    stage = load_stage(Path(hssd_root), scene_id)
    floor = float(np.mean([region.floor_height for region in regions]))
    from reverberate.geometry.apartment import wall_footprint

    walls = wall_footprint(stage, floor + WALL_SECTION_HEIGHT)
    doorways = find_doorways([region.polygon_xz.buffer(0) for region in regions], walls)
    rooms = merge_closets(regions, doorways)
    raster = _fill_unclaimed(_rasterise(rooms, np.asarray(xv), np.asarray(zv)))
    if int(raster.min()) < 0:
        raise AssertionError("the room partition left cells with no owner")
    return Partition(rooms=tuple(rooms), raster=raster, xv=np.asarray(xv), zv=np.asarray(zv))
