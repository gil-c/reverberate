"""Which room each point of a scene belongs to, in the everyday sense.

HSSD's `region_annotations` are floor polygons with a label, and nothing in them
says whether a wall stands between two neighbours, so the dataset cuts one
continuous body of air into several "rooms". On `102344403` the `living room` it
names is 81 m2 of a space that runs on into a 26.7 m2 `kitchen` through an
opening 6.97 m wide, and simulating that region alone seals a rigid wall across
it. Four rules put the air back together:

1. the passage must be wider than a door, :data:`DOOR_MAX_M`;
2. a wall separates whatever the hole in it, so the shared boundary must carry
   less than :data:`WALL_MAX_M` of wall;
3. a hallway counts as a door, and is then attached to the largest room it
   opens onto;
4. nothing under :data:`MIN_ROOM_M2` is a room.

**Both thresholds are read off measurements, not chosen**, and the reasoning
behind every one of these -- and what they replace -- is
`docs/adr/0010-a-room-is-not-a-region.md`. Read it before changing a number
here. `scripts/room_openings.py` is how they were measured.

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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from reverberate.geometry.apartment import (
    OUTDOOR_LABELS,
    WALL_SECTION_HEIGHT,
    clean_outline,
    find_doorways,
    load_stage,
    wall_footprint,
)
from reverberate.geometry.hssd_room import RoomRegion, load_regions

__all__ = [
    "DOOR_MAX_M",
    "MIN_ROOM_M2",
    "WALL_MAX_M",
    "Boundary",
    "Partition",
    "RoomPartition",
    "everyday_rooms",
    "partition_of_scene",
    "room_holding",
    "rooms_of_scene",
    "shared_boundaries",
]

#: Widest gap a single door leaf can fill. A passage wider than this has no
#: door in it. Set from the empty band between the measured 0.96 m and 1.33 m.
DOOR_MAX_M = 1.10

#: Wall a shared boundary may carry and still dissolve. Set from the empty band
#: between the measured 0.79 m and 1.35 m.
WALL_MAX_M = 1.05

#: Floor area under which a region is not a room but part of the next one. A
#: decision, not a measurement: 2 m2 of air has no acoustic life of its own.
MIN_ROOM_M2 = 2.0

#: Circulation. Separates the rooms it serves, then joins the largest of them.
#: An entrance hall is circulation on the same terms as a corridor, and the
#: dataset uses both labels: 162 regions are `hallway`, 58 are the other.
HALLWAY_LABELS = frozenset({"hallway", "entryway/foyer/lobby"})

#: A run of open boundary shorter than this is noise from the section rather
#: than a passage.
MIN_GAP_M = 0.25

#: How far a region's outline reaches across the wall band to meet its
#: neighbour's. Must exceed the thickest partition; measured, gaps start at
#: 0.15 m.
BOUNDARY_REACH_M = 0.16


@dataclass(frozen=True)
class Boundary:
    """What two neighbouring regions share, and how much of it stands open."""

    contact_m: float
    open_m: float
    #: The widest single run of opening, which is what a door has to fill.
    widest_m: float

    @property
    def walled_m(self) -> float:
        return self.contact_m - self.open_m

    @property
    def dissolves(self) -> bool:
        """One room, by rules 1 and 2."""
        return self.widest_m > DOOR_MAX_M and self.walled_m < WALL_MAX_M


def shared_boundaries(
    polygons: list[Polygon], walls: Polygon | MultiPolygon
) -> dict[tuple[int, int], Boundary]:
    """Every pair of regions that touch, with the opening the walls leave.

    The wall band is read from the same horizontal section of the stage that
    :func:`~reverberate.geometry.apartment.find_doorways` uses, so a doorway is
    measured rather than invented.
    """
    found: dict[tuple[int, int], Boundary] = {}
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            # ``boundary`` rather than ``exterior``: two scenes of the 168 have a
            # region whose authored loop self-intersects, so cleaning it returns
            # a MultiPolygon, which has no exterior ring. Regions carry no holes,
            # so for every other one the two are the same line.
            line = polygons[i].boundary.intersection(polygons[j].buffer(BOUNDARY_REACH_M))
            if line.length < 0.3:
                continue
            free = line.difference(walls)
            parts = free.geoms if hasattr(free, "geoms") else [free]
            runs = [part.length for part in parts if part.length > MIN_GAP_M]
            found[i, j] = Boundary(
                contact_m=float(line.length),
                open_m=float(sum(runs)),
                widest_m=float(max(runs, default=0.0)),
            )
    return found


def everyday_rooms(regions: list[RoomRegion], walls: Polygon | MultiPolygon) -> list[RoomPartition]:
    """Group regions into the rooms a person standing in them would name.

    Outdoor regions are grouped too, and never merge into a room, so a caller
    that wants only the inside can drop the rooms whose ``outdoor`` is set. Pass
    them in rather than filtering first: a balcony store belongs with the
    balcony, and without the balcony there is nothing for it to belong to.
    """
    polygons = [region.polygon_xz.buffer(0) for region in regions]
    labels = [region.label for region in regions]
    outdoors = [label in OUTDOOR_LABELS for label in labels]
    shared = shared_boundaries(polygons, walls)

    parent = list(range(len(regions)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for (i, j), boundary in shared.items():
        if outdoors[i] or outdoors[j]:
            continue
        if labels[i] in HALLWAY_LABELS or labels[j] in HALLWAY_LABELS:
            continue
        if boundary.dissolves:
            union(i, j)

    def group_area(i: int) -> float:
        return float(sum(polygons[k].area for k in range(len(regions)) if find(k) == find(i)))

    def attach(
        i: int,
        prefer: Callable[[int, Boundary], tuple[Any, ...]],
        *,
        outdoor_ok: bool,
    ) -> None:
        """Join ``i`` to the neighbour it opens onto that ``prefer`` ranks highest."""
        best, key = None, None
        for j in range(len(regions)):
            if j == i or find(j) == find(i):
                continue
            if outdoors[j] and not outdoor_ok:
                continue
            boundary = shared.get((min(i, j), max(i, j)))
            if boundary is None or boundary.widest_m <= 0.0:
                continue
            rank = prefer(j, boundary)
            if key is None or rank > key:
                best, key = j, rank
        if best is not None:
            union(i, best)

    # Rule 3. Attached only once the rooms are formed, which is what stops a
    # corridor carrying a merge from one of its ends to the other.
    for i, label in enumerate(labels):
        if label in HALLWAY_LABELS and not outdoors[i]:
            attach(
                i,
                lambda j, boundary: (labels[j] not in HALLWAY_LABELS, group_area(j)),
                outdoor_ok=False,
            )

    # Rule 4. Indoors for preference: a balcony store that opens only onto the
    # balcony belongs outdoors, not through a solid wall into a bedroom.
    for i in sorted(range(len(regions)), key=lambda k: polygons[k].area):
        if outdoors[i] or group_area(i) >= MIN_ROOM_M2:
            continue
        attach(
            i,
            lambda j, boundary: (not outdoors[j], boundary.widest_m, boundary.contact_m),
            outdoor_ok=True,
        )

    grouped: dict[int, list[int]] = {}
    for i in range(len(regions)):
        grouped.setdefault(find(i), []).append(i)

    rooms = []
    for members in grouped.values():
        members.sort(key=lambda k: -polygons[k].area)
        rooms.append(
            RoomPartition(
                name="-".join(regions[k].name for k in members),
                label=labels[members[0]],
                polygon=_joined(polygons, members, walls),
                regions=tuple(regions[k].name for k in members),
                outdoor=any(outdoors[k] for k in members),
            )
        )
    return sorted(rooms, key=lambda room: -room.area_m2)


def _joined(
    polygons: list[Polygon], members: list[int], walls: Polygon | MultiPolygon
) -> Polygon | MultiPolygon:
    """One shape for a room, its own passages included.

    HSSD's regions do not tile: neighbours are held 0.15 m apart by the wall
    band, so a plain union of the parts of a room comes back disconnected --
    three pieces for the 126 m2 of `102344403`, which is not a volume anything
    can be simulated in. The gap is closed the way
    :func:`~reverberate.geometry.apartment.build_storey` closes it for a whole
    storey, by adding back the pieces of that band the walls leave open. The
    outer boundary therefore stays the authored polygons plus real doorways,
    never a dilated silhouette no wall corresponds to.
    """
    parts = [polygons[k] for k in members]
    shape = unary_union([*parts, *find_doorways(parts, walls)]).buffer(0)
    return clean_outline(shape)


def rooms_of_scene(hssd_root: Path, scene_id: str) -> list[RoomPartition]:
    """The everyday rooms of a whole scene, outdoor ones included."""
    regions = load_regions(
        Path(hssd_root) / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
    )
    if not regions:
        raise ValueError(f"{scene_id} has no annotated regions")
    stage = load_stage(Path(hssd_root), scene_id)
    floor = float(np.mean([region.floor_height for region in regions]))
    return everyday_rooms(regions, wall_footprint(stage, floor + WALL_SECTION_HEIGHT))


def room_holding(rooms: list[RoomPartition], name: str) -> RoomPartition:
    """The room known by ``name``, or the one that folded that region in.

    A run asks for `living room` and gets the whole open space it belongs to.
    Naming a region that is no longer a room of its own is the ordinary case,
    not an error: that is the point of the partition.
    """
    for room in rooms:
        if room.name == name or name in room.regions:
            return room
    known = ", ".join(sorted(r.name for r in rooms))
    raise ValueError(f"no room named {name!r}; this scene has {known}")


@dataclass(frozen=True)
class RoomPartition:
    """One everyday room: the regions that make it up, joined into one shape."""

    name: str
    label: str
    polygon: Polygon | MultiPolygon
    #: The region names folded in, the largest first.
    regions: tuple[str, ...]
    #: Not interior air, so not a room to simulate. Kept rather than dropped so
    #: a caller can say what it left out instead of quietly losing it.
    outdoor: bool = False

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
    rooms = [room for room in rooms_of_scene(hssd_root, scene_id) if not room.outdoor]
    if not rooms:
        raise ValueError(f"{scene_id} has no interior regions to partition")
    raster = _fill_unclaimed(_rasterise(rooms, np.asarray(xv), np.asarray(zv)))
    if int(raster.min()) < 0:
        raise AssertionError("the room partition left cells with no owner")
    return Partition(rooms=tuple(rooms), raster=raster, xv=np.asarray(xv), zv=np.asarray(zv))
