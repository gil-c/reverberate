"""A storey's stations and rails: where sources may stand, sit and travel.

A layout is a function of the dwelling and of nothing random. It holds

- **the floor** the format's rules are checked against: the free floor of
  :func:`reverberate.experiments.w40_volume_field.plan.free_floor`, the
  walkable outline whose boundary is the walls, the rooms of ADR 0010 and the
  footprints of the furniture one sits on;
- **the stations**: one seat or more per piece of seating, standing spots
  picked on the listening lattice's 0.40 m pitch about a metre apart, and a
  waypoint in every door;
- **the rails**: a sparse graph over the standing spots and the doors, each
  rail the shortest way across the free floor, and one or two rails from every
  seat to the spots next to it.

**Clearance.** Rails are routed on the free floor pulled in by a further
:data:`RAIL_MARGIN_M`, 3 cm: 0.28 m from a wall and 0.25 m from furniture.
The margin is what lets a point be rounded to the millimetre and stay on the
floor the validation checks, which is the free floor itself.

Heights are distances above the storey's floor, which is not assumed to be at
``y = 0``: every position is written ``floor_y_m + height``.

**Fixtures** are the objects a fixed source belongs to, for recipes of
version 2: a television, a washing machine, a counter, a shower, a window.
They are what the export labels and no more (:data:`FIXTURE_CATEGORIES`),
each with the point in the air next to it that a source is put at. They are
kept beside the stations, in :attr:`Layout.fixtures`, and are no part of the
graph: nobody walks to one, and the first generator never sees them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from reverberate.scenes.kinematics import polyline_length, yaw_of_direction
from reverberate.scenes.recipe import Dwelling, Heights, Rail, Station

__all__ = [
    "RAIL_MARGIN_M",
    "RAIL_PITCH_M",
    "Fixture",
    "Floor",
    "Layout",
    "LayoutSettings",
    "Navigator",
    "Room",
    "SeatObject",
    "build_layout",
    "load_hssd_floor",
    "load_hssd_layout",
]

#: Rails and the listener's walk keep this much inside the free floor.
RAIL_MARGIN_M = 0.03

#: The spacing of the positions the low band is solved at along a rail.
RAIL_PITCH_M = 0.08

#: Slack of the free floor tests, for numbers rounded to the millimetre.
FLOOR_TOLERANCE_M = 1e-6

#: A rail may leave the free floor over this length towards a seat, and a seat
#: lies within this of its object's footprint (rules 3 and 4 of the format).
SEAT_REACH_M = 0.60

#: A seat is at least this far from any wall (rule 3).
SEAT_WALL_M = 0.30

#: HSSD's categories of things one sits on, and how many seats each may give.
SEAT_CATEGORIES = {
    "seat": 1,
    "chair": 1,
    "armchair": 1,
    "stool": 1,
    "couch": 3,
    "sofa": 3,
    "bed": 2,
}


#: The categories of HSSD a fixed source may belong to. ``window`` and ``outer_door``
#: are not HSSD's: its openings carry no category, and are told apart by whether they
#: reach the floor.
FIXTURE_CATEGORIES = (
    "tv",
    "washing_machine_and_dryer",
    "kitchen_counter",
    "counter",
    "shower",
    "toilet",
    "piano",
    "desk",
    "table",
    "shelf",
    "nightstand",
    "window",
    "outer_door",
)

#: A fixed source stands this far in front of a panel, or above a piece of furniture.
FIXTURE_STANDOFF_M = 0.25
FIXTURE_ABOVE_M = 0.15


@dataclass(frozen=True)
class Fixture:
    """An object a fixed source belongs to, and where that source is put."""

    name: str
    category: str
    #: ``(x, y, z)`` of the emitting point, in the air next to the object.
    position: tuple[float, float, float]
    #: The way the object faces the room, a unit ``(x, z)``, or ``None``.
    front: tuple[float, float] | None = None


@dataclass(frozen=True)
class Room:
    """A room of ADR 0010: its name and its outline in ``(x, z)``."""

    name: str
    polygon: Any


@dataclass(frozen=True)
class SeatObject:
    """A piece of furniture one sits on."""

    name: str
    category: str
    footprint: Any
    #: The way the piece faces, a unit ``(x, z)``, or ``None`` when it has no front.
    front: tuple[float, float] | None = None


@dataclass(frozen=True)
class Floor:
    """What the format's geometric rules are checked against."""

    floor_y_m: float
    #: ``plan.free_floor``: the walkable outline less 0.25 m at the walls and
    #: the furniture grown by 0.22 m.
    free: Any
    #: The walkable outline itself; its boundary is the walls.
    walkable: Any
    rooms: tuple[Room, ...]
    #: Footprints of the seating, by instance name.
    objects: tuple[SeatObject, ...] = ()
    #: What a fixed source may belong to (version 2).
    fixtures: tuple[Fixture, ...] = ()

    def __post_init__(self) -> None:
        grown = self.free.buffer(FLOOR_TOLERANCE_M)
        shapely.prepare(grown)
        object.__setattr__(self, "_grown", grown)

    def on_free(self, xz: np.ndarray) -> np.ndarray:
        """Whether each ``(x, z)`` is on the free floor."""
        points = np.atleast_2d(np.asarray(xz, dtype=float))
        grown: Any = self._grown  # type: ignore[attr-defined]
        return np.asarray(shapely.covers(grown, shapely.points(points)), dtype=bool)

    def legs_on_free(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Whether each straight leg from ``start`` to ``end`` stays on the free floor."""
        a = np.atleast_2d(np.asarray(start, dtype=float))
        b = np.atleast_2d(np.asarray(end, dtype=float))
        grown: Any = self._grown  # type: ignore[attr-defined]
        lines = shapely.linestrings(np.stack([a, b], axis=1))
        return np.asarray(shapely.covers(grown, lines), dtype=bool)

    def room_of(self, x: float, z: float) -> str | None:
        here = Point(x, z)
        for room in self.rooms:
            if room.polygon.buffer(FLOOR_TOLERANCE_M).covers(here):
                return room.name
        return None

    def wall_distance(self, x: float, z: float) -> float:
        return float(self.walkable.boundary.distance(Point(x, z)))

    def footprint(self, name: str) -> Any | None:
        for item in self.objects:
            if item.name == name:
                return item.footprint
        return None


@dataclass(frozen=True)
class LayoutSettings:
    """The few lengths a layout is built with. The defaults are the project's."""

    heights: Heights = Heights()
    #: The listening lattice's pitch; standing spots are picked on it.
    lattice_m: float = 0.40
    #: Least distance between two standing spots.
    stand_spacing_m: float = 1.0
    #: The pitch of the lattice routes are searched on.
    route_pitch_m: float = 0.10
    #: A rail is added while the graph's detour between its ends exceeds this.
    stretch: float = 1.3
    #: No rail is longer.
    max_rail_m: float = 12.0
    #: Rails from each seat to the spots next to it.
    seat_links: int = 2


@dataclass
class Layout:
    """The stations and rails of a storey, and the floor they were laid on."""

    dwelling: Dwelling
    floor: Floor
    heights: Heights
    stations: tuple[Station, ...]
    rails: tuple[Rail, ...]
    #: For every seat, the point of the free floor one steps from to sit down.
    access: dict[str, tuple[float, float]]
    settings: LayoutSettings = field(default_factory=LayoutSettings)
    #: A station of kind ``fixture`` for each of the floor's fixtures, for recipes of
    #: version 2. Not among :attr:`stations`: no rail reaches one.
    fixtures: tuple[Station, ...] = ()

    @property
    def walk_area(self) -> Any:
        """Where rails and the listener go: the free floor less the margin."""
        return self.floor.free.buffer(-RAIL_MARGIN_M)

    def station(self, ident: str) -> Station:
        for station in self.stations:
            if station.id == ident:
                return station
        raise KeyError(ident)

    def summary(self) -> str:
        kinds = [station.kind for station in self.stations]
        metres = sum(polyline_length(np.asarray(rail.points)) for rail in self.rails)
        return (
            f"{self.dwelling.name}: {kinds.count('seat')} seats, {kinds.count('stand')} standing "
            f"spots, {kinds.count('waypoint')} doors, {len(self.rails)} rails, {metres:.1f} m "
            f"of rail, free floor {self.floor.free.area:.1f} m2"
        )


# --------------------------------------------------------------------------
# routes over the free floor
# --------------------------------------------------------------------------


class Navigator:
    """Shortest ways across an area, on a fine lattice, then pulled straight."""

    def __init__(self, area: Any, pitch_m: float = 0.10) -> None:
        self.area = area
        shapely.prepare(self.area)
        min_x, min_z, max_x, max_z = area.bounds
        xs = np.arange(min_x, max_x + 1e-9, pitch_m)
        zs = np.arange(min_z, max_z + 1e-9, pitch_m)
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
        inside = np.asarray(shapely.contains_xy(area, grid_x, grid_z), dtype=bool)
        number = np.full(inside.shape, -1, dtype=np.int64)
        number[inside] = np.arange(int(inside.sum()))
        self.nodes = np.stack([grid_x[inside], grid_z[inside]], axis=-1)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        costs: list[np.ndarray] = []
        nx, nz = inside.shape
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            first = number[: nx - di, max(0, -dj) : nz - max(0, dj)]
            second = number[di:, max(0, dj) : nz - max(0, -dj)]
            both = (first >= 0) & (second >= 0)
            a, b = first[both], second[both]
            legs = shapely.linestrings(np.stack([self.nodes[a], self.nodes[b]], axis=1))
            keep = np.asarray(shapely.covers(self.area, legs), dtype=bool)
            rows.append(a[keep])
            cols.append(b[keep])
            costs.append(np.full(int(keep.sum()), pitch_m * math.hypot(di, dj)))
        count = len(self.nodes)
        self.graph = coo_matrix(
            (np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
            shape=(count, count),
        ).tocsr()
        _, labels = connected_components(self.graph, directed=False)
        sizes = np.bincount(labels) if count else np.zeros(0, dtype=np.int64)
        #: Nodes of the largest connected part; the rest cannot be reached.
        self.reachable = labels == int(np.argmax(sizes)) if count else np.zeros(0, dtype=bool)

    def node_near(self, xz: Sequence[float]) -> int | None:
        """The nearest reachable node seen from ``xz`` in a straight line."""
        point = np.asarray(xz, dtype=float)
        distance = np.linalg.norm(self.nodes - point, axis=1)
        distance[~self.reachable] = np.inf
        for index in np.argsort(distance, kind="stable")[:12]:
            if not np.isfinite(distance[index]):
                break
            if distance[index] < 1e-9 or self.area.covers(LineString([point, self.nodes[index]])):
                return int(index)
        return None

    def distances(self, node: int) -> np.ndarray:
        return np.asarray(dijkstra(self.graph, directed=False, indices=node), dtype=float)

    def path(self, start: Sequence[float], end: Sequence[float]) -> np.ndarray | None:
        """A polyline from ``start`` to ``end`` inside the area, or ``None``."""
        first, last = self.node_near(start), self.node_near(end)
        if first is None or last is None:
            return None
        _, before = dijkstra(self.graph, directed=False, indices=first, return_predecessors=True)
        chain = [last]
        while chain[-1] != first:
            previous = int(before[chain[-1]])
            if previous < 0:
                return None
            chain.append(previous)
        points = [np.asarray(start, dtype=float), *self.nodes[chain[::-1]]]
        points.append(np.asarray(end, dtype=float))
        return self.pull(np.asarray(points))

    def pull(self, points: np.ndarray) -> np.ndarray:
        """Drop every point a straight leg inside the area can skip."""
        kept = [points[0]]
        index = 0
        while index < len(points) - 1:
            ahead = points[index + 1 : index + 401]
            legs = shapely.linestrings(
                np.stack([np.broadcast_to(points[index], ahead.shape), ahead], axis=1)
            )
            seen = np.flatnonzero(np.asarray(shapely.covers(self.area, legs), dtype=bool))
            index += int(seen[-1]) + 1 if seen.size else 1
            kept.append(points[index])
        out = np.asarray(kept)
        moved = np.concatenate([[True], np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9])
        pulled: np.ndarray = out[moved]
        return pulled


# --------------------------------------------------------------------------
# stations
# --------------------------------------------------------------------------


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _round(xz: Sequence[float]) -> tuple[float, float]:
    return (round(float(xz[0]), 3) + 0.0, round(float(xz[1]), 3) + 0.0)


@dataclass(frozen=True)
class _Seat:
    ident: str
    owner: SeatObject
    xz: tuple[float, float]
    access: tuple[float, float]
    yaw: float
    room: str


def _seats_of(item: SeatObject, floor: Floor, walk: Any) -> list[tuple[float, _Seat]]:
    """Candidate seats on one piece, best first: where one sits and steps from."""
    limit = SEAT_CATEGORIES.get(item.category, 0)
    if limit == 0 or item.footprint.is_empty:
        return []
    reach = SEAT_REACH_M - 0.02
    centre = item.footprint.centroid
    # One steps from a centimetre inside the walk area, not from its edge,
    # which a rounding to the millimetre would cross.
    walk = walk.buffer(-0.01)
    if walk.is_empty:
        return []
    if limit == 1:
        # One sits in the middle and steps from the front where the floor is
        # free there, from the nearest free floor otherwise.
        ahead = [0.6, 0.45, 0.3] if item.front is not None else []
        front = item.front or (0.0, 0.0)
        aims = [Point(centre.x + front[0] * d, centre.y + front[1] * d) for d in ahead]
        steps = [nearest_points(walk, aim)[0] for aim in [*aims, centre]]
        near = [step for step in steps if step.distance(centre) <= reach]
        pairs = [(centre, near[0] if near else steps[-1])]
    else:
        inner = item.footprint.buffer(-0.30)
        ring = (inner if not inner.is_empty else centre.buffer(0.01)).exterior
        spots = [ring.interpolate(s) for s in np.arange(0.0, ring.length, 0.05)]
        pairs = [(spot, nearest_points(walk, spot)[0]) for spot in spots]
    found: list[tuple[float, _Seat]] = []
    for spot, step in pairs:
        gap = step.distance(spot)
        if gap < 1e-6:
            continue
        # Sit no further from the floor one steps from than a rail may leave it.
        pull = min(1.0, reach / gap)
        seat = _round((step.x + (spot.x - step.x) * pull, step.y + (spot.y - step.y) * pull))
        access = _round((step.x, step.y))
        outward = np.array([access[0] - seat[0], access[1] - seat[1]])
        length = float(np.linalg.norm(outward))
        if length < 0.05 or length > SEAT_REACH_M - 0.005:
            continue
        if floor.wall_distance(*seat) < SEAT_WALL_M + 0.005:
            continue
        room = floor.room_of(*access)
        if room is None or not floor.on_free(np.array(access))[0]:
            continue
        facing = np.asarray(item.front) if item.front is not None else outward / length
        score = float(np.dot(outward / length, facing))
        # A couch is sat on from its front; a bed or a stool from any side.
        if limit > 1 and item.front is not None and item.category != "bed" and score < 0.5:
            continue
        yaw = round(float(yaw_of_direction(facing[0], facing[1])), 2) + 0.0
        found.append((score, _Seat("", item, seat, access, yaw, room)))
    found.sort(key=lambda entry: (-round(entry[0], 3), entry[1].xz))
    return found


def _seats(floor: Floor, walk: Any) -> list[_Seat]:
    seats: list[_Seat] = []
    for item in sorted(floor.objects, key=lambda entry: entry.name):
        limit = SEAT_CATEGORIES.get(item.category, 0)
        chosen: list[_Seat] = []
        for _, seat in _seats_of(item, floor, walk):
            if len(chosen) >= limit:
                break
            apart = all(math.dist(seat.xz, other.xz) >= 0.65 for other in [*seats, *chosen])
            if apart:
                chosen.append(seat)
        chosen.sort(key=lambda seat: seat.xz)
        for index, seat in enumerate(chosen):
            suffix = "" if len(chosen) == 1 else "_" + "abcdefgh"[index]
            seats.append(
                _Seat(_slug(item.name) + suffix, item, seat.xz, seat.access, seat.yaw, seat.room)
            )
    return seats


def _doors(floor: Floor, walk: Any, navigator: Navigator) -> list[tuple[float, float]]:
    """A point of the walk area in every passage between two rooms."""
    doors: list[tuple[float, float]] = []
    for index, room in enumerate(floor.rooms):
        for other in floor.rooms[index + 1 :]:
            gap = walk.intersection(room.polygon.buffer(0.35)).intersection(
                other.polygon.buffer(0.35)
            )
            parts = list(gap.geoms) if hasattr(gap, "geoms") else [gap]
            for part in parts:
                if part.is_empty or part.area < 1e-4:
                    continue
                # The node of the passage nearest its middle that a room holds:
                # a waypoint names its room, and the sill is in none.
                middle = np.array([part.centroid.x, part.centroid.y])
                order = np.argsort(np.linalg.norm(navigator.nodes - middle, axis=1), kind="stable")
                here = None
                for node in order[:200]:
                    spot = _round(navigator.nodes[node])
                    if not navigator.reachable[node] or not part.buffer(0.3).covers(Point(spot)):
                        continue
                    if floor.room_of(*spot) is not None:
                        here = spot
                        break
                if here is None:
                    continue
                if all(math.dist(here, door) >= 0.5 for door in doors):
                    doors.append(here)
    return sorted(doors)


def _stands(
    floor: Floor,
    walk: Any,
    navigator: Navigator,
    settings: LayoutSettings,
    taken: list[tuple[tuple[float, float], float]],
) -> list[tuple[tuple[float, float], str]]:
    """Standing spots on the listening lattice, the clearest first, kept apart."""
    from reverberate.experiments.w40_volume_field.plan import grid_points

    points, labels = grid_points(walk, list(floor.rooms), pitch_m=settings.lattice_m, height_m=0.0)
    if not len(points):
        return []
    xz = points[:, [0, 2]]
    clear = np.asarray(shapely.distance(walk.boundary, shapely.points(xz)), dtype=float)
    order = sorted(
        range(len(xz)),
        key=lambda i: (-round(min(float(clear[i]), 0.6), 2), float(xz[i, 0]), float(xz[i, 1])),
    )
    chosen: list[tuple[tuple[float, float], str]] = []
    for index in order:
        here = _round(xz[index])
        if any(math.dist(here, other) < apart for other, apart in taken):
            continue
        if any(math.dist(here, other) < settings.stand_spacing_m for other, _ in chosen):
            continue
        node = navigator.node_near(here)
        if node is None or not floor.on_free(np.array(here))[0]:
            continue
        chosen.append((here, labels[index]))
    return sorted(chosen)


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------


def build_layout(
    dwelling: Dwelling, floor: Floor, settings: LayoutSettings | None = None
) -> Layout:
    """Stations and rails for a storey. Deterministic: nothing is drawn."""
    settings = settings or LayoutSettings()
    heights = settings.heights
    walk = floor.free.buffer(-RAIL_MARGIN_M)
    if walk.is_empty:
        raise ValueError("no free floor is left once the rails' margin is taken")
    navigator = Navigator(walk, settings.route_pitch_m)
    standing_y = round(dwelling.floor_y_m + heights.standing_m, 3)
    seated_y = round(dwelling.floor_y_m + heights.seated_m, 3)

    seats = [seat for seat in _seats(floor, walk) if navigator.node_near(seat.access) is not None]
    doors = _doors(floor, walk, navigator)
    taken = [(door, 0.6) for door in doors]
    taken += [(seat.access, 0.5) for seat in seats] + [(seat.xz, 0.6) for seat in seats]
    stands = _stands(floor, walk, navigator, settings, taken)

    stations: list[Station] = []
    for index, (xz, room) in enumerate(stands, start=1):
        stations.append(
            Station(
                f"stand_{index:03d}", "stand", (xz[0], standing_y, xz[1]), "standing", room, 0.0
            )
        )
    for index, xz in enumerate(doors, start=1):
        room = floor.room_of(*xz) or ""
        stations.append(
            Station(
                f"door_{index:02d}", "waypoint", (xz[0], standing_y, xz[1]), "standing", room, 0.0
            )
        )
    hubs = list(stations)
    if not hubs:
        raise ValueError("the free floor holds no standing spot")

    # Distances over the floor between hubs, and a greedy spanner over them:
    # the shortest pairs first, a rail wherever the graph so far is a detour.
    nodes = np.array([navigator.node_near(hub.xz) for hub in hubs], dtype=np.int64)
    over_floor = np.array([navigator.distances(int(node))[nodes] for node in nodes])
    count = len(hubs)
    through = np.full((count, count), np.inf)
    np.fill_diagonal(through, 0.0)
    pairs = sorted(
        (float(over_floor[i, j]), i, j)
        for i in range(count)
        for j in range(i + 1, count)
        if over_floor[i, j] <= settings.max_rail_m
    )
    rails: list[Rail] = []
    for length, i, j in pairs:
        if through[i, j] <= settings.stretch * length:
            continue
        rail = _rail(navigator, floor, hubs[i], hubs[j], None, settings)
        if rail is None:
            continue
        rails.append(rail)
        via = np.minimum(
            through[:, [i]] + length + through[[j], :], through[:, [j]] + length + through[[i], :]
        )
        through = np.minimum(through, via)

    access: dict[str, tuple[float, float]] = {}
    for seat in seats:
        node = navigator.node_near(seat.access)
        assert node is not None
        reach = navigator.distances(node)[nodes]
        station = Station(
            seat.ident,
            "seat",
            (seat.xz[0], seated_y, seat.xz[1]),
            "seated",
            seat.room,
            seat.yaw,
            seat.owner.name,
        )
        linked = 0
        for index in np.argsort(reach, kind="stable"):
            if linked >= settings.seat_links or not np.isfinite(reach[index]):
                break
            if linked and reach[index] > 3.0:
                break
            rail = _rail(navigator, floor, hubs[int(index)], station, seat.access, settings)
            if rail is not None:
                rails.append(rail)
                linked += 1
        if linked:
            stations.append(station)
            access[station.id] = seat.access

    return Layout(
        dwelling=dwelling,
        floor=floor,
        heights=heights,
        stations=tuple(stations),
        rails=tuple(sorted(rails, key=lambda rail: rail.id)),
        access=access,
        settings=settings,
        fixtures=_fixture_stations(floor),
    )


def _fixture_stations(floor: Floor) -> tuple[Station, ...]:
    """A station for each fixture: where its source stands, and the way it faces."""
    out = []
    for item in sorted(floor.fixtures, key=lambda entry: entry.name):
        x, z = _round((item.position[0], item.position[2]))
        room = floor.room_of(x, z)
        if room is None and floor.rooms:
            here = Point(x, z)
            room = min(floor.rooms, key=lambda r: (float(r.polygon.distance(here)), r.name)).name
        yaw = 0.0
        if item.front is not None:
            yaw = round(float(yaw_of_direction(item.front[0], item.front[1])), 2) + 0.0
        out.append(
            Station(
                _slug(item.name),
                "fixture",
                (x, round(float(item.position[1]), 3) + 0.0, z),
                "fixed",
                room or "",
                yaw,
                item.name,
            )
        )
    return tuple(out)


def _rail(
    navigator: Navigator,
    floor: Floor,
    start: Station,
    end: Station,
    access: tuple[float, float] | None,
    settings: LayoutSettings,
) -> Rail | None:
    """The rail from a hub to another, or to a seat through the point one steps from."""
    target = access if access is not None else end.xz
    path = navigator.path(start.xz, target)
    if path is None:
        return None
    points = [_round(point) for point in path]
    if access is not None:
        points.append(end.xz)
    points = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
    if len(points) < 2:
        return None
    on_floor = points[:-1] if access is not None else points
    legs = np.asarray(on_floor)
    if len(legs) > 1 and not floor.legs_on_free(legs[:-1], legs[1:]).all():
        return None
    if polyline_length(np.asarray(points)) > settings.max_rail_m:
        return None
    return Rail(f"{start.id}__{end.id}", start.id, end.id, tuple(points), RAIL_PITCH_M)


# --------------------------------------------------------------------------
# HSSD: the one place that reads files
# --------------------------------------------------------------------------


def load_hssd_floor(hssd_root: Path, scene_id: str) -> Floor:
    """The floor of a scene's main storey, read from the HSSD download."""
    from reverberate.experiments.w40_volume_field.plan import free_floor
    from reverberate.geometry.apartment import build_apartment, instances_on_storey
    from reverberate.geometry.hssd_assets import category_for_template
    from reverberate.geometry.hssd_room import load_object_instances
    from reverberate.geometry.placement import footprint_of
    from reverberate.geometry.sim_geometry import obstacle_collider

    storey, free, rooms = free_floor(hssd_root, scene_id)
    storeys = build_apartment(hssd_root, scene_id)
    instances = instances_on_storey(
        load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json"),
        storeys[0],
        storeys,
    )
    objects = []
    for index, instance in enumerate(instances):
        category = category_for_template(hssd_root, instance.template_name) or "unknown"
        if category not in SEAT_CATEGORIES:
            continue
        base = obstacle_collider(hssd_root, instance.template_name)
        if base is None:
            continue
        mesh = base.copy()
        matrix = instance.transform_matrix()
        mesh.apply_transform(matrix)
        footprint = footprint_of(mesh)
        if footprint.is_empty:
            continue
        # HSSD's pieces face their own +z; the instance's matrix turns it.
        front = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
        norm = math.hypot(front[0], front[2])
        objects.append(
            SeatObject(
                name=f"{category}_{index}",
                category=category,
                footprint=footprint,
                front=(float(front[0] / norm), float(front[2] / norm)) if norm > 1e-6 else None,
            )
        )
    fixtures = []
    ceiling = float(storey.ceiling_height)
    for index, instance in enumerate(instances):
        found_category = category_for_template(hssd_root, instance.template_name)
        category = found_category or "unknown"
        bounds = None
        if found_category is None:
            bounds = _opening_bounds(hssd_root, instance)
            if bounds is not None:
                # A door reaches the floor; a window does not.
                low = float(bounds[0][1]) - float(storey.floor_height)
                category = "window" if low > 0.3 else "outer_door"
        if category not in FIXTURE_CATEGORIES:
            continue
        if bounds is None:
            base = obstacle_collider(hssd_root, instance.template_name)
            if base is None:
                continue
            mesh = base.copy()
            mesh.apply_transform(instance.transform_matrix())
            bounds = np.asarray(mesh.bounds, dtype=float)
        found = _fixture_point(category, bounds, storey.walkable, ceiling)
        if found is not None:
            fixtures.append(Fixture(f"{category}_{index}", category, found[0], found[1]))
    return Floor(
        floor_y_m=float(storey.floor_height),
        free=free,
        walkable=storey.walkable,
        rooms=tuple(Room(room.name, room.polygon) for room in rooms),
        objects=tuple(objects),
        fixtures=tuple(fixtures),
    )


def _opening_bounds(hssd_root: Path, instance: Any) -> np.ndarray | None:
    """The box of a door or a window in the scene's frame, or ``None`` for anything else."""
    import trimesh

    from reverberate.geometry.hssd_assets import resolve_asset

    asset = resolve_asset(hssd_root / "objects", instance.template_name)
    glb = hssd_root / "objects" / "openings" / f"{instance.template_name}.glb"
    if asset is None or not glb.exists():
        return None
    mesh = trimesh.load(glb, force="mesh")
    mesh.apply_transform(instance.transform_matrix())
    return np.asarray(mesh.bounds, dtype=float)


def _fixture_point(
    category: str, bounds: np.ndarray, walkable: Any, ceiling: float
) -> tuple[tuple[float, float, float], tuple[float, float] | None] | None:
    """Where the source of an object stands, and the way it faces: in the air next to it.

    A panel, a television or an opening, has its source a little in front
    of it on the side that is inside the dwelling and further from a wall;
    anything else has it a little above its top.
    """
    low, high = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    centre = 0.5 * (low + high)
    if category in ("tv", "window", "outer_door"):
        thin = 0 if high[0] - low[0] < high[2] - low[2] else 2
        reach = 0.5 * float(high[thin] - low[thin]) + FIXTURE_STANDOFF_M
        best = None
        for side in (1.0, -1.0):
            at = centre.copy()
            at[thin] += side * reach
            here = Point(float(at[0]), float(at[2]))
            if not walkable.covers(here):
                continue
            clear = float(walkable.boundary.distance(here))
            if best is None or clear > best[0]:
                front = (side, 0.0) if thin == 0 else (0.0, side)
                best = (clear, (float(at[0]), float(at[1]), float(at[2])), front)
        return None if best is None else (best[1], best[2])
    top = min(float(high[1]) + FIXTURE_ABOVE_M, ceiling - 0.3)
    return (float(centre[0]), top, float(centre[2])), None


def load_hssd_layout(
    hssd_root: Path, dwelling: str, settings: LayoutSettings | None = None
) -> Layout:
    """The layout of a dwelling named as this project names it, ``hssd_0076``."""
    from reverberate.geometry.scene_ids import scene_of

    scene_id, storey = scene_of(dwelling)
    if storey not in (None, 1):
        raise ValueError(f"{dwelling}: only a scene's main storey has a layout so far")
    floor = load_hssd_floor(hssd_root, scene_id)
    return build_layout(
        Dwelling(dwelling, scene_id, round(floor.floor_y_m, 3) + 0.0), floor, settings
    )
