"""Where the source and the listener stand, and what kind of source it is.

Two things this module exists to fix, both found by reading the previous
sampler rather than by guessing:

**Furniture was not excluded.** The walkable outline is architectural, a union
of room polygons and the doorways joining them, so nothing stopped a source
being sampled *inside* a sofa or a wardrobe. Acoustically that is a source
sealed inside an obstacle, and the Sabine/Eyring guard does not reliably catch
it. Here the obstacles' own footprints are subtracted from the sampling area,
so the defect is impossible by construction rather than filtered afterwards.

**Height was hardcoded at 1.2 m for both ends**, which is precisely the
variable worth modelling. A source is now drawn from an archetype carrying a
plausible height, directivity and orientation.

Orientation is sampled, not assumed: a cardioid source aimed at the listener
against one aimed away measured a 20 dB swing in direct-to-reverberant ratio
on an 8x3x5 m box. Leaving it unsampled would make it a hidden variable the
model cannot learn.

``REFERENCE_OMNI`` is deliberately the most frequent archetype. It is the
control: it isolates the effect of geometry from the effect of directivity, it
keeps the Sabine and Eyring baselines comparable (both assume a diffuse field,
hence an omnidirectional source), and it gives a clean subset on which to
check that the model has not merely learned to guess where a source points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reverberate.geometry.apartment import Storey
from reverberate.geometry.hssd_room import FurnitureInstance
from reverberate.geometry.sim_geometry import (
    MIN_WALL_DISTANCE,
    OBSTACLE_FACE_BUDGET,
    room_of,
    simulation_collider,
)

#: Ear height of a standing and of a seated listener, in metres.
STANDING_EAR_HEIGHT = 1.6
SEATED_EAR_HEIGHT = 1.2


@dataclass(frozen=True)
class SourceArchetype:
    """A kind of sound source, with the height and directivity that go with it.

    ``weight`` is a relative sampling frequency, not a probability: the weights
    are normalised when drawing.
    """

    name: str
    min_height: float
    max_height: float
    directive: bool
    weight: float

    def sample_height(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.min_height, self.max_height))


#: A person speaking, from seated to standing. Directional, and the most
#: legible case to listen to in the demo.
VOICE = SourceArchetype("voice", SEATED_EAR_HEIGHT, STANDING_EAR_HEIGHT, True, 1.0)

#: A loudspeaker on a stand or a shelf.
SPEAKER = SourceArchetype("speaker", 0.8, 1.2, True, 1.0)

#: A noisy appliance, from floor level to a worktop. Treated as omnidirectional
#: because its radiation is dominated by its cabinet rather than by an aimed
#: driver.
APPLIANCE = SourceArchetype("appliance", 0.3, 0.9, False, 1.0)

#: The control. Omnidirectional, fixed height, no orientation dependence, and
#: weighted to make up about half the dataset so that "the model captures the
#: geometry" cannot be confused with "the model captures the directivity".
REFERENCE_OMNI = SourceArchetype("reference_omni", 1.5, 1.5, False, 3.0)

SOURCE_ARCHETYPES: tuple[SourceArchetype, ...] = (VOICE, SPEAKER, APPLIANCE, REFERENCE_OMNI)


@dataclass(frozen=True)
class Placement:
    """One source or listener: where it stands, where it looks, what it is."""

    position: np.ndarray
    azimuth: float
    room: str
    archetype: SourceArchetype | None = None

    @property
    def directive(self) -> bool:
        return self.archetype is not None and self.archetype.directive

    def directivity(self) -> pra.directivities.Directivity | None:
        """The pyroomacoustics directivity for this placement, if any.

        ``None`` for an omnidirectional archetype, which is what
        ``add_source``/``add_microphone`` already expect for "no directivity",
        so callers need no special case.
        """
        if not self.directive:
            return None
        return pra.directivities.Cardioid(
            orientation=pra.directivities.DirectionVector(
                azimuth=self.azimuth, colatitude=90.0, degrees=True
            )
        )


@dataclass(frozen=True)
class PlacedPair:
    """A source and a listener, with the provenance needed to group the dataset."""

    source: Placement
    listener: Placement

    @property
    def same_room(self) -> bool:
        """Intra-room pairs; the others travel through a doorway."""
        return self.source.room == self.listener.room

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.source.position - self.listener.position))


def footprint_of(mesh: trimesh.Trimesh) -> Polygon:
    """The ground footprint of a placed obstacle, as a convex hull in XZ.

    Convex rather than exact on purpose: it can only ever *over*-estimate the
    space a piece of furniture occupies, so the error direction is "refuses a
    valid position" rather than "puts the source inside the sofa". An L-shaped
    sectional therefore blocks the corner it wraps around, which is a cost
    worth paying for a guarantee that holds without a containment test.
    """
    points = np.asarray(mesh.vertices)[:, [0, 2]]
    if len(points) < 3:
        return Polygon()
    hull = Polygon(points).convex_hull
    if isinstance(hull, Polygon) and not hull.is_empty:
        return hull
    return Polygon()


#: What a person can stand on rather than walk around, in metres above the
#: floor. A rug, a floor mat or a threshold strip is under this; a bed frame is
#: not. Measured need: in HSSD bedroom.001 a single rug's convex hull covered
#: 11.63 m2 of a 13.61 m2 room, so treating every placed object as an obstacle
#: left **no** free floor at all and the sampler could not place anything. The
#: rug is still simulated as a surface; it is simply not a thing you walk
#: around.
STEP_OVER_HEIGHT = 0.15

#: Full stature, taller than :data:`STANDING_EAR_HEIGHT` because a person's head
#: is above their ears. An object entirely above this, such as a ceiling lamp or
#: a high shelf, does not stop anyone standing under it.
STANDING_HEIGHT = 1.8


def occupies_standing_space(
    mesh: trimesh.Trimesh,
    floor_height: float,
    step_over: float = STEP_OVER_HEIGHT,
    stature: float = STANDING_HEIGHT,
) -> bool:
    """Whether a placed object actually stops a person standing there.

    The previous rule was "any placed object blocks", which is wrong in both
    directions and was measured to be wrong: it excluded rugs, which you stand
    on, and it would equally exclude the floor under a pendant lamp. An object
    blocks when it rises above what can be stepped onto *and* begins below the
    top of a standing person.
    """
    low, high = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])
    return high > floor_height + step_over and low < floor_height + stature


def furniture_footprints(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    face_budget: int = OBSTACLE_FACE_BUDGET,
    floor_height: float | None = None,
) -> list[Polygon]:
    """The XZ footprint of every obstacle, under its instance matrix.

    Uses ``simulation_collider``, the same mesh the simulator and the acoustic
    view read, so the space treated as occupied is the space that is actually
    simulated as occupied.

    Pass ``floor_height`` to keep only the objects that occupy standing space;
    leaving it ``None`` keeps every object, which is the older and blunter
    behaviour and is retained only for callers that want the raw footprints.
    """
    footprints = []
    for instance in instances:
        base = simulation_collider(hssd_root, instance.template_name, face_budget)
        if base is None:
            continue
        mesh = base.copy()
        mesh.apply_transform(instance.transform_matrix())
        if floor_height is not None and not occupies_standing_space(mesh, floor_height):
            continue
        polygon = footprint_of(mesh)
        if not polygon.is_empty:
            footprints.append(polygon)
    return footprints


def sampling_area(
    storey: Storey,
    footprints: list[Polygon] | None = None,
    min_wall_distance: float = MIN_WALL_DISTANCE,
    clearance: float = 0.0,
) -> Polygon | MultiPolygon:
    """The floor area a source or listener may stand on.

    The walkable outline, pulled in from the walls, minus the furniture. Taking
    the difference here rather than rejecting bad samples later means the
    guarantee is structural: there is no position in the returned area that is
    inside an obstacle.
    """
    area = storey.walkable.buffer(-min_wall_distance)
    if footprints:
        blocked = unary_union([polygon.buffer(clearance) for polygon in footprints])
        area = area.difference(blocked)
    if area.is_empty:
        raise ValueError("no free floor area left once walls and furniture are excluded")
    return area


def _sample_positions(
    area: Polygon | MultiPolygon,
    count: int,
    rng: np.random.Generator,
    attempts_per_point: int = 400,
) -> list[tuple[float, float]]:
    """Rejection sampling of XZ points inside an arbitrary area.

    The area is a polygon with holes once furniture is removed, so there is no
    closed form; the apartment is dense enough that rejection converges.
    """
    min_x, min_z, max_x, max_z = area.bounds
    points: list[tuple[float, float]] = []
    for _ in range(count * attempts_per_point):
        if len(points) == count:
            break
        x = float(rng.uniform(min_x, max_x))
        z = float(rng.uniform(min_z, max_z))
        if area.contains(Point(x, z)):
            points.append((x, z))
    if len(points) < count:
        raise ValueError(f"only placed {len(points)} of {count} points in the free area")
    return points


def choose_archetype(
    rng: np.random.Generator,
    archetypes: tuple[SourceArchetype, ...] = SOURCE_ARCHETYPES,
) -> SourceArchetype:
    weights = np.array([archetype.weight for archetype in archetypes], dtype=float)
    index = int(rng.choice(len(archetypes), p=weights / weights.sum()))
    return archetypes[index]


def sample_pair(
    storey: Storey,
    rng: np.random.Generator,
    area: Polygon | MultiPolygon | None = None,
    same_room: bool | None = None,
    archetype: SourceArchetype | None = None,
    min_wall_distance: float = MIN_WALL_DISTANCE,
    max_attempts: int = 200,
) -> PlacedPair:
    """A source and a listener, both standing somewhere they could really stand.

    ``area`` should be the result of ``sampling_area`` with the storey's
    furniture; it is accepted as an argument rather than rebuilt here because
    subtracting a few hundred footprints is the expensive part and callers
    sampling many pairs must pay it once. Passing ``None`` falls back to walls
    only, which is the old behaviour and is only appropriate for a storey with
    no furniture.

    ``same_room=None`` accepts whatever comes out, ``True`` keeps drawing until
    both land in the same annotated room and ``False`` until they land in
    different ones, which is how a response through a doorway gets sampled
    deliberately rather than by luck.
    """
    if area is None:
        area = sampling_area(storey, min_wall_distance=min_wall_distance)
    kind = archetype if archetype is not None else choose_archetype(rng)

    for _ in range(max_attempts):
        (source_x, source_z), (listener_x, listener_z) = _sample_positions(area, 2, rng)
        source = Placement(
            position=np.array([source_x, storey.floor_height + kind.sample_height(rng), source_z]),
            azimuth=float(rng.uniform(0.0, 360.0)),
            room=room_of(storey, source_x, source_z),
            archetype=kind,
        )
        standing = bool(rng.integers(2))
        listener = Placement(
            position=np.array(
                [
                    listener_x,
                    storey.floor_height + (STANDING_EAR_HEIGHT if standing else SEATED_EAR_HEIGHT),
                    listener_z,
                ]
            ),
            azimuth=float(rng.uniform(0.0, 360.0)),
            room=room_of(storey, listener_x, listener_z),
        )
        pair = PlacedPair(source=source, listener=listener)
        if same_room is None or pair.same_room == same_room:
            return pair
    raise ValueError(f"could not sample a pair with same_room={same_room} in this storey")


@dataclass(frozen=True)
class PlacedGroup:
    """Several sources and several receivers sharing one room, and one run.

    The asymmetry is a cost fact, not a modelling preference. The solver's cost
    is set by the grid and the number of time steps, so a receiver added to an
    existing run is nearly free while a second source is a second run. W20's
    2 x 6 is that arithmetic: twelve responses for the price of two.
    """

    room: str
    sources: tuple[Placement, ...]
    receivers: tuple[Placement, ...]
    seed: int

    @property
    def response_count(self) -> int:
        return len(self.sources) * len(self.receivers)

    def record(self) -> dict[str, object]:
        """The placement as it goes into the run record, positions and all."""
        return {
            "room": self.room,
            "seed": self.seed,
            "sources": [
                {
                    "position": [round(float(value), 4) for value in source.position],
                    "azimuth_deg": round(source.azimuth, 2),
                    "archetype": source.archetype.name if source.archetype else None,
                }
                for source in self.sources
            ],
            "receivers": [
                {
                    "position": [round(float(value), 4) for value in receiver.position],
                    "azimuth_deg": round(receiver.azimuth, 2),
                }
                for receiver in self.receivers
            ],
        }


def room_sampling_area(
    storey: Storey,
    room: str,
    area: Polygon | MultiPolygon,
) -> Polygon | MultiPolygon:
    """The part of ``area`` that lies inside one annotated room.

    Intersected with the free area rather than taken raw, so the wall setback
    and the furniture subtraction still hold inside the room.
    """
    for region in storey.rooms:
        if region.name == room:
            free = area.intersection(region.polygon_xz.buffer(0))
            if free.is_empty:
                raise ValueError(f"room {room!r} has no free floor left once furniture is removed")
            return free
    raise ValueError(f"no room named {room!r} on this storey")


def largest_free_room(storey: Storey, area: Polygon | MultiPolygon) -> str:
    """The room with the most free floor, which is where a group will fit.

    Free floor rather than gross area: a large bedroom whose floor is entirely
    taken by a bed is a worse candidate than a small empty one, and it is free
    floor that the sampler actually draws from.
    """
    best, best_area = "", 0.0
    for region in storey.rooms:
        free = area.intersection(region.polygon_xz.buffer(0)).area
        if free > best_area:
            best, best_area = region.name, float(free)
    if not best:
        raise ValueError("no room on this storey has any free floor")
    return best


def sample_group(
    storey: Storey,
    rng: np.random.Generator,
    area: Polygon | MultiPolygon,
    *,
    room: str | None = None,
    sources: int = 2,
    receivers: int = 6,
    min_separation: float = 0.5,
    seed: int = 0,
    max_attempts: int = 200,
) -> PlacedGroup:
    """Sources and receivers at head height, all in one room, all apart.

    ``min_separation`` keeps every pair of placements apart in the horizontal
    plane. Two reasons, and neither is aesthetic. Receivers are not grid
    constrained, so each is interpolated from the 8 nodes around it; two
    receivers closer than a cell would read very nearly the same nodes and
    return the same response twice while costing storage as if they were two
    measurements. And a receiver sitting on a source is not a measurement of a
    room at all. Half a metre is far larger than the 8.2 mm cell of a 4 kHz
    working point, so this is a statement about what is worth measuring, not
    about what the grid can resolve.

    Nothing here re-checks walls or furniture: ``area`` is already the free
    floor, so exclusion is structural, as ``sampling_area`` intends.
    """
    chosen = room if room is not None else largest_free_room(storey, area)
    free = room_sampling_area(storey, chosen, area)
    total = sources + receivers

    for _ in range(max_attempts):
        points = _sample_positions(free, total, rng)
        if _all_apart(points, min_separation):
            break
    else:
        raise ValueError(
            f"could not place {total} points at least {min_separation} m apart in {chosen!r}; "
            f"its free floor is {free.area:.2f} m2"
        )

    placed_sources = tuple(
        Placement(
            position=np.array([x, storey.floor_height + kind.sample_height(rng), z]),
            azimuth=float(rng.uniform(0.0, 360.0)),
            room=room_of(storey, x, z),
            archetype=kind,
        )
        for (x, z), kind in ((point, choose_archetype(rng)) for point in points[:sources])
    )
    placed_receivers = tuple(
        Placement(
            position=np.array(
                [
                    x,
                    storey.floor_height
                    + (STANDING_EAR_HEIGHT if bool(rng.integers(2)) else SEATED_EAR_HEIGHT),
                    z,
                ]
            ),
            azimuth=float(rng.uniform(0.0, 360.0)),
            room=room_of(storey, x, z),
        )
        for x, z in points[sources:]
    )
    return PlacedGroup(room=chosen, sources=placed_sources, receivers=placed_receivers, seed=seed)


def _all_apart(points: list[tuple[float, float]], minimum: float) -> bool:
    coordinates = np.asarray(points)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return bool(distances.min() >= minimum)
