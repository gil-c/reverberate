"""The geometry handed to pyroomacoustics, built from what the viewer shows.

The rule this module exists to enforce is "what you see is what is simulated".
There is deliberately no second reconstruction here: the shell is
``extrude_storey`` of the same walkable outline the viewer walks on, and each
obstacle is the same collider file, under the same instance matrix, as the
acoustic view draws. If the two ever disagree, it is a bug, and
``describe_geometry`` exists so that disagreement can be measured rather than
argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import trimesh
from shapely.geometry import Point

from reverberate.geometry.absorption import (
    AbsorptionAudit,
    CompensatedMaterial,
    audit,
    compensate,
)
from reverberate.geometry.apartment import (
    Storey,
    build_apartment,
    extrude_storey,
    instances_on_storey,
)
from reverberate.geometry.decimation import (
    DETAIL_LEVELS,
    DetailLevel,
    level_for,
)
from reverberate.geometry.envelope import acoustic_envelope
from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import FurnitureInstance, load_object_instances
from reverberate.geometry.materials import material_for_label
from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.viz.room_surfaces import shell_surface_labels


@dataclass
class GeometrySummary:
    """What the simulator is about to receive, in numbers a human can check."""

    shell_faces: int
    shell_volume: float
    shell_watertight: bool
    obstacle_count: int
    obstacle_faces: int
    unresolved: list[str]
    absorption: AbsorptionAudit | None = None

    @property
    def total_walls(self) -> int:
        """pyroomacoustics builds one wall per triangle, so this is the real cost."""
        return self.shell_faces + self.obstacle_faces

    def summary(self) -> str:
        absorption = f", {self.absorption.summary()}" if self.absorption is not None else ""
        return (
            f"shell {self.shell_faces} faces ({self.shell_volume:.0f} m3, "
            f"watertight={self.shell_watertight}), {self.obstacle_count} obstacles "
            f"totalling {self.obstacle_faces} faces, {self.total_walls} pra walls"
            f"{absorption}"
        )


#: Face budget per furniture obstacle. pyroomacoustics builds one wall per
#: triangle, and an undecimated apartment comes to over 400k of them, which is
#: not simulable. Decimation therefore happens *here*, in the single place both
#: the simulator and the acoustic view read from, so that reducing the cost
#: never turns the picture into a flattering version of the real input.
OBSTACLE_FACE_BUDGET = 150


def decimate(mesh: trimesh.Trimesh, face_budget: int) -> trimesh.Trimesh:
    """Reduce an obstacle to the face budget, keeping its overall shape."""
    if len(mesh.faces) <= face_budget:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=face_budget)
    except Exception:
        # A mesh the decimator cannot handle is passed through whole rather
        # than dropped: an expensive obstacle is better than a missing one.
        return mesh


@lru_cache(maxsize=512)
def simulation_collider(hssd_root: Path, template: str, face_budget: int) -> trimesh.Trimesh | None:
    """The mesh that both the simulator and the acoustic view use for a template.

    Cached because a room usually places the same template several times, and
    decimation is the expensive part.
    """
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return None
    mesh = trimesh.load(asset.collider, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return decimate(mesh, face_budget)


@lru_cache(maxsize=1024)
def reduced_collider(
    hssd_root: Path, template: str, detail_length: float
) -> tuple[trimesh.Trimesh, float] | None:
    """The decimated obstacle mesh and the surface area it had before.

    Both halves are needed together and both are expensive, so they are cached
    as a pair: the reduced mesh is what the simulator and the acoustic view
    draw, and the original area is what the absorption compensation divides by.
    Returning the area rather than the original mesh keeps the cache small.
    """
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return None
    mesh = trimesh.load(asset.collider, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    # The raw collider is a convex decomposition whose triangle area counts
    # faces buried between adjacent convex pieces, where sound never reaches:
    # 1316 m2 claimed against 698 m2 real on one apartment. ``acoustic_envelope``
    # returns the outer surface and its true area, and it refuses to approximate
    # an object whose shape the approximation would misrepresent.
    #
    # The gate is geometric deviation, not area: simulation showed that once an
    # envelope strays far from the real surface, rescaling absorption to match
    # does not recover the acoustics and can make them worse (RT60 -59% without
    # compensation, -87% with it, on an envelope 77 cm off). Compensation is a
    # small correction for a good approximation, never a licence for a bad one.
    envelope = acoustic_envelope(mesh, max_deviation=detail_length / 2.0)
    return envelope.mesh, reference_area(hssd_root, template)


@lru_cache(maxsize=1024)
def reference_area(hssd_root: Path, template: str) -> float:
    """The obstacle's true outer surface, measured once at the finest rung.

    This is what absorption is compensated *against*, and it deliberately does
    not depend on the rung being built. Comparing a coarse envelope with its own
    area yields a factor of 1 and no compensation at all, which is how a whole
    apartment silently lost half its absorbing power (370 m2 down to 188 m2)
    the first time levels of detail were switched on: the far rooms were
    coarsened, their surface went with them, and nothing put it back.
    """
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return 0.0
    mesh = trimesh.load(asset.collider, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return 0.0
    return float(acoustic_envelope(mesh).area)


def shell_assignments(storey: Storey, seed: int = 0) -> list[MeshMaterialAssignment]:
    """The apartment shell, split into floor, wall and ceiling materials.

    Split by face normal rather than given one material for the whole
    enclosure: carpet underfoot and plasterboard overhead sit at opposite ends
    of the absorption range, and averaging them away would flatten the signal
    the model is meant to learn.
    """
    rng = np.random.default_rng(seed)
    shell = extrude_storey(storey)
    labels = shell_surface_labels(shell)
    assignments = []
    for surface in ("floor", "wall", "ceiling"):
        selected = labels == surface
        if not selected.any():
            continue
        part = shell.submesh([np.flatnonzero(selected)], append=True)
        assert isinstance(part, trimesh.Trimesh)
        assignments.append(
            MeshMaterialAssignment(
                mesh=part,
                material=material_for_label(surface, rng),
                name=f"shell_{surface}",
            )
        )
    return assignments


def obstacle_assignments(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    seed: int = 0,
    listener: np.ndarray | None = None,
    level: DetailLevel | None = None,
    storey: Storey | None = None,
) -> tuple[list[MeshMaterialAssignment], list[str], AbsorptionAudit]:
    """Every piece of furniture, as its collider under its instance matrix.

    ``resolve_asset`` falls back to the render mesh when an object ships no
    ``.collider.glb``, which is HSSD's own rule and is what keeps doors and
    windows in the simulation instead of silently dropping them.

    Each obstacle is decimated to the detail its distance from ``listener``
    justifies, then has its absorption rescaled so that decimating it does not
    also delete the absorption it was supposed to provide. Pass ``level`` to
    force one rung for every obstacle, which is what a caller wanting a
    listener-independent geometry should do. Passing neither uses the finest
    rung, so the default is the most conservative option rather than the
    cheapest.

    The returned :class:`AbsorptionAudit` is the check on the whole scheme:
    absorbing power in and out should agree, and any gap is capping.
    """
    rng = np.random.default_rng(seed)
    assignments: list[MeshMaterialAssignment] = []
    unresolved: list[str] = []
    entries: list[CompensatedMaterial] = []
    base_materials: list[pra.Material] = []

    for index, instance in enumerate(instances):
        category = category_for_template(hssd_root, instance.template_name) or "unknown"
        chosen = level if level is not None else _level_for_instance(instance, listener, storey)
        loaded = reduced_collider(hssd_root, instance.template_name, chosen.detail_length)
        if loaded is None:
            unresolved.append(instance.template_name)
            continue
        base, original_area = loaded
        mesh = base.copy()
        mesh.apply_transform(instance.transform_matrix())

        material = material_for_label(category, rng)
        # Areas are compared after the instance matrix so that a non-uniform
        # scale is reflected in both, rather than compensating for a scaling
        # that never happened.
        scale = _area_scale(base, mesh)
        entry = compensate(
            material,
            original_area=original_area * scale,
            reduced_area=float(mesh.area),
            base_key=category,
        )
        entries.append(entry)
        base_materials.append(material)
        assignments.append(
            MeshMaterialAssignment(
                mesh=mesh,
                material=entry.material,
                name=f"{category}_{index}",
                compensation=entry,
            )
        )
    return assignments, unresolved, audit(entries, base_materials)


def _area_scale(base: trimesh.Trimesh, placed: trimesh.Trimesh) -> float:
    """How much the instance matrix changed the mesh's area."""
    base_area = float(base.area)
    if base_area <= 0:
        return 1.0
    return float(placed.area) / base_area


def _level_for_instance(
    instance: FurnitureInstance,
    listener: np.ndarray | None,
    storey: Storey | None = None,
) -> DetailLevel:
    """The detail rung this obstacle sits on for a given listener position.

    Room membership is looked up rather than assumed. An earlier version passed
    ``same_room=True`` unconditionally, which meant the coarsest rung was never
    reached and every obstacle in the flat was simulated at the resolution
    reserved for the ones next to the listener.
    """
    if listener is None:
        return DETAIL_LEVELS[0]
    position = np.asarray(instance.translation, dtype=float)
    distance = float(np.linalg.norm(position - np.asarray(listener, dtype=float)))
    same_room = True
    if storey is not None:
        same_room = room_of(storey, float(position[0]), float(position[2])) == room_of(
            storey, float(listener[0]), float(listener[2])
        )
    return level_for(distance, same_room=same_room)


def simulation_geometry(
    hssd_root: Path,
    storey: Storey,
    instances: list[FurnitureInstance],
    seed: int = 0,
    storeys: list[Storey] | None = None,
    listener: np.ndarray | None = None,
    level: DetailLevel | None = None,
) -> tuple[list[MeshMaterialAssignment], GeometrySummary]:
    """Everything pyroomacoustics receives for one apartment storey.

    ``instances`` is filtered to this storey here rather than trusting the
    caller to have remembered: on a multi-storey scene, passing a whole
    scene's furniture would otherwise simulate the floor above as well.
    Filtering is idempotent, so pre-filtered input is safe. Pass ``storeys``
    when the scene has more than one, so pieces in the overlap band between a
    ceiling and the floor above are assigned to exactly one of them.

    ``listener`` selects each obstacle's level of detail by distance; ``level``
    forces one rung for all of them. The geometry therefore depends on where
    the listener stands, which is why the viewer must be shown the mesh for the
    pair being simulated rather than for wherever its camera happens to be.
    """
    instances = instances_on_storey(instances, storey, storeys)
    shell = shell_assignments(storey, seed=seed)
    obstacles, unresolved, absorption = obstacle_assignments(
        hssd_root, instances, seed=seed, listener=listener, level=level, storey=storey
    )
    whole_shell = extrude_storey(storey)
    summary = GeometrySummary(
        shell_faces=sum(len(assignment.mesh.faces) for assignment in shell),
        shell_volume=float(whole_shell.volume),
        shell_watertight=bool(whole_shell.is_watertight),
        obstacle_count=len(obstacles),
        obstacle_faces=sum(len(assignment.mesh.faces) for assignment in obstacles),
        unresolved=unresolved,
        absorption=absorption,
    )
    return [*shell, *obstacles], summary


def apartment_geometry(
    hssd_root: Path,
    scene_id: str,
    storey_index: int = 0,
    seed: int = 0,
    include_outdoor: bool = False,
    listener: np.ndarray | None = None,
    level: DetailLevel | None = None,
) -> tuple[list[MeshMaterialAssignment], GeometrySummary, Storey]:
    """One call from a scene id to everything the simulator needs.

    This is the entry point for the acoustic pipeline: it assembles the
    apartment, keeps the furniture standing on the chosen storey, and returns
    the mesh-plus-material list ``pra_room.build_room`` already expects.
    Storeys come largest first, so the default is the main floor.

    ``listener`` picks each obstacle's level of detail by distance; ``level``
    forces one rung throughout. The returned assignments are the *same* meshes
    the viewer draws in its acoustic mode, which is the property worth
    preserving: if the simulation and the picture ever disagree, one of them
    stopped calling this function.
    """
    storeys = build_apartment(hssd_root, scene_id, include_outdoor=include_outdoor)
    if not storeys:
        raise ValueError(f"scene {scene_id} has no walkable storey")
    storey = storeys[storey_index]
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    assignments, summary = simulation_geometry(
        hssd_root, storey, instances, seed=seed, storeys=storeys, listener=listener, level=level
    )
    return assignments, summary, storey


def build_pra_room(
    assignments: list[MeshMaterialAssignment], fs: int = 16000, max_order: int = 1
) -> pra.Room:
    """Hand the geometry to pyroomacoustics unchanged.

    ``max_order`` defaults low on purpose: with furniture colliders included,
    the image source model becomes impractical well before it becomes
    inaccurate (see the roadmap's note on max_order 3).
    """
    from reverberate.geometry.pra_room import build_room

    return build_room(assignments, fs=fs, max_order=max_order)


#: How far a source or receiver is kept from any wall, in metres. Close to a
#: surface the image source model produces very early, very strong reflections
#: that dominate the response and are not what a listener in the room hears.
MIN_WALL_DISTANCE = 0.5


@dataclass(frozen=True)
class SourceReceiver:
    """One source and one receiver placed in the apartment, with provenance."""

    source: np.ndarray
    receiver: np.ndarray
    source_room: str
    receiver_room: str

    @property
    def same_room(self) -> bool:
        """Whether the pair is intra-room; inter-room pairs travel through doorways."""
        return self.source_room == self.receiver_room


def room_of(storey: Storey, x: float, z: float) -> str:
    """Which annotated room a point falls in, or ``"doorway"`` between them."""
    for region in storey.rooms:
        if region.polygon_xz.buffer(0).contains(Point(x, z)):
            return region.name
    return "doorway"


def sample_points(
    storey: Storey,
    count: int,
    rng: np.random.Generator,
    height: float = 1.2,
    min_wall_distance: float = MIN_WALL_DISTANCE,
) -> list[np.ndarray]:
    """Points inside the walkable area, kept clear of the walls.

    Rejection sampling over the outline's bounding box: the walkable area is
    an arbitrary polygon with holes, so there is no closed form, and the
    apartment is dense enough that rejection converges quickly.
    """
    interior = storey.walkable.buffer(-min_wall_distance)
    if interior.is_empty:
        raise ValueError("no point in this storey is far enough from every wall")
    min_x, min_z, max_x, max_z = interior.bounds
    points: list[np.ndarray] = []
    for _ in range(count * 400):
        if len(points) == count:
            break
        x = rng.uniform(min_x, max_x)
        z = rng.uniform(min_z, max_z)
        if interior.contains(Point(x, z)):
            points.append(np.array([x, storey.floor_height + height, z]))
    if len(points) < count:
        raise ValueError(f"only placed {len(points)} of {count} points in this storey")
    return points


def sample_source_receiver(
    storey: Storey,
    rng: np.random.Generator,
    same_room: bool | None = None,
    min_wall_distance: float = MIN_WALL_DISTANCE,
) -> SourceReceiver:
    """A source/receiver pair, optionally constrained to one room or to two.

    ``same_room=None`` accepts whatever comes out, ``True`` keeps drawing until
    both land in the same annotated room, and ``False`` until they land in
    different ones, which is how an inter-room response through a doorway gets
    sampled deliberately rather than by luck.
    """
    for _ in range(200):
        source, receiver = sample_points(storey, 2, rng, min_wall_distance=min_wall_distance)
        pair = SourceReceiver(
            source=source,
            receiver=receiver,
            source_room=room_of(storey, float(source[0]), float(source[2])),
            receiver_room=room_of(storey, float(receiver[0]), float(receiver[2])),
        )
        if same_room is None or pair.same_room == same_room:
            return pair
    raise ValueError(f"could not sample a pair with same_room={same_room} in this storey")
