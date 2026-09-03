"""The geometry handed to the solver, built from what the viewer shows.

The rule this module exists to enforce is "what you see is what is simulated".
There is deliberately no second reconstruction here: the shell is
``extrude_storey`` of the same walkable outline the viewer walks on, and each
obstacle is the same collider file, under the same instance matrix, as the
acoustic view draws. If the two ever disagree, it is a bug, and
``describe_geometry`` exists so that disagreement can be measured rather than
argued about.

**Nothing is simplified here any more.** Envelope fitting, level-of-detail
decimation and the absorption compensation that paid for them are gone. They
were built when the target was pyroomacoustics, which charges one wall per
triangle; the wave solver charges for the bounding box instead and is
indifferent to triangle count. Roadmap section 6.3 retires their thresholds,
and a reduction whose justification has been retired is a transformation
standing between the picture and the solver for no reason. Voxelisation cost
does still scale with triangle count, and that is now the only thing paying:
roughly six minutes for a bedroom at 16 kHz, against a solve measured in hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import trimesh
from shapely.geometry import Point

from reverberate.geometry.apartment import (
    Storey,
    build_apartment,
    extrude_storey,
    instances_on_storey,
)
from reverberate.geometry.hssd_assets import category_for_template, resolve_asset
from reverberate.geometry.hssd_room import FurnitureInstance, load_object_instances
from reverberate.geometry.materials import material_for_label
from reverberate.geometry.orientation import BOTH, orient_for_air
from reverberate.geometry.pra_room import MeshMaterialAssignment
from reverberate.geometry.sealed import SealedReport, sealed_regions
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
    #: Faces whose orientation could not be derived and which are therefore
    #: exported as active on both sides. Reported rather than hidden: it is the
    #: share of the scene whose absorption depends on a claim about geometry
    #: that nothing was able to check.
    unoriented_faces: int = 0
    #: Obstacles whose convex bodies could not be unioned into one outer
    #: surface, so they still carry their original buried interior faces. Named
    #: rather than counted only, because ``outer_surface``'s own docstring
    #: promises the count is reported and not hidden -- and a mesh that kept
    #: its buried faces is exactly the kind ``sealed_regions`` cannot be
    #: trusted to classify correctly.
    unmerged: list[str] = field(default_factory=list)
    #: The air the solver will seal inside closed bodies, and any body whose
    #: inside could not be told from its outside. Carried here so the picture
    #: can show what the simulation stopped carrying sound through: sealing is
    #: correct, and it must not therefore be silent.
    sealed: SealedReport = field(default_factory=SealedReport)

    @property
    def total_walls(self) -> int:
        """pyroomacoustics builds one wall per triangle, so this is the real cost."""
        return self.shell_faces + self.obstacle_faces

    def summary(self) -> str:
        oriented = self.total_walls - self.unoriented_faces
        return (
            f"shell {self.shell_faces} faces ({self.shell_volume:.0f} m3, "
            f"watertight={self.shell_watertight}), {self.obstacle_count} obstacles "
            f"totalling {self.obstacle_faces} faces, {self.total_walls} pra walls, "
            f"{oriented} faces oriented, {len(self.unmerged)} unmerged, "
            f"{self.sealed.summary()}"
        )


def outer_surface(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """The union of a collider's convex bodies, and whether the union succeeded.

    HSSD ships colliders as convex decompositions whose bodies interpenetrate.
    Every contact between two bodies leaves a pair of faces *inside* the solid,
    where sound never reaches, and the voxeliser's adjacency graph around them
    is a tangle of shards rather than one sealed surface.

    That is not cosmetic. PFFDTD deliberately does not fill solids -- it builds
    an adjacency graph so it can accept non-watertight scenes -- so the air
    inside every object is simulated, bounded by nodes marked rigid, which is a
    cavity with no absorption at all and a correspondingly enormous Q. Measured
    on one bedroom: 403 such pockets holding 3.77 m3, 11 per cent of the room's
    own air, and 211 of them resonating between 1 and 4 kHz at a mean size of
    8.6 cm. The responses show the consequence as narrow lines near 2 kHz that
    only emerge below -25 dB and drag the late decay to three times the early
    one.

    The union removes the buried faces exactly rather than approximately. It
    conserves volume to the digit, so nothing about the shape is given up; only
    the interior area goes, which is the area that was never reachable.

    Returns the original mesh unchanged, and False, when the boolean engine
    cannot do it, or when it returns something that is not actually one sealed
    solid: an obstacle with its buried faces is still better than no obstacle,
    and the caller reports the count rather than hiding it. Watertightness is
    checked here rather than assumed, because "conserves volume to the digit"
    is a claim about a *closed* solid, and a union that comes back open has not
    earned it.
    """
    if mesh.body_count <= 1:
        return mesh, True
    try:
        united = trimesh.boolean.union(list(mesh.split(only_watertight=False)))
    except Exception:
        return mesh, False
    if not isinstance(united, trimesh.Trimesh) or len(united.faces) == 0:
        return mesh, False
    if not united.is_watertight:
        return mesh, False
    return united, True


@lru_cache(maxsize=512)
def _load_and_union(hssd_root: Path, template: str) -> tuple[trimesh.Trimesh, bool] | None:
    """The unioned collider for a template, and whether the union held.

    ``None`` only when the asset itself cannot be resolved or loaded. Cached
    because a room usually places the same template several times and the
    union is the expensive part; the ``bool`` is cached alongside the mesh so
    a caller that needs to know about a failed union is not forced to redo the
    union just to learn that.
    """
    asset = resolve_asset(hssd_root / "objects", template)
    if asset is None:
        return None
    mesh = trimesh.load(asset.collider, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return outer_surface(mesh)


def obstacle_collider(hssd_root: Path, template: str) -> trimesh.Trimesh | None:
    """The mesh that both the solver and the acoustic view use for a template.

    HSSD's collider, reduced to its outer surface by :func:`outer_surface`.
    ``resolve_asset`` falls back to the render mesh when an object has no
    ``.collider.glb``, which is HSSD's own rule and is what keeps doors and
    windows in the simulation instead of silently dropping them.

    A thin wrapper over :func:`_load_and_union`'s cache for callers that only
    want the mesh. :func:`obstacle_assignments` calls ``_load_and_union``
    directly instead, because it is the one place a failed union has to be
    reported rather than silently accepted.
    """
    result = _load_and_union(hssd_root, template)
    return result[0] if result is not None else None


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
    # Orientation is derived on the whole enclosure, before it is cut into
    # floor, wall and ceiling: a submesh of a box is an open sheet with no
    # inside, and asking each part on its own would throw away the one thing
    # that makes the answer knowable. The shell's air is on the inside.
    oriented = orient_for_air(shell, "inside")
    shell = oriented.mesh
    assignments = []
    for surface in ("floor", "wall", "ceiling"):
        selected = labels == surface
        if not selected.any():
            continue
        faces = np.flatnonzero(selected)
        part = shell.submesh([faces], append=True)
        assert isinstance(part, trimesh.Trimesh)
        assignments.append(
            MeshMaterialAssignment(
                mesh=part,
                material=material_for_label(surface, rng),
                name=f"shell_{surface}",
                sides=oriented.sides[faces],
            )
        )
    return assignments


def obstacle_assignments(
    hssd_root: Path,
    instances: list[FurnitureInstance],
    seed: int = 0,
) -> tuple[list[MeshMaterialAssignment], list[str], list[str]]:
    """Every piece of furniture, as its collider under its instance matrix.

    The collider goes to the solver whole. Nothing is decimated, no envelope is
    fitted, and no absorption is rescaled to make up for either, because
    nothing is taken away to make up for.

    Returns the assignments, the templates that could not be resolved at all,
    and the *assignment* names (``owner``-shaped, like ``"seat_3"``, not the
    template id) whose convex bodies could not be unioned into one outer
    surface -- placed anyway, on their original, buried-face mesh, because an
    obstacle with a defect is better than a missing obstacle, but named so the
    defect is visible rather than indistinguishable from a clean union, and so
    :func:`~reverberate.geometry.sealed.sealed_regions` can tell which placed
    piece it is looking at rather than which template it came from.
    """
    rng = np.random.default_rng(seed)
    assignments: list[MeshMaterialAssignment] = []
    unresolved: list[str] = []
    unmerged: list[str] = []

    for index, instance in enumerate(instances):
        category = category_for_template(hssd_root, instance.template_name) or "unknown"
        loaded = _load_and_union(hssd_root, instance.template_name)
        if loaded is None:
            unresolved.append(instance.template_name)
            continue
        base, merged = loaded
        mesh = base.copy()
        mesh.apply_transform(instance.transform_matrix())
        # After the instance matrix, not before: a mirroring transform flips
        # the winding, so an orientation derived in the template's own frame
        # would be exactly backwards for half the placements.
        oriented = orient_for_air(mesh, "outside")
        mesh = oriented.mesh

        name = f"{category}_{index}"
        if not merged:
            unmerged.append(name)
        assignments.append(
            MeshMaterialAssignment(
                mesh=mesh,
                material=material_for_label(category, rng),
                name=name,
                sides=oriented.sides,
            )
        )
    return assignments, unresolved, unmerged


def simulation_geometry(
    hssd_root: Path,
    storey: Storey,
    instances: list[FurnitureInstance],
    seed: int = 0,
    storeys: list[Storey] | None = None,
) -> tuple[list[MeshMaterialAssignment], GeometrySummary]:
    """Everything pyroomacoustics receives for one apartment storey.

    ``instances`` is filtered to this storey here rather than trusting the
    caller to have remembered: on a multi-storey scene, passing a whole
    scene's furniture would otherwise simulate the floor above as well.
    Filtering is idempotent, so pre-filtered input is safe. Pass ``storeys``
    when the scene has more than one, so pieces in the overlap band between a
    ceiling and the floor above are assigned to exactly one of them.

    The result depends only on the scene and the seed. It used to depend on
    where the listener stood, through the level-of-detail ladder, which meant
    the viewer had to be shown the mesh for the pair being simulated rather
    than the one scene. One geometry per storey now, for every pair.
    """
    instances = instances_on_storey(instances, storey, storeys)
    shell = shell_assignments(storey, seed=seed)
    obstacles, unresolved, unmerged = obstacle_assignments(hssd_root, instances, seed=seed)
    whole_shell = extrude_storey(storey)
    everything = [*shell, *obstacles]
    summary = GeometrySummary(
        shell_faces=sum(len(assignment.mesh.faces) for assignment in shell),
        shell_volume=float(whole_shell.volume),
        shell_watertight=bool(whole_shell.is_watertight),
        obstacle_count=len(obstacles),
        obstacle_faces=sum(len(assignment.mesh.faces) for assignment in obstacles),
        unresolved=unresolved,
        unmerged=unmerged,
        unoriented_faces=sum(
            int(np.count_nonzero(assignment.sides == BOTH))
            for assignment in everything
            if assignment.sides is not None
        ),
        sealed=sealed_regions(list(everything), unmerged=frozenset(unmerged)),
    )
    return everything, summary


def apartment_geometry(
    hssd_root: Path,
    scene_id: str,
    storey_index: int = 0,
    seed: int = 0,
    include_outdoor: bool = False,
) -> tuple[list[MeshMaterialAssignment], GeometrySummary, Storey]:
    """One call from a scene id to everything the simulator needs.

    This is the entry point for the acoustic pipeline: it assembles the
    apartment, keeps the furniture standing on the chosen storey, and returns
    the mesh-plus-material list ``pra_room.build_room`` already expects.
    Storeys come largest first, so the default is the main floor.

    The returned assignments are the *same* meshes the viewer draws in its
    acoustic mode, which is the property worth preserving: if the simulation
    and the picture ever disagree, one of them stopped calling this function.
    """
    storeys = build_apartment(hssd_root, scene_id, include_outdoor=include_outdoor)
    if not storeys:
        raise ValueError(f"scene {scene_id} has no walkable storey")
    storey = storeys[storey_index]
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    assignments, summary = simulation_geometry(
        hssd_root, storey, instances, seed=seed, storeys=storeys
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
