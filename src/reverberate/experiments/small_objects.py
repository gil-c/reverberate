"""Does simulating a small object as solid at every band actually matter?

pyroomacoustics takes one geometry shared by all seven octave bands, so a
0.8 m potted plant reflects at 125 Hz exactly as it does at 8 kHz. Physically
it should not: at 125 Hz the wavelength is 274 cm and sound flows around an
object that size rather than off it. Geometrical acoustics has no diffraction,
so this is not a bug we introduced and not one we can fix from inside the
simulator: the coefficients are the only lever, and there is no value of alpha
that means "let the energy through". Alpha 0 reflects it all, alpha 1 absorbs
it all, and neither is transmission. This was confirmed at source level in
pyroomacoustics 0.10.1: ``wall.cpp`` sets ``transmission = sqrt(1 - alpha)``,
which is the amplitude *reflection* coefficient carried along the ray, and
``room.cpp`` always takes the nearest wall and bounces. Nothing crosses a wall.

**This experiment has not produced a trustworthy result yet.** It was written
before the ray count was found to be set 147x below the minimum pyroomacoustics
computes for itself, so the one run attempted measured mostly sampling noise
and was discarded. It is kept because the protocol is the reusable part, but it
must be re-run over several seeds and reported against the measured spread, not
against the JND alone, before any number from it is quoted.

So the question this module answers is not "how do we correct it" but "how
large is it", because that decides whether it needs correcting at all. The
bound is measured by removing the object entirely: an object that is
acoustically transparent and an object that is absent are the same thing to a
ray tracer, so *present minus absent* is exactly the error the low bands
carry. At 8 kHz the same difference is the object working as intended, which
gives the measurement its own control: the effect must be large at the top of
the range and is only a defect where it survives at the bottom.

Two deliberate choices about the setup:

**One room, not the apartment.** The plant is compared against everything else
that absorbs, so the fewer competing surfaces there are the larger its share.
An isolated living room is therefore the *upper* bound on its influence, and a
bound that is under the JND here is under it in the whole flat as well.

**Source and receiver either side of it, close in.** The pair is placed on a
line through the object at head-ish height, which is the geometry that
maximises occlusion of the direct path. A pair drawn at random would mostly
measure the diffuse field, where one plant is negligible by construction and
the answer would be "no effect" for the wrong reason.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.geometry.apartment import Storey, build_apartment, instances_on_storey
from reverberate.geometry.hssd_room import load_object_instances
from reverberate.geometry.placement import footprint_of
from reverberate.geometry.pra_room import MeshMaterialAssignment, PairResponse, simulate_pairs
from reverberate.geometry.sim_geometry import MIN_WALL_DISTANCE, room_of, simulation_geometry
from reverberate.metrics import JND_C50_DB, JND_DRR_DB, JND_RT60_RELATIVE

#: Categories treated as "small objects" for this measurement. Deliberately
#: short: the roadmap names the potted plant as the designated test case, and a
#: list that grew to cover everything would dilute the result rather than
#: strengthen it.
SMALL_CATEGORIES = ("plant", "vase")


def isolated_storey(storey: Storey, room_name: str) -> Storey:
    """One room of a storey, on its own, as a storey the pipeline accepts.

    Reuses the production shell and material path rather than building a
    special test geometry: the room is the same extruded polygon, with the same
    floor/wall/ceiling split, that the apartment would have given it.
    """
    rooms = [room for room in storey.rooms if room.name == room_name]
    if not rooms:
        raise ValueError(f"no room named {room_name!r} on this storey")
    room = rooms[0]
    return Storey(
        floor_height=storey.floor_height,
        ceiling_height=storey.ceiling_height,
        walkable=room.polygon_xz.buffer(0),
        rooms=rooms,
        doorways=0,
    )


def free_area(
    storey: Storey,
    assignments: list[MeshMaterialAssignment],
    exclude: str,
    min_wall_distance: float = MIN_WALL_DISTANCE,
) -> Polygon | MultiPolygon:
    """Floor a source or receiver may stand on, ignoring one named obstacle.

    The object under test is excluded on purpose: the whole point is to stand
    either side of it, and its own footprint would otherwise push the pair
    outwards until it no longer occludes anything.
    """
    blocked = [
        footprint_of(assignment.mesh)
        for assignment in assignments
        if assignment.name != exclude and not assignment.name.startswith("shell_")
    ]
    area = storey.walkable.buffer(-min_wall_distance)
    if blocked:
        area = area.difference(
            unary_union([polygon for polygon in blocked if not polygon.is_empty])
        )
    return area


@dataclass(frozen=True)
class OccludingPair:
    """A source and a receiver placed either side of one obstacle."""

    source: np.ndarray
    receiver: np.ndarray
    obstacle: str
    half_distance: float
    azimuth: float

    @property
    def pair(self) -> tuple[np.ndarray, np.ndarray]:
        return self.source, self.receiver


def occluding_pair(
    mesh: trimesh.Trimesh,
    name: str,
    area: Polygon | MultiPolygon,
    half_distance: float,
    azimuths: int = 36,
) -> OccludingPair | None:
    """Place a source and a receiver on a line through an obstacle's centroid.

    Height comes from the obstacle rather than from a fixed 1.2 m: a pair at
    head height either side of a low object has an unobstructed direct path and
    would measure nothing. The azimuth is swept because most orientations run
    into a wall or another piece of furniture; the first one where both ends
    land on free floor is taken, and ``None`` is returned when none does, so a
    plant wedged into a corner is reported as unmeasurable instead of being
    quietly measured somewhere else.
    """
    centre = np.asarray(mesh.bounds, dtype=float).mean(axis=0)
    for step in range(azimuths):
        angle = 2.0 * np.pi * step / azimuths
        offset = np.array([np.cos(angle), 0.0, np.sin(angle)]) * half_distance
        source = centre + offset
        receiver = centre - offset
        if all(
            area.contains(Point(float(point[0]), float(point[2]))) for point in (source, receiver)
        ):
            return OccludingPair(
                source=source,
                receiver=receiver,
                obstacle=name,
                half_distance=half_distance,
                azimuth=angle,
            )
    return None


@dataclass(frozen=True)
class SmallObjectEffect:
    """What one obstacle does to one pair, band by band.

    ``present`` is what we simulate today and ``absent`` is what an
    acoustically transparent object would give, so the difference is the error
    at the bands where the object should have been transparent and the intended
    behaviour at the bands where it should not.
    """

    obstacle: str
    category: str
    size: float
    half_distance: float
    bands: tuple[int, ...]
    rt60_relative: np.ndarray
    c50_db: np.ndarray
    drr_db: np.ndarray
    direct_level_db: float

    def exceeds_jnd(self, band_index: int) -> bool:
        return bool(
            self.rt60_relative[band_index] > JND_RT60_RELATIVE
            or abs(self.c50_db[band_index]) > JND_C50_DB
            or abs(self.drr_db[band_index]) > JND_DRR_DB
        )

    def rows(self) -> list[str]:
        lines = []
        for index, band in enumerate(self.bands):
            verdict = "OVER JND" if self.exceeds_jnd(index) else "under"
            lines.append(
                f"  {band:>5} Hz  RT60 {self.rt60_relative[index]:+7.2%}  "
                f"C50 {self.c50_db[index]:+6.2f} dB  DRR {self.drr_db[index]:+6.2f} dB  {verdict}"
            )
        return lines


def effect_of(
    present: PairResponse,
    absent: PairResponse,
    obstacle: str,
    category: str,
    size: float,
    half_distance: float,
) -> SmallObjectEffect:
    """Compare the two responses, with the transparent case as the reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rt60 = np.abs(present.bands.rt60 - absent.bands.rt60) / absent.bands.rt60
    return SmallObjectEffect(
        obstacle=obstacle,
        category=category,
        size=size,
        half_distance=half_distance,
        bands=OCTAVE_BANDS,
        rt60_relative=np.nan_to_num(rt60, nan=0.0, posinf=0.0),
        c50_db=present.bands.c50 - absent.bands.c50,
        drr_db=present.bands.drr - absent.bands.drr,
        direct_level_db=present.bands.direct_level - absent.bands.direct_level,
    )


def small_obstacles(
    assignments: list[MeshMaterialAssignment], max_size: float
) -> list[tuple[str, str, float]]:
    """Named obstacles in the chosen categories, with their largest dimension.

    Names carry the category because ``obstacle_assignments`` builds them as
    ``f"{category}_{index}"``, which is also what makes an assignment
    removable by name without touching anyone else's material draw.
    """
    found = []
    for assignment in assignments:
        category = assignment.name.rsplit("_", 1)[0]
        if category not in SMALL_CATEGORIES:
            continue
        size = float(np.max(assignment.mesh.extents))
        if size <= max_size:
            found.append((assignment.name, category, size))
    return found


def run(
    hssd_root: Path,
    scene_id: str,
    room_name: str,
    half_distances: tuple[float, ...] = (0.5, 1.0),
    max_size: float = 1.2,
    seed: int = 0,
    max_order: int = 2,
    n_rays: int = 10000,
) -> list[SmallObjectEffect]:
    """Measure every small object in one room, against the same room without it.

    The geometry is built **once**, with every obstacle at the finest detail
    rung, and the object under test is then dropped from the assignment list.
    Rebuilding the geometry from a shortened instance list would instead have
    shifted every subsequent material draw by one, so the two runs would have
    differed in far more than the plant.
    """
    storeys = build_apartment(hssd_root, scene_id)
    if not storeys:
        raise ValueError(f"scene {scene_id} has no walkable storey")
    storey = storeys[0]
    instances = load_object_instances(hssd_root / "scenes" / f"{scene_id}.scene_instance.json")
    instances = instances_on_storey(instances, storey, storeys)
    in_room = [
        instance
        for instance in instances
        if room_of(storey, float(instance.translation[0]), float(instance.translation[2]))
        == room_name
    ]
    room_storey = isolated_storey(storey, room_name)
    assignments, summary = simulation_geometry(hssd_root, room_storey, in_room, seed=seed)
    print(f"{scene_id} / {room_name}: {summary.summary()}")

    targets = small_obstacles(assignments, max_size)
    if not targets:
        raise ValueError(f"no small object under {max_size} m in {room_name}")

    by_name = {assignment.name: assignment for assignment in assignments}
    probes: list[tuple[OccludingPair, str, float]] = []
    for name, category, size in targets:
        area = free_area(room_storey, assignments, exclude=name)
        for half_distance in half_distances:
            pair = occluding_pair(by_name[name].mesh, name, area, half_distance)
            if pair is None:
                print(f"  {name} at {half_distance} m: no free line through it, skipped")
                continue
            probes.append((pair, category, size))

    if not probes:
        raise ValueError("no obstacle could be probed: every line through one was blocked")

    print(f"simulating {len(probes)} pairs with every object present")
    present = simulate_pairs(
        assignments, [probe.pair for probe, _, _ in probes], max_order=max_order, n_rays=n_rays
    )

    effects = []
    for index, (probe, category, size) in enumerate(probes):
        reduced = [assignment for assignment in assignments if assignment.name != probe.obstacle]
        print(f"simulating {probe.obstacle} at {probe.half_distance} m, absent")
        absent = simulate_pairs(reduced, [probe.pair], max_order=max_order, n_rays=n_rays)
        effects.append(
            effect_of(
                present[index], absent[0], probe.obstacle, category, size, probe.half_distance
            )
        )
    return effects


def report(effects: list[SmallObjectEffect]) -> str:
    """The result as a human reads it, with the verdict stated rather than implied."""
    lines = []
    low_bands = [index for index, band in enumerate(OCTAVE_BANDS) if band <= 500]
    worst_low = 0.0
    for effect in effects:
        lines.append(
            f"{effect.obstacle} ({effect.size:.2f} m across), pair at "
            f"+/-{effect.half_distance:.2f} m, direct level "
            f"{effect.direct_level_db:+.2f} dB"
        )
        lines.extend(effect.rows())
        worst_low = max(
            worst_low,
            max(abs(effect.c50_db[index]) / JND_C50_DB for index in low_bands),
            max(abs(effect.drr_db[index]) / JND_DRR_DB for index in low_bands),
            max(effect.rt60_relative[index] / JND_RT60_RELATIVE for index in low_bands),
        )
    failing = [
        effect for effect in effects if any(effect.exceeds_jnd(index) for index in low_bands)
    ]
    lines.append("")
    lines.append(
        f"low bands (125-500 Hz): worst effect is {worst_low:.2f} JND, "
        f"{len(failing)} of {len(effects)} probes over the threshold"
    )
    lines.append(
        "verdict: the low-frequency solidity error is audible, small objects must be "
        "removed and their absorbing power folded onto neighbouring surfaces"
        if failing
        else "verdict: the low-frequency solidity error is below the JND everywhere; "
        "record the limitation and leave the geometry alone"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--scene", default="102344049")
    parser.add_argument("--room", default="living room")
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--n-rays", type=int, default=10000)
    parser.add_argument("--half-distance", type=float, nargs="+", default=[0.5, 1.0])
    args = parser.parse_args(argv)

    effects = run(
        args.hssd_root,
        args.scene,
        args.room,
        half_distances=tuple(args.half_distance),
        max_order=args.max_order,
        n_rays=args.n_rays,
    )
    print()
    print(report(effects))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
