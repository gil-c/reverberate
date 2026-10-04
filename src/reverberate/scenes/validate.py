"""The numbered rules of ``docs/formats/scene-recipe.md``, each with its number.

:func:`validate` returns every violation it finds; :func:`check` raises the
first. Rule 1, the shape, is the parser's (:mod:`reverberate.scenes.recipe`):
a recipe that reached this module has it.

**The floor is not in the recipe.** Rules 3, 4 and 6 ask whether a point is on
the free floor, in a room or near a wall, and that is the dwelling's geometry.
Pass the :class:`~reverberate.scenes.layout.Floor`; without it those parts are
skipped; :data:`FLOOR_RULES` names them, so a caller can say that it ran half
a validation and not take it for a whole one.

Rules 6 to 9 are checked where the format says: every 50 ms and at every
keyframe and segment boundary, on the positions of
:mod:`reverberate.scenes.kinematics` and no others.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.scenes.kinematics import (
    listener_state,
    polyline_length,
    rail_length,
    sample_times,
    source_state,
)
from reverberate.scenes.layout import SEAT_REACH_M, SEAT_WALL_M, Floor
from reverberate.scenes.recipe import (
    Recipe,
    RecipeError,
    Rise,
    Source,
    Travel,
    decimals_of,
    parse_recipe,
)

__all__ = [
    "FLOOR_RULES",
    "HEAD_CLEARANCE_M",
    "LOW_BAND_TEMPERATURE_C",
    "MAX_SPEED_M_S",
    "MAX_TURN_DEG_S",
    "MIN_RISE_S",
    "SOURCE_CLEARANCE_M",
    "Violation",
    "check",
    "validate",
    "validate_text",
]

#: Rule 7.
MAX_SPEED_M_S = 1.5
MIN_RISE_S = 1.0
MAX_TURN_DEG_S = 360.0

#: Rule 8: a mouth and the head, and two sources.
HEAD_CLEARANCE_M = 0.50
SOURCE_CLEARANCE_M = 0.40

#: Rule 10: the temperature the low band is solved at.
LOW_BAND_TEMPERATURE_C = 20.0

#: Rule 4.
RAIL_PITCH_M = 0.08

#: Slack on comparisons of numbers that were rounded to their step.
_EPS = 1e-6

_DIGEST = re.compile(r"[0-9a-f]{64}")

#: The rules whose geometric half needs the floor.
FLOOR_RULES = (3, 4, 6)


@dataclass(frozen=True)
class Violation:
    rule: int
    message: str

    def __str__(self) -> str:
        return f"rule {self.rule}: {self.message}"


def validate_text(text: str | bytes, floor: Floor | None = None) -> list[Violation]:
    """Parse and validate; a recipe rule 1 refuses comes back as that violation."""
    try:
        recipe = parse_recipe(text)
    except RecipeError as error:
        return [Violation(error.rule, error.message)]
    return validate(recipe, floor)


def check(recipe: Recipe, floor: Floor | None = None) -> None:
    """Raise the first violation as a :class:`RecipeError`."""
    found = validate(recipe, floor)
    if found:
        raise RecipeError(found[0].rule, found[0].message)


def validate(recipe: Recipe, floor: Floor | None = None) -> list[Violation]:
    """Every violation of rules 2 to 11, in the order of the rules."""
    found: list[Violation] = []
    found += _time(recipe)
    found += _stations(recipe, floor)
    found += _rails(recipe, floor)
    found += _graph(recipe)
    structural = bool(found)
    found += _assets(recipe)
    found += _numbers(recipe)
    if not structural:
        # The trajectories exist only once the structure holds.
        found += _listener(recipe, floor)
        found += _speeds(recipe)
        found += _clearances(recipe)
        found += _collisions(recipe)
    return sorted(found, key=lambda violation: violation.rule)


# --------------------------------------------------------------------------
# rule 2: time
# --------------------------------------------------------------------------


def _time(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(2, message))

    duration = recipe.duration_s
    if not duration > 0:
        bad(f"duration_s is {duration}, expected more than 0")
        return out
    seen: set[str] = set()
    for source in recipe.sources:
        if source.id in seen:
            out.append(Violation(1, f"source id {source.id!r} is used twice"))
        seen.add(source.id)
        segments = source.segments
        if not segments:
            bad(f"source {source.id} has no segment")
            continue
        if segments[0].start_s != 0.0:
            bad(f"source {source.id} starts at {segments[0].start_s}, expected 0")
        if segments[-1].end_s != duration:
            bad(f"source {source.id} ends at {segments[-1].end_s}, expected {duration}")
        for index, segment in enumerate(segments):
            if not segment.end_s > segment.start_s:
                bad(f"source {source.id} segment {index} does not end after it starts")
            if index and segment.start_s != segments[index - 1].end_s:
                kind = "overlap" if segment.start_s < segments[index - 1].end_s else "gap"
                bad(f"source {source.id} has a {kind} before segment {index}")
        last = 0.0
        for index, interval in enumerate(source.activity):
            if not interval.end_s > interval.start_s:
                bad(f"source {source.id} activity {index} does not end after it starts")
            if interval.start_s < 0 or interval.end_s > duration:
                bad(f"source {source.id} activity {index} lies outside the scene")
            if interval.start_s < last:
                bad(f"source {source.id} activity {index} overlaps the one before")
            if interval.clip_offset_s < 0:
                bad(f"source {source.id} activity {index} starts before its clip")
            last = max(last, interval.end_s)
    frames = recipe.listener.keyframes
    if len(frames) < 2:
        bad("the listener has fewer than two keyframes")
        return out
    if frames[0].t_s != 0.0:
        bad(f"the first keyframe is at {frames[0].t_s}, expected 0")
    if frames[-1].t_s != duration:
        bad(f"the last keyframe is at {frames[-1].t_s}, expected {duration}")
    for index in range(1, len(frames)):
        if not frames[index].t_s > frames[index - 1].t_s:
            bad(f"keyframe {index} does not come after keyframe {index - 1}")
            break
    return out


# --------------------------------------------------------------------------
# rules 3 and 4: stations and rails
# --------------------------------------------------------------------------


def _stations(recipe: Recipe, floor: Floor | None) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(3, message))

    seen: set[str] = set()
    for station in recipe.stations:
        if station.id in seen:
            bad(f"station id {station.id!r} is used twice")
        seen.add(station.id)
        seated = station.kind == "seat"
        if (station.height == "seated") != seated:
            bad(f"station {station.id} is a {station.kind} and says {station.height}")
        expected = recipe.dwelling.floor_y_m + recipe.heights.of(station.height)
        if abs(station.position[1] - expected) > 1e-3 + _EPS:
            bad(f"station {station.id} is at y = {station.position[1]}, expected {expected:.3f}")
        if floor is None:
            continue
        x, z = station.xz
        if not seated:
            if not floor.on_free(np.array([x, z]))[0]:
                bad(f"station {station.id} is not on the free floor")
            elif floor.room_of(x, z) != station.room:
                bad(f"station {station.id} is in {floor.room_of(x, z)!r}, not {station.room!r}")
            continue
        footprint = floor.footprint(station.object) if station.object else None
        if footprint is None:
            bad(f"seat {station.id} names no object of the dwelling ({station.object!r})")
            continue
        from shapely.geometry import Point

        away = float(footprint.distance(Point(x, z)))
        if away > SEAT_REACH_M + _EPS:
            bad(f"seat {station.id} is {away:.2f} m from the footprint of {station.object}")
        if floor.wall_distance(x, z) < SEAT_WALL_M - _EPS:
            bad(f"seat {station.id} is {floor.wall_distance(x, z):.2f} m from a wall")
    return out


def _stretch(points: np.ndarray, low: float, high: float) -> np.ndarray:
    """The part of a polyline between two arc lengths."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arcs = np.concatenate([[0.0], np.cumsum(steps)])
    inner = points[(arcs > low) & (arcs < high)]
    ends = [[np.interp(arc, arcs, points[:, axis]) for axis in range(2)] for arc in (low, high)]
    return np.vstack([ends[0], inner.reshape(-1, 2), ends[1]])


def _rails(recipe: Recipe, floor: Floor | None) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(4, message))

    known = {station.id: station for station in recipe.stations}
    seen: set[str] = set()
    for rail in recipe.rails:
        if rail.id in seen:
            bad(f"rail id {rail.id!r} is used twice")
        seen.add(rail.id)
        if rail.pitch_m != RAIL_PITCH_M:
            bad(f"rail {rail.id} has pitch_m {rail.pitch_m}, expected {RAIL_PITCH_M}")
        if rail.a not in known or rail.b not in known or rail.a == rail.b:
            bad(f"rail {rail.id} does not join two stations ({rail.a!r}, {rail.b!r})")
            continue
        if rail.points[0] != known[rail.a].xz or rail.points[-1] != known[rail.b].xz:
            bad(f"rail {rail.id} does not start at {rail.a} and end at {rail.b}")
        points = np.asarray(rail.points, dtype=float)
        length = polyline_length(points)
        if not length > 0:
            bad(f"rail {rail.id} has no length")
            continue
        if floor is None:
            continue
        low = SEAT_REACH_M if known[rail.a].kind == "seat" else 0.0
        high = length - (SEAT_REACH_M if known[rail.b].kind == "seat" else 0.0)
        if high <= low:
            continue
        checked = _stretch(points, low, high)
        if not floor.legs_on_free(checked[:-1], checked[1:]).all():
            bad(f"rail {rail.id} leaves the free floor further than 0.60 m from a seat")
    return out


# --------------------------------------------------------------------------
# rule 5: sources on the graph
# --------------------------------------------------------------------------


def _graph(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    stations = {station.id: station for station in recipe.stations}
    rails = {rail.id: rail for rail in recipe.rails}
    for source in recipe.sources:
        problem = _walk(source, stations, rails)
        if problem:
            out.append(Violation(5, f"source {source.id}: {problem}"))
    return out


def _walk(source: Source, stations: dict[str, Any], rails: dict[str, Any]) -> str | None:
    at: str | None = None
    posture: str | None = None
    for index, segment in enumerate(source.segments):
        where = f"segment {index}"
        if isinstance(segment, Travel):
            rail = rails.get(segment.rail)
            if rail is None:
                return f"{where} names no rail of the recipe ({segment.rail!r})"
            if {segment.origin, segment.to} != {rail.a, rail.b} or segment.origin == segment.to:
                return f"{where} does not go from one end of {rail.id} to the other"
            if at is not None and segment.origin != at:
                return f"{where} leaves from {segment.origin} while the source is at {at}"
            if posture == "seated":
                return f"{where} travels while the source is seated"
            at, posture = segment.to, "standing"
            continue
        station = stations.get(segment.station)
        if station is None:
            return f"{where} names no station of the recipe ({segment.station!r})"
        if at is not None and segment.station != at:
            return f"{where} is at {segment.station} while the source is at {at}"
        if isinstance(segment, Rise):
            if station.kind != "seat":
                return f"{where} rises at {station.id}, which is not a seat"
            if posture == segment.to:
                return f"{where} rises to {segment.to} from {posture}"
            at, posture = segment.station, segment.to
            continue
        if station.kind == "waypoint":
            return f"{where} dwells at {station.id}, a waypoint where nobody stays"
        if segment.height == "seated" and station.kind != "seat":
            return f"{where} is seated at {station.id}, which is not a seat"
        if posture is not None and segment.height != posture:
            return f"{where} dwells {segment.height} while the source is {posture}"
        at, posture = segment.station, segment.height
    return None


# --------------------------------------------------------------------------
# rule 6: the listener on the floor
# --------------------------------------------------------------------------


def _listener(recipe: Recipe, floor: Floor | None) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(6, message))

    stations = {station.id: station for station in recipe.stations}
    frames = recipe.listener.keyframes
    base = recipe.dwelling.floor_y_m
    low, high = recipe.heights.seated_m, recipe.heights.standing_m
    seats: list[str | None] = []
    for index, frame in enumerate(frames):
        station = stations.get(frame.station) if frame.station is not None else None
        if frame.station is not None and station is None:
            bad(f"keyframe {index} names no station of the recipe ({frame.station!r})")
        seat = station is not None and station.kind == "seat"
        seats.append(station.id if station is not None and seat else None)
        height = frame.position[1] - base
        if height < low - _EPS or height > high + _EPS:
            bad(f"keyframe {index}: the head is {height:.3f} m above the floor")
        elif height <= low + _EPS and not seat:
            bad(f"keyframe {index}: the head is at the seated height and not at a seat")
        if station is not None and (
            math.dist((frame.position[0], frame.position[2]), station.xz) > _EPS
            or (seat and abs(frame.position[1] - station.position[1]) > _EPS)
        ):
            bad(f"keyframe {index} names {station.id} and is not at it")
    if floor is None or out:
        return out
    xz = np.array([[frame.position[0], frame.position[2]] for frame in frames])
    on_floor = floor.on_free(xz)
    legs = floor.legs_on_free(xz[:-1], xz[1:])
    for index in range(len(frames)):
        if not on_floor[index] and seats[index] is None:
            bad(f"keyframe {index} is not on the free floor")
    for index in range(len(frames) - 1):
        here, there = seats[index], seats[index + 1]
        if legs[index] or (here is not None and here == there):
            continue
        # Sitting down or getting up: one end is the seat, the other the free
        # floor within a rail's reach of it.
        if (here is None) != (there is None):
            other = index if here is None else index + 1
            reach = float(np.linalg.norm(xz[index + 1] - xz[index]))
            if on_floor[other] and reach <= SEAT_REACH_M + _EPS:
                continue
        bad(f"the listener leaves the free floor between keyframes {index} and {index + 1}")
    return out


# --------------------------------------------------------------------------
# rule 7: speeds
# --------------------------------------------------------------------------


def _speeds(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(7, message))

    for source in recipe.sources:
        if source.turn_rate_deg_s < 0:
            bad(f"source {source.id} has a negative turn rate")
        for index, segment in enumerate(source.segments):
            span = segment.end_s - segment.start_s
            if isinstance(segment, Rise) and span < MIN_RISE_S - _EPS:
                bad(f"source {source.id} segment {index} rises in {span:.3f} s")
            if isinstance(segment, Travel):
                peak = rail_length(recipe.rail(segment.rail)) / span
                peak *= 1.5 if segment.profile == "smoothstep" else 1.0
                if peak > MAX_SPEED_M_S + _EPS:
                    bad(f"source {source.id} segment {index} peaks at {peak:.3f} m/s")
    frames = recipe.listener.keyframes
    for index in range(len(frames) - 1):
        here, there = frames[index], frames[index + 1]
        span = there.t_s - here.t_s
        speed = math.dist(here.position, there.position) / span
        if speed > MAX_SPEED_M_S + _EPS:
            bad(f"the listener moves at {speed:.3f} m/s after keyframe {index}")
        turn = max(
            abs(there.yaw_deg - here.yaw_deg),
            abs(there.pitch_deg - here.pitch_deg),
            abs(there.roll_deg - here.roll_deg),
        )
        if turn / span > MAX_TURN_DEG_S + _EPS:
            bad(f"the head turns at {turn / span:.1f} degrees a second after keyframe {index}")
    return out


# --------------------------------------------------------------------------
# rules 8 and 9: clearances and collisions
# --------------------------------------------------------------------------


def _instants(recipe: Recipe) -> np.ndarray:
    edges = [frame.t_s for frame in recipe.listener.keyframes]
    for source in recipe.sources:
        edges += [segment.start_s for segment in source.segments]
    return np.unique(np.concatenate([sample_times(recipe), np.asarray(edges, dtype=float)]))


def _clearances(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    times = _instants(recipe)
    head = listener_state(recipe, times).position
    mouths = [source_state(recipe, source.id, times).position for source in recipe.sources]
    for index, source in enumerate(recipe.sources):
        gap = np.linalg.norm(mouths[index] - head, axis=1)
        worst = int(np.argmin(gap))
        if gap[worst] < HEAD_CLEARANCE_M - _EPS:
            out.append(
                Violation(
                    8,
                    f"source {source.id} is {gap[worst]:.3f} m from the head at "
                    f"t = {times[worst]:.3f} s",
                )
            )
        for other in range(index + 1, len(recipe.sources)):
            gap = np.linalg.norm(mouths[index] - mouths[other], axis=1)
            worst = int(np.argmin(gap))
            if gap[worst] < SOURCE_CLEARANCE_M - _EPS:
                out.append(
                    Violation(
                        8,
                        f"sources {source.id} and {recipe.sources[other].id} are "
                        f"{gap[worst]:.3f} m apart at t = {times[worst]:.3f} s",
                    )
                )
    return out


def _collisions(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    travels: dict[str, list[tuple[str, Travel]]] = {}
    for source in recipe.sources:
        for segment in source.segments:
            if isinstance(segment, Travel):
                travels.setdefault(segment.rail, []).append((source.id, segment))
    for rail, users in travels.items():
        for index, (first, one) in enumerate(users):
            for second, two in users[index + 1 :]:
                if first == second or one.origin == two.origin:
                    continue
                if one.start_s < two.end_s and two.start_s < one.end_s:
                    out.append(
                        Violation(
                            9,
                            f"sources {first} and {second} travel {rail} in opposite "
                            f"directions from t = {max(one.start_s, two.start_s):.3f} s",
                        )
                    )
    return out


# --------------------------------------------------------------------------
# rules 10 and 11: assets and canonical numbers
# --------------------------------------------------------------------------


def _assets(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    for source in recipe.sources:
        for index, interval in enumerate(source.activity):
            if not _DIGEST.fullmatch(interval.clip.sha256):
                out.append(
                    Violation(
                        10,
                        f"source {source.id} activity {index}: the clip's digest is not "
                        "64 hexadecimal characters",
                    )
                )
                break
        model = source.directivity.model
        if model != "omni" and model not in recipe.assets.directivity:
            out.append(
                Violation(
                    10,
                    f"source {source.id} names the directivity {model!r}, which has no "
                    "digest under assets.directivity",
                )
            )
    if recipe.atmosphere.temperature_c != LOW_BAND_TEMPERATURE_C:
        out.append(
            Violation(
                10,
                f"the temperature is {recipe.atmosphere.temperature_c} C; the low band was "
                f"solved at {LOW_BAND_TEMPERATURE_C}",
            )
        )
    return out


def _off_step(value: Any, key: str, where: str) -> str | None:
    if isinstance(value, dict):
        for name, item in value.items():
            if name == "generator":
                continue
            found = _off_step(item, name, f"{where}.{name}" if where else name)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _off_step(item, key, f"{where}[{index}]")
            if found:
                return found
    elif type(value) is float:
        decimals = decimals_of(key)
        if decimals is not None and round(value, decimals) != value:
            return f"{where} is {value!r}, not a multiple of {10.0**-decimals:g}"
    return None


def _numbers(recipe: Recipe) -> list[Violation]:
    found = _off_step(recipe.to_dict(), "", "")
    return [Violation(11, found)] if found else []
