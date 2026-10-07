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

Rules 12 to 17 are version 2's and are asked of a version 2 recipe alone:
what may be carried and what is fixed, who is of the listener's
conversation, how far a sway may reach, that a turn's effort is its level,
that no turn of the conversation is under the scene's floor of speech to
noise, and that the record of the gaze is in order.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from reverberate.scenes.kinematics import (
    anchor_at,
    listener_state,
    polyline_length,
    rail_length,
    sample_times,
)
from reverberate.scenes.layout import SEAT_REACH_M, SEAT_WALL_M, Floor
from reverberate.scenes.levels import conversation_snr, effort_range_db
from reverberate.scenes.recipe import (
    LISTENER,
    VOICE_KINDS_V2,
    Dwell,
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
    "SWAY_RADIUS_M",
    "SWAY_YAW_DEG",
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

#: Rule 4: the pitch a rail has unless another is asked for, and the pitches it may have.
#: Two positions half a wavelength apart hold a frequency: 0.20 m is 850 Hz, under the
#: crossover, and nothing wider is of use.
RAIL_PITCH_M = 0.08
RAIL_PITCH_MIN_M = 0.04
RAIL_PITCH_MAX_M = 0.20

#: Rule 14: the furthest a sway may carry a mouth or the head from where the low band
#: is read, the sum of its amplitudes; and the most a sway may turn a source.
SWAY_RADIUS_M = 0.05
SWAY_YAW_DEG = 30.0
SWAY_PERIOD_MIN_S = 0.5
#: Rule 12: a stride, and how far the wearer's mouth is from the centre of the head.
STRIDE_M = (0.3, 1.0)
MOUTH_OFFSET_M = (0.05, 0.25)
#: Rule 3: a fixture stands at most this far outside the walkable outline, and under this height.
FIXTURE_OUTSIDE_M = 0.5
FIXTURE_HEIGHT_MAX_M = 3.5

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
    """Every violation of rules 2 to 11, and of 12 to 17 in version 2, in the rules' order."""
    second = recipe.schema_version >= 2
    found: list[Violation] = []
    found += _time(recipe)
    found += _stations(recipe, floor)
    found += _rails(recipe, floor)
    found += _graph(recipe)
    if second:
        found += _kinds(recipe)
        found += _roles(recipe)
    structural = bool(found)
    found += _assets(recipe)
    found += _numbers(recipe)
    if second:
        found += _sway(recipe)
        found += _efforts(recipe)
        found += _gaze(recipe)
    if not structural:
        # The trajectories exist only once the structure holds.
        found += _listener(recipe, floor)
        found += _speeds(recipe)
        found += _clearances(recipe)
        found += _collisions(recipe)
        if second:
            found += _audible(recipe)
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
        if source.attach is not None:
            # A carried source is where its carrier is: rule 12 asks that it has no segment.
            segments = ()
        elif not segments:
            bad(f"source {source.id} has no segment")
            continue
        if segments and segments[0].start_s != 0.0:
            bad(f"source {source.id} starts at {segments[0].start_s}, expected 0")
        if segments and segments[-1].end_s != duration:
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
        if station.kind == "fixture" or station.height == "fixed":
            # Version 2: where a fixed source stands, at a height of its own.
            height = station.position[1] - recipe.dwelling.floor_y_m
            if station.kind != "fixture" or station.height != "fixed":
                bad(f"station {station.id} is a {station.kind} and says {station.height}")
            elif not 0.0 < height <= FIXTURE_HEIGHT_MAX_M:
                bad(f"fixture {station.id} is {height:.3f} m above the floor")
            elif floor is not None:
                from shapely.geometry import Point

                away = float(floor.walkable.distance(Point(*station.xz)))
                if away > FIXTURE_OUTSIDE_M + _EPS:
                    bad(f"fixture {station.id} is {away:.2f} m outside the dwelling")
            continue
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
        if not RAIL_PITCH_MIN_M - _EPS <= rail.pitch_m <= RAIL_PITCH_MAX_M + _EPS:
            bad(
                f"rail {rail.id} has pitch_m {rail.pitch_m}, outside "
                f"{RAIL_PITCH_MIN_M} to {RAIL_PITCH_MAX_M}"
            )
        elif rail.pitch_m != recipe.rails[0].pitch_m:
            bad(
                f"rail {rail.id} has pitch_m {rail.pitch_m}, and the recipe's first rail "
                f"{recipe.rails[0].pitch_m}"
            )
        if rail.a not in known or rail.b not in known or rail.a == rail.b:
            bad(f"rail {rail.id} does not join two stations ({rail.a!r}, {rail.b!r})")
            continue
        if "fixture" in (known[rail.a].kind, known[rail.b].kind):
            bad(f"rail {rail.id} ends at a fixture, where nobody walks")
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
        if source.attach is not None:
            continue
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
        if (segment.height == "fixed") != (station.kind == "fixture"):
            return f"{where} is {segment.height} at {station.id}, a {station.kind}"
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
    mouths = [anchor_at(recipe, source.id, times) for source in recipe.sources]
    # A carried source is where its carrier is, and a footstep is under a walker: neither
    # is a body to keep clear of. Its carrier is held apart in its stead.
    carried = [source.attach is not None for source in recipe.sources]
    for index, source in enumerate(recipe.sources):
        if carried[index]:
            continue
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
            if carried[other]:
                continue
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


# --------------------------------------------------------------------------
# version 2. Rule 12: what is carried and what is fixed
# --------------------------------------------------------------------------


def _kinds(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(12, message))

    by_id = {source.id: source for source in recipe.sources}
    stations = {station.id: station for station in recipe.stations}
    if sum(source.kind == "own_voice" for source in recipe.sources) > 1:
        bad("more than one source is the wearer's own voice")
    for source in recipe.sources:
        who = f"source {source.id}"
        if source.id == LISTENER:
            bad(f"a source is named {LISTENER!r}, the name an attachment gives the listener")
        attach = source.attach
        people = source.kind == "noise" and source.subtype in ("body", "steps")
        if attach is not None:
            if source.segments:
                bad(f"{who} is carried and has segments of its own")
            if attach.to != LISTENER:
                carrier = by_id.get(attach.to)
                if carrier is None or carrier.attach is not None or carrier.id == source.id:
                    bad(f"{who} is carried by {attach.to!r}, which is no source that walks")
                elif carrier.kind != "voice":
                    bad(f"{who} is carried by {attach.to}, a {carrier.kind} and not a person")
            if attach.at == "floor":
                stride = attach.stride_m or 0.0
                if not STRIDE_M[0] - _EPS <= stride <= STRIDE_M[1] + _EPS:
                    bad(f"{who} has a stride of {stride} m, outside {STRIDE_M[0]} to {STRIDE_M[1]}")
            elif attach.offset_m is not None:
                reach = math.sqrt(sum(v * v for v in attach.offset_m))
                if not MOUTH_OFFSET_M[0] - _EPS <= reach <= MOUTH_OFFSET_M[1] + _EPS:
                    bad(f"{who} is {reach:.3f} m from the centre of the head")
        if source.kind == "own_voice" and (
            attach is None or attach.to != LISTENER or attach.at != "mouth"
        ):
            bad(f"{who} is the wearer's own voice and is not carried at the listener's mouth")
        by_listener = attach is not None and attach.offset_m is not None
        if source.kind != "own_voice" and by_listener and not people:
            bad(f"{who} is a {source.kind} carried by the listener")
        if source.kind == "voice" and attach is not None:
            bad(f"{who} is a voice and is carried; a voice walks by itself")
        if people:
            want = "floor" if source.subtype == "steps" else "mouth"
            if attach is None or attach.at != want:
                bad(
                    f"{who} is somebody's noise ({source.subtype}) and is not carried at the {want}"
                )
        elif source.kind in ("media_voice", "noise"):
            # Every other noise is fixed: one station, the whole scene.
            if attach is not None:
                bad(f"{who} is a {source.subtype} and is carried; only a noise of somebody is")
            elif any(not isinstance(segment, Dwell) for segment in source.segments) or (
                len({segment.station for segment in source.segments if isinstance(segment, Dwell)})
                > 1
            ):
                bad(f"{who} is a {source.subtype} and moves; only a noise of somebody does")
        outside = source.kind == "noise" and source.subtype == "outside"
        if outside != (source.opening is not None):
            bad(f"{who}: a noise of outside names the opening it comes in by, and no other does")
        elif source.opening is not None and source.segments:
            first = source.segments[0]
            station = stations.get(first.station) if isinstance(first, Dwell) else None
            if (
                station is None
                or station.kind != "fixture"
                or station.object != source.opening.object
            ):
                bad(f"{who} is not at the fixture of its opening, {source.opening.object}")
    return out


# --------------------------------------------------------------------------
# rule 13: who is of the listener's conversation
# --------------------------------------------------------------------------


def _covers(intervals: Any, duration: float) -> str | None:
    """Why these intervals do not cover the scene in order, or ``None``."""
    if not intervals:
        return "says nothing"
    if intervals[0].start_s != 0.0 or intervals[-1].end_s != duration:
        return "does not run from the scene's start to its end"
    for index, interval in enumerate(intervals):
        if not interval.end_s > interval.start_s:
            return f"interval {index} does not end after it starts"
        if index and interval.start_s != intervals[index - 1].end_s:
            return f"has a gap or an overlap before interval {index}"
    return None


def _roles(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []

    def bad(message: str) -> None:
        out.append(Violation(13, message))

    mine = recipe.listener.conversation
    problem = _covers(mine, recipe.duration_s)
    if problem:
        bad(f"the listener's conversation {problem}")
        return out
    for source in recipe.sources:
        if source.kind != "voice":
            continue
        problem = _covers(source.roles, recipe.duration_s)
        if problem:
            bad(f"source {source.id}: its roles {problem.replace('says nothing', 'say nothing')}")
            continue
        edges = sorted({m.start_s for m in mine} | {r.start_s for r in source.roles})
        edges.append(recipe.duration_s)
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            middle = 0.5 * (start + end)
            group = next(m.group for m in mine if m.start_s <= middle < m.end_s)
            role = next(r for r in source.roles if r.start_s <= middle < r.end_s)
            together = role.group is not None and role.group == group
            if together != (role.role == "conversation"):
                bad(
                    f"source {source.id} is {role.role!r} in group {role.group!r} from "
                    f"t = {start:.3f} s, while the listener is in {group!r}"
                )
                break
    return out


# --------------------------------------------------------------------------
# rule 14: a sway stays where the low band holds
# --------------------------------------------------------------------------


def _sway(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    bodies: list[tuple[str, Any]] = [("the listener", recipe.listener.sway or ())]
    bodies += [(f"source {source.id}", source.sway) for source in recipe.sources]
    for who, sway in bodies:
        if any(part.amplitude < 0 or part.period_s < SWAY_PERIOD_MIN_S - _EPS for part in sway):
            out.append(
                Violation(
                    14,
                    f"{who} has a sway of negative amplitude or of a period under "
                    f"{SWAY_PERIOD_MIN_S} s",
                )
            )
            continue
        reach = sum(part.amplitude for part in sway if part.axis != "yaw")
        turn = sum(part.amplitude for part in sway if part.axis == "yaw")
        if reach > SWAY_RADIUS_M + _EPS:
            out.append(
                Violation(
                    14, f"{who} sways up to {reach:.3f} m, further than the {SWAY_RADIUS_M} m held"
                )
            )
        if turn > SWAY_YAW_DEG + _EPS:
            out.append(Violation(14, f"{who} sways up to {turn:.1f} degrees of yaw"))
    for source in recipe.sources:
        if source.attach is not None and source.sway:
            out.append(Violation(14, f"source {source.id} is carried and has a sway of its own"))
    return out


# --------------------------------------------------------------------------
# rule 15: a turn's effort is its level
# --------------------------------------------------------------------------


def _efforts(recipe: Recipe) -> list[Violation]:
    out: list[Violation] = []
    for source in recipe.sources:
        level = source.level_spl_1m_db
        if level is None or not 0.0 <= level <= 100.0:
            out.append(Violation(15, f"source {source.id} is {level} dB SPL at 1 m"))
            continue
        if source.kind not in VOICE_KINDS_V2:
            continue
        for index, interval in enumerate(source.activity):
            if interval.effort is None:
                continue
            low, high = effort_range_db(interval.effort)
            spoken = level + interval.gain_db
            if not low - 0.01 <= spoken <= high + 0.01:
                out.append(
                    Violation(
                        15,
                        f"source {source.id} activity {index} is {spoken:.2f} dB SPL at 1 m "
                        f"and says {interval.effort}",
                    )
                )
                break
    return out


# --------------------------------------------------------------------------
# rule 16: the conversation is heard
# --------------------------------------------------------------------------


def _audible(recipe: Recipe) -> list[Violation]:
    if recipe.scene is None:
        return []
    floor = recipe.scene.snr_floor_db
    worst = min(conversation_snr(recipe), key=lambda row: row[2], default=None)
    if worst is None or worst[2] >= floor - 0.01:
        return []
    interval = recipe.source(worst[0]).activity[worst[1]]
    return [
        Violation(
            16,
            f"source {worst[0]} speaks at t = {interval.start_s:.3f} s at {worst[2]:.1f} dB over "
            f"the noise at the listener, in free field; the scene's floor is {floor:.1f} dB",
        )
    ]


# --------------------------------------------------------------------------
# rule 17: the record of the gaze
# --------------------------------------------------------------------------


def _gaze(recipe: Recipe) -> list[Violation]:
    known = {source.id for source in recipe.sources}
    last = 0.0
    for index, interval in enumerate(recipe.listener.gaze):
        problem = None
        if not interval.end_s > interval.start_s:
            problem = "does not end after it starts"
        elif interval.start_s < last or interval.end_s > recipe.duration_s:
            problem = "overlaps the one before or lies outside the scene"
        elif interval.target is not None and interval.target not in known:
            problem = f"names no source of the recipe ({interval.target!r})"
        elif (interval.mode in ("talker", "glance", "event")) != (interval.target is not None):
            problem = f"is a {interval.mode} and " + (
                "names a target" if interval.target is not None else "names no target"
            )
        if problem:
            return [Violation(17, f"the listener's gaze interval {index} {problem}")]
        last = interval.end_s
    return []
