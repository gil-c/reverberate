"""Where everything is at time ``t``: the one definition, for every reader.

The validation, the page's scene view and the trace all ask this module, so a
source cannot be in one place for the rule that checks it and in another for
the engine that renders it. Everything is vectorised over an array of times.

**A source's yaw has a memory.** The format gives a facing and a turn rate:
the yaw starts at the facing's value and follows it by the shorter way round,
no faster than the rate. That is a recurrence, and a recurrence needs a step
to be the same number for two readers. The step is :data:`YAW_STEP_S`, the
pack's 50 ms: the yaw is advanced on that grid from the scene's start and is
linear between two grid instants. It is not wrapped, like the listener's.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from reverberate.scenes.recipe import Dwell, Rail, Recipe, Rise, Source

__all__ = [
    "AUDIBLE_TAIL_S",
    "YAW_STEP_S",
    "ListenerState",
    "LowBandPositions",
    "SourceState",
    "audible_steps",
    "listener_state",
    "low_band_source_positions",
    "polyline_length",
    "rail_arc_lengths",
    "rail_length",
    "rail_samples",
    "sample_times",
    "seat_rail_heights",
    "source_state",
    "yaw_of_direction",
]

#: The step a source's yaw is advanced on, and the step rules 6 to 9 are
#: sampled at: the scene pack's.
YAW_STEP_S = 0.05

#: A source stays audible this long after an activity interval ends: the
#: length of a low band response, the pack's ``low_samples / low_sample_rate_hz``.
AUDIBLE_TAIL_S = 1.2

#: A last sample closer than this to ``b`` is ``b``; a micrometre.
_SAME_M = 1e-6


def yaw_of_direction(dx: np.ndarray | float, dz: np.ndarray | float) -> np.ndarray:
    """The yaw, in degrees, that faces along ``(dx, dz)``: 0 is ``+x``, +90 is ``-z``."""
    yaw: np.ndarray = np.degrees(
        np.arctan2(0.0 - np.asarray(dz, dtype=float), np.asarray(dx, dtype=float))
    )
    return yaw


def _cumulative(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def polyline_length(points: np.ndarray) -> float:
    return float(_cumulative(np.asarray(points, dtype=float))[-1])


def rail_length(rail: Rail) -> float:
    return polyline_length(np.asarray(rail.points, dtype=float))


def _arc_lengths(length: float, pitch: float) -> np.ndarray:
    count = int(np.floor(length / pitch + 1e-9))
    arcs = np.arange(count + 1, dtype=float) * pitch
    if length - arcs[-1] > _SAME_M:
        arcs = np.append(arcs, length)
    else:
        arcs[-1] = length
    return arcs


def rail_arc_lengths(rail: Rail) -> np.ndarray:
    """Arc lengths from ``a`` of the rail's sampled positions.

    ``j * pitch_m`` for ``j = 0 .. floor(L / pitch_m)``, then ``L`` itself;
    when the last multiple is ``L`` to a micrometre, ``b`` is not repeated.
    """
    return _arc_lengths(rail_length(rail), rail.pitch_m)


def _along(points: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    cumulative = _cumulative(points)
    return np.stack(
        [np.interp(arcs, cumulative, points[:, 0]), np.interp(arcs, cumulative, points[:, 1])],
        axis=-1,
    )


def rail_samples(rail: Rail) -> np.ndarray:
    """The ``(x, z)`` the low band is solved at along a rail, ``[sample, 2]``.

    A function of the rail and of nothing else, so a cache can be keyed on it.
    Their height is the standing one.
    """
    return _along(np.asarray(rail.points, dtype=float), rail_arc_lengths(rail))


def seat_rail_heights(recipe: Recipe) -> np.ndarray:
    """Heights above the floor of the samples of a seat's vertical rail.

    From the seated end, every ``0.08`` m, then the standing height itself. The
    same for every seat of a recipe.
    """
    pitch = recipe.rails[0].pitch_m if recipe.rails else 0.08
    span = recipe.heights.standing_m - recipe.heights.seated_m
    return recipe.heights.seated_m + _arc_lengths(span, pitch)


def sample_times(recipe: Recipe, step_s: float = YAW_STEP_S) -> np.ndarray:
    """Every ``step_s`` from the start, and the end of the scene."""
    count = int(np.floor(recipe.duration_s / step_s + 1e-9))
    times = np.arange(count + 1, dtype=float) * step_s
    if recipe.duration_s - times[-1] > 1e-9:
        times = np.append(times, recipe.duration_s)
    else:
        times[-1] = recipe.duration_s
    return times


# --------------------------------------------------------------------------
# the listener
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ListenerState:
    """The head at each time. Angles in degrees, the yaw not wrapped."""

    t: np.ndarray
    position: np.ndarray  # [time, 3]
    yaw_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    #: Above the storey's floor.
    height_m: np.ndarray
    #: Index of the keyframe at or before each time (the last interval for the end).
    keyframe: np.ndarray


def listener_state(recipe: Recipe, t: ArrayLike) -> ListenerState:
    """The listener's head at ``t``: linear between keyframes, held outside them."""
    times = np.atleast_1d(np.asarray(t, dtype=float))
    frames = recipe.listener.keyframes
    knots = np.array([frame.t_s for frame in frames])
    positions = np.array([frame.position for frame in frames], dtype=float)
    position = np.stack([np.interp(times, knots, positions[:, axis]) for axis in range(3)], axis=-1)
    index = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, max(len(frames) - 2, 0))
    return ListenerState(
        t=times,
        position=position,
        yaw_deg=np.interp(times, knots, [frame.yaw_deg for frame in frames]),
        pitch_deg=np.interp(times, knots, [frame.pitch_deg for frame in frames]),
        roll_deg=np.interp(times, knots, [frame.roll_deg for frame in frames]),
        height_m=position[:, 1] - recipe.dwelling.floor_y_m,
        keyframe=index,
    )


# --------------------------------------------------------------------------
# a source
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceState:
    """A source's mouth at each time.

    ``sample_a``, ``sample_b`` and ``weight`` say which solved positions the
    source lies between: the low band there is ``(1 - weight)`` of the first
    and ``weight`` of the second. On a ``travel`` they index
    :func:`rail_samples` of the segment's rail; on a ``rise`` they index
    :func:`seat_rail_heights`; on a ``dwell`` the source is at its station, or
    at one end of the seat's vertical rail, and both are ``-1``.
    """

    t: np.ndarray
    position: np.ndarray  # [time, 3]
    yaw_deg: np.ndarray
    #: Above the storey's floor.
    height_m: np.ndarray
    #: Index into ``source.segments``.
    segment: np.ndarray
    sample_a: np.ndarray
    sample_b: np.ndarray
    weight: np.ndarray


def _ease(profile: str, tau: np.ndarray) -> np.ndarray:
    if profile == "smoothstep":
        return 3.0 * tau**2 - 2.0 * tau**3
    return tau


def _progress(start: float, end: float, times: np.ndarray) -> np.ndarray:
    if end <= start:
        return np.ones_like(times)
    return np.clip((times - start) / (end - start), 0.0, 1.0)


def _between(arcs: np.ndarray, arc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.clip(np.searchsorted(arcs, arc, side="right") - 1, 0, len(arcs) - 2)
    span = arcs[low + 1] - arcs[low]
    weight = np.clip((arc - arcs[low]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    return low, low + 1, weight


def _segment_index(source: Source, times: np.ndarray) -> np.ndarray:
    starts = np.array([segment.start_s for segment in source.segments])
    return np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)


def _place(
    recipe: Recipe, source: Source, times: np.ndarray, listener_xz: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Position, facing target (NaN where the yaw holds), segment, samples, weight."""
    floor = recipe.dwelling.floor_y_m
    heights = recipe.heights
    index = _segment_index(source, times)
    position = np.zeros((times.size, 3))
    target = np.full(times.size, np.nan)
    sample_a = np.full(times.size, -1, dtype=np.int64)
    sample_b = np.full(times.size, -1, dtype=np.int64)
    weight = np.zeros(times.size)
    for number, segment in enumerate(source.segments):
        here = index == number
        if not here.any():
            continue
        moment = times[here]
        tangent: np.ndarray | None = None
        if isinstance(segment, Dwell):
            x, z = recipe.station(segment.station).xz
            xz = np.tile([x, z], (moment.size, 1))
            height = np.full(moment.size, heights.of(segment.height))
        elif isinstance(segment, Rise):
            x, z = recipe.station(segment.station).xz
            xz = np.tile([x, z], (moment.size, 1))
            span = heights.standing_m - heights.seated_m
            eased = _ease("smoothstep", _progress(segment.start_s, segment.end_s, moment))
            arc = span * (eased if segment.to == "standing" else 1.0 - eased)
            height = heights.seated_m + arc
            rungs = seat_rail_heights(recipe) - heights.seated_m
            sample_a[here], sample_b[here], weight[here] = _between(rungs, arc)
        else:
            rail = recipe.rail(segment.rail)
            points = np.asarray(rail.points, dtype=float)
            length = polyline_length(points)
            eased = _ease(segment.profile, _progress(segment.start_s, segment.end_s, moment))
            forward = segment.origin == rail.a
            arc = length * (eased if forward else 1.0 - eased)
            xz = _along(points, arc)
            height = np.full(moment.size, heights.standing_m)
            sample_a[here], sample_b[here], weight[here] = _between(rail_arc_lengths(rail), arc)
            cumulative = _cumulative(points)
            leg = np.clip(np.searchsorted(cumulative, arc, side="right") - 1, 0, len(points) - 2)
            tangent = (points[leg + 1] - points[leg]) * (1.0 if forward else -1.0)
        position[here, 0] = xz[:, 0]
        position[here, 1] = floor + height
        position[here, 2] = xz[:, 1]
        if isinstance(segment, Rise):
            continue
        facing = segment.facing
        if facing.mode == "fixed":
            target[here] = facing.yaw_deg
        elif facing.mode == "travel" and tangent is not None:
            target[here] = yaw_of_direction(tangent[:, 0], tangent[:, 1])
        elif facing.mode == "listener" and listener_xz is not None:
            delta = listener_xz[here] - xz
            aim = yaw_of_direction(delta[:, 0], delta[:, 1])
            target[here] = np.where(np.linalg.norm(delta, axis=1) > _SAME_M, aim, np.nan)
    return position, target, index, sample_a, sample_b, weight


def _follow(target: np.ndarray, grid: np.ndarray, rate_deg_s: float) -> np.ndarray:
    """The yaw on the grid: towards the target by the shorter way, at most at the rate."""
    known = np.flatnonzero(~np.isnan(target))
    yaw = np.zeros(grid.size)
    current = float(target[known[0]]) if known.size else 0.0
    yaw[0] = current
    reach = np.diff(grid) * max(rate_deg_s, 0.0)
    values = target.tolist()
    reaches = reach.tolist()
    for step in range(1, grid.size):
        aim = values[step]
        if aim == aim:  # not NaN: a facing is in force
            turn = (aim - current + 180.0) % 360.0 - 180.0
            limit = reaches[step - 1]
            current += max(-limit, min(limit, turn))
        yaw[step] = current
    return yaw


class _Tracks:
    """The yaw of every source of one recipe on the 50 ms grid, computed once."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.grid = sample_times(recipe)
        self._listener_xz = listener_state(recipe, self.grid).position[:, [0, 2]]
        self._yaw: dict[str, np.ndarray] = {}

    def yaw(self, source: Source) -> np.ndarray:
        if source.id not in self._yaw:
            _, target, *_ = _place(self.recipe, source, self.grid, self._listener_xz)
            self._yaw[source.id] = _follow(target, self.grid, source.turn_rate_deg_s)
        return self._yaw[source.id]


#: A few recipes' tracks, each kept with its recipe so an ``id`` is not reused.
_CACHE: OrderedDict[int, _Tracks] = OrderedDict()
_CACHE_SIZE = 4


def _tracks(recipe: Recipe) -> _Tracks:
    key = id(recipe)
    found = _CACHE.get(key)
    if found is None or found.recipe is not recipe:
        found = _Tracks(recipe)
        _CACHE[key] = found
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return found


def source_state(recipe: Recipe, source_id: str, t: ArrayLike) -> SourceState:
    """Where the source's mouth is at ``t``, which way it faces, and between which samples."""
    source = recipe.source(source_id)
    times = np.atleast_1d(np.asarray(t, dtype=float))
    position, _, index, sample_a, sample_b, weight = _place(recipe, source, times, None)
    tracks = _tracks(recipe)
    return SourceState(
        t=times,
        position=position,
        yaw_deg=np.interp(times, tracks.grid, tracks.yaw(source)),
        height_m=position[:, 1] - recipe.dwelling.floor_y_m,
        segment=index,
        sample_a=sample_a,
        sample_b=sample_b,
        weight=weight,
    )


# --------------------------------------------------------------------------
# what the low band is solved from
# --------------------------------------------------------------------------


def audible_steps(
    recipe: Recipe, source_id: str, *, step_s: float = YAW_STEP_S, tail_s: float = AUDIBLE_TAIL_S
) -> np.ndarray:
    """Per step of :func:`sample_times`, whether the source is audible: the pack's ``audible``.

    A step is audible when it lies in an activity interval or within
    ``tail_s`` after one (``docs/formats/scene-pack.md``): the sound a source
    made still rings for the length of a response after it falls silent.
    """
    times = sample_times(recipe, step_s)
    heard = np.zeros(times.size, dtype=bool)
    for interval in recipe.source(source_id).activity:
        heard |= (times >= interval.start_s - 1e-9) & (times <= interval.end_s + tail_s + 1e-9)
    return heard


@dataclass(frozen=True)
class LowBandPositions:
    """The source positions the band under the crossover is solved from, each once."""

    #: ``[position, 3]`` in metres, scene frame, on whole millimetres.
    positions: np.ndarray
    #: Per position: ``"station"``, ``"rail"`` or ``"seat_rail"``. An end a rail
    #: shares with a station is the rail's.
    kind: tuple[str, ...]
    #: The rows each source reads, sorted.
    by_source: dict[str, tuple[int, ...]]

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def counts(self) -> dict[str, int]:
        return {
            "stations": self.kind.count("station"),
            "rail_samples": self.kind.count("rail"),
            "seat_rail_samples": self.kind.count("seat_rail"),
        }


_KIND_RANK = {"rail": 0, "seat_rail": 1, "station": 2}


def low_band_source_positions(
    recipe: Recipe,
    *,
    audible_only: bool = True,
    step_s: float = YAW_STEP_S,
    tail_s: float = AUDIBLE_TAIL_S,
) -> LowBandPositions:
    """Every position a wave solve is needed from for this recipe.

    A source at rest is at its station, at its height; on a ``travel`` it
    lies between two of :func:`rail_samples`; on a ``rise`` between two of
    :func:`seat_rail_heights`. With ``audible_only`` a position counts only
    if some audible step (:func:`audible_steps`) reads it, the two samples
    either side of the source at that step: a rail walked in silence costs
    nothing. Without it every station dwelt at and every sample of every
    rail travelled counts, heard or not, which is what a cache filled once
    per rail would hold.

    Positions are told apart to the millimetre, so a rail's end and the
    standing station it starts from are one solve.
    """
    floor = recipe.dwelling.floor_y_m
    standing = floor + recipe.heights.standing_m
    rungs = floor + seat_rail_heights(recipe)
    found: dict[tuple[int, int, int], tuple[int, str]] = {}
    by_source: dict[str, set[tuple[int, int, int]]] = {}

    def add(mine: set[tuple[int, int, int]], points: np.ndarray, kind: str) -> None:
        for point in np.asarray(points, dtype=float).reshape(-1, 3):
            key = (
                round(float(point[0]) * 1000.0),
                round(float(point[1]) * 1000.0),
                round(float(point[2]) * 1000.0),
            )
            held = found.get(key)
            if held is None:
                found[key] = (len(found), kind)
            elif _KIND_RANK[kind] < _KIND_RANK[held[1]]:
                found[key] = (held[0], kind)
            mine.add(key)

    def on_rail(rail: Rail, chosen: np.ndarray) -> np.ndarray:
        flat = rail_samples(rail)[chosen]
        return np.stack([flat[:, 0], np.full(flat.shape[0], standing), flat[:, 1]], axis=1)

    for source in recipe.sources:
        mine = by_source.setdefault(source.id, set())
        if audible_only:
            times = sample_times(recipe, step_s)
            heard = audible_steps(recipe, source.id, step_s=step_s, tail_s=tail_s)
            if not heard.any():
                continue
            position, _, index, sample_a, sample_b, weight = _place(
                recipe, source, times[heard], None
            )
        for number, segment in enumerate(source.segments):
            if audible_only:
                here = index == number
                if not here.any():
                    continue
                # Slot 1 is read only where it has a weight, slot 0 only where it has one.
                chosen = np.unique(
                    np.concatenate(
                        [sample_a[here][weight[here] < 1.0], sample_b[here][weight[here] > 0.0]]
                    )
                )
            if isinstance(segment, Dwell):
                x, z = recipe.station(segment.station).xz
                add(mine, np.array([x, floor + recipe.heights.of(segment.height), z]), "station")
            elif isinstance(segment, Rise):
                x, z = recipe.station(segment.station).xz
                heights = rungs[chosen] if audible_only else rungs
                points = np.stack(
                    [np.full(heights.size, x), heights, np.full(heights.size, z)], axis=1
                )
                add(mine, points, "seat_rail")
            else:
                rail = recipe.rail(segment.rail)
                every = np.arange(len(rail_arc_lengths(rail)))
                add(mine, on_rail(rail, chosen if audible_only else every), "rail")
    order = sorted(found, key=lambda key: found[key][0])
    row = {key: n for n, key in enumerate(order)}
    return LowBandPositions(
        positions=np.array(order, dtype=float).reshape(-1, 3) / 1000.0,
        kind=tuple(found[key][1] for key in order),
        by_source={
            name: tuple(sorted(row[key] for key in keys)) for name, keys in by_source.items()
        },
    )
