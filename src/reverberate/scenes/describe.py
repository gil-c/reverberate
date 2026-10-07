"""A recipe in words: who goes where, how far, and how much of the time they speak."""

from __future__ import annotations

import numpy as np

from reverberate.scenes.kinematics import (
    listener_state,
    low_band_source_positions,
    rail_length,
    source_state,
)
from reverberate.scenes.levels import conversation_snr, in_conversation
from reverberate.scenes.recipe import Dwell, Recipe, Rise, Travel, canonical_bytes, recipe_sha256

__all__ = ["describe", "gaze_share", "low_band_positions"]


def gaze_share(recipe: Recipe) -> float:
    """Of the time somebody else of the conversation holds the floor, the share looked at.

    Who holds the floor is the voice of the listener's conversation whose
    turn started last; the listener looks at it while an interval of its
    ``gaze`` is a ``talker``'s and names that voice. Read every 50 ms.
    ``nan`` where nobody of the conversation ever talks.
    """
    grid = np.arange(0.0, recipe.duration_s, 0.05)
    holder = np.full(grid.size, -1, dtype=np.int64)
    turns = sorted(
        (interval.start_s, interval.end_s, number)
        for number, source in enumerate(recipe.sources)
        if source.kind == "voice"
        for interval, inside in zip(
            source.activity,
            in_conversation(source, np.array([a.start_s for a in source.activity])),
            strict=True,
        )
        if inside and interval.event == "turn"
    )
    for start, end, number in turns:
        holder[(grid >= start) & (grid < end)] = number
    looked = np.full(grid.size, -2, dtype=np.int64)
    index = {source.id: number for number, source in enumerate(recipe.sources)}
    for interval in recipe.listener.gaze:
        if interval.mode == "talker" and interval.target in index:
            looked[(grid >= interval.start_s) & (grid < interval.end_s)] = index[interval.target]
    talking = holder >= 0
    return float((looked[talking] == holder[talking]).mean()) if talking.any() else float("nan")


def _second(recipe: Recipe) -> list[str]:
    """What a recipe of version 2 says beyond the first: the scene, the levels, the gaze."""
    scene = recipe.scene
    assert scene is not None
    lines = []
    ratios = np.array([row[2] for row in conversation_snr(recipe)])
    heard = (
        f"least {ratios.min():.1f} dB, median {np.median(ratios):.1f} dB over {ratios.size} turns"
        if ratios.size
        else "no turn"
    )
    lines.append(
        f"  calmness {scene.calmness:g}; the conversation over the noise at the listener, in free "
        f"field: {heard} (floor {scene.snr_floor_db:g} dB)"
    )
    said = [a for s in recipe.sources if s.kind in ("voice", "own_voice") for a in s.activity]
    efforts = [
        f"{sum(a.effort == name for a in said)} {name}"
        for name in sorted({a.effort or "" for a in said})
    ]
    events = [
        f"{sum(a.event == name for a in said)} {name}"
        for name in ("turn", "backchannel", "laughter")
    ]
    lines.append(f"  voice intervals: {', '.join(events)}; efforts: {', '.join(efforts) or 'none'}")
    spans = ", ".join(
        f"{m.group or 'nobody'} {m.start_s:.0f} to {m.end_s:.0f} s"
        for m in recipe.listener.conversation
    )
    changes = sum(len(s.roles) - 1 for s in recipe.sources if s.kind == "voice")
    lines.append(
        f"  the listener talks with: {spans}; {changes} change(s) of role among the voices"
    )
    modes: dict[str, float] = {}
    for interval in recipe.listener.gaze:
        modes[interval.mode] = modes.get(interval.mode, 0.0) + interval.end_s - interval.start_s
    shares = ", ".join(
        f"{name} {100 * t / recipe.duration_s:.0f} %" for name, t in sorted(modes.items())
    )
    lines.append(
        f"  gaze: on the talker {100 * gaze_share(recipe):.0f} % of the time somebody else of the "
        f"conversation talks; of the scene: {shares}"
    )
    return lines


def low_band_positions(recipe: Recipe, rail_positions: int = 2) -> dict[str, int]:
    """How many source positions the low band is solved at for this recipe.

    ``all`` is every position the sources pass through, each once: a station
    dwelt at, the samples of a rail travelled, the samples of the vertical
    rail of a seat a source rises at; ``stations``, ``rail_samples`` and
    ``seat_rail_samples`` split it, an end a rail shares with a station being
    the rail's. ``audible`` is how many of them a source is audible at, which
    is what a trace solves (:func:`~.kinematics.low_band_source_positions`).

    ``rail_positions`` is the solved positions a source on a rail reads, as
    a trace's ``--rail-positions``: two are the samples either side of it.
    With more, ``audible`` is counted by the trace's own plan
    (:func:`reverberate.trace.plan.tracks_of`), which reads the nearest
    that many at every audible step, so that this count, the generator
    panel's and a dry run's are one number.
    """
    every = low_band_source_positions(recipe, audible_only=False)
    if int(rail_positions) == 2:
        heard = low_band_source_positions(recipe, audible_only=True).count
    else:
        from reverberate.trace.plan import Profile, tracks_of

        tracks = tracks_of(recipe, Profile(rail_positions=int(rail_positions)))
        heard = int(tracks.positions.shape[0])
    return {**every.counts(), "all": every.count, "audible": heard}


def describe(recipe: Recipe, rail_positions: int = 2) -> str:
    """A summary a person reads: per source, stations, metres, share active.

    ``rail_positions`` counts the low band's positions as a trace of that
    ``--rail-positions`` solves them.
    """
    duration = recipe.duration_s
    times = np.arange(0.0, duration + 1e-9, 0.5)
    head = listener_state(recipe, times).position
    size = len(canonical_bytes(recipe))
    kinds = [station.kind for station in recipe.stations]
    metres = sum(rail_length(rail) for rail in recipe.rails)
    lines = [
        f"recipe {recipe_sha256(recipe)}",
        f"  {recipe.dwelling.name} ({recipe.dwelling.scene_id}), seed {recipe.seed}, "
        f"{duration:.1f} s, {size} bytes canonical",
        f"  {len(recipe.stations)} stations in use ({kinds.count('seat')} seats, "
        f"{kinds.count('stand')} standing, {kinds.count('waypoint')} doors), "
        f"{len(recipe.rails)} rails, {metres:.1f} m of rail",
    ]
    positions = low_band_positions(recipe, rail_positions)
    pitch = recipe.rails[0].pitch_m if recipe.rails else 0.08
    lines.append(
        f"  low band source positions: {positions['all']} "
        f"({positions['stations']} stations, {positions['rail_samples']} on rails, "
        f"{positions['seat_rail_samples']} on seats' vertical rails), "
        f"{positions['audible']} where a source is audible"
        # Said only when it is not the first rule, whose line stays as it was.
        + (
            ""
            if int(rail_positions) == 2 and pitch == 0.08
            else f" (rails every {pitch:g} m, {int(rail_positions)} positions read)"
        )
    )
    spoken = moving = 0.0
    for source in recipe.sources:
        if source.kind in ("noise", "media_voice"):
            continue
        for interval in source.activity:
            spoken += interval.end_s - interval.start_s
            for segment in source.segments:
                if not isinstance(segment, Dwell):
                    moving += max(
                        0.0,
                        min(interval.end_s, segment.end_s) - max(interval.start_s, segment.start_s),
                    )
    lines.append(
        f"  speech on the move: {moving:.1f} s of {spoken:.1f} s spoken "
        f"({100 * moving / max(spoken, 1e-9):.1f} %)"
    )
    if recipe.generator is not None:
        lines.append(f"  generator {recipe.generator.name} {recipe.generator.version}")
    if recipe.schema_version >= 2:
        lines += _second(recipe)

    frames = recipe.listener.keyframes
    walked = sum(
        float(np.hypot(b.position[0] - a.position[0], b.position[2] - a.position[2]))
        for a, b in zip(frames[:-1], frames[1:], strict=True)
    )
    # A rest names its station; in version 2 a standing spot is named too, and is no seat.
    station_kind = {station.id: station.kind for station in recipe.stations}
    seated = sum(
        b.t_s - a.t_s
        for a, b in zip(frames[:-1], frames[1:], strict=True)
        if a.station is not None
        and a.station == b.station
        and station_kind.get(a.station) != "stand"
    )
    seats = sorted(
        {f.station for f in frames if f.station and station_kind.get(f.station) != "stand"}
    )
    lines.append(
        f"listener: {len(frames)} keyframes, {walked:.1f} m walked, "
        f"{100 * seated / duration:.0f} % seated, seats {', '.join(seats) or 'none'}"
    )

    lines.append(
        f"{'source':<12} {'kind':<18} {'stn':>3} {'metres':>8} {'seated':>7} {'active':>7} "
        f"{'spurts':>6}   to the head, m (min / median / max)"
    )
    for source in recipe.sources:
        visited: list[str] = []
        travelled = 0.0
        sat = 0.0
        for segment in source.segments:
            if isinstance(segment, Travel):
                travelled += rail_length(recipe.rail(segment.rail))
            elif isinstance(segment, Dwell):
                if not visited or visited[-1] != segment.station:
                    visited.append(segment.station)
                if segment.height == "seated":
                    sat += segment.end_s - segment.start_s
            elif isinstance(segment, Rise):
                travelled += recipe.heights.standing_m - recipe.heights.seated_m
        active = sum(interval.end_s - interval.start_s for interval in source.activity)
        gap = np.linalg.norm(source_state(recipe, source.id, times).position - head, axis=1)
        kind = source.kind if source.subtype is None else f"{source.kind}/{source.subtype}"
        lines.append(
            f"{source.id:<12} {kind:<18} {len(set(visited)):>3} {travelled:>8.1f} "
            f"{100 * sat / duration:>6.0f}% {100 * active / duration:>6.0f}% "
            f"{len(source.activity):>6}   {gap.min():.2f} / {np.median(gap):.2f} / {gap.max():.2f}"
        )
        lines.append(f"             visits: {' > '.join(visited)}")
    return "\n".join(lines)
