"""A recipe in words: who goes where, how far, and how much of the time they speak."""

from __future__ import annotations

import numpy as np

from reverberate.scenes.kinematics import (
    listener_state,
    low_band_source_positions,
    rail_length,
    source_state,
)
from reverberate.scenes.recipe import Dwell, Recipe, Rise, Travel, canonical_bytes, recipe_sha256

__all__ = ["describe", "low_band_positions"]


def low_band_positions(recipe: Recipe) -> dict[str, int]:
    """How many source positions the low band is solved at for this recipe.

    ``all`` is every position the sources pass through, each once: a station
    dwelt at, the samples of a rail travelled, the samples of the vertical
    rail of a seat a source rises at; ``stations``, ``rail_samples`` and
    ``seat_rail_samples`` split it, an end a rail shares with a station being
    the rail's. ``audible`` is how many of them a source is audible at, which
    is what a trace solves (:func:`~.kinematics.low_band_source_positions`).
    """
    every = low_band_source_positions(recipe, audible_only=False)
    heard = low_band_source_positions(recipe, audible_only=True)
    return {**every.counts(), "all": every.count, "audible": heard.count}


def describe(recipe: Recipe) -> str:
    """A summary a person reads: per source, stations, metres, share active."""
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
    positions = low_band_positions(recipe)
    lines.append(
        f"  low band source positions: {positions['all']} "
        f"({positions['stations']} stations, {positions['rail_samples']} on rails, "
        f"{positions['seat_rail_samples']} on seats' vertical rails), "
        f"{positions['audible']} where a source is audible"
    )
    spoken = moving = 0.0
    for source in recipe.sources:
        if source.kind == "noise":
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

    frames = recipe.listener.keyframes
    walked = sum(
        float(np.hypot(b.position[0] - a.position[0], b.position[2] - a.position[2]))
        for a, b in zip(frames[:-1], frames[1:], strict=True)
    )
    seated = sum(
        b.t_s - a.t_s
        for a, b in zip(frames[:-1], frames[1:], strict=True)
        if a.station is not None and a.station == b.station
    )
    seats = sorted({frame.station for frame in frames if frame.station is not None})
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
