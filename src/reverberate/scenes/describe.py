"""A recipe in words: who goes where, how far, and how much of the time they speak."""

from __future__ import annotations

import numpy as np

from reverberate.scenes.kinematics import (
    listener_state,
    rail_arc_lengths,
    rail_length,
    seat_rail_heights,
    source_state,
)
from reverberate.scenes.recipe import Dwell, Recipe, Rise, Travel, canonical_bytes, recipe_sha256

__all__ = ["describe", "low_band_positions"]


def low_band_positions(recipe: Recipe) -> dict[str, int]:
    """How many source positions the low band is solved at for this recipe.

    A station dwelt at is one; a rail travelled is its samples; a seat a
    source rises at is the samples of its vertical rail. The ends a rail
    shares with a station are counted with the rail.
    """
    dwelt: set[tuple[str, str]] = set()
    rails: set[str] = set()
    risen: set[str] = set()
    for source in recipe.sources:
        for segment in source.segments:
            if isinstance(segment, Dwell):
                dwelt.add((segment.station, segment.height))
            elif isinstance(segment, Travel):
                rails.add(segment.rail)
            else:
                risen.add(segment.station)
    return {
        "stations": len(dwelt),
        "rail_samples": sum(len(rail_arc_lengths(recipe.rail(rail))) for rail in sorted(rails)),
        "seat_rail_samples": len(risen) * len(seat_rail_heights(recipe)),
    }


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
        f"  low band source positions: {sum(positions.values())} "
        f"({positions['stations']} stations, {positions['rail_samples']} on rails, "
        f"{positions['seat_rail_samples']} on seats' vertical rails)"
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
