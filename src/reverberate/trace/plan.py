"""What a recipe asks of the machine, decided on the laptop before anything is rented.

A plan is pure: the recipe, the occluders of the dwelling's mirror scene and
nothing else. It holds

- **the tracks**: the listener and every source at every step of the pack,
  from the recipe's own kinematics, and the solved positions each audible
  step of a source lies between (:func:`tracks_of`);
- **the listening cells**: where an order 7 array must stand for the band
  under the crossover (:func:`listening_cells`). One at every place the
  listener rests, at the height it has there; one every 0.15 m of its path;
  and more where the cell rule of ``spatial.translate`` asks for them, which
  is where a source comes close: a cell may be read from 0.15 of its
  distance to the source at most, so two cells 0.15 m apart cannot both
  serve a head 0.5 m from a mouth. There the path is sampled every 0.10 m,
  and where that is not enough a cell is put on the head itself;
- **the pairs**: the (source position, cell) couples some audible step reads,
  and no other (:func:`assign`, :func:`pairs_of`);
- **the tail's sites and cells** (:func:`tail_sites_of`, :func:`tail_cells`);
- **the cost**, before any rental (:func:`estimate`).

The machine runs :func:`assign` again, on the centres the arrays really got,
so the plan and the pack are one rule applied twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from reverberate.mirror.tails import SPACING_M, TailSites, source_weights, tail_sites
from reverberate.render.pack import STEP_S
from reverberate.scenes import (
    Recipe,
    audible_steps,
    listener_state,
    low_band_source_positions,
    rail_samples,
    recipe_sha256,
    seat_rail_heights,
    source_state,
)
from reverberate.scenes.recipe import Rise, Travel
from reverberate.spatial.lowband import solve_fmax_hz
from reverberate.spatial.translate import (
    EXACT_UNDER_M,
    FUSE_WITHIN_M,
    MODE_EXACT,
    MODE_FUSED,
    MODE_INAUDIBLE,
    MODE_TRANSLATED,
    SOURCE_SHARE,
    choose_cells,
    clearance_m,
    serving_radius_m,
)

__all__ = [
    "ARRAY_RADIUS_M",
    "DENSE_PITCH_M",
    "PATH_PITCH_M",
    "Assignment",
    "CellSet",
    "Patch",
    "Plan",
    "Profile",
    "SourceTrack",
    "Tracks",
    "assign",
    "busiest_window",
    "estimate",
    "listening_cells",
    "make_plan",
    "pairs_of",
    "patch_cells",
    "tail_cells",
    "tail_sites_of",
    "tracks_of",
]

#: Cells along the listener's path, and where a near source asks for more.
PATH_PITCH_M = 0.15
DENSE_PITCH_M = 0.10
#: A path sample this close to a cell already there is that cell.
MERGE_M = 0.03
#: The ball an array needs free of surfaces and of sources: twelve steps of
#: the grid to 1500 Hz (``accel.pairs.OUTER_RADIUS_M``).
ARRAY_RADIUS_M = 0.26
#: A source slot's weight under this is no weight.
WEIGHT_FLOOR = 1e-9

#: ``/cells/kind`` of the format.
KIND_SEAT, KIND_ADDED = 1, 2
#: Why a cell is there: a place the listener rests, its path, the denser
#: rule, the head itself, the validation patch.
ORIGIN_REST, ORIGIN_PATH, ORIGIN_DENSE, ORIGIN_HEAD, ORIGIN_PATCH = 0, 1, 2, 3, 4

#: The validation patch: a square of this side, sampled this finely, at both heights.
PATCH_SIDE_M = 0.80
PATCH_PITCH_M = 0.04
#: The patch's source stands this far from its centre or further, as the dense line's did.
PATCH_SOURCE_M = 1.2

# The mirror's projection for a card (``docs/adr/0016-appendix-moving-mirror-cost.md``):
# none of it measured on a card. A step-pair of the batched trace, a distance
# field of the diffracted onset on the host, a source site's rays, and a
# pair's levelling, taken as a lattice point's render (138 s for 437 points).
CARD_PAIR_S = 1.0e-3
ONSET_FIELD_S = 0.12
RAYS_SITE_S = 16.0
LEVEL_PAIR_S = 0.3
#: Provisioning, the bundle's push and the grid: what a rental pays before it computes.
FIXED_S = 900.0
#: The first whole fetch of this project: 21 GB in 45 minutes.
FETCH_BYTES_PER_S = 7.8e6
PAIR_BYTES = 64 * 4800 * 4
HISTOGRAM_BYTES = 286_000


@dataclass(frozen=True)
class Profile:
    """What of the recipe is traced: all of it, or a smoke run's first seconds and sources."""

    #: This many seconds of the scene alone; ``None`` for the whole of it.
    seconds: float | None = None
    #: At most this many sources, the moving ones first; ``None`` for all.
    sources: int | None = None
    #: The dense patch of the off-line translation validation, solved with the scene.
    patch: bool = False
    #: Where the window starts in the scene. Not zero, the pack's step ``k`` is
    #: the scene at ``start_s + k step_s``: a pack to exercise the stages on a
    #: stretch where something moves, not one to play the recipe's clips with.
    start_s: float = 0.0

    @property
    def smoke(self) -> bool:
        return self.seconds is not None or self.sources is not None

    def record(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "sources": self.sources,
            "patch": self.patch,
            "start_s": self.start_s,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Profile:
        return cls(
            record.get("seconds"),
            record.get("sources"),
            bool(record.get("patch", False)),
            float(record.get("start_s", 0.0)),
        )


# --------------------------------------------------------------------------
# the tracks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTrack:
    """One source at every step, and the solved positions it lies between."""

    id: str
    position: np.ndarray
    yaw_deg: np.ndarray
    audible: np.ndarray
    #: ``[step, 2]``: rows of :attr:`Tracks.positions`; ``-1`` for a slot not read.
    slot: np.ndarray
    #: ``[step]``: the weight of slot 1, zero where it is not read.
    weight: np.ndarray
    #: ``[step]``: whether the source is on a rail or rising, not at rest.
    moving: np.ndarray
    #: ``[step]``: index into the recipe's segments of the source.
    segment: np.ndarray


@dataclass(frozen=True)
class Tracks:
    """The scene at every step of the pack."""

    duration_s: float
    #: The scene's time at step 0: zero but for a smoke run's window.
    start_s: float
    listener: np.ndarray
    #: ``[step, 3]``: yaw, pitch and roll in degrees, the recipe's convention.
    orientation: np.ndarray
    sources: dict[str, SourceTrack]
    #: ``[position, 3]``: the source positions the low band is solved from, in whole millimetres.
    positions: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.listener.shape[0])

    @property
    def times(self) -> np.ndarray:
        return self.start_s + np.arange(self.steps) * STEP_S


def _millimetres(points: np.ndarray) -> np.ndarray:
    return np.asarray(np.rint(np.asarray(points, dtype=float) * 1000.0), dtype=np.int64)


def _slots(recipe: Recipe, source_id: str, times: np.ndarray) -> tuple[Any, np.ndarray, Any, Any]:
    """The source's state, its two solved positions at every step, the weight, whether it moves."""
    source = recipe.source(source_id)
    state = source_state(recipe, source_id, times)
    floor = recipe.dwelling.floor_y_m
    standing = floor + recipe.heights.standing_m
    rungs = floor + seat_rail_heights(recipe)
    slots = np.repeat(state.position[:, None, :], 2, axis=1)
    weight = np.array(state.weight, dtype=float)
    moving = np.zeros(times.size, dtype=bool)
    for number, segment in enumerate(source.segments):
        here = state.segment == number
        if not here.any():
            continue
        rows = (state.sample_a[here], state.sample_b[here])
        if isinstance(segment, Travel):
            flat = rail_samples(recipe.rail(segment.rail))
            for slot in range(2):
                at = flat[rows[slot]]
                slots[here, slot] = np.stack(
                    [at[:, 0], np.full(at.shape[0], standing), at[:, 1]], axis=1
                )
            moving[here] = True
        elif isinstance(segment, Rise):
            x, z = recipe.station(segment.station).xz
            for slot in range(2):
                height = rungs[rows[slot]]
                slots[here, slot] = np.stack(
                    [np.full(height.size, x), height, np.full(height.size, z)], axis=1
                )
            moving[here] = True
        else:
            weight[here] = 0.0
    # A source on a solved position reads it alone, from slot 0.
    on_second = weight >= 1.0 - WEIGHT_FLOOR
    slots[on_second, 0] = slots[on_second, 1]
    weight[on_second | (weight <= WEIGHT_FLOOR)] = 0.0
    return state, slots, weight, moving


def tracks_of(recipe: Recipe, profile: Profile | None = None) -> Tracks:
    """The listener and the traced sources at every 50 ms step of the pack."""
    profile = profile or Profile()
    start = float(profile.start_s)
    first = int(round(start / STEP_S))
    duration = float(recipe.duration_s) - start
    if profile.seconds is not None:
        duration = min(duration, float(profile.seconds))
    steps = int(round(duration / STEP_S)) + 1
    if (
        steps < 2
        or abs(duration - (steps - 1) * STEP_S) > 1e-9
        or abs(start - first * STEP_S) > 1e-9
    ):
        raise ValueError(
            f"{duration} s from {start} s is not a whole number of steps of {STEP_S} s"
        )
    times = start + np.arange(steps) * STEP_S
    head = listener_state(recipe, times)
    orientation = np.stack([head.yaw_deg, head.pitch_deg, head.roll_deg], axis=1)
    found: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for source in recipe.sources:
        audible = audible_steps(recipe, source.id)[first : first + steps]
        if profile.smoke and not audible.any():
            continue
        state, slots, weight, moving = _slots(recipe, source.id, times)
        found.append((source.id, state, slots, weight, moving, audible))
    if profile.sources is not None:
        # A smoke run is there to exercise every stage: the sources heard on the move first.
        ranked = sorted(found, key=lambda item: not bool((item[4] & item[5]).any()))
        kept = {item[0] for item in ranked[: profile.sources]}
        found = [item for item in found if item[0] in kept]
    read = [
        np.concatenate([slots[audible, 0], slots[audible & (weight > 0.0), 1]])
        for _, _, slots, weight, _, audible in found
    ]
    millimetres = (
        np.unique(_millimetres(np.concatenate(read)), axis=0)
        if read and sum(len(r) for r in read)
        else np.zeros((0, 3), dtype=np.int64)
    )
    row_of = {tuple(int(v) for v in key): row for row, key in enumerate(millimetres)}
    sources: dict[str, SourceTrack] = {}
    for name, state, slots, weight, moving, audible in found:
        slot = np.full((steps, 2), -1, dtype=np.int32)
        keys = _millimetres(slots)
        for step in np.flatnonzero(audible):
            slot[step, 0] = row_of[tuple(int(v) for v in keys[step, 0])]
            if weight[step] > 0.0:
                slot[step, 1] = row_of[tuple(int(v) for v in keys[step, 1])]
        sources[name] = SourceTrack(
            id=name,
            position=state.position,
            yaw_deg=state.yaw_deg,
            audible=audible,
            slot=slot,
            weight=np.where(audible, weight, 0.0),
            moving=moving,
            segment=np.asarray(state.segment),
        )
    return Tracks(
        duration_s=duration,
        start_s=start,
        listener=head.position,
        orientation=orientation,
        sources=sources,
        positions=millimetres.astype(float) / 1000.0,
    )


def busiest_window(recipe: Recipe, seconds: float, sources: int, every_s: float = 5.0) -> float:
    """Where a smoke run of ``seconds`` exercises most: the start, in seconds, of its window.

    A scene is mostly at rest, and a window at rest reads every cell
    exactly and no rail. The score of a window is the audible steps over
    which the head moves and those over which the source itself moves, each
    counted to a hundred, for its ``sources`` best sources; the earliest of
    the best windows is returned.
    """
    whole = tracks_of(recipe)
    span = int(round(seconds / STEP_S))
    stride = max(1, int(round(every_s / STEP_S)))
    head = np.zeros(whole.steps, dtype=bool)
    head[1:] = np.linalg.norm(np.diff(whole.listener, axis=0), axis=1) > 1e-9
    best, best_score = 0, -1.0
    for first in range(0, max(whole.steps - span, 1), stride):
        window = slice(first, first + span + 1)
        scores = sorted(
            (
                min(int((track.audible[window] & head[window]).sum()), 100)
                + min(int((track.audible[window] & track.moving[window]).sum()), 100)
                for track in whole.sources.values()
            ),
            reverse=True,
        )
        score = float(sum(scores[:sources]))
        if score > best_score:
            best, best_score = first, score
    return best * STEP_S


# --------------------------------------------------------------------------
# the cells, and which serve a step
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSet:
    """The listening cells asked for: where, of which kind, and why."""

    position: np.ndarray
    kind: np.ndarray
    origin: np.ndarray
    clearance_m: np.ndarray

    @property
    def count(self) -> int:
        return int(self.position.shape[0])

    def with_cells(
        self, positions: np.ndarray, kind: int, origin: int, triangles: np.ndarray
    ) -> CellSet:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not positions.shape[0]:
            return self
        return CellSet(
            position=np.concatenate([self.position, positions]),
            kind=np.concatenate([self.kind, np.full(len(positions), kind, dtype=np.uint8)]),
            origin=np.concatenate([self.origin, np.full(len(positions), origin, dtype=np.uint8)]),
            clearance_m=np.concatenate([self.clearance_m, clearance_m(positions, triangles)]),
        )


@dataclass(frozen=True)
class Assignment:
    """One source's ``low/mode`` and ``low/cell``, and the steps the rule could not serve."""

    mode: np.ndarray
    cell: np.ndarray
    #: ``[step]``: the rule refused every cell and the nearest serves in mode 2.
    fallback: np.ndarray


def _choose(
    head: np.ndarray,
    source: np.ndarray,
    cells: np.ndarray,
    clearance: np.ndarray,
    tree: cKDTree,
    rows: np.ndarray,
) -> tuple[int, tuple[int, int]]:
    """:func:`choose_cells` over the cells within reach of the head, with its ``LookupError``."""
    near = np.asarray(sorted(tree.query_ball_point(head, FUSE_WITHIN_M + 1e-9)), dtype=np.int64)
    if not near.size:
        raise LookupError(f"no cell within {FUSE_WITHIN_M} m of the head")
    radius = serving_radius_m(cells[near], clearance[near], source)
    mode, chosen = choose_cells(head, cells[near], radius)
    return mode, (
        int(rows[near[chosen[0]]]),
        int(rows[near[chosen[1]]]) if chosen[1] >= 0 else -1,
    )


def assign(
    tracks: Tracks,
    cells: np.ndarray,
    clearance: np.ndarray,
    *,
    usable: np.ndarray | None = None,
) -> tuple[dict[str, Assignment], list[tuple[str, int]]]:
    """Every source's cells at every audible step, by the rule of ``spatial.translate``.

    ``cells`` are the arrays' centres and ``clearance`` their distance to
    the nearest surface; ``usable`` leaves out the cells no array stands on
    and those of the validation patch. A step the rule refuses falls back to
    the nearest cell alone, translated, and is returned by name: the scene is
    not failed for it.
    """
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    rows = np.arange(cells.shape[0]) if usable is None else np.flatnonzero(usable)
    if not rows.size:
        raise ValueError("no cell may be used")
    held, held_clearance = cells[rows], np.asarray(clearance, dtype=float)[rows]
    tree = cKDTree(held)
    steps = tracks.steps
    found: dict[str, Assignment] = {}
    refused: list[tuple[str, int]] = []
    for name, track in tracks.sources.items():
        mode = np.full(steps, MODE_INAUDIBLE, dtype=np.uint8)
        cell = np.full((steps, 2), -1, dtype=np.int32)
        fallback = np.zeros(steps, dtype=bool)
        heard = np.flatnonzero(track.audible)
        if heard.size:
            both = np.concatenate([tracks.listener[heard], track.position[heard]], axis=1)
            jobs, job_of = np.unique(both, axis=0, return_inverse=True)
            job_of = np.asarray(job_of).reshape(-1)
            for job, row in enumerate(jobs):
                here = heard[job_of == job]
                try:
                    chosen_mode, chosen = _choose(
                        row[:3], row[3:], held, held_clearance, tree, rows
                    )
                except LookupError:
                    nearest = int(tree.query(row[:3])[1])
                    away = float(np.linalg.norm(held[nearest] - row[:3]))
                    chosen_mode = MODE_EXACT if away < EXACT_UNDER_M else MODE_TRANSLATED
                    chosen = (int(rows[nearest]), -1)
                    fallback[here] = True
                    refused.extend((name, int(step)) for step in here)
                mode[here] = chosen_mode
                cell[here] = chosen
        found[name] = Assignment(mode, cell, fallback)
    return found, refused


def pairs_of(tracks: Tracks, low: dict[str, Assignment]) -> list[list[int]]:
    """Per source position, the cells it is heard at: a pair is there only if a step reads it."""
    heard: list[set[int]] = [set() for _ in range(tracks.positions.shape[0])]
    for name, track in tracks.sources.items():
        chosen = low[name]
        for step in np.flatnonzero(track.audible):
            for position in track.slot[step]:
                if position >= 0:
                    heard[int(position)].update(int(c) for c in chosen.cell[step] if c >= 0)
    return [sorted(cells) for cells in heard]


def _runs(listener: np.ndarray) -> list[tuple[int, int]]:
    """The stretches of steps over which the head moves, each from its first to its last step."""
    moved = np.linalg.norm(np.diff(listener, axis=0), axis=1) > 1e-9
    runs: list[tuple[int, int]] = []
    start = None
    for step, moving in enumerate(moved):
        if moving and start is None:
            start = step
        if not moving and start is not None:
            runs.append((start, step))
            start = None
    if start is not None:
        runs.append((start, len(moved)))
    return runs


def _along(points: np.ndarray, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A polyline sampled every ``pitch_m`` from its start, its end included: points and arcs."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    at = np.append(np.arange(0.0, arc[-1] - 1e-9, pitch_m), arc[-1])
    sampled = np.stack([np.interp(at, arc, points[:, axis]) for axis in range(3)], axis=1)
    return sampled, at


def _merged(new: np.ndarray, held: np.ndarray, within_m: float = MERGE_M) -> np.ndarray:
    """Of ``new``, in whole millimetres, those on which no cell of ``held`` or of ``new`` stands."""
    kept: list[np.ndarray] = []
    points = [np.asarray(held, dtype=float).reshape(-1, 3)]
    for point in _millimetres(new).astype(float) / 1000.0:
        every = np.concatenate(points)
        if every.shape[0] and float(np.linalg.norm(every - point, axis=1).min()) <= within_m:
            continue
        kept.append(point)
        points.append(point[None, :])
    return np.asarray(kept, dtype=float).reshape(-1, 3)


def listening_cells(
    recipe: Recipe, tracks: Tracks, triangles: np.ndarray
) -> tuple[CellSet, dict[str, Any]]:
    """The cells a scene needs, and the record of how the rule was met.

    ``triangles`` are the occluders of the dwelling's mirror scene,
    ``[triangle, 3, 3]``: what a cell's clearance is measured to.
    """
    listener = tracks.listener
    seats = np.asarray(
        [s.position for s in recipe.stations if s.kind == "seat"], dtype=float
    ).reshape(-1, 3)
    still = np.ones(tracks.steps, dtype=bool)
    moved = np.linalg.norm(np.diff(listener, axis=0), axis=1) > 1e-9
    still[:-1] &= ~moved
    still[1:] &= ~moved
    rest = _millimetres(listener[still])
    rest = rest[np.sort(np.unique(rest, axis=0, return_index=True)[1])].astype(float) / 1000.0
    kind = np.full(rest.shape[0], KIND_ADDED, dtype=np.uint8)
    if seats.shape[0] and rest.shape[0]:
        on_seat = np.linalg.norm(rest[:, None, :] - seats[None, :, :], axis=2).min(axis=1)
        kind[on_seat < EXACT_UNDER_M] = KIND_SEAT
    cells = CellSet(
        position=rest,
        kind=kind,
        origin=np.full(rest.shape[0], ORIGIN_REST, dtype=np.uint8),
        clearance_m=clearance_m(rest, triangles) if rest.shape[0] else np.zeros(0),
    )
    runs = _runs(listener)
    for first, last in runs:
        sampled, _ = _along(listener[first : last + 1], PATH_PITCH_M)
        cells = cells.with_cells(
            _merged(sampled, cells.position), KIND_ADDED, ORIGIN_PATH, triangles
        )
    record: dict[str, Any] = {
        "rest_places": int(rest.shape[0]),
        "seats": int((kind == KIND_SEAT).sum()),
        "path_runs": len(runs),
        "path_m": round(float(np.linalg.norm(np.diff(listener, axis=0), axis=1).sum()), 2),
        "path_cells": cells.count - int(rest.shape[0]),
    }
    if not cells.count:
        raise ValueError("the listener is nowhere: the scene has no cell")

    # The denser rule, where the cells every 0.15 m are refused by a near source.
    _, refused = assign(tracks, cells.position, cells.clearance_m)
    record["steps_refused_at_0.15_m"] = len(refused)
    run_of = np.full(tracks.steps, -1, dtype=np.int64)
    for number, (first, last) in enumerate(runs):
        run_of[first : last + 1] = number
    dense: list[np.ndarray] = []
    for number in sorted({int(run_of[step]) for _, step in refused if run_of[step] >= 0}):
        first, last = runs[number]
        sampled, _ = _along(listener[first : last + 1], DENSE_PITCH_M)
        heads = listener[sorted({step for _, step in refused if run_of[step] == number})]
        near = np.linalg.norm(sampled[:, None, :] - heads[None, :, :], axis=2).min(axis=1)
        dense.append(sampled[near <= PATH_PITCH_M])
    before = cells.count
    if dense:
        cells = cells.with_cells(
            _merged(np.concatenate(dense), cells.position, 0.01),
            KIND_ADDED,
            ORIGIN_DENSE,
            triangles,
        )
    record["dense_cells"] = cells.count - before

    # Still refused: a cell on the head itself, where an array can stand.
    _, refused = assign(tracks, cells.position, cells.clearance_m)
    record["steps_refused_at_0.10_m"] = len(refused)
    before = cells.count
    for name, step in refused:
        head = _millimetres(listener[step]).astype(float) / 1000.0
        track = tracks.sources[name]
        tree = cKDTree(cells.position)
        try:
            _choose(
                listener[step],
                track.position[step],
                cells.position,
                cells.clearance_m,
                tree,
                np.arange(cells.count),
            )
            continue  # a cell added for an earlier step serves this one
        except LookupError:
            pass
        mouths = np.asarray([t.position[step] for t in tracks.sources.values() if t.audible[step]])
        free = float(clearance_m(head[None, :], triangles)[0])
        if free < ARRAY_RADIUS_M or (
            mouths.size and float(np.linalg.norm(mouths - head, axis=1).min()) < ARRAY_RADIUS_M
        ):
            continue
        cells = cells.with_cells(head[None, :], KIND_ADDED, ORIGIN_HEAD, triangles)
    record["head_cells"] = cells.count - before
    record["cells"] = cells.count
    record["cells_whose_ball_is_not_free"] = int((cells.clearance_m < ARRAY_RADIUS_M).sum())
    record["source_share"] = SOURCE_SHARE
    return cells, record


# --------------------------------------------------------------------------
# the tail
# --------------------------------------------------------------------------


def tail_sites_of(recipe: Recipe, track: SourceTrack) -> TailSites:
    """Where one source's rays are traced from: the places it rests, and its rails."""
    source = recipe.source(track.id)
    floor = recipe.dwelling.floor_y_m
    standing = floor + recipe.heights.standing_m
    seated = floor + recipe.heights.seated_m
    stations: list[np.ndarray] = []
    rails: list[np.ndarray] = []
    seen: set[str] = set()
    for number, segment in enumerate(source.segments):
        here = (track.segment == number) & track.audible
        if not here.any():
            continue
        if isinstance(segment, Travel):
            if segment.rail not in seen:
                seen.add(segment.rail)
                flat = np.asarray(recipe.rail(segment.rail).points, dtype=float)
                rails.append(
                    np.stack([flat[:, 0], np.full(flat.shape[0], standing), flat[:, 1]], axis=1)
                )
        elif isinstance(segment, Rise):
            if f"rise:{segment.station}" not in seen:
                seen.add(f"rise:{segment.station}")
                x, z = recipe.station(segment.station).xz
                rails.append(np.array([[x, seated, z], [x, standing, z]]))
        else:
            stations.append(track.position[np.flatnonzero(here)[0]])
    held = np.asarray(stations, dtype=float).reshape(-1, 3)
    if held.shape[0]:
        keys = _millimetres(held)
        held = held[np.sort(np.unique(keys, axis=0, return_index=True)[1])]
    return tail_sites(held, rails)


def tail_cells(cells: np.ndarray, kind: np.ndarray, spacing_m: float = SPACING_M) -> np.ndarray:
    """The rows of ``cells`` the tail's spheres stand on: every seat, then none within 0.80 m."""
    cells = np.asarray(cells, dtype=float).reshape(-1, 3)
    chosen = [int(row) for row in np.flatnonzero(np.asarray(kind) == KIND_SEAT)]
    for row in range(cells.shape[0]):
        if row in chosen:
            continue
        if (
            not chosen
            or float(np.linalg.norm(cells[chosen] - cells[row], axis=1).min()) >= spacing_m
        ):
            chosen.append(row)
    return np.asarray(sorted(chosen), dtype=np.int32)


# --------------------------------------------------------------------------
# the validation patch
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """The dense square of the off-line translation validation: its cells and its one source."""

    centre: np.ndarray
    cells: np.ndarray
    #: Row of :attr:`Tracks.positions` the patch is solved from.
    source: int
    min_clearance_m: float


def _square(centre_xz: np.ndarray, heights: tuple[float, ...]) -> np.ndarray:
    count = int(round(PATCH_SIDE_M / PATCH_PITCH_M)) + 1
    offsets = (np.arange(count) - (count - 1) / 2.0) * PATCH_PITCH_M
    x, z = np.meshgrid(centre_xz[0] + offsets, centre_xz[1] + offsets, indexing="ij")
    layers = [np.stack([x.ravel(), np.full(x.size, y), z.ravel()], axis=1) for y in heights]
    return _millimetres(np.concatenate(layers)).astype(float) / 1000.0


def patch_cells(
    recipe: Recipe,
    tracks: Tracks,
    cells: CellSet,
    triangles: np.ndarray,
    centre_xz: tuple[float, float] | None = None,
) -> Patch:
    """A square of 0.80 m every 4 cm at both heights, 882 cells, as near a surface as arrays stand.

    Without ``centre_xz`` the square is centred on the listening cell where
    its coarse outline (five points a side) is nearest a wall or furniture
    while an array still stands on each. Its source is the solved position
    nearest the centre that is 1.2 m away or more.
    """
    floor = recipe.dwelling.floor_y_m
    heights = (floor + recipe.heights.standing_m, floor + recipe.heights.seated_m)
    if centre_xz is None:
        count = 5
        offsets = (np.arange(count) - (count - 1) / 2.0) * (PATCH_SIDE_M / (count - 1))
        x, z = np.meshgrid(offsets, offsets, indexing="ij")
        best: tuple[float, np.ndarray] | None = None
        seen: set[tuple[int, int]] = set()
        for cell in cells.position:
            key = (int(round(cell[0] * 10)), int(round(cell[2] * 10)))
            if key in seen:
                continue
            seen.add(key)
            outline = np.concatenate(
                [
                    np.stack([cell[0] + x.ravel(), np.full(x.size, y), cell[2] + z.ravel()], axis=1)
                    for y in heights
                ]
            )
            least = float(clearance_m(outline, triangles).min())
            if least >= ARRAY_RADIUS_M and (best is None or least < best[0]):
                best = (least, cell[[0, 2]])
        if best is None:
            raise ValueError("no listening cell has a patch round it on which arrays stand")
        centre = np.asarray(best[1], dtype=float)
    else:
        centre = np.asarray(centre_xz, dtype=float)
    square = _square(centre, heights)
    if not tracks.positions.shape[0]:
        raise ValueError("the patch needs a source position and the scene has none audible")
    middle = np.array([centre[0], heights[0], centre[1]])
    away = np.linalg.norm(tracks.positions - middle[None, :], axis=1)
    far = np.flatnonzero(away >= PATCH_SOURCE_M)
    source = int(far[np.argmin(away[far])]) if far.size else int(np.argmax(away))
    return Patch(
        centre=middle,
        cells=square,
        source=source,
        min_clearance_m=float(clearance_m(square, triangles).min()),
    )


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Everything the machine must compute for one recipe under one profile."""

    recipe_sha256: str
    profile: Profile
    tracks: Tracks
    cells: CellSet
    low: dict[str, Assignment]
    #: Per source position, the cells it is heard at: rows of :attr:`all_cells`.
    heard_at: list[list[int]]
    #: Rows of the scene's cells the tail's receiver spheres stand on.
    tail_cells: np.ndarray
    sites: dict[str, TailSites]
    patch: Patch | None = None
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def all_cells(self) -> np.ndarray:
        """The scene's cells, then the patch's: what the pairs campaign stands arrays on."""
        if self.patch is None:
            return self.cells.position
        return np.concatenate([self.cells.position, self.patch.cells])

    @property
    def pairs(self) -> int:
        return sum(len(cells) for cells in self.heard_at)

    def save(self, directory: Path) -> None:
        """``plan.json``, the record, and ``plan.npz``, the cells the machine starts from."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plan.json").write_text(json.dumps(self.record, indent=1, sort_keys=True))
        np.savez(
            directory / "plan.npz",
            cells=self.cells.position,
            kind=self.cells.kind,
            origin=self.cells.origin,
            clearance_m=self.cells.clearance_m,
            patch_cells=self.patch.cells if self.patch is not None else np.zeros((0, 3)),
            patch_source=np.asarray(-1 if self.patch is None else self.patch.source),
        )


def make_plan(
    recipe: Recipe,
    triangles: np.ndarray,
    profile: Profile | None = None,
    *,
    patch_centre_xz: tuple[float, float] | None = None,
) -> Plan:
    """The plan of ``recipe``: deterministic, and free of any file but those it is given."""
    profile = profile or Profile()
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    tracks = tracks_of(recipe, profile)
    cells, cell_record = listening_cells(recipe, tracks, triangles)
    low, refused = assign(tracks, cells.position, cells.clearance_m)
    heard_at = pairs_of(tracks, low)
    sites = {name: tail_sites_of(recipe, track) for name, track in tracks.sources.items()}
    used: set[tuple[int, ...]] = set()
    for name, track in tracks.sources.items():
        if track.audible.any():
            slots, weight = source_weights(track.position[track.audible], sites[name])
            rows = np.unique(np.concatenate([slots[:, 0], slots[weight > 0.0, 1]]))
            used.update(
                tuple(int(v) for v in key) for key in _millimetres(sites[name].positions[rows])
            )
    receivers = tail_cells(cells.position, cells.kind)
    patch = None
    if profile.patch:
        patch = patch_cells(recipe, tracks, cells, triangles, patch_centre_xz)
        added = range(cells.count, cells.count + patch.cells.shape[0])
        heard_at[patch.source] = sorted(set(heard_at[patch.source]) | set(added))
    audible = {name: int(track.audible.sum()) for name, track in tracks.sources.items()}
    jobs = 0
    for track in tracks.sources.values():
        both = np.concatenate([tracks.listener, track.position], axis=1)[track.audible]
        jobs += int(np.unique(both, axis=0).shape[0]) if both.shape[0] else 0
    modes = np.concatenate(
        [low[name].mode[track.audible] for name, track in tracks.sources.items()]
        or [np.zeros(0, dtype=np.uint8)]
    )
    whole = low_band_source_positions(recipe)
    record = {
        "recipe_sha256": recipe_sha256(recipe),
        "dwelling": recipe.dwelling.name,
        "profile": profile.record(),
        "duration_s": tracks.duration_s,
        "steps": tracks.steps,
        "sources": list(tracks.sources),
        "audible_steps": audible,
        "audible_steps_total": int(sum(audible.values())),
        "step_pairs": jobs,
        "cells": cell_record,
        "modes": {
            "exact": int((modes == MODE_EXACT).sum()),
            "translated": int((modes == MODE_TRANSLATED).sum()),
            "fused": int((modes == MODE_FUSED).sum()),
        },
        "fallback_steps": [[name, step] for name, step in refused],
        "source_positions": int(tracks.positions.shape[0]),
        "source_positions_of_the_whole_recipe": whole.count,
        "pairs": sum(len(c) for c in heard_at),
        "pairs_of_the_patch": 0 if patch is None else int(patch.cells.shape[0]),
        "tail_sites": len(used),
        "tail_cells": int(receivers.size),
        "patch": None
        if patch is None
        else {
            "centre_m": [round(float(v), 3) for v in patch.centre],
            "cells": int(patch.cells.shape[0]),
            "source_m": [round(float(v), 3) for v in tracks.positions[patch.source]],
            "source_distance_m": round(
                float(np.linalg.norm(tracks.positions[patch.source] - patch.centre)), 3
            ),
            "min_clearance_m": round(patch.min_clearance_m, 3),
        },
    }
    return Plan(
        recipe_sha256=recipe_sha256(recipe),
        profile=profile,
        tracks=tracks,
        cells=cells,
        low=low,
        heard_at=heard_at,
        tail_cells=receivers,
        sites=sites,
        patch=patch,
        record=record,
    )


# --------------------------------------------------------------------------
# the cost
# --------------------------------------------------------------------------


def estimate(
    plan: Plan | dict[str, Any], *, rate_usd_per_hour: float, low_engine: str = "lowband"
) -> dict[str, Any]:
    """Machine-seconds and USD of a plan at an hourly rate, stage by stage.

    The low band is priced by the engine that solves it: the batched solver
    (:func:`reverberate.wave.lowband.pairs.estimate`, measured on one RTX
    3080) or, with ``low_engine`` ``pffdtd``, PFFDTD a source position
    (:func:`reverberate.accel.pairs.estimate`, measured on 2 x A100). The
    rate is the caller's and is right only for the card the terms were
    measured on, which the result names. The mirror's stages are the
    projection of ADR 0016's cost appendix, which no card has run. A plan's
    record is enough.
    """
    from reverberate.accel import pairs as present
    from reverberate.wave.lowband import pairs as batched

    engines: dict[str, Any] = {"lowband": batched.estimate, "pffdtd": present.estimate}
    if low_engine not in engines:
        raise ValueError(f"unknown low band engine {low_engine!r}")
    pairs_estimate = engines[low_engine]

    record = plan.record if isinstance(plan, Plan) else plan
    positions, pairs = int(record["source_positions"]), int(record["pairs"])
    scene_pairs = pairs - int(record.get("pairs_of_the_patch", 0))
    low = pairs_estimate(
        positions, pairs, fmax_hz=solve_fmax_hz(), rate_usd_per_hour=rate_usd_per_hour
    )
    histograms = int(record["tail_sites"]) * int(record["tail_cells"])
    pack_bytes = scene_pairs * PAIR_BYTES + histograms * HISTOGRAM_BYTES
    seconds = {
        "fixed": FIXED_S,
        "low": float(low["seconds"]),
        "paths": (int(record["step_pairs"]) + scene_pairs) * CARD_PAIR_S,
        "diffraction": positions * ONSET_FIELD_S,
        "rays": int(record["tail_sites"]) * RAYS_SITE_S,
        "level": scene_pairs * LEVEL_PAIR_S,
        "write": pack_bytes / 200e6,
        "transfer": (pack_bytes + pairs * PAIR_BYTES) / FETCH_BYTES_PER_S,
    }
    total = float(sum(seconds.values()))
    return {
        "billed_rate_usd_per_hour": float(rate_usd_per_hour),
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
        "usd": {
            name: round(value / 3600.0 * rate_usd_per_hour, 3) for name, value in seconds.items()
        },
        "total_s": round(total, 1),
        "total_usd": round(total / 3600.0 * rate_usd_per_hour, 2),
        "pack_gb": round(pack_bytes / 1e9, 2),
        "pair_cache_gb": round(pairs * PAIR_BYTES / 1e9, 2),
        "low": low,
        "low_engine": low_engine,
        "measured_on": str(low.get("measured_on", "2 x A100")),
        "measured": ["low"],
        "projected": ["paths", "diffraction", "rays", "level", "write", "transfer", "fixed"],
    }
