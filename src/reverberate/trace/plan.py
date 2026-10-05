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
    rail_arc_lengths,
    rail_samples,
    recipe_sha256,
    seat_rail_heights,
    source_state,
)
from reverberate.scenes.recipe import Rail, Rise, Travel
from reverberate.spatial.lowband import LOW_DURATION_S, solve_fmax_hz
from reverberate.spatial.rail import band_limited_weights, knots_hz, nearest_samples
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
    "RAIL_POSITIONS",
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
    "read_arcs",
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
#: The solved positions a source on a rail reads unless more are asked for: the two round
#: it, weighted linearly. More are read by ``spatial.rail.band_limited_weights``.
RAIL_POSITIONS = 2

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

# What a stage costs a unit, and the card and host it was measured on
# (``docs/adr/0016-appendix-trace-cost.md``, the ledger these are read from). A stage whose
# constant no machine has run is not in :data:`MEASURED_ON`, and the estimate
# says so on its line.
#: The scene, its facets and occluders' grid, then the occupancy and its graph: once a trace.
PATHS_FIXED_S = 10.0
#: A distinct (source, head) position of the batched trace, its diffracted onset included:
#: 26 ms on a window of 8257 of them, 20 ms on one of 18 809.
PATH_JOB_S = 0.0223
#: A tail site's rays through the tree, in double precision, and what a tail cell adds to
#: them: 0.58 s a site over 53 cells on an RTX 3090 Ti (2026-10-05), of which 0.49 s are the
#: launch, whatever the cells, and 0.056 s the 53 entries written. Through the grid it was
#: 12.8 s a site over 18 cells and 14.5 s over 29 on an RTX 3090.
RAYS_SITE_S = 0.53
RAYS_SITE_CELL_S = 0.001
#: The rays those were measured with; a site of another count is priced in proportion.
RAYS_MEASURED = 100_000
#: A pair's seam and onset, the tails' tables of the pairs included: 62 and 83 ms.
LEVEL_PAIR_S = 0.083
#: A pair's 64 channels through the air and the masks, into the pack: 7 ms a pair on 1894
#: pairs, read and file included. A second window took 164 ms a pair with the card idle
#: and 5 ms of them in the pair's own work: the ledger says what that was taken to be.
WRITE_PAIR_S = 0.010
#: The check that reads the pack's structure, and what the full one adds: ten seconds of
#: three sources on both array modules, 41 to 77 s.
CHECK_READ_S = 1.0
CHECK_FULL_S = 60.0
#: Provisioning, the bundle's push, and what the watcher's five minutes leave unbilled to
#: no stage: the first smoke's rental (2 x P100, 2026-10-04).
FIXED_S = 340.0
#: What comes home through Vast's ssh proxy on four streams, bytes a second: the two
#: whole scenes of 2026-10-05, a pack each from California (one stream 2.0 to 2.3 MB/s,
#: eight 5.9, on a line that carries 17 MB/s to the United States and 70 to Europe: the
#: proxy is the limit). The 8.1 MB/s of the first card box (2026-10-04) was not seen again.
FETCH_BYTES_PER_S = 4.4e6
#: The card and the host each constant was measured on; a stage absent here is projected.
_BOX = "one RTX 3090, 2026-10-04"
MEASURED_ON = {
    "low": "2 x A100 (accel.pairs.estimate)",
    "fixed": "2 x Tesla P100 (instance 54194831, 2026-10-04)",
    "paths": _BOX,
    "rays": _BOX,
    "level": _BOX,
    "write": _BOX,
    "check": _BOX,
    "transfer_pack": "8 x and 4 x RTX 3090, 2026-10-05, through the proxy on four streams",
    "transfer_pairs": "8 x and 4 x RTX 3090, 2026-10-05, through the proxy on four streams",
}
PAIR_BYTES = 64 * 4800 * 4
HISTOGRAM_BYTES = 286_000
#: A pair of the pack with ``bins,int16``: 8 339 007 472 bytes for the 18 992 rows of the
#: first scene's pack, every pair the same (``python -m reverberate.render compact``).
PACK_PAIR_BYTES_BINS_INT16 = 439_080
#: What each other lever divides a pair's bytes by, on 600 pairs of the first scene
#: (``docs/open-questions/low-band-compact.md``): the bins alone, 16 bits of them, each
#: degree cut where it has decayed 60 dB, the degrees let go 50 dB under the whole. The
#: decay's and the degree's figures are those two settings'; another is priced as they are.
_LEVER_FACTOR = {"bins": 1.41, "int16": 1.985, "decay": 2.54, "degree": 1.14}
#: A pair's encoding into those bins on the machine, beside its row's write: 5 ms on the
#: laptop's core (2026-10-05); the machine's has not been read apart.
WRITE_COMPACT_PAIR_S = 0.008


def pack_pair_bytes(low_levers: str | None) -> float:
    """What a pair takes in a pack written with ``low_levers`` (``None``, ``none``: samples)."""
    from reverberate.render.compact import Levers

    levers = Levers.parse(low_levers)
    if levers.off:
        return float(PAIR_BYTES)
    if levers.sample == "int16":
        size = float(PACK_PAIR_BYTES_BINS_INT16)
    else:
        size = PAIR_BYTES / _LEVER_FACTOR["bins"]
    if levers.decay_db:
        size /= _LEVER_FACTOR["decay"]
    if levers.degree_db:
        size /= _LEVER_FACTOR["degree"]
    return size


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
    #: The solved positions a source on a rail reads: two, weighted linearly,
    #: or more, weighted per frequency (``docs/open-questions/rail-interpolation.md``).
    rail_positions: int = RAIL_POSITIONS

    def __post_init__(self) -> None:
        if not 2 <= int(self.rail_positions) <= 16:
            raise ValueError(f"a source reads 2 to 16 positions, not {self.rail_positions}")

    @property
    def smoke(self) -> bool:
        return self.seconds is not None or self.sources is not None

    def record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "seconds": self.seconds,
            "sources": self.sources,
            "patch": self.patch,
            "start_s": self.start_s,
        }
        # Named only when it is not the first rule's, whose records it leaves as they were.
        if self.rail_positions != RAIL_POSITIONS:
            record["rail_positions"] = int(self.rail_positions)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Profile:
        return cls(
            record.get("seconds"),
            record.get("sources"),
            bool(record.get("patch", False)),
            float(record.get("start_s", 0.0)),
            int(record.get("rail_positions", RAIL_POSITIONS)),
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
    #: ``[step, position]``: every solved position a step reads, the two of
    #: :attr:`slot` first, and ``[step, position, knot]``, their weights at the
    #: frequencies of :attr:`Tracks.rail_knots_hz`. ``None`` under the first
    #: rule, where :attr:`slot` and :attr:`weight` say it all.
    rail_slot: np.ndarray | None = None
    rail_weight: np.ndarray | None = None

    def read(self, step: int) -> np.ndarray:
        """The rows of :attr:`Tracks.positions` the step reads, ``-1`` for a slot not read."""
        table = self.slot if self.rail_slot is None else self.rail_slot
        return np.asarray(table[step])


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
    #: The frequencies the sources' ``rail_weight`` are held at, in Hz.
    rail_knots_hz: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return int(self.listener.shape[0])

    @property
    def times(self) -> np.ndarray:
        return self.start_s + np.arange(self.steps) * STEP_S


def _millimetres(points: np.ndarray) -> np.ndarray:
    return np.asarray(np.rint(np.asarray(points, dtype=float) * 1000.0), dtype=np.int64)


def _slots(
    recipe: Recipe, source_id: str, times: np.ndarray, count: int = RAIL_POSITIONS
) -> tuple[Any, np.ndarray, Any, Any, np.ndarray | None, np.ndarray | None]:
    """The source's state, its two solved positions at every step, the weight, whether it moves.

    With a ``count`` over two, also the ``count`` positions round it and
    which of them are positions (:func:`_rail_reading`); ``None`` otherwise.
    """
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
    nodes = held = None
    if count > RAIL_POSITIONS:
        nodes, held = _rail_reading(recipe, source_id, state, slots, weight, count)
    # A source on a solved position reads it alone, from slot 0.
    on_second = weight >= 1.0 - WEIGHT_FLOOR
    slots[on_second, 0] = slots[on_second, 1]
    weight[on_second | (weight <= WEIGHT_FLOOR)] = 0.0
    return state, slots, weight, moving, nodes, held


def _even(lengths: np.ndarray, pitch_m: float) -> np.ndarray:
    """Arc lengths that cut each leg of a path into equal gaps of ``pitch_m`` at most."""
    arcs = [np.zeros(1)]
    start = 0.0
    for length in np.asarray(lengths, dtype=float):
        gaps = max(int(np.ceil(length / pitch_m - 1e-9)), 1)
        arcs.append(start + length * np.arange(1, gaps + 1) / gaps)
        start += float(length)
    return np.concatenate(arcs)


def read_arcs(rail: Rail) -> np.ndarray:
    """Arc lengths of a rail's solved positions when more than two are read at a step.

    Every corner is one, and each straight leg between two corners is cut
    into equal gaps of the rail's pitch at most. Two things the samples of
    :func:`reverberate.scenes.rail_arc_lengths` do not give: between two
    positions either side of a corner the source is on neither's line, and
    the weights of ``spatial.rail`` hold the field there no better than a
    straight chord does; and a last sample a centimetre short of the rail's
    end makes two positions that say the same thing, whose weights then
    grow to tell them apart.
    """
    points = np.asarray(rail.points, dtype=float)
    return _even(np.linalg.norm(np.diff(points, axis=0), axis=1), rail.pitch_m)


def _rail_reading(
    recipe: Recipe, source_id: str, state: Any, slots: np.ndarray, weight: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``count`` solved positions round a source at every step: ``[step, count, 3]``, valid.

    The positions of the rail it travels (:func:`read_arcs`), or of the
    seat's vertical rail it rises on, cut the same way, nearest it along the
    rail: the two round it first, which replace ``slots`` and ``weight`` of
    those steps. A rail of fewer than ``count`` positions gives what it
    has. At rest nothing is valid: the station is read alone.
    """
    source = recipe.source(source_id)
    floor = recipe.dwelling.floor_y_m
    standing = floor + recipe.heights.standing_m
    rungs = seat_rail_heights(recipe)
    pitch = recipe.rails[0].pitch_m if recipe.rails else float(rungs[1] - rungs[0])
    steps = state.position.shape[0]
    nodes = np.zeros((steps, count, 3))
    held = np.zeros((steps, count), dtype=bool)
    for number, segment in enumerate(source.segments):
        here = state.segment == number
        if not here.any() or not isinstance(segment, Travel | Rise):
            continue
        first, second, share = state.sample_a[here], state.sample_b[here], state.weight[here]
        if isinstance(segment, Travel):
            rail = recipe.rail(segment.rail)
            sampled = rail_arc_lengths(rail)
            arcs = read_arcs(rail)
            points = np.asarray(rail.points, dtype=float)
            walked = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            )
            # A corner is read at its own place, not at one a rounding short of it.
            every = np.stack(
                [
                    np.interp(arcs, walked, points[:, 0]),
                    np.full(arcs.size, standing),
                    np.interp(arcs, walked, points[:, 1]),
                ],
                axis=1,
            )
        else:
            x, z = recipe.station(segment.station).xz
            sampled = rungs - rungs[0]
            arcs = _even(sampled[-1:], pitch)
            every = np.stack(
                [np.full(arcs.size, x), floor + rungs[0] + arcs, np.full(arcs.size, z)], axis=1
            )
        at = sampled[first] * (1.0 - share) + sampled[second] * share
        lower = np.clip(np.searchsorted(arcs, at, side="right") - 1, 0, arcs.size - 2)
        span = arcs[lower + 1] - arcs[lower]
        chosen = nearest_samples(arcs, at, count, lower=lower)
        held[here] = chosen >= 0
        nodes[here] = every[np.maximum(chosen, 0)]
        slots[here] = nodes[here][:, :2]
        weight[here] = np.clip((at - arcs[lower]) / span, 0.0, 1.0)
    return nodes, held


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
    count = int(profile.rail_positions)
    # More than two positions: which, per source, and whether each is one.
    rails: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source in recipe.sources:
        audible = audible_steps(recipe, source.id)[first : first + steps]
        if profile.smoke and not audible.any():
            continue
        state, slots, weight, moving, nodes, held = _slots(recipe, source.id, times, count)
        found.append((source.id, state, slots, weight, moving, audible))
        if nodes is not None and held is not None:
            # A source on a solved position, or at rest, reads it alone.
            held &= (audible & (weight > 0.0))[:, None]
            rails[source.id] = (nodes, held)
    if profile.sources is not None:
        # A smoke run is there to exercise every stage: the sources heard on the move first.
        ranked = sorted(found, key=lambda item: not bool((item[4] & item[5]).any()))
        kept = {item[0] for item in ranked[: profile.sources]}
        found = [item for item in found if item[0] in kept]
    read = [
        np.concatenate([slots[audible, 0], slots[audible & (weight > 0.0), 1]])
        for _, _, slots, weight, _, audible in found
    ]
    read += [rails[name][0][rails[name][1]] for name, *_ in found if name in rails]
    millimetres = (
        np.unique(_millimetres(np.concatenate(read)), axis=0)
        if read and sum(len(r) for r in read)
        else np.zeros((0, 3), dtype=np.int64)
    )
    row_of = {tuple(int(v) for v in key): row for row, key in enumerate(millimetres)}
    knots = knots_hz(solve_fmax_hz()) if count > RAIL_POSITIONS else None
    sources: dict[str, SourceTrack] = {}
    for name, state, slots, weight, moving, audible in found:
        slot = np.full((steps, 2), -1, dtype=np.int32)
        keys = _millimetres(slots)
        for step in np.flatnonzero(audible):
            slot[step, 0] = row_of[tuple(int(v) for v in keys[step, 0])]
            if weight[step] > 0.0:
                slot[step, 1] = row_of[tuple(int(v) for v in keys[step, 1])]
        rail_slot = rail_weight = None
        if knots is not None:
            nodes, held = rails[name]
            rail_slot = np.full((steps, count), -1, dtype=np.int32)
            rail_slot[:, 0] = slot[:, 0]
            rail_weight = np.zeros((steps, count, knots.size), dtype=np.float32)
            rail_weight[audible, 0, :] = 1.0
            between = np.flatnonzero(held[:, 0])
            if between.size:
                rail_keys = _millimetres(nodes[between])
                for row, step in enumerate(between):
                    for position in np.flatnonzero(held[step]):
                        rail_slot[step, position] = row_of[
                            tuple(int(v) for v in rail_keys[row, position])
                        ]
                rail_weight[between] = band_limited_weights(
                    nodes[between], state.position[between], knots, valid=held[between]
                )
        sources[name] = SourceTrack(
            id=name,
            position=state.position,
            yaw_deg=state.yaw_deg,
            audible=audible,
            slot=slot,
            weight=np.where(audible, weight, 0.0),
            moving=moving,
            segment=np.asarray(state.segment),
            rail_slot=rail_slot,
            rail_weight=rail_weight,
        )
    return Tracks(
        duration_s=duration,
        start_s=start,
        listener=head.position,
        orientation=orientation,
        sources=sources,
        positions=millimetres.astype(float) / 1000.0,
        rail_knots_hz=knots,
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
            for position in track.read(int(step)):
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
        # What a card's record memory is counted against: a position heard at more cells
        # than a card holds records for is solved more than once.
        "cells_a_position": [len(c) for c in heard_at],
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
    plan: Plan | dict[str, Any],
    *,
    rate_usd_per_hour: float,
    fetch_pairs: bool = True,
    check: str | None = None,
    low_engine: str = "lowband",
    low_ppw: float | None = None,
    low_seconds: float | None = None,
    rays: int | None = None,
    low_levers: str | None = None,
) -> dict[str, Any]:
    """Machine-seconds and USD of a plan at an hourly rate, stage by stage.

    ``low_levers`` is the form the pack's low band is written in
    (:func:`pack_pair_bytes`; ``None`` and ``none`` are the samples): the
    pack's size and its way home follow, and where it keeps the bins the
    pair cache is priced compact too
    (:data:`reverberate.accel.pairs.COMPACT_PAIR_BYTES`).

    This is the plan on the machines the constants were measured on, one
    card; :func:`reverberate.trace.machines.predict` is the plan on an
    offered machine. The batched solver's solves are counted as that card
    makes them: a source position heard at more cells than its 20 GB hold
    records for is solved more than once (``low.solves``,
    ``extra_solves``). ``low_ppw`` prices the batched solver on another
    Cartesian grid than the bundle's. ``low_seconds`` prices a solve of
    fewer seconds than the pack's 1.2: the solve and a cell's records go
    as them, and so does the pair cache's way home; the pack is as long.
    ``rays`` prices a tail site of another count than the 100 000 the
    rays' constants were measured at, in proportion.

    The low band is priced by the engine that solves it: the batched solver
    (:func:`reverberate.wave.lowband.pairs.estimate`, measured on one RTX
    3080) or, with ``low_engine`` ``pffdtd``, PFFDTD a source position
    (:func:`reverberate.accel.pairs.estimate`, measured on 2 x A100). The
    rate is the caller's and is right only for the card the terms were
    measured on, which the result names stage by stage.

    Every other stage is a count of the plan times a constant of this
    module, measured on the machine :data:`MEASURED_ON` names
    (``docs/adr/0016-appendix-trace-cost.md``); ``measured_on_by_stage`` repeats it,
    and ``projected`` lists the stages no machine has run. A card of
    another kind moves the rays most: their kernel is double precision,
    which a consumer card computes many times slower than its single.

    ``fetch_pairs`` prices the pair cache's way home, which is as long as
    the pack's; ``check`` is ``full`` or ``read``, a smoke run's default
    being the first and the whole scene's the second. A plan's record is
    enough.
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
    solves = positions
    window = LOW_DURATION_S if low_seconds is None else float(low_seconds)
    if not 0.0 < window <= LOW_DURATION_S:
        raise ValueError(f"the low band is solved for up to {LOW_DURATION_S} s, not {window}")
    cast = RAYS_MEASURED if rays is None else int(rays)
    if low_engine == "lowband":
        counts = [int(c) for c in record.get("cells_a_position", [])]
        counted = batched.solves_needed(
            counts, batched.MEASURED_CARD_GIB, ppw=low_ppw, duration_s=window
        )
        solves = counted if counts and counted is not None else positions
        low = batched.estimate(
            solves,
            pairs,
            fmax_hz=solve_fmax_hz(),
            duration_s=window,
            rate_usd_per_hour=rate_usd_per_hour,
            ppw=low_ppw,
        )
    else:
        if low_ppw is not None or low_seconds is not None:
            raise ValueError("only the batched solver is priced on another grid or duration")
        low = pairs_estimate(
            positions, pairs, fmax_hz=solve_fmax_hz(), rate_usd_per_hour=rate_usd_per_hour
        )
    histograms = int(record["tail_sites"]) * int(record["tail_cells"])
    from reverberate.render.compact import Levers

    compact = not Levers.parse(low_levers).off
    pack_bytes = scene_pairs * pack_pair_bytes(low_levers) + histograms * HISTOGRAM_BYTES
    jobs = int(record["step_pairs"]) + scene_pairs
    cached = float(present.COMPACT_PAIR_BYTES if compact else PAIR_BYTES)
    pair_bytes = pairs * cached * (window / LOW_DURATION_S)
    if check is None:
        check = "full" if dict(record.get("profile", {})).get("seconds") is not None else "read"
    seconds = {
        "fixed": FIXED_S,
        "low": float(low["seconds"]),
        "paths": (PATHS_FIXED_S if jobs else 0.0) + jobs * PATH_JOB_S,
        "rays": int(record["tail_sites"])
        * (RAYS_SITE_S + int(record["tail_cells"]) * RAYS_SITE_CELL_S)
        * (cast / RAYS_MEASURED),
        "level": scene_pairs * LEVEL_PAIR_S,
        "write": scene_pairs * (WRITE_PAIR_S + (WRITE_COMPACT_PAIR_S if compact else 0.0)),
        "check": CHECK_READ_S + (CHECK_FULL_S if check == "full" else 0.0),
        "transfer_pack": pack_bytes / FETCH_BYTES_PER_S,
        "transfer_pairs": pair_bytes / FETCH_BYTES_PER_S if fetch_pairs else 0.0,
    }
    total = float(sum(seconds.values()))
    measured = [name for name in seconds if name in MEASURED_ON]
    return {
        "billed_rate_usd_per_hour": float(rate_usd_per_hour),
        "seconds": {name: round(value, 1) for name, value in seconds.items()},
        "usd": {
            name: round(value / 3600.0 * rate_usd_per_hour, 3) for name, value in seconds.items()
        },
        "total_s": round(total, 1),
        "total_usd": round(total / 3600.0 * rate_usd_per_hour, 2),
        "non_solve_s": round(total - seconds["low"], 1),
        "non_solve_usd": round((total - seconds["low"]) / 3600.0 * rate_usd_per_hour, 3),
        "pack_gb": round(pack_bytes / 1e9, 2),
        "pair_cache_gb": round(pair_bytes / 1e9, 2),
        "pairs_fetched": bool(fetch_pairs),
        "check": check,
        "low": low,
        "low_engine": low_engine,
        "low_ppw": low_ppw,
        # Named only when they are not the reference's, whose records they leave as they were.
        **({} if low_seconds is None else {"low_seconds": window}),
        **({} if rays is None else {"rays": cast}),
        **({"low_levers": str(low_levers)} if compact else {}),
        "solves": solves,
        "extra_solves": solves - positions,
        "measured": measured,
        "projected": [name for name in seconds if name not in MEASURED_ON],
        # The low band's card, as its engine names it; then every measured stage's.
        "measured_on": str(low.get("measured_on", "2 x A100")),
        "measured_on_by_stage": {
            **{name: MEASURED_ON[name] for name in measured},
            "low": str(low.get("measured_on", MEASURED_ON["low"])),
        },
    }
