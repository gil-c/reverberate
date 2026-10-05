"""The diffracted onset of the steps whose source the listener does not see.

What :func:`reverberate.mirror.diffract.diffracted_paths` gives a lattice
point without a direct path, given to a step: the geodesic round the
occluders, the single edges beside it, and the reflections of the edge in
the listener's own room. The parts that cost are shared or batched:

- *once per scene* (:func:`onset_field`): the occupancy grid, its graph of
  free cells, the diffracting edges;
- *once per cell the source stands in*: Dijkstra's distances from it. A
  source at a station is one of them however long it speaks;
- *once per block of steps*: the legs of every (step, edge) pair against the
  occluders, and the reflections of every step's edge, through the same
  validation as the image paths (:func:`reverberate.mirror.moving._validate`).

A step's onset is the present pipeline's for that source and that listener
**alone**. The pipeline's own answer at a lattice point also depends on
which other points were asked with it: it frees the occupancy round every
one of them, and points whose edges stand within 0.25 m share the image
tree of the first. Here the cells freed are the field's, fixed for the
scene, and every step reflects its own edge.

The rows carry what the pack asks beyond the pipeline's paths: the departure
direction, towards the first corner from the source, and an identity made
of the edges the path bends round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import to_numpy
from reverberate.mirror.edges import (
    MAX_PATHS,
    Edges,
    _aperture,
    _nearest_on_segment,
    diffracting_edges,
    snap_to_edges,
)
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import IsmSettings, _gains
from reverberate.mirror.moving import (
    KIND_DIFFRACTED,
    KIND_DIFFRACTED_REFLECTED,
    MovingScene,
    _blocked,
    _Rows,
    _unit,
    _validate,
    path_id,
)
from reverberate.mirror.occupancy import (
    CLEAR_CELLS,
    GROW_CELLS,
    MIN_DETOUR_M,
    SAMPLES_PER_EDGE,
    DiffractionSettings,
    Occupancy,
    _clear,
    _detours,
    _graph,
    _pull,
    maekawa_db,
    occupancy_of,
)
from reverberate.mirror.shared import Store, name_of

__all__ = ["OnsetField", "onset_field", "onset_rows"]

#: Legs to and from an edge are shrunk by this, as ``edges.edge_paths`` shrinks them.
EDGE_EPSILON_M = 0.01
#: Distance fields kept, one per cell a source stood in.
ROOTS_KEPT = 32
#: Distance fields solved by one call of Dijkstra. An undirected search transposes
#: the graph at every call, which costs four times one field's search: a call a
#: field spent four fifths of the onsets' time on it (RTX 3090's host, hssd_0076).
ROOTS_PER_CALL = 16


@dataclass
class OnsetField:
    """What the onsets of one scene share. Built by :func:`onset_field`."""

    occupancy: Occupancy
    graph: csr_matrix
    edges: Edges
    settings: DiffractionSettings
    sound_speed_m_s: float
    #: A field is kept as its predecessors alone: a cell is reached when it has one.
    _roots: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    solved: int = 0
    #: Fields found in the store instead of solved.
    read: int = 0
    #: What names the field, and where its distance fields are kept between processes.
    key: str = ""
    store: Store | None = None

    def root(self, start: int) -> np.ndarray:
        """Dijkstra's predecessors from cell ``start``: negative where the cell is not reached.

        The start itself has none either, and is reached.
        """
        found = self._roots.get(start)
        if found is None:
            self.solve([start])
            found = self._roots[start]
        return found

    def solve(self, starts: Any) -> None:
        """The fields of ``starts`` that are not held, :data:`ROOTS_PER_CALL` to a call.

        A field is the same whichever call solved it, and whichever process:
        each start is its own search, and with a store a field another
        process solved is mapped and not solved again. The last
        :data:`ROOTS_KEPT` are kept, those just asked for among them.
        """
        wanted = list(dict.fromkeys(int(s) for s in starts))
        missing = [s for s in wanted if s not in self._roots]
        store = self.store
        names = {start: name_of("distance field 1", self.key, start) for start in missing}
        claims: dict[int, Any] = {}
        theirs: list[int] = []
        if store is not None:
            for start in list(missing):
                if self._read(start, names[start]):
                    missing.remove(start)
                    continue
                claims[start] = store.claim("fields", names[start])
                if claims[start] is None:
                    # Another process is solving this field now: read when it has.
                    theirs.append(start)
                    missing.remove(start)
                elif self._read(start, names[start]):
                    claims.pop(start).close()
                    missing.remove(start)
        try:
            for first in range(0, len(missing), ROOTS_PER_CALL):
                self._search(missing[first : first + ROOTS_PER_CALL], names)
        finally:
            for claim in claims.values():
                if claim is not None:
                    claim.close()
        for start in theirs:
            assert store is not None
            store.claim("fields", names[start], wait=True).close()
            if not self._read(start, names[start]):
                # The process that was solving it is gone: solved here.
                self._search([start], names)
        for start in wanted:
            self._roots[start] = self._roots.pop(start)
        while len(self._roots) > max(ROOTS_KEPT, len(wanted)):
            self._roots.pop(next(iter(self._roots)))

    def _read(self, start: int, name: str) -> bool:
        """The field of ``start`` from the store, when it is there."""
        kept = None if self.store is None else self.store.load("fields", name)
        if kept is None:
            return False
        self._roots[start] = kept["predecessor"]
        self.read += 1
        return True

    def _search(self, share: list[int], names: dict[int, str]) -> None:
        """One call of Dijkstra for the starts of ``share``, each field kept and stored."""
        _, predecessor = dijkstra(
            self.graph, directed=False, indices=share, return_predecessors=True
        )
        for row, start in enumerate(share):
            self._roots[start] = np.asarray(predecessor[row])
            if self.store is not None:
                self.store.save("fields", names[start], {"predecessor": self._roots[start]})
        self.solved += len(share)


def onset_field(
    catalogue: DerivedScene,
    clear: np.ndarray,
    *,
    sound_speed_m_s: float,
    settings: DiffractionSettings | None = None,
    store: Store | None = None,
) -> OnsetField:
    """The scene's occupancy, graph and edges.

    ``clear`` is every position a source or the listener takes in the
    scene, ``[point, 3]``: the cells round each are freed, as the pipeline
    frees them round its source and its points, so that a mouth or a head
    close to a surface is not walled in by the grid. Give the whole
    recipe's positions and every trace of it reads one field.

    With a ``store`` the first process that asks makes the field and the
    others map it, and so with each distance field solved on it.
    """
    settings = settings or DiffractionSettings()
    clear = np.atleast_2d(np.asarray(clear, dtype=float))
    key = name_of(
        "onset field 1",
        catalogue.key,
        settings.record(),
        [GROW_CELLS, SAMPLES_PER_EDGE, CLEAR_CELLS],
        clear,
    )

    def build() -> dict[str, np.ndarray]:
        lo = np.minimum(clear.min(axis=0), catalogue.bmin) - 0.3
        hi = np.maximum(clear.max(axis=0), catalogue.bmax) + 0.3
        occupancy = occupancy_of(catalogue, lo, hi, settings)
        _clear(occupancy, clear, CLEAR_CELLS)
        graph = _graph(occupancy)
        edges = diffracting_edges(catalogue)
        return {
            "blocked": occupancy.blocked,
            "origin": occupancy.origin,
            "cell_m": np.asarray(occupancy.cell_m),
            "graph_data": graph.data,
            "graph_indices": graph.indices,
            "graph_indptr": graph.indptr,
            **{f"edges_{name}": getattr(edges, name) for name in _EDGE_FIELDS},
        }

    made = build() if store is None else store.make("onsets", key, build)[0]
    occupancy = Occupancy(
        np.asarray(made["blocked"]), np.asarray(made["origin"]), float(made["cell_m"])
    )
    return OnsetField(
        occupancy=occupancy,
        graph=csr_matrix(
            (made["graph_data"], made["graph_indices"], made["graph_indptr"]),
            shape=(occupancy.blocked.size, occupancy.blocked.size),
        ),
        edges=Edges(
            a=np.asarray(made["edges_a"]),
            b=np.asarray(made["edges_b"]),
            label=np.asarray(made["edges_label"]),
            facet=np.asarray(made["edges_facet"]),
            length_m=np.asarray(made["edges_length_m"]),
        ),
        settings=settings,
        sound_speed_m_s=float(sound_speed_m_s),
        key=key,
        store=store,
    )


_EDGE_FIELDS = ("a", "b", "label", "facet", "length_m")


@dataclass
class _Onset:
    """One step's onset as it is built: the pipeline's columns, and the pack's beside them."""

    length: np.ndarray
    direction: np.ndarray
    gain: np.ndarray
    order: np.ndarray
    sequence: np.ndarray
    #: Per row: the edges it bends round from the source, or ``None`` when a corner is the grid's.
    names: list[tuple[int, ...] | None]
    #: Per row: the first point of the path after the source.
    leaves_by: np.ndarray
    #: The corner the listener's room reflects: where, how far the sound has
    #: come to it, what the edges took, its edge and where the path left the source by.
    corner: np.ndarray
    before_m: float
    loss_db: np.ndarray
    corner_name: tuple[int, ...] | None
    corner_leaves_by: np.ndarray


def _edge_of(point: np.ndarray, edges: Edges) -> int:
    """The edge a point stands on: the nearest, the first of those that share a rim."""
    feet = _nearest_on_segment(point, edges.a, edges.b)
    return int(np.argmin(np.linalg.norm(feet - point, axis=1)))


def _geodesic(
    held: OnsetField, source: np.ndarray, receiver: np.ndarray, width: int
) -> _Onset | None:
    """The way round the occluders: the body of ``diffracted_paths``' loop, for one step."""
    occupancy = held.occupancy
    shape = occupancy.blocked.shape
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    c = held.sound_speed_m_s
    start = int(occupancy.flat(occupancy.cell_of(source))[0])
    predecessor = held.root(start)
    node = int(occupancy.flat(occupancy.cell_of(receiver))[0])
    if node != start and predecessor[node] < 0:
        return None
    nodes = [node]
    while nodes[-1] != start and predecessor[nodes[-1]] >= 0:
        nodes.append(int(predecessor[nodes[-1]]))
    cells = np.stack(np.unravel_index(np.asarray(nodes), shape), axis=1)
    chain = occupancy.centre_of(cells)
    chain[0] = receiver
    chain[-1] = source
    corners = _pull(occupancy, chain)
    loose = _detours(corners)
    corners, on_edge = snap_to_edges(corners, held.edges)
    legs = [float(np.linalg.norm(b - a)) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    length = float(sum(legs))
    loss = np.zeros(len(bands))
    bends = 0
    tight = _detours(corners)
    named: list[int] | None = []
    for k in range(1, len(corners) - 1):
        detour = max(tight[k], loose[k]) if on_edge[k] else tight[k]
        if detour < MIN_DETOUR_M and not on_edge[k]:
            continue
        bends += 1
        loss += maekawa_db(detour, bands, c)
        if named is not None:
            named = [*named, _edge_of(corners[k], held.edges)] if on_edge[k] else None
    if bends == 0:
        loss += maekawa_db(0.0, bands, c)
        named = None
    towards = corners[1] - receiver
    direction = towards / max(float(np.linalg.norm(towards)), 1e-9)
    gain = 10.0 ** (-loss / 20.0) / max(length, 1e-3)
    # The corners run from the listener to the source; a name runs from the source.
    name = None if named is None else tuple(reversed(named))
    leaves_by = np.asarray(corners[-2], dtype=float)
    near_edge = (_edge_of(corners[1], held.edges),) if len(corners) > 2 and on_edge[1] else None
    return _Onset(
        length=np.asarray([length]),
        direction=direction[None, :],
        gain=gain[None, :],
        order=np.zeros(1, dtype=np.int64),
        sequence=np.full((1, width), -1, dtype=np.int32),
        names=[name],
        leaves_by=leaves_by[None, :],
        corner=np.asarray(corners[1], dtype=float),
        before_m=length - legs[0],
        loss_db=loss,
        corner_name=near_edge,
        corner_leaves_by=leaves_by,
    )


def _through_the_edges(
    ms: MovingScene,
    held: OnsetField,
    onsets: dict[int, _Onset],
    job_source: np.ndarray,
    job_listener: np.ndarray,
    xp: Any,
) -> int:
    """``edges._through_the_edges`` for every step at once; how many steps found an edge."""
    edges = held.edges
    jobs = sorted(onsets)
    if edges.count == 0 or not jobs:
        return 0
    count = edges.count
    a = np.tile(edges.a, (len(jobs), 1))
    b = np.tile(edges.b, (len(jobs), 1))
    source = np.repeat(job_source[jobs], count, axis=0)
    receiver = np.repeat(job_listener[jobs], count, axis=0)
    point = _aperture(a, b, source, receiver)
    first = np.linalg.norm(point - source, axis=1)
    second = np.linalg.norm(point - receiver, axis=1)
    total = first + second
    straight = np.asarray([float(np.linalg.norm(job_listener[j] - job_source[j])) for j in jobs])
    near = total <= np.repeat(straight, count) + held.settings.max_edge_detour_m
    near &= second > 1e-3
    tried = np.flatnonzero(near)
    device = ms.on(xp)
    block = ms.moving.tests_per_block
    clear = np.zeros(total.size, dtype=bool)
    cut = to_numpy(
        _blocked(
            xp, device, xp.asarray(source[tried]), xp.asarray(point[tried]), EDGE_EPSILON_M, block
        )
    )
    tried = tried[~cut]
    cut = to_numpy(
        _blocked(
            xp,
            device,
            xp.asarray(point[tried]),
            xp.asarray(receiver[tried]),
            EDGE_EPSILON_M,
            block,
        )
    )
    clear[tried[~cut]] = True
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    found = 0
    for k, job in enumerate(jobs):
        rows = slice(k * count, (k + 1) * count)
        picked = np.flatnonzero(near[rows])
        ordered = picked[np.argsort(total[rows][picked])]
        keep = ordered[clear[rows][ordered]]
        if keep.size == 0:
            continue
        lengths = total[rows][keep]
        apertures = point[rows][keep]
        towards = apertures - job_listener[job]
        directions = towards / np.maximum(np.linalg.norm(towards, axis=1, keepdims=True), 1e-9)
        gains = np.zeros((keep.size, bands.size))
        for i, index in enumerate(keep):
            loss = maekawa_db(float(total[rows][index] - straight[k]), bands, held.sound_speed_m_s)
            gains[i] = 10.0 ** (-loss / 20.0) / max(float(total[rows][index]), 1e-3)
        # Two facets can share one rim; one arrival is enough.
        marker = np.round(np.column_stack([lengths, directions]), 3)
        _, first_of = np.unique(marker, axis=0, return_index=True)
        pick = np.sort(first_of)
        pick = pick[np.argsort(lengths[pick])[:MAX_PATHS]]
        lengths, directions, gains = lengths[pick], directions[pick], gains[pick]
        apertures, which = apertures[pick], keep[pick]
        onset = onsets[job]
        nearest = int(np.argmin(lengths))
        aperture = apertures[nearest]
        before = float(np.linalg.norm(aperture - job_source[job]))
        loss_db = -20.0 * np.log10(np.maximum(gains[nearest] * float(lengths[nearest]), 1e-12))
        names: list[tuple[int, ...] | None] = [(int(e),) for e in which]
        leaves_by = apertures
        if float(onset.length[0]) < float(lengths.min()) - 0.01:
            lengths = np.concatenate([onset.length[:1], lengths])
            directions = np.vstack([onset.direction[:1], directions])
            gains = np.vstack([onset.gain[:1], gains])
            names = [onset.names[0], *names]
            leaves_by = np.vstack([onset.leaves_by[:1], leaves_by])
        # One barrier's worth of sound, shared over the ways round it.
        share = np.sqrt(np.sum(onset.gain[0] ** 2) / max(float(np.sum(gains**2)), 1e-300))
        onsets[job] = _Onset(
            length=lengths,
            direction=directions,
            gain=gains * share,
            order=np.zeros(lengths.size, dtype=np.int64),
            sequence=np.full((lengths.size, onset.sequence.shape[1]), -1, dtype=np.int32),
            names=names,
            leaves_by=leaves_by,
            corner=aperture,
            before_m=before,
            loss_db=loss_db[: len(OCTAVE_BANDS)],
            corner_name=(int(which[nearest]),),
            corner_leaves_by=aperture,
        )
        found += 1
    return found


def _reflect_the_edges(
    ms: MovingScene,
    held: OnsetField,
    onsets: dict[int, _Onset],
    job_listener: np.ndarray,
    xp: Any,
) -> int:
    """``edges._reflect_the_edges`` with every step's own edge as the source; the paths added."""
    if held.settings.reflections > 1:
        raise NotImplementedError(
            "a moving onset reflects its edge once; the pipeline's default is one reflection"
        )
    jobs = sorted(onsets)
    facets = ms.normals.shape[0]
    if not jobs or facets == 0:
        return 0
    ism = IsmSettings(max_order=1, flutter_order=0)
    corner = np.repeat(np.stack([onsets[j].corner for j in jobs]), facets, axis=0)
    receiver = np.repeat(job_listener[jobs], facets, axis=0)
    sequence = np.tile(np.arange(facets, dtype=np.int64), len(jobs))[:, None]
    index, image, _ = _validate(
        xp,
        ms,
        xp.asarray(corner),
        xp.asarray(receiver),
        xp.asarray(sequence),
        xp.ones(sequence.shape[0], dtype=xp.int64),
        xp.ones(sequence.shape[0], dtype=xp.int64),
        region=None,
        epsilon_m=ism.epsilon_m,
    )
    added = 0
    width = next(iter(onsets.values())).sequence.shape[1]
    for k, job in enumerate(jobs):
        mine = index[(index >= k * facets) & (index < (k + 1) * facets)]
        if mine.size == 0:
            continue
        chosen = np.isin(index, mine)
        onset = onsets[job]
        facet = sequence[mine]
        after = np.linalg.norm(image[chosen] - job_listener[job][None, :], axis=1)
        direction = (image[chosen] - job_listener[job][None, :]) / np.maximum(after, 1e-12)[:, None]
        gain = _gains(ms.catalogue, facet, after)
        total = after + onset.before_m
        # One point source at the edge: the spreading is over the whole way,
        # and the edges' loss applies.
        scale = (after / np.maximum(total, 1e-6))[:, None] * 10.0 ** (-onset.loss_db / 20.0)[
            None, :
        ]
        padded = np.full((mine.size, width), -1, dtype=np.int32)
        padded[:, 0] = facet[:, 0]
        lengths = np.concatenate([onset.length, total])
        pick = np.argsort(lengths)[: min(MAX_PATHS, lengths.size)]
        names = [*onset.names, *([onset.corner_name] * mine.size)]
        onsets[job] = _Onset(
            length=lengths[pick],
            direction=np.vstack([onset.direction, direction])[pick],
            gain=np.vstack([onset.gain, gain * scale])[pick],
            order=np.concatenate([onset.order, np.ones(mine.size, dtype=np.int64)])[pick],
            sequence=np.concatenate([onset.sequence, padded])[pick],
            names=[names[int(p)] for p in pick],
            leaves_by=np.vstack(
                [onset.leaves_by, np.repeat(onset.corner_leaves_by[None, :], mine.size, axis=0)]
            )[pick],
            corner=onset.corner,
            before_m=onset.before_m,
            loss_db=onset.loss_db,
            corner_name=onset.corner_name,
            corner_leaves_by=onset.corner_leaves_by,
        )
        added += int(np.count_nonzero(pick >= onset.length.size))
    return added


def onset_rows(
    ms: MovingScene,
    held: OnsetField,
    job_source: np.ndarray,
    job_listener: np.ndarray,
    shadowed: np.ndarray,
    xp: Any,
    rows: _Rows,
) -> dict[str, Any]:
    """The onsets of the jobs in ``shadowed``, added to ``rows``; the record of what was found."""
    width = ms.width
    solved = held.solved
    onsets: dict[int, _Onset] = {}
    # By the cell the source stands in: a field is solved once, with its neighbours.
    shadowed = np.asarray(shadowed, dtype=np.int64).reshape(-1)
    occupancy = held.occupancy
    start = np.asarray(occupancy.flat(occupancy.cell_of(job_source[shadowed])), dtype=np.int64)
    cells = np.unique(start)
    for first in range(0, cells.size, ROOTS_KEPT):
        share = cells[first : first + ROOTS_KEPT]
        held.solve(share)
        for job in (int(j) for j in shadowed[np.isin(start, share)]):
            onset = _geodesic(held, job_source[job], job_listener[job], width)
            if onset is not None:
                onsets[job] = onset
    with_edges = 0
    reflected = 0
    if held.settings.edges:
        with_edges = _through_the_edges(ms, held, onsets, job_source, job_listener, xp)
    if held.settings.reflections > 0:
        reflected = _reflect_the_edges(ms, held, onsets, job_listener, xp)
    for job in sorted(onsets):
        onset = onsets[job]
        count = onset.length.size
        kind = np.where(onset.order == 0, KIND_DIFFRACTED, KIND_DIFFRACTED_REFLECTED)
        # A path without a name takes its rank by delay among the step's such paths.
        ids = np.zeros(count, dtype=np.uint64)
        nameless = {KIND_DIFFRACTED: 0, KIND_DIFFRACTED_REFLECTED: 0}
        for row in (int(r) for r in np.argsort(onset.length, kind="stable")):
            facets = onset.sequence[row][: int(onset.order[row])]
            name = onset.names[row]
            if name is None:
                ids[row] = path_id(int(kind[row]), facets, rank=nameless[int(kind[row])])
                nameless[int(kind[row])] += 1
            else:
                ids[row] = path_id(int(kind[row]), facets, name)
        ids = _apart(ids, kind, onset)
        rows.add(
            np.full(count, job),
            path_id=ids,
            length_m=onset.length,
            arrival=onset.direction,
            departure=_unit(onset.leaves_by - job_source[job][None, :]),
            gain=onset.gain,
            order=onset.order,
            kind=kind,
            sequence=onset.sequence,
            rank=np.arange(count),
        )
    return {
        "settings": held.settings.record(),
        "asked": int(len(shadowed)),
        "found": len(onsets),
        "distance_fields": held.solved - solved,
        "with_edge_paths": with_edges,
        "reflected_paths": reflected,
    }


def _apart(ids: np.ndarray, kind: np.ndarray, onset: _Onset) -> np.ndarray:
    """Identities made distinct within the step: a repeated name takes a rank instead.

    Two rims on one line carry two edge indices and may yield one name
    after rounding; the pack holds no identity twice in a step.
    """
    seen: set[int] = set()
    out = ids.copy()
    spare = 1 << 20
    for row in (int(r) for r in np.argsort(onset.length, kind="stable")):
        while int(out[row]) in seen:
            facets = onset.sequence[row][: int(onset.order[row])]
            out[row] = path_id(int(kind[row]), facets, rank=spare)
            spare += 1
        seen.add(int(out[row]))
    return out
