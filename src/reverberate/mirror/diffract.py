"""The first arrival at a receiver the source does not see: round the obstacles, by their edges.

The geodesic round the occluders (:mod:`reverberate.mirror.occupancy`) gives
the onset's time, direction and loss; the edges (:mod:`reverberate.mirror.edges`)
give the other ways round and pull the geodesic's corners onto real edges;
the last corner's own image tree adds the receiver's room's reflections of it.
It is an estimate of the first arrival, not a wave solution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.csgraph import dijkstra

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.edges import (
    _reflect_the_edges,
    _through_the_edges,
    diffracting_edges,
    snap_to_edges,
)
from reverberate.mirror.geometry import DerivedScene
from reverberate.mirror.ism import Paths
from reverberate.mirror.occupancy import (
    CLEAR_CELLS,
    MIN_DETOUR_M,
    DiffractionSettings,
    _clear,
    _detours,
    _graph,
    _pull,
    maekawa_db,
    occupancy_of,
)

__all__ = ["DiffractionSettings", "diffracted_paths"]


def diffracted_paths(
    scene: DerivedScene,
    source: np.ndarray,
    receivers: np.ndarray,
    indices: list[int],
    *,
    sound_speed_m_s: float,
    settings: DiffractionSettings | None = None,
    say: Any = None,
) -> tuple[dict[int, Paths], dict[str, Any]]:
    """For each receiver in ``indices``: the diffracted onset as a one-path ``Paths``.

    Receivers the geodesic cannot reach are left out. The path's ``order`` is
    0 for the rendering (it carries no reflection) but it is not a direct
    path: the caller keeps it apart.
    """
    settings = settings or DiffractionSettings()
    source = np.asarray(source, dtype=float)
    receivers = np.asarray(receivers, dtype=float)
    chosen = receivers[indices] if indices else np.zeros((0, 3))
    # The whole storey: the way round may leave the receivers' box.
    lo = np.minimum(np.minimum(receivers.min(axis=0), source), scene.bmin) - 0.3
    hi = np.maximum(np.maximum(receivers.max(axis=0), source), scene.bmax) + 0.3
    occupancy = occupancy_of(scene, lo, hi, settings)
    _clear(occupancy, np.vstack([source[None, :], chosen]), CLEAR_CELLS)
    selected = diffracting_edges(scene)
    graph = _graph(occupancy)
    start = int(occupancy.flat(occupancy.cell_of(source))[0])
    distance, predecessor = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    bands = np.asarray(OCTAVE_BANDS, dtype=float)
    shape = occupancy.blocked.shape
    out: dict[int, Paths] = {}
    secondary: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    lengths = []
    for index in indices:
        receiver = receivers[index]
        node = int(occupancy.flat(occupancy.cell_of(receiver))[0])
        if not np.isfinite(distance[node]):
            continue
        nodes = [node]
        while nodes[-1] != start and predecessor[nodes[-1]] >= 0:
            nodes.append(int(predecessor[nodes[-1]]))
        cells = np.stack(np.unravel_index(np.asarray(nodes), shape), axis=1)
        chain = occupancy.centre_of(cells)
        chain[0] = receiver
        chain[-1] = source
        corners = _pull(occupancy, chain)
        loose = _detours(corners)
        corners, on_edge = snap_to_edges(corners, selected)
        legs = [
            float(np.linalg.norm(b - a)) for a, b in zip(corners[:-1], corners[1:], strict=True)
        ]
        length = float(sum(legs))
        loss = np.zeros(len(bands))
        edges = 0
        tight = _detours(corners)
        for k in range(1, len(corners) - 1):
            # A corner on an edge is a bend whatever kink it has left: pulling
            # the chain tight takes the kink out and not the wall. Its detour
            # keeps the grid's slack, which still knows how deep in the shadow
            # the receiver is.
            detour = max(tight[k], loose[k]) if on_edge[k] else tight[k]
            if detour < MIN_DETOUR_M and not on_edge[k]:
                # A corner the grid made, not an edge the sound bends round.
                continue
            edges += 1
            loss += maekawa_db(detour, bands, sound_speed_m_s)
        if edges == 0:
            # Pulled straight within the grid's slack: the shadow boundary's 4.8 dB.
            loss += maekawa_db(0.0, bands, sound_speed_m_s)
        towards = corners[1] - receiver
        direction = towards / max(float(np.linalg.norm(towards)), 1e-9)
        gain = 10.0 ** (-loss / 20.0) / max(length, 1e-3)
        points = np.repeat(receiver[None, None, :], 5, axis=1)
        points[0, 0] = source
        out[index] = Paths(
            receiver=receiver.copy(),
            image=np.asarray([-1], dtype=np.int64),
            order=np.asarray([0], dtype=np.int64),
            length_m=np.asarray([length]),
            direction=direction[None, :],
            gain=gain[None, :],
            points=points,
            sequence=np.full((1, 3), -1, dtype=np.int64),
        )
        # The last corner is a source in the receiver's own room; keep what the
        # room needs to reflect it: where it stands, how far the sound already
        # travelled to reach it, and what the edges took out on the way.
        secondary[index] = (np.asarray(corners[1], dtype=float), length - legs[0], loss)
        lengths.append(length - float(np.linalg.norm(receiver - source)))
    reflected = 0
    trees = 0
    edge_record: dict[str, Any] = {}
    if settings.edges:
        out, secondary, edge_record = _through_the_edges(
            scene, source, receivers, out, secondary, sound_speed_m_s, settings
        )
    if settings.reflections > 0 and secondary:
        out, trees, reflected = _reflect_the_edges(scene, receivers, out, secondary, settings)
    record = {
        "settings": settings.record(),
        "grid": list(shape),
        "blocked_fraction": round(float(occupancy.blocked.mean()), 4),
        "asked": len(indices),
        "found": len(out),
        "detour_m_median": round(float(np.median(lengths)), 3) if lengths else None,
        "edge_trees": trees,
        "reflected_paths": reflected,
        **edge_record,
    }
    if say is not None:
        say(
            f"diffraction: {len(out)} of {len(indices)} points, grid {shape}, "
            f"{trees} edge trees, {reflected} reflected paths"
        )
    return out, record
