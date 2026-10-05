"""The air outside a dwelling's outer walls: what it is in a grid, and a grid without it.

An exported storey is wrapped in a shell a little beyond its outer walls,
and the voxel grid holds the air between the two: a ring all round the
dwelling, as wide as the gap and as high as the storey, closed on every side
by nodes of the shell's material. Where an outer wall has an opening that
nothing fills (an entrance whose leaf stops under its lintel, a window), a
source inside reaches that ring, and the solver then updates it: on
hssd_0076 to 1500 Hz it is 5.3 million of the 47.4 million nodes a step
writes, a cavity of 55 cubic metres, 0.37 m wide and 57 m round, lined like
a wall. Outdoors is not that. Sound that leaves by an open door does not
come back a second later from a corridor that does not exist.

Two things are separated here, and neither is done unless asked:

- :func:`outside_air` finds the ring in the air a campaign's sources reach,
  from the grid alone. From each face of the box inwards, a row of nodes
  first meets the shell's own air, which starts at the shell and ends at
  the outer wall, the same number of nodes on nearly every row of that
  face; the rows that pass through an opening go on into the dwelling, and
  are cut where the others end. A face whose rows do not agree on such a
  width, or agree on one wider than a gap can be, has no ring and is left
  alone.
- :func:`without` gives the engine's arrays with those nodes cut off: a
  node of air beside them becomes a boundary node that no longer reads
  them, and carries either nothing (``"rigid"``: the opening is a wall) or
  a material whose admittance is one (``"open"``: the opening absorbs a
  wave that meets it squarely, the open window of room acoustics, and what
  leaves does not return). A problem built from them never reaches the
  ring.

Which of the two an opening should be is a fact of the dwelling (a door
that is shut, a window that is glass) and not of the solver; the grid only
shows that it is neither today.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from reverberate.wave.lowband.problem import EngineArrays, _copies, reached_nodes

__all__ = ["CLOSURES", "MAX_GAP_M", "OPEN_BRANCHES", "outside_air", "reach_of", "without"]

#: What an opening onto the ring becomes.
CLOSURES = ("open", "rigid")
#: The widest gap between a shell and an outer wall that is taken for one, m.
MAX_GAP_M = 0.8
#: Of a face's rows, the share that must end the shell's air on the same node.
AGREEING_ROWS = 0.5
#: One branch, a resistance of the air's own impedance: an admittance of one at every frequency.
OPEN_BRANCHES = np.array([[0.0, 1.0, 0.0]])


def _adjacency(arrays: EngineArrays) -> np.ndarray:
    """Every node's bit mask of the neighbours it reads, all set for a node of plain air."""
    bits = np.zeros(arrays.bn_ixyz.size, dtype=np.uint16)
    for j in range(len(arrays.scheme.neighbours)):
        bits |= arrays.adj_bn[:, j].astype(np.uint16) << np.uint16(j)
    adjacency = np.full(arrays.points, 0xFFFF, dtype=np.uint16)
    adjacency[np.asarray(arrays.bn_ixyz, dtype=np.int64)] = bits
    return adjacency


def reach_of(arrays: EngineArrays, seeds: np.ndarray) -> np.ndarray:
    """The nodes sources at ``seeds`` reach, ``[nx, ny, nz]`` bool: the solver's own reach."""
    nx, ny, nz = arrays.shape
    offsets = np.array(
        [(dx * ny + dy) * nz + dz for dx, dy, dz in arrays.scheme.neighbours], np.int64
    )
    reached = reached_nodes(
        arrays.shape,
        offsets,
        _adjacency(arrays),
        _copies(arrays.shape, arrays.fcc_flag),
        np.asarray(seeds, dtype=np.int64),
    )
    return np.asarray(reached.reshape(arrays.shape))


def outside_air(
    reached: np.ndarray, step_m: float, *, max_gap_m: float = MAX_GAP_M
) -> tuple[np.ndarray, dict[str, Any]]:
    """The reached nodes that lie between the shell and the outer walls, and what was found.

    ``reached`` is ``[nx, ny, nz]`` bool. Each of the box's six faces is
    read on its own. A row of nodes from the face inwards has a first
    reached node and a first run of them; on a face with a ring nearly every
    row's first run ends on the same node, the outer wall's, a gap's width
    from the first node any row reaches. That node is the face's, and on
    each row whose first run starts within the gap, the run is the ring as
    far as it. A gap has the dwelling behind its wall: where most of those
    rows reach nothing further in, the face holds a shallow room and not a
    ring. The record says, a face, the gap found and the rows that agreed,
    or why none was.
    """
    reached = np.asarray(reached, dtype=bool)
    outside = np.zeros(reached.shape, dtype=bool)
    faces = []
    most = int(np.floor(max_gap_m / step_m))
    for axis in range(3):
        for flipped in (False, True):
            rows = np.moveaxis(reached, axis, -1)
            marks = np.moveaxis(outside, axis, -1)
            if flipped:
                rows, marks = rows[..., ::-1], marks[..., ::-1]
            name = f"{'xyz'[axis]}{'+' if flipped else '-'}"
            any_reached = rows.any(axis=-1)
            if not bool(any_reached.any()):
                faces.append({"face": name, "ring": False, "why": "nothing reached"})
                continue
            length = rows.shape[-1]
            index = np.arange(length)
            first = np.argmax(rows, axis=-1)
            gone = ~rows & (index >= first[..., None])
            end = np.where(gone.any(axis=-1), np.argmax(gone, axis=-1), length)
            del gone
            start = int(first[any_reached].min())
            ends = end[any_reached & (first <= start + most)] - start
            counts = np.bincount(ends[ends <= most], minlength=1)
            width = int(np.argmax(counts)) if counts.size else 0
            agreeing = int(counts[max(width - 1, 0) : width + 2].sum()) if counts.size else 0
            share = agreeing / max(int(ends.size), 1)
            wall = start + width
            # A gap has the dwelling behind its wall: a room that is merely shallow has not.
            gap_rows = any_reached & (np.abs(end - wall) <= 1)
            behind = (rows & (index > end[..., None])).any(axis=-1) & gap_rows
            backed = int(np.count_nonzero(behind)) / max(int(np.count_nonzero(gap_rows)), 1)
            found = {
                "face": name,
                "first_node": start,
                "gap_nodes": width,
                "gap_m": round(width * step_m, 4),
                "rows": int(ends.size),
                "rows_agreeing": round(share, 4),
                "rows_with_air_behind": round(backed, 4),
            }
            if width < 1 or share < AGREEING_ROWS:
                faces.append({**found, "ring": False, "why": "the rows agree on no gap"})
                continue
            if backed < AGREEING_ROWS:
                faces.append({**found, "ring": False, "why": "no air behind the wall"})
                continue
            chosen = any_reached & (first < wall)
            mark = (
                (index >= first[..., None])
                & (index < np.minimum(end, wall)[..., None])
                & chosen[..., None]
            )
            before = int(np.count_nonzero(marks))
            marks |= mark
            # The rows that go on past the wall's node: an opening's, and the ring's own
            # where it turns a corner of the dwelling.
            past = int(np.count_nonzero(chosen & (end > wall + 1)))
            faces.append(
                {
                    **found,
                    "ring": True,
                    "nodes": int(np.count_nonzero(marks)) - before,
                    "rows_past_the_wall": past,
                }
            )
    count = int(np.count_nonzero(outside))
    return outside, {
        "nodes": count,
        "share_of_reached": count / max(int(np.count_nonzero(reached)), 1),
        "volume_m3": round(count * step_m**3, 2),
        "faces": faces,
    }


def without(
    arrays: EngineArrays, outside: np.ndarray, *, closure: str = "open"
) -> tuple[EngineArrays, dict[str, Any]]:
    """``arrays`` with the nodes of ``outside`` cut off from the rest, and the cut's record.

    A node of air that reads a node of ``outside`` becomes a boundary node
    that reads its other neighbours, its surface the links cut; a boundary
    node that read one, on the opening's edge, no longer does. With
    ``closure`` ``"rigid"`` the new nodes carry nothing and the edge keeps
    its wall's material. With ``"open"`` both carry
    :data:`OPEN_BRANCHES`, a material added after the entry's own, over
    the faces cut: an edge node has one material, and its wall's own loss
    there is let go. The cut is made from both sides: a node of ``outside``
    no longer reads the dwelling either, so nothing reaches it and it is not
    stepped. The record's ``closing_m2`` is the area of the cut, a link a
    node's face.
    """
    if closure not in CLOSURES:
        raise ValueError(f"an opening is closed as one of {CLOSURES}, not {closure!r}")
    if arrays.fcc_flag:
        raise NotImplementedError("the outside is cut on the Cartesian grid only")
    nx, ny, nz = arrays.shape
    outside = np.asarray(outside, dtype=bool).reshape(arrays.shape)
    neighbours = arrays.scheme.neighbours
    # Per node, the links that cross the cut, either way: bit j for neighbour j. A node of
    # the outside must not read the dwelling either, or it would be reached, and stepped.
    cut = np.zeros(arrays.shape, dtype=np.uint16)
    for j, (dx, dy, dz) in enumerate(neighbours):
        beside = outside.copy()
        source = outside[
            max(dx, 0) : nx + min(dx, 0), max(dy, 0) : ny + min(dy, 0), max(dz, 0) : nz + min(dz, 0)
        ]
        beside[
            max(-dx, 0) : nx + min(-dx, 0),
            max(-dy, 0) : ny + min(-dy, 0),
            max(-dz, 0) : nz + min(-dz, 0),
        ] = source
        cut |= (beside != outside).astype(np.uint16) << np.uint16(j)
    cut_flat = cut.reshape(-1)
    bn = np.asarray(arrays.bn_ixyz, dtype=np.int64)
    is_boundary = np.zeros(arrays.points, dtype=bool)
    is_boundary[bn] = True
    count = len(neighbours)
    before = np.asarray(arrays.adj_bn, dtype=bool)
    held = np.stack([(cut_flat[bn] >> np.uint16(j)) & np.uint16(1) == 1 for j in range(count)], 1)
    # The links a boundary node read and no longer reads: the opening's own edge, where the
    # node already stood against a wall.
    lost = before & held
    adj = before & ~held
    fresh = np.flatnonzero((cut_flat > 0) & ~is_boundary)
    fresh_lost = np.stack(
        [(cut_flat[fresh] >> np.uint16(j)) & np.uint16(1) == 1 for j in range(count)], axis=1
    )
    in_dwelling = ~outside.reshape(-1)
    materials = list(arrays.materials)
    material = np.asarray(arrays.mat_bn, dtype=np.int8).copy()
    saf = np.asarray(arrays.saf_bn, dtype=np.float64).copy()
    fresh_material = np.full(fresh.size, -1, dtype=np.int8)
    closing = in_dwelling[fresh]
    edge = in_dwelling[bn] & lost.any(axis=1)
    if closure == "open":
        fresh_material[closing] = len(materials)
        # A node of the opening's edge has one material: the opening's, over the faces cut.
        material[edge] = len(materials)
        saf[edge] = lost[edge].sum(axis=1)
        materials.append(OPEN_BRANCHES.copy())
    index = np.concatenate([bn, fresh])
    order = np.argsort(index, kind="stable")
    cut_arrays = replace(
        arrays,
        bn_ixyz=index[order],
        adj_bn=np.concatenate([adj, ~fresh_lost])[order],
        mat_bn=np.concatenate([material, fresh_material])[order],
        saf_bn=np.concatenate([saf, fresh_lost.sum(axis=1).astype(np.float64)])[order],
        materials=materials,
    )
    faces = int(fresh_lost[closing].sum()) + int(lost[edge].sum())
    return cut_arrays, {
        "closure": closure,
        "closing_nodes": int(np.count_nonzero(closing)) + int(np.count_nonzero(edge)),
        "closing_m2": round(faces * arrays.h**2, 3),
    }
