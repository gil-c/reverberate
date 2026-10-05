"""Which side of a pair to solve from: the source position, the listening cell, or a mix.

A pair is one source position heard at one cell. A forward solve from a
source position gives it at every cell; by reciprocity a solve from a cell
gives one of its 64 channels at every source position, so a cell costs 64
solves. With a solver that runs many solves at once a solve is a solve
whichever side it starts from, and the question is only how many are needed.

Every pair must be covered by its source position or by its cell. Choosing
the cells to solve reciprocally is a minimum weight vertex cover of the
bipartite graph of pairs, a source position weighing one solve and a cell
64, which a maximum flow solves exactly (:func:`solve_counts`). The answer
on a recipe is a count, not an opinion.

**What a reciprocal solve would need**, and why it is not built unless the
count asks for it. A channel of a cell is the fit's row applied to the
pressure at the array's thousand nodes: per frequency, a weighted sum. The
reciprocal source is therefore the array itself, each node driven with the
impulse response of its weight, and the reading is the pressure at the
source position's eight nodes with the source's trilinear weights. The fit
is the same operator transposed, so the two sides agree to rounding in
exact arithmetic: no multipole is approximated beyond what the forward fit
already approximates. What it costs: a thousand driven nodes with their own
time signals per solve (the fit's operator brought to the grid's rate, about
130 MB a solve at the grid to 1500 Hz), and the differentiated drive a
single precision solve needs, applied to each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

__all__ = ["solve_counts"]


def solve_counts(heard_at: list[list[int]], cells: int, *, channels: int = 64) -> dict[str, Any]:
    """Solves needed forward, reciprocally, and by the best mix of both.

    ``heard_at[s]`` lists the cells source position ``s`` is heard at.
    Returns the three counts, and for the mix which cells are solved
    reciprocally and how many source positions remain to solve forward.
    """
    positions = [s for s, listed in enumerate(heard_at) if listed]
    used = sorted({int(c) for listed in heard_at for c in listed})
    if any(c < 0 or c >= cells for c in used):
        raise ValueError("a pair names a cell that does not exist")
    forward = len(positions)
    reciprocal = channels * len(used)
    row_of = {s: i for i, s in enumerate(positions)}
    column_of = {c: i for i, c in enumerate(used)}
    # Nodes: 0 the origin, then the source positions, then the cells, then the sink.
    first_cell = 1 + forward
    sink = first_cell + len(used)
    heads, tails, capacity = [], [], []
    for s in positions:
        heads.append(0)
        tails.append(1 + row_of[s])
        capacity.append(1)
        for c in sorted(set(heard_at[s])):
            heads.append(1 + row_of[s])
            tails.append(first_cell + column_of[int(c)])
            capacity.append(channels + 1)
    for c in used:
        heads.append(first_cell + column_of[c])
        tails.append(sink)
        capacity.append(channels)
    size = sink + 1
    graph = csr_matrix(
        (np.asarray(capacity, dtype=np.int32), (heads, tails)), shape=(size, size), dtype=np.int32
    )
    flow = maximum_flow(graph, 0, sink)
    # The cover is read off the minimum cut: a source position still reachable from the
    # origin in the residual graph is not solved forward, and a reachable cell is.
    residual = graph - flow.flow
    residual.eliminate_zeros()
    reachable = np.zeros(size, dtype=bool)
    reachable[breadth_first_order(residual, 0, return_predecessors=False)] = True
    reciprocal_cells = [c for c in used if reachable[first_cell + column_of[c]]]
    forward_left = [s for s in positions if not reachable[1 + row_of[s]]]
    mixed = len(forward_left) + channels * len(reciprocal_cells)
    if mixed != int(flow.flow_value):
        raise AssertionError("the cover read off the cut is not the size of the flow")
    per_cell = np.bincount(
        np.asarray([c for listed in heard_at for c in set(listed)], dtype=np.int64),
        minlength=cells,
    )
    return {
        "pairs": int(sum(len(set(listed)) for listed in heard_at)),
        "source_positions": forward,
        "cells_used": len(used),
        "forward_solves": forward,
        "reciprocal_solves": reciprocal,
        "mixed_solves": mixed,
        "mixed_reciprocal_cells": reciprocal_cells,
        "mixed_forward_positions": len(forward_left),
        "cells_heard_from_more_than_channels": int((per_cell > channels).sum()),
        "most_positions_at_one_cell": int(per_cell.max()) if per_cell.size else 0,
    }
