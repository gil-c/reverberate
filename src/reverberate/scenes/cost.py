"""What a recipe will cost to trace, counted from the recipe alone.

A trace's bill is the wave solves, one for every place a source is heard
from under the crossover, and then what each solve is heard at: the cells
that stand where the listener rests and every 0.15 m of its way. Both are
functions of the recipe, and :func:`counts` gives them without the
dwelling's geometry, so that a generator's rules can be judged before
anything is rented. :func:`predict` hands them to the trace's own predictor,
:func:`reverberate.trace.machines.predict`, whose constants are the measured
ones.

**What is exact and what is not.** The source positions are those of
:func:`~.kinematics.low_band_source_positions`, exact. The cells are the
rule's first stage, the listener's rests and its way (``trace.plan``,
``listening_cells``): the cells a trace adds where a source comes very near
the way need the dwelling's surfaces and are not counted; the first scene
had none. A pair is a position and a cell that hears it: counted here with
the cell nearest the head at each audible step, two while the head moves,
which is right at rest and an estimate on the move. The wearer's own voice
is counted at no solve (``docs/open-questions/recipes-v2.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from reverberate.scenes.kinematics import (
    audible_steps,
    listener_state,
    low_band_source_positions,
    sample_times,
    source_state,
)
from reverberate.scenes.recipe import LISTENER, Recipe

__all__ = ["CELL_MERGE_M", "CELL_PITCH_M", "REFERENCE_MACHINE", "counts", "predict"]

#: The trace's rule: a cell every this far along the listener's way, none within the
#: second of another (``trace.plan.PATH_PITCH_M``, ``MERGE_M``), and a tail's sphere on
#: the cells this far apart (``SPACING_M``).
CELL_PITCH_M = 0.15
CELL_MERGE_M = 0.03
TAIL_SPACING_M = 0.80

#: The machine the first scene was traced on, run A of 2026-10-05: eight RTX 3090 billed
#: 1.382 USD an hour together (``docs/open-questions/performance-audit.md``, section 2).
REFERENCE_MACHINE = {"gpu_name": "RTX 3090", "num_gpus": 8, "gpu_ram_gb": 24.0, "dph_total": 1.382}


def _cells(head: np.ndarray) -> tuple[np.ndarray, float]:
    """The cells of a head's track on the 50 ms grid, and the metres it walks."""
    moved = np.linalg.norm(np.diff(head, axis=0), axis=1) > 1e-9
    still = np.ones(head.shape[0], dtype=bool)
    still[:-1] &= ~moved
    still[1:] &= ~moved
    rest = np.rint(head[still] * 1000.0).astype(np.int64)
    cells = [row.astype(float) / 1000.0 for row in np.unique(rest, axis=0)]
    edges = np.flatnonzero(np.diff(np.concatenate([[False], moved, [False]]).astype(int)))
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        way = head[first : last + 1]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1))])
        at = np.append(np.arange(0.0, arc[-1] - 1e-9, CELL_PITCH_M), arc[-1])
        for point in np.stack([np.interp(at, arc, way[:, axis]) for axis in range(3)], axis=1):
            point = np.rint(point * 1000.0) / 1000.0
            if not cells or np.linalg.norm(np.asarray(cells) - point, axis=1).min() > CELL_MERGE_M:
                cells.append(point)
    walked = float(np.linalg.norm(np.diff(head, axis=0), axis=1).sum())
    return np.asarray(cells, dtype=float).reshape(-1, 3), walked


def counts(recipe: Recipe) -> dict[str, Any]:
    """What a trace of the recipe solves and holds: the record its price is read from."""
    times = sample_times(recipe)
    head = listener_state(recipe, times).position
    cells, walked = _cells(head)
    moving = np.zeros(times.size, dtype=bool)
    moving[1:] = np.linalg.norm(np.diff(head, axis=0), axis=1) > 1e-9
    near = cKDTree(cells).query(head, k=min(2, cells.shape[0]))[1].reshape(times.size, -1)
    low = low_band_source_positions(recipe)
    heard: list[set[int]] = [set() for _ in range(low.count)]
    tree = cKDTree(low.positions) if low.count else None
    audible_total = 0
    for source in recipe.sources:
        audible = audible_steps(recipe, source.id)
        audible_total += int(audible.sum())
        own = source.attach is not None and source.attach.to == LISTENER
        if tree is None or not audible.any() or (own and source.attach.at == "mouth"):  # type: ignore[union-attr]
            continue
        where = source_state(recipe, source.id, times[audible]).position
        rows = tree.query(where)[1]
        for row, cell, walks in zip(rows, near[audible], moving[audible], strict=True):
            heard[int(row)].update(int(c) for c in (cell if walks else cell[:1]))
    # A tail's sphere stands on the cells 0.80 m apart, as the trace picks them.
    spheres: list[np.ndarray] = []
    for cell in cells:
        if (
            not spheres
            or np.linalg.norm(np.asarray(spheres) - cell, axis=1).min() >= TAIL_SPACING_M
        ):
            spheres.append(cell)
    kinds = low.counts()
    return {
        "duration_s": float(recipe.duration_s),
        "steps": int(times.size),
        "source_positions": low.count,
        "source_positions_by_kind": kinds,
        "cells": int(cells.shape[0]),
        "path_m": round(walked, 2),
        "pairs": int(sum(len(c) for c in heard)),
        "cells_a_position": [len(c) for c in heard],
        "pairs_of_the_patch": 0,
        # A place a source rests at, or a footfall, is a site of its own; a rail is a few.
        "tail_sites": kinds["stations"]
        + kinds.get("footfalls", 0)
        + -(-kinds["rail_samples"] // 8),
        "tail_cells": len(spheres),
        # With a sway no two audible steps are one position of the early trace.
        "step_pairs": audible_total,
        "audible_steps_total": audible_total,
        "profile": {},
    }


def predict(recipe: Recipe | dict[str, Any], **machine: Any) -> dict[str, Any]:
    """Hours and USD of the recipe's trace on a machine, by the trace's own predictor.

    ``recipe`` is a recipe or a record of :func:`counts` (or a plan's own
    record, which has the same keys). ``machine`` is what
    :func:`reverberate.trace.machines.predict` takes of an offer; left out,
    the machine of the first scene (:data:`REFERENCE_MACHINE`) and the grid
    a trace takes unless told, 7.2 points a wavelength
    (:data:`reverberate.trace.plan.LOW_PPW`).
    """
    from reverberate.trace import machines

    from reverberate.trace.plan import LOW_PPW

    record = counts(recipe) if isinstance(recipe, Recipe) else dict(recipe)
    # On the grid a trace takes unless told; ``low_ppw=10.5`` is the reference.
    found = machines.predict(record, **{**REFERENCE_MACHINE, "low_ppw": LOW_PPW, **machine})
    if found is None:
        raise ValueError("the trace has no price for this machine")
    return {
        "source_positions": int(record["source_positions"]),
        "cells": record.get("cells"),
        "pairs": int(record["pairs"]),
        "hours": round(float(found["hours"]), 3),
        "usd": round(float(found["usd"]), 2),
        "solve_usd": round(
            float(found["work"]["solve_card_s"])
            / 3600.0
            * found["rate_usd_per_hour"]
            / found["cards"],
            2,
        ),
        "seconds": found["seconds"],
        "note": found["note"],
    }
