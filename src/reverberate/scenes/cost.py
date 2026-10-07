"""What a recipe will cost to trace, counted from the recipe alone.

A trace's bill is the wave solves, one for every place a source is heard
from under the crossover, and then what each solve is heard at: the cells
that stand where the listener rests and every 0.15 m of its way.
:func:`counts` gives them without the dwelling's geometry, so that a
generator's rules can be judged before anything is rented, and
:func:`predict` hands them to the trace's own predictor,
:func:`reverberate.trace.machines.predict`, whose constants are the measured
ones.

**One implementation.** :func:`counts` is the trace's own plan,
:func:`reverberate.trace.plan.make_plan`, made without the dwelling's
surfaces (:data:`~reverberate.trace.plan.NO_SURFACES`): the same tracks,
the same cells, the same rule that says which cell serves a step and so
which pairs are solved, the same sites of the tail. What the dwelling adds
to it is all the two can differ by: a cell whose ball a wall cuts gets no
array, a cell is added where a source comes very near the listener's way
and a wall keeps the others from serving, and a fixture too near a surface
is given to the mirror alone. The first scene had none of the three.

**What is not solved.** A source the mirror renders alone
(:func:`reverberate.scenes.wave_band`: the wearer's own voice, a person's
breath and footsteps, the pane of a closed window) is counted at no solve
and no pair; its early trace and its tail's sites are counted, in seconds
of the host and of a card's rays.
"""

from __future__ import annotations

from typing import Any

from reverberate.scenes.kinematics import low_band_source_positions
from reverberate.scenes.recipe import Recipe

__all__ = ["REFERENCE_MACHINE", "counts", "predict"]

#: The machine the first scene was traced on, run A of 2026-10-05: eight RTX 3090 billed
#: 1.382 USD an hour together (``docs/open-questions/performance-audit.md``, section 2).
REFERENCE_MACHINE = {"gpu_name": "RTX 3090", "num_gpus": 8, "gpu_ram_gb": 24.0, "dph_total": 1.382}


def counts(recipe: Recipe, *, low_ppw: float | None = None) -> dict[str, Any]:
    """What a trace of the recipe solves and holds: the plan's record, without the dwelling.

    ``low_ppw`` is the low band's grid the pairs are counted on, its arrays
    on the grid's nodes (:func:`reverberate.trace.plan.on_the_grid`); left
    out, the grid a trace takes unless told
    (:data:`reverberate.trace.plan.LOW_PPW`). ``path_m`` and
    ``source_positions_by_kind`` are added to the plan's record for a
    reader of the recipe.
    """
    from reverberate.trace.plan import LOW_PPW, NO_SURFACES, make_plan

    plan = make_plan(recipe, NO_SURFACES, low_ppw=LOW_PPW if low_ppw is None else float(low_ppw))
    return {
        **plan.record,
        "cell_rule": plan.record["cells"],
        "cells": int(plan.cells.count),
        "path_m": plan.record["cells"]["path_m"],
        "source_positions_by_kind": low_band_source_positions(recipe).counts(),
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
    from reverberate.trace.plan import LOW_PPW, as_made

    # On the grid a trace takes unless told; ``low_ppw=10.5`` is the reference.
    asked = {**REFERENCE_MACHINE, "low_ppw": LOW_PPW, **machine}
    record = (
        counts(recipe, low_ppw=asked["low_ppw"]) if isinstance(recipe, Recipe) else dict(recipe)
    )
    found = machines.predict(record, **asked)
    if found is None:
        raise ValueError("the trace has no price for this machine")
    return {
        "source_positions": int(record["source_positions"]),
        "cells": record.get("cells"),
        # As a machine makes them, its arrays on the grid's nodes: what is priced.
        "pairs": int(as_made(record)["pairs"]),
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
