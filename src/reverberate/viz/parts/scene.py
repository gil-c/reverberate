"""Where everything is while a scene plays, for a view that shows a listener only that.

Read from a scene pack and from nothing else: the head and each source's
mouth are the pack's own tables, a value a step, so what is drawn is where
the engine rendered them. The dwelling is drawn from the pack too: the cells
the trace solved the low band at are the floor a head can stand on, 0.40 m
apart, each with its room's name. No mesh is loaded and no dataset is
needed.

Positions are in the scene's frame, ``(x, y up, z)``, and yaws in the
recipe's convention: ``0`` faces ``+x``, positive is counter clockwise seen
from above.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from reverberate.render.pack import ScenePack
from reverberate.viz.parts.media import colour_of

__all__ = ["scene_view"]


def _rounded(values: np.ndarray, digits: int = 3) -> list[Any]:
    return list(np.round(np.asarray(values, dtype=float), digits).tolist())


def scene_view(
    pack: ScenePack, window_s: tuple[float, float] | None = None, *, every: int = 2
) -> dict[str, Any]:
    """The floor, the listener and the sources of ``pack`` over ``window_s``, a value in ``every``.

    Times in the answer run from the window's start, as a player's do.
    """
    h = pack.header
    last = h.steps - 1
    start_s, stop_s = window_s or (0.0, last * h.step_s)
    first = int(round(max(start_s, 0.0) / h.step_s))
    stop = int(round(min(stop_s, last * h.step_s) / h.step_s))
    steps = slice(first, stop + 1, every)
    floor_y = float(dict(json.loads(pack.recipe).get("dwelling") or {}).get("floor_y_m", 0.0))
    cells = np.asarray(pack.cells.position, dtype=float)
    rooms = sorted(set(pack.cells.room))
    # A cell stands at several heights; the floor has it once.
    seen: dict[tuple[float, float], int] = {}
    for place, room in zip(cells, pack.cells.room, strict=True):
        seen.setdefault((round(float(place[0]), 3), round(float(place[2]), 3)), rooms.index(room))
    counts = {"voice": 0, "noise": 0}
    sources = []
    for name, source in pack.sources.items():
        kind = "noise" if source.kind == "noise" else "voice"
        sources.append(
            {
                "id": name,
                "kind": kind,
                "said": source.subtype or source.kind.replace("_", " "),
                "colour": colour_of(counts[kind], kind),
                "position": _rounded(np.asarray(source.position)[steps]),
                "yaw_deg": _rounded(np.asarray(source.yaw_deg)[steps], 1),
                "audible": np.asarray(source.audible)[steps].astype(int).tolist(),
            }
        )
        counts[kind] += 1
    orientation = np.asarray(pack.listener.orientation, dtype=float)[steps]
    return {
        "dwelling": h.dwelling,
        "window_s": [first * h.step_s, stop * h.step_s],
        "step_s": h.step_s * every,
        "floor": {
            "y": floor_y,
            "pitch_m": float(pack.cells.grid_step_m[0]),
            "rooms": rooms,
            "cells": [[x, z, room] for (x, z), room in seen.items()],
        },
        "listener": {
            "position": _rounded(np.asarray(pack.listener.position)[steps]),
            "yaw_deg": _rounded(orientation[:, 0], 1),
            "pitch_deg": _rounded(orientation[:, 1], 1),
        },
        "sources": sources,
    }
