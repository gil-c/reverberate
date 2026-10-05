"""Two rooms and a doorway, voxelised at two steps: what the A/B view is proved on.

Two real grids of one dwelling exist only once two traces have been rented.
Until then the comparison (:func:`reverberate.viz.computed_grid.compare`)
is exercised on a dwelling small enough to voxelise anywhere: a box of two
rooms parted by a sheet with a doorway in it, written as the model the
project's own voxeliser reads (:func:`reverberate.accel.voxelise.voxelise_scene`,
on ``numpy``) and voxelised at whatever step is asked. Nothing here is a
dwelling of the dataset, and a page that shows these grids says so.

The model is in the scene's frame, ``y`` up. ``origin`` moves it, so that it
can be stood inside a real scene's frame for a check by eye.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.trace.computed import GridRecord, grid_record

__all__ = ["ROOMS_M", "WALL_X_M", "rooms_grid", "write_rooms"]

#: The box of the two rooms, m: ``x`` by ``y`` (up) by ``z``.
ROOMS_M = (4.0, 2.4, 3.0)
#: The partition's place along ``x``, and the doorway's first edge along ``z`` and its top.
WALL_X_M = 2.0
DOOR_Z_M = 1.0
DOOR_TOP_M = 2.0

_BRANCHES = np.array([[4.24161203e-01, 1.17781161e02, 6.54109887e04]])


def _rectangle(a: tuple[float, float, float], b: tuple[float, float, float]) -> list[list[float]]:
    """The four corners of an axis aligned rectangle of constant ``x`` between two corners."""
    return [[a[0], a[1], a[2]], [a[0], b[1], a[2]], [a[0], b[1], b[2]], [a[0], a[1], b[2]]]


def write_rooms(
    path: Path, *, door_m: float = 0.9, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Path:
    """The model of the two rooms, with a doorway ``door_m`` wide; returns ``path``."""
    sx, sy, sz = ROOMS_M
    pts = [
        [0, 0, 0],
        [sx, 0, 0],
        [sx, sy, 0],
        [0, sy, 0],
        [0, 0, sz],
        [sx, 0, sz],
        [sx, sy, sz],
        [0, sy, sz],
    ]
    # Wound so that the area normal points into the box, as the unit box of the tests.
    tris = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [3, 6, 7],
        [3, 2, 6],
        [0, 7, 4],
        [0, 3, 7],
        [1, 6, 2],
        [1, 5, 6],
    ]
    sheets = [
        _rectangle((WALL_X_M, 0.0, 0.0), (WALL_X_M, sy, DOOR_Z_M)),
        _rectangle((WALL_X_M, 0.0, DOOR_Z_M + door_m), (WALL_X_M, sy, sz)),
        _rectangle((WALL_X_M, DOOR_TOP_M, DOOR_Z_M), (WALL_X_M, sy, DOOR_Z_M + door_m)),
    ]
    wall_pts = [corner for sheet in sheets for corner in sheet]
    wall_tris = [
        [4 * k + i for i in corners]
        for k in range(len(sheets))
        for corners in ((0, 1, 2), (0, 2, 3))
    ]
    shift = np.asarray(origin, dtype=float)[None, :]
    mats = {
        "shell": {
            "pts": (np.asarray(pts, dtype=float) + shift).tolist(),
            "tris": tris,
            "sides": [2] * len(tris),
            "color": [200, 200, 200],
        },
        "partition": {
            "pts": (np.asarray(wall_pts, dtype=float) + shift).tolist(),
            "tris": wall_tris,
            # A sheet: it parts the air on both of its sides.
            "sides": [3] * len(wall_tris),
            "color": [120, 160, 200],
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mats_hash": mats, "sources": [], "receivers": []}))
    return path


def rooms_grid(
    work: Path,
    step_m: float,
    *,
    door_m: float = 0.9,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sources: np.ndarray | None = None,
    key: str = "",
) -> tuple[GridRecord, Path, dict[str, Any]]:
    """The two rooms voxelised at ``step_m`` under ``work``: the record, the entry, the report.

    The reach starts from ``sources``, by default one source in the first
    room: the second room is reached through the doorway or not at all.
    """
    from reverberate.accel.voxelise import voxelise_scene

    work = Path(work)
    model = write_rooms(work / "rooms.json", door_m=door_m, origin=origin)
    for label in ("shell", "partition"):
        with h5py.File(work / f"{label}.h5", "w") as handle:
            handle.create_dataset("DEF", data=_BRANCHES)
    entry = work / "entry"
    ppw = 10.5
    report = voxelise_scene(
        model,
        entry,
        mat_folder=work,
        mat_files={"shell": "shell.h5", "partition": "partition.h5"},
        fmax=343.2 / (step_m * ppw),
        ppw=ppw,
        nh=4,
        xp=np,
        seal_pockets=False,
    )
    if sources is None:
        sources = np.asarray(origin, dtype=float)[None, :] + np.array([[1.0, 1.2, 1.5]])
    record = grid_record(
        entry, sources, key=key or f"two-rooms-{step_m * 1000:.0f}mm", labels=("partition", "shell")
    )
    return record, entry, report.record()
