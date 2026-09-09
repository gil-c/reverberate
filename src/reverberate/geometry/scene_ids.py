"""This project's own name for every HSSD scene.

HSSD's identifiers are unwieldy in two ways. 135 of the 168 are compound --
`105515175_173104107` -- so anything appended to one reads as part of it, and
none of them says anything a reader can hold on to. A scene here is
``hssd_0011``, and a storey of a scene with more than one takes a suffix:
``hssd_0001_1`` is its ground floor. **The suffix is safe on our names where it
was not on theirs**, because a local name is always ``hssd_`` and four digits.

**The table is a record, not a rule.** ``data/scene_ids.csv`` was written once,
in sorted order of HSSD id, and the numbers in it never move: a name printed in
a report or used as a run directory has to keep meaning the same scene. A scene
added later takes the next free number, wherever it sorts. Renumbering would
silently rewrite history.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["SCENE_IDS_FILE", "SceneName", "local_name", "scene_names", "scene_of"]

SCENE_IDS_FILE = Path(__file__).parent / "data" / "scene_ids.csv"


@dataclass(frozen=True)
class SceneName:
    """One HSSD scene under this project's name for it."""

    local: str
    scene_id: str
    #: How many storeys its region annotations were grouped onto. A scene with
    #: one has no suffix; the storeys of any other are numbered from the ground.
    storeys: int


@lru_cache(maxsize=1)
def scene_names() -> dict[str, SceneName]:
    """Every scene, keyed by its HSSD id."""
    with SCENE_IDS_FILE.open() as handle:
        rows = list(csv.DictReader(handle))
    table = {
        row["scene_id"]: SceneName(row["local"], row["scene_id"], int(row["storeys"]))
        for row in rows
    }
    if len(table) != len(rows) or len({e.local for e in table.values()}) != len(table):
        raise ValueError(f"{SCENE_IDS_FILE} names a scene, or uses a name, twice")
    return table


def local_name(scene_id: str, storey: int | None = None) -> str:
    """This project's name for a scene, or for one storey of it, 1-based.

    Passing 1 for a single-storey scene is accepted, so a caller looping over
    storeys need not special-case the common one. Asking for a storey the scene
    does not have is refused rather than quietly answered with the base name,
    which would turn an off-by-one into a plausible name for the wrong thing.
    """
    entry = scene_names().get(scene_id)
    if entry is None:
        raise KeyError(f"{scene_id!r} is not in {SCENE_IDS_FILE.name}")
    if storey is not None and not 1 <= storey <= entry.storeys:
        raise ValueError(f"{entry.local} has {entry.storeys} storeys, asked for {storey}")
    if storey is None or entry.storeys <= 1:
        return entry.local
    return f"{entry.local}_{storey}"


def scene_of(local: str) -> tuple[str, int | None]:
    """The HSSD scene id behind a local name, and which storey it means."""
    by_local = {entry.local: entry for entry in scene_names().values()}
    base, _, tail = local.rpartition("_")
    if base.startswith("hssd_") and tail.isdigit() and len(tail) < 4:
        entry, storey = by_local.get(base), int(tail)
    else:
        entry, storey = by_local.get(local), None
    if entry is None:
        raise KeyError(f"{local!r} is not a name in {SCENE_IDS_FILE.name}")
    if storey is not None and not 1 <= storey <= entry.storeys:
        raise ValueError(f"{entry.local} has {entry.storeys} storeys, asked for {storey}")
    return entry.scene_id, storey
