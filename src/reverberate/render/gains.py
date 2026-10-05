"""A source's gain in a traced pack, set in place and put back: no pair is solved again.

A source's gain is one scalar the engine multiplies its dry signal by,
the attribute ``gain_db`` of ``/sources/<id>``, which the trace copied
from the recipe. Nothing else of a pack depends on it: the responses are
those of a unit source. So the gains of a traced scene are a choice that
can be made after it, which is how the first whole scene's noises, drawn
12 dB too low (``docs/open-questions/first-scene-defects.md``), are
brought to where they are heard.

:func:`set_gains`, ``python -m reverberate.render gains PACK``, writes the
attribute **in place**. What the trace wrote is kept beside it the first
time, as the attribute ``gain_db_traced``, and every gain asked as a
change (``--kind noise --add 12``) is a change from that one, so asking
twice does not add twice. ``--undo`` puts every source back and removes
the copies.

**The pack stays honest.** The embedded recipe and its digest are not
touched: they are the scene that was drawn and traced, and what ties the
pack to its recipe's file and to the other packs of that recipe. That the
pack no longer renders that recipe's gains is said in its provenance,
under ``gains``: per source that differs, the recipe's gain and the one
rendered. A reader that wants the recipe's levels reads ``gain_db_traced``
where it is, ``gain_db`` elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["TRACED", "set_gains"]

#: The attribute that keeps the gain the trace wrote, in a pack whose gain was set.
TRACED = "gain_db_traced"


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def set_gains(
    path: Path,
    *,
    gains_db: Mapping[str, float] | None = None,
    add_db: Mapping[str, float] | None = None,
    undo: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Set the gains of the pack at ``path`` in place; returns what every source has.

    ``gains_db`` maps a source's id to its gain in dB; ``add_db`` maps a
    kind of source to what is added to the traced gain of every source of
    that kind. A source named in the first is not moved by the second. A
    source neither names keeps what it has. ``undo`` puts the traced
    gains back; ``dry_run`` writes nothing.
    """
    import h5py

    gains_db, add_db = dict(gains_db or {}), dict(add_db or {})
    report: dict[str, Any] = {"pack": str(path), "written": not dry_run, "sources": {}}
    with h5py.File(Path(path), "r" if dry_run else "r+") as f:
        sources = f["sources"]
        unknown = sorted(set(gains_db) - set(sources))
        if unknown:
            raise KeyError(f"the pack holds no source {', '.join(unknown)}")
        kinds = {_text(group.attrs["kind"]) for group in sources.values()}
        absent = sorted(set(add_db) - kinds)
        if absent:
            raise KeyError(f"the pack holds no source of kind {', '.join(absent)}")
        changed: dict[str, Any] = {}
        for name, group in sources.items():
            kind = _text(group.attrs["kind"])
            has = float(group.attrs["gain_db"])
            traced = float(group.attrs[TRACED]) if TRACED in group.attrs else has
            if undo:
                wanted = traced
            elif name in gains_db:
                wanted = float(gains_db[name])
            elif kind in add_db:
                wanted = traced + float(add_db[kind])
            else:
                wanted = has
            if not dry_run:
                group.attrs["gain_db"] = wanted
                if wanted == traced:
                    group.attrs.pop(TRACED, None)
                else:
                    group.attrs[TRACED] = traced
            if wanted != traced:
                changed[name] = {"recipe_db": traced, "db": wanted}
            row = {"kind": kind, "traced_db": traced, "was_db": has, "db": wanted}
            report["sources"][name] = row
        provenance = json.loads(_text(f.attrs["provenance_json"]))
        provenance.pop("gains", None)
        if changed:
            provenance["gains"] = {
                "recipe": "the sources named are rendered at another gain than the recipe's",
                "sources": changed,
            }
        if not dry_run:
            f.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        report["gains"] = provenance.get("gains")
    return report
