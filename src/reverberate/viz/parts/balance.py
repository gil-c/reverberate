"""The balance a listener sets on a track list, kept for whoever writes the next recipes.

A recipe gives each source a gain. Whether the scene that follows is one a
listener would set that way is known only once he has moved the faders:
what he sets is written here, a source a line, so that the generator's
ranges can be drawn towards it. The file (:data:`SCHEMA`) holds for each
source its fader in dB, whether it is muted and whether it is soloed, and
what the balance was set on.

:func:`gains` is the one place a balance becomes factors: a solo silences
every source that is not soloed, a mute silences its own, and a fader at
the bottom of its travel is silence and not -60 dB.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reverberate.viz.parts.server import HttpError

__all__ = ["FADER_DB", "SCHEMA", "clean", "gains", "load", "save"]

SCHEMA = "reverberate.apps.balance"
#: A fader's travel, dB; at its bottom the source is silent.
FADER_DB = (-60.0, 12.0)


def clean(body: Any, sources: Iterable[str]) -> dict[str, dict[str, Any]]:
    """A balance as the page sends it, held to ``sources`` and to a fader's travel."""
    if not isinstance(body, Mapping) or not isinstance(body.get("sources"), Mapping):
        raise HttpError(400, "a balance is {sources: {id: {gain_db, mute, solo}}}")
    given = body["sources"]
    out: dict[str, dict[str, Any]] = {}
    for name in sources:
        row = given.get(name) or {}
        if not isinstance(row, Mapping):
            raise HttpError(400, f"the balance of {name} is not an object")
        try:
            level = float(row.get("gain_db", 0.0))
        except (TypeError, ValueError) as error:
            raise HttpError(400, f"the gain of {name} is not a number") from error
        if level != level:
            raise HttpError(400, f"the gain of {name} is not a number")
        out[name] = {
            "gain_db": min(max(level, FADER_DB[0]), FADER_DB[1]),
            "mute": bool(row.get("mute", False)),
            "solo": bool(row.get("solo", False)),
        }
    unknown = sorted(set(given) - set(out))
    if unknown:
        raise HttpError(400, f"no source named {', '.join(unknown)}")
    return out


def gains(balance: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """The factor of each source under ``balance``: what the mix multiplies its stem by."""
    soloed = any(row.get("solo") for row in balance.values())
    out = {}
    for name, row in balance.items():
        level = float(row.get("gain_db", 0.0))
        silent = bool(row.get("mute")) or (soloed and not row.get("solo")) or level <= FADER_DB[0]
        out[name] = 0.0 if silent else 10.0 ** (level / 20.0)
    return out


def save(
    path: Path, balance: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write ``balance`` at ``path``, whole; the document, returned."""
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "saved": datetime.now(UTC).isoformat(timespec="seconds"),
        "context": dict(context or {}),
        "sources": {name: dict(row) for name, row in balance.items()},
        "gains": gains(balance),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".part")
    scratch.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    scratch.replace(path)
    return document


def load(path: Path, sources: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The balance at ``path`` for ``sources``; every fader at 0 dB where there is no file."""
    names = list(sources)
    path = Path(path)
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            kept = {k: v for k, v in dict(document.get("sources") or {}).items() if k in names}
            return clean({"sources": kept}, names)
        except (ValueError, HttpError):
            pass
    return clean({"sources": {}}, names)
