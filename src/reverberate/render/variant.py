"""What a pack is a variant of: the small file beside it, and what it cost.

``python -m reverberate.trace variants`` writes ``variant.json`` in each
variant's home: its name, the options it takes, what it was predicted to
cost on the whole scene and on the excerpt. A pack comes home in
``<home>/pulled/pack.h5``, so the file is looked for beside the pack and
one directory up. What the pack did cost is read in its own provenance,
where the homecoming stamps the rental's records
(:func:`reverberate.trace.driver.stamp_cost`).

Read by the listening kit: the comparison of several packs
(:mod:`reverberate.render.check.many`) names its files by it, and the audit
page shows it beside the pack heard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["VARIANT_FILE", "label_of", "measured_cost", "read_variant", "summary"]

#: Beside a variant's pack: its name, its flags, what it was predicted to cost.
VARIANT_FILE = "variant.json"


def read_variant(pack: Path) -> dict[str, Any] | None:
    """The :data:`VARIANT_FILE` of a pack, or ``None``: beside it, or beside its ``pulled``."""
    pack = Path(pack)
    for directory in (pack.parent, pack.parent.parent):
        path = directory / VARIANT_FILE
        if path.is_file():
            try:
                return dict(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                return None
    return None


def measured_cost(provenance: dict[str, Any]) -> dict[str, Any] | None:
    """What a pack's rental was billed, from its provenance's cost records; ``None`` without."""
    records = provenance.get("cost")
    if not isinstance(records, list) or not records:
        return None
    usd = seconds = 0.0
    for record in records:
        if isinstance(record, dict):
            usd += float(record.get("usd") or 0.0)
            seconds += float(record.get("seconds") or 0.0)
    return {"usd": round(usd, 3), "seconds": round(seconds, 1), "records": len(records)}


def label_of(pack: Path, taken: set[str] | None = None) -> str:
    """A pack's name among others: its variant's, else its home's, made distinct."""
    pack = Path(pack)
    variant = read_variant(pack)
    if variant and variant.get("name"):
        base = str(variant["name"])
    elif pack.parent.name == "pulled":
        base = pack.parent.parent.name
    else:
        base = pack.stem
    base = "".join(c if c.isalnum() or c in "-_." else "-" for c in base) or "pack"
    label, count = base, 2
    while taken is not None and label in taken:
        label, count = f"{base}-{count}", count + 1
    if taken is not None:
        taken.add(label)
    return label


def summary(pack: Path, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the page shows beside a pack: its name, flags and costs, each ``None`` when unknown."""
    variant = read_variant(pack) or {}
    predicted = dict(variant.get("predicted") or {})
    return {
        "name": variant.get("name"),
        "says": variant.get("says"),
        "flags": dict(variant.get("flags") or {}),
        "predicted_scene_usd": dict(predicted.get("scene") or {}).get("usd"),
        "predicted_excerpt_usd": dict(predicted.get("excerpt") or {}).get("usd"),
        "measured": variant.get("measured") or measured_cost(provenance or {}),
    }
