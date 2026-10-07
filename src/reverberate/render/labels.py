"""The labels of a pack: what each stem is, interval by interval, beside the stems.

A rendered scene is a mixture and its stems, and what a model is trained to
keep is not said by the audio: it is the recipe's. :func:`labels` reads it
from the pack, which holds its recipe, and gives one document
(``docs/formats/scene-signal.md``, "The labels"): per source, in the pack's
order, which is the stems' order, its kind and subtype, what it is to the
listener over each interval (``roles``, the training label of a recipe of
version 2), every interval it sounds in with its vocal effort and the level
it is emitted at, and what the stem holds of it. :func:`write_labels` writes
it; ``python -m reverberate.render labels PACK [OUT]`` and ``render mix``
call it.

**Levels are said once and applied once.** A clip is stored at the level
its library says (``docs/formats/clip-library.md``): a voice at 60 dB SPL
at 1 m, a noise at its entry's level, full scale standing for 86 dB SPL at
1 m. The engine multiplies it by the source's ``gain_db`` and the
interval's, and by nothing else. A version 2 recipe's ``level_spl_1m_db``
is the stored level and the source's gain together, said so that a reader
need not know the library: it is a label, and no stage applies it. An
interval's ``level_spl_1m_db`` here is that plus the interval's gain: what
the source is at 1 m in free field while the interval lasts, full scale of
the rendered signal read as :data:`FULL_SCALE_SPL_1M_DB`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.render.pack import ScenePack, read_pack

__all__ = [
    "FULL_SCALE_SPL_1M_DB",
    "LABELS_SUFFIX",
    "SCHEMA",
    "SCHEMA_VERSION",
    "labels",
    "labels_path",
    "write_labels",
]

SCHEMA = "reverberate.scene-labels"
SCHEMA_VERSION = 1
#: What full scale of a clip, and so of a rendered signal, stands for at 1 m from a
#: source in free field (``reverberate.scenes.clips.FULL_SCALE_SPL_1M_DB``).
FULL_SCALE_SPL_1M_DB = 86.0
#: Beside ``<name>.f32`` and ``<name>.json``, the labels are ``<name>.labels.json``.
LABELS_SUFFIX = ".labels.json"

#: What a stem holds of its source.
WHOLE = "whole"
ROOM = "room"


def _role_at(roles: list[dict[str, Any]], t: float) -> dict[str, Any]:
    for role in roles:
        if float(role["start_s"]) <= t < float(role["end_s"]):
            return role
    return roles[-1] if roles and t >= float(roles[-1]["end_s"]) else {}


def _source(pack: ScenePack, tree: dict[str, Any]) -> dict[str, Any]:
    held = pack.sources[str(tree["id"])]
    level = tree.get("level_spl_1m_db")
    roles = list(tree.get("roles", []))
    intervals = []
    for interval in tree.get("activity", []):
        made: dict[str, Any] = {
            "start_s": interval["start_s"],
            "end_s": interval["end_s"],
            "clip": interval["clip"]["name"],
            "clip_library": interval["clip"]["library"],
            "gain_db": interval["gain_db"],
        }
        if level is not None:
            made["level_spl_1m_db"] = round(float(level) + float(interval["gain_db"]), 2)
        for key in ("effort", "event"):
            if key in interval:
                made[key] = interval[key]
        if roles:
            # What the voice is to the listener when the interval starts.
            role = _role_at(roles, float(interval["start_s"]))
            made["role"] = role.get("role")
            if "group" in role:
                made["group"] = role["group"]
        intervals.append(made)
    out: dict[str, Any] = {
        "id": tree["id"],
        "kind": tree["kind"],
        "subtype": tree.get("subtype"),
        # ``whole``: the source as heard in the room. ``room``: the room's answer alone,
        # the direct sound left to the device stage (the wearer's own voice).
        "stem": WHOLE if held.direct else ROOM,
        # What rendered the band under the crossover: the wave solver, or the mirror.
        "low_band": "mirror" if held.mirror_only else "wave" if held.low is not None else "none",
        "gain_db": tree["gain_db"],
        "level_spl_1m_db": level,
        "activity": intervals,
    }
    if held.mirror_only:
        out["mirror_only_because"] = held.mirror_only_because
    if roles:
        out["roles"] = roles
    for key in ("attach", "opening"):
        if key in tree:
            out[key] = tree[key]
    if held.band_gain_db:
        out["band_gain_db"] = [float(v) for v in held.band_gain_db]
        out["band_gain_of"] = held.band_gain_of
    return out


def labels(pack: ScenePack) -> dict[str, Any]:
    """The labels of ``pack``: a document of plain values, the sources in the pack's order."""
    recipe = pack.recipe_json()
    trees = {str(tree["id"]): tree for tree in recipe.get("sources", [])}
    listener = dict(recipe.get("listener", {}))
    h = pack.header
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "recipe_sha256": h.recipe_sha256,
        "recipe_version": int(recipe.get("schema_version", 1)),
        "dwelling": h.dwelling,
        "duration_s": h.duration_s,
        "sample_rate_hz": h.sample_rate_hz,
        "levels": {
            "full_scale_spl_1m_db": FULL_SCALE_SPL_1M_DB,
            "applied": "the source's gain_db and the interval's, on the clip as stored",
        },
        "scene": recipe.get("scene"),
        "listener": {
            # The group the listener talks in, and what the head was turned to: the
            # other half of a voice's role.
            "conversation": listener.get("conversation", []),
            "gaze": listener.get("gaze", []),
        },
        "sources": [_source(pack, trees[name]) for name in pack.sources if name in trees],
    }


def labels_path(signal: Path) -> Path:
    """Where the labels of a signal written at ``signal`` are: ``<name>.labels.json``."""
    signal = Path(signal)
    stem = signal.with_suffix("") if signal.suffix in (".f32", ".json") else signal
    return stem.with_name(stem.name + LABELS_SUFFIX)


def write_labels(target: Path, pack: ScenePack | Path) -> Path:
    """The labels of ``pack`` at ``target``, sorted keys, whole or not at all; the path."""
    target = Path(target)
    if isinstance(pack, ScenePack):
        document = labels(pack)
    else:
        with read_pack(Path(pack)) as held:
            document = labels(held)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(_plain(document), indent=1, sort_keys=True) + "\n")
    partial.replace(target)
    return target


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
