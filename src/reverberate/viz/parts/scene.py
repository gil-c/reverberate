"""Where everything is while a scene plays, and who is what to the listener.

Two readers give one document, which ``static/scene-view.js`` and
``static/timeline.js`` draw:

- :func:`pack_scene` reads a scene pack: the head and each source's mouth are
  the pack's own tables, a value a step, so what is drawn is where the engine
  rendered them;
- :func:`recipe_scene` reads a recipe alone, through
  :mod:`reverberate.scenes.kinematics`, so that a recipe is looked at before
  anything is traced.

Who is what is the recipe's in both (a pack holds its recipe): :func:`cast`
gives each source a name a person reads, a shape and a colour, its intervals
and, for a voice, whether each is of the listener's conversation, which is
the label a model is trained on (``docs/formats/scene-signal.md``, "The
labels").

The dwelling is :func:`plan_of` a layout, the data the inspector's scene
view draws (``viz.scene_api.layout_payload``): the rooms' outlines, the
walkable floor whose boundary is the walls, the seating's footprints. Where
no dataset is at hand a pack still has a floor, the cells its low band was
solved at.

Positions are in the scene's frame, ``(x, y up, z)``, and yaws in the
recipe's convention: ``0`` faces ``+x``, positive is counter clockwise seen
from above.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from reverberate.render.pack import ScenePack
from reverberate.viz.parts.media import VOICE_COLOURS

__all__ = [
    "NOISE_COLOURS",
    "OWN_COLOUR",
    "STEP_S",
    "cast",
    "head_track",
    "pack_scene",
    "plan_of",
    "recipe_scene",
]

#: The step of what is drawn, seconds.
STEP_S = 0.1
#: Colours of what is not a person, in the order they come: duller than the voices'.
NOISE_COLOURS = ("#a8743a", "#5d7f9e", "#8d8a35", "#86659b", "#4f8f86", "#a85d6e")
#: The wearer's own voice: the listener's head is the white one.
OWN_COLOUR = "#f2f2f2"
#: Kinds drawn as a head: a person. Everything else is a marker.
PEOPLE = ("voice", "own_voice", "near_voice", "far_voice")


def _rounded(values: Any, digits: int = 3) -> list[Any]:
    return list(np.round(np.asarray(values, dtype=float), digits).tolist())


def _words(name: str) -> str:
    """An identifier as words: ``kitchen_counter_53`` is ``kitchen counter``."""
    return re.sub(r"[_\s]+", " ", re.sub(r"_\d+$", "", name)).strip()


def _role_at(roles: Sequence[Mapping[str, Any]], t: float) -> Mapping[str, Any]:
    for role in roles:
        if float(role["start_s"]) <= t < float(role["end_s"]):
            return role
    return roles[-1] if roles else {}


def _label(tree: Mapping[str, Any], number: int, stations: Mapping[str, Any]) -> str:
    kind = str(tree["kind"])
    if kind == "own_voice":
        return "Your voice"
    if kind in PEOPLE:
        found = re.fullmatch(r"[a-z_]*?_?(\d+)", str(tree["id"]))
        return f"Talker {found.group(1) if found else number}"
    said = str(tree.get("subtype") or kind).replace("_", " ").capitalize()
    places = [s.get("station") for s in tree.get("segments", []) if s.get("station") in stations]
    where = _words(str(places[0])) if places else ""
    if tree.get("opening"):
        where = f"{tree['opening'].get('state', '')} window".strip()
    return f"{said}, {where}" if where and where.lower() != said.lower() else said


def cast(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Who is in a recipe (its tree, of either version) and what each is to the listener.

    ``sources`` come people first, in the recipe's order, then the rest:
    the order of a track list and of a timeline's lanes. An interval's
    ``role`` is ``conversation`` (a voice of the listener's conversation
    when it starts, and the wearer's own voice), ``outside`` (any other
    voice) or ``noise`` (what is not a person).
    """
    stations = {str(s["id"]): s for s in recipe.get("stations", []) if s.get("kind") == "fixture"}
    rows: list[dict[str, Any]] = []
    people = things = 0
    for tree in recipe.get("sources", []):
        kind = str(tree["kind"])
        person = kind in PEOPLE
        roles = [dict(role) for role in tree.get("roles", [])]
        if person and kind != "own_voice":
            colour = VOICE_COLOURS[people % len(VOICE_COLOURS)]
            people += 1
        elif kind == "own_voice":
            colour = OWN_COLOUR
        else:
            colour = NOISE_COLOURS[things % len(NOISE_COLOURS)]
            things += 1
        intervals = []
        for interval in tree.get("activity", []):
            start = float(interval["start_s"])
            if kind in ("own_voice", "near_voice"):
                role = "conversation"
            elif not person:
                role = "noise"
            else:
                role = str(_role_at(roles, start).get("role") or "outside")
            made = {"start_s": start, "end_s": float(interval["end_s"]), "role": role}
            for key in ("event", "effort"):
                if interval.get(key):
                    made[key] = interval[key]
            intervals.append(made)
        rows.append(
            {
                "id": str(tree["id"]),
                "what": kind,
                "subtype": tree.get("subtype"),
                "kind": "voice" if person else "noise",
                "shape": "self" if kind == "own_voice" else "head" if person else "marker",
                "label": _label(tree, people, stations),
                "colour": colour,
                "roles": roles,
                "intervals": intervals,
                "level_spl_1m_db": tree.get("level_spl_1m_db"),
                "gain_db": tree.get("gain_db"),
            }
        )
    rows.sort(key=lambda row: row["kind"] != "voice")
    listener = dict(recipe.get("listener") or {})
    names = {row["id"]: row["label"] for row in rows}
    spans = []
    for span in listener.get("conversation", []):
        group, start = span.get("group"), float(span["start_s"])
        members = [
            row["id"]
            for row in rows
            if row["roles"]
            and _role_at(row["roles"], start).get("role") == "conversation"
            and _role_at(row["roles"], start).get("group") == group
        ]
        spans.append(
            {
                "start_s": start,
                "end_s": float(span["end_s"]),
                "group": group,
                "members": members if group else [],
            }
        )
    gaze = [
        {
            "start_s": float(g["start_s"]),
            "end_s": float(g["end_s"]),
            "mode": g.get("mode"),
            "target": g.get("target"),
            "said": (
                f"looking at {names.get(str(g.get('target')), g.get('target'))}"
                if g.get("mode") in ("talker", "glance") and g.get("target")
                else {"reading": "reading", "away": "looking away", "walk": "walking"}.get(
                    str(g.get("mode")), str(g.get("mode") or "")
                )
            ),
        }
        for g in listener.get("gaze", [])
    ]
    return {"sources": rows, "conversation": spans, "gaze": gaze}


def plan_of(layout: Any) -> dict[str, Any]:
    """A dwelling's floor and walls, from its layout: what the inspector's scene view draws."""
    from reverberate.viz.scene_api import layout_payload

    told = layout_payload(layout)
    return {
        "floor_y": float(layout.floor.floor_y_m),
        "rooms": told["rooms"],
        #: The walkable floor: its boundary, holes and all, is the walls.
        "walls": told["walkable"],
        "seating": [
            {"name": _words(item["name"]), "footprint": item["footprint"]}
            for item in told["seating"]
        ],
    }


def _cells(pack: ScenePack, floor_y: float) -> dict[str, Any]:
    """The floor a pack knows by itself: the cells its low band was solved at, each once."""
    cells = np.asarray(pack.cells.position, dtype=float)
    rooms = sorted(set(pack.cells.room))
    seen: dict[tuple[float, float], int] = {}
    for place, room in zip(cells, pack.cells.room, strict=True):
        seen.setdefault((round(float(place[0]), 3), round(float(place[2]), 3)), rooms.index(room))
    return {
        "y": floor_y,
        "pitch_m": float(pack.cells.grid_step_m[0]),
        "rooms": rooms,
        "cells": [[x, z, room] for (x, z), room in seen.items()],
    }


def head_track(pack: ScenePack) -> dict[str, Any]:
    """The scene's own head a step of the pack, for a player that follows it.

    The yaw is unwrapped, so that a head that passes 180 degrees between two
    steps is not turned the long way round between them.
    """
    orientation = np.asarray(pack.listener.orientation, dtype=float)
    yaw = np.degrees(np.unwrap(np.radians(orientation[:, 0])))
    return {
        "step_s": float(pack.header.step_s),
        "yaw_deg": _rounded(yaw, 2),
        "pitch_deg": _rounded(orientation[:, 1], 2),
        "roll_deg": _rounded(orientation[:, 2], 2),
    }


def pack_scene(pack: ScenePack, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole scene of ``pack``, a value every :data:`STEP_S`; ``plan`` is its dwelling's."""
    h = pack.header
    every = max(1, int(round(STEP_S / h.step_s)))
    steps = slice(0, h.steps, every)
    recipe = dict(pack.recipe_json())
    # A synthetic pack's recipe says no kind: the pack's own table does.
    recipe["sources"] = [
        {**tree, "kind": tree.get("kind") or pack.sources[str(tree["id"])].kind}
        for tree in recipe.get("sources", [])
        if str(tree["id"]) in pack.sources
    ]
    made = cast(recipe)
    floor_y = float(dict(recipe.get("dwelling") or {}).get("floor_y_m", 0.0))
    sources = []
    for row in made["sources"]:
        source = pack.sources[row["id"]]
        sources.append(
            {
                **row,
                "position": _rounded(np.asarray(source.position)[steps]),
                "yaw_deg": _rounded(np.asarray(source.yaw_deg)[steps], 1),
            }
        )
    orientation = np.asarray(pack.listener.orientation, dtype=float)[steps]
    return {
        "dwelling": h.dwelling,
        "duration_s": float(h.duration_s),
        "step_s": float(h.step_s * every),
        "plan": None if plan is None else dict(plan),
        "floor": _cells(pack, floor_y),
        "listener": {
            "position": _rounded(np.asarray(pack.listener.position)[steps]),
            "yaw_deg": _rounded(np.degrees(np.unwrap(np.radians(orientation[:, 0]))), 1),
            "pitch_deg": _rounded(orientation[:, 1], 1),
            "conversation": made["conversation"],
            "gaze": made["gaze"],
        },
        "sources": sources,
    }


def recipe_scene(recipe: Any, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The whole scene of a recipe, with no pack: where its kinematics put everything."""
    import json

    from reverberate.scenes import canonical_bytes
    from reverberate.viz.scene_api import tracks

    told = tracks(recipe, STEP_S)
    made = cast(json.loads(canonical_bytes(recipe)))
    by_id = {source["id"]: source for source in told["sources"]}
    sources = []
    for row in made["sources"]:
        track = by_id[row["id"]]
        sources.append(
            {
                **row,
                "position": [list(p) for p in zip(track["x"], track["y"], track["z"], strict=True)],
                "yaw_deg": track["yaw_deg"],
            }
        )
    head = told["listener"]
    return {
        "dwelling": recipe.dwelling.name,
        "duration_s": float(recipe.duration_s),
        "step_s": STEP_S,
        "plan": None if plan is None else dict(plan),
        "floor": None,
        "listener": {
            "position": [list(p) for p in zip(head["x"], head["y"], head["z"], strict=True)],
            "yaw_deg": head["yaw_deg"],
            "pitch_deg": head["pitch_deg"],
            "conversation": made["conversation"],
            "gaze": made["gaze"],
        },
        "sources": sources,
    }
