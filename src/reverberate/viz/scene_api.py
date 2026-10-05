"""What the page's scene view asks the server: layouts, recipes, and where things are.

The scene view draws a recipe (``docs/formats/scene-recipe.md``) and replays it
without sound. The page holds no rule of its own about a scene: everything it
shows is computed here by :mod:`reverberate.scenes`, the package the trace
reads recipes through, so what is audited by eye is what will be rendered.

Under ``/api/scene/``:

- ``GET  schema``: the generator's parameters, their defaults and the limits a
  value may take, read off :class:`reverberate.scenes.Parameters`.
- ``GET  layout/<dwelling>``: stations, rails, the free floor, rooms, seating.
- ``POST generate``: ``{dwelling, seed, parameters}`` gives a recipe and its
  report. No acoustics is computed.
- ``POST tracks``: ``{canonical, step_s}`` gives every source and the listener
  on a time grid, by ``kinematics.source_state`` and ``listener_state``.
- ``GET  recipes/<dwelling>``: the recipes saved for a dwelling.
- ``GET  recipes/<dwelling>/<name>``: one of them, with its report.
- ``POST recipes/<dwelling>/<name>``: ``{canonical, overwrite}`` saves one.

Saving is the only thing here that writes, and only when asked: recipes go to
``<data root>/recipes/<dwelling>/<name>.json`` in their canonical form.

**A recipe comes back from the page as text.** A report carries the recipe
twice: ``recipe``, the tree the page draws from, and ``canonical``, its
canonical form as a string. The page hands the string back untouched. It
could not hand back the tree: a browser writes ``12.0`` as ``12``, which rule
1 of the format refuses, because one scene would then have two identities.

No library of pinned clips and no asset keys exist yet, so a recipe generated
here names placeholder clips and placeholder assets, and its report says so: a
trace refuses it.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.scenes import (
    GenerationError,
    Layout,
    Parameters,
    Recipe,
    RecipeError,
    canonical_bytes,
    describe,
    generate,
    listener_state,
    load_hssd_layout,
    low_band_positions,
    parse_recipe,
    recipe_sha256,
    sample_times,
    save_recipe,
    source_state,
    validate,
)
from reverberate.scenes.layout import Floor
from reverberate.scenes.validate import (
    FLOOR_RULES,
    HEAD_CLEARANCE_M,
    MAX_SPEED_M_S,
    MAX_TURN_DEG_S,
    MIN_RISE_S,
    RAIL_PITCH_M,
)

__all__ = [
    "PREFIX",
    "SceneError",
    "SceneService",
    "layout_payload",
    "parameter_schema",
    "parameters_from",
    "report",
    "tracks",
]

#: The path every endpoint lives under.
PREFIX = "api/scene"

#: The step tracks are sampled at unless the page asks otherwise, and the
#: range it may ask for. The page interpolates linearly between two samples.
TRACK_STEP_S = 0.2
TRACK_STEP_RANGE_S = (0.05, 2.0)

#: A body larger than this is not a recipe.
MAX_BODY_BYTES = 8 << 20

_DWELLING = re.compile(r"[a-z0-9_]+")
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


#: The widest pitch the panel offers, and the readings whose solves it counts: two positions
#: read linearly, and the eight ``docs/open-questions/rail-interpolation.md`` measured.
RAIL_PITCH_WIDEST_M = 0.12
RAIL_POSITIONS_OFFERED = (2, 8)


class SceneError(Exception):
    """A request that cannot be served, with the HTTP status that says why."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# the generator's parameters
# --------------------------------------------------------------------------

#: What a value may be, by the unit its name ends with: lowest, highest, and
#: the step a control moves by. The first suffix that matches is taken, so
#: ``_m_s`` and ``_deg_s`` are read before ``_s``. The speeds and the turn
#: rates stop at what rule 7 of the format allows.
_LIMITS_BY_UNIT: tuple[tuple[str, float, float, float], ...] = (
    ("count", 0, 16, 1),
    ("_m_s", 0.1, MAX_SPEED_M_S, 0.05),
    ("_deg_s", 0.0, MAX_TURN_DEG_S, 5.0),
    ("_deg", -90.0, 90.0, 1.0),
    ("_db", -60.0, 12.0, 0.5),
    ("_share", 0.0, 1.0, 0.05),
    ("_s", 0.1, 3600.0, 0.1),
    ("_m", 0.1, 30.0, 0.1),
)

#: The few parameters a rule of the format bounds more tightly than their unit.
_LIMITS_BY_FIELD: dict[str, tuple[float, float, float]] = {
    "duration_s": (10.0, 3600.0, 10.0),
    "rise_s": (MIN_RISE_S, 10.0, 0.1),
    "near_distance_m": (HEAD_CLEARANCE_M, 30.0, 0.1),
    "far_distance_m": (HEAD_CLEARANCE_M, 30.0, 0.1),
    "listener_gaze_jitter_deg": (0.0, 90.0, 1.0),
    # Rule 4: a rail's pitch, from the one two positions read linearly need to the widest
    # a trace of more positions (``--rail-positions 8``) was measured at.
    "rail_pitch_m": (RAIL_PITCH_M, RAIL_PITCH_WIDEST_M, 0.01),
}


def _leaves(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, dict):
        found: dict[str, Any] = {}
        for key, value in tree.items():
            found.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return found
    return {prefix: tree}


def _moved(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(type(item)(item + 1 + index) for index, item in enumerate(value))
    return type(value)(value + 1)


def _record_paths() -> dict[str, str]:
    """Where ``Parameters.record`` writes each field, found by moving one field at a time."""
    defaults = Parameters()
    base = _leaves(defaults.record())
    paths: dict[str, str] = {}
    for field in fields(Parameters):
        moved = replace(defaults, **{field.name: _moved(getattr(defaults, field.name))})
        changed = [path for path, value in _leaves(moved.record()).items() if base[path] != value]
        if len(changed) != 1:
            raise RuntimeError(f"Parameters.{field.name} is written at {changed} of the record")
        paths[field.name] = changed[0]
    return paths


def _limits(name: str, key: str) -> tuple[float, float, float]:
    if name in _LIMITS_BY_FIELD:
        return _LIMITS_BY_FIELD[name]
    for suffix, low, high, step in _LIMITS_BY_UNIT:
        if key.endswith(suffix):
            return low, high, step
    raise RuntimeError(f"Parameters.{name} has a unit the scene view does not know: {key}")


def parameter_schema() -> dict[str, Any]:
    """Every field of :class:`Parameters`: its kind, default and limits.

    Nothing is listed by hand: the fields and defaults are the dataclass's,
    ``path`` is where ``Parameters.record`` writes the field in a recipe's
    ``generator.parameters`` (and so its name in the format document), the
    unit is the key's suffix, and the limits follow from the unit.
    """
    defaults = Parameters()
    listed = []
    for name, path in _record_paths().items():
        default = getattr(defaults, name)
        is_range = isinstance(default, tuple)
        sample = default[0] if is_range else default
        group, _, key = path.rpartition(".")
        low, high, step = _limits(name, key)
        integer = isinstance(sample, int)
        listed.append(
            {
                "name": name,
                "path": path,
                "group": group,
                "key": key,
                "kind": "range" if is_range else "scalar",
                "type": "integer" if integer else "number",
                "default": list(default) if is_range else default,
                "min": int(low) if integer else low,
                "max": int(high) if integer else high,
                "step": int(step) if integer else step,
                "fixed": low == high,
            }
        )
    return {"parameters": listed, "seed": {"min": 0, "max": 2**53 - 1, "default": 0}}


def parameters_from(flat: Any) -> Parameters:
    """:class:`Parameters` from ``{field: value}``; a field left out keeps its default.

    A name that is not a field, a value outside its limits or a range written
    backwards is refused, naming the field.
    """
    if flat is None:
        return Parameters()
    if not isinstance(flat, dict):
        raise SceneError(400, "parameters is not an object")
    schema = {entry["name"]: entry for entry in parameter_schema()["parameters"]}
    unknown = sorted(set(flat) - set(schema))
    if unknown:
        raise SceneError(400, f"unknown parameter: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for name, given in flat.items():
        entry = schema[name]
        cast: Callable[[Any], Any] = int if entry["type"] == "integer" else float
        items = given if entry["kind"] == "range" else [given]
        if not isinstance(items, list) or len(items) != (2 if entry["kind"] == "range" else 1):
            raise SceneError(400, f"{name} is not a {entry['kind']}")
        try:
            numbers = [cast(item) for item in items]
        except (TypeError, ValueError) as error:
            raise SceneError(400, f"{name} is not a number") from error
        for number in numbers:
            if not entry["min"] <= number <= entry["max"]:
                raise SceneError(
                    400, f"{name} = {number} is outside {entry['min']} to {entry['max']}"
                )
        if entry["kind"] == "range" and numbers[0] > numbers[1]:
            raise SceneError(400, f"{name}: the minimum {numbers[0]} exceeds {numbers[1]}")
        values[name] = tuple(numbers) if entry["kind"] == "range" else numbers[0]
    return replace(Parameters(), **values)


def _flat(parameters: Parameters) -> dict[str, Any]:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in asdict(parameters).items()
    }


# --------------------------------------------------------------------------
# a dwelling's layout
# --------------------------------------------------------------------------


def _ring(coordinates: Any) -> list[list[float]]:
    return [[round(float(x), 3), round(float(z), 3)] for x, z in coordinates]


def _parts(geometry: Any) -> list[dict[str, Any]]:
    """A polygon or several as ``[{exterior, holes}]``, the shape of a manifest's outline."""
    if geometry is None or geometry.is_empty:
        return []
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return [
        {
            "exterior": _ring(polygon.exterior.coords),
            "holes": [_ring(hole.coords) for hole in polygon.interiors],
        }
        for polygon in polygons
        if polygon.geom_type == "Polygon"
    ]


def layout_payload(layout: Layout) -> dict[str, Any]:
    """What the page draws of a dwelling before any recipe exists."""
    floor = layout.floor
    return {
        "dwelling": layout.dwelling.to_dict(),
        "heights": layout.heights.to_dict(),
        "summary": layout.summary(),
        "stations": [station.to_dict() for station in layout.stations],
        "rails": [rail.to_dict() for rail in layout.rails],
        "free": _parts(floor.free),
        "walkable": _parts(floor.walkable),
        "rooms": [{"name": room.name, "outline": _parts(room.polygon)} for room in floor.rooms],
        "seating": [
            {"name": item.name, "category": item.category, "footprint": _parts(item.footprint)}
            for item in floor.objects
        ],
    }


# --------------------------------------------------------------------------
# a recipe, reported and sampled
# --------------------------------------------------------------------------


def report(recipe: Recipe, floor: Floor | None) -> dict[str, Any]:
    """A recipe with everything the page shows beside it."""
    positions = low_band_positions(recipe)
    restored: dict[str, Any] | None = None
    if recipe.generator is not None:
        try:
            restored = _flat(Parameters.from_record(recipe.generator.parameters))
        except (KeyError, TypeError, ValueError, IndexError):
            # A recipe of another generator: shown, but its panel is not restored.
            restored = None
    libraries = sorted({a.clip.library for source in recipe.sources for a in source.activity})
    return {
        "recipe": recipe.to_dict(),
        # What the page sends back to name this recipe: see the module's note.
        "canonical": canonical_bytes(recipe).decode(),
        "recipe_sha256": recipe_sha256(recipe),
        "describe": describe(recipe),
        "violations": [
            {"rule": violation.rule, "message": violation.message}
            for violation in validate(recipe, floor)
        ],
        # Without the dwelling, rules 3, 4 and 6 are checked in part only.
        "floor_checked": floor is not None,
        "floor_rules": list(FLOOR_RULES),
        # ``total`` is every position, each once: the three kinds it is split into. It was
        # the sum of every count of the record, ``all`` and ``audible`` among them.
        "low_band_positions": {**positions, "total": positions["all"]},
        # What a trace solves, by the positions a source on a rail reads: a dry run's count.
        "low_band_solved": {
            "pitch_m": recipe.rails[0].pitch_m if recipe.rails else RAIL_PITCH_M,
            "by_rail_positions": {
                str(count): low_band_positions(recipe, count)["audible"]
                for count in RAIL_POSITIONS_OFFERED
            },
        },
        "parameters": restored,
        "placeholder_clips": "placeholder" in libraries,
    }


def _column(values: np.ndarray, decimals: int) -> list[float]:
    rounded: list[float] = np.round(values, decimals).tolist()
    return rounded


def _listener_movement(recipe: Recipe) -> list[dict[str, Any]]:
    """What the listener does between keyframes, neighbours of one kind joined."""
    seated = recipe.heights.seated_m + recipe.dwelling.floor_y_m
    spans: list[dict[str, Any]] = []
    frames = recipe.listener.keyframes
    for a, b in zip(frames[:-1], frames[1:], strict=True):
        low = abs(a.position[1] - seated) < 1e-6 and abs(b.position[1] - seated) < 1e-6
        if a.position == b.position:
            kind = "rest"
        elif a.position[1] != b.position[1]:
            kind = "rise"
        else:
            kind = "walk"
        span = {
            "type": kind,
            "start_s": a.t_s,
            "end_s": b.t_s,
            "height": "seated" if low else "standing",
        }
        if kind == "rise":
            span["to"] = "seated" if b.position[1] < a.position[1] else "standing"
        station = a.station if a.station is not None and a.station == b.station else None
        if station is not None:
            span["station"] = station
        last = spans[-1] if spans else None
        if last is not None and all(
            last.get(key) == span.get(key) for key in ("type", "height", "to", "station")
        ):
            last["end_s"] = b.t_s
        else:
            spans.append(span)
    return spans


def tracks(recipe: Recipe, step_s: float = TRACK_STEP_S) -> dict[str, Any]:
    """Every source and the listener on a time grid, and what each is doing.

    The grid is every ``step_s`` from the start, then the end of the scene.
    Positions are the scene's own ``(x, y, z)`` in metres and angles are the
    recipe's, in degrees, the yaw not wrapped. ``movement`` restates a
    source's segments and ``activity`` its intervals, so the page lays out a
    lane without reading the recipe's structure.
    """
    low, high = TRACK_STEP_RANGE_S
    if not low <= step_s <= high:
        raise SceneError(400, f"step_s = {step_s} is outside {low} to {high}")
    times = sample_times(recipe, step_s)
    head = listener_state(recipe, times)
    sources = []
    for source in recipe.sources:
        state = source_state(recipe, source.id, times)
        movement = []
        for segment in source.segments:
            span = segment.to_dict()
            span.pop("facing", None)
            span.pop("profile", None)
            movement.append(span)
        sources.append(
            {
                "id": source.id,
                "kind": source.kind,
                "subtype": source.subtype,
                "x": _column(state.position[:, 0], 3),
                "y": _column(state.position[:, 1], 3),
                "z": _column(state.position[:, 2], 3),
                "yaw_deg": _column(state.yaw_deg, 2),
                "movement": movement,
                "activity": [[a.start_s, a.end_s] for a in source.activity],
            }
        )
    return {
        "recipe_sha256": recipe_sha256(recipe),
        "duration_s": recipe.duration_s,
        "step_s": step_s,
        "t": _column(times, 3),
        "floor_y_m": recipe.dwelling.floor_y_m,
        "heights": recipe.heights.to_dict(),
        "sources": sources,
        "listener": {
            "x": _column(head.position[:, 0], 3),
            "y": _column(head.position[:, 1], 3),
            "z": _column(head.position[:, 2], 3),
            "yaw_deg": _column(head.yaw_deg, 2),
            "pitch_deg": _column(head.pitch_deg, 2),
            "roll_deg": _column(head.roll_deg, 2),
            "movement": _listener_movement(recipe),
        },
    }


# --------------------------------------------------------------------------
# the service the server routes to
# --------------------------------------------------------------------------


def _recipe_of(body: Any) -> Recipe:
    text = body.get("canonical") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise SceneError(400, "the body holds no recipe: canonical, its text, is missing")
    try:
        return parse_recipe(text)
    except RecipeError as error:
        raise SceneError(422, str(error)) from error


class SceneService:
    """The scene view's endpoints over one HSSD download and one recipes folder."""

    def __init__(
        self,
        hssd_root: Path | None,
        recipes_root: Path,
        layout_of: Callable[[str], Layout] | None = None,
    ) -> None:
        self.hssd_root = hssd_root
        self.recipes_root = recipes_root
        self._layout_of = layout_of
        self._layouts: dict[str, Layout] = {}
        # One request at a time: a layout is built once, and the kinematics
        # keep a small cache of their own that is not made for two threads.
        self._lock = threading.Lock()

    # -- dwellings ----------------------------------------------------------

    def layout(self, dwelling: str) -> Layout:
        if not _DWELLING.fullmatch(dwelling):
            raise SceneError(400, f"{dwelling!r} is not a dwelling's name")
        if dwelling not in self._layouts:
            try:
                if self._layout_of is not None:
                    self._layouts[dwelling] = self._layout_of(dwelling)
                elif self.hssd_root is not None:
                    self._layouts[dwelling] = load_hssd_layout(self.hssd_root, dwelling)
                else:
                    raise KeyError(dwelling)
            except (KeyError, OSError, ValueError) as error:
                raise SceneError(404, f"no layout for {dwelling}: {error}") from error
        return self._layouts[dwelling]

    def _floor(self, recipe: Recipe) -> Floor | None:
        try:
            return self.layout(recipe.dwelling.name).floor
        except SceneError:
            return None

    # -- recipes on disk ----------------------------------------------------

    def _path(self, dwelling: str, name: str) -> Path:
        if not _DWELLING.fullmatch(dwelling):
            raise SceneError(400, f"{dwelling!r} is not a dwelling's name")
        if not _NAME.fullmatch(name):
            raise SceneError(400, f"{name!r} is not a recipe's name: letters, digits, _ and -")
        return self.recipes_root / dwelling / f"{name}.json"

    def saved(self, dwelling: str) -> list[dict[str, Any]]:
        """The recipes saved for a dwelling, by name."""
        folder = self._path(dwelling, "x").parent
        listed = []
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            entry: dict[str, Any] = {"name": path.stem}
            try:
                tree = json.loads(path.read_bytes())
                entry["seed"] = tree["seed"]
                entry["duration_s"] = tree["duration_s"]
                entry["sources"] = len(tree["sources"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry["error"] = str(error)
            listed.append(entry)
        return listed

    def load(self, dwelling: str, name: str) -> dict[str, Any]:
        path = self._path(dwelling, name)
        if not path.is_file():
            raise SceneError(404, f"no recipe {name} for {dwelling}")
        try:
            recipe = parse_recipe(path.read_bytes())
        except RecipeError as error:
            raise SceneError(422, f"{name}: {error}") from error
        return {"name": name, **report(recipe, self._floor(recipe))}

    def save(self, dwelling: str, name: str, body: Any) -> dict[str, Any]:
        """Write a recipe in its canonical form; an existing name is kept unless told."""
        recipe = _recipe_of(body)
        path = self._path(dwelling, name)
        if recipe.dwelling.name != dwelling:
            raise SceneError(400, f"the recipe is of {recipe.dwelling.name}, not of {dwelling}")
        if path.exists() and not (isinstance(body, dict) and body.get("overwrite") is True):
            raise SceneError(409, f"{name} exists for {dwelling}")
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = save_recipe(recipe, path)
        return {"name": name, "recipe_sha256": digest, "path": str(path)}

    # -- generation ---------------------------------------------------------

    def generate(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise SceneError(400, "the body is not an object")
        dwelling = body.get("dwelling")
        seed = body.get("seed", 0)
        if not isinstance(dwelling, str):
            raise SceneError(400, "dwelling is missing")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**53:
            raise SceneError(400, "seed is not an integer in 0 <= seed < 2^53")
        parameters = parameters_from(body.get("parameters"))
        layout = self.layout(dwelling)
        try:
            recipe = generate(
                layout,
                parameters,
                seed,
                allow_placeholder_clips=True,
                allow_placeholder_assets=True,
            )
        except GenerationError as error:
            raise SceneError(422, str(error)) from error
        except Exception as error:  # noqa: BLE001
            # Ranges nobody tried: the page shows what broke and stays up.
            raise SceneError(422, f"{type(error).__name__}: {error}") from error
        return {**report(recipe, layout.floor), "placeholder_assets": True}

    # -- routing ------------------------------------------------------------

    def handle(self, method: str, parts: list[str], body: Any = None) -> Any:
        """Answer ``parts``, the path below :data:`PREFIX`; raises :class:`SceneError`."""
        with self._lock:
            match (method, parts):
                case ("GET", ["schema"]):
                    return parameter_schema()
                case ("GET", ["layout", dwelling]):
                    return layout_payload(self.layout(dwelling))
                case ("POST", ["generate"]):
                    return self.generate(body)
                case ("POST", ["tracks"]):
                    step = body.get("step_s", TRACK_STEP_S) if isinstance(body, dict) else None
                    if not isinstance(step, int | float) or isinstance(step, bool):
                        raise SceneError(400, "step_s is not a number")
                    return tracks(_recipe_of(body), float(step))
                case ("GET", ["recipes", dwelling]):
                    self._path(dwelling, "x")
                    return self.saved(dwelling)
                case ("GET", ["recipes", dwelling, name]):
                    return self.load(dwelling, name)
                case ("POST", ["recipes", dwelling, name]):
                    return self.save(dwelling, name, body)
        raise SceneError(404, f"no such endpoint: {method} /{PREFIX}/{'/'.join(parts)}")
