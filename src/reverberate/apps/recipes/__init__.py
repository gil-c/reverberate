"""The recipes of a folder: listed with what matters, one drawn, one thrown away, one looked at.

``python -m reverberate.apps.recipes [FOLDER]`` finds every scene recipe
under ``FOLDER`` (``docs/formats/scene-recipe.md``) and shows a line each:
how long it is, how many people talk and in how many groups, the noises, how
calm it is, the conversation over the noise, and what its trace is predicted
to cost. For the one chosen it lists every rule broken. It draws a new one
from a dwelling, a seed, one of the three presets and a length, moves one to
the folder's ``trash`` (nothing is ever deleted), and opens one in a viewer:
the scene player's view and timeline, with no sound and no pack, so that a
recipe is judged before any money is spent on it.

It holds no rule of its own: the summary is
:func:`reverberate.scenes.describe`'s and :mod:`reverberate.scenes.levels`',
the rules are :func:`reverberate.scenes.validate`'s, the cost is
:func:`reverberate.scenes.cost.predict`'s, the draw is
:func:`reverberate.scenes.generate_social`'s, and where everything is in the
viewer is :mod:`reverberate.scenes.kinematics`'.
"""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.scenes import (
    GenerationError,
    Layout,
    Recipe,
    RecipeError,
    SocialParameters,
    canonical_bytes,
    describe,
    generate_social,
    load_clip_library,
    load_hssd_layout,
    parse_recipe,
    placeholder_assets,
    recipe_sha256,
    save_recipe,
    validate,
)
from reverberate.viz.parts.scene import cast, plan_of, recipe_scene
from reverberate.viz.parts.server import AppServer, HttpError, Request

__all__ = ["LIBRARY", "PRESETS", "STATIC", "TRASH", "Recipes", "build"]

STATIC = Path(__file__).parent / "static"
#: A file larger than this is not read: a recipe of twenty minutes is a third of a megabyte.
MAX_BYTES = 8 << 20
#: How deep under the folder recipes are looked for.
DEPTH = 4
MARK = b'"reverberate.scene-recipe"'
#: Where a recipe thrown away goes, under the folder; never listed, never emptied here.
TRASH = "trash"
PRESETS = ("quiet", "medium", "lively")
#: The clips a recipe drawn here names: the library of the first scene.
LIBRARY = Path(__file__).parents[2] / "scenes" / "library" / "clarify_v1.json"
#: The lengths a recipe may be drawn at, seconds.
DURATION_S = (20.0, 1200.0)
_DWELLING = re.compile(r"[a-z0-9_]{1,40}")


class Recipes:
    """The recipes under a folder, each read once for as long as its file does not change."""

    def __init__(self, folder: Path, *, hssd_root: Path | None = None) -> None:
        self.folder = Path(folder).resolve()
        self.hssd_root = hssd_root if hssd_root is not None and Path(hssd_root).is_dir() else None
        self._lock = threading.Lock()
        self._read: dict[tuple[str, int, int], Recipe | RecipeError] = {}
        self._reports: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._layouts: dict[str, Layout | None] = {}

    # --- the folder ---------------------------------------------------------------------------

    def names(self) -> list[str]:
        """Every recipe under the folder, by its path from it; what is in the trash is not one."""
        found: list[str] = []
        if not self.folder.is_dir():
            return found
        for path in sorted(self.folder.rglob("*.json")):
            relative = path.relative_to(self.folder)
            if len(relative.parts) > DEPTH or relative.parts[0] == TRASH or not path.is_file():
                continue
            if path.stat().st_size > MAX_BYTES or MARK not in path.read_bytes():
                continue
            found.append(str(relative))
        return found

    def _path(self, name: str) -> Path:
        path = (self.folder / name).resolve()
        if self.folder not in path.parents or not path.is_file() or path.suffix != ".json":
            raise HttpError(404, f"no recipe named {name!r}")
        if path.relative_to(self.folder).parts[0] == TRASH:
            raise HttpError(404, f"no recipe named {name!r}")
        return path

    def _key(self, name: str) -> tuple[str, int, int]:
        stat = self._path(name).stat()
        return name, stat.st_size, stat.st_mtime_ns

    def _recipe(self, name: str) -> Recipe | RecipeError:
        key = self._key(name)
        with self._lock:
            if key not in self._read:
                try:
                    self._read[key] = parse_recipe(self._path(name).read_bytes())
                except RecipeError as error:
                    self._read[key] = error
            return self._read[key]

    def _layout(self, dwelling: str) -> Layout | None:
        """The dwelling's layout from the dataset, read once; nothing where it cannot be."""
        with self._lock:
            if dwelling not in self._layouts:
                made: Layout | None = None
                if self.hssd_root is not None and _DWELLING.fullmatch(dwelling):
                    try:
                        made = load_hssd_layout(self.hssd_root, dwelling)
                    except (KeyError, OSError, ValueError):
                        made = None
                self._layouts[dwelling] = made
            return self._layouts[dwelling]

    # --- a line a recipe ----------------------------------------------------------------------

    @staticmethod
    def row(name: str, recipe: Recipe | RecipeError) -> dict[str, Any]:
        """What a recipe says of itself at once: who is in it, how long, how calm."""
        if isinstance(recipe, RecipeError):
            return {"name": name, "error": f"rule {recipe.rule}: {recipe.message}"}
        import json

        made = cast(json.loads(canonical_bytes(recipe)))
        people = [s for s in made["sources"] if s["shape"] == "head"]
        groups = sorted({r["group"] for s in people for r in s["roles"] if r.get("group")})
        return {
            "name": name,
            "error": None,
            "sha256": recipe_sha256(recipe),
            "version": recipe.schema_version,
            "dwelling": recipe.dwelling.name,
            "seed": recipe.seed,
            "duration_s": recipe.duration_s,
            "people": len(people),
            "own_voice": any(s["shape"] == "self" for s in made["sources"]),
            "groups": len(groups),
            "noises": [
                str(s["subtype"] or s["what"]).replace("_", " ")
                for s in made["sources"]
                if s["shape"] == "marker"
            ],
            "calmness": None if recipe.scene is None else recipe.scene.calmness,
        }

    def listing(self) -> list[dict[str, Any]]:
        return [self.row(name, self._recipe(name)) for name in self.names()]

    def report(self, name: str, recipe: Recipe | RecipeError) -> dict[str, Any]:
        """A recipe's line, and what takes a moment: the rules, the levels, the cost."""
        row = self.row(name, recipe)
        if isinstance(recipe, RecipeError):
            broken = [{"rule": recipe.rule, "message": recipe.message}]
            return {**row, "valid": False, "violations": broken}
        layout = self._layout(recipe.dwelling.name)
        try:
            found = validate(recipe, None if layout is None else layout.floor)
            summary = describe(recipe)
        except Exception as error:  # noqa: BLE001
            # A recipe so far from the format that the rules themselves fail on it: said, and
            # the list stays up.
            said = f"the rules could not be applied: {type(error).__name__}: {error}"
            return {**row, "valid": False, "violations": [{"rule": 0, "message": said}]}
        placeholders = []
        if dict(
            dict(recipe.generator.parameters if recipe.generator else {}).get("clips") or {}
        ).get("placeholder"):
            placeholders.append("clips")
        if recipe.assets == placeholder_assets():
            placeholders.append("the dwelling's computed assets")
        return {
            **row,
            "summary": summary,
            "valid": not found,
            "violations": [{"rule": v.rule, "message": v.message} for v in found],
            "floor_checked": layout is not None,
            "speech_over_noise": _speech_over_noise(recipe),
            "cost": _cost(recipe),
            "placeholders": placeholders,
        }

    def one(self, name: str) -> dict[str, Any]:
        key = self._key(name)
        recipe = self._recipe(name)
        with self._lock:
            held = self._reports.get(key)
        if held is None:
            held = self.report(name, recipe)
            with self._lock:
                self._reports[key] = held
        return held

    # --- one more, one less, one looked at ----------------------------------------------------

    def _assets(self, dwelling: str) -> Any:
        """The assets a recipe of ``dwelling`` in this folder names, which a trace can find."""
        for name in self.names():
            recipe = self._recipe(name)
            if (
                isinstance(recipe, Recipe)
                and recipe.dwelling.name == dwelling
                and recipe.assets != placeholder_assets()
            ):
                return recipe.assets
        return None

    def draw(self, body: Any) -> dict[str, Any]:
        """A recipe from ``{dwelling, seed, preset, duration_s}``, written in the folder."""
        if self.hssd_root is None:
            raise HttpError(409, "no dataset here: start with --hssd-root to generate")
        body = body if isinstance(body, dict) else {}
        dwelling, seed = body.get("dwelling"), body.get("seed", 0)
        preset, duration = body.get("preset", "medium"), body.get("duration_s", 180.0)
        if not isinstance(dwelling, str) or not _DWELLING.fullmatch(dwelling):
            raise HttpError(400, "the dwelling is missing, or is not a dwelling's name")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**53:
            raise HttpError(400, "the seed is not a whole number from 0")
        if preset not in PRESETS:
            raise HttpError(400, f"the preset is one of {', '.join(PRESETS)}")
        if isinstance(duration, bool) or not isinstance(duration, int | float):
            raise HttpError(400, "the length is not a number of seconds")
        if not DURATION_S[0] <= float(duration) <= DURATION_S[1]:
            raise HttpError(400, f"the length is from {DURATION_S[0]:g} to {DURATION_S[1]:g} s")
        name = f"{dwelling}_{preset}_seed{seed}_{float(duration):g}s.json"
        path = self.folder / name
        if path.exists():
            raise HttpError(409, f"{name} is there already: it is the same recipe")
        layout = self._layout(dwelling)
        if layout is None:
            raise HttpError(422, f"the dataset holds no dwelling named {dwelling}")
        assets = self._assets(dwelling)
        try:
            recipe = generate_social(
                layout,
                SocialParameters.preset(preset, float(duration)),
                seed,
                assets=assets,
                clips=load_clip_library(LIBRARY),
                allow_placeholder_assets=assets is None,
            )
        except (GenerationError, KeyError, OSError, ValueError) as error:
            raise HttpError(422, f"{type(error).__name__}: {error}") from error
        self.folder.mkdir(parents=True, exist_ok=True)
        save_recipe(recipe, path)
        return self.one(name)

    def discard(self, body: Any) -> dict[str, Any]:
        """Move a recipe to the folder's trash, under a name that says when; nothing is deleted."""
        name = body.get("name") if isinstance(body, dict) else None
        if not isinstance(name, str):
            raise HttpError(400, "name is missing")
        path = self._path(name)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self.folder / TRASH / f"{stamp}_{name.replace('/', '__')}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise HttpError(409, f"{target} is there already")
        path.rename(target)
        return {"name": name, "moved_to": str(target)}

    def view(self, name: str) -> dict[str, Any]:
        """Where everything is over the recipe, for the viewer: the recipe alone, and the walls."""
        recipe = self._recipe(name)
        if isinstance(recipe, RecipeError):
            raise HttpError(422, f"rule {recipe.rule}: {recipe.message}")
        layout = self._layout(recipe.dwelling.name)
        scene = recipe_scene(recipe, None if layout is None else plan_of(layout))
        scene["name"] = name
        scene["note"] = (
            "" if layout is not None else "No dataset here: the dwelling's walls are not drawn."
        )
        return scene


def _speech_over_noise(recipe: Recipe) -> dict[str, Any] | None:
    """The conversation over the noise at the listener, in free field: the least and the median."""
    if recipe.schema_version < 2:
        return None
    from reverberate.scenes.levels import conversation_snr

    ratios = np.array([row[2] for row in conversation_snr(recipe)])
    if not ratios.size:
        return None
    return {
        "least_db": round(float(ratios.min()), 1),
        "median_db": round(float(np.median(ratios)), 1),
        "turns": int(ratios.size),
    }


def _cost(recipe: Recipe) -> dict[str, Any]:
    """What the trace is predicted to take, on the machine the first scene was traced on."""
    from reverberate.scenes import cost

    try:
        found = cost.predict(recipe)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "usd": found["usd"],
        "hours": found["hours"],
        "source_positions": found["source_positions"],
        "pairs": found["pairs"],
        "note": found["note"],
    }


def build(folder: Path, *, hssd_root: Path | None = None) -> tuple[AppServer, Recipes]:
    recipes = Recipes(folder, hssd_root=hssd_root)
    server = AppServer("recipes", STATIC)

    def read(request: Request) -> Any:
        if not request.parts:
            rows = recipes.listing()
            return {
                "folder": str(recipes.folder),
                "trash": str(recipes.folder / TRASH),
                "recipes": rows,
                "can_generate": recipes.hssd_root is not None,
                "dwellings": sorted({row["dwelling"] for row in rows if not row["error"]}),
                "presets": list(PRESETS),
            }
        return recipes.one("/".join(request.parts))

    server.route("GET", "api/recipes", read)
    server.route("GET", "api/view", lambda request: recipes.view("/".join(request.parts)))
    server.route("POST", "api/generate", lambda request: recipes.draw(request.body))
    server.route("POST", "api/delete", lambda request: recipes.discard(request.body))
    return server, recipes
