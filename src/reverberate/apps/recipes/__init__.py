"""The recipes on disk: listed, validated, summarised, and one more generated. A first cut.

``python -m reverberate.apps.recipes FOLDER`` finds every scene recipe under
``FOLDER`` (``docs/formats/scene-recipe.md``) and shows, for the one chosen,
what :func:`reverberate.scenes.describe` says of it and every rule
:func:`reverberate.scenes.validate` finds broken. With the dataset at hand
(``--hssd-root``, or the one ``walk.toml`` names) it also draws a recipe
from a dwelling and a seed, with the generator's own defaults, and saves it
in ``--save-in`` when asked.

It holds no rule of its own: everything is the public functions of
:mod:`reverberate.scenes`, whose generator is being rewritten. The rules
that need the dwelling's floor are checked only for a recipe generated
here, which has its layout; a recipe read from disk is checked without.
``docs/apps.md`` lists what remains.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

from reverberate.scenes import (
    GenerationError,
    Recipe,
    RecipeError,
    describe,
    generate,
    load_hssd_layout,
    parse_recipe,
    recipe_sha256,
    save_recipe,
    validate,
)
from reverberate.viz.parts.server import AppServer, HttpError, Request

__all__ = ["STATIC", "Recipes", "build"]

STATIC = Path(__file__).parent / "static"
#: A file larger than this is not read: a recipe of twenty minutes is a third of a megabyte.
MAX_BYTES = 8 << 20
#: How deep under the folder recipes are looked for.
DEPTH = 4
MARK = b'"reverberate.scene-recipe"'


class Recipes:
    """The recipes under a folder, each read once."""

    def __init__(
        self, folder: Path, *, hssd_root: Path | None = None, save_in: Path | None = None
    ) -> None:
        self.folder = Path(folder).resolve()
        self.hssd_root = hssd_root
        self.save_in = None if save_in is None else Path(save_in)
        self._lock = threading.Lock()
        self._read: dict[str, Recipe | RecipeError] = {}
        self._reports: dict[str, dict[str, Any]] = {}
        self._drawn: Recipe | None = None

    def names(self) -> list[str]:
        """Every recipe under the folder, by its path from it."""
        found = []
        for path in sorted(self.folder.rglob("*.json")):
            relative = path.relative_to(self.folder)
            if len(relative.parts) > DEPTH or not path.is_file():
                continue
            if path.stat().st_size > MAX_BYTES or MARK not in path.read_bytes():
                continue
            found.append(str(relative))
        return found

    def _recipe(self, name: str) -> Recipe | RecipeError:
        with self._lock:
            if name not in self._read:
                path = (self.folder / name).resolve()
                if self.folder not in path.parents or not path.is_file():
                    raise HttpError(404, f"no recipe named {name!r}")
                try:
                    self._read[name] = parse_recipe(path.read_bytes())
                except RecipeError as error:
                    self._read[name] = error
            return self._read[name]

    def listing(self) -> list[dict[str, Any]]:
        return [self._row(name, self._recipe(name)) for name in self.names()]

    @staticmethod
    def _row(name: str, recipe: Recipe | RecipeError) -> dict[str, Any]:
        if isinstance(recipe, RecipeError):
            return {"name": name, "error": f"rule {recipe.rule}: {recipe.message}"}
        kinds = Counter(source.kind for source in recipe.sources)
        return {
            "name": name,
            "error": None,
            "sha256": recipe_sha256(recipe),
            "dwelling": recipe.dwelling.name,
            "seed": recipe.seed,
            "duration_s": recipe.duration_s,
            "sources": dict(kinds),
        }

    @staticmethod
    def report(name: str, recipe: Recipe | RecipeError, floor: Any = None) -> dict[str, Any]:
        """A recipe in words and every rule it breaks."""
        row = Recipes._row(name, recipe)
        if isinstance(recipe, RecipeError):
            return {
                **row,
                "valid": False,
                "violations": [{"rule": recipe.rule, "message": recipe.message}],
            }
        try:
            broken = validate(recipe, floor)
            summary = describe(recipe)
        except Exception as error:  # noqa: BLE001
            # A recipe so far from the format that the rules themselves fail on it: said, and
            # the list stays up.
            said = f"the rules could not be applied: {type(error).__name__}: {error}"
            return {**row, "valid": False, "violations": [{"rule": 0, "message": said}]}
        return {
            **row,
            "summary": summary,
            "valid": not broken,
            "violations": [{"rule": v.rule, "message": v.message} for v in broken],
            "floor_checked": floor is not None,
        }

    def one(self, name: str) -> dict[str, Any]:
        recipe = self._recipe(name)
        with self._lock:
            held = self._reports.get(name)
        if held is None:
            held = self.report(name, recipe)
            with self._lock:
                self._reports[name] = held
        return held

    def draw(self, body: Any) -> dict[str, Any]:
        """A recipe from ``{dwelling, seed}`` and the generator's defaults; kept for a save."""
        if self.hssd_root is None:
            raise HttpError(409, "no dataset here: start with --hssd-root to generate")
        dwelling, seed = (body or {}).get("dwelling"), (body or {}).get("seed", 0)
        if not isinstance(dwelling, str) or not dwelling:
            raise HttpError(400, "dwelling is missing")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**53:
            raise HttpError(400, "seed is not an integer in 0 <= seed < 2^53")
        try:
            layout = load_hssd_layout(self.hssd_root, dwelling)
            recipe = generate(
                layout, None, seed, allow_placeholder_clips=True, allow_placeholder_assets=True
            )
        except (GenerationError, KeyError, OSError, ValueError) as error:
            raise HttpError(422, f"{type(error).__name__}: {error}") from error
        self._drawn = recipe
        return {
            **self.report(f"{dwelling}_seed{seed}.json", recipe, layout.floor),
            "placeholders": "the clips and the assets are placeholders: a trace refuses it",
            "can_save": self.save_in is not None,
        }

    def save(self, body: Any) -> dict[str, Any]:
        """Write the recipe last drawn in ``save_in``, under the name it was reported with."""
        if self._drawn is None:
            raise HttpError(409, "nothing was generated")
        if self.save_in is None:
            raise HttpError(409, "start with --save-in to keep what is generated")
        recipe = self._drawn
        path = self.save_in / f"{recipe.dwelling.name}_seed{recipe.seed}.json"
        if path.exists() and not (body or {}).get("overwrite"):
            raise HttpError(409, f"{path} is there already")
        self.save_in.mkdir(parents=True, exist_ok=True)
        return {"saved": str(path), "sha256": save_recipe(recipe, path)}


def build(
    folder: Path, *, hssd_root: Path | None = None, save_in: Path | None = None
) -> tuple[AppServer, Recipes]:
    recipes = Recipes(folder, hssd_root=hssd_root, save_in=save_in)
    server = AppServer("recipes", STATIC)

    def read(request: Request) -> Any:
        if not request.parts:
            return {
                "folder": str(recipes.folder),
                "recipes": recipes.listing(),
                "can_generate": recipes.hssd_root is not None,
            }
        return recipes.one("/".join(request.parts))

    server.route("GET", "api/recipes", read)
    server.route("POST", "api/generate", lambda request: recipes.draw(request.body))
    server.route("POST", "api/save", lambda request: recipes.save(request.body))
    return server, recipes
