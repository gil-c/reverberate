"""Command line: draw a recipe, check one, read one.

python -m reverberate.scenes generate --dwelling hssd_0076 --seed N --out recipe.json
    [--duration S] [--parameters PARAMETERS.json] [--assets ASSETS.json]
    [--clips CLIPS.json | --placeholder-clips] [--placeholder-assets] [--hssd-root DIR]
python -m reverberate.scenes validate recipe.json [--hssd-root DIR] [--no-floor]
python -m reverberate.scenes describe recipe.json

``generate`` and ``validate`` read the dwelling's geometry from the HSSD
download, ``<data root>/raw/hssd-hab`` unless ``--hssd-root`` says otherwise.
``validate --no-floor`` checks what a recipe says of itself and skips the
halves of rules 3, 4 and 6 that need the floor; it says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from reverberate.scenes.describe import describe
from reverberate.scenes.generate import (
    GenerationError,
    Parameters,
    generate,
    load_clip_library,
)
from reverberate.scenes.layout import load_hssd_floor, load_hssd_layout
from reverberate.scenes.recipe import Assets, RecipeError, load_recipe, save_recipe
from reverberate.scenes.validate import FLOOR_RULES, validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reverberate.scenes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="draw a recipe on a dwelling")
    p.add_argument("--dwelling", required=True, help="this project's name, hssd_0076")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=float, default=None, help="seconds; 1200 unless said")
    p.add_argument("--parameters", type=Path, default=None, help="a generator.parameters tree")
    p.add_argument("--assets", type=Path, default=None, help="the recipe's assets block, as JSON")
    p.add_argument("--clips", type=Path, default=None, help="a clip library, as JSON")
    p.add_argument(
        "--placeholder-clips",
        action="store_true",
        help="name clips that are not audio; no trace will accept the recipe",
    )
    p.add_argument(
        "--placeholder-assets",
        action="store_true",
        help="write asset keys that match nothing; no trace will accept the recipe",
    )
    p.add_argument("--hssd-root", type=Path, default=None)

    p = sub.add_parser("validate", help="check a recipe against the format's rules")
    p.add_argument("recipe", type=Path)
    p.add_argument("--hssd-root", type=Path, default=None)
    p.add_argument("--no-floor", action="store_true", help="skip what needs the dwelling")

    p = sub.add_parser("describe", help="a recipe in words")
    p.add_argument("recipe", type=Path)
    return parser


def _hssd_root(given: Path | None) -> Path:
    if given is not None:
        return given
    from reverberate.settings import data_root

    return data_root() / "raw" / "hssd-hab"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        print(describe(load_recipe(args.recipe)))
        return 0

    if args.command == "validate":
        try:
            recipe = load_recipe(args.recipe)
        except RecipeError as error:
            print(error)
            return 1
        floor = (
            None
            if args.no_floor
            else load_hssd_floor(_hssd_root(args.hssd_root), recipe.dwelling.scene_id)
        )
        found = validate(recipe, floor)
        for violation in found:
            print(violation)
        if floor is None:
            rules = ", ".join(str(rule) for rule in FLOOR_RULES)
            print(f"the floor was not read: rules {rules} were checked in part")
        print("valid" if not found else f"{len(found)} violation(s)")
        return 1 if found else 0

    parameters = Parameters()
    if args.parameters is not None:
        parameters = Parameters.from_record(json.loads(args.parameters.read_text()))
    if args.duration is not None:
        parameters = replace(parameters, duration_s=args.duration)
    assets = None
    if args.assets is not None:
        assets = Assets.from_dict(json.loads(args.assets.read_text()))
    clips = load_clip_library(args.clips) if args.clips is not None else None
    layout = load_hssd_layout(_hssd_root(args.hssd_root), args.dwelling)
    print(layout.summary())
    try:
        recipe = generate(
            layout,
            parameters,
            args.seed,
            assets=assets,
            clips=clips,
            allow_placeholder_clips=args.placeholder_clips,
            allow_placeholder_assets=args.placeholder_assets,
        )
    except (GenerationError, ValueError) as error:
        print(error)
        return 1
    digest = save_recipe(recipe, args.out)
    if clips is None:
        print("the clips are placeholders: the recipe names no audio")
    if assets is None:
        print("the asset keys are placeholders: the recipe matches no trace")
    print(f"{args.out}  recipe_sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
