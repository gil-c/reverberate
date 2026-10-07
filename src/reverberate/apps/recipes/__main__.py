"""``python -m reverberate.apps.recipes [FOLDER]``: the recipes of a folder, and a viewer of one."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reverberate.apps.recipes import build
from reverberate.settings import data_root
from reverberate.viz.walk_config import find_config, load_config

PORT = 8784


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reverberate.apps.recipes", description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        help="where recipes are looked for and written; <data root>/recipes unless said",
    )
    parser.add_argument("--hssd-root", type=Path, help="the dataset, for the walls and to generate")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    hssd_root = args.hssd_root
    if hssd_root is None:
        found = find_config(None)
        hssd_root = load_config(found).hssd_root if found is not None else None
    if hssd_root is None:
        hssd_root = data_root() / "raw" / "hssd-hab"
    server, recipes = build(args.folder or data_root() / "recipes", hssd_root=hssd_root)
    print(f"{len(recipes.names())} recipe(s) under {recipes.folder}")
    if recipes.hssd_root is None:
        print(f"no dataset at {hssd_root}: nothing is generated and no wall is drawn")
    server.serve(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
