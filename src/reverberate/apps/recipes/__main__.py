"""``python -m reverberate.apps.recipes``: the recipes under a folder, validated and summarised."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reverberate.apps.recipes import build
from reverberate.viz.walk_config import find_config, load_config

PORT = 8782


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reverberate.apps.recipes", description=__doc__)
    parser.add_argument("folder", type=Path, help="where recipes are looked for")
    parser.add_argument("--hssd-root", type=Path, help="the dataset, to generate; walk.toml's")
    parser.add_argument("--save-in", type=Path, help="where a generated recipe is saved")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    hssd_root = args.hssd_root
    if hssd_root is None:
        found = find_config(None)
        hssd_root = load_config(found).hssd_root if found is not None else None
    if hssd_root is not None and not Path(hssd_root).is_dir():
        hssd_root = None
    server, recipes = build(args.folder, hssd_root=hssd_root, save_in=args.save_in)
    print(f"{len(recipes.names())} recipe(s) under {recipes.folder}")
    server.serve(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
