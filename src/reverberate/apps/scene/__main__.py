"""``python -m reverberate.apps.scene PACK``: a whole scene, everybody in sight, a fader on each."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from reverberate.apps.scene import build
from reverberate.settings import data_root
from reverberate.viz.audit_dry import DrySources
from reverberate.viz.parts.stems import AHEAD_S, MAX_WORKERS, NICE, trim
from reverberate.viz.walk_config import find_config, load_config

PORT = 8783
#: What the stems kept on disk may take before the oldest scenes are forgotten, GB.
CACHE_GB = 12.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reverberate.apps.scene", description=__doc__)
    parser.add_argument("pack", type=Path, help="a scene pack, pack.h5")
    parser.add_argument(
        "--balance", type=Path, help="where the balance is written; beside the pack unless said"
    )
    parser.add_argument(
        "--cache", type=Path, help="where stems are kept; <data root>/cache/scene_player"
    )
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=CACHE_GB,
        help="the stems kept are forgotten at start, the oldest scene first, beyond this",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS, help=f"at most {MAX_WORKERS} processes render"
    )
    parser.add_argument("--nice", type=int, default=NICE, help="added to a worker's niceness")
    parser.add_argument(
        "--ahead", type=float, default=AHEAD_S, help="seconds rendered past what is heard"
    )
    parser.add_argument("--hssd-root", type=Path, help="the dataset, for the walls; walk.toml's")
    parser.add_argument("--clips", type=Path, help="the dry clips; <data root>/clips")
    parser.add_argument("--measured-head", type=Path, help="the SOFA head order 7 is decoded with")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not args.pack.is_file():
        parser.error(f"{args.pack} is not a file")
    hssd_root = args.hssd_root
    if hssd_root is None:
        found = find_config(None)
        hssd_root = load_config(found).hssd_root if found is not None else None
    if hssd_root is None:
        hssd_root = data_root() / "raw" / "hssd-hab"
    cache = args.cache or data_root() / "cache" / "scene_player"
    freed = trim(cache, budget_gb=args.cache_gb)
    with tempfile.TemporaryDirectory(prefix="reverberate-scene-") as scratch:
        server, streamed, media = build(
            args.pack,
            Path(scratch) / "decoders",
            cache=cache,
            balance=args.balance,
            hssd_root=hssd_root,
            measured_head=args.measured_head,
            workers=args.workers,
            nice=args.nice,
            ahead_s=args.ahead,
            dry=DrySources(args.clips or data_root() / "clips", data_root() / "voices"),
        )
        # The scene's first seconds are rendered while the browser opens: pressing play is heard.
        streamed.want(0)
        told = streamed.status()
        print(f"playing {streamed.pack_path}: {', '.join(streamed.sources)}")
        print(
            f"stems in {told['cache']}, {told['workers']} worker(s) at nice +{told['nice']}, "
            f"{told['ahead_s']:g} s ahead"
            + (f"; {freed / 1e9:.1f} GB of stems kept from earlier were forgotten" if freed else "")
        )
        print(f"the balance is written in {media.balance_path} when it is saved")
        server.serve(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
