"""``python -m reverberate.apps.scene_player``: a scene, its sources in sight, a fader on each."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from reverberate.apps.scene_player import build

PORT = 8781


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reverberate.apps.scene_player", description=__doc__
    )
    parser.add_argument("folder", type=Path, help="a folder of the listening kit, with order 7")
    parser.add_argument("--pack", type=Path, help="the scene's pack, when the folder's is gone")
    parser.add_argument(
        "--results", type=Path, help="where the balance is written; <folder>_listening unless said"
    )
    parser.add_argument("--measured-head", type=Path, help="the SOFA head order 7 is decoded with")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="reverberate-scene-player-") as scratch:
        server, kit, media = build(
            args.folder,
            Path(scratch) / "decoders",
            pack=args.pack,
            results=args.results,
            measured_head=args.measured_head,
        )
        print(f"playing {kit.folder}: {', '.join(media.sources)}")
        print(f"the balance is written in {media.balance_path}")
        server.serve(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
