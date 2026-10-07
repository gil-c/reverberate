"""``python -m reverberate.apps.compare``: open a comparison, or write one from packs first."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from reverberate.apps.compare import build

PORT = 8780


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reverberate.apps.compare", description=__doc__)
    parser.add_argument(
        "folder", type=Path, nargs="?", help="a folder written by render check --against"
    )
    parser.add_argument(
        "--packs",
        type=Path,
        nargs="+",
        metavar="PACK",
        help="write the folder first from these packs of one scene, the reference first",
    )
    parser.add_argument("--names", nargs="+", help="with --packs: their names")
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"), help="seconds")
    parser.add_argument("--sources", nargs="+", help="with --packs: the sources rendered")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["clips", "clicks", "pink"],
        help="with --packs: what the sources are fed",
    )
    parser.add_argument("--out", type=Path, help="with --packs: where the folder is written")
    parser.add_argument(
        "--results", type=Path, help="where blind tests are written; <folder>_listening unless said"
    )
    parser.add_argument("--measured-head", type=Path, help="the SOFA head order 7 is decoded with")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    folder = args.folder
    if args.packs:
        if len(args.packs) < 2:
            parser.error("--packs takes the reference and at least one other pack")
        if args.window is None:
            parser.error("--packs needs --window: order 7 is 12.3 MB a second a source sounds")
        folder = _written(args)
    if folder is None:
        parser.error("give a folder, or --packs to write one")
    with tempfile.TemporaryDirectory(prefix="reverberate-compare-") as scratch:
        server, kit, media = build(
            folder,
            Path(scratch) / "decoders",
            results=args.results,
            measured_head=args.measured_head,
        )
        print(f"comparing {', '.join(kit.variants)} in {kit.folder}")
        for note in kit.notes:
            print(f"note: {note}")
        print(f"blind tests are written in {media.results}")
        server.serve(args.port, open_browser=not args.no_browser)
    return 0


def _written(args: argparse.Namespace) -> Path:
    """The comparison of ``--packs``, written by the listening kit with its order 7 stems."""
    from reverberate.render.check import many
    from reverberate.render.check.report import defaults
    from reverberate.render.check.run import CheckSettings

    out = args.out or Path(tempfile.mkdtemp(prefix="reverberate-compare-kit-"))
    found = defaults(None, None, args.measured_head, None)
    many.run(
        args.packs[0],
        args.packs[1:],
        out,
        names=args.names,
        clips_root=found["clips_root"],
        manifest=found["manifest"],
        window_s=(args.window[0], args.window[1]),
        sources=args.sources,
        measured_head=found["measured_head"],
        settings=CheckSettings(workers=2),
        ambisonic=True,
        signals=args.signals,
        say=lambda text: print(text, flush=True),
    )
    return Path(out)


if __name__ == "__main__":
    sys.exit(main())
