"""Serve the reconstructed-room web viewer for one HSSD region.

Exports both views of *our* reconstruction (``render.glb`` and
``acoustic.glb``) into a temporary directory alongside the static viewer, then
serves that directory over plain HTTP. The browser does the rendering, so this
process needs no renderer, no GPU binding and no simulator: the whole
visualisation is a standard glTF web component that can be lifted into the
Gradio demo later, or replaced, without touching the reconstruction code.

Run as
``python -m reverberate.viz.serve_room <hssd_root> <scene_id> [--region NAME]``.
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import sys
import tempfile
import webbrowser
from pathlib import Path

from reverberate.viz.scene_export import ExportReport, export_region

STATIC_DIR = Path(__file__).parent / "web"


def prepare_site(
    hssd_root: Path, scene_id: str, region_name: str | None, target: Path
) -> tuple[ExportReport, ExportReport]:
    """Write the viewer plus both exported scenes into ``target``."""
    shutil.copytree(STATIC_DIR, target, dirs_exist_ok=True)
    render_report = export_region(hssd_root, scene_id, region_name, "render", target / "render.glb")
    acoustic_report = export_region(
        hssd_root, scene_id, region_name, "acoustic", target / "acoustic.glb"
    )
    (target / "scene.json").write_text(
        json.dumps(
            {
                "title": f"{scene_id} / {render_report.region_name}",
                "hint": render_report.summary(),
            }
        )
    )
    return render_report, acoustic_report


def serve(directory: Path, port: int, open_browser: bool) -> None:
    handler = _handler_for(directory)
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"serving {url} (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def _handler_for(directory: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)  # type: ignore[arg-type]

        def log_message(self, format: str, *args: object) -> None:
            """Quiet by default; the useful output is the export report."""

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--region", default=None, help="defaults to the busiest region")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", type=Path, default=None, help="write the site and exit")
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.build_only is not None:
        render_report, acoustic_report = prepare_site(
            arguments.hssd_root, arguments.scene_id, arguments.region, arguments.build_only
        )
        print(f"render:   {render_report.summary()}")
        print(f"acoustic: {acoustic_report.summary()}")
        print(f"wrote {arguments.build_only}")
        return 0

    with tempfile.TemporaryDirectory(prefix="reverberate-viewer-") as temporary:
        target = Path(temporary)
        render_report, acoustic_report = prepare_site(
            arguments.hssd_root, arguments.scene_id, arguments.region, target
        )
        print(f"render:   {render_report.summary()}")
        print(f"acoustic: {acoustic_report.summary()}")
        serve(target, arguments.port, not arguments.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
