"""Serve the apartment viewer over the whole HSSD dataset.

One running process browses every apartment: the viewer asks for a scene, the
server assembles it on demand and caches the result. Building all 168 scenes up
front is not an option, and neither is restarting the process per scene, which
is what the previous single-region viewer forced.

The browser does the rendering, so this process needs no renderer, no GPU
binding and no simulator: the whole visualisation is a standard glTF web
component that can be lifted into the Gradio demo later, or replaced, without
touching the reconstruction code. Furniture assets are symlinked in place and
decoded by the browser, because their KTX2 textures do not survive a Python
side merge.

Run as ``python -m reverberate.viz.serve_room <hssd_root> [--scene ID]``.
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from reverberate.viz.scene_manifest import ManifestReport, write_manifest

STATIC_DIR = Path(__file__).parent / "web"


def list_apartments(hssd_root: Path, first: str | None = None) -> list[dict[str, str]]:
    """Every scene in the dataset, as an apartment the selector can offer.

    ``first`` is put at the head of the list so that the apartment already
    assembled is the one the viewer opens with, rather than paying for a second
    assembly on load.
    """
    semantics = hssd_root / "semantics" / "scenes"
    apartments = [
        {"id": scene_id, "label": scene_id}
        for scene_id in sorted(
            path.name.split(".")[0] for path in (hssd_root / "scenes").glob("*.scene_instance.json")
        )
        # A scene without region annotations has no rooms to assemble.
        if (semantics / f"{scene_id}.semantic_config.json").is_file()
    ]
    if first is not None:
        apartments.sort(key=lambda apartment: apartment["id"] != first)
    return apartments


class SiteBuilder:
    """Assembles apartments into the served directory, once each."""

    def __init__(self, hssd_root: Path, target: Path, first: str | None = None) -> None:
        self.hssd_root = hssd_root
        self.target = target
        self._lock = threading.Lock()
        self._built: dict[str, ManifestReport] = {}
        shutil.copytree(STATIC_DIR, target, dirs_exist_ok=True)
        (target / "apartments.json").write_text(json.dumps(list_apartments(hssd_root, first)))

    def ensure(self, scene_id: str) -> ManifestReport:
        # Serialised deliberately: two browser requests for the same scene must
        # not both run the assembly, which is the slow part.
        with self._lock:
            if scene_id not in self._built:
                report = write_manifest(self.hssd_root, scene_id, self.target / "scenes" / scene_id)
                print(f"{scene_id}: {report.summary()}")
                print(f"{scene_id}: {report.storey}")
                self._built[scene_id] = report
            return self._built[scene_id]


def _handler_for(builder: SiteBuilder) -> type[http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(builder.target), **kwargs)  # type: ignore[arg-type]

        def do_GET(self) -> None:  # noqa: N802
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "scenes" and parts[2] == "manifest.json":
                try:
                    builder.ensure(parts[1])
                except Exception as error:  # noqa: BLE001
                    self.send_error(500, f"could not assemble {parts[1]}: {error}")
                    return
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            """Quiet by default; the useful output is the assembly report."""

    return Handler


def serve(builder: SiteBuilder, port: int, open_browser: bool) -> None:
    # Without this, restarting the viewer on the same port fails for a minute
    # while the previous socket sits in TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), _handler_for(builder)) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"serving {url} (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hssd_root", type=Path)
    parser.add_argument("--scene", default=None, help="assemble this apartment before serving")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", type=Path, default=None, help="write the site and exit")
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.build_only is not None:
        builder = SiteBuilder(arguments.hssd_root, arguments.build_only, arguments.scene)
        if arguments.scene:
            builder.ensure(arguments.scene)
        print(f"wrote {arguments.build_only}")
        return 0

    with tempfile.TemporaryDirectory(prefix="reverberate-viewer-") as temporary:
        builder = SiteBuilder(arguments.hssd_root, Path(temporary), arguments.scene)
        if arguments.scene:
            builder.ensure(arguments.scene)
        serve(builder, arguments.port, not arguments.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
