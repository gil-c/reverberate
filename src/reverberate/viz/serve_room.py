"""Serve the walk-through app over the whole HSSD dataset.

One running process browses every apartment: the app asks for a scene, the
server assembles it on demand and caches the result. Building all 168 scenes up
front is not an option, and neither is restarting the process per scene, which
is what the previous single-region viewer forced.

The browser does the rendering, so this process needs no renderer, no GPU
binding and no simulator. Furniture assets are symlinked in place and decoded
by the browser, because their KTX2 textures do not survive a Python side merge.

It also serves the solver runs the app can open: those carrying a ``walk.json``,
see :mod:`reverberate.viz.app_payload`. A run names the scene it was simulated
in, so it is offered with that apartment rather than as a separate page.

And it answers the page's scene view under ``/api/scene/``
(:mod:`reverberate.viz.scene_api`): layouts, the generator, recipes and where
everything is over time. Saving a recipe is the one request that writes, to
``<data root>/recipes`` unless ``--recipes`` says otherwise.

And it makes a scene heard, under ``/api/audit/``
(:mod:`reverberate.viz.audit_api`): a scene pack found beside its recipe, or
named with ``--pack``, is rendered by the signal engine in worker processes,
its stems kept under ``<data root>/cache/audit_stems`` unless
``--audit-cache`` says otherwise, and streamed to the page at order 7.

And it shows a pack's geometry as the computation used it, under
``/api/computed/`` (:mod:`reverberate.viz.computed_api`): the wave solver's
grid, the arrays and the sources on their nodes, the mirror's facets, a
sample of the rays. ``--grid`` and ``--mirror`` offer a grid or a mirror's
scene that is not beside its pack; what is derived is kept under
``<data root>/cache/computed`` unless ``--computed-cache`` says otherwise.

Run it: ``python src/reverberate/viz/serve_room.py``, or the run button on
this file. Everything it needs is in ``walk.toml``
(:mod:`reverberate.viz.walk_config`); any command line argument overrides the
file for one run.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a file, from PyCharm's run button or `python serve_room.py`:
    # put `src` on the path so the package imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import http.server
import json
import shutil
import socketserver
import tempfile
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from reverberate.geometry.scene_ids import scene_of
from reverberate.settings import data_root
from reverberate.store import ObjectStore, shared_store
from reverberate.viz.app_payload import WalkRun, build_run, discover_walk_runs
from reverberate.viz.assemble_dataset import every_storey
from reverberate.viz.audit_api import PREFIX as AUDIT_PREFIX
from reverberate.viz.audit_api import AuditService, Binary
from reverberate.viz.audit_dry import DrySources
from reverberate.viz.audit_stems import StemService
from reverberate.viz.computed_api import PREFIX as COMPUTED_PREFIX
from reverberate.viz.computed_api import ComputedService
from reverberate.viz.decoders import export_decoders
from reverberate.viz.scene_api import MAX_BODY_BYTES, PREFIX, SceneError, SceneService
from reverberate.viz.scene_cache import (
    SceneEntry,
    cache_root,
    ensure_scene,
    published_key,
    scene_key,
)
from reverberate.viz.voices import voices_dir
from reverberate.viz.walk_config import CONFIG_NAME, WalkConfig, find_config, load_config

STATIC_DIR = Path(__file__).parent / "app"


def list_apartments(hssd_root: Path, store: ObjectStore | None = None) -> list[dict[str, object]]:
    """Every storey, by the project's name, with whether it opens without assembling.

    ``local`` is the short name the page asks the server for; ``scene_id`` is
    what runs are keyed by. ``ready`` means an entry exists in the published
    catalogue or on this disk; assembling takes minutes, so the page offers
    only what is ready.
    """
    rows = every_storey()
    if not (hssd_root / "semantics" / "scenes").is_dir():
        # No dataset here: only what the catalogue holds can be opened at all.
        rows = [
            r
            for r in rows
            if store is not None and published_key(store, r.scene_id, r.storey_index)
        ]
    return [
        {
            "local": row.local,
            "scene_id": row.scene_id,
            "ready": _ready(hssd_root, store, row.scene_id, row.storey_index),
        }
        for row in rows
    ]


def _ready(hssd_root: Path, store: ObjectStore | None, scene_id: str, storey: int) -> bool:
    if store is not None and published_key(store, scene_id, storey):
        return True
    try:
        return (
            cache_root() / scene_key(hssd_root, scene_id, storey=storey) / "entry.json"
        ).is_file()
    except (OSError, ValueError):
        return False


def _resolve(name: str) -> tuple[str, int]:
    """The HSSD scene id and storey index behind either name; an id means the ground floor."""
    try:
        scene_id, storey = scene_of(name)
    except KeyError:
        return name, 0
    return scene_id, (storey - 1) if storey else 0


def attach_runs(
    apartments: Sequence[Mapping[str, object]], runs: Sequence[WalkRun], first: str | None = None
) -> list[dict[str, object]]:
    """Tell each apartment which solver runs exist for it, and order the list.

    The selector needs this before any apartment is opened, so that a run is
    discoverable rather than something you have to already know about. The
    page opens the first apartment: ``first`` (a short name or a scene id),
    then those with a run, then the rest.
    """
    by_scene: dict[str, list[str]] = {}
    for run in runs:
        by_scene.setdefault(run.scene_id, []).append(run.name)
    attached = [
        {**apartment, "runs": by_scene.get(str(apartment["scene_id"]), [])}
        for apartment in apartments
    ]
    attached.sort(key=lambda a: (first not in (a["local"], a["scene_id"]), not a["runs"]))
    return attached


class SiteBuilder:
    """Assembles apartments into the served directory, once each."""

    def __init__(
        self,
        hssd_root: Path,
        target: Path,
        first: str | None = None,
        runs_root: Path | None = None,
        rebuild: bool = False,
        lead: str | None = None,
        voices: Path | None = None,
        measured_head: Path | None = None,
        recipes: Path | None = None,
        packs: Sequence[Path] = (),
        audit_cache: Path | None = None,
        audit_workers: int | None = None,
        clips: Path | None = None,
        grids: Sequence[Path] = (),
        mirrors: Sequence[Path] = (),
        computed_cache: Path | None = None,
    ) -> None:
        self.hssd_root = hssd_root
        self.target = target
        # Nothing is created here: the folder appears with the first recipe saved.
        self.scenes = SceneService(hssd_root, recipes or data_root() / "recipes")
        print(f"recipes: {self.scenes.recipes_root}")
        # Sound for a scene: the engine's stems, rendered when a pack is opened.
        # Nothing is created or started here either.
        stems = StemService(
            audit_cache or data_root() / "cache" / "audit_stems",
            workers=audit_workers,
            dry=DrySources(
                clips if clips is not None else data_root() / "clips",
                voices if voices is not None else voices_dir(),
            ),
        )
        self.audit = AuditService(
            stems,
            folders=[self.scenes.recipes_root, *(p for p in packs if p.is_dir())],
            packs=[p for p in packs if p.is_file()],
        )
        print(f"packs: {len(self.audit.packs())} found; stems in {stems.cache_root}")
        # What a pack was computed on: nothing is read until the page asks.
        self.computed = ComputedService(
            self.audit,
            computed_cache or data_root() / "cache" / "computed",
            grids=grids,
            mirrors=mirrors,
            vox_cache=data_root() / "cache" / "vox",
        )
        self.rebuild = rebuild
        self.lead = lead
        self._lock = threading.Lock()
        self._built: dict[str, SceneEntry] = {}
        # The store is where the whole dataset was assembled to, so an
        # apartment opened here is fetched rather than rebuilt. A machine
        # without credentials gets None and can open only what it has.
        self.store = shared_store()
        print("store: " + ("reachable" if self.store else "not reachable, local entries only"))
        shutil.copytree(STATIC_DIR, target, dirs_exist_ok=True)

        # The page decodes with the library's own filters, designed here at
        # start: a second of work, and the one place the head is chosen.
        if measured_head is None:
            raise SystemExit("set measured_head in walk.toml: the page listens with that head only")
        heads = export_decoders(target / "decoders", measured_path=measured_head)
        print("decoders: " + ", ".join(f"{h['name']} (order {h['order']})" for h in heads))
        # Voices are fetched by `python -m reverberate.viz.voices`, once; the
        # site links the directory and says when it is empty.
        voices = voices if voices is not None else voices_dir()
        if (voices / "voices.json").is_file():
            link = target / "voices"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(voices.resolve(), target_is_directory=True)
            count = len(json.loads((voices / "voices.json").read_text()))
            print(f"voices: {count} in {voices}")
        else:
            (target / "voices").mkdir(exist_ok=True)
            (target / "voices" / "voices.json").write_text("[]")
            print(f"voices: none; run python -m reverberate.viz.voices (looked in {voices})")

        self.runs = discover_walk_runs(runs_root) if runs_root is not None else []
        # Runs are built up front, unlike apartments: there are a handful of
        # them and the payload is a link and a small file, so paying for it
        # here keeps the failure visible at startup.
        built = []
        for run in self.runs:
            try:
                record = build_run(run, target / "runs" / run.name)
            except (OSError, ValueError) as error:
                # One broken run must not take the whole app down at start.
                print(f"{run.name}: not offered, {error}")
                continue
            meshes = ", ".join(f"{k} Hz" for k in record["meshes"]) or "no mesh"
            print(f"{run.name}: {len(run.sources)} sources, {meshes} (scene {run.scene_id})")
            built.append(run)
        self.runs = built
        (target / "runs.json").write_text(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "scene_id": r.scene_id,
                        "url": f"runs/{r.name}",
                        # Which run the app opens on. Whoever started the
                        # server had a run in mind and it is rarely the one
                        # sorting last.
                        "lead": r.name == self.lead,
                    }
                    for r in self.runs
                ]
            )
        )

        lead_run = next((r for r in self.runs if r.name == self.lead), None)
        first = first or (lead_run.scene_id if lead_run else None)
        apartments = attach_runs(list_apartments(hssd_root, self.store), self.runs, first)
        (target / "apartments.json").write_text(json.dumps(apartments))

    def ensure(self, name: str) -> SceneEntry:
        """Put one apartment under ``scenes/<name>/``, by either of its names."""
        # Serialised deliberately: two browser requests for the same scene must
        # not both run the assembly, which is the slow part.
        scene_id, storey = _resolve(name)
        with self._lock:
            if name not in self._built:
                cached = ensure_scene(
                    self.hssd_root, scene_id, storey=storey, force=self.rebuild, store=self.store
                )
                # The site is a temporary directory and the entry is not: the
                # scene is linked into place rather than copied, because its
                # asset symlinks are absolute and a copy would duplicate the
                # exported meshes for the lifetime of the process.
                # Linked under both names for the ground floor: the page asks
                # by short name, a run's scene id still resolves.
                for alias in {name} | ({scene_id} if storey == 0 else set()):
                    link = self.target / "scenes" / alias
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink():
                        link.unlink()
                    elif link.exists():
                        shutil.rmtree(link)
                    link.symlink_to(cached.path, target_is_directory=True)
                print(f"{scene_id}: {cached.summary()}")
                print(f"{scene_id}: {cached.storey()}")
                print(f"{scene_id}: cache entry {cached.key}")
                self._built[name] = cached
            return self._built[name]


def _handler_for(builder: SiteBuilder) -> type[http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(builder.target), **kwargs)  # type: ignore[arg-type]

        def do_GET(self) -> None:  # noqa: N802
            if self._api("GET"):
                return
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "scenes" and parts[2] == "manifest.json":
                try:
                    builder.ensure(parts[1])
                except Exception as error:  # noqa: BLE001
                    self.send_error(500, f"could not assemble {parts[1]}: {error}")
                    return
            wanted = self.headers.get("Range")
            if wanted and self._send_range(wanted):
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if not self._api("POST"):
                self.send_error(404, "nothing takes a POST here")

        def _api(self, method: str) -> bool:
            """Answer a request of the scene view or of its sound; false when it is neither.

            In JSON, but for the audio itself, which is the engine's bytes.
            """
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            scene, audit = parts[:2] == PREFIX.split("/"), parts[:2] == AUDIT_PREFIX.split("/")
            computed = parts[:2] == COMPUTED_PREFIX.split("/")
            if not scene and not audit and not computed:
                return False
            scenes: SceneService | None = getattr(builder, "scenes", None)
            sound: AuditService | None = getattr(builder, "audit", None)
            used: ComputedService | None = getattr(builder, "computed", None)
            status, answer = 200, None
            try:
                if (scenes if scene else used if computed else sound) is None:
                    raise SceneError(404, "this server has no scene view")
                body = None
                if method == "POST":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_BODY_BYTES:
                        raise SceneError(413, "the body is larger than a recipe")
                    try:
                        body = json.loads(self.rfile.read(length))
                    except ValueError as error:
                        raise SceneError(400, f"the body is not JSON: {error}") from error
                if scene:
                    assert scenes is not None
                    answer = scenes.handle(method, parts[2:], body)
                elif computed:
                    assert used is not None
                    answer = used.handle(method, parts[2:], dict(parse_qsl(url.query)))
                else:
                    assert sound is not None
                    answer = sound.handle(method, parts[2:], dict(parse_qsl(url.query)), body)
            except SceneError as error:
                status, answer = error.status, {"error": error.message}
            extra: Mapping[str, str] = {}
            if isinstance(answer, Binary):
                payload, kind, extra = answer.payload, "application/octet-stream", answer.headers
            else:
                payload, kind = json.dumps(answer).encode(), "application/json"
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            for name, value in extra.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)
            return True

        def _send_range(self, wanted: str) -> bool:
            """Serve ``bytes=a-b`` of a file, which is how the page reads a field.

            A response of gigabytes is read one cell at a time straight out of
            the HDF5, so the server has to answer ranges; the standard handler
            does not. Anything it cannot satisfy falls through to a whole
            file, which the page treats as an error rather than as a cell.
            """
            path = Path(self.translate_path(self.path))
            if not path.is_file() or not wanted.startswith("bytes="):
                return False
            first, _, last = wanted[len("bytes=") :].partition("-")
            size = path.stat().st_size
            try:
                start = int(first)
                end = int(last) if last else size - 1
            except ValueError:
                return False
            if start < 0 or start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return True
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = handle.read(min(remaining, 1 << 20))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            return True

        def end_headers(self) -> None:
            # The page and its modules are rebuilt from the source tree on every
            # start. Without this a browser keeps the previous module and runs
            # code that is no longer anywhere on disk, which looks exactly like
            # an edit having had no effect.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, format: str, *args: object) -> None:
            """Quiet by default; the useful output is the assembly report."""

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    """Threaded, because assembling an apartment takes seconds.

    On a single threaded server that assembly blocks every other request, so
    switching apartment freezes the whole page: the audio of the solver mode
    stops mid playback and the run payload cannot be fetched. Assembly itself is
    still serialised by the builder's own lock, so threading adds concurrency
    where it helps and none where it would duplicate work.
    """

    # Without this, restarting the viewer on the same port fails for a minute
    # while the previous socket sits in TIME_WAIT.
    allow_reuse_address = True
    daemon_threads = True


def _bind(builder: SiteBuilder, port: int, tries: int = 10) -> _Server:
    """The server on ``port``, or on the next free one when that is taken.

    Several sessions run a viewer on this machine at once, and a port taken by
    one of them is a start-up failure and not a reason to stop: the port in
    use is printed and the browser is opened on it.
    """
    for candidate in range(port, port + tries):
        try:
            return _Server(("127.0.0.1", candidate), _handler_for(builder))
        except OSError as error:
            if error.errno not in (48, 98):  # EADDRINUSE on macOS and Linux
                raise
            print(f"port {candidate} is in use, trying {candidate + 1}")
    raise OSError(f"no free port between {port} and {port + tries - 1}")


def serve(builder: SiteBuilder, port: int, open_browser: bool) -> None:
    with _bind(builder, port) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"serving {url} (ctrl-c to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            audit: AuditService | None = getattr(builder, "audit", None)
            if audit is not None:
                audit.stems.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help=f"a {CONFIG_NAME} to read")
    parser.add_argument("hssd_root", type=Path, nargs="?", default=None)
    parser.add_argument("--scene", default=None, help="open on this apartment")
    parser.add_argument(
        "--runs", type=Path, default=None, help="directory of runs with a walk.json"
    )
    parser.add_argument("--run", default=None, help="open on this run")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--voices", type=Path, default=None, help="default: <data root>/voices")
    parser.add_argument("--measured-head", type=Path, default=None, help="a SOFA head to add")
    parser.add_argument(
        "--recipes", type=Path, default=None, help="where recipes are saved; <data root>/recipes"
    )
    parser.add_argument(
        "--pack",
        type=Path,
        action="append",
        default=[],
        help="a scene pack to offer, or a folder of them; may be repeated",
    )
    parser.add_argument(
        "--audit-cache", type=Path, default=None, help="default: <data root>/cache/audit_stems"
    )
    parser.add_argument(
        "--audit-workers", type=int, default=None, help="processes that render stems"
    )
    parser.add_argument("--clips", type=Path, default=None, help="default: <data root>/clips")
    parser.add_argument(
        "--grid",
        type=Path,
        action="append",
        default=[],
        help="a voxel cache entry or an as_computed.npz to offer beside the packs' own grids",
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        action="append",
        default=[],
        help="a mirror's scene directory, for a pack whose bundle is not beside it",
    )
    parser.add_argument(
        "--computed-cache", type=Path, default=None, help="default: <data root>/cache/computed"
    )
    parser.add_argument("--build-only", type=Path, default=None, help="write the site and exit")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="reassemble the scene even if cached"
    )
    arguments = parser.parse_args(argv)

    found = find_config(arguments.config)
    if found is None and arguments.hssd_root is None:
        print(
            f"no {CONFIG_NAME} found (looked in the working directory and the main checkout) "
            "and no hssd_root given",
            file=sys.stderr,
        )
        return 2
    config = (
        load_config(found)
        if found is not None
        else WalkConfig(hssd_root=arguments.hssd_root, open_browser=not arguments.no_browser)
    )
    config.apply_environment()
    hssd_root = arguments.hssd_root or config.hssd_root
    scene = arguments.scene or config.scene
    runs = arguments.runs or config.runs
    run = arguments.run or config.run
    port = arguments.port or config.port
    voices = arguments.voices or config.voices
    head = arguments.measured_head or config.measured_head
    rebuild = arguments.rebuild
    open_browser = config.open_browser and not arguments.no_browser
    if found is not None:
        print(f"config: {found}")

    def build(target: Path) -> SiteBuilder:
        builder = SiteBuilder(
            hssd_root,
            target,
            scene,
            runs,
            rebuild,
            run,
            voices,
            head,
            arguments.recipes,
            packs=arguments.pack,
            audit_cache=arguments.audit_cache,
            audit_workers=arguments.audit_workers,
            clips=arguments.clips,
            grids=arguments.grid,
            mirrors=arguments.mirror,
            computed_cache=arguments.computed_cache,
        )
        if scene:
            builder.ensure(scene)
        return builder

    if arguments.build_only is not None:
        build(arguments.build_only)
        print(f"wrote {arguments.build_only}")
        return 0
    with tempfile.TemporaryDirectory(prefix="reverberate-viewer-") as temporary:
        serve(build(Path(temporary)), port, open_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
