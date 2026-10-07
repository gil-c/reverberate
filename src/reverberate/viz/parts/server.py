"""A small application's server: routes, static folders, and nothing of its purpose.

An application names its own folder of static files and adds routes; the
components' files, the page's decoder and the measured head's filters are
mounted for it. A route is a function of a :class:`Request` that returns
what ``json.dumps`` takes, or a :class:`Binary`; it raises
:class:`HttpError` to refuse. :meth:`AppServer.handle` answers a request
with no socket, which is how the tests ask.

.. code-block:: python

    server = AppServer("compare", Path(__file__).parent / "static")
    server.route("GET", "api/items", lambda request: library.describe())
    server.serve(8770)

Bound to the loopback address only, one thread a request, nothing kept by
the browser between two starts: as :mod:`reverberate.viz.serve_room`.
"""

from __future__ import annotations

import http.server
import json
import mimetypes
import socketserver
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

__all__ = [
    "PARTS_STATIC",
    "AppServer",
    "Binary",
    "HttpError",
    "Request",
    "Response",
    "mount_decoders",
]

#: The components' own files, served under ``parts/``.
PARTS_STATIC = Path(__file__).parent / "static"
#: The inspector's files, served under ``reuse/`` for the decoder they hold; never written.
INSPECTOR_STATIC = Path(__file__).parents[1] / "app"
#: A body larger than this is not a request of a small application.
MAX_BODY_BYTES = 1 << 20


class HttpError(Exception):
    """A request refused: the status and what the page is told."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


@dataclass(frozen=True)
class Request:
    method: str
    #: What follows the route's own path, between slashes.
    parts: tuple[str, ...]
    query: Mapping[str, str]
    #: The body, read as JSON; ``None`` for a GET.
    body: Any = None

    def number(self, name: str, default: float | None = None) -> float:
        """A parameter of the query as a number; refused when it is not one."""
        text = self.query.get(name)
        if text is None:
            if default is None:
                raise HttpError(400, f"{name} is missing")
            return default
        try:
            return float(text)
        except ValueError as error:
            raise HttpError(400, f"{name} is not a number: {text!r}") from error

    def text(self, name: str) -> str:
        if name not in self.query:
            raise HttpError(400, f"{name} is missing")
        return self.query[name]


@dataclass(frozen=True)
class Binary:
    """An answer that is not JSON: its bytes, its type and what goes in the headers."""

    payload: bytes
    kind: str = "application/octet-stream"
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Response:
    status: int
    kind: str
    payload: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.payload)


Handler = Callable[[Request], Any]


class AppServer:
    """Routes and static folders of one application."""

    def __init__(self, name: str, static: Path, *, parts: bool = True) -> None:
        self.name = name
        self._routes: dict[tuple[str, tuple[str, ...]], Handler] = {}
        self._mounts: list[tuple[tuple[str, ...], Path]] = []
        self._closing: list[Callable[[], None]] = []
        self.mount("", static)
        if parts:
            self.mount("parts", PARTS_STATIC)
            self.mount("reuse", INSPECTOR_STATIC)

    def mount(self, prefix: str, directory: Path) -> None:
        """Serve the files of ``directory`` under ``prefix``; the longest prefix answers."""
        self._mounts.append((_split(prefix), Path(directory).resolve()))
        self._mounts.sort(key=lambda mount: -len(mount[0]))

    def route(self, method: str, path: str, handler: Handler) -> None:
        """``handler`` answers ``method`` on ``path`` and on everything under it."""
        self._routes[(method.upper(), _split(path))] = handler

    def on_close(self, action: Callable[[], None]) -> None:
        """Something to do once the server has stopped."""
        self._closing.append(action)

    def handle(self, method: str, url: str, body: bytes | None = None) -> Response:
        """The answer to one request, as the socket would send it."""
        parsed = urlsplit(url)
        parts = _split(unquote(parsed.path))
        try:
            for size in range(len(parts), -1, -1):
                handler = self._routes.get((method.upper(), parts[:size]))
                if handler is not None:
                    request = Request(
                        method.upper(),
                        parts[size:],
                        dict(parse_qsl(parsed.query, keep_blank_values=True)),
                        _json(body) if method.upper() != "GET" else None,
                    )
                    return _answer(handler(request))
            if method.upper() != "GET":
                raise HttpError(404, f"nothing takes a {method.upper()} at /{'/'.join(parts)}")
            return self._file(parts)
        except HttpError as error:
            payload = json.dumps({"error": error.message}).encode()
            return Response(error.status, "application/json", payload)

    def _file(self, parts: tuple[str, ...]) -> Response:
        for prefix, directory in self._mounts:
            if parts[: len(prefix)] != prefix:
                continue
            rest = parts[len(prefix) :] or ("index.html",)
            path = directory.joinpath(*rest).resolve()
            # A name may not climb out of the folder it is served from.
            if directory not in path.parents and path != directory:
                break
            if path.is_dir():
                path = path / "index.html"
            if path.is_file():
                kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if path.suffix in (".js", ".mjs"):
                    kind = "text/javascript"
                return Response(200, kind, path.read_bytes())
        raise HttpError(404, f"no file at /{'/'.join(parts)}")

    def serve(self, port: int, *, open_browser: bool = True, tries: int = 10) -> None:
        """Serve until interrupted, on ``port`` or on the next one that is free."""
        with _bind(self, port, tries) as server:
            url = f"http://127.0.0.1:{server.server_address[1]}/"
            print(f"{self.name}: serving {url} (ctrl-c to stop)", flush=True)
            if open_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nstopped")
            finally:
                for action in self._closing:
                    action()


def _split(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part)


def _json(body: bytes | None) -> Any:
    if not body:
        return None
    if len(body) > MAX_BODY_BYTES:
        raise HttpError(413, "the body is larger than a small application takes")
    try:
        return json.loads(body)
    except ValueError as error:
        raise HttpError(400, f"the body is not JSON: {error}") from error


def _answer(made: Any) -> Response:
    if isinstance(made, Response):
        return made
    if isinstance(made, Binary):
        return Response(200, made.kind, made.payload, made.headers)
    return Response(200, "application/json", json.dumps(made).encode())


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_for(app: AppServer) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes | None) -> None:
            response = app.handle(self.command, self.path, body)
            self.send_response(response.status)
            self.send_header("Content-Type", response.kind)
            self.send_header("Content-Length", str(len(response.payload)))
            # The page is the source tree's: a browser that keeps a module runs code that is
            # no longer on disk.
            self.send_header("Cache-Control", "no-store")
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.payload)

        def do_GET(self) -> None:  # noqa: N802
            self._send(None)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self.send_error(413, "the body is larger than a small application takes")
                return
            self._send(self.rfile.read(length))

        def log_message(self, format: str, *args: object) -> None:
            """Quiet: what an application says, it prints itself."""

    return Handler


def _bind(app: AppServer, port: int, tries: int) -> _Server:
    for candidate in range(port, port + tries):
        try:
            return _Server(("127.0.0.1", candidate), _handler_for(app))
        except OSError as error:
            if error.errno not in (48, 98):  # EADDRINUSE on macOS and Linux
                raise
            print(f"port {candidate} is in use, trying {candidate + 1}")
    raise OSError(f"no free port between {port} and {port + tries - 1}")


def mount_decoders(app: AppServer, target: Path, measured_head: Path | None) -> list[Any]:
    """Design the page's decoder into ``target`` and serve it under ``decoders/``.

    The head is the inspector's: ``measured_head``, or the one ``walk.toml``
    names, or ``<data root>/raw/hrtf/HRIR_L2702.sofa``. Without it the folder
    holds an empty list, and a player says that it plays two ears only.
    """
    from reverberate.settings import data_root
    from reverberate.viz.walk_config import find_config, load_config

    head = measured_head
    if head is None:
        found = find_config(None)
        head = load_config(found).measured_head if found is not None else None
    if head is None:
        head = data_root() / "raw" / "hrtf" / "HRIR_L2702.sofa"
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    records: list[Any] = []
    if Path(head).is_file():
        from reverberate.viz.decoders import export_decoders

        records = export_decoders(target, measured_path=Path(head))
    else:
        (target / "decoders.json").write_text("[]")
        print(f"no measured head at {head}: order 7 is not decoded, two ears are played as written")
    app.mount("decoders", target)
    return records
