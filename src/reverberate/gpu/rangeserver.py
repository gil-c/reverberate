"""A run's output folder over HTTPS, in ranges, to whoever holds the run's token.

This file runs **on the rented machine**, by the system's ``python3`` and
nothing else: no package of this repository, none of ``pip``'s. The laptop
sends it down the ssh connection whose host keys it pinned
(:func:`reverberate.gpu.transfer.serve`) and starts it there. It exists
because a pack of 9.5 GB comes home on four ssh streams, each bounded by
one window of ``sshd``'s, and the host's ``sshd`` refuses many more; ranges
over TLS have neither bound.

What it is allowed, all of it:

- **one folder**, the run's output. A path is resolved, links followed,
  and served only where the result is a regular file under that folder;
  anything else is 404, a folder included: there is no listing;
- **reading**: ``GET`` and ``HEAD``, whole or one range. No other method;
- **to the bearer of the run's token**, compared in constant time, before
  the path is looked at: without it every request is 401 and says nothing;
- **over TLS 1.2 or later**, with a key made on the machine for this run.
  The laptop reads the certificate back over the pinned ssh connection and
  believes that one alone;
- **for a time**: it ends itself at ``--lifetime``, and where ``--forget``
  is given it removes the key's and the token's files once they are read,
  so that neither stays on the disk.

``--limit-stream``, ``--limit-total``, ``--drop-after`` and ``--drops`` are
for the tests and the bench on 127.0.0.1: a line that carries so much a
stream or in all, and so many connections cut after so many bytes of a
body.
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import os
import socketserver
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

#: What is read from the file and written to the connection at once.
BLOCK = 1 << 20
#: A connection that says nothing for this long is closed, s.
IDLE_S = 120.0
#: The path that answers 204 to the token's bearer and serves nothing: is the way open.
PING = "/.ping"


class Budget:
    """So many bytes a second, shared by whoever asks: the line of the tests."""

    def __init__(self, bytes_per_s: float) -> None:
        self.rate = float(bytes_per_s)
        self._lock = threading.Lock()
        self._free_at = time.monotonic()

    def take(self, count: int) -> None:
        if self.rate <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(self._free_at, now)
            self._free_at = start + count / self.rate
            wait = self._free_at - now
        if wait > 0.0:
            time.sleep(wait)


def resolve(root: str, path: str) -> str | None:
    """The regular file ``path`` names under ``root``, or ``None``.

    ``root`` is already resolved. The path is the URL's, decoded once; it
    is joined, resolved with its links, and kept only where what it
    resolves to is still under ``root`` and is a regular file.
    """
    decoded = unquote(urlsplit(path).path)
    if "\x00" in decoded or not decoded.startswith("/"):
        return None
    found = os.path.realpath(os.path.join(root, decoded.lstrip("/")))
    if found != root and not found.startswith(root.rstrip(os.sep) + os.sep):
        return None
    return found if os.path.isfile(found) else None


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """``bytes=a-b`` as the first byte and the count, within ``size``; ``None`` when refused.

    One range, with its first byte given. No header is the whole file.
    """
    if header is None:
        return 0, size
    unit, _, spec = header.partition("=")
    if unit.strip() != "bytes" or "," in spec:
        return None
    first, _, last = spec.strip().partition("-")
    if not first.isdigit() or (last and not last.isdigit()):
        return None
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        return None
    return start, end - start + 1


class Handler(BaseHTTPRequestHandler):
    """One connection: the token, then a file's bytes."""

    protocol_version = "HTTP/1.1"
    server_version = "rv"
    sys_version = ""
    timeout = IDLE_S

    def _plain(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        if status == 401:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _authorised(self) -> bool:
        given = self.headers.get("Authorization", "")
        wanted = "Bearer " + self.server.token  # type: ignore[attr-defined]
        return hmac.compare_digest(given.encode("utf-8", "replace"), wanted.encode())

    def _serve(self, body: bool) -> None:
        server: Any = self.server
        if not self._authorised():
            self._plain(401)
            return
        if urlsplit(self.path).path == PING:
            self._plain(204)
            return
        found = resolve(server.root, self.path)
        if found is None:
            self._plain(404)
            return
        try:
            handle = open(found, "rb")  # noqa: SIM115 - closed below, after its bytes
        except OSError:
            self._plain(404)
            return
        with handle:
            size = os.fstat(handle.fileno()).st_size
            asked = self.headers.get("Range")
            span = byte_range(asked, size) if size else (0, 0)
            if span is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, count = span
            self.send_response(206 if asked is not None and size else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(count))
            if asked is not None and size:
                self.send_header("Content-Range", f"bytes {start}-{start + count - 1}/{size}")
            self.end_headers()
            if not body:
                return
            handle.seek(start)
            own = Budget(server.limit_stream)
            sent = 0
            while sent < count:
                block = handle.read(min(BLOCK if not server.throttled else 1 << 16, count - sent))
                if not block:
                    # The file is shorter than it was: the body cannot be what was promised.
                    self.close_connection = True
                    return
                own.take(len(block))
                server.total.take(len(block))
                if sent + len(block) > server.drop_after > 0 and server.cut():
                    self.wfile.write(block[: max(0, server.drop_after - sent)])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.close()
                    return
                self.wfile.write(block)
                sent += len(block)

    def do_GET(self) -> None:  # noqa: N802 - the name ``http.server`` calls
        self._serve(True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(False)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the parent's name
        # The request's line and its status; never a header, so never the token.
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        sys.stderr.write(
            "{} {} {}\n".format(time.strftime("%H:%M:%S"), self.client_address[0], format % args)
        )


class Server(ThreadingHTTPServer):
    """The listener: TLS on every connection, a thread a connection."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        *,
        root: str,
        token: str,
        certificate: str,
        key: str,
        limit_stream: float = 0.0,
        limit_total: float = 0.0,
        drop_after: int = 0,
        drops: int = 1,
        quiet: bool = False,
    ) -> None:
        if len(token) < 32:
            raise ValueError("a token of 32 characters or more")
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise ValueError("the folder to serve does not exist: " + root)
        self.token = token
        self.limit_stream = float(limit_stream)
        self.total = Budget(limit_total)
        self.throttled = limit_stream > 0.0 or limit_total > 0.0
        self.drop_after = int(drop_after)
        self.drops = int(drops)
        self._cuts = threading.Lock()
        self.quiet = quiet
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certificate, key)
        # An old TLS library offers no curve for the key exchange unless told one.
        with contextlib.suppress(AttributeError, ValueError, ssl.SSLError):
            context.set_ecdh_curve("prime256v1")
        self.context = context
        super().__init__(address, Handler)

    def server_bind(self) -> None:
        # The parent's asks the resolver for this host's name, which nothing here reads
        # and which took five seconds on a host whose address has no name.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

    def cut(self) -> bool:
        """Whether one more body is to be cut: ``drops`` of them are, then none."""
        with self._cuts:
            if self.drops <= 0:
                return False
            self.drops -= 1
            return True

    def get_request(self) -> Any:
        # Accepted here and shaken hands with in the connection's own thread
        # (``finish_request``): a client that opens and says nothing holds no one up.
        return self.socket.accept()

    def finish_request(self, request: Any, client_address: Any) -> None:
        request.settimeout(IDLE_S)
        try:
            secured = self.context.wrap_socket(request, server_side=True)
        except (OSError, ssl.SSLError):
            return
        try:
            super().finish_request(secured, client_address)
        except (OSError, ssl.SSLError):
            # A client that left in the middle of a body: its own affair.
            pass
        finally:
            with contextlib.suppress(OSError):
                secured.close()

    def handle_error(self, request: Any, client_address: Any) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="the one folder that is served")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bind", default="0.0.0.0")  # noqa: S104 - the host maps this port
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--pid-file", default="")
    parser.add_argument("--lifetime", type=float, default=6 * 3600.0, help="ends itself, s")
    parser.add_argument("--forget", action="store_true", help="remove the key and the token")
    parser.add_argument("--limit-stream", type=float, default=0.0, help="bytes/s a connection")
    parser.add_argument("--limit-total", type=float, default=0.0, help="bytes/s in all")
    parser.add_argument("--drop-after", type=int, default=0, help="cut a body after these bytes")
    parser.add_argument("--drops", type=int, default=1, help="bodies that are cut so")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    with open(args.token_file) as handle:
        token = handle.read().strip()
    server = Server(
        (args.bind, args.port),
        root=args.root,
        token=token,
        certificate=args.cert,
        key=args.key,
        limit_stream=args.limit_stream,
        limit_total=args.limit_total,
        drop_after=args.drop_after,
        drops=args.drops,
        quiet=args.quiet,
    )
    if args.forget:
        for path in (args.key, args.token_file):
            with contextlib.suppress(OSError):
                os.unlink(path)
    if args.pid_file:
        with open(args.pid_file, "w") as handle:
            handle.write(str(os.getpid()))
    timer = threading.Timer(max(1.0, args.lifetime), server.shutdown)
    timer.daemon = True
    timer.start()
    sys.stderr.write(f"serving {server.root} on {args.bind}:{server.server_address[1]}\n")
    sys.stderr.flush()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if args.pid_file:
            with contextlib.suppress(OSError):
                os.unlink(args.pid_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
