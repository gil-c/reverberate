"""The fastest way a run's files come home from a machine, among those that work on it.

A pack of 9.5 GB came home on four ssh streams
(:mod:`reverberate.gpu.homecoming`), at 14 to 57 MB/s where the instance's
own address answered. What bounds that is in
``docs/open-questions/direct-connection.md``: a stream of ssh is one TCP
connection and one window of ``sshd``'s, and the host's ``sshd`` refuses
connections past a few. This module is what goes past both, and what
chooses:

- :func:`serve` starts :mod:`reverberate.gpu.rangeserver` on the machine,
  on the port the rental asked the host to map (:data:`HOMECOMING_PORT`):
  the run's output folder alone, over TLS, to the bearer of a token made
  here for this run. It is started over the **pinned** ssh connection and
  no other, and the certificate the machine made comes back on that
  connection: :class:`Served` believes that certificate alone;
- :class:`HttpsTransport` reads a file's ranges there, a connection a
  worker, and asks everything else (sizes, digests, listings) over ssh as
  it was asked. A file fetched so ends as every other: its SHA-256 is the
  one the machine computes, said over ssh;
- :class:`Lanes` is how many workers a transfer uses: it starts with a
  few and adds more while each addition still brings the bytes faster,
  and takes them back when a connection is refused;
- :func:`ways` is the order they are tried in: HTTPS ranges, ssh at the
  machine's own address, ssh through the proxy. What
  ``python -m reverberate.gpu.transfer_bench`` measured on the host
  (:func:`verdict_of`) changes the order and the counts;
- :class:`Early` looks at the machine while its campaign still runs: the
  pack is brought as soon as it is written, and the campaign's end is
  seen in seconds and not at the next look of five minutes.

**Nothing is put on the machine that it must keep secret from us, and
nothing of ours that it could use elsewhere**: the token opens the run's
own output and dies with the server; the key is the machine's own, made
there. No storage credential, no agent, no key of the laptop.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import http.client
import json
import posixpath
import secrets
import shlex
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from reverberate.gpu import rangeserver
from reverberate.gpu.homecoming import (
    STALLED_BYTES_PER_S,
    WORKERS,
    SshTransport,
    Transport,
    fastest,
)
from reverberate.wave.remote import ConnectionLost, RemoteError, connection_level

__all__ = [
    "HOMECOMING_PORT",
    "HTTPS",
    "SSH_DIRECT",
    "SSH_PROXY",
    "Early",
    "HttpsTransport",
    "Lanes",
    "Served",
    "Way",
    "minutes_home",
    "serve",
    "stop",
    "verdict_of",
    "verdict_path",
    "ways",
]

#: The port of the instance the range server listens on, which a rental asks the host to
#: map (:func:`reverberate.gpu.vast.rent`). The host's own port for it is another, said by
#: the API's record and by the instance's environment.
HOMECOMING_PORT = 8443
#: Where the server's own files are on the machine: beside the run, never inside what it serves.
REMOTE_SERVE = "/root/.rv-serve"
#: The server ends itself after this long whatever happens to the laptop, s.
SERVE_LIFETIME_S = 12 * 3600.0
#: A connection of a range that brings nothing for this long is lost, s.
READ_TIMEOUT_S = 60.0

#: The names of the ways, as the bench's table and a fetch's record say them.
HTTPS = "https"
SSH_DIRECT = "direct ssh"
SSH_PROXY = "proxy ssh"

#: Workers a way starts with and the most it grows to (:class:`Lanes`). Ranges over HTTPS
#: pass no ``sshd``: sixteen to start, thirty-two at most. Direct ssh starts with the four
#: that were measured and may reach eight, one connection opened at a time: what refused
#: twelve streams was twelve handshakes at once (``MaxStartups 10:30:100``), not twelve
#: sessions. The proxy keeps its four: more brought little and were refused together.
LANES = {HTTPS: (16, 32), SSH_DIRECT: (WORKERS, 8), SSH_PROXY: (WORKERS, WORKERS)}


# --------------------------------------------------------------------------
# how many workers
# --------------------------------------------------------------------------


class Lanes:
    """How many workers of a transfer may work at once, found as it goes.

    A transfer starts with ``start`` workers. Each time a window has
    passed (``window_s``, and a chunk a worker at least), the bytes a
    second it brought are compared with what the count before it brought:
    ``gain`` times more or better, and the count is doubled, up to
    ``most``; less, and the climb ends, on the count before where the new
    one brought less. A connection that fails takes the count back by a
    quarter, never under ``least``, and ends the climb: the host said it
    has too many. Failures of one window are one failure: four streams cut
    in the same second are one event, not four.

    Bytes are counted when a chunk is whole, so a window is a coarse
    measure: what is asked of it is whether twice the connections still
    bring clearly more, not what they bring.
    """

    def __init__(
        self,
        start: int,
        most: int,
        *,
        least: int = 1,
        window_s: float = 4.0,
        gain: float = 1.15,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.least = max(1, int(least))
        self.most = max(self.least, int(most))
        self.allowed = min(max(self.least, int(start)), self.most)
        self.window_s, self.gain = float(window_s), float(gain)
        self._clock = clock
        self._lock = threading.Lock()
        self._since = clock()
        self._bytes = 0
        self._chunks = 0
        self._failed_at: float | None = None
        #: The count before this one and the rate it brought, while the climb lasts.
        self._before: tuple[int, float] | None = None
        self.settled = self.allowed >= self.most
        #: Every count tried, with the bytes a second its window brought.
        self.history: list[tuple[int, float]] = []

    def may(self, worker: int) -> bool:
        """Whether worker ``worker`` (counted from 0) is one of those that work now."""
        return worker < self.allowed

    def brought(self, count: int) -> None:
        """A chunk of ``count`` bytes arrived whole."""
        with self._lock:
            self._bytes += int(count)
            self._chunks += 1
            now = self._clock()
            if self.settled or now - self._since < self.window_s:
                return
            if self._chunks < self.allowed:
                return
            rate = self._bytes / max(now - self._since, 1e-9)
            self.history.append((self.allowed, rate))
            if self._before is not None and rate < self.gain * self._before[1]:
                # The workers added brought nothing worth their connections.
                if rate < self._before[1]:
                    self.allowed = self._before[0]
                self.settled = True
            else:
                self._before = (self.allowed, rate)
                self.allowed = min(self.most, 2 * self.allowed)
                self.settled = self.allowed == self._before[0]
            self._since, self._bytes, self._chunks = now, 0, 0

    def failed(self) -> None:
        """A connection was lost or refused: fewer of them, and no more are added."""
        with self._lock:
            now = self._clock()
            self.settled = True
            if self._failed_at is not None and now - self._failed_at < self.window_s:
                return
            self._failed_at = now
            self.allowed = max(self.least, self.allowed - max(1, self.allowed // 4))
            self._since, self._bytes, self._chunks = now, 0, 0


# --------------------------------------------------------------------------
# the range server, started over the pinned connection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Served:
    """A range server that answers: where, with which token, and the one certificate believed."""

    host: str
    port: int
    token: str = field(repr=False)
    #: The machine's certificate, PEM, as it came back over the pinned ssh connection.
    certificate: str = field(repr=False)
    #: The folder of the machine that is served, as the machine names it.
    root: str = ""

    @property
    def fingerprint(self) -> str:
        """The SHA-256 of the certificate, in hexadecimal."""
        return hashlib.sha256(ssl.PEM_cert_to_DER_cert(self.certificate)).hexdigest()

    def path(self, remote: str) -> str:
        """The URL's path for the machine's file ``remote``, which is under the folder served."""
        inside = posixpath.relpath(posixpath.normpath(remote), posixpath.normpath(self.root))
        if inside.startswith(".."):
            raise RemoteError(f"{remote} is not under the folder that is served")
        return "/" + quote(inside)

    def context(self) -> ssl.SSLContext:
        """A TLS context that believes this certificate and no other authority."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # The certificate names no host: it is believed for what it is, byte for byte.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cadata=self.certificate)
        return context


class _Pinned(http.client.HTTPSConnection):
    """A connection to a :class:`Served`, refused unless it shows that certificate."""

    def __init__(self, served: Served, timeout: float) -> None:
        super().__init__(served.host, served.port, context=served.context(), timeout=timeout)
        self._wanted = served.fingerprint

    def connect(self) -> None:
        super().connect()
        shown = self.sock.getpeercert(binary_form=True)
        if not shown or not hmac.compare_digest(hashlib.sha256(shown).hexdigest(), self._wanted):
            self.close()
            raise RemoteError("the range server did not show the certificate the machine made")


def _serve_script(root: str, port: int, where: str, lifetime_s: float, extra: str) -> str:
    """The shell that installs and starts the server; its stdin is the token, then the source."""
    name = f"VAST_TCP_PORT_{int(port)}"
    return (
        f"set -e; umask 077; mkdir -p {shlex.quote(where)}; cd {shlex.quote(where)};"
        ' if [ -f pid ]; then kill "$(cat pid)" 2>/dev/null || true; rm -f pid; fi;'
        " IFS= read -r token; printf '%s' \"$token\" > token; unset token; cat > rangeserver.py;"
        " command -v python3 >/dev/null 2>&1 || { echo 'no python3 on the machine' >&2; exit 3; };"
        " command -v openssl >/dev/null 2>&1 || { echo 'no openssl on the machine' >&2; exit 3; };"
        # An RSA key: the one kind every TLS library a machine may hold shakes hands with
        # (a key on a curve was refused by Python 3.9 on LibreSSL 2.8, tried on 127.0.0.1).
        " openssl req -x509 -newkey rsa:2048 -nodes"
        " -keyout key.pem -out cert.pem -days 30 -subj /CN=rv-homecoming >/dev/null 2>&1"
        " || { echo 'openssl made no certificate' >&2; exit 3; };"
        # Detached with every stream of its own, so that the ssh session ends without it.
        " ( $(command -v setsid || true) python3 rangeserver.py"
        f" --root {shlex.quote(root)} --port {int(port)}"
        " --cert cert.pem --key key.pem --token-file token --pid-file pid"
        f" --lifetime {float(lifetime_s):g} --forget --quiet {extra}"
        " > log 2>&1 < /dev/null & );"
        " n=0; while [ ! -f pid ] && [ $n -lt 80 ]; do sleep 0.1; n=$((n + 1)); done;"
        " [ -f pid ] || { echo 'the range server did not start:' >&2; tail -c 400 log >&2;"
        " exit 4; };"
        # The host's port for it, as Vast says it to the instance; empty where it says none.
        f" m=$(printenv {name} 2>/dev/null || true);"
        ' [ -n "$m" ] || m=$(tr "\\0" "\\n" < /proc/1/environ 2>/dev/null'
        f" | sed -n 's/^{name}=//p' | head -1 || true);"
        ' echo "mapped $m"; cat cert.pem'
    )


def _ask_with(machine: Any, command: str, data: bytes, timeout: float = 90.0) -> str:
    """Run ``command`` on ``machine`` with ``data`` on its stdin; what it printed."""
    argv = machine.ssh_command(command)
    try:
        done = subprocess.run(argv, input=data, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RemoteError(f"the range server was not started in {timeout:g} s") from None
    said = done.stderr.decode(errors="replace").strip()
    if done.returncode == 0:
        return done.stdout.decode(errors="replace")
    if connection_level(argv, done.returncode, said):
        raise ConnectionLost(f"range server: the connection was lost: {said[-300:]}", 255)
    raise RemoteError(f"range server: {said[-400:]}", done.returncode)


def _certificate_in(text: str) -> str:
    begin, end = "-----BEGIN CERTIFICATE-----", "-----END CERTIFICATE-----"
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < start:
        raise RemoteError("the machine sent no certificate back")
    return text[start : stop + len(end)] + "\n"


def serve(
    machine: Any,
    root: str,
    *,
    host_port: int | None = None,
    port: int = HOMECOMING_PORT,
    where: str = REMOTE_SERVE,
    lifetime_s: float = SERVE_LIFETIME_S,
    extra: str = "",
    ask: Callable[[Any, str, bytes], str] | None = None,
    ping: bool = True,
    ending: Callable[..., None] | None = None,
) -> Served:
    """Start the range server on ``machine`` for the folder ``root``; where it answers.

    ``machine`` is the instance at its own address **with its host keys
    pinned** (:class:`reverberate.gpu.direct.DirectMachine`): the token goes
    down that connection's stdin and the certificate comes back on it, so
    what is believed afterwards rests on the keys read on the machine, and
    on nothing an address could show. Any other machine is refused.

    ``host_port`` is the host's port for ``port``; left out, it is the one
    the machine was given when rented (``machine.mapped``), or the one the
    instance's own environment names. A server that was started before is
    ended first: one token is good at a time. Raises
    :class:`reverberate.wave.remote.RemoteError` where the way is not open
    (no mapped port, no ``python3``, no ``openssl``, a port that does not
    answer from here), which the caller takes as "not this way"; a server
    that started and cannot be reached is ended before that is said
    (``ending``, :func:`stop` unless a test hands another).
    """
    from reverberate.gpu.direct import DirectMachine

    if not isinstance(machine, DirectMachine) or machine.known_hosts is None:
        raise RemoteError("a range server is started over pinned host keys, or not at all")
    token = secrets.token_urlsafe(32)
    source = Path(rangeserver.__file__).read_bytes()
    said = (ask or _ask_with)(
        machine,
        _serve_script(root, port, where, lifetime_s, extra),
        token.encode() + b"\n" + source,
    )
    try:
        certificate = _certificate_in(said)
        outside = host_port or dict(getattr(machine, "mapped", ()) or ()).get(int(port))
        if not outside:
            for line in said.splitlines():
                word = line.partition("mapped ")[2].strip()
                if line.startswith("mapped ") and word.isdigit():
                    outside = int(word)
        if not outside:
            raise RemoteError(f"the host maps no port to the instance's {port}")
        served = Served(str(machine.host), int(outside), token, certificate, root)
        if ping:
            _ping(served)
    except Exception:
        # Nothing is left listening that nobody will read from.
        (ending or stop)(machine, where=where)
        raise
    return served


def _ping(served: Served) -> None:
    """One request with the token and one without: the first is answered, the second refused."""
    for token, wanted in ((served.token, 204), ("", 401)):
        connection = _Pinned(served, 20.0)
        try:
            connection.request(
                "GET", rangeserver.PING, headers={"Authorization": f"Bearer {token}"}
            )
            status = connection.getresponse().status
        except ssl.SSLCertVerificationError as error:
            raise RemoteError(
                f"the range server's certificate is not the machine's: {error}"
            ) from None
        except (OSError, http.client.HTTPException) as error:
            raise RemoteError(
                f"the range server does not answer at {served.host}:{served.port}: {error}"
            ) from None
        finally:
            connection.close()
        if status != wanted:
            raise RemoteError(f"the range server answered {status} where {wanted} was expected")


def stop(machine: Any, *, where: str = REMOTE_SERVE, run: Callable[..., str] | None = None) -> None:
    """End the range server on ``machine`` and remove what it left; never raises."""
    if run is None:
        from reverberate.wave.remote import run_on

        run = run_on
    command = (
        f'cd {shlex.quote(where)} 2>/dev/null && {{ [ -f pid ] && kill "$(cat pid)" 2>/dev/null;'
        f" cd / && rm -rf {shlex.quote(where)}; }}; true"
    )
    with contextlib.suppress(Exception):
        run(machine, command, what="stop the range server", timeout=60)


class HttpsTransport:
    """A file's ranges over HTTPS, a kept connection a worker; the rest as ``control`` does it.

    ``control`` is the ssh transport of the same machine: the size, the
    digests and the listings are asked there, so a file's SHA-256 never
    comes from the server whose bytes it judges.
    """

    def __init__(
        self, served: Served, control: Transport, *, timeout: float = READ_TIMEOUT_S
    ) -> None:
        self.served, self.control, self.timeout = served, control, float(timeout)
        self._kept: dict[int, _Pinned] = {}
        self._lock = threading.Lock()

    def size(self, remote: str) -> int:
        return self.control.size(remote)

    def sha256(self, remote: str) -> str:
        return self.control.sha256(remote)

    def chunk_digests(self, remote: str, chunk_bytes: int, chunks: Sequence[int]) -> list[str]:
        return self.control.chunk_digests(remote, chunk_bytes, chunks)

    def listing(self, remote_dir: str, exclude: Collection[str]) -> list[tuple[str, int]]:
        return self.control.listing(remote_dir, exclude)

    def files(
        self,
        remote_dir: str,
        names: Sequence[str],
        target: Path,
        worker: int,
        timeout: float | None,
    ) -> None:
        self.control.files(remote_dir, names, target, worker, timeout)

    def _path(self, remote: str) -> str:
        return self.served.path(remote)

    def _connection(self, worker: int) -> _Pinned:
        with self._lock:
            if worker not in self._kept:
                self._kept[worker] = _Pinned(self.served, self.timeout)
            return self._kept[worker]

    def _drop(self, worker: int) -> None:
        with self._lock:
            lost = self._kept.pop(worker, None)
        if lost is not None:
            with contextlib.suppress(Exception):
                lost.close()

    def read(self, remote: str, offset: int, count: int, target: Path, worker: int) -> None:
        path = self._path(remote)
        headers = {
            "Authorization": f"Bearer {self.served.token}",
            "Range": f"bytes={int(offset)}-{int(offset) + int(count) - 1}",
        }
        limit = max(120.0, count / STALLED_BYTES_PER_S)
        began = time.time()
        connection = self._connection(worker)
        try:
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            if response.status != 206:
                response.read()
                if response.status in (401, 403, 404, 416):
                    raise RemoteError(f"range of {remote}: the server answered {response.status}")
                raise ConnectionLost(f"range of {remote}: status {response.status}", None)
            promised = int(response.getheader("Content-Length") or 0)
            buffer = memoryview(bytearray(1 << 20))
            came = 0
            with target.open("wb") as handle:
                while True:
                    got = response.readinto(buffer)
                    if not got:
                        break
                    handle.write(buffer[:got])
                    came += got
                    if time.time() - began > limit:
                        raise ConnectionLost(f"a chunk did not come in {limit:g} s", None)
            if came < promised:
                # Cut in the middle of its body: the connection is not one to ask again on.
                raise ConnectionLost(f"range of {remote}: {came} bytes of {promised} came", None)
        except ssl.SSLCertVerificationError as error:
            self._drop(worker)
            raise RemoteError(
                f"the range server's certificate is not the machine's: {error}"
            ) from None
        except (OSError, http.client.HTTPException) as error:
            self._drop(worker)
            raise ConnectionLost(
                f"range of {remote}: {type(error).__name__}: {error}", None
            ) from None
        except RemoteError:
            self._drop(worker)
            raise

    def close(self) -> None:
        for worker in list(self._kept):
            self._drop(worker)
        self.control.close()


# --------------------------------------------------------------------------
# which way, on this host
# --------------------------------------------------------------------------


@dataclass
class Way:
    """One way a file comes home: its name, the machine as it is reached, its workers."""

    name: str
    #: The machine on the route this way takes: what ``fetch_file`` is handed.
    through: Any
    start: int
    most: int
    #: Whether it goes past Vast's proxy.
    direct: bool = True
    #: The transport it opens, where that is not plain ssh to ``through``.
    transport: Callable[[], Transport] | None = None

    def options(self) -> dict[str, Any]:
        """What :func:`reverberate.gpu.homecoming.fetch_file` is given beside the machine."""
        if self.most <= self.start and self.transport is None:
            # As it always was: so many workers, all of them at work.
            return {"workers": self.start}
        # Never fewer than the four a fetch always had, whatever fails.
        given: dict[str, Any] = {
            "lanes": Lanes(self.start, self.most, least=min(self.start, WORKERS))
        }
        if self.transport is not None:
            given["transport"] = self.transport()
        return given


def verdict_path(machine: Any) -> Path | None:
    """Where the bench's verdict on ``machine``'s host is kept: beside its pinned keys."""
    pinned = getattr(machine, "known_hosts", None)
    return None if pinned is None else Path(str(pinned) + ".transfer.json")


def verdict_of(machine: Any) -> dict[str, Any]:
    """What ``transfer_bench`` measured on ``machine``'s host; empty where it was not run."""
    path = verdict_path(machine)
    if path is None or not path.is_file():
        return {}
    try:
        held = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def ways(
    machine: Any,
    root: str,
    *,
    direct: Any = None,
    say: Any = None,
    verdict: dict[str, Any] | None = None,
    serving: Callable[..., Served] | None = None,
) -> tuple[list[Way], Callable[[], None]]:
    """The ways ``root``'s files come home from ``machine``, the first to try first; and the end.

    Without a verdict: HTTPS ranges where the server starts and answers,
    then ssh at the machine's own address where that answers, then ssh
    through the proxy, which is the way that is known to work. A verdict
    (:func:`verdict_of`, read when none is given) puts the ways in the
    order of what each brought on this host, with the count of streams it
    brought it on, and names the cipher ssh was the fastest with. A way
    the verdict does not name keeps its place after those it names. The
    second value ends what was started: call it when the transfers are
    over. ``direct`` is the machine at its own address where the caller
    has already asked (:func:`reverberate.gpu.homecoming.fastest`).
    """
    say = say or (lambda message: None)
    if direct is None:
        direct = fastest(machine, say=say)
    serving = serving or serve
    found: list[Way] = []
    started: list[Any] = []
    if verdict is None:
        verdict = verdict_of(direct)
    cipher = verdict.get("cipher") or None
    if direct is not machine:
        try:
            served = serving(direct, root)
        except Exception as error:  # noqa: BLE001 - ssh is the way that is known to work
            say(f"  no range server ({str(error)[:160]}): by ssh")
        else:
            started.append(direct)
            say(
                f"  range server at {served.host}:{served.port}, its certificate"
                f" {served.fingerprint[:16]} read over the pinned connection"
            )
            found.append(
                Way(
                    HTTPS,
                    direct,
                    *LANES[HTTPS],
                    transport=lambda: HttpsTransport(served, SshTransport(direct)),
                )
            )
        found.append(
            Way(
                SSH_DIRECT,
                direct,
                *LANES[SSH_DIRECT],
                transport=(lambda: SshTransport(direct, cipher=cipher)) if cipher else None,
            )
        )
    found.append(Way(SSH_PROXY, machine, *LANES[SSH_PROXY], direct=False))
    order = [str(name) for name in verdict.get("order", [])]
    if order:
        found.sort(key=lambda way: order.index(way.name) if way.name in order else len(order))
        said = ", ".join(order)
        say(f"  the order measured on this host: {said}")
    for way in found:
        streams = dict(verdict.get("streams", {})).get(way.name)
        if streams:
            # What the bench found is where the transfer starts; it may still add a few.
            way.start = max(1, min(int(streams), way.most))

    def end() -> None:
        for served_on in started:
            stop(served_on)

    return found, end


def minutes_home(gigabytes: float, megabytes_per_s: float, *, overhead_s: float = 0.0) -> float:
    """Minutes ``gigabytes`` take at ``megabytes_per_s``, with what is not the line's added."""
    return (gigabytes * 1e9 / (megabytes_per_s * 1e6) + overhead_s) / 60.0


# --------------------------------------------------------------------------
# while the campaign still runs
# --------------------------------------------------------------------------


class Early:
    """A look at the machine every few seconds while its campaign runs.

    Two things are waited for. **A file of ``items`` that appears**: a
    campaign writes its pack beside its place and renames it, so the name
    is the whole file; ``bring`` is called with it at once, in this
    thread, while the campaign checks what it wrote. **The campaign's
    end**, ``campaign.done`` or ``campaign.failed``: :attr:`ended` is then
    set, once an appearance, and whoever waits on it looks at once instead
    of at the end of its pause. The looks go on one kept connection, which
    costs a tenth of a second where a new one costs two.

    A ``bring`` that fails is said and tried again at the next look, three
    times at most; a look that fails is taken again. Nothing here ends a
    run.
    """

    TRIES = 3

    def __init__(
        self,
        machine: Any,
        remote_dir: str,
        bring: Callable[[str], Any],
        *,
        items: Sequence[str] = ("pack.h5",),
        poll_s: float = 20.0,
        say: Any = None,
        ask: Callable[..., str] | None = None,
    ) -> None:
        self.machine, self.remote_dir, self.bring = machine, remote_dir, bring
        self.items, self.poll_s = tuple(items), float(poll_s)
        self.say = say or (lambda message: None)
        if ask is None:
            from reverberate.wave.remote import run_on

            ask = run_on
        self._ask = ask
        self.ended = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tried: dict[str, int] = {}
        self._marked = False
        #: The items brought, each with the time it was home at.
        self.home: dict[str, float] = {}

    def look(self) -> None:
        """One look, and what follows from it."""
        names = [*self.items, "campaign.done", "campaign.failed"]
        said = self._ask(
            self.machine,
            f"cd {shlex.quote(self.remote_dir)} 2>/dev/null &&"
            f" ls -1d {' '.join(shlex.quote(name) for name in names)} 2>/dev/null; true",
            what="early look",
            timeout=60,
        )
        there = {line.strip() for line in str(said).splitlines()}
        marked = bool(there & {"campaign.done", "campaign.failed"})
        if marked and not self._marked:
            self.ended.set()
        self._marked = marked
        for item in self.items:
            if item not in there or item in self.home:
                continue
            if self._tried.get(item, 0) >= self.TRIES or self._stop.is_set():
                continue
            self._tried[item] = self._tried.get(item, 0) + 1
            self.say(f"  {item} is written on the machine: brought now, while the campaign goes on")
            try:
                self.bring(item)
            except Exception as error:  # noqa: BLE001 - the fetch at the end brings it
                self.say(f"  {item} not home early ({str(error)[:160]})")
            else:
                self.home[item] = time.time()

    def _loop(self) -> None:
        # The first look after a pause: a campaign that was just launched has written nothing.
        while not self._stop.wait(self.poll_s):
            with contextlib.suppress(Exception):
                self.look()

    def start(self) -> Early:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="early homecoming")
        self._thread.start()
        return self

    def wait(self, seconds: float) -> None:
        """Wait ``seconds``, or less if the campaign's end appears meanwhile."""
        if self.ended.wait(max(0.0, seconds)):
            self.ended.clear()

    def finish(self, wait: bool = True) -> None:
        """No more looks; a file that is on its way is waited for, unless ``wait`` is false."""
        self._stop.set()
        if self._thread is not None and wait:
            self._thread.join()
        self._thread = None
