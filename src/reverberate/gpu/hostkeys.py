"""The ssh host keys of a rented machine, pinned before its address is trusted.

An instance reached at its host's own address is a new name to ``ssh``, and
``StrictHostKeyChecking=accept-new`` would take whatever key answers there the
first time: a machine in the middle of that path would be believed. So the
keys are not learnt from the address. They are **read on the machine itself,
through the channel that is already authenticated** (Vast's ssh proxy, which
admits the account's key alone), written to a file of their own, one per
instance, and the direct connection is made with ``StrictHostKeyChecking=yes``
against that file and no other (:mod:`reverberate.gpu.direct`). A key that
differs is a refusal, never a question.

What this trusts is said in ``docs/open-questions/direct-connection.md``: the
proxy and the host's operator, who hold the machine whatever the route.

:func:`read_through` opens no connection of its own: it is handed the
function that runs a command on a machine. :func:`confirmed_by_api` reads
the instance's log from Vast's API, a second channel that does not rest on
the first connection to the proxy.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverberate.settings import runs_dir

__all__ = [
    "HostKey",
    "READ_COMMAND",
    "confirmed_by_api",
    "forget",
    "host_pattern",
    "known_hosts_text",
    "parse",
    "pin",
    "pinned_path",
    "read_through",
]

#: What is asked of the machine: the public halves of its sshd's keys. Public
#: by nature; the private halves never leave the machine and are never read.
READ_COMMAND = "cat /etc/ssh/ssh_host_*_key.pub"

#: The kinds of key kept, the strongest first, which is the order ``ssh`` is
#: left to prefer them in. DSA is not kept.
KINDS = (
    "ssh-ed25519",
    "ecdsa-sha2-nistp521",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp256",
    "ssh-rsa",
)

_LINE = re.compile(r"^\s*([a-z0-9@.-]+)\s+([A-Za-z0-9+/]+={0,2})(?:\s.*)?$")


@dataclass(frozen=True)
class HostKey:
    """One public host key: its kind and its blob, as ``sshd`` publishes them."""

    kind: str
    blob: str

    @property
    def fingerprint(self) -> str:
        """``SHA256:...``, as ``ssh`` and ``ssh-keygen -l`` print it."""
        digest = hashlib.sha256(base64.b64decode(self.blob)).digest()
        return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def _names_itself(kind: str, blob: str) -> bool:
    """Whether ``blob`` is a key of ``kind``: a key's blob opens with its kind's name.

    What keeps a line of a login banner, or of anything else a host prints,
    from being pinned as a key.
    """
    try:
        raw = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(raw) < 4:
        return False
    length = int.from_bytes(raw[:4], "big")
    return raw[4 : 4 + length] == kind.encode() and len(raw) > 4 + length


def parse(text: str) -> list[HostKey]:
    """The host keys in what a machine printed, the strongest kind first, each once.

    Lines that are not a key of a kind in :data:`KINDS` are dropped without a
    word: a host prints a banner on every login.
    """
    found: dict[tuple[str, str], HostKey] = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if match is None:
            continue
        kind, blob = match.group(1), match.group(2)
        if kind in KINDS and _names_itself(kind, blob):
            found.setdefault((kind, blob), HostKey(kind, blob))
    return sorted(found.values(), key=lambda key: (KINDS.index(key.kind), key.blob))


def read_through(machine: Any, run: Callable[..., str] | None = None) -> list[HostKey]:
    """The machine's host keys, read on it over the route ``machine`` already is.

    ``machine`` is the proxy's route: the one whose authentication is not in
    question. ``run`` is :func:`reverberate.wave.remote.run_on` unless a test
    hands another.
    """
    if run is None:
        from reverberate.wave.remote import run_on

        run = run_on
    return parse(run(machine, READ_COMMAND, what="host keys", timeout=60))


#: Has the machine write its keys' fingerprints to its container's output,
#: which is what Vast keeps as the instance's log.
LOG_COMMAND = (
    "for f in /etc/ssh/ssh_host_*_key.pub; do ssh-keygen -lf $f; done > /proc/1/fd/1; echo written"
)


def _fetch(client: Any, url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(  # noqa: S310 - https only, checked by the caller
        url, timeout=20, context=client._ssl_context()
    ) as response:
        return str(response.read().decode(errors="replace"))


def confirmed_by_api(
    client: Any,
    instance_id: int,
    machine: Any,
    keys: Iterable[HostKey],
    *,
    run: Callable[..., str] | None = None,
    fetch: Callable[[Any, str], str] | None = None,
    tries: int = 5,
    pause_s: float = 4.0,
) -> bool:
    """Whether Vast's API, a channel of its own, shows ``keys`` as the instance's.

    The keys read through the proxy are believed as far as the proxy's first
    connection was, and that one took the key it was shown. This closes it:
    the machine at the end of ``machine``'s route writes its fingerprints to
    its container's output, and the instance's log is asked of the API, over
    TLS that the system's certificates check. A machine that only pretends
    to be the instance cannot write to the instance's log; so every
    fingerprint found there is the instance's own.

    ``False`` when the log does not show every key, or could not be had: the
    caller is told nothing was confirmed, never that something was wrong.
    """
    if run is None:
        from reverberate.wave.remote import run_on

        run = run_on
    wanted = {key.fingerprint for key in keys}
    if not wanted:
        return False
    try:
        run(machine, LOG_COMMAND, what="fingerprints to the log", timeout=60)
        answer = client.request(
            "PUT", f"/instances/request_logs/{int(instance_id)}/", {"tail": "300"}
        )
    except Exception:  # noqa: BLE001 - unconfirmed, whatever stood in the way
        return False
    url = str((answer or {}).get("result_url") or "")
    if not url.startswith("https://"):
        return False
    for _ in range(max(1, tries)):
        time.sleep(pause_s)
        try:
            text = (fetch or _fetch)(client, url)
        except OSError:
            continue
        if all(mark in text for mark in wanted):
            return True
    return False


def host_pattern(host: str, port: int = 22) -> str:
    """How ``known_hosts`` names an address: bare on port 22, bracketed otherwise."""
    if not host or any(c.isspace() or c in ",*?!#|" for c in host):
        raise ValueError(f"not an address a known_hosts line can name: {host!r}")
    return host if int(port) == 22 else f"[{host}]:{int(port)}"


def known_hosts_text(keys: Iterable[HostKey], host: str, port: int = 22) -> str:
    """The ``known_hosts`` lines that pin ``keys`` to one address and nothing else."""
    pattern = host_pattern(host, port)
    lines = [f"{pattern} {key.kind} {key.blob}" for key in keys]
    if not lines:
        raise ValueError("no host key to pin: an empty file would be a file that trusts nothing")
    return "\n".join(lines) + "\n"


def pinned_path(instance_id: int, directory: Path | None = None) -> Path:
    """Where an instance's pinned keys are kept: one file an instance, never shared."""
    return (directory or runs_dir() / "known_hosts") / f"instance_{int(instance_id)}"


def pin(keys: Iterable[HostKey], host: str, port: int, path: Path) -> Path:
    """Write the file that pins ``keys`` to ``host`` and ``port``, replacing any before it.

    Written beside its place and renamed, readable by the user alone. The
    file is replaced and never appended to: an address a host gave to
    another instance yesterday must not be believed today.
    """
    text = known_hosts_text(keys, host, port)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)
    return path


def forget(instance_id: int, directory: Path | None = None) -> None:
    """Remove an instance's pinned keys, once it is destroyed."""
    with contextlib.suppress(FileNotFoundError):
        pinned_path(instance_id, directory).unlink()
