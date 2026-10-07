"""A rented machine's own address, believed only with the keys the machine holds.

Every connection of this repository went to ``sshN.vast.ai``, a relay Vast
runs between the laptop and the host, and the relay is the slow part of a
run whose result is large. An instance is now created with a port of its
host mapped to its ssh (:data:`reverberate.gpu.vast.RUNTYPE_DIRECT`), a
:class:`~reverberate.wave.remote.Machine` carries that address beside the
proxy's, and the transfers try it first
(:func:`reverberate.gpu.homecoming.fastest`). What was measured on each
route is in ``docs/open-questions/direct-connection.md``.

This module is what makes that address safe to use, and the whole of the
seam with the code that transfers:

- :func:`upgrade` takes the machine that answered through the proxy and
  hands it back with its host keys pinned (:mod:`reverberate.gpu.hostkeys`):
  a :class:`PinnedMachine`, still on the proxy, whose
  :meth:`~PinnedMachine.directly` is a :class:`DirectMachine` that checks
  them. Where the keys cannot be pinned, or the address shows another key,
  the machine comes back **without** its direct address. It never fails a
  run: the proxy is the fallback. :func:`reverberate.gpu.vast.wait_for_ssh`
  calls it, so every rented machine has passed through it.
- :func:`ssh_options` and :func:`rsync_shell` say how any machine is
  addressed, on either route, for code that builds its own command.
- :func:`rank` orders offers for a run whose result must come home: hosts
  with open ports first, then the regions the caller prefers.

**What a connection to a rented machine may do** is fixed here and not left
to the caller's ssh configuration. It authenticates with the account's key
and nothing else: no password, no keyboard, no other key of the agent
offered. It forwards nothing: no agent, no X11, no port in either
direction, whatever ``~/.ssh/config`` grants elsewhere. On the direct route
it believes the pinned keys alone. The rented machine is somebody else's
computer; nothing of the laptop is lent to it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reverberate.gpu import hostkeys
from reverberate.wave.remote import (
    CONNECT_TIMEOUT_S,
    CONTROL_PERSIST_S,
    Machine,
    RemoteError,
    _run,
)

__all__ = [
    "DirectMachine",
    "PinnedMachine",
    "add_arguments",
    "is_pinned",
    "rank",
    "regions",
    "rsync_shell",
    "ssh_options",
    "upgrade",
]

#: What no connection to a rented machine is allowed, whatever the user's own
#: ssh configuration grants elsewhere. The first word wins in ``ssh``, so
#: these stand before anything a configuration file says.
_LEND_NOTHING = (
    "ForwardAgent=no",
    "ForwardX11=no",
    "ClearAllForwardings=yes",
    "PermitLocalCommand=no",
    "Tunnel=no",
)
#: Authentication is the account's key, named, and nothing else is offered.
_KEY_ONLY = (
    "BatchMode=yes",
    "IdentitiesOnly=yes",
    "PreferredAuthentications=publickey",
    "PasswordAuthentication=no",
    "KbdInteractiveAuthentication=no",
)
#: The pinned file is the only word on who the host is: no file of the
#: system's, no key added because the server offered one, no question asked.
_PINNED_ONLY = (
    "StrictHostKeyChecking=yes",
    "GlobalKnownHostsFile=/dev/null",
    "UpdateHostKeys=no",
    "CheckHostIP=no",
)


def _words(options: Iterable[str]) -> list[str]:
    argv: list[str] = []
    for option in options:
        argv += ["-o", option]
    return argv


@dataclass(frozen=True)
class DirectMachine(Machine):
    """A machine at its host's own address, answering with the keys pinned for it.

    A :class:`~reverberate.wave.remote.Machine` everywhere one is taken:
    ``ssh_command``, ``scp_command``, ``sharing`` and ``close_command`` are
    the parent's, with the options below in place of the parent's
    ``accept-new``.
    """

    #: The file that pins this instance's host keys (:func:`hostkeys.pin`).
    known_hosts: Path | None = None
    #: The instance's ports the host maps, each with the host's port for it
    #: (:func:`reverberate.gpu.vast.mapped_ports`): where a range server is reached.
    mapped: tuple[tuple[int, int], ...] = ()

    def _options(self) -> list[str]:
        if self.known_hosts is None:
            raise ValueError("a direct route without pinned host keys is not connected to")
        words = [
            *_LEND_NOTHING,
            *_KEY_ONLY,
            *_PINNED_ONLY,
            f"UserKnownHostsFile={self.known_hosts}",
            f"ConnectTimeout={CONNECT_TIMEOUT_S}",
            "ServerAliveInterval=30",
            "ServerAliveCountMax=4",
        ]
        if self.control:
            words += [
                "ControlMaster=auto",
                f"ControlPath={self.control}",
                f"ControlPersist={CONTROL_PERSIST_S}",
            ]
        return _words(words)


@dataclass(frozen=True)
class PinnedMachine(Machine):
    """A machine through the proxy, whose own address is believed by its pinned keys.

    Commands go where they always went. :meth:`directly` is the difference:
    the parent's connects to the address and takes the key it is shown
    (``accept-new``); this one returns a :class:`DirectMachine`, which
    refuses any key but those read on the machine.
    """

    known_hosts: Path | None = None
    mapped: tuple[tuple[int, int], ...] = ()

    def _options(self) -> list[str]:
        return [*_words([*_LEND_NOTHING, *_KEY_ONLY]), *super()._options()]

    def directly(self) -> Machine | None:
        """This machine at its own address, its keys checked; ``None`` where it has none."""
        if self.direct is None or self.known_hosts is None:
            return None
        return DirectMachine(
            host=self.direct[0],
            port=int(self.direct[1]),
            user=self.user,
            identity=self.identity,
            control=self.control,
            known_hosts=self.known_hosts,
            mapped=self.mapped,
        )


# --------------------------------------------------------------------------
# the seam: how a machine is addressed, whatever its route
# --------------------------------------------------------------------------


def is_pinned(machine: Any) -> bool:
    """Whether ``machine`` is a direct route with pinned keys, or holds one."""
    if isinstance(machine, DirectMachine):
        return machine.known_hosts is not None
    return isinstance(machine, PinnedMachine) and machine.directly() is not None


def ssh_options(machine: Any) -> list[str]:
    """The ``ssh`` options that go with ``machine`` on the route it is on, as argv."""
    return list(machine._options())


def rsync_shell(machine: Any) -> str:
    """The remote shell ``rsync -e`` is given for ``machine``, as one string.

    For a plain machine this is, word for word, what
    :func:`reverberate.wave.remote_voxelise.rsync` has always built. For a
    pinned or a direct one it carries that machine's own options. ``rsync``
    splits the string on spaces, so no word of it may hold one.
    """
    identity = ["-i", str(machine.identity)] if machine.identity else []
    if isinstance(machine, DirectMachine | PinnedMachine):
        words = ["ssh", "-p", str(machine.port), *machine._options(), *identity]
    else:
        words = ["ssh", "-p", str(machine.port), "-o", "StrictHostKeyChecking=accept-new"]
        words += identity
    spaced = [word for word in words if any(c.isspace() for c in word)]
    if spaced:
        raise ValueError(f"rsync splits its remote shell on spaces, and these hold one: {spaced}")
    return " ".join(words)


# --------------------------------------------------------------------------
# pinning the address before it is used
# --------------------------------------------------------------------------

#: What ``ssh`` says when the key that answered is not the pinned one.
_MISMATCH = ("HOST IDENTIFICATION HAS CHANGED", "Host key verification failed")


def upgrade(
    machine: Any,
    instance_id: int,
    *,
    client: Any = None,
    confirm: bool = False,
    directory: Path | None = None,
    say: Callable[[str], None] = print,
    run: Callable[..., str] | None = None,
    probe: Callable[..., str] | None = None,
    mapped: Sequence[tuple[int, int]] = (),
) -> Any:
    """``machine`` with its own address pinned, or without that address.

    ``machine`` is the instance as it answered through the proxy, carrying
    the address the API gave (``machine.direct``). Three things must hold
    for the address to be kept, and each is said when it does not:

    1. the machine's host keys are read, on it, through the proxy;
    2. they are pinned to the address in the instance's own file;
    3. a command answers at the address with ``StrictHostKeyChecking=yes``
       against that file.

    Then a :class:`PinnedMachine` comes back. Otherwise the machine comes
    back with ``direct`` unset, so that nothing later connects to an
    address whose keys were never checked; a key that does not match is
    said for what it is. A machine that had no address comes back as it
    is. Nothing here raises: the caller holds a machine that works.

    ``confirm`` asks one thing more of a pinned machine: that its keys be
    found in the instance's log as Vast's API gives it
    (:func:`hostkeys.confirmed_by_api`, which needs ``client``), a check
    that no longer rests on the proxy's first connection. It costs one
    call to the API and some ten seconds; it is off until it has been seen
    on more hosts than the one it was tried on.

    ``mapped`` is the instance's ports the host maps, as the API's record
    gives them (:func:`reverberate.gpu.vast.mapped_ports`); the pinned
    machine carries them, for a range server
    (:func:`reverberate.gpu.transfer.serve`).
    """
    address = getattr(machine, "direct", None)
    if address is None:
        return machine
    host, port = str(address[0]), int(address[1])
    without = replace(machine, direct=None)
    try:
        keys = hostkeys.read_through(machine, run=run)
    except Exception as error:  # noqa: BLE001 - the proxy's route still stands
        say(f"  direct route not taken: host keys not read ({str(error)[:80]})")
        return without
    if not keys:
        say("  direct route not taken: the machine showed no host key")
        return without
    try:
        path = hostkeys.pin(keys, host, port, hostkeys.pinned_path(instance_id, directory))
        pinned = PinnedMachine(
            host=machine.host,
            port=machine.port,
            user=machine.user,
            identity=machine.identity,
            direct=(host, port),
            control=machine.control,
            known_hosts=path,
            # The ports the host maps beside ssh, for a range server (``gpu.transfer``).
            mapped=tuple((int(a), int(b)) for a, b in mapped),
        )
        direct = pinned.directly()
        if direct is None:
            raise ValueError("no route came of the pinned file")
    except Exception as error:  # noqa: BLE001 - a file that cannot be written is not a run lost
        say(f"  direct route not taken: its keys could not be pinned ({str(error)[:80]})")
        return without
    try:
        (probe or _run)(direct.ssh_command("true"), what="direct ssh probe", timeout=45, attempts=2)
    except Exception as error:  # noqa: BLE001 - whatever stood in the way, the proxy stands
        if isinstance(error, RemoteError) and any(text in str(error) for text in _MISMATCH):
            say(
                f"  DIRECT ROUTE REFUSED: {host}:{port} did not show the key the machine"
                " holds; something stands on that path. The proxy is the route."
            )
        else:
            say(f"  direct route not taken: {host}:{port} did not answer ({str(error)[-120:]})")
        hostkeys.forget(instance_id, directory)
        return without
    say(f"  direct route {host}:{port}, pinned {keys[0].kind} {keys[0].fingerprint}")
    if confirm:
        if client is not None and hostkeys.confirmed_by_api(
            client, instance_id, direct, keys, run=run
        ):
            say("  the pinned keys are the instance's own, by its log from Vast's API")
        else:
            say(
                "  the pinned keys were not confirmed through Vast's API: believed as the proxy was"
            )
    return pinned


# --------------------------------------------------------------------------
# which offers, for a run whose result is large
# --------------------------------------------------------------------------


def regions(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """A ``--prefer-region`` value as its list: ``"FR,GB,Quebec"`` or several of them."""
    if value is None:
        return ()
    parts = [value] if isinstance(value, str) else list(value)
    return tuple(word.strip() for part in parts for word in part.split(",") if word.strip())


def region_rank(location: str, prefer: Sequence[str]) -> int:
    """Where ``location`` stands in ``prefer``, or its length when it is not there.

    An offer's location reads ``"Quebec, CA"`` or ``"France, FR"``. Two
    letters name the country code after the comma; anything longer is looked
    for in the whole, so ``"California"`` and ``"Quebec"`` name a place.
    Case is not read.
    """
    place = location.strip().lower()
    code = place.rsplit(",", 1)[-1].strip()
    for index, wanted in enumerate(prefer):
        word = wanted.strip().lower()
        if not word:
            continue
        if (len(word) == 2 and word == code) or (len(word) > 2 and word in place):
            return index
    return len(prefer)


def rank(offers: Iterable[Any], prefer_regions: Sequence[str] = ()) -> list[Any]:
    """``offers`` with open ports first, then by preferred region; their own order within.

    A preference and not a filter: an offer is never dropped, so a search
    that found machines still finds them. The order the offers came in,
    which is the price's, is kept among equals.
    """
    return sorted(
        offers,
        key=lambda offer: (
            0 if int(getattr(offer, "direct_ports", 0) or 0) > 0 else 1,
            region_rank(str(getattr(offer, "location", "") or ""), prefer_regions),
        ),
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``--prefer-region``, for a command that rents."""
    parser.add_argument(
        "--prefer-region",
        action="append",
        default=[],
        metavar="REGIONS",
        help="regions to rent in first, nearest the place the result comes home to: country"
        " codes or place names, comma separated (FR,GB,Quebec). A preference, not a filter;"
        " the rates measured from each are in docs/open-questions/direct-connection.md",
    )
