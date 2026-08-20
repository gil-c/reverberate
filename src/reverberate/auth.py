"""Secrets via KeePassXC's browser-integration protocol.

Usage::

    import reverberate.auth as auth
    auth.inject()                    # all entries -> os.environ
    auth.inject(["VASTAI_API_KEY"])  # only this one

Existing environment variables always win and are never overwritten. In CI
(``GITHUB_ACTIONS`` or ``CI`` set) this is a no-op, since secrets are already
injected as environment variables there.

This is a thin, reverberate-specific wrapper around the shared `keepassify
<https://github.com/gil-c/keepassify>`_ package, which implements the generic
KeePassXC Browser Integration protocol (NaCl-encrypted JSON over a local
socket/pipe). Only the bits specific to this project - which KeePassXC entry
URLs to query - live here. See ROADMAP.md section 6.1: application code never
imports this module directly; it reads validated values from ``os.environ``
via :mod:`reverberate.settings` instead. This module is the local, developer
side tool that populates the environment before that happens.

Secrets are stored in KeePassXC as ordinary entries: the URL is the
namespace, the username is the environment variable name, and the password is
the value. Edit ``URLS`` below to point at a different project's namespace(s).

CLI: ``python -m reverberate.auth associate`` registers this machine with
KeePassXC; ``python -m reverberate.auth list`` prints the known secret names.
"""
from __future__ import annotations

import os
import sys

from keepassify import KeePassXCError, SecretStore, write_association

__all__ = ["KeePassXCError", "inject"]

#: This project's KeePassXC entry-URL namespace. Later URLs win over earlier
#: ones when the same variable name appears under multiple URLs.
URLS = ("https://dev-common.local", "https://reverberate.local")

TIMEOUT = 60.0

_cache: dict[str, str] | None = None


def _running_in_ci() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))


def _make_store(urls: tuple[str, ...] = URLS) -> SecretStore:
    """Build a fresh SecretStore, re-reading env vars (e.g. KEEPASSXC_SECRETS_HOME,
    TIMEOUT) on every call rather than baking them in once at import time."""
    return SecretStore(urls=urls, timeout=TIMEOUT, skip_in_ci=False)


def _fetch_secrets(urls: tuple[str, ...] = URLS) -> dict[str, str]:
    try:
        return _make_store(urls).fetch()
    except KeePassXCError as exc:
        if "no KeePassXC association" in str(exc):
            raise KeePassXCError(f"{exc}; run `python -m reverberate.auth associate`") from exc
        raise


def _from_keepassxc() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {} if _running_in_ci() else _fetch_secrets()
    return _cache


def inject(names: list[str] | None = None) -> int:
    """Copy KeePassXC secrets into os.environ and return how many were added.

    If *names* is given, only those variable names are considered; it is not
    an error for some of them to already be set. Existing environment
    variables always take priority and are never overwritten. No-op
    returning 0 in CI.

    Raises KeePassXCError if KeePassXC is unreachable or stays locked.
    """
    if _running_in_ci():
        return 0
    secrets = _from_keepassxc()
    items = secrets.items() if names is None else (
        (name, secrets[name]) for name in names if name in secrets
    )
    added = 0
    for name, value in items:
        if name not in os.environ:
            os.environ[name] = value
            added += 1
    return added


def _main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["list"])[0]
    if command == "associate":
        print("Select the target database in KeePassXC, then approve the dialog.")
        store = _make_store()
        association = store.associate()
        path = write_association(association, store.association_path)
        print(f"Associated as {association['id']!r}; saved to {path}")
    elif command == "list":
        for name, value in sorted(_fetch_secrets().items()):
            print(f"{name:<24} {len(value)} chars")
    else:
        print("usage: python -m reverberate.auth [associate|list]")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except KeePassXCError as error:
        print(f"error: {error}")
        raise SystemExit(1) from None
