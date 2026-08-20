"""The single module through which application code reads secrets.

Per ROADMAP.md section 6.1: code accesses secrets *only* through
``os.environ``, via this module, and nowhere else in the codebase reads
environment variables directly. This module never falls back to a default
and never continues in a degraded mode: a missing required variable is a
loud, immediate failure that names the variable.

Secret values are never logged, never printed and never included in an
exception message; only the variable *name* is ever referenced.

Locally, secrets are supplied by the developer's KeePassXC secret manager via
:mod:`reverberate.auth` (run ``python -m reverberate.auth associate`` once,
then have your shell/IDE run configuration call ``auth.inject()`` before
launching, or run scripts through ``uv run python -m reverberate.auth`` style
tooling). In CI, GitHub Actions secrets are injected directly as environment
variables by the workflow. This module does not know or care which of those
happened; it only reads ``os.environ``.
"""
from __future__ import annotations

import os


class MissingSecretError(RuntimeError):
    """Raised when a required environment variable is not set."""


def require(name: str) -> str:
    """Return the value of the required environment variable *name*.

    Raises :class:`MissingSecretError` immediately, naming the variable, if
    it is unset or empty. Never returns a default.
    """
    value = os.environ.get(name)
    if not value:
        raise MissingSecretError(
            f"required environment variable {name!r} is not set. "
            "Locally: `uv run python -m reverberate.auth associate` once, "
            "then inject it before running (see reverberate.auth.inject()). "
            "In CI: add it to the workflow as `secrets.{name}` and to "
            "`.env.example`.".format(name=name)
        )
    return value


def optional(name: str, default: str | None = None) -> str | None:
    """Return the value of environment variable *name*, or *default* if unset."""
    return os.environ.get(name, default)
