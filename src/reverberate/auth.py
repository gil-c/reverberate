"""KeePassXC secrets: inject() into os.environ, require() to read them back."""

from __future__ import annotations

import os
import sys

from keepassify import KeePassXCError, SecretStore, write_association

__all__ = ["KeePassXCError", "inject", "require"]

#: KeePassXC entry-URL namespaces: the "sites" these secrets are stored under
#: in the vault (get-logins looks entries up by URL, same as a browser would).
#: ``dev-common.local`` holds credentials shared across the owner's projects,
#: such as the Vast.ai key; ``reverberate.local`` holds this project's own. The
#: project namespace is searched last, so it wins on a name collision.
URLS = ("https://dev-common.local", "https://reverberate.local")

_cache: dict[str, str] | None = None


def _running_in_ci() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))


def _fetch() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {} if _running_in_ci() else SecretStore(urls=list(URLS), skip_in_ci=False).fetch()
    return _cache


def inject(names: list[str] | None = None) -> int:
    """Copy KeePassXC secrets into os.environ (never overwriting existing ones).

    Returns the number of variables added. No-op in CI, where secrets are
    already environment variables.
    """
    if _running_in_ci():
        return 0
    secrets = _fetch()
    items = secrets.items() if names is None else ((n, secrets[n]) for n in names if n in secrets)
    added = 0
    for name, value in items:
        if name not in os.environ:
            os.environ[name] = value
            added += 1
    return added


def require(name: str) -> str:
    """Return os.environ[name], raising loudly if it is unset."""
    value = os.environ.get(name)
    if not value:
        raise KeePassXCError(f"required environment variable {name!r} is not set")
    return value


def _main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["list"])[0]
    store = SecretStore(urls=list(URLS))
    if command == "associate":
        association = store.associate()
        path = write_association(association, store.association_path)
        print(f"Associated as {association['id']!r}; saved to {path}")
    elif command == "list":
        for name, value in sorted(store.fetch().items()):
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
