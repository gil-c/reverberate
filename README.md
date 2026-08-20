# reverberate

A machine learning surrogate that predicts the acoustic response of a room directly
from its 3D mesh and the acoustic properties of its surfaces, without running a wave
or ray simulation.

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync --extra dev
```

Opening a new worktree runs this automatically via a git hook (see
`.githooks/post-checkout`); enable it once per clone with:

```sh
git config core.hooksPath .githooks
```

### Secrets

Local secrets are supplied through [KeePassXC's Browser Integration
protocol](https://github.com/gil-c/keepassify), never through a `.env` file or
anything checked into the repository. See `.env.example` for the list of variables
the project expects in the process environment, and `src/reverberate/auth.py` for the
local pairing/injection tool. Application code reads secrets only through
`src/reverberate/settings.py`, which fails loudly at startup if a required variable
is missing.

One-time local pairing:

```sh
uv run python -m reverberate.auth associate
```
