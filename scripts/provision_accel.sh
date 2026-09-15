#!/usr/bin/env bash
# Provision one rented CUDA machine for the whole campaign: the engine, and an
# interpreter for reverberate.accel with cupy on top.
#
# Two interpreters, on purpose. PFFDTD needs numpy below 2 and Python 3.10, and
# scripts/build_pffdtd.sh gives it exactly that; it is never imported by this
# project's code. reverberate needs Python 3.11 or later and numpy 2, so it
# gets its own environment from uv, which fetches a CPython of the requested
# version in seconds and installs the pinned requirements in about a minute.
#
# Idempotent: every step checks for what it made.
#
# Usage, on the rented machine, with /root/requirements-remote.txt uploaded:
#   bash provision_accel.sh
set -euo pipefail

VENV="${ACCEL_VENV:-/root/accel-venv}"
PYTHON_VERSION="${ACCEL_PYTHON:-3.12}"
REQUIREMENTS="${1:-/root/requirements-remote.txt}"

log() { printf '\n==> %s\n' "$*"; }

log "uv"
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
UV="$HOME/.local/bin/uv"

log "CPython ${PYTHON_VERSION}"
"$UV" python install "${PYTHON_VERSION}" >/dev/null 2>&1 || true

log "environment ${VENV}"
if [ ! -x "${VENV}/bin/python" ]; then
  "$UV" venv --python "${PYTHON_VERSION}" "${VENV}" >/dev/null
fi
"$UV" pip install --python "${VENV}/bin/python" -r "${REQUIREMENTS}" >/dev/null

log "the card, as cupy sees it"
"${VENV}/bin/python" - <<'PY'
import cupy, numpy
n = cupy.cuda.runtime.getDeviceCount()
for i in range(n):
    p = cupy.cuda.runtime.getDeviceProperties(i)
    name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
    print(f"    device {i}: {name}, sm_{p['major']}{p['minor']}, {p['totalGlobalMem'] / 1e9:.1f} GB")
print(f"    cupy {cupy.__version__}, numpy {numpy.__version__}, runtime {cupy.cuda.runtime.runtimeGetVersion()}")
x = cupy.arange(10, dtype=cupy.float64)
assert float((x * x).sum()) == 285.0
print("    ok  a kernel ran")
PY

log "Done"
