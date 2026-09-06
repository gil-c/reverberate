#!/usr/bin/env bash
# Build PFFDTD on a freshly rented CUDA machine.
#
# PFFDTD was last touched in 2021 and does not run as shipped on a 2026 CUDA
# stack. Four things break, in this order, each only visible after the previous
# one is fixed; B1 lost about an hour rediscovering them on paid hardware. They
# are fixed here so nobody rediscovers them again.
#
#   1. The CUDA Makefile targets -arch=sm_35, which nvcc 12.x refuses outright.
#   2. myfuncs.py calls np.finfo(np.float); the np.float alias is gone. Repaired
#      by patch 7 in reverberate.wave.vendored, not here: a sed in a build
#      script is invisible in review and a silent no-op the day upstream
#      reformats the line it matches.
#   3. The code uses np.bool8, removed in numpy 2, so numpy is pinned to 1.26.
#   4. The voxeliser closes SharedMemory blocks while numpy views are still
#      alive, which raises BufferError on Python >= 3.9.
#
# Idempotent: safe to re-run on a machine that is already built.
#
# Usage, on the rented machine:
#   bash build_pffdtd.sh [install_dir]      # default /root/pffdtd

set -euo pipefail

PFFDTD_REPO="${PFFDTD_REPO:-https://github.com/bsxfun/pffdtd.git}"
# Pinned: the commit B1 was measured against. Do not float this without
# re-running the B1 sweep, because the numbers in data/runs/b1_pffdtd_cost
# describe this tree and no other.
PFFDTD_COMMIT="${PFFDTD_COMMIT:-aa319f6c86517cb95aabfae8656277da62c3ead5}"
# sm_89 is Ada (RTX 4090). Set CUDA_ARCH for other cards: 86 = Ampere/3090,
# 90 = Hopper/H100.
CUDA_ARCH="${CUDA_ARCH:-89}"

INSTALL_DIR="${1:-/root/pffdtd}"
VENV_DIR="${VENV_DIR:-$(dirname "${INSTALL_DIR}")/pffdtd-venv}"

log() { printf '\n==> %s\n' "$*"; }

log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libhdf5-dev is not in the CUDA image and PFFDTD will not build without it.
apt-get install -y -qq git build-essential libhdf5-dev python3-venv python3-dev >/dev/null

log "Source at ${PFFDTD_COMMIT}"
if [ ! -d "${INSTALL_DIR}/.git" ]; then
  git clone --quiet "${PFFDTD_REPO}" "${INSTALL_DIR}"
fi
git -C "${INSTALL_DIR}" fetch --quiet origin
git -C "${INSTALL_DIR}" checkout --quiet "${PFFDTD_COMMIT}"

log "Patch 1: CUDA arch sm_35 -> sm_${CUDA_ARCH}"
sed -i "s/-arch=sm_[0-9]*/-arch=sm_${CUDA_ARCH}/g" "${INSTALL_DIR}/c_cuda/Makefile"


log "Python environment"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "${INSTALL_DIR}/python/pip_requirements.txt" || \
  echo "    (pip_requirements.txt partially applied; the pins below win)"
# Patch 3: numpy < 2, because the code still uses np.bool8. These are the exact
# versions B1 measured on. Installed last so nothing upgrades numpy underneath.
pip install --quiet "numpy==1.26.4" "scipy==1.15.3" "h5py==3.16.0" "numba==0.67.0"

log "Patch 4: tolerant SharedMemory.close()"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "${SITE_PACKAGES}/pffdtd_compat.py" <<'PY'
"""Make SharedMemory.close() tolerant of live numpy views.

PFFDTD's voxeliser closes a shared memory block while numpy arrays still
reference its buffer. Before Python 3.9 that was silently allowed; now
mmap.close() raises BufferError and the run dies after voxelisation has already
been paid for. The blocks are unlinked immediately afterwards anyway, so
swallowing this specific error is safe here.
"""

from multiprocessing import shared_memory

_original_close = shared_memory.SharedMemory.close


def _tolerant_close(self):
    try:
        _original_close(self)
    except BufferError:
        pass


shared_memory.SharedMemory.close = _tolerant_close
PY

# It must be a .pth file, not sitecustomize.py: /usr/lib/python3.10 comes before
# site-packages on sys.path, so a system sitecustomize.py silently shadows ours.
echo "import pffdtd_compat" > "${SITE_PACKAGES}/zz_pffdtd_compat.pth"

log "Build the solvers"
make -C "${INSTALL_DIR}/c_cuda" clean >/dev/null 2>&1 || true
make -C "${INSTALL_DIR}/c_cuda" all

log "Verify"
missing=0
for binary in fdtd_main_cpu_single fdtd_main_cpu_double fdtd_main_gpu_single fdtd_main_gpu_double; do
  if [ -x "${INSTALL_DIR}/c_cuda/${binary}.x" ]; then
    echo "    ok  ${binary}.x"
  else
    echo "    MISSING ${binary}.x"
    missing=1
  fi
done
python - <<'PY'
import multiprocessing.shared_memory as sm

import numpy as np

block = sm.SharedMemory(create=True, size=64)
view = np.ndarray((8,), dtype=np.float64, buffer=block.buf)
view[0] = 1.0
block.close()  # BufferError here means patch 4 did not take
del view
block.unlink()
print(f"    ok  numpy {np.__version__}, tolerant SharedMemory.close")
PY
[ "${missing}" -eq 0 ] || { echo "build incomplete"; exit 1; }

log "Done"
cat <<EOF
    source ${VENV_DIR}/bin/activate
    export PYTHONPATH=${INSTALL_DIR}/python

  Sanity check the toolchain on the shipped church example before trusting any
  measurement; it takes about 5 minutes on an RTX 4090.
EOF
