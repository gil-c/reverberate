#!/usr/bin/env bash
#
# Patch 6's acceptance test: the indexed triangle search must change nothing.
#
# Voxelise one scene twice against the same PFFDTD checkout, once with our
# replacement vox_grid_base.py installed and once with upstream's, and compare
# vox_out.h5 byte for byte. Any difference means the index is wrong, and no
# amount of wall clock makes that a good trade -- a patch measured only on speed
# has not been checked at all.
#
# Measured on the bedroom at 4 kHz, 8 processes, a 10 core laptop:
#
#   upstream scan   fill  8.37 s   44 831 tris x 40 832 vox = 1.8e9 comparisons
#   indexed         fill  1.33 s   460 790 (triangle, voxel) pairs, 0.11 s to bin
#   vox_out.h5      88 826 940 bytes, sha256 2fdad989..., both
#
# The saving is not the bedroom. On the whole apartment at 4 kHz the scan is
# 61 626 s of one core and the indexed fill is 67 s.
#
# Usage: scripts/check_vox_index.sh [model.json] [fmax]
set -euo pipefail

MODEL="${1:-$PFFDTD_DIR/../../runs/w25_union/models/bedroom_only.json}"
FMAX="${2:-4000}"
NPROCS="${NPROCS:-8}"

: "${PFFDTD_DIR:?point PFFDTD_DIR at the vendored PFFDTD checkout}"
PFFDTD_PY="${PFFDTD_PY:-$PFFDTD_DIR/../pffdtd-venv/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/run.py" <<'PYTHON'
import sys, os, multiprocessing
multiprocessing.set_start_method("fork", force=True)
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
from pathlib import Path
from common.room_geo import RoomGeo
from fdtd.sim_consts import SimConsts
from voxelizer.cart_grid import CartGrid
from voxelizer.vox_grid import VoxGrid
from voxelizer.vox_scene import VoxScene

model, fmax, out, nprocs = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
geo = RoomGeo(model, az_el=[0.0, 0.0])
consts = SimConsts(Tc=20.0, rh=50.0, fmax=fmax, PPW=10.5, fcc=False)
grid = CartGrid(h=consts.h, offset=3.5, bmin=geo.bmin, bmax=geo.bmax, fcc=False)
voxels = VoxGrid(geo, grid)
voxels.fill(Nprocs=nprocs)
scene = VoxScene(geo, grid, voxels, fcc=False)
scene.calc_adj(Nprocs=nprocs)
scene.check_adj_full()
scene.save(str(out), compress=None)
PYTHON

run () {  # run <label>
    rm -rf "${WORK:?}/mmap_dat" "$WORK/$1"
    ( cd "$WORK" && PYTHONPATH="$PFFDTD_DIR/python" "$PFFDTD_PY" run.py "$MODEL" "$FMAX" "$WORK/$1" "$NPROCS" ) \
        2>&1 | tr '\r' '\n' | grep -E "TIMED voxgrid (index|fill)|indexed .* pairs" || true
}

echo "== upstream, the scan =="
git -C "$PFFDTD_DIR" checkout -- python/voxelizer/vox_grid_base.py
run upstream

echo "== patched, the index =="
cp "$REPO/src/reverberate/wave/vendored/vox_grid_base.py" "$PFFDTD_DIR/python/voxelizer/vox_grid_base.py"
run patched

echo "== vox_out.h5 =="
UP="$(shasum -a 256 "$WORK/upstream/vox_out.h5" | cut -d' ' -f1)"
PA="$(shasum -a 256 "$WORK/patched/vox_out.h5" | cut -d' ' -f1)"
if [ "$UP" = "$PA" ]; then
    echo "IDENTICAL  $UP  ($(wc -c < "$WORK/patched/vox_out.h5") bytes)"
else
    echo "DIFFERENT -- the index is wrong"
    echo "  upstream $UP"
    echo "  patched  $PA"
    exit 1
fi
