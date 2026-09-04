#!/usr/bin/env bash
#
# The acceptance for slabbed voxelisation: how a grid is cut up must not change
# what comes out of it.
#
# Voxelise one scene several ways -- whole, in slabs, at another voxel side, and
# both at once -- and compare the four datasets the engine reads. Datasets, not
# file bytes: a slabbed run appends to resizable HDF5 datasets and a single pass
# writes contiguous ones, so the containers differ while every number in them is
# the same. That is the honest comparison and it is the one that matters.
#
# Measured on the bedroom at 4 kHz, 3 861 276 boundary nodes:
#
#   single         37.6 s
#   slabs 2        33.1 s   IDENTICAL
#   slabs 5        34.4 s   IDENTICAL
#   nh 20          39.7 s   IDENTICAL
#   slabs 3, nh 20 35.3 s   IDENTICAL
#
# Usage: scripts/check_slabs.sh <model.json> <materials dir> [fmax]
set -euo pipefail

MODEL="${1:?a model json}"
MATS="${2:?a directory of impedance .h5 files}"
FMAX="${3:-4000}"
: "${PFFDTD_DIR:?point PFFDTD_DIR at the vendored PFFDTD checkout}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PYTHONPATH="$REPO/src" "${PYTHON:-$REPO/.venv/bin/python}" \
    "$REPO/scripts/_slab_identity.py" "$MODEL" "$MATS" "$FMAX" "$WORK" \
    '{"slabs":2}' '{"slabs":5}' '{"nh":20}' '{"slabs":3,"nh":20}'
