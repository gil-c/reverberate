"""Concatenate the float32 slices of one band in row order.

    python merge.py part0.h5 part1.h5 ... merged.h5

A band whose output does not fit the host's RAM is solved as consecutive
slices of its receiver rows; the merged file has the rows of the whole plan.
"""

import sys

import h5py
import numpy as np

parts, dst = sys.argv[1:-1], sys.argv[-1]
shapes = []
for p in parts:
    with h5py.File(p, "r") as a:
        shapes.append(a["u_out"].shape)
n, t = sum(s[0] for s in shapes), shapes[0][1]
with h5py.File(dst, "w") as b:
    d = b.create_dataset("u_out", shape=(n, t), dtype=np.float32, chunks=(min(1021, n), t))
    at = 0
    for p in parts:
        with h5py.File(p, "r") as a:
            u = a["u_out"]
            for s in range(0, u.shape[0], 8000):
                blk = u[s : s + 8000]
                d[at : at + blk.shape[0]] = blk
                at += blk.shape[0]
print("merged", n, t)
