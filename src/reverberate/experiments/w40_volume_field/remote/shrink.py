"""Rewrite the engine's float64 output as float32, chunked one array per chunk.

    python shrink.py sim_outs.h5 pressure.h5

Runs on the card's host under PFFDTD's interpreter. Halves the disk and the
transfer; the encoder never needed more than single precision.
"""

import sys

import h5py
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as a, h5py.File(dst, "w") as b:
    u = a["u_out"]
    n, t = u.shape
    d = b.create_dataset("u_out", shape=(n, t), dtype=np.float32, chunks=(min(1021, n), t))
    for s in range(0, n, 8000):
        d[s : s + 8000] = np.asarray(u[s : s + 8000], dtype=np.float32)
print("shrunk", n, t)
