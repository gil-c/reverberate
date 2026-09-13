"""Cut one band's pressure into shards by point ranges, for the encode boxes.

    python split.py pressure.h5 '{"bounds": [[0, 1021], ...], "pattern": "x.shard{k}.h5"}'

The pressure is row-contiguous per point, so a shard is a slice of rows and
every box pulls only its own.
"""

import json
import sys

import h5py
import numpy as np

src, spec = sys.argv[1], json.loads(sys.argv[2])
with h5py.File(src, "r") as a:
    u = a["u_out"]
    for k, (lo, hi) in enumerate(spec["bounds"]):
        dst = spec["pattern"].format(k=k)
        with h5py.File(dst, "w") as b:
            d = b.create_dataset(
                "u_out",
                shape=(hi - lo, u.shape[1]),
                dtype=np.float32,
                chunks=(min(1021, hi - lo), u.shape[1]),
            )
            for s in range(lo, hi, 8000):
                e = min(s + 8000, hi)
                d[s - lo : e - lo] = u[s:e]
        print("shard", k, lo, hi, flush=True)
