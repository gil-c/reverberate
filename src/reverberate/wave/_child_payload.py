"""Build the viewer's payload beside the grid, on whatever machine holds it.

Run by :func:`reverberate.wave.remote_voxelise.build_payload_remote` as a
subprocess, never imported. Same contract as ``_child_voxelise``: a JSON job on
stdin, a JSON report on stdout, files in ``out_dir``.

**This exists so the grid never travels.** ``vox_out.h5`` is 25 GB for the flat
at 16 kHz and the payload the browser fetches is about 185 MB, so building the
payload where the grid already is turns a transfer that has failed into one that
takes seconds. It also keeps it off the laptop, which is the standing rule.

It needs only numpy and h5py, both of which PFFDTD's own interpreter already
has, so a machine that can voxelise can do this immediately afterwards with no
second provisioning step. ``vox_view`` is copied across beside this file rather
than installed as a package for the same reason.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    job = json.loads(sys.stdin.read())
    modules = Path(job["module_dir"])
    sys.path.insert(0, str(modules))

    # ``vox_view`` reaches for ``reverberate.wave.comms.transpose_order``. The
    # real module is copied across beside it and made importable under its own
    # name, rather than reimplemented here as a one-line stub: the permutation
    # has to agree with what ``rotate_sim_data`` did to the grid, and a copy
    # that drifts would mis-rotate every node silently.
    for package in ("reverberate", "reverberate/wave"):
        init = modules / package / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.touch()
    (modules / "reverberate" / "wave" / "comms.py").write_bytes((modules / "comms.py").read_bytes())

    import vox_view  # type: ignore[import-not-found]

    cache_dir = Path(job["cache_dir"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = int(job["target_cubes"])

    timings: dict[str, float] = {}

    def timed(name: str, call: Any) -> Any:
        start = time.time()
        result = call()
        timings[name] = round(time.time() - start, 3)
        return result

    surface = timed("surface_s", lambda: vox_view.read_surface(cache_dir, budget))
    record = timed(
        "write_s",
        lambda: vox_view.write_voxel_payload(surface, list(job["labels"]), out_dir),
    )

    blocks = surface.blocks
    report = {
        "timings": timings,
        "quads": surface.quads,
        "triangles": surface.triangles,
        "blocks": blocks.drawn,
        "total_nodes": blocks.total_nodes,
        "cell_m": blocks.cell_m,
        "h_m": blocks.h_m,
        "lossless": not blocks.aggregated,
        "payload_bytes": sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()),
        "note": record.get("note", ""),
    }
    sys.stdout.write("\n@@REVERBERATE@@" + json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
