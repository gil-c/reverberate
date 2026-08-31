"""Voxelisation, in PFFDTD's own interpreter, one stage at a time.

Run by :mod:`reverberate.wave.voxelise` as a subprocess, never imported: PFFDTD
needs numpy below 2 and a patched install of its own, which is not what the rest
of this project runs on. The only contract is the one in this file's docstring:
a JSON job on stdin, a JSON report on stdout, files in ``out_dir``.

This is ``sim_setup`` unrolled rather than called, for two reasons. It is timed
**per stage**, which W8 item 5 asks for and a single call cannot give; and the
source and receiver step is separated out, so the cache entry it leaves behind
holds only what depends on the scene and the grid. ``comms_out.h5`` is built
here only because ``rotate_sim_data`` and ``sort_sim_data`` read and rewrite it
alongside ``vox_out.h5``, and it is deleted before the entry is published:
:mod:`reverberate.wave.comms` regenerates it per source and receiver pair for
nothing.

The sequence, the arguments and the order are copied from ``sim_setup``; any
divergence is a bug here, and ``tests/test_wave_comms.py`` compares the output
against a real ``sim_setup`` run.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def main() -> int:
    job = json.loads(sys.stdin.read())

    pffdtd = Path(job["pffdtd_dir"])
    sys.path.insert(0, str(pffdtd / "python"))
    # PFFDTD's voxeliser hands closures to multiprocessing, which only works
    # under fork. macOS defaults to spawn, so a laptop voxelisation dies at the
    # first worker without this. The rented Linux box forks by default.
    multiprocessing.set_start_method("fork", force=True)
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    import numpy as np
    from common.room_geo import RoomGeo
    from fdtd.rotate_sim_data import fold_fcc_sim_data, rotate_sim_data, sort_sim_data
    from fdtd.sim_comms import SimComms
    from fdtd.sim_consts import SimConsts
    from fdtd.sim_mats import SimMats
    from voxelizer.cart_grid import CartGrid
    from voxelizer.vox_grid import VoxGrid
    from voxelizer.vox_scene import VoxScene

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fcc = bool(job["fcc"])
    compress = job.get("compress")
    nprocs = job.get("nprocs")
    timings: dict[str, float] = {}

    def timed(name: str, call: Callable[[], Any]) -> Any:
        start = time.time()
        result = call()
        timings[name] = round(time.time() - start, 3)
        return result

    bmin = np.array(job["bmin"], dtype=np.float64) if job.get("bmin") else None
    bmax = np.array(job["bmax"], dtype=np.float64) if job.get("bmax") else None

    room_geo = timed(
        "room_geo_s",
        lambda: RoomGeo(
            job["model_json"], az_el=job.get("rot_az_el", [0.0, 0.0]), bmin=bmin, bmax=bmax
        ),
    )
    room_geo.print_stats()

    sim_consts = SimConsts(Tc=job["Tc"], rh=job["rh"], fmax=job["fmax"], PPW=job["ppw"], fcc=fcc)
    sim_consts.save(str(out_dir))

    sim_mats = SimMats(save_folder=str(out_dir))
    sim_mats.package(
        mat_files_dict=job["mat_files"],
        mat_list=room_geo.mat_str,
        read_folder=job["mat_folder"],
    )

    cart_grid = CartGrid(
        h=sim_consts.h, offset=3.5, bmin=room_geo.bmin, bmax=room_geo.bmax, fcc=fcc
    )
    cart_grid.print_stats()
    cart_grid.save(str(out_dir))

    # A placeholder source and receiver, purely so the rotate and sort passes
    # below have the file they insist on rewriting. Deleted afterwards.
    sim_comms = SimComms(save_folder=str(out_dir))
    sim_comms.prepare_source_pts(room_geo.Sxyz[0])
    sim_comms.prepare_receiver_pts(room_geo.Rxyz)
    sim_comms.prepare_source_signals(job.get("probe_duration", 0.001), sig_type="impulse")
    sim_comms.save(compress=compress)

    vox_grid = VoxGrid(room_geo, cart_grid, Nvox_est=job.get("nvox_est"), Nh=job.get("nh"))
    timed("vox_grid_fill_s", lambda: vox_grid.fill(Nprocs=nprocs))
    vox_grid.print_stats()

    vox_scene = VoxScene(room_geo, cart_grid, vox_grid, fcc=fcc)
    timed("vox_scene_adj_s", lambda: vox_scene.calc_adj(Nprocs=nprocs))
    vox_scene.check_adj_full()
    timed("vox_scene_save_s", lambda: vox_scene.save(str(out_dir), compress=compress))

    def _to_engine_space() -> None:
        rotate_sim_data(str(out_dir))
        if fcc:
            fold_fcc_sim_data(str(out_dir))
        sort_sim_data(str(out_dir))

    timed("to_engine_space_s", _to_engine_space)

    (out_dir / "comms_out.h5").unlink()

    report = {
        "timings": timings,
        "voxelisation_s": round(
            sum(timings[k] for k in ("vox_grid_fill_s", "vox_scene_adj_s", "vox_scene_save_s")),
            3,
        ),
        "h_m": float(sim_consts.h),
        "sample_rate_hz": float(sim_consts.SR),
        "grid_shape": [int(n) for n in cart_grid.Nxyz],
        "grid_points": int(np.prod(np.asarray(cart_grid.Nxyz, dtype=np.int64))),
        "boundary_nodes": int(vox_scene.bn_ixyz.size),
        "triangles": int(room_geo.tris.shape[0]),
        "bmin": [float(v) for v in room_geo.bmin],
        "bmax": [float(v) for v in room_geo.bmax],
        "nprocs": int(nprocs) if nprocs else None,
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    sys.stdout.write("\n@@REVERBERATE@@" + json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
