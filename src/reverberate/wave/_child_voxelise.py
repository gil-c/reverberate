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


def _slabbed_adj(
    out_dir: Path,
    room_geo: Any,
    cart_grid: Any,
    vox_grid: Any,
    fcc: bool,
    nprocs: int | None,
    slabs: int,
) -> int:
    """``calc_adj`` a slab of voxels at a time, appending to ``vox_out.h5``.

    Why this is exact rather than approximate, in three facts about the code it
    drives, each checked in ``tests/test_slabbed.py``:

    1. **A voxel emits only its core.** ``vox_scene.py`` sets
       ``in_mask[1:-1,1:-1,1:-1]`` and then clears ``vox_bp`` outside it, so the
       one-cell halo is *read* -- which is what makes the adjacency at the core's
       edge correct -- and never *emitted*. The cores tile the grid with no
       overlap and no gap, which is why upstream can assert
       ``np.unique(bn_ixyz).size == bn_ixyz.size``. A slab is a whole number of
       voxels, so it inherits that property: no node is produced twice and none
       is missed.
    2. **Nothing written to the file is a global reduction.** ``adj_bn`` comes
       from ray hits on the node's own six legs; ``saf_bn`` from those hits and
       the triangle normal; ``mat_bn`` from the nearest triangle's material.
       ``calc_adj`` does compute ``mat_approx_sa`` by summing over every node,
       but that figure is only printed -- it never feeds back into what is
       saved. So a node's row does not depend on which slab it was computed in.
    3. **The engine's flat index is outermost-axis-major.** A slab cut along the
       axis ``rotate_sim_data`` puts first is therefore a contiguous range of
       flat indices, so sorting inside each slab and concatenating in order
       gives exactly the globally sorted array -- no merge, no seam.

    The rotate and sort passes are done here, per slab, rather than by
    ``rotate_sim_data``/``sort_sim_data`` afterwards: those read the whole of
    ``adj_bn`` and ``bn_ixyz`` into memory, which is 15 GB at 16 kHz on this
    flat and would put back exactly the ceiling the slabbing exists to remove.

    ``check_adj_full`` is skipped. It memory-maps one byte per grid point --
    151 GB for that same scene -- and it is a verification of the whole grid,
    which a slab cannot do a piece of. The report says ``slabs`` so a reader
    knows this entry did not have it.

    Returns the total number of boundary nodes written.
    """
    import h5py
    import numpy as np
    from common.myfuncs import ind2sub3d
    from voxelizer.vox_scene import VoxScene

    grid = np.asarray(cart_grid.Nxyz, dtype=np.int64)
    # The same rule rotate_sim_data uses, so the file this writes is in the
    # space the engine expects: axes in descending extent, biggest first.
    order = np.argsort(grid)[::-1]
    rotated = grid[order]
    axes = [cart_grid.xv, cart_grid.yv, cart_grid.zv]

    # Column permutation for adj_bn, lifted from rotate_sim_data: the six
    # neighbour directions have to follow the axes they point along.
    steps = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    columns = np.argsort(
        [int(np.flatnonzero(np.all(step[order] == steps, axis=-1))[0]) for step in steps]
    )

    # Cut along the axis that ends up outermost, which is what makes a slab a
    # contiguous range of engine indices.
    axis = int(order[0])
    nonempty = list(vox_grid.nonempty_idx)
    starts = np.array([vox_grid.voxels[i].ixyz_start[axis] for i in nonempty], dtype=np.int64)
    edges = np.linspace(0, int(grid[axis]), slabs + 1)[1:-1]
    groups: list[list[int]] = [[] for _ in range(slabs)]
    for voxel, start in zip(nonempty, starts, strict=True):
        groups[int(np.searchsorted(edges, start, side="right"))].append(voxel)

    handle = h5py.File(out_dir / "vox_out.h5", "w")
    datasets = {
        "bn_ixyz": handle.create_dataset("bn_ixyz", (0,), maxshape=(None,), dtype=np.int64),
        "adj_bn": handle.create_dataset(
            "adj_bn", (0, 6), maxshape=(None, 6), dtype=bool, chunks=(1 << 16, 6)
        ),
        "mat_bn": handle.create_dataset("mat_bn", (0,), maxshape=(None,), dtype=np.int8),
        "saf_bn": handle.create_dataset("saf_bn", (0,), maxshape=(None,), dtype=np.float64),
    }
    total = 0
    for number, group in enumerate(groups):
        print(f"--SLAB {number + 1}/{slabs}: {len(group)} non-empty voxels", flush=True)
        if not group:
            continue
        vox_grid.nonempty_idx = group
        scene = VoxScene(room_geo, cart_grid, vox_grid, fcc=fcc)
        scene.calc_adj(Nprocs=nprocs)

        subs = ind2sub3d(scene.bn_ixyz, *grid)
        moved = (subs[order[0]] * rotated[1] + subs[order[1]]) * rotated[2] + subs[order[2]]
        keep = np.argsort(moved)
        rows = {
            "bn_ixyz": moved[keep],
            "adj_bn": scene.adj_bn[:, columns][keep],
            "mat_bn": scene.mat_bn[keep],
            "saf_bn": scene.saf_bn[keep],
        }
        written = int(keep.size)
        for name, data in rows.items():
            dataset = datasets[name]
            dataset.resize(total + written, axis=0)
            dataset[total : total + written] = data
        total += written
        # Let the slab go before the next one is built, which is the whole
        # point: peak memory is one slab and not the scene.
        del scene, subs, moved, keep, rows

    for name, values in zip("xyz", [axes[i] for i in order], strict=True):
        handle.create_dataset(f"{name}v", data=values)
    handle.create_dataset("h", data=np.float64(cart_grid.h))
    for name, size in zip(("Nx", "Ny", "Nz"), rotated, strict=True):
        handle.create_dataset(name, data=np.int64(size))
    handle.create_dataset("Nb", data=np.int64(total))
    handle.close()
    return total


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

    slabs = int(job.get("slabs") or 1)
    if slabs > 1:
        boundary_nodes = timed(
            "vox_scene_adj_s",
            lambda: _slabbed_adj(out_dir, room_geo, cart_grid, vox_grid, fcc, nprocs, slabs),
        )
        timings["vox_scene_save_s"] = 0.0
        timings["to_engine_space_s"] = 0.0
        (out_dir / "comms_out.h5").unlink()
    else:
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
        boundary_nodes = int(vox_scene.bn_ixyz.size)

    report = {
        "timings": timings,
        "slabs": slabs,
        "voxelisation_s": round(
            sum(timings[k] for k in ("vox_grid_fill_s", "vox_scene_adj_s", "vox_scene_save_s")),
            3,
        ),
        "h_m": float(sim_consts.h),
        "sample_rate_hz": float(sim_consts.SR),
        "grid_shape": [int(n) for n in cart_grid.Nxyz],
        "grid_points": int(np.prod(np.asarray(cart_grid.Nxyz, dtype=np.int64))),
        "boundary_nodes": boundary_nodes,
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
