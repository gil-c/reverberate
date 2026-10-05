"""Where a site's time goes on a card, launch against host: the pieces, timed one by one.

    PYTHONPATH=src python scripts/profile_rays_site.py --bundle RUN/bundle/trace \
        [--sites 8] [--rays 100000] [--precision double] [--out site.json]

A site of the tail, through the tree, as ``mirror.tails.histograms`` makes it:
the launch's arrays (the receivers' tree, the source), the launch itself, the
counts brought home, turned to energies, and kept a (site, cell) at a time.
Each piece is timed over ``--sites`` sites after one that pays the upload and
the compilation, and the whole of ``histograms`` beside their sum.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.compute import Devices
from reverberate.mirror import tracer
from reverberate.mirror.engine import histogram_tree_on_device, tree_upload
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import Histogram
from reverberate.mirror.tails import TailCache, histograms, tail_key
from reverberate.trace.assets import MirrorAssets
from reverberate.trace.plan import tail_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--rays", type=int, default=100_000)
    parser.add_argument("--precision", default="double")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    import cupy

    assets = MirrorAssets.load(args.bundle / "mirror")
    settings: MirrorSettings = replace(
        assets.settings,
        rays=replace(assets.settings.rays, rays=args.rays, precision=args.precision),
    )
    rays = settings.traced_rays()
    with np.load(args.bundle / "plan.npz") as plan:
        cells = np.asarray(plan["cells"], dtype=float)
        cells = cells[tail_cells(cells, plan["kind"])]
    positions = np.load(args.bundle / "positions.npy").reshape(-1, 3)
    take = np.unique(np.rint(np.linspace(0, positions.shape[0] - 1, args.sites + 1)).astype(int))
    sites = positions[take]
    spent: dict[str, float] = {}

    def clocked(name: str, started: float) -> None:
        spent[name] = spent.get(name, 0.0) + time.time() - started

    with tempfile.TemporaryDirectory() as scratch:
        cache = TailCache(Path(scratch) / "tails", keep=4)
        from reverberate.mirror.tails import shared_scene

        shared = shared_scene(cache, assets.catalogue, settings)
        scene = shared["scene"]
        started = time.time()
        held = tracer.prepare(scene)
        build_s = time.time() - started
        started = time.time()
        on_card = tree_upload(scene, held, device=0, single=args.precision == "single")
        cupy.cuda.Stream.null.synchronize()
        upload_s = time.time() - started
        shared["uploads"]["tree"] = held
        shared["uploads"][("tree", 0, args.precision == "single")] = on_card
        for number, site in enumerate(sites):
            first = number == 0
            stats: dict[str, int] = {}
            t0 = time.time()
            counted = histogram_tree_on_device(
                scene,
                site,
                cells,
                rays,
                held,
                on_card,
                device=0,
                ray_start=0,
                ray_count=rays.rays,
                stats=stats,
            )
            if first:
                continue
            clocked("launch_call", t0)
            spent["kernel"] = spent.get("kernel", 0.0) + stats["kernel_us"] / 1e6
            t0 = time.time()
            tracer.shot_arrays(scene, held, site, cells, rays)
            clocked("launch_arrays", t0)
            t0 = time.time()
            histogram = Histogram.from_counts(
                *counted, bin_s=rays.bin_s, bands_hz=scene.materials.bands_hz, order=3, rays=1
            )
            clocked("counts_to_energies", t0)
            t0 = time.time()
            keys = [tail_key(shared["key"], site, cell, rays) for cell in cells]
            clocked("keys", t0)
            t0 = time.time()
            cache.put(keys, histogram)
            clocked("entries_written", t0)
        # The whole, as a trace's job makes it, on sites not yet kept.
        other = positions[np.clip(take[1:] + 1, 0, positions.shape[0] - 1)]
        t0 = time.time()
        import os

        os.environ[tracer.STRUCTURE_VARIABLE] = "tree"
        histograms(assets.catalogue, settings, other, cells, devices=Devices((0,), 1), cache=cache)
        whole_s = (time.time() - t0) / len(other)
    count = len(sites) - 1
    report: dict[str, Any] = {
        "precision": args.precision,
        "rays": args.rays,
        "cells": int(cells.shape[0]),
        "sites": count,
        "tree_build_s": round(build_s, 2),
        "upload_s": round(upload_s, 3),
        "a_site_s": {name: round(value / count, 4) for name, value in spent.items()},
        "a_site_through_histograms_s": round(whole_s, 4),
    }
    print(json.dumps(report, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
