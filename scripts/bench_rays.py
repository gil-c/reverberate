"""The rays of the tail on a real dwelling: what a site costs, and that the tail is the same.

    PYTHONPATH=src python scripts/bench_rays.py --bundle RUN/bundle/trace \
        [--sites 20] [--rays 100000,1000000] [--out rays.json] [--cpu] [--no-grid]

``--bundle`` is a trace's bundle: its ``mirror`` directory (the scene and its
calibration), ``plan.npz`` (the cells, of which the tail's are taken as a run
takes them) and ``positions.npy`` (source positions of the dwelling, of which
``--sites`` are taken evenly). Nothing is written but ``--out``.

Three ways of casting the same rays are measured on the cards of the machine
(or, with ``--cpu``, on one core by the C text; the grid's twin is then left
out, being a Python loop):

- ``grid``: the present kernel, a uniform grid in double precision;
- ``tree``: the hierarchy of boxes, the triangle test in double precision;
- ``single``: the hierarchy, everything in single precision.

**Cost.** For each count of ``--rays``: seconds a site (the whole call, and the
launches alone), segments a second, nodes and triangle tests a segment, rays
that met no triangle per million. The first site of each way is cast once
before the clock starts: it pays the compilation and the upload.

**Proof**, at the first count of ``--rays`` on every site:

1. *identity*: the tree's histograms against the grid's, the same seed:
   crossings, energies and moments, the largest difference in counts of 2^40.
   The two are the same arithmetic on the same hits and must be equal (the
   moments to a count or two).
2. *statistics*: single precision against double at one seed, beside double
   at another seed against double: per band, the level's difference in bins of
   2 ms and in windows of 50 ms, as a root mean square over the bins that
   hold crossings and as the worst cell, band and window. Single precision is
   another draw of the same tail if it stands no farther from double than
   another seed does.
3. *escapes*: rays that met no triangle, per million, each way.

A machine's line is printed as soon as it is measured; the whole is the JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.compute import Devices, device_report
from reverberate.mirror import tracer
from reverberate.mirror.engine import histogram_on_devices
from reverberate.mirror.ism import occluder_grid
from reverberate.mirror.parameters import apply_parameters
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.trace.assets import MirrorAssets
from reverberate.trace.plan import tail_cells

#: A bin or a window is read where the reference holds this many crossings at least.
CROSSINGS = 20
WINDOW_BINS = 25


def say(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def cast(
    way: str,
    scene: Any,
    site: np.ndarray,
    cells: np.ndarray,
    rays: RaySettings,
    devices: Devices,
    held: dict[Any, Any],
    grid: Any,
) -> tuple[Histogram, dict[str, int], float]:
    """One site's histogram one way, its counts, and the seconds the whole call took."""
    settings = replace(rays, precision="single" if way == "single" else "double")
    stats: dict[str, int] = {}
    started = time.time()
    histogram = histogram_on_devices(
        scene,
        site,
        cells,
        settings,
        devices=devices,
        grid=grid if way == "grid" else None,
        held=held,
        structure="grid" if way == "grid" else "tree",
        stats=stats,
    )
    return histogram, stats, time.time() - started


def levels_db(histogram: Histogram, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, window and band: the level in dB, and the crossings the window holds."""
    bins = (histogram.energy.shape[1] // window) * window
    shape = histogram.energy.shape
    energy = histogram.energy[:, :bins].reshape(shape[0], bins // window, window, shape[2])
    hits = histogram.hits[:, :bins].reshape(shape[0], bins // window, window)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(energy.sum(axis=2)), hits.sum(axis=2)


def distance_db(one: Histogram, reference: Histogram) -> dict[str, float]:
    """How far ``one`` stands from ``reference``: bins of 2 ms and windows of 50 ms, dB."""
    out: dict[str, float] = {}
    for name, window in (("bins_2ms", 1), ("windows_50ms", WINDOW_BINS)):
        a, _ = levels_db(one, window)
        b, held = levels_db(reference, window)
        read = (held >= CROSSINGS)[:, :, None] & np.isfinite(a) & np.isfinite(b)
        gap = np.where(read, a - b, 0.0)
        count = max(int(read.sum()), 1)
        out[f"{name}_rms_db"] = round(float(np.sqrt((gap**2).sum() / count)), 4)
        out[f"{name}_worst_db"] = round(float(np.abs(gap).max(initial=0.0)), 3)
        # The worst band, every cell and window of it together: never the median.
        per_band = np.sqrt((gap**2).sum(axis=(0, 1)) / np.maximum(read.sum(axis=(0, 1)), 1))
        out[f"{name}_worst_band_rms_db"] = round(float(per_band.max(initial=0.0)), 4)
        out[f"{name}_read"] = int(read.sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sites", type=int, default=20)
    parser.add_argument("--rays", default="100000,1000000")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="one core of the host, the C text")
    parser.add_argument("--no-grid", action="store_true", help="leave the present kernel out")
    parser.add_argument("--grid-sites", type=int, default=3, help="sites the grid is timed on")
    args = parser.parse_args()
    counts = [int(v) for v in str(args.rays).split(",") if v]
    devices = Devices.host(1) if args.cpu else Devices.detect()
    if not devices.gpu and not tracer.available():
        raise SystemExit("no card and no C compiler: nothing to time here")
    ways = ["tree", "single"] if (args.no_grid or not devices.gpu) else ["grid", "tree", "single"]

    assets = MirrorAssets.load(args.bundle / "mirror")
    scene = apply_parameters(assets.catalogue, assets.settings.parameters)
    base = assets.settings.traced_rays()
    with np.load(args.bundle / "plan.npz") as plan:
        cells = np.asarray(plan["cells"], dtype=float)
        cells = cells[tail_cells(cells, plan["kind"])]
    positions = np.load(args.bundle / "positions.npy").reshape(-1, 3)
    take = np.linspace(0, positions.shape[0] - 1, min(args.sites, positions.shape[0]))
    sites = positions[np.unique(np.rint(take).astype(int))]
    report: dict[str, Any] = {
        "bundle": str(args.bundle),
        "devices": device_report() if devices.gpu else {"host": "one core, the C text"},
        "triangles": int(scene.occluder_vertices.shape[0]),
        "cells": int(cells.shape[0]),
        "sites": int(sites.shape[0]),
        "settings": base.record(),
        "cost": [],
    }
    say(f"{report['triangles']} triangles, {cells.shape[0]} tail cells, {sites.shape[0]} sites")

    held: dict[Any, Any] = {}
    started = time.time()
    held["tree"] = tracer.prepare(scene)
    report["tree"] = {
        "build_s": round(time.time() - started, 1),
        "nodes": held["tree"].tree.nodes,
        "depth": held["tree"].tree.depth,
        "node_bytes": held["tree"].tree.nodes * 64,
        "triangle_bytes_double": int(held["tree"].triangles.nbytes),
        "triangle_bytes_single": int(held["tree"].triangles.nbytes) // 2,
    }
    say(f"the tree: {report['tree']}")
    grid = None
    if "grid" in ways:
        started = time.time()
        grid = occluder_grid(scene, base.cell_m)
        report["grid_build_s"] = round(time.time() - started, 1)

    # ---- cost -------------------------------------------------------------------------------
    for rays_count in counts:
        rays = replace(base, rays=rays_count)
        for way in ways:
            timed = sites[: args.grid_sites] if way == "grid" else sites
            warm = time.time()
            cast(way, scene, timed[0], cells, rays, devices, held, grid)
            warm_s = time.time() - warm
            total: dict[str, int] = {}
            whole = 0.0
            for site in timed:
                _, stats, seconds = cast(way, scene, site, cells, rays, devices, held, grid)
                whole += seconds
                for name, value in stats.items():
                    total[name] = total.get(name, 0) + value
            segments = max(total.get("segments", 0), 1)
            kernel_s = total.get("kernel_us", 0) / 1e6
            line = {
                "way": way,
                "rays": rays_count,
                "sites": int(len(timed)),
                "first_site_s": round(warm_s, 2),
                "site_s": round(whole / len(timed), 4),
                "site_kernel_s": round(kernel_s / len(timed), 4),
                "segments_a_site": int(segments / len(timed)),
                "segments_a_second": round(segments / max(kernel_s, 1e-9)),
                "nodes_a_segment": round(total.get("nodes", 0) / segments, 2),
                "tests_a_segment": round(total.get("tests", 0) / segments, 2),
                "escapes_per_million_rays": round(
                    total.get("escapes", 0) / (rays_count * len(timed)) * 1e6, 2
                ),
            }
            report["cost"].append(line)
            say(f"cost: {line}")

    # ---- proof ------------------------------------------------------------------------------
    rays = replace(base, rays=counts[0])
    other = replace(rays, seed=rays.seed + 1)
    identity = {"hits": 0, "energy_counts": 0, "moment_counts": 0, "sites": 0}
    measures: dict[str, list[dict[str, float]]] = {"single": [], "seed": []}
    escapes = dict.fromkeys(ways, 0)
    for number, site in enumerate(sites):
        found = {}
        for way in ways:
            found[way], stats, _ = cast(way, scene, site, cells, rays, devices, held, grid)
            escapes[way] += stats.get("escapes", 0)
        again, _, _ = cast("tree", scene, site, cells, other, devices, held, grid)
        if "grid" in found:
            identity["sites"] += 1
            identity["hits"] = max(
                identity["hits"], int(np.abs(found["tree"].hits - found["grid"].hits).max())
            )
            for name, key in (("energy", "energy_counts"), ("moments", "moment_counts")):
                gap = np.abs(getattr(found["tree"], name) - getattr(found["grid"], name))
                identity[key] = max(identity[key], int(round(float(gap.max()) * 2.0**40)))
        measures["single"].append(distance_db(found["single"], found["tree"]))
        measures["seed"].append(distance_db(again, found["tree"]))
        say(
            f"proof: site {number} single against double {measures['single'][-1]};"
            f" another seed {measures['seed'][-1]}"
        )
    worst = {
        name: {key: max(row[key] for row in rows) for key in rows[0] if not key.endswith("_read")}
        for name, rows in measures.items()
    }
    report["proof"] = {
        "rays": counts[0],
        "identity_tree_against_grid": identity if "grid" in ways else "the grid was left out",
        "worst_site_single_against_double": worst["single"],
        "worst_site_another_seed_against_double": worst["seed"],
        "single_is_within_a_seed": all(
            worst["single"][key] <= worst["seed"][key] for key in worst["single"]
        ),
        "escapes_per_million_rays": {
            way: round(count / (counts[0] * len(sites)) * 1e6, 2) for way, count in escapes.items()
        },
        "per_site": measures,
    }
    say(f"proof: identity {report['proof']['identity_tree_against_grid']}")
    say(f"proof: single against double, worst site {worst['single']}")
    say(f"proof: another seed against double, worst site {worst['seed']}")
    say(f"proof: single within a seed: {report['proof']['single_is_within_a_seed']}")
    say(f"proof: escapes per million rays {report['proof']['escapes_per_million_rays']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        say(f"written {args.out}")


if __name__ == "__main__":
    main()
