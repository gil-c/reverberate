"""The moving mirror on a small synthetic dwelling, on the host: time a step, and the tail's error.

    PYTHONPATH=src python scripts/bench_moving_mirror.py [--steps 400] [--rays 1000] [--out x.json]

Two things are measured, both on numpy and on one core, in about a minute.

1. **Time per step-pair.** A source and a listener walk through two rooms
   joined by a doorway. The batched trace (``mirror.moving.trace_early``)
   is timed over the whole walk, cold (its candidate trees and short lists
   are grown on the way) and warm; the present pipeline's work for one pair
   (``ism.grow_tree`` then ``ism.paths_for``, the numpy twin of the card's
   kernel) is timed on a few of the steps. The paths of those steps are
   checked equal.

2. **The tail's interpolation error.** Rays are traced from two source
   positions 0.80 m apart and from the point half way; the level of the mean
   of the two histograms is compared with the level traced at the middle,
   band by band, at every cell; the same for a head half way between two
   cells 0.80 m apart. The worst band is reported, and beside it what two
   traces of the same position with different seeds differ by, which is
   the floor a figure here can be read against.

The scene is synthetic and small, the host is not a card: the figures say
how the two ways scale against each other, and
``docs/adr/0016-appendix-moving-mirror-cost.md`` says what follows for the
card and under which assumptions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import Devices
from reverberate.mirror.geometry import DerivedScene, Facet, GeometryRules, MaterialTable
from reverberate.mirror.ism import IsmSettings, grow_tree, occluder_grid, paths_for
from reverberate.mirror.moving import MovingSettings, prepare, trace_early
from reverberate.mirror.moving_onset import onset_field
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.tails import TailCache, histograms, interpolation_error_db

#: The dwelling: two rooms side by side along x, a doorway in the wall between them.
SIZE = np.array([8.0, 2.6, 5.0])
WALL_X = 4.0
DOOR = (2.0, 3.0, 2.1)


def _quad(corners: np.ndarray) -> np.ndarray:
    return np.asarray([corners[[0, 1, 2]], corners[[0, 2, 3]]], dtype=float)


def dwelling() -> DerivedScene:
    """Two rooms, a doorway, and a dozen panels of furniture that reflect on both sides."""
    quads: list[tuple[np.ndarray, np.ndarray, str, int]] = []
    sx, sy, sz = SIZE
    shell = [
        ([1.0, 0, 0], [[0, 0, 0], [0, 0, sz], [0, sy, sz], [0, sy, 0]], "shell_wall"),
        ([-1.0, 0, 0], [[sx, 0, 0], [sx, sy, 0], [sx, sy, sz], [sx, 0, sz]], "shell_wall"),
        ([0, 1.0, 0], [[0, 0, 0], [sx, 0, 0], [sx, 0, sz], [0, 0, sz]], "shell_floor"),
        ([0, -1.0, 0], [[0, sy, 0], [0, sy, sz], [sx, sy, sz], [sx, sy, 0]], "shell_ceiling"),
        ([0, 0, 1.0], [[0, 0, 0], [0, sy, 0], [sx, sy, 0], [sx, 0, 0]], "shell_wall"),
        ([0, 0, -1.0], [[0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]], "shell_wall"),
    ]
    for normal, corners, kind in shell:
        quads.append((np.asarray(normal), np.asarray(corners, dtype=float), kind, 2))
    z0, z1, top = DOOR
    # The partition, in three pieces round the doorway, reflecting on both sides.
    for (a0, a1), (b0, b1) in (
        ((0.0, sy), (0.0, z0)),
        ((0.0, sy), (z1, sz)),
        ((top, sy), (z0, z1)),
    ):
        corners = [[WALL_X, a0, b0], [WALL_X, a1, b0], [WALL_X, a1, b1], [WALL_X, a0, b1]]
        quads.append((np.array([1.0, 0, 0]), np.asarray(corners), "shell_wall", 3))
    rng = np.random.default_rng(7)
    for _ in range(12):
        centre = rng.uniform([0.6, 0.5, 0.6], [sx - 0.6, 1.6, sz - 0.6])
        if abs(centre[0] - WALL_X) < 0.7:
            centre[0] += 1.4
        w, d = rng.uniform(0.4, 0.9, size=2)
        if rng.random() < 0.5:  # a table top
            corners = centre + np.array([[-w, 0, -d], [w, 0, -d], [w, 0, d], [-w, 0, d]])
            normal = np.array([0.0, 1.0, 0.0])
        else:  # a cupboard's side
            corners = centre + np.array([[0, -w, -d], [0, w, -d], [0, w, d], [0, -w, d]])
            normal = np.array([1.0, 0.0, 0.0])
        quads.append((normal, corners, "furniture", 3))
    triangles = np.concatenate([_quad(corners) for _, corners, _, _ in quads])
    facets = tuple(
        Facet(
            label=1 if kind == "furniture" else 0,
            normal=normal,
            offset=float(normal @ corners[0]),
            area=float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[3] - corners[0]))),
            triangles=np.arange(2 * k, 2 * k + 2, dtype=np.int32),
            kind=kind,
            sides=sides,
        )
        for k, (normal, corners, kind, sides) in enumerate(quads)
    )
    labels = np.repeat(np.asarray([f.label for f in facets], dtype=np.int16), 2)
    materials = MaterialTable(
        ("shell", "furniture"),
        # Duller as the band rises, so the bands do not all read alike.
        np.stack(
            [np.linspace(0.08, 0.20, len(OCTAVE_BANDS)), np.linspace(0.2, 0.5, len(OCTAVE_BANDS))]
        ),
        np.array([0.2, 0.4]),
    )
    return DerivedScene(
        labels=("shell", "furniture"),
        materials=materials,
        facets=facets,
        reflector_vertices=triangles,
        reflector_facet=np.repeat(np.arange(len(facets), dtype=np.int32), 2),
        occluder_vertices=triangles.copy(),
        occluder_label=labels,
        occluder_sides=np.repeat(np.asarray([f.sides for f in facets], dtype=np.int8), 2),
        rules=GeometryRules(),
        bmin=np.zeros(3),
        bmax=SIZE.copy(),
    )


def walks(steps: int) -> tuple[np.ndarray, np.ndarray]:
    """A source pacing its room and a listener going to the other room and back, 50 ms a step."""
    u = np.linspace(0.0, 1.0, steps)[:, None]
    source = np.array([1.0, 1.5, 1.0]) + np.abs(np.sin(2 * np.pi * u)) * np.array([2.2, 0.0, 2.6])
    there = np.sin(np.pi * u) ** 2
    listener = np.array([2.4, 1.6, 2.5]) + there * np.array([4.2, 0.0, 0.0])
    listener[:, 2] += 1.2 * np.sin(4 * np.pi * u[:, 0]) * there[:, 0]
    return source, listener


def time_the_trace(steps: int, compare: int) -> dict[str, Any]:
    catalogue = dwelling()
    settings = MirrorSettings(ism=IsmSettings(max_order=3, flutter_order=6))
    source, listener = walks(steps)
    region = (np.zeros(3) - 0.5, SIZE + 0.5)
    started = time.perf_counter()
    ms = prepare(catalogue, settings, MovingSettings())
    held = onset_field(catalogue, np.concatenate([source, listener]), sound_speed_m_s=343.2)
    prepared = time.perf_counter() - started
    started = time.perf_counter()
    table = trace_early(ms, source, listener, region=region, onsets=held)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    table = trace_early(ms, source, listener, region=region, onsets=held)
    warm = time.perf_counter() - started
    started = time.perf_counter()
    plain = trace_early(ms, source, listener, region=region)
    without_onsets = time.perf_counter() - started
    grid = occluder_grid(ms.scene)
    picks = np.linspace(0, steps - 1, compare).astype(int)
    grow = validate = 0.0
    images = []
    for step in picks:
        started = time.perf_counter()
        tree = grow_tree(ms.scene, source[step], ms.ism, region=region)
        grow += time.perf_counter() - started
        started = time.perf_counter()
        found = paths_for(ms.scene, tree, listener[step], ms.ism, grid=grid)
        validate += time.perf_counter() - started
        images.append(tree.count)
        mine, _ = plain.paths(int(step))
        if not np.array_equal(mine.sequence, found.sequence):
            raise SystemExit(f"step {step}: the batched trace and the twin disagree")
    per_step = np.diff(table.offsets)
    return {
        "scene": catalogue.summary(),
        "steps": steps,
        "record": table.record,
        "rows_per_step_median": float(np.median(per_step)),
        "images_per_tree_median": float(np.median(images)),
        "prepare_s": round(prepared, 3),
        "batched_ms_per_step_pair": {
            "cold_with_onsets": round(1e3 * cold / steps, 3),
            "warm_with_onsets": round(1e3 * warm / steps, 3),
            "warm_image_paths_only": round(1e3 * without_onsets / steps, 3),
        },
        "present_ms_per_pair": {
            "grow_tree": round(1e3 * grow / compare, 3),
            "paths_for": round(1e3 * validate / compare, 3),
            "pairs_timed": int(compare),
        },
        "speedup_validation": round((validate / compare) / (without_onsets / steps), 1),
    }


def tail_error(rays: int) -> dict[str, Any]:
    catalogue = dwelling()
    settings = MirrorSettings(
        rays=RaySettings(rays=rays, duration_s=0.4, bin_s=0.002, receiver_radius_m=0.3)
    )
    # Cells 0.80 m apart in both rooms; the odd ones are the heads half way.
    cells = np.array(
        [[x, 1.6, z] for z in (1.5, 3.5) for x in (1.2, 1.6, 2.0, 2.4, 2.8, 5.2, 5.6, 6.0)]
    )
    low, high = np.array([1.0, 1.5, 2.2]), np.array([1.8, 1.5, 2.2])
    middle = 0.5 * (low + high)
    cache = TailCache()
    started = time.perf_counter()
    at_low, at_high, at_middle = histograms(
        catalogue,
        settings,
        np.stack([low, high, middle]),
        cells,
        devices=Devices.host(1),
        cache=cache,
    )
    again = histograms(
        catalogue,
        replace(settings, rays=replace(settings.rays, seed=1)),
        middle[None, :],
        cells,
        devices=Devices.host(1),
        cache=cache,
    )[0]
    seconds = time.perf_counter() - started
    from_bin = int(round(0.02 / settings.rays.bin_s))
    source_side = np.stack(
        [
            interpolation_error_db(
                at_low.energy[c], at_high.energy[c], at_middle.energy[c], 0.5, from_bin=from_bin
            )
            for c in range(len(cells))
        ]
    )
    floor = np.stack(
        [
            interpolation_error_db(
                again.energy[c], again.energy[c], at_middle.energy[c], 0.5, from_bin=from_bin
            )
            for c in range(len(cells))
        ]
    )
    # Heads at cells 1 and 3 and their like, read from the cells 0.40 m either side.
    heads = [k for k in range(len(cells)) if k % 8 in (1, 3, 6)]
    listener_side = np.stack(
        [
            interpolation_error_db(
                at_middle.energy[c - 1],
                at_middle.energy[c + 1],
                at_middle.energy[c],
                0.5,
                from_bin=from_bin,
            )
            for c in heads
        ]
    )

    def summary(errors: np.ndarray) -> dict[str, Any]:
        worst_band = int(np.argmax(np.abs(errors).max(axis=0)))
        return {
            "worst_band_hz": int(OCTAVE_BANDS[worst_band]),
            "worst_db": round(float(np.abs(errors).max()), 2),
            "worst_per_band_db": [round(float(v), 2) for v in np.abs(errors).max(axis=0)],
            "median_abs_db": round(float(np.median(np.abs(errors))), 2),
        }

    return {
        "rays": rays,
        "duration_s": settings.rays.duration_s,
        "cells": int(len(cells)),
        "level_from_s": 0.02,
        "trace_seconds": round(seconds, 1),
        "source_half_way_between_two_0_80_m_apart": summary(source_side),
        "head_half_way_between_two_cells_0_80_m_apart": summary(listener_side),
        "same_position_another_seed": summary(floor),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--compare", type=int, default=12, help="pairs the present way is timed on")
    parser.add_argument("--rays", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = {"trace": time_the_trace(args.steps, args.compare), "tail": tail_error(args.rays)}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
