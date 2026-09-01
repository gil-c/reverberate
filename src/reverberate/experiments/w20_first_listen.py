"""W20, the first listen: twelve impulse responses from one furnished bedroom.

Nothing in this project has ever been *heard*. Every result so far is a number
in a record: a decay curve, a bit exactness check, a cost per point update. This
item exists because a wave solver that has never been listened to is a solver
whose failures are invisible, and because the roadmap's own history is a list of
defects found only by looking at what was actually fed in.

**Why two sources and six receivers.** The solver's cost is set by the grid and
the number of time steps, not by the number of receivers: a receiver is eight
interpolation weights read out of a field that is being computed anyway. A
second source is a second field, so a second run. Twelve responses therefore
cost two runs, and the shape of the experiment is a cost fact rather than a
preference.

**Why an existing voxelisation.** The cache already holds this room at a 4 kHz
working point, keyed on its geometry, its materials and its grid. Re-voxelising
would change nothing and cost minutes, and the point of W8's split was exactly
that a placement is cheap once the grid exists. The bounds are checked against
the cache rather than assumed to match.

**What this is not.** Two bare points in a room are not a pair of ears. There is
no head, no torso, no pinna and no interaural anything, so the result is
monophonic pressure at two locations and every artefact this writes says so.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point

from reverberate.experiments.engine import Engine, sim_consts, write_record
from reverberate.experiments.run import entry_from_key, execute
from reverberate.experiments.small_objects import isolated_storey
from reverberate.geometry.apartment import build_apartment, instances_on_storey
from reverberate.geometry.hssd_room import load_object_instances
from reverberate.geometry.placement import (
    PlacedGroup,
    furniture_footprints,
    sample_group,
    sampling_area,
)
from reverberate.wave.comms import write_comms
from reverberate.wave.voxelise import cache_root

#: The room the pipeline has already exported, voxelised and costed. Named here
#: rather than passed by default from a shell history nobody kept.
SCENE_ID = "102344022"
ROOM = "bedroom.001"

#: Measured on W20's own first source: 7.57e7 points x 109228 steps, so
#: 8.27e12 point updates, in about 5880 s of local CPU on ten cores. That is
#: 1.4e9 point updates per second, or about 1.4e8 per core. Used only to decide
#: whether to rent, never reported as a result.
#:
#: This replaces 2.6e10, which was read off the B1 sweep's ``engine_s`` column
#: and was wrong by a factor of about thirty seven. B1's own row for this exact
#: grid, 7.57e7 points and 43692 steps in 90.8 s, implies 3.6e10 updates per
#: second, or 3.6e9 per core, roughly twenty five times what this machine's
#: cores actually retire. Those rows are not local CPU measurements, whatever
#: their column name says. The estimate they produced said 636 s and the run
#: took 5880 s, so the rental decision was taken on a number that was not a
#: measurement of the thing it named. Reported here rather than quietly
#: corrected, because B1's conclusions rest on it.
LOCAL_UPDATES_PER_SECOND = 1.4e9

#: The roadmap's measured A100 figure, for the same decision.
GPU_UPDATES_PER_SECOND = 6.7e10

#: Past this, the working agreement says announce an instance type and an hourly
#: rate before renting rather than letting a laptop grind.
RENTAL_TRIGGER_S = 15 * 60


@dataclass(frozen=True)
class Cost:
    """What a run will cost, before it is started rather than after."""

    grid_points: int
    steps: int
    #: How many solver runs the figures below cover. One run per source, so a
    #: two source experiment is twice the updates of a one source one. It is a
    #: field rather than an implicit one because the times were reported for
    #: the whole experiment while the update count was reported for a single
    #: source, which made the two disagree by exactly this factor.
    sources: int
    local_s: float
    gpu_s: float

    @property
    def updates(self) -> float:
        return float(self.grid_points) * float(self.steps) * float(self.sources)

    def record(self) -> dict[str, Any]:
        return {
            "grid_points": self.grid_points,
            "steps": self.steps,
            "sources": self.sources,
            "point_updates": self.updates,
            "point_updates_per_source": float(self.grid_points) * float(self.steps),
            "estimated_local_s": round(self.local_s, 1),
            "estimated_gpu_s": round(self.gpu_s, 1),
            "rental_trigger_s": RENTAL_TRIGGER_S,
            "would_rent": self.local_s > RENTAL_TRIGGER_S,
        }


def estimate(entry_path: Path, duration_s: float, sources: int) -> Cost:
    """Point updates and wall time for the whole experiment, both machines."""
    manifest = json.loads((entry_path / "manifest.json").read_text())
    points = int(manifest["grid_points"])
    steps = int(round(duration_s * float(manifest["sample_rate_hz"])))
    updates = float(points) * steps * sources
    return Cost(
        grid_points=points,
        steps=steps,
        sources=sources,
        local_s=updates / LOCAL_UPDATES_PER_SECOND,
        gpu_s=updates / GPU_UPDATES_PER_SECOND,
    )


def place(
    hssd_root: Path, entry_path: Path, seed: int, sources: int, receivers: int
) -> PlacedGroup:
    """Draw the placements, then check them against the grid that will be used.

    The check is the point. A placement is sampled from the *architecture*, and
    the solver runs on a *voxelisation* whose bounds were fixed when it was
    built. If the two ever disagree, a position lands outside the grid and the
    interpolation reads whatever is at the edge, which is a silent wrong answer
    rather than a crash.
    """
    storeys = build_apartment(hssd_root, SCENE_ID)
    storey = storeys[0]
    instances = instances_on_storey(
        load_object_instances(hssd_root / "scenes" / f"{SCENE_ID}.scene_instance.json"),
        storey,
        storeys,
    )
    room_storey = isolated_storey(storey, ROOM)
    outline = room_storey.walkable.buffer(0)
    inside = [
        instance
        for instance in instances
        if outline.contains(Point(float(instance.translation[0]), float(instance.translation[2])))
    ]
    footprints = furniture_footprints(hssd_root, inside, floor_height=room_storey.floor_height)
    area = sampling_area(room_storey, footprints)
    group = sample_group(
        room_storey,
        np.random.default_rng(seed),
        area,
        room=ROOM,
        sources=sources,
        receivers=receivers,
        seed=seed,
    )
    _check_inside_grid(group, entry_path)
    return group


def _check_inside_grid(group: PlacedGroup, entry_path: Path, margin_cells: float = 2.0) -> None:
    manifest = json.loads((entry_path / "manifest.json").read_text())
    low = np.asarray(manifest["bmin"], dtype=float)
    high = np.asarray(manifest["bmax"], dtype=float)
    margin = margin_cells * float(manifest["h_m"])
    for kind, placements in (("source", group.sources), ("receiver", group.receivers)):
        for index, placement in enumerate(placements):
            position = np.asarray(placement.position, dtype=float)
            if np.any(position < low + margin) or np.any(position > high - margin):
                raise ValueError(
                    f"{kind} {index} at {position.round(3).tolist()} is outside the voxelised "
                    f"bounds {low.round(3).tolist()}..{high.round(3).tolist()} "
                    f"with a {margin_cells} cell margin"
                )


def prepare_runs(
    entry_path: Path,
    group: PlacedGroup,
    out_dir: Path,
    duration_s: float,
    *,
    double_precision: bool,
) -> list[Path]:
    """One comms file per source, all sharing the six receivers.

    They are written to ``comms/`` rather than into the run directory on
    purpose. The engine deletes every input but ``sim_consts.h5`` once it has
    finished, and the comms file is the only record of two things the response
    cannot be read without: the eight interpolation weights of each receiver,
    and whether the source was differentiated. Written into the run directory
    it would be deleted as the engine's own input; written beside it, the
    engine deletes only its copy.
    """
    receivers = np.vstack([np.asarray(r.position, dtype=float) for r in group.receivers])
    comms_dir = out_dir / "comms"
    comms_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, source in enumerate(group.sources):
        (out_dir / f"source{index}").mkdir(parents=True, exist_ok=True)
        paths.append(
            write_comms(
                entry_path,
                np.asarray(source.position, dtype=float),
                receivers,
                duration_s,
                diff_source=not double_precision,
                out_path=comms_dir / f"source{index}.h5",
            )
        )
    return paths


def solve(
    entry_path: Path,
    comms_paths: list[Path],
    out_dir: Path,
    *,
    engine: Engine,
    double_precision: bool,
) -> list[dict[str, Any]]:
    """Run the solver once per source, in the source's own directory."""
    records = []
    for index, comms in enumerate(comms_paths):
        run_dir = out_dir / f"source{index}"
        files = [
            entry_path / "sim_consts.h5",
            entry_path / "vox_out.h5",
            comms,
            entry_path / "sim_mats.h5",
        ]
        record = execute(files, run_dir, engine=engine, double_precision=double_precision)
        record["source_index"] = index
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--cache-key", required=True, help="voxelisation the runs read")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--receivers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20250101)
    parser.add_argument("--engine", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--double", action="store_true", help="double precision engine")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="place, cost and stop, so the rental decision is taken with numbers",
    )
    args = parser.parse_args(argv)

    entry = entry_from_key(args.cache_key)
    cost = estimate(entry.path, args.duration, args.sources)
    group = place(args.hssd_root, entry.path, args.seed, args.sources, args.receivers)

    args.out.mkdir(parents=True, exist_ok=True)
    write_record(
        args.out,
        "plan.json",
        {
            "scene_id": SCENE_ID,
            "room": ROOM,
            "cache_key": args.cache_key,
            "cache_root": str(cache_root()),
            "duration_s": args.duration,
            "sample_rate_hz": sim_consts(entry.path).sample_rate,
            "responses": group.response_count,
            "binaural": False,
            "binaural_note": "two bare points in a room are not a pair of ears",
            "placement": group.record(),
            "cost": cost.record(),
        },
    )
    if args.plan_only:
        return 0

    comms = prepare_runs(entry.path, group, args.out, args.duration, double_precision=args.double)
    records = solve(entry.path, comms, args.out, engine=args.engine, double_precision=args.double)
    write_record(args.out, "solve.json", {"runs": records})
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
