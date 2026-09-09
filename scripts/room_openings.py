"""Print every shared boundary of a scene, and say which rule decided it.

The numbers behind `DOOR_MAX_M` and `WALL_MAX_M` come from this: run it over
the scenes in play and look for the empty band each threshold sits in.

    PYTHONPATH=src python scripts/room_openings.py 102344403
"""

import argparse
from pathlib import Path

import numpy as np

from reverberate.geometry.apartment import (
    OUTDOOR_LABELS,
    WALL_SECTION_HEIGHT,
    load_stage,
    wall_footprint,
)
from reverberate.geometry.hssd_room import load_regions
from reverberate.geometry.rooms import HALLWAY_LABELS, everyday_rooms, shared_boundaries

ROOT = Path("/Users/gilles/Developer/reverberate/data/raw/hssd-hab")


def report(hssd_root: Path, scene_id: str) -> None:
    regions = load_regions(hssd_root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json")
    stage = load_stage(hssd_root, scene_id)
    floor = float(np.mean([region.floor_height for region in regions]))
    walls = wall_footprint(stage, floor + WALL_SECTION_HEIGHT)
    polygons = [region.polygon_xz.buffer(0) for region in regions]
    owner = {name: room for room in everyday_rooms(regions, walls) for name in room.regions}

    print(f"== {scene_id}")
    print(f"{'paire':40s} {'contact':>8s} {'passage':>8s} {'mur':>7s}   verdict")
    boundaries = shared_boundaries(polygons, walls)
    for (i, j), boundary in sorted(boundaries.items(), key=lambda kv: -kv[1].widest_m):
        pair = f"{regions[i].name} | {regions[j].name}"
        if owner[regions[i].name] is owner[regions[j].name]:
            verdict = "une seule piece"
        elif regions[i].label in OUTDOOR_LABELS or regions[j].label in OUTDOOR_LABELS:
            verdict = "dehors"
        elif regions[i].label in HALLWAY_LABELS or regions[j].label in HALLWAY_LABELS:
            verdict = "couloir, separe"
        elif boundary.widest_m <= 0:
            verdict = "mur plein"
        elif not boundary.dissolves:
            verdict = "porte" if boundary.widest_m <= 1.10 else "mur perce"
        else:
            verdict = "separe"
        print(
            f"{pair:40s} {boundary.contact_m:8.2f} {boundary.widest_m:8.2f} "
            f"{boundary.walled_m:7.2f}   {verdict}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--hssd-root", type=Path, default=ROOT)
    args = parser.parse_args()
    for scene in args.scenes:
        report(args.hssd_root, scene)
