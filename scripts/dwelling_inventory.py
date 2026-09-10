"""Every dwelling in the HSSD scene set, one row per storey.

A scene with two storeys is two dwellings' worth of geometry as far as the
solver is concerned -- the storeys are separate air volumes, joined only by a
stairwell the region annotations do not describe -- so each is listed on its
own as ``hssd_0001_1``, ``hssd_0001_2``, ground floor first, under this
project's own name for the scene. See
:mod:`reverberate.geometry.scene_ids`: the HSSD id is carried beside it in
``scene_id``, never appended to, because 135 of the 168 are already of the form
``105515175_173104107`` and a trailing ``_1`` on one of those reads as part of
the id rather than as a storey.

Rooms are counted by the four rules of :mod:`reverberate.geometry.rooms`, not
by `region_annotations`: what the dataset calls eleven rooms is often four.

    PYTHONPATH=src python scripts/dwelling_inventory.py --out data/interim
"""

import argparse
import csv
import json
import traceback
from collections import Counter
from pathlib import Path

from shapely.ops import unary_union

from reverberate.geometry.apartment import build_apartment
from reverberate.geometry.scene_ids import local_name

ROOT = Path("/Users/gilles/Developer/reverberate/data/raw/hssd-hab")

#: Labels that make a scene something other than somewhere people live. A scene
#: carrying one of these is still listed, and still flagged, rather than
#: dropped on a guess.
NON_RESIDENTIAL = frozenset({"classroom", "meetingroom/conferenceroom", "library", "bar"})

#: A storey with a bed on it is somewhere people sleep. Reported beside
#: ``residential`` rather than folded into it, because the ground floor of a
#: house has no bedroom and is still part of a dwelling.
SLEEPING = "bedroom"


def rows_for(hssd_root: Path, scene_id: str) -> list[dict[str, object]]:
    storeys = build_apartment(hssd_root, scene_id)
    storeys.sort(key=lambda storey: storey.floor_height)
    rows = []
    for index, storey in enumerate(storeys, start=1):
        rooms = storey.everyday
        labels = Counter(region.label for region in storey.rooms)
        area = float(unary_union([room.polygon for room in rooms]).area)
        height = storey.ceiling_height - storey.floor_height
        rows.append(
            {
                "dwelling": local_name(scene_id, index),
                "scene_id": scene_id,
                "hssd_storey": scene_id if len(storeys) == 1 else f"{scene_id}#{index}",
                "storey": index,
                "storeys": len(storeys),
                "floor_height_m": round(storey.floor_height, 2),
                "ceiling_height_m": round(height, 2),
                "interior_area_m2": round(area, 1),
                "interior_volume_m3": round(area * height, 1),
                "rooms": len(rooms),
                "regions": len(storey.rooms),
                "bedrooms": labels["bedroom"],
                "largest_room_m2": round(max((r.area_m2 for r in rooms), default=0.0), 1),
                "largest_room": rooms[0].name if rooms else "",
                "residential": not (NON_RESIDENTIAL & set(labels)),
                "sleeps": labels[SLEEPING] > 0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("scenes", nargs="*")
    args = parser.parse_args()

    scenes = args.scenes or sorted(
        path.name.split(".")[0]
        for path in (args.hssd_root / "semantics" / "scenes").glob("*.semantic_config.json")
    )
    rows: list[dict[str, object]] = []
    failed: dict[str, str] = {}
    for number, scene in enumerate(scenes, start=1):
        try:
            rows.extend(rows_for(args.hssd_root, scene))
        except Exception:  # noqa: BLE001 - one bad scene must not stop the census
            failed[scene] = traceback.format_exc(limit=1).strip().splitlines()[-1]
        print(f"{number:4d}/{len(scenes)}  {scene}  ({len(rows)} rows, {len(failed)} failed)")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "dwellings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "dwellings_failed.json").write_text(json.dumps(failed, indent=2))
    print(f"{len(rows)} dwellings from {len(scenes)} scenes, {len(failed)} scenes failed")


if __name__ == "__main__":
    main()
