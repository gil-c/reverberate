"""What the walk-through app reads from a solver run.

A run is a directory holding ``walk.json``:

.. code-block:: json

    {"dwelling": "hssd_0002",
     "sources": [{"id": "S1", "name": "voice", "position": [-6.05, 1.7, -3.89],
                  "directivity": "omni", "field": "field/S1.h5"}],
     "meshes": {"4000": "../w36_audit_4k/voxels"}}

``dwelling`` is the project's name for the scene (``scene_id``, the HSSD id,
is accepted instead); ``field`` names the source's impulse response field
(:mod:`reverberate.viz.field_payload`); ``meshes`` maps a band limit in hertz
to a tiered audit payload, both relative to the run. A run without
``walk.json`` is not offered, which keeps every earlier run out of the app.
Positions are in the scene frame, metres, y up.

:func:`write_synthetic_run` writes a run of a box room with mock fields, for
building and testing the app without a solve.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.geometry.scene_ids import local_name, scene_of
from reverberate.viz import field_payload
from reverberate.viz.vox_view import VoxelCloud, surface_of, write_quads

__all__ = [
    "WALK_MANIFEST",
    "WalkRun",
    "build_run",
    "check_run",
    "discover_walk_runs",
    "write_synthetic_run",
]

#: The file whose presence makes a directory a run the app can open.
WALK_MANIFEST = "walk.json"

#: What a source may be. Only the first is drawn today; the glyph module keys
#: on this string so a cardioid can be added without renaming anything.
DIRECTIVITIES = ("omni", "cardioid")


@dataclass(frozen=True)
class WalkRun:
    """A run the app can open: where it is and what its manifest says."""

    name: str
    path: Path
    scene_id: str
    #: The project's name for the scene or storey, `hssd_0011`; empty when the
    #: scene is not in the table.
    dwelling: str
    room: str
    sources: list[dict[str, Any]]
    #: Band limit in hertz, as text because it is a JSON key, to the payload
    #: directory relative to ``path``.
    meshes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> WalkRun:
        manifest = json.loads((path / WALK_MANIFEST).read_text())
        sources = list(manifest.get("sources") or [])
        for source in sources:
            if source.get("directivity", "omni") not in DIRECTIVITIES:
                raise ValueError(
                    f"{path.name}: source {source.get('id')!r} has directivity "
                    f"{source.get('directivity')!r}; known: {', '.join(DIRECTIVITIES)}"
                )
            if len(source.get("position", ())) != 3:
                raise ValueError(f"{path.name}: source {source.get('id')!r} needs [x, y, z]")
        if "dwelling" in manifest:
            dwelling = str(manifest["dwelling"])
            scene_id, _ = scene_of(dwelling)
        elif "scene_id" in manifest:
            scene_id = str(manifest["scene_id"])
            try:
                dwelling = local_name(scene_id)
            except KeyError:
                dwelling = ""
        else:
            raise ValueError(f"{path.name}: {WALK_MANIFEST} names neither dwelling nor scene_id")
        return cls(
            name=path.name,
            path=path,
            scene_id=scene_id,
            dwelling=dwelling,
            room=str(manifest.get("room", "")),
            sources=sources,
            meshes={str(k): str(v) for k, v in (manifest.get("meshes") or {}).items()},
        )


def discover_walk_runs(runs_root: Path) -> list[WalkRun]:
    """Every run under ``runs_root`` that carries a ``walk.json``, by name."""
    runs = []
    for path in sorted(Path(runs_root).iterdir()):
        if (path / WALK_MANIFEST).is_file():
            runs.append(WalkRun.read(path))
    return runs


def _mesh_record(run: WalkRun, fmax: str, relative: str, target: Path) -> dict[str, Any]:
    """Link one mesh payload into the site and say which rooms it holds.

    The room list is what lets the page grey out a band limit before the
    listener walks into a room it was never built for. The link rather than a
    copy: a payload is gigabytes and the site is a temporary directory.
    """
    source = (run.path / relative).resolve()
    rooms_file = source / "rooms.json"
    if not rooms_file.is_file():
        raise FileNotFoundError(f"{run.name}: mesh {fmax} Hz names {relative}, no rooms.json there")
    index = json.loads(rooms_file.read_text())
    link = target / "meshes" / fmax
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(source, target_is_directory=True)
    rooms = index.get("rooms") or []
    return {
        "url": f"meshes/{fmax}",
        "h_m": index.get("h_m"),
        "coarse_m": rooms[0]["coarse"]["cell_m"] if rooms else None,
        # Name and regions both: the page matches a payload room to a room of
        # the plan by either, so a payload built before a room was renamed
        # still lines up with the plan it is drawn in.
        "rooms": [
            {"name": room["name"], "regions": list(room.get("regions") or [room["name"]])}
            for room in rooms
        ],
        "quads": int(sum(room["fine"]["quads"] + room["coarse"]["quads"] for room in rooms)),
    }


def _field_record(run: WalkRun, source: dict[str, Any], target: Path) -> dict[str, Any] | None:
    """Index a source's field into the site: a small index and a link to the file."""
    relative = source.get("field")
    if not relative:
        return None
    field = (run.path / relative).resolve()
    if not field.is_file():
        raise FileNotFoundError(f"{run.name}: source {source.get('id')!r} names {relative}, absent")
    site = target / "fields" / str(source.get("id"))
    if site.is_symlink():
        site.unlink()
    index = field_payload.build_site(field, site)
    return {
        "url": f"fields/{source.get('id')}",
        "cells": len(index["cell_index"]),
        "order": index["order"],
        "samples": index["samples"],
    }


def build_run(run: WalkRun, target: Path) -> dict[str, Any]:
    """Write ``run.json`` under ``target`` and link the payloads it names."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "name": run.name,
        "scene_id": run.scene_id,
        "dwelling": run.dwelling,
        "room": run.room,
        "sources": [
            {
                "id": str(source.get("id", f"S{i + 1}")),
                "name": str(source.get("name", "")),
                "position": [float(v) for v in source["position"]],
                "directivity": str(source.get("directivity", "omni")),
                "field": _field_record(run, source, target),
            }
            for i, source in enumerate(run.sources)
        ],
        "meshes": {
            fmax: _mesh_record(run, fmax, relative, target)
            for fmax, relative in sorted(run.meshes.items(), key=lambda item: float(item[0]))
        },
    }
    (target / "run.json").write_text(json.dumps(record))
    return record


# --- a run that exists only so the app can be built before the real one ------


def _shell_blocks(lo: np.ndarray, hi: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Block centres of a hollow box between ``lo`` and ``hi``, with a label each.

    Floor 0, ceiling 1, walls 2: three materials, so the picture has more than
    one colour in it and a face on the wrong side of a wall would show.
    """
    counts = np.maximum(np.rint((hi - lo) / cell).astype(int), 2)
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in counts), indexing="ij")
    on_floor = iy == 0
    on_ceiling = iy == counts[1] - 1
    on_wall = (ix == 0) | (ix == counts[0] - 1) | (iz == 0) | (iz == counts[2] - 1)
    keep = on_floor | on_ceiling | on_wall
    label = np.where(on_floor, 0, np.where(on_ceiling, 1, 2))[keep]
    centres = np.stack([ix[keep], iy[keep], iz[keep]], axis=1) * cell + lo + cell / 2
    return centres.astype(np.float32), label.astype(np.int8)


def _solid_blocks(lo: np.ndarray, hi: np.ndarray, cell: float, label: int) -> np.ndarray:
    counts = np.maximum(np.rint((hi - lo) / cell).astype(int), 1)
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in counts), indexing="ij")
    centres = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1) * cell + lo + cell / 2
    return np.asarray(centres, dtype=np.float32)


def _tier(
    room_lo: np.ndarray, room_hi: np.ndarray, cell: float, target: Path, stem: str
) -> dict[str, Any]:
    """One tier of the synthetic room: a box shell and one block of furniture."""
    shell, shell_label = _shell_blocks(room_lo, room_hi, cell)
    table_lo = room_lo + np.array([1.0, cell, 1.0])
    table_hi = table_lo + np.array([1.2, 0.7, 0.8])
    table = _solid_blocks(table_lo, table_hi, cell, 3)
    positions = np.concatenate([shell, table])
    material = np.concatenate([shell_label, np.full(len(table), 3, dtype=np.int8)])
    cloud = VoxelCloud(
        positions=positions,
        material=material,
        inert=np.zeros(len(positions), dtype=bool),
        total_nodes=int(len(positions)),
        cell_m=cell,
        h_m=cell,
        bounds_lo=positions.min(axis=0) - cell / 2,
        bounds_hi=positions.max(axis=0) + cell / 2,
    )
    surface = surface_of(cloud)
    record = write_quads(surface, None, target, stem)
    corners = surface.corners.reshape(-1, 3)
    record["bounds"] = [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]
    return {
        "span": 1,
        "cell_m": cell,
        "blocks": cloud.drawn,
        "nodes": cloud.drawn,
        "quads": surface.quads,
        "sealed_left_out": 0,
        "tiles": [record],
        "aggregated": False,
    }


#: The sources the producer session announced for `hssd_0002` on 2026-09-10,
#: in the scene frame. Its lattice: one height, 1.70 m, 0.40 m step, 524
#: cells over 85.6 m2 of free floor.
PRODUCER_SOURCES: tuple[tuple[str, str, str, tuple[float, float, float]], ...] = (
    (
        "S1",
        "voice, living room by the sofa",
        "living room-hallway-dining room-kitchen",
        (-6.054, 1.695, -3.887),
    ),
    (
        "S2",
        "voice, living room by the kitchen",
        "living room-hallway-dining room-kitchen",
        (-2.9, 1.695, -3.9),
    ),
    ("S3", "voice, bedroom", "bedroom-closet.001", (-18.1, 1.642, 1.979)),
    ("S4", "voice, bathroom", "bathroom.001", (-8.467, 1.696, 0.33)),
    ("S5", "voice, small bedroom", "bedroom.001-closet.005-closet.003", (-1.13, 1.695, -1.107)),
)


def write_synthetic_run(
    target: Path,
    scene_id: str = "102344022",
    room: str = "living room-hallway-dining room-kitchen",
    centre: Sequence[float] = (-4.5, 0.0, -3.9),
    size: Sequence[float] = (6.0, 2.8, 5.0),
    fine_m: float = 0.1,
    coarse_m: float = 0.4,
    fields: bool = True,
    field_step_m: float = 0.40,
    field_box_m: float = 2.4,
) -> WalkRun:
    """A run made of nothing the solver produced, so the app can be built now.

    A box room of ``size`` centred on ``centre`` (its floor at ``centre[1]``),
    with one block of furniture in it, drawn at two cell sizes as two tiers of
    one mesh at a nominal 4000 Hz; the producer's five omnidirectional sources
    at their announced positions, each with a mock field over a box of
    ``field_box_m`` around it (``fields``), one listening height, the whole
    response per cell. Every file has the shape the real payload will have,
    and nothing in it is a measurement of anything.
    """
    target = Path(target)
    centre_a = np.asarray(centre, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2
    lo = centre_a - np.array([half[0], 0.0, half[2]])
    hi = centre_a + np.array([half[0], 2 * half[1], half[2]])
    payload = target / "audit_4k" / "voxels"
    room_dir = payload / room.replace(" ", "_")
    fine = _tier(lo, hi, fine_m, room_dir, "fine")
    coarse = _tier(lo, hi, coarse_m, room_dir, "coarse")
    coarse["span"] = int(round(coarse_m / fine_m))
    coarse["aggregated"] = True
    outline = [
        [
            [float(lo[0]), float(lo[2])],
            [float(hi[0]), float(lo[2])],
            [float(hi[0]), float(hi[2])],
            [float(lo[0]), float(hi[2])],
            [float(lo[0]), float(lo[2])],
        ]
    ]
    (payload / "rooms.json").write_text(
        json.dumps(
            {
                "cache_key": "synthetic",
                "scene_id": scene_id,
                "h_m": fine_m,
                "coarse_span": coarse["span"],
                "labels": ["floor", "ceiling", "wall", "table"],
                "note": "synthetic box, not a voxelisation of anything",
                "rooms": [
                    {
                        "name": room,
                        "label": room,
                        "regions": room.split("-"),
                        "area_m2": round(float(size[0] * size[2]), 3),
                        "outline": outline,
                        "stand": [float(centre_a[0]), float(centre_a[2]), float(half[0])],
                        "dir": room_dir.name,
                        "fine": fine,
                        "coarse": coarse,
                    }
                ],
            }
        )
    )
    sources = [
        (source_id, name, list(position)) for source_id, name, _, position in PRODUCER_SOURCES
    ]
    if fields:
        for index, (source_id, _, position) in enumerate(sources):
            half_box = field_box_m / 2
            field_payload.mock_field(
                target / "fields" / f"{source_id}.h5",
                source_id=source_id,
                source_position=position,
                box_lo=[position[0] - half_box, float(lo[1]), position[2] - half_box],
                box_hi=[position[0] + half_box, float(hi[1]), position[2] + half_box],
                step_m=field_step_m,
                margin_m=0.0,
                scene_id=scene_id,
                dwelling=local_name(scene_id, 1),
                room_name=PRODUCER_SOURCES[index][2],
                seed=index,
            )
    manifest = {
        "dwelling": local_name(scene_id, 1),
        "room": room,
        "synthetic": True,
        "sources": [
            {
                "id": source_id,
                "name": name,
                "position": position,
                "directivity": "omni",
                **({"field": f"fields/{source_id}.h5"} if fields else {}),
            }
            for source_id, name, position in sources
        ],
        "meshes": {"4000": "audit_4k/voxels"},
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / WALK_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return WalkRun.read(target)


def check_run(path: Path) -> list[str]:
    """Everything the app would refuse in a run directory, so the producer
    can fix it before the swap rather than at the first launch."""
    problems: list[str] = []
    path = Path(path)
    if not (path / WALK_MANIFEST).is_file():
        return [f"no {WALK_MANIFEST} in {path}"]
    try:
        run = WalkRun.read(path)
    except (ValueError, KeyError) as error:
        return [str(error)]
    if not run.sources:
        problems.append("no sources")
    seen: set[str] = set()
    for source in run.sources:
        source_id = str(source.get("id"))
        if source_id in seen:
            problems.append(f"source id {source_id!r} used twice")
        seen.add(source_id)
        relative = source.get("field")
        if not relative:
            problems.append(f"source {source_id}: no field, it will be seen and not heard")
            continue
        field = path / relative
        if not field.is_file():
            problems.append(f"source {source_id}: field {relative} is absent")
            continue
        for problem in field_payload.check(field):
            problems.append(f"source {source_id}: {problem}")
        header = field_payload.read_header(field) if not problems else None
        if header and header.source_id != source_id:
            problems.append(f"source {source_id}: its field says source_id {header.source_id!r}")
        if header and header.scene_id != run.scene_id:
            problems.append(f"source {source_id}: its field is of scene {header.scene_id}")
    if not run.meshes:
        problems.append("no meshes: the acoustic view will be disabled")
    for fmax, relative in run.meshes.items():
        rooms = path / relative / "rooms.json"
        if not rooms.is_file():
            problems.append(f"mesh {fmax} Hz: no rooms.json under {relative}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write or check a run for the walk-through app")
    parser.add_argument("target", type=Path, help="run directory to create, or to check")
    parser.add_argument("--check", action="store_true", help="check an existing run and exit")
    parser.add_argument("--scene", default="102344022")
    parser.add_argument("--room", default="living room-hallway-dining room-kitchen")
    parser.add_argument(
        "--centre",
        type=float,
        nargs=3,
        default=(-4.5, 0.0, -3.9),
        metavar=("X", "FLOOR_Y", "Z"),
        help="centre of the box on the floor, scene frame",
    )
    parser.add_argument("--no-fields", action="store_true", help="skip the mock fields")
    arguments = parser.parse_args(argv)
    if arguments.check:
        problems = check_run(arguments.target)
        for problem in problems:
            print(f"  {problem}")
        if not problems:
            run = WalkRun.read(arguments.target)
            print(
                f"{arguments.target}: {run.dwelling or run.scene_id}, {len(run.sources)} sources, "
                f"meshes {', '.join(run.meshes) or 'none'} Hz"
            )
        return 1 if problems else 0
    run = write_synthetic_run(
        arguments.target,
        arguments.scene,
        arguments.room,
        arguments.centre,
        fields=not arguments.no_fields,
    )
    print(f"wrote {run.path} ({len(run.sources)} sources, meshes {', '.join(run.meshes)} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
