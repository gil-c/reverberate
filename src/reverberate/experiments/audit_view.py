"""The audit view: a whole flat, drawn at the grid's own step in one room at a time.

Constraint 9 says the viewer reads what the solver reads. At 4 kHz the whole
flat is drawable losslessly and that is already published. At 16 kHz it is not:
1 089 464 499 boundary nodes merge to roughly 35 M quads and 2.8 GB, which no
browser will hold. **So the picture is tiered rather than thinned.** The room a
reader stands in draws at the grid's own 2.04 mm, and every other room draws at
8.17 mm, which is the 4 kHz cell size. Both tiers come from the *same*
voxelisation: mixing a 16 kHz grid with a separate 4 kHz one would put a seam at
every room boundary that belongs to the mixing and not to the geometry.

Nothing is dropped and nothing is smoothed. The only reductions are the two the
merge already makes -- faces between touching blocks are not drawn, coplanar
faces of one material become one rectangle -- and the aggregation of the coarse
tier, which the payload states in so many words.

**Rooms are processed one at a time, and the grid is re-read for each.** The
whole flat's node list is 1.089e9 rows and does not fit beside the arrays the
merge needs; one room's does. Re-reading the 25 GB entry twelve times costs
about three minutes on an NVMe and needs no scratch file, no shard format and
no resume logic, and it keeps every step of the build under two minutes.

Usage::

    PYTHONPATH=src .venv/bin/python -m reverberate.experiments.audit_view \\
        --cache-key 9ef0bd3e... --scene-id 102344022 \\
        --hssd-root data/raw/hssd-hab --out data/runs/w34_audit_16k
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.experiments.run import entry_from_key
from reverberate.geometry.rooms import Partition, _rasterise, partition_of_scene
from reverberate.viz.vox_view import (
    blocks_from_nodes,
    scene_subs,
    surface_of,
    write_quads,
)

__all__ = [
    "BUILD_NOTE",
    "SEALED_LEFT_OUT",
    "TierPlan",
    "build",
    "halo_for",
    "main",
    "membership_mask",
]

#: The coarse tier's block, as a multiple of the grid step. Four at 16 kHz is
#: 8.17 mm, which is exactly the 4 kHz cell, so "the other rooms at the 4 kHz
#: resolution" is literally true rather than approximately.
COARSE_SPAN = 4

#: The most quads one tile may hold. Eighty bytes a quad, so this is 40 MB.
#:
#: Measured rather than chosen: at 2 000 000 the living room came out as eight
#: tiles of 1.7 M, and a browser holding eight million quads could fit only two
#: of them, so a reader standing in the middle of the room saw a quarter of it
#: with the rest missing. The tile is the grain at which the view can follow a
#: reader, so it wants to be well under the budget rather than near it.
QUAD_BUDGET = 500_000

#: How many nodes are read from the grid at once. Sixteen million rows is about
#: 220 MB of transient across the three arrays, and the read is sequential.
CHUNK = 1 << 24

#: Why the sealed insides are not in the payload, said on the page.
SEALED_LEFT_OUT = (
    "The sealed insides are merged and then left out. They are the inward face "
    "of every closed body's shell -- PFFDTD stores boundary nodes only, so the "
    "middle of a solid is not in the grid and that face has nothing behind it "
    "to hide it. Every one of them sits behind the material face in front, so "
    "no reader ever sees one, and on this flat they are 42 per cent of the "
    "merged area. They are counted per room rather than silently absent. What "
    "is *not* dropped is a block that is rigid and still coupled, which is the "
    "defect sealing exists to fix: that carries a different label and is drawn."
)

BUILD_NOTE = (
    "Two tiers of one grid. The selected room draws at the solver's own step; "
    "every other room draws aggregated, at the cell size of the band below. "
    "Faces between touching blocks are not drawn and coplanar faces of the same "
    "material are merged into rectangles, so the solid is the same one the "
    "solver reads. Nothing is smoothed: the staircase is what the wave equation "
    "was solved on. Grey is rigid, a block carrying no material at all. Two "
    "materials of this flat's fifty-one can still look alike, so the legend "
    "names them and the colour only separates them."
)


@dataclass(frozen=True)
class TierPlan:
    """One tier of one room: what was drawn, at what step, and out of what."""

    span: int
    cell_m: float
    blocks: int
    nodes: int
    quads: int
    files: list[dict[str, Any]]
    #: Sealed faces merged and then left out of the payload. Counted and
    #: published rather than simply absent: a picture that quietly drops 42 per
    #: cent of what it merged is the kind of thing this view exists to catch.
    sealed_left_out: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "span": self.span,
            "cell_m": self.cell_m,
            "blocks": self.blocks,
            "nodes": self.nodes,
            "quads": self.quads,
            "sealed_left_out": self.sealed_left_out,
            "tiles": self.files,
            "aggregated": self.span > 1,
        }


def halo_for(span: int) -> int:
    """How far past a room its working set must reach, in grid cells.

    Two spans, and the second one was measured rather than reasoned. A block
    anchored on the room's own edge runs ``span`` cells outward, and the block
    beyond *that* runs another ``span``: unless one of its nodes is in the
    working set the face between them is drawn, because nothing says it is
    hidden. At ``span + 1`` the coarse tier came out 0.068 m2 heavy on this
    flat, 0.0024 per cent, which is small and is exactly the class of defect an
    audit view must not have -- a face the solver does not see.
    """
    return 2 * span


def membership_mask(partition: Partition, halo: int) -> np.ndarray:
    """A bitset per lattice cell naming every room within ``halo`` cells of it.

    A room's working set is not its own nodes. Two things need its neighbours:
    a face between a block of this room and a block of the next is hidden in
    the grid and must be hidden here too, and a coarse block straddling the
    partition has to be built from all the nodes in it whichever side they
    fall. Both are answered by dilating each room's cells by ``halo`` and
    keeping every node inside the result.

    Packed as one integer per cell rather than one boolean raster per room:
    thirteen rasters of the flat's 104 M cells is 1.25 GB and one ``uint16`` is
    209 MB, and it costs one lookup per node instead of thirteen.
    """
    from scipy import ndimage

    if len(partition.rooms) > 16:
        raise ValueError(f"{len(partition.rooms)} rooms will not pack into a uint16 mask")
    cross = ndimage.generate_binary_structure(2, 1)
    mask = np.zeros(partition.raster.shape, dtype=np.uint16)
    for index in range(len(partition.rooms)):
        near = ndimage.binary_dilation(partition.raster == index, cross, iterations=halo)
        mask |= near.astype(np.uint16) << np.uint16(index)
    return mask


#: The band of heights a standing point must be clear through, in metres above
#: the grid's floor. A reader's eye is at 1.6 m and their head is not a point.
HEAD_BAND_M = (1.2, 1.9)

#: Side of the cell the standing search works on, in metres. Coarse on purpose:
#: it is looking for the middle of the floor, not for a gap between two chairs.
STAND_CELL_M = 0.05


def stand_points(
    cache_dir: Path,
    partition: Partition,
    axes: list[np.ndarray],
    ny: int,
    nz: int,
    shape: tuple[int, int, int],
) -> dict[str, list[float]]:
    """The most open point of each room's floor, measured against the grid.

    A reader dropped into a room they are auditing must not start inside the
    wardrobe. The room outline cannot say where the wardrobe is -- it is a floor
    plan -- and standing at the outline's most open point put the camera inside
    solid geometry, which fills the view with the pink of a sealed interior and
    reads as a broken page rather than as a camera inside the furniture.

    So the grid answers instead. Every boundary node between
    :data:`HEAD_BAND_M` is stamped onto a coarse plan of the flat, and each room
    takes the cell of its own that is farthest from anything stamped. That is
    the middle of the free floor at head height, furniture included, and it is
    the same measurement whatever the band or the scene.

    Returns, per room, ``[x, z, clearance]`` in metres. The clearance travels
    because a room with none -- a closet packed to the ceiling -- should say so
    rather than look like a room the reader failed to find.
    """
    from scipy import ndimage

    low = float(axes[1][0]) + HEAD_BAND_M[0]
    high = float(axes[1][0]) + HEAD_BAND_M[1]
    band = (np.searchsorted(axes[1], low), np.searchsorted(axes[1], high))
    step = max(1, int(round(STAND_CELL_M / float(axes[0][1] - axes[0][0]))))
    plan = np.zeros((-(-shape[0] // step), -(-shape[2] // step)), dtype=bool)

    with h5py.File(cache_dir / "vox_out.h5", "r") as handle:
        total = int(handle["bn_ixyz"].shape[0])
        for start in range(0, total, CHUNK):
            index = np.asarray(handle["bn_ixyz"][start : min(start + CHUNK, total)])
            subs = scene_subs(index, ny, nz, shape)
            del index
            at_head = (subs[1] >= band[0]) & (subs[1] < band[1])
            plan[subs[0][at_head] // step, subs[2][at_head] // step] = True
            del subs

    # Distance to the nearest occupied cell, in metres, over the whole plan;
    # each room then takes the best cell it owns. One transform for every room
    # rather than one each, because clearance does not stop at a partition: a
    # doorway's free space belongs to whichever room the cell is in.
    clear = ndimage.distance_transform_edt(~plan) * step * float(axes[0][1] - axes[0][0])
    # Inside the room's own polygon, not inside its share of the partition. The
    # partition fills every cell of the bounding box, so a room on the outside
    # wall owns a slab of empty ground beyond it -- and that slab has the
    # largest clearance in the flat, which is where the first version stood the
    # reader: eleven metres clear, in the garden, eight metres from the bedroom.
    coarse = _rasterise(list(partition.rooms), axes[0][::step], axes[2][::step])
    coarse = coarse[: plan.shape[0], : plan.shape[1]]

    found: dict[str, list[float]] = {}
    for position, room in enumerate(partition.rooms):
        mine = coarse == position
        if not mine.any():
            continue
        masked = np.where(mine, clear, -1.0)
        flat = int(np.argmax(masked))
        ix, iz = divmod(flat, plan.shape[1])
        found[room.name] = [
            round(float(axes[0][min(ix * step, shape[0] - 1)]), 3),
            round(float(axes[2][min(iz * step, shape[2] - 1)]), 3),
            round(float(masked[ix, iz]), 3),
        ]
    return found


def _scan_room(
    cache_dir: Path, mask: np.ndarray, room: int, shape: tuple[int, int, int], ny: int, nz: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Every node of the grid within one room's dilated footprint.

    Reads the whole entry and keeps a slice of it. Sequential on an NVMe at
    about 1.9 GB/s measured, so the scan costs about twenty seconds and the
    alternative -- a per-room shard file written in one pass -- costs 18 GB of
    scratch and a resume protocol to save eight minutes over the whole build.
    """
    bit = np.uint16(1) << np.uint16(room)
    subs_out: list[list[np.ndarray]] = [[], [], []]
    material_out: list[np.ndarray] = []
    inert_out: list[np.ndarray] = []
    with h5py.File(cache_dir / "vox_out.h5", "r") as handle:
        total = int(handle["bn_ixyz"].shape[0])
        for start in range(0, total, CHUNK):
            stop = min(start + CHUNK, total)
            index = np.asarray(handle["bn_ixyz"][start:stop])
            subs = scene_subs(index, ny, nz, shape)
            del index
            keep = (mask[subs[0], subs[2]] & bit) != 0
            if not keep.any():
                continue
            for axis in range(3):
                subs_out[axis].append(subs[axis][keep])
            material_out.append(np.asarray(handle["mat_bn"][start:stop], dtype=np.int8)[keep])
            adjacency = np.asarray(handle["adj_bn"][start:stop], dtype=bool)
            inert_out.append(np.asarray(~adjacency.any(axis=1))[keep])
            del subs, adjacency
    subs_all = [np.concatenate(part) if part else np.zeros(0, np.int32) for part in subs_out]
    material = np.concatenate(material_out) if material_out else np.zeros(0, np.int8)
    inert = np.concatenate(inert_out) if inert_out else np.zeros(0, bool)
    return subs_all, material, inert


def _outline(polygon: Any) -> list[list[list[float]]]:
    """A room's footprint as plain rings, ready for a point-in-polygon in JS."""
    shapes = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    return [
        [[round(float(x), 2), round(float(z), 2)] for x, z in piece.exterior.coords]
        for piece in shapes
        if not piece.is_empty
    ]


def _tile(corners: np.ndarray, budget: int, quads: int) -> tuple[np.ndarray, int]:
    """Assign each quad to a tile, cutting the room's box until each fits.

    Cut on the two horizontal axes only. A room is wider than it is tall, so
    cutting the height would make tiles a reader has to load in pairs to see
    one wall.
    """
    if quads <= budget:
        return np.zeros(quads, dtype=np.int32), 1
    centre = corners.mean(axis=1)
    lo, high = centre.min(axis=0), centre.max(axis=0)
    extent = np.maximum(high - lo, 1e-9)
    cuts = np.array([1, 1, 1])
    while quads / float(cuts[0] * cuts[2]) > budget:
        axis = 0 if extent[0] / cuts[0] >= extent[2] / cuts[2] else 2
        cuts[axis] += 1
    ix = np.minimum(((centre[:, 0] - lo[0]) / extent[0] * cuts[0]).astype(np.int32), cuts[0] - 1)
    iz = np.minimum(((centre[:, 2] - lo[2]) / extent[2] * cuts[2]).astype(np.int32), cuts[2] - 1)
    return ix * cuts[2] + iz, int(cuts[0] * cuts[2])


def _write_tier(
    subs: list[np.ndarray],
    material: np.ndarray,
    inert: np.ndarray,
    room_of: np.ndarray,
    span: int,
    axes: list[np.ndarray],
    h_m: float,
    room: int,
    target: Path,
    stem: str,
    budget: int,
) -> TierPlan:
    """Merge one room's working set at one span and write the quads it owns."""
    Path(target).mkdir(parents=True, exist_ok=True)
    cloud = blocks_from_nodes(subs, material, inert, room_of, span, axes, h_m, int(material.size))
    surface = surface_of(cloud)
    assert surface.quad_room is not None
    mine = surface.quad_room == room
    # The sealed faces are merged and then not written. They are the inward
    # side of every closed body's shell, and they are 42 per cent of the area:
    # each one sits behind the material face in front of it, so no reader ever
    # sees one, and shipping them costs the browser almost half its memory for
    # nothing. See SEALED_LEFT_OUT.
    #
    # Merged first and dropped after, never excluded from the merge: a sealed
    # block still hides the back of the material block in front of it, and
    # removing it earlier would expose that face and draw a surface the solver
    # does not have.
    sealed = surface.label.reshape(-1, 4)[:, 0] == -2
    left_out = int((mine & sealed).sum())
    mine = mine & ~sealed
    corners = surface.corners.reshape(-1, 4, 3)[mine]
    tile_of, count = _tile(corners, budget, int(mine.sum()))
    # A rebuild at a different tile size leaves the previous one's files behind,
    # and they are indistinguishable from the current ones on disk. The index
    # would not list them, so nothing would draw them, but a reader looking at
    # the directory to check what was published would be looking at two answers.
    for stale in Path(target).glob(f"{stem}*"):
        stale.unlink()
    files: list[dict[str, Any]] = []
    kept = np.flatnonzero(mine)
    for tile in range(count):
        chosen = kept[tile_of == tile]
        if chosen.size == 0:
            continue
        record = write_quads(surface, chosen, target, f"{stem}_{tile}" if count > 1 else stem)
        box = surface.corners.reshape(-1, 4, 3)[chosen].reshape(-1, 3)
        record["bounds"] = [box.min(axis=0).tolist(), box.max(axis=0).tolist()]
        files.append(record)
    return TierPlan(
        span=span,
        cell_m=float(cloud.cell_m),
        blocks=int((cloud.room == room).sum()) if cloud.room is not None else cloud.drawn,
        nodes=int((room_of == room).sum()),
        quads=int(mine.sum()),
        files=files,
        sealed_left_out=left_out,
    )


def build(
    cache_key: str,
    hssd_root: Path,
    scene_id: str,
    out: Path,
    coarse_span: int = COARSE_SPAN,
    quad_budget: int = QUAD_BUDGET,
    only: list[str] | None = None,
    index_only: bool = False,
) -> dict[str, Any]:
    """Build both tiers for every room and write them under ``out``.

    ``index_only`` rewrites ``rooms.json`` from the payloads already on disk and
    builds nothing. Everything in it beyond the tier records -- the room's
    regions, its outline, where a reader should stand -- is derived from the
    partition and the grid, so it is refreshed on every run anyway; this is the
    way to refresh it without spending the thirteen minutes again. It still
    reads the grid once, because the standing point is measured against it."""
    entry = entry_from_key(cache_key)
    cache_dir = entry.path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_dir / "vox_out.h5", "r") as handle:
        _, ny, nz = (int(handle[k][()]) for k in ("Nx", "Ny", "Nz"))
        h_m = float(handle["h"][()])
        total = int(handle["bn_ixyz"].shape[0])
    with h5py.File(cache_dir / "cart_grid.h5", "r") as handle:
        axes = [np.asarray(handle[k][:], dtype=np.float64) for k in ("xv", "yv", "zv")]
    shape = (axes[0].size, axes[1].size, axes[2].size)

    partition = partition_of_scene(Path(hssd_root), scene_id, axes[0], axes[2])
    mask = membership_mask(partition, halo_for(coarse_span))
    labels = sorted(json.loads((cache_dir / "manifest.json").read_text()).get("materials") or {})
    standing = stand_points(cache_dir, partition, axes, ny, nz, shape)

    rooms: list[dict[str, Any]] = []
    for index, room in enumerate(partition.rooms):
        if index_only or (only and room.name not in only):
            continue
        started, spent = time.time(), time.process_time()
        subs, material, inert = _scan_room(cache_dir, mask, index, shape, ny, nz)
        room_of = partition.raster[subs[0], subs[2]]
        target = out / "voxels" / room.name.replace(" ", "_")
        fine = _write_tier(
            subs, material, inert, room_of, 1, axes, h_m, index, target, "fine", quad_budget
        )
        coarse = _write_tier(
            subs,
            material,
            inert,
            room_of,
            coarse_span,
            axes,
            h_m,
            index,
            target,
            "coarse",
            quad_budget,
        )
        rooms.append(
            {
                "name": room.name,
                "label": room.label,
                "regions": list(room.regions),
                "area_m2": round(room.area_m2, 3),
                # The floor outline, so the page can say which room the reader
                # is standing in rather than making them pick from a list. Two
                # decimals is a centimetre, which is finer than the question
                # "am I in the kitchen" can be asked at.
                "outline": _outline(room.polygon),
                "stand": standing.get(room.name),
                "dir": target.name,
                "fine": fine.record(),
                "coarse": coarse.record(),
                "build_s": round(time.time() - started, 1),
                # Wall clock *and* processor time, because the two disagree for
                # a reason worth seeing. A laptop that suspends mid-build makes
                # a room look a hundred times slower while it computed at the
                # usual rate: 5848 s of wall against 37 s of work is a lid that
                # closed, not a slow merge, and one number cannot tell which.
                "cpu_s": round(time.process_time() - spent, 1),
                "working_set": int(material.size),
            }
        )
        print(
            f"{room.name:16s} {fine.nodes:>12,} nodes  fine {fine.quads:>9,} quads "
            f"in {len(fine.files)} tile(s)  coarse {coarse.quads:>8,}  "
            f"{rooms[-1]['cpu_s']:.0f}s cpu of {rooms[-1]['build_s']:.0f}s wall",
            flush=True,
        )

    # A partial build merges into what is already published rather than
    # replacing it. Rooms are independent and the largest is minutes of work,
    # so building them a few at a time is the normal way to run this; an index
    # that forgot the rooms built an hour ago would make that unusable.
    index_path = out / "voxels" / "rooms.json"
    if (only or index_only) and index_path.is_file():
        previous = json.loads(index_path.read_text())
        fresh = {room["name"] for room in rooms}
        rooms = [room for room in previous.get("rooms", []) if room["name"] not in fresh] + rooms
        order = {room.name: position for position, room in enumerate(partition.rooms)}
        rooms.sort(key=lambda published: order.get(str(published["name"]), len(order)))
        # Everything a room carries beyond its own tiers is derived from the
        # partition, and the partition is rebuilt on every invocation. Refresh
        # it rather than trusting what an earlier one wrote: a room whose
        # outline came from an older mapping would put the reader in the wrong
        # room, and the tiers beside it would look like the answer to that.
        for published in rooms:
            source = partition.rooms[order[str(published["name"])]]
            published["label"] = source.label
            published["regions"] = list(source.regions)
            published["area_m2"] = round(source.area_m2, 3)
            published["outline"] = _outline(source.polygon)
            published["stand"] = standing.get(source.name)

    record = {
        "cache_key": cache_key,
        "scene_id": scene_id,
        "h_m": h_m,
        "grid_shape": list(shape),
        "total_nodes": total,
        "coarse_span": coarse_span,
        "quad_budget": quad_budget,
        "labels": labels,
        "note": BUILD_NOTE,
        "sealed_note": SEALED_LEFT_OUT,
        "sealed_left_out": sum(
            int(room[tier]["sealed_left_out"]) for room in rooms for tier in ("fine", "coarse")
        ),
        "rooms": rooms,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coarse-span", type=int, default=COARSE_SPAN)
    parser.add_argument("--quad-budget", type=int, default=QUAD_BUDGET)
    parser.add_argument("--only", nargs="*", default=None, help="build these rooms only")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="rewrite rooms.json from the payloads on disk, merging nothing",
    )
    args = parser.parse_args(argv)

    record = build(
        args.cache_key,
        args.hssd_root,
        args.scene_id,
        args.out,
        args.coarse_span,
        args.quad_budget,
        args.only,
        args.index_only,
    )
    fine = sum(int(room["fine"]["quads"]) for room in record["rooms"])
    coarse = sum(int(room["coarse"]["quads"]) for room in record["rooms"])
    dropped = int(record["sealed_left_out"])
    print(
        f"\n{len(record['rooms'])} rooms, {record['total_nodes']:,} nodes: "
        f"fine {fine:,} quads ({fine * 80 / 1e6:.0f} MB), "
        f"coarse {coarse:,} quads ({coarse * 80 / 1e6:.0f} MB); "
        f"{dropped:,} sealed quads merged and left out "
        f"({100 * dropped / max(1, dropped + fine + coarse):.0f} per cent, "
        f"{dropped * 80 / 1e6:.0f} MB saved)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - a command line entry point
    raise SystemExit(main())
