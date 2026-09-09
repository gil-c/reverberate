"""Air pockets in a voxelised grid, counted and sealed on the grid itself.

W25 found that PFFDTD does not fill solids, so the inside of every closed
object was simulated as air behind a rigid wall and rang. The fix was a census
of closed bodies on the **meshes** (:mod:`reverberate.geometry.sealed`) and a
patch that seals the non-air side of every closed surface. That census cannot
see what only exists on the grid: an object whose mesh is not closed but whose
voxels enclose air at this cell size, two objects that touch once the cells are
coarse, a door leaf one cell thick with air inside it. On the 81 m2 living
room at 1 kHz that left 344 pockets coupled to the field, and one of them, a
0.49 m wide closet interior, rang at 700 Hz for the whole response, 21 dB above
its neighbours.

**The census is on the grid, and it is the last word.** Every node that is not a
boundary node is air; the six connected components of that air are the regions
the wave can occupy. Two of them are large, the room or the storey and the box
of exterior air around the shell; everything else is the inside of something.
**A component smaller than a room is sealed**, :data:`ROOM_MIN_M3`, whatever
surrounds it: its cells become sealed boundary nodes, and the bit that pointed
into them from every neighbouring live node is cleared, so the room neither
drives them nor hears them. The report names every component sealed with its
volume, its lowest mode and the materials around it, so a reader can check
that the line that vanished was a pocket and not a room.

**Why the materials do not decide.** The first rule tried kept any component
that touched the shell, on the argument that a room touches its walls. The
closet that rang touched the wall behind it, a cabinet with no back, and
stayed; the car in the garage, 7.7 m3 bounded by nothing but "car", would have
been sealed either way. What separates an inside from a room is that a room
is bigger than 10 m3: the smallest room in the reference flat is 13 m3, and a
walk-in closet behind a closed door, sealed by this rule, was never part of
the domain the listener is in. Under a couch, at 33 mm cells, the gap closes
and the space becomes a cavity coupled to the room only by numerical leakage;
sealing it is the smaller error, and at 8 mm the gap is open and nothing is
sealed there.

The grid is read whole, so this tool fits the low and mid bands of a storey on
a laptop and needs the rented machine for a room at 16 kHz; :func:`census`
says how many bytes it will take before it takes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import ndimage

__all__ = ["Pocket", "PocketCensus", "census", "seal", "seal_in_place"]

#: Cells to labels: ``uint8`` air mask plus ``int32`` labels.
BYTES_PER_CELL = 5

#: Below this an air component is the inside of something, not a room.
ROOM_MIN_M3 = 10.0


@dataclass(frozen=True)
class Pocket:
    """One connected region of air, and what surrounds it."""

    label: int
    cells: int
    volume_m3: float
    span_m: tuple[float, float, float]
    origin_m: tuple[float, float, float]
    live_nodes: int
    sealed_nodes: int
    materials: tuple[int, ...]
    #: Whether the census decided to seal it: smaller than a room.
    sealed: bool

    @property
    def lowest_mode_hz(self) -> float:
        """``c / 2L`` on the longest side, at 343.2 m/s."""
        return 343.2 / (2.0 * max(self.span_m))

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cells": self.cells,
            "volume_m3": round(self.volume_m3, 5),
            "span_m": [round(v, 4) for v in self.span_m],
            "origin_m": [round(v, 4) for v in self.origin_m],
            "lowest_mode_hz": round(self.lowest_mode_hz, 1),
            "live_boundary_nodes": self.live_nodes,
            "sealed_boundary_nodes": self.sealed_nodes,
            "materials": list(self.materials),
            "sealed": self.sealed,
        }


@dataclass
class PocketCensus:
    """Every air component of a grid, the big ones named and the pockets listed."""

    shape: tuple[int, int, int]
    step_m: float
    components: int
    largest: list[dict[str, Any]]
    pockets: list[Pocket] = field(default_factory=list)

    def to_seal(self) -> list[Pocket]:
        return [pocket for pocket in self.pockets if pocket.sealed]

    def record(self) -> dict[str, Any]:
        sealed = self.to_seal()
        return {
            "shape": list(self.shape),
            "step_m": self.step_m,
            "components": self.components,
            "largest": self.largest,
            "pockets_coupled": len(self.pockets),
            "pockets_sealed": len(sealed),
            "pockets_sealed_volume_m3": round(sum(p.volume_m3 for p in sealed), 4),
            "room_min_m3": ROOM_MIN_M3,
            "sealed": [p.record() for p in sorted(sealed, key=lambda p: -p.cells)[:200]],
            "kept": [p.record() for p in self.pockets if not p.sealed][:50],
            "rule": (
                f"air components of the grid; every component under {ROOM_MIN_M3:g} m3 is the "
                "inside of something and is sealed, whatever surrounds it; the materials "
                "around each are reported so the decision can be read"
            ),
        }


def _load(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return {
            "shape": tuple(int(handle[k][()]) for k in ("Nx", "Ny", "Nz")),
            "h": float(handle["h"][()]),
            "bn": np.asarray(handle["bn_ixyz"][:], dtype=np.int64),
            "adj": np.asarray(handle["adj_bn"][:], dtype=bool),
            "mat": np.asarray(handle["mat_bn"][:]),
            "axes": tuple(np.asarray(handle[k][:], dtype=float) for k in ("xv", "yv", "zv")),
        }


def census(
    vox_out: Path,
    *,
    room_min_m3: float = ROOM_MIN_M3,
    max_cells: int = 2_000_000_000,
    min_cells: int = 8,
) -> PocketCensus:
    """Count the air components of a grid and say which are smaller than a room.

    ``max_cells`` refuses a grid this machine should not read whole.
    """
    grid = _load(Path(vox_out))
    nx, ny, nz = grid["shape"]
    total = nx * ny * nz
    if total > max_cells:
        raise MemoryError(
            f"{total:.3g} cells at {BYTES_PER_CELL} bytes each is "
            f"{total * BYTES_PER_CELL / 1e9:.0f} GB; run the census where that fits"
        )
    bn, adj, mat = grid["bn"], grid["adj"], grid["mat"]
    air = np.ones(total, dtype=bool)
    air[bn] = False
    labels, count = ndimage.label(
        air.reshape(nx, ny, nz), structure=ndimage.generate_binary_structure(3, 1)
    )
    labels = np.asarray(labels, dtype=np.int32).ravel()
    del air
    sizes = np.bincount(labels)
    sizes[0] = 0
    step = grid["h"]
    cell = step**3

    live = adj.any(axis=1)
    strides = (ny * nz, nz, 1)
    touched_live: dict[int, int] = {}
    touched_sealed: dict[int, int] = {}
    materials: dict[int, set[int]] = {}
    for stride in strides:
        for sign in (1, -1):
            neighbour = bn + sign * stride
            inside = (neighbour >= 0) & (neighbour < total)
            found = np.zeros(bn.size, dtype=np.int32)
            found[inside] = labels[neighbour[inside]]
            for mask, bucket in ((live, touched_live), (~live, touched_sealed)):
                hit = found[mask]
                for label, n in zip(*np.unique(hit[hit > 0], return_counts=True), strict=True):
                    bucket[int(label)] = bucket.get(int(label), 0) + int(n)
            hit_live = found[live]
            mats_live = mat[live]
            keep = hit_live > 0
            for label, material in zip(hit_live[keep], mats_live[keep], strict=True):
                materials.setdefault(int(label), set()).add(int(material))

    objects = ndimage.find_objects(labels.reshape(nx, ny, nz))
    axes = grid["axes"]
    order = np.argsort(sizes)[::-1]
    largest = [
        {
            "label": int(label),
            "cells": int(sizes[label]),
            "volume_m3": round(float(sizes[label] * cell), 3),
        }
        for label in order[:3]
        if sizes[label] > 0
    ]
    pockets: list[Pocket] = []
    max_cells_pocket = int(room_min_m3 / cell)
    for label, live_n in touched_live.items():
        n = int(sizes[label])
        if n < min_cells:
            continue
        slices = objects[label - 1]
        span = tuple(float((s.stop - s.start) * step) for s in slices)
        origin = tuple(float(axis[s.start]) for axis, s in zip(axes, slices, strict=True))
        mats = tuple(sorted(materials.get(label, set())))
        pockets.append(
            Pocket(
                label=int(label),
                cells=n,
                volume_m3=float(n * cell),
                span_m=(span[0], span[1], span[2]),
                origin_m=(origin[0], origin[1], origin[2]),
                live_nodes=int(live_n),
                sealed_nodes=int(touched_sealed.get(label, 0)),
                materials=mats,
                sealed=n <= max_cells_pocket,
            )
        )
    pockets.sort(key=lambda p: -p.cells)
    return PocketCensus(
        shape=(nx, ny, nz), step_m=step, components=int(count), largest=largest, pockets=pockets
    )


def seal(vox_out: Path, out: Path, pockets: list[Pocket]) -> dict[str, Any]:
    """Write a copy of the grid with the given pockets sealed.

    Every cell of a sealed pocket becomes a boundary node with no adjacency,
    no material and no surface, which is exactly what W25's patch made of the
    inside of a closed body; and every live boundary node next to one has the
    bit that pointed into it cleared, so its ``K_bn`` counts real air only.
    ``bn_ixyz`` stays sorted, which the engine and its multi GPU split require.
    """
    grid = _load(Path(vox_out))
    nx, ny, nz = grid["shape"]
    total = nx * ny * nz
    bn, adj, mat = grid["bn"], grid["adj"], grid["mat"]
    with h5py.File(vox_out, "r") as handle:
        saf = np.asarray(handle["saf_bn"][:])
    air = np.ones(total, dtype=bool)
    air[bn] = False
    labels, _ = ndimage.label(
        air.reshape(nx, ny, nz), structure=ndimage.generate_binary_structure(3, 1)
    )
    labels = np.asarray(labels, dtype=np.int32).ravel()
    del air
    wanted = np.zeros(int(labels.max()) + 1, dtype=bool)
    for pocket in pockets:
        wanted[pocket.label] = True
    pocket_cells = np.flatnonzero(wanted[labels])
    del labels

    # Bits pointing into a pocket, cleared. adj_bn columns are the six
    # directions in the order the voxeliser writes them, +x -x +y -y +z -z.
    is_pocket = np.zeros(total, dtype=bool)
    is_pocket[pocket_cells] = True
    strides = (ny * nz, nz, 1)
    cleared = 0
    column = 0
    for stride in strides:
        for sign in (1, -1):
            neighbour = bn + sign * stride
            inside = (neighbour >= 0) & (neighbour < total)
            into = np.zeros(bn.size, dtype=bool)
            into[inside] = is_pocket[neighbour[inside]]
            cleared += int((adj[:, column] & into).sum())
            adj[:, column] &= ~into
            column += 1
    del is_pocket
    # A wall that faced nothing but the pocket now has no air side at all. The
    # engine asserts that such a node carries no material (fdtd_data.h,
    # ``if (all_not_adj) assert(mat_bn[i]==-1)``), so it becomes static too.
    buried = ~adj.any(axis=1)
    buried_walls = int((buried & (mat != -1)).sum())
    mat = np.where(buried, -1, mat).astype(mat.dtype)
    saf = np.where(buried, 0, saf).astype(saf.dtype)

    new_bn = np.concatenate([bn, pocket_cells])
    new_adj = np.concatenate([adj, np.zeros((pocket_cells.size, 6), dtype=bool)])
    new_mat = np.concatenate([mat, np.full(pocket_cells.size, -1, dtype=mat.dtype)])
    new_saf = np.concatenate([saf, np.zeros(pocket_cells.size, dtype=saf.dtype)])
    order = np.argsort(new_bn, kind="stable")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(vox_out, "r") as source, h5py.File(out, "w") as target:
        for key in source:
            if key in ("bn_ixyz", "adj_bn", "mat_bn", "saf_bn", "Nb"):
                continue
            source.copy(key, target)
        target.create_dataset("bn_ixyz", data=new_bn[order])
        target.create_dataset("adj_bn", data=new_adj[order])
        target.create_dataset("mat_bn", data=new_mat[order])
        target.create_dataset("saf_bn", data=new_saf[order])
        target.create_dataset("Nb", data=np.int64(new_bn.size))
    return {
        "pockets_sealed": len(pockets),
        "cells_sealed": int(pocket_cells.size),
        "volume_sealed_m3": round(float(pocket_cells.size * grid["h"] ** 3), 4),
        "bits_cleared": cleared,
        "walls_made_static": buried_walls,
        "boundary_nodes_before": int(bn.size),
        "boundary_nodes_after": int(new_bn.size),
    }


def seal_in_place(vox_out: Path, *, max_cells: int = 2_000_000_000) -> dict[str, Any]:
    """Census a fetched grid and replace it with its sealed copy, or say why not.

    What the chain calls on every grid it brings home. A grid too large to
    label on this machine is left as it is and the record says so, rather
    than the chain failing after a two hour voxelisation.
    """
    vox_out = Path(vox_out)
    try:
        found = census(vox_out, max_cells=max_cells)
    except MemoryError as too_big:
        return {"sealed": False, "why": str(too_big)}
    record = found.record()
    to_seal = found.to_seal()
    if not to_seal:
        record["seal"] = {"pockets_sealed": 0}
        return record
    sealed = vox_out.with_name("vox_out.sealed.h5")
    record["seal"] = seal(vox_out, sealed, to_seal)
    sealed.replace(vox_out)
    return record


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vox_out", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="the entry's manifest.json")
    parser.add_argument("--room-min-m3", type=float, default=ROOM_MIN_M3)
    parser.add_argument("--out", type=Path, help="write the sealed grid here; omit to census only")
    parser.add_argument("--report", type=Path, help="write the census as JSON")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    labels = sorted(manifest["materials"])
    found = census(args.vox_out, room_min_m3=args.room_min_m3)
    record = found.record()
    to_seal = found.to_seal()
    print(
        f"{found.components} air components, {len(found.pockets)} coupled pockets, "
        f"{len(to_seal)} under {args.room_min_m3:g} m3 sealed, holding "
        f"{record['pockets_sealed_volume_m3']} m3; kept {len(found.pockets) - len(to_seal)}"
    )
    for pocket in to_seal[:15]:
        names = [labels[i] if 0 <= i < len(labels) else "_RIGID" for i in pocket.materials]
        print(
            f"  {pocket.cells} cells, {pocket.volume_m3:.3f} m3, span "
            f"{[round(v, 2) for v in pocket.span_m]} m, lowest mode "
            f"{pocket.lowest_mode_hz:.0f} Hz, materials {names}"
        )
    if args.out is not None:
        record["seal"] = seal(args.vox_out, args.out, to_seal)
        print(json.dumps(record["seal"]))
    if args.report is not None:
        args.report.write_text(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
