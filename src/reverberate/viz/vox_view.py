"""The voxelisation, as points a browser can draw.

The mesh view answers "did the right geometry leave the exporter". This one
answers the question after it: "did the solver read what the exporter sent".
They are not the same question, and the gap between them is where a defect
hides -- a surface whose material landed on the wrong side, or an interior the
voxeliser left coupled to the room, look identical in a picture of triangles.

**Aggregated, because the grid is not drawable.** A bedroom at 16 kHz has 63
million boundary nodes at 2 mm. Taking one in two hundred and drawing it at
true size gives 2 mm specks thirty millimetres apart -- dust, not a wall, and
unreadable however it is shaded. So the grid is binned into coarser cells and
one cube is drawn per occupied cell, coloured by the material most of the nodes
in it carry.

That changes what a cube means, and the change is the honest part: a cube is
"this block of the grid holds boundary nodes", not "here is a node". The cell
size and the node count behind it travel with the payload so the picture cannot
be read as finer than it is.

**Inert nodes are drawn, and drawn differently.** They are the ones this
project seals, and sealing stops the simulation carrying sound through a
region. Their share of the picture is the visible form of that decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "VoxelCloud",
    "VoxelSurface",
    "read_surface",
    "read_voxels",
    "surface_of",
    "write_voxel_payload",
]

#: The most blocks the grid may be binned into, which is what picks the block
#: size. Smaller blocks are a finer picture and a longer first build, and the
#: merging in :func:`surface_of` is what makes the choice free of the frame
#: rate: a million triangles draws anywhere.
#:
#: Measured on this bedroom's 16 kHz grid, which is 63 430 624 boundary nodes
#: at 2.04 mm:
#:
#: ===========  =============  ==========  ============  =========
#: block        blocks         quads       triangles     build
#: ===========  =============  ==========  ============  =========
#: 16.3 mm          545 631      30 060        60 120       10 s
#:  8.2 mm        3 122 239     120 888       241 776      118 s
#:  4.1 mm       13 917 266     475 740       951 480     1088 s
#: ===========  =============  ==========  ============  =========
#:
#: Halving the block does not quite quadruple the count, which is why these
#: are measured rather than derived: the surface is not flat, and corners and
#: thin features merge less than a flat one would.
#:
#: The build is cached beside the voxelisation, so the wait is paid once per
#: grid and the block size is chosen for the picture rather than for the wait.
#:
#: Nothing is ever dropped at any of these. Every block holding boundary nodes
#: is drawn, so what a coarser block costs is where inside it the boundary
#: was, and which material won a majority in it, and nothing else.
#:
#: The grid's own 2 mm is one block per node and is not reachable yet:
#: :func:`_dense_blocks` materialises the whole lattice, 9.2 GB of labels at
#: that resolution, and the build runs into hours. Slicing it is the work that
#: unlocks native, and it is not done.
TARGET_CUBES = 20_000_000


@dataclass
class VoxelCloud:
    """Occupied blocks of the grid, with what each one mostly is."""

    #: (n, 3) float32 centres in metres, in the scene's own frame.
    positions: np.ndarray
    #: (n,) int8 material index, the commonest in the block; -1 is rigid.
    material: np.ndarray
    #: (n,) bool, true where most of the block's nodes have no adjacency left.
    #: These are the sealed insides of solid objects: present in the grid,
    #: inert in the solve.
    inert: np.ndarray
    #: Every boundary node the voxelisation holds, before aggregation.
    total_nodes: int
    #: Side of a drawn cube, in metres.
    cell_m: float
    #: Grid step in metres. ``cell_m / h_m`` is how many cells a cube spans.
    h_m: float

    @property
    def drawn(self) -> int:
        return int(self.positions.shape[0])

    @property
    def lattice(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the blocks sit on a dense lattice: its origin and its shape.

        Blocks are a sparse set of centres, and both reductions in
        :func:`surface_of` need to ask whether a neighbour exists.
        """
        low = self.positions.min(axis=0).astype(np.float64)
        high = self.positions.max(axis=0).astype(np.float64)
        shape = np.rint((high - low) / self.cell_m).astype(np.int64) + 1
        return low, shape

    def summary(self) -> str:
        return (
            f"{self.drawn:,} cubes of {self.cell_m * 1000:.0f} mm for "
            f"{self.total_nodes:,} boundary nodes at {self.h_m * 1000:.2f} mm, "
            f"{int(self.inert.sum()):,} inert"
        )


def read_voxels(cache_dir: Path, target_cubes: int = TARGET_CUBES) -> VoxelCloud:
    """Read a cached voxelisation and bin it into drawable blocks.

    The block size is derived from the node count rather than fixed, so a
    coarse grid is not thrown away and a fine one does not melt the browser.
    It is a power of two multiple of the grid step, which keeps the blocks
    aligned to the grid rather than straddling it.

    The result is kept beside the voxelisation it came from. Binning reads 890
    MB and takes fifteen seconds on a bedroom at 16 kHz, and the viewer does it
    on every start; the entry is content addressed, so a cached answer cannot
    belong to a different grid.
    """
    cached = _cached_blocks(Path(cache_dir), target_cubes)
    if cached is not None:
        return cached
    cloud = _bin_voxels(Path(cache_dir), target_cubes)
    _write_cached_blocks(Path(cache_dir), target_cubes, cloud)
    return cloud


def read_surface(cache_dir: Path, target_cubes: int = TARGET_CUBES) -> VoxelSurface:
    """The merged mesh for a cached voxelisation, built once and kept.

    Merging is the long part -- eighteen minutes at 4 mm on one bedroom -- so
    it is cached beside the voxelisation it came from, which is content
    addressed, so a cached answer cannot belong to a different grid.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"viewer_mesh_{target_cubes}.npz"
    if path.is_file():
        try:
            with np.load(path) as data:
                blocks = _cached_blocks(cache_dir, target_cubes)
                if blocks is not None:
                    return VoxelSurface(
                        corners=data["corners"],
                        index=data["index"],
                        label=data["label"],
                        blocks=blocks,
                        quads=int(data["quads"]),
                    )
        except (OSError, KeyError, ValueError):
            pass  # rebuild rather than refuse to draw
    surface = surface_of(read_voxels(cache_dir, target_cubes))
    staging = path.with_suffix(".partial.npz")
    try:
        np.savez(
            staging,
            corners=surface.corners,
            index=surface.index,
            label=surface.label,
            quads=surface.quads,
        )
        staging.replace(path)
    except OSError:
        staging.unlink(missing_ok=True)
    return surface


def _blocks_path(cache_dir: Path, target_cubes: int) -> Path:
    """Where the binned blocks live. The target is in the name because it
    decides the block size, so two targets are two different pictures."""
    return cache_dir / f"viewer_blocks_{target_cubes}.npz"


def _cached_blocks(cache_dir: Path, target_cubes: int) -> VoxelCloud | None:
    path = _blocks_path(cache_dir, target_cubes)
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return VoxelCloud(
                positions=data["positions"],
                material=data["material"],
                inert=data["inert"],
                total_nodes=int(data["total_nodes"]),
                cell_m=float(data["cell_m"]),
                h_m=float(data["h_m"]),
            )
    except (OSError, KeyError, ValueError):
        # A half written or stale file is worth rebuilding rather than
        # refusing to draw over.
        return None


def _write_cached_blocks(cache_dir: Path, target_cubes: int, cloud: VoxelCloud) -> None:
    path = _blocks_path(cache_dir, target_cubes)
    staging = path.with_suffix(".partial.npz")
    try:
        np.savez(
            staging,
            positions=cloud.positions,
            material=cloud.material,
            inert=cloud.inert,
            total_nodes=cloud.total_nodes,
            cell_m=cloud.cell_m,
            h_m=cloud.h_m,
        )
        staging.replace(path)
    except OSError:
        # A read-only or full cache is a slow viewer, not a broken one.
        staging.unlink(missing_ok=True)


def _block_keys(subs: list[np.ndarray], span: int, shape: np.ndarray) -> np.ndarray:
    """One integer per node, naming the block it falls in.

    Packed over the coarse lattice rather than hashed, so the subscripts come
    back by division; and computed on the nodes rather than on the lattice, so
    nothing here is ever the size of the grid.
    """
    coarse = -(-shape // span)
    return np.asarray(
        (subs[0] // span) * (coarse[1] * coarse[2])
        + (subs[1] // span) * coarse[2]
        + (subs[2] // span)
    )


def _bin_voxels(cache_dir: Path, target_cubes: int) -> VoxelCloud:
    """The binning itself. See :func:`read_voxels`."""
    from reverberate.wave.comms import transpose_order

    cache_dir = Path(cache_dir)
    with h5py.File(cache_dir / "vox_out.h5", "r") as handle:
        _, ny, nz = (int(handle[k][()]) for k in ("Nx", "Ny", "Nz"))
        h_m = float(handle["h"][()])
        index = np.asarray(handle["bn_ixyz"][:], dtype=np.int64)
        material = np.asarray(handle["mat_bn"][:], dtype=np.int8)
        inert = ~np.asarray(handle["adj_bn"][:], dtype=bool).any(axis=1)
    with h5py.File(cache_dir / "cart_grid.h5", "r") as handle:
        axes = [np.asarray(handle[k][:], dtype=np.float64) for k in ("xv", "yv", "zv")]

    total = int(index.size)
    # Two conventions meet here and neither is guessable from the other. The
    # index is flat over the engine's grid with the last axis contiguous, and
    # the engine's axes are ``cart_grid``'s permuted into descending extent by
    # ``rotate_sim_data``. Undo the permutation, so what comes out is in the
    # scene's own frame and can be drawn against the mesh.
    subs = [index // (ny * nz), (index // nz) % ny, index % nz]
    order = transpose_order((axes[0].size, axes[1].size, axes[2].size))
    scene_subs: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * 3
    for engine_axis, cart_axis in enumerate(order):
        scene_subs[int(cart_axis)] = subs[engine_axis]

    # Boundary nodes cover a surface, so halving the block size roughly
    # quadruples the count: step through powers of two rather than solving for
    # one, which would need the surface area this is being used to estimate.
    #
    # Everything below is sized by the number of *occupied* blocks, never by
    # the number of cells. A first version counted densely over the lattice,
    # which at the grid's own 2 mm is a 34 GB occupancy array and a 481 GB
    # material tally: it does not fail, it swaps, which is worse than failing.
    shape = np.array([axis.size for axis in axes], dtype=np.int64)
    kinds = int(material.max()) + 2  # -1 rigid, then 0..max
    span = 1
    while span < 4096:
        if np.unique(_block_keys(scene_subs, span, shape)).size <= target_cubes:
            break
        span *= 2

    # ``inverse`` numbers the occupied blocks 0..n-1, so every count that
    # follows is over blocks that exist rather than cells that might.
    unique_keys, inverse, occupancy = np.unique(
        _block_keys(scene_subs, span, shape), return_inverse=True, return_counts=True
    )
    found = unique_keys.size
    tally = np.bincount(
        inverse * kinds + (material.astype(np.int64) + 1), minlength=found * kinds
    ).reshape(found, kinds)

    # A block is sealed only when *nothing* in it carries a material and every
    # node in it has lost its adjacency. A block at a surface holds both sides
    # of it, and calling that sealed because half its nodes are would paint
    # every wall in the room the colour of the sealing.
    #
    # The adjacency is read rather than inferred from the material because the
    # two only agree after patch 5. On a grid voxelised before it, a node can
    # be rigid and still coupled to its neighbours -- which is the defect that
    # patch exists for, and the view must be able to show it rather than
    # quietly relabel it as sealed.
    inert_count = np.bincount(inverse, weights=inert, minlength=found)
    block_inert = (tally[:, 1:].sum(axis=1) == 0) & (inert_count >= occupancy)
    # Otherwise it takes the commonest material it actually has, ignoring the
    # rigid nodes, which are the far side of a boundary the room cannot hear.
    block_material = tally[:, 1:].argmax(axis=1).astype(np.int8)
    block_material[block_inert] = -1

    # Back from the packed key to a centre in the scene's own frame.
    coarse = -(-shape // span)
    ia = unique_keys // (coarse[1] * coarse[2])
    ib = (unique_keys // coarse[2]) % coarse[1]
    ic = unique_keys % coarse[2]
    # A block of ``span`` cells starting at ``idx`` holds the nodes idx to
    # idx+span-1, so its centre is half of *that* span past the first, not half
    # a block past it: with no aggregation the cube must sit on the node.
    half = 0.5 * (span - 1) * h_m
    positions = np.stack(
        [
            axes[0][np.minimum(ia * span, shape[0] - 1)] + half,
            axes[1][np.minimum(ib * span, shape[1] - 1)] + half,
            axes[2][np.minimum(ic * span, shape[2] - 1)] + half,
        ],
        axis=1,
    ).astype(np.float32)

    return VoxelCloud(
        positions=positions,
        material=block_material,
        inert=np.asarray(block_inert),
        total_nodes=total,
        cell_m=span * h_m,
        h_m=h_m,
    )


@dataclass
class VoxelSurface:
    """The blocks as a plain mesh: the fewest quads that draw the same solid."""

    #: (4q, 3) float32 corners, four per quad, in the scene's own frame.
    corners: np.ndarray
    #: (6q,) uint32 triangle indices, two triangles per quad.
    index: np.ndarray
    #: (4q,) int16 label per corner: -2 sealed, -1 rigid, 0.. material.
    label: np.ndarray
    #: What it was built from, so the payload can quote both.
    blocks: VoxelCloud
    #: How many quads survived merging.
    quads: int

    @property
    def triangles(self) -> int:
        return int(self.index.size // 3)

    def summary(self) -> str:
        cubes = self.blocks.drawn
        return (
            f"{self.quads:,} quads, {self.triangles:,} triangles for "
            f"{cubes:,} blocks of {self.blocks.cell_m * 1000:.1f} mm "
            f"({cubes * 12:,} triangles as solid cubes)"
        )


#: A face label meaning "no face here", outside the range of any material once
#: the sealed flag is folded in as -2.
NO_FACE = -99


def surface_of(cloud: VoxelCloud) -> VoxelSurface:
    """Turn blocks into the fewest quads that draw exactly the same solid.

    Two reductions, and neither changes the picture by one pixel.

    **Hidden faces are not drawn.** A block touching another hides the face
    between them. Measured on this bedroom at 2 mm, six faces per block is
    369 M triangles and the visible ones are 132 M: most of a voxel shell's
    faces face another voxel.

    **Coplanar faces of the same label merge into rectangles.** A floor is one
    quad, not one per cell. This is where the axis that costs most is cheapest
    to fix: the same grid has 30 M faces on the vertical axis against 16 M on
    x, because floor and ceiling are perpendicular to it, and those are exactly
    the two surfaces that collapse to a handful of quads.

    The staircase survives both, on purpose. It is what the solver works on,
    and smoothing it here would draw a room the wave equation was not solved
    in.
    """
    occupied = _dense_blocks(cloud)
    label = _dense_labels(cloud)
    corners: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    quads = 0

    for axis in range(3):
        near = np.moveaxis(occupied, axis, 0)
        near_label = np.moveaxis(label, axis, 0)
        for side in (0, 1):
            for i in range(near.shape[0]):
                behind = None
                if side == 0 and i > 0:
                    behind = near[i - 1]
                elif side == 1 and i + 1 < near.shape[0]:
                    behind = near[i + 1]
                visible = near[i] if behind is None else near[i] & ~behind
                if not visible.any():
                    continue
                merged = _greedy_quads(np.where(visible, near_label[i], NO_FACE))
                if merged.size == 0:
                    continue
                corners.append(_corners_of(cloud, axis, side, i, merged))
                labels.append(np.repeat(merged[:, 4].astype(np.int16), 4))
                quads += int(merged.shape[0])

    if not corners:
        empty = np.zeros((0, 3), np.float32)
        return VoxelSurface(empty, np.zeros(0, np.uint32), np.zeros(0, np.int16), cloud, 0)
    points = np.concatenate(corners).astype(np.float32)
    base = (np.arange(quads, dtype=np.uint32) * 4)[:, None]
    index = (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).ravel()
    return VoxelSurface(points, index, np.concatenate(labels), cloud, quads)


def _greedy_quads(kind: np.ndarray) -> np.ndarray:
    """Merge one slice of face labels into the fewest rectangles.

    Returns ``(n, 5)`` of ``u0, v0, u1, v1, label``, ends exclusive.

    Two passes, and the second is what makes a floor one rectangle rather than
    one strip per row: runs along u are found first, then runs spanning exactly
    the same u with the same label in consecutive v are merged. A full 2D
    greedy mesher does no better on axis-aligned geometry, which is what a
    voxel grid is made of.

    Vectorised over the whole slice: the obvious loop over rows and runs is
    twenty times slower, and the grid's own resolution has a thousand slices
    per axis.
    """
    width = kind.shape[0]
    changed = np.empty_like(kind, dtype=bool)
    changed[0] = True
    changed[1:] = kind[1:] != kind[:-1]
    run_u, run_v = np.nonzero(changed)
    labels = kind[run_u, run_v]
    keep = labels != NO_FACE
    run_u, run_v, labels = run_u[keep], run_v[keep], labels[keep]
    if run_u.size == 0:
        return np.zeros((0, 5), dtype=np.int64)

    # A run ends at the next change in its own column, or at the edge.
    ends = np.full((width + 1, kind.shape[1]), width, dtype=np.int64)
    rows = np.nonzero(changed)[0]
    ends[:-1][changed] = rows
    ends = np.minimum.accumulate(ends[::-1], axis=0)[::-1]
    run_end = ends[run_u + 1, run_v]

    # Runs merge down v when start, end and label match and v is consecutive,
    # so sorting by those three puts every mergeable chain together.
    order = np.lexsort((run_v, labels, run_end, run_u))
    u0, v0, u1, lab = run_u[order], run_v[order], run_end[order], labels[order]
    breaks = np.empty(u0.size, dtype=bool)
    breaks[0] = True
    breaks[1:] = (
        (u0[1:] != u0[:-1]) | (u1[1:] != u1[:-1]) | (lab[1:] != lab[:-1]) | (v0[1:] != v0[:-1] + 1)
    )
    start = np.flatnonzero(breaks)
    stop = np.r_[start[1:], u0.size] - 1
    return np.stack([u0[start], v0[start], u1[start], v0[stop] + 1, lab[start]], axis=1)


def _dense_blocks(cloud: VoxelCloud) -> np.ndarray:
    """The blocks back on a dense lattice, so neighbours can be looked up."""
    origin, shape = cloud.lattice
    cells = np.rint((cloud.positions - origin) / cloud.cell_m).astype(np.int64)
    grid = np.zeros(tuple(shape), dtype=bool)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return grid


def _dense_labels(cloud: VoxelCloud) -> np.ndarray:
    """Material per lattice cell, with sealed folded in as its own label."""
    origin, shape = cloud.lattice
    cells = np.rint((cloud.positions - origin) / cloud.cell_m).astype(np.int64)
    grid = np.full(tuple(shape), NO_FACE, dtype=np.int16)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = np.where(
        cloud.inert, np.int16(-2), cloud.material.astype(np.int16)
    )
    return grid


def _corners_of(cloud: VoxelCloud, axis: int, side: int, i: int, merged: np.ndarray) -> np.ndarray:
    """The four corners of every merged face of one slice, all at once."""
    origin, _ = cloud.lattice
    cell = cloud.cell_m
    other = [a for a in range(3) if a != axis]
    plane = origin[axis] + (i + side) * cell - 0.5 * cell
    lo_u = origin[other[0]] + merged[:, 0] * cell - 0.5 * cell
    lo_v = origin[other[1]] + merged[:, 1] * cell - 0.5 * cell
    hi_u = origin[other[0]] + merged[:, 2] * cell - 0.5 * cell
    hi_v = origin[other[1]] + merged[:, 3] * cell - 0.5 * cell
    out = np.empty((merged.shape[0], 4, 3), dtype=np.float64)
    out[:, :, axis] = plane
    out[:, :, other[0]] = np.stack([lo_u, hi_u, hi_u, lo_u], axis=1)
    out[:, :, other[1]] = np.stack([lo_v, lo_v, hi_v, hi_v], axis=1)
    return out.reshape(-1, 3)


def write_voxel_payload(
    surface: VoxelSurface, labels: list[str], target: Path
) -> dict[str, object]:
    """Write the mesh where the browser can fetch it, and describe it.

    Binary rather than JSON: 476 000 quads is 38 MB of typed arrays and about
    ten times that as text, and the browser wants typed arrays at the end of it
    either way.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "voxels.f32").write_bytes(surface.corners.tobytes())
    (target / "voxels_index.u32").write_bytes(surface.index.tobytes())
    (target / "voxels_label.i16").write_bytes(surface.label.tobytes())
    blocks = surface.blocks
    sealed_quads = int(np.count_nonzero(surface.label[::4] == -2))
    record = {
        "quads": surface.quads,
        "triangles": surface.triangles,
        "blocks": blocks.drawn,
        "total_nodes": blocks.total_nodes,
        "cell_m": blocks.cell_m,
        "h_m": blocks.h_m,
        "sealed_quads": sealed_quads,
        "labels": labels,
        "corners_url": "voxels.f32",
        "index_url": "voxels_index.u32",
        "label_url": "voxels_label.i16",
        "note": (
            f"The solver's own grid at {blocks.h_m * 1000:.2f} mm, in blocks of "
            f"{blocks.cell_m * 1000:.1f} mm. Faces between touching blocks are "
            "not drawn and coplanar faces of the same material are merged into "
            "rectangles, so this is the same solid as "
            f"{blocks.drawn * 12:,} cube triangles in {surface.triangles:,}. "
            "Nothing is dropped: the merged area matches the visible face area "
            "exactly. Pink is sealed -- blocks holding no material at all, the "
            "insides of solid objects, which the solver carries no sound "
            "through."
        ),
    }
    (target / "voxels.json").write_text(json.dumps(record))
    return record
