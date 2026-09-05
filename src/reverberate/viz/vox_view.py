"""The voxelisation, as points a browser can draw.

The mesh view answers "did the right geometry leave the exporter". This one
answers the question after it: "did the solver read what the exporter sent".
They are not the same question, and the gap between them is where a defect
hides -- a surface whose material landed on the wrong side, or an interior the
voxeliser left coupled to the room, look identical in a picture of triangles.

**Aggregated, because the grid is not drawable.** A bedroom at 16 kHz has 63
million boundary nodes at 2 mm. Taking one in two hundred and drawing it at
true size gives 2 mm specks thirty millimetres apart -- dust, not a wall, and
unreadable however it is shaded. So the grid is binned into coarser blocks,
each coloured by the material most of the nodes in it carry, and adjacent
blocks' hidden faces and coplanar quads of the same material are merged into
the fewest rectangles that draw the same solid (see :func:`surface_of`) --
never into fewer or coarser blocks than the binning chose.

That changes what a block means, and the change is the honest part: a block is
"this much of the grid holds boundary nodes", not "here is a node". The block
size and the node count behind it travel with the payload so the picture cannot
be read as finer than it is.

**Inert nodes are drawn, and drawn differently.** They are the ones this
project seals, and sealing stops the simulation carrying sound through a
region. Their share of the picture is the visible form of that decision.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    #: The native grid's own renderable extent per axis, in the scene's own
    #: frame: its first and last boundary node, each expanded by half a native
    #: cell. A block is aggregated as if it held a full ``cell_m`` worth of
    #: cells even along an axis the grid's shape does not divide evenly by, so
    #: without this a boundary block would draw past where the grid actually
    #: ends -- see :func:`_corners_of`, which clips to it.
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray

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

    @property
    def aggregated(self) -> bool:
        """True when a drawn cube is bigger than the grid step it stands for.

        The block size is chosen to fit a cube budget, so a fine grid is drawn
        coarser than it is: the 16 kHz bedroom is 63 430 624 nodes of 2.04 mm
        drawn as 13 917 266 blocks of 4.09 mm. That is a picture of the grid,
        not the grid, and a viewer comparing it against the mesh needs to be
        told which of the two they are looking at -- a thin object aggregated
        away reads exactly like a thin object the voxeliser missed.
        """
        return self.cell_m > self.h_m * 1.0001

    def summary(self) -> str:
        scale = (
            f", aggregated {self.cell_m / self.h_m:.0f}x from {self.h_m * 1000:.2f} mm"
            if self.aggregated
            else " at the grid's own step"
        )
        return (
            f"{self.drawn:,} cubes of {self.cell_m * 1000:.2f} mm for "
            f"{self.total_nodes:,} boundary nodes at {self.h_m * 1000:.2f} mm"
            f"{scale}, {int(self.inert.sum()):,} inert"
        )


def _commonest_material(
    inverse: np.ndarray, material: np.ndarray, found: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per block: the material most of its nodes carry, and whether it has none.

    By sorting rather than by tallying. The tally this replaces was a dense
    ``(blocks, materials)`` count, which is fine for a bedroom's thirteen
    materials at a coarse block and is **28 GB** for the flat's fifty-one at the
    grid's own step -- so it, and not the browser, was what stopped a whole flat
    being drawn at full resolution.

    Sorting on ``block * kinds + material`` puts every (block, material) pair
    together, so the run lengths *are* the counts and the longest run in each
    block is its answer. Peak memory is a handful of arrays the length of the
    node list, whatever the material count.

    Rigid nodes, ``material == -1``, are excluded from the vote: they are the
    far side of a boundary the room cannot hear. A block with nothing but rigid
    nodes has no material at all, which the second return value marks.
    """
    kinds = int(material.max()) + 1
    voting = material >= 0
    carried = np.zeros(found, dtype=np.int8)
    if not voting.any():
        return carried, np.ones(found, dtype=bool)

    pairs = inverse[voting] * kinds + material[voting].astype(np.int64)
    pairs.sort()
    # Run boundaries, hence run lengths, hence the count of each material in
    # each block -- without ever materialising the blocks-by-materials grid.
    edges = np.flatnonzero(np.diff(pairs))
    starts = np.concatenate(([0], edges + 1))
    counts = np.diff(np.concatenate((starts, [pairs.size])))
    run_block, run_material = np.divmod(pairs[starts], kinds)

    # The winner per block: order by block, then by count, then by *descending*
    # material, and take the last run of each block. The last key is what makes
    # a tie fall to the lower material index, which is what ``argmax`` over the
    # dense tally did -- without it the two disagree on every block where two
    # materials are level, which random blocks hit immediately.
    picked = np.lexsort((-run_material, counts, run_block))
    block_sorted = run_block[picked]
    last = np.flatnonzero(np.diff(block_sorted))
    last = np.concatenate((last, [block_sorted.size - 1]))
    carried[block_sorted[last]] = run_material[picked][last].astype(np.int8)

    all_rigid = np.ones(found, dtype=bool)
    all_rigid[run_block] = False
    return carried, all_rigid


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
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
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
                bounds_lo=data["bounds_lo"],
                bounds_hi=data["bounds_hi"],
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
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
            bounds_lo=cloud.bounds_lo,
            bounds_hi=cloud.bounds_hi,
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


def _read_nodes(handle: Any, total: int, chunk: int = 1 << 24) -> tuple[Any, Any, Any]:
    """``bn_ixyz``, ``mat_bn`` and the inert flag, without three copies of the grid.

    ``adj_bn`` is six bytes a node and is only ever reduced to one boolean, so
    reading it whole costs 6.5 GB on the flat at 16 kHz to produce 1.1 GB. Read
    in chunks, reduce each, and the transient is the chunk.

    ``bn_ixyz`` is read at its stored width rather than widened to int64 on the
    way in: it is 1 089 464 499 values on that grid, and the difference between
    asking h5py for int64 and letting the caller widen only where it must is
    8.7 GB of resident array.
    """
    index = np.asarray(handle["bn_ixyz"][:])
    material = np.asarray(handle["mat_bn"][:], dtype=np.int8)
    adjacency = handle["adj_bn"]
    inert = np.empty(total, dtype=bool)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        inert[start:stop] = ~np.asarray(adjacency[start:stop], dtype=bool).any(axis=1)
    return index, material, inert


def _bin_voxels(cache_dir: Path, target_cubes: int) -> VoxelCloud:
    """The binning itself. See :func:`read_voxels`."""
    from reverberate.wave.comms import transpose_order

    cache_dir = Path(cache_dir)
    with h5py.File(cache_dir / "vox_out.h5", "r") as handle:
        _, ny, nz = (int(handle[k][()]) for k in ("Nx", "Ny", "Nz"))
        h_m = float(handle["h"][()])
        total = int(handle["bn_ixyz"].shape[0])
        index, material, inert = _read_nodes(handle, total)
    with h5py.File(cache_dir / "cart_grid.h5", "r") as handle:
        axes = [np.asarray(handle[k][:], dtype=np.float64) for k in ("xv", "yv", "zv")]
    # Two conventions meet here and neither is guessable from the other. The
    # index is flat over the engine's grid with the last axis contiguous, and
    # the engine's axes are ``cart_grid``'s permuted into descending extent by
    # ``rotate_sim_data``. Undo the permutation, so what comes out is in the
    # scene's own frame and can be drawn against the mesh.
    #
    # Held as int32. A subscript is bounded by its axis, 11 549 at the worst on
    # the flat at 16 kHz, so int64 buys nothing and costs 13 GB: three of these
    # for 1 089 464 499 nodes is 26 GB as int64 and 13 as int32, against 32 GB
    # of memory. The arithmetic that builds them is still done in the index's
    # own width, one axis at a time, so nothing overflows on the way.
    order = transpose_order((axes[0].size, axes[1].size, axes[2].size))
    scene_subs: list[np.ndarray] = [np.empty(0, dtype=np.int32)] * 3
    plane = np.int64(ny) * np.int64(nz)
    for engine_axis, divisor, modulus in (
        (0, plane, None),
        (1, np.int64(nz), np.int64(ny)),
        (2, None, np.int64(nz)),
    ):
        part = index if divisor is None else index // divisor
        if modulus is not None:
            part = part % modulus
        scene_subs[int(order[engine_axis])] = part.astype(np.int32, copy=False)
        del part

    # Boundary nodes cover a surface, so halving the block size roughly
    # quadruples the count: step through powers of two rather than solving for
    # one, which would need the surface area this is being used to estimate.
    #
    # Everything below is sized by the number of *occupied* blocks, never by
    # the number of cells. A first version counted densely over the lattice,
    # which at the grid's own 2 mm is a 34 GB occupancy array and a 481 GB
    # material tally: it does not fail, it swaps, which is worse than failing.
    shape = np.array([axis.size for axis in axes], dtype=np.int64)
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
    carried, all_rigid = _commonest_material(inverse, material, found)
    block_inert = all_rigid & (inert_count >= occupancy)
    # Otherwise it takes the commonest material it actually has, ignoring the
    # rigid nodes, which are the far side of a boundary the room cannot hear.
    #
    # A block with no material nodes at all is rigid regardless of whether it
    # is also sealed: ``block_inert`` additionally demands every node have
    # lost adjacency, which a rigid-but-still-coupled block -- the very defect
    # patch 5 exists to fix -- does not satisfy. Gating on ``block_inert``
    # here would leave such a block's ``argmax`` of an all-zero row, material
    # index 0, standing as if it were commonly that material.
    block_material = carried
    block_material[all_rigid] = -1

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

    # The grid's own extent, half a native cell past its outermost nodes on
    # each side -- not derived from the aggregated block positions, which are
    # exactly the values a boundary block's own size would otherwise overshoot.
    bounds_lo = np.array([axis[0] - 0.5 * h_m for axis in axes], dtype=np.float64)
    bounds_hi = np.array([axis[-1] + 0.5 * h_m for axis in axes], dtype=np.float64)

    return VoxelCloud(
        positions=positions,
        material=block_material,
        inert=np.asarray(block_inert),
        total_nodes=total,
        cell_m=span * h_m,
        h_m=h_m,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
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
    # Computed once and threaded through rather than re-derived by each of
    # _dense_labels and _corners_of: cloud.lattice is a min/max reduction over
    # every block position, and _corners_of alone is called once per emitted
    # quad batch, so re-deriving it there repeats that reduction thousands of
    # times over a grid with thousands of slices per axis.
    origin, shape = cloud.lattice
    label = _dense_labels(cloud, origin, shape)
    # label's NO_FACE sentinel already says which cells are occupied, so a
    # second dense array built by _dense_blocks purely to answer that would be
    # a redundant lattice-sized allocation and scatter alongside this one.
    occupied = label != NO_FACE
    corners: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    quads = 0

    for axis in range(3):
        near = np.moveaxis(occupied, axis, 0)
        near_label = np.moveaxis(label, axis, 0)
        # One slice's shape is the same for every side and every i along this
        # axis, so these are allocated once per axis and overwritten in place
        # for each slice below, rather than each slice reallocating its own --
        # _greedy_quads runs roughly two sides times a thousand slices per
        # axis on the shipped grid, and its largest scratch array alone is
        # multiple megabytes.
        kind_buf = np.empty(near.shape[1:], dtype=np.int16)
        changed_buf = np.empty(near.shape[1:], dtype=bool)
        ends_buf = np.empty((near.shape[1] + 1, near.shape[2]), dtype=np.int64)
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
                kind_buf[:] = NO_FACE
                kind_buf[visible] = near_label[i][visible]
                merged = _greedy_quads(kind_buf, changed_buf, ends_buf)
                if merged.size == 0:
                    continue
                corners.append(_corners_of(cloud, axis, side, i, merged, origin))
                labels.append(np.repeat(merged[:, 4].astype(np.int16), 4))
                quads += int(merged.shape[0])

    if not corners:
        empty = np.zeros((0, 3), np.float32)
        return VoxelSurface(empty, np.zeros(0, np.uint32), np.zeros(0, np.int16), cloud, 0)
    points = np.concatenate(corners).astype(np.float32)
    base = (np.arange(quads, dtype=np.uint32) * 4)[:, None]
    index = (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).ravel()
    return VoxelSurface(points, index, np.concatenate(labels), cloud, quads)


def _greedy_quads(
    kind: np.ndarray,
    changed: np.ndarray | None = None,
    ends: np.ndarray | None = None,
) -> np.ndarray:
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

    ``changed`` (``kind.shape``, bool) and ``ends`` (``(kind.shape[0] + 1,
    kind.shape[1])``, int64) are optional scratch buffers, both fully
    overwritten before they are read. Passed in and reused rather than
    allocated fresh, a caller invoking this once per slice of a much larger
    volume -- :func:`surface_of` does, thousands of times per axis -- turns
    thousands of transient multi-megabyte allocations into one pair per axis.
    """
    width = kind.shape[0]
    if changed is None:
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
    if ends is None:
        ends = np.empty((width + 1, kind.shape[1]), dtype=np.int64)
    ends[:] = width
    rows = np.nonzero(changed)[0]
    ends[:-1][changed] = rows
    # In place: accumulate reads and writes the same (reversed) positions in
    # the same sequential order, so this is safe, and it is what keeps this
    # from being a second fresh (width + 1, height) array every call.
    np.minimum.accumulate(ends[::-1], axis=0, out=ends[::-1])
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


def _dense_labels(cloud: VoxelCloud, origin: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Material per lattice cell, with sealed folded in as its own label.

    ``origin``/``shape`` come from :attr:`VoxelCloud.lattice`, taken as
    arguments rather than read again here, so a caller looping over many
    slices of the result -- :func:`surface_of` does -- pays for that
    reduction once.
    """
    cells = np.rint((cloud.positions - origin) / cloud.cell_m).astype(np.int64)
    grid = np.full(tuple(shape), NO_FACE, dtype=np.int16)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = np.where(
        cloud.inert, np.int16(-2), cloud.material.astype(np.int16)
    )
    return grid


def _corners_of(
    cloud: VoxelCloud, axis: int, side: int, i: int, merged: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """The four corners of every merged face of one slice, all at once.

    Clipped to the grid's own extent, ``cloud.bounds_lo``/``bounds_hi``. Every
    block is sized as if it held a full ``cell_m`` worth of native cells, which
    a block at the far edge of an axis the grid's shape does not divide evenly
    by does not: its rendered quad would otherwise reach past where the grid
    actually ends. The clip is a no-op away from the boundary, where a block's
    true extent already matches ``cell_m``.

    ``origin`` is :attr:`VoxelCloud.lattice`'s, passed in rather than read
    again here: this runs once per emitted quad batch, and re-deriving it
    would repeat that reduction over every block position for every batch.
    """
    cell = cloud.cell_m
    other = [a for a in range(3) if a != axis]
    plane = origin[axis] + (i + side) * cell - 0.5 * cell
    lo_u = origin[other[0]] + merged[:, 0] * cell - 0.5 * cell
    lo_v = origin[other[1]] + merged[:, 1] * cell - 0.5 * cell
    hi_u = origin[other[0]] + merged[:, 2] * cell - 0.5 * cell
    hi_v = origin[other[1]] + merged[:, 3] * cell - 0.5 * cell
    out = np.empty((merged.shape[0], 4, 3), dtype=np.float64)
    out[:, :, axis] = np.clip(plane, cloud.bounds_lo[axis], cloud.bounds_hi[axis])
    out[:, :, other[0]] = np.clip(
        np.stack([lo_u, hi_u, hi_u, lo_u], axis=1),
        cloud.bounds_lo[other[0]],
        cloud.bounds_hi[other[0]],
    )
    out[:, :, other[1]] = np.clip(
        np.stack([lo_v, lo_v, hi_v, hi_v], axis=1),
        cloud.bounds_lo[other[1]],
        cloud.bounds_hi[other[1]],
    )
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
            (
                f"Aggregated {blocks.cell_m / blocks.h_m:.0f}x: this is a "
                f"picture of the grid, not the grid. A feature thinner than "
                f"{blocks.cell_m * 1000:.1f} mm may be here and not drawn. "
                if blocks.aggregated
                else "Drawn at the grid's own step, one cube per node. "
            )
            + f"The solver's own grid at {blocks.h_m * 1000:.2f} mm, in blocks of "
            f"{blocks.cell_m * 1000:.2f} mm. Faces between touching blocks are "
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
