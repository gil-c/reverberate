"""What the low band was computed on, kept beside the pack: ``as_computed.npz``.

A pack holds responses. It does not hold the staircase the wave solver made
of the dwelling, the nodes each listening array really sampled, nor the
nodes a source was spread on, and the voxel grid they come from is tens of
gigabytes on the machine that is destroyed when the trace ends. The audit
must show all three as they were used (``reverberate.viz.computed_grid``),
so the trace writes them here, small enough to come home with the pack:

- **the grid as the solver read it**: one bit a node for the air a source
  can reach, found by :func:`reverberate.wave.lowband.problem.reached_nodes`
  from the nodes the sources are placed on, which is how the solver cuts its
  own problem; one bit a node for the reached boundary nodes; and for each
  of those, in ascending order, the six links it may read and its material.
  Sixty-three million nodes are 8 MB as bits before deflation, and a
  dwelling's bits are long runs: the low grid of hssd_0076 to 1 kHz, 19.5
  million nodes of which 1.2 million reached boundary nodes, is 0.3 MB;
- **the grid's place**: origin, step, shape, and the engine's axis order,
  so a node's position is exact;
- **the arrays**: per listening cell the position asked for, the node the
  expansion is about and every node sampled;
- **the sources**: per solved position the eight nodes it is spread on and
  their weights (:func:`reverberate.wave.comms.interp_weights`).

Everything is in the scene's frame and the scene's axis order: a node is
``origin + h * (ix, iy, iz)`` and its flat index ``(ix * ny + iy) * nz + iz``.
The engine's own order (:func:`reverberate.wave.comms.transpose_order`) is
recorded and undone here once, so no reader has to.

A face centred grid is not recorded (its nodes have twelve links and a fold);
the archive then says why and still holds the arrays and the sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DIRECTIONS",
    "NAME",
    "SCHEMA",
    "AsComputed",
    "GridRecord",
    "grid_record",
    "read_as_computed",
    "save_as_computed",
    "snap_arrays",
    "snap_sources",
    "write_as_computed",
]

#: The archive's name beside ``pack.h5``.
NAME = "as_computed.npz"
SCHEMA = "reverberate.as-computed"
VERSION = 1

#: The six links of a Cartesian node, in the order of the adjacency bits.
DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@dataclass(frozen=True)
class GridRecord:
    """A Cartesian voxel grid as the low band solver read it, in the scene's axes."""

    key: str
    #: ``[3]``: the position of node ``(0, 0, 0)``, m.
    origin_m: np.ndarray
    step_m: float
    time_step_s: float
    #: Engine axis ``k`` is scene axis ``order[k]``.
    order: tuple[int, int, int]
    #: ``[nx, ny, nz]`` bool: the nodes a source can make other than zero.
    reached: np.ndarray
    #: Reached boundary nodes, ascending flat index, with the links each may
    #: read (bit ``d`` of :data:`DIRECTIONS`) and its material, ``-1`` rigid.
    boundary: np.ndarray
    adjacency: np.ndarray
    material: np.ndarray
    labels: tuple[str, ...] = ()
    #: How many nodes the reach was started from.
    seeds: int = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.reached.shape
        return int(nx), int(ny), int(nz)

    @property
    def nodes(self) -> int:
        return int(self.reached.size)

    def axis(self, which: int) -> np.ndarray:
        """The node coordinates along one scene axis."""
        return np.asarray(self.origin_m[which] + self.step_m * np.arange(self.shape[which]))

    def subs(self, flat: np.ndarray) -> np.ndarray:
        """``[n, 3]`` node subscripts of flat indices."""
        return np.stack(np.unravel_index(np.asarray(flat, dtype=np.int64), self.shape), axis=1)

    def positions(self, flat: np.ndarray) -> np.ndarray:
        """``[n, 3]`` node positions of flat indices, m."""
        return np.asarray(self.origin_m[None, :] + self.step_m * self.subs(flat))

    def flat_of(self, positions: np.ndarray) -> np.ndarray:
        """The flat index of the node at each position; a position off the lattice is refused."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        subs = np.rint((positions - self.origin_m[None, :]) / self.step_m).astype(np.int64)
        if (subs < 0).any() or (subs >= np.asarray(self.shape)[None, :]).any():
            raise ValueError("a node lies outside the grid")
        back = self.origin_m[None, :] + self.step_m * subs
        if not np.allclose(back, positions, rtol=0.0, atol=1e-6 * self.step_m):
            raise ValueError("a position is not a node of the grid")
        return np.asarray(np.ravel_multi_index(tuple(subs.T), self.shape), dtype=np.int64)

    def engine_index(self, flat: np.ndarray) -> np.ndarray:
        """The engine's flat index of nodes given by this record's."""
        subs = self.subs(flat)
        dims = [self.shape[axis] for axis in self.order]
        moved = [subs[:, axis] for axis in self.order]
        return np.asarray((moved[0] * dims[1] + moved[1]) * dims[2] + moved[2], dtype=np.int64)


def grid_record(
    entry_dir: Path | str, sources: np.ndarray, *, key: str = "", labels: tuple[str, ...] = ()
) -> GridRecord:
    """The grid of a cache entry as the solver cuts it for ``sources``, ``[n, 3]`` in metres.

    The reach is the solver's own (:func:`reached_nodes` over the entry's
    adjacency, started at the nodes the sources are spread on). Raises
    ``NotImplementedError`` on a face centred grid.
    """
    from reverberate.wave.comms import engine_indices, interp_weights, load_grid, transpose_order
    from reverberate.wave.lowband.problem import _copies, reached_nodes, read_entry

    entry_dir = Path(entry_dir)
    arrays = read_entry(entry_dir)
    if arrays.fcc_flag:
        raise NotImplementedError("a face centred grid is not recorded: twelve links and a fold")
    grid = load_grid(entry_dir)
    order = [int(v) for v in transpose_order(grid.shape)]
    if tuple(int(grid.shape[axis]) for axis in order) != tuple(arrays.shape):
        raise ValueError(f"{entry_dir}: the engine's shape is not its grid's, rotated")
    sources = np.asarray(sources, dtype=float).reshape(-1, 3)
    seeds = np.concatenate(
        [engine_indices(interp_weights(p, grid)[1], grid) for p in sources]
        or [np.zeros(0, dtype=np.int64)]
    )
    ex, ey, ez = arrays.shape
    offsets = np.array([(dx * ey + dy) * ez + dz for dx, dy, dz in DIRECTIONS], dtype=np.int64)
    bn = np.asarray(arrays.bn_ixyz, dtype=np.int64)
    bits = np.zeros(bn.size, dtype=np.uint16)
    for d in range(6):
        bits |= arrays.adj_bn[:, d].astype(np.uint16) << np.uint16(d)
    adjacency = np.full(arrays.points, 0xFFFF, dtype=np.uint16)
    adjacency[bn] = bits
    reached = reached_nodes(
        arrays.shape, offsets, adjacency, _copies(arrays.shape, arrays.fcc_flag), seeds
    )
    del adjacency
    # From the engine's axes to the scene's: engine axis k is scene axis order[k].
    back = [order.index(axis) for axis in range(3)]
    scene_reached = np.ascontiguousarray(reached.reshape(arrays.shape).transpose(back))
    kept = reached[bn]
    engine_subs = np.unravel_index(bn[kept], arrays.shape)
    flat = np.ravel_multi_index(tuple(engine_subs[back[axis]] for axis in range(3)), grid.shape)
    scene_bits = np.zeros(int(kept.sum()), dtype=np.uint8)
    held = arrays.adj_bn[kept]
    for axis in range(3):
        for sign in range(2):
            scene_bits |= held[:, 2 * back[axis] + sign].astype(np.uint8) << np.uint8(
                2 * axis + sign
            )
    ascending = np.argsort(flat, kind="stable")
    if not labels:
        manifest = entry_dir / "manifest.json"
        if manifest.is_file():
            named = json.loads(manifest.read_text()).get("materials", {})
            labels = tuple(sorted(named)) if isinstance(named, dict) else ()
    return GridRecord(
        key=str(key or entry_dir.name),
        origin_m=np.array([grid.xv[0], grid.yv[0], grid.zv[0]], dtype=float),
        step_m=float(grid.h),
        time_step_s=float(grid.Ts),
        order=(order[0], order[1], order[2]),
        reached=scene_reached,
        boundary=np.asarray(flat[ascending], dtype=np.int64),
        adjacency=scene_bits[ascending],
        material=np.asarray(arrays.mat_bn, dtype=np.int8)[kept][ascending],
        labels=tuple(labels),
        seeds=int(seeds.size),
    )


def snap_sources(positions: np.ndarray, grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per source position the eight nodes it is spread on and their weights.

    ``grid`` is a :class:`reverberate.wave.comms.Grid`. The nodes are flat
    indices in the scene's axes, ``[n, 8]``; the weights sum to one.
    """
    from reverberate.wave.comms import interp_weights

    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    nodes = np.zeros((positions.shape[0], 8), dtype=np.int64)
    weights = np.zeros((positions.shape[0], 8))
    for row, position in enumerate(positions):
        weights[row], nodes[row] = interp_weights(position, grid)
    return nodes, weights


def snap_arrays(designs: list[Any], grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell the node its array is about and the nodes it samples.

    ``designs`` holds a :class:`reverberate.spatial.array.ArrayDesign` or
    ``None`` a cell. Returns the centres ``[cell, 3]`` (``nan`` where no
    array stands), the offsets ``[cell + 1]`` into the nodes, and the nodes
    as flat indices in the scene's axes.
    """
    origin = np.array([grid.xv[0], grid.yv[0], grid.zv[0]], dtype=float)
    shape = (int(grid.xv.size), int(grid.yv.size), int(grid.zv.size))
    centres = np.full((len(designs), 3), np.nan)
    offsets = np.zeros(len(designs) + 1, dtype=np.int64)
    nodes: list[np.ndarray] = []
    for cell, design in enumerate(designs):
        if design is not None:
            centres[cell] = design.centre
            subs = np.rint((np.asarray(design.positions) - origin[None, :]) / grid.h).astype(
                np.int64
            )
            nodes.append(np.ravel_multi_index(tuple(subs.T), shape))
        offsets[cell + 1] = offsets[cell] + (0 if design is None else int(design.count))
    held = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int64)
    return centres, offsets, held


@dataclass(frozen=True)
class AsComputed:
    """The archive, read: what the pack beside it was computed on."""

    record: dict[str, Any]
    grid: GridRecord | None
    #: Per cell: the position asked for, the array's centre (``nan`` without
    #: an array) and, where a grid was at hand, the nodes sampled.
    cell_asked: np.ndarray
    cell_centre: np.ndarray
    cell_offsets: np.ndarray
    cell_nodes: np.ndarray
    #: Per solved source position: where it is, its nodes and their weights.
    source_position: np.ndarray
    source_nodes: np.ndarray
    source_weights: np.ndarray
    path: Path | None = field(default=None, compare=False)

    def nodes_of(self, cell: int) -> np.ndarray:
        """The flat indices of a cell's array; empty where none stands or none was recorded."""
        if self.cell_offsets.size <= cell + 1:
            return np.zeros(0, dtype=np.int64)
        start, stop = int(self.cell_offsets[cell]), int(self.cell_offsets[cell + 1])
        return np.asarray(self.cell_nodes[start:stop], dtype=np.int64)


def _small(values: np.ndarray, count: int) -> np.ndarray:
    """Flat indices in four bytes where the grid allows it."""
    return np.asarray(values, dtype=np.int32 if count < 2**31 else np.int64)


def save_as_computed(
    path: Path | str,
    *,
    record: dict[str, Any],
    grid: GridRecord | None,
    cell_asked: np.ndarray,
    cell_centre: np.ndarray,
    cell_offsets: np.ndarray | None = None,
    cell_nodes: np.ndarray | None = None,
    source_position: np.ndarray | None = None,
    source_nodes: np.ndarray | None = None,
    source_weights: np.ndarray | None = None,
) -> Path:
    """Write the archive; the file appears whole."""
    path = Path(path)
    cells = int(np.asarray(cell_asked).reshape(-1, 3).shape[0])
    count = grid.nodes if grid is not None else 2**40
    arrays: dict[str, np.ndarray] = {
        "cell_asked": np.asarray(cell_asked, dtype=float).reshape(-1, 3),
        "cell_centre": np.asarray(cell_centre, dtype=float).reshape(-1, 3),
        "cell_offsets": np.asarray(
            cell_offsets if cell_offsets is not None else np.zeros(cells + 1), dtype=np.int64
        ),
        "cell_nodes": _small(cell_nodes if cell_nodes is not None else np.zeros(0), count),
        "source_position": np.asarray(
            source_position if source_position is not None else np.zeros((0, 3)), dtype=float
        ).reshape(-1, 3),
        "source_nodes": _small(
            source_nodes if source_nodes is not None else np.zeros((0, 8)), count
        ).reshape(-1, 8),
        "source_weights": np.asarray(
            source_weights if source_weights is not None else np.zeros((0, 8)), dtype=float
        ).reshape(-1, 8),
    }
    held = {"schema": SCHEMA, "version": VERSION, **record, "grid": None}
    if grid is not None:
        mask = np.zeros(grid.nodes, dtype=bool)
        mask[grid.boundary] = True
        arrays.update(
            grid_reached=np.packbits(grid.reached.reshape(-1)),
            grid_boundary=np.packbits(mask),
            grid_adjacency=np.asarray(grid.adjacency, dtype=np.uint8),
            grid_material=np.asarray(grid.material, dtype=np.int8),
        )
        held["grid"] = {
            "key": grid.key,
            "shape": list(grid.shape),
            "origin_m": [float(v) for v in grid.origin_m],
            "step_m": grid.step_m,
            "time_step_s": grid.time_step_s,
            "order": list(grid.order),
            "labels": list(grid.labels),
            "seeds": grid.seeds,
            "axes": "scene: node = origin + step * (ix, iy, iz), flat = (ix * ny + iy) * nz + iz",
            "adjacency_bits": [list(d) for d in DIRECTIONS],
        }
    arrays["record"] = np.frombuffer(json.dumps(held, sort_keys=True).encode(), dtype=np.uint8)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    partial.replace(path)
    return path


def read_as_computed(path: Path | str) -> AsComputed:
    """The inverse of :func:`save_as_computed`; a file of another schema is a ``ValueError``."""
    path = Path(path)
    with np.load(path) as held:
        data = {name: np.asarray(held[name]) for name in held.files}
    record = json.loads(bytes(data["record"]).decode())
    if record.get("schema") != SCHEMA or int(record.get("version", 0)) != VERSION:
        raise ValueError(f"{path} is not an as-computed archive of version {VERSION}")
    grid = None
    told = record.get("grid")
    if told:
        shape = tuple(int(v) for v in told["shape"])
        count = int(np.prod(shape))
        reached = np.unpackbits(data["grid_reached"], count=count).astype(bool).reshape(shape)
        boundary = np.flatnonzero(np.unpackbits(data["grid_boundary"], count=count))
        grid = GridRecord(
            key=str(told["key"]),
            origin_m=np.asarray(told["origin_m"], dtype=float),
            step_m=float(told["step_m"]),
            time_step_s=float(told["time_step_s"]),
            order=(int(told["order"][0]), int(told["order"][1]), int(told["order"][2])),
            reached=reached,
            boundary=boundary.astype(np.int64),
            adjacency=data["grid_adjacency"],
            material=data["grid_material"],
            labels=tuple(str(v) for v in told.get("labels", [])),
            seeds=int(told.get("seeds", 0)),
        )
    return AsComputed(
        record=record,
        grid=grid,
        cell_asked=data["cell_asked"],
        cell_centre=data["cell_centre"],
        cell_offsets=data["cell_offsets"],
        cell_nodes=data["cell_nodes"].astype(np.int64),
        source_position=data["source_position"],
        source_nodes=data["source_nodes"].astype(np.int64),
        source_weights=data["source_weights"],
        path=path,
    )


def write_as_computed(trace: Any) -> dict[str, Any]:
    """``as_computed.npz`` beside the pack a trace has just written; what it holds, or why not.

    Called at the end of the write stage. It reads the campaign the pairs
    were solved by (its grid entry, its arrays, its source positions) and
    never stops a trace: a failure is said and recorded, and the pack stands.
    """
    try:
        return _write(trace)
    except Exception as error:  # noqa: BLE001 - the pack is written; this is its picture
        trace.journal.say(f"as computed: not written, {error!r}"[:300])
        return {"written": False, "why": repr(error)[:300]}


def _write(trace: Any) -> dict[str, Any]:
    engine = trace.engine
    campaign = getattr(engine, "campaign", None)
    scene_cells = int(trace.asked.shape[0])
    asked = np.concatenate([trace.asked, trace.patch_cells.reshape(-1, 3)])
    record: dict[str, Any] = {
        "voxel_low_key": str(engine.voxel_low_key),
        "solver": str(engine.solver),
        "cells_in_pack": scene_cells,
        "why_no_grid": "",
    }
    target = Path(trace.out) / NAME
    entry = getattr(campaign, "entry_path", None)
    designs = getattr(campaign, "designs", None)
    if campaign is None or entry is None or designs is None:
        record["why_no_grid"] = "the low band was not solved on a grid"
        centres = np.full(asked.shape, np.nan)
        centres[:scene_cells][trace.placed] = trace.cells[trace.placed]
        save_as_computed(target, record=record, grid=None, cell_asked=asked, cell_centre=centres)
        return {"written": True, "bytes": target.stat().st_size, "grid": False}
    from reverberate.wave.comms import load_grid

    lattice = load_grid(entry)
    sources = np.asarray(campaign.sources, dtype=float).reshape(-1, 3)
    centres, offsets, nodes = snap_arrays(list(designs), lattice)
    source_nodes, source_weights = snap_sources(sources, lattice)
    grid: GridRecord | None = None
    try:
        grid = grid_record(entry, sources, key=str(engine.voxel_low_key))
    except NotImplementedError as error:
        record["why_no_grid"] = str(error)
    save_as_computed(
        target,
        record=record,
        grid=grid,
        cell_asked=np.asarray(campaign.cells, dtype=float),
        cell_centre=centres,
        cell_offsets=offsets,
        cell_nodes=nodes,
        source_position=sources,
        source_nodes=source_nodes,
        source_weights=source_weights,
    )
    size = target.stat().st_size
    trace.journal.say(
        f"as computed: {NAME}, {size / 1e6:.2f} MB"
        + (f", grid {grid.shape} with {grid.boundary.size} boundary nodes" if grid else "")
    )
    return {"written": True, "bytes": size, "grid": grid is not None}
