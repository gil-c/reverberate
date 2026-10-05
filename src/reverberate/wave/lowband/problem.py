"""A voxelised dwelling as the solver holds it: the air a source can reach, in columns.

The engine's four files describe a box: ``Nx Ny Nz`` nodes, of which the
boundary nodes are listed with the neighbours each may read and the material
it carries. A storey is not a box. Round it lie the margin of the grid and
whatever the bounding box holds beside the dwelling, and inside it lie the
nodes buried in walls and furniture; a source in a room moves none of them.
The present engine updates them all.

Here the grid is cut to **the nodes a source can reach**. A node reads its
neighbours, a boundary node only those its adjacency allows, so from the
nodes the campaign's sources are placed on, the set of nodes that can ever
be other than zero is found by following who reads whom
(:func:`reached_nodes`). The columns (one ``x, y`` of the engine's grid, its
contiguous ``z`` whole) that hold a reached node are kept and numbered, with
one more column of zeros standing for every column that is not kept, and a
table gives each kept column its neighbours. Nothing else is stored: a
source's two fields are ``8`` bytes a kept node.

Everything the engine derives when it loads its files is derived here in
the engine's own arithmetic (``fdtd_data.h``): the stencil's coefficients,
the boundary branches' four numbers from each material's ``D E F``, the
surface factor's rescaling on the face centred grid, the nodes of the
absorbing layer just inside the box and the halo beyond it. A problem built
with ``seeds=None`` keeps the whole box, which is the engine's own problem
and what the tests compare the cut one with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.wave.lowband.scheme import Scheme, scheme_of

__all__ = [
    "EPS_SINGLE",
    "EngineArrays",
    "Problem",
    "build_problem",
    "load_problem",
    "prune_branches",
    "reached_nodes",
    "read_entry",
]

#: ``EPS`` of ``fdtd_common.h`` in single precision: it scales the centre's weight for stability.
EPS_SINGLE = 1.19209289e-07


@dataclass
class EngineArrays:
    """What the engine reads of a voxelisation, in the engine's index space."""

    shape: tuple[int, int, int]
    fcc_flag: int
    courant: float
    courant2: float
    ts: float
    h: float
    bn_ixyz: np.ndarray
    #: ``[boundary node, neighbour]`` bool: whether the node reads that neighbour.
    adj_bn: np.ndarray
    #: The material of each boundary node, ``-1`` for a rigid one.
    mat_bn: np.ndarray
    saf_bn: np.ndarray
    #: One ``[branch, 3]`` array of ``D E F`` per material.
    materials: list[np.ndarray] = field(default_factory=list)

    @property
    def scheme(self) -> Scheme:
        return scheme_of(self.fcc_flag)

    @property
    def points(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz


def read_entry(entry_dir: Path | str) -> EngineArrays:
    """The three files of a cache entry the engine reads beside its comms."""
    entry_dir = Path(entry_dir)
    with h5py.File(entry_dir / "sim_consts.h5", "r") as handle:
        courant = float(handle["l"][()])
        courant2 = float(handle["l2"][()])
        ts = float(handle["Ts"][()])
        h = float(handle["h"][()])
        fcc_flag = int(handle["fcc_flag"][()])
    with h5py.File(entry_dir / "vox_out.h5", "r") as handle:
        shape = (int(handle["Nx"][()]), int(handle["Ny"][()]), int(handle["Nz"][()]))
        bn_ixyz = np.asarray(handle["bn_ixyz"][...], dtype=np.int64)
        adj_bn = np.asarray(handle["adj_bn"][...], dtype=bool)
        mat_bn = np.asarray(handle["mat_bn"][...], dtype=np.int8)
        saf_bn = np.asarray(handle["saf_bn"][...], dtype=np.float64)
    materials = []
    with h5py.File(entry_dir / "sim_mats.h5", "r") as handle:
        for index in range(int(handle["Nmat"][()])):
            materials.append(np.asarray(handle[f"mat_{index:02d}_DEF"][...], dtype=np.float64))
    return EngineArrays(
        shape=shape,
        fcc_flag=fcc_flag,
        courant=courant,
        courant2=courant2,
        ts=ts,
        h=h,
        bn_ixyz=bn_ixyz,
        adj_bn=adj_bn,
        mat_bn=mat_bn,
        saf_bn=saf_bn,
        materials=materials,
    )


def prune_branches(
    materials: list[np.ndarray], fmax_hz: float, *, above: float
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Each material without the branches that resonate above ``above`` times ``fmax_hz``.

    A branch is a mass, a resistance and a stiffness in series, resonant at
    ``sqrt(F / D) / 2 pi``; the materials are fitted with one an octave from
    16 Hz to 16 kHz. Far under its resonance a branch is a spring, and its
    admittance ``i w / F`` is small beside the branches in the band. The
    record gives, per material, the largest change of the wall's admittance
    under ``fmax_hz``, relative to the admittance itself: what dropping them
    costs, to be read before it is used.
    """
    frequency = np.geomspace(20.0, fmax_hz, 96)
    s = 2j * np.pi * frequency
    kept, worst = [], []
    for branches in materials:
        resonance = np.sqrt(branches[:, 2] / np.maximum(branches[:, 0], 1e-300)) / (2.0 * np.pi)
        keep = resonance <= above * fmax_hz
        if not keep.any():
            keep[np.argmin(resonance)] = True

        def admittance(rows: np.ndarray) -> np.ndarray:
            d, e, f = rows[:, 0:1], rows[:, 1:2], rows[:, 2:3]
            return np.asarray(np.sum(1.0 / (d * s[None, :] + e + f / s[None, :]), axis=0))

        full = admittance(branches)
        worst.append(float(np.max(np.abs(admittance(branches[keep]) - full) / np.abs(full))))
        kept.append(np.ascontiguousarray(branches[keep]))
    return kept, {
        "above": above,
        "branches": [int(b.shape[0]) for b in kept],
        "worst_relative_admittance_change": max(worst, default=0.0),
    }


def _copies(shape: tuple[int, int, int], fcc_flag: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The halo's copies of a step, in the engine's order: ``(destination, source)`` flat indices.

    The fold's face first on the folded face centred grid, then the two
    ``z`` faces, the ``y`` faces and the ``x`` faces; each group reads what
    the groups before it wrote, as the engine's consecutive loops do.
    """
    nx, ny, nz = shape
    index = np.arange(nx * ny * nz, dtype=np.int64).reshape(nx, ny, nz)
    groups = []
    if fcc_flag == 2:
        groups.append((index[:, ny - 1, :].ravel(), index[:, ny - 2, :].ravel()))
    groups.append(
        (
            np.concatenate([index[:, :, 0].ravel(), index[:, :, nz - 1].ravel()]),
            np.concatenate([index[:, :, 2].ravel(), index[:, :, nz - 3].ravel()]),
        )
    )
    if fcc_flag == 2:
        groups.append((index[:, 0, :].ravel(), index[:, 2, :].ravel()))
    else:
        groups.append(
            (
                np.concatenate([index[:, 0, :].ravel(), index[:, ny - 1, :].ravel()]),
                np.concatenate([index[:, 2, :].ravel(), index[:, ny - 3, :].ravel()]),
            )
        )
    groups.append(
        (
            np.concatenate([index[0].ravel(), index[nx - 1].ravel()]),
            np.concatenate([index[2].ravel(), index[nx - 3].ravel()]),
        )
    )
    return groups


def _interior(shape: tuple[int, int, int]) -> np.ndarray:
    """Whether each node is one the engine updates: not on the box's outer layer."""
    nx, ny, nz = shape
    inside = np.zeros(shape, dtype=bool)
    inside[1 : nx - 1, 1 : ny - 1, 1 : nz - 1] = True
    return inside.ravel()


def _absorbing(shape: tuple[int, int, int], fcc_flag: int) -> np.ndarray:
    """``Q`` of each node: on how many faces of the box's inner layer it lies, 0 to 3.

    The engine's absorbing nodes are the layer one node inside the box. On
    the folded face centred grid the far ``y`` face is folded onto the near
    one, and the fold itself is not a face.
    """
    nx, ny, nz = shape
    q = np.zeros(shape, dtype=np.uint8)
    q[[1, nx - 2], :, :] += 1
    q[:, [1] if fcc_flag == 2 else [1, ny - 2], :] += 1
    q[:, :, [1, nz - 2]] += 1
    return np.asarray(q.ravel() * _interior(shape), dtype=np.uint8)


def reached_nodes(
    shape: tuple[int, int, int],
    offsets: np.ndarray,
    adjacency: np.ndarray,
    copies: list[tuple[np.ndarray, np.ndarray]],
    seeds: np.ndarray,
) -> np.ndarray:
    """Every node that a field started at ``seeds`` can make other than zero, as a bool mask.

    ``adjacency`` is each node's bit mask of the neighbours it reads, every
    bit set for a node of plain air. A node ``i`` reads ``i + offsets[d]``
    when its bit ``d`` is set, so the nodes reached from ``j`` in a step are
    the interior ``j - offsets[d]`` whose bit ``d`` is set; a halo node is
    reached through its copy. Followed until nothing new is reached.
    """
    points = int(np.prod(shape))
    interior = _interior(shape)
    reached = np.zeros(points, dtype=bool)
    frontier = np.unique(np.asarray(seeds, dtype=np.int64))
    reached[frontier] = True
    while frontier.size:
        found = []
        for d, offset in enumerate(offsets):
            reader = frontier - int(offset)
            reader = reader[(reader >= 0) & (reader < points)]
            reader = reader[interior[reader] & ~reached[reader]]
            reader = reader[(adjacency[reader] >> np.uint16(d)) & np.uint16(1) == 1]
            reached[reader] = True
            found.append(reader)
        for dst, src in copies:
            copied = dst[reached[src] & ~reached[dst]]
            reached[copied] = True
            found.append(copied)
        frontier = np.concatenate(found)
    return reached


@dataclass
class Problem:
    """One grid, cut to the columns its sources reach, with the engine's coefficients."""

    scheme: Scheme
    shape: tuple[int, int, int]
    fcc_flag: int
    h: float
    ts: float
    courant: float
    #: The stencil's coefficients as the engine casts them: centre, neighbour, scaled ``l^2``.
    a1: np.float32
    a2: np.float32
    sl2: np.float32
    lo2: np.float32
    l32: np.float32
    #: Kept columns: ``column_of[ix * Ny + iy]`` is the column's number or ``-1``.
    column_of: np.ndarray
    columns: np.ndarray
    #: ``[column, lateral]``: the column at each lateral offset, ``columns.size`` where none is.
    lateral: np.ndarray
    #: The lateral offsets ``(dx, dy)`` of the table, and each neighbour as ``(slot, dz)``;
    #: slot ``-1`` is the node's own column.
    lateral_offsets: tuple[tuple[int, int], ...]
    stencil: tuple[tuple[int, int], ...]
    #: ``[column, z]`` uint8: 0 not updated as air, else ``1 + Q``.
    kind: np.ndarray
    #: The halo's copies, compact indices, in the engine's order.
    copies: list[tuple[np.ndarray, np.ndarray]]
    #: Boundary nodes that are reached: compact index, column, ``z``, adjacency bits.
    bn_index: np.ndarray
    bn_column: np.ndarray
    bn_z: np.ndarray
    bn_adjacency: np.ndarray
    #: The row of each boundary node among the lossy ones, ``-1`` for a rigid node.
    bn_lossy: np.ndarray
    #: Per lossy node: its material and its scaled surface factor.
    lossy_material: np.ndarray
    lossy_ssaf: np.ndarray
    #: Per material: branches, ``[branch, 4]`` of ``b, b d, b Dh, b Fh``, and ``beta``.
    branches: np.ndarray
    quads: np.ndarray
    beta: np.ndarray
    box_nodes: int
    reached: int
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def nz(self) -> int:
        return int(self.shape[2])

    @property
    def column_count(self) -> int:
        return int(self.columns.size)

    @property
    def nodes(self) -> int:
        """Stored nodes of one field: the kept columns and the column of zeros."""
        return (self.column_count + 1) * self.nz

    @property
    def max_branches(self) -> int:
        return int(self.quads.shape[1])

    @property
    def lossy(self) -> int:
        return int(self.lossy_material.size)

    @property
    def updated(self) -> int:
        """Nodes written every step: the air and the boundary."""
        return int(np.count_nonzero(self.kind)) + int(self.bn_index.size)

    def bytes_per_source(self) -> int:
        """Card memory of one source: two fields and the boundary branches' two states."""
        return 8 * self.nodes + 8 * self.lossy * (self.max_branches + 1)

    def bytes_shared(self) -> int:
        """Card memory of the grid itself, whatever the number of sources."""
        # A node's mask on the card is two bytes, and a lossy node's own value is kept twice.
        arrays = (
            self.kind,
            self.kind,
            self.lateral,
            self.bn_column,
            self.bn_z,
            self.bn_adjacency,
            self.bn_lossy,
            self.lossy_material,
            self.lossy_ssaf,
        )
        return int(sum(a.nbytes for a in arrays))

    def compact(self, engine_index: np.ndarray) -> np.ndarray:
        """Engine flat indices as compact ones; a node of a column not kept is ``-1``."""
        engine_index = np.asarray(engine_index, dtype=np.int64)
        column = self.column_of[engine_index // self.nz]
        return np.where(column >= 0, column.astype(np.int64) * self.nz + engine_index % self.nz, -1)

    def neighbours_of(self, column: np.ndarray, z: np.ndarray) -> np.ndarray:
        """``[neighbour, node]`` compact indices of the nodes each of these nodes reads."""
        column = np.asarray(column, dtype=np.int64)
        z = np.asarray(z, dtype=np.int64)
        out = np.empty((len(self.stencil), column.size), dtype=np.int64)
        for j, (slot, dz) in enumerate(self.stencil):
            other = column if slot < 0 else self.lateral[column, slot].astype(np.int64)
            out[j] = other * self.nz + z + dz
        return out


def _coefficients(arrays: EngineArrays) -> dict[str, np.float32]:
    """``load_sim_data``'s update coefficients, single precision, in its own order."""
    scheme = arrays.scheme
    lfac = scheme.laplacian
    dsl2 = (1.0 + EPS_SINGLE) * lfac * arrays.courant2
    da1 = 2.0 - dsl2 * len(scheme.neighbours)
    da2 = lfac * arrays.courant2
    return {
        "a1": np.float32(da1),
        "a2": np.float32(da2),
        "sl2": np.float32(dsl2),
        "lo2": np.float32(0.5 * arrays.courant),
        "l32": np.float32(arrays.courant),
    }


def _materials(materials: list[np.ndarray], ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The branches' four numbers and ``beta`` per material, as ``load_sim_data`` makes them."""
    count = len(materials)
    most = max((int(m.shape[0]) for m in materials), default=0)
    quads = np.zeros((count, max(most, 1), 4), dtype=np.float32)
    beta = np.zeros(count, dtype=np.float32)
    branches = np.zeros(count, dtype=np.int8)
    for i, rows in enumerate(materials):
        branches[i] = rows.shape[0]
        for j, (d, e, f) in enumerate(np.asarray(rows, dtype=np.float64)):
            dh, eh, fh = d / ts, e, f * ts
            b = 1.0 / (2.0 * dh + eh + 0.5 * fh)
            bd = b * (2.0 * dh - eh - 0.5 * fh)
            if not (np.isfinite(b) and np.isfinite(bd)):
                raise ValueError(f"material {i} branch {j} has no finite coefficients")
            quads[i, j] = (b, bd, b * dh, b * fh)
            beta[i] = np.float32(beta[i] + np.float32(b))
    return branches, quads, beta


def build_problem(arrays: EngineArrays, seeds: np.ndarray | None = None) -> Problem:
    """The solver's problem from the engine's arrays, cut to what ``seeds`` reach.

    ``seeds`` are engine flat indices, the nodes sources are placed on. With
    ``None`` the whole box is kept and every node is updated as the engine
    updates it.
    """
    scheme = arrays.scheme
    if arrays.fcc_flag == 1:
        raise NotImplementedError("a face centred grid must be folded, as the card's engine needs")
    nx, ny, nz = arrays.shape
    points = arrays.points
    offsets = np.array([(dx * ny + dy) * nz + dz for dx, dy, dz in scheme.neighbours], np.int64)
    bn = np.asarray(arrays.bn_ixyz, dtype=np.int64)
    bits = np.zeros(bn.size, dtype=np.uint16)
    for j in range(len(scheme.neighbours)):
        bits |= arrays.adj_bn[:, j].astype(np.uint16) << np.uint16(j)
    mat_bn = np.asarray(arrays.mat_bn, dtype=np.int8)
    saf_bn = np.asarray(arrays.saf_bn, dtype=np.float64)
    if bn.size and bool((np.diff(bn) < 0).any()):
        # The grid's order, a column after another: the order a card walks the branches in.
        ascending = np.argsort(bn, kind="stable")
        bn, bits, mat_bn, saf_bn = (
            bn[ascending],
            bits[ascending],
            mat_bn[ascending],
            saf_bn[ascending],
        )
    interior = _interior(arrays.shape)
    if bn.size and not interior[bn].all():
        raise ValueError("a boundary node lies on the box's outer layer")
    copies = _copies(arrays.shape, arrays.fcc_flag)
    is_boundary = np.zeros(points, dtype=bool)
    is_boundary[bn] = True
    if seeds is None:
        reached = np.ones(points, dtype=bool)
    else:
        adjacency = np.full(points, 0xFFFF, dtype=np.uint16)
        adjacency[bn] = bits
        seeds = np.asarray(seeds, dtype=np.int64)
        if is_boundary[seeds].any():
            raise ValueError("a source is placed on a boundary node")
        reached = reached_nodes(arrays.shape, offsets, adjacency, copies, seeds)
        del adjacency

    kept_columns = np.asarray(reached.reshape(nx * ny, nz).any(axis=1))
    columns = np.flatnonzero(kept_columns)
    column_of = np.full(nx * ny, -1, dtype=np.int32)
    column_of[columns] = np.arange(columns.size, dtype=np.int32)

    laterals = tuple(
        dict.fromkeys((dx, dy) for dx, dy, _ in scheme.neighbours if (dx, dy) != (0, 0))
    )
    stencil = tuple(
        (-1 if (dx, dy) == (0, 0) else laterals.index((dx, dy)), dz)
        for dx, dy, dz in scheme.neighbours
    )
    ix, iy = columns // ny, columns % ny
    lateral = np.full((columns.size + 1, len(laterals)), columns.size, dtype=np.int32)
    for slot, (dx, dy) in enumerate(laterals):
        jx, jy = ix + dx, iy + dy
        inside = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
        other = column_of[np.where(inside, jx * ny + jy, 0)]
        lateral[: columns.size, slot] = np.where(inside & (other >= 0), other, columns.size)

    air = reached & interior & ~is_boundary
    q = _absorbing(arrays.shape, arrays.fcc_flag)
    if bool(np.any(reached[bn] & (q[bn] > 0))):
        raise NotImplementedError("a reached boundary node lies on the box's absorbing layer")
    kind_box = np.where(air, 1 + q, 0).astype(np.uint8).reshape(nx * ny, nz)
    kind = np.zeros((columns.size + 1, nz), dtype=np.uint8)
    kind[: columns.size] = kind_box[columns]

    def compact(index: np.ndarray) -> np.ndarray:
        return np.asarray(column_of[index // nz].astype(np.int64) * nz + index % nz)

    kept_copies = []
    for dst, src in copies:
        both = reached[dst] & kept_columns[src // nz]
        if both.any():
            kept_copies.append((compact(dst[both]), compact(src[both])))

    chosen = reached[bn]
    bn_kept = bn[chosen]
    material = mat_bn[chosen]
    saf = saf_bn[chosen]
    # ``ssaf_bn``: the surface factor cast to single, rescaled on the face centred grid.
    if arrays.fcc_flag > 0:
        ssaf = (np.float32(0.5 / np.sqrt(2.0)) * saf).astype(np.float32)
    else:
        ssaf = saf.astype(np.float32)
    is_lossy = material >= 0
    bn_lossy = np.full(bn_kept.size, -1, dtype=np.int32)
    bn_lossy[is_lossy] = np.arange(int(is_lossy.sum()), dtype=np.int32)
    branches, quads, beta = _materials(arrays.materials, arrays.ts)
    if is_lossy.any() and int(material.max()) >= len(arrays.materials):
        raise ValueError("a boundary node names a material the entry does not hold")
    coefficients = _coefficients(arrays)
    return Problem(
        scheme=scheme,
        shape=arrays.shape,
        fcc_flag=arrays.fcc_flag,
        h=arrays.h,
        ts=arrays.ts,
        courant=arrays.courant,
        column_of=column_of,
        columns=columns,
        lateral=lateral,
        lateral_offsets=laterals,
        stencil=stencil,
        kind=kind,
        copies=kept_copies,
        bn_index=compact(bn_kept),
        bn_column=column_of[bn_kept // nz].astype(np.int32),
        bn_z=(bn_kept % nz).astype(np.int32),
        bn_adjacency=bits[chosen],
        bn_lossy=bn_lossy,
        lossy_material=material[is_lossy].astype(np.int32),
        lossy_ssaf=ssaf[is_lossy],
        branches=branches,
        quads=quads,
        beta=beta,
        box_nodes=points,
        reached=int(np.count_nonzero(reached)),
        record={
            "scheme": scheme.name,
            "box_nodes": points,
            "box_shape": list(arrays.shape),
            "reached_nodes": int(np.count_nonzero(reached)),
            "kept_columns": int(columns.size),
            "box_columns": int(nx * ny),
            "boundary_nodes": int(bn.size),
            "boundary_nodes_reached": int(bn_kept.size),
            "lossy_nodes_reached": int(is_lossy.sum()),
            "halo_copies": int(sum(d.size for d, _ in kept_copies)),
            "branches": int(quads.shape[1]),
        },
        **coefficients,
    )


def load_problem(
    entry_dir: Path | str,
    seeds: np.ndarray | None = None,
    *,
    prune_above: float | None = None,
    fmax_hz: float | None = None,
    walls: Any = None,
    walls_file: Path | str | None = None,
    outside: str | None = None,
) -> Problem:
    """:func:`build_problem` of a cache entry, as it is unless told otherwise.

    ``prune_above`` drops the high branches. ``walls``, a
    :class:`reverberate.wave.lowband.walls.WallFit`, fits every material
    again for the band with fewer branches; the fit is kept in
    ``walls_file`` when given, so that the workers of a campaign read one
    fit and do not each make their own. ``outside`` (``"open"`` or
    ``"rigid"``) cuts off the air between the dwelling's outer walls and
    its shell (:mod:`reverberate.wave.lowband.outside`) and needs
    ``seeds``. Each changes what is solved, and is in the problem's record.
    """
    arrays = read_entry(entry_dir)
    record: dict[str, Any] = {}
    if prune_above is not None:
        if fmax_hz is None:
            raise ValueError("pruning the branches needs the band's fmax")
        arrays.materials, record["pruned"] = prune_branches(
            arrays.materials, fmax_hz, above=prune_above
        )
    if outside is not None:
        from reverberate.wave.lowband.outside import outside_air, reach_of, without

        if seeds is None:
            raise ValueError("the outside is what the sources reach beyond the walls: give seeds")
        mask, found = outside_air(reach_of(arrays, seeds), arrays.h)
        cut: dict[str, Any] = {"closure": outside, "closing_nodes": 0, "closing_m2": 0.0}
        if bool(mask.any()):
            arrays, cut = without(arrays, mask, closure=outside)
        record["outside"] = {**found, **cut}
    if walls is not None:
        from reverberate.wave.lowband.walls import refit_kept

        arrays.materials, record["walls"] = refit_kept(arrays.materials, walls, walls_file)
    problem = build_problem(arrays, seeds)
    problem.record.update(record)
    problem.record["branches"] = problem.max_branches
    return problem
