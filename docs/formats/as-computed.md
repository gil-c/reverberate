# What a pack was computed on: `as_computed.npz`

A scene pack (`scene-pack.md`) holds responses. It does not hold the geometry
the low band was solved on, and the voxel grid that geometry comes from stays
on the rented machine. The trace therefore writes, at the end of its write
stage and beside `pack.h5`, a small archive of what the wave solver used, and
the fetch brings it home with the pack (`reverberate.accel.bundle.HOME_ITEMS`).
The audit draws it (`reverberate.viz.computed_api`).

The archive is not part of the pack and no engine reads it. A pack without
one renders exactly as before; the audit then says what it cannot show.

Written and read by `reverberate.trace.computed`. Schema
`reverberate.as-computed`, version 1. A NumPy `.npz`, deflated.

## Frame

Everything is in the scene's frame and the scene's axis order, `y` up. A node
is `origin_m + step_m * (ix, iy, iz)` and its flat index is
`(ix * ny + iy) * nz + iz`. The engine's own axis order (descending extent,
`reverberate.wave.comms.transpose_order`) is recorded as `order` and is
already undone: engine axis `k` is scene axis `order[k]`.

## Members

| Name | Type | Shape | Content |
|---|---|---|---|
| `record` | uint8 | `[bytes]` | JSON, see below |
| `grid_reached` | uint8 | `[ceil(nodes / 8)]` | packed bits, one a node in flat order: the node can be made other than zero by a source |
| `grid_boundary` | uint8 | `[ceil(nodes / 8)]` | packed bits: the node is a reached boundary node |
| `grid_adjacency` | uint8 | `[boundary]` | per reached boundary node in ascending flat index: bit `d` set when the node may read its neighbour `d`, in the order `+x -x +y -y +z -z` |
| `grid_material` | int8 | `[boundary]` | the node's material, an index into `labels`; `-1` for a rigid node |
| `cell_asked` | float64 | `[cell, 3]` | the position each listening cell was asked at, m |
| `cell_centre` | float64 | `[cell, 3]` | the node its array is expanded about; `nan` where no array could stand |
| `cell_offsets` | int64 | `[cell + 1]` | cell `c` owns `cell_nodes[cell_offsets[c] : cell_offsets[c + 1]]` |
| `cell_nodes` | int32 | `[sampled]` | flat indices of the nodes each array sampled, in the array's order |
| `source_position` | float64 | `[source, 3]` | every source position solved, m |
| `source_nodes` | int32 | `[source, 8]` | flat indices of the eight nodes a source is spread on |
| `source_weights` | float64 | `[source, 8]` | their weights, summing to one |

The four `grid_*` members are absent when `record.grid` is `null`. Indices are
`int64` on a grid of more than `2^31` nodes.

`record` holds `schema`, `version`, `voxel_low_key`, `solver`, `cells_in_pack`
(the first cells are the pack's `/cells`, in its order; the rest are the
validation patch's), `why_no_grid` (empty, or the reason: a face centred
grid, or a low band that was not solved on a grid) and `grid`:

```json
{
 "key": "<the grid's cache key>",
 "shape": [1150, 148, 1287],
 "origin_m": [-15.77, -0.09, -17.92],
 "step_m": 0.0217905,
 "time_step_s": 3.662e-05,
 "order": [2, 0, 1],
 "labels": ["bed", "book"],
 "seeds": 24
}
```

## How the grid is read

**Reached.** The solver cuts its problem to the nodes a source can reach
(`reverberate.wave.lowband.problem.reached_nodes`): from the nodes the sources
are spread on, following who reads whom. The archive holds that set, started
from every source position of the bundle. A node that is not reached is
inside a wall, in a sealed pocket, or outside the dwelling.

**Walls are links.** A boundary node is an air node beside a surface; the
surface itself is the set of links the node may not read. A wall of the model
is often a sheet without thickness, with reached nodes on both of its sides,
so a reader must take a wall from `grid_adjacency` and never from
`grid_reached` alone.

**The absorbing layer** is not stored: it is the layer one node inside the
grid's box, whatever the dwelling.

## Size

One bit a node twice over, and two bytes a reached boundary node, before
deflation; a dwelling's bits are long runs. The low grid of hssd_0076 to
1 kHz (19.5 million nodes, 1.18 million reached boundary nodes) is 0.3 MB.
A cell's array is about a thousand nodes, four bytes each.
