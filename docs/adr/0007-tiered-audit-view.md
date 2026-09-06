# 0007: the whole flat at 16 kHz, one room at a time

Status: accepted.

## Context

The viewer's acoustic mode draws the solver's own grid, which is the picture
that catches the defects a triangle mesh cannot: a material landed on the wrong
side of a surface, an interior left coupled to the room. That picture exists for
one bedroom and for the whole flat at 4 kHz. It does not exist for the whole
flat at **16 kHz**, and 16 kHz is the band where staircasing, thin features and
sealing are hardest and therefore most worth auditing.

The obstacle is size. The flat's 16 kHz voxelisation is **1 089 464 499 boundary
nodes** on an 11 549 x 1 415 x 9 035 grid at 2.043 mm. Merged losslessly it is
about 28 M quads and 2.2 GB. No browser holds that.

Four rented attempts at building it failed. Their common shape was to narrow a
constant -- an `int64` to an `int32`, a whole read to a chunked one -- and rent a
larger machine, against two costs that are proportional to the **lattice** and
not to the picture:

- `surface_of` allocated the block lattice as an `int16` volume. That is 1.14 GB
  for one bedroom at 4.09 mm blocks, 9.2 GB for the same bedroom at 2.04 mm and
  **295 TB for the flat**. Named as undone in the code since W25.
- `_bin_voxels` searched for the block size by calling `np.unique` on the full
  node key array once per candidate: about five sorts of 8.7 GB, which is where
  7 h 37 at 15 per cent CPU with 34.4 GB of swap went.

## Decision

**Three changes, and the third is the one the user agreed to.**

**The merge is sparse.** Each slice is meshed from the cells it actually holds
rather than from the plane they sit in, so the cost follows the blocks. Measured
on the bedroom's own 16 kHz blocks: identical quads, 22.1 s to 6.0 s, and the
lattice-sized allocation gone. The dense mesher is kept in the test suite as the
oracle it now is.

**The block size is given, not searched.** At the grid's own step a block *is* a
node, so the grouping is skipped entirely rather than made cheaper. Where a
budget is still given, each coarser span is counted from the finer span's own
unique blocks, so one sort answers for all of them.

**The picture is tiered by room, not thinned.** Every room draws at 8.17 mm --
the 4 kHz cell -- and the room the reader is standing in draws at the solver's
own 2.043 mm. Both tiers come from the *same* voxelisation.

Three consequences follow and each was measured rather than assumed.

- **A room is the everyday one.** HSSD splits the wardrobe from the bedroom it
  opens into; seven of this scene's nineteen interior regions are such closets.
  They are folded into the room their **doorway** reaches, read out of the stage
  mesh by `apartment.find_doorways`. Shared boundary is the obvious criterion
  and it is the wrong one: W34 measured the two disagreeing on five of seven,
  because a closet shares 2.10 m with the room behind it and 2.10 m with the
  room in front. The derived mapping reproduces W34's stated one exactly, and
  gives **12 rooms**, not the 13 W34 records: that count included the garage
  twice.
- **The room joins the merge key.** Without it a rectangle spans a shared wall
  and belongs to neither room. With it the rectangle stops at the partition,
  which costs 0.6 per cent more quads at the grid's own step and 2.5 per cent at
  the coarse one, and the total face area is unchanged to float32 rounding.
- **A room too heavy for one file is cut into tiles, and the cut is applied to
  the finished quads.** Cutting the block set first would leave each piece's
  edge blocks unable to see their neighbours, so they would emit faces the grid
  does not have. Assigning a merged quad to a tile by its centroid cannot.

## What this rests on, and what it costs

The partition is asserted **total**: the per-room node counts sum to the grid's
own, on every grid built. The tiering is asserted **exact**: the summed face
area of the per-room payloads equals the area of one untiered merge of the same
grid, which is the assertion that separates merging from dropping.

The halo each room is built with is `2 * span` cells, and that figure is
measured rather than reasoned. At `span + 1` the coarse tier came out 0.068 m2
heavy, 0.0024 per cent -- a block anchored on the room's edge runs `span` cells
outward and the block beyond it another `span`, so at `span + 1` nothing said
the face between them was hidden.

## What was rejected

**Drawing the other rooms from the 4 kHz voxelisation.** It is free, the payload
already exists, and it puts a seam at every room boundary: two voxelisations
staircase differently, so the join would be an artefact of the mixing and a
reader auditing geometry would have to learn to ignore it. Aggregating the
16 kHz grid by four gives the same 8.17 mm cell out of one grid.

**Renting.** The plan budgeted 1.5 to 2.5 h on a 64 GB machine and about
1.50 USD. With the sparse merge and the per-room split the whole flat at 16 kHz
builds in about **13 minutes** on the laptop, peaking at 16 GB, in per-room steps
of under three minutes. Renting a machine to run an algorithm that now fits is
the mistake this record exists to close.

**A shard file per room.** One pass over the grid writing 18 GB of scratch, then
a merge per room. It saves about eight minutes of re-reading and costs a scratch
format, a resume protocol and 18 GB. The entry re-reads at 1.9 GB/s measured, so
twelve scans are 3.3 minutes.
