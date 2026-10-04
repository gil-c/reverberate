# How far a listening cell can be read from, under 1 kHz

Date: 2026-10-04

Status: measured on the data the project owns; three points are left open
and are listed at the end with the solve that would settle each. Written for
lot L4 of ADR 0016. The code is `reverberate.spatial.translate`; the
measurement is `python -m reverberate.experiments.w44_interpolation
line-channels`, run on the dense line of hssd_0076
(`data/runs/w44_clarify_interpolation/line_home2/pulled/field/S1.h5`, 341
points 2 cm apart, source S1) with that run's `plan.json` and the mirror's
`scene.npz` of `w42_gpu_hssd_0076`.

## What is read, and how the error is counted

A listening cell holds the order 7 expansion of the wave field about its
centre. A head that is not on a cell reads one cell's expansion re-expanded
at the head (`translation_operator`), or two cells' fused
(`fusion_operator`). Both give all 64 channels; W44 had measured the
pressure, channel 0, alone.

The error is the energy of the difference over the energy of the solved
response, over the 50 ms after the onset, per octave from 125 Hz to 1 kHz.
For 64 channels it is read **on the head's sphere**: each channel is
weighted by `j_n(k a)`, `a = 0.10 m`, which is the pressure the expansion
gives on a sphere of the head's radius. Unweighted, N3D counts every degree
alike and the figure is ruled by degrees a head does not hear: on a tenth of
the line's points the channels of degrees 4 to 7 hold 20 to 48 dB more
energy than channel 0 under 500 Hz, the near field of the source as the
encoder's 60 dB gain cap lets it through. The worst octave of the worst case
is what is read below, with the ninth decile beside it.

## The floor was the position, not the solver

W44 found a floor of -20 to -30 dB at every spacing, 2 cm included, and
left it unexplained. Three things make it, and the first two are not the
solver's noise:

1. **An array stands on a node of its grid, and a field records the point
   asked for.** The line's points are 2 cm apart and the low grid's nodes
   32.7 mm: 341 points are 209 distinct expansions under 800 Hz, each up to
   17 mm (median 9.5 mm) from where the field puts it. Reading one cell a
   few centimetres away, channel 0, cells 1.2 m or more from the source,
   median then worst case of the worst octave:

   | offsets taken from | distance | 125 | 250 | 500 | 1k | worst |
   | --- | --- | --- | --- | --- | --- | --- |
   | the field's positions | 0.020 m | -32.0 | -27.3 | -20.5 | -22.1 | -16.1 |
   | the field's positions | 0.040 m | -36.3 | -31.8 | -25.4 | -28.9 | -14.1 |
   | the arrays' own centres (`plan.json`) | 0.033 m | -46.2 | -46.2 | -46.2 | -40.5 | -30.5 |

   Two adjacent points that share a centre are the same solve at the same
   nodes.
2. **A field levels each point's low band onto its mid band, point by
   point.** Those pairs of points with one centre still differ under 707 Hz,
   by one scalar: -45.3 dB in the median and -31.2 dB at worst, the same in
   every octave. It is the per point level ratio of the three band assembly.
   A campaign that solves the low band alone has no such scalar.
3. **On the lattice, a quarter of the low band's arrays are not where the
   point is.** The low band's array is 0.39 m in radius on the grid to
   1 kHz and its ball must be free air, so the plan moves it up to 0.40 m
   to find room: of the 437 points of `w42_gpu_hssd_0076`, 115 have their
   low band expanded more than 3 cm from the point, the ninth decile is
   19.6 cm and the worst 41.5 cm. Those are the points with the least
   clearance. ADR 0016's reading of the lattice, "the ninth decile is the
   clearance", is in part this: under 800 Hz those points were translated
   from somewhere else. The same field is what the page plays.

The 18 points the line shares with the lattice match the earlier field to
2e-6 of the peak (ADR 0016): the solver repeats itself. What is left with
the true centres, far from the source, is -22 to -25 dB at worst and -26 to
-30 dB at the ninth decile under 500 Hz at every distance from 10 to 39 cm,
the same figure in the three octaves, which is the mark of a scalar, item
2, and not of a translation.

## One cell: `k d` up to 3.6

Cells 1.2 m or more from the source, 61 to 128 cases a distance; their
clearance to the nearest surface is 0.51 to 1.10 m. All 64 channels on the
head's sphere, worst octave: ninth decile, then worst case. `k d` is at
1 kHz.

| distance | `k d` | one cell, p90 | one cell, worst | two cells, p90 | two cells, worst |
| --- | --- | --- | --- | --- | --- |
| 0.033 m | 0.6 | -37.3 | -32.4 | -37.5 | -33.9 |
| 0.065 m | 1.2 | -32.7 | -28.4 | -36.8 | -32.3 |
| 0.098 m | 1.8 | -30.1 | -25.4 | -32.8 | -29.4 |
| 0.131 m | 2.4 | -28.3 | -25.5 | -29.4 | -27.5 |
| 0.163 m | 3.0 | -26.1 | -22.9 | -28.6 | -25.9 |
| 0.196 m | 3.6 | -21.7 | **-21.1** | -27.6 | -24.8 |
| 0.229 m | 4.2 | -16.7 | -15.6 | -26.4 | -24.7 |
| 0.261 m | 4.8 | -13.9 | -11.5 | -26.9 | -23.4 |
| 0.294 m | 5.4 | -10.6 | -10.2 | -27.6 | **-24.1** |
| 0.327 m | 6.0 | -8.3 | -7.7 | -26.9 | -23.4 |
| 0.392 m | 7.2 | -5.6 | -5.3 | -11.2 | -8.1 |

"Two cells" are one either side of the head on the line, each at the
distance of the row.

- **One cell holds -20 dB up to 0.20 m, `k d = 3.6`**, as the reading of the
  pressure in ADR 0016 said (3.5). The pressure alone goes a little further:
  -22.6 dB at 0.229 m and -18.1 dB at 0.261 m.
- **The upper degrees fail first.** At 0.196 m and 1 kHz the error of one
  cell is carried by degrees 1 to 3 (-28, -25 and -26 dB of the whole, at
  worst) and not by degree 0 (-23 dB); from degree 5 up the error is under
  -42 dB because the head's sphere holds nothing there. An output degree `n`
  needs the input's degrees up to about `n + k d`, and the input stops at 7.
- **Two cells either side hold -23 dB out to 0.33 m each, `k d = 6.0`**, and
  fail at 0.39 m (-8 dB at 1 kHz on 64 channels, though the pressure still
  reads -21 dB there, which is what W44 saw).
- **The two need not be at one distance, and must be either side.** Cells
  at 0.098 m and 0.294 m either side of the head: -26.9 dB at worst, where
  the near one alone gives -26.6 dB. Two on the same side, at 0.098 m and
  0.196 m: -24.7 dB, worse than the near one alone. A second cell is taken
  only from the other side.

## The source: a share of its distance, not a radius

The pack format let a cell serve a head anywhere inside its free ball. The
line says otherwise. Every case of every distance up to 0.20 m, grouped by
the distance read over the distance from the cell to the source; worst
octave, ninth decile then worst case:

| `d / r_s` | cases | one cell, 64 channels | two cells, 64 channels | two cells, pressure |
| --- | --- | --- | --- | --- |
| under 0.05 | 232 | -30.5, -24.6 | -34.2, -27.8 | -32.8, -27.8 |
| 0.05 to 0.10 | 269 | -22.5, -14.7 | -28.1, -21.6 | -28.5, -24.8 |
| 0.10 to 0.15 | 110 | -21.3, -17.3 | -27.2, **-22.6** | -29.3, -25.4 |
| 0.15 to 0.20 | 58 | -14.6, -8.3 | -20.2, -13.0 | -31.0, -28.5 |
| 0.20 to 0.25 | 36 | -11.6, -6.9 | -15.6, -12.4 | -26.4, **-25.6** |
| 0.25 to 0.33 | 37 | -5.5, -1.6 | -10.2, -7.2 | -21.5, -16.6 |
| 0.33 to 0.50 | 40 | +5.1, +8.8 | +0.2, +5.9 | -4.8, +1.2 |

- **64 channels need the distance read under 0.15 of the source's
  distance** (two cells), the pressure under 0.25. Inside the free ball but
  past those shares the error is that of no prediction at all.
- **One cell alone is the weaker mode near a source**: under the same share
  of 0.15 its worst case is -14.7 dB and its ninth decile -21 dB. The
  trace should give a head two cells wherever a source is within 1.2 m.
- The reason is the near field: about a centre `r_s` from a point source the
  coefficients of degree `n` grow as `h_n(k r_s)`, so the cut at degree 7
  loses a share that grows with `d / r_s`, and the expansions themselves are
  capped by the encoder there.
- **A near voice is 0.5 to 1.5 m from the head.** At 0.5 m a cell serves
  within 7.5 cm; at 1 m within 15 cm; at 2 m within the 0.30 m that two
  cells allow anyway. ADR 0016's risk 7 asked for the dense cells this needs:
  **cells 0.15 m apart along the listener's path** hold every source from
  0.50 m, the recipe's minimum, out.

## Surfaces: nothing under 0.51 m was measured

Every cell of the line is 0.51 m or more from the nearest surface, median
0.61 m, and the far cells above hold at 0.294 m: a share of 0.58 of the
clearance. So a surface is not what a source is, and a cell may be read
from half its clearance. **Nearer a surface than 0.51 m the line says
nothing**, and the lattice cannot say it either, because its low band arrays
were moved where the clearance is small (item 3 above).

## What the library does with it

`reverberate.spatial.translate`, one offset convention throughout,
`d = target - centre`:

| constant | value | from |
| --- | --- | --- |
| `TRANSLATE_WITHIN_M` | 0.20 m | one cell, -21.1 dB at 0.196 m |
| `FUSE_WITHIN_M` | 0.30 m | two cells, -24.1 dB at 0.294 m |
| `SOURCE_SHARE` | 0.15 | two cells, -22.6 dB between 0.10 and 0.15 |
| `SURFACE_SHARE` | 0.5 | held at 0.58 where the clearance is 0.51 m or more |

`serving_radius_m` is the smaller of the two shares, and `choose_cells`
gives the pack's `mode`: exact on a cell; fused from the nearest cell that
may serve and the nearest that may on the other side of the head;
translated from one cell within 0.20 m when there is no second; an error
naming the head when no cell may serve. `cell_stride` says how many cells of
a line a source may skip: a source 2 m away reads every fourth cell of a
line at 0.15 m.

## The lattice's pitch

- **0.40 m is enough far from every source, on the lines the data covers.**
  A head at the centre of a square is 0.283 m from two diagonal cells that
  lie either side of it on one line, which is the row at 0.294 m: -24.1 dB
  on 64 channels, where one cell at that distance gives -10 dB. So fusion of
  two holds what ADR 0016's risk 9 feared, with the source 1.9 m or more
  away (0.283 / 0.15).
- **Three or four cells were not measured**, and neither was a head off both
  the lattice's lines and its diagonals, whose two nearest cells are not on
  one line with it. The line is one dimension.
- **Near a source the lattice would have to be 0.20 m or finer** (its centre
  0.141 m from a cell, a source from 0.94 m), and 0.10 m for a voice at
  0.5 m: 1 750 or 7 000 cells over the storey's free floor against 437.
- **Recommended: no lattice for a scene.** The listener's path is in the
  recipe. Cells are stood on it every 0.15 m, at every place it rests, and
  on every seat: 895 cells for the 140 m walked in the first recipe of lot
  L2, against 437 for the lattice; each source reads the cells of its own stride. A head is
  then always between two cells on one line, the one geometry measured, and
  a pair exists only where the listener goes. If a lattice is kept so that
  recipes share cells, 0.20 m, with path cells added where a source is
  nearer than 0.94 m.
- **Seats.** The array of a cell is a ball of 0.26 m on the grid to 1500 Hz
  (0.39 m on the grid to 1 kHz) and must be free air. A seated head against
  a backrest has no such ball; the campaign then stands the array up to
  0.10 m aside and records where, and the head is read from there, which the
  first row of the table says costs nothing so long as no source is within
  0.7 m of it.

## For the engine: kernels of 32 taps at 4 kHz

`translation_operator` and `fusion_operator` return weights per frequency
for a batch of offsets, in `numpy` or `cupy`. Sampled on the 17 frequencies
of `rfftfreq(32, 1 / 4000)`, `kernels` makes them 32 tap filters at the low
band's rate. Against the operator on the response's own 0.83 Hz grid, under
1414 Hz:

| distance | 16 taps | 32 taps | 64 taps |
| --- | --- | --- | --- |
| 0.10 m | -43.2 dB | -55.8 dB | -67.5 dB |
| 0.20 m | -39.6 dB | -52.3 dB | -64.1 dB |
| 0.30 m | -35.8 dB | -49.0 dB | -61.0 dB |

A translation moves an arrival by `d / c` at most, 0.9 ms or under four
samples, which is why so few taps hold it. One offset is 0.52 MB at 32 taps
(64 by 64 by 32, float32) and takes 1.2 ms of one laptop core to make;
fused, 64 by 128, twice that, 5 ms a head once `fusion_inverse` has factored
the pair of cells (10 ms, once per pair). The weights on the response's own
grid would be 1698 frequencies, a hundred times the size, for an error that
is already 25 dB under the method's own. The quadrature of degree 26 is the
operator to 1e-7 at `k d = 3.6` and 6e-4 at 7.3.

## What the data cannot settle

1. **Surfaces nearer than 0.51 m, heads off the lines, and three or four
   cells.** One solve settles the three: a patch 0.80 m square every 4 cm,
   441 cells, one corner against the sofa and one edge 0.3 m from a wall,
   source S1, low band alone. With `reverberate.accel.pairs` on the grid to
   1500 Hz that is one source position and 441 pairs: about 9 minutes of
   2 x A100, **0.27 USD**, and 0.5 GB to bring home
   (`low-band-pairs-cost.md`).
2. **A second height.** Every figure here is at 1.70 m. The patch solved
   again at 1.20 m, where furniture is nearer, is the same price.
3. **Taking the direct sound out before translating.** The source's share
   comes from the near field of the direct path alone, which is known in
   closed form at every head position (`spatial.field.monopole_coefficients`
   and the mirror's signature). Subtracting it from the cell, translating
   the rest and adding the direct sound at the head should lift the 0.15
   towards the surfaces' 0.5. Not tried; the dense line can test it without
   a solve.
