# The rays of the tail through a hierarchy of boxes

Date: 2026-10-05

Status: **designed, written and proven on the host; no card has run it.**
Lot L24 of ADR 0016, after the performance audit
(`performance-audit.md`, section 3) counted the tail's rays at 130 to 1300
times their floor. Every figure says where it comes from: a count made on
the laptop on the storey's own mirror (`hssd_0076`, 1 478 245 triangles, the
bundle of the first scene), a test of the suite, or a prediction with its
basis. What a card must still say is in section 11, with the commands. The
grid's kernel stays the one a trace uses until then:
`REVERBERATE_RAY_STRUCTURE=tree` selects what is described here.

## 1. The answer

- **The same histograms from a twelfth of the triangle tests.** A segment
  of a ray on the storey visits 44.8 nodes of a tree and tests 8.45
  triangles, where it walked 28 cells of a grid and tested 140. The tree's
  boxes are single precision and conservative, the triangle test is the
  twin's own in double precision, and of two triangles at one distance the
  lower index is kept: **the nearest hit does not depend on the structure**,
  so the histograms are the grid's to the count.
- **Shown on the storey itself, on the laptop.** The new text, compiled for
  the host, cast the 100 000 rays of two sites of the first scene (14.06
  million segments each) and was held against what the card's grid kernel
  wrote in that scene's pack: at the first site every one of 54 600 energy
  bins and 873 600 moments is equal; at the second, one crossing of 78 553
  differs (section 9).
- **Single precision is an option, proven as another draw of the same
  tail,** not the default: with the triangle tests cut to 8 a segment, the
  double precision that was 1.39e9 tests a site is 1.19e8, and what single
  precision still buys is for the card to say. Its triangle test is
  watertight by construction, where Moller and Trumbore's is not.
- **A histogram is kept a (site, cell) at a time.** A second recipe of the
  dwelling casts rays only from the sites, and for the cells, no earlier
  recipe read.
- **One C text for a host's core and for a card.** A machine without a
  card casts a site of 100 000 rays in 22 s on one core of the laptop,
  which is the order of what the card's present kernel takes (12.8 to
  14.5 s).
- **Predicted on an RTX 3090**: 0.1 to 1 s a site in double precision, 0.05
  to 0.5 s in single, against 12.8 to 14.5 s: the tail of a recipe in 20 to
  200 card seconds (2600 today), and the tail of a dwelling's 340 sites
  once, in a minute or a few, for every recipe that follows. The bracket is
  wide because it is a prediction; section 10 says what it rests on.
- **Found on the way, and not this lot's to repair** (section 9): at
  100 000 rays the tail is known to a few tenths of a decibel in its first
  30 dB and to nothing better than 5 to 9 dB from 50 dB down, with every
  cell's crossings together. It is the estimator, not the precision; rays
  a hundred times cheaper are the first remedy.

## 2. What a segment must do

A ray is a chain of segments. For each: find the first triangle along it;
find the receivers' spheres it enters before that; lose energy, draw the next
direction. On the storey (site of the first scene, 100 000 rays, counted by
the text itself):

| | the grid (audit, site 100) | the tree (site of scene 1) |
| --- | --- | --- |
| segments a ray | 99 | 140.6 |
| structure steps a segment | 28 cells | 44.8 nodes of 64 bytes |
| triangle tests a segment | 140 | 8.45 |
| receiver steps a segment | 28 cells, again | 1.07 nodes |
| crossings a segment | "a few in a hundred" | 0.0106 |
| a site of 100 000 rays | 9.9e6 segments, 1.39e9 tests | 14.06e6 segments, 6.30e8 nodes, 1.19e8 tests |

(The two sites differ: the audit followed 40 rays of another site. Rays end
by the energy floor in both: 99 756 of 100 000 here, 242 by the 1.2 s.)

What costs, in the order it costs:

1. **The triangle tests in double precision**: 60 operations each. 507 a
   segment where there were 8400.
2. **The nodes**: one read of 64 bytes and two box tests of some 30 single
   precision operations. Memory, not arithmetic: the tree is 57 MB and a
   segment touches 45 nodes of it that its neighbours in the warp do not.
3. **The bounce**: an absorption per band, a normal, one to five draws of
   the hash, a reflection: some 150 operations, once a segment.
4. **The spheres and the histogram**: one node a segment in the mean (the
   root of the receivers' tree refuses nearly every segment), and one
   crossing in 94 segments, each 119 atomic additions (7 bands, 16 moments
   a band): 1.3 additions a segment.

## 3. The structure

`reverberate.mirror.bvh`. Decided, with what was weighed:

- **A binary tree of bounding boxes, cut by the surface area heuristic over
  16 bins**, each node holding its two children's boxes, so one read of 64
  bytes decides both (the layout of Aila and Laine, 2009). The scene is
  1.48 million triangles of 1.8 cm median edge with a few of metres (the
  largest edge is 3.98 m): half the area is in 60 446 triangles and nine
  tenths in 269 979. A uniform grid cannot serve both ends, a tree cut by
  area does. A tree ordered by a space-filling curve (LBVH) builds faster
  and is worse exactly on a few large triangles among many small; the
  build is paid once a dwelling, so the better tree was taken.
- **Built on the host, in numpy, a whole level at a time**: 7.7 s for the
  storey's tree and 8 to 12 s with the triangles' attributes, 2.4 GB at its
  peak, deterministic
  (no random number, stable sorts, the lower index on a tie), **kept in
  the store of lot L15b** (`mirror/shared.py`, kind `rays_tree`) under a
  name made of the triangles, their labels and the facets, and of no
  material: a calibration finds the same tree. 893 666 nodes, 30 deep, 57
  MB; the triangles in leaf order beside it, 106 MB in double precision or
  53 MB in single; 12 MB of labels and flags.
- **Leaves of two triangles at most.** Counted on the storey:

  | leaf | nodes | nodes a segment | tests a segment |
  | --- | --- | --- | --- |
  | 1 | 1 478 244 (95 MB) | 48.4 | 6.97 |
  | 2 | 893 666 (57 MB) | 44.8 | 8.45 |
  | 4 | 494 025 (32 MB) | 41.3 | 12.31 |
  | 8 | 261 839 (17 MB) | 38.1 | 20.96 |

  A test is double precision and a node single: small leaves. Thirty-two
  bins in place of 16 change the nodes by a fiftieth and the tests by
  nothing (40.5 and 12.33 at leaves of four). **Seven tests a
  segment is the floor of this scene**, not of the tree: a ray starts on a
  surface, among the boxes of that surface's neighbours, and ends on
  another.
- **A stack a ray, the nearer child first, the farther pushed with its
  entry distance and dropped at the pop when a hit has come nearer.** 96
  entries; the build refuses a tree deeper than 88 and halves segments
  from depth 48 on so that none is.
- **Not taken**: nodes of eight children with quantised boxes (Ylitie and
  others, 2017) and the card's own ray tracing units (OptiX). Both are
  faster by a factor of two to ten on a card, and neither can be a plain C
  text that a host compiles and numpy can be held against operation for
  operation, which is what makes the histograms provable here. They are
  the next step if the card's measure says the nodes are what is left
  (section 12).

## 4. Double precision: the same hits, by construction

Three things make the tree's histogram the grid's:

1. **A box's test only ever adds candidates.** The boxes are rounded
   outwards and grown by 50 micrometres (sixteen units in the last place of
   the scene's largest coordinate, where that is more), entry distances are
   taken short and exits long by two parts in a million, and the
   comparison with the best hit so far is `<=`: a box the ray in double
   precision enters, or that holds a triangle at exactly the best distance,
   is never refused.
2. **The triangle test is the twin's**, Moller and Trumbore in double
   precision on the unit direction, a hit farther than a micrometre.
3. **The tie rule is the twin's**: the nearest hit, then the lower index of
   the scene. The grid's twin takes `argmin` over its candidates sorted by
   index, which is that rule; the tree visits triangles in another order and
   compares `(distance, index)`. Ties are not rare: a model holds sheets
   twice.

When is the identity exact? **Against the grid's twin, always**: the tests
hold the twin through the tree to the grid's twin on a furnished room with
and without the image tree's skip: crossings and energies to the count,
moments to two counts of 2^40 (a reflected direction may differ in its last
bit, because numpy takes a scalar product through the machine's BLAS and the
text writes it out). **Against the card's grid kernel, to a ray in a hundred
thousand**: that kernel tests the triangle on the direction scaled by the
ray's remaining reach, not on the unit direction, so its distances differ
from the twin's in their last bit, and on a storey of curved furniture a
last bit grows over 140 bounces into another path for a ray now and then:
one crossing of 78 553 at the second site of section 9, none of 149 145 at
the first.

## 5. Single precision: another draw of the same tail

`RaySettings.precision = "single"`: positions, directions, energies and the
triangle test in single precision, coordinates taken from the scene's middle
(a single holds half a micrometre at 8 m). The hash keeps its 64 bits and
gives its 24 highest; the histograms stay 64 bit integers. A precision is
part of a histogram's key.

- **The triangle test is Woop, Benthin and Wald's (2013)**: the ray's own
  frame, three edge functions that are each a difference of two products of
  the sheared corners. Two triangles that share an edge take the same two
  products for it, so a ray is never outside both; one that is on the edge
  to the last bit is settled in double precision. Moller and Trumbore has
  no such property: **in double precision, 171 of 5292 rays aimed exactly
  at the edges and corners of a tessellated room pass between two
  triangles**; none of 15 876 does with this test in single
  (`test_no_ray_aimed_at_an_edge_or_a_corner_leaves_a_closed_room`). A ray
  cast at random does not come that near an edge in double precision,
  which is why no tail showed it.
- **The hit is the triangle's own point**, the three corners weighted by
  the edge functions, on the triangle's plane to the rounding of a
  coordinate however long the segment; `origin + t * direction` would be
  off the plane by the rounding of `t`.
- **A ray's own surface is refused, and no wall that meets it.** The origin
  may be a rounding behind the surface the ray leaves, and would meet the
  coplanar neighbour of the triangle it left. A hit is refused when it lies
  within 20 micrometres of that surface *on a triangle parallel to it*
  (cosine over 0.999). The first version refused every hit within that
  distance: rays then went through the wall at every corner of a room they
  landed within 40 micrometres of, 10 in a million segments. With the
  parallel rule: section 9.

**A finding about the reference.** The double precision's bound of a
micrometre on a hit's distance is the same kind of rule, and lets a ray
through a wall it lands within a micrometre of: 17 rays per million leave a
closed furnished room in double precision (5 of 300 000), and 10 to 35 per
million meet no triangle on the storey. It is 0.0002 dB and not a defect to
repair in this lot; it is why "zero escapes" is not the measure of single
precision, and "no more than double precision" is.

## 6. Whether the scene should be simplified first

No. With a tree, the cost of a segment grows with the logarithm of the
triangles: the same room cut in 150 or in 864 triangles a wall (5.8 times
more) costs 13.8 or 18.6 nodes a segment and 3.98 tests in both. A storey
decimated four times would save about four of 45 nodes, a tenth, and
nothing of the tests; and it would change which triangle a ray hits and
with which normal, on furniture whose scattering is a coefficient fitted
with these triangles. A change of the rendered tail for a tenth is not
proposed. (The grid is what made detail expensive: 4074 triangles in its
worst cell.)

## 7. How the rays are laid on the card

- **A thread a ray, the ray's whole life in it**, as today. The audit
  proposed to advance every living ray by a segment a launch and compact
  them in between, so that no lane of a warp waits. What a lane waited for
  was the grid: 8 triangles in the median cell and 4074 in the worst. In
  the tree every leaf holds one or two. What is left to wait for is a ray
  that outlives its warp: the audit's site has 99 segments a ray in the
  mean and 116 at most, so a warp of 32 idles about a tenth of its lane
  time. A wavefront would pay a state of 150 bytes a ray written and read
  at every bounce, and a launch a bounce, for that tenth. Kept as a remedy
  if the card says otherwise.
- **Receivers**: a second small tree over the spheres' boxes (53 cells,
  some dozens of nodes), walked for each segment up to the hit. All the cells of a launch
  at once; a cell's histogram does not depend on the others
  (`test_shares_of_the_rays_through_the_tree_sum_to_the_whole`).
- **Histogram writes**: atomic additions on 64 bit integers, as today. 1.3
  a segment; partial histograms a thread would cost 30 MB a thread.
- **Random numbers**: the counter based hash of (seed, ray, bounce, draw),
  unchanged, so a ray is the same in the twin, on a core and on a card, in
  any share of the rays.

## 8. A histogram a (site, cell)

`mirror/tails.py`. The key of a histogram held a digest of the recipe's
whole set of cells, so a second recipe of the dwelling cast every site
again. Now:

- `tail_key(scene, position, cell, rays)`: one cell. An entry is one
  cell's rows, `tails/<two characters>/<key>.npz`, written whole or not at
  all.
- `histograms` asks for a site over its cells, reads what is kept, casts
  the site once over the cells that are not, and keeps each on its own.
  That a cell's rows are the same whichever cells were cast with it is a
  property of the tracer and is tested.
- A trace's `rays` job is still a site (`site_key`), queued when a cell at
  least is absent. The report counts both: `histograms_traced` (sites) and
  `site_cells_traced`.
- Under `REVERBERATE_MIRROR_STORE` the cache is the dwelling's
  (`<store>/tails`), beside the trees and the fields a second scene
  already finds there; otherwise the run's own `tails`, as before.
- Migration: none. An entry under the old key is not found and is cast
  again.
- **Not changed**: what an entry holds. 0.6 MB a (site, cell) in double
  precision, 30 MB a site of 53 cells of 7 bands, as before. The audit's fifth remedy
  (single precision, compressed, 4 MB a site) changes what the levelling
  reads by a part in ten million and is left to its own lot; with the rays
  cut to a fraction of a second, writing 35 MB a site is now a visible part
  of a site's time, and the card's run measures it (`site_s` against
  `site_kernel_s`).

## 9. What is proven on the host

Tests (`tests/test_mirror_tracer.py`, `test_mirror_tails.py`,
`test_mirror_rays.py`), 7 s together:

| what | how |
| --- | --- |
| the tree returns the nearest hit and the lower index on a tie | against a test of every triangle, 2 scenes of 420 random triangles of which 20 are there twice, 300 rays each |
| the build is deterministic, holds every item once, survives nothing, one item and 60 items on one point | arrays equal twice; leaves counted |
| the twin through the tree counts what the grid's twin counts | a furnished room (a closed cabinet in it), with and without the skip of the image tree's rays: crossings and energies equal, moments within 2 counts |
| the C text counts what the twin counts | 1500 rays, the same histograms to the count and the same numbers of nodes visited and triangles tested |
| the C text's first hit is the twin's | 2000 rays on 630 random triangles: index and distance equal |
| single precision is watertight | 15 876 rays aimed at edges and corners of a closed room: none leaves |
| single precision gives the double's tail | 8000 rays, 412 000 segments: per band and window of 50 ms, single against double at one seed is nearer than the farthest two of three seeds of double are, worst band and window; no ray leaves |
| shares of the rays sum to the whole; one core and three give one histogram; a cell alone is its rows of three | integers equal |
| a (site, cell) is cast once | a second set of cells at the same site reads two and casts one; the result is the whole trace's |

The two tests of the grid's twin that took 59 s of the suite
(`test_covered_rays_*`) cast their rays through the tree's twin and take
0.3 s; that the two twins count alike with and without the skip is the
third row.

On the storey (`scene1/A`, the laptop, one core, the C text):

| | |
| --- | --- |
| a site of 100 000 rays, double precision | 22 to 48 s (0.29 to 0.63 million segments a second, the lower figure with other lots at work on the machine) |
| against the pack's histograms (the card's grid kernel, single precision floats) | site (-14.058, 1.7, -16.565), 13 cells: 54 600 bins and 873 600 moments equal. Site (-14.058, 1.7, -15.965), 7 cells: one bin of 29 400 differs, by one ray's energy (8.8e-10 of the source's) |
| rays that meet no triangle | double precision: 2 and 1 of 100 000 at those sites, 35 per million over two others; single precision at those two: 30 per million |
| single precision against double, one seed, a site of 53 cells | the whole energy a band: 0.002 to 0.003 dB (two seeds of double: 0.03 to 0.10 dB) |

**Single precision against two seeds of double**, at that site (100 000
rays, 53 cells, 452 000 crossings; a level is read where the reference
holds a hundred crossings; root mean square, then the worst band's, then
the worst single value, dB):

| | single against double | another seed against double |
| --- | --- | --- |
| a cell's bins of 2 ms, the 30 dB under its loudest | 0.00, 0.00, 0.00 | 0.51, 0.56, 1.27 |
| a cell's windows of 50 ms, the same 30 dB | 0.42, 0.56, 3.5 | 1.88, 2.91, 25.2 |
| every cell's crossings together, windows of 50 ms, 30 dB | 0.04, 0.07, 0.15 | 0.16, 0.26, 0.51 |
| the same, 60 dB | 0.23, 0.39, 1.08 | 2.57, 3.95, 11.0 |
| the same, the whole histogram | 1.25, 2.37, 8.5 | 3.87, 6.49, 14.0 |

Single precision stands four to ten times nearer to double than another
seed of double does, in every row: the two share their rays' first bounces
and part ways late. The audit's figures for the seed's spread (0.5 dB in
bins of 2 ms, 0.14 dB in windows of 50 ms) are the first and the third
rows' right column.

**What the right column says of the tail itself.** The level falls 14 dB
every 100 ms at that site. A ray's energy is its start times the product of
what a hundred surfaces left, band by band, and those products spread over
orders of magnitude: late in the tail a window's energy is a few rays that
met the least absorbing surfaces. With every cell's crossings together
(26 000 a window) two seeds differ by 0.1 dB at -14 dB, 0.4 dB at -28 dB,
0.5 dB at -39 dB, and 7.8 dB at -53 dB; at one cell 1.6 dB already at -36
dB. Under -90 dB the counts are zero in the middle bands while the 125 Hz
band keeps the ray alive. So: **100 000 rays give the first 30 to 40 dB of
a tail and a guess below.** More rays help as their square root at best.
The estimator's own remedy is to let absorption decide whether a ray lives
(with a common survival a band's weights stay within a narrow range) and
not only what it weighs, which is a change of the tracer's statistics for a
lot of its own and the owner's decision: it changes every tail. It was not
visible before because nobody could afford a second seed of a storey.

## 10. Predicted on an RTX 3090, and on what it rests

The present kernel makes 1.0e6 segments a second, each 140 double precision
tests and 28 cells: about 8.8e9 double precision operations a second, 1.6
per cent of the card's 0.556e12.

| | a segment | if the card stays at 1.6 % of its double rate | if the lanes' wait was most of the loss |
| --- | --- | --- | --- |
| tree, double precision | 507 operations of tests, some 200 of the rest | 1.25e7 segments a second: **1.1 s a site of 14e6 segments, 0.8 s of the audit's 9.9e6** | ten times that: **0.1 s a site** |
| tree, single precision | 45 nodes of 64 bytes | bound by memory, not arithmetic: published software traversals reach 1e8 to 4e8 incoherent rays a second on cards of this class with tuned wide trees; a plain binary tree in portable C is taken at a third to a tenth: 3e7 to 1e8 segments a second: **0.15 to 0.5 s a site** | |

So: **0.1 to 1 s a site in double precision, 0.05 to 0.5 s in single**:
13 to 130 times the present kernel. The audit's floor (0.01 to 0.1 s) is
the tuned wide tree's or OptiX's.

| the tail of | today | double | single |
| --- | --- | --- | --- |
| a site, 100 000 rays | 12.8 to 14.5 s | 0.1 to 1 s | 0.05 to 0.5 s |
| a site, 1 000 000 rays | 128 to 145 s | 1 to 10 s | 0.5 to 5 s |
| a recipe (202 sites) | 2600 card s, 0.125 USD | 20 to 200 s | 10 to 100 s |
| a dwelling (340 sites, once, with the (site, cell) cache) | once a recipe | 35 to 340 s, then nothing a recipe | 17 to 170 s |

**More rays instead of fewer.** A million rays a site would cost what
100 000 cost today at the slow end of the bracket and a tenth of it at the
fast end, and take the seed-to-seed spread of a 2 ms bin from 0.5 dB to
0.16 dB. It multiplies nothing else: the histograms are the same size. If
the card confirms a second a site or less, a million rays is affordable for
a dwelling (340 sites, 6 to 60 minutes of one card, once) and is the owner's
to choose; it changes every tail and its key.

## 11. The card's run

One RTX 3090, the code of this branch, the bundle of the first scene
(`runs/w45_clarify_scene/scene1/A/bundle`, 132 MB of mirror), about ten
minutes, of which the grid's own baseline is most:

```
python -m pytest tests/test_mirror_engine.py tests/test_mirror_tracer.py -q -p no:cacheprovider
PYTHONPATH=src python scripts/bench_rays.py --bundle BUNDLE/trace \
    --sites 20 --rays 100000,1000000 --grid-sites 2 --out rays_3090.json
```

The first line holds the card's text to the twin and to the grid's kernel
on a furnished room (crossings equal, energies and moments within two
counts, the same numbers of nodes and tests) and counts the rays single
precision lets out of a closed room in five million segments. The second
prints, for the grid, the tree in double and the tree in single:

- seconds a site at 100 000 and at 1 000 000 rays (the whole call, and
  the launches alone), segments a second, nodes and tests a segment;
- **identity**: the tree's histograms against the grid's at 20 sites, the
  largest difference in crossings and in counts. Expected: zero crossings
  at most sites and one ray's here and there (section 4);
- **statistics**: single against double beside another seed against double,
  per band, in bins of 2 ms and windows of 50 ms, root mean square and
  worst cell, band and window. Accepted when single stands no farther than
  the other seed (`single_is_within_a_seed`); the audit and lot L12
  measured that spread at 0.5 dB in bins and 0.14 dB in windows;
- **escapes** per million rays, the three ways.

With `--cpu` the same script runs on a core without a card (section 9's
figures are from it).

What follows the run: the tree becomes the default structure and the plan's
`RAYS_SITE_S` is set from it (`trace/plan.py`); single precision and the
count of rays are the owner's, with the figures.

## 12. Risks, and what is not done

- **The card's text has not been compiled by a card's compiler.** It is the
  host's text (plain C that is also C++) under the same options as the
  other kernels; the atomic additions, the infinities and the bits of a
  float are the four lines that differ. First thing the run does.
- **The prediction's bracket is a factor of ten.** If double precision
  lands at its slow end (1 s a site), the remedies in order: single
  precision (proven here as a draw of the same tail); a single precision
  test before the double one, conservative as the boxes are, which keeps
  the identity and leaves one or two double tests a segment; nodes of
  eight children.
- **The host's side of a site is now visible**: 30 MB of counts brought
  home, turned to double precision and written as 53 files. Measured by the
  run; the remedy is the audit's fifth (smaller entries), a lot of its own.
- **A site whose cells are partly kept is cast whole** for the cells that
  are not: rays cost the same for one cell or 53. With sites at a second
  or less this is cheaper than any bookkeeping of which rays went where.
- **The tail's defects of the chain audit are untouched and no harder to
  repair**: D8 (the shell's scattering differs between images and rays) is
  the materials', which the tree does not hold and the text reads a label
  at a time as before; D10 (`skip_specular_order`) is a setting the text
  takes as the kernel did; D4, D5 and D9 are the renderer's.
- **Not done**: the plan's cost constants (after the card); the
  histograms' storage; the viewer's sample of rays (`viz/computed_rays.py`)
  still walks the grid's twin.
