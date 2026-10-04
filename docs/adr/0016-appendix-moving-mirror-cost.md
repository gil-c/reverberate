# 0016, appendix: what the moving mirror costs

Lot L5 of ADR 0016: the band above the crossover, for a scene in which the
source and the listener both move. Measured on 2026-10-04, on a laptop's CPU
(numpy, one core); no card was rented for this lot. The projection for a card
is at the end, with what it assumes. The code is `reverberate.mirror.moving`,
`mirror.moving_onset`, `mirror.tails` and `mirror.directivity`; the benchmark
is `scripts/bench_moving_mirror.py`.

## The figure to beat

The lattice pipeline of ADR 0014 validates, for one source and one point,
every image of the source's tree: 128 352 images on hssd_0076 for source S1.
On one RTX 3090 that was 8.9 s for 437 points, 20 ms a pair (24 ms with the
tree and the grid). A 20 minute scene of 14 sources each audible half the
time is 168 000 pairs: 67 minutes of the card.

## What is computed when

| when | what | on hssd_0076, S1 |
| --- | --- | --- |
| once per scene | facet planes, spheres, mutual visibility, buckets, the occluders' grid at 0.10 m | 0.8 s, 1.4 GB |
| once per scene | the onset's occupancy grid, its graph, the diffracting edges | 2.0 s |
| once per source anchor (0.5 m grid) | the candidate sequences of every source of the cell, and each image's affine map | 208 227 candidates, about 0.1 s |
| once per pair of anchors (source cell, listener cell of 0.5 m) | the short list: a walk of the unfolded path with the cells' size as margin | 37 000 of them, 25 to 55 ms |
| once per step | the sieve: the same walk over the short list, images through their affine maps | 600 left |
| once per step | the pipeline's own validation of what the sieve left | 10 to 49 paths |
| once per cell a source stands in, when a step is shadowed | Dijkstra's distances from it | 0.12 s |
| once per source site (station, rail every 0.80 m) | the rays' histograms over the tail cells | the pipeline's rays, unchanged |

A step whose source and listener stand where an earlier step's did is not
computed again.

## At rest, the present pipeline

- Synthetic room with a doorway (`tests/test_mirror_moving.py`): the same
  paths in the same order, lengths, directions and gains equal to the bit,
  and `render.early_signals` of the table equal **sample for sample** to
  that of the pipeline's paths, with the diffracted onset where the point is
  shadowed.
- The same room turned off the axes: the same paths; the rendered early
  part within 1e-12 of its peak. The pipeline grows its images with a matrix
  product and the batch with elementwise sums, which differ by a few units
  in the last place when a plane's normal has three components.
- hssd_0076, source S1, **all 437 lattice points**, against the paths the
  card wrote for the field of 2026-09-19 (`mirror/c10/paths_S1.npz`,
  calibration `c3cec6aab28bb582`): the same facet sequences at 437 points of
  437, 6092 paths; lengths within 3.6e-15 m, gains within 5e-16 relative,
  the departure direction within 8e-16 of the direction of the saved first
  reflection point; the early part rendered at order 3 within 4.4e-13 of its
  peak. 8.0 s on the laptop, 18 ms a pair, 3.6 GB at the peak.
- The field of 2026-09-18 (`c9`), which the owner listened to, was traced
  before the image tree was corrected (116 662 images then, 128 352 since):
  against its paths the present code, batched or not, finds one path more at
  some points. The comparison above is with the present code's own field.
- The diffracted onset of a shadowed step is the pipeline's on that one pair.
  The lattice pipeline's onset at a point also depends on the other points of
  its run (`docs/formats/scene-pack.md`), so it is not compared point by
  point with the field's.

## Time per step-pair, measured

hssd_0076, the laptop's CPU, a listener walking at 1 m/s among the lattice
points, image paths and onsets:

| scene | ms a step-pair | of which |
| --- | --- | --- |
| source at its station, 400 steps | 20 | sieve 9, occluders 10 |
| source walking too, 400 steps, 234 of them shadowed | 27, and 57 with the onsets | 97 distance fields, 12 s |

Synthetic dwelling of `scripts/bench_moving_mirror.py` (21 facets, 3000
images a tree, 400 steps, both walking, 233 steps shadowed):

| | ms a pair |
| --- | --- |
| batched, image paths | 1.9 |
| batched, with the diffracted onsets | 3.9 |
| present, `ism.paths_for` alone, the tree given | 238 |
| present, `ism.grow_tree` | 1.0 |

126 times the present validation on the same host. That ratio is of two
numpy codes, one of which loops in Python: it says the batch is not slow, not
what a card gains.

## The tail's interpolation, measured

Same script: 1000 rays, 0.4 s, sixteen cells in two rooms, level summed from
20 ms on, per band, the worst cell read.

| | worst band | worst | median |
| --- | --- | --- | --- |
| source half way between two positions 0.80 m apart | 8 kHz | 3.6 dB | 0.5 dB |
| head half way between two cells 0.80 m apart | 8 kHz | 2.3 dB | 0.4 dB |
| the same position, another seed | 8 kHz | 2.2 dB | 0.5 dB |

The third line is the floor: at 1000 rays, which is what the Python tracer
gives in a minute, two traces of one position differ by 2.2 dB at the worst
cell. The head's figure is at that floor; the source's is 1.4 dB over it, at
the worst cell and band. The medians agree with the 0.3 to 0.7 dB this
project measured on the wave field. **The worst case is not resolved**: it
needs the card's 100 000 rays, and is for the first trace on a card to
measure before the 0.80 m spacing is taken as settled.

## Projection for one RTX 3090

Not measured. It rests on these assumptions, each of which the first trace on
a card checks:

1. `cupy` runs the array code as numpy does. It was written against the
   functions both have and has not been run on a card.
2. An elementwise operation on a card costs 15 to 50 times less per element
   than on one laptop core, once arrays hold a million elements.
3. One array operation on a card costs 30 to 50 microseconds to launch.

Then, per step-pair:

- **Sieve**: 37 000 pairs through about 50 operations, 9 ms on the laptop,
  **0.2 to 0.6 ms**.
- **Validation of what is left**: the occluders are walked cell by cell,
  every leg in step, about 40 000 launches a call whatever the number of
  legs, 1.2 to 2 s. A call must therefore hold many steps:
  `MovingSettings.pairs_per_validation` at 3 000 000 is 5000 steps and 1 GB,
  **0.25 to 0.4 ms** a step. At the default of 300 000, chosen for a laptop,
  it would be ten times that: set it for the card.
- **Host**: the rows' gains, identities and sorting, 0.2 ms. The short
  lists, 25 to 55 ms each on the host, are computed on `xp` too, a few
  listener cells at a time.

About **1 ms a pair, 3 minutes for the 168 000**, against 67: twenty times,
with the order of magnitude asked for as the floor of the range rather than
its middle. What is not in it:

- **The onsets are host work.** A shadowed step of a source that walks
  costs a Dijkstra per cell entered, 0.12 s here. If a fifth of the
  168 000 pairs are shadowed and a tenth of those have a source on a rail,
  that is 3400 distance fields, 7 minutes of one core, more than the card's
  part. Stations cost one each. It parallelises over cores and is the first
  thing to move if it shows in the ledger.
- **The rays**: 16 s a source site on the card, unchanged. A source with
  three stations and 12 m of rail is 18 sites, 5 minutes; fourteen such
  sources are over an hour, before the dwelling's cache. This is now the
  mirror's largest cost, and L10's.

If assumption 1 or 3 fails, the second lever is ready: the sieve leaves 600
candidates of 128 000 for the pipeline's own CUDA kernel
(`mirror.kernels.PATHS_KERNEL`), which reads a tree of any shape; fed the
sieve's survivors as a tree of chains it would validate two hundred times
fewer pairs than it does today.

## Open

- Nothing here ran on a card.
- `voice_v1` is a parametric fit (`mirror.directivity`), flagged for the
  owner.
- A source at a station is given the 0.5 m anchor of a moving one: its
  candidate tree is 208 000 sequences where its own is 128 000. An anchor on
  the station itself, with no slack, would shorten its lists.
- A reflection order above one for the diffracted onset is refused in the
  moving trace; the pipeline's default is one.
- The seated height was not exercised: nothing in the trace reads a height,
  but the calibration comes from 1.70 m (ADR 0016).
