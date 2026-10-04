# A solver built for the low band, and what it should cost

Date: 2026-10-04

Status: written and proved on `numpy`, on small grids. **Nothing below has
run on a card**: every speed is a prediction with its assumptions beside it,
and the commands that replace them with measurements are at the end. Written
for lot L10a of ADR 0016. The code is `reverberate.wave.lowband`; the
question it answers was left open by `low-band-pairs-cost.md`.

## The use, on the recipe the owner wants

The realistic recipe (hssd_0076, seed 20261004, 20 minutes, 14 sources,
`dwell_s` 20 to 180, the listener resting 15 to 120 s, planned by
`reverberate.trace.plan.make_plan`):

| | |
| --- | --- |
| source positions audible | 1646 |
| listening cells | 831: 13 rest places, 818 along 140 m of path |
| pairs | 16 529, ten a source position; half the positions have four or fewer, five have more than 750 |
| the present engine, 2 x A100 at 1.74 USD/h | 26 240 s of stencil, 14 020 s of pairs: **19.5 USD** |

## What a solve is made of

Read on the grid of hssd_0076 to 1 kHz (19.5 M nodes) and scaled to the
grid to 1500 Hz (65.9 M nodes, 32 769 steps):

- **A source reaches 72 % of the box.** The rest is the grid's margin, the
  air outside the shell and the nodes buried in walls and furniture. The
  present engine updates all of it.
- **The boundary is as dear as the air.** 6 % of the nodes at 1 kHz, 4 % at
  1500 Hz, are lossy boundary nodes, and each carries 11 branches of two
  states: 176 bytes read and written a step against 12 for a node of air.
  On the grid to 1500 Hz that is 0.51 GB a step for the boundary and 0.57 GB
  for the air it bounds. **Any scheme that thins the air leaves the boundary
  as the larger part.**
- **The branches cannot simply be dropped.** Each material is 11 resonators
  an octave apart from 16 Hz to 16 kHz, fitted together. Leaving out those
  resonant above 3 kHz changes a wall's admittance under 1500 Hz by up to
  68 % (`problem.prune_branches`, on the grid's own materials). A cheaper
  boundary needs a new fit of fewer branches for this band, and its own
  proof on a furnished room; it is not done here, and the engine's boundary
  is kept to the operation.

## The scheme

Budget: the present engine's own, 1.03 % of phase velocity at `fmax` along
the worst direction (`scheme.velocity_error`).

| scheme | points per wavelength for 1 % | Courant limit | nodes | steps | node updates |
| --- | --- | --- | --- | --- | --- |
| Cartesian, 7 points (present) | 10.5 | 0.577 | 65.9 M | 32 769 | 2.16e12 |
| Cartesian at 7.2 | 2.3 % at `fmax`, 1.0 % at 1 kHz | 0.577 | 21.2 M | 22 470 | 4.8e11 |
| face centred cubic, 13 points | 7.7 | 1 | 13.0 M | 13 874 | 1.8e11 |

The face centred grid is the cubic grid's nodes of even index sum, each read
with twelve neighbours. It holds 1 % at 7.7 points, is stable to a Courant
number of 1, and PFFDTD carries it with the same boundary model: **twelve
times fewer node updates**. Considered and not taken: the interpolated
wideband scheme (27 points, no boundary model or voxeliser for it in
PFFDTD) and compact implicit schemes (a linear solve a step, and boundaries
of their own).

**Proved on `numpy`**: one step of the solver advances a plane wave as each
scheme's relation says, to 2e-5, in three directions, through the face
centred grid's fold; a monopole in free air has the level `1 / 4 pi r` on
both grids and the scheme's speed along its worst direction; a rigid box of
20 x 15 x 11 nodes rings at its analytic modes within 1.2 % to 1500 Hz on
the Cartesian grid.

**The face centred grid's weakness is its walls.** A boundary node there
loses four of its twelve links and keeps a whole node's mass, so a wave
running along a wall meets a quarter less stiffness; the Cartesian node
loses one link of six and none along the wall. It is PFFDTD's boundary, and
this solver reproduces it. Measured: the lowest mode of a rigid box 0.6 m
long is 4.1 % low, and of one 1.2 m long 2.0 % low, about `0.8 h / L`. In a
room 4 m across that is 0.5 %, half the dispersion budget again and of one
sign; in a corridor 1 m wide, 1.5 %. So the face centred grid is a choice to
measure on the dwelling and not a default, and the Cartesian grid at fewer
points is the other candidate: its walls stay where they are and its
dispersion grows.

## What was built

`reverberate.wave.lowband`, `numpy` for the tests and CUDA kernels for a
card, the same arithmetic in both and in a third, scalar, transcription:

- **the engine's step, exactly** (`solver`): the air, the absorbing layer,
  the rigid and the lossy boundary nodes with every branch, in the CPU
  engine's order of operations, single precision, on either grid, read from
  the engine's own four files. On the bundle's grid the result is the
  present engine's to rounding (PFFDTD's card kernel sums in another order
  and fuses its products, so not to the bit);
- **the reached columns alone** (`problem`): the nodes a source can make
  other than zero are found by following who reads whom from the sources'
  nodes, and only the columns holding one are stored. Equal to the whole box
  bit for bit;
- **a batch of sources a launch**: a field is `[node, source]`, the grid and
  its boundary shared, one kernel launch for all. Equal to each source
  solved alone, bit for bit;
- **receivers on the card**: a record is a node of a source, kept at the
  grid's rate in single precision on the device and filtered, resampled to
  4 kHz and fitted there. No comms file, no pressure file, no host copy;
- **the fit as an operator** (`fit`): the normal equations of each bin are
  solved once per array geometry against the weighted array matrix, 1.8 GB
  on the card; a pair is then two transforms and a product, not 3600
  factorisations. Equal to the solve to 1e-9;
- **the campaign** (`pairs.LowbandPairs`, `trace.engines.BatchedPairs`): the
  pairs campaign with its solve replaced. The bundle, the arrays, the cache
  form, the index and the keys are unchanged, so the trace runs as it is. A
  pair's key names the solver and the grid, and another grid (`--low-scheme
  fcc`, `--low-ppw`) is voxelised and cached under its own key.

## Memory, and how many at once

A source is two fields of the stored columns and two states a branch a lossy
node; a record is 4 bytes a node a step. With ten cells a source (9840
nodes), 80 % of a card for the batch and 2.5 GB kept for the fit:

| grid | a source | its records | 16 GB | 24 GB | 48 GB | 80 GB |
| --- | --- | --- | --- | --- | --- | --- |
| Cartesian 10.5 | 0.69 GB | 1.29 GB | 5 | 8 | 18 | 30 |
| Cartesian 7.2 | 0.26 GB | 0.88 GB | 9 | 14 | 31 | 53 |
| face centred 7.7 | 0.15 GB | 0.55 GB | 14 | 23 | 51 | 87 |

The records are the larger part. A source heard at 831 cells needs 107 GB of
them on the Cartesian grid: it is solved several times, each for some of its
cells (`pairs.LowbandPairs.items`), which costs the five such sources of the
recipe five to thirty solves more, by the card. Decimating the records as they are made
would cut them three to seven times; it changes the filters' numbers and is
left until a card says the records bind.

## Reciprocity, counted

A cell costs 64 solves and serves every source position; a source position
costs one and serves every cell. Every pair must be covered by one or the
other, which is a minimum weight cover of the pairs' graph, solved exactly
(`reciprocity.solve_counts`). On the recipe above:

| what is solved | solves |
| --- | --- |
| every audible source position | 1646 |
| every cell, reciprocally: 64 x 831 | 53 184 |
| the best mix: 3 rest places reciprocally (192), 1338 positions forward | 1530 |

**The source side is solved.** The best mix saves 7 %, and would need the
array driven as a source (a thousand nodes, each with the fit's weight as a
time signal, 130 MB a solve). Ten cells are heard from more than 64
positions; only three are heard from enough positions that are heard
nowhere else. Not built.

## What it should cost, and on what assumption

The assumption: a card's speed is its memory traffic, 12 bytes a node of air
and 192 a lossy node a step, as the present engine's 15.9 s is 1.37 GB a
step. Nothing else enters these figures.

| grid | traffic a step | steps | a source position | against 15.9 s |
| --- | --- | --- | --- | --- |
| present engine | 1.37 GB | 32 769 | 15.9 s | 1 |
| Cartesian 10.5, reached nodes | 1.08 GB | 32 769 | 12.6 s | 1.3 |
| Cartesian 7.2 | 0.42 GB | 22 470 | 3.4 s | 4.7 |
| face centred 7.7 | 0.25 GB | 13 874 | 1.2 s | 13 |

A pair costs 0.85 s today (the engine's copy and write 0.43, comms 0.02, the
fit 0.40). Here the first two are gone and the fit is a product; what is
left is three filter passes and the resampling of a thousand records,
guessed at 0.05 to 0.1 s. For the recipe, 2 x A100 at 1.74 USD/h:

| engine | stencil | pairs | USD |
| --- | --- | --- | --- |
| present | 26 240 s | 14 020 s | 19.5 |
| batched, the bundle's grid | 20 700 s | 830 s | 10.4 |
| batched, Cartesian 7.2 | 5 600 s | 830 s | 3.1 |
| batched, face centred | 2 000 s | 400 s | 1.2 |

**An order of magnitude needs the face centred grid, or a cheaper boundary
on the Cartesian one, and each must pass the comparison below.** The
bundle's grid alone, which is the present engine to rounding, is predicted
at half the price, most of it the pairs.

## The pair cache's size

16 529 pairs are 20.3 GB, more than the solve is worth in transfer. The
cache form is 64 channels of 4800 samples in single precision, and nothing
in it is redundant: the fit is made on 2.4 s and cut to 1.2 s, so the 4800
samples hold more than the bins under 1500 Hz would; keeping it at 3 kHz
instead of 4 would save a quarter and is exact only to the level of the
response at its cut. Half precision is not acceptable without a proof and
none is offered. What would help is outside this lot: the machine publishes
the cache to the store and brings home the pack alone.

## Commands

On the laptop, the reference and a bundle of the dense line's 341 cells:

```
python -m reverberate.wave.lowband line \
    --field data/runs/w44_clarify_interpolation/line_home2/pulled/field/S1.h5 \
    --plan data/runs/w44_clarify_interpolation/line_home2/pulled/plan.json --out L
python -m reverberate.accel pairs-bundle --out B --scene-id 104862621_172226772 \
    --models-from data/runs/w44_clarify_interpolation/bundle_line_0076/models/storey \
    --sources L/sources.npy --cells L/cells.npy
```

On the machine, with `B` and `L/line.npz` pushed and `R` its hourly rate:

```
# The kernels against numpy bit for bit, and against PFFDTD's binary on the same files.
python -m reverberate.wave.lowband verify --out O/verify --pffdtd /root/pffdtd

# The bundle's grid against the present engine on it, which is run first into O/ref.
python -m reverberate.wave.lowband compare --bundle B --out O/a \
    --reference O/ref --pffdtd /root/pffdtd --rate R
# The cheaper grids against the same reference, and against the three band line.
python -m reverberate.wave.lowband compare --bundle B --out O/c72 --ppw 7.2 --reference O/ref --rate R
PFFDTD_DIR=/root/pffdtd PFFDTD_PYTHON=/root/pffdtd-venv/bin/python \
python -m reverberate.wave.lowband compare --bundle B --out O/fcc --scheme fcc --reference O/ref --rate R
python -m reverberate.wave.lowband compare --bundle B --out O/line --line line.npz --rate R

# The ledger: node updates a second, seconds and USD a source position and a pair.
python -m reverberate.wave.lowband cost --bundle B --out O/cost --rate R --batches 1,2,4,8,16,32,64
python -m reverberate.wave.lowband cost --bundle B --out O/cost_fcc --scheme fcc --rate R
```

`compare` prints, per third octave from 100 to 1250 Hz and per ambisonic
degree, the error's median, ninth decile and worst cell: the energy of the
difference over the reference's in the 50 ms after the onset, on a head's
sphere (`low-band-translation.md`), and the level over the whole response.
An expansion about another node is moved to the reference's first. A scene
is run with the solver by `--campaign-args "--low-engine lowband"` on
`python -m reverberate.trace rent`, with `--low-scheme` and `--low-ppw` for
another grid.

## What is not known

- **Every speed.** The traffic model has not met a card, nor has a batch:
  whether thirty sources a launch run a card nearer its memory's speed than
  one engine process does is what `cost` measures.
- **Whether a cheaper grid passes.** The face centred grid's walls and the
  Cartesian grid's dispersion at 7.2 points both move a room's modes by half
  a per cent to a per cent. Against the present engine that is a difference
  in every band above the first modes, and whether it is under the level
  that matters is read on the worst third octave of `compare`, not here.
- **The face centred voxelisation of a storey.** The card's voxeliser makes
  the Cartesian grid only, so PFFDTD's runs on the host, once a dwelling;
  its time on 13 M nodes is not measured.
- **The pair's cost**, guessed above.
