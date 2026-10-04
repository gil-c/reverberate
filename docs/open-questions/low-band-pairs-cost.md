# What a source position costs under 1 kHz

Date: 2026-10-04

Status: a model fitted to one logged campaign, not yet a measurement of the
campaign it prices. Written for lot L4 of ADR 0016. The code is
`reverberate.accel.pairs`; its `estimate` is this note as a function. The
first campaign of pairs replaces the fitted terms with its own
(`report.json`, one record per source position).

## The use

A scene has about two thousand source positions (stations, and rails every
8 cm) and each is heard at the few listening cells the listener is near while that
source sounds. A pair is one source position and one cell: 64 channels, 1.2 s,
kept at 4 kHz, 1.23 MB. A field is the opposite shape, one source and 437
cells, and the campaign that makes fields was priced for that.

## What was logged

The line campaign of 2026-10-03
(`data/runs/w44_clarify_interpolation/line_home2/`, `onebox.json`,
`pulled/campaign.log`, `pulled/report.json`): instance 54075335, 2 x A100
80 GB PCIe, 1.74 USD/h, 1.187 h billed, 2.189 USD. One source, 341 cells.

| stage | seconds | of the time billed |
| --- | --- | --- |
| rent, build the engine, provision, push | 81 | 2 % |
| voxelise three grids | 240 | 6 % |
| audit view | 144 | 3 % |
| plan the arrays | 34 | 1 % |
| low band: comms 7, engine 100, encode 196 | 306 | 7 % |
| mid band: comms 9, engine 380, encode 108 | 498 | 12 % |
| high band: comms 12, engine 1758, encode 72 | 1842 | 43 % |
| assemble the field | 268 | 6 % |
| fetch 5 GB to the laptop | 768 | 18 % |

The low band is the only line a scene keeps. Its 306 s were: 6.8 s writing
the receivers' file for 335 544 nodes; about 72 s of time stepping (the
engine's log reads 41.3 % at 30 s and 84.1 % at 60 s) and 28 s writing
29.3 GB of pressure; 104.4 s preparing the encoder; 91 s encoding 341 cells,
0.267 s each.

## The fit

The stencil alone was never timed on the low grid. Three solves of one
campaign give it: each is `updates / r + receiver samples x a`.

| band | node updates | receiver samples | engine, s |
| --- | --- | --- | --- |
| low | 4.264e11 | 7.33e9 | 100.3 |
| mid | 3.321e13 | 1.014e10 | 379.7 |
| high | 1.962e14 | 7.61e9 | 1757.9 |

The low and the mid band give `r = 1.354e11` node updates a second on the
two cards and `a = 13.25 ns` per receiver node and step, the engine's copy
to the host and its write. The high band, not used in the fit, is then
predicted at 1550 s and took 1758: the largest grid runs 13 % under that
rate. On the grid to 1 kHz the stencil is therefore **3.15 s** a source
position; the 6.4 s of ADR 0014 was the plan's estimate. A field's low band
spent 97 % of its engine's time on receivers.

## Where a field's campaign is wrong for pairs

For 275 source positions and 6000 pairs, the scene the brief expected, on the
grid to 1 kHz:

- **The encoder is prepared once per solve**: 104 s each, 28 710 s of
  33 007 s, 13.9 of 15.95 USD. `PairsCampaign` prepares it once for the
  campaign: every cell's array is the same shells about a node.
- **Every cell is written as 57 600 samples at 48 kHz**, 14.7 MB, then
  band limited at assembly. Pairs are fitted at 4 kHz on 4800 samples and
  written in that form, 1.23 MB. The fit's cost goes as its bins, which the
  rate does not change; the transforms and the file do.
- **The air and the three band assembly are applied on the machine.** For
  pairs neither is: the air is the recipe's and the masks are the
  crossover's, and the trace applies both
  (`reverberate.spatial.lowband.to_stored`). A pair's key holds neither.
- **Not wrong, and kept**: one source per solve, and the receivers' records
  held on the host. Both are the engine's. The second costs 0.43 s a pair.

## Which side is solved

By reciprocity the response from a source to a cell could be solved from the
cell. A cell's channel is a filtered sum over the thousand nodes of its
array, so the reciprocal source is that array driven by the encoder's
weights, one solve per channel: **64 C solves for C cells against S for S
source positions**, each reciprocal solve giving every source position at
once, as point receivers that cost nothing.

On the first recipe of lot L2 (hssd_0076, seed 20261004, 20 minutes, 14
sources; `reverberate.scenes.low_band_positions` counts 2151 positions on
its 60 stations and 144.6 m of rail, of which the sources pass through
1919 and are audible at 1636): the listener walks 140 m and rests at 13
places for four fifths of the time. Cells every 0.15 m along that path are
895.

| what is solved | solves |
| --- | --- |
| every source position the sources are audible at | 1636 |
| every cell, reciprocally: 64 x 908 | 58 112 |
| the 13 resting places reciprocally (832), and forward the positions heard while the listener walks (1167) | 1999 |

**The source side is solved.** Reciprocity loses even where it is at its
best, a listener at rest: 469 positions are heard at rest only, and the 13
cells that would spare them cost 832 solves. It would win for a listener who
rests at fewer than 8 places, and it needs an excitation the engine does not
have.

## The grid: to 1500 Hz, not 1000

The crossover's low masks reach zero at 1414 Hz
(`Crossover.band_hz`). A field's low band stops at 1000 Hz: band limited
there by a filter 6 dB down at 1000 Hz, and not fitted by the encoder above
it. The validated field took its wave data over 800 Hz from the grid to
4 kHz (`spatial.bands.CROSSOVER_OF_FMAX`). **A low only solve must be valid
to 1414 Hz**, and `spatial.lowband.solve_fmax_hz` gives it 1500 Hz:

| `fmax` | step | nodes | steps | VRAM | low side lost, 707 to 1414 Hz | stencil |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 Hz | 32.7 mm | 19.5 M | 21 846 | 2.3 GB | 2.75 dB, nothing over 1 kHz | 3.15 s |
| 1414 Hz | 23.1 mm | 55.2 M | 30 895 | 2.6 GB | 0.32 dB | 12.6 s |
| **1500 Hz** | 21.8 mm | 65.9 M | 32 769 | 2.7 GB | 0.21 dB | **15.9 s** |
| 1768 Hz | 18.5 mm | 107.9 M | 38 624 | 3.1 GB | 0.06 dB | 30.8 s |

1768 Hz puts the ramp's top at 0.8 of `fmax`, the rule of a field's seams,
at twice the price of 1500 Hz for 0.15 dB.

## The model

On 2 x A100 at 1.74 USD/h, the grid to 1500 Hz:

| term | seconds | US cents |
| --- | --- | --- |
| per source position: the stencil | 15.9 | 0.77 |
| per pair: the engine's output 0.43, comms 0.02, the fit 0.40 | 0.85 | 0.041 |
| once: the encoder's preparation | 157 | 7.6 |
| once: rent, build, provision, push, voxelise one grid | about 200 | 10 |
| fetch, per GB, at the 6.5 MB/s of the line campaign | 154 | 7.4 |

**One 20 minute scene, 14 sources**, the recipe above. The pairs are counted
on it step by step: two path cells a step for each audible source, each
source reading the cells of its own stride (`low-band-translation.md`):
8635 pairs.

| source positions | pairs | stencil | pairs | total | USD | cache |
| --- | --- | --- | --- | --- | --- | --- |
| 1636, the audible ones | 8635 | 7.2 h | 2.0 h | 9.3 h | 16.2 | 10.6 GB |
| 1919, every one passed | 8635 | 8.5 h | 2.0 h | 10.6 h | 18.4 | 10.6 GB |
| 2200, the central case | 8635 | 9.7 h | 2.0 h | 11.8 h | 20.6 | 10.6 GB |

and 27 minutes more, 0.8 USD, to bring the cache home. **The stencil is four
fifths of it: 0.77 cents and 15.9 s a source position, 17 USD for 2200.**
The brief's 150 to 400 positions would have been 3 to 7 USD; the recipe has
six times as many, because every metre of rail is 12.5 solves. The account
held 31.89 USD on 2026-10-04.

What lowers it without a new engine, each a decision and not a default:

- **Solve only where a source is audible**: 1636 of 1919, 15 % less. The
  trace knows; the cache then fills per recipe, not per rail.
- **The grid to 1414 Hz**: 12.6 s for 15.9, 21 % less, for 0.11 dB of the
  low side.
- **A wider rail pitch is not open.** Two positions 8 cm apart averaged hold
  1 kHz to -13.4 dB; at 16 cm, -3.0 dB, and aligned on the direct delay
  first, -6.2 dB (`line_gaps/summary_gaps.json`).

## What is not known

- **The stencil's rate on the low grid alone**, and on one card. The fit
  assumes the low and the mid band run at one rate. A grid of 66 M nodes
  may not fill two A100s.
- **Any other card.** The grid needs 2.7 GB, so every NVIDIA card holds it,
  and a card at a tenth of the price may solve it at a third of the speed.
  The encoder fits in float64, which consumer cards do slowly; nothing here
  prices that.
- **Several engines at once.** `--solvers` runs one engine per card, or
  more. It cannot raise the cards' node updates a second; it hides the
  engines' start and their writes behind each other. Not measured.

## Is the engine too dear, and what would replace it

Yes, for 2200 positions: 20 USD and half a day for one scene, two thirds of
the account, and every change the owner asks for after listening that moves
a rail pays again for its positions.

**Solving many sources on one card does not by itself lower it.** The
stencil is bound by the card's memory traffic: 2200 solves of 2.16e12 node
updates are 4.75e15 updates however they are batched, 9.7 h of two A100s.
The grid is small enough for many at once: a source is two pressure fields
of float32, 8 bytes a node, and the boundaries' state, with the geometry
shared; on the grid to 1500 Hz that is 0.53 GB a source plus about 0.2 GB,
**about 100 sources on an A100 80 GB and 30 on an RTX 3090** (about 280 and
80 on the grid to 1 kHz). What that buys is the rest: no engine start, no
file and no log per source, and a cheap card run full.

What a purpose-built low band solver must do to lower the bill, in the
order it pays:

1. **Fewer node updates per source**, which is the only thing that moves the
   17 USD. A scheme that holds its dispersion at 7 points per wavelength
   instead of 10.5 (PFFDTD's own face centred cubic grid is one) has 3.4
   times fewer nodes; updating the air inside the dwelling's hull and not
   its bounding box saves about a quarter more. Together four to five
   times: **about 3.5 s and 0.17 cents a source position, 3.7 USD for
   2200**. The voxeliser and the encoder's dispersion model follow the
   scheme, and the validated field is the reference it must reproduce.
2. **Receivers and the fit on the card.** The engine copies one sample a
   receiver a step to the host and writes them at the end, 0.43 s a pair,
   and the fit is 0.40 s more: 3.5 USD of the scene. Kept on the card, the
   arrays' records need no copy and no file; the fit stays.
3. **Many sources in one run**, as counted above, for the fixed costs per
   solve this note could not measure and for cheaper cards.

With 1 and 2 the scene is about 5 USD and three hours. Until then the
present engine can make the first scene for about 16 USD if only audible
positions are solved; nothing in `reverberate.accel.pairs` has to change
for the new solver but the function it calls to solve.

## Command lines

On the laptop, a bundle from a recipe or from a list of positions, and its
estimate:

```
python -m reverberate.accel pairs-bundle --out B --scene-id 104862621_172226772 \
    --models-from <an earlier export's storey directory> \
    (--recipe recipe.json | --sources positions.npy) --cells cells.npy [--heard-at heard.json]
python -m reverberate.accel pairs-estimate --sources 1636 --pairs 8635 --rate 1.74
```

The rental, by the orchestrator, with the driver a field uses:

```
python -m reverberate.gpu.onebox --bundle B --home H --hours 12 --max-dph 2.0 --gpu A100 \
    --no-cache --yes
```

On the machine the driver runs `python -m reverberate.accel campaign
--bundle B --out O`, which reads what the bundle is. It resumes: a pair in
the cache is not solved, and a source position with none to make is not
solved at all. At home:

```
python -m reverberate.accel pairs-install --pulled H/pulled/pairs [--publish]
```

puts the pairs in `<data root>/cache/low-pairs/<voxel_low_key>/` and, with
`--publish`, in the store under `low-pairs/<voxel_low_key>/`.
