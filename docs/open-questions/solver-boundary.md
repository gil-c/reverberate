# The wave solver's walls, and its other waste

Date: 2026-10-05

Status: **phase one of lot L25 of ADR 0016: designed, written and proven on
the laptop's cores. No card was rented and no card has run any of it.** The
performance audit (`performance-audit.md`, sections 4 to 6) found that the
walls are 51 % of the bytes a step of the low band solver moves, that a
material needs seven branches under 1500 Hz where eleven are kept, that the
fit after a solve is 13 to 37 times its floor, that 11 % of the nodes are air
outside the dwelling, and that five recipes of a dwelling ask three times the
solves they need. This note says what was built for each, what the laptop
could prove, and what a card must still say (section 8, with its commands).
Nothing here changes a rendered number by default: every option that does is
off, and is in the pairs' keys when on.

## 1. The answer

| | what | proven here | changes responses | default |
| --- | --- | --- | --- | --- |
| A | a lossy node's branches updated in the stencil's own thread | the card's own text, compiled for the host, equals the `numpy` step to the bit: both grids, both ways of the boundary, a batch and its sources alone | no | **on** |
| B | each material fitted again for 40 to 1500 Hz with fewer branches | 7 branches hold all 41 materials under both bars; 6 leave six of them over | yes | off (11) |
| C | the air outside the outer walls found and cut off | found from the grid alone on both grids of hssd_0076: 12.3 % of the updated nodes, 25.5 % of the lossy ones | yes | off |
| D | the filters and the resampling as products in the fit's spectra | 62 dB under the time chain, which is the resampler's own 0.006 dB | yes (0.006 dB) | off |
| E | the longest launches first, the last of one source; the grid on the card once; a probe of the card | by test | no | on |
| F | several recipes' bundles of a dwelling as one | by test | no | a command |

Predicted on an RTX 3090 from the bytes a step moves, at the 517 GB/s the
card moved on the first whole scene (88 s a source position, 0.173 USD a card
hour):

| | walls' bytes a lossy node | GB a step | walls' share | s a position | USD a thousand positions |
| --- | --- | --- | --- | --- | --- |
| before this lot | 264 | 1.390 | 51 % | 88.0 | 4.23 |
| A | 184 | 1.184 | 42 % | 75.0 | 3.60 |
| A, 8 branches | 136 | 1.055 | 35 % | 66.8 | 3.21 |
| A, 7 branches | 120 | 1.012 | 32 % | 64.1 | 3.08 |
| A, the outside cut | 184 | 0.974 | 38 % | 61.6 | 2.96 |
| A, 7 branches, the outside cut | 120 | 0.846 | 28 % | 53.5 | 2.57 |

and the fit from 0.55 s a pair (6.1 s a position at eleven pairs a position)
to a predicted 0.02 to 0.05 s. With a campaign over a dwelling's recipes
(section 7) the count of positions falls to a third at five recipes.

## 2. A: the boundary in the stencil's pass

**What it was.** One kernel walked every stored node and made a boundary
node's rigid update; a second, one thread a lossy node, then read that value
back from the field (one value in a 32 byte sector of the card's memory,
fetched and written back: 64 bytes), took the value of two steps before from
an array kept for that alone (8 bytes), found its node through a column and a
`z` (8 bytes), and updated the branches (16 bytes a branch, 176 at eleven).

**What it is.** The first kernel goes on, for a lossy node, into the
branches: the thread still holds the rigid update and the value of two steps
before. The node's row in the branch arrays is a base a column and the count
of the column's lossy bits under its `z` (`solver.lossy_rows`): four bytes
and `Nz / 8` a column in place of an index a node. The branch arrays are
where they were, in the grid's order, which is the order the kernel walks
(`problem.build_problem` now keeps the boundary nodes in that order whatever
the entry's). The arithmetic of a node is the old kernel's line for line.
The old way is kept as `boundary="apart"`, for the measurement of one
against the other on one card.

**The proof without a card** (`wave/lowband/hostkernel.py`). The CUDA text
is compiled as C++ by the host's compiler with `-ffp-contract=off`, which is
to it what `-fmad=false` is to `nvcc`, and each launch is run as its grid of
blocks and threads. `CardStepper` takes the compiler as an argument, so the
class itself runs, with its launch grids and its arguments, on `numpy`
arrays. The tests hold, on a lossy room that straddles the face centred
grid's fold: the fields, the branches and the records of `stencil` and of
`apart` equal to the `numpy` step's to the bit, on the Cartesian and the
face centred grid, on the whole box (the halo's copies, the absorbing layer)
and on the cut one, a batch of two and each source alone. The scalar
transcription (`reference_step`) is rewritten as the new kernel, row lookup
included, and is held to the `numpy` step as before. The `numpy` step itself
is not touched: it never had the second pass.

What this does not prove: `nvcc`'s own choices and threads at once. `verify`
on a card says that (section 8); until it has, `--boundary apart` is the
kernel every cached pair was made by.

**Bytes a step** (`solver.step_bytes`, from the arrays' sizes; the reference
grid: 47 371 003 updated nodes, 2 687 916 lossy, 383 884 columns of 146):

| | apart | stencil |
| --- | --- | --- |
| the air: 12 B a node written | 568 MB | 568 MB |
| masks, 2 B a stored node; lossy bits and bases | 112 MB | 121 MB |
| branch states | 473 MB | 473 MB |
| material and surface factor (and column, `z`) | 43 MB | 22 MB |
| the field read back and written, one sector a node | 172 MB | 0 |
| the value of two steps before | 22 MB | 0 |
| **a step** | **1390 MB** | **1184 MB** |

**Per campaign, not per launch.** Each launch built a stepper, which sent
the masks, the neighbour table and the lossy nodes' arrays to the card
again: 160 MB a launch. A worker now keeps one stepper's grid for the
campaign (`LowbandPairs.grid_on_card`) and a launch shares it. The kernels
were already compiled once a process.

## 3. B: a wall fitted again for the band

`wave/lowband/walls.py`. Each material's own admittance, as its eleven
branches give it, is fitted over 40 to 1500 Hz by `M` branches whose three
numbers are free and positive. A branch `D s + E + F / s` with positive
numbers is passive, a sum of them is, and the scheme's update of such a
branch is stable for any of them: nothing is to be proven again about
stability, only that the wall is the same wall.

**The fitter.** A branch is its resonance, its quality and its conductance
at resonance, in logarithms. The start is the eleven branches; they are
taken away one at a time, each time the one whose loss the others make up
best after a fit of the rest (the best three candidates are fitted), down
to `M`. A fit is a least squares of the error relative to the admittance's
modulus with its analytic Jacobian, then Lawson's reweighting towards the
least worst error. Nothing is drawn at random. 1.5 s a material; a campaign
fits once and its workers read the file (`state/walls.json`).

**The bars.** Absorption coefficients are octave values known to a few
hundredths, so a refit may move one by a hundredth at most
(`ABSORPTION_BAR`). The solver is held against the present engine at -30 dB
in the worst third octave over the 50 ms after the onset; a reflection's
error at each of a handful of bounces stays 10 dB under that when it is
under -46 dB (`REFLECTION_BAR_DB`).

**The table**: the 41 materials of hssd_0076, the worst third octave from 40
to 1500 Hz of the worst material (`python -m reverberate.wave.lowband walls
--materials BUNDLE/pairs/models/materials`):

| branches | admittance, relative | reflection at normal incidence | absorption, normal | absorption, random | materials over a bar | worst |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.5 % | -67.7 dB | 0.0006 | 0.0009 | 0 | painting |
| **7** | **3.7 %** | **-51.3 dB** | **0.0043** | **0.0068** | **0** | plant |
| 6 | 11.5 % | -41.9 dB | 0.012 | 0.017 | 6: bed, book, couch, seat, shell, window shade | bed |
| 5 | 12.1 % | -39.7 dB | 0.018 | 0.024 | 25 | couch |

The shell, which two lossy nodes in three carry: 0.5 % and -66 dB at seven;
5.2 %, -46.5 dB and 0.012 of random incidence absorption at six. The
audit's plain least squares gave 1.3 % at seven and 16 % at six; a better
fitter moves six to 11.5 % and not under the bars. **Seven is proposed.**
Six is one branch fewer for a wall that is then another wall by a hundredth
of absorption, on the material that is most of the dwelling.

The option is `walls=N` of `LowbandPairs` (`--walls N` of `compare` and
`cost`); the default is the materials as they are until the card's
comparison passes (section 8). The solver's name in a pair's key then says
"walls of N branches fitted from 40 to 1500 Hz", so the caches never mix.

## 4. C: the air outside the outer walls

**What it is**, read in the first scene's `as_computed.npz` (both grids):
the export wraps the storey in a shell 0.39 to 0.41 m beyond the outer
walls' outer faces. The grid holds the air between: a ring all round the
dwelling, 18 nodes wide, from floor to ceiling (node layers 5 to 133),
57 m round, 60 m3, closed on both sides and at both ends by nodes of the
shell's own material. It has no opening to the box's absorbing layer: it is
a corridor, not outdoors. Two openings of the outer walls lead into it, one
on each long side, each about 1.7 m wide and the full height of the storey;
a thin leaf stands in each, within the wall's thickness, as high as 2.1 m
and with gaps. Above the leaf the opening is whole. So the audit's
"joined between 2.1 and 2.6 m" is the part with no leaf; through the leaf's
gaps the ring is joined at most heights (one component of reached air at
nearly every layer from 0.3 m up).

**What it costs.** 5 829 632 of the 47 371 003 updated nodes (12.3 %), and
685 565 of the 2 687 916 lossy ones (25.5 %): a ring is nearly all wall.
17 % of a step's bytes with the boundary in the stencil's pass. No listening
cell and no source is in it (counted: 0 of 817 704 array nodes, 0 of 12 232
source nodes). On the grid at 7.2 points the same 12.3 %.

**What it does to the response**, as far as a laptop says. The dwelling on
a grid six times coarser (0.13 m, to 230 Hz, 219 000 nodes, materials by
the commonest of each block, the thin leaves lost), one source, 24 cells,
1.2 s, four ways; energy in windows of 0.1 s, median over the cells, dB re
the first window:

| 40 to 100 Hz | at 0.45 s | at 0.95 s | slope, first 0.5 s | slope, last 0.6 s |
| --- | --- | --- | --- | --- |
| as exported, openings whole | -39.0 | -84.0 | -102 dB/s | -77 dB/s |
| as exported, a leaf to 2.1 m | -37.3 | -79.3 | -98 | -70 |
| the outside cut, openings open | -41.1 | -88.5 | -108 | -71 |
| the outside cut, openings rigid | -33.1 | -72.1 | -89 | -72 |

| 100 to 230 Hz | at 0.45 s | at 0.95 s | slope, first 0.5 s | slope, last 0.6 s |
| --- | --- | --- | --- | --- |
| as exported, openings whole | -27.7 | -66.5 | -72 dB/s | -76 dB/s |
| as exported, a leaf to 2.1 m | -27.3 | -64.5 | -71 | -76 |
| the outside cut, openings open | -30.6 | -72.9 | -79 | -80 |
| the outside cut, openings rigid | -23.0 | -58.2 | -58 | -68 |

Read: under 230 Hz the ring is not a reservoir that makes a second slope.
The decay has one slope in all four, within its own curvature. The ring
behaves as a poor absorber: against openings that let the sound out it
leaves 4 to 9 dB more after a second, and against a wall in their place 6
to 12 dB less. **The two slopes the audit read on the whole band (40 dB in
0.43 s, then 20 dB in 0.62 s) are not reproduced here in any of the four,
so the ring is not shown to be their cause; they are strongest from 500 to
1400 Hz, where this coarse grid says nothing.** That is one comparison on a
card (section 8).

**What is right.** Outdoors is free field: what leaves by an opening does
not come back. A ring lined like a wall that returns a part of it a second
later is not a dwelling's outside. Whether each opening is open (a door
ajar, a window with no glass) or shut (then it is a wall of the leaf's
material from floor to lintel) is a fact of the dwelling, to be said by the
export; the grid shows it is neither today. The two differ by 15 dB after a
second at low frequencies: it matters more than the ring does.

**What is built** (`wave/lowband/outside.py`). `outside_air` finds the ring
from the reached air alone: from each face of the box inwards, a row of
nodes first meets the shell's air, which ends on the outer wall the same
number of nodes in on nearly every row (83 to 94 % here); the rows through
an opening go on, and are cut where the others end. A face is a ring only
if that gap is under 0.8 m and most of its rows have air behind the wall,
so a shallow room is not taken for one, and a dwelling that does not leak
has none. `without` cuts those nodes off from both sides: the nodes of the
opening's plane become boundary nodes that read the dwelling only, with
nothing on them (`"rigid"`) or an admittance of one (`"open"`: a wave that
meets the opening squarely is absorbed, 45 dB in a duct under its first
cross mode by test). `outside=` of `LowbandPairs` and `--outside` of
`compare` and `cost`; off by default; in the pairs' keys.

## 5. D: the fit in its spectra

`fit.SpectralChain`. The integration with the 40 Hz low cut, the zero phase
low pass at 1500 Hz and the resampling to 4 kHz are linear and do not
depend on time, and the fit begins with a transform and reads 3600 bins. So
the records are transformed once as they are, in single precision, their
spectrum is read at the fit's frequencies, multiplied there by the filters'
exact responses (the sections' own, by `sosfreqz`), and handed to the fit's
operator. The fit's frequencies are not bins of a transform at the grid's
rate (27 307.1 Hz against 4 kHz over 9600); a record is 32 769 samples and
nothing more, so its spectrum between the bins of a transform padded four
times is read by a Kaiser windowed sinc of sixteen taps, to a millionth.
One code for `numpy` and for a card: a transform, sixteen gathers, a
product. No recurrence, no weights a sample, no double precision record.

**Against the time chain**, on a record that decays 60 dB with a direct
sound (the test; 1.2 s here):

- 62 to 63 dB under the response, window by window of 50 ms each against
  its own energy, for the first second. That difference is one thing: the
  resampler's passband is 0.006 dB over one (measured on its own: `7.0e-4`
  from 0 to 1480 Hz), and an exact cut of the band is not.
- in the last tenth of a second the two part, to -39 dB of the window's own
  energy: the time chain stops its filters with the record, the spectra let
  the low cut ring on. The response is 60 dB down there; the difference is
  94 dB under the response's start.
- on a window shorter than the low cut's own memory (25 ms) the two are not
  comparable, and a campaign's window never is.

On `numpy`, 40 records of 32 769 samples: 2.3 s by the chain, 0.22 s by the
spectra. On a card: a single precision transform of 984 by 131 072 and a
double precision product of 3600 by 64 by 984: predicted 0.02 to 0.05 s a
pair, to be measured by `cost --fit spectra`. `fit="spectra"` of
`LowbandPairs`; off by default; in the pairs' keys, since 0.006 dB is not
the same bits.

## 6. E: the order of a run, and a probe

- **Longest first, the last short** (`pairs.last_short`). A queue's
  launches go by their count of sources, most first; the last two a card
  are parted into launches of one source, so that a card waits half a
  source position at the end and not half a launch of eight (run A: 4440
  card seconds idle). A source costs the same card seconds alone. Launches
  of one length keep the positions' order, so neighbouring pairs still come
  home together. A run of no more than two launches a card is left as it
  is.
- **A probe** (`pairs.probe`, 10 s before a worker's first launch, on the
  campaign's own grid with one source): node updates a second, the bytes
  they are, and for a card whose memory rate is known what the bytes
  predict at the 55 % the kernels were measured at. Under 70 % of that the
  log says the card is slow. It is in `lowband.json` (`probes`) and in
  `state/probe.*.json`. Scene B's 1.0e10 updates a second where the bytes
  give 1.4e10 would have been said in its first minute.

## 7. F: a dwelling's recipes solved once

A pair's key holds its two positions in whole millimetres, the grid, the
encoder, the solver and the window, and no recipe. So nothing in the cache
or the campaign needs to change: `pairs.merge_bundles` (`python -m
reverberate.wave.lowband union --bundles B1 B2 ... --out U`) writes one
bundle whose source positions are the union to the millimetre, whose cells
are the union, and where a position is heard at every cell any recipe hears
it at. A campaign over it solves each position once and writes every
recipe's pairs under the names that recipe's trace will ask for; `items`
already groups a position's cells into one solve, split only when their
records pass the host's memory.

What the trace must pass (`trace/`, not this lot's): build the union before
the rental and hand its campaign's pair cache to each recipe's trace
(`--reuse-from` is that hand); and name the solver as the campaign does,
`low_grid(..., walls=, outside=, fit=)`, when any of the three options is
on, or the bundle's keys will not be the machine's.

## 8. What a card must say, and the commands

One RTX 3090 for about 45 minutes, 0.15 USD. `B` is a pairs bundle of the
dense line (341 cells, one source), `O` an output directory, on the machine.

```
# A: the kernel on a card equals numpy, both ways (2 min)
python -m reverberate.wave.lowband verify --out O/verify
#   card_equals_numpy, boundary_apart_equals_numpy, batch_equals_singles_on_card: all true

# A: before and after on the same card, with the probe (2 x 4 min)
python -m reverberate.wave.lowband cost --bundle B --out O/cost_apart --boundary apart --rate 0.173
python -m reverberate.wave.lowband cost --bundle B --out O/cost_stencil --rate 0.173
#   predicted: 88 to 75 s a position; gb_per_s says whether the card's rate held

# B: the refitted walls against the present engine on the dense line (4 x 6 min)
python -m reverberate.wave.lowband compare --bundle B --out O/w11 --reference O/ref --pffdtd /root/pffdtd
python -m reverberate.wave.lowband compare --bundle B --out O/w8 --walls 8 --reference O/ref
python -m reverberate.wave.lowband compare --bundle B --out O/w7 --walls 7 --reference O/ref
python -m reverberate.wave.lowband compare --bundle B --out O/w6 --walls 6 --reference O/ref
python -m reverberate.wave.lowband between --reference O/w11 --candidate O/w7
#   the bar: worst third octave -30 dB and level within 0.5 dB, as w11 itself passes it
python -m reverberate.wave.lowband cost --bundle B --out O/cost_w7 --walls 7 --rate 0.173

# C: the outside cut, against the grid as exported (2 x 6 min)
python -m reverberate.wave.lowband compare --bundle B --out O/open --outside open --reference O/ref
python -m reverberate.wave.lowband between --reference O/w11 --candidate O/open
#   then the decay of both caches, 500 to 1400 Hz: is the second slope the ring's

# D: the fit in its spectra (6 min)
python -m reverberate.wave.lowband compare --bundle B --out O/spectra --fit spectra --reference O/ref
python -m reverberate.wave.lowband between --reference O/w11 --candidate O/spectra
python -m reverberate.wave.lowband cost --bundle B --out O/cost_spectra --fit spectra --rate 0.173
#   predicted: 60 dB under w11; a pair in 0.02 to 0.05 s
```

## 9. What is the owner's to decide

1. **Seven branches** as the default once `compare` passes, or eight for a
   margin of 15 dB more at 4 % of a step.
2. **What an outer opening is**: open, shut, or as exported. It changes the
   low frequencies' late level by up to 15 dB and is 17 % of a step. The
   correctness audit should see the slices first.
3. **The fit in its spectra** as the default: it moves every pair by 0.006
   dB, so every cached pair is then another pair.
4. Whether the union of a dwelling's recipes is the trace's default.

## 10. Not done

- Nothing ran on a card.
- The trace does not pass the new options (`trace/engines.py`,
  `trace/bundle.py`): they are reachable from `LowbandPairs` and from
  `compare` and `cost`.
- The 45 % of the card's memory rate the kernels do not use (the audit's
  section 5) is not looked for: it needs a card.
- `Problem.bytes_per_source` still counts the value of two steps before: an
  upper figure for both kernels, 21 MB a source on the reference grid.
- Where the export's shell comes from, and whether every storey has the
  ring, is for the scenes' lot; `python -m reverberate.wave.lowband outside
  --entry E --source X Y Z` says it for any entry.
