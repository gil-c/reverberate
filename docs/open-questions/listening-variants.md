# Which cheaper trace sounds the same

Date: 2026-10-05

Status: the options exist, each leaves the default pack as it was, and none
is decided: the owner decides by ear, A against B, on one excerpt. Written
for lot L12 of ADR 0016. This note holds what was measured before the
variants were offered, and how they are put side by side.

## The reference and its variants

The reference is the Cartesian grid at 10.5 points per wavelength to
1500 Hz, rails every 8 cm read linearly from two positions, a low response
of 1.2 s, 100 000 rays in double precision. The set of the first scene is
`src/reverberate/trace/sets/listening_v1.json`:

| variant | what it takes | whole scene, USD predicted | saved |
| --- | --- | --- | --- |
| reference | nothing | 7.54 | |
| grid-7.2 | `--low-ppw 7.2` | 2.80 | 63 % |
| rail-10cm-x8 | rails every 10 cm, `--rail-positions 8` | 6.70 | 11 % |
| rail-12cm-x8 | rails every 12 cm, `--rail-positions 8` | 5.98 | 21 % |
| low-0.8s | `--low-seconds 0.8` | 5.07 | 33 % |
| rays-50k | `--rays 50000` | 7.47 | 1 % |
| all-cheap | 7.2, 12 cm read from 8, 0.8 s | 1.81 | 76 % |

Priced on the realistic recipe of hssd_0076 (1200 s, 14 sources) at
0.136 USD/h, the rate of the card the batched solver was measured on: one
card's machine-seconds, not an offer's wall time. A rail read from eight
positions is solved at fewer positions (1096 at 12 cm against 1529) and
heard at more pairs (23 219 against 16 887): a pair's filters and its way
home are part of what it saves less than its solves suggest.

`python -m reverberate.trace variants --recipe R.json --window START|auto
SECONDS --home DIR --mirror ...` prints that table for any recipe, and the
`trace rent --smoke` line of each variant's excerpt.

## A rail's pitch is the recipe's

A variant of the rails is another recipe: the reference's with `pitch_m`
changed on every rail and in its `generator` block. Checked on the
realistic recipe: at 0.10 and at 0.12 m that rewrite is byte for byte what
the generator draws from the same seed and the same parameters with that
pitch. The stations, the movements and the clips are the reference's, so
the comparison (`render check --against`) and the page accept the two packs
as one scene: the same steps, the same sources and the same head at every
step.

## A low response of fewer seconds

`--low-seconds S` makes the solver simulate `S` seconds. The stored
response keeps the pack's 4800 samples: it is faded to nothing over the
20 ms before `S` (a raised cosine) and is silent after. A pair's key names
its window, so a cache never mixes two durations.

Measured on 894 pairs of the scene's own run (hssd_0076, 1500 Hz, 0.7 to
13.4 m, far and shadowed pairs among them), channel 0, the energy after the
cut over the pair's whole energy:

| after | worst pair | median |
| --- | --- | --- |
| 0.6 s | -42.5 dB | -49.7 dB |
| 0.8 s | -45.3 dB | -54.5 dB |
| 1.0 s | -47.5 dB | -58.2 dB |

Per third octave, the energy after the cut over that band's own energy,
worst pair (median):

| band | 0.6 s | 0.8 s | 1.0 s |
| --- | --- | --- | --- |
| 50 Hz | -35.5 (-50.2) | -40.2 (-56.4) | -45.1 (-61.6) |
| 63 Hz | -33.5 (-55.1) | -36.6 (-57.7) | -39.9 (-61.9) |
| 80 Hz | -35.5 (-52.0) | -37.9 (-57.8) | -39.4 (-62.4) |
| 100 Hz | -34.4 (-47.2) | -41.2 (-58.2) | -43.5 (-64.3) |
| 125 Hz | -35.7 (-48.2) | -43.9 (-59.9) | -46.9 (-66.3) |
| 160 Hz | -36.0 (-46.9) | -46.1 (-60.0) | -51.7 (-68.1) |
| 200 Hz | -34.4 (-46.8) | -50.2 (-61.1) | -54.1 (-70.0) |
| 250 Hz | -46.3 (-57.4) | -53.8 (-67.9) | -57.0 (-72.0) |
| 315 Hz | -37.2 (-55.2) | -54.6 (-68.1) | -59.9 (-73.8) |
| 400 Hz | -50.9 (-64.0) | -55.3 (-72.4) | -56.7 (-75.5) |
| 500 Hz | -53.3 (-69.8) | -54.5 (-74.5) | -57.2 (-77.4) |
| 630 Hz | -50.5 (-66.3) | -58.4 (-75.1) | -61.0 (-78.5) |
| 800 Hz | -56.2 (-69.0) | -58.3 (-76.9) | -60.4 (-79.7) |
| 1000 Hz | -57.6 (-72.1) | -60.6 (-77.8) | -63.1 (-80.0) |
| 1250 Hz | -55.4 (-66.6) | -62.7 (-75.8) | -65.4 (-78.1) |

What this says, and what it does not:

- The dense line gave -62 dB at worst after 0.8 s. The scene's own pairs
  give -45 dB: the far and shadowed pairs decay more slowly, and the worst
  band is 63 Hz at -36.6 dB of its own energy.
- In absolute terms the part cut at 0.8 s is at most -44 dB under the
  direct sound of a source at 1 m (median -61 dB). The 1.2 s kept today is
  itself a cut: its last 50 ms are at -54 dB at worst on the same scale.
- **The mirror's tail does not mask it.** In the three packs at hand
  (`smoke1`, `v1_near`, `v1_far`) the tail's histograms hold nothing after
  0.6 to 0.8 s in the bands from 1 kHz: the rays have fallen under their
  energy floor. After 0.8 s the low band is the only thing still sounding,
  at -58 to -72 dB under its own first 0.2 s in those pairs. Five pairs are
  not a sample; the direction is clear.
- All 64 channels summed give far worse figures (-17 dB at worst after
  0.8 s) that are not the room's: the high orders hold the fit's noise at
  frequencies an array of 0.26 m cannot resolve, which rings to the end of
  the response. Channel 0 is read here for that reason, and the question of
  what the engine does with those channels is open.

Recommended to listen to: 0.8 s. Going from 1.2 to 1.0 s buys 2 dB of
what 0.8 s costs and saves half as much. 0.6 s is 3 dB worse than 0.8 s at
worst and 14 dB worse in the bands from 125 to 315 Hz.

## Fewer rays, and single precision

`--rays N` casts `N` rays a tail site. A ray carries one part in `N` of the
source's energy, so nothing downstream is scaled. The noise, measured on
the twin (the Python tracer, on a box of 25 % absorption and 40 %
scattering, three receivers), as the difference of two histograms in dB
over the bins and windows within 60 dB of the loudest:

| case | 2 ms bins, RMS (worst) | 50 ms windows, RMS (worst) | whole tail, worst |
| --- | --- | --- | --- |
| 100 000, seed to seed: the yardstick | 0.49 (2.4) | 0.14 (0.3) | 0.19 |
| 50 000 against 100 000 of another seed | 0.59 (3.8) | 0.15 (0.4) | 0.27 |
| 25 000 against 100 000 of another seed | 0.75 (3.1) | 0.18 (0.5) | 0.36 |
| 50 000, seed to seed | 0.69 (4.0) | 0.16 (0.3) | 0.30 |
| 25 000, seed to seed | 0.93 (3.8) | 0.22 (0.6) | 0.45 |

Two things qualify those figures. The twin traces 400 rays a second, so
the counts were scaled: 12 800 rays on a receiver sphere of 0.559 m stand
for 100 000 on the tracer's 0.20 m, which keeps the hits a bin. And the
box's absorption is the same in every band, so the seven bands read alike:
a dwelling's bands differ by how long each lasts, not by this count.

**Single precision is not offered.** The rays' kernel is the bit for bit
twin of the Python tracer: it shares its geometry with the paths' kernel,
draws its uniforms as 53 bits of a hash and counts its histograms in
integers so that a card and a host agree exactly. A single precision
variant is another kernel and another twin, and its identity could not be
measured on the host's path without rewriting that path in single
precision first. The rays are 0.14 USD of the scene's 7.54: the option
would save a part of 2 %.

## Hearing them

`python -m reverberate.render check REF.h5 --against V1.h5 V2.h5 ... --out
DIR` writes `listen/<name>_mix.wav` for every pack (the same window, head
and clips, one gain for all), `variants.md` (each variant less the
reference per third octave and per source, early and late, and the largest
of those under the crossover and over it) and `blind/`: the mixes as
`X1.wav` ... in an order drawn at random, with `key.sealed`, which `python
-m reverberate.render unseal` reads.

`python -m reverberate.apps.compare DIR` (`docs/apps.md`) is the page made
for this decision and nothing else: the variants as buttons switched at the
same instant, the sonograms, the tables above as curves, and a blind test
(ABX or a ranking) whose result is written beside `DIR`. With `--ambisonic`
the check also keeps each source's order 7 stem, which that page decodes
under the head its listener turns, and with `--signals clips clicks pink`
it feeds the sources clicks and pink noise beside the speech.

On the inspector's page, every pack given by `--pack` that is the scene of the pack
heard has a button in the sound bar. A press, or `[` and `]`, changes the
pack heard at the same instant; the note beside it says which variant is
heard, what the whole scene is predicted to cost with it and what its own
rental was billed. The names and costs come from the `variant.json` the
`variants` command writes in each variant's home. The "As computed" fold
takes any two of those packs as A and B.

## Not verified without a card

- That a launch refused its memory is parted and tried again on a real
  card, and that the pool gives its blocks back as intended
  (`reverberate.wave.lowband.pairs`): tested with an allocator that
  refuses, not with cupy's.
- A low band solved for 0.8 s end to end: the solver's steps, the fit's
  operator at 3200 samples, the trace's fade. Each part is tested on the
  host; the whole was not run.
- What `--rays` saves on a card: priced in proportion to the count.
