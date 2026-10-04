# 0016: a scene moves, and one engine renders it for the audit and for training

Status: proposed. Builds on ADR 0014, also proposed, whose joined field is the
render the owner validated by ear. Supersedes part of ADR 0013 and reopens two
of the roadmap's non-goals; which parts is said below, and the wording
proposed for the roadmap is in `0016-appendix-roadmap-text.md`.

## Context

Clarify trains speech separation for hearing aids. It needs acoustic scenes
in which everything moves: near voices, far voices, noise sources, the
listener, over sequences of twenty minutes. This project renders one fixed
source over a lattice of listening points 0.40 m apart (ADR 0015), and its
page lets one listener walk that lattice by swapping the nearest cell's
response (ADR 0013). Neither moves a source, and neither gives a signal: the
page plays to two ears, and a field is 437 responses, not a sequence.

The owner's requirements, 2026-10-04: sources and listener move; the output
is the order 7 ambisonic signal at the listener, 48 kHz; the band under
1 kHz keeps the wave solver's resonances, for moving sources too; the
crossover and the mirror above it are those validated by ear; the cost of a
scene is brought to its minimum, on a card, rewriting what has to be; and
everything simulated can be audited, by the engine that will feed training
and no other.

## What was measured

On hssd_0076, source S1. The data and scripts are in
`data/runs/w44_clarify_interpolation/`: `leave_one_out/` works on the 437
points of `w42_gpu_hssd_0076/field/S1.h5`; `line_gaps/` on a line solved for
the purpose, 341 points 2 cm apart, 0.31 to 4.16 m from the source
(`line_home2/pulled/field/S1.h5`; 0.97 h of 2 x A100 at 1.74 USD/h,
2.19 USD; its 18 lattice points match the earlier field to 2e-6 of the
peak). The figure is the energy of the error over the 50 ms after the onset, channel
0, against the solved response's, in dB per octave, 125 Hz to 1 kHz unless
said. Median first, ninth decile in brackets where it changes the reading.

| what | spacing | 125 | 250 | 500 | 1k | file |
| --- | --- | --- | --- | --- | --- | --- |
| average of the two neighbours | 0.40 m | -20.1 | -11.3 | -0.6 | +3.5 | `line_gaps/summary_gaps.json` |
| average of the two neighbours | 0.08 m | -42.9 | -39.1 | -27.7 | -13.4 (-12.0) | same |
| one order 7 point, read half a spacing away | 0.40 m | -31.9 | -31.6 | -27.2 (-4.8) | -23.0 (-6.7) | same |
| one order 7 point, read half a spacing away | 0.80 m | -27.7 | -24.8 | -16.1 (+19.9) | -4.8 | same |
| two points fused by plane waves, at their middle | 0.40 m | -31.8 | -30.6 | -26.9 (-6.7) | -23.5 (-8.7) | same |
| one point, read a whole pitch away, lattice | 0.40 m | -25.1 | -23.6 | -18.9 (-0.6) | -6.2 | `leave_one_out/summary.json` |
| two opposite neighbours fused, lattice | 0.80 m apart | -27.5 | -25.8 | -20.8 (-3.4) | -22.8 (-10.7) | `leave_one_out/summary_planewave.json` |

- **Averaging waveforms fails** from 500 Hz at 0.40 m. At 8 cm it holds
  1 kHz to -13 dB and gives out at 2 kHz (-3.0 dB).
- **An order 7 expansion translates** while `k d` stays near 3.5, `d` being
  the distance translated: 0.20 m at 1 kHz, 0.12 m at 2 kHz (-18.5 dB),
  0.04 m at 4 kHz (-24.7 dB). That is half a pitch `p` with `k p = 7`. A
  whole pitch of 0.40 m at 1 kHz, `k d = 7.3`, gives -6 dB.
- **Fusing two neighbours** recovers 1 kHz where one neighbour a whole pitch
  away gives -6 dB. Four neighbours gain nothing under 1 kHz (-23.0) and
  amplify from 2 kHz up (+9 to +14 dB).
- **The ninth decile is the clearance.** A tenth of the cases err by -5 dB
  or worse at 500 Hz. In the thirds from 400 to 630 Hz, the share of
  translations worse than -6 dB is 65 to 79 per cent when the nearest
  surface or the source is closer than 0.4 m to the expansion's centre, 19
  per cent at 0.6 to 0.8 m, none beyond 1 m; every case within 0.8 m of the
  source fails (`diag500.py`). An expansion translates inside its own free
  ball and no further. The lattice's median clearance is 0.65 m.
- **By reciprocity the first two rows are the source's side**: a source
  between two solved positions is their average, which holds 1 kHz only at
  8 cm.
- **The late level** differs between lattice neighbours by 0.3 to 0.7 dB in
  the median, 2.1 dB at the ninth decile, 4.6 dB at worst.
- **A floor** of -20 to -30 dB remains at every spacing, 2 cm included. Not
  explained; the solver's sub-cell receiver placement is the hypothesis and
  was not tested.
- **The mirror**, one RTX 3090, 437 points (records of 2026-09-17,
  `w42_gpu_hssd_0076/mirror/cost_study_2026-09-17`): the image tree 1.1 s a
  source, the paths 24 ms a source and point pair, the rays 16 s a source.
- **The wave solve to 1 kHz** is 6.4 s of the card's stencil (ADR 0014); the
  rest of a campaign's time is the receivers' output and its encoding.

## Decision

**A scene is a recipe, traced once on a rented card into a pack, and
rendered from the pack by one signal engine.**

```
recipe  ->  TRACE (rented card)  ->  scene pack  ->  SIGNAL (numpy or cupy)  ->  order 7, 48 kHz
               |                                        |- the laptop: the audit, in the application
        dwelling assets (cache)                         `- later: the card that trains
```

1. **The recipe** is a few kilobytes of JSON
   (`docs/formats/scene-recipe.md`): the dwelling and the versions of its
   assets, stations and rails, every source's movement, facing and activity
   with the clips named by digest, the listener's free trajectory and head,
   the air, the seed, and the ranges it was drawn from. It is deterministic
   and it is **the only thing stored**. Packs and signals are regenerated.

2. **The crossover stays at 1 kHz and the mirror above it stays the
   validated one**: `mirror.hybrid.Crossover` unchanged, image sources to
   order 3, 100 000 rays, the calibration's parameters and the settings
   pinned with them (ADR 0014).

3. **Under 1 kHz the wave solver answers, one solve per source position.**
   - *The listener is continuous by translation.* Each listening cell holds
     an order 7 expansion; the field at the head is one cell's expansion
     translated, or two cells' fused by a minimum norm plane wave fit. The
     cells are chosen **by clearance**: a cell serves a head only inside
     its own free ball, bounded by the nearest surface and by the source.
     Where the 0.40 m lattice leaves a head with no such cell, a cell is
     added.
   - *The source is continuous by sampling.* A source rests at **stations**
     and travels on **rails**. A station is solved once. A rail is solved
     every 8 cm and the source between two positions is their weighted
     average, the one case where averaging waveforms was measured to hold
     1 kHz.
   - *A seat is an exact listening position*: an array is stood where the
     seated head is and nothing is translated, because next to furniture a
     translation has no free ball. Standing and seated are two heights on
     the same grids, 1.70 m and 1.20 m.

4. **Above 1 kHz, the early part is recomputed at every step.** The image
   tree follows the source and each path is validated again; a path keeps
   one identity across steps, so its delay moves continuously and is heard
   as the Doppler shift it is. The step is 50 ms.

5. **Above 1 kHz, the tail is the rays', interpolated in energy.** Exact at
   stations; along rails and between listening cells the histograms are
   weighted in energy, never the waveforms.

6. **The join is `mirror.hybrid`'s, per step**: pressure masks over the
   onset, power masks after, one levelling scalar, here interpolated along
   the trajectories.

7. **Air absorption** is ISO 9613-1 as it exists
   (`audio.apply_air_absorption`, ADR 0009), with the recipe's humidity and
   pressure.

8. **A voice is directional by default, with a switch.** The pack holds
   omnidirectional gains and each path's departure direction; the engine
   applies the directivity, so one pack gives both renders and the switch
   costs no trace. The pattern is normalised to the omnidirectional source's
   power. It acts on the early arrivals above 1 kHz; the tail and the low
   band stay omnidirectional.

9. **The pack** (`docs/formats/scene-pack.md`) holds, per source and per
   step, the arrivals (delay, arrival and departure directions, band gains,
   identity), the histograms the tail is made from, the low band responses
   used and which cells the head is read from, and the levelling scalar. It
   holds the cells and not the translation operators: stored, those would be
   200 GB a scene.

10. **One signal engine, `reverberate.render`, on `numpy` or `cupy`.** It
    applies the pack to the dry clips and writes the order 7 signal at the
    head, in the scene's fixed frame, in blocks whose size does not change
    the result. The head's rotation and the binaural decode stay where
    ADR 0013 put them, in the page.

11. **The audit is that engine.** The responses come from the rented card;
    the signal is rendered on the laptop's processor at full resolution, and
    the application plays what the engine wrote, checked by checksum. The
    pack is the same bytes for the audit and for training. **Host and card
    agree to 1e-6 of the output's peak, and are not bit identical**: a
    transform and a sum do not round alike on the two. The tolerance is
    measured and is a criterion.

12. **The scope is one scene.** Twenty minutes on hssd_0076, and no other
    scene and no dataset until that scene is generated, the audit is
    finished, the owner has listened through it and validated it, and every
    stage's cost is at its minimum. The owner expects to ask for changes
    after listening.

13. **Everything may be rewritten** where it lowers the cost and reproduces
    the first scene within the stated tolerance: the mirror's kernels, the
    encoders, the transfers, and the wave solver itself. The project is not
    bound to PFFDTD.

## What this supersedes

**ADR 0013, for a scene.** Four of its decisions stop at the field walk,
where they stay in force:

- *Whatever depends on where the listener stands is computed in the page.*
  For a scene the engine computes it. What depends on which way the listener
  faces is still the page's.
- *The response is swapped by crossfade, never interpolated*, and its
  rejection of interpolating between cells. The reason it gave stands and is
  now measured: two cells' waveforms cannot be averaged. A scene does not
  average them; it translates an expansion under 1 kHz and moves paths
  above.
- *The propagation delay is one delay line in front of the convolutions.* A
  scene has one moving delay per path.
- *The nearest cell out to 2.2 m, and the late part every two metres.* A
  scene has no nearest cell rule: a head with no cell whose free ball holds
  it is an error of the trace, not a fallback.

Its worklet, its ring, its decode and its head rotation are untouched.

**ADR 0014, one sentence.** Its render is equal between host and card "in
law only", the tail's noise being drawn by two generators. The signal engine
draws the tail's noise from one generator on both.

**The roadmap.** Its non-goal on a second solver, which ADR 0014 already
crossed and this decision builds on. The condition the roadmap sets for
reopening it, a failed validation, is not met: the reasons are the cost law
and a requirement the roadmap did not have. Its non-goal on real time
simulation, in part: the acoustics are still computed offline, but the
signal engine renders them at about the speed they play. Its statements
that one engine covers the band and that the geometric engine is removed,
its order by band, and its dataset budget, none of which describe the
project any longer.

## Consequences

- Six lots code against the two formats: recipes (L2), the scene view (L3),
  the low band and its translation (L4), the moving mirror (L5), the trace
  (L6), the engine (L7).
- The dwelling gains a **low band pair cache**: the wave response under
  1 kHz between one source position and one cell, keyed by content, filled
  only with the pairs a recipe uses and shared by every recipe on the
  dwelling.
- A pack is about 8 GB for 20 minutes and 14 sources under the assumptions
  of `scene-pack.md`, nearly all of it low band responses, against 14.7 GB
  for the signal. Its size is a cost like any other and is in the ledger.
- A campaign of low band pairs solves many source positions and few
  receivers, the opposite of a field (ADR 0015). The solver's time is then
  its output and encoding, not its stencil, which is where a rewrite pays.
- Cost is recorded per stage from the first lot that computes: seconds,
  card, hourly rate, USD. There is no dataset budget in this decision.
- Nothing here is validated until the owner has listened. The criteria
  before that: at rest the engine returns the validated joined field; along
  the solved line it meets the 341 solved responses, listener then source;
  host and card agree; two processes give one output. The worst band is
  read, never the median, and a threshold does not move.

## Risks

1. **The cost of a source position under 1 kHz.** The engine as it stands
   spends its time writing receivers. L4 measures it before any campaign.
2. **The time the signal engine takes on the laptop** for 20 minutes at
   order 7: estimated near real time, not measured. The fallback is to
   render windows round the cursor.
3. **The calibration is one source, at 1.70 m, in one dwelling.** Seated
   heights, other source positions and the levelling scalar along a
   trajectory are not validated.
4. **Directivity departs from the validated render.** The switch gives the
   comparison. Under 1 kHz and in the tail the voice stays omnidirectional.
5. **The floor of -20 to -30 dB** of the translation is not explained.
6. **The fusion was measured on channel 0 only.** The engine needs all 64
   channels at the head; that the higher orders translate as well is
   expected and not measured.
7. **Near a source the low band has no free ball.** Every measured
   translation within 0.8 m of the source failed; a near voice is 0.5 to
   1.5 m from the head. The recipe keeps 0.50 m between mouth and head, and
   L4 must show the dense cells this needs or say that it cannot.
8. **The late reverberation follows a moving source at once**, the
   rendering being quasi-static. Unheard so far.
9. **Off the lattice's lines the head is further from a cell** than
   anything measured: 0.28 m at the centre of a square of the 0.40 m
   lattice, `k d = 5.2` at 1 kHz, where the line measured 0.20 m. L4
   measures it, and the remedy is the same as for the clearance: a cell
   added.

## Rejected

**Interpolating waveforms**, for the listener or as the general rule for
the source: -0.6 dB at 500 Hz between cells 0.40 m apart, and still -3 dB
at 2 kHz at 8 cm.

**The nearest cell, crossfaded**, ADR 0013's rule, for the low band: at
0.40 m the nearest neighbour errs by +1.6 dB at 500 Hz.

**A crossover at 500 Hz**, which would have let source positions stand
0.24 m apart (-7.7 dB at 500 Hz) and cut the wave solve's cost. The mirror
is not validated between 500 Hz and 1 kHz: its judgement reads from 1 kHz
(ADR 0014), and the render the owner validated by ear joins at 1 kHz.

**Fusing four neighbours**: no better than two under 1 kHz, and it amplifies
above.

**Storing the rendered signal.** Order 7 at 48 kHz in float32 is 14.7 GB
for twenty minutes. The recipe is a few kilobytes and gives it back.

**Storing the translation operators in the pack.** 8.5 MB a step.

**Baking the directivity into the pack's gains.** The switch would then
cost a second trace.

**Bit identity between host and card.** Not attainable, and promising it
would make the tolerance a secret.
