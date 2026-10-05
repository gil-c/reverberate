# 0014, appendix: the measurements behind the decision

The record the mirror's development kept, 2026-09-16 to 18, moved here from
#45 so it is not lost. Each figure is of the code and the fields of its date;
some settings it describes were later reverted or removed, and it says so
where that was known then. The decision itself is ADR 0014.

The last figures of that record: on the fields of 2026-09-18, every second
point (219 of 437), the criteria reading from 1 kHz on the worst band, the
joined field met 0.366 of the sixteen criteria and the mirror alone 0.357.
The review of #47 then corrected the image tree (a facet reflecting on both
sides sends images to its back) and the edges (a rim line stops at a
doorway), which changes those fields; they were rendered again on
2026-09-19 with calibration `c3cec6aab28bb582` and are not yet measured.

## 2026-09-17: what the model needed to meet the reference

Measured on hssd_0076 against the same wave field; the numbers are in the
run's `mirror/metrics_c/S1.json` and the report to the owner.

- The whole band, the reference chain's air absorption and the reference's
  own source signature (a minimum phase filter from its median direct
  spectrum) are part of the mirror, not options of a listening test.
- The tail's level is set against the direct pulse's whole band energy
  and the energy a receiver sphere catches of the direct rays,
  `r^2 / (4 d^2)`; its band energies pass through the inverse of what the
  analysis bank reads of shaped noise. Three reading errors (reflections in
  the direct bins, a 1 ms window on ringing filters, a bank applied twice)
  had been paid for by 4 to 8 dB of calibrated gain.
- Rays whose bounces are all specular on reflectors, within the image
  tree's order and window, are left to the images.
- The shell has its own scattering coefficient, a calibrated parameter: a
  specular bounce keeps a ray's elevation, and without mixing the late
  field lay flat and decayed slowly, which the calibration had paid for
  with 1.3 to 1.7 times the catalogue's absorption. With the shell at 0.4
  the calibration keeps the catalogue's absorption within 4 % and the tail
  gain within 1.4 dB. The images keep the class's scattering and have their
  own absorption scale, set by the matched reflections.
- A point without a direct path gets a diffracted onset: the geodesic
  round the occluders with Maekawa's loss per edge.
- The calibration is a band by band fixed point on a cost that reads every
  band (T30 to absorption, colour to tail gain, reflection level to the
  images' absorption), a few evaluations where the fifteen-coordinate
  search took forty.
- Order 4 of the image tree was measured and not kept: three points of
  recall for twelve times the paths.

## 2026-09-17, morning: above 1 kHz, and what a campaign costs

- The mirror answers above 1 kHz; a wave solve will answer below. The
  judgement reads decay, colour, seam and echogram on the octave bands from
  1 kHz, and the reflections and the late field's directions on the
  response high passed there. Per band, the night's C erred 3.9 dB in colour
  at 250 Hz, 2.4 at 500 and 2.0 at 1 kHz.
- The interaural coherence is an energy weighted mean of its 20 ms frames.
  The plain mean counted frames 50 to 130 dB down, where the reference's
  ears grow coherent and nothing is heard, as much as the first ones.
- The judgement's own floor, two renders of one model with other seeds:
  T30 11 %, EDT 15 %, colour 1.0 dB, seam 3.6 dB, sector energy 1.4 dB. The
  T30 target (5 %) and the seam target (1 dB) are under it; the mirror's
  T30 gap to the reference (9.5 %) is at it.
- A whole campaign of hssd_0076 (437 points, order 7) takes 68 s on one
  RTX 3090 with 1e5 rays and 90 s with 3e5, 0.30 and 0.40 US cents billed:
  1/350 and 1/265 of the optimised wave campaign's machine time. The
  quality's plateau starts at 1e5 rays and order 3, the cheapest setting
  that loses nothing measurable (runbook part 6).
- What stays open: points without a direct path (185 of 437) meet 23 % of
  the criteria against 44 % elsewhere; their diffracted onset arrives 1.6 ms
  after the reference's first arrival (median), the 10 cm grid's dilation,
  and a finer grid without dilation leaks through thin walls. Recall
  (0.60 where there is a direct path) and precision (0.67) are limited by
  paths the model does not have, not by a bias (near misses fall both
  sides of the 0.2 ms gate).

## 2026-09-18, night: the two solvers joined, and what a threshold means

- **The tail's grain was the rays' own noise.** The reference's broadband
  envelope fluctuates 1.9 dB about its decay between 0.15 and 0.45 s; the
  mirror's fluctuated 4.0 dB. A 2 ms histogram bin holds some sixty ray
  crossings whose energies are unequal enough to leave 3.5 dB on top of the
  field's own. A moving mean over the histogram's bins, whose half width
  grows as a tenth of the time since the tail began and stops at 50 ms,
  brings it to 2.2 to 2.9 dB. A flat window of the same width costs 0.05 of
  early decay time; the growing one costs nothing, because the first bins
  stay as the rays wrote them.
- **The two solvers are joined per point** (`mirror hybrid`). Over the octave
  round 1 kHz the wave field and the mirror correlate 0.83 to 0.98 over the
  direct sound and under 0.25 after 200 ms, so the onset is joined with
  masks that add to one in pressure and the rest with masks that add to one
  in power. Both are spectra, so no arrival moves. The step at the join is
  measured per point: the mirror sits 2.4 dB under the wave field there,
  2.1 dB where there is a direct path and 3.2 dB where there is not, and one
  scalar per point takes it out.
- **What that buys.** The wave field of 0076 plans 6.4 s of card time for
  its band to 1 kHz, 496 s for the band to 4 kHz and 2929 s for the band to
  8 kHz. A campaign that solves only what it keeps drops the two largest
  lines; the crossover's own frequency hardly matters between 500 Hz and
  1 kHz, so it is chosen on quality alone.
- **The measurement has a floor, and the thresholds do not move for it.**
  One mirror judged against another that differs only in its ray seed reads,
  on 24 points of 0076, a reverberation time 8.4 per cent apart on the worst
  band, an early decay time 8.8, a colour 1.06 dB and a seam 3.6 dB at the
  10 ms window. Several perceptual thresholds sit under that floor. The
  thresholds stay where they are: they state the quality wanted, not the
  quality this measurement can currently resolve. A criterion that fails
  because its reading is noisy is a criterion asking for a quieter reading,
  and the work is to give it one -- more rays, a longer window inside the
  metric's own definition, a better estimator -- not to raise the bar's
  height. **Recorded and reverted 2026-09-18:** a second per point target
  set derived from this floor was added and removed the same day, on the
  owner's rule that one improves the outcome, never the yardstick.
- **Smoothing the histogram over time was measured and removed.** A moving
  mean over the bins, half width a tenth of the time since the tail began
  and at most 50 ms, took the tail's grain from 4.0 to 2.1 dB between 1 and
  12 kHz and lowered the seed to seed floor (reverberation time 11.1 to
  8.0 per cent apart). It also smoothed what the reference has: the wave
  field's tail shows vertical stripes, arrivals common to every band in a
  5 ms frame, and on the storey each point's stripe depth moved further
  from the reference's (1.92 dB of gap for B, unsmoothed, against 2.43 for
  the smoothed C), while the seam went from 3.76 to 4.12 dB. Monte Carlo
  noise is independent from one direction and one band to the next; the
  room's arrivals are not. A smoothing belongs across those, never across
  time. **Removed 2026-09-18** with `tail_smooth_s` and
  `tail_smooth_fraction`.
- **The shadowed points' first arrival is one geodesic and should be
  several.** Its time is right, its direction is right half the time: the
  elevation and the azimuth of the onset are unbiased (median error 0) but
  spread +-35 and +-70 degrees, because the reference's first arrival comes
  round several edges at once and the geodesic picks one. The corner's own
  reflections are now added (order 1, trees shared between nearby corners),
  and so is every edge the point can see the source round: 651 of them on
  0076, chosen by length (at least 30 cm) and by material (under 0.6 of
  absorption at 1 kHz), each carrying Maekawa's loss for its own detour and
  sharing the geodesic's energy rather than claiming a whole barrier's. The
  share of criteria met on 32 shadowed points goes from 0.230 to 0.309.
- **The plateau is still at 1e5 rays.** One evaluation of
  the calibration's own cost on 24 points, same parameters, only the ray
  count moving: 19.09 at 3e5, 19.53 at 1e5, 20.26 at 3e4, 21.10 at 1e4. The
  moving mean takes out the estimator's noise, not its bias, so fewer rays
  still read a shorter decay (the reverberation time ratio falls to 0.93 at
  1e4). 1e5 rays and order 3 remain the cheapest setting that loses nothing.
- **A deeper flutter was measured and not kept.** The wave field's tail does
  hold a periodic structure of its own, 5 to 32 ms depending on where the
  listener stands (autocorrelation of the broadband envelope, 0.2 to 0.4),
  which is the floor to ceiling flutter the image tree already extends to
  order 6. Taking that extension to 10 costs 20 per cent more images
  (139 957 against 116 662) and reads 0.570 of the criteria against 0.568 on
  the 24 calibration points: the same, within the seeds' own spread.
- **The join was measured, third octave by third octave.** The hybrid's level
  against the wave field's, median over 19 points of 0076, from 315 Hz to
  2.5 kHz. With the crossover as it stands (1 kHz, one octave, the onset in
  pressure, the step levelled per point): 0.00 dB up to 630 Hz, then +0.27,
  +0.33, -0.86, -0.75, +0.67, -0.57. Nothing at the join stands out from
  what the mirror does on its own above it. Each choice was worth what it
  claims: without the pressure masks over the onset the join gains 1.6 dB at
  the cutoff, without the levelling it loses 1.2 dB there and 3.5 dB just
  above, and a half octave ramp digs a 1.6 dB hole at 1.25 kHz. Two octaves
  read the same as one.
- **Scaling a shadowed point's tail on its own onset was tried and
  removed** (`tail_scale_on_onset`, 2026-09-18). The idea was that the rays and the
  diffracted onset came round the same doorway, so their ratio is that
  point's own and not the storey's median. Measured on 32 shadowed points it
  is worse: the share of criteria met falls from 0.316 to 0.293, the tail's
  colour error from 8.5 to 11.2 dB and the early decay time from 0.19 to
  0.47. The histogram's first bin behind a door holds too few crossings to
  be a scale, and the onset's own level carries the edges' loss twice over.
- **Most shadowed points bend more than once.** On 0076, of the 185 points
  with no direct path, 5 per cent bend not at all, 21 per cent bend once,
  54 per cent twice, 20 per cent three times or more. A single edge path
  cannot reach three quarters of them, which is why the geodesic still
  earns its place and why the edges help them less than their geometry
  suggests. In the delivered C the edges reach **81 of the 185**; the other
  104 keep the geodesic alone (`points_with_edge_paths` in the run's
  diffraction record).
- **Snapping the geodesic's corners onto the edges, first measured.** It shortens the way round by 0.11 m in the
  median and moves the onset's direction error from 4.31 to 1.40 degrees,
  its level error from 1.61 to 0.92 dB, the interaural level error from
  2.12 to 1.49 dB: the geometry is plainly better. But 0.11 m less detour is
  0.4 dB less of Maekawa's loss, and the tail gains were calibrated against
  the grid's own longer way: the early decay time's error goes from 0.19 to
  0.37, the colour's from 8.5 to 9.2 dB, and the share of criteria met from
  0.316 to 0.299.
- **And the level is not what is wrong with it.** `DiffractionSettings`
  gained a `gain_db`, decibels on every diffracted path, so the two effects
  could be separated. Swept with the corners snapped: -3 dB brings the share
  of criteria met back to 0.3164, exactly the unsnapped figure, and leaves
  the early decay time's error at 0.367 against 0.189. A flat gain buys back
  the aggregate and not the decay, so what snapping costs is not loudness.
  The likeliest cause is that a shorter geodesic flips which paths survive
  the "shorter than every edge" test and changes the whole set. The next
  point says what it actually was. `gain_db` was **removed 2026-09-18**: a
  flat fudge with no physics, and its only use was to cancel a loss that
  turned out to be a counting error.
- **What snapping cost was a counting rule, and it is fixed.** Pulling the
  chain tight takes the kink out of a bend, and a bend was counted by the
  kink it had left (`min_detour_m`, there to reject corners the grid
  invented). 115 of the 185 shadowed points lost at least one bend that way:
  the median count fell from 2 to 1 and the loss at 2 kHz from 28.7 to
  18.2 dB, while the loss per counted bend held at 15 to 18 dB. A corner
  that stands on a selected edge is a bend whatever kink is left, and with
  that rule `snap_corners` is on: on 32 shadowed points the onset's
  direction error goes from 4.31 to 1.40 degrees, its level error from 1.61
  to 0.92 dB, the interaural level error from 2.12 to 1.49 dB, the tail's
  colour from 8.53 to 8.40 dB, and the share of criteria met and the early
  decay time do not move (0.3164 and 0.186). Measured again on 40 shadowed
  points once the tail smoothing was gone: 0.2297 of criteria met with the
  corners snapped against 0.2250 without, the onset's level error 2.96
  against 4.07 dB. Snapping is no longer a setting; the corners are always
  pulled onto the edges within `snap_m`.
- **Which detour the loss is read from** (`loss_from`) was measured too:
  keeping the grid's own slack for a snapped bend reads 0.3164 of the
  criteria, using what is left after tightening 0.3105, and one loss from
  the whole way round against the straight line 0.3008 (early decay time
  0.186, 0.200, 0.375). The grid's slack wins because it is the only one of
  the three that still knows how deep in the shadow the receiver is. On
  the 40 points of the second measurement the grid's slack and the
  tightened detour read the same (0.2297) and the whole way round 0.2281.
  **Removed 2026-09-18** as a setting: the grid's slack is the code. So are
  `max_loss_db` (24, 32, 40 and 60 dB read the same; 24 is Maekawa's own
  limit and stays as the function's cap), `edge_gain` (one value ever
  used; the other counted a doorway's sound once per edge) and
  `same_arrival_s`, `same_arrival_deg` (declared, never read).
- **The seam criterion reads less than its own noise, and that is a fact
  about the estimator.** It reads the level step across the mixing time on
  10 ms either side, per octave, and takes the worst band. Ten milliseconds
  of an octave band holds a handful of independent samples: two mirrors
  differing only in their ray seed sit 3.59 dB apart there, while the
  mirror's own distance to the reference is 3.44 dB. Signal over floor:
  0.96. Measured against the window's width, 24 points, worst band then
  median band:

  | window | floor, worst | error, worst | floor, median | error, median |
  | --- | --- | --- | --- | --- |
  | 10 ms | 3.59 | 3.44 | 1.73 | 1.79 |
  | 20 ms | 2.51 | 3.49 | 1.34 | 1.49 |
  | 40 ms | 1.93 | 2.53 | 0.90 | 1.54 |
  | 80 ms | 2.04 | 3.76 | 1.09 | 2.42 |
  | 160 ms | 2.05 | 3.67 | 1.04 | 2.42 |

  **Recorded and reverted 2026-09-18.** The window was widened to 80 ms and
  the statistic changed to the median; both were put back. The table is kept
  because it says what the mirror must do to be judged on the seam at all:
  bring the ray estimator's variance down until 10 ms of an octave band is
  a measurement. Widening the window measures a different quantity, and
  taking the median of four bands hides the band that is out of place.
- **The decay times are read on the worst band, and stay there.** Taking
  the worst of four bands puts the noise of all four into the reading.
  Measured on 24 points, floor then distance to the reference:

  | statistic | floor | error | ratio |
  | --- | --- | --- | --- |
  | reverberation time, worst band | 0.084 | 0.102 | 1.21 |
  | reverberation time, median band | 0.040 | 0.055 | 1.37 |
  | early decay time, worst band | 0.088 | 0.260 | 2.95 |
  | early decay time, median band | 0.049 | 0.132 | 2.71 |

  **Recorded and reverted 2026-09-18.** Both were moved to the median band
  and put back. The median halves the floor for the reverberation time and
  makes the early decay time's ratio *worse*, so it was not even a
  consistent gain; and it hides what it averages. On 219 points of 0076,
  of those whose median band met 5 per cent, 17 per cent had a band more
  than 10 per cent out, the worst at 27. A listener walking the storey
  hears that band. The worst band stays.
## 2026-10-05: the alignment step is withdrawn from a pack

The mirror was put on the wave field's clock and scale by a lead, a gain
and a signature, each read on the direct sound of a field of the dwelling
(`mirror.files.align_to_reference`, `mirror.direct.measure_signature`).
The audit of the chain (`docs/open-questions/chain-audit.md`) found that
the gain and the signature are not a scale:

- **the gain** is the ratio of two energies inside 0.5 ms round the direct
  sound (`mirror.direct.DIRECT_WINDOW_S`), and the two pulses are not the
  same pulse. The wave field's is spread by its grid: 5.0 to 6.1 dB of a
  unit pulse's energy lie outside the window on the ten nearest points of
  hssd_0076, 1.6 to 3.5 dB outside 4 ms. The mirror's is the signature, a
  minimum phase filter of unit energy, all of it inside (-0.3 dB). The
  gain comes out 2.4 dB low at every frequency: -5.22 dB over the field's
  unit, with a signature of +2.85 dB at 1 kHz, where the mirror before
  either renders a unit source as `1 / d` within 0.06 dB and the wave
  field's direct sound is `FIELD_UNIT_AT_1M / d` within 0.1 dB;
- **the signature** is the reference's direct spectrum, the band limit of
  its grid at 8 kHz with it: -1.8 dB at 8 kHz, -3.3 at 12 kHz, -5.0 at
  16 kHz re 1 kHz, laid on a band that grid does not hold.

Nothing of the kind needs fitting: both solvers are physical before
anything is measured. So in a scene pack **the gain is one and the
signature a unit pulse, once for all, whatever the dwelling**
(`trace.assets.Normalisation`, `PHYSICAL`); the lead stays, it is a clock
and was never fitted. The seam between the two bands, which had hidden
the 2.4 dB under a levelling of 1.9 dB a pair (and read 1.07 dB of its
own low, the band limit of a solve to 1500 Hz taken for a level), is read
where both bands are whole, and its median on the first whole scene falls
from +1.89 to +0.52 dB (18 219 pairs; +0.27 dB on the 255 with a direct
path under 1 m). The constant of the tapered join is then zero, in code
(`render.seam.SEAM_CONSTANT_DB`).

Measured on the pack at rest that holds a lattice point of the validated
wave field (S1 of hssd_0076, 1.99 m, a direct path;
`python -m reverberate.render check --reference-point`), the pack's level
over the wave field's, a third octave at a time:

| | 1k | 1.26k | 1.6k | 2k | 2.5k | 3.2k | 4k | 5k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the image paths' window, as traced under its pair's seam of +1.98 dB | -0.51 | -0.78 | -0.89 | -0.62 | -0.85 | -1.27 | -0.61 | -0.50 |
| the same, on the physical scale | -0.23 | -0.02 | +0.03 | +0.08 | +0.03 | +0.38 | +0.07 | +0.19 |
| the first 50 ms, as traced | +1.68 | -0.83 | -0.34 | -0.35 | +0.10 | -0.94 | -0.15 | -0.73 |
| the first 50 ms, on the physical scale | +1.87 | -0.20 | +0.57 | +0.32 | +0.96 | +0.72 | +0.54 | -0.04 |

and after 50 ms, per octave at 1, 2 and 4 kHz: +0.04, -2.06 and -2.24 dB
as traced, +0.38, -1.28 and -1.31 dB on the physical scale. The direct
sound is on the wave field within 0.1 dB at five third octaves of eight.
What is left is not a scale: the first 50 ms are 0.3 to 1 dB over the
field from 1.6 to 4 kHz and 1.9 dB over it at 1 kHz (the join in power of
reflections that are still coherent, and the images the calibration makes
loud), and the late part is 1.3 dB under it at 2 and 4 kHz. At the point
without a direct path (4.75 m), which its own seam of +3.98 dB had
levelled on every band, the first 50 ms read -1.35, -2.18 and -0.52 dB
against the field at 1, 2 and 4 kHz, from -0.61, -0.86 and +0.81 dB, and
the late part -0.57, +0.66 and -1.62 dB from +0.30, +1.86 and -0.53 dB:
a pack of one pair has no scene to take a median from, and stands at
`1 / d`. Those are the tail's defects and the calibration's (D4 to D6,
D8, D9), which one fit on fields traced with `coincident_facets = "once"`
is to settle.

**The static fields keep their alignment.** `mirror.pipeline` and
`write_field` are unchanged, and a pack traced with the bundle's
`normalisation = "aligned"` carries every number as before: the pack at
rest against `render_point` then `blend` is still held sample for sample
(`tests/test_trace.py`). A field of the mirror written today is therefore
still 2.4 dB under the wave field before its seam; an alignment read on a
band and not on a window would be its remedy, and is not made here.

**The calibration is not moved by this.** It reads every level against
the same response's own direct sound, band by band, so neither a gain nor
a filter common to the whole response enters its numbers.
