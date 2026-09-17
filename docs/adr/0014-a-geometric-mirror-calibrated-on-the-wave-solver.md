# 0014: a geometric mirror, calibrated on the wave solver, beside the field

Status: proposed, 2026-09-16, for the owner to accept or reject. Reopens the
question ADR 0004 closed (the geometric engine is removed) on the terms that
record set: not a second solver of the dataset, a model calibrated against
the wave solver's own fields.

## Context

The wave solver's field of one source over one storey costs about 1 USD and
a quarter of an hour on one card (ADR 0012), and a dataset of hundreds of
flats and sources is still days of rented cards. The owner asked for the
cheapest way to produce more responses from the same geometry, given a
calibration set of a few wave fields, and decided against a neural
surrogate: the linear and classical methods of room acoustics have
parameters that mean something, can be audited, and can be calibrated on
a hundred dollars of solver data.

ADR 0004 removed the geometric engine because no library could be built
with the materials the project needed under a usable licence, and because
a plausibility-tuned engine cannot be separated from its budget afterwards.
Neither objection holds for an engine of our own: the image-source and
ray-tracing formalisms are textbook, the geometry and materials are the
export the solver reads, and every choice is a parameter written to the
run.

What a geometric model cannot do is also known: diffraction round a
doorway, which is how the solver reaches a bedroom from the living room
(185 of the 437 points of hssd_0076 have no line of sight to the source
and no image path at order 3), and the wave-like behaviour of small
reflectors at low frequencies. The criteria have to measure this rather
than hide it.

## Decision

1. **The mirror is our own engine, on the card, twinned in numpy.** The
   image-source tree with beam pruning and a flutter extension, the path
   validation, and the stochastic rays with integer histograms run as CUDA
   kernels compiled from Python (`reverberate.mirror.kernels`, `engine`);
   the same algorithms in numpy (`ism`, `rays`) are the twins the kernels
   are held to, bit-exact on the paths and on the histograms' counts.
   Cards are split over receivers and rays and merged in a fixed order, so
   one card or four give the same field.
2. **The geometry is derived from the solver's export by stated rules**
   (`mirror.geometry`): planar facets above an area threshold reflect,
   closed meshes are decimated to a distance bound to occlude, the rest is
   counted as diffuse; the census of what was kept and left out is written
   beside the scene and shown in the app.
3. **Early reflections and the tail are judged separately** by part A
   (discrete reflections: recall and precision against the reference's
   detected reflections, their time, direction and level errors, the
   spatial echogram distance) and part B (decay, tail colour, diffuseness,
   anisotropy, coherence, seam), in `mirror.criteria`; the solver's own
   floor across two of its grids is printed beside every report so no
   candidate is asked for more than the reference reproduces of itself.
4. **The mirror is calibrated on the reference, never tuned by hand.** An
   absorption scale per octave band, a scattering scale and a tail gain
   per band are searched with the criteria as the cost, on a few dozen
   points of one field; the parameters are one JSON keyed by their digest
   and every field carries the key it was rendered with. Material tables
   are never edited; a campaign's variants (rigid, empty, permuted) are
   overrides recorded per campaign.
5. **The mirror runs as a stage of the campaign on the machine that solved
   the field**, after the assembly, in two phases: the card phase needs the
   derived geometry and the lattice's positions and writes the paths and
   the histogram; the host phase needs the reference field and writes the
   mirror field in the reference's format (`field_mirror/<S>.h5`), the
   metrics per point and the report. Nothing large travels.
6. **The app compares them A against B**: the same cell, the same level,
   the wave field or its mirror, with a dashboard of the criteria and the
   mirror's audit layers (reflectors, occluders, paths at the cell).

## Consequences

- One source of hssd_0076 (437 points, 116 662 images at order 3 with the
  flutter to 6, 1e6 rays) takes 13 min on one RTX 3090 for the card phase;
  the host phase renders and judges on the cores. The measured timings and
  the transfer rule are in `docs/runbook-rented-machines.md`, part 6.
- The first raw gap before calibration, 16 points of hssd_0076: recall
  median 0.49 with the matched reflections at 0.04 ms, 0.55 degrees and
  0.82 dB; the solver's own floor is 0.88. The calibrated figures are in
  the run's `mirror/metrics/S1.json` and the report to the owner.
- Two findings for the owner that are not the mirror's: the export
  averages floor, walls and ceiling into one `shell` material, and rooms
  without a line of sight are reached by the solver through diffraction
  the mirror does not model. Both are reported per point, not hidden.
- ADR 0004's objections are answered by ownership, not by a library: the
  licence and the budget-tuning objections do not apply to code of our
  own. Its conclusion that a geometric engine is not a second solver of
  the dataset stands: the mirror is a calibrated model beside the field,
  never a replacement for it.
- The roadmap's list of non goals keeps "no second solver"; the mirror is
  reported under it as the calibrated model this record describes, for the
  owner to place.

## Update, 2026-09-17: what the model needed to meet the reference

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

## Update, 2026-09-17, morning: above 1 kHz, and what a campaign costs

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

## Update, 2026-09-18, night: the two solvers joined, and what a threshold means

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
- **A threshold is not a statistic.** The perceptual thresholds (5 per cent
  of a decay time, a decibel, 20 microseconds, 0.075 of coherence) are right
  and were being asked of one point's measurement, which is noisier than
  they are. They now judge the storey's median, whose noise is the floor
  over the square root of the point count. Per point each criterion sits at
  the next round number over the measured floor: reverberation time 0.10,
  early decay time 0.15, seam 4 dB, the others unchanged.
- **The smoothing lowered the floor as well as the error**: two seeds now
  read a reverberation time 8.0 per cent apart against 11.1, an early decay
  time 10.2 against 14.9, a colour 0.81 dB against 1.05 and a seam 3.12 dB
  against 3.64.
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
- **The plateau is still at 1e5 rays**, smoothing or not. One evaluation of
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
- **Scaling a shadowed point's tail on its own onset was tried and left
  off** (`tail_scale_on_onset`). The idea was that the rays and the
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
- **Snapping the geodesic's corners onto the edges was written, measured and
  left off** (`snap_corners`). It shortens the way round by 0.11 m in the
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
  the "shorter than every edge" test and changes the whole set. That is a
  session of its own; `snap_corners` stays off until it has had one.
