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
