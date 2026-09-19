# 0014: a geometric mirror, calibrated on the wave solver and joined to it above 1 kHz

Status: proposed. Reopens the question ADR 0004 closed for third party engines:
this one is the project's own, so no licence stands in the way.

## Context

A wave solve costs as the fourth power of the highest frequency it resolves.
On hssd_0076 the band to 1 kHz takes 6.4 s of card time, the band to 4 kHz
496 s and the band to 8 kHz 2 929 s. Above a few hundred hertz a room is
mostly arrivals and their decay, which a geometric model can compute for
almost nothing; below, it is resonances, which only the wave solver resolves.

## Decision

**The wave solver answers under a crossover at 1 kHz, a geometric mirror of it
above.** `reverberate.mirror`:

- **Geometry.** The solver's own model, turned into planar reflecting facets
  (0.4 m² and up), decimated occluders and a room shell.
- **Early part.** Image sources on the facets to order 3, with the parallel
  pairs followed to order 6; each validated path is a band limited pulse from
  its direction, laid through the project's octave bank.
- **Tail.** 100 000 rays leave the image sources' specular paths alone (to
  their order, 3, within their 80 ms window) and fill a directional energy
  histogram in 2 ms bins; the tail is noise bursts per bin, drawn from the
  directions the bin's moments give.
- **Points in shadow.** The geodesic round the occluders on a 10 cm grid with
  Maekawa's loss per bend, its corners pulled onto real edges, the other edges
  the point sees the source round, and the last corner's reflections in the
  point's own room.
- **Calibration.** A band by band fixed point on 24 points of the wave field,
  judged above the crossover only: absorption from the decay time, a tail gain
  from the colour, the image sources' own absorption from the matched
  reflections' levels. The file it writes also pins the ray and tail settings
  it was fitted under, and a render uses them.
- **Join.** Per point, in the frequency domain: masks adding to one in
  pressure over the onset, where both carry the same arrival, and in power
  after it; the mirror levelled onto the wave field over the crossover band.
- **Devices.** Every stage runs on the host's cores, one card or several
  (`reverberate.compute.Devices`). The paths and the histogram are the same on
  all of them: shares come back in order and the histogram is a sum of
  integers binned with one divisor. The responses are the same on any number
  of cards, and on any number of cores; between the host and a card they are
  equal in law only, the tail's noise being drawn by a different generator and
  the low cut applied as a spectrum on the card.

## Rejected

The measurements are in `0014-appendix-measurements.md`; the costs and the
failures of the rented machines in `docs/runbook-rented-machines.md`, part 6.


Each measured on hssd_0076 and removed: image order 4 (twelve times the path
time, precision lower); 300 000 rays (no criterion moved over 100 000); a
Nelder-Mead search (135 s an evaluation, forty of them, against 31 s and
eight for the fixed point); smoothing the histogram over time (it removed the
vertical stripes of the wave field's tail along with the ray noise); a
statistical tail from Eyring and Barron (replaced by the rays); scaling a
shadowed point's tail on its own onset; separate floor and ceiling
scattering; a flat gain on diffracted paths; judging a per point target set
derived from the mirror's own noise (a threshold states the quality wanted,
not the one a reading can resolve).

## Consequences

- By ear on hssd_0076 the joined field is the closest of the geometric renders
  to the wave field. Its criteria, measured on the code before review, are in
  the appendix; the fields of this code are measured with the campaign that
  runs it (`python -m reverberate.mirror judge`).
- A campaign needs the wave solve under 1 kHz only.
- **Open:** the calibration is fitted on one dwelling. Whether its parameters
  transfer to another is not measured; until it is, the mirror is only as good
  as the wave field it was calibrated on. The source signature and the clock
  are also read from a wave field.
