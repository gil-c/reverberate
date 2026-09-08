# The three-band economy and the ambisonic array do not share a grid

Date: 2026-09-08

Status: open. Written at the merge of the W10 (ambisonic) and W37 (solver
economy) branches, deliberately not solved there. It is a physics question,
not a merge conflict, and it is handed to a session of its own.

## What each branch assumes

**W37, `reverberate.bands` and `experiments.w37_three_band`.** Roadmap
section 4.2: three solves on three grids. Whole apartment at the bottom with
the full decay, whole apartment in the middle with a long window, one room at
the top with a short one. Grid steps at 10.5 points per wavelength:

| band | top | grid step |
| --- | --- | --- |
| low | 1 kHz | 32.7 mm |
| mid | 4 kHz | 8.2 mm |
| high | 16 kHz | 2.04 mm |

The bands are recombined per receiver with filters built by subtraction, a
level ratio of `20 log10(fmax_a / fmax_b)` between solves, air absorption
applied per leg, and a synthetic tail spliced on per receiver with independent
noise per channel.

**W10, `reverberate.spatial`, ADR 0008.** One solve at 16 kHz. The array is
built **in node index space**: six shells from 1.2 to 16 cm, about 140
directions each, every receiver snapped to a grid node, 841 nodes at 2.04 mm.
Snapping is not a convenience. It removes the trilinear interpolation error of
up to -1.2 dB at the top of the band, which would otherwise be read as a radius
dependent gain, which is the quantity the encoder fits. Each frequency is then
fitted only from the shells whose `k r` the truncated expansion still
describes: at 16 kHz only nodes inside about 20 mm are admitted, and at 1 kHz
the outer 16 cm shell does the work.

## Why they cannot be combined by construction

1. **The array is a function of the grid step.** `default_shells(grid_step_m)`
   derives the radii and node counts from the step. On the 8.2 mm mid grid a
   12 mm shell holds a handful of nodes and on the 32.7 mm low grid it does not
   exist. The inner cloud that serves the top of the band can only be built on
   the fine grid, and the same `ArrayDesign` cannot be reproduced on the other
   two. Three solves therefore mean three different arrays, not one array
   sampled three times.

2. **The node sets differ, so the receivers differ.** Recombination in
   `bands.recombine` is per receiver and assumes the same receiver in every
   band. With node snapping, receiver `q` of the high grid has no counterpart
   on the mid grid. The only representation shared across the three solves is
   the spherical harmonic coefficient set `a_nm(omega)`, which suggests
   **encoding each band with its own array first and recombining in the
   ambisonic domain**. That is linear and the subtraction filters apply
   unchanged per channel, but it has not been built or measured.

3. **Two grids decorrelate, and the crossover blends them per channel.** W3
   measured a -6 dB response level floor between runs differing by a sub cell
   offset. `bands.py` already records that inside a crossover the sum is
   neither run's waveform. In the ambisonic domain that blend happens in every
   channel, so a direction of arrival read across a crossover may move. How
   much, and whether the decode hides or exposes it, is unmeasured.

4. **The effective order is measured per grid, and has only been measured on
   one.** ADR 0008 replaced section 7.1's `k r` rule with a conditioning
   measurement: order 7 holds from 1 kHz to 16 kHz on the 2.04 mm array. On
   the 32.7 mm low grid a 16 cm shell is five cells across, so the fit is
   from far fewer nodes and the aliasing gate, the noise gain cap and the
   receivers-against-unknowns guard all have to be re-run. The low band may
   support only a low order, which is what section 7.1 expected and is
   physically fine since interaural cues below 1 kHz are carried by delay, but
   the report must say so per band.

5. **The dispersion correction is per grid.** `encode.numerical_wavenumber`
   fits with the scheme's own direction averaged wavenumber. At 16 kHz the gate
   made it worth 0.005 degrees because the outer shells were already dropped.
   Each band sits at 10.5 points per wavelength at its own top, so the mid and
   low bands see the same one per cent at their own top frequency, with their
   own gate deciding whether it matters.

6. **The clash rule scales with the cell.** No receiver node may sit within one
   cell of a boundary node. `check_clearance` demands the whole ball be in free
   air with a 2 cm margin at 2.04 mm. At 32.7 mm the exclusion shell around
   every surface is 33 mm, so a 16 cm ball at ear height near a headboard or
   a wall may be refused on the low grid where it passed on the high one. The
   listening positions the dataset can use are the intersection over the three
   grids, and nobody has counted it.

7. **The domain and window tricks see a ball, not a point.** The
   source-receiver ellipsoid (`experiments.w37_ellipsoid`) puts a focus on the
   receiver. With an array the receiver is a ball of radius 16 cm, so the
   exactness window shrinks by `r / c`, about 0.47 ms, and the high band
   domain must contain the whole ball inside the room's real walls. Small, but
   it belongs in the record of every economy run.

8. **The synthetic tail must be drawn in the ambisonic domain.** `tail.synthesise`
   draws independent noise per receiver, which is correct for two ears above
   1 kHz. For an ambisonic response the diffuse tail has a defined structure:
   equal energy per channel under N3D normalisation and no correlation between
   channels. Drawing it per receiver node and then encoding would fit a random
   field of 841 pressures at 16 kHz to order 7, which is not the same thing.
   Drawing it per spherical harmonic channel is, and then the binaural decode
   gives the interaural coherence for free, which is section 5.5's spatial gift.

9. **Air absorption and recombination do not commute exactly.** Air is a time
   varying filter, recombination is time invariant. Applying air once, after
   recombination, is the exact form. W37 applies it per leg, which is exact
   away from the crossovers and differs inside them by an amount nobody has
   measured; it is likely negligible and should be stated as such.

## What the next session should do first

- State the target: an ambisonic response per band, recombined in the
  spherical harmonic domain, decoded once. Or the alternative, one array on
  the fine grid only for the high band and lower orders on the other two.
- Build the three arrays for `bedroom.001` from `default_shells` at each step
  and run `check_clearance` on each. Report how many of the run's listening
  positions survive all three.
- Run the conditioning measurement of ADR 0008 on the mid and low grids with
  the analytic monopole, and report the effective order per band.
- Measure the direction of arrival across a crossover on the rehearsal box
  before renting anything: two solves of the box at 4 kHz and 16 kHz cost
  minutes on the laptop.

Nothing here spends GPU time until the last item, and it may not need to.
