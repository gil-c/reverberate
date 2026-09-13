# 0011: a field is a campaign, solved on the whole storey and cut by points

Status: accepted.

## Context

The web viewer walks a listener through a flat and must hear every source
without lag at any point. That is an order 7 ambisonic response at every
point of a grid, for every source: hundreds of points, each a whole array of
about a thousand grid nodes per band, so one solve's output is hundreds of
gigabytes and does not fit a laptop, a card's host, or one machine's night.

Two campaigns taught what the pipeline has to be. The first (hssd_0002,
2026-09-10) solved the high band in the source's room with its doorways
sealed, and 44 of 524 points lost their high band to an 8 dB disagreement
with the storey-wide mid band. It also kept the card up for 18 h to do 2 h of
work. The second (hssd_0076, 2026-09-12) solved the storey at 8 kHz on one
H200, sliced the mid band in two to fit the host's RAM, pushed the pressure to
four boxes and encoded there; every point kept its high band, and the card
still billed three hours for 1.3 h of solving because the transfer ran on its
clock.

## Decision

1. **The high band is solved on the whole storey** whenever a host's cards
   hold the grid (`9.027 B` a node; 8.6e9 nodes on one 80 GB card, more on
   an H200 or two A100s). A room-only high band is the fallback, and every
   point without a high band says so in the field.
2. **A band that does not fit the host's RAM is solved in receiver slices**,
   each a full solve of the storey over a contiguous range of the plan's
   rows, merged on the host in row order. The card's time multiplies; the
   grid does not shrink. The rule is `2.2 x output + 8 GB` of host RAM.
3. **The pressure never comes home.** It is shrunk to float32 on the card's
   host, cut into shards by point ranges, pushed to cheap fast-core boxes
   through each box's own ssh proxy, and encoded there with as many workers
   as the box's memory holds at 7 GB each. Only the 64 channel signals are
   fetched, a few megabytes a point.
4. **A campaign is one command that resumes from its state on disk**
   (export, grids and their viewer payloads, `plan.json`, comms,
   `solve.json`, shards, field). Rentals check the
   credit against the remaining plan, tried offers are never retried, hosts
   holding data are never destroyed by a stage that did not verify a copy,
   and every long remote job runs detached and is polled on a marker.
5. **A point without a low band array of its own borrows its nearest
   neighbour's, levelled onto its own mid band** on the calibration octaves
   the pair check uses, and is flagged. Three points of 437 on hssd_0076.

## Consequences

- One source over a 160 m2 storey costs about 7 USD and 6 h with the driver
  as it stands, 5 USD and 5 h once the store carries the pressure instead of
  the card's proxy. The measured rules and the failures behind them are in
  `docs/runbook-rented-machines.md`.
- The audio at a point is what `w38_ambisonic_bands` produces for a single
  point: same bands, windows, encoder and assembly. The field adds nothing
  to the physics and removes the doorway seam.
- Encoding on the card itself, restricted to the octaves each band keeps,
  would end the transfer altogether; it is an open question, not part of
  this decision.
- The field format is `docs/formats/ambisonic-field.md`.
