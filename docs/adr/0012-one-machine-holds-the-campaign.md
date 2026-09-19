# 0012: one machine holds the whole campaign, and the card does the work bought by the core

Status: accepted, 2026-09-14.

## Context

ADR 0015 made a field one resumable campaign and measured what it cost: on
hssd_0076 the card solved for 1.3 h and billed 3 h, four CPU boxes encoded
for 2 to 3 h, a fifth box had voxelised for 2 h the day before, and 128 GB
of pressure crossed the network between them at 22 MB/s on the card's
clock. The laptop planned, audited and assembled in between. Every stage
ran where its kind of machine was cheapest, and the campaign paid for the
seams: five to ten times its compute, two nights running.

Two stages were bought by the core because they were written for the core.
PFFDTD's voxeliser answers, for every grid node, which of its six legs a
triangle crosses and which triangle is nearest, one voxel and one triangle
at a time in twelve processes that spill to disk: 27 minutes of ray tests
and 9 of merging on 40 cores for the storey at 8 kHz. The ambisonic encoder
fits, per frequency bin, a regularised least squares whose matrix depends
on the array's geometry and not on the pressure, and rebuilt that matrix
from spherical Bessel functions for every one of a thousand points: 16 to
125 seconds a point on the boxes.

## Decision

1. **One machine.** The campaign runs on the host that holds the card(s),
   from a bundle the laptop prepares (the storey's mesh and materials, the
   listening grid and its rooms, the sources, the parameters and the cache
   keys) to a run directory the laptop fetches (the fields, the audit view,
   ``walk.json``, the plan, the logs, and the grids as a cache). Nothing
   else crosses the network. ``reverberate.accel`` is the library that runs
   there; ``reverberate.gpu.onebox`` is the only module that knows about
   Vast, and it watches the machine every five minutes rather than waiting
   for a marker.
2. **The voxeliser runs on the card and reproduces PFFDTD's byte for byte.**
   One thread block per (voxel, triangle) pair, order-independent atomics,
   the same float64 arithmetic in the same order with fused multiply-adds
   disabled, upstream's near-hit band and its per-voxel shortcut kept, and
   the pocket census of the chain applied unchanged afterwards. Measured:
   the 1 kHz storey grid in 14 s and the 4 kHz grid in 99 s on an RTX 3090,
   against 1693 s and 2380 s on the rented CPU box, every dataset equal.
3. **The encoder prepares a band once and runs on the card.** The matrix,
   its Gram and the gate weights are computed once per band and array
   geometry by the CPU module's own functions, kept on the card, and every
   point costs two transforms, one batched product and one batched solve.
   The four filters before the fit are kernels that do the CPU's arithmetic
   in the CPU's order. The pressure is read from the engine's output on the
   same disk, rounded to float32 as the shrink did, and deleted once it is
   signals.
4. **The engine is driven locally, in receiver slices when the host's RAM
   asks for it, and each slice is encoded before the next is solved**, so
   the disk holds one slice of pressure at a time.
5. **Every common NVIDIA card is a target.** The engine is built for the
   card it finds, the kernels are plain CUDA C compiled at run time, the
   working sets are cut into slabs sized to the card, and the rental chooses
   a host by the grid's VRAM in total over its cards. What a card cannot do
   is hold a grid; nothing here is tied to an A100.
6. **The engine writes its own precision.** Upstream keeps and writes the
   receiver records as float64 whatever the build; patch 8 keeps and writes
   ``Real``, so the single precision engine the campaign runs holds and
   writes float32, the numbers it computed. Half the host RAM, half the
   disk and half the writing for the same file rounded as the encoder
   rounded it; the campaign sizes its slices from the built engine.
7. **What is reproduced, and to what.** The grids are equal to the last
   bit. The encoded signals are held to a stated tolerance, measured per
   campaign against the CPU path on the same pressure and against the
   reference field, because a transform and a solve on a card are not the
   CPU's arithmetic and cannot be made so.

## Consequences

- A fresh export of the same scene is not the same mesh: the export of
  2026-09-14 has 12 460 fewer triangles than the one of 2026-09-12, so a
  campaign that must reproduce an earlier field starts from that earlier
  export (``prepare_bundle(models_from=...)``), and the cache key says so.
- The engine's float64 output is not float32-exact; the shrink of ADR 0015
  rounded it, and the card path rounds it the same way so the encoded
  signals match. Keeping the full precision is a one-line change that
  would change every field by about one part in ten million, and is not
  taken here.
- The synthesised tails of a field are a function of the low band's length
  in samples, which is the engine's step count plus one and not the plan's;
  a reproduction that is exact to 1e-7 for 393 ms and then different was
  three samples short. Recorded in the runbook, pinned by a test.
- The pocket census stays on the host's CPU, unchanged; a grid too large to
  label there is left unsealed as before, and the manifest says so.
- The assembly's tail synthesis ran the whole octave bank on every noise
  draw and kept one band; it now filters each draw through its band alone,
  in one transform per channel, and the crossovers in one FFT convolution.
  The field is the same to 4e-12 of its peak, not to the bit; the batched
  transform rounds differently in the last place. Recorded so nobody
  expects bit equality from a re-assembly.
- The measured cost and time of a campaign with this decision are recorded
  in ``docs/runbook-rented-machines.md``, part 5.
