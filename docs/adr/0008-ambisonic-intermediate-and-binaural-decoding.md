# 8. Encode the field from a volumetric array of grid nodes, decode it to ears with a head

Date: 2026-09-07

Status: accepted

## Context

Roadmap section 7 fixes ambisonics as the spatial intermediate format, decoded
offline, because it is the only representation that survives a head rotation or
a change of head morphology without resimulating. It also says plainly that
"two bare microphones do not make binaural". Nothing of either existed: every
receiver was a bare omnidirectional point, and every audio file this project
had written carried a warning saying so.

Section 7.1 also sized the array, and its sizing is the thing this decision
had to revisit. It applies the microphone rule `N <~ k r <~ N+1` and concludes
that a fixed array supports order 1 over one octave, or order 3 from five
concentric spheres of sixteen microphones, "the honest ceiling".

## Decision

**Encode from a volumetric array of grid nodes, and let the measurement decide
the order.** Six shells from 1.2 to 16 cm, 140 near uniform directions each,
deduplicated onto the solver's own nodes: 841 receivers at the 16 kHz working
point. Order 7 out, fitted at order 10, Tikhonov regularised with the noise
gain capped at a stated 60 dB, and each frequency fitted only from the shells
whose `k r` the truncated fit can still describe.

**Receivers are grid nodes, not interpolated points.** One output row each
rather than eight.

**Air absorption is applied as post processing**, ISO 9613-1 with the
temperature and the humidity declared.

**Decode with a rigid sphere head**, magnitude least squares above 2 kHz and a
diffuse field covariance constraint, both after Schorkhuber, Zaunschirm and
Holdrich 2018.

## Why the order is measured rather than ruled

The `N <~ k r` rule is a statement about a **microphone**: below `k r = N` the
radial term `j_n(k r)` falls off so steeply that the order disappears into the
microphone's own noise. There is no microphone here. The solver's floor is
numerical, about -100 dB, so an order sitting 60 dB down is still there.

Measured, by encoding an analytic monopole at 1.5 m through this array with
noise at that floor and comparing per order against the closed form:

| order | 125 Hz | 500 Hz | 1 kHz | 4 kHz | 16 kHz |
| --- | --- | --- | --- | --- | --- |
| 0 | -126 | -143 | -131 | -116 | -89 |
| 3 | -43 | -89 | -105 | -89 | -73 |
| 5 | -0 | -46 | -82 | -58 | -60 |
| 7 | -0 | -0 | -48 | -43 | -48 |

Error in dB relative to the order's own energy. **Order 7 holds from 1 kHz to
16 kHz with 43 dB of margin**, against section 7.1's order 3 ceiling. Below
500 Hz the high orders genuinely are gone, and that is physics rather than a
defect: at 125 Hz and 16 cm, `k r` is 0.37.

**What binds instead is aliasing, and it is a single number.** A shell whose
`k r` exceeds the fitted order is one the truncated expansion cannot describe,
and its residue folds into the orders that are kept. Measured at 16 kHz, order
7 comes back at **-20 dB** when shells are admitted up to `k r = fit order`,
and at **-48 dB** with a margin of four orders. So the gate stops four orders
below the fit, tapering over two, and the outer shells serve the bottom of the
band while the inner cloud serves the top.

## Why the receivers are nodes

PFFDTD interpolates a receiver trilinearly from the eight nodes around it,
which is exact for a field linear across a cell. At the top of the band it is
not one: at 10.5 points per wavelength a 16 kHz wave read halfway between two
nodes loses `cos(k h / 2)` per axis, 0.956, up to **-1.2 dB in three**. That
error is a smooth function of position, so it does not average out over an
array. It becomes a radius dependent gain, which is exactly the quantity the
encoder is fitting, and it would be read as a field that is not there.

Snapping to nodes removes it exactly. The shells are then not exactly
spherical, so the fit is given each node's true radius rather than the nominal
one. The other gain is arithmetic: one output row per receiver instead of
eight, so 841 receivers cost what 106 would have cost.

## Why air absorption is post processing, and exact

In an impulse response every sample arriving at time `t` has travelled exactly
`c t` of path, whatever route it took. Air absorption is therefore precisely a
per sample frequency dependent gain `exp(-m(f) c t)`: a time varying filter,
not a diffuse field approximation. It applies retroactively to every response
already computed, costs no rental, and needs no patch to the pinned solver.

Verified against the roadmap's own table, which is ISO 9613-1 at 20 C and
50 per cent relative humidity, and reproduced digit for digit: 0.0047, 0.0099,
0.0297, 0.1053 and 0.3645 dB/m at 1, 2, 4, 8 and 16 kHz. On a synthetic 0.29 s
decay it takes **-37.4 per cent of T60 at 16 kHz** against the roadmap's own
predicted -37.7, -14.1 at 8 kHz against -14.3, and nothing at 500 Hz.

**Humidity is a declared parameter of a response, not a detail.** At 16 kHz the
coefficient runs from 0.25 dB/m at 80 per cent to 0.47 at 30, a factor of 1.9.

## Why a head at all, and which one

An order 7 decode against an order 7 head is not the same as a full decode. A
rigid sphere at 16 kHz has content out to order 45; truncated at order 7 its
own reconstruction is 33 dB down. Measured on a plane wave from the left:

| band | plain least squares | magnitude least squares |
| --- | --- | --- |
| 4 to 8 kHz, shadowed ear | +5.6 dB | +2.8 dB |
| 8 to 16 kHz, shadowed ear | -0.7 dB | +0.4 dB |
| 8 to 16 kHz interaural level difference | 19.0 dB | 16.5 dB against a true 16.9 |

And on a diffuse field, the deficit in level a plain truncated decode returns:

| order | plain | magnitude least squares | and the covariance constraint |
| --- | --- | --- | --- |
| 1 | -15.9 dB | -0.09 dB | -0.07 dB |
| 7 | -4.7 dB | -0.04 dB | -0.04 dB |

**The covariance constraint buys nothing measurable on top of magnitude least
squares here**, and that is reported rather than dressed up: it is kept because
it is cheap, because it is the published method, and because it constrains the
cross term rather than only the level, but it is not what fixes the deficit.

The sphere is the tested default because it is analytic: its coefficients have
a closed form, `H_nm = i^(n-1) Y_nm(ear) / (mu^2 h_n'(mu))`, so the numerical
projection has something exact to be checked against, and the decoded delay can
be checked against Woodworth. A measured KU100 is supported through the same
path for listening, and roadmap section 7.2's simulated heads are the eventual
answer and a different item.

## What was rejected

**Trilinear receivers.** -1.2 dB at 16 kHz at mid cell, and eight times the
output rows for the privilege.

**One array per band.** The shells are gated per frequency instead, which is
the same idea without a seam at the crossover and without a second run.

**Order 3, section 7.1's ceiling.** Measured to be four orders pessimistic for
a wave solver, for the reason given above. Section 7.1 should be corrected.

**A hard gate.** A cut moving from bin to bin is a comb filter. Raised cosine.

**Fitting at the order that is kept.** Fitting two orders above gives the
shells' own residue somewhere to go other than into the reported orders. Ten
against seven costs 1.46 times the solve and buys 8 dB at 8 kHz.

**Solving with a complex system matrix.** The geometry carries no phase, so
`G` is real and the two halves of each bin are solved together as two real
right hand sides. Half the work, and a fit that cannot invent a phase.

## Consequences

An ambisonic response is a new artefact with a new shape, and
`docs/formats/ambisonic-response.md` declares it. It is one rigid array, so
unlike the loose point receivers of ADR 0005 it is written to SOFA as **one**
measurement whose receiver type is `spherical harmonics`.

The dataset's spatial resolution is now set by the head model rather than by
the room recording, which reverses the roadmap section 7.2 statement that the
ambisonic order caps the chain at about 40 degrees.

841 receivers over 0.5 s at the 16 kHz grid rate is 0.98 GB of `sim_outs.h5`,
fifteen times W29's. It costs nothing on the card, since the engine copies
`Nr` floats per step, and it costs a fetch.
