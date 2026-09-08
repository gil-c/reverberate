# 0009: atmospheric absorption is applied after the solve, not inside it

Status: accepted.

Numbered 0009 rather than 0008 because the W10 branch, which merges first by
agreement of both sessions, holds 0008 for its ambisonic decoding decision. Two
files carrying one number is not a merge conflict and nothing would have said
so: git merges them as two distinct paths and the collision only appears the
first time somebody writes "ADR 0008" and means one of them.

## Context

The solver has no viscosity term. PFFDTD is pinned at `aa319f6c` and its CUDA
kernel carries no Stokes filter, so every response this project holds decays
only by its boundaries. `audio.py` has said so since W8 and every `report.json`
carries it as a machine-readable omission.

W30 measured what that omission costs, and it is not a refinement. At 20 C and
50 per cent relative humidity, ISO 9613-1 gives 0.3645 dB/m at 16 kHz, which is
a T60 of 0.48 s from air alone. Against the measured bedroom's 0.28 s of
surface decay the two together read 0.19 s, a fall of 35 per cent. W29's whole
mid-to-high result moved by 5.3 per cent, so the term that is missing is five
times the effect the experiment was built to see.

The obvious place to put it is the kernel, since that is where the physics is.
That is the option this decision rejects.

## Decision

**Atmospheric absorption is applied to responses the solver has already
produced, in `reverberate.air`, and never in the engine.**

The justification is that in an impulse response it is not an approximation.
Every sample arriving at time `t` has travelled exactly `c t` of path, whatever
geometry it took to get there, because that is what the speed of sound means.
Atmospheric absorption is therefore *exactly* a per-sample, frequency-dependent
gain `exp(-m(f) c t)`: a time-varying filter on a signal that already exists,
not something a diffuse field assumption has to excuse.

Four consequences follow, and they are the reason rather than a bonus.

- **It costs no rental.** It is arithmetic on a few megabytes.
- **It applies retroactively.** `w20_first_listen`, `w27_sealed` and `w29_16k`
  can each be corrected from their stored responses, and the responses
  themselves were never wrong.
- **It needs no patch to the pinned solver.** A Stokes term in the inner loop
  would be a fifth vendored patch on the hot path of a float32 kernel, and
  section 12.4 already carries four.
- **Humidity becomes a declared parameter of a run rather than a compiled-in
  constant.** At 16 kHz the coefficient runs from 0.252 dB/m at 80 per cent
  relative humidity to 0.466 at 30, a factor of 1.85. One post-processing pass
  can be repeated at another humidity; a kernel constant cannot.

`Atmosphere` is a frozen dataclass with a `record()` method precisely so that a
run states its air rather than inheriting a default nobody wrote down. The
roadmap's own "43 dB per 500 ms" is the 80 per cent figure, the mildest of the
range, so section 10 carries a table rather than one number.

## The implementation, and why the short-time transform is not a second model

The gain varies in both time and frequency, so it is applied on a short-time
spectrum: each analysis frame is multiplied by `exp(-m(f) c t)` at that frame's
centre time and its own bin frequencies, then overlap-added back. The frame
length trades time smearing against frequency resolution.

That could be a model parameter, and it is not, because two things pin it. A
pure tone at `f` is checked to decay at exactly `m(f) c` decibels per second,
which it does to within 0.14 per cent at 2, 8 and 16 kHz. And halving the frame
changes a measured band energy by under 1 per cent, far inside the 3.1 per cent
seed-to-seed noise floor W3 measured. So the transform is an implementation
detail with an analytic acceptance test, not a knob a run has to declare.

## What was rejected

**A Stokes term in the kernel.** Correct in principle and the only option if
the field inside the domain were wanted rather than the response at a receiver.
It costs a fifth patch to a pinned solver, a rebuild on every rented machine, a
recomputation of every existing run, and it bakes the humidity into the binary.
The reciprocal case would be a solver whose output is not an impulse response,
and this one's is.

**A single broadband gain, or one per octave band with no time dependence.**
Wrong in shape: the whole point is that the attenuation accumulates with path
length, so it is what separates an early reflection from a late one. A static
band gain colours the response without shortening the tail, which is the
opposite of the effect.

**Applying it during the octave-band analysis in `metrics.py`.** That would
correct the measurement and leave the response wrong, so the audio and the
metrics would disagree, and any consumer reading the stored response would get
the uncorrected one. The correction belongs to the signal.

## Consequences

`w20_first_listen`, `w27_sealed` and `w29_16k` must be re-reported from their
stored responses, and the `no air absorption` entry comes off their omissions
once they are. Any run published before that carries a high-frequency tail
roughly twice as long as it should be, and the binaural and ambisonic rendering
work inherits that until this lands.

The mid-to-high extrapolation of section 5.5 is reframed by this. Once air is
applied, the difference in decay between the mid and the high band is dominated
by a formula with two declared parameters, and what is left to test is the
*material* term, whose uncertainty is the catalogue's own extrapolated 8 and
16 kHz columns.
