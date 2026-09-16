# 0013: the listener is rendered in the browser

Status: accepted. Supersedes the split of ADR 0006, which kept every piece of
signal processing in Python and had the page draw and play files.

## Context

The owner wants to stand in a simulated flat, choose sources, walk and turn,
and hear the binaural render follow without lag, so as to judge the reference
wave solver by ear and by eye before other solvers are calibrated against it.
ADR 0006 was written for a different question, "what did the solver return for
this run", and its answer -- analyse in Python at build time, draw in the page
-- cannot follow a moving listener: head yaw was baked at four angles and every
response was one file.

The input is an impulse response **field**: one ambisonic response per
listening cell of a lattice over the flat's air, per source, produced by a
parallel session with the reference solver (`docs/formats/response-field.md`).
From it the response at any cell and any head yaw is a rotation and a decode,
both linear and both cheap.

## Decision

**Whatever depends on where the listener stands or which way they face is
computed in the page.** Python still builds every input, once: the decoder
filters (`reverberate.viz.decoders`, from the library's own
`spatial.binaural.design_decoder`), the field unpacked into raw chunks
(`reverberate.viz.field_payload`), the voices (`reverberate.viz.voices`).
The page does the rotation (`audio/sh.js`), the decode to two ears
(`audio/brir.worker.js`), the plots of the response (`audio/plots.worker.js`),
and plays through the browser's own convolvers (`audio/engine.js`).

**The decode is done in the frequency domain, per cell and per head
orientation.** A cell's channels are transformed once when it is first stood in
and kept; a head update rotates the kept spectra by the exact block-diagonal
rotation of the real harmonics (yaw and pitch together, `sh.rotationBlocks`,
built by projection on a quadrature exact for twice the order, 1 ms), multiplies
by the decoder's spectra and sums into two ears. At order 7 that is 680 weights
over one 8192-bin spectrum and 128 complex multiply-accumulates: measured at
15 to 40 ms on the laptop, in a worker, so the picture never waits for the
sound. Pitch is in the audio from the start rather than "later", because the
general rotation turned out no harder to test than the yaw-only one: a delta
must land where the rotated direction says, to rounding.

**The response is decoded in two stages, and is exact at rest.** A whole
second at order 7 costs ten times the early part to rotate, which is the lag
the owner excluded. So the first 150 ms are re-decoded at every head update
in one worker, and the rest in a worker per source: once when a cell is
entered, at the head of that moment, and again, exactly, once the head has
been still for 200 ms; a stillness decode is cancelled if the head moves
first. Both times are settings. At rest the sum of the two parts is the exact
decode of the whole response to rounding. While turning, the part after
150 ms lags the head: above a kilohertz that part is diffuse and has no
direction to lag by; below, the late field has some structure, and during a
turn it stays with the head for a fraction of a second. That is the one
compromise, and it ends when the head stops.

**The response is swapped by crossfade, never interpolated.** Waveforms of two
neighbouring cells cannot be averaged above about a kilohertz, so the listener
hears the nearest cell, and a new cell or a new yaw arrives on an idle
`ConvolverNode` and takes over by a 50 ms equal-power crossfade.

**The plots are of the response at the cell, not of the voice and not of
the decode.** The omnidirectional channel of the cell the listener stands in
gives the spectrogram and the Schroeder decay, in absolute decibels against
the cell nearest the source, so walking away plots lower; the covariance of
all channels gives where the early and the whole energy arrive from, drawn in
the head's frame. Rebuilt when the cell changes, in a worker of their own;
changing the voice or turning the head changes nothing in the levels.

## What was rejected

**An `AudioWorklet` with its own partitioned convolution.** The browser's
`ConvolverNode` already runs a partitioned FFT convolution on a thread of its
own; a hand-written one would have been slower and unverifiable here. The
worklet is worth revisiting only if a browser turns out to click on buffer
swaps despite the crossfade.

**Interpolating responses between cells.** Comb filtering above the first
wavelength that fits between two cells; the producer's plan says the same.

**Order 3 in the page.** Four times cheaper, and the owner chose order 7,
which the measured update time supports.

## Consequences

The tests cannot run the page, but node runs its modules: `tests/test_sh_js.py`
checks the harmonics, the rotation and the head convention against the
library, `tests/test_analysis_js.py` the direction diagram and the absolute
levels, `tests/test_grid_js.py` the admission of grid tiles. The page exposes
`window.reverberate` so the rest can be driven in a browser, and it was.

`run_view.py`, the build-time analysis of ADR 0006, is gone with the page
that read it.
