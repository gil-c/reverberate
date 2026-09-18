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
and plays the early part through a convolver of its own on the audio thread
(`audio/early.worklet.js`, `audio/partitioned.js`) and the late part through
the browser's `ConvolverNode`s (`audio/engine.js`).

**The decode is done in the frequency domain, per cell and per head
orientation.** A cell's channels are transformed once when it is first stood in
and kept; a head update rotates the kept spectra by the exact block-diagonal
rotation of the real harmonics (yaw and pitch together, `sh.rotationBlocks`,
built by projection on a quadrature exact for twice the order, 1 ms), multiplies
by the decoder's spectra and sums into two ears. At order 7 that is 680 weights
over one 8192-bin spectrum and 128 complex multiply-accumulates, of which only
the half below the Nyquist bin is computed, the rest being its conjugate: 3 ms
for a pure yaw, where the rotation of each degree mixes only two coefficients,
and 10 to 15 ms with any pitch in it, in a worker, so the picture never waits
for the sound. Pitch is in the audio from the start rather than "later", because the
general rotation turned out no harder to test than the yaw-only one: a delta
must land where the rotated direction says, to rounding.

**The response is decoded in two stages, and is exact at rest.** A whole
second at order 7 costs ten times the early part to rotate, which is the lag
the owner excluded. So the first 150 ms are re-decoded at every head update,
at most every 40 ms, in one worker, and the rest in a worker per source: once when a cell is
entered, at the head of that moment, and again, exactly, once the head has
been still for 200 ms; a stillness decode is cancelled if the head moves
first. Both times are settings. At rest the sum of the two parts is the exact
decode of the whole response to rounding, because the two are faded into
each other on the field, before the decode. They used to be faded on the two
ears after it, and the binaural decoder spreads each sample over 512 taps
centred on the 256th, so a fade on the decoded early part took away field
samples from well before the seam that the late part never had: near the
seam the parts missed the whole by half its size, for as long as this ADR
has said they did not (`tests/test_decode_js.py`). While turning, the part after
150 ms lags the head: above a kilohertz that part is diffuse and has no
direction to lag by; below, the late field has some structure, and during a
turn it stays with the head for a fraction of a second. That is the one
compromise, and it ends when the head stops.

**The response is swapped by crossfade, never interpolated.** Waveforms of two
neighbouring cells cannot be averaged above about a kilohertz, so the listener
hears the nearest cell, and a new cell or a new yaw arrives in an idle slot
and takes over by a crossfade whose gains sum to one.

**The early part is convolved by the page's own `AudioWorklet`, which keeps
one history of the input for every response** (`audio/early.worklet.js`,
`audio/partitioned.js`). Uniformly partitioned overlap-save in blocks of the
render quantum, so it adds no latency: the input is transformed once per
block into a ring of spectra, and every response in play is applied to that
same ring. The decode worker hands the responses over already cut into
blocks and transformed, so the audio thread only multiplies and adds -- six
per cent of its budget per source, measured. The late part stays on the
browser's `ConvolverNode`s: it is replaced every two metres at most, and it is
a second of diffuse reverberation.

A `ConvolverNode` given a buffer has no past: it convolves only what arrives
after, as if the voice had been switched on at that instant, and each tap of
the response adds a small step as the input reaches it, heard at whatever gain
the fade in has reached. Replaced twenty-five times a second, and with a real
early response carrying energy across all of its 150 ms, that was a crackle
no shape, length or scheduling of the fade removed. With one history, a
response sounds from its first sample as though it had always been there,
and a crossfade between two is the crossfade of two complete convolutions,
exact to single precision (`tests/test_partitioned_js.py`). Counted in the
browser, through a 220 Hz tone, as frames whose waveform curvature stands out
from its neighbourhood:

| | frames with a transient, walking | turning | median frame, walking |
| --- | --- | --- | --- |
| standing still | -- | -- | 4.5 dB |
| `ConvolverNode`s | 428 of 1133 | 146 of 1161 | 14.3 dB |
| **the worklet** | **2 to 31 of 1150** | **0 of 1171** | **4.5 dB** |

A tone, because through noise a click is masked, for a detector as for an
ear: walked through pink noise the same page looked clean to a detector that
missed steps 20 dB down, where through the tone it sees one 40 dB down every
time. It is on quiet or tonal material, speech between words, that one is
heard.

**`audio/ring.js` schedules every gain, and a gain that fades reaches exactly
zero.** A superseded response fades once, to zero, over one fade, and is left
alone; the incoming one takes whatever the others leave, so the gains sum to
one at every instant; a slot is given a new response only once it is silent.
The worklet reads its gains straight off the ring, the engine plays its
curves on the tail's gains, and the offline harness plays the same curves.
The engine first re-ramped every fading gain towards zero at every swap
instead. That sums to one too, but it restarts each fade, so a gain
approaches zero and never arrives: every slot looked equally busy, a rule
choosing the one free longest collapsed three slots into two, and every early
response was cut short -- the sound played on and stopped following the
listener. The harness, which took its slots in turn, said all was well; it
now asks the ring, and so does every test of it.

**The fade is 120 ms long, over six slots, against a response every 40 ms.**
Longer than the interval, so one response is always fading into the next and
the filter never stops moving; with a fade shorter than the interval it moves
in bursts between plateaus, and a head turn is heard to step on them. Longer
still is smoother, but a crossfade is an average over the poses it spans:
120 ms of it is seven and a half degrees behind a head turning at the speed
of the arrow keys, where the page already was. The fade is a smootherstep,
an S with no first or second derivative at either end; it measured smoother
than the raised cosine it replaced on every timbre number.

**The propagation delay is a delay line in front of the convolutions, and
every cell's response is cut at its own direct arrival.** The lattice is forty
centimetres wide, so the time the direct sound takes to arrive steps by up to
a millisecond and a half from one cell to the next; left in the response,
that step comb filters the whole crossfade that follows it. So
`direct_path_m / c` is taken out of each cell (`spatial.js`) and put back as
one `DelayNode` that follows the listener's real distance to the source
(`engine.js`), which is also what gives a walk towards a source its Doppler.
Each ramp of it lands at its own pose's time plus a constant: its slope is
the pitch shift, and a pose reaches the engine late by up to an update
interval, so a ramp of fixed length after the call made the pitch stray from
the Doppler by up to six cents, heard as a pitch shift that came and went.
Anchored to the pose, it strays by under one.

**The early part starts where the sound does.** A response comes out of the
solver a few hundred samples after its geometric arrival -- its source pulse
has a delay of its own -- and out of the binaural decoder five milliseconds
later again, because the decoder is designed centred. Neither is needed, and
both are cut off: the solver's latency measured once per field on the cell
nearest the source, less a margin for its spread across cells; the decoder's
quiet run-up, with the late part placed as much earlier to meet it. The sound
reaches the listener ten milliseconds sooner.

**The late part is fetched again every two metres, not at every cell,** no
sooner than 0.7 s apart and again whenever the room changes, and crossfaded
over a quarter of a second. Each swap of it clears a `ConvolverNode` and
throws away a second of reverberation, which then builds up again from
nothing; at every cell, at walking pace, it never finished building and was
heard as a flutter. The late field is diffuse and hardly changes across a
room. The early part, which carries the direction, still follows every cell.

**The listener keeps the nearest cell out to 2.2 m, and is given one within
1.1 m to begin with.** The solver writes no cell inside the furniture or
against the walls -- on `hssd_0076` there are more empty lattice points than
full ones -- while the picture lets the listener stand anywhere a room
contains them. The page fell silent wherever no cell was within two lattice
steps, and walking a room grazes such places constantly: a fade out and a
fade in a tenth of a second apart, the loudest crack measured. Hearing the
room from a metre away is wrong; hearing nothing is more wrong, and audible.
The two distances differ so that the edge of the solved air cannot chatter.

These decisions were made against measurement. `tests/js/smoothness.mjs`
walks a field offline -- the scheduler of `spatial.js`, the convolutions of
`engine.js`, the page's own decode -- and reports how far the binaural filter
moves from one ten millisecond frame to the next, as the second difference
of its level, of its interaural delay and level, of the energy of its late
part and of when it starts, and as the step of its spectrum. On six seconds
of walking across the living room of `w41_field_hssd_0076`, each decision
undone in turn:

| | level dB | timbre dB | reverberation dB | interaural delay, samples | interaural level dB | arrival, samples |
| --- | --- | --- | --- | --- | --- | --- |
| the page before, near enough | 9.0 | 6.5 | 4.2 | 5.8 | 12.5 | 79 |
| without the delay line | 0.7 | 1.2 | 0.2 | 5.2 | 0.2 | 33 |
| on `ConvolverNode`s | 1.1 | 1.4 | 0.2 | 1.6 | 1.8 | 18 |
| with a fade shorter than the interval | 4.4 | 3.5 | 0.2 | 1.3 | 1.0 | 24 |
| with the late part at every cell | 0.6 | 1.1 | 1.9 | 1.2 | 0.2 | 17 |
| **as it ships** | **0.7** | **1.1** | **0.2** | **1.2** | **0.2** | **17** |
| the lattice on its own | 1.2 | 3.5 | -- | 1.4 | 2.8 | 39 |

The last row is the nearest cell decoded at the true pose at every frame,
with no schedule and no crossfade: not a floor, since walking the page is
smoother than it -- a crossfade spanning two cells averages them. Turning,
the page is at it: 0.02, 0.14 and 0.01 on level, timbre and interaural
level, against 0, 0.22 and 0.01, where on `ConvolverNode`s it measured 0.63,
1.63 and 0.72. A fade of 80 ms instead of 120 would put the sound a degree
nearer the head and leave turning as it is; walking, it gives back what the
worklet gained.

**The plots are of the response at the cell, not of the voice and not of
the decode.** The omnidirectional channel of the cell the listener stands in
gives the spectrogram and the Schroeder decay, in absolute decibels against
the cell nearest the source, so walking away plots lower; the covariance of
all channels gives where the early and the whole energy arrive from, drawn in
the head's frame. Rebuilt when the cell changes, in a worker of their own;
changing the voice or turning the head changes nothing in the levels.

## What was rejected

**Interpolating responses between cells.** Comb filtering above the first
wavelength that fits between two cells; the producer's plan says the same.
Measured since: blending the four cells of the lattice box halved the timbre
step and the arrival jerk of a walk, at four times the decode, and was not
what cost the listener anything once the delay line was in.

**An equal power crossfade.** Its gains square to one, which is right for two
unrelated responses; two a head update apart are nearly the same signal, and
it doubled the level jerk of a turn.

**Turning a cell's response by the angle between the source seen from the
cell and from where the listener really is.** A rotation the decode already
does, and it draws a continuous interaural delay where the lattice draws a
staircase; it also turns every reflection by that angle, which is false, and
it moved the interaural jerk of a measured walk by one and a half per cent.

**A warm up before each fade in**, so that a new response's strongest taps
pass through its empty `ConvolverNode` unheard. It halved the moderate
transients, doubled the large ones and cost 25 ms of latency; the worklet
made the question moot.

**`ConvolverNode`s for the early part, as this ADR first decided.** It
rejected a worklet then: the browser's convolver already runs a partitioned
convolution on a thread of its own, and a hand-written one would have been
slower and unverifiable. It named the condition for revisiting that, a
browser that clicks on buffer swaps despite the crossfade; the condition was
met, and the arithmetic is verified in node against direct convolution.

**Order 3 in the page.** Four times cheaper, and the owner chose order 7,
which the measured update time supports.

## Consequences

The tests cannot run the page, but node runs its modules: `tests/test_sh_js.py`
checks the harmonics, the rotation and the head convention against the
library, `tests/test_analysis_js.py` the direction diagram and the absolute
levels, `tests/test_grid_js.py` the admission of grid tiles,
`tests/test_decode_js.py` the decode against a direct time-domain one and the
seam, `tests/test_partitioned_js.py` the early convolver against direct
convolution, and `tests/test_audio_smoothness_js.py` the ring, the rules of
`spatial.js` and `engine.js`, and the harness. The page exposes
`window.reverberate` so the rest can be driven in a browser, and it was.

`run_view.py`, the build-time analysis of ADR 0006, is gone with the page
that read it.

The decode is `audio/decode.js`, and `audio/brir.worker.js` the message shell
around it, so the tests and the harness drive the page's own arithmetic with
no browser.
