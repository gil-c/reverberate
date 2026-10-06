# How fast can the signal engine render a scene on the laptop, and what bounds it

Date: 2026-10-05

Status: measured on the first whole scene (`w45_clarify_scene/scene1/A`,
twenty minutes, fourteen sources, order 7, 48 kHz). The engine has a second
set of parts, `reverberate.render.fast`, which is the default; the parts it
replaces are kept as `RenderSettings(engine="reference")` and the tests hold
one against the other. Written for lot L20 of ADR 0016. The owner's
question: the first scene took 3 h 26 min to render on the laptop, and a
render that is "only convolution and gain" should take about two minutes.

## The answer

The owner is right about the order of magnitude, and the engine was wrong
by most of it; **two minutes is not reached**. Three figures, each of the
whole scene:

- **one fast core, alone**: about 1 100 s of it (0.08 to 0.11 s a
  source-second while the listener rests, 0.30 while he walks), against
  6 100 s for the engine as it was: **5.5 times less work**, and the
  whole scene in 18 minutes on one core where it took 100;
- **the mix, as it was timed** (six processes, the laptop shared with
  other agents' jobs): **11 minutes**, 656 s, at a load of 2 to 5, and 14
  minutes, 837 s, at a load of 5 to 10, against 3 h 26 min for the same
  scene's stems on the same shared laptop: **15 to 19 times less**;
- **the audit's stems**, the same way: 17 to 30 minutes at the rates
  measured, against 3 h 26 min.

What is left is not a constant to tune. It is, in this order: that the
laptop has four fast cores and six slow ones that are worth one and a half
between them (a slow core takes 4.5 times a fast one on a transform), so
that six processes give 2.1 times one while the machine is shared; and the
listener's walk, a fifth of the scene's time and half of its cost, where
the format asks for new responses of every source every few steps. The
last section says what would close each.

## The floor: what cannot be avoided

- **The output.** 64 channels by 48 000 samples by 1 200 s are 3.69e9
  samples, 14.7 GB in float32. The laptop copies memory at 54 GB/s (one
  core, measured) and hashes at about 2 GB/s: writing and checking the file
  is under 10 s and is not what bounds anything.
- **The sources.** The fourteen sources sound 8 360 seconds between them
  (a voice a third of the time, a noise most of it). Silence costs nothing,
  in the old engine as in the new: the unit of cost is the *source-second*.
- **A convolution engine's count, per source-second.** A step's response
  is 64 channels by 1.2 s. Rendered by brute force, as a workstation's
  convolution plug-in would (uniform partitions of one step, 24 of them, a
  cross-fade between two responses), a source-second is
  `20 blocks x 24 partitions x 64 channels x 2401 bins x 8 x 2` = 1.2e9
  floating point operations, and 64 inverse transforms a block besides:
  about 1.5e9, 0.15 s of a core that does 1e10 a second. But the response
  must also *exist*: 3.7 million numbers a source and a step, transformed
  (4.4e8 operations a step, 9e9 a source-second) whenever anything moves.
  Brute force is two minutes for a scene at rest and hours for one that
  walks.
- **The structure the pack already has** is cheaper than brute force, and
  is the floor that matters:
  - *early*: 3 to 15 arrivals a step, each one mono signal read at a moving
    delay (12 taps at two ends) and laid on 64 harmonics that turn (two
    products a channel): about 200 operations a path and a sample, 4e7 to
    1.4e8 a source-second, plus one short transform a row for its bands;
  - *low*: 4 kHz, twelve times fewer samples: the convolutions are nothing;
    raising 64 channels to 48 kHz is 29 taps a sample, 9e7 a source-second,
    once for a mix; a head off its cell applies a 64 by 64 or 64 by 128
    matrix a frequency, 75 frequencies, 80 frames a second: 5e7 to 2e8;
  - *tail*: 45 plane waves, each a convolution with 1.2 s of shaped noise:
    45 inverse transforms, 2.1 ns a sample each on this machine, times how
    much longer the transform is than what it gives.
  Summed, 2e8 to 5e8 operations a source-second where nothing moves: 20 to
  50 ms of one core, **170 to 420 s of one core for the scene, 30 to 70 s
  on the laptop's cores if they all pulled**. That is the two minutes the
  owner expected, and it is the floor for a scene at rest.
- **What moves has another floor.** A listener who walks changes every
  source's low band cells every 0.3 s and its tail histograms as often:
  each new histogram is a response to build (noise shaped in 8 bands and 45
  directions, brought to what the bank reads, transformed: about 30 ms),
  each new pair of cells an inverse (16 ms) and each new response of the
  low band a decode (1.4 ms). These are not per sample; they are per
  *change*, and the scene has thousands.

## The engine as it was, on the real pack

One process, one thread, the window 1065 s to 1095 s (the listener at rest
for 18.5 s, walking for 11.5 s), all fourteen sources, 238 source-seconds:
173 s. By part, seconds of one core per source-second:

| | early | low | tail | total |
| --- | --- | --- | --- | --- |
| reference engine, the window | 0.17 | 0.36 | 0.20 | 0.73 |

8 360 source-seconds at 0.73 are 6 100 s of one core; six processes that
gave four times one (`scene-pack.md`) and shared the machine made the 3 h
26 min. Where each part spent ten to a hundred times its floor:

- **early**: the dry signal filtered into *every* combination of band,
  mask and air distance, 96 signals at twice the rate over the whole run,
  whether the step had three arrivals or fifty; then each path a product
  of a hundred coefficients by those signals, twice (5 200 operations a
  path and a sample for a voice with four arrivals, against 200); in
  double precision, with a 270 MB table of coefficients a noise source;
- **low**: a transform as long as a response (1.2 s, 64 or 128 channels) to
  keep 87 ms of it, twenty times a second; and the head moved off its cell
  by a quadrature of 400 plane waves, 6e7 operations a frame, 80 frames a
  second, *whether or not the head had moved since the last frame*: a
  head that stands still paid the same operator again every 12.5 ms;
- **tail**: a transform of 1.7 s to give 0.5 s, in double precision; and,
  for a mix, every source's 45 inverse transforms and its encoding on 64
  channels paid apart;
- **all three**: 6 to 166 MB arrays allocated and copied per run, which is
  why six processes gave four times one.

## The leads, expected and measured

Seconds of one core per source-second are of the same window unless said;
"at rest" is 290 s to 310 s (113 source-seconds), "walking" 1084 s to
1096 s (94 source-seconds).

| lead | expected | measured |
| --- | --- | --- |
| a row's bands as one filter on the two steps it is read over, not 96 signals over the run | 5 to 15 times on the bank | early 0.17 to 0.034 with the next |
| the read and the encoding as a loop in C over float32 (`render/native.py`) | 3 to 5 times on them | 0.8 ms a path and a step |
| each low band response convolved once over a run, not a response a step | 20 times on the convolutions | low at rest 0.33 to 0.02 with the next two |
| the translation in closed form (`spatial.translate.translation_matrices`): `sum_l i^l j_l(k d) M_l(direction)`, fifteen real matrices from a table of triple products | 25 times on forming an operator, no quadrature error | 0.5 ms an operator against 50; equals the quadrature's to 1e-7 where that is exact |
| the operator kept: a head that stands still applies one matrix to every frame of every source | the whole of it at rest | 0.04 ms a frame |
| two cells' inverse by blocks (`pair_inverse`): a 64 by 64 Schur complement | 5 times | 16 ms against 50, equal to 1e-11 |
| the low bands of a mix summed at 4 kHz and raised once, in C | 13/14 of the raising | 0.1 s in 20 s of scene |
| the tail in float32, its transforms two seconds long (`tail_steps`) | 2 by 2 | 0.20 to 0.047, then to 0.025 a source alone |
| the tails of a mix summed as spectra wherever a response is heard at one level, transformed and encoded **once** for all the sources | 14 times on the tail at rest | less: the mix at rest is 0.078 a source-second against 0.097 for its stems apart. A source at one level the whole two seconds costs a product and no transform; a voice that starts or stops in them is still transformed apart, and a third of the voices' runs hold such an edge |
| the carrier drawn in C, and mapped by every process of a render from one file | 0.9 s to 0.15 s a source; 1.3 GB once, not a process | as expected |
| silence | nothing: already skipped | unchanged |
| float32 end to end | 2 on memory traffic | inside the figures above |
| the reference's table of coefficients made only when read; dry segments held once | memory, not time | a process from 9.3 GB to about 2 GB |
| Accelerate's transform (vDSP) in place of scipy's | 2 on the transforms | not built: no binding without a new dependency |
| a response built per step and convolved by partitions (the workstation's design) | worse: 9e9 a source-second while anything moves | not built, counted above |
| stems stored at a lower order or rate | the audit's disk | not built: see the audit below |

What did **not** pay, measured: Accelerate's matrix product does not scale
across processes on this machine (four processes each take 2.7 times as
long as one: the cores of a cluster share its matrix unit), so the parts
avoid it in their inner loops; and a scene's dry signals are filtered
twenty seconds at a time, which a process handed every sixth ten seconds
did again each time: a render's shares are now a minute long.

## What changes in the samples

The fast parts are the reference's mathematics in another order and in
single precision. On the real pack, the mix of all fourteen sources over
1076 s to 1088 s (at rest, then walking), fast against reference:

| part | largest difference, of the peak | rms, of the signal |
| --- | --- | --- |
| early | 2.9e-7 | -134 dB |
| tail (the same noise, drawn from the same seed) | 4.2e-7 | -129 dB |
| low | 8.3e-5 | -84 dB |
| the mix | 6.1e-5 | -86 dB |

- The early part and the tail differ by the rounding of float32. The
  tail's noise is the *same* realisation: the generator is exact in both.
- The low band differs by the reference's own quadrature. With the
  quadrature's operator put in the fast part the difference falls to the
  rounding; the closed form is the exact one. On the synthetic density
  pack, whose responses are noise to 2 kHz read 0.3 m from two cells, the
  quadrature's error is 6e-3 of the peak: the reference's error, not the
  fast part's.
- A walking head's fusion is solved in two products where the reference
  spreads on plane waves; kept in double precision between the two (an
  inverse is five hundred times its matrix), it is the same to 2e-7.
- A mix is no longer the sum of its stems to the bit: it is their sum to
  4e-7 of the peak, because what the sources share is summed before it is
  transformed. The audit sums stems, as before.
- A run's length (`chunk_steps`) and the tail's (`tail_steps`) move the
  samples by 5e-7, where the reference's moved them by 2e-8.
- The C text and its `numpy` twin give the same bits (no fused product):
  a machine without a compiler renders the same file, more slowly.

## The whole scene, timed

`write_mix` (`python -m reverberate.render mix <pack> <out>`), the order 7
mix of the fourteen sources to a file under a scratch folder, its SHA-256
taken as it is written; the file removed afterwards.

| what | processes | wall | processor | a process holds | load of the machine |
| --- | --- | --- | --- | --- | --- |
| the mix, 0 to 1200 s (8 361 source-seconds) | 6 | **837 s** (1.4 times real time) | 4 065 s | 5.1 GB at most | 5.3 before, 6.6 after; 10 during |
| the same, again, with less held a process | 6 | **656 s** (1.8 times real time) | 3 471 s | 4.6 GB at most | 2.4 before, 4.8 after |
| the mix, 280 to 340 s, at rest (362 source-seconds) | 4 | 20 s (3.0 times real time) | 57 s | 4.7 GB | 1.9 before, 2.3 after |
| the mix, 300 to 360 s, at rest, one process, warm | 1 | 42 s (1.4 times real time) | 42 s | | 15 |
| the mix, 290 to 310 s, at rest, one process, warm | 1 | 8.8 s (2.3 times real time) | 8.8 s | | 2 to 3 |
| the mix, 1084 to 1096 s, walking, one process, warm | 1 | 28.7 s (0.42 times real time) | 28.7 s | | 2 to 3 |
| the audit's stems, 90 s of its service from 280 s | 6 | 16.6 s of stem a second: the scene in 17 min at that rate | | 3.4 GB | 6 to 10 |

The same bytes whatever the processes and whenever, **on the laptop**:
the two whole renders have one SHA-256 and the window of a minute has one
with four processes and with eight. **Not on the CI's Linux, and it is not
rounding.** The test that holds a process started afresh to the test's own
process (`tests/test_render_fast.py::test_several_processes_write_the_file_one_engine_writes`)
failed in four of the nine runs of the whole suite that held it between
its arrival and 2026-10-06, two of them on branches that had not touched
the engine and were run again until they passed. The test's tolerance is
as it was, 1e-12 of the peak. It is marked `quarantine`: out of the run a
pull request waits for, and run three times on every pull request by a job
of its own that is red when it fails and holds nothing.

What that job measured (lot L27, eight runs of the CI):

- Alone in a process of its own, which then renders for the first time
  itself, it fails in 16 runs of 28. After the other tests of its file,
  when only the processes it starts are fresh, in 1 of 5; in the whole
  suite, the same case, in the 4 of 9 above. So it is not what the tests
  before it leave behind, and either side of the comparison can be the
  one that is wrong: whichever renders for the first time.
- The files differ in 15 to 29 frames of 14 400, all inside one step of
  2 400 samples, on all 64 channels, by up to 6.6e31 of the peak, and by
  not a number in two runs. That is memory read after it was given back,
  not a sum rounded otherwise. The 1.196e-03 of the first report is the
  same thing on a milder day, and came back to the digit.
- With `MALLOC_PERTURB_` set, which fills memory as it is given back, it
  fails in 6 runs of 7 and every frame of a step differs (2 400 of
  2 400), and nothing else of
  `tests/test_render_fast.py` or `tests/test_render_engine.py` fails.
- The parts rendered alone by another fresh process are this process's
  in most failures, and in two the early part is off by 1e13 and 1e28:
  it is the early part, and it is a race.
- The same test with each process it starts having loaded the C text and
  made the delay table before its engine starts a thread
  (`test_processes_readied_before_their_threads_write_the_file_one_engine_writes`,
  which runs after the first and so in a process that has rendered)
  passed 29 runs of 29, with and without `MALLOC_PERTURB_`. At one
  failure in five with nothing readied, 29 passes by chance are one in
  several hundred.
- It is not that test's alone. Under one process a core every worker is a
  fresh process, and in the first full run of four workers
  `tests/test_render_seam.py::test_the_engine_takes_the_level_a_band_and_the_scalar_when_told`
  failed on two renders of one pack that were not one, 3 337 samples of
  921 600 apart from frame 57 on, the frames of the list above. Whichever
  test renders first in a process is exposed, and one that holds its
  render against nothing does not notice. `tests/conftest.py` now makes
  the table before any test of a process, which keeps every test out of
  the way of it and is to go when the defect does. The quarantine's job
  runs without that file (`--noconftest`), so that its test stays as
  exposed as it was measured; with it, 6 runs of 6 passed.

The reading, from those and from the text: `render/native.py` makes the
delay table on first use with no lock (`_table`), and `early_interval`
gives the loop in C the table's address without keeping the table for the
length of the call. In a fresh process the first run's intervals are
rendered by threads, each finds no table and makes one, and the one stored
last frees the one a loop in C is still reading, which the allocator hands
to the next request. `_library` has the same shape (it says it has tried
before it has built), which costs a thread one call through the numpy twin
and nothing else. An engine with `workers=1`, which is what the mix's
processes are given, starts no thread and is not exposed; an engine left
at its default is, in the first run of every fresh process, on any
machine whose allocator reuses the block at once. The remedy is three
lines in `render/native.py` (a lock about the two first uses, and the
table held in a name across the call); it is not made in the lot that
found it, which was not to touch the engine, and when it is made the mark
comes off the test. Until then "the same bytes" is a property of an engine
on one thread or of a process that has rendered before.

One more thing the same lot met, once: in the first run of the CI that
held the libraries' products to one thread a process (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`), the reference engine's child, started with an
environment of its own and so left to take four, did not write its
parent's digest
(`tests/test_render_engine.py::test_two_processes_give_one_output`, which
had not failed before). The child is now given its parent's count and
has passed since. If the count is the cause, as it looks, "the same bytes
on any machine" needs it said, and no document says it yet.
Neither whole
render had the laptop to itself (other agents' jobs, and a photo analysis
daemon at 180 per cent of a core during the first). The processor time,
3 471 and 4 065 s, is 3.2 to 3.7 times the 1 100 s one fast core needs
alone: that is the measure of what the processes cost one another, and it
is the first thing that bounds the render. Measured apart, the same ten
seconds rendered by six processes at once take each 2.8 times what one
takes alone (the tail 3.3 times): six processes give 2.1 times one. A
quiet laptop was not available and no figure is claimed for one.

## What bounds it, and what would close it

**The count of fast cores, and what processes cost one another.** 1 100 s
of a fast core is 2 minutes only on nine of them. The laptop has four, and
six slow ones that take 4.5 times as long on a transform (measured: 12.4
against 56 ms for the tail's 45 transforms of 153 600 points); and six
processes side by side gave 2.1 times one, not six: the parts move 27 MB
arrays through transforms that no longer fit a core's cache when six run,
and each process still holds 4.6 GB. This is a factor of four to five
between what was measured and the target, and it is the first thing to
work on: fewer, larger things held (the tail's spectra, a source's tables
of arrivals), buffers kept and not allocated a run, and threads in one
process in place of processes.

**The walk.** While the listener walks, a scene-second costs 2.3 s of a
fast core against 0.45 at rest: the walk is 246 s of the scene and half of
its cost. By part, of the 28.7 s the walking window took: the tail 15 s
(every new histogram is a response: its noise shaped and brought to the
bank's reading, 13 ms, transformed, 12 ms, and convolved, 12 ms; four new
ones a source-second), the low band 10 s (a new response decoded and
transformed, 1.4 ms, seventeen a source-second; a new inverse a pair of
cells, 16 ms; two products a frame in double precision), the early part
3.5 s. None of this is per sample: it is per change of what the pack
points at, and it is the format's own interpolation (a step reads up to
four histograms and four low band responses, with weights that move).

**What would close it**, in the order of what each is worth:

1. *A card.* The design maps to one unchanged in its mathematics: the
   parts are transforms, products and one gather, all of which `cupy`
   has, and the reference parts already run there. The fast parts were
   written for the host (the read and the encoding are a C loop, the
   transforms `scipy`'s); their card twin is the same three files with
   `xp` in place of `numpy` and the C loop as one `ElementwiseKernel`.
   Not built in this lot, and so not measured: no figure is claimed.
2. *More fast cores*, or a quiet machine: see above.
3. *The walk's histograms.* A histogram's response is rebuilt whenever a
   walk meets it; kept on disk beside the pack (45 by 57 600 float32,
   10 MB each, 3 400 of them in the first scene: 34 GB) it would be read
   and not made. Too large as it is; the bank's reading (`_norm`, a third
   of the cost) is 64 bytes a histogram and could be kept in the pack by
   the trace for nothing. Not built: it changes the pack.
4. *Accelerate's transforms.* The transforms are half of what is left at
   rest; vDSP is about twice `scipy`'s on this processor and has no
   binding in the project. A `ctypes` call into it is a page of code and
   a dependency on the system's framework on one platform only.
5. *The audit's stems as two files a source*, the low band at 4 kHz and
   the rest at 48 kHz, raised when a mix is served: a twelfth of the low
   band's samples and no raising per source. Worth 10 per cent of the
   stems' time and 0 of the disk (the high part is the 64 channels).
   Not built.

## The audit

The audit renders stems, one source at a time, and keeps them: what a mix
shares between its sources (the tails' transforms, the low bands' raising)
is not shared there, which is the price of a solo that is instant. What
changes for the owner:

- **The cache is rendered again.** The engine's code is part of a stem's
  key: the 112 GB of the first scene's stems are stale and the page
  renders new ones as it is used, or
  `python -m reverberate.viz.audit_stems <pack.h5>` does it beforehand:
  17 to 30 minutes at the rates measured (16.6 s of stem a second with
  six workers on the shared laptop), where it took 3 h 26 min.
- **A worker keeps every source's engine** (sixteen, where it kept six and
  built a source's engine again every other chunk: 9.6 s of stem a second
  before that one change, 16.6 after), and the workers of a pack share
  the sources' noise as files under `<cache>/<pack>/carriers` (1.3 GB a
  scene, drawn once).
- **What he hears is the same** to -86 dB of the mix: the same arrivals,
  the same noise in the tails, a low band that differs from the old one
  by the old one's quadrature. The sound check gives the verdicts below
  on both engines.
- **The stems stay, at order 7.** Rendering on demand instead of keeping
  stems needs one source faster than real time on one core with room to
  spare: it is (0.1 s a source-second at rest), but fourteen at once
  while the listener walks are 2.3 s of a fast core a second, and a seek
  would wait. The cache is still what makes solo and mute instant; it is
  simply filled ten times sooner.

## The sound check, both engines

`python -m reverberate.render check <pack> --window 1078 1090 --sources
noise_3 far_2 near_2` (the listener at rest, then walking), once with each
engine (`REVERBERATE_ENGINE=reference` for the old parts): **186 verdicts,
the same 186 on both**: 138 pass, 13 warn, 2 fail, 32 inform, 1 skipped.
Of the report's 2 172 numbers, ten differ by more than 0.05: nine are
levels of a silence, -142 to -156 dB with the fast engine where the
reference's are -143 to -170 dB (single precision's floor under a signal
that has stopped), and one is the sample at which an extreme is found. The
two failures are the scene's own, as `first-scene-defects.md` has them.
