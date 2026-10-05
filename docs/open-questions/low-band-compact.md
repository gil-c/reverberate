# Are the low band responses a lever, and what does each way of pulling it cost

Date: 2026-10-05

Status: measured on the first scene's own pairs and on the dense line; the
form that passed is built as an option that is off (`low/compact`,
`docs/formats/scene-pack.md`) and nothing is decided: the owner hears a
pack both ways first. Written for lot L14a of ADR 0016. The code is
`reverberate.render.compact`; the measurements are
`python -m reverberate.experiments.w46_compact_low pairs | rails | degrees | pitch`.

## The answer

Yes. The first scene's pack is 24 GB of which 20.7 GB are `low/ir`:
16 887 pairs of 1.23 MB. Kept as the bins of its transform, in 16 bits,
each degree cut where it has decayed 60 dB, a pair is 174 kB in the mean
and the 20.7 GB are 2.9 GB, with nothing changed that a measurement at the
head can find over -42 dB of the response, and that only in its tail.

Factors are of 1.23 MB, on 600 pairs of the scene taken evenly over their
order of distance; an error is the worst third octave from 100 to 1250 Hz
of the worst of 150 pairs, on the head's sphere, with the head 0, 0.10 and
0.20 m from the cell (the worst of the three), early then whole response;
the ninth decile is in brackets. Bytes and minutes are of the first scene,
at the 6 MB/s its pack came home at (57.6 minutes for its `low/ir`).

| lever | factor | early, dB | whole, dB | level | saves | way home | on the machine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bins` | 1.41 | -142 | -147 | 0.00 dB | 6.0 GB | 17 min | the cache and the transfer, not the solve |
| `bins,int16` | 2.80 | -80 (-93) | -78 (-84) | 0.00 dB | 13.3 GB | 37 min | the same |
| `bins,degree=50` | 1.64 | -43 (-56) at 0.20 m, -70 on the cell | -40 (-53) | 0.01 dB | 8.1 GB | 22 min | the same |
| `bins,decay=60` | 3.62 | -79 (-87) | -42 (-49) | 0.01 dB | 15.0 GB | 42 min | the same; the solve only as `--low-seconds` |
| `bins,int16,decay=60` | 7.06 | -77 (-87) | -42 (-49) | 0.01 dB | 17.8 GB | 49 min | |
| `bins,int16,degree=50,decay=60` | 8.01 | -43 (-56) | -40 (-48) | 0.01 dB | 18.2 GB | 50 min | |
| a rail's positions in low rank | 4, were it kept | 0 to -2 above 800 Hz | | | | | not worth a lot of its own |

**Recommended as the candidate default: `bins,int16,decay=60`.** The degree
lever is sound (on the dense line it is no worse than today in any third
octave) but it adds an eighth to what the three others give and is the one
lever that takes channels out of the engine's output; it stays an option.

None of this lowers the memory of the wave solve or its time: a cell's
receiver records are its 984 nodes by the solve's steps, whatever is kept of
the 64 channels fitted from them. What lowers them is fewer steps
(`--low-seconds`, lot L12) and fewer cells (the pitch, below). Applied
where the pairs are written, on the machine, the form takes the pair cache
and its way home down by the same factors: the hour is eight minutes.

## How an error is read

As `docs/open-questions/low-band-translation.md` reads it. A cell's 64
channels are moved to the head as the engine moves them
(`spatial.translate`), read on a sphere of the head's radius, 0.10 m (each
channel weighed by `j_n(k a)`), and the energy of the difference is set
over the energy of the truth per third octave, on the 50 ms after the
arrival and on the whole response. The worst case stands beside the ninth
decile. A level is the change in a third octave's energy, before and after
50 ms.

Two sets of data:

- **150 pairs of the first scene** (600 for the sizes), 0.60 to 13.6 m
  from their source, from the pairs the scene's run brought home, put in
  the stored form by `trace.level.pair_low`. There is no truth beside a
  pair, so what is read is what a lever changes: the pair kept, read back,
  and the difference moved to a head 0, 0.10 and 0.20 m away. A pair is
  read from a distance only if the serving rule lets the engine read it
  from there (0.15 of the source's distance).
- **The dense line** of W44 (hssd_0076, 209 nodes 33 mm apart, the source
  0.31 to 4.15 m away), where the solved response at the head is known.
  Thirty cases a geometry, those the serving rule allows.

## 1. The sample: 16 bits want the bins and a scale per 100 Hz

The response decays; what a format adds must lie under it at every moment.
On the 150 pairs the pressure falls 74.5 dB from its loudest 20 ms to its
last 100 ms in the median, 67 dB at the tenth percentile and 53 dB in the
slowest pair. "Under the decay" is the least, over the whole response from
the arrival on, of the response's level over the noise's, on the head's
sphere; "floor" is the noise under the response's peak.

| kept as | bytes | early, dB | whole, dB | floor, dB | under the decay, dB |
| --- | --- | --- | --- | --- | --- |
| samples, float16 | 2.0 | -35 (-52) | -34 (-50) | 67 | **-0.7** |
| samples, int16, a scale a pair | 2.0 | -46 (-61) | -37 (-50) | 64 | **-4.7** |
| samples, int16, a scale a channel | 2.0 | -57 (-71) | -45 (-59) | 87 | -0.2 |
| samples, int16, a scale a degree | 2.0 | -51 (-67) | -40 (-55) | 82 | -1.1 |
| bins, float16 | 2.83 | -65 (-79) | -68 (-72) | 80 | -1.3 |
| bins, int16, a scale a pair | 2.83 | -36 (-58) | -25 (-46) | 53 | -24.6 |
| bins, int16, a scale a channel | 2.83 | -48 (-67) | -36 (-56) | 78 | 2.1 |
| bins, int16, a scale a degree | 2.83 | -42 (-62) | -30 (-50) | 72 | -5.1 |
| **bins, int16, a scale a channel and 100 Hz** | **2.80** | **-80 (-93)** | **-78 (-84)** | **94** | **16.9** |

- **Every way of keeping the samples in 16 bits ends over the response's
  tail** in some pair. Their noise is as loud at the end as at the
  arrival, and as loud in a third octave of 23 Hz as in one of 290 Hz.
  `float16` of the samples also breaks the format: it puts 1e-4 and more
  of the spectrum's peak above 1414 Hz, where a reader may assume nothing
  (`render.pack.validate` refuses the synthetic pack so written).
- **One scale a pair, or a degree, is ruled by the loudest channel.** In
  a pair of ten the channels of degree 4 hold 21 dB more than channel 0 at
  100 Hz (below), and every other channel pays for them.
- **One scale a channel and a block of 100 Hz** follows the response in
  frequency, which is where it has its range (the crossover's ramp takes
  it to nothing between 707 and 1414 Hz): 78 dB under the response in the
  worst third octave
  of the worst pair, a floor 94 dB under the peak, and never nearer than
  17 dB to the response's own decay. The scales are 4 kB a pair, one per
  cent. It is what `int16` is.

## 2. Degree against frequency

**The rule is in `j_n`, not in `k R` over `n`.** Within `R` of the cell a
plane wave is the sum over the degrees of `(2n + 1) j_n(k R)^2`, and
`j_n(x) <= x^n / (2n + 1)!!`. Degree `n` is kept from the frequency where
that bound on its share is a stated level under the whole, at `R = 0.30 m`
(a head's 0.10 m at the 0.20 m one cell is moved), through a raised cosine
100 Hz wide:

| level | degree 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60 dB | 0.3 | 15 | 62 | 136 | 229 | 333 | 445 Hz |
| 50 dB | 1 | 27 | 91 | 182 | 288 | 404 | 525 Hz |
| 40 dB | 3 | 47 | 134 | 243 | 363 | 489 | 619 Hz |
| 30 dB | 10 | 84 | 196 | 323 | 457 | 592 | 729 Hz |

A degree that would enter under 100 Hz is left whole (there is nothing to
save). The rule first proposed, `k R >= alpha n`, lets the low degrees go
too early: at `k R = 0.5` degree 1 still holds 8 per cent of the energy.

**On the dense line**, the prediction at the head with each rule against
the solved truth, worst third octave of the worst case, early (the whole
response reads the same to 0.3 dB). "Changed" is the prediction with the
rule against the prediction without, over the truth.

| geometry | today | 60 dB | 50 dB | 40 dB | 30 dB | `k R = 0.5 n` | `k R = n` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one cell, 0.098 m | -17.3 | -17.3 | -17.3 | -17.3 | -17.3 | -17.3 | -8.6 |
| one cell, 0.196 m | -17.7 | -17.7 | -17.7 | -17.7 | -17.7 | -17.7 | -4.3 |
| two cells 0.16 m apart, head 0.065 m from one | -25.9 | -25.9 | -25.9 | -25.9 | -25.9 | -25.9 | -13.3 |
| two cells 0.39 m apart, head half way | -25.3 | -25.3 | -25.3 | -25.3 | -25.3 | **-21.7** | -12.2 |
| two cells 0.59 m apart, head half way | -24.1 | -24.1 | -24.0 | -24.0 | -23.6 | **-19.6** | -7.8 |
| changed, the worst of the five | | -61 | -50 | -43 | -32 | -20 | -4 |

- **At 60 and 50 dB the rule is no worse than today in any third octave,
  worst case or ninth decile, to 0.1 dB**, and in no degree's worst band.
  At 40 dB the same to 0.2 dB. At 30 dB a third octave is 0.5 dB worse at
  the furthest fusion, 0.7 dB at the ninth decile. The rule in `k R` over
  `n` loses 4 to 5 dB at the two furthest fusions from `alpha = 0.5` and 9
  to 16 dB everywhere at 1.
- **It is not better either.** Band by band and degree by degree a cell of
  the table moves at 50 dB by up to 21 dB one way and 26 dB the other, but
  only cells that hold -55 dB of the truth or less. The upper degrees
  under their frequency carry no field the line can see, and no harm.
- Read on the scene's own pairs, `degree=50` changes the head's field by
  -70 dB on the cell, -58 dB at 0.10 m and **-43 dB at 0.20 m** (ninth
  decile -56 dB), and no third octave's level by more than 0.01 dB. The
  scene's arrays are 0.26 m in radius where the line's are 0.39 m under
  800 Hz, so their upper degrees hold more of what follows.

**What the upper degrees hold under their frequency** is the encoder's gain
and the room's slowest modes, not the field at the head. Mean channel of a
degree over channel 0, as stored, median pair then ninth decile:

| third octave | degree 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- |
| 100 Hz | +5.9, +21.5 dB | -21.6, +2.2 | -52.9, -27.6 | -64.7, -27.6 |
| 200 Hz | +2.2, +7.6 | +1.9, +15.8 | -20.2, +1.6 | -48.3, -20.6 |
| 315 Hz | +0.7, +3.2 | +1.6, +6.3 | +0.7, +11.9 | -16.9, +3.8 |
| 500 Hz | +0.1, +2.7 | +0.5, +2.7 | +0.7, +3.1 | +0.9, +6.1 |
| 800 Hz | -0.1, +1.6 | -0.1, +1.6 | 0.0, +1.7 | +0.2, +2.2 |

A field gives every degree the pressure's level; a degree is over it just
under the frequency it enters at (where the array can barely tell it) and
far under it lower still (where the encoder's regularisation gives up).
After 0.4 s degree 4 at 100 Hz is 15 dB over channel 0 in the median pair
and 24 dB at the ninth decile: in the three small packs the channels of
degrees 3 to 5 stop decaying at 35 to 50 dB under the pressure's peak from
0.6 s to the window's end, which is the long ring lot L12 saw. On the
head's sphere it is nothing.

**What it saves is small**: 1.16 times at 50 dB, 1.22 at 40 dB, on top of
the bins; 1.13 times on top of the three other levers. And it has a price
none of them has: **under its frequency a degree is gone from the engine's
own 64 channels**, since a head's degrees are the cell's. In the two real
packs, whose head is 5 mm from its cell, the raw channels change by -13 dB
while the
pressure changes by -151 dB and the head's sphere by -66 dB. A reader of
the raw order 7 signal under 525 Hz would see it; a head does not.

## 3. Length

What is left of a response late, energy in a fifth of a second over the
whole, on the head's sphere, 150 pairs:

| third octave | 0.6 to 0.8 s | 0.8 to 1.0 s | 1.0 to 1.2 s |
| --- | --- | --- | --- |
| 100 Hz | -48 (-42), worst -38 | -60 (-54), worst -36 | -64 (-57), worst -38 |
| 250 Hz | -57 (-52), worst -46 | -70 (-64), worst -57 | -72 (-65), worst -54 |
| 500 Hz | -70 (-64), worst -57 | -78 (-71), worst -57 | -79 (-72), worst -56 |
| 1 kHz | -71 (-64), worst -57 | -83 (-75), worst -61 | -84 (-76), worst -61 |

(median, ninth decile in brackets, then the slowest pair.)

| cut | factor, with the bins | whole, dB | early, dB | level |
| --- | --- | --- | --- | --- |
| every pair at 0.8 s (`--low-seconds 0.8`) | 2.12 | -34 (-52) | -83 | 0.00 dB |
| each degree 50 dB under the pressure's peak | 4.42 | -34 (-41) | -79 | 0.01 dB |
| **each degree 60 dB under** | **3.62** | **-42 (-49)** | **-79** | **0.01 dB** |
| each degree 70 dB under | 3.03 | -51 (-58) | -80 | 0.00 dB |

- **A length per pair and per degree is both smaller and nearer the
  response than one length for all.** At 0.8 s the slowest pair loses
  -34 dB of its 100 Hz third octave; cut where it has itself fallen 60 dB,
  it loses -42 dB, and the median pair is 0.75 s of pressure, 0.55 s of
  degree 2 and 0.40 s of degree 7. The pressure is kept whole in one pair
  of 26.
- **A degree is cut as it is heard within reach**: its channels weighed by
  the bound of section 2, against the loudest 10 ms of the pressure. Read
  as stored, degrees 3 to 5 never fall 60 dB (the floor above) and nothing
  is saved: 1.2 times, against 2.57 times so read.
- **At 70 dB the cut meets the response's own floor** (74.5 dB in the
  median pair): a third of the pressures are whole and the factor falls to
  2.15 over the bins.
- **What masks the cut, and what does not.** Lot L12 found the mirror's
  tail empty after 0.6 to 0.8 s above 1 kHz in the three small packs, so
  the high side masks nothing there and nothing else is playing in a
  response to an impulse. What is cut is itself 60 dB under the response's
  loudest 10 ms and 42 dB or more under the response's energy in its own
  third octave; under running speech it lies under the speech that
  followed. It is the end of a tail at -60 dB and is the owner's to hear:
  `render check` gives the two files.
- **Only a shorter solve lowers the machine's memory and time**, and a
  solve has one length for every pair of its source position, the slowest
  one's: the fixed cut's worst pair is 18 dB worse than its ninth decile.

## 4. The bins, the compact form, and what the engine pays

`low/ir` is zero above the top of the crossover's ramp, 1414 Hz, and is
kept as 4800 samples that reach 2 kHz. The bins of its transform up to
there, 1699 complex numbers, are the same response to its float32
(-142 dB, the rounding): 1.41 times. The three other levers are then
natural in that form, a table of bins per degree:

- a degree kept from a frequency is its bins from that frequency;
- a degree cut at `N` samples is its transform at `N`: half the length,
  half the bins. `N` is a multiple of 50 ms;
- 16 bits are of the bins, with their scales.

A cut degree ends in time, so it reaches a little over 1414 Hz (1.2e-4 to
1.8e-4 of the spectrum's peak, over the 1e-4 a reader may assume); the
decoder brings it back under on the transform of the whole length.

**The engine reads `ir[row]` as it did** (`render.low` is untouched): the
pack's reader decodes a pair when it is asked for
(`render.compact.CompactIr`), and the engine keeps the spectra of the pairs
in use as before. Decoding a pair with every lever is 3.1 ms of one core
against 0.4 ms to read 1.23 MB (2.2 ms with `bins,int16`). The first scene
reads 16 887 pairs in 1200 s by 14 sources, one pair per second of scene
and source: **3 ms per second of scene and source**, against 40 to 180 ms
for the low band and 220 to 810 ms for a source's whole render
(`scene-pack.md`). Nothing is held: a decoded pair is 1.23 MB while the
engine turns it into the spectrum it keeps.

## 5. Neighbours on a rail: not as a code

Fourteen runs of twelve rail positions 8 cm apart heard at one cell exist
in the pairs that are home; eight were read (the source 1.5 to 10 m
away). Their twelve responses, on the head's sphere, reduced to their first
singular vectors:

| rank kept | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| energy left, dB | -3 to -6 | -5 to -10 | -8 to -14 | -11 to -19 | -20 to -35 | -31 to -43 | -41 to -53 |

At a quarter of the rank, which is the 4 times, the error per position is
0 to -2 dB above 800 Hz in the worst position of every run and -12 to
-23 dB at 100 Hz at the ninth decile. -30 dB needs 6 to 8 of 12 and -40 dB
needs 7 to 10: one
and a half times at most. Putting the twelve on one arrival time first is
worse (-30 dB needs 8 to 11): what differs between them is the room, not
the direct sound.

Twelve positions over 0.88 m hold about `2 L f / c + 1` independent
responses at `f`: 2 at 200 Hz, 5 at 700 Hz, 8 at 1400 Hz. That is a rank
per frequency, and it is what lot L11 already reads at the source of the
data (`--rail-positions`: a rail solved every 10 to 12 cm and read from
eight positions with weights per frequency). **A code of its own for the
rails is not worth a lot**: its gain is under two at an error that can be
stated, and the solves L11 leaves out are worth more than the bytes.

## 6. The cells' pitch, for the lot that owns it

Two cells a pitch apart, fused, the head half way (the node, or the two
nodes, furthest from both), eight pairs of cells a class of the nearer
cell's distance to the source; worst third octave of the worst pair, early,
ninth decile in brackets. A figure in bold is of pairs the serving rule
(0.15 of the source's distance) refuses.

| pitch | source 0.65 to 0.9 m | 0.95 to 1.3 m | 1.4 to 1.8 m | 1.9 to 2.5 m | from 2.5 m |
| --- | --- | --- | --- | --- | --- |
| 0.16 m | **-22.9**; -25.6 where allowed (6 of 8) | -34.2 (-35.0) | -30.4 (-33.8) | -26.2 (-27.6) | -30.2 (-32.5) |
| 0.20 m | **-22.7**; -25.3 where allowed (6 of 8) | -35.3 (-36.0) | -31.1 (-32.0) | -35.5 (-37.0) | -29.2 (-30.0) |
| 0.26 m | **-17.8**; -30.3 where allowed (2 of 8) | -28.7 (-29.3) | -27.2 (-28.1) | -30.3 (-31.6) | -30.4 (-30.6) |
| 0.29 m | **-13.7** (none allowed) | **-25.3**; -30.4 where allowed (4 of 8) | -27.7 (-27.8) | -27.1 (-29.3) | -26.9 (-28.5) |

- **From 0.95 m out, 0.20 m holds what 0.16 m holds** (-29 to -35 dB
  against -26 to -34 dB), and 0.26 m holds -27 dB. 0.29 m holds -27 dB from
  1.4 m, and -25 dB between 0.95 and 1.3 m, where the rule refuses half
  its pairs.
- **Under 0.9 m the pitch is the source's**: -23 dB at 0.16 and 0.20 m,
  -18 dB at 0.26 m, -14 dB at 0.29 m. The rule's share is what keeps the
  pitch near a voice, and it is right to.
- A path cell every 0.20 m in place of every 0.15 m is a quarter fewer
  cells, pairs and receiver records; every 0.26 m, 42 per cent fewer, at
  up to 6 dB of the error between 0.95 and 1.8 m. The line is one line, one
  height and one source: the open points of `low-band-translation.md`
  stand. The plan is not changed here.

## The three small packs, heard both ways

`python -m reverberate.render compact IN.h5 OUT.h5 --levers ...`, then
`python -m reverberate.render check IN.h5 --against OUT... --names ...`.
The engine's low band from each rewritten pack against the original's, on
white noise over three seconds in which the source is heard, on the head's
sphere, all third octaves then the worst; the checker's own table (the
level of each third octave of the pressure, early, late and on the clips)
reads 0.0 dB in every cell for every variant of the three packs.

| pack | `bins` | `bins,int16` | `bins,degree=50` | `bins,decay=60` | `bins,int16,decay=60` | all four |
| --- | --- | --- | --- | --- | --- | --- |
| `v1_near`, a voice 2.0 m from its cell | -152 | -93, -85 | -67, -71 | -63, -54 | -63, -54; 7.2 times | -61, -54; 8.2 times |
| `v1_far` | -152 | -91, -79 | -66, -70 | -57, -46 | -57, -46; 6.5 times | -57, -46; 7.4 times |
| `smoke1` (three voices; traced before the lead) | -152 | -94, -88 | -54 to -35, **-19** | -59, -51 | -59, -51; 2.8 times | -56 to -35, **-19**; 6.6 times |

`smoke1` is the pack of the first smoke run, traced before
`/mirror`'s `lead_s` was laid into `low/ir`: its responses arrive 2 to
5 ms into their window, where the trace now leaves 11 ms and more, and its
nearest voice (0.79 m) holds degree 3 at 40 to 70 Hz 13 dB over the whole
of the pressure. A degree's ramp has no room before such an
arrival, and the degree lever moves a third octave of that voice by -19 dB
(0.02 dB of level); and the pressure itself stands at -56 dB of its peak in
the window's last millisecond, what its masks rang before the arrival come
round, so that two of its three pairs are not cut at all. It is not a pack
the trace writes any more. The pairs of sections 1 to 3 are the scene's,
with the lead.

## What was built

- `reverberate.render.compact`: `Levers` (`bins`, `int16`, `degree=DB`,
  `decay=DB`; none by default), `encode` and `decode` of one response,
  `CompactIr` (the reader's `ir`), `write_compact`, and `compact_pack`,
  the converter.
- `reverberate.render.pack`: `PackWriter(low_levers=...)` writes
  `low/compact` in place of `low/ir`; `read_pack` gives `Low.ir` as
  before. With no lever the writer's bytes are today's, and the converter
  copies the file.
- `python -m reverberate.render compact IN.h5 OUT.h5 --levers bins,int16,decay=60`.
- **The trace's hook** is the writer's argument: where
  `reverberate.trace.run` opens its `PackWriter`, one more keyword,
  `low_levers=Levers.parse(<the option's text>)`. On the machine the same
  `encode` takes a cached pair as it is (`top_hz` 1500, the solve's own
  limit, in place of the crossover's 1414), which is what would make the
  way home seven times shorter; that is the pair cache's file, another
  lot's. (Lot L15a placed the hook, `--low-levers`, and built the cache's
  file: with `top_hz` 2000, see the last section.)

## What the data cannot settle

1. **Whether the cut at 60 dB is heard.** Every figure says no; the
   owner's ears say. `v1_near` and `v1_far` are ten seconds of one voice;
   the full scene's pack was not home when this was written, and only its
   pairs were read.
2. **The raw channels.** Nothing here reads the engine's 64 channels
   other than as a head hears them. If training reads the raw order 7
   signal under 525 Hz, the degree lever is not for it. The decay lever
   touches them too, less: it ends the late floor of section 2 with the
   degree that holds it, -28 and -40 dB of the raw 64 channels in the two
   real packs, where a head hears -57 and -63 dB.
3. **A truth for the scene's arrays.** The line's arrays are 0.39 m in
   radius under 800 Hz and the scene's 0.26 m: what the degree lever
   removes from a scene's pair (-43 dB at 0.20 m) is larger than on the
   line (-55 dB), and whether that is the encoder's gain leaving, which is
   what the table of section 2 suggests, or field, needs a dense line
   solved with the scene's own array.
4. **The pitch off the line**: one line, one height, one source.
5. **A pair cache kept this way.** The cache holds a pair before its masks
   and its air; the form applies to it as it is, and was measured on the
   stored form only.

   **Measured since (lot L15a), and built.** Not with `top_hz` 1500. A
   cached pair of the first scene holds 45 dB under its energy above
   1500 Hz in the median and 32 dB in the worst of forty; `low/ir` is cut
   from it in time, by the onset's window, before its masks, so what is
   dropped above 1500 Hz comes back under 1414 Hz: with the bins cut at
   1500 Hz the stored response differs from the plain pair's by -48 dB of
   its energy in the worst pair, -46 dB in its worst third octave, and one
   onset moved by a sample. With every bin to the cache form's 2 kHz
   (2400 bins a channel, 621 kB a pair, a factor of 1.98 and not 2.64) it
   differs by -88 dB, -83 dB in the worst third octave of sixty pairs, and
   no onset moves; channel 0 alone, which the levelling reads, by -75 dB at
   the worst. That is the form `accel.pairs.PairCache` writes where a
   trace's bundle says so (`bins,int16`, never the decay or the degree: a
   cache serves every pack of its dwelling), 5 ms a pair to write and 2 ms
   to read on the laptop's core, 0.3 ms for channel 0. **Not yet run on a
   machine.**
