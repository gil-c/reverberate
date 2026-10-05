# What the sound check found in the first whole scene, and what of it is the scene's

Date: 2026-10-05

Status: every class of failure has its cause, measured on the scene's own
pack. Most were the check reading a room it had never met; two things it
did not flag are real and can be heard; one fix waits for the owner's ear
(`relevel`), one for the next trace (the anchor), and the instants to listen
at are at the end. Written for lot L17 of ADR 0016. Section 10, lot L18,
says what the levelling scalar of section 8 is made of: one error of scale,
found, and the ratio of two interference patterns; and that one number
replaces it.

The scene is `w45_clarify_scene/scene1`, trace A: hssd_0076, 20 minutes,
14 sources (3 near voices, 6 far voices, 5 noises), 18 219 pairs of the low
band. The check of its window from 660 to 720 s read **60 FAIL, 141 WARN,
492 PASS**; the same check read 0 FAIL on the small pack at rest. Nothing
here was solved again: the machines are gone, and about 12 000 of the
scene's pair responses are on the laptop, which is what the measurements on
pairs below were made on.

## The answer, class by class

| class | FAIL | cause | what was done |
| --- | --- | --- | --- |
| `level_step` | 11 | the check: one tone crossing the nulls of its own reflections | read on a band (a comb of tones) |
| `band_alignment` | 19 | the check: a band's loudest arrival taken for its first | read where the pack puts the direct sound; told where there is none |
| `pre_arrival_energy` | 4 | the check: "before the first arrival" of a band whose first arrival it misread | read before the straight line's time |
| `reverberation_reference` | 18 | the check: a field whose source is rooms away | told beyond 1 m from that source; the field's own range judged instead |
| `interval_edge` | 3 | the check: a clip's loop read as a cut | a join is held to go on, an edge to start from zero |
| `zipper` | 1 | the check: a reflection's Doppler shift read as a line | lines inside the Doppler spread are not read |
| `audibility` | 1 | the check: one second of the end of an utterance | told under 3 s of speech |
| `level_at_listener` | 1 | the clip: a washing machine whose level is under 80 Hz; and the check's distance | distance of the window; the clip stands (below) |
| `level_distance` | 1 | the check: a source going behind a wall | read where the source is seen |
| `binaural_front_back` | 1 | the check: no direct path, no one direction | warns there, as `direction` does |

And what no test said:

| found | size | where it is |
| --- | --- | --- |
| everything above the crossover of a source goes up and down as anything walks | 5 to 8 dB within a second at the worst | the trace: each pair levelled on its own (`relevel --seam constant`, section 10) |
| the mirror's direct sound is under `1 / d` | 2.3 to 2.7 dB, every band | the alignment's gain, read in 0.5 ms of energy (section 10) |
| a near voice 1.3 m from the head, behind a wall | 30 dBA, 24 dB under itself | the recipe's generator |
| a far pair joined 5 to 20 ms after its direct sound | 0.1 to 1.4 dB at 1 kHz, early part | the trace's anchor (changed, needs a trace) |
| the noises are not heard | 5 to 9 dB under a near voice together, two of five under 25 dBA | the generator's default and two clips |
| the band under the crossover moves in stairs of a step | lines 20 Hz apart 33 to 38 dB under a 400 Hz tone | the engine, left as it is |

## 1. `level_step`: a tone is not a level

The check sent 400 Hz and 2500 Hz through each source and read the level
of each tone every 10 ms. Eleven sources jumped by 3.6 to 11.2 dB between
one frame and the next but one.

**What changes in the pack at those steps is small.** Over the window, per
source, between two audible steps:

| table | ninth decile | 99th percentile | largest |
| --- | --- | --- | --- |
| `level/high_gain_db` | up to 0.28 dB | up to 0.80 dB | up to 1.64 dB |
| `level/onset_s` less the first arrival | up to 6.0 ms | up to 22.6 ms | up to 45.7 ms |
| the early part's energy at 2 kHz | up to 0.58 dB | up to 16.4 dB | up to 41.5 dB |
| the direct path's gain, where there is one | under 0.2 dB | under 0.3 dB | 0.30 dB |

(each the worst of the fourteen sources) and per source in the 1200 steps:
`low/pair` changes at 0 to 187 steps,
`low/cell` at 0 to 187, `mode` at 0 to 14, the tail's histograms at 4 to
26, a path is born or dies at 0 to 147 steps, the direct path comes or goes
at 0 to 3. The anchor's jumps are real (section 2) and act on the
crossover's octave only, where neither tone is. None of this is a jump of
7 dB at 2.5 kHz.

**What the tone does is the room.** Rendered part by part round each
failing instant:

- far_1, 2500 Hz, 706.69 s: the early part alone falls from -11.7 dB to
  -37.5 dB and comes back to -14.5 dB in 190 ms, smoothly, while nothing
  changes in the pack but the listener's place (24 rows before and after,
  `high_gain_db` -3.69 to -3.71 dB, the same two pairs). The listener walks
  at 0.5 m/s; 2.5 kHz is 13.7 cm long; two reflections that reach a moving
  head from opposite sides beat at twice the speed over the wavelength,
  7.6 Hz, a null every 130 ms. It is a null of the room, crossed.
- noise_1, 707.4 s: the early part and the tail are each steady within
  2 dB (-25.6 to -23.1 and -24.9 to -24.6 dB) and their sum falls to
  -44 dB: the tone of the one cancels the tone of the other for 30 ms.
- far_2, 400 Hz, 674.5 s: the band under the crossover falls from -35.5 to
  -45.8 dB over the three steps in which the source walks its last 24 cm,
  and stays there. A quarter of the wavelength of 400 Hz is 21 cm, and the
  listener does not see the source: a standing wave from its side to its
  node.

A tone in a room where anything moves does this; a voice is not a tone,
and a listener follows the level of a band. So the test is rewritten: it
reads the level on **two combs of tones 100 Hz apart**, 200 to 700 Hz and
1.5 to 6.4 kHz, whose own power is the same in every 10 ms (each tone, and
every beat of two, has whole periods in a frame). Each tone fades on its
own and their sum keeps the band's level. The limits have not moved: 1 dB,
warn to 3 dB. A switch of 3.5 dB on a comb through two moving paths is
found where it is and fails; one tone through the same two paths reads
more than 10 dB (`tests/test_render_check_scene.py`). The tone's own
reading stays in the result's detail.

On the pack: far_1 reads 0.83 dB where its tone read 6.95, noise_4 0.85
where its tone read 6.37, and **far_3 reads 2.61 dB**, a warning, at
709.89 s. That one is the scene's: the listener does not see far_3 there,
what it has above the crossover is its tail, the tail's four histograms
are exchanged for four others at 709.90 s, and `high_gain_db` climbs from
-4.69 to -2.27 dB over the 0.35 s before. It is the breathing of section 8,
caught once.

A comb is periodic in 10 ms, so two paths whose lengths differ by a
multiple of 3.43 m to within 7 cm fade all its tones together. In a room
that is a small share of the pairs of paths; it is the reason a comb may
warn where a noise would not.

## 2. `band_alignment` and `pre_arrival_energy`: three things

**(a) The check's reading.** It took a band's first arrival to be the first
peak of its envelope that reaches half the band's largest. Under the
crossover, at a far place, the direct sound is not half the largest: of
the scene's 9669 pairs with a direct path the loudest sample comes more
than 5 ms after the direct sound in 1 per cent within 2 m, 24 per cent from
2 to 4 m, 72 per cent from 4 to 6 m and 70 to 76 per cent beyond. The check
then compared a reflection under the crossover with the direct sound over
it. Read where the pack puts it (the first peak of each band's envelope
within 3 ms of the straight line's time, as `trace.clock.read_direct` reads
a pair), on eleven probes with a direct path:

| probe | distance | the old reading | the direct sound under less over the crossover |
| --- | --- | --- | --- |
| far_3 at 675.67 s | 5.45 m | +1.38 ms, FAIL | 0.00 ms |
| far_4 at 660.02 s | 4.98 m | +2.69 ms, FAIL | +0.04 ms |
| noise_2 at 719.92 s | 8.52 m | +4.85 ms, FAIL | +0.08 ms |
| noise_4 at 719.92 s | 4.80 m | +4.44 ms, FAIL | +0.17 ms |
| noise_3 at 660.02 s | 3.47 m | +0.75 ms, WARN | +0.77 ms |
| noise_4 at 660.02 s | 4.65 m | +0.56 ms, WARN | +0.56 ms |
| five others, 1.7 to 7.4 m | | +0.04 to +0.31 ms | +0.04 to +0.42 ms |

The two bands are on one clock. The two that still warn have a reflection
within a millisecond of the direct sound, which a band that ends at
1.4 kHz cannot part from it.

`pre_arrival_energy` read what came 1 ms before "the first arrival" of the
band over the crossover, found by the same half of the largest: behind a
corner that was a reflection 4 to 23 ms after the first path, and
everything before it counted. It is now read before the straight line
between the source and the head could bring anything, which nothing
precedes whatever stands in the way.

**(b) What the late anchor does.** The trace anchored the window in which
the two bands are joined in pressure on the loudest sample of the pair's
response (`mirror.hybrid.blend`'s rule, right where the loudest sample is
the direct sound). Late by 5 to 20 ms, the window is longer, not misplaced:
the direct sound is inside it under the crossover (baked in `low/ir`) and
over it (the engine's window follows `level/onset_s`, the same anchor).
What changes is what lies between the direct sound and the loudest sample:
joined in pressure, where the two solvers' reflections are not the same
arrival and should be joined in power. Computed on eight pairs of the
cache, 3.9 to 11.5 m apart, the anchor 6 to 22 ms late: the stored
response and the engine's high band summed, against the same with both
anchored on the direct sound (the row made again from the cached pair, the
engine given the other `onset_s`):

| | the pack less the right join, third octaves 315 Hz to 3.15 kHz |
| --- | --- |
| the direct sound, 3 ms either side | 0.0 dB in every band, on all eight |
| the first 50 ms | -0.1 to -1.4 dB at 1 kHz; -0.5 dB at the most at 800 Hz and -0.3 dB at 1.25 kHz; 0.0 elsewhere |
| the whole response | -0.1 to -1.2 dB at 1 kHz |

Nothing is doubled and nothing is lost of the direct sound. Up to 1.4 dB
is missing at 1 kHz in the early reflections of a far source. The second
cost is that the anchor jumps: between two steps of a walk the loudest
sample changes from one arrival to another, and the window moves by more
than 5 ms 95 to 442 times a source over the scene, each time taking a few
reflections from one mask to the other over a step.

**(c) Where there is no direct path** the mirror's first path is a bend
round an edge, 34 to 124 dB under a source at 1 m on the fifteen probes of
the window (a direct sound at their distances is 10 to 20 dB under it), and
the band under the crossover is a wave through the openings, whose first
arrival comes 14 to 31 ms after that path and is the louder by far. The two
have no arrival in common and are not to be aligned: above 1.4 kHz such a
source is its reverberation. The check now says so and judges only what
comes before the straight line's time.

**The rule, and what is baked.** A pair is now anchored on **its own direct
sound**: the first peak of its response within 3 ms of the straight line's
time, where that peak is at least a fifth of the loudest sample, and the
loudest sample where it holds none (`trace.level.pair_anchor_s`). For a
pair the mirror gives a direct path it is the clock's reading; it needs
nothing of the mirror, so a row is still made as soon as its pair is
solved. Of the existing pack:

- `level/onset_s` is a table the engine reads and could be written again;
- **the rows of `low/ir` are stored with their masks taken on the old
  anchor**, and making them again needs the pairs. A third of them are not
  on the laptop.

Moving the table alone would join what lies between the two anchors in
power over the crossover and in pressure under it: half the fault for the
reflections that differ, and a new one for those that agree. It is not
done. **The next trace has the rule; this pack keeps its 1.4 dB.** A
re-trace needs no solve where the pair cache holds the pairs (the levelling
and the rows are made again: `level.jsonl` and the rows' files are named by
the rule and are not read from before).

## 3. Levels

**near_3 at 32.7 dBA** is one second of the end of an utterance (its
interval ends at 661.08 s): an active speech level of 38 dB at 1 m was read
on it, and the voice heard 6.6 dB over its free field. A voice of which the
window holds under 3 s is now told, not judged.

**noise_2 at 17 dB SPL** is the washing machine, and it is the clip: 25 dB
of its level lies under 80 Hz, which neither band renders (the solve's
band starts at 80 Hz and the mirror's chain cuts there). Stored at 58 dB
SPL at 1 m, it is 30 dBA at 1 m:

| clip | stored, dB SPL at 1 m | dBA at 1 m |
| --- | --- | --- |
| `appliance_washing_machine` | 58 | 30.0 |
| `other_kitchen` | 47 | 36.4 |
| `appliance_extractor_fan` | 52 | 39.5 |
| `street_traffic` | 58 | 47.7 |
| `other_boiling`, `other_frying`, `appliance_range_hood` | 50 to 60 | 47.6 to 51.4 |
| the 12 of `music`, the 5 of `television` | 57 to 62 | 50.0 to 57.8 |
| `water_tap`, `water_bath_filling`, `water_shower` | 58 to 62 | 57.0 to 60.4 |

The check also held a window's level against the distance of the whole
scene (4.75 m for a noise that stood 5.0 to 8.5 m away in the window); it
reads the window's now.

**Far voices 10 dB over the free field** are the room. A furnished room's
critical distance is under a metre (0.057 times the root of the volume
over the reverberation time: 0.6 to 0.7 m for 40 to 80 cubic metres and
0.4 to 0.5 s), and at 5 m in one closed room the reverberant field would
stand 17 dB over the direct sound; through doorways it is less, and the
window reads +7 to +11 dB. Past 2.5 m only a level under the free field is
judged.

**Every source over the 20 minutes**, from the pack's tables alone: the
energy of the arrivals and of the tail's histograms from 1 to 4 kHz with
`high_gain_db`, on the clip's A weighted level at 1 m and the source's
gain, every half second it sounds. Against the window's rendered levels
this reads 3 to 10 dB low where the source is seen (it leaves out the band
under the crossover) and 16 dB low where it is not, so the columns compare
sources and the last gives the true level where one was rendered.

| source | gain | sounds | at 1 m, dBA | at the head, tables | median distance | seen | rendered, 660 to 720 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| near_1 | -3.4 dB | 186 s | 53.1 | 49.6 | 1.6 m | 83 % | 56.3 dBA |
| near_2 | -5.5 dB | 220 s | 51.0 | 46.5 | 1.9 m | 77 % | 55.6 dBA |
| near_3 | -3.0 dB | 242 s | 53.5 | 48.7 | 2.5 m | 69 % | (1 s) |
| far_1 | -3.2 dB | 434 s | 53.3 | 39.1 | 7.2 m | 26 % | 50.6 dBA |
| far_2 | -1.3 dB | 427 s | 55.2 | 42.0 | 6.4 m | 66 % | 38.2 dBA |
| far_3 | -4.0 dB | 435 s | 52.5 | 44.7 | 7.1 m | 63 % | 47.0 dBA |
| far_4 | -4.0 dB | 446 s | 52.5 | 41.2 | 6.1 m | 62 % | 49.6 dBA |
| far_5 | -2.8 dB | 432 s | 53.8 | 43.5 | 6.4 m | 45 % | 52.0 dBA |
| far_6 | -2.8 dB | 432 s | 53.7 | 40.8 | 7.1 m | 23 % | 36.4 dBA |
| noise_1, shower | -7.7 dB | 940 s | 52.7 | 43.1 | 5.4 m | 76 % | 47.4 dBA |
| noise_2, washing machine | -10.5 dB | 829 s | 19.6 | 9.2 | 6.3 m | 69 % | 11.5 dBA |
| noise_3, street | -8.3 dB | 1200 s | 39.4 | 30.6 | 4.8 m | 79 % | 35.1 dBA |
| noise_4, kitchen | -9.5 dB | 937 s | 27.0 | 18.3 | 4.7 m | 75 % | 24.7 dBA |
| noise_5, television | -10.6 dB | 1200 s | 46.1 | 35.8 | 4.1 m | 38 % | 44.3 dBA |

A near voice while it speaks, against the rest, on the same reading:

| | first decile | median | ninth decile |
| --- | --- | --- | --- |
| over the five noises together | -17 to -31 dB | +4.7 to +8.9 dB | +12 to +25 dB |
| over the other eight voices | -19 to -33 dB | +2.8 to +6.7 dB | +32 to +38 dB |
| over everything else | -29 to -34 dB | -1.0 to +0.8 dB | +8 to +16 dB |

The first decile is a near voice the listener has walked away from or does
not see (section 9). In the median a near voice stands 0 dB over the rest,
and the rest is voices: the noises are 5 to 9 dB under it together, and
two of the five are under 25 dBA at the head.

**What is recommended, and what was changed.** The target for a scene meant
to train the separation of speech in noise: a near voice **0 to +5 dB over
everything else in the median, -5 to +15 dB from the first to the ninth
decile**, with the noises together within 5 dB of the far voices together,
so that neither alone is the difficulty.

- `sources.noise.gain_db`: **-6 to +6 dB**, in place of -18 to -6. A noise
  is drawn about the level its clip is stored at, which is its source's at
  1 m. This is the generator's default now (`scenes.generate.Parameters`).
  It raises the noises by 12 dB in the mean: at the middle of the range the
  television of this scene would stand at 55 dBA at the head and its shower
  at 55, where the near voices are, and the scene is then a hard one.
- `sources.gain_db` of the voices stays at -6 to 0 dB.
- The two clips that are under 40 dBA at 1 m (`appliance_washing_machine`,
  `other_kitchen`) should be levelled on their A weighted level, or on
  their level above 100 Hz, in the library's selection. Not done: it is the
  library's choice of levels, and a clip stored 28 dB higher by its own
  measure would pass full scale.
- For this scene, with no new trace: the engine applies a source's
  `gain_db` as the pack's attribute says. Raising the five noises by 12 dB
  there is the default above.

## 4. `interval_edge`: a loop is a join

The three failures are a noise's clip coming round: the generator writes a
looping clip as one interval a turn, each starting on the sample the one
before ends on, and the engine fades neither side of such a join
(`render.dry.DryTrack.from_recipe`), by design. The check read the end of
one and the start of the next as edges: a sample at the clip's own level.
The 5 ms fade is skipped nowhere it should apply.

What a join must do is go on. The step across each, over the rms of the
steps within 10 ms either side:

| clip | joins in the scene | the step across |
| --- | --- | --- |
| `water_shower`, 27 s | 34 | 0.5 |
| `street_traffic`, 288 s | 4 | 0.5 |
| `other_kitchen`, 288 s | 3 | 1.9 |
| `appliance_washing_machine`, 120 s | 6 | 2.3 |
| `television_01` to `05`, a playlist | 9 | 0 (each starts and ends on zero) |

No click. The check now holds a join to that (`interval_join`, the limits
of the click in noise, 6.5 and 8) and an edge to zero, and reads the
intervals that meet its window, not the scene's.

## 5. `zipper`: the listener's own Doppler

noise_4 does not move; the listener walks at 0.52 m/s through the probe.
A reflection that reaches a moving head is shifted by up to the speed over
the wavelength, 3.8 Hz at 2.5 kHz, and the check looked for lines 2, 4, 6,
8 and 10 Hz from the tone: it found a reflection. It is not the run
schedule of ten steps (a run's edge would be a click, and `tone_click`
reads -67 dB) and not the tail. Lines no further from the tone than the
source's and the head's speeds can shift it are no longer read; with that,
noise_4 reads -72 dB.

The three warnings at 20 Hz round 400 Hz (-33 to -38 dB) are the engine.
The band under the crossover takes each step's responses for the frames
nearest the step: between two steps the response changes during 37.5 ms
and holds for 12.5 ms, a stair a step long with rounded risers, where the
early part's gains are linear. A walking source's 400 Hz tone that falls
3.5 dB a step falls 3.6 dB within 20 ms of it. Reading each frame's
responses between its two steps would remove it; it is not done here,
because the rails read with more than two positions a step
(`rail-interpolation.md`) were measured on the present frames and the
engine's stems are being made with them. It stays a warning.

## 6. The validated field speaks for its own source

`reverberation_reference` and `late_spectrum_reference` held every source
to the field of S1, which stands in the living room, at the lattice point
nearest the head. For a source rooms away the decay is another: the 18
failures are 35 to 70 per cent in T20. Both are now judged where the
pack's source stands within 1 m of the field's own and told elsewhere,
with the distance. In their place every probe is held to **what the field
itself reads over its 437 points**, the T20 per
octave of nine tenths of them, its 5th and 95th percentiles
(`reverberation_range`, the same 20 and 35 per cent past the nearer end),
and the report tells what the pack's own probes span
(`reverberation_spread`). One source of the scene stands within a metre of
S1, the kitchen's noise_4, and is still judged: its T20 is within 15 per
cent of the field's on three probes, and its late colour within 6.8 dB
(two warnings).

## 7. The rest

- `level_distance`, far_6, -40.9 dB a decade: the listener never sees far_6
  in the window. The law is read where the pack gives a direct path, which
  is what the limit's reason always said.
- `binaural_front_back`, far_4, and `binaural_left_right`: no direct path;
  the window holds several bends and no one direction. They warn there, as
  `direction` did.
- `doppler`: told only where the listener does not see the source
  throughout the probe.
- `level_at_listener`: read on the window's distance; told for a voice
  with under 3 s of speech and for a level over the free field past 2.5 m.

## 8. Not flagged: the levelling scalar breathes

`level/high_gain_db` multiplies everything a source has above the
crossover. It is the mirror's gain plus the seam of the step's pairs, and
each pair is levelled on its own: over the scene's pairs a seam is +0.4 dB
at the first decile, +1.9 dB in the median and +3.4 dB at the ninth. Along
a walk the step reads a new pair every 8 cm of a rail and every 0.15 m of
the head's path. Per source, over the whole scene, within half a second of
a walk:

| | ninth decile | 99th percentile | largest |
| --- | --- | --- | --- |
| as traced | 1.7 to 2.4 dB | 2.6 to 4.9 dB | 3.7 to 6.8 dB |
| steadied over 1 s either side | 0.6 to 0.9 dB | 1.1 to 2.1 dB | 1.2 to 3.3 dB |
| steadied over 2 s either side | 0.3 to 0.5 dB | 0.5 to 1.5 dB | 0.6 to 1.9 dB |

and within a second at the worst 4.4 to 8.2 dB a source. A steady street
noise 5 m from the walking head goes from -7.6 to -1.0 dB between 487.75
and 488.75 s. The check sends its tones over one stretch of 5 s a source
and met it once (far_3, section 1).

No room does that. What a seam measures is how far the two solvers
disagree about a place, which changes over metres, plus what one octave of
one response happens to hold at one point, which changes over centimetres
and is no property of the band above. `reverberate.render.relevel` keeps
the first: `python -m reverberate.render relevel PACK` averages the table
under a raised cosine 2 s either side within every run of audible steps,
**in place**, keeps the trace's table in the pack and puts it back with
`--undo`. A source at rest before a head at rest keeps its scalar to the
bit. It is **off**: nothing was written to the scene's pack, and the trace
writes each step's own seam as before. If the owner's ear keeps it, the
trace's part is `--seam smooth`.

"Changes over metres" was a guess, and section 10 measures it: nothing of a
seam changes over metres. What is not the pattern of one point is one
constant, and the averaging is not needed.

## 9. Not flagged: a near voice behind a wall

near_1 stands 1.32 m from the head from 672.7 to 696.1 s and the pack gives
it no direct path: the mirror's shortest way round is 10.5 m, 124 dB under
a source at 1 m, and the wave's first arrival comes 54 ms after it sounds. It speaks there at **30 dBA** at the ears, and at 54 to 55 dBA
when it is seen at 707 to 720 s. The generator places a near voice 0.6 to
2 m from where the listener rests "on the floor plan", and a wall is on
the floor plan. Over the scene the three near voices speak unseen within
2.5 m for 15, 26 and 2 s, and unseen at all for 32, 50 and 76 s of their
186, 220 and 242 s (the listener walks 246 s of the 1200).

The check now says where a near voice speaks unseen (`near_voice_seen`,
a warning: it is the recipe's doing). The generator is not changed here: a
near voice's station wants a line of sight to the rest it is near, which
is the mirror's test on the dwelling and belongs to the scenes' lot.

## 10. One calibration, not a levelling a position

The owner's question (2026-10-05): "For the smoothing, I wonder why we
need it. Is it a question of calibration? I thought the adjustment was
made once and for all, and not dynamically at each position." It is a
question of calibration, the adjustment should be made once, and the
smoothing of section 8 is not needed. Lot L18; nothing was solved again,
and every number is read on the scene's pack, on its 18 219 seams
(`level.jsonl`) and on the 12 351 pair responses the laptop holds.

**What a seam is read on.** `mirror.hybrid.seam_db`: the energy of the
omnidirectional channel over the crossover's octave, 707 to 1414 Hz, over
the **whole response**, 1.2 s, of the wave side (with its air, before its
masks) against the mirror rendered at the same pair (its paths and its
tail). One ratio of two energies in one octave, direct sound and
reverberation together; and what it gives multiplies everything from
1 kHz to 20 kHz.

### What explains a seam

Over the 18 219 pairs: mean 1.91 dB, median 1.90, standard deviation 1.27.

| what the pairs are grouped by | groups | share of the variance | what is left, dB |
| --- | --- | --- | --- |
| a direct path or none | 2 | 1.6 % | 1.26 |
| the source's room and the head's | 4 | 1.5 % | 1.26 |
| the rooms and the direct path | 8 | 5.6 % | 1.23 |
| the distance, ten classes from 1 to 10 m | 10 | 1.4 % | 1.26 |
| the source | 14 | 0.8 % | 1.27 |
| the source's height, seated or standing | 3 | 0.1 % | 1.27 |
| the head's height | 2 | 0.0 % | 1.27 |
| rooms, direct path and distance together | 63 | 11.4 % | 1.20 |

Nothing of a room explains it. The mean is the same wherever it is read:
1.68 dB under 1 m (255 pairs, deviation 0.46), 1.71 from 1 to 1.5 m, 1.83
from 2 to 3 m, 1.80 from 4 to 5 m, 1.89 from 5 to 6 m with a direct path;
2.08 without one, 1.76 with. Of the eight groups of rooms and direct path
seven have their median between 1.6 and 2.4 dB, and one, 507 pairs seen
from the living room into the bedroom through its door, has 0.8. The
dwelling has two rooms the rule names (the bedroom, and the living room
with its kitchen and entry), so "by pair of rooms" is four numbers that
are one number.

**A model with structure is refused**: by rooms and line of sight it
removes 0.04 dB of 1.27, and the group that stands out is one doorway of
one dwelling. It was not built.

### The constant is one error of scale, and it is found

A pack is physical: a source of unit gain reads `1 / d` at the listener,
in both bands, with no calibration at all. The wave side does: its direct
sound reads +0.3 dB against `1 / d` on the scene's pairs (`trace.clock`),
-0.1 to -0.3 dB from 300 Hz to 5.6 kHz on the validated field. The mirror
does not:

| the direct sound against `1 / d`, dB | 300 to 800 Hz | 1 kHz | 2 kHz | 4 kHz | 5.6 to 8 kHz |
| --- | --- | --- | --- | --- | --- |
| the wave field of hssd_0076 (12 nearest points, 0.18 to 0.75 m) | -0.24 | -0.14 | -0.17 | -0.30 | -2.05 |
| the mirror's field, the same points | -2.57 | -2.49 | -2.83 | -2.93 | -3.69 |
| the mirror, from the pack (`alignment_gain` times the signature) | -2.32 | -2.45 | -2.79 | -2.89 | -3.61 |

The mirror's direct sound is **2.3 to 2.7 dB under the wave's in every
band from 300 Hz to 5.6 kHz**: 2.65 dB from 500 Hz to 4 kHz in the median
of the field's 40 nearest points (quartiles 2.60 and 2.70), and 2.1 to
2.3 dB in the third octaves from 630 Hz to 1 kHz on the scene's 271 pairs
under 1.2 m (quartiles 0.1 dB apart). A number that tight, at every
distance, is not a room's.

It is in `mirror.files.align_to_reference`. The alignment's gain is the
root of the ratio of two energies, the reference's direct pulse and the
mirror's, each read in a window of 0.5 ms (`mirror.direct.DIRECT_WINDOW_S`).
The mirror's pulse is its signature, a minimum phase filter: all of it is
in the window (-0.13 dB). The wave solver's pulse is band limited and
rings both ways: the window holds **-2.2 dB** of it, 13 per cent before
and 11 after. The two energies are made equal (+0.3 dB), so the mirror is
set 2.2 to 2.7 dB low, the same in every band. The seam has been putting
it back, a pair at a time, with the pair's own noise on top: 1.9 dB in
the scene's median and 2.3 on the validated field, less than 2.65 because
the pairs are solved to 1500 Hz and are already 2 dB down in the top third
of the seam's octave (the wave's direct sound over the mirror's reads
+2.0, +1.8 and 0 to +0.3 dB in the thirds at 800, 1000 and 1250 Hz, +1.2
to +1.4 dB over the octave).

**So the constant is not the dwelling's.** The signature has unit energy
over 24 kHz and 55 per cent of it under 8 kHz, which is +2.9 to +2.3 dB
from 300 Hz to 4 kHz, 2.4 dB over the band from 500 Hz to 4 kHz; a mirror
whose direct sound is `1 / d` has `alignment_gain` times that equal to
one: `alignment_gain` **0.76**, -2.4 dB, where the pack holds 0.548,
-5.2 dB. The measurement against the wave field gives 0.744 (0.548 raised
by 2.65 dB; a field's 0.0144 becomes 0.0195). The two agree to 0.2 dB, and
the first needs no reference field: it is a normalisation, the same for
every dwelling.

### What is left is two interference patterns

How far two seams differ, by how far apart their pairs are (half the mean
square difference, as its root, dB). The source along its rail and the head
on one cell, 968 rails with a direct path and 852 without; and the source
on one position, the head from cell to cell, 33 and 34 positions:

| apart | seam, direct | seam, none | wave side, direct | mirror side, direct |
| --- | --- | --- | --- | --- |
| 0.08 m | 0.43 | 0.46 | 0.35 | 0.31 |
| 0.16 m | 0.60 | 0.64 | 0.50 | 0.44 |
| 0.25 m | 0.66 | 0.75 | 0.58 | 0.53 |
| 0.4 m | 0.70 | 0.89 | 0.68 | 0.59 |
| 0.6 m | 0.75 | 1.05 | 0.85 | 0.71 |
| 0.85 m | 0.78 | 1.16 | 1.00 | 0.79 |
| 1.25 m | 0.82 | 1.23 | 1.28 | 1.03 |
| 2 m | 0.80 | 1.35 | 1.73 | 1.47 |
| 3.2 m | 0.91 | 1.30 | 1.97 | 1.98 |

(the head from cell to cell reads the same: 0.50 at 0.08 m, 0.69 at 0.4 m,
0.77 at 1.25 m, 0.87 at 3.2 m with a direct path.)

- **A seam is decided within 0.3 m.** Two thirds of the variance between
  two seams is there at 0.25 m and all of it at 1 m; from 1 m to 4 m it
  does not grow. The wavelength at 1 kHz is 0.34 m. Were it an error
  of the model about a room it would grow over metres, as each side does.
- **Each side alone goes on growing** with the distance, 2 dB at 3 m, and
  the two go together: over all pairs the wave's level in the octave and
  the mirror's are correlated at 0.98 (0.92 with a direct path). The
  mirror follows the room.
- **At 8 cm they are not the same pattern.** The wave side moves by
  0.35 dB and the mirror by 0.31, and their difference by 0.43, where
  two things that moved together would leave nothing and two that are
  independent 0.47. The mirror sums its image paths in pressure and
  draws its tail as noise, so it has an interference pattern of its own
  in that octave; it is not the wave field's, and cannot be: the two
  solvers' reflections differ by centimetres.

So the scatter, 1.1 dB with a direct path and 1.4 without, is the ratio of
two interference patterns in one octave at one point. A real room has such
a pattern in every octave, and the pattern of one octave says nothing of
the next: laid on everything above 1 kHz it makes a source breathe as no
room does. It belongs to neither band's level.

The two packs at rest (`v1_near`, `v1_far`) hold one pair each: +1.98 dB
at 2.0 m with a direct path and +3.98 dB behind a wall, the first on the
median and the second at the ninth decile of the pairs without a direct
path. The validated field's own seams are not stored a point; its
provenance holds their median and deciles (2.3, 0.9, 4.1 dB).

### Four ways of writing the table, measured

`reverberate.render.relevel`, and `--seam` of the trace. Per source, the
range over the fourteen; "left at the join" is the step's own seam less
what the table gives the band at 1 kHz, over the 55 572 steps where a
source or the head moves:

| | moves within half a second, 90 %, 99 %, largest | within a second, largest | left at the join: half of the steps under, 90 %, 99 %, largest |
| --- | --- | --- | --- |
| `pair` (the trace's) | 1.7 to 2.4, 2.6 to 4.9, 3.7 to 6.8 dB | 4.3 to 8.2 dB | nothing |
| `smooth`, 2 s either side | 0.3 to 0.5, 0.5 to 1.5, 0.6 to 1.9 | 1.1 to 3.6 | 0.3, 0.9, 1.9, 4.6 dB |
| `constant` | nothing | nothing | 0.7, 1.9, 3.3, 7.5 dB |
| `tapered`, at 1 kHz and under | as `pair` | as `pair` | nothing |
| `tapered`, at 2 kHz | 0.8 to 1.2, 1.3 to 2.4, 1.8 to 3.4 | 2.2 to 4.1 | |
| `tapered`, from 4 kHz | nothing | nothing | |

And a pair at a time, what `constant` leaves between the two bands at the
join: within 1 dB for 62 per cent of the pairs and within 2 dB for 90
(69 and 94 with a direct path, 54 and 85 without); 3.8 dB at the 99th
percentile.

`constant` is the median of the pack's pairs' seams (1.90 dB here) unless
a number is given. `tapered` is the constant from 4 kHz up, the step's own
seam to 1 kHz and half way between the two, in decibels, at 2 kHz; it is a
level a band, which a pack's table does not hold, and is given to the
engine at the render (`Engine(high_gain_db=...)`, `[step, band]`), the
early part by its bands and the late part a group of bands at a time.

### Which one

**`constant`, and its number in the calibration.**

- What `pair` does is heard and is wrong: everything above 1 kHz of a
  source goes up and down by 2 to 5 dB within half a second, 8 within a
  second, as anything walks. A step of 1 dB of a broad band is heard; this
  is a modulation of the whole upper spectrum at the rate of the walk.
- What `constant` leaves is not a motion of anything. Each band has the
  level its own solver gives it. Where they meet, the octave under 1 kHz
  stands 0.7 dB from the octave over it in the median and 1.9 dB at the
  ninth decile, differently from one place to the next: a tilt of a
  decibel or two across one join, which is what a room's own response does
  between any two octaves, and which no listener names.
- `smooth` keeps what changes over metres. The data hold nothing that
  does: it is the constant with a remainder of the pattern in it.
- `tapered` keeps the join exact and lets the octaves at 1 and 2 kHz
  breathe, by 2 to 5 and 1 to 2.4 dB. It protects a continuity at the
  join that has no physical meaning (the ratio of two patterns) at the
  price of half the fault. It is there to be heard.

It is the owner's ear that decides; the files are below.

### What the calibration becomes

- **Now, on this pack**: `python -m reverberate.render relevel PACK --seam
  constant` (the median of its seams, 1.90 dB), or `--constant-db 2.65`
  for the direct sound on `1 / d`. In place, `--undo` puts the trace's
  table back to the bit.
- **The alignment** (`mirror.files.align_to_reference`): the gain read
  **in the band**, not in 0.5 ms of energy: both pulses under one window
  wide enough for both (3 ms either side of each one's arrival, on points
  under 1 m, where nothing else is in it), the ratio of their mean power
  from 500 Hz to 4 kHz, the median over the points. On hssd_0076 that is
  0.0195 for a field and 0.744 for a pack.
- **Once for all**: the signature given unit gain in its band, 500 Hz to
  4 kHz, in place of unit energy (`mirror.direct.measure_signature`), and
  `MirrorAssets.pack_gain` one. No field of the dwelling is read for the
  scale. The lead, 10.67 ms, is the solver's pulse and not the dwelling's
  either.
- **The trace**, then: `--seam constant --seam-constant-db 0`.
  `low/seam_db` stays in the pack and in the report as a **check**: the
  median of the pairs under 1.5 m with a direct path within 0.5 dB of
  nothing says the scale holds.

**What of the seam is the reverberation's.** With the direct sound exact,
the pairs the reverberation decides stand a little over those the direct
sound decides: 2.08 dB without a direct path against 1.76 with one, 1.83
at 4 to 6 m against 1.71 under 1 m. The mirror's reverberation is 0.1 to
0.4 dB under the wave's in the octave at 1 kHz. That is a third of the
scatter of the pairs it is read on, and **the data do not support a
correction of the absorption
table or of the tail's gains**; they speak for one octave, in one
dwelling, with the table fitted on one source of that dwelling.

**What stands between this and a calibration made once for all**:

1. the normalisation above, in code, and the mirror's own tests and the
   validated hybrid of hssd_0076 read again with it (the hybrid was
   validated with each point's own seam, 2.3 dB in the median: the new
   scale is that field within 0.35 dB);
2. a second dwelling. A hundred pairs of it, near, far and behind a wall,
   with the scale by construction and no alignment: the median seam of the
   near pairs with a direct path within 0.5 dB of nothing confirms the
   normalisation, and that of the pairs without a direct path within 1 dB
   confirms the absorption table's reverberant level at 1 kHz;
3. above 1 kHz only a wave field solved to 8 kHz in that dwelling says
   whether the tail's gains by band, fitted on one source of hssd_0076,
   hold elsewhere: one field, at the cost of the one of hssd_0076.

**A wider crossover instead** was not built. The wave solver valid an
octave higher costs sixteen times the solves (the fourth power of the
frequency) and twice the pack's low band; the mirror valid an octave
lower is not validated there (it was fitted and judged above the
crossover, and under 500 Hz a room is modes and bends, not rays). Neither
removes the scatter: wherever the two meet, a seam a pair is the ratio of
two patterns.

### The files

`w45_clarify_scene/scene1/listen_seam/` (`scripts/listen_seams.py`): four
passages of ten seconds, the source alone and the mix of every source
heard there, under `pair`, `smooth`, `constant` and `tapered`, through the
page's decode with the scene's head, at one gain a passage; and a blind
set of the source alone with its key sealed. Its `README.md` says what to
listen for.

## The check on 700 to 720 s, before and after

The 14 sources, the measured head, the validated field of S1, two threads
on the laptop; the same pack and the same window, the check before this
lot and after it (`python -m reverberate.render check PACK --window 700
720`).

| | FAIL | WARN | PASS | INFO | SKIP |
| --- | --- | --- | --- | --- | --- |
| before | 48 | 109 | 497 | 37 | 11 |
| after | 1 | 82 | 517 | 130 | 13 |

By test, where anything changed, as FAIL / WARN / PASS / INFO / SKIP:

| test | before | after |
| --- | --- | --- |
| `level_step` | 10 / 2 / 1 / 0 / 0 | 0 / 6 / 7 / 0 / 0 |
| `band_alignment` | 14 / 1 / 22 / 0 / 0 | 0 / 1 / 24 / 12 / 0 |
| `pre_arrival_energy` | 8 / 1 / 28 / 0 / 0 | 0 / 0 / 37 / 0 / 0 |
| `reverberation_reference` | 4 / 9 / 24 / 0 / 0 | 0 / 0 / 3 / 34 / 0 |
| `late_spectrum_reference` | 2 / 19 / 16 / 0 / 0 | 0 / 2 / 1 / 34 / 0 |
| `reverberation_range` | (new) | 0 / 1 / 36 / 0 / 0 |
| `reverberation_spread` | (new) | 0 / 0 / 0 / 1 / 0 |
| `interval_edge` | 3 / 1 / 9 / 0 / 0 | 0 / 0 / 11 / 0 / 0 |
| `interval_join` | (new) | 0 / 0 / 3 / 0 / 0 |
| `zipper` | 1 / 3 / 9 / 0 / 0 | 0 / 3 / 10 / 0 / 0 |
| `binaural_front_back` | 3 / 6 / 28 / 0 / 0 | 0 / 9 / 28 / 0 / 0 |
| `binaural_left_right` | 1 / 4 / 32 / 0 / 0 | 0 / 5 / 32 / 0 / 0 |
| `level_distance` | 1 / 1 / 8 / 0 / 3 | 0 / 0 / 8 / 0 / 5 |
| `level_at_listener` | 1 / 6 / 4 / 2 / 1 | 1 / 0 / 5 / 7 / 1 |
| `audibility` | 0 / 1 / 12 / 0 / 0 | 0 / 1 / 10 / 2 / 0 |
| `doppler` | 0 / 11 / 1 / 0 / 1 | 0 / 7 / 0 / 5 / 1 |
| `direct_band_balance` | 0 / 0 / 25 / 0 / 0 | 0 / 2 / 23 / 0 / 0 |
| `near_voice_seen` | (new) | 0 / 1 / 1 / 0 / 0 |

Unchanged: `arrival_time`, `direct_level`, `direction`, `late_echo`,
`seam_third_octaves`, `tone_click`, `noise_click`, `duplicate_arrivals`,
`reverberation_plausible`, `c50`, `silence`, `dc`, `binaural_peak`,
`playback_level`, `speech_to_rest`.

What is left:

- **The one failure is the washing machine**: 16.5 dB SPL (13.9 dBA) at
  6.6 m where its free field is 32.2 dB, the listener seeing it. It is the
  clip (section 3) and it stands until the clip is levelled on what is
  rendered of it.
- `level_step`: six warnings from 1.0 to 2.6 dB (far_3 2.61, far_5 1.46,
  noise_5 1.34, near_1 1.09, far_6 1.07, near_2 1.00), where the tones read
  2.5 to 11.2 dB. Two are on the comb under the crossover (the stairs of
  section 5), four over it (the scalar and the tail's histograms, section
  8). `far_2`, which does not move there, reads 0.00 dB.
- `zipper`: the three warnings at 20 Hz round 400 Hz, -33 to -38 dB.
- `band_alignment`: 24 probes with a direct path within 0.5 ms and one at
  0.79 ms (noise_4, a reflection within a millisecond of its direct
  sound); 12 without one, told.
- `direct_band_balance` warns twice where it passed: it reads the direct
  sound under the crossover where the direct sound is, and no longer a
  louder reflection in its place.
- `reverberation_range`: 36 of the 37 probes within 20 per cent of what
  nine tenths of the field's 437 points read (0.54 to 0.71 s at 250 Hz,
  0.43 to 0.59 at 500 Hz, 0.39 to 0.56 at 1 kHz, 0.38 to 0.60 at 2 kHz,
  0.33 to 0.53 at 4 kHz); noise_2 at 719.98 s is 24 per cent over at 4 kHz
  and warns. Between them the probes span 0.32 to 0.74 s, the longest
  twice the shortest in one octave. The run itself read this test against
  the field's shortest and longest (0.31 to 1.85 s at 2 kHz, a few points
  read on their floor), which every probe passes and which says nothing;
  the row above is the run's own T20 against the percentiles, as the check
  now reads it, and is the only number of the table not printed by the
  run.
- `near_voice_seen`: near_1 speaks 1.4 s unseen from 706.30 s, 2.5 m away.

The number of results told and not judged went from 37 to 130: 68 are the
two tests against a field whose source is elsewhere, 12 the alignment of
bands that share no arrival. Each still gives its number.

## What to listen for, and where

Times are the scene's. The files of the first check
(`check_660/listen/<source>.wav`) start at 660 s.

| what | source and time | if the finding is right |
| --- | --- | --- |
| the scalar breathing (8) | noise_3, 487.7 to 488.8 s, 401.5 to 402.5 s and 109.5 to 110.5 s (a steady street noise, the listener walks) | its hiss, everything above 1 kHz, swells by 6 dB within the second and sinks back; steady after `relevel` |
| the same, four ways (10) | `listen_seam/`: noise_3 483 to 493 s, far_6 92 to 102 s, far_5 509 to 520 s, near_2 779 to 789 s | `pair` breathes; `constant` does not and its tone does not wander; `tapered` breathes less and lower; `smooth` is `constant` to the ear |
| the same on voices | far_6 96.5 to 97.5 s (8.0 dB); far_5 514 to 515 s (7.4 dB); far_1 374 to 375 s (6.2 dB, it walks); near_2 783.7 to 784.7 s (5.9 dB) | consonants brighten and dull as the speaker or the listener walks |
| a near voice behind a wall (9) | near_1, 672.7 to 675.7 s and 685.1 to 696.1 s; near_2, 859.5 to 870.0 s | a murmur from the next room at arm's length; the same voice is present at 707.7 to 712.6 s |
| a voice never seen (2c) | far_6, the whole of 660 to 720 s | dull and without consonants' edge: under 1.4 kHz the wave, above it reverberation only |
| a tone's nulls, not steps (1) | far_1 706.5 to 706.8 s; far_5 705.6 to 705.8 s; noise_1 707.3 to 707.5 s; near_1 708.6 to 708.8 s | nothing: no jump of the voice or the noise. A jump heard there refutes section 1 |
| the one step that is left (1) | far_3, 709.5 to 710.3 s | its reverberation brightens by 2 to 3 dB over a third of a second |
| a loop's join (4) | noise_1 at 690.125 s (and every 26.973 s); noise_4 at 752.296 s; noise_3 at 864.000 s | nothing: no click |
| no pulsing at the run's rate (5) | noise_4, 704 to 709 s | a steady noise, no flutter twice a second |
| the stairs under the crossover (5) | far_3, 708.3 to 713.3 s, its low vowels | at most a faint roughness; on speech probably nothing |
| noises not heard (3) | noise_2 anywhere; noise_4 anywhere | noise_2 is silence at any level one would listen at; noise_4 is at the edge of hearing |
| the late anchor (2b) | far_2 at 1100 s; noise_2 at 280 s (the largest, 1.4 dB) | nothing a listener would name: a third octave at 1 kHz, early reflections only |

## What a re-trace needs

- **The anchor**: no solve. The pairs are in the dwelling's cache; the
  levelling and the rows are made again under their new names and the pack
  is written. On the laptop a third of the pairs are missing, so it is a
  run on a machine that holds the cache.
- **The scalar**: nothing; `relevel --seam constant` on the pack, and
  `--seam constant` of the trace from then on (section 10). The scale
  itself is the mirror's alignment, in code, before the next dwelling.
- **The noises' levels**: nothing; a recipe's gains, applied by the engine.
- **A near voice in sight of its listener**: a new recipe, so a new trace
  with its solves.
