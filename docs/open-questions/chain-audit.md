# The audio chain, link by link, against truths that are not its own

Date: 2026-10-05

Status: the 2 dB are located and are a normalisation error of the mirror,
not of the wave band; eleven other defects of the same kind are listed with
their place in the code, four of which can be heard; the split between the
images and the rays is sound in its design and loses energy in one place;
the chain renders from 45 Hz, not from 80. Nothing of the chain was changed:
every defect is pinned by a test of `tests/test_chain_audit.py`, marked
`xfail` with the error measured, and the remedies belong to the lots that
own the files (L20 the engine, L21 the levelling) or to the owner. Written
for lot L19 of ADR 0016.

Everything here was measured on a laptop: small exact cases, the solver's
CPU twin on a box of 25 litres, the tracer's twin on shoeboxes, and, read
only, the validated fields of `w42_gpu_hssd_0076` (the wave field solved to
8 kHz, an independent truth for the mirror from 1 to 4 kHz at 173 of its
points), the first scene's pack and `level.jsonl`, and about 12 000 of its
pair responses.

## The four answers

**1. Where the 2 dB come from.** From one line:
`mirror.files.align_to_reference` (the ratio at `files.py:163-164`) reads
the level of a direct sound as its energy over 0.5 ms
(`mirror.direct.DIRECT_WINDOW_S`), and the two pulses it compares are not
the same pulse. The wave field's direct sound is spread by its grid: half
of its energy lies outside 0.5 ms (-5.5 to -6.1 dB of a unit pulse inside
the window, ten nearest points). The mirror's is the minimum phase
signature of `mirror.direct.measure_signature`, of unit energy, all of it
inside the window (-0.3 dB). The gain comes out 2.4 dB low at every
frequency, and the pack carries it: `alignment_gain` times the signature is
-2.36 dB re `1 / d` at 1 kHz, -2.61 at 2 kHz, -2.51 at 4 kHz. The wave band
is right: its direct sound is `FIELD_UNIT_AT_1M / d` within 0.1 dB in both
the field solved to 8 kHz and the pairs solved to 1500 Hz. The account to
the tenth is in section 2.

**2. Other errors of this type.** Eleven, in section 3. The four that can
be heard: the signature takes 1.8 to 5 dB of treble from everything above
the crossover (it copies the band limit of the 8 kHz grid into a pack that
is physical); a voice on its axis is 3.0 dB (1 kHz) to 6.3 dB (8 kHz) over
its clip's level while the band under the crossover stays at 0 dB; every
tail has a comb of 2.7 dB an octave; and the calibration in use lets a
bounce on the shell return 117 % of what it received.

**3. The split and the join.** The split is not a time: the images render
the specular paths of orders 1 to 3 inside 80 ms and the rays leave out
exactly those (-0.03 dB between the two over 80 ms, tracer's twin), so it
follows the room by construction and no time needs tuning to the room's
size. `tail_from_s` is not the split, it is a gate, and it is the one place
energy is lost: what the rays hold in the first 10 ms after the direct
sound is dropped, 0.3 to 3.1 dB of the reflections of those 10 ms. The join
in time is right at the direct sound (the two bands within 0.04 ms, the
masks adding to one within 0.01 dB) and 1.5 dB hot at 1 kHz from 10 to
20 ms, where it joins in power two bands that are still half coherent.
Section 4.

**4. The bottom of the spectrum.** The chain renders from 45 Hz: the wave
band's low cut is -1 dB at 43.7 Hz, -3 dB at 40 Hz, -10 dB at 35 Hz, and
nothing else removes level under 200 Hz. "Neither band renders under
80 Hz" is true of the free field stand-in alone. The washing machine is
quiet because 98 % of its clip's energy lies between 10 and 20 Hz. The
energy the owner saw round 20 Hz is the pressure zone of a sealed room, a
true answer of the model and not a dwelling's. Section 5.

## 1. The audit table

Errors are the chain over the truth, in dB, positive when the chain is
loud. "Changes validated output" means the field of ADR 0014 or a pack
already traced.

| link | compared with | measured | verdict |
| --- | --- | --- | --- |
| octave bank, bands summed in pressure | a unit pulse | within 0.02 dB from 354 Hz to 23.5 kHz; -0.19 dB at 177 Hz, -2.7 dB at 125 Hz, both under the crossover | PASS |
| `mirror.render.early_signals`, one path | `1 / d`, 500 Hz to 20 kHz | -0.06 dB at every frequency (the windowed sinc), on its sample | PASS |
| `FIELD_UNIT_AT_1M`, field solved to 8 kHz | the direct sound of the ten nearest points, times `d` | -0.02 to +0.08 dB at 1, 2 and 4 kHz | PASS |
| `fmax / 8000` scaling, pairs solved to 1500 Hz | the direct sound of 92 pairs under 0.9 m, times `d` | -0.02 dB at 500 Hz, -0.06 at 707 Hz, -0.09 at 800 Hz | PASS |
| the solve's own band limit | the same pairs | -0.45 dB at 1 kHz, -1.38 at 1189 Hz, -5.97 at 1414 Hz; 1.8 dB more at 1414 Hz than the fourth order filter run twice gives | PASS under the mask (0.2 dB heard); see D7 |
| the alignment's gain | a reference pulse of the same level | 0.0 dB for a pulse of one sample; -2.7 dB for a pulse spread over 0.5 ms; -2.35, -2.68, -2.61 dB at 1, 2, 4 kHz on the validated field, constant from 0.2 to 6 m | **D1** |
| the signature above the crossover | a unit source, flat | -1.8 dB at 8 kHz, -3.3 at 12 kHz, -5.0 at 16 kHz, re 1 kHz | **D2** |
| the two bands' direct sounds in time | each other | 0.00 ms in the median, -0.02 to +0.04 ms (126 points of the field); -0.03 ms on the scene's pairs under 2 m | PASS |
| the anchor of the pressure window | the direct sound | on it under 2 m; 6.8 ms after it in the median from 4 to 8 m | fixed in e93109ed, needs a trace |
| pressure masks in the window | a pulse both bands carry | 0.01 dB from 500 Hz to 2 kHz | PASS |
| power masks after the window | a reflection both bands carry | +3.0 dB at 1 kHz if wholly coherent; measured coherence 0.77, 0.41, 0.19 at 5-10, 10-20, 20-50 ms: +1.5 dB at 1 kHz from 10 to 20 ms, +0.75 dB from 20 to 50 ms | **D6** |
| `seam_db` of two bands on one scale | 0 dB | -0.9 dB on a synthetic pair, -1.07 dB on 92 pairs | **D7** |
| N3D and ACN, channel 0 and the 64 channels | a unit plane wave from six axes | 1 and 64, to 1e-9 | PASS |
| the binaural decoder, both heads | a unit plane wave | sphere: -0.18 to +0.22 dB under 300 Hz; measured head (HRIR_L2702): +0.5 dB at 125 Hz, +0.1 at 250 Hz, diffuse field within +1.6 and -0.6 dB to 8 kHz; left is left | PASS: full scale at the ear is full scale on channel 0, 86 dB SPL at 1 m |
| the image sources of a shoebox | pyroomacoustics (`tests/test_mirror_ism.py`) | positions to 1e-5 m, one path an image, the oracle's factor | PASS (already pinned) |
| the receiver sphere | the cap a sphere subtends | the tracer: -0.01 dB at 0.25 m to +-0.3 dB at 1.5 m (40 000 rays); the renderer's `r^2 / 4 d^2`: +0.97 dB at 0.25 m, +0.18 at 0.5 m, +0.04 at 1 m | **D9** |
| the split: images of orders 1 to 3 plus the rays that skip them | rays that skip nothing | -0.03 dB over 0.5 to 80 ms (box of 9 x 6 x 2.6 m, 3 m, tracer's twin, 6000 rays) | PASS |
| the flutter images of orders 4 to 6 | the same | in the images and in the rays: 0.5 to 3.8 % of the images' energy | **D10**, 0.02 to 0.16 dB |
| the tail's first 10 ms | the first bounce's Lambert part, integrated over the walls | dropped: -0.3 to -1.5 dB on the reflections of those 10 ms at scattering 0.2, -0.9 to -3.1 dB at 0.4; the twin's rays -0.72 dB where the integral gives -0.74 | **D5** |
| a bounce under the calibration in use | `1 - alpha` | the shell at 1 kHz: scattered by the rays 0.33, mirrored by the images 0.85, of 0.82 kept | **D8** |
| the tail's energy | its histogram's | -0.19 dB over 707 Hz to 11.3 kHz | PASS |
| the tail's colour | a flat histogram | +1.0 dB at the bands' centres, -1.7 dB at their edges | **D4** |
| the mirror against the wave field, 10 to 50 ms | the wave field to 8 kHz, each side re its own direct sound | +0.8 dB at 1 kHz, +0.4 at 2 kHz, +0.2 at 4 kHz | PASS within 1 dB |
| the mirror against the wave field, 50 to 400 ms | the same | +0.6 dB at 1 kHz, 0.0 at 2 kHz, -1.0 at 4 kHz, no slope with distance | PASS within 1 dB |
| the wave band's low cut | a plain integrator | -0.14 dB at 50 Hz, -3.0 at 40 Hz, -17 at 31.5 Hz, -49 at 20 Hz | PASS: whole from 50 Hz |
| the free field stand-in under 160 Hz | the solve's band | nothing under 80 Hz, -6 dB at 120 Hz | **D11** |
| air absorption | once a band | the low band at the trace (`pair_low`), the early part and the tail in the engine, the cache before its air | PASS by reading |
| a voice on its axis | the clip's level | +0.96, +1.41, +2.26, +3.04, +4.08, +5.26, +6.25 dB from 125 Hz to 8 kHz; the low band 0 dB | **D3**, the owner's |

Not audited: the engine's own arithmetic between steps (L20 is rebuilding
it), the compact form's 16 bits, the 5 ms clip fades, the page's gain.

## 2. The account of the 2 dB

The mirror renders a unit source as `1 / d` (-0.06 dB). It is then
multiplied by the alignment's gain and convolved with the signature. On
the physical scale of a pack:

| | at 1 kHz | over 707 to 1414 Hz |
| --- | --- | --- |
| `alignment_gain` over `FIELD_UNIT_AT_1M` | -5.22 dB | -5.22 dB |
| the signature, a filter of unit energy | +2.85 dB | +2.77 dB |
| the mirror's direct sound re `1 / d` | **-2.36 dB** | **-2.45 dB** |

A gain that put the mirror on the wave field would be -2.85 dB, the
signature's own level undone. It is -5.22 dB because of what the window
holds of each pulse:

| | inside 0.5 ms | inside 4 ms |
| --- | --- | --- |
| the wave field's direct pulse, ten nearest points, re a unit pulse | -5.0 to -6.1 dB | -1.6 to -3.5 dB |
| the mirror's, signature applied | -0.3 dB | -0.2 dB |

The wave field's pulse is not the signature's pulse: the grid delays the
top of its band, the signature has the same magnitude and minimum phase.
The reference loses 2.4 dB of its energy to the window and the mirror
loses none.

**On the validated field** (the wave field to 8 kHz against
`field_mirror_c10`, octave of 1 kHz, 126 points with a direct path):

| part of the response | wave over mirror | of which |
| --- | --- | --- |
| the direct sound, +-1.5 ms | +2.35 dB, from 0.2 to 6 m | the gain: 2.4 dB |
| 1.5 to 10 ms | +2.90 dB | the gain, and 0.5 dB of the gate (D5) |
| 10 to 50 ms | +1.52 dB | the gain, less 0.8 dB the mirror is hot by |
| 50 to 400 ms | +1.76 dB | the gain, less 0.6 dB |
| the whole response | **+2.09 dB** | |
| the same, 47 points without a direct path | +3.02 dB | |

which is the field's median seam of +2.3 dB. It is a constant with
distance and with time: a normalisation, not a physical model.

**On the first scene** (pairs solved to 1500 Hz; median +1.90 dB, +1.78 on
the 9669 pairs with a direct path, +2.11 on the 8550 without):

| term | dB | where |
| --- | --- | --- |
| the mirror's direct sound under `1 / d` | +2.45 | `mirror.files.align_to_reference`, carried by `trace.assets.MirrorAssets.pack_gain` |
| the seam reads the solve's band limit as a level | -1.07 | `mirror.hybrid.seam_db`: one weight for every frequency from 707 to 1414 Hz, the wave side before its mask (D7) |
| **predicted for a pair that is its direct sound** | **+1.38** | |
| measured, 255 pairs under 1 m | +1.65 | |
| measured, every pair with a direct path | +1.78 | |
| left, the reverberant part | +0.3 to +0.4 | not measured to the tenth: see below |

The 0.3 to 0.4 dB left are not located by a measurement. Two causes are
known and neither could be separated on the laptop: the gate (0.5 dB on the
first 10 ms), and the calibration, which was fitted on fields traced with
`coincident_facets = "twice"` (every reflection on a sheet 6 dB loud) and
is used by a scene traced with `"once"`, so the scene's images hold less
than the fit balanced. A render of the mirror at 200 pairs of the scene,
part by part, would settle it (section 7).

**What is heard today.** The seam raises the mirror by 1.9 dB in the
median where it is 2.45 dB low: above the crossover a scene is about
0.6 dB under the band below it, and each pair is corrected by its own
number, from +0.4 to +3.4 dB between the deciles, which moves as things
move. A pack without a low band has no seam and is 2.4 dB low.

**The one fix.** In a pack nothing needs to be fitted: the mirror is
physical before the gain and the wave band is physical after
`FIELD_UNIT_AT_1M`. `alignment_gain` becomes 1 and the signature a unit
pulse (which removes D2 as well); `lead_s` stays, it is a clock and was
never fitted. The static fields keep their alignment, or have it measured
on a band and not on a window. With the seam's reading made unbiased (D7,
L21's), the median of the seams is then the reverberant residue alone,
about 0.3 dB, and what is left of a pair's seam is that pair's physics.

## 3. The defects

Ranked by what can be heard. "Fix changes" says whether a remedy moves
output that was validated.

| | defect | where | size | fix changes | whose |
| --- | --- | --- | --- | --- | --- |
| D1 | the mirror is 2.4 dB under the wave band | `mirror/files.py`, `align_to_reference`, the window of `DIRECT_WINDOW_S` (`mirror/direct.py:29`); carried by `trace/assets.py:73` `pack_gain` | -2.36 dB at 1 kHz, -2.6 dB at 2 and 4 kHz, every pair | every pack; no field if only `pack_gain` moves | L21 with the levelling |
| D2 | the mirror's treble is the 8 kHz grid's | `mirror/direct.py`, `measure_signature`; applied by `render/dry.py`, `DryTrack.high`, from `render/engine.py:89` | -1.8 dB at 8 kHz, -3.3 at 12 kHz, -5.0 at 16 kHz re 1 kHz, on every source | every pack, brighter | L20 and L21; the owner hears it |
| D3 | a voice's axis is over its clip | `mirror/directivity.py`, `model` (unit mean power) | on the axis +3.0 dB at 1 kHz to +6.3 dB at 8 kHz, behind -4.0 to -11.8 dB, and 0 dB under the crossover: a step of 3 dB at 1 kHz for a talker who faces the listener | every voice | the owner (section 6) |
| D4 | the tail is a comb | `mirror/render.py`, `tail_from_histogram`: one noise a band through a bank whose bands add in pressure, raised by `bank_reading`'s inverse; the same in `render/tail.py` | +1.0 dB at 1, 2, 4, 8 kHz and -1.7 dB at 1.4, 2.8, 5.7 kHz | every tail, by 2.7 dB a half octave | L20 |
| D5 | the tail's first 10 ms are dropped | `mirror/render.py`, `tail_from_histogram`, `from_bin`; `RenderSettings.tail_from_s` | the reflections of the first 10 ms 0.3 to 3.1 dB short; 0.1 to 0.2 dB of the whole response | the early part of every pair; the calibration | L20, after the owner's choice (section 4) |
| D6 | a reflection both bands carry is joined in power | `mirror/hybrid.py`, `Crossover.coherent_s` and `coherent_fade_s` | +1.5 dB at 1 kHz from 10 to 20 ms, +0.75 dB from 20 to 50 ms, in the crossover's octave | the join | L21: a narrower overlap makes it smaller |
| D7 | the seam reads a band limit as a level | `mirror/hybrid.py`, `seam_db`; `trace/level.py`, `pair_seam_db` | the scene's seams 1.07 dB low | every seam | L21 |
| D8 | a bounce returns more than the wall keeps | `mirror/parameters.py`, `image_scene` against `apply_parameters`: `shell_scattering` for the rays, the class's for the images, and two absorptions | the shell under c3cec6aab28bb582: images 2.4 dB a bounce over what the rays left them at 1 kHz (1.0 dB at 125 Hz, 2.7 dB at 8 kHz); 117 % returned | the calibration | the owner, then one fit |
| D9 | the tail's scale is the small angle's | `mirror/render.py:484`, `trace/level.py:260`, the engine's tail | the tail +0.97 dB at 0.25 m, +0.18 dB at 0.5 m | sources nearer than 0.5 m | L20 |
| D10 | flutter images are also rays | `mirror/pipeline.py`, `traced_rays`: `skip_specular_order = max_order`, not `flutter_order` | 0.02 to 0.16 dB | no | whoever touches the rays next |
| D11 | the free field stand-in starts at 80 Hz | `trace/engines.py:342-349`, `FreeFieldPairs.response`; `render/pack.py:1109`, `SYNTHETIC_HIGHPASS_HZ` | nothing under 80 Hz, -6 dB at 120 Hz, where a solve is whole from 50 Hz | synthetic packs only | L20 |
| D12 | a clip levelled on what cannot be heard | `noise/appliance_washing_machine`: -0.1 dB of its level between 10 and 20 Hz, -17.5 dB over 20 Hz, -24.7 dB over 40 Hz | the clip 25 dB under its stated level once rendered | the library | the library's lot |

Two smaller things read in passing. The calibration in use has
`image_absorption_scale` at its floor of 0.05 at 8 kHz and under 0.4 from
500 Hz up: the fit asks the images for walls that absorb almost nothing,
which is what D8 looks like from inside the fit. And `trace.level.mirror_omni`
renders the mirror with the default atmosphere while the wave side takes
the recipe's: nothing at 1 kHz, and a difference if the seam is ever read
higher.

## 4. The split between the images and the rays, and the join

**The design.** An arrival belongs to the images if it is specular on
reflector facets, of order 1 to 3, with at most one bounce on furniture,
and arrives inside 80 ms; the rays count every crossing but those
(`RaySettings.skip_specular_order`, `skip_window_s`, set from the tree by
`MirrorSettings.traced_rays`). The two overlap in time from the first
reflection to 80 ms and never in content. So the question "is a fixed time
right for a small room and a large one" has no object: by the lattice of a
shoebox's images, at absorption 0.2,

| box, distance | first order-4 image | last order-3 image | share of the specular energy the images hold: 0-10, 10-20, 20-30, 30-50 ms |
| --- | --- | --- | --- |
| 4 x 3 x 2.5 m, 1 m | +15.8 ms | +33.8 ms | 100, 85, 10, 1 % |
| 4 x 3 x 2.5 m, 3 m | +9.7 ms | +26.3 ms | 98, 26, 7, 0 % |
| 9 x 6 x 2.6 m, 1 m | +25.9 ms | +75.8 ms | 100, 100, 87, 22 % |
| 9 x 6 x 2.6 m, 3 m | +18.6 ms | +75.7 ms | 100, 89, 54, 16 % |
| 9 x 6 x 2.6 m, 6 m | +14.2 ms | +61.5 ms | 100, 72, 17, 2 % |

the hand-over happens where the room puts it, 10 to 20 ms in the small
room and 20 to 50 ms in the large one, without a number that says so. The
80 ms window holds every order-3 image of both but one, 0.1 % of their
energy.

**The loss.** `tail_from_s = 10 ms` silences the tail until 10 ms after
the direct sound. It exists because the direct rays are in the histogram
(they were its scale) and a noise that starts on the direct sound smears
it. What it drops is not the direct rays alone: everything scattered at
the first bounce, and everything off a surface the tree does not mirror.
The first bounce's Lambert part, integrated over the six walls
(absorption 0.2):

| box, distance | scattering 0.2: dropped re the direct sound, and the first 10 ms of reflections | scattering 0.4, the shell's in use |
| --- | --- | --- |
| 4 x 3 x 2.5 m, 1 m | -8.7 dB, -1.0 dB | -5.6 dB, -2.4 dB |
| 4 x 3 x 2.5 m, 3 m | -2.2 dB, -0.4 dB | +0.8 dB, -1.3 dB |
| 9 x 6 x 2.6 m, 1 m | -11.9 dB, -1.5 dB | -8.8 dB, -3.1 dB |
| 9 x 6 x 2.6 m, 3 m | -6.0 dB, -0.7 dB | -3.0 dB, -1.9 dB |
| 9 x 6 x 2.6 m, 6 m | -5.1 dB, -0.3 dB | -2.1 dB, -0.9 dB |

and on the validated field the mirror is 0.5 dB short of the wave field
from 1.5 to 10 ms in the median, 4 to 5 dB at the ninth decile.

**What should change.** Three things, in this order.

1. The direct rays leave the histogram (the tracer counts crossings from
   the first bounce on; the scale is the sphere's cap, which the tracer
   reproduces to 0.05 dB, D9), and the tail starts at the direct sound.
   `tail_from_s` goes to zero and is no longer a parameter of the physics.
2. Before that, D8 is decided. Under the calibration in use the rays
   scatter 40 % of a bounce on the shell and the images still render 90 %
   of it: the scattered part of the first bounces is in good part energy
   that the images already hold. The gate hides some of that, which is why
   the mirror is only 0.5 dB short where the integral says 1 to 3 dB.
   Opening the gate without making the two shares add to one would make
   the first 10 ms too loud.
3. The noise that carries the first milliseconds is 24 bursts a bin of
   2 ms. Whether a first bounce's scattered part may be a noise is the
   owner's ear to judge; if not, it is the images' first order that takes
   it back (their factor without the scattering's loss).

**The join in time.** At the direct sound the two bands coincide (0.00 ms
in the median, 0.04 ms at the ninth decile) and the pressure masks add to
one (0.01 dB). The mirror's lead of 512 samples covers the 256 the zero
phase bank rings before a pulse. The window is `coherent_s = 5 ms` then a fade of 5 ms, and
the coherence of the two bands over 707 to 1414 Hz, measured on 63 points
of the validated field, is 0.96 at the direct sound, 0.86 from 1.5 to
5 ms, 0.77 from 5 to 10 ms, 0.41 from 10 to 20 ms, 0.19 from 20 to 50 ms
and nothing after. A pressure join is right to 10 ms and a power join from
50 ms; between the two the truth is neither, and the power join is 1.5 dB
hot at 1 kHz from 10 to 20 ms. No regime mismatch of a fixed 3 dB was
found anywhere: no path is attenuated twice. The remedy is the width of
the overlap, not the window: the error is the product of the two masks,
and a join tapered over a third of an octave has a third of it.

## 5. The bottom of the spectrum

**What the low chain passes.** One filter acts under 200 Hz: the fit's
integrator and low cut (`accel.pairs.LOWCUT_HZ, LOWCUT_ORDER = 40.0, 8`,
designed by `accel.dsp.lowcut_sos`). Against a plain integrator:

| Hz | 20 | 25 | 31.5 | 40 | 50 | 63 | 80 to 200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dB | -49.0 | -33.3 | -17.2 | -3.6 | -0.14 | 0.00 | 0.00 |

-1 dB at 43.7 Hz, -3 dB at 40.1 Hz, -10 dB at 35.0 Hz. The decimation to
4 kHz keeps the bins, the low masks are one under 707 Hz, the compact form
keeps degree 0 whole, and the decoder is +0.5 dB at 125 Hz. On the first
scene's cache (120 pairs at 0.5 to 1.5 m, 120 at 3 to 6 m, mean power of
channel 0 a third octave, re the free field level): from the 40 Hz third
octave to 1 kHz the level stays within the room's own 0 to +5 dB (near)
and +9 to +13 dB (far); the 31.5 Hz third octave is 8 to 10 dB under its
neighbour, 25 Hz 27 dB, 20 Hz 40 dB. There is no step at 80 Hz. The mirror
band's own low cut (the same 40 Hz, `render/dry.py`) acts on a band the
high masks have already emptied under 707 Hz.

**Where "80 Hz" came from.** `trace.engines.FreeFieldPairs` and
`render.pack.synthetic_free_field` keep nothing under 80 Hz and raise a
cosine to 160 Hz (D11). A level read on a free field pack is the
stand-in's.

**The washing machine.** Its clip holds 97.7 % of its energy between 10
and 20 Hz, where no one hears and no microphone of a dwelling is trusted:
-17.5 dB of its level over 20 Hz, -24.4 dB over 31.5 Hz, -25.0 dB over
80 Hz. Through the chain's low cut it is -24.6 dB, which is the 25 dB the
sound check found. A render down to 20 Hz would give it 7 dB back, all of
it between 20 and 31.5 Hz at about 40 dB SPL, 20 to 35 dB under the threshold
of hearing there. The remedy is the clip's: its level read over 40 Hz, or
A-weighted.

**What the energy round 20 Hz was.** The solver's unit source is a step of
volume velocity. In free air its pressure is a pulse, `1 / (4 pi d)`. In a
closed room the volume it goes on giving has nowhere to go: the wave
equation integrated over the room gives a mean pressure that climbs by
`c^2 / V` a second, under the room's first resonance, until what the walls
absorb balances it. On the CPU twin, a rigid box of 25 litres: the plain
integral of the record stands at 8 to 11 after 55 to 110 ms, alike at the
three receivers, where the direct pulse is under 1 for one step; its slope
is 145 a second where `c^2 / V` gives 175; the low cut leaves a mean of
5e-5 (`test_a_sealed_room_keeps_the_volume_its_source_gave_it`). In the
frequency domain it is a response that rises 12 dB an octave towards zero
under the first mode, 14 Hz for a dwelling 12 m long, with the first modes
on top of it between 15 and 30 Hz, hardly damped because the materials are
fitted from 125 Hz up. That is the band the owner saw.

It is not a defect of the scheme, of the hard source or of a boundary: it
is the sealed room's own answer, the cabin gain of a car. It is not a
dwelling's: a dwelling leaks under its doors and through its ventilation,
and its windows and light walls give, which caps the pressure zone and
damps the first modes; the model has neither. The low cut hides a missing
piece of physics, not an error, and a zero-mean excitation would not
remove it: the source is not at fault.

**A render valid to 20 Hz would need** a low cut at 20 Hz of low order
(order 2 at 15 Hz keeps the washing machine within 2.2 dB, order 8 at
20 Hz within 11.7 dB); a leak and a wall compliance in the model, without
which 20 to 40 Hz is the sealed room's and too loud; absorption under
125 Hz, which the catalogue does not hold; and a longer window than 1.2 s,
since a mode at 20 Hz with the present walls rings longer. Nothing of the
grid changes. What would stay untrustworthy is the level of everything
under the first mode of each room, which depends on what is open. The
recommendation is to leave the cut where it is until a leak is modelled,
and to say in `scene-pack.md` that a pack is valid from 45 Hz.

## 6. For the owner

1. **The direct sound of a voice.** A clip is recorded in front of the
   talker, so it is the axis. The directivity is normalised to unit mean
   power, which puts the axis 3 to 6 dB over the clip above the crossover
   and leaves the band below at the clip's level. Normalised on the axis
   instead, a voice that faces the listener is its clip and the tail
   falls by the directivity index (3.0 dB at 1 kHz), but the band under
   the crossover stays omnidirectional and is then 1 to 2 dB loud. Either
   way the two bands disagree until the wave band has a directive source.
2. **The mirror's treble.** Removing the signature from a pack makes
   everything above 6 kHz brighter by 2 to 5 dB. It is the truth of a unit
   source; it is not what was listened to so far.
3. **Images and rays on the shell (D8).** One scattering for both, or the
   present two with the rays' share cut to what the images do not render.
   Then one fit of the calibration.
4. **The first 10 ms of the tail**: noise, or the images' own.
5. **20 Hz**: leave the cut at 40 Hz until the model has a leak.

## 7. The order of the work, and what could not be measured

The order matters because each step changes what the next one fits.

1. Normalisation: `pack_gain` to 1 and no signature in a pack (D1, D2).
   The calibration reads every level against the same response's own
   direct sound, band by band, so this step moves none of its numbers.
2. The seam's reading without its bias (D7) and the tapered join (D6): L21.
3. The decisions of section 6, then D4, D5, D9 and D10 in the mirror and
   the engine.
4. One fit of the calibration, on fields traced with
   `coincident_facets = "once"` and with the shares of step 3: the present
   one was fitted on doubled sheet reflections and on the gate.
5. Then the seams' median is zero within the reverberant residue, and a
   seam that is not is a pair to look at.

Not measured, and what would settle each:

| question | what it needs | cost |
| --- | --- | --- |
| the 0.3 to 0.4 dB of the scene's seam that are not on the direct sound | `mirror_omni` at 200 pairs of the scene, its parts apart, against the pairs' own responses, on the bundle the scene was traced from | one card for a quarter of an hour, under 1 USD |
| the reverberant level and T60 against diffuse field theory in a shoebox, per band | the card's tracer at 1e6 rays on three boxes; the twin takes 35 s for 6000 rays | minutes of one card |
| the mirror against the wave field in a room that was not calibrated on | the wave field to 8 kHz from a second source of hssd_0076 | the cost of a field, about 30 USD by the last one |
| the pressure zone of the dwelling itself | one solve of 3 s without the low cut, the raw record kept | one pair's solve, cents |
| the engine between steps, the stems' sum, the compact form | left to L20's own tests on the rebuilt engine | none |
