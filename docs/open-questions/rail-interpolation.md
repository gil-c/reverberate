# How far apart a moving source's solved positions may be

Date: 2026-10-05

Status: measured on the dense line and implemented as an option whose
default changes nothing. Written for lot L11 of ADR 0016. The weights are
`reverberate.spatial.rail`; the measurement is `python -m
reverberate.experiments.w44_interpolation rail`, run on the dense line of
hssd_0076 (`data/runs/w44_clarify_interpolation/line_home2/pulled/field/S1.h5`,
341 points 2 cm apart, source S1) with that run's `plan.json`; the option is
`python -m reverberate.trace ... --rail-positions 8` on a recipe generated
at another `rails.pitch_m`.

## The question

A twenty minute scene costs 7.54 USD to trace and 7.07 of it is the low
band's wave solves, one for every position a source is heard from: 1529
(only 6 % of the audible steps are a source on the move, but a rail is
many positions and a station one),
because a rail is solved every 8 cm and the engine reads the two positions
either side of the source linearly. The 8 cm came from that linear reading.
The sampling theorem asks for less: at a frequency `f` the pressure over
space holds no wavenumber above `2 pi f / c`, so positions half a
wavelength apart determine it, 17 cm at 1 kHz and 12.1 cm at 1414 Hz, where
the crossover's low mask ends.

## What is read, and how the error is counted

By reciprocity the pressure a fixed source leaves along a line is what a
source moving along that line leaves at a fixed omnidirectional listener.
Some points of the line are taken as the rail's solved positions and the
others are predicted from them. Channel 0 alone is a reciprocal test.

- **The centres are the arrays' own** (`low-band-translation.md`). Under
  707 Hz the field holds its low solve, whose 341 arrays stand on 209
  distinct nodes 32.7 mm apart: the third octaves from 100 to 630 Hz are
  read there, at pitches that are multiples of 32.7 mm. From 891 Hz it
  holds its mid solve, 2 cm apart to within 4 mm: the third octaves at
  1 kHz and 1250 Hz are read there, at multiples of 2 cm, each position
  taken where its array stood: a nominal 12 cm there is gaps of 11.4 to
  12.3 cm, and 12.3 cm is half a wavelength at 1400 Hz, the top edge of
  the 1250 Hz third octave. **The 800 Hz third octave lies across the seam
  of the two solves**
  and reads -13 to -16 dB at every pitch, 4 cm included: it is the seam and
  is given for completeness only.
- **The error** is the energy of the difference over the energy of the
  solved response, per third octave, on the 50 ms after the onset and on
  the whole response. `masked` is the same with the crossover's low mask
  on the error alone (the pressure mask on the early part, the power mask
  on the whole): the error over what is heard in that band once the mirror
  has given the rest. The mask is 6 dB down at 1 kHz and 22 dB down at
  1250 Hz in pressure.
- **The floor.** At the densest pitch (6.5 cm under the seam, 4 cm over it)
  the best interpolators read -36 dB at worst under the seam and -44 dB
  over it. Under the seam that is the scalar a three band field levels
  each point's low band with (-45 dB in the median, -31 dB at worst,
  `low-band-translation.md`), not an interpolation error; a campaign that
  solves the low band alone has no such scalar.
- **Cases.** `interior`: the whole line as one rail, targets with four
  positions or more either side (330 and 915 far cases at 13 and
  12 cm). `end`: rails 1.2 m long cut from the line, sampled as
  `scenes.rail_samples` does, targets in the first or the last gap, read
  from one side only past the two round them (36 and 114 far cases).
  `near`: the target is less than 1.2 m from the source, which the line
  passes at 0.31 m. The worst case over the targets is what is read; the
  median and the ninth decile are given beside it.

## The interpolators

`linear` is the pack's first rule. `Lagrange` is the polynomial through 4
or 6 positions. `sinc` is a sinc at the pitch under a raised cosine over 4,
6 or 8. `band limited` is `spatial.rail.band_limited_weights` over 4, 6 or
8: the weights that leave the least error on a field whose spectrum over
space is flat inside `|k| <= 1.05 * 2 pi f / c`, so they change with
frequency; `one weight` is the same made once, for 1414 Hz. Early part,
interior, far from the source, worst case, dB.

Under the seam (the low solve):

| pitch | read | 125 | 250 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -37.0 | -35.0 | -29.4 | -27.6 | -23.3 |
| 6.5 cm | Lagrange 3, 4 | -36.8 | -36.7 | -37.0 | -37.1 | -36.8 |
| 6.5 cm | Lagrange 5, 6 | -37.1 | -36.9 | -37.1 | -37.0 | -37.0 |
| 6.5 cm | sinc, 4 | -36.9 | -37.4 | -34.1 | -31.6 | -27.5 |
| 6.5 cm | sinc, 6 | -36.9 | -36.0 | -34.9 | -34.5 | -33.2 |
| 6.5 cm | sinc, 8 | -37.5 | -38.1 | -37.6 | -37.0 | -36.3 |
| 6.5 cm | band limited, one weight, 8 | -37.6 | -37.3 | -37.4 | -37.2 | -37.1 |
| 6.5 cm | band limited, 4 | -36.3 | -37.1 | -37.9 | -37.5 | -36.4 |
| 6.5 cm | band limited, 6 | -37.3 | -37.2 | -37.4 | -37.2 | -37.0 |
| 6.5 cm | **band limited, 8** | -35.8 | -36.4 | -37.2 | -37.0 | -36.7 |
| 9.8 cm | linear, 2 | -34.8 | -30.2 | -23.6 | -22.3 | -17.8 |
| 9.8 cm | Lagrange 3, 4 | -34.4 | -34.5 | -35.0 | -33.9 | -28.6 |
| 9.8 cm | Lagrange 5, 6 | -34.6 | -34.5 | -34.4 | -34.6 | -34.4 |
| 9.8 cm | sinc, 4 | -35.0 | -33.9 | -29.1 | -26.9 | -22.0 |
| 9.8 cm | sinc, 6 | -34.3 | -33.2 | -31.5 | -31.5 | -30.8 |
| 9.8 cm | sinc, 8 | -35.0 | -35.4 | -34.3 | -34.6 | -34.8 |
| 9.8 cm | band limited, one weight, 8 | -28.4 | -30.4 | -27.0 | -30.3 | -31.2 |
| 9.8 cm | band limited, 4 | -34.9 | -35.0 | -33.5 | -33.8 | -34.2 |
| 9.8 cm | band limited, 6 | -34.6 | -34.6 | -34.4 | -34.3 | -34.4 |
| 9.8 cm | **band limited, 8** | -33.9 | -34.6 | -34.5 | -33.8 | -33.7 |
| 13.1 cm | linear, 2 | -31.6 | -25.5 | -17.9 | -16.7 | -12.2 |
| 13.1 cm | Lagrange 3, 4 | -32.6 | -31.7 | -30.5 | -26.3 | -20.1 |
| 13.1 cm | Lagrange 5, 6 | -32.8 | -32.0 | -32.7 | -32.1 | -26.0 |
| 13.1 cm | sinc, 4 | -32.4 | -29.1 | -22.3 | -20.2 | -15.2 |
| 13.1 cm | sinc, 6 | -32.1 | -29.5 | -28.5 | -28.5 | -28.9 |
| 13.1 cm | sinc, 8 | -33.5 | -33.6 | -32.9 | -33.1 | -31.2 |
| 13.1 cm | band limited, one weight, 8 | -25.2 | -26.6 | -23.8 | -28.2 | -23.9 |
| 13.1 cm | band limited, 4 | -33.6 | -31.0 | -31.2 | -31.8 | -28.5 |
| 13.1 cm | band limited, 6 | -32.9 | -32.3 | -32.4 | -32.5 | -31.5 |
| 13.1 cm | **band limited, 8** | -31.3 | -32.0 | -31.3 | -31.5 | -31.4 |
| 16.3 cm | linear, 2 | -28.9 | -22.0 | -14.3 | -13.3 | -8.9 |
| 16.3 cm | Lagrange 3, 4 | -29.2 | -28.9 | -25.7 | -20.3 | -14.2 |
| 16.3 cm | Lagrange 5, 6 | -29.2 | -28.6 | -30.5 | -25.7 | -18.5 |
| 16.3 cm | sinc, 4 | -29.2 | -25.6 | -18.3 | -16.3 | -11.2 |
| 16.3 cm | sinc, 6 | -29.1 | -26.3 | -26.3 | -26.4 | -20.4 |
| 16.3 cm | sinc, 8 | -29.3 | -29.9 | -29.3 | -28.7 | -28.2 |
| 16.3 cm | band limited, one weight, 8 | -18.4 | -16.3 | -17.1 | -11.4 | -4.7 |
| 16.3 cm | band limited, 4 | -30.4 | -27.5 | -30.0 | -26.3 | -22.0 |
| 16.3 cm | band limited, 6 | -29.2 | -28.3 | -29.3 | -29.3 | -28.1 |
| 16.3 cm | **band limited, 8** | -28.5 | -28.8 | -28.9 | -29.4 | -28.9 |
| 19.6 cm | linear, 2 | -27.7 | -18.8 | -10.9 | -10.3 | -5.9 |
| 19.6 cm | Lagrange 3, 4 | -27.8 | -28.0 | -19.7 | -15.2 | -9.1 |
| 19.6 cm | Lagrange 5, 6 | -27.6 | -27.4 | -26.6 | -19.5 | -11.8 |
| 19.6 cm | sinc, 4 | -28.1 | -22.3 | -14.2 | -12.4 | -7.3 |
| 19.6 cm | sinc, 6 | -27.6 | -24.7 | -26.7 | -20.6 | -12.4 |
| 19.6 cm | sinc, 8 | -27.5 | -27.0 | -27.2 | -27.3 | -17.6 |
| 19.6 cm | band limited, one weight, 8 | -10.9 | -4.2 | 1.6 | -0.6 | -0.3 |
| 19.6 cm | band limited, 4 | -28.0 | -26.2 | -27.5 | -21.0 | -14.4 |
| 19.6 cm | band limited, 6 | -27.2 | -26.7 | -28.1 | -27.4 | -21.3 |
| 19.6 cm | **band limited, 8** | -27.2 | -27.2 | -28.4 | -27.9 | -25.8 |
| 26.1 cm | linear, 2 | -23.6 | -15.1 | -6.3 | -5.7 | -1.7 |
| 26.1 cm | Lagrange 3, 4 | -25.7 | -23.9 | -11.0 | -7.4 | -2.1 |
| 26.1 cm | Lagrange 5, 6 | -25.6 | -24.4 | -15.5 | -9.0 | -2.5 |
| 26.1 cm | sinc, 4 | -25.1 | -18.5 | -8.3 | -6.6 | -1.9 |
| 26.1 cm | sinc, 6 | -25.4 | -21.4 | -16.7 | -9.0 | -2.5 |
| 26.1 cm | sinc, 8 | -25.5 | -25.4 | -24.9 | -12.4 | -3.0 |
| 26.1 cm | band limited, one weight, 8 | 2.8 | 1.9 | 1.4 | 1.4 | 1.0 |
| 26.1 cm | band limited, 4 | -26.4 | -23.0 | -18.6 | -9.8 | -2.2 |
| 26.1 cm | band limited, 6 | -25.6 | -24.1 | -25.6 | -16.8 | -3.5 |
| 26.1 cm | **band limited, 8** | -25.4 | -24.0 | -25.8 | -20.1 | -4.6 |
| 32.7 cm | linear, 2 | -21.3 | -11.6 | -2.8 | -3.0 | 1.2 |
| 32.7 cm | Lagrange 3, 4 | -25.0 | -18.8 | -4.9 | -4.0 | 2.1 |
| 32.7 cm | Lagrange 5, 6 | -25.3 | -23.2 | -7.0 | -4.2 | 2.7 |
| 32.7 cm | sinc, 4 | -23.1 | -14.4 | -3.8 | -3.5 | 1.7 |
| 32.7 cm | sinc, 6 | -24.7 | -24.2 | -7.1 | -4.3 | 2.8 |
| 32.7 cm | sinc, 8 | -24.6 | -25.0 | -11.7 | -4.5 | 3.5 |
| 32.7 cm | band limited, one weight, 8 | 2.6 | 2.6 | 2.6 | 2.5 | 1.8 |
| 32.7 cm | band limited, 4 | -25.4 | -23.9 | -8.7 | -4.1 | 1.5 |
| 32.7 cm | band limited, 6 | -25.0 | -25.3 | -16.1 | -4.7 | 1.1 |
| 32.7 cm | **band limited, 8** | -25.3 | -24.9 | -21.8 | -5.2 | 1.0 |

Over it (the mid solve):

| pitch | read | 1000 | 1250 |
| --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -12.5 | -9.6 |
| 8.0 cm | Lagrange 3, 4 | -21.4 | -15.2 |
| 8.0 cm | Lagrange 5, 6 | -29.4 | -20.1 |
| 8.0 cm | sinc, 4 | -15.3 | -11.6 |
| 8.0 cm | sinc, 6 | -24.5 | -20.7 |
| 8.0 cm | sinc, 8 | -24.5 | -22.7 |
| 8.0 cm | band limited, one weight, 8 | -41.2 | -38.1 |
| 8.0 cm | band limited, 4 | -31.6 | -22.3 |
| 8.0 cm | band limited, 6 | -41.6 | -36.4 |
| 8.0 cm | **band limited, 8** | -42.0 | -40.8 |
| 10.0 cm | linear, 2 | -9.1 | -6.0 |
| 10.0 cm | Lagrange 3, 4 | -15.0 | -9.0 |
| 10.0 cm | Lagrange 5, 6 | -20.2 | -11.4 |
| 10.0 cm | sinc, 4 | -11.4 | -7.2 |
| 10.0 cm | sinc, 6 | -19.5 | -11.2 |
| 10.0 cm | sinc, 8 | -23.0 | -15.1 |
| 10.0 cm | band limited, one weight, 8 | -27.4 | -23.5 |
| 10.0 cm | band limited, 4 | -23.2 | -13.1 |
| 10.0 cm | band limited, 6 | -36.2 | -20.7 |
| 10.0 cm | **band limited, 8** | -38.8 | -31.0 |
| 12.0 cm | linear, 2 | -6.4 | -3.4 |
| 12.0 cm | Lagrange 3, 4 | -10.0 | -4.6 |
| 12.0 cm | Lagrange 5, 6 | -13.1 | -5.4 |
| 12.0 cm | sinc, 4 | -8.0 | -3.9 |
| 12.0 cm | sinc, 6 | -13.5 | -5.3 |
| 12.0 cm | sinc, 8 | -18.7 | -6.4 |
| 12.0 cm | band limited, one weight, 8 | -16.7 | -9.3 |
| 12.0 cm | band limited, 4 | -15.5 | -5.8 |
| 12.0 cm | band limited, 6 | -24.7 | -8.0 |
| 12.0 cm | **band limited, 8** | -31.2 | -9.6 |
| 14.0 cm | linear, 2 | -3.9 | -1.0 |
| 14.0 cm | Lagrange 3, 4 | -5.9 | -0.9 |
| 14.0 cm | Lagrange 5, 6 | -7.4 | -0.8 |
| 14.0 cm | sinc, 4 | -4.8 | -1.0 |
| 14.0 cm | sinc, 6 | -7.3 | -0.8 |
| 14.0 cm | sinc, 8 | -9.5 | -0.6 |
| 14.0 cm | band limited, one weight, 8 | -3.7 | -0.7 |
| 14.0 cm | band limited, 4 | -8.8 | -0.9 |
| 14.0 cm | band limited, 6 | -12.5 | -1.3 |
| 14.0 cm | **band limited, 8** | -18.0 | -1.8 |
| 16.0 cm | linear, 2 | -2.0 | 0.6 |
| 16.0 cm | Lagrange 3, 4 | -3.0 | 1.3 |
| 16.0 cm | Lagrange 5, 6 | -3.5 | 1.7 |
| 16.0 cm | sinc, 4 | -2.5 | 0.9 |
| 16.0 cm | sinc, 6 | -3.5 | 1.8 |
| 16.0 cm | sinc, 8 | -4.2 | 2.4 |
| 16.0 cm | band limited, one weight, 8 | 0.3 | -0.4 |
| 16.0 cm | band limited, 4 | -3.9 | 1.0 |
| 16.0 cm | band limited, 6 | -4.5 | 0.9 |
| 16.0 cm | **band limited, 8** | -5.2 | 1.0 |

From 20 cm up every interpolator reads 0 dB or worse at 1 kHz and 1250 Hz.

- **Half a wavelength is a wall, and it is the frequency's, not the
  band's centre.** At 14 cm (1226 Hz) the 1250 Hz third octave is lost
  whatever reads it; at 16 cm (1072 Hz) the upper half of the 1 kHz one;
  at 20 cm (858 Hz) everything from 800 Hz; at 26 cm (656 Hz) the 630 Hz
  one. The direct sound of a source walking towards the listener, or away,
  runs along the rail at the whole of `k`: the favourable case of a field
  spread over directions does not apply to it.
- **Only weights that change with frequency reach the wall at every
  frequency.** One weight a position is as good at 8 cm and collapses
  under 630 Hz from 13 cm (-24 dB, -5 dB at 16 cm). A polynomial is right
  at low frequencies and 10 to 20 dB short at the top. A windowed sinc
  needs a rail's uniform spacing and is poor at its ends.
- **Eight positions over six over four** by 3 to 10 dB in the last octave
  under the wall, and nothing below it.

## Linear every 8 cm against eight positions, band limited

Early part, worst case, dB. Under the seam the nearest pitches measured to
8 cm are 6.5 and 9.8 cm; linear at 8 cm lies between them (-20 dB at
630 Hz).

Interior, far from the source:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -36.7 | -37.0 | -36.6 | -37.5 | -35.0 | -34.8 | -29.4 | -27.6 | -23.3 |
| 6.5 cm | **band limited, 8** | -35.7 | -35.8 | -35.7 | -36.1 | -36.4 | -36.1 | -37.2 | -37.0 | -36.7 |
| 9.8 cm | linear, 2 | -33.7 | -34.8 | -33.8 | -33.2 | -30.2 | -30.1 | -23.6 | -22.3 | -17.8 |
| 9.8 cm | **band limited, 8** | -34.0 | -33.9 | -33.9 | -34.1 | -34.6 | -34.2 | -34.5 | -33.8 | -33.7 |
| 13.1 cm | linear, 2 | -30.5 | -31.6 | -30.6 | -29.1 | -25.5 | -25.0 | -17.9 | -16.7 | -12.2 |
| 13.1 cm | **band limited, 8** | -32.2 | -31.3 | -32.2 | -31.1 | -32.0 | -32.3 | -31.3 | -31.5 | -31.4 |
| 16.3 cm | linear, 2 | -28.2 | -28.9 | -28.1 | -26.0 | -22.0 | -22.3 | -14.3 | -13.3 | -8.9 |
| 16.3 cm | **band limited, 8** | -30.1 | -28.5 | -30.0 | -28.3 | -28.8 | -29.2 | -28.9 | -29.4 | -28.9 |
| 19.6 cm | linear, 2 | -26.4 | -27.7 | -25.2 | -23.4 | -18.8 | -19.3 | -10.9 | -10.3 | -5.9 |
| 19.6 cm | **band limited, 8** | -28.5 | -27.2 | -28.4 | -26.9 | -27.2 | -27.9 | -28.4 | -27.9 | -25.8 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -11.7 | -12.5 | -9.6 |
| 8.0 cm | **band limited, 8** | -15.4 | -42.0 | -40.8 |
| 10.0 cm | linear, 2 | -8.3 | -9.1 | -6.0 |
| 10.0 cm | **band limited, 8** | -13.0 | -38.8 | -31.0 |
| 12.0 cm | linear, 2 | -7.1 | -6.4 | -3.4 |
| 12.0 cm | **band limited, 8** | -14.2 | -31.2 | -9.6 |
| 14.0 cm | linear, 2 | -4.2 | -3.9 | -1.0 |
| 14.0 cm | **band limited, 8** | -13.9 | -18.0 | -1.8 |
| 16.0 cm | linear, 2 | -3.0 | -2.0 | 0.6 |
| 16.0 cm | **band limited, 8** | -13.9 | -5.2 | 1.0 |

Interior, less than 1.2 m from the source:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -35.3 | -35.6 | -35.5 | -35.5 | -33.9 | -33.1 | -29.9 | -26.7 | -23.6 |
| 6.5 cm | **band limited, 8** | -34.6 | -34.4 | -34.3 | -34.2 | -34.1 | -34.7 | -35.0 | -35.1 | -34.3 |
| 9.8 cm | linear, 2 | -32.3 | -33.3 | -32.1 | -32.0 | -30.5 | -28.8 | -25.0 | -21.2 | -18.2 |
| 9.8 cm | **band limited, 8** | -32.9 | -32.2 | -31.5 | -31.5 | -31.4 | -32.1 | -32.4 | -32.0 | -29.9 |
| 13.1 cm | linear, 2 | -30.0 | -31.2 | -29.1 | -29.0 | -26.9 | -24.1 | -19.3 | -16.0 | -12.6 |
| 13.1 cm | **band limited, 8** | -33.0 | -32.8 | -31.8 | -32.2 | -32.3 | -31.0 | -30.1 | -31.2 | -30.4 |
| 16.3 cm | linear, 2 | -27.8 | -27.9 | -26.0 | -26.3 | -23.4 | -20.9 | -15.9 | -12.7 | -9.3 |
| 16.3 cm | **band limited, 8** | -30.0 | -31.6 | -30.8 | -31.4 | -30.3 | -29.3 | -27.6 | -29.1 | -28.9 |
| 19.6 cm | linear, 2 | -25.8 | -25.9 | -23.5 | -23.9 | -20.9 | -17.7 | -12.5 | -9.5 | -6.3 |
| 19.6 cm | **band limited, 8** | -27.5 | -29.7 | -28.6 | -29.6 | -29.1 | -27.6 | -26.7 | -26.9 | -22.9 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -13.4 | -12.8 | -9.6 |
| 8.0 cm | **band limited, 8** | -16.1 | -38.7 | -39.2 |
| 10.0 cm | linear, 2 | -10.1 | -8.9 | -6.1 |
| 10.0 cm | **band limited, 8** | -14.2 | -36.5 | -29.3 |
| 12.0 cm | linear, 2 | -8.9 | -6.3 | -3.3 |
| 12.0 cm | **band limited, 8** | -15.1 | -31.9 | -14.6 |
| 14.0 cm | linear, 2 | -6.1 | -3.6 | -1.0 |
| 14.0 cm | **band limited, 8** | -14.8 | -20.3 | -2.6 |
| 16.0 cm | linear, 2 | -4.9 | -2.0 | 0.7 |
| 16.0 cm | **band limited, 8** | -14.2 | -6.0 | 0.9 |

The first or last gap of a rail 1.2 m long, far:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -41.0 | -40.7 | -41.4 | -39.3 | -37.5 | -38.0 | -31.3 | -28.9 | -24.4 |
| 6.5 cm | **band limited, 8** | -40.6 | -40.2 | -41.4 | -40.3 | -38.3 | -39.2 | -37.8 | -38.0 | -36.8 |
| 9.8 cm | linear, 2 | -39.7 | -36.0 | -37.4 | -34.6 | -34.1 | -32.8 | -24.5 | -23.4 | -18.8 |
| 9.8 cm | **band limited, 8** | -35.3 | -34.9 | -35.5 | -35.6 | -36.3 | -36.2 | -36.7 | -36.5 | -36.8 |
| 13.1 cm | linear, 2 | -35.3 | -31.6 | -33.1 | -29.8 | -27.5 | -26.3 | -18.3 | -17.5 | -12.6 |
| 13.1 cm | **band limited, 8** | -31.2 | -30.1 | -31.4 | -30.3 | -32.0 | -31.0 | -31.3 | -30.0 | -30.4 |
| 16.3 cm | linear, 2 | -31.6 | -30.8 | -30.0 | -28.1 | -22.6 | -22.5 | -14.7 | -13.7 | -8.9 |
| 16.3 cm | **band limited, 8** | -30.7 | -29.7 | -30.2 | -29.3 | -28.0 | -29.9 | -30.3 | -29.7 | -24.7 |
| 19.6 cm | linear, 2 | -30.5 | -29.4 | -27.6 | -24.9 | -19.8 | -19.0 | -11.3 | -10.3 | -5.9 |
| 19.6 cm | **band limited, 8** | -29.1 | -28.3 | -28.5 | -27.0 | -25.1 | -27.8 | -27.5 | -25.5 | -20.0 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -14.5 | -12.5 | -9.4 |
| 8.0 cm | **band limited, 8** | -14.1 | -41.3 | -36.5 |
| 10.0 cm | linear, 2 | -11.6 | -9.6 | -6.8 |
| 10.0 cm | **band limited, 8** | -15.3 | -34.0 | -21.7 |
| 12.0 cm | linear, 2 | -8.9 | -6.0 | -3.3 |
| 12.0 cm | **band limited, 8** | -13.5 | -24.8 | -7.9 |
| 14.0 cm | linear, 2 | -7.4 | -5.2 | -2.0 |
| 14.0 cm | **band limited, 8** | -14.4 | -13.4 | -2.0 |
| 16.0 cm | linear, 2 | -4.8 | -2.9 | -0.4 |
| 16.0 cm | **band limited, 8** | -12.9 | -4.8 | -0.6 |

The same, near:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -37.1 | -36.3 | -36.1 | -36.5 | -35.3 | -33.4 | -30.9 | -27.1 | -24.7 |
| 6.5 cm | **band limited, 8** | -32.5 | -32.1 | -32.2 | -32.1 | -32.5 | -32.5 | -33.0 | -33.0 | -33.3 |
| 9.8 cm | linear, 2 | -34.1 | -33.6 | -33.1 | -34.0 | -31.0 | -29.2 | -25.3 | -21.8 | -18.8 |
| 9.8 cm | **band limited, 8** | -31.5 | -30.4 | -29.8 | -29.6 | -29.2 | -29.9 | -30.3 | -30.9 | -30.8 |
| 13.1 cm | linear, 2 | -32.6 | -33.3 | -31.2 | -31.7 | -28.5 | -24.8 | -19.9 | -16.6 | -13.3 |
| 13.1 cm | **band limited, 8** | -31.4 | -33.5 | -33.1 | -34.8 | -32.6 | -31.8 | -30.6 | -28.8 | -32.4 |
| 16.3 cm | linear, 2 | -27.8 | -27.9 | -26.2 | -26.8 | -24.5 | -21.4 | -16.5 | -13.1 | -9.9 |
| 16.3 cm | **band limited, 8** | -28.4 | -29.4 | -30.4 | -29.3 | -30.1 | -30.4 | -29.6 | -23.2 | -21.9 |
| 19.6 cm | linear, 2 | -26.0 | -26.4 | -24.5 | -25.1 | -22.0 | -18.7 | -13.4 | -9.6 | -7.0 |
| 19.6 cm | **band limited, 8** | -24.9 | -29.3 | -29.0 | -30.3 | -28.0 | -25.9 | -26.9 | -15.5 | -14.0 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -13.8 | -13.6 | -9.9 |
| 8.0 cm | **band limited, 8** | -16.1 | -37.0 | -33.0 |
| 10.0 cm | linear, 2 | -10.1 | -9.8 | -6.5 |
| 10.0 cm | **band limited, 8** | -13.6 | -31.4 | -19.4 |
| 12.0 cm | linear, 2 | -9.1 | -7.1 | -3.6 |
| 12.0 cm | **band limited, 8** | -13.5 | -23.1 | -9.8 |
| 14.0 cm | linear, 2 | -6.6 | -4.7 | -1.8 |
| 14.0 cm | **band limited, 8** | -13.1 | -13.6 | -2.0 |
| 16.0 cm | linear, 2 | -5.3 | -2.7 | -0.4 |
| 16.0 cm | **band limited, 8** | -13.5 | -5.5 | 0.1 |

Interior and far, the median:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -49.2 | -50.0 | -47.7 | -45.0 | -41.3 | -41.4 | -35.0 | -31.3 | -26.4 |
| 6.5 cm | **band limited, 8** | -52.5 | -51.5 | -52.6 | -51.9 | -52.3 | -51.9 | -50.0 | -50.3 | -50.0 |
| 9.8 cm | linear, 2 | -46.0 | -45.8 | -42.8 | -39.7 | -35.5 | -35.6 | -29.4 | -25.1 | -20.6 |
| 9.8 cm | **band limited, 8** | -49.1 | -49.1 | -48.9 | -48.3 | -47.9 | -47.6 | -46.8 | -46.4 | -45.3 |
| 13.1 cm | linear, 2 | -42.2 | -42.0 | -38.8 | -35.6 | -31.3 | -31.4 | -25.2 | -21.3 | -16.8 |
| 13.1 cm | **band limited, 8** | -44.5 | -44.7 | -44.8 | -44.5 | -44.1 | -44.3 | -42.6 | -43.3 | -43.4 |
| 16.3 cm | linear, 2 | -39.6 | -38.8 | -35.8 | -32.6 | -27.7 | -28.4 | -21.9 | -18.7 | -13.5 |
| 16.3 cm | **band limited, 8** | -42.0 | -42.2 | -42.0 | -41.7 | -41.9 | -42.6 | -40.2 | -41.8 | -41.4 |
| 19.6 cm | linear, 2 | -37.1 | -36.4 | -32.7 | -30.1 | -24.4 | -25.9 | -19.2 | -16.4 | -10.2 |
| 19.6 cm | **band limited, 8** | -41.1 | -41.4 | -41.7 | -41.3 | -40.8 | -41.9 | -39.5 | -40.6 | -34.9 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -18.9 | -16.6 | -13.2 |
| 8.0 cm | **band limited, 8** | -24.6 | -52.5 | -51.2 |
| 10.0 cm | linear, 2 | -16.3 | -12.9 | -9.8 |
| 10.0 cm | **band limited, 8** | -24.4 | -50.6 | -36.7 |
| 12.0 cm | linear, 2 | -14.1 | -10.2 | -7.2 |
| 12.0 cm | **band limited, 8** | -24.9 | -41.5 | -16.1 |
| 14.0 cm | linear, 2 | -11.8 | -7.9 | -5.0 |
| 14.0 cm | **band limited, 8** | -24.5 | -25.3 | -6.5 |
| 16.0 cm | linear, 2 | -9.9 | -6.2 | -3.6 |
| 16.0 cm | **band limited, 8** | -24.4 | -10.9 | -4.4 |

The ninth decile:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -42.3 | -42.0 | -41.6 | -40.7 | -38.0 | -37.5 | -31.5 | -29.4 | -25.1 |
| 6.5 cm | **band limited, 8** | -42.2 | -42.0 | -42.1 | -42.4 | -42.5 | -42.4 | -42.9 | -42.2 | -42.4 |
| 9.8 cm | linear, 2 | -38.7 | -39.0 | -38.1 | -36.3 | -33.1 | -32.0 | -26.0 | -23.6 | -19.1 |
| 9.8 cm | **band limited, 8** | -41.1 | -40.6 | -40.7 | -40.7 | -40.5 | -40.5 | -40.4 | -39.8 | -39.1 |
| 13.1 cm | linear, 2 | -36.4 | -36.4 | -34.9 | -32.3 | -28.4 | -28.1 | -21.3 | -18.6 | -14.3 |
| 13.1 cm | **band limited, 8** | -37.6 | -37.8 | -37.5 | -37.3 | -37.0 | -37.6 | -37.0 | -36.3 | -36.0 |
| 16.3 cm | linear, 2 | -33.5 | -34.0 | -31.9 | -28.7 | -25.1 | -24.7 | -17.9 | -15.0 | -10.7 |
| 16.3 cm | **band limited, 8** | -34.9 | -34.8 | -34.8 | -34.6 | -34.5 | -35.1 | -34.3 | -34.5 | -33.6 |
| 19.6 cm | linear, 2 | -30.8 | -31.6 | -28.9 | -25.7 | -21.9 | -21.7 | -14.7 | -12.0 | -7.6 |
| 19.6 cm | **band limited, 8** | -33.2 | -33.2 | -33.5 | -33.4 | -32.9 | -33.7 | -33.2 | -33.5 | -29.7 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -15.9 | -14.3 | -10.9 |
| 8.0 cm | **band limited, 8** | -18.7 | -48.3 | -47.8 |
| 10.0 cm | linear, 2 | -13.4 | -10.7 | -7.7 |
| 10.0 cm | **band limited, 8** | -17.7 | -46.7 | -32.8 |
| 12.0 cm | linear, 2 | -11.2 | -7.9 | -5.1 |
| 12.0 cm | **band limited, 8** | -18.4 | -37.0 | -13.0 |
| 14.0 cm | linear, 2 | -8.8 | -5.6 | -2.9 |
| 14.0 cm | **band limited, 8** | -18.2 | -21.5 | -3.7 |
| 16.0 cm | linear, 2 | -6.9 | -3.6 | -1.1 |
| 16.0 cm | **band limited, 8** | -18.1 | -7.8 | -1.4 |

The whole response, worst case:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -36.7 | -37.0 | -36.8 | -37.8 | -35.5 | -35.2 | -31.6 | -28.5 | -23.9 |
| 6.5 cm | **band limited, 8** | -35.7 | -35.8 | -35.7 | -36.0 | -36.3 | -36.1 | -37.2 | -36.9 | -36.6 |
| 9.8 cm | linear, 2 | -34.6 | -34.8 | -34.0 | -34.0 | -30.4 | -30.5 | -26.0 | -23.3 | -18.4 |
| 9.8 cm | **band limited, 8** | -33.9 | -34.0 | -34.0 | -34.1 | -34.6 | -34.2 | -34.5 | -34.0 | -33.7 |
| 13.1 cm | linear, 2 | -31.7 | -32.0 | -30.7 | -30.4 | -25.9 | -25.9 | -20.2 | -17.8 | -12.8 |
| 13.1 cm | **band limited, 8** | -32.3 | -31.4 | -32.3 | -31.6 | -32.1 | -32.2 | -31.4 | -31.6 | -31.5 |
| 16.3 cm | linear, 2 | -29.4 | -29.2 | -27.8 | -27.3 | -22.8 | -23.1 | -16.8 | -14.4 | -9.5 |
| 16.3 cm | **band limited, 8** | -29.7 | -28.5 | -29.9 | -28.8 | -28.9 | -29.1 | -29.0 | -29.3 | -29.0 |
| 19.6 cm | linear, 2 | -27.2 | -27.9 | -24.8 | -24.1 | -19.8 | -20.0 | -13.5 | -11.1 | -6.5 |
| 19.6 cm | **band limited, 8** | -28.1 | -27.1 | -28.4 | -27.4 | -27.4 | -27.8 | -28.3 | -27.9 | -25.9 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -13.0 | -12.7 | -10.0 |
| 8.0 cm | **band limited, 8** | -16.1 | -41.3 | -40.4 |
| 10.0 cm | linear, 2 | -9.5 | -9.4 | -6.8 |
| 10.0 cm | **band limited, 8** | -13.5 | -38.8 | -31.5 |
| 12.0 cm | linear, 2 | -8.3 | -6.7 | -4.0 |
| 12.0 cm | **band limited, 8** | -15.4 | -32.0 | -11.0 |
| 14.0 cm | linear, 2 | -5.6 | -4.2 | -1.9 |
| 14.0 cm | **band limited, 8** | -14.3 | -18.5 | -2.3 |
| 16.0 cm | linear, 2 | -4.3 | -2.4 | -0.1 |
| 16.0 cm | **band limited, 8** | -14.1 | -5.8 | 0.6 |

Under the low mask, early part, worst case:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -36.7 | -37.0 | -36.6 | -37.5 | -35.0 | -34.8 | -29.4 | -27.6 | -23.3 |
| 6.5 cm | **band limited, 8** | -35.7 | -35.8 | -35.7 | -36.1 | -36.4 | -36.1 | -37.2 | -37.0 | -36.7 |
| 9.8 cm | linear, 2 | -33.7 | -34.8 | -33.8 | -33.2 | -30.2 | -30.1 | -23.6 | -22.3 | -17.8 |
| 9.8 cm | **band limited, 8** | -34.0 | -33.9 | -33.9 | -34.1 | -34.6 | -34.2 | -34.5 | -33.8 | -33.7 |
| 13.1 cm | linear, 2 | -30.5 | -31.6 | -30.6 | -29.1 | -25.5 | -25.0 | -17.9 | -16.7 | -12.2 |
| 13.1 cm | **band limited, 8** | -32.2 | -31.3 | -32.2 | -31.1 | -32.0 | -32.3 | -31.3 | -31.5 | -31.4 |
| 16.3 cm | linear, 2 | -28.2 | -28.9 | -28.1 | -26.0 | -22.0 | -22.3 | -14.3 | -13.3 | -8.9 |
| 16.3 cm | **band limited, 8** | -30.1 | -28.5 | -30.0 | -28.3 | -28.8 | -29.2 | -28.9 | -29.4 | -28.9 |
| 19.6 cm | linear, 2 | -26.4 | -27.7 | -25.2 | -23.4 | -18.8 | -19.3 | -10.9 | -10.3 | -5.9 |
| 19.6 cm | **band limited, 8** | -28.5 | -27.2 | -28.4 | -26.9 | -27.2 | -27.9 | -28.4 | -27.9 | -25.8 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -12.5 | -18.9 | -27.1 |
| 8.0 cm | **band limited, 8** | -15.6 | -48.6 | -59.2 |
| 10.0 cm | linear, 2 | -9.1 | -15.1 | -23.4 |
| 10.0 cm | **band limited, 8** | -13.2 | -46.2 | -53.6 |
| 12.0 cm | linear, 2 | -8.0 | -12.3 | -20.7 |
| 12.0 cm | **band limited, 8** | -14.4 | -40.4 | -39.3 |
| 14.0 cm | linear, 2 | -5.3 | -10.3 | -18.1 |
| 14.0 cm | **band limited, 8** | -14.1 | -27.7 | -24.0 |
| 16.0 cm | linear, 2 | -4.2 | -8.1 | -16.6 |
| 16.0 cm | **band limited, 8** | -14.1 | -14.0 | -16.3 |

The same at a rail's ends:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | -41.0 | -40.7 | -41.4 | -39.3 | -37.5 | -38.0 | -31.3 | -28.9 | -24.4 |
| 6.5 cm | **band limited, 8** | -40.6 | -40.2 | -41.4 | -40.3 | -38.3 | -39.2 | -37.8 | -38.0 | -36.8 |
| 9.8 cm | linear, 2 | -39.7 | -36.0 | -37.4 | -34.6 | -34.1 | -32.8 | -24.5 | -23.4 | -18.8 |
| 9.8 cm | **band limited, 8** | -35.3 | -34.9 | -35.5 | -35.6 | -36.3 | -36.2 | -36.7 | -36.5 | -36.8 |
| 13.1 cm | linear, 2 | -35.3 | -31.6 | -33.1 | -29.8 | -27.5 | -26.3 | -18.3 | -17.5 | -12.6 |
| 13.1 cm | **band limited, 8** | -31.2 | -30.1 | -31.4 | -30.3 | -32.0 | -31.0 | -31.3 | -30.0 | -30.4 |
| 16.3 cm | linear, 2 | -31.6 | -30.8 | -30.0 | -28.1 | -22.6 | -22.5 | -14.7 | -13.7 | -8.9 |
| 16.3 cm | **band limited, 8** | -30.7 | -29.7 | -30.2 | -29.3 | -28.0 | -29.9 | -30.3 | -29.7 | -24.7 |
| 19.6 cm | linear, 2 | -30.5 | -29.4 | -27.6 | -24.9 | -19.8 | -19.0 | -11.3 | -10.3 | -5.9 |
| 19.6 cm | **band limited, 8** | -29.1 | -28.3 | -28.5 | -27.0 | -25.1 | -27.8 | -27.5 | -25.5 | -20.0 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | -15.4 | -20.2 | -27.3 |
| 8.0 cm | **band limited, 8** | -14.3 | -46.3 | -55.9 |
| 10.0 cm | linear, 2 | -13.0 | -16.4 | -24.6 |
| 10.0 cm | **band limited, 8** | -15.5 | -40.1 | -46.4 |
| 12.0 cm | linear, 2 | -10.2 | -13.6 | -20.9 |
| 12.0 cm | **band limited, 8** | -13.7 | -33.0 | -33.1 |
| 14.0 cm | linear, 2 | -8.9 | -11.5 | -20.9 |
| 14.0 cm | **band limited, 8** | -14.6 | -22.6 | -22.8 |
| 16.0 cm | linear, 2 | -6.2 | -9.1 | -19.1 |
| 16.0 cm | **band limited, 8** | -13.1 | -13.1 | -18.8 |

The level: the energy of the prediction over the truth's, early part,
worst case, dB either way:

| pitch | read | 100 | 125 | 160 | 200 | 250 | 315 | 400 | 500 | 630 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.5 cm | linear, 2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.3 | 0.5 |
| 6.5 cm | **band limited, 8** | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 9.8 cm | linear, 2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | 0.4 | 0.5 | 1.0 |
| 9.8 cm | **band limited, 8** | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| 13.1 cm | linear, 2 | 0.3 | 0.2 | 0.2 | 0.3 | 0.4 | 0.4 | 0.8 | 1.1 | 1.9 |
| 13.1 cm | **band limited, 8** | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| 16.3 cm | linear, 2 | 0.3 | 0.3 | 0.3 | 0.4 | 0.6 | 0.6 | 1.3 | 1.7 | 3.0 |
| 16.3 cm | **band limited, 8** | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| 19.6 cm | linear, 2 | 0.4 | 0.3 | 0.5 | 0.5 | 0.9 | 0.8 | 1.8 | 2.5 | 4.1 |
| 19.6 cm | **band limited, 8** | 0.3 | 0.4 | 0.3 | 0.4 | 0.4 | 0.3 | 0.3 | 0.3 | 0.3 |

| pitch | read | 800 | 1000 | 1250 |
| --- | --- | --- | --- | --- |
| 8.0 cm | linear, 2 | 1.6 | 2.1 | 3.1 |
| 8.0 cm | **band limited, 8** | 0.5 | 0.1 | 0.1 |
| 10.0 cm | linear, 2 | 2.5 | 3.2 | 4.5 |
| 10.0 cm | **band limited, 8** | 0.5 | 0.1 | 0.1 |
| 12.0 cm | linear, 2 | 3.0 | 4.6 | 5.7 |
| 12.0 cm | **band limited, 8** | 0.6 | 0.1 | 1.2 |
| 14.0 cm | linear, 2 | 4.1 | 6.5 | 6.7 |
| 14.0 cm | **band limited, 8** | 0.7 | 0.5 | 4.0 |
| 16.0 cm | linear, 2 | 4.7 | 7.9 | 7.8 |
| 16.0 cm | **band limited, 8** | 0.6 | 2.5 | 5.6 |

- **Today's reading loses level as well as shape**: a source half way
  between two positions 8 cm apart is 2.1 dB down at 1 kHz and 3.1 dB at
  1250 Hz at worst, a level that dips between every two positions as the
  source walks. Eight positions hold it to 0.3 dB under 630 Hz and at
  1 kHz at every pitch to 12 cm.
- **The near field is no worse than the far.** A listener 0.31 to 1.2 m
  from the rail reads within 3 dB of the far cases at every pitch. The
  least band the weights are made for is tied to the gap (0.4 of `pi` over
  it) so that a 100 Hz weight is not made for a 3.4 m wave: with a floor of
  3 rad/m the near cases lose up to 5 dB under 400 Hz.
- **An end costs 5 to 8 dB at the top** (1 kHz at 12 cm: -24.8 dB against
  -31.2 dB) and little under 630 Hz. Most rails of a recipe are 1.2 to
  2 m long, so most of a walk is within eight positions of an end.
- **Under 315 Hz everything reads the floor**, -29 to -37 dB, and there
  eight positions are up to 5 dB above two at a rail's end near the
  source (-29.6 dB against -34.0 dB at 200 Hz and 9.8 cm): past an end
  their weights are larger and carry more of the field's own scalar. It is
  the one place where they read worse than the linear rule, 20 dB under
  the bands that decide.

## A corner is solved, and no gap is a sliver

The weights depend on distances alone, so a corner of a rail can be read
with the same formula, and a free field says what that gives (`tests/
test_rail_interpolation.py`, seven plane waves, 12 cm, 500 Hz and 1 kHz
together): on a right angle half way between two samples, eight positions
leave -7 dB, where the two round the corner alone leave -7 dB, and the
weights' own estimate at 1 kHz says -13 dB. A position in two dimensions is
not determined by samples of two lines it is on neither of.

So a rail read from more than two positions is solved differently
(`trace.plan.read_arcs`): **every corner is a position, and each straight
leg between two corners is cut into equal gaps of the pitch at most.**
Every place of the rail is then between two positions of its own line; the
gaps either side of the corner read -22 and -24 dB and the next -36 dB.

Equal gaps matter as much as the corner. A rail's samples
(`scenes.rail_samples`) are every pitch from one end and then the other
end itself: a rail 1.21 m long ends on a gap of 1 cm, and a seat's vertical
rail of 0.50 m at 12 cm on one of 2 cm. Two positions that close say the
same thing, and the weights that tell them apart grow: with the samples
and the corners together, the realistic recipe at 12 cm asked for a
weight of 17 at its worst step, and at 196 of its 10 653 moving steps for
weights four times or more what the linear rule takes: 12 to 25 dB more
of whatever the solves do not share. With equal gaps the largest weight is
1.2, on the recipe at every pitch from 8 to 16 cm. The realistic recipe has 72 corners
on its 90 rails.

## The engine, on a free monopole

`tests/test_rail_interpolation.py` walks a monopole 1.2 m at 1 m/s, 0.8 m
straight at the listener and 0.4 m across after a right angle, and renders
the low band against a pack that holds the response of every step. Third
octaves of channel 0, dB, exact pitches, ends and corner included:

| read | 200 | 250 | 315 | 400 | 500 | 630 | 800 | 1000 | 1250 | 64 channels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 cm, two, linear | -44.5 | -40.6 | -36.6 | -31.8 | -29.3 | -24.4 | -20.9 | -17.9 | -14.8 | -24.9 |
| 8 cm, eight | -56.1 | -55.0 | -54.2 | -54.1 | -55.0 | -52.4 | -52.6 | -50.5 | -42.6 | -53.2 |
| 12 cm, four | -45.4 | -45.4 | -45.6 | -46.1 | -47.0 | -35.7 | -29.9 | -22.8 | -15.1 | -32.1 |
| 12 cm, eight | -53.8 | -53.1 | -53.3 | -53.2 | -52.5 | -49.4 | -43.1 | -33.2 | -22.3 | -42.3 |
| 14 cm, eight | -48.5 | -48.8 | -49.4 | -49.9 | -50.2 | -43.5 | -34.6 | -23.4 | -11.4 | -31.9 |
| 16 cm, eight | -48.5 | -48.6 | -48.0 | -48.5 | -46.1 | -34.2 | -22.6 | -10.2 | -3.4 | -21.0 |

In free air the 1250 Hz third octave at 12 cm is 7 dB better than today's
reading; on the line it is equal. The line's figure is the one to plan
with.

## The decision, and what each choice costs

The realistic recipe (`data/runs/w45_clarify_scene/scene1/recipe.json`)
regenerated at each pitch is the same scene to the byte but for its rails'
`pitch_m`, and at 0.08 the same bytes. `python -m reverberate.trace rent
... --dry-run`, 0.136 USD/h on one RTX 3080:

| rail pitch | positions read | solves | pairs | low band | whole trace | pack | against today's worst third octave |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 cm | 2, linear (today) | 1529 | 16 887 | 7.07 USD | 7.54 USD | 23.8 GB | -9.6 dB at 1250 Hz, -12.5 dB at 1 kHz |
| 8 cm | 8 | 1588 | 25 508 | 7.43 USD | 8.04 USD | 34.4 GB | 27 dB better or more at 1 kHz and 1250 Hz: a variant for the ear, not for the cost |
| **10 cm** | **8** | **1272** | 23 956 | 6.11 USD | **6.70 USD** | 32.5 GB | **no loss**: 9.5 dB better or more from 315 Hz up in every case; its own worst is -19.4 dB |
| 12 cm | 8 | 1096 | 23 219 | 5.41 USD | 5.98 USD | 31.6 GB | equal at 1250 Hz (-9.6 dB interior, -7.9 dB against -9.4 dB at a rail's ends), 10 to 19 dB better at 1 kHz and below |
| 12 cm | 4 | 1071 | 17 910 | 5.18 USD | 5.66 USD | 25.1 GB | 1 kHz -15.5 dB: 3 dB better than today, 1250 Hz 4 dB worse |
| 14 cm | 8 | 963 | 22 530 | 4.88 USD | 5.45 USD | 30.8 GB | 1250 Hz lost (-2 dB against -9.6); 1 kHz equal to 5 dB better; equal under the mask |
| 16 cm | 8 | 849 | 22 090 | 4.43 USD | 4.98 USD | 30.2 GB | 1 kHz lost (-5 dB against -12.5), 6 dB worse under the mask; 9 dB better at 630 Hz |

- **The option that loses nothing is 10 cm read from eight positions**: by
  the worst third octave of the worst case, raw, it reads -19.4 dB where
  today reads -9.6 dB, and from 315 Hz up it is better than today in every
  band and every case measured. It saves 17 % of the solves and 0.84 USD
  of 7.54.
- **12 cm is the widest pitch the mask allows** (half a wavelength at
  1430 Hz) and loses nothing under the mask, where its worst band is the
  floor's -29 dB against today's -19 dB. Raw, its 1250 Hz third octave is
  today's to 1.5 dB. It saves 28 % of the solves and 1.56 USD.
- **14 and 16 cm trade the top of the band for the solves**: 14 cm gives
  up the 1250 Hz third octave, which the mask holds 22 dB down, and 16 cm
  the upper half of the 1 kHz one, which it does not. They save 2.09 and
  2.56 USD.
- **Fewer solves are not as many fewer as the pitch says.** A pitch half
  as wide again is 28 % of the solves fewer at 12 cm, not a third:
  stations stay, each rail keeps its two ends, eight positions round a step
  reach further along a rail than two, and corners are added. And the
  pairs grow by 40 %, because eight positions a step are heard at the step's cells: the
  pack is 8 GB larger and everything but the solves 0.10 USD dearer, which
  the figures above include.

## What the engine pays

A step's response is `sum_s w_s(f) H_s(f)` in the transform the engine
already takes of a step, the weights read between knots 50 Hz apart. A
source at rest reads one position as before, on a transform 11 % longer
(5760 samples against 5184). A source on a rail read from eight positions
costs 2.2 times its low band read from two: 102 ms against 47 ms a second
of signal on one laptop core, with one cell.

## Two other levers, measured here and not implemented

- **The length of a response.** On the line, what lies after 0.8 s of a
  low band response is -62 dB of it at worst and -70 dB in the median,
  after 0.6 s -54 dB and -59 dB. A solve's time is its steps: stopping at
  0.8 s instead of 1.2 s is a third of every solve, 2.3 USD of this scene,
  and a third of the pack. The line is one source in its own room; a far
  voice heard through two doors decays later and was not measured.
- **Solving only to the end of a steeper mask.** A solve's cost goes as
  the fourth power of its top frequency. The mask's ramp is an octave,
  707 to 1414 Hz, and the solve goes to 1500 Hz. A ramp of half an octave,
  841 to 1189 Hz, would be solved to 1261 Hz: half the cost of every solve,
  and a rail every 14 cm with nothing lost. It changes the crossover the
  owner validated by ear and is a change of the format: said, not done.

## What the data cannot settle

1. **The 1250 Hz third octave at an exact 12 cm in a room.** The line's
   positions are 11.2 to 12.8 cm apart there. A rail solved every 12 cm and
   every 2 cm, low band alone, settles it for the price of a smoke run.
2. **A corner in a room.** The line is straight; the corner's figures are a
   free field's.
3. **The seated rails.** A rise is read by the same weights along its
   vertical rail and was not measured: the line is horizontal.
4. **Taking the direct sound out before interpolating.** It is known in
   closed form at every position and is what runs along the rail at the
   whole of `k`; the rest of the field might then be read from positions
   further apart than half a wavelength. Not tried.
