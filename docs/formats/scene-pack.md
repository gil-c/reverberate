# The scene pack

Status: proposed with [ADR 0016](../adr/0016-a-scene-moves-and-one-engine-renders-it.md).
Written by the trace stage on a rented card (lot L6, from L4 and L5), read by
the signal engine (`reverberate.render`, L7). `reverberate.render.pack` is
the one implementation of this document: its types, its writer
(`PackWriter`, `write_pack`), its reader (`read_pack`) and its invariants
(`validate`, which both the writer and the reader run). The trace is
`reverberate.trace`: the tables of `early`, `tail` and `/directivity` are
made by `reverberate.mirror.moving`, `mirror.tails` and
`mirror.directivity`, whose `pack()` methods give the datasets below in the
types below, `low/ir` by `reverberate.spatial.lowband`, and the three
scalars that level the two sides by `reverberate.trace.level`. No card has
run it yet: what it writes was made on `numpy`, with a monopole in free air
where a card would solve.

A pack is everything the acoustics of one recipe
([`scene-recipe.md`](scene-recipe.md)) come to, sampled along the
trajectories: for every source and every step, the arrivals above the
crossover, what the tail is made from, which solved responses carry the band
under it, and the scalar that levels the two. The signal engine applies it to
dry audio and needs nothing else: no geometry, no solver, no card. It is a
cache. Its identity is the recipe's digest, the asset keys and the code
version; it is regenerated, not archived.

One file, `pack.h5`, HDF5 as the fields are. The reasons: the tables are
ragged and large and are read a source and a block at a time, which HDF5
does by slice and `.npz` does not; `h5py` is already what every field is read
with; and one file is one thing to copy from a rented machine. Uncompressed
and little endian, for the reason of `impulse-response.md`.

## Conventions

Everything of `ambisonic-response.md` and `scene-recipe.md` holds. In
addition:

| quantity | choice |
| --- | --- |
| positions and directions | scene frame, `(x, y up, z)`; a direction is a unit vector |
| ambisonic frame | reached by `spatial.sh.scene_to_ambisonic` in the engine, nowhere in the pack |
| step | `step_s = 0.05`; step `k` is the scene at `t_k = k step_s`; `steps = round(duration_s / step_s) + 1` |
| output | `steps - 1` intervals of one step each, `(steps - 1) * 2400` samples: the last step ends the scene |
| bands | `bands_hz = (125, 250, 500, 1000, 2000, 4000, 8000)`, `acoustics.OCTAVE_BANDS`: what a path's gain and a histogram are written on |
| bank bands | `bank_bands_hz = metrics.band_centres(48000)`, eight, the last at 16 kHz; bank band `i` reads the band nearest its centre, as `mirror.render._band_map` |
| gains | pressure, linear, unless the name ends in `_db` |
| clock | the wave field's: a response of `low/ir` starts `lead_s` before its geometric time zero |
| missing index | `-1` |

**The step is 50 ms** because that is the time the fastest source the
recipe allows, 1.5 m/s, takes to cross one 8 cm rail sample (53 ms): a
moving source is on a new solved position at about every step and never
skips one. It is 2400 samples at 48 kHz and 200 at the low band's rate, both
whole. A listener at 1.5 m/s moves 7.5 cm in a step, under a fifth of the
0.40 m lattice.

**Rendering is quasi-static.** At time `t` the engine plays what the scene
frozen as it is at `t` would give for the signal emitted so far: a delay is
the path's length now, a response is the response between the positions
now. That is what makes a moving delay a Doppler shift, and it is the rule
of ADR 0013's page. Its error is of the order of speed over sound speed,
0.4 per cent, except for the late reverberation, which follows the source at
once instead of over its own decay.

## Root

Attributes:

| attribute | type | value |
| --- | --- | --- |
| `schema` | str | `"reverberate.scene-pack"` |
| `schema_version` | int | `1` |
| `profile` | str | `"trace"`, `"synthetic-free-field"`, or `"synthetic-density"`: random tables at a real pack's size, for the cost benchmark, whose render means nothing |
| `recipe_sha256` | str | the recipe's identity |
| `dwelling`, `scene_id` | str | as the recipe |
| `duration_s`, `step_s` | float | |
| `steps` | int | |
| `sample_rate_hz` | float | `48000`, the rate of the output |
| `order` | int | `7`: 64 channels |
| `ordering`, `normalisation` | str | `"ACN"`, `"N3D"` |
| `sound_speed_m_s` | float | `343.2`, the solver's at the recipe's temperature |
| `bands_hz`, `bank_bands_hz` | int array | as above |
| `has_low`, `has_tail` | bool | whether the two groups exist in every source |
| `low_sample_rate_hz` | float | `4000` |
| `low_samples` | int | `4800`, 1.2 s |
| `fusion_json` | str | how two cells are fused and which may serve: `{"quadrature_degree": 26, "lambda": 0.001, "exact_under_m": 0.001, "translate_within_m": 0.2, "fuse_within_m": 0.3, "source_share": 0.15, "surface_share": 0.5}` |
| `provenance_json` | str | see Provenance |

Dataset `/recipe`, `uint8 [byte]`: the recipe's canonical bytes, whose
SHA-256 is `recipe_sha256`. The engine reads the sources' activity, clips
and gains from it, so a pack is rendered without its recipe file.

## `/listener`

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `position` | float64 | `[step, 3]` | the centre of the head |
| `orientation` | float32 | `[step, 3]` | yaw, pitch, roll in degrees, the recipe's convention, yaw not wrapped |

The orientation is carried for the decoder and is **not applied**: the
engine's output is the order 7 field at the head's centre in the scene's
fixed ambisonic frame. The head turns at decode, where it costs nothing and
where the page already does it.

## `/cells`

The listening positions the low band was solved at and the scene uses: an
order 7 array stood at each. Shared by every source.

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `position` | float64 | `[cell, 3]` | the array's centre: the node of the low grid the array stood on, not the point asked for |
| `kind` | uint8 | `[cell]` | `0` lattice, `1` seat, `2` added: on the listener's path, or where the lattice is too coarse |
| `lattice_index` | int32 | `[cell, 3]` | `(i, layer, k)` on the lattice; `-1, -1, -1` off it |
| `clearance_m` | float32 | `[cell]` | distance from the centre to the nearest surface of `mirror/scene.npz` |
| `room` | str | `[cell]` | by ADR 0010: the room of the recipe's station nearest the cell, the trace having the recipe's rooms and not the dwelling's polygons |

Attributes `grid_origin_m`, `grid_step_m` (0.40 in `x` and `z`) and
`layers_y_m` (the two heights). A seat's cell is asked for at the seat
exactly; its `position` is the node the array got, half a grid step's
diagonal away at most (19 mm on the grid to 1500 Hz), or up to 0.10 m aside
where the seat leaves the array's ball no free air. The engine translates
from `position`, so the difference costs nothing; a field that recorded the
point asked for instead carried a floor of -25 dB
(`docs/open-questions/low-band-translation.md`).

## `/sources/<id>`

One group per source of the recipe, named by its `id`. Attributes: `kind`,
`subtype` (empty for a voice), `directivity_model`, `directivity_enabled`,
`gain_db`, `tail_seed` (uint64, the first eight bytes, little endian, of
`sha256("<seed>:tail:<id>")`, the seed written in decimal).

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `position` | float64 | `[step, 3]` | the mouth |
| `yaw_deg` | float32 | `[step]` | the facing, not wrapped |
| `audible` | bool | `[step]` | the step lies in an activity interval or within `low_samples / low_sample_rate_hz` after one (`scenes.audible_steps`) |

A step that is not `audible` has no arrivals and `-1` in every index below;
the trace does not compute it.

### `early`: the arrivals above the crossover

One table, every audible step's arrivals one after the other.

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `offsets` | int64 | `[step + 1]` | step `k` owns rows `offsets[k]` to `offsets[k + 1]` |
| `path_id` | uint64 | `[row]` | the path's identity, the same at every step it exists |
| `delay_s` | float64 | `[row]` | `length_m / sound_speed_m_s`, geometric: `lead_s` is not in it |
| `arrival` | float32 | `[row, 3]` | unit vector from the listener towards where the sound comes from (`Paths.direction`) |
| `departure` | float32 | `[row, 3]` | unit vector from the source along the path's first leg |
| `gain` | float32 | `[row, band]` | spreading times the reflection factors, `Paths.gain` after `regain` with the calibration's image materials; omnidirectional, without air |
| `order` | uint8 | `[row]` | reflections on the path |
| `kind` | uint8 | `[row]` | `0` direct, `1` specular, `2` diffracted round edges, `3` diffracted then reflected |

Within a step the rows are sorted by `path_id`, and no `path_id` repeats.

**`path_id`** is the first eight bytes, little endian, of the SHA-256 of:
the `kind` as one byte, then the facets of the path in bounce order as
little endian `int32` (`Paths.sequence` without its padding), then for kinds
2 and 3 one `int32` of `-1` and the indices of the edges it bends round in
order from the source (`mirror.edges.diffracting_edges`), as `int32`. The
`-1` keeps a facet from being read as an edge. An image is its facet
sequence wherever the source stands, so the identity survives the tree
being grown again at every position. A diffracted path one of whose corners
is the grid's and not an edge's has no stable name: in place of the edges it
takes `-1, -1` and its rank by delay among the step's such paths of its
kind, and the jump rule below catches a rank that changed hands. A
diffracted then reflected path (kind 3) names the edge its reflection is
taken from, the one nearest the listener. `render.pack.path_id` is the
function, and the trace (`mirror.moving`) names its rows with it.

**`departure`**, for an image path, points from the source at its first
reflection point; for the direct path, at the listener; for a diffracted
path, at the first corner the path bends round, counted from the source.

**A step without a direct path** holds the reflections that reach it and
the diffracted onset the mirror gives a point its source does not see
(`mirror.diffract`), as kinds 2 and 3: at most 24 rows, the shortest.
The onset is computed for the step's source and listener alone. The lattice
field of ADR 0014 computed it for all its points together, which shares
one image tree between edges 0.25 m apart and frees the occupancy grid round
every point; so at such a point the pack's onset is the mirror's own on that
pair, and may differ from that field's in its reflected rows.

**Between two steps** the engine moves each path as follows, with
`u = (t - t_k) / step_s`, `l(t)` the listener's position and `c` the sound
speed.

- The path's **apparent source** at a step is
  `q_k = l_k + c delay_s arrival`: the image itself for kinds 0 and 1.
  `arrival` is float32 and unit to 1e-7 only; the engine makes it unit in
  float64 first, so that the delay at a step is `delay_s` and not `delay_s`
  times that error.
  `q(t)` is linear from `q_k` to `q_{k+1}` and `l(t)` linear from `l_k` to
  `l_{k+1}`.
- **Delay**: `|q(t) - l(t)| / c`, plus `lead_s`. Continuous, equal to the
  stored delay at each step, and exact for an image when source and listener
  move in straight lines within the step, an image being an affine function
  of its source.
- **Arrival direction**: `(q(t) - l(t))` normalised, then
  `scene_to_ambisonic`, then `real_sh` at order 7. On the sphere by
  construction.
- **Gain**: linear in `u`, band by band. Everything else that scales a
  path is read at the two steps and multiplied into that gain before it is
  interpolated: the directivity's gain for the step's `departure` and
  `yaw_deg`, the air's loss for the step's `delay_s`, the crossover's
  window for the step's `delay_s` and `level/onset_s`, and `high_gain_db`.
  The product is linear in `u`; none of its factors is interpolated on its
  own, and the departure direction is not interpolated at all.
- **Birth and death**: a `path_id` present at one of the two steps only has
  gain zero at the other and keeps the apparent source of the step where it
  exists, so it fades over one step and its delay still follows the
  listener.
- **Jump**: a `path_id` present at both steps whose apparent source moved
  more than 0.30 m is two paths, one dying and one born. A real image moves
  at the source's speed, 7.5 cm a step at most.

A path's delay line therefore belongs to its `path_id`: it is created at
birth, read at the moving delay with an interpolator of the engine's choice
(the mirror's own is a 33 tap windowed sinc, `mirror.render.DELAY_HALF_TAPS`),
and released after death.

**The engine's interpolator** (`reverberate.render.delay`) is a Kaiser
windowed sinc of 12 taps (shape 11) on the filtered dry signal held at
twice the rate, tabulated at 1024 fractions of a sample and read linearly
between two. Measured against the exact delayed tone, worst over the
fraction and the frequency: **-94.0 dB (2.0e-5) from 1 kHz to 20 kHz**,
-115.6 dB under 1 kHz, and 0.0002 dB of level. A delay of a whole number of
samples is exact. Holding the signal at twice the rate needs its band to end
before its Nyquist frequency: the early part is faded out by a raised cosine
from 22 kHz to 24 kHz, which `mirror.render` does not do, so at rest the two
agree to the pack's float32 (3e-7) under 22 kHz and differ above.

**The direction** is evaluated at 8 instants a step and its harmonics are
linear between two: a path that turns by `a` radians in a step errs by
`(7 a / 8)^2 / 8` of its top order at most.

### `low`: the band under the crossover

The responses, deduplicated:

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `ir` | float32 | `[pair, channel, sample]` | the wave response between one source position and one cell, order 7, at `low_sample_rate_hz` |
| `pair_position` | float64 | `[pair, 3]` | the source position it was solved from |
| `pair_cell` | int32 | `[pair]` | row of `/cells` |
| `pair_key` | bytes, 64 | `[pair]` | the pair's key in the dwelling's cache |
| `seam_db` | float32 | `[pair]` | `mirror.hybrid.seam_db` between the pair's wave response and the mirror rendered at the same pair, omnidirectional, before any levelling |
| `onset_s` | float64 | `[pair]` | time of the loudest sample of channel 0 of the wave response at 48 kHz, on the pack's clock |

Both are read as `mirror.hybrid.blend` reads them, on the wave response
**with its air and before its masks** (`spatial.lowband.with_air` of the
cache form), not on `ir`, whose masks have taken most of the seam's octave.
The mirror of the seam is `mirror.render.render_point` at the pair: the
source on `pair_position`, the head on the cell's centre, the pair's own
image paths and diffracted onset, the tail the pack gives that pair (its
histograms, each on its own `scale`, summed with the weights a step at rest
there would have), then the air, the low cut and the signature, on the wave
field's clock and scale (`/mirror`'s `lead_s` and `alignment_gain`). Its
noise is drawn with the mirror's seed plus the cell's row, as a field draws
a point's (`reverberate.trace.level`).

and per step:

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `pair` | int32 | `[step, 2, 2]` | rows of `ir`: source slot by cell slot |
| `position_weight` | float32 | `[step]` | weight of source slot 1; slot 0 has one minus it |
| `cell` | int32 | `[step, 2]` | rows of `/cells`, slot 0 the nearer |
| `mode` | uint8 | `[step]` | `0` inaudible, `1` exact, `2` translated from one cell, `3` fused from two |

**What `ir` is, and its scale.** `spatial.lowband.to_stored` of the pair's
response in the dwelling's cache, which holds it at 4 kHz before its masks
and its air. **Its samples are the 48 kHz response's own**, read every
twelfth: the kept bins transformed back at 4800 samples and divided by 12
(`spatial.lowband.decimate`), so the cache, the pack and a field are on one
scale and `spatial.lowband.from_stored` gives the 48 kHz response back. The
engine therefore multiplies by `sample_rate_hz / low_sample_rate_hz`, 12,
when it convolves at 4 kHz: the dry signal decimated at unit gain and
convolved with `12 ir` at 4 kHz is at the response's level. It
is the wave response with the crossover's low side already taken, as
`mirror.hybrid.blend` takes it: the part within the onset window
through the pressure mask, the rest through the power mask
(`Crossover.masks`), and the air absorption of `/atmosphere` applied
(`audio.apply_air_absorption`). Its spectrum is therefore zero above
`cutoff_hz * 2^(width_octaves / 2)`, 1414 Hz, which is what lets it be kept
at 4 kHz: the decimation is exact (the bins of the 48 kHz transform under
2 kHz, transformed back at 4800 samples), and the engine's interpolation
back to 48 kHz has the whole of 1414 to 2586 Hz for its transition. At
3 kHz the first image would start at 1586 Hz and that transition would be
seven times narrower. One response is 1.23 MB against 14.7 MB at 48 kHz.

**Source side.** At rest at a station, slot 0 is the station's position and
`position_weight` is 0. On a rail, slots 0 and 1 are the two solved
positions either side of the source and the weight is linear in arc length
between them. The engine crossfades the two responses' outputs with those
weights. That is averaging two waveforms, which this project measured to
hold 1 kHz only when the two are 8 cm apart; it is why rails are solved
every 8 cm.

**Listener side.** `mode` says how the field at the cell becomes the field
at the head:

- `1`, exact: the head is within `exact_under_m` of cell slot 0. The
  response is used unchanged. A seat, or a listener at rest on a lattice
  point.
- `2`, translated: cell slot 0 alone. The cell's 64 channels are translated
  by `d0 = l(t) - position[cell 0]`.
- `3`, fused: both cells, by the minimum norm plane wave estimator.

**The pack stores the cells, not the weights**, and the engine derives the
operator from the offsets `d_j = l(t) - position[cell j]`. With the
directions `s` and weights `w` of `spatial.sh.quadrature(quadrature_degree)`
normalised to sum to one, `Y = real_sh(7, s)`, `k = 2 pi f / c`, and
`A_j = Y^T diag(exp(-i k s . r_j))` the expansion at cell `j` of unit plane
waves, where `r_j = scene_to_ambisonic(d_j)`:

```
G(f) = Y^T W A^H (A W A^H + lambda tr(A W A^H) / n I)^-1        A = [A_0; A_1], n its rows
b_listener(f) = G(f) [b_0(f); b_1(f)]
```

For one cell the same expression with `A = A_0`. This is the estimator of
`w44_clarify_interpolation/leave_one_out/loo_planewave.py`, there evaluated
for channel 0 only, here for all 64. Stored, `G` is a 64 by 128 complex
matrix per frequency and per step: 8.5 MB a step on 129 frequencies, 200 GB
a scene. Derived, the matrix to invert depends only on `r_0 - r_1`, the
vector between the two cells, which takes a few values on a lattice; it is
factored once per vector and frequency, and a step costs one product. So
the offsets cost the engine little and the weights would cost the pack
everything.

For one cell `A W A^H` is the identity and the estimator is the plain
translation over `1 + lambda`. Mode 2 takes the plain translation,
`T(f) = Y^T W diag(exp(+i k s . r_0)) Y`
(`spatial.translate.translation_operator`), without the `1 + lambda`: a
head on its cell is then the cell itself.

**Between two steps** the engine renders the low band in frames one step
long, four to a step (centred every 12.5 ms), under a square root Hann
window at analysis and at synthesis, so the frames add to one. A frame
takes the `pair`, the `position_weight`, the `cell` and the `mode` of **the
step nearest its centre**, and the listener's offsets `d_j` **at its
centre**, `l(t)` being linear between steps. The passage from one step's
responses to the next is therefore the frames' own overlap, a raised cosine
one step long, and nothing of the low band is interpolated by index: a
weight or a cell that changes hands between two steps is two sets of frames
cross-fading. The frame's transform has 101 bins of 20 Hz, of which the 75
under `1.05 * 1414` Hz are given to the operator.

**The translation is behind one interface**
(`reverberate.render.translate.Translation`): the cells' fields and the
head's offsets in, the head's field out. Its default (`SpatialTranslation`)
is `reverberate.spatial.translate`, the one implementation of the
mathematics above, applied without forming `T` or `G`
(`apply_translation`, `apply_fusion`): the fields are solved against
`(A W A^H + mu I)`, which depends only on the vector between the two cells
and is inverted once per vector (`fusion_inverse`), spread on the
quadrature's 378 plane waves, each turned by its own phase, and gathered.
Forming `G` first gives the same field at twenty times the cost. A function
that returns `G` is swapped in through `OperatorTranslation`.

The trace chooses the cells **by clearance**
(`spatial.translate.choose_cells`): a cell may serve a head at distance `d`
only if `d` is under its serving radius, the smaller of `surface_share` of
`clearance_m` and `source_share` of its distance to the source, and under
`fuse_within_m`. Slot 0 is the nearest cell that may serve; slot 1 the
nearest that may on the other side of the head. With both, `mode` is 3;
with slot 0 alone and within `translate_within_m`, 2; otherwise a cell is
needed there (`kind` 2). The shares are measured: inside the free ball
itself, past 0.15 of the source's distance, the 64 channels are no
prediction (`docs/open-questions/low-band-translation.md`).

**Where the cells are** (`reverberate.trace.plan`): one at every place the
listener rests, at the height it has there; one every 0.15 m of its path;
where a source is so near that the rule refuses those, the path every
0.10 m; and where that is refused too, one on the head itself, if an array
can stand there (0.26 m free of surfaces and of mouths). The plan makes the
rule hold at every audible step before anything is rented. The machine
applies it again on the centres the arrays really got. **A step no cell may
serve does not fail the scene**: it reads the nearest cell alone, mode 2,
and is counted in the provenance (`fallback_steps`); a mouth nearer the head
than an array's radius is the case.

### `tail`: what the late part is made from

The histograms of the mirror's rays, deduplicated:

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `energy` | float32 | `[hist, bin, band]` | `Histogram.energy`: energy per 2 ms bin, in units of the source's |
| `moments` | float32 | `[hist, bin, band, 16]` | `Histogram.moments`: its order 3 directional moments, ACN and N3D, ambisonic frame |
| `scale` | float64 | `[hist, bank band]` | `scale_per_band` of `mirror.render.render_point`: what turns the histogram's energy into the response's |
| `hist_position` | float64 | `[hist, 3]` | the source position the rays left from |
| `hist_cell` | int32 | `[hist]` | row of `/cells`: where the receiver sphere stood |

and per step:

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `hist` | int32 | `[step, 2, 2]` | rows of `energy`: source slot by cell slot |
| `position_weight` | float32 | `[step]` | weight of source slot 1, in energy |
| `cell_weight` | float32 | `[step]` | weight of cell slot 1, in energy |

600 bins of `histogram_bin_s`, on the geometric clock. The step's tail
holds the weighted sum of the four's energies, bin by bin, band by band and
direction by direction: an interpolation in energy, never of waveforms (the
engine adds four independent noises, see below). Rays are traced from every
station and from rail positions every 0.80 m; the receiver spheres stand on
cells of `/cells` no nearer one another than 0.80 m, and on every seat:
the trace chooses them among `/cells` and `hist_cell` is their row there.
The weights are linear in arc length on the source side and in distance on
the listener's: slot 1 of the cells has `d0 / (d0 + d1)`. The two cells are
the nearest two the head sees, among the six nearest (a cell behind a wall
is near and is another room's tail); where it sees none, the nearest alone.
`scale` is the cell's own where it sees the source position and rays reached
it; elsewhere the median of the position's cells that do, as the lattice
field gives its points without a direct path. The late level measured between neighbours 0.40 m apart
differs by 0.3 to 0.7 dB in the median, so the tail is sampled more coarsely
than anything else in the pack.

The engine makes the tail from the same quantities as
`mirror.render.tail_from_histogram`: per bin and bank band an energy, the
calibration's `tail_gain_db` on top of `scale`, through the inverse of the
bank's own reading (clipped at zero), times the air's share, and a density
over the 45 directions of `spatial.sh.quadrature(2 * histogram_order + 2)`,
the moments' reconstruction clipped at zero times the quadrature's weights
(the weights alone where nothing is left). Each histogram's energy is
multiplied by **its own** `scale`; the moments are read as a density only.
The bins are on the geometric clock and the tail is laid `lead_s` later,
rounded to a whole sample. `tail_bursts` is the reference renderer's and is
not read by the engine.

**One carrier per source, already through the bank.** For every bank band
`b` and direction `d` the source has one stream of noise, drawn from
`tail_seed`, passed through the band's filter once and brought to unit
variance. A histogram's response is, on each direction, the sum over the
bands of the stream times a gain that is constant over a bin: the square
root of the energy the bin holds for that band and direction, over the
bin's 96 samples. The energy is first spread over the neighbouring bins
with the shares the band's filter gives a bin of white noise (at 125 Hz
half of it goes to the two bins either side and two thousandths to the
third; at 16 kHz six thousandths go to the next), so that **the expected energy of
every bin, band and direction is that of shaped white noise through the
bank**, which is what the reference renderer makes. The bands are then
scaled, one gain a band, so that **the octave bank reads on the first
channel of the rendered response what the histogram holds**: what the
reference renderer's tail reads in expectation (the energies asked, through
the bank's reading of white noise shaped by each band's filter), against the
reading of this very draw, each band's share and what two shares have in
common, both through the crossover's power mask when the pack has a low
band, since that is how the tail is heard. A band of which the mask leaves
less than a thousandth (125 and 250 Hz) keeps a gain of one, and a band the
others already fill past its due is given none. One draw of the reference
renderer's tail reads within 0.8 dB of its expectation (1.2 dB once in
eight draws of a 0.1 s tail); the engine's reads it to 2e-6. The noise is
not drawn again at each step, or the tail would be a different room twenty
times a second.

**The generator** (`reverberate.render.noise`) is Threefry 2x32 with twenty
rounds: 32 bit additions, rotations and exclusive ors, the same on every
array library, checked against Random123's known answers. The key is
`tail_seed`'s low and high words; the counter is `(index, stream)`.
Direction `d` of bank band `b` reads the stream `45 b + d`: its noise at
`index` is the sum of the eight bytes of the two words, less 1020, over
`sqrt(43690)` (unit variance, normal to an excess kurtosis of -0.15; exact
in integers, where a logarithm would round differently on a card). The
sample `t` of the carrier is the full convolution of the stream with the
band's 512 taps at `t + 511`, over the root of the taps' energy.

**A step's response is the sum of its histograms' responses, each times
the square root of its weight.** That is the interpolation in energy
because the histograms of one step read the carrier from different
**places**: place `p` starts `6 p` bins into the carrier (576 samples, more
than the filter's 511, so two places hold independent noise), and there
are 16. A histogram's place is fixed for the scene: the histograms are
taken in the order of their rows and each takes the lowest place that no
lower row it shares a step with holds (its row modulo 16 if every place is
held, in which case the step's response is multiplied by the one gain that
brings its expected energy back to the weighted sum). The energies of
independent noises add, so the energy of the step is the weighted sum of
the four in every bin, band and direction, and a histogram is the same
noise each time the scene comes back to it.

**Where the tail starts.** `tail_from_s` after the smallest `delay_s` of
the step's rows: the energy of the bins before it is removed before it is
spread. A step with no row starts at its histograms' own first bins, with
nothing removed.

**Between two steps** the outputs of the two steps' responses are
cross-faded linearly, each times its own `high_gain_db`: the quasi-static
rule.

**What this replaced, and what it changes in the samples** (lot L7b,
2026-10). The first engine made one response per step from the four
histograms mixed in energy: 24 bursts of white noise per bin and band, each
on a direction drawn from the density, then the band's filter, 8 transforms
of 45 by 62 500 points for every step whose weights had moved. A walk cost
3.3 s of one core per second of scene and source, and four processes
rendered 0.7 s of stem a second together. The tail is now a different draw of the same law, so it is
compared by its energies, on the `synthetic-density` pack with everything
moving (31 histograms in 5 s, their decay times drawn apart between 0.3 and
0.6 s per band):

- *Expected energy of a step's whole tail, new over old, worst step*: 0.00
  to 0.01 dB in every band from 500 Hz to 16 kHz. In the 250 Hz band, 1.2 dB
  at the worst step and 0.1 dB in the mean: the bank's inverted reading is
  clipped at zero per histogram and no longer once for the mix, and the two
  differ where a band holds less than the ninth of its lower neighbour that
  the bank leaks into it. After the first 100 ms of the tail, for the same
  reason, 0.8 dB at the worst step and band (2 kHz), 0.1 dB in the mean;
  and at 250 Hz there are steps where the old tail held nothing by then and
  the new one holds what one of the histograms is left with.
- *Directions*: at the worst step and band 1 per cent of the energy lies on
  other directions than before (9 per cent at 250 Hz, the same clipping).
- *The rendered signal through the bank*, six draws of each engine, white
  noise in, quarter seconds, first channel, new over old: worst quarter
  second 0.74, 0.69, 0.36, 0.23 and 0.20 dB at 1, 2, 4, 8 and 16 kHz, where
  two halves of the old engine's own draws differ by 0.96, 0.61, 0.50, 0.24
  and 0.52 dB; over the 5 s, +0.18, -0.40, -0.13, +0.05 and +0.06 dB. Under
  the crossover the tail holds nothing to compare.
- *The channels above the first*: the sixteen first channels together, over
  sixteen times the first, read -0.28, -0.35 and -0.19 dB at 1, 2 and 4 kHz
  in the old engine while everything moved, and -0.16, +0.32 and +0.14 dB
  now; at rest both read within 0.1 dB of it. The old engine drew some
  bursts on another direction whenever the weights moved and cross-faded
  the two, which is the likely cost; a histogram's directions no longer
  move.
- *One thing is different by design*: between two histograms the old tail
  was one noise whose envelope moved; it is now two noises cross-faded in
  energy over the 0.80 m that separate them, as the fine structure of a
  real room's tail changes over half a wavelength. Two histograms that hold
  the same energy are not the same samples.
- *The ring of the bank before the tail's first bin* is as before in
  expectation: two bins at 125 Hz and a trace in the third.
- *Against the reference renderer*, where the old engine was one more draw
  of the same law: on the traced room of `tests/test_trace.py` the bank
  reads the engine's tail within 0.30 dB of the reference's draw in every
  band (0.68 before), and on the box of
  `tests/test_scene_pack_integration.py` within 0.41 dB (0.56 before). The
  500 Hz band after the crossover is a residue 30 dB under the 1 kHz band,
  in which two draws of the reference itself differ by up to 2.4 dB.

### `level`

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `high_gain_db` | float32 | `[step]` | what multiplies everything of `early` and `tail` at this step |
| `onset_s` | float64 | `[step]` | where the crossover's coherent window is anchored, on the pack's clock |

`high_gain_db` is `20 log10(alignment_gain)` plus the step's seam: the
`low/seam_db` of the step's pairs, weighted by `low/position_weight` across
source slots and by inverse distance across cells, in decibels. It is the
scalar of `mirror.hybrid.blend`, one per point there, one per step here.
`onset_s` is the step's smallest `delay_s` plus what the wave response's
loudest sample trails the mirror's first arrival by: each pair's
`low/onset_s` less the smallest delay of the mirror's paths at that pair
(the direct path's where there is one, the diffracted onset's where the
pair is shadowed, the straight line's where the mirror finds nothing),
under the same weighting. Neither is interpolated between steps on its own:
each is read at the step and enters the path's gain there (see `early`),
and the tail's.

**At rest**, the source on a solved position and the head on a cell, a step
reads one pair with weight one: `high_gain_db` is
`20 log10(alignment_gain) + seam_db` of that pair and `onset_s` its
`low/onset_s`, which are `blend`'s own gain and the anchor of its window.
Measured on a small room (`tests/test_trace.py`), the pack rendered by the
engine against `render_point` then `blend` of the present pipeline at the
same point, with the same wave response:

| part | what is equal | measured |
| --- | --- | --- |
| the three scalars | `seam_db` to float32, `onset_s` to the sample | 1e-4 dB, exact |
| `low` | sample for sample, 64 channels | 3.2e-5 of the response's peak |
| `early`, under 20 kHz | sample for sample to the engine's approximations | 1.2 per cent of the early part's peak inside the onset window, 0.17 per cent after it; the error's energy -39.5 dB |
| `early`, whole band | as above, plus the engine's taper from 22 kHz | 2.2 per cent, -24 dB |
| `tail` | energy per bank band: it is noise, two draws | 0.7 dB at worst |

The early part's residue is the engine's: its masks are filters of 85 ms
where `blend` multiplies a spectrum of 1.2 s, and a path takes the window's
value at its arrival where `blend` windows the samples.

## `/mirror`, `/crossover`, `/atmosphere`

`/mirror`: dataset `signature`, `float64 [tap]`, the source's minimum phase
signature (`mirror.direct.measure_signature`, 128 taps). Attributes
`lead_s` and `alignment_gain` (`mirror.files.Alignment`: the clock and scale
of the wave field; `lead_s` is the alignment's lead **rounded to a whole
sample at 48 kHz**, 512 samples for 10.6744 ms on hssd_0076, because that is
the shift the mirror's field was written with and the hybrid was validated
at), `lowcut_hz` (40) and `lowcut_order` (8), `tail_from_s`,
`tail_bursts`, `tail_gain_db` (seven), `histogram_bin_s`, `histogram_order`
(3), `receiver_radius_m`, and `settings_json`, the whole
`MirrorSettings.record()`.

`/crossover`: attributes `cutoff_hz` (1000), `width_octaves` (1),
`coherent_s` (0.005), `coherent_fade_s` (0.005): `mirror.hybrid.Crossover`.

`/atmosphere`: attributes `temperature_c`, `humidity_percent`,
`pressure_kpa`, and `enabled`. `low/ir` already carries the air; nothing
else in the pack does.

## `/directivity/<model>`

One group per model a source names.

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `gain_db` | float32 | `[band, angle]` | the pattern, on `bands_hz` |

Attribute `angles_deg`, 0 to 180 in steps of 5: the angle between the
departure direction and the source's facing, `(cos yaw, 0, -sin yaw)` in the
scene frame. A model is a figure of revolution about the facing. **It is
normalised to unit mean power over the sphere in every band**, so a
directional source radiates what the omnidirectional one does, and the tail
and the low band, which are omnidirectional, keep their level. The table is
read linearly in decibels between its angles, and the mean power is that of
the table so read. `omni` is zeros.

`voice_v1` (`mirror.directivity.voice_v1`) is **a parametric fit, to be
confirmed by the owner**: `-back_db (1 - cos angle) / 2` before
normalisation, with `back_db = 2, 3, 5, 7, 10, 14, 18` on the seven bands,
round figures for the front to back difference of speech of the order
reported by Chu and Warnock, *Detailed directivity of sound fields around
human talkers*, NRC Canada, IRC-RR-104, 2002, and, at 8 kHz, by Monson,
Hunter and Story, J. Acoust. Soc. Am. 132 (1), 2012. Their tables were not
reproduced; replacing the figures by them changes the digest and nothing
else. A model's digest, in the recipe's `assets.directivity`, is the
SHA-256 of `gain_db`'s bytes.

## What the engine does with it

Per source, at every output sample, in this order:

1. **Early.** Each living path: the dry signal read at the path's delay,
   through the octave bank with the path's band gains times the
   directivity's (interpolated in angle, when `directivity_enabled`), times
   the air's `exp(-m(f) c delay)`, encoded on the arrival direction.
2. **Tail.** As above, with the air's loss at each bin's time.
3. **High side.** The sum of 1 and 2, through the signature and the low cut,
   times `high_gain_db`, through the crossover's high masks: an arrival at
   pack time `tau` takes `w hi_press + (1 - w) hi_power`, `w` being the
   value at `tau` of `Crossover.onset_window` anchored on `level/onset_s`.
   The tail starts after the window and takes the power mask.
4. **Low side.** The dry signal at `low_sample_rate_hz` convolved with the
   step's responses, crossfaded by `position_weight`, moved to the head by
   `mode`, brought to 48 kHz, times the ratio of the two rates (see the
   scale of `ir`).
5. **Sum**, times the source's `gain_db` and the interval's.

The dry signal of an activity interval is the clip from its offset, faded in
and out over 5 ms inside the interval (a raised cosine), so that a source
never comes on or goes off away from zero; an end on which another interval
of the source starts, to the sample, is not faded, nor is that start
(`render.dry.DryTrack.from_recipe`, and `clip-library.md`).

and the sources are summed, in the order the pack holds them. At rest, with
one source and directivity off, steps 1 to 4 are
`mirror.render.render_point` followed by `mirror.hybrid.blend`.

**What the engine does in another order, and what it costs.**

- *The fixed filters are applied to the dry signal, before the delay line*:
  the octave bank, the signature, the low cut, the crossover's masks, the
  air. A fixed filter and a delay that moves commute to the order of the
  speed over the sound speed, 0.4 per cent, the error the quasi-static rule
  already has; at rest they commute exactly. It is one filtering of a mono
  clip instead of one per output channel.
- *The crossover's masks are filters of 85 ms* (4097 taps under a Hann
  window, from the mask on the 1.2 s grid `mirror.hybrid.blend` uses), a
  mask being a spectrum and a scene having no length to take one over. The
  mask is smoothed by 23 Hz; on the synthetic level B the two sides add
  back to the whole band within 0.5 per cent.
- *The air* on a path is `exp(-m(f) c delay)` exactly at distances 8 m
  apart (further apart beyond 88 m, twelve at most) and linear between two:
  3 per cent of the 20 kHz component at worst, 0.1 per cent at 8 kHz. On the
  tail it is one factor per bin and bank band, the share of the band's
  energy the air leaves at the bin's time.
- *The tail* takes the power mask and has no window: it starts after it.
- *The output does not depend on how it is asked for.* A source is rendered
  in runs of ten steps that start at multiples of ten from the scene's
  start, each from the pack and the dry signal alone. A block, a stem and a
  seek are slices of those runs: blocks of any size give the same samples to
  the bit. The run's length is a setting of the render; another length moves
  the samples by 2e-8 of the peak.

- *What it costs*, on the laptop the audit runs on (ten cores, four of them
  fast), measured by `python -m reverberate.render benchmark --workers 1
  --processes 1,4,6,8,10` on the `synthetic-density` pack: seconds of one
  core per second of scene and source, then seconds of stem rendered per
  second of wall clock by processes of one thread each, a run at a time as
  the audit renders them.

  | scene | early | low | tail | total | 1 process | 4 | 6 | 8 | 10 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | at rest | 0.13 | 0.04 | 0.05 | 0.22 | 5.0 | 14.0 | 16.5 | 17.2 | 16.3 |
  | everything moving | 0.15 | 0.18 | 0.48 | 0.81 | 1.3 | 3.5 | 4.1 | 4.0 | 3.5 |

  Six processes are the most that help: the four fast cores give three
  times one, the six slow ones a little more, and beyond that the processes
  wait on one another. The scene of the size table below is 8 400 seconds
  of stem: 8 minutes at rest, 34 with everything moving. A process holds
  0.7 GB at rest and 1.2 GB moving besides the pack. In the moving scene
  six new histograms a second each cost a response (30 ms), and each of the
  six to eight histograms a run reads costs a convolution (14 ms); a scene
  that comes back to its histograms pays the first once.

The signal is written as `docs/formats/scene-signal.md` says.

## Provenance

`provenance_json`, sorted keys:

| field | meaning |
| --- | --- |
| `recipe_sha256` | again, so the JSON stands alone |
| `assets` | the recipe's `assets`, as found: a trace refuses a mismatch |
| `code_version` | the commit of `reverberate` that traced |
| `solver` | the low band engine and its commit or version |
| `assets_mismatched` | the names of the recipe's asset keys that are not the trace's; empty unless the trace was told to allow them |
| `low_pairs` | how many pairs were read from the cache (`cached`), carried by the bundle (`carried`) and solved (`solved`) |
| `created_utc` | |
| `profile` | what of the recipe was traced: `{"seconds", "sources", "patch", "start_s"}`, all `null`, `false` and `0` for the whole scene |
| `fallback_steps` | how many audible steps no cell could serve by the rule and read the nearest alone |
| `device`, `timings_s` | the card and the seconds of each stage, as the machine saw them |
| `cost` | a list, one record per stage |

A cost record: `stage` (`low`, `paths`, `rays`, `level`, `write`, `check`,
`transfer`, `rental`), `seconds`, `card` (the model's name), `cards`,
`billed_rate_usd_per_hour`, `usd`, `instance`. `paths` holds the diffracted
onsets, which the batched trace computes with the image paths; `check` is
the pack read back and rendered on the host and on the card; `rental` is
what the machine was billed for outside the stages: provisioning, the
bundle's push, the watcher's polls. A cost without its rate is not written
(roadmap constraint 10): the machine does not know its rate, so it writes
`"cost": []` and the laptop that rented it adds the records when the pack is
home (`reverberate.trace.driver.stamp_cost`), which rewrites this attribute
and nothing else.

**A smoke pack** (`profile.seconds` not `null`) holds the first seconds of
the scene and some of its sources, under the whole recipe's digest and
bytes. With `profile.start_s` not zero its step `k` is the scene at
`start_s + k step_s`: it exercises the stages on a stretch where something
moves and is rendered with dry signals given to the engine, not with the
recipe's clips, whose times are the scene's.

**A pair's key** is the first 64 hexadecimal characters of the SHA-256 of
the canonical JSON of: `voxel_low_key`, the source position and the cell's
position (the one asked for) in millimetres as integers, the encoder's
settings, the solver's version, the window in seconds
(`spatial.lowband.pair_key`). `voxel_low_key` is the key of a grid to
1500 Hz, not of a field's low grid to 1 kHz, which stops under the
crossover's ramp. It is the same for every recipe that uses
the pair, which is what makes the dwelling's cache fill once.

## Size

For 20 minutes, `steps = 24 001`, and 14 sources. The per step tables are
fixed; the rest depends on the scene, and the figures below are for a scene
assumed as follows until the first one is measured: each source audible
half the time; 16 arrivals a step (the validated field has a median of 10
and at most 49, `mirror/report_S1_c9.json`); 400 low band pairs and 240
histograms a source.

| part | per unit | per source | 14 sources |
| --- | --- | --- | --- |
| per step tables | 102 B a step | 2.4 MB | 34 MB |
| `early` | 70 B a row | 13 MB | 190 MB |
| `low/ir` | 1.23 MB a pair | 490 MB | 6.9 GB |
| `tail` | 286 kB a histogram | 69 MB | 0.96 GB |
| **the pack** | | | **about 8 GB** |

The rendered order 7 signal of the same scene is 14.7 GB. The pack is
`low/ir`: pairs times 1.23 MB, and the number of pairs is the number of
(source position, cell) couples the scene passes through while the source is
audible. A source at one station heard from 150 cells is 150 pairs; a walk
down a 6 m rail is 75 positions, each heard from one or two cells.

What the trace computes, counted and not priced: 14 sources by 12 000
audible steps is 168 000 path validations, 67 minutes of one RTX 3090 at the
24 ms a pair measured on 2026-09-17 for the lattice pipeline. The batched
trace of L5 and what it is expected to cost are in
[the cost appendix](../adr/0016-appendix-moving-mirror-cost.md).

## What a reader may assume

1. Every per step dataset has `steps` rows; `offsets` has `steps + 1`,
   starts at 0, never decreases, and ends at the row count.
2. Step `k` is the scene at `k step_s` exactly, source and listener at the
   same instant.
3. Where `audible` is false: no rows, `mode` 0, every index `-1`.
4. Where `audible` is true and `has_low`: `mode` is 1, 2 or 3; `pair[k, 0, 0]`
   is valid; `pair[k, 1, :]` is valid exactly when `position_weight > 0`;
   `pair[k, :, 1]` is valid exactly when `mode` is 3. The same for `hist`
   with its two weights.
5. `ir[pair[k, a, b]]` was solved from `pair_position` at `cell[k, b]`.
6. Directions are unit vectors to 1e-6. Gains are finite and not negative.
   `delay_s` is positive and under `low_samples / low_sample_rate_hz`.
7. Rows of a step are sorted by `path_id`, without repeats.
8. Every weight is in `[0, 1]`.
9. `ir` is zero above 1414 Hz to rounding, and every channel of it is on one
   scale: no per channel or per pair normalisation was applied.
10. Nothing in the pack depends on the head's orientation or on any clip.
11. Two traces of one identity agree in every dataset the mirror writes
    (`early`, `tail`): the paths and the histograms are the same on any
    device (ADR 0014). `low/ir` agrees to the solver's own repeatability and
    is not promised to the bit.

## The synthetic profile

`profile = "synthetic-free-field"`: a pack a test builds in memory, with no
trace, whose render is known in closed form. Lot L7 is developed against
it. Its two cells are where the builder is told to put them and its `mode`
is what the builder is asked for: it does not follow the rule a trace
chooses cells by, so that one cell can be measured further than a trace
would read it.

- No room: one source, omnidirectional, at a fixed position; a listener at
  rest or on a straight line at constant speed.
- `has_tail = false`. `/mirror/signature` is `[1.0]`, `lead_s = 0`,
  `alignment_gain = 1`, `lowcut_hz = 0` (no low cut).
  `/atmosphere.enabled = false`. `level/high_gain_db` is 0 everywhere.
- `early`: one row a step, `kind` 0, `order` 0, `path_id` that of the
  direct path (the digest of the single byte `0`), `delay_s = d / c`,
  `gain = 1 / d` in every band, `arrival` the unit vector from listener to
  source, `departure` its opposite, `d` the distance at the step.
- **Level A**, `has_low = false`: the crossover is not applied and the early
  part is the whole band. The output is the dry signal delayed by `d(t) / c`,
  divided by `d(t)`, through the octave bank with equal gains, on the
  harmonics of the source's direction. At rest that is
  `mirror.render.early_signals` of one path convolved with the dry signal.
- **Level B**, `has_low = true`: `/cells` holds two cells 0.40 m apart on
  the listener's line; `low/ir` is the monopole's interior expansion at each
  (`spatial.field.monopole_coefficients`, divided by `i^n` per degree to make
  it the ambisonic signal, scaled by `4 pi` so its far field is `1 / d`,
  on the scale of `ir`)
  through the low **pressure** mask (one arrival lies wholly in the onset
  window, and `level/onset_s` is its delay, so the early part takes the high
  pressure mask and the two add to one) and through a raised cosine high
  pass from 80 to 160 Hz: under it the expansion of a source at `r` grows as
  `(k r)^-(n + 1)` and no float32 holds degree 7. `seam_db` is 0. The
  translated and fused fields at the head are then known from the same
  function at the head itself. A head within `exact_under_m` of a cell is
  mode 1; elsewhere mode 2 from the nearer cell, or mode 3 from both when
  the builder is asked to fuse.
- `/recipe` holds a recipe of the sources and no activity; a test gives the
  engine its dry signal directly.

Measured on level B, source 3 m away, dry noise from 200 to 1300 Hz, the
error's energy over the monopole's at the head, per ambisonic degree 0 to 7,
in dB (`tests/test_render_engine.py`):

| the head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| on a cell, mode 1 | -120 | -120 | -120 | -120 | -120 | -120 | -120 | -120 |
| 0.10 m from one cell, mode 2 | -59 | -56 | -54 | -53 | -44 | -31 | -19 | -9 |
| 0.20 m from one cell, mode 2 | -51 | -49 | -41 | -32 | -23 | -16 | -9 | -4 |
| 0.20 m from each of two, mode 3 | -52 | -54 | -58 | -51 | -45 | -44 | -35 | -23 |

Mode 1 is the filter between the two rates (1e-6 of ripple). **An order 7
expansion moved from one cell keeps its low degrees and loses its top
ones**: what reaches degree 7 at the head comes from degrees the cell does
not hold. Two cells fused recover them to -23 dB. This is the estimator of
this document measured on all 64 channels, which ADR 0016 lists as its
risk 6; a free field says nothing of a room.

A synthetic pack carries `provenance_json` with `"cost": []` and
`code_version` of the test.
