# The scene signal

Status: proposed with [ADR 0016](../adr/0016-a-scene-moves-and-one-engine-renders-it.md).
Written by the signal engine (`reverberate.render.output.write_signal`), read
by the audit's server (lot L8, `open_signal`) and later by training.

The order 7 ambisonic signal at the listener over a whole scene, or one
source's share of it (a stem). It is a cache like the pack: its identity is
the pack's, the sources summed and the engine's settings, and it is
regenerated, not archived.

Two files side by side:

| file | what |
| --- | --- |
| `<name>.f32` | the frames one after the other, each its 64 channels in ACN order, little endian float32; frame `n` starts at byte `256 n` |
| `<name>.json` | the header below |

## Conventions

Those of [`scene-pack.md`](scene-pack.md): ACN, N3D, 48 kHz, order 7. The
signal is the field at the centre of the head **in the scene's fixed
ambisonic frame**. The head's rotation is not in it: the decoder applies
`/listener/orientation`, where the page already does it.

Twenty minutes are 57 600 000 frames and 14.7 GB.

## The header

| key | type | meaning |
| --- | --- | --- |
| `schema`, `schema_version` | str, int | `"reverberate.scene-signal"`, `1` |
| `data` | str | the name of the frames' file, beside the header |
| `dtype`, `layout` | str | `"<f4"`, `"frames of channels"` |
| `sample_rate_hz` | float | `48000` |
| `order`, `channels`, `frames` | int | `7`, `64`, the scene's samples |
| `ordering`, `normalisation`, `frame` | str | `"ACN"`, `"N3D"`, the scene's fixed frame |
| `recipe_sha256` | str | the pack's identity |
| `sources` | list | the ids summed, in the pack's order; one id for a stem |
| `peak` | float | the largest absolute sample |
| `sha256` | str | the digest of the frames' file |
| `complete` | bool | `true` |

The header is written last, to a scratch name, and renamed. **A signal
without its header is a render that stopped**, and a reader refuses a
header whose `frames` and `channels` do not give the file's size.

## Why two plain files and not HDF5

The pack is HDF5 because its tables are ragged and read a source and a
block at a time. The signal is one rectangle written once from its start to
its end and read by time range, and for that:

- frames are appended as the engine yields them, one block in memory, with
  no chunk cache and no index to rewrite;
- a time range is one seek and one read, so a server streams it with
  `numpy.memmap` or with a byte range and needs no library to do it;
- the checksum the audit compares, the page's stream against the engine's
  signal, is the digest of the file itself, computed while it is written;
- a WAV cannot hold it: its size field ends at 4 GB.

Uncompressed, for the reason of `impulse-response.md`.

## The audit's stems

The audit's server (`reverberate.viz.audit_stems`) keeps one signal per
source of a pack, so that the page mixes any of them at once. Each is this
format with three things added, because it is rendered a chunk at a time and
out of order, the chunk under the play cursor first:

```
<cache>/<pack sha256, 16 hex>/<source id>/<key, 16 hex>/
    key.json      what the key is the digest of
    stem.f32      the frames, at their place from the start
    chunks.jsonl  one line per chunk rendered
    stem.json     the header above, written once every chunk is there
```

- **A chunk is one run of the engine**, `chunk_steps` steps: half a second,
  24 000 frames, 6.1 MB. The engine renders a run from the pack and the dry
  signal alone, so a stem's chunks put end to end are a one-shot
  `Engine.stem` of the source, byte for byte, in whatever order they were
  rendered.
- **`stem.f32` has its full length from the first chunk and is sparse.** A
  chunk in which the engine returned only zeros is recorded and not written:
  it stays a hole and reads as the zeros it is. A stem takes the disk of the
  time its source sounds, 12.3 MB a second.
- **`chunks.jsonl`** has one JSON object per chunk: `chunk`, `sha256` (of the
  engine's samples as float32 frames, taken before they are written),
  `frames`, `peak`, `silent`, `levels_db` (the omnidirectional channel's level
  per step) and `seconds` (what the render took). A chunk is in the cache
  when its line is; a last line cut short is a chunk to render again.
- **The key** is the SHA-256 of `key.json`: the digest of the pack's bytes,
  the source, a digest of the engine's code, the settings that change a
  sample (`chunk_steps`, `direction_nodes`, and `workers`, which the cache
  fixes at 1 because the transforms' thread count moves the last bits),
  whether the source's directivity is applied, and the digest of what the
  source was fed. The owner's directivity switch therefore moves the stems
  of the sources it concerns and no other. A header's `stem_key` repeats it.

A mix served to the page is the chosen stems summed in float64 in the pack's
order and brought back to float32: the engine's mix to within the rounding
of each stem to float32 (rule 4 below), and a single stem untouched.

## The sound check

`python -m reverberate.render check <pack.h5> [--recipe R.json] [--clips DIR]
[--manifest M.json] [--out DIR] [--window START STOP] [--sources ...]
[--measured-head H.sofa] [--reference FIELD.h5] [--reference-point]
[--probe-seconds S]`
(`reverberate.render.check`) renders what it needs through the engine and
holds it to what a listener would reject. It writes, in `--out`:

- `check.json`: every result (`test`, `source`, `status`, `value`, `unit`,
  `threshold`, `reason`, `note`, `detail`), the limits with their reasons,
  the levels at the listener and the files written;
- `check.md`: the same as tables, the faults first, in the order a listener
  meets them;
- `listen/mix.wav` and `listen/<source>.wav`: the two ears, 48 kHz, 24 bit,
  for the scene's own head, through the page's decoder
  (`spatial.binaural.design_decoder` on the measured head, turned by the
  head's matrix, in blocks of 512 with a crossfade when the head moves, as
  `viz/app/scene/sound-decode.js`), at the page's default level, a gain of
  one. When the mix peaks under -26 dB re full scale a second set is
  written, `*_plusNNdB.wav`, every file by the same whole tens of decibels;
- with `--reference-point`, these alone: `reference_point.md` and
  `reference_point.json` (`reverberate.render.check.reference`). For a pack
  whose source stands within 5 cm of the validated field's own, with its
  directivity off, and whose listener rests within 5 cm of a lattice point
  of that field for 1.2 s while it sounds: the pack rendered there against
  the field's response, both on the field's clock and scale. Per third
  octave, the image paths (from 5 ms before the first arrival to 1 ms
  before the tail starts): the pack's level over the field's and the
  energy of their difference over the field's; the level of the first
  50 ms; the same two for the whole response under the crossover; and per
  octave the level after 50 ms and T20. It refuses a pack whose source is
  elsewhere: that comparison is the check's own `late_spectrum_reference`,
  which cannot tell the source's place from the pack;
- `impulse.png`, `continuity.png`, `levels.png` where `matplotlib` is
  installed.

It exits 1 when a test fails. A status is `PASS`, `WARN`, `FAIL`, `INFO`
(told, not judged) or `SKIP` (nothing to read: a source at rest has no
Doppler shift).

Three families of tests. **Impulse probes**: the dry signal is one sample,
at one instant a source at rest and three a source that moves, so what is
rendered is the scene's response. **Continuity**: two steady tones (400 Hz
and 2.5 kHz, one each side of the crossover), then pink noise, over the
stretch of the source's audible time in which the most changes in the pack.
**The mix**: the recipe's clips, over the window.

| test | what is read | passes to | warns to |
| --- | --- | --- | --- |
| `arrival_time` | first arrival over the crossover against `lead_s` plus the distance over the sound speed | 0.1 ms | 0.5 ms |
| `band_alignment` | first arrival under the crossover against over it | 0.5 ms | 1 ms |
| `direct_level` | the direct sound, 2 to 8 kHz, against the convention of `clip-library.md`: the source's gain and pattern over the distance | 3 dB | 6 dB |
| `direct_band_balance` | the direct sound under the crossover against over it, each re the convention | 3 dB | 6 dB |
| `direction` | the intensity vector of the direct sound against the geometry | 3 degrees | 10 degrees |
| `binaural_left_right` | the head turned to put the source 60 degrees left, then right: which ear leads, which is louder | near ear leads by 0.25 to 0.9 ms and by 3 dB | signs right |
| `binaural_front_back` | the ears' level difference per third octave, source 30 degrees front left, against the decoder's own front and back | nearer the front, within 2 dB | nearer the front |
| `pre_arrival_energy` | energy more than 1 ms before the first arrival, over the whole | -40 dB | -30 dB |
| `late_echo` | where the level per 100 ms rises by 6 dB: what it rises to, re the loudest | -60 dB | -40 dB |
| `seam_third_octaves` | third octaves 630 Hz to 1.6 kHz against the line through 315-500 Hz and 2-3.15 kHz; `:whole` and `:early` (first 50 ms, warns only) | 3 dB | 6 dB |
| `late_spectrum_reference` | third octaves of the part after 50 ms, 250 Hz to 4 kHz, less their mean, against the validated field's at the nearest lattice point | 4 dB | 8 dB |
| `reverberation_reference` | T20 per octave after 50 ms, 250 Hz to 4 kHz, against the same | 20 per cent | 35 per cent |
| `reverberation_plausible` | T20 per octave | 0.15 to 1.0 s | (warns outside) |
| `c50` | early to late per octave, 500 Hz to 4 kHz, a source seen within 2.5 m | 0 to 40 dB | (warns outside) |
| `duplicate_arrivals` | rows of one step with one delay, direction and gain under two names | none | fails when the pair passes the direct sound |
| `tone_click` | what `x[n] - 2 cos(w) x[n-1] + x[n-2]` leaves of each tone, on W, X, Y, Z and the two ears, re the tone | -50 dB | -40 dB |
| `noise_click` | largest sample difference of the noise over the local rms of the differences | 6.5 | 8 |
| `level_step` | a tone's level between a 10 ms frame and the next but one | 1 dB | 3 dB |
| `zipper` | lines 20 Hz and 2 Hz apart round a tone, 6 dB over what lies between | -40 dB | -26 dB |
| `doppler` | the 2.5 kHz tone's frequency per 100 ms against the radial speed (warns only in a room) | 20 per cent | 50 per cent |
| `level_distance` | slope of the noise's level against the distance, where it changes by a quarter | -26 to 0 dB a decade | -30 to +10 |
| `level_at_listener` | active speech level (P.56) of a voice, level while it sounds of a noise, less the clip's at 1 m over the distance | 6 dB | 12 dB |
| `audibility` | the same, A weighted | 20 to 85 dBA, a near voice 45 to 80 | from 10 dBA |
| `binaural_peak`, `playback_level` | the two ears' peak at the page's default level | -30 to -3 dB re full scale | -40 to 0 |
| `dc` | the mean over the rms | -40 dB | -20 dB |
| `silence` | largest sample where the recipe gives the source nothing, re its peak | -90 dB | -60 dB |
| `interval_edge` | the dry signal's first and last sample of an interval, re its rms | -40 dB | -20 dB |
| `speech_to_rest` | a voice over everything else while it speaks, A weighted | told | |

The reasons are in `reverberate.render.check.run.LIMITS` and are printed in
`check.md`. The false alarm rates are stated there: Gaussian noise passes
6.5 once in 1.2e10 samples; a steady tone leaves -250 dB; a crossfade over
one step from nothing to full leaves -58 dB.

**Placeholders.** A recipe generated without a clip library names
`placeholder` clips. The check then plays `voice_NN` from the `NN`-th
speaker of the manifest (in the order of the speakers' names) and
`noise_<subtype>` from that subtype's clips, each set end to end in the
order of the clips' names, looped, read at the interval's `clip_offset_s`;
and says so beside the source. Such an interval is not cut on an utterance.

**What it does not tell.** Timbre and naturalness. Whether the validated
field is itself right. A fault where no probe was sent. Faults each under
its limit that add up. Anything of the headphones.

## What a reader may assume

1. The frames' file is exactly `frames * channels * 4` bytes.
2. Frame `n` is the scene at `n / sample_rate_hz` seconds; frame 0 is the
   scene's start.
3. The samples do not depend on the size of the blocks they were rendered
   in (`tests/test_render_engine.py`).
4. Stems of one pack rendered with the same settings, summed in float64 in
   the pack's order, are the mix before its conversion to float32.
