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

## What a reader may assume

1. The frames' file is exactly `frames * channels * 4` bytes.
2. Frame `n` is the scene at `n / sample_rate_hz` seconds; frame 0 is the
   scene's start.
3. The samples do not depend on the size of the blocks they were rendered
   in (`tests/test_render_engine.py`).
4. Stems of one pack rendered with the same settings, summed in float64 in
   the pack's order, are the mix before its conversion to float32.
