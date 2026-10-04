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

## What a reader may assume

1. The frames' file is exactly `frames * channels * 4` bytes.
2. Frame `n` is the scene at `n / sample_rate_hz` seconds; frame 0 is the
   scene's start.
3. The samples do not depend on the size of the blocks they were rendered
   in (`tests/test_render_engine.py`).
4. Stems of one pack rendered with the same settings, summed in float64 in
   the pack's order, are the mix before its conversion to float32.
