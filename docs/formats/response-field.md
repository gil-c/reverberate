# The impulse response field

One HDF5 per source, named by `walk.json` at the root of a run
(`reverberate.viz.app_payload`). Reader and checker: `reverberate.viz.field_payload`;
`python -m reverberate.viz.field_payload <file>` says what the app would refuse.
Agreed with the producer session on 2026-09-10.

Frames: positions in the scene's own frame, metres, `(x, y up, z)`; ambisonic
channels in the frame of `spatial.sh.scene_to_ambisonic`, x front, y left,
z up. ACN ordering, N3D normalisation; anything else is refused by name.

| dataset | dtype | shape | meaning |
| --- | --- | --- | --- |
| `/ir` | float32 | `[position, channel, sample]` | the whole response at each cell, 48 kHz |
| `/positions` | float64 | `[position, 3]` | the cell's real centre |
| `/cell_index` | int32 | `[position, 3]` | `(i, j, k)` on the lattice; a refused cell is absent |
| `/direct_path_m` | float64 | `[position]` | source to cell distance |
| `/rooms`, `/has_high`, `/solved_to_hz` | str, bool, float | `[position]` | optional: the room, whether the high band was solved there, and above which frequency the response is synthesised |

Root attributes: `order`, `ordering = "ACN"`, `normalisation = "N3D"`,
`sample_rate_hz`, `source_id`, `source_position`, `directivity`, `gain` (one
gain for the whole source, never one per cell), `grid_origin_m`, `grid_step_m`,
`grid_shape`, `scene_id`, `dwelling`, `provenance_json`. A lattice of one
layer has `grid_shape[1] = 1` and may carry a zero step in y.

**Storage.** The page reads a cell by an HTTP range request straight out of
the file, so `/ir` must be uncompressed little-endian float32, either
contiguous or chunked as one chunk per position, `(1, channels, samples)`.
At order 7 a second of response is 12.3 MB per cell.

**What the page does with it.** `field_payload.build_site` writes an
`index.json` (everything above except the samples, plus every cell's byte
offset) beside a link to the file. The page splits a response at 150 ms: the
part before turns with the head at every update, the rest when a cell is
entered and again, exactly, once the head has been still (ADR 0012).
