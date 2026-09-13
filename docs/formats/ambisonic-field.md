# The ambisonic field on disk

Status: current. Written by `reverberate.experiments.w40_volume_field.assemble`.
One file per source, `field/<source>.h5`, and one index beside them,
`walk.json`. The response at one point follows
[`ambisonic-response.md`](ambisonic-response.md) in every convention
(ACN, N3D, frame, transform); this document covers the grid.

## `field/<source>.h5`

| dataset | shape | meaning |
| --- | --- | --- |
| `ir` | `[point, channel, sample]` float32 | the order 7 response at every listening point, 64 channels at 48 kHz, 1.2 s |
| `positions` | `[point, 3]` | scene coordinates `(x, height, z)` of the listening point, ear height |
| `point_index` | `[point]` | index into the plan's points, for joining with `plan.json` |
| `cell_index` | `[point, 3]` | integer cell of the listening grid; `grid_origin_m`, `grid_step_m` and `grid_shape` in the attributes place it |
| `rooms` | `[point]` bytes | the room the point stands in, by the rules of ADR 0010 |
| `direct_path_m` | `[point]` | source to point distance, so a consumer can align the direct arrival between neighbours |
| `has_high` | `[point]` bool | the point has a solved high band; otherwise its top octaves are synthesised from 0.8 x the mid band's `fmax` |
| `solved_to_hz` | `[point]` | the frequency the response is solved to (8000 with a high band, 3200 without) |
| `high_dropped` | `[point]` bool | the point had a high band that failed the level check against its mid band and was dropped |
| `low_borrowed` | `[point]` bool | the point had no low band array of its own and uses its nearest neighbour's, levelled onto its own mid band |

Attributes: `order`, `ordering` (ACN), `normalisation` (N3D), `sample_rate_hz`,
`axes`, `directivity` (omni), `source_id`, `source_name`, `source_room`,
`source_position`, `scene_id`, `dwelling`, `gain` (1.0), `mid_on_high_gain`
(the median gain that put the mid band on the high band's scale, so points
without a high band sit at the same level), `high_dropped_count`,
`low_borrowed_count`, `absorption_for_tail`, and `provenance_json` (the run,
the cache keys per band, the encoder settings).

The three bands are levelled on each other by the ratio their source
bandwidths predict, so the level is the mid band's own; a consumer that plays
several sources at once applies the same `gain` to all.

## `walk.json`

```json
{
  "dwelling": "hssd_0076",
  "scene_id": "104862621_172226772",
  "height_m": 1.7,
  "pitch_m": 0.4,
  "sources": [{"id": "S1", "name": "...", "room": "...", "position": [x, y, z],
               "directivity": "omni", "field": "field/S1.h5"}],
  "voxel_cache_keys": {"low": "...", "mid": "...", "high": "..."},
  "meshes": {"1000": "audit/1000/voxels", "4000": "audit/4000/voxels", "8000": "audit/8000/voxels"},
  "meshes_scene_id": "104862621_172226772"
}
```

`voxel_cache_keys` names the grids the field was solved on, in the vox cache
and on the store. `meshes` names the audit view the campaign built from each
grid, one per band, relative to the run: the tiered payload of ADR 0007
(`experiments.audit_view`, a `rooms.json` and a directory per room, each room
in two tiers of tiles). The room the reader stands in is the grid itself, one
cube per boundary node, coplanar faces of one material merged into one
rectangle and nothing else; every other room is decimated: four nodes a
cube at 8 kHz, two at 4 kHz, none at 1 kHz (`aggregation` on each tier says
how many, `1` being the grid itself). `meshes_scene_id` is
the scene they were built from, so a viewer can refuse a mesh of another
scene. A campaign that built no audit view writes `{}` and `null`; the app
then opens the dwelling in its colour view, with the sound.
