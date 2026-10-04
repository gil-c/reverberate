# The scene recipe

Status: proposed with [ADR 0016](../adr/0016-a-scene-moves-and-one-engine-renders-it.md).
Written and validated by `reverberate.scenes` (lot L2), drawn by the
application (L3), read by the trace stage (L6). Nothing reads it yet.

A recipe is one JSON document of a few kilobytes that says everything a moving
scene is: the dwelling, where sources may stand and travel, what each source
does over time, what the listener does, the air, and the seed and versions
that make it reproducible. It holds no audio and no acoustics. It is the only
thing of a scene that is stored; the pack
([`scene-pack.md`](scene-pack.md)) and the rendered signal are regenerated
from it.

## Conventions, stated once

| quantity | choice |
| --- | --- |
| frame | the scene's own, as every field of this project: `(x, y up, z)`, right handed |
| lengths, positions | metres |
| time | seconds from the start of the scene, `0 <= t <= duration_s` |
| angles | degrees |
| yaw | about the up axis; `0` faces scene `+x`; positive is counter clockwise seen from above, so `+90` faces scene `-z` |
| pitch | positive looks up |
| roll | positive lowers the right ear |
| order of the three | yaw, then pitch about the head's own left axis, then roll about its own front axis |
| levels | decibels, pressure (`20 log10`) |

The yaw is the head's own, not the field's: a head at yaw `psi` hears the
field rotated by `-psi` (`spatial.sh.rotate_yaw`). In the ambisonic frame
(`x` front, `y` left, `z` up, reached by `spatial.sh.scene_to_ambisonic` and
no other function) the head's orientation is
`R = Rz(yaw) Ry(-pitch) Rx(roll)`, which for a zero roll is the matrix of
`viz/app/audio/sh.js`, `headMatrix`. The page has no roll today; a recipe may
carry one and a decoder that cannot apply it must say so.

A height is a distance above the storey's floor, `dwelling.floor_y_m`. Every
position in a recipe is nevertheless written whole, `[x, y, z]` with
`y = floor_y_m + height`, so no reader has to add anything.

## Serialisation and identity

**The canonical form** is
`json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`
encoded as UTF-8, followed by one line feed. **The recipe's identity is the
SHA-256 of those bytes**, written `recipe_sha256` wherever it is quoted. A
recipe on disk may be indented for reading; its identity is always computed
on the canonical form of what it parses to.

**Numbers are quantised before they are written**, so that Python's shortest
representation gives the same characters on every machine:

| quantity | step |
| --- | --- |
| positions, lengths, heights | 0.001 m |
| times | 0.001 s |
| angles | 0.01 degree |
| levels | 0.01 dB |
| speeds | 0.001 m/s |

A value that is a whole number of its step is written as the float it is
(`12.5`, `1.0`), never as an integer; counts, seeds and versions are integers.

**Determinism.** The same seed, the same generator parameters, the same asset
keys and the same generator version give the same canonical bytes, in two
processes under different `PYTHONHASHSEED`. Every random draw comes from a
generator seeded with the first eight bytes of
`sha256("<seed>:<label>")`, the label naming what is drawn (`"source:v1"`,
`"listener"`), so adding a source does not move the others.

## Top level

| key | type | meaning |
| --- | --- | --- |
| `schema` | string | `"reverberate.scene-recipe"` |
| `schema_version` | int | `1` |
| `dwelling` | object | which storey |
| `assets` | object | the versions of everything the recipe was generated against |
| `seed` | int | `0 <= seed < 2^53` |
| `duration_s` | float | length of the scene |
| `output` | object | `{"order": 7, "sample_rate_hz": 48000}`; ACN and N3D are not options |
| `atmosphere` | object | the air |
| `heights` | object | `{"standing_m": 1.7, "seated_m": 1.2}` |
| `stations` | array | where a source or the listener may be at rest |
| `rails` | array | the lines sources travel on |
| `sources` | array | what emits |
| `listener` | object | who hears |
| `generator` | object or `null` | what produced the recipe, with its parameter ranges |

Unknown keys are refused at every level: a recipe that says something the
reader does not understand is not rendered.

### `dwelling`

| key | type | meaning |
| --- | --- | --- |
| `name` | string | this project's name, `hssd_0076` (`geometry.scene_ids`) |
| `scene_id` | string | the HSSD scene, `104862621_172226772` |
| `floor_y_m` | float | the storey's floor, `storey.floor_height` of `geometry.apartment` |

### `assets`

| key | type | meaning |
| --- | --- | --- |
| `export_sha256` | string | digest of the storey's export, `apartment_full.json` followed by `manifest.json`, as `mirror.pipeline.derive_scene` computes it |
| `voxel_low_key` | string | cache key of the grid the low band is solved on (`campaign.json`, `bands.low.cache_key`) |
| `mirror_scene_key` | string | `DerivedScene.key` |
| `calibration_key` | string | `mirror.parameters.Parameters.key` |
| `directivity` | object | `{"<model>": "<sha256 of its table>"}` for every model a source names |
| `rooms_rule` | string | `"adr-0010"`, the rule that named the rooms |

A recipe whose keys do not match the assets a trace finds is refused, by key
name. Clips are not listed here: each activity interval carries its own
clip's digest.

### `atmosphere`

| key | type | meaning |
| --- | --- | --- |
| `temperature_c` | float | must equal the temperature the low band was solved at, 20.0 today |
| `humidity_percent` | float | relative humidity, `0` to `100` |
| `pressure_kpa` | float | `101.325` at sea level |

The three fields are `reverberate.audio.Atmosphere`'s. The temperature is not
free because the wave solve fixes the sound speed
(`experiments.run.sound_speed`, 343.2 m/s at 20 C) and a mirror at another
speed would not meet it at the crossover. Humidity and pressure only enter
the air absorption, which is post processing (ADR 0009), and are free.

### `stations`

A station is a place where somebody can be at rest.

| key | type | meaning |
| --- | --- | --- |
| `id` | string | unique in the recipe, `[a-z0-9_]+` |
| `kind` | string | `"stand"`, `"seat"` or `"waypoint"` (a rail junction where nobody stays) |
| `position` | `[x, y, z]` | the mouth or the ears: `y = floor_y_m + heights.standing_m` for `stand` and `waypoint`, `floor_y_m + heights.seated_m` for `seat` |
| `height` | string | `"standing"` or `"seated"`; restates the kind so a reader need not know the rule |
| `room` | string | the room's name by the rules of ADR 0010 |
| `facing_yaw_deg` | float | the natural facing: a seat's front; for a stand, any |
| `object` | string or absent | the HSSD instance a seat belongs to |

**A seat is an exact listening position.** When the listener sits there, the
trace computes the low band on an array centred on the seat itself and
translates nothing (ADR 0016). A seat also has a standing position straight
above it, at `heights.standing_m`, where a rail may end.

### `rails`

A rail is a polyline on the free floor joining two stations. Rails that share
a station form the graph sources travel on.

| key | type | meaning |
| --- | --- | --- |
| `id` | string | unique |
| `a`, `b` | string | the station at each end |
| `points` | array of `[x, z]` | the polyline, first point at `a`, last at `b`, at least two |
| `pitch_m` | float | `0.08`: the spacing of the positions the low band is solved at |

A rail is walked standing: every point of it is at
`y = floor_y_m + heights.standing_m`, and it ends above a seat, not on it.
**The rail's sampled positions** are at arc lengths `j * pitch_m` from `a`,
`j = 0 .. floor(L / pitch_m)`, and at `b` itself; they are a function of the
recipe and nothing else, so the low band cache can be keyed on them.

Every seat has an implicit vertical rail between its seated and its standing
position, sampled at the same pitch from the seated end, on which a source
rises and sits.

### `sources`

| key | type | meaning |
| --- | --- | --- |
| `id` | string | unique |
| `kind` | string | `"near_voice"`, `"far_voice"` or `"noise"` |
| `subtype` | string or absent | for a noise, what it is: `"appliance"`, `"television"`, `"music"`, `"water"`, `"street"`, `"other"` |
| `directivity` | object | `{"model": "voice_v1", "enabled": true}`; `{"model": "omni", "enabled": false}` for a noise |
| `gain_db` | float | applied to everything the source emits |
| `turn_rate_deg_s` | float | how fast the source turns towards a new facing |
| `segments` | array | where the source is, over the whole scene |
| `activity` | array | when it emits, and what |

`kind` is a label carried to the pack and to training. What makes a voice
near or far is the generator's distance range, recorded under `generator`,
not a rule of the format.

`directivity.enabled` is the owner's switch: off, the source is rendered
omnidirectional whatever the model, as the render validated by ear. The
models and their normalisation are in `scene-pack.md`.

**`segments`** are in time order, touch end to start, and cover
`[0, duration_s]`. Each has `type`, `start_s`, `end_s` and:

| `type` | other keys | where the source is |
| --- | --- | --- |
| `"dwell"` | `station`, `height`, `facing` | at the station; `height` is `"seated"` only at a seat |
| `"travel"` | `rail`, `from`, `to`, `profile`, `facing` | on the rail, from one end station to the other |
| `"rise"` | `station`, `to` | on the seat's vertical rail, towards `to` = `"standing"` or `"seated"` |

Along a rail of length `L` the arc length is `L e(tau)`, with
`tau = (t - start_s) / (end_s - start_s)` and `e(tau) = tau` for
`profile = "constant"`, `3 tau^2 - 2 tau^3` for `"smoothstep"` (peak speed 1.5
times the mean). A `rise` is a smoothstep. The speed is therefore derived
from the two times and the rail, and never stored beside them.

`facing` is one of `{"mode": "fixed", "yaw_deg": v}`, `{"mode": "travel"}`
(along the rail's tangent) or `{"mode": "listener"}` (towards the listener's
position at that instant). The source's yaw is the facing's value at the
scene's start, and from there moves towards the facing's current value by
the shorter way round at no more than `turn_rate_deg_s`. A source has no
pitch and no roll.

**`activity`** is a list of intervals in time order that do not overlap:

| key | type | meaning |
| --- | --- | --- |
| `start_s`, `end_s` | float | when the source emits |
| `clip` | object | `{"library": "ears", "name": "<clip_id>", "sha256": "<digest of the clip's bytes>"}` |
| `clip_offset_s` | float | where in the clip the interval starts |
| `gain_db` | float | on top of the source's own |

**Audio is never embedded.** A clip is named and pinned by the digest of its
bytes as fetched (`clarify_library.fetch_clip`); a reader that finds other
bytes under the name refuses the recipe. A clip is mono; one at another rate
is resampled to the output rate by the engine's loader. The clip must be at
least `clip_offset_s + end_s - start_s` long.

### `listener`

| key | type | meaning |
| --- | --- | --- |
| `interpolation` | string | `"linear"` |
| `keyframes` | array | the trajectory |

Each keyframe:

| key | type | meaning |
| --- | --- | --- |
| `t_s` | float | strictly increasing; the first is `0`, the last `duration_s` |
| `position` | `[x, y, z]` | the centre of the head |
| `yaw_deg`, `pitch_deg`, `roll_deg` | float | the head; yaw is not wrapped, so `350` to `370` is a turn of 20 degrees |
| `station` | string or absent | the seat or stand the listener is at, exactly |

Between two keyframes the position is linear in time and each angle is
linear in time. Linear, because a spline may leave the free floor between two
points that are both on it, and then no rule below could be checked on the
keyframes alone. A smooth walk is a dense one: a curve is written as the
keyframes it needs.

Two consecutive keyframes naming the same `station` are a rest there. A
listener is seated only at a seat.

### `generator`

`null` for a recipe written by hand. Otherwise:

| key | type | meaning |
| --- | --- | --- |
| `name` | string | `"reverberate.scenes"` |
| `version` | string | the generator's own version; a change of output for one seed changes it |
| `parameters` | object | every range a draw was taken from |

`parameters` is a tree whose leaves are scalars or two element `[min, max]`
ranges, with the unit as the key's suffix. It is recorded so the audit can
show what the scene was drawn from, and so the same parameters and seed
give the recipe back. The trace and the signal engine do not read it. The
keys the first generator writes, nested at each dot:

```
sources.near_voice.count, sources.far_voice.count, sources.noise.count     [min, max]
sources.near_voice.distance_m, sources.far_voice.distance_m                [min, max], to the listener, at rest
sources.speed_m_s, sources.dwell_s, sources.gain_db                        [min, max]
sources.seated_share, sources.speech_s, sources.pause_s                    [min, max]
listener.speed_m_s, listener.rest_s, listener.turn_rate_deg_s              [min, max]
listener.seated_share                                                      [min, max]
rails.pitch_m, rails.max_length_m                                          scalar
```

## Validation

A recipe that breaks one of these is refused with the rule's number.
`W = 0.25 m` and `F = 0.22 m` are `WALL_SETBACK_M` and `FURNITURE_CLEARANCE_M`
of `experiments.w40_volume_field.plan`, whose `free_floor` is the floor
meant here.

1. **Shape.** Every key above is present with its type, and no other.
2. **Time.** `duration_s > 0`. Every source's segments cover the scene
   without gap or overlap. Activity intervals lie inside the scene and do not
   overlap. Keyframes start at `0`, end at `duration_s` and increase.
3. **Stations.** Unique ids. A `stand` or a `waypoint` lies on the free floor
   and in the room it names. A `seat` lies within 0.60 m of the footprint of
   its `object` and at least 0.30 m from any wall.
4. **Rails.** Both ends are stations, and the first and last points are their
   `x, z`. Every segment of the polyline lies on the free floor, except its
   last 0.60 m towards a seat. `pitch_m` is `0.08`.
5. **Sources on the graph.** A `dwell` is the first segment or follows one
   that ends at its station. A `travel` goes from one end of its rail to the other. A `rise`
   is at a seat; a `dwell` there is `seated` after a rise to `seated` and
   `standing` otherwise.
6. **Listener on the floor.** Every keyframe, and so every point between two,
   is on the free floor, unless both keyframes name the same seat. The head's
   height above the floor is between `heights.seated_m` and
   `heights.standing_m`, and equals `heights.seated_m` only at a seat.
7. **Speeds.** A source's peak speed on a rail is at most 1.5 m/s, a rise
   takes at least 1.0 s, the listener's speed between two keyframes is at
   most 1.5 m/s and the head turns at most 360 degrees a second.
8. **Clearances.** At every instant a source's mouth and the listener's head
   are at least 0.50 m apart, and two sources at least 0.40 m.
9. **Collisions.** No two sources travel the same rail segment in opposite
   directions at the same time.
10. **Assets.** Every clip's digest is 64 hexadecimal characters; every
    directivity model named has a digest under `assets.directivity`; the
    atmosphere's temperature is the low band's.
11. **Canonical numbers.** Every number is a multiple of its step.

Rules 6 to 9 are checked on the trajectories sampled every 50 ms, the pack's
step, and at every keyframe and segment boundary.

## Example

A complete recipe of twenty seconds: a near voice that speaks from an
armchair, then rises and walks to the kitchen; a television; a listener who
walks towards the voice and turns to it. Indented here; its identity is that
of its canonical form. The positions are illustrative and were not checked
against the dwelling.

```json
{
  "schema": "reverberate.scene-recipe",
  "schema_version": 1,
  "dwelling": {
    "name": "hssd_0076",
    "scene_id": "104862621_172226772",
    "floor_y_m": 0.0
  },
  "assets": {
    "export_sha256": "3b5d0c1e9a7f42a68d1c0e5b7f9a2c4e6d8b0a1c3e5f7092b4d6f8a0c2e4f601",
    "voxel_low_key": "90222a626eb68ed0008f3005f63b4d44",
    "mirror_scene_key": "455c538ec2a6a74db2f0f2381f285f6e",
    "calibration_key": "c3cec6aab28bb582",
    "directivity": {
      "voice_v1": "7a1f3c5e9b2d4f60817a3c5e9b2d4f60817a3c5e9b2d4f60817a3c5e9b2d4f60"
    },
    "rooms_rule": "adr-0010"
  },
  "seed": 20261004,
  "duration_s": 20.0,
  "output": {"order": 7, "sample_rate_hz": 48000},
  "atmosphere": {"temperature_c": 20.0, "humidity_percent": 50.0, "pressure_kpa": 101.325},
  "heights": {"standing_m": 1.7, "seated_m": 1.2},
  "stations": [
    {
      "id": "armchair",
      "kind": "seat",
      "position": [-10.4, 1.2, -11.15],
      "height": "seated",
      "room": "living room-kitchen-entryway",
      "facing_yaw_deg": 90.0,
      "object": "armchair_3"
    },
    {
      "id": "counter",
      "kind": "stand",
      "position": [-7.2, 1.7, -12.35],
      "height": "standing",
      "room": "living room-kitchen-entryway",
      "facing_yaw_deg": 180.0
    },
    {
      "id": "tv",
      "kind": "stand",
      "position": [-12.6, 1.7, -9.0],
      "height": "standing",
      "room": "living room-kitchen-entryway",
      "facing_yaw_deg": 0.0
    }
  ],
  "rails": [
    {
      "id": "armchair_counter",
      "a": "armchair",
      "b": "counter",
      "points": [[-10.4, -11.15], [-9.1, -11.6], [-7.2, -12.35]],
      "pitch_m": 0.08
    }
  ],
  "sources": [
    {
      "id": "v1",
      "kind": "near_voice",
      "directivity": {"model": "voice_v1", "enabled": true},
      "gain_db": 0.0,
      "turn_rate_deg_s": 180.0,
      "segments": [
        {
          "type": "dwell",
          "station": "armchair",
          "height": "seated",
          "start_s": 0.0,
          "end_s": 9.0,
          "facing": {"mode": "listener"}
        },
        {"type": "rise", "station": "armchair", "to": "standing", "start_s": 9.0, "end_s": 10.5},
        {
          "type": "travel",
          "rail": "armchair_counter",
          "from": "armchair",
          "to": "counter",
          "profile": "smoothstep",
          "start_s": 10.5,
          "end_s": 15.5,
          "facing": {"mode": "travel"}
        },
        {
          "type": "dwell",
          "station": "counter",
          "height": "standing",
          "start_s": 15.5,
          "end_s": 20.0,
          "facing": {"mode": "fixed", "yaw_deg": 180.0}
        }
      ],
      "activity": [
        {
          "start_s": 1.0,
          "end_s": 8.0,
          "clip": {
            "library": "ears",
            "name": "p004/sentences_02_regular",
            "sha256": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
          },
          "clip_offset_s": 0.0,
          "gain_db": 0.0
        },
        {
          "start_s": 11.0,
          "end_s": 19.0,
          "clip": {
            "library": "ears",
            "name": "p004/sentences_02_regular",
            "sha256": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
          },
          "clip_offset_s": 7.0,
          "gain_db": 0.0
        }
      ]
    },
    {
      "id": "n1",
      "kind": "noise",
      "subtype": "television",
      "directivity": {"model": "omni", "enabled": false},
      "gain_db": -12.0,
      "turn_rate_deg_s": 180.0,
      "segments": [
        {
          "type": "dwell",
          "station": "tv",
          "height": "standing",
          "start_s": 0.0,
          "end_s": 20.0,
          "facing": {"mode": "fixed", "yaw_deg": 0.0}
        }
      ],
      "activity": [
        {
          "start_s": 0.0,
          "end_s": 20.0,
          "clip": {
            "library": "noise",
            "name": "television/news_01",
            "sha256": "a0b1c2d3e4f5061728394a5b6c7d8e9fa0b1c2d3e4f5061728394a5b6c7d8e9f"
          },
          "clip_offset_s": 3.5,
          "gain_db": 0.0
        }
      ]
    }
  ],
  "listener": {
    "interpolation": "linear",
    "keyframes": [
      {"t_s": 0.0, "position": [-8.0, 1.7, -9.5], "yaw_deg": 200.0, "pitch_deg": 0.0, "roll_deg": 0.0},
      {"t_s": 4.0, "position": [-8.0, 1.7, -9.5], "yaw_deg": 215.0, "pitch_deg": -5.0, "roll_deg": 0.0},
      {"t_s": 8.0, "position": [-9.4, 1.7, -10.4], "yaw_deg": 215.0, "pitch_deg": -15.0, "roll_deg": 0.0},
      {"t_s": 20.0, "position": [-9.4, 1.7, -10.4], "yaw_deg": 300.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    ]
  },
  "generator": {
    "name": "reverberate.scenes",
    "version": "0.1.0",
    "parameters": {
      "sources": {
        "near_voice": {"count": [1, 1], "distance_m": [0.5, 1.5]},
        "far_voice": {"count": [0, 0], "distance_m": [2.0, 8.0]},
        "noise": {"count": [1, 1]},
        "speed_m_s": [0.4, 1.2],
        "dwell_s": [4.0, 60.0],
        "gain_db": [-12.0, 0.0],
        "seated_share": [0.3, 0.7],
        "speech_s": [2.0, 12.0],
        "pause_s": [0.5, 6.0]
      },
      "listener": {
        "speed_m_s": [0.3, 1.2],
        "rest_s": [2.0, 30.0],
        "turn_rate_deg_s": [20.0, 180.0],
        "seated_share": [0.0, 0.0]
      },
      "rails": {"pitch_m": 0.08, "max_length_m": 12.0}
    }
  }
}
```
