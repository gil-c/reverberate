# The impulse response on disk

Status: current. Implemented by `reverberate.response` and `reverberate.store`.
Decided in [ADR 0005](../adr/0005-storage-format-and-remote-store.md).

A response is written twice, to two files that agree, and the split is not
redundancy.

| file | what it is | who reads it |
| --- | --- | --- |
| `response.sofa` | AES69-2022, `SingleRoomSRIR` convention | anyone, including outside this project |
| `response.h5` | the raw record at the solver's own rate, with typed provenance | this project |

## Units, stated once

Positions in metres. Sample rates in hertz. Durations in seconds. Sound speed in
metres per second. Grid step in metres.

**Impulse responses are pressure in arbitrary units.** The solver's excitation is
not calibrated to a source power, so an absolute level would be a fiction. What
is comparable is the relative level between two responses of the same run, and
that is preserved exactly because no per response normalisation is applied
anywhere in the writer.

## Coordinates

The scene is Y up, as HSSD and glTF are, and every position in this project is
`(x, height, z)`. SOFA is Z up. The conversion is one right handed rotation of
+90 degrees about X:

```
(x, y, z)  ->  (x, -z, y)
```

It lives in `to_sofa_coordinates` and its inverse, is applied only on write and
read of the SOFA file, and is tested both for being its own inverse and for
preserving handedness. Performing it in two places is how a dataset acquires a
mirrored axis that nobody notices until a spatial model is trained on it.

## `response.h5`

HDF5, no compression, schema version in the root attributes.

| path | type | shape | meaning |
| --- | --- | --- | --- |
| `/ir` | float64 | `[receiver, sample]` | pressure, arbitrary units, at the solver's rate |
| `/source_position` | float64 | `[3]` | metres, scene coordinates, Y up |
| `/receiver_positions` | float64 | `[receiver, 3]` | metres, scene coordinates, Y up |

Root attributes: `schema_version`, `sample_rate_hz`, `created_utc`,
`room_volume_m3` when known, and `provenance_json`.

`provenance_json` is `reverberate.response.Provenance` serialised with sorted
keys:

| field | meaning |
| --- | --- |
| `scene_sha256` | digest of the serialised surface list handed to the solver |
| `mats_hash` | the voxelisation cache key |
| `engine` | `cuda` or `cpu` |
| `band` | which of the roadmap's three bands |
| `fmax_hz`, `grid_step_m`, `points_per_wavelength` | the grid |
| `sound_speed_m_s` | as handed to the solver |
| `seed` | the seed that produced the placement |
| `run_id` | the run directory, which is also the key under `runs/` in the store |
| `solver_commit` | PFFDTD commit |
| `billed_rate_usd_per_hour`, `wall_clock_s` | present only for a rented run |
| `notes` | what a reader must be told and no field carries |

`scene_sha256` is the anchor for the roadmap's constraint 9. The viewer renders
the object with that digest, so a picture and a response can be **proved** to be
of the same thing rather than assumed to be.

`billed_rate_usd_per_hour` exists because roadmap constraint 10 says a cost
figure without its hourly rate is not a measurement.

## `response.sofa`

`SingleRoomSRIR`, version 1.0, written through `sofar`.

- `Data.IR` is `[M, R, N]` with `R = 1`. **One receiver becomes one measurement**,
  not one measurement with several receivers. The receivers here are independent
  points in a room, not elements of one rigid array, and SOFA's
  `ReceiverPosition` is relative to the listener: describing six loose points as
  six elements of one listener would claim a rigid geometry that does not exist.
- `ListenerPosition` is `[M, 3]`, the receiver positions in SOFA coordinates.
- `SourcePosition` is `[M, 3]`. The convention requires one per measurement; a
  single static source is tiled.
- `GLOBAL_Comment` carries the same `provenance_json`, because a SOFA reader
  outside this project has nothing else.

`SingleRoomMIMOSRIR` is the right convention once the roadmap's section 8
receiver batching puts several sources in one file. It is deliberately not used
yet: nothing writes several sources into one file, and a convention nobody
exercises is a convention nobody checks.

`SimpleFreeFieldHRIR` is where the simulated head related transfer functions of
roadmap section 7.2 will go.

## Why uncompressed

One response of 1.5 s at 48 kHz float32 is 288 kB; the raw record at the mid
band grid rate of 72.4 kHz in float64 is 869 kB. Projecting to a full dataset of
300 scenes by 80 receivers by 1.5 s at third order ambisonics, 16 channels,
gives about **111 GB**, against a voxelisation cache the roadmap already sizes
at approaching **1 TB**.

FLAC on 24 bit converted room responses typically returns 30 to 50 per cent. At
Backblaze B2's 6.00 USD per TB per month that saves **under 0.30 USD a month**,
and costs exactness on the quantity this project exists to measure. So responses
are stored uncompressed, and this paragraph is the reason.

## Where it lives

Bucket `Clarify` on Backblaze B2, endpoint
`https://s3.eu-central-003.backblazeb2.com`. **The bucket is shared with another
project of the owner's**, so every key this project writes sits under a
`reverberate/` prefix, which `reverberate.store.PREFIX` makes non optional.

```
reverberate/
  vox/<mats_hash>/       sim_consts.h5, vox_out.h5, sim_mats.h5, cart_grid.h5, manifest.json
  ir/<sha256>/           response.sofa, response.h5, manifest.json
  scenes/<sha256>/       the serialised surface list the solver and the viewer share
  runs/<item>/<run_id>/  logs, records, renders, audio
  staging/<sha256>       an object waiting for its size to be confirmed
```

Keys are content addressed, so an upload is idempotent. An object is written
under `staging/` and copied to its real key only once its size is confirmed,
because an interrupted multi gigabyte upload would otherwise leave a truncated
object that `exists()` reports as present and the cache would serve for ever.
Its SHA-256 travels in object metadata and is checked on read. A full digest
check on write would mean downloading a gigabyte back to verify a gigabyte just
sent.

**The remote is the source of truth and the local disk is a read-through cache.**
The lookup order is local, then remote, then compute. That ordering is what makes
a rented GPU instance a first class client of the cache: the alternative, a local
truth mirrored outwards, gives a rented instance nothing to read until somebody
remembers to push.
