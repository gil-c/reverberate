# What is in `data/`, and what it would cost to lose it

313 GB at the time of writing. The disk went down to about 100 GB free before
this measurement, which is why everything below was published. The point of
this page is still the other question: which of it is evidence that exists once,
and which is a cache that regenerates itself. Since 2026-11-03 the answer to the
first is "nothing in `runs`, `cache/vox` or `external`": every file there is also
in the object store, checked by size and digest.

Measured 2026-11-03 (the previous measurement, 2026-09-09, had 120 GB). `data/`
is not versioned; this page is.

| Directory | Size | What it is | In the store | If it were lost |
|---|---|---|---|---|
| `raw/hssd-hab` | 23 GB | The scene dataset. | As `datasets/hssd-hab.tar`. | Re-fetch from the store, or re-download. |
| `raw/hrtf` | 3.3 MB | `HRIR_L2702.sofa`, the measured head. Carries its own CC 3.0 BY-SA licence, which does not compose with this project's. | No. | Re-download. |
| `cache/vox` | 72 GB | Voxelisations, content addressed, 25 entries. | **All 25**, every file. | Re-fetch, or recompute. |
| `cache/carve`, `cache/scenes` | 1.2 GB | Derived from HSSD and the code. | Scenes as manifests. | Recompute. |
| `cache/colliders` | 3.0 GB | One simulated mesh per HSSD template, 15 684 of them. **The expensive one**: about 266 hours of one core. | Yes. | Re-fetch from the store, or two days of rented cores. |
| `cache/scene_assets` | 0 B here | The render pool. On this machine each entry is a symlink into `raw/hssd-hab`. | As `scene_pool/`. | Re-fetch, or re-link. |
| `cache/low-pairs`, `cache/scene_player` | 4.4 GB | Caches of the clarify simulator. | No. | Recompute. |
| `runs` | 146 GB | The measurements. See below. | **All 44 run directories**, every file, including `w40_volume_field`, which is no longer on the disk. | Re-fetch. |
| `external` | 63 GB | Published measurements (Motus, RAF, BRAS) and the calibration's state. | **Everything**, every file. | Re-fetch. |
| `vendor` | 419 MB | PFFDTD source and its virtualenv. | No. | Rebuild. |
| `clips`, `voices` | 815 MB | Listening material. | Voices as `voices/`. | Re-record or re-fetch. |
| `interim/materials` | 204 KB | Impedance fits per material class. | No. | Recompute. |
| `littrature`, `patches`, `archive` | 350 KB | References and rejected work. | No. | Irreplaceable, trivial to keep. |

`raw/3d-front-midi` and `raw/structured3d`, dead on the grounds the previous
version of this page gave, are gone.

## Where it sits in the store

Every key is under `reverberate/`, because the bucket is shared. A file keeps
its path under `data/`, with one rename:

    data/runs/<run>/<path>          ->  runs/<run>/<path>
    data/cache/vox/<key>/<file>     ->  vox/<key>/<file>
    data/external/<path>            ->  external/<path>

Each object carries its SHA-256 in the `sha256` metadata, which `put_file` writes
and `get_file` checks. Two run files had changed locally after their first
publication, `w20_first_listen/report.json` and `w37_three_band/plan.json`. The
published version was left as it was, and the local one sits next to it as
`<name>.local-<first 12 hex of its digest>`.

Each backed-up item also has a manifest, `manifests/<item>.json`: every file's
path under `data/`, its key, size and digest, and any symlink the directory
held. An item is one run directory (`runs/<run>`), one voxel entry
(`vox/<key>`), or one entry of a published source (`external/<source>/<entry>`).
The loose files at the top of `runs/` are `runs/top-level_files`.

The bucket keeps object versions, and a lifecycle rule deletes a hidden version
one day after it is hidden. Publishing goes through `staging/` and a copy, so a
large publication briefly holds its bytes twice; that settles within two days.
In the store, after this publication: `runs` 177 GB, `vox` 77 GB, `external`
68 GB, `scene_pool` 12 GB, `datasets` 11 GB, 344 GB in all, about 2 USD a month.

## Bringing an item back

From the main checkout:

    PYTHONPATH=src .venv/bin/python scripts/restore_data.py runs/w40_volume_field
    PYTHONPATH=src .venv/bin/python scripts/restore_data.py --list

The script reads the manifest and fetches each file through the store, so the
digest is checked before the file takes its name. A file already present with
the right size is skipped, so an interrupted restore resumes.

## What was removed locally, and what was not

Removed on 2026-11-03, after every file was found in the store with the same
size and digest:

| Item | Size | Restore |
|---|---|---|
| `runs/w40_volume_field` | 19 GB | `PYTHONPATH=src .venv/bin/python scripts/restore_data.py runs/w40_volume_field` |

The walk app lists only the runs it finds on the disk, so `w40_volume_field`
no longer appears in it until it is restored.

Everything else stays, **although it is in the store**, because something reads
it from the disk with no fallback to the store:

- `runs/w41_field_hssd_0076` is the run `walk.toml` opens first.
- `runs/w42_gpu_hssd_0076` holds the fields that `w44_clarify_interpolation`
  and `w45_clarify_scene` name by absolute path, and the meshes the
  `w42_gpu_hssd_0076_hybrid` walk points into.
- `runs/w44_clarify_interpolation` and `runs/w45_clarify_scene` are the
  clarify simulator's work in progress.
- `cache/vox`: `entry_from_key` in `experiments/run.py` opens an entry by key on
  the disk and treats absence as an error. `fetch_entry` exists, but only the
  voxelise lookup calls it. Giving `entry_from_key` the same fallback would free
  about 65 GB; the three entries `w44_clarify_interpolation` cites would stay.
- `external`: the scripts on the real-data branch read `motus/sh_rirs.zip`,
  `motus/meta`, the BRAS archives and `raf/rir` from the paths given on their
  command line. `raf/rir` (43 GB) is raw archive parts whose result is already
  `raf/decay_*.json`; it is the next candidate.
- Runs under 100 MB, and every run the roadmap quotes its constants from: they
  cost nothing and they are the evidence.

## The runs are not symmetrical

Every run directory is now in the object store. The ones the roadmap quotes its
measured constants from were, on 2026-09-09, only on this laptop:

    w34_audit_16k  1766 MB     w29_16k           295 MB
    w39_living_8k  1379 MB     w2_lateral_margin 205 MB
    w10_bedroom_16k 1302 MB    w25_union         137 MB
    w39_audit_4k    299 MB     w3_noise_floor    122 MB

`w3_noise_floor` is where the 10 per cent absolute and 3 per cent differential
error bars of section 9.1 come from. `w25_union` is the T30/T20 of 1.00 to 1.11
that replaced 1.68 to 1.80. Deleting them does not make the roadmap wrong; it
makes it **unfalsifiable**, which for a portfolio piece read by people who build
simulators for a living is worse. They are kept locally as well.

**So the order is: publish, verify, then clear.** Not: clear.

## Starting the simulations over

ADR 0010 changed what a room is, so every room-scoped result is against the old
definition. That is a reason to redo them, and it is *not* a reason to delete
the old ones: the comparison between the two definitions is itself a
measurement, and `w39_living_8k` is the only record of what the old one gave.

Of the voxel cache, the four `room_only` entries -- 4.4 GB, formerly named
`bedroom_only` -- are keyed on geometry the new rules no longer produce. They
are dead for future runs. The `apartment_full` entries are storey geometry and
survive, except where a scene has a `porch/terrace/deck/driveway` region, which
`102344022` does not.

## Naming

The exported models are `apartment_full.json` and `room_only.json`. The second
was `bedroom_only.json` and held, in W39, a living room; see ADR 0010.

**Old run directories keep their names.** The roadmap cites them as W3, W25,
W34, and renaming them would break every reference for no gain.
