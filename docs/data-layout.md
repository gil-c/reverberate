# What is in `data/`, and what it would cost to lose it

120 GB at the time of writing, on a disk with 1.3 TiB free, so **nothing here is
urgent for space**. The point of this page is the other question: which of it is
evidence that exists once, and which is a cache that regenerates itself.

Measured 2026-09-09. `data/` is not versioned; this page is.

| Directory | Size | What it is | If it were lost |
|---|---|---|---|
| `raw/hssd-hab` | 23 GB | The scene dataset. | Re-download. Slow, free. |
| `raw/3d-front-midi` | 27 GB | **Dead.** Referenced only by `viz/mesh_viewer.py`, a standalone viewer; the roadmap never mentions it and ADR 0004 removed the geometric engine that wanted it. | Nothing. |
| `raw/structured3d` | 38 MB | Dead on the same grounds. | Nothing. |
| `raw/hrtf` | 3.3 MB | `HRIR_L2702.sofa`, the measured head. Carries its own CC 3.0 BY-SA licence, which does not compose with this project's. | Re-download. |
| `cache/vox` | 59 GB | Voxelisations, content addressed. **14 of the 16 entries are already in the object store**, which the store module calls the source of truth against a local read-through cache. The two that are not are 0.2 GB local variants. | Re-download or recompute. |
| `cache/carve`, `cache/scenes` | 680 MB | Derived from HSSD and the code. | Recompute. |
| `cache/colliders` | 3.0 GB | One simulated mesh per HSSD template, 15 684 of them, shared by every apartment that places the piece. **The expensive one**: about 266 hours of one core for the whole dataset, which is why it was built on rented machines and published. | Re-fetch from the store, or two days of rented cores. |
| `cache/scene_assets` | 0 B here | The render pool. On this machine each entry is a symlink into `raw/hssd-hab`, so it costs nothing; on a machine without the dataset it holds the 8.5 GB fetched from the store. | Re-fetch, or re-link. |
| `runs` | 10 GB | The measurements. See below. | **Some of it, everything.** |
| `vendor` | 419 MB | PFFDTD source and its virtualenv. | Rebuild. |
| `interim/materials` | 204 KB | Impedance fits per material class. | Recompute. |
| `littrature`, `patches`, `archive` | 350 KB | References and rejected work. | Irreplaceable, trivial to keep. |

## The assembled dataset is in the store

`reverberate/scenes/` holds all 176 storeys as manifests, `reverberate/scene_pool/`
holds the two mesh pools, and `reverberate/scenes/index.json` is the catalogue,
keyed by this project's own names. `reverberate/datasets/hssd-hab.tar` is the
dataset itself, 11.2 GB, put there so a rented machine can read it from a
presigned URL rather than be handed a credential. See ADR 0011.

An apartment therefore opens on any machine without HSSD and without an
assembly. What the local disk keeps is a read-through cache of that, which is
the ordering `store.py` argues for.

## The runs are not symmetrical

Of 37 run directories, **11 are in the object store and 26 exist only on this
laptop**. The unpublished ones are 6.8 GB and include exactly those the roadmap
quotes its measured constants from:

    w34_audit_16k  1766 MB     w29_16k           295 MB
    w39_living_8k  1379 MB     w2_lateral_margin 205 MB
    w10_bedroom_16k 1302 MB    w25_union         137 MB
    w39_audit_4k    299 MB     w3_noise_floor    122 MB

`w3_noise_floor` is where the 10 per cent absolute and 3 per cent differential
error bars of section 9.1 come from. `w25_union` is the T30/T20 of 1.00 to 1.11
that replaced 1.68 to 1.80. Deleting them does not make the roadmap wrong; it
makes it **unfalsifiable**, which for a portfolio piece read by people who build
simulators for a living is worse.

**So the order is: publish, then clear.** Not: clear.

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
