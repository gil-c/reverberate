# ADR 0011: The dataset is assembled once, and shared through the store

**Status.** Accepted, 2026-09-10.

## Problem

Opening a flat in the viewer assembled it first. The cost is about a minute per
distinct template: the boolean union of its collision proxy's convex bodies, then
the flood fill of `geometry.carve`. HSSD's 168 scenes place 53 021 pieces drawn from
**15 684 distinct templates**, about **266 hours of one core** for the dataset, and
every flat paid again for furniture the flat next door also owns.

## Decision

**Three caches, each keyed on what it depends on.**

1. `geometry.collider_cache` keeps one template's simulated mesh, keyed on the source
   of `carve.py`, `hssd_assets.py`, `orientation.py` and `outer_surface.py` -- not on
   the whole of `sim_geometry.py`, which is why `outer_surface` has a file of its own.
   It is a property of the template, so every apartment shares it.
2. `viz.scene_pool` keeps render meshes the same way: a symlink into HSSD where the
   dataset is present, the fetched file where it is not.
3. `viz.scene_cache` keeps an assembled storey: a manifest, two shells and symlinks
   into both pools. Its key covers the material tables, since the manifest carries
   each instance's absorption.

**`viz.scene_store` publishes and fetches them**, local then remote then build, the
order `store.py` argues for. A catalogue, `scenes/index.json`, names every storey by
its short name; without HSSD on the machine the viewer takes keys and its list of
apartments from it.

**The pool is built on rented cores**, sharded by template, never by scene, by
`scripts/remote_assemble.py`. The instance reads the dataset through a presigned URL
for one object; no credential leaves the laptop.

## Consequences

All 176 storeys are published. On a machine with no copy of HSSD an apartment opens
in 3 to 15 s cold and at once warm. The pool cost 5.48 USD across ten rentals at
0.055 to 0.344 USD/h.

A change to `viz/scene_manifest.py` or to the material tables invalidates every
storey, which is correct and now cheap: with the pool warm, all 176 re-assemble and
republish in minutes. A change to the four files the pool is keyed on costs the 266
core-hours again.

## Rejected

**Serving the store to the browser directly.** It needs the bucket public or a
signing endpoint, and HSSD is CC BY-NC: a public bucket of its meshes is a licence
question. The Python server fetches and serves; the bucket stays private.

**Hashing each HSSD asset into the pool key.** Reading the glTF to decide whether to
skip reading the glTF gives the saving back, and HSSD's template names are already
hashes of the assets.
