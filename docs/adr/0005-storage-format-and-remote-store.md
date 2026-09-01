# 5. Store impulse responses as SOFA beside a raw record, on a shared B2 bucket

Date: 2026-09-01

Status: accepted

## Context

Three things were missing at once, and they turn out to be one decision.

**No declared response format.** The solver writes `sim_outs.h5` at the grid rate
in its own index space, and the trail stopped there. The roadmap's first binding
requirement is "the full impulse response, not a decay envelope", spatially
resolved and re-decodable to arbitrary microphone positions, and nothing said
what that is on disk.

**No shared cache.** A voxelisation costs minutes of CPU and of the order of a
gigabyte per room per band. It "currently lives per worktree with no sharing
mechanism", so a rented GPU instance cannot reach one, which is how B0 came to
spend 52.3 minutes of a GPU rental on voxelisation, about 80 per cent of that
bill buying CPU on an idle card.

**No place for either.** The roadmap sizes the cache at approaching a terabyte.

The owner's Backblaze B2 credentials are already in KeePassXC and already reach a
bucket that another of the owner's projects uses.

## Decision

**SOFA, AES69-2022, `SingleRoomSRIR`, is the canonical interchange format**, with
a raw internal HDF5 beside it carrying the solver's own sample rate and a typed
provenance block. Field by field in
[`docs/formats/impulse-response.md`](../formats/impulse-response.md).

**Backblaze B2 is the source of truth and the local disk is a read-through
cache.** Lookup is local, then remote, then compute; a computed voxelisation is
published. Keys are content addressed. An object lands under `staging/` and is
copied to its real key only once its size is confirmed.

**Everything this project writes sits under a `reverberate/` prefix**, because
the bucket is shared.

## Consequences

A rented instance becomes a first class client of the cache instead of a special
case, which is the direct fix for B0's 80 per cent.

The dataset is readable outside this repository without importing a line of it.

The store is one more moving part that can fail, and its failure mode is a
truncated object that looks present. The staging prefix and the size check are
the answer, and there is a test with a deliberately truncating client that proves
a short upload is never promoted.

Egress from B2 is free only up to three times the average monthly stored bytes.
At the current scale that is not binding, and it is worth a line in the ledger
rather than a mechanism.

## What was rejected

**Bare WAV plus a JSON sidecar.** Cheap and immediately playable, but it makes
the response and its provenance two files that can be separated, has no place for
receiver geometry, and is not what the field reads.

**SOFA only.** SOFA has free-form global attributes, so provenance could live
there as a string. Burying the cache key, the seed and the billed rate in a
string is what makes `reverberate audit` impossible, and the roadmap asks for
every response to be traceable to the exact mesh, orientation derivation,
material assignment, solver settings, seed, cached voxelisation and billed hourly
rate.

**A compressed codec, FLAC or otherwise.** The arithmetic is in the format
document: at the projected full dataset size a lossless codec saves under 0.30
USD a month against a cache that costs a hundred times more, and buys a
conversion step on the quantity this project exists to measure.

**`b2sdk`, the native Backblaze backend.** The sibling project's local notes say
the S3 compatible API rejects this key and only the native backend works. **That
note is stale**: a listing against the live bucket succeeded with the current
application key before this module was written. Recorded here because it would
otherwise send the next reader down a second client library for no reason. A
marked, opt-in test asserts it rather than leaving it as a paragraph.

**A local truth mirrored outwards.** It gives a rented instance nothing to read
until somebody remembers to push, which is the problem restated rather than
solved.

## Note on this file's neighbours

`docs/adr/0001` through `0004` are **not present on `main`**, although the
roadmap cites `docs/adr/0004-geometric-engine-survey.md` as the record of the
geometric engine decision. ADR 0001 exists on an unmerged branch. The numbering
here continues the roadmap's, so this file does not collide with them if they
land. **The missing ADRs are a real gap and are reported rather than
reconstructed.**
