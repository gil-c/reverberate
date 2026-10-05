# What every stage of the chain costs against what it needs

Date: 2026-10-05

Status: **an audit by reading, counting and small laptop measurements; no
card was rented.** Written for lot L22 of ADR 0016, after the signal engine
was found to take 3 h 26 for what about two minutes should do. Every figure
says where it comes from: the two whole-scene ledgers
(`docs/adr/0016-appendix-trace-cost.md`), the first scene's own files
(`lowband.json`, `as_computed.npz`, the pair cache, the bundle), the code, or
a count made for this note on the laptop (section 12). What a card must
still say is listed with its price in section 11. Nothing here changes a
rendered number; the remedies are for the lots that follow.

## 1. The answer

The reference job is one scene of twenty minutes in hssd_0076: 14 sources,
1529 audible low band source positions, 16 887 pairs, 831 cells, 202 tail
sites. It was billed 11.68 USD and 8.45 h on 8 x RTX 3090 (run A). With what
is merged since and not yet run whole it is predicted at 8.6 USD and 6.2 h.

**Nine tenths of that bill is the wave solve, and the wave solve is not
flagrant in its kernel: it moves bytes at 55 % of what the card can move.**
It is flagrant in what it is asked to do. Five findings, the first two the
ones the owner singled out:

1. **The ray tracer is 130 to 1300 times its floor** (section 3). A site is
   9.9 million ray segments, and the card makes 1.0 million a second. The
   mirror's scene is 1 478 245 triangles, not a few thousand; each segment
   tests 140 of them in double precision, on a card whose double precision
   rate is one sixty-fourth of its single. A single precision tracer with a
   proper hierarchy makes 1e8 to 1e9 segments a second on that card. In USD
   it is small on one recipe (0.13 USD of 8.6); it is the largest ratio of
   the chain.
2. **The walls cost 55 % of a step because they are 51 % of the bytes a
   step moves** (section 4): a lossy node moves 278 bytes a step and a node
   of air 14. The kernel is not at fault; the state is. Updating the lossy
   nodes inside the stencil's pass removes 15 % of a step with the same
   bits, and a material needs 7 branches under 1500 Hz, not 11 and not 3,
   which removes 13 % more with a proof.
3. **Five recipes of one dwelling ask for 7857 solves, of which 2534 are
   distinct** (section 6, measured here). Two recipes share 70 % of their
   source positions, because sources live on the dwelling's rails and
   stations, and share 1 % of their pairs, because the listener's cells are
   its own. Today each recipe solves every position again. A dwelling
   solved as one campaign costs a third at five recipes and a sixth at ten.
   **This is the largest lever of mass production and it needs no new
   physics.**
4. **The fit after a solve is 13 to 37 times its floor** (section 5): 0.26
   to 0.74 s a pair, 10 066 card seconds a scene, to filter and resample a
   thousand records in double precision, one thread a record, before a fit
   that is itself a product of spectra. Done in the spectra it is 0.02 s.
5. **Money that buys no computation** (section 7): an engine compiled on
   every rental that the default solver never runs (216 to 498 s), a disk
   of 219 GB for 53 GB used (0.83 and 1.75 USD), eight cards billed while
   one laptop line brings the pack home (0.9 USD predicted, 1.93 billed),
   cards idle at the end because the launches go in the positions' order
   (0.21 USD), and three hosts that never answered kept seven minutes each
   (0.27 USD). Together 2 to 3.5 USD of an 8.6 USD scene.

On the laptop, beside the signal engine (lot L20): the stems are rendered
through the 47 % of the scene where a source is silent, the sound check is
the engine's speed and nothing else, and the test suite runs in one process
under coverage.

**After the plan of section 9, the reference scene on its validated grid is
predicted at 3.3 USD and 2.5 h on the same machine, and at about 0.55 USD a
recipe when a dwelling's ten recipes are solved together.** With the two
listening decisions that are the owner's (the coarser grid, the rail
interpolation) it is 1.3 USD and one hour alone, and about 0.2 USD a recipe
in mass.

## 2. The table of all stages

Measured: run A unless said (8 x RTX 3090, 1.382 USD/h billed, 0.173 USD a
card hour, 3.84e-4 USD a machine second). "Now" is the prediction for the
code as merged (the queue, the compiled paths, the compact pack), which no
whole scene has run. A floor is what the operation needs on that hardware,
computed in the section named. Risk is to the rendered result: none (same
bits, or nothing rendered is touched), proof (a comparison must pass), ear
(the owner's to hear).

### On the cards

| stage | what it is | measured | floor | ratio | cause | remedy | gain | risk | effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wave stepping | 47.4 M nodes x 32 769 steps x 1529 positions | 141 160 card s, 6.78 USD, 88 s a position | 61 s a position at 80 % of the card's memory rate; 44 s with the remedies of section 4 | 1.45; 2.0 | 55 % of the memory rate; 51 % of the bytes are the walls' state | sections 4 and 5 | 16 % same bits; 28 % with 7 branches; 50 % at the floor | none, then proof | weeks |
| the number of solves | one a source position a recipe | 1529 (to 1789) a recipe | 2500 to 2700 a dwelling, whatever the recipes | 3.1 at five recipes | each recipe solved alone | one campaign a dwelling (section 6) | 68 % at five recipes, 83 % at ten | none | days |
| the fit | 984 records of 32 769 samples to 64 channels of 4800 | 10 066 card s, 0.48 USD, 0.26 to 0.74 s a pair | 0.02 s a pair | 13 to 37 | double precision filters walking each record, a resampling of about 850 weights a sample on a thousand records, then the fit | filters and resampling inside the fit's spectra (section 5) | 9500 card s, 0.46 USD, 20 min | proof | days |
| idle cards at the end | the last launches are the longest | 4440 card s, 0.21 USD | half the shortest launch a card | 4 | launches in the positions' order | longest launches first, the short ones last | 0.15 USD, 6 min | none | hours |
| rays | 100 000 rays a site, 99 segments a ray | 12.8 to 14.5 s a site; 2600 card s now, 0.13 USD (1436 s of 8 cards as run, 0.55 USD) | 0.01 to 0.1 s a site | 130 to 1300 | section 3 | section 3 | all of it | none, then proof | 1 to 2 weeks |
| voxelise, grid cut, fit's operator | once a dwelling | 43 s, then 18 s and about 15 s a worker process | 60 s once | 1 to 8 | each worker cuts its own grid and prepares its own operator | cut once and share the file; operator kept beside the grid | 0.05 USD | none | hours |

### On the machine's cores

| stage | what it is | measured | now | floor | ratio | remedy | risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paths | 45 864 positions through an image tree of 208 000 candidates | 1076 s, one process, 0.41 USD | about 250 core s, under the solves | 20 s (9e8 sieve tests) | 12 | none needed: it is free while cards solve | none |
| onset distance fields | up to 1274 Dijkstra fields on 484 532 cells | 64 s a scene | same | once a dwelling | 1 a recipe | key a field by its cell, not by the scene's positions | none |
| level | one scalar a pair: wave against mirror over 707 to 1414 Hz | 1417 s, 0.54 USD | 25 ms a pair, 420 core s | 0.1 ms a pair: the octave's energy read on the arrivals and the histogram | 250 | closed form (section 5); free under the solves as it is | proof |
| tails' tables | a site's histograms weighted to each source's cells | 342 s on each resume | same | 0 on a resume | n/a | an existence check before rebuilding (`trace/run.py`, `_tail_job`) | none |
| rows and write | 20.7 GB of `low/ir` | 153 s serial, 0.06 USD | 240 core s and 170 s serial | 40 s at 1 GB/s, once | 4 | rows written in the pack's form, not as `.npy` read again | none |
| the pair file | 64 x 4800 float32 | read whole by the level, again by the rows | same | once | 2 | the level's channel 0 beside the file, or both in one job | none |

### Orchestration and transfers

| stage | measured | floor | ratio | cause | remedy | gain a rental | risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rental to first line | 489 s (0.19 USD), 804 s on run B | 170 s: ssh at 109 s, wheels 45 s, push 15 s | 3 to 5 | PFFDTD compiled, 216 to 498 s; the default solver never runs it | skip the build unless `--low-engine pffdtd` or `--low-scheme fcc`; then an image with the interpreter | 0.08 to 0.19 USD, 4 to 8 min, every rental and every smoke | none |
| hosts that never answer | 3 x 420 s, 0.27 USD | the ready time's ninth decile | 2 to 3 | one timeout for all hosts | record every host's ready time; time out at 180 to 240 s | 0.1 to 0.15 USD | none |
| a failure | 826 s waiting (0.32 USD), stages redone | 30 s | 25 | seen at the 300 s poll, relaunched by the driver | the machine's own loop relaunches at once; the watcher reads a kept connection every 30 s | 0.3 USD a failure | none |
| the disk | 219 GB rented, 53 GB used: 0.83 USD (A), 1.75 (B) | 90 GB, 60 with the compact pack | 3.6 | a field campaign's formula (`gpu/onebox.py`) | size it from the plan: pairs, tails, pack, a margin | 0.6 to 1.3 USD | none |
| the pack home | 5027 s, 1.93 USD (as run); 2400 s, 0.91 USD (compact, proxy) | 60 s to the store on the datacentre's line; the laptop then reads the store with no machine billed | 40 | the machine waits for the laptop's line | direct route (merged: 3 to 8 min); then the store | 0.7 to 0.9 USD | none |
| the end of the run | 150 s mean of a 300 s poll | 10 s | 15 | the poll | a done marker read on a kept connection | 0.05 USD | none |
| the bundle up | 215 MB, 15 to 19 s; minutes on a slow uplink | under 1 MB a recipe | 400 in bytes | 214 of 215 MB are the dwelling's (88 MB of models, 126 MB of mirror) | the dwelling's part on the store, fetched by the machine once a dwelling | seconds; matters at N recipes | none |
| the pair cache home | 12 351 of 18 219 pairs in 6 h at 0.3 to 0.5 MB/s | 0: the audit does not read it | n/a | brought by default on a whole scene | to the store, compact (621 kB a pair), or left | 20 GB and the wait | none |
| the pack's size | 9.5 GB | 3.5 GB with `decay=60` | 2.7 | an owner's listening decision | `low-band-compact.md` | 6 GB a recipe | ear |

### On the laptop

| stage | measured | floor | ratio | cause | remedy | risk |
| --- | --- | --- | --- | --- | --- | --- |
| the stems | 3 h 26 for 14 x 20 min | about 2 min | 100 | lot L20 | lot L20 | L20's |
| the stems' silent half | 47 % of the source seconds are inaudible (179 652 audible steps of 336 014) and are rendered, then not written | 0 | 1.9 | `viz/audit_stems.py` asks the engine for every chunk | skip a chunk whose steps are all inaudible in the pack's own table | none |
| the stem cache | 119 GB on disk, 206 GB logical, hashed whole at the seal; lost when any of 16 engine files changes | none kept, if the mix renders faster than it plays | n/a | a cache sized for a slow engine | render the chosen sources by block on demand once L20's engine holds 2 x real time for 14 sources with margin; else keep it without the whole-file hash | none |
| a chunk of mix | 14 files of 6.1 MB read and summed in double precision each half second, 172 MB/s of disk | the same sum in single precision, or none (on demand) | 2 | per-source stems mixed on each request | as above | none |
| opening a pack | SHA-256 of 9.5 GB on the request's thread; up to 8 copies of the early tables in memory | a digest of the header and the tables' own digests | 20 s to 0 | the file's bytes as its name | the pack's recorded digest | none |
| `packs()` | a recursive listing a request, twice a second | once, then on a change | n/a | no memo | memo with the directory's time | none |
| the tracks | 4.8 MB of text, 756 000 numbers, not compressed, never cached | 0.4 MB: only what moves, as binary, compressed | 12 | every source at every tenth of a second | a source at rest once; `gzip` | none |
| the sound check | 14 min for 60 s of 14 sources | 20 s on ten cores after L20 | 40 | 1200 to 1300 source seconds through one thread of the engine; 57 to 85 engines built; the page's decoder run a source, not on the mix; silent sources rendered | one engine, sources in processes, decode the mix once | none |
| the worklet | 375 M operations a second in JavaScript at order 7 | the same | 1 | n/a | nothing: it is a twentieth of a core | n/a |
| recipe, plan | 4 s, 5 s | 1 s | 5 | Python loops a step and a source, `assign` three times | nothing now: linear in the scene's length (30 s for two hours) | n/a |
| the test suite | 380 to 620 s here, 6.5 to 11 min in CI, where the rule is 60 s | 60 to 90 s on ten cores | 6 | one process, coverage traced on every run, whole traces in fixtures | section 8 | none |

## 3. The ray tracer

### What it does

`mirror/kernels.py`, `RAYS_KERNEL`, launched by `mirror/engine.py`: one card
thread a ray, the whole of a ray's life in that thread. A ray leaves the
site, and at each bounce:

1. **walks a uniform grid of 0.10 m cells** (`walk_cells`, a 3D DDA) from its
   position towards the end of its reach, and in each cell tests every
   triangle listed there by Moller and Trumbore, in double precision, until a
   hit precedes the next cell's entry;
2. **walks the same cells again** over the segment found, for the receivers'
   spheres listed in them (0.20 m of radius, one a tail cell: 53 on the
   scene, binned in the occluders' grid), and for each sphere entered adds
   the ray's energy to an integer histogram: 8 bands, and 16 moments a band
   (order 3), by 137 atomic additions of 64 bits;
3. absorbs, draws specular or Lambert from a 64 bit hash (two rounds of
   `mix64` a uniform, 53 bits), and goes on until every band is under
   1e-6 of its start or 411.8 m are travelled (1.2 s).

All 53 cells are in one launch, and a site's histograms are traced once and
serve every step and every pair. That part is right. A site's result is 32.6
MB of 64 bit integers brought to the host, turned to double precision and
saved uncompressed: 35 MB a site, 7 GB a scene.

### What it costs, counted

Measured here on the scene's own mirror (`bundle/trace/mirror/scene`), by
following 40 rays of site 100 with the kernel's own walk:

| | |
| --- | --- |
| triangles of the occluders | 1 478 245; the median one is 2.3 cm across |
| grid | 136 x 33 x 152 cells; 132 615 hold triangles; 25 a cell in the mean, 8 in the median, 49 at the ninth decile, 4074 at most |
| segments a ray | 99 (98 in the median, 116 at most); every ray ends by the energy floor |
| mean segment | 1.80 m |
| cells walked a segment | 28 |
| triangles tested a segment | 140 |
| a site of 100 000 rays | 9.9e6 segments, 2.8e8 cells, 1.39e9 triangle tests |

A site takes 12.8 to 14.5 s on an RTX 3090 with the card to itself (the plan's
model is 10 s and 0.155 s a tail cell). So the card makes **about 1.0e6
segments a second and 1.4e8 triangle tests a second.**

### Why

- **Double precision on a consumer card.** An RTX 3090 computes 35.6e12
  single precision operations a second and 0.556e12 double: one
  sixty-fourth. 1.39e9 tests of about 60 operations are 8.4e10 operations,
  0.15 s at the card's full double rate. The two Tesla P100 of the first
  smoke, whose doubles run at half their singles, traced a site in about 6
  s beside its upload: the older card was faster.
- **The card runs at about a seventieth of even that rate**, because of how
  the work is laid on it. A warp is 32 rays, each in its own cell with its
  own number of triangles, and it advances at the pace of its slowest: with
  8 triangles in the median cell and 4074 in the largest, the 32 lanes wait
  for the unlucky one at every cell, for 99 bounces. Each test reads 72
  bytes of a triangle that no neighbouring lane reads.
- **140 tests a segment where 3 to 6 are needed.** A uniform grid cannot be
  fine enough for furniture modelled at 2 cm and coarse enough for the air
  of a room: a ray crosses 28 cells and meets cells of dozens of triangles
  near every surface. A triangle that spans several cells is tested in each.
- **1.48 million triangles describe detail the rays cannot use.** The tail
  is what remains after the image tree's three orders: diffuse, late energy
  above the crossover. Its scattering is a coefficient a material, not the
  2 cm facets of a plant.

It is not the random numbers (one hash of two rounds a draw, a few a
bounce), not the atomics (a ray enters a sphere on a few segments in a
hundred), and not the host: the kernel is one launch a site.

### The floor

A bounding volume hierarchy over 1.5 million triangles is 21 levels; an
incoherent ray visits 30 to 60 nodes and tests 2 to 6 triangles. Aila and
Laine (2009, "Understanding the efficiency of ray traversal on GPUs")
measured 40 to 140 million diffuse rays a second on a GTX 285 with a
software traversal in single precision on scenes of 0.2 to 1 million
triangles; an RTX 3090 is ten to twenty times that card, so **5e8 to 2e9
segments a second in software, and more through the card's own ray tracing
units (OptiX)**. Kept conservative at 1e8 to 1e9: a site is 0.01 to 0.1 s,
and a scene's 202 sites are 2 to 20 s of one card where they are 2600 s.
**Ratio 130 to 1300.**

### Remedies, in the order to take them

| step | what | a site | a recipe (202 sites) | risk and its proof |
| --- | --- | --- | --- | --- |
| 0 | today | 12.8 s | 2600 card s, 0.125 USD | |
| 1 | a hierarchy in place of the grid, still double precision; ties between two triangles at one distance broken by the lower index, in the twin too | about 1.5 s (140 tests to 5; lanes of a warp do the same work) | 300 card s | none once the twin has the same tie rule: the same hits, the same counts. `tests/test_mirror_rays.py` holds the kernel to the twin already |
| 2 | wavefront: a launch advances every living ray by one segment, the living compacted between launches | about 1 s | 200 card s | none: the same arithmetic a ray |
| 3 | single precision traversal and tests, the double kernel kept as the twin; the integer histograms and the hash's 53 bits kept | 0.05 to 0.1 s | 10 to 20 card s | proof: a hit at grazing incidence may change, so the identity is statistical. The histograms of the two kernels must differ by less than two seeds of the double kernel do, band by band and bin by bin, worst cell read |
| 4 | 50 000 rays, if the seed-to-seed spread says so as reported | half | half | proof: the same test, the present 100 000 as reference |
| 5 | the histograms kept in single precision, compressed, 4 MB a site and not 35 | | 0.8 GB of disk, not 7 | none: the pack holds them in single precision already |
| 6 | a site's key without the recipe's set of cells: one histogram a (site, cell), kept in the dwelling's store | | 0 for a pair of site and cell another recipe traced | none |

After step 3 the stage is bound by what surrounds the kernel, and step 6
matters more than any further speed: **tail sites are source positions of
the dwelling** (stations, and rails every 0.80 m: about 340 of them on
hssd_0076), and today the key holds a digest of the recipe's whole set of
tail cells (`mirror/tails.py`, `tail_key`), so a second recipe traces every
site again.

**Without tracing every site?** Three formulations were weighed:

- *Reciprocity.* A ray traced from the listener's cell arrives at the site's
  sphere with the same energy, and its direction at the listener is the
  direction it was sent in, so the moments are kept. 53 cells would replace
  202 sites, a factor of 3.8. It changes the estimator (a statistical
  identity again) for a factor that step 3 makes worthless. Not worth a lot.
- *A decay model a room.* After the mixing time the tail is an exponential a
  band in each room, coupled through the doors: a few numbers a pair of
  rooms, calibrated on the tracer, in place of 600 bins x 8 bands x 17
  moments. It would cut the storage and the tails' tables, not the cost,
  which is gone by then; and it is a change of what is rendered, for the
  owner's ear. Not for speed.
- *Reuse over a dwelling.* Step 6. This is the one that scales.

**Predicted cost of the tail stage**: 0.125 USD a recipe today; 0.015 after
step 1; under 0.001 after step 3; and for a dwelling of N recipes after step
6, about 340 sites once, 20 to 40 card seconds, whatever N.

**The small card run that settles it** (0.05 USD, ten minutes on one RTX
3090): the present kernel on one site with the launch's time split from the
host's (the model's 10 s a site that do not depend on the cells: what are
they), and the count of segments it makes, against the 9.9e6 counted here.

## 4. The boundary of the wave solver

### Why 4 % of the nodes cost 55 % of a step

Read in `wave/lowband/solver.py`. On the reference grid (`lowband.json`):
63.3 M nodes in the box, 56.0 M stored, 47.4 M reached and updated, of which
2.69 M lossy boundary nodes (5.7 % of the updated nodes, 4 % of the box);
11 branches a material; 32 769 steps.

**A node of air** (`lowband_air`, one block a column, threads along `z` and
the batch): it reads its mask (2 B), its own value two steps back in `u0` (4
B), its value and six neighbours in `u1`, and writes `u0` (4 B). Each value of
`u1` is read by seven threads and fetched once: the two neighbours along the
column are in the same stream, the four lateral ones are other columns at
most 400 kB away, which the card's cache holds. **14 bytes a node a step,
all of it streamed.**

**A lossy node** is first updated as rigid by that kernel (the same 14 B),
then by `lowband_lossy`, one thread a node:

| what | bytes | how |
| --- | --- | --- |
| its column, its `z`, its material, its surface factor | 16 | streamed |
| its value in `u0`, read, and written back | 8 of its own, 64 moved | **scattered**: one node alone in its 32 byte sector, fetched and written back |
| the value it held two steps before (`before`) | 8 | streamed; kept because the air's kernel has overwritten it |
| 11 branches x 2 states (`vh`, `gh`), read and written | 176 | streamed, in 22 streams |
| its material's 44 coefficients | 0 | in the cache: 41 materials |
| **in all, beside the 14** | **264** | |

So a step moves 0.68 GB for the air (47.4 M x 12 B and 56.0 M masks) and
0.71 GB for the walls (2.69 M x 264 B): **the walls are 51 % of the bytes,
and the step is its bytes.** Measured on the RTX 3080: 1.5 ms for the air
and 1.8 ms for the branches, 453 and 394 GB/s of a card that moves 760; on
the RTX 3090 the whole step is 2.69 ms a source, 517 GB/s of 936. Both
kernels run at 52 to 60 % of the card's memory rate. The arithmetic is
nothing: 180 single precision operations a lossy node.

**It is the state, not the layout.** The two layouts already tried and
dropped (`[node, branch, source]`, and the boundary gathering its own
neighbours) were worse for the reason this table gives: more scattered
bytes. Sorting the nodes by material would change nothing (the coefficients
are in the cache). The lossy rows are already in the grid's order.

### Remedies

Bytes a source step, and the step predicted from them at the measured 517
GB/s of the RTX 3090 (88 s a position today):

| remedy | walls' bytes a node | step, GB | walls' share | s a position | risk |
| --- | --- | --- | --- | --- | --- |
| today | 264 | 1.39 | 51 % | 88 | |
| A. the branches updated in the stencil's own thread | 184 | 1.17 | 42 % | 74 | none |
| B. 7 branches a material, alone | 200 | 1.22 | 44 % | 77 | proof |
| A and B | 120 | 1.00 | 32 % | 63 | proof |
| A, and 4 branches | 72 | 0.87 | 22 % | 55 | proof, then ear |
| A and B, at 80 % of the card's rate | 120 | 1.00 | 32 % | 44 | proof |

**A. In the stencil's pass.** The air's kernel already makes a boundary
node's rigid update from its mask. Let it go on, for a lossy node, into the
branches: the thread holds the value of two steps back (it has just read
it), so `before` and the scattered read and write of `u0` both disappear,
and so do the four index arrays. What it needs is the node's row in the
branch arrays: one row base a column, and the count of lossy nodes under
`z` in the column, from a mask of the column's lossy bits (146 bits, five
words a column, in the cache). The branch arrays stay as they are, in the
grid's order, which is the order the kernel walks. The arithmetic is the
present one in the present order: **the same bits**, held by `python -m
reverberate.wave.lowband verify` on a card (two minutes, 0.01 USD) and by
`tests/test_wave_lowband.py` on `numpy`. A warp that holds a lossy node runs
the branches with its other lanes masked: 180 operations, not bytes. This
is the reverse of the layout that was dropped (the boundary gathering the
air), not that layout.

**B. Fewer branches.** Each material is 11 mass, resistance and stiffness
branches an octave apart from 16 Hz to 16 kHz. Measured here: each of the
dwelling's materials fitted again with M free branches (positive `D`, `E`,
`F`) on its own admittance from 40 to 1500 Hz, 140 frequencies, least
squares of the relative error, four starts; the eight materials that carry
88 % of the lossy nodes (the shell alone is 66 %):

| branches | worst relative error of the admittance, any material, any frequency | weighted by the nodes | worst change of the absorption coefficient |
| --- | --- | --- | --- |
| 3 | 35 % | 26 % | 0.049 |
| 4 | 20 % | 12 % | 0.028 |
| 5 | 13 % | 10 % | 0.025 |
| 6 | 16 % | 4 % | 0.012 |
| 7 | 1.3 % | 1.0 % | 0.005 |
| 8 | 1.1 % | 0.6 % | 0.001 |

**Two or three branches do not hold a wall under 1500 Hz; seven do, to one
per cent.** The seven are what one expects: the six resonant in the band and
one that stands for the stiffness of the five above it. The fit here is a
plain least squares and a better one (vector fitting, passivity kept) may
reach the same with six; four is a change of the walls of 12 to 20 %, about
1 dB of a reflection, which no proof here admits.

**What must be validated again**: the harness of lot L10a, unchanged.
`python -m reverberate.wave.lowband compare --bundle B --out O --reference
O/ref` on the dense line's 341 cells, the refitted materials against the
present engine on the same grid, read on the worst third octave and the
worst degree over the 50 ms after the onset, bar -30 dB and 0.5 dB of
level. Since the grid and the arrays are the same, only the walls differ
and the bar is the right one. One solve each way and 341 fits: ten minutes
of an RTX 3090, 0.03 USD. A fit of fewer branches is a file beside the
materials (`sim_mats.h5`) and a key of the pair cache; the solver reads the
branch count a material already.

**Weighed and not kept:**

- *Materials that are rigid under 1500 Hz treated as rigid.* Only the
  shower's tiles stay under 0.02 of absorption from 40 to 1500 Hz: 1.7 % of
  the lossy nodes. With the toilet, the lamps, the piano and the flowerpots
  (under 0.02 above 60 Hz) it is 6 %. The solver takes the branch count a
  material, so it costs nothing to try; it gains 3 % of the walls.
- *A frequency-independent impedance a band, with a solve a band.* It
  multiplies the solves by the bands. No.
- *A digital impedance filter of lower order.* A parallel bank of M second
  order branches is a filter of order 2M with 2M states: it is remedy B. A
  free rational fit of order 8 to 10 might match seven branches; its
  stability in the scheme must then be proven afresh, where a positive
  branch is stable by construction.
- *Half precision states.* Not acceptable without a proof, and none is
  offered: the branches integrate over 32 769 steps.
- *The branches sorted by material or by position.* Already in the grid's
  order; the material's coefficients are in the cache.

## 5. The rest of the wave solve

**The kernel against the card.** 517 GB/s of 936 on the RTX 3090: 55 %. A
stencil that streams reaches 75 to 85 % of a card's memory rate. So 1.45
is the most a better kernel can give on the same bytes; everything else must
come from fewer bytes (section 4), fewer nodes, fewer steps or fewer solves.
Where the 45 % goes is not known: the one-block-a-column launch leaves 43 %
of a block's threads without a node at a batch of one (146 nodes in 256
threads), 8.7 M stored nodes are visited to read a mask of zero, and the
mask is a third stream. `python -m reverberate.wave.lowband cost` with the
two kernels timed apart on an RTX 3090 says which (0.03 USD).

**One thing the ledgers show and nothing explains.** Scene B, the same card
model on another host, on the grid at 7.2 points: 1.00e10 node updates a
second where scene A made 1.76e10 (`lowband.json` of each, medians over 96
and 210 launches). By the bytes it should have been 1.4e10: 0.55 GB a step
in 1.53 ms is 360 GB/s, where scene A's cards moved 517. The RTX 3080 showed a part of it (36 s a
position measured where the bytes give 30). Either the kernel loses a third
on a grid of shorter columns, or that host's cards were slow. **A third of
scene B's 3.26 USD of solves is in that question**, and a probe of the
kernel in the first minute of every rental (ten seconds of steps, compared
with the card's table in `trace/machines.py`) answers it for every run: a
card at 60 % of its rate is given back at once.

**The steps.** The Courant number is at its limit (0.577, 36.6 µs a step);
32 769 steps are 1.2 s. Measured here on 300 pairs of run A's cache, channel
0, the energy left after a time:

| band | -30 dB at (median, ninth decile) | -40 dB | -60 dB |
| --- | --- | --- | --- |
| whole | 0.30 s, 0.39 s | 0.43 s, 0.51 s | 1.05 s, 1.15 s |
| under 200 Hz | 0.37 s, 0.43 s | 0.50 s, 0.55 s | 0.84 s, 1.10 s |
| 200 to 500 Hz | 0.32 s, 0.39 s | 0.43 s, 0.50 s | 0.67 s, 0.75 s |
| 500 to 1400 Hz | 0.23 s, 0.30 s | 0.32 s, 0.39 s | 0.60 s, 0.93 s |

The first 40 dB fall in 0.43 s and the next 20 take 0.62 s more: the decay
has two slopes. So the window cannot be cut for free (a stop when -60 dB is
reached saves 10 %), **and the last 0.6 s of every solve, half its steps,
computes what lies 40 dB under the response's energy.** Two ways to stop
paying for it, both for the owner's ear: a window of 0.6 s with the tail
continued by its own fitted decay a band, or the late part on a grid for
500 Hz (27 times fewer nodes, three times fewer steps). Before either: the
second slope may not be the room's. `as_computed.npz` shows 19.7 m2 of air
outside the shell, between the outer walls and the box's edge, all round
the dwelling and under the ceiling's slab, joined to the inside between 2.1
and 2.6 m above the floor (one component of air at node layer 125, two at layer 100). **It
is 11 % of the nodes every solve updates (5.3 M of 47.4 M)**, no listening
cell is in it, and a ring of air that the dwelling leaks into is a candidate
for a slow second slope. Whether the leak is the dwelling's or the
voxeliser's is lot L19's question; closing it, if it is an artefact, is 8 %
of a step and perhaps the late decay.

**The fit.** After a launch each cell's 984 records of 32 769 samples are
turned to double precision, filtered forward and back by cascaded sections
(a recurrence: one thread a record walks it alone, so a card runs a thousand
threads), resampled to 4 kHz by about 850 weights an output sample in double
precision (4e9 products a pair), and only then fitted, by a transform, a
product with the operator and a transform back (`wave/lowband/fit.py`). It
is 0.26 s a pair in long launches and 0.74 s in short ones, 10 066 card
seconds on run A, while the card steps nothing. Every one of these
operations is linear and invariant in time, and the fit works on spectra:
one transform of the raw single precision records (984 x 65 536 points, a
few milliseconds), the two filters' responses and the band's cut as
multiplications of the 3600 bins kept, the operator, one inverse transform
at 4 kHz. About 5e9 operations a cell, **0.02 s**. It is not the same bits
(the recurrences' start and the resampler's window differ at the edges), so
it is held by `compare` at the same bar, where it should stand 50 dB under
it: 0.03 USD.

**The level's closed form.** The scalar that joins a pair's two bands is
the ratio of two energies over 707 to 1414 Hz. The mirror's side is found
by drawing 11 million normal numbers a cell, shaping them into 1.2 s of
noise at 48 kHz in eight bands, adding the early pulses, filtering, and
transforming the result to read one octave of it (`trace/level.py`,
`mirror_omni`). The expected energy of that octave is a sum: the arrivals'
gains squared through the band's response, and the histogram's bins times
the band filters' energy in the octave. No noise need be drawn. It would
move `seam_db` in its last digits (the noise's own sample variance goes),
so it needs a proof, and since lot L15b it costs 25 ms a pair on idle
cores: **flagrant in ratio (250), worth 0 USD today**, worth doing when the
solves are short enough that the host is the bound.

**The count of solves.** Looked for and not found: a formulation that makes
one solve serve several source positions of one recipe. Sources launched
together in one grid cannot be told apart afterwards by any code: the
system is linear and invariant in time, so B sources need B solves or B
times the window. A solve in the frequency domain is a linear system of 47
M unknowns a frequency, 3600 of them. A solve a room with couplings is the
same nodes. Reciprocity is counted in `low-band-solver.md` (64 solves a
cell, a gain of 7 %). **What does change the count is that it is taken a
recipe when it belongs to the dwelling**: section 6.

## 6. Mass production: a dwelling is the unit

Measured here: four more recipes of hssd_0076 drawn with the first one's
parameters (seeds 20261005 to 20261008, twenty minutes, 14 sources) and
planned on the laptop by `python -m reverberate.trace bundle`.

| recipes | source positions asked | distinct | of the sum | pairs asked | distinct | cells, distinct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1529 | 1529 | 1.00 | 16 887 | 16 887 | 831 |
| 2 | 3318 | 2181 | 0.66 | 37 701 | 37 535 | 1722 |
| 3 | 4816 | 2338 | 0.49 | 50 870 | 50 390 | 2391 |
| 4 | 6503 | 2461 | 0.38 | 66 512 | 65 950 | 3197 |
| 5 | 7857 | 2534 | 0.32 | 78 265 | 77 497 | 3777 |

The fifth recipe brings 73 positions the first four had not (of 1354). Two
recipes share 63 to 75 % of the smaller one's positions; by capture and
recapture the dwelling holds about 2400 to 2700 in all: 131 rails at 8 cm
(210.6 m), 74 stations, at eight heights of which each place has its own.
**The pairs do not repeat (99 % distinct), because a listening cell is
placed where that recipe's listener passed.** The pair cache, keyed by the
pair, therefore gives a second recipe 166 of its 20 814 pairs, 0.8 %, while
64 % of its solves compute a field the first run computed and kept at
eleven cells.

**The remedy is a campaign a dwelling.** Draw the dwelling's N recipes
first, plan them all, and solve each distinct position once with the union
of the cells that hear it. A solve costs the same card seconds for one cell
or for eighty (run A: 86 to 92 s); the records are on the host since lot
L13; the fits go as the pairs, as they do now. Each recipe's trace then
finds every one of its pairs in the cache and runs its other stages. No
format changes, no physics: a driver that unions `heard_at` over recipes and
hands the pair cache to each trace (`--reuse-from` is that hand). The same
holds for the tail sites (section 3, step 6), the image trees and the
distance fields, which are keyed by the dwelling's places.

For recipes that come later, one at a time, the same economy needs every
solve to keep its whole field over the dwelling's listening floor: about
5000 cells on a lattice, 94 GB a position at 4 kHz. That is a design of its
own (records decimated on the card, an expansion made on demand) and is not
proposed here: planning a dwelling's recipes together gets the same count
with nothing new.

### What is paid once, and what is paid again today

| once a | what | today |
| --- | --- | --- |
| machine | rental, the interpreter and wheels, the kernels compiled, the queue's start | once a recipe: 489 to 804 s |
| dwelling | the voxel grid (43 s), its cut (18 s), the fit's operator, the mirror's prepared scene (126 MB), the models (88 MB), the image trees, the distance fields, the solves of its 2500 positions, its 340 tail sites | the grid and the trees are cached by key if the same machine and store are kept; the bundle's 214 MB go up a recipe; the solves and the tail sites are made again a recipe; the distance fields are keyed by the recipe |
| recipe | its fits (a pair), its paths, its tails' tables, its level, its rows, its pack | as it should be |

### The steady state

On 8 x RTX 3090 at 0.173 USD a card hour, the present kernel and grid, and
the frequency-domain fit; a recipe's own stages run under the solves of the
next:

| | solves a recipe | card s a recipe | USD a recipe | USD an hour of scene |
| --- | --- | --- | --- | --- |
| a recipe alone, as merged | 1571 (mean of five) | 138 000, and 0.9 h of machine round it | 8.6 | 26 |
| N = 5 a dwelling | 507 | 44 600 | 2.3 | 6.9 |
| N = 10 | about 265 | 23 300 | 1.25 | 3.8 |
| N = 30 | about 90 | 7900 | 0.50 | 1.5 |
| N = 10, after section 9's steps 1 to 8 | 265 | 10 700 | 0.55 | 1.7 |
| N = 10, and the two listening decisions | 207 | 3300 | 0.2 | 0.6 |

**Scheduling.** One queue over the recipes and the dwellings of a machine,
not a queue a recipe: the cards take the next dwelling's launches while the
cores finish the last recipe's rows, so the 555 s a card of idle end, the
fetch and the start are paid once a machine and not once a recipe. The
queue of lot L13 already takes jobs of any kind; what is missing is the
driver above it and a done marker a recipe.

**Should the results leave the machine?** Not through the laptop. A pack is
9.5 GB (3.5 with `decay=60`); the order 7 mix of the same scene is 14.7 GB
in single precision, and 14 stems are 103 GB. The pack is the smallest
thing that holds everything, and it is the only one from which another mix
can be made (other clips, other levels, another head): **the pack is what
is kept, on the store, sent from the machine over the datacentre's line,
and the laptop or the training machine reads the store.** Rendering the
features on the rented machine is right only when the training itself is
there; it must then render faster than lot L10b measured on a card (0.72 to
1.07 s of compute a second of source, bound by the host's interpreter),
which is lot L20's result to carry to a card. Sending to the store needs a
key on a rented machine: a key that can only write under one prefix of one
bucket limits what a lost machine could do, and it is the owner's to allow.

## 7. Orchestration and transfers

The table of section 2 holds the figures; what follows is what the code
says.

- **The engine build.** `gpu/onebox.py`, `provision_machine`, runs
  `scripts/build_pffdtd.sh` first, always: packages, a clone, eight
  patches, a virtual environment, and `make -C c_cuda all` for four
  binaries. The default low band (`wave/lowband`, the Cartesian grid)
  voxelises on the card by `accel.voxelise` and steps by its own kernels;
  nothing it runs opens `/root/pffdtd`. `--low-scheme fcc` needs PFFDTD's
  Python voxeliser (not its binaries) and `--low-engine pffdtd` needs all of
  it. `wave/remote_voxelise.py` already says the build "compiles the CUDA
  binaries, which this path does not need". The image is
  `nvidia/cuda:12.4.1-devel-ubuntu22.04`, which holds the compiler the
  mirror's compiled paths want. Then `scripts/provision_accel.sh`: `uv`, a
  CPython, the wheels of `requirements-remote.txt`, a `cupy` kernel
  compiled as a test: no build. An image of the project's own with the
  interpreter and wheels in it leaves the 109 s a host takes to answer and
  the push.
- **The disk** is `1.2 x largest + 0.05 x outputs + 80` GB with a field's
  outputs (`gpu/onebox.py`); the runbook says a trace's sizing "is not
  done". It is billed by the hour whether written or not.
- **A resume** makes again: the arrays' placement (`place`, never read
  back), `assign` in the parent and in every worker, each worker's grid cut
  and fit operator (33 s a worker), and every source's tails table
  (`_tail_job`, no existence check: the 342 s). The pairs, the early
  tables, the histograms, the levels and the rows are kept.
- **The watcher** polls every 300 s and a command through the proxy costs
  2.9 s, against 0.12 s on a kept direct connection
  (`direct-connection.md`).
- **The fetch** is direct first since lot L16 (20 to 57 MB/s measured on
  100 MB; the west coast and a transfer of gigabytes are not measured), in
  chunks of 32 MB on four streams. The offers are still priced with the
  proxy's 4.4 MB/s unless `--line` says otherwise, which makes a host with
  a fast line look as dear as one without.

## 8. The laptop

**The signal engine** is lot L20's and is not read here. Three things round
it that its speed does not settle:

- `viz/audit_stems.py` asks the engine for every chunk of every source and
  drops the chunk when its peak is zero. The pack's own table says which
  steps a source is audible at: 179 652 of 336 014. Not asking is a factor
  of 1.9 on the whole pre-render, on any engine.
- The cache's key holds a digest of 16 engine files: every commit of lot
  L20 voids 119 GB, and the old folders are never removed. With an engine
  that renders the chosen sources faster than they play, the page asks for
  blocks and nothing is kept; the threshold is L20's measured figure for 14
  moving sources on this laptop, with a margin of two. Until then the seal
  should not hash 206 GB of mostly holes.
- The sound check is 1200 to 1300 source seconds of engine in one thread
  (840 of the mix, 270 of three renders of one window, 70 to 200 of
  impulses), at the engine's 0.22 to 0.81 s a source second: 14 minutes are
  the engine and nothing else. What is the check's own: an engine built for
  each of 57 to 85 renders, the page's decoder run on each source where the
  decoder is linear and the mix could be decoded once, and silent sources
  rendered.

**The audit server** has no stage one or two orders over its floor in a
way a user feels, beside the pack's digest (20 s on the first open, on the
request's thread) and the stems. The "As computed" views derive on request
and keep on disk; `walls` and `facts` pass over the whole boundary a
request (2.7 M nodes, tens of milliseconds) and `slice` loops a row in
Python. The worklet at order 7 is 64 transforms of 1024 points a block of
512 samples in JavaScript, 375 M operations a second: a twentieth of a
core, and it does not grow with the sources because the server mixes.

**The test suite**: 1235 tests in 127 files, one process, `pytest --cov` on
every run, no `conftest.py`, no parallel runner in `requirements-dev.txt`.
CI is one job of 6.5 to 11 minutes. The suite now prints its forty slowest
tests (`--durations=40` in the `Makefile`, this lot's only change of code),
so that the list is read in every run and in CI. **Read in this lot's own CI
run: 1290 tests in 533 s, of which the forty slowest are 300 s and five are
122 s:**

| test | seconds |
| --- | --- |
| `test_mirror_rays.py::test_covered_rays_count_again_after_the_tree_s_window` | 35.4 |
| `test_w38_ambisonic_bands.py::test_three_grids_assemble_into_one_drawable_run_that_points_at_the_source` | 33.6 |
| `test_mirror_rays.py::test_covered_rays_leave_the_direct_and_lose_the_specular_reflections` | 23.6 |
| `test_mirror_render.py::test_rendering_is_deterministic_for_a_seed` | 17.1 |
| `test_spatial_encode.py::test_the_filter_does_not_wrap_its_own_pre_ring_onto_the_end_of_the_record` | 12.7 |

The two first of `test_mirror_rays.py` are the tracer's Python twin walking
its grid a cell at a time, 59 s for two properties that fewer rays in a
smaller room would hold; they are the first to shorten, in the lot that
rewrites the tracer. Then come 35 tests of 3.4 to 10 s: whole traces, whole
reports of `tests/test_w10_render.py` (six of them, 3.4 to 8.2 s each), the
encoder's child processes. By reading, the other slow ones
are whole traces built in module fixtures (`tests/test_trace.py`, about 18
plans and a full trace; `tests/test_trace_pool.py`), the sound check on
packs of 2 to 3 s (`tests/test_render_check.py`,
`tests/test_render_check_scene.py`), seven stem services with their servers
(`tests/test_audit_sound.py`), and Python processes started to import the
stack again. Remedies in the order of their gain: a parallel runner
(`pytest -n auto`: ten cores here, four in CI; the tests write under
`tmp_path` and the servers must take a free port), coverage in CI alone,
the traces' fixtures shared a session. Not done here: every other lot is
editing tests.

## 9. The ranked plan

The owner's two first; then by USD and wall time a unit of effort. The
predicted cost is the reference scene alone on run A's machine (8 x RTX
3090; 1.382 USD/h, 1.30 once the disk is sized), each step added to those
above it. The prediction for the code as merged is the cost appendix's:
22 170 s, 6.2 h, 8.5 USD.

| | step | risk | effort | after it: wall | USD |
| --- | --- | --- | --- | --- | --- |
| 1 | **The ray tracer** (section 3, steps 1 to 3, 5 and 6): hierarchy, wavefront, single precision with the double twin, histograms in single precision, a key a site and cell | none, then proof | 1 to 2 weeks | 6.1 h | 8.4 |
| 2 | **The boundary in the stencil's pass** (section 4, A) | none | 1 week | 5.3 h | 7.3 |
| 3 | **Seven branches** (section 4, B), by `compare` | proof | days | 4.7 h | 6.5 |
| 4 | What buys no computation (section 7): no engine build, the disk sized, the fetch direct, the longest launches first, a relaunch and an end seen in 30 s, the tails' tables kept on a resume | none | 2 to 3 days | 4.0 h | 5.1 |
| 5 | The fit in its spectra (section 5) | proof | days | 3.8 h | 4.9 |
| 6 | The air outside the shell, if lot L19 finds it an artefact | proof | days | 3.5 h | 4.6 |
| 7 | **A campaign a dwelling** (section 6); the pack to the store | none; the store's key is the owner's | 1 week | N = 10: 0.5 h a recipe | 0.9 a recipe |
| 8 | The kernels to 80 % of the card's rate, and a probe of the card at the rental | none | weeks, after the measurement | 2.5 h alone | 3.3 alone; 0.55 a recipe at N = 10 |
| 9 | The owner's two listening decisions: the grid at 7.2 points (0.39 of a position's seconds), the rail interpolation (0.78 of the positions); then `decay=60` and the late window | ear | none: built | 1.0 h alone | 1.3 alone; 0.2 a recipe at N = 10 |
| 10 | On the laptop: silent spans not rendered, blocks on demand after L20, the sound check in processes, the tests in parallel | none | days | | |

**The floor of the whole chain.** With steps 1 to 8, what is left of a
recipe alone is 65 000 card seconds of stepping at the card's memory rate,
600 of fits, 30 of rays, a quarter of an hour of machine round them: 3.3
USD, of which 2.9 are bytes that the scheme, the grid and the walls' seven
branches require. That is the cost of the physics as it is validated today.
Under it there is only the count of solves (step 7: a sixth at ten recipes)
and what the owner's ear allows (step 9: a third).

## 10. How each stage scales

| stage | scene's length | sources | movement | dwelling's volume | frequency limit | cards and cores |
| --- | --- | --- | --- | --- | --- | --- |
| wave stepping | none directly | by their positions | linear in the rail positions visited, to the dwelling's own count | linear in the nodes; the walls with the surface | fourth power | linear in the cards (the queue: 1.84 at two, 3.26 at four, 7.5 to 9 % serial) |
| fits | by the pairs | by the pairs | by the cells along the listener's path | none | linear | on the solving card: serial with it |
| rays | none | by the tail sites | sites every 0.80 m of rail | segments a ray go as the surface over the volume | none | a site a card |
| paths | linear in the distinct (source, head) positions | linear | linear | by the facets | none | a process a core; 8.2 at 16, falling at 28 (before lot L15b) |
| level, rows | linear in the pairs | linear | linear | none | rows linear | a process a core |
| write | linear in the pairs | linear | linear | none | linear | serial |
| transfers | linear in the pairs | linear | linear | none | linear | the line |
| stems, sound check | linear | linear | 3.7 times dearer moving | none | none | L20's |
| recipe, plan | linear, in Python | linear | linear | none | none | one core |

Where the time is not yet linear in the computing power: the fit (it stops
its card), the write (one process), each worker's start (a grid cut and an
operator each), the fetch (one line), the signal engine on a card (one
interpreter), and the host stages on a machine whose cores are slow (a Xeon
core at a tenth of the laptop's).

## 11. The paid measurements

| | what | where | time | USD |
| --- | --- | --- | --- | --- |
| a | `lowband cost` on both grids with the air's and the walls' kernels timed apart, and the card's memory use read beside | one RTX 3090 | 10 min | 0.03 |
| b | the same on a second host of the same card: is scene B's 1.0e10 the host's | another RTX 3090 | 10 min | 0.03 |
| c | one tail site with the launch timed apart from the host, and the kernel's count of segments | the machine of a | 5 min | 0.02 |
| d | `compare` of the seven-branch materials and of the fit in its spectra, 341 cells each | the machine of a | 25 min | 0.08 |
| e | `verify` of the boundary in the stencil's pass, when written | any card | 2 min | 0.01 |
| f | 10 GB by the direct route from a west coast host, and to the store from the same host | the machine of a | 10 min | 0.03 |
| g | the queue on a whole scene, never run: the 300 s window on two cards first | 2 x RTX 3090 | 40 min | 0.25 |

All but g: one card for an hour, 0.2 USD.

## 12. How this note's own counts were made

On the laptop, one process, from files of the first scene (read only):

- *The grid*: `scene1/A/pulled/as_computed.npz`, the reached and boundary
  bits unpacked to the box (623 x 146 x 696); the air's components a
  horizontal layer by `scipy.ndimage.label`.
- *The decay*: 300 pairs drawn from the 12 351 of
  `scene1/A/pulled/pairs`, channel 0, Schroeder's integral whole and in
  three bands (fourth order Butterworth).
- *The rays*: the bundle's `trace/mirror/scene`, the tracer's own grid
  (`mirror.rays.triangle_grid`, 0.10 m), 40 rays from source position 100
  followed with the kernel's walk and its stopping rule; Lambert draws from
  `numpy`'s generator, so the counts are statistics and not the kernel's own
  rays.
- *The branches*: the bundle's `pairs/models/materials/*.h5`, the lossy
  nodes a material from `as_computed.npz`, `scipy.optimize.least_squares` on
  the logarithms of `D`, `E`, `F`.
- *The recipes*: `python -m reverberate.scenes generate --dwelling hssd_0076
  --seed N --duration 1200 --parameters scene1/params.json
  --placeholder-clips`, then `python -m reverberate.trace bundle` with the
  first scene's mirror and models; positions compared at a millimetre.
- *The launches*: `lowband.json` of both runs, a median a batch size.

## 13. What is the owner's to decide

1. **Whether a dwelling's recipes are drawn and solved together** (section
   6). It is the largest saving of mass production and costs no rendered
   number; it fixes the order of work: recipes first, a dwelling at a time.
2. **A key of the store on a rented machine**, write only, one prefix: the
   pack and the pair cache leave by the datacentre's line and the machine
   is given back at once.
3. **Four branches or seven** only if seven's proof passes and more is
   wanted: four is a change of the walls for the ear.
4. **The late window** (the last 0.6 s at 40 dB under the response), after
   lot L19 has said what the air outside the shell does to it.
5. Already waiting: the grid at 7.2 points, the rail interpolation,
   `decay=60`.
