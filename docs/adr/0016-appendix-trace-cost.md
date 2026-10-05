# 0016, appendix: what the trace costs on a card, stage by stage

Lot L10b of ADR 0016: everything a scene trace does that is not the low
band's wave solve, measured on a rented card and cut where it paid. The
solves are L10a's. Measured on 2026-10-04 and 05 on one machine: one RTX
3090 (24 GB), 14 vCPU of a Xeon E5-2697 v4 at 2.3 GHz, 0.174 USD an hour
(instance 54204430). The code is `reverberate.trace`, `mirror.moving_onset`,
`mirror.occupancy`, `mirror.tails` and `mirror.engine`; the constants of
`trace.plan.estimate` are this ledger's.

## How it was measured

`python -m reverberate.trace run --bundle B --out O --free-field` runs every
stage on the card with a monopole in free air in place of each solve: the
paths, the onsets, the rays, the levelling, the pack and the check are the
real ones, and no solve is paid. `python -m reverberate.trace bundle` builds
`B` on the laptop without renting.

The scene is the first recipe with short dwells (seed 20261004, `dwell_s`
20 to 180, `listener_rest_s` 15 to 120): 14 sources, 24 001 steps, 179 652
audible steps of which 45 864 distinct (source, head) positions, 831
listening cells, 1529 source positions, 16 887 pairs, 202 tail sites over 53
tail cells. Two windows of it were traced, placed where most moves
(`--smoke-start auto`), all 14 sources:

| window | step-pairs | pairs | jobs of the early trace | tail sites x cells |
| --- | --- | --- | --- | --- |
| 60 s from 370 s | 6363 | 1894 | 8257 | 70 x 18 |
| 300 s | 14 223 | 4586 | 18 809 | 139 x 29 |

A job is a distinct (source, head) position or a pair at rest; 54 per cent
of the first window's and 45 per cent of the second's had no direct path
and took a diffracted onset.

**The host is shared and its clock is not steady.** Two of this lot's own
processes on the card at once doubled each other's time, and a profile
taken that way named a culprit that was not one (a sparse transpose at
0.56 s a call that costs 0.03 s alone). Every figure below is from a run
that had the machine to itself, unless it says otherwise.

## The ledger

Seconds per unit before (commit 540d3fee) and after (this lot), on that
machine; the scene's count; the scene's seconds and USD after, at 0.174
USD an hour.

| stage | unit | scene | before | after | scene, after | USD |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | a rental | 1 | 340 s | 340 s | 340 s | 0.016 |
| paths, once | a trace | 1 | 10 s | 10 s, 0 when every table is cached | 10 s | 0.000 |
| paths | a job | 62 751 | 32 ms | 22 ms | 1 400 s | 0.068 |
| rays | a tail site | 202 | 26 s (18 cells) | 12.8 s (18 cells), 14.5 s (29 cells) | 3 680 s at 53 cells | 0.178 |
| level | a pair | 16 887 | 139 ms | 62 ms (60 s window), 83 ms (300 s) | 1 400 s | 0.068 |
| write | a pair | 16 887 | 46 ms | 7 ms (60 s window); 164 ms (300 s), see below | 170 s | 0.008 |
| check | a trace | 1 | a minute of every source on both modules: not ended after 35 min | read: 0.3 s; full: 41 to 77 s | 1 s | 0.000 |
| transfer, pack | 8.1 MB/s | 23.8 GB | 2 940 s | 2 940 s | 2 940 s | 0.142 |
| transfer, pair cache | 8.1 MB/s | 20.8 GB | 2 560 s | 0 unless asked | 0 | 0.000 |

**All but the solves, the whole scene, on this machine: 9 940 s, 2.8
hours, 0.48 USD with the pair cache left on the machine; 12 500 s, 0.60 USD
with it brought home.** Before, in the same units: 340 + 2 020 (paths) +
7 580 (rays, at the ratio measured over 18 cells) + 2 350 (level) + 780
(write) + over 2 100 (check) + 5 500 (both transfers), over 20 600 s and
1.00 USD.

Where a figure is not this machine's: `fixed` is the first smoke's rental
(2 x Tesla P100, instance 54194831): 111 s of provisioning, 14 s of push,
and the rest what the watcher's five minute poll leaves between the run's
end and the fetch. The rays' 26 s before were measured while another
process of this lot used the card part of the time (the kernel alone took
19 to 23 s a site while shared), so the gain of the stage is not all the
code's: 3.7 s a site are, by the profile of item 1 below. The
scene's rays are an extrapolation in the number of tail cells from two
points (10.0 s and 0.155 s a cell), which the first whole trace measures.

## What was cut, and what it gave

Each keeps the values it had: the tests of `tests/test_trace_cost.py` hold
the identities, and the early tables of the 60 s window are the same bytes
before and after.

1. **The rays' scene is uploaded once a card and keyed once.** Every site
   re-uploaded the scene (3.3 s, most of it `reflector_of_triangles` on the
   host) and re-hashed its 1.6 million triangles for the cache key (0.4 s).
   `TailCache` now carries what a scene's calls share: 3.7 s a site less.
   The rest of 26 to 12.8 s is the card no longer shared.
2. **The levelling reads channel 0.** A pair's seam and onset are read on
   channel 0 with its air; the air was applied to the 64 channels first
   (48 ms of host transforms a pair). `trace.level.pair_omni` gives the
   same two numbers to the bit from the one channel. The low cut's
   response, 13 ms of `sosfreqz` a pair, is read once a transform length.
   Pairs are levelled by six threads on a card. 139 to 62 ms a pair.
3. **`low/ir` is made on the card.** The air and the masks of a pair's 64
   channels: 44 ms on the host, 8.4 ms on the card, equal within 2.2e-13
   of the response's peak (measured on 10 pairs; a full check measures it
   again on 8 and holds it to 1e-6).
4. **The onsets' distance fields are solved 16 to a call** and the chain is
   pulled tight in one pass per corner (`occupancy._pull`, the scalar
   walk's own samples). Fields and corners are those of before. With item
   5: 32 to 26 ms a job on the 60 s window, 275 to 229 s for the stage.
5. **The onsets' occupancy is built when a table is to be traced**, not on
   a resume whose tables are all on disk.
6. **The check is an option.** `full` (a smoke run's default) reads the
   pack whole, renders ten seconds of the three sources heard longest on
   `numpy` and on `cupy`, traces a source's early part on both and brings
   eight pairs' `low/ir` through both. `read` (the whole scene's default)
   reads the pack's structure. `--check` on `bundle`, `rent` and `run`.
7. **A container's cores are counted by its quota.** `os.cpu_count()` said
   72 on a container given 13.8; the engine's transforms started 72
   threads there. `compute.usable_cores` reads the cgroup's quota.
8. **The pair cache stays on the machine unless asked** (`--fetch-pairs`,
   or `--publish-pairs`; the whole scene's default brings it), and the
   pack and the pairs are sent without `-z`.

## V4: the engine on the host against the engine on the card

The first smoke reported the two renders equal to 8e-16 of the peak. They
were two devices' renders: that trace's engine (commit f35681ca) made its
tail in double precision, and two double precision transforms of one
signal do agree to a few units in the last place of the peak. Measured
here, per part, on that smoke's own pack and on the benchmark's density
pack, with the engine of commit 540d3fee:

| pack | early | low | tail | all, over the peak |
| --- | --- | --- | --- | --- |
| first smoke, 20 s, 3 sources at rest | 5.3e-16 | 6.7e-16 | 6.5e-9 | 6.5e-9 |
| density, at rest | 9.4e-16 | 9.1e-16 | 1.5e-7 of the tail's own peak | 7.4e-10 |
| density, moving | 9.5e-16 | 1.5e-13 | 1.8e-7 of the tail's own peak | 6.2e-10 |
| 60 s window, 10 s of 3 sources | | | | 4.3e-8 |
| 300 s window, 10 s of 3 sources | | | | 3.1e-8 |

The tail's difference is its carrier, single precision since the engine's
speed work. All pass at 1e-6. The check's record now proves the device
rather than assuming it: the module each engine holds, the module of the
arrays its renderer returns before they come to the host (`cupy`), the
card's name, the bytes the card's pool held (1.3 to 1.8 GB), and
`identical`, which is false and must be: a difference of exactly zero does
not pass. A card that was promised and is not there raises; before, the
check silently rendered on the host alone and reported no verdict.

## The signal engine on the card

`python -m reverberate.render benchmark`, the density pack, seconds of
compute per second of scene and source:

| where | at rest | moving |
| --- | --- | --- |
| the laptop, one thread (`scene-pack.md`) | 0.22 | 0.81 |
| this machine's host, one thread | 2.24 | 6.50 |
| this machine's card (`--gpu`) | 0.72 | 1.07 |

Nothing broke on the card. The card is three to six times its own host
and slower than one thread of the laptop: the engine on a card is bound
by the host's interpreter (one core at 100 per cent, a core ten times
slower than the laptop's), not by the card.

The scene's 14 stems are 8 400 seconds of stem. On this card: 1.7 to 2.5
hours, 0.29 to 0.43 USD, and 103 GB of order 7 in single precision to
bring home at 8.1 MB/s, 3.5 hours more, 0.61 USD. On the laptop: 34
minutes with everything moving, at no cost. **The audit renders on the
laptop.** A training feed that mixes on its own card and brings nothing
home pays the first figure only, five to seven times slower than real
time on a machine like this one; what it needs is a host with a fast
core, or sources rendered in processes side by side on one card, which
was not measured.

## Transfers

Measured between this machine and the laptop, 512 MB of incompressible
data: 8.1 MB/s down with or without `-z`, 4.3 MB/s up. Solved pairs
deflate by 5 per cent (`gzip`), and by 17 per cent as `low/ir` under HDF5's
shuffle and `gzip`, at 17 ms a pair: 3.5 GB less to send against 5 minutes
more to write, no gain. The bundle (215 MB, 63 MB on the wire) goes up in
15 to 19 s.

So the bytes are the cost, and the scheme is to send fewer: the pack holds
every response a render reads, with its air and its masks; the pair cache
holds them before both, which is what a trace of another crossover, or of
another scene of the dwelling, would not solve again. A smoke run leaves it;
the whole scene brings it unless told (`--no-fetch-pairs`), and the estimate
prices the two apart.

Not done: sending the cache from the machine to the store over the
datacentre's line needs the store's key on a rented machine, which is the
owner's to decide. `low/ir` in half precision would halve the pack (the
error is -70 dB of each response's peak) and is a change of the format.

## What remains the largest, and what would cut it

1. **The rays, 3 680 s.** The kernel is double precision, and a consumer
   card computes doubles many times slower than singles: the card drew
   209 W of 350 at 100 per cent, twice as many rays in one launch took
   twice as long (19.6 s for 100 000, 39.2 s for 200 000, the card shared),
   and the first smoke's two Tesla P100, older cards with full rate
   doubles, took 29 s for 3 sites with the upload this lot removed. Either
   a card of that class is rented for the rays, or the kernel is written
   in single precision with the integer histograms kept, which changes
   which rays hit and is a statistical identity to be measured, not a bit
   identity.
2. **The pack's way home, 2 940 s** at the laptop's 8.1 MB/s. (The first
   whole scenes brought it at 2 MB/s a stream, 4.4 on four, through Vast's
   proxy: the ledgers below. The pack is now written in 0.39 of the bytes
   and fetched in chunks.)
3. **The levelling and the early trace, 1 400 s each.** Both are bound by
   the host: a pair's levelling is a dozen small launches, and six threads
   give 2.2 times one. The mirror's render at a pair, batched over pairs
   in one launch, is the next step; the tails' tables of the pairs (35 of
   the 83 ms) are host work to batch likewise.

   **Taken another way (lot L13,
   `0016-appendix-every-card-every-core.md`).** Both stages, and the
   pack's rows, are now `numpy` jobs of one queue, a process a core, while
   the cards solve: the early trace costs a laptop's core what it cost the
   RTX 3090 with one core, so the card was never its bound. The units of
   the ledger above are a card's and one process's; the queue's are in that
   appendix, measured on a host and, in phase two, on a machine of several
   cards.

## The write of the 300 s window, not explained to the end

The 60 s window wrote its 1894 pairs in 13 s. The 300 s window took 754 s
for 4586, and 454 s again when run a second time, the card idle and one
core busy. The profile of the second: 24 s in the pairs' own work (5 ms a
pair: 4.7 ms of read, the air and the masks on the card), 4 s in the file,
and **401 s in the stage's own lines**, which hold nothing but the rows'
copy into the array `low/ir` is written from. By then this lot's runs had
written about 100 GB of files on the machine and its page cache stood at
119 GB against a container limit of 116 GB: every new page of that array
had to be reclaimed from the cache first. That reading fits the profile
and was not proven before the machine was gone. The estimate takes the
first window's figure; a whole scene writes about 50 GB of files (pairs,
histograms, pack), under that limit on this machine and over it on one
with less memory. If the first whole trace shows it again, the rows go to
the file a pair at a time and the array is not held.

**Done before the first whole trace, without waiting to see it again**
(lot PF): `trace.run.PairRows` hands the pack's writer one pair when it
asks, and the writer copies a table that is not an array a row at a time.
One pair, 1.2 MB, is all that is held, where a source's rows were 5.6 GB
for 4586 pairs. The file is the same byte for byte
(`tests/test_trace.py`). Whether it removes the 164 ms a pair is for the
first whole trace to say: the reading above was not proven.

## The first whole scenes: the real ledgers (2026-10-05)

The scene of twenty minutes (hssd_0076, 14 sources, 1529 source positions,
831 cells; 16 887 pairs by the plan, 18 219 on the validated grid's arrays
and 15 187 on the coarser grid's) was traced twice, on the code of before
the queue. Each run failed twice and was resumed twice on its machine.
Seconds are the wall's, read in each run's own `campaign.log` and in this
machine's rental ledger (`python -m reverberate.trace ledger --home H ...`
prints the stages); USD are at the rate the rental was billed.

**Scene A**: instance 54262814, 8 x RTX 3090 24 GB, 72 vCPU, California;
offered at 1.284 USD/h and **billed 1.382** with its 219 GB of disk.

| stage | seconds | cards working | USD | |
| --- | --- | --- | --- | --- |
| rental to the trace's first line | 489 | 0 | 0.19 | the instance answering, the engine built (3.6 min), the bundle pushed |
| voxelise, plan, assign, three times | 246 | 0 | 0.09 | 108 s, then 70 and 68 s with the grid cached |
| solve | 19 784 | 8 | 7.59 | 1600 solves in 304 launches: 141 160 card seconds of steps, 88.2 s a position, and 10 066 of fits; the cards busy 0.955 of the wall |
| paths | 1 076 | 1 | 0.41 | one process |
| rays | 1 436 | 8, a site at a time | 0.55 | 1 094 s for the 202 sites, 342 s reading the tails' tables again in the resume |
| level | 1 417 | 1 | 0.54 | one process; ended by the clock's check |
| write | 153 | 0 | 0.06 | 24.5 GB |
| between attempts | 826 | 0 | 0.32 | 605 s after the first failure, 221 s after the second |
| from the pack to the machine's end | 5 027 | 0 | 1.93 | below |
| **billed** | **30 433** | | **11.68** | 8.45 h |

**Scene B**, the same recipe at 7.2 points per wavelength: instance
54273116, 4 x RTX 3090 24 GB, 36 vCPU, California; offered at 0.543 USD/h
and **billed 0.797**.

| stage | seconds | cards working | USD | |
| --- | --- | --- | --- | --- |
| rental to the trace's first line | 804 | 0 | 0.18 | the engine built in 8.3 min |
| voxelise, plan, assign, three times | 659 | 0 | 0.15 | the plan is 3 minutes on this grid, each time |
| solve | 14 745 | 4 | 3.26 | 1562 solves in 152 launches: 54 147 card seconds of steps, 34.7 s a position, and 3 166 of fits; busy 0.975 |
| paths | 1 052 | 1 | 0.23 | |
| rays | 1 625 | 4, a site at a time | 0.36 | 1 266 s, and 359 s reading the tables again |
| level | 1 156 | 1 | 0.26 | ended by the same check |
| write | 120 | 0 | 0.03 | 20.5 GB |
| between attempts | 825 | 0 | 0.18 | |
| from the pack to the machine's end | 3 905 | 0 | 0.86 | below |
| **billed** | **24 872** | | **5.51** | 6.91 h |

Besides, 0.27 USD of three hosts that never answered on ssh and were
destroyed after seven minutes each.

**The launches.** A launch held as many positions as its card held records
for: one position heard at 84 cells, or fourteen heard at one. A position
cost the same card seconds in either (86 to 92 s on scene A), so a
launch's length went as its positions:

| scene A, launches of | launches | pairs a launch | minutes a launch | fit, s a pair |
| --- | --- | --- | --- | --- |
| 1 position (first attempt) | 93 | 84 | 2.7 | 0.74 |
| 2 to 4 | 61 | 65 | 4.1 to 6.6 | 0.74 to 0.48 |
| 5 to 8 | 76 | 55 | 7.6 to 11.7 | 0.43 to 0.29 |
| 9 to 12 | 64 | 34 | 13.4 to 17.7 | 0.33 to 0.27 |
| 13 and 14 | 10 | 14 | 19.0 to 20.7 | 0.28 |

The launches went by their records' size, so the last hour of each run
was its longest launches: 25 launches of 18.8 minutes in the mean on scene
A (319 positions heard at 498 cells), 14 of 18.5 minutes on scene B. They
are not the positions heard at many cells, which were the first and the
shortest; they are those heard at one or two, fourteen and thirty four to
a launch. The cards then ended one by one with nothing left to give them:
**555 s a card idle on scene A and 255 s on scene B**, 4 440 and 1 020 card
seconds, 3 and 2 per cent of the solves. A prediction that divides the card
seconds by the cards is short by that, half a launch a card, and by the 55 s
before the first launch; the queue's launches of eight positions are 12
minutes on this card and leave 6.

**Predicted and billed.** At the rental the driver printed 7.16 h and 9.19
USD for scene A and 5.56 h and 3.02 USD for scene B. They were billed 8.45 h
and 11.68 USD, 6.91 h and 5.51 USD: 2.49 USD more each. Where it is, the
seconds over the prediction at the rate billed:

| cause | scene A | scene B |
| --- | --- | --- |
| the rate, on the seconds predicted: the disk's storage, which the offer's price did not hold | +0.70 | +1.41 |
| the first failure (a launch refused its memory, half an hour in): the wait for the relaunch, the stages before the solves made again | +0.26 | +0.09 |
| the second (the clock's check, after every solve): the wait, the stages again, the tails' tables read again | +0.24 | +0.27 |
| from the pack to the machine's end, against the 2 940 s priced for its fetch | +0.80 | +0.21 |
| the solves: the cards' idle end and the fits on scene A; on scene B the card itself, 34.7 s a position where 30 were priced | +0.28 | +0.42 |
| the rays, a site over every card and not a site a card | +0.24 | +0.08 |
| the early trace and the levelling faster than priced, the start slower | -0.03 | +0.01 |

The three silent hosts, 0.27 USD, were other instances' bills.

**From the pack to the machine's end** is what bringing 24.5 GB home
through the proxy cost with the machine billing: the driver's own fetch
at 2 MB/s, stopped; the pack rewritten on the machine as `bins,int16`
(4 minutes, 9.51 and 8.07 GB); those fetched in chunks of 32 MB on four
streams each, both at once, at 5.2 and 5.0 MB/s. Scene B's was home and
verified in 30 minutes. Scene A's stopped at 252 chunks of 284 with a
stream that brought nothing for ten minutes, and its pack left by another
machine. **With the pack written compact by the trace and fetched in
chunks from the first minute, that stage is 36 minutes and 0.83 USD on
scene A's host, not 84 minutes and 1.93 USD.**

**The prediction as it is now** (`trace.machines`, the queue, the pack as
`bins,int16` through the proxy, the pair cache left), on the same hosts at
the rates they billed: scene A 6.2 h and 8.6 USD, of which the solves 5.2 h
and the fetch 40 minutes and 0.91 USD; scene B 5.3 h and 4.3 USD. Against
the runs as they ran, without their failures and with the pack as samples:
8.2 h and 11.3 USD for scene A, 7.1 h and 5.6 USD for scene B. No run has
yet been made with the queue on a whole scene: the rays a site a card and
the host's stages under the solves are the two terms that rest on the
windows of the every card appendix alone.

**What the disk cost.** The rental asked for 219 GB, a field campaign's
sizing, and the traces used 53 GB (27 GB before the pack, 24.5 GB of
pack). At what these two hosts ask for storage that is 0.10 and 0.25 USD
an hour: 0.83 and 1.75 USD, the largest single difference on scene B. The
offers are now priced with their disk; sizing the disk for a trace is not
done.

## The command for the whole scene

```
M="--mirror-from data/runs/w42_gpu_hssd_0076/mirror \
   --calibration data/runs/w42_gpu_hssd_0076/mirror/calibration/c3cec6aab28bb582.json \
   --lead-s 0.0106744 --gain 0.014396307939042263"
X="--models-from data/runs/w44_clarify_interpolation/bundle_line_0076/models/storey"

python -m reverberate.trace rent --recipe R.json --home H $M --dry-run --rate 0.174
python -m reverberate.trace rent --recipe R.json --home H $M $X \
    --hours 8 --max-dph 0.6 --gpu 3090 --avoid ID ... --yes
```

The whole scene checks by reading and brings its pair cache home; `--check
full` and `--no-fetch-pairs` say otherwise. The low band is solved by the
batched solver unless `--low-engine pffdtd` is given. `--dry-run` prints
each stage with the card its constant was measured on; the totals of this
appendix are at 0.174 USD an hour and were printed with PFFDTD's price for
the solves, which is not this lot's to state.

Since lot PF the machine is not named by `--gpu` and `--hours`: the offers
are priced for the plan and the lowest predicted total within `--max-hours`
is rented, with the watchdog taken from the prediction. The commands as
they are now, and the table of cards the prediction rests on, are section 7
of `docs/runbook-rented-machines.md`.
