# 0016, appendix: the whole trace on every card and every core of the machine it is given

Status: phase one, 2026-10-05. Built and proved on the host's path (no card);
the cards' figures are phase two's, and each is named as such where it is
missing.

The first whole scene (1529 wave solves, 16 887 pairs) left a machine of
eight cards and about seventy cores with one card and one core at work for
two to three hours after its solves: the early trace, the levelling and the
pack's rows were one process, and only the rays spread. This appendix is
what replaces that: one queue for every stage, a process a card and a
process a core, and what the wave solver needs to run on a small card or on
several.

## The design

**A resource model read from the machine** (`reverberate.trace.resources`).
The cards are listed by `nvidia-smi`, without a context opened to count
them, each with the memory it really has free; the cores are the
container's quota (`compute.usable_cores`), the memory its limit. Each
worker then measures the device it holds for a second or two: the wave
solver's own step on a box of free air, in node updates a second, and a
transform of the fit's size. No table of card names is read on the machine.

**One queue** (`reverberate.trace.pool`). A stage is a set of jobs that do
not need each other:

| stage | a job | runs on | waits for |
| --- | --- | --- | --- |
| solve | a launch of the wave solver, 8 source positions at most | a card | nothing |
| rays | a tail site's histograms | a card | nothing |
| paths | 256 audible steps of one source; 64 pairs at rest | a core | nothing |
| tails | one source's tail table | a core | the sites it reads |
| level | 64 pairs | a core | their early block, the sites, the launches of its pairs |
| rows | 64 pairs as `low/ir` keeps them | a core | the launches of its pairs |

A job's message is its name and a few numbers. A worker takes the first job
it may, of whatever stage, so no card and no core waits while a job it could
run is ready, and the stages overlap: the cards cast the rays, then solve,
while the cores trace the early paths, then level each block of pairs as
its launches come home and make its rows. The pack's write and its check
are what is left for one process at the end.

**Processes, not threads.** A worker is a process started fresh (`spawn`),
shown one card (`CUDA_VISIBLE_DEVICES`) or none, its transforms told to use
one thread. It builds its own trace from the bundle and from the centres
the run wrote (`state/centres.json`): the scene is read from the files the
page cache shares, never pickled. With a machine of no card the same jobs
run on the cores; with `--workers 0`, in one process.

**What a card does, and what a core does.** The early trace took 22 ms a
position on an RTX 3090 with one core and takes 25 ms on one core of a
laptop without it: it was never the card's work. The levelling and the rows
are transforms of a few thousand samples, a dozen a pair. These three are
now `numpy` on every machine, a process a core, which has two consequences:
the cards are left to the solver and the rays, and **the tables are the
same bytes on a machine with eight cards, one, or none**.

**Order cannot change the pack.** A job's result is a file under the job's
name, written whole or not at all, and a stage's merge reads the files in
the jobs' order. The blocks' bounds are constants of the code (256 steps,
64 pairs), never the workers' count. `trace.run.pack_digest` is a digest of
every dataset and attribute of a pack but its date, its seconds and its
device; the tests hold one digest across one process, a machine of two
cards and two cores, real processes, a resumed run, and a run in which five
jobs were failed on purpose and made again.

**A job that fails.** A worker refused its memory hands its launch back in
two halves (`wave.lowband.pairs.halves`), which the queue hands out again
and what waited for the launch waits for; any other failure sends the job
to another worker, three times at most, and then stops the run with the
worker's traceback. A worker that dies loses its job to another.

**Launches for any mix of cards.** A launch's records now go to the host
while it runs (below), so what a card holds is the grid, its sources'
fields and the fit. The launches are sized for the smallest card of the
machine: any card takes any launch, and none waits for one of its size. A
source costs the same card seconds alone or among seven (3.2 ms a step a
source on the RTX 3080), so a larger card loses nothing by it; a launch is
capped at 8 sources so that what the last cards do at a run's end is short.

**The prediction, on the machine.** Before the first long job is handed
out the run says what it will take here (`prediction.json`, the log): the
launches really planned, over the cards' measured rates; the host's units
over its workers and its measured slowness against the reference host; the
rays and the write. Given `--max-hours` (the driver now passes what its
watchdog leaves), a run predicted over it stops before its long work.
**Until phase two the solve's rate is the box's and not the grid's**
(`resources.REFERENCE["solve_over_box"]` is not measured): the prediction
says so, and an uncalibrated prediction is said and does not stop a rented
machine.

## What was bound by the host, and what was done

Profiled on the laptop, `numpy`, one process, on a window of the realistic
scene: 20 s of hssd_0076 from 1075 s, three sources, 373 audible steps, 72
cells, 96 solved positions, 284 pairs, 17 tail sites over 12 cells, a
monopole in free air where a card would solve (`--free-field`).

| stage | seconds | a unit | where the seconds are |
| --- | --- | --- | --- |
| paths | 17.4 | 26 ms a position (630: 346 of the sources, 284 pairs at rest) | once a process, the mirror prepared 0.9 s and the onsets' occupancy 2.1 s; then the sieve's walk 4.1 s, the validation 6.0 s of which the occlusion tests 5.1, the onsets' 202 distance fields 2.3 s, the image trees and the short lists 2.6 s |
| level | 21.3 | 76 ms a pair | the mirror's render at the pair: the tail's noise from its histogram, its band filters, the air |
| rows | 4.0 | 14 ms a pair | the air and the masks of 64 channels, and the file |
| rays, on the host | two minutes a site and more at 2000 rays | | the host's tracer, which a machine with a card does not run; not in the figures below |

**They are bound by the host, and not by its interpreter.** In each the
time is inside `numpy`'s own loops, at 4.2 instructions a cycle and 12 per
cent of system time (6.5 GB of fresh pages for a process that holds 3.5 GB
at its largest). One loop in Python remains, the host's air absorption,
which transforms its frames one at a time. So a thread more could not help
(six gave 2.2 times one), and a launch more on a card could not either: the
card answered in microseconds and waited for the host. What was done:

1. the early trace, the levelling and the rows are jobs of the queue, a
   process a core, on `numpy` on every machine, while the cards solve;
2. the levelling reads channel 0 of a pair from the mapped file, 19 kB of
   its 1.2 MB;
3. the pack's rows are made as their launches come home, a block a file,
   and the write copies them from the mapped files: a pair's air and masks
   (14 ms here, 44 ms on the first card machine's host) are no longer in
   what one process does at the end;
4. a worker keeps twelve histograms in memory and no more (a site's is tens
   of megabytes, and every worker reads every site), and a histogram is
   written whole or not at all;
5. a table's blocks go first to the worker that traced its last block,
   which holds its image trees.

Not done, and why: batching the mirror's render over pairs in one launch
(the render is `numpy`'s work, not the interpreter's, and it now hides under
the solves on every machine with a few cores); the air's frames at once
(about a third of a pair's row and a sixth of its levelling: worth doing
for a machine of one card and two cores); the onsets' distance fields
shared between workers (each solves those its blocks need: 2.3 s of 17
here).

## Scaling on the laptop

`python -m reverberate.trace scaling --bundle B --out O --free-field --cpu
--workers 1,2,4,8 --seed-from RUN`: the window run whole at each count, the
histograms handed to every run. Work is the seconds of a stage's jobs summed
over the workers, and in brackets against one worker's.

| workers | wall s | ideal s | speed-up | paths work | level work | rows work | free field work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 46.7 | 46.7 | 1.00 | 17.0 | 21.3 | 4.0 | 1.9 |
| 2 | 39.4 | 23.4 | 1.19 | 33.7 (1.98) | 30.6 (1.44) | 5.8 (1.46) | 2.6 (1.42) |
| 4 | 42.2 | 11.7 | 1.11 | 79.0 (4.64) | 53.7 (2.52) | 11.6 (2.92) | 4.7 (2.55) |
| 8 | 50.6 | 6.0 | 0.94 | 231.7 (13.2) | 59.3 (2.75) | 17.8 (4.34) | 9.6 (5.00) |

Every count wrote the same pack. **The curve is this laptop's, and it
cannot show the queue's.** The proof is without the queue: the same stages
in one process, and N such processes started together that share nothing.

| processes at once | paths s, each | level s, each | cycles, each | instructions, each |
| --- | --- | --- | --- | --- |
| 1 | 16.0 | 21.0 | 170e9 | 720e9 |
| 2 | 22.8 | 29.8 | 177e9 | 716e9 |
| 4 | 49.0 | 58.8 | 210e9 | 723e9 |

Each does the same instructions in nearly the same cycles, 4 per cent more
at two and 23 at four, and takes 1.4 and 3.0 times the seconds: the machine
gives a second process a slower clock (four performance cores and six
efficiency cores, a desktop in use; 3.6 GHz for one process, 1.9 GHz each
for four). Its whole throughput for this work is 1.40 times one process at
two and 1.33 at four. The queue's 1.19 and 1.11 are 85 and 83 per cent of
that.

**What the queue itself costs, on this window** (the serial fraction asked
of each stage; they are constants, and do not grow with the scene but the
write):

| what | seconds | whose |
| --- | --- | --- |
| the workers' start: the interpreter, the imports, the measure | 2.5 | once a run, in parallel |
| the jobs planned before the first is handed out: the mirror prepared in the queue's process, the scene's key, the launches | 2 | one process; **on the real grid the launches' plan reads the grid's cut, about a minute, and no card has a job until it ends** |
| paths: a worker's own preparation (the mirror, the onsets' occupancy) | 2.9 a worker | in parallel; then image trees and distance fields grown again by a worker for anchors another already has: a block of the pairs costs 0.7 to 1.4 s on the worker that traced the sources and 1.7 to 3.2 s on another |
| paths, level, tails: the merge | under 0.1 a stage | one process |
| rows | 0 | |
| write | 0.2 (10 ms a pair) | one process |

Of 47 s, 5 are the queue's own and one process's: a serial fraction of a
tenth on a window of twenty seconds, of under a hundredth on the scene of
twenty minutes. What interference the workers have with each other in
cycles is memory traffic, and a machine of many cores is where it is to be
measured (`scaling --workers 1,2,4,8,16,32`, below).

## The wave solver on a small card, and on several

### The records leave the card as they are made (built, bit for bit)

A cell's records are 984 nodes by 32 769 steps by 4 bytes, 129 MB, and
were the larger part of a launch: 84 cells are 10.8 GB beside 0.69 GB of
fields a source, which is why a card under 16 GB was of no use.
`solver.solve(records_on="host")` keeps 512 steps of them on the card and
copies each block to the host's memory while the next is computed; the fit
is then handed the records back a few cells at a time, as it already took
them. The numbers are the same single precision values: the tests hold the
records equal for blocks of 1, 7, 64 and more steps on both grids, and a
campaign's cache equal to the one the present path writes.

What crosses: 84 cells' rows are 0.33 MB a step, about 20 MB/s at a
launch's pace of 60 steps a second for 5 sources. A card's link carries
several GB/s; phase two measures that the stepping does not slow
(`python -m reverberate.wave.lowband slabs` prints both rates).

What a card then needs on the grid to 1500 Hz (63 M nodes, 47 M reached):
0.2 GB of grid, 1.8 GB of fit, 0.69 GB a source and 0.17 GB of blocks, and
1.5 GB free for the fit's transforms once the fields are gone. **About
4 GB for one source at a time: a 6 GB card solves the grid, an 8 GB card
takes launches of 6 sources, a 12 GB card of 8.** Its throughput is its
own stencil rate; the queue measures it and no longer cares about its
name. The records are bounded by the host instead: half its memory over
the cards (7.5 GB a card on 120 GB and 8 cards, 58 cells a launch), and a
source heard at more cells than that is still solved more than once.

**Decimating on the card instead was measured and is not adopted.** The
fit's chain is a forward pass (low cut and fourth order low pass), the
same low pass backwards, and resampy to 4 kHz; only the forward pass can
be streamed. Keeping every second sample after it, with the backward pass
redesigned at the halved rate, differs from today's responses by -25 dB in
the worst band (707 to 1414 Hz), -44 dB in the best, on a lossy room at
the grid's rate; by 3, -18 dB: what a fourth order pass leaves above the
new Nyquist folds back. Keeping the forward pass in single precision
alone costs 1.6e-8 of the peak. A sharper streamed filter would be another
chain with other numbers; the copy to the host is today's chain to the
bit and costs nothing, so the question is closed unless a host's memory
binds, and then the block goes to its disk the same way.

### One solve over several cards (built on `numpy`, to the bit; measured in phase two)

For a grid one card cannot hold: a small card, a larger dwelling, a higher
`fmax`. Independent launches already divide a campaign by its cards, so
this buys capability and never speed. `wave.lowband.slabs` cuts the stored
columns in runs along `x`; a slab is a `Problem` of its own whose halo, one
plane either side, is of the kind that is never written, so the solver's
steppers run on it unchanged. A step is the solver's with one exchange in
it, after the box's own copies and before the update: each cut's two
planes, through `slabs.move`, the one function where a field leaves a
device. The records and the final field are the single solve's to the bit
on both grids, in two and three slabs, with and without the grid cut to its
sources' reach.

The cost, estimated: a plane of the grid to 1500 Hz is about 0.1 M nodes,
so a cut moves 0.75 MB a source a step, 6 MB for a launch of 8, 32 769
times. Through the host that is two copies each way at a few GB/s and a
fraction of a millisecond of latency each: 1 to 2 ms a step against 13 ms
of stepping a card for 8 sources on half the grid, **7 to 15 per cent, half
of that from card to card**; for one source alone, 20 to 30 per cent. The
command below measures it both ways.

### Out of core on one card (estimated, not built)

Swapping the grid through the host needs every node on the card once a
step: 0.4 GB a source each way, about 70 ms at 12 GB/s against 3.4 ms of
stepping, twenty times slower. Advancing a slab sixteen steps a visit, with
a halo sixteen planes deep recomputed each time, brings the transfer to 4 ms
a step and the stepping to about 4: **2.5 times slower at best**, with the
boundary's branches swapped too and a second solver to keep equal to the
first. Not worth building: a card that cannot hold one source of a grid is
under 5 GB here, two such cards hold it in slabs at a tenth of the cost in
time, and nothing rented is that small.

## What is proved without a card

- the queue: what waits is not started early; a job answered in parts is
  waited for through its parts; a failed job goes to another worker; one
  refused its memory is split; a launch too large for a card waits for one
  that holds it (`tests/test_trace_pool.py`);
- one pack on any machine: the digest above, in one process, on machines
  that are not there, in real processes, resumed, and with failures;
- a table cut in blocks is the table traced whole, field by field;
- records brought to the host are the records; a solve in slabs is the
  solve; the launches of a queue write the cache a campaign writes.

Not proved without a card, and phase two's to say: that `cupy`'s kernels
give one card's worker the bits another card's gives (the steppers are
compiled without fused multiply-add and each is already held to `numpy`);
that eight processes on eight cards do not slow each other through the
host; that `slabs.move` between cards is what it is on `numpy`; what the
records' way to the host costs; and every rate of the prediction.

## The commands for a machine with several cards

`B` is a bundle built on the laptop with its pairs
(`python -m reverberate.trace bundle --recipe R.json --mirror ... --models-from ...
--smoke 20 --smoke-start auto --smoke-sources 3 --out B`), `O` a directory
on the machine.

```
# The solver against numpy on this machine's cards, as before.
python -m reverberate.wave.lowband verify --out O/verify

# Records on the host against records on the card: the same bits, and both rates.
# Then one solve over two cards, from card to card and through the host.
python -m reverberate.wave.lowband slabs --out O/slabs --batch 4 --cards 0,1
python -m reverberate.wave.lowband slabs --out O/slabs_host --batch 4 --cards 0,1 --through-host
python -m reverberate.wave.lowband slabs --out O/slabs_one --batch 1 --cards 0,1

# The trace as the driver launches it: every card, every core, the prediction first.
python -m reverberate.accel campaign --bundle B --out O/all

# Every stage but the solve on every card and core, without paying for a solve.
python -m reverberate.trace run --bundle B --out O/free --free-field --check read

# Scaling: the bundle whole on 1, 2, 4, 8 cards (8 host workers each time), then on
# 1 to 32 host workers with every card. scaling.json and a table; fails if two counts
# wrote different packs.
python -m reverberate.trace scaling --bundle B --out O/by_cards --cards 1,2,4,8 --workers 8
python -m reverberate.trace scaling --bundle B --out O/by_cores --workers 1,2,4,8,16,32 \
    --free-field --seed-from O/free --seed tails

# One pack from a card machine and from the laptop's free field run of the same bundle.
python -m reverberate.trace digest O/free/pack.h5 LAPTOP/pack.h5
```

## How near to linear, and what bounds it

None of this is measured on a card; it is what the design and the host's
figures say, to be held against phase two.

- **The solves, nine tenths of a scene, go as the sum of the cards'
  rates**, whatever their mix and from 6 GB a card: independent launches on
  one queue. Lost: the plan and each worker's grid and fit, two to three
  minutes once; and at the end one launch a card at most, 15 minutes of an
  RTX 3080 for 8 sources, so up to 4 per cent of a three hour run on eight
  cards (halved by launches of 4).
- **The rays go as the cards**: 202 sites, one a card at a time.
- **The host's stages hide under the solves.** The scene's are 62 751
  positions, 16 887 pairs to level and as many rows: 3 150 seconds of the
  reference core, perhaps 8 000 of a rented Xeon's. Five host workers (the
  13.8 core machine with eight cards) do that in half an hour and sixty in
  three minutes, against three hours of solves. On a machine of one card and
  two cores they are what is left after the solves, as before.
- **One process**: the grid's voxelisation (59 s), the plan, the write
  (170 s, a copy now) and the check: five to eight minutes a scene.

So the machine's time should be the cards' throughput to within a few per
cent and those minutes, and its price nearly the same on eight cards as on
one. What will bound it, in the order expected:

1. **the pack's way home**, 49 minutes at 8 MB/s and far more at what the
   first run saw: not this lot's, and now the largest thing that is not a
   solve;
2. **the host's memory**, not the cards': a launch's records are half the
   memory over the cards, and a source heard at more cells than that is
   solved again, as it was for a card's memory;
3. **the host's cores and their memory traffic** where the cards are many
   and the cores few (the 13.8 core machine): measured here only as 23 per
   cent more cycles at four processes;
4. what is not known at all: eight processes each with a context of their
   own on eight cards, the kernels compiled eight times at a first run, and
   whether each spins a core while its card works.

## The offers' table

`trace.machines.CARDS` stays where it is needed, to choose an offer before
anything is rented, and is not read on the machine. Proposed for after
phase two, not done here while two paid traces run on the present
selection: `machines.predict` takes the queue's shape (the host's stages
under the solves, over the offer's cores; no solve added for the records; a
card from 6 GB), and a card without a row is priced as the slowest known
instead of being left out, since the machine measures itself and stops a
run it predicts over its hours.
