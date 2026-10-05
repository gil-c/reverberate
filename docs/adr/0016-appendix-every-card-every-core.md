# 0016, appendix: the whole trace on every card and every core of the machine it is given

Status: phase two, 2026-10-05. Built and proved on the host's path, then
run and measured on a machine of 4 x RTX 3090 (instance 54299322, 32 threads
of a Threadripper PRO 3975WX, 0.484 USD/h) and on the 38 lent cores of a host
whose cards could not be opened (instance 54294466). **The section "Measured
on cards" is the authority where it and a later section differ**: the later
sections are phase one's, kept with what they expected. **The section "The
host's stages, compiled and shared" is later than both** (lot L15b, the same
day, on the laptop alone): it is the authority on what the early trace and
the levelling cost and on how their jobs are cut, and no card has run it.

The first whole scene (1529 wave solves, 16 887 pairs) left a machine of
eight cards and about seventy cores with one card and one core at work for
two to three hours after its solves: the early trace, the levelling and the
pack's rows were one process, and only the rays spread. This appendix is
what replaces that: one queue for every stage, a process a card and a
process a core, and what the wave solver needs to run on a small card or on
several.

## Measured on cards (phase two)

### The whole trace through the queue

A window of the realistic scene with its real low band (2 s where three
sources move, 53 solved positions, 122 pairs, 12 tail sites; solves of
0.3 s, `--low-seconds`, on the grid to 1500 Hz, 47.4 M reached nodes) ran
end to end on the four cards at its first launch: four card workers, 26
host workers, no job failed. `python -m reverberate.trace scaling --cards
1,2,4 --low-batch 2`, the bundle whole at each count, 26 host workers each
time:

| cards | work wall s | ideal s | speed-up | solve wall | rays wall | solve work | serial fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1596 | 1596 | 1.00 | 1422 | 139 | 1421 | |
| 2 | 868 | 798 | 1.84 | 759 | 76 | 1495 (1.05) | 0.088 |
| 4 | 489 | 399 | 3.26 | 414 | 42 | 1572 (1.11) | 0.075 |

**The three counts wrote the same pack** (one digest): a card's worker
gives the bits another card's gives, and the order the jobs end in changes
nothing. The levelling and the rows ended with the last launch at every
count (their wall is the solves'), the early trace in 16 s: **the stages
that are not solves or rays no longer add to the wall**. What keeps four
cards at 3.26 and not 4: the launches planned in the queue's process (22 s
for the grid's cut) and each worker's own grid and fit before its first
launch (33 s a worker, once), which are the serial 7 to 9 per cent of an
eight minute run and are 2 minutes of a scene of hours; and 5 to 11 per
cent more card seconds a solve as cards are added, measured while another
measurement used the host's cores.

So, to the owner's question: **on this window the wall goes as the cards
to within the two minutes a run starts with**, and the rest hides under the
solves.

### The host's stages against the cores

`trace scaling --workers ... --free-field --cpu`, histograms handed to
every run. On the 4.4 GHz machine (16 cores of two threads, 30.7 lent), a
window of 60 s of every source (8257 positions, 1894 pairs: 90 blocks of
the early trace, 31 of the levelling):

| workers | wall s | ideal s | speed-up | paths work s | level work s | rows work s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 826 | 826 | 1.00 | 392 | 338 | 52 |
| 2 | 449 | 413 | 1.84 | 440 (1.12) | 359 (1.06) | 53 |
| 4 | 246 | 207 | 3.36 | 504 (1.29) | 365 (1.08) | 58 |
| 8 | 143 | 103 | 5.8 | 621 (1.58) | 383 (1.13) | 59 |
| 16 | 101 | 52 | 8.2 | 966 (2.46) | 429 (1.27) | 64 |
| 28 | 117 | 30 | 7.1 | 2050 (5.2) | 566 (1.67) | 76 |

Same pack at every count. The card measurements above and below ran on
the same machine during these counts, so the larger ones are pessimistic.
On the other host (old Xeons, 38 lent cores, other tenants at a load of
40), the laptop's window of 20 s (5 blocks of the levelling, 8 of the early
trace: too few jobs past 8 workers): 203 s, 121, 77, 62, 60, 62 for 1, 2,
4, 8, 16, 32 workers, the levelling's work within 6 per cent of one
worker's at every count.

**What bounds each curve.** The levelling and the rows scale as the cores
do (13 per cent more work at 8 workers, 27 at 16) and lose only where two
workers share a core. **The early trace does not**: its work is 1.6 times
one worker's at 8 and 2.5 times at 16. Every worker prepares the mirror and
the onsets' occupancy for itself and grows again the image trees and the
distance fields of anchors another worker has, and its arrays are
gigabytes: that is the stage to work on next (one preparation shared from a
file; the trees kept by anchor on disk). A process a thread of a core is
worse than a process a core, so the pool now counts cores
(`resources.Machine.threads_a_core`): 15 host workers on that machine, not
26.

Against the first card machine's one process on one RTX 3090, the same
60 s window's stages that are not solves: 182 s of early trace, 895 s of
rays, 157 s of levelling and 19 s of write by its ledger, 1250 s one after
the other. Through the queue with two of the cards and 28 workers: 450 s,
all of it the rays (843 card seconds, 12 s a site, a site a card); the
early trace took 92 s of it, the levelling 20, the rows 5, the write 7.

### The prediction, calibrated

A card's rate on the grid to 1500 Hz is 0.46 of its rate on the box of free
air it measures at its start (1.72e10 node updates a second against 3.74e10
on an RTX 3090, launches of 8): `resources.REFERENCE["solve_over_box"]`.
A source of 1.2 s is then 90 s of an RTX 3090 (110 s on the RTX 3080 of the
solver's own measurement). A launch costs 8.6 s besides its steps and a run
55 s before its first step. With those the run of 1 card above is predicted
at 1650 s and took 1596; before them, 1364.

### What a card needs: 4 GB, proved

`REVERBERATE_CARD_LIMIT_GB` holds every card of a run to that much (the
pool refuses the rest). A window of solves of the full 1.2 s (23 positions,
32 769 steps, records on the host) ran with every card held to **4 GB**:
23 launches of one source, none parted, 3.18 to 3.44 GB at the fullest by
`nvidia-smi`, the context included, 1.41e10 node updates a second against
1.72e10 in launches of 8: **a card of 4 GB solves the grid to 1500 Hz at 82
per cent of the rate a 24 GB card of its kind has.** Two faults had to go
first, and both were paid by every card of every run before:

- the fit's preparation held every chunk of its weighted matrices at once,
  3.4 GB beside the 1.8 GB it keeps; each chunk is now solved as it is made
  (`accel.encode.prepare_band(reduce=...)`), the same numbers;
- each kept part was cut on the card from a block a chunk had just freed
  and held the whole of that block: the 1.8 GB operator held 3.9 GB of the
  card. The parts now wait on the host until the preparation ends: 1.84 GB
  held for 1.84 GB used.

Records on the host against records on the card, same launch, same card:
2.74e10 against 2.70e10 node updates a second on the 17 M node room (the
command below), the same bits. They cost nothing.

### One solve over two cards

`python -m reverberate.wave.lowband slabs --cards 0,1 --steps 2000`, a
lossy room of 17 M nodes, two RTX 3090. **The records are the single
solve's to the bit in every case.**

| launch | planes cross | two cards against one | lost against twice one card |
| --- | --- | --- | --- |
| 4 sources | card to card | 1.71 | 14 % |
| 4 sources | through the host | 1.50 | 25 % |
| 1 source | card to card | 1.56 | 22 % |
| 1 source | through the host | 1.12 | 44 % |
| 8 sources | card to card | 1.76 | 12 % |

1.68 MB cross a step for four sources. Phase one expected 7 to 15 per cent
through the host and half of it card to card; it is 12 to 14 card to card
and 25 through the host. It stays what it was meant to be, a way to hold a
grid no card holds, and it is not wired into the trace.

### What broke, and what was done

1. **A host whose cards cannot be opened** (machine 152135: `nvidia-smi`
   lists four cards, `cuInit` answers 999, `/dev/nvidia-uvm` cannot be
   opened). The campaign took `numpy` in silence. Now `accel campaign`
   opens the cards before anything else and stops with the reason unless
   `--cpu` asked for the host; the driver asks `cuInit` with the cards'
   memory before it pushes anything, and a host that fails is destroyed and
   avoided as one whose cards are held; the host is in
   `vast.KNOWN_BAD_HOSTS`.
2. **The levelling waited for nearly every launch.** The launches were
   ordered by their records' size, so a block of 64 neighbouring pairs had
   its sources in launches all over the queue. They go in the positions'
   order now.
3. **The fit's memory**, above.
4. **Workers a thread, not a core**, above.

Nothing of the queue itself failed on cards: four contexts in four
processes, the kernels compiled by each worker at the first run, a worker's
core at 76 per cent while its card works.

### The clock's check, and what a far pair's onset is

Two windows of far sources stopped at the end of the levelling here, and
the two whole scenes of the same day did after all their solves: "the low
band and the mirror are not on one clock", on the median of what the pairs'
loudest sample trails their direct sound by. The clock was right. Pairs at
2.6 to 3.4 m trail by 10.64 to 10.66 ms, the lead; pairs at 4 to 9 m by
15.5 to 18.8 ms, falling evenly with the distance (18.5 ms at 4.4 m, 15.5
at 8.7), some by 28 to 30.

**The wave's direct sound is there at those pairs, at its time and at its
level, and it is not the loudest thing in the response.** Read in the pair
cache, channel 0, the peak within 3 ms after the straight line's time,
times the distance: 0.0012 at every near pair (22 of them, 2.6 to 3.4 m),
and 0.0012 in the median at the far ones (129 pairs at 4.4 to 9.7 m; 0.0010
to 0.0019), with nothing before it (0.0001). The response's own peak comes
5 to 8 ms after the line at those pairs, sometimes 17 to 20, and is 1.5
times the direct sound (the direct sound is 0.67 of it in the median, 0.53
at the least). So the mirror and the grid agree that the path is open; the
low band's loudest sample at a far pair is a later arrival.

The check now reads the first decile of the trails, where the direct sound
is the loudest, and the report keeps the median and the spread by distance
(`level.trail_s_at_the_first_decile`, `trail_s_by_distance`). **A window
with no near pair at all is still stopped**: the 5 s window of three far
voices here has its first decile at 15.53 ms, every one of its 129 direct
pairs being at 4.4 m or more. The reading that would hold for any scene is
the one measured above, the response's peak within a millisecond or two of
where the mirror puts the direct sound, and not its loudest sample. **It is
built since lot L15a** (`trace.clock`): the first peak within 3 ms of the
mirror's time and the level there against `1 / d`, on every pair with a
direct path; on the two whole scenes' own pairs it reads -0.02 to +0.17 ms
and +0.3 to -1.8 dB at every distance, at 8 m and more too, and a cache
1 ms or 6 dB off stops the trace with a message that says which. **What the check leaves open is not the check's**: `low/onset_s` of a pair is its loudest
sample, and at a far pair the pack therefore anchors the join of the two
bands 5 to 20 ms after the direct sound. Whether that is heard is for a
listening of far sources to say; it was so before this lot.

### The pack across machines

The same bundle gives one pack on 1, 2 and 4 cards and on 1 to 32 workers
of one machine. **It does not give the same pack on the laptop and on the
rented host** (another processor and another numpy: 2.4.6 on arm64, 2.5.3
on x86-64): the digest differs, as two transforms of two libraries differ
in their last bits. What is promised is the machine, not the world.

### The realistic scene, predicted with the queue

The scene of twenty minutes on the bundle's grid: 1529 solves of 1.2 s,
16 887 pairs, 202 tail sites over 53 cells, 62 751 positions.

| | 8 x RTX 3090 24 GB (1.38 USD/h) | 4 x RTX 3090 24 GB (0.80 USD/h) |
| --- | --- | --- |
| solves, 90 s a source | 4.79 h | 9.58 h |
| launches, start, fit (0.43 s a pair) | 0.31 h | 0.6 h |
| rays, 18 s a site | 0.13 h | 0.26 h |
| host stages: 6 450 core seconds of that Threadripper | under the solves | under the solves |
| voxelise, plan, write, check | 0.08 h | 0.08 h |
| **on the machine** | **5.3 h, 7.3 USD** | **10.5 h, 8.4 USD** |

On 4 cards of 16 GB the hours are those of four cards at those cards' own
rate, which no run here measured (the Tesla V100 are on the host that could
not open them); 16 GB is four times what a solve needs.

Against the code of before, as the two whole scenes of that day ran it
(their own `campaign.log`). Run A, the eight cards: after its solves the
early trace took 17.9 min, the rays 18.2, the levelling 23.6 and the write
2.6, 62 minutes of eight cards one stage after the other. Run B, four
cards: 17.5 min of early trace and 21.1 of rays, then the same levelling
and write. Through the queue the early trace and the levelling are under
the solves, and the rays are a site a card: 7.6 minutes of eight cards, 15
of four. **The queue removes about 50 of those 62 minutes on eight cards
(1.2 USD of the 8.5 the run cost) and as many on four, and adds nothing to
the solves**, which were already a thread a card: run A's took 4.9 h for
what its first attempt had left, about 100 s a source in launches that
held 84 cells' records on the card, against 90 s measured here. What is
left to gain on a scene is in the solves themselves (a card's rate, the
solves counted) and in the pack's way home.

### What still idles, and the next change

- Cards: none while a launch waits. At a run's end the last launches of 8
  sources leave cards idle for up to 12 minutes; launches of 4 would halve
  it for 2 per cent more solving.
- Cores: most of them, most of the run. The host's stages are 6 450 core
  seconds against 4.8 hours of eight cards; a machine of a dozen cores does
  them, and seventy cores are not needed.
- The 22 s to 2 minutes at the start in which no card has a job: the plan
  of the launches could follow the rays' jobs instead of preceding every
  job.
- The early trace's workers preparing the same things, above: the one
  stage whose work grows with its workers. Closed by L15b, below.

## The host's stages, compiled and shared (lot L15b, phase one: the laptop, no card)

The queue hid the early trace and the levelling under the solves. The solves
are being cut (a coarser grid, fewer positions, shorter responses), and on a
machine of one card and two cores the host's stages are what is left. This
lot makes them cheaper **without changing a table**: every figure below is
of runs whose pack has the digest the code of before writes.

Everything is measured on the 20 s window of phase one (hssd_0076 from
1075 s: 373 audible steps and 284 pairs at rest, 657 positions of which 630
are distinct, 72 cells, 34 cells a source stands in while it is shadowed),
`--free-field --cpu`, histograms handed in. **The laptop was not alone**: it
rendered audio through the afternoon at a load of 4 to 5, and the same run
took 7 s at one moment and 17 s at another. So a before and its after are
always two runs taken minutes apart, counts are given beside seconds, and
no rented machine has run any of this.

### Where the seconds were

One process, `cProfile`, the code of before:

| early trace, 15.2 s, 23 ms a position | seconds | what it is |
| --- | --- | --- |
| validation of the sieved pairs | 5.25 | 314 317 (step, image) pairs on flat arrays: the legs' walk through the occluders' cells is 4.47 of it, 6 705 turns of a loop in Python, each a dozen temporaries the length of the block |
| the step's sieve | 2.5 | 13 435 436 pairs through `_walk`, a million at a time: a gigabyte of temporaries a block |
| distance fields | 2.1 | Dijkstra from each cell a shadowed source stands in, about 50 ms a field on a graph of 484 532 free cells, in `scipy`'s own compiled code |
| the onsets' occupancy, graph and edges | 1.8 | once a process |
| image trees | 1.1 | 25 anchors, 45 ms each, once a process and again when a tree leaves its sixteen |
| short lists | 1.1 | a tree of 209 506 images walked from a listener's cell, 27 ms a list |
| the mirror prepared | 0.8 | which facets face which, the occluders' grid: once a process |
| geodesics, edges, rows, identities | 0.9 | Python a path |

| levelling, 21.7 s, 76 ms a pair | ms a pair | what it is |
| --- | --- | --- |
| the tail's noise | 46 | 11 059 200 normal deviates from the pair's seed (39 ms) and their squares summed a bin (7 ms) |
| the tail's directions and filters | 15 | a direction a burst from the moments (3.6), the bank's eight filters over 1.2 s at 48 kHz (5.2, of which the filters' own transform, the same every time, is half), the bursts laid (1.3) |
| the air, twice | 6 | 472 transforms a pair, a frame at a time in Python: the loop phase one left |
| the pair's file | 4.5 | mapped, and saved a sample after the other by the free field: channel 0 touched every page |
| the rest | 4 | the early part, the signature, the low cut designed again a pair, the seam |

Peak memory of the process: 3.8 GB, the sieve's blocks.

### What was done

1. **The sieve and the validation in C, a pair at a time**
   (`reverberate.mirror.native`). The tests of `moving._walk`,
   `moving._validate`, `_inside_facet` and `_blocked` written out for one
   pair, leaving at the first test the pair fails: no temporaries, and a
   leg's walk is a loop of the processor. One text, built once a machine by
   its own compiler (`cc -O2 -ffp-contract=off`, 0.25 s) into a directory
   named by the text, the flags and the compiler, and called through
   `ctypes`, which lets go of the interpreter's lock for the call. **Every
   operation that decides a path is the twin's, in the twin's order, and
   none is fused with its neighbour** (`ism.dot3`, the lesson of PR #57).
   What the twin takes from an arc sine and an arc cosine, the tree's beam,
   is not in the text: `numpy` is asked for it, of the few pairs that passed
   everything else, so no machine's own library decides a grazing path.
   The window's 13.4 M sieved pairs and 314 k validations: 7.8 s on flat
   arrays, **0.85 s** in the text. A short list: 27 ms, **3.1 ms**. A
   machine without a compiler, or with `REVERBERATE_NO_NATIVE=1`, takes the
   twin and its log says so.
2. **The same text for a card** (`native.DEVICE_SOURCE`): the tests are
   plain C that is also C++, and two launches, a thread a pair, call them
   under the project's kernel options. A worker that holds a card takes the
   early trace's jobs when it has nothing of its own (the queue already
   handed them to it, to run on `numpy`); with `REVERBERATE_PATHS_ON_CARD=1`
   it sieves and validates on its card. **Off by default: no card has run
   it.** What is proved is that the card's text, built for the host with a
   loop in place of the threads, gives the host's rows
   (`tests/test_mirror_native.py`).
3. **One preparation for every process** (`reverberate.mirror.shared`). A
   store is a directory of arrays named by a digest of what they were made
   from: the prepared mirror (161 MB here, the occluders' triangles), the
   onsets' field (65 MB), each anchor's tree (14 MB), each short list
   (85 kB), each distance field (3.1 MB, its predecessors alone). An entry
   is written under another name and renamed, read by mapping its files
   (one copy in the page cache for every worker), and made under a lock of
   its own name, so that a process that asks while another makes it waits
   and maps. The queue's process makes the mirror and the onsets' field
   before the first job. The store is the run's `mirror_store`, or
   `REVERBERATE_MIRROR_STORE` for one that outlives the run: a resumed run
   finds everything, and a second scene of the dwelling finds what does not
   depend on its positions (the mirror, the trees and the lists of the
   cells it shares; not the onsets' field, which frees the cells round the
   scene's own positions).
4. **The levelling's jobs by cell.** A pair's tail is drawn from the seed of
   its cell, so the pairs of one cell draw the same eleven million
   deviates. A job was 64 pairs in the pairs' order, which on the whole
   scene (16 887 pairs, 831 cells) is 12 918 (job, cell) meetings: nearly
   every pair drew its own. A job is now the pairs of whole cells, 64 or
   more, levelled a cell after the other with the cell's draws kept
   (`render.TailNoise`): 831 draws for the scene. Such a job waits for more
   launches than a job of neighbours did, and starts later in the solves;
   what is left of the levelling when the last launch ends is a few
   seconds of each host worker.
5. **What was recomputed, kept; what was looped, taken at once.** The
   bank's filters' transform (`render._bank_spectra`), the low cut's
   sections, the bank's bands; a tail at order 0 does not look for the
   direction of a burst whose encoding does not depend on it; the air's
   frames are transformed in one call each way and their gains kept by
   length (`audio.apply_air_absorption`), which the pack's rows gain from
   as well. Each is the same array to the bit, held by a test against what it was.

### Before and after

One process, the window, the store empty and then filled (taken between
15:25 and 15:50 that afternoon):

| | before | after, a first run | after, the store filled |
| --- | --- | --- | --- |
| early trace | 15.2 s, 23 ms a position | 7.4 s (the mirror 0.9, the onsets' field 2.0, 18 trees 1.1, 34 fields 1.9, 57 lists 0.2; the text 0.85) | **1.25 s, 1.9 ms a position** |
| levelling | 21.7 s, 76 ms a pair | **7.0 s, 25 ms a pair** (63 draws 3.1 s; 13.8 ms a pair besides) | the same |
| peak memory of the process | 3.8 GB | 3.1 GB (it builds the onsets' field) | 1.7 GB |

Through the queue, each count a whole run with a store of its own, the two
codes run one after the other (15:57 to 16:00; seconds of the jobs summed
over the workers, and against one worker's):

| workers | early trace, before | after | levelling, before | after | rows, before | after |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 19.3 | 6.0 | 27.0 | 7.4 | 5.8 | 4.1 |
| 2 | 38.8 (2.02) | 8.5 (1.41) | 30.2 (1.12) | 7.8 (1.05) | 5.7 (0.98) | 4.3 (1.06) |
| 4 | 76.3 (3.96) | 14.9 (2.47) | 48.5 (1.80) | 12.3 (1.67) | 9.5 (1.62) | 6.5 (1.59) |

**Six runs, one pack**: the digest of the three counts of before is that of
the three of after.

**Whether the work still grows with the workers, this laptop cannot say in
seconds**: at four processes it gives each 1.6 times the seconds for the
same work (the rows, which share nothing, are the control: 1.59 and 1.62).
What it can say is counted. At 1, 2 and 4 workers the run made **the same
entries: 1 mirror, 1 onsets' field, 18 trees, 57 lists, 34 distance fields**
(`trace_report.json`, `mirror_store`), the others read (19 trees, 43 lists
and 27 fields at four). The seconds a job spends as a process
(`paths_cpu_s`: what it waits for another's entry is not in it) are 4.7,
5.0 and 6.6: 1.05 and 1.39 times one worker's, under the control's 1.06 and
1.59. What the table's 2.47 holds beyond that is waiting: on a window of
twelve jobs every worker's first job wants the trees and the fields
another is making at that moment. A machine of many cores and a longer
window is where the curve is to be read (the commands below).

### The whole scene, from these figures

The plan of the first scene, counted on the laptop (`trace bundle` without
`--smoke`): 179 652 audible steps, 16 887 pairs over 831 cells, 231 source
anchors, 2 959 pairs of anchors, at most 1 274 cells a source stands in. In
seconds of the laptop's core, one process:

| | before | after | what it is made of |
| --- | --- | --- | --- |
| early trace | 1 830 (79 600 positions at 23 ms: the first run's 62 751 and the pairs, taken as distinct) | about 250 | 160 of positions at 2 ms; and once a scene, whatever the workers: 64 of fields at most, 10 of trees, 9 of lists, 3 of mirror and field |
| levelling | 1 250 | about 270 | 233 of pairs at 13.8 ms, 38 of 831 draws |
| rows | 340 | about 240 | |

The host of the eight cards took 17.9 and 23.6 minutes for the first two in
one process (`campaign.log` of run A): by the same ratios, about 2.5 and 5
minutes of one of its cores, and under a minute of wall on eight workers.
On the store's disk: 231 trees of 14 MB, up to 1 274 fields of 3.1 MB and
the lists, **about 8 GB for the scene**, beside the pair cache.

**The target was ten times less work in each stage. The early trace is at
twelve once its store is filled and about seven on a first run of the
scene; the levelling is at four to five, and stops there while its numbers
may not change.** What bounds each:

- *the early trace*: the distance fields. A field is `scipy`'s Dijkstra
  over the whole dwelling, 50 ms, and its predecessors are read along one
  chain: a search that stopped at the listener, or another heap, would
  break equal lengths another way and bend the onset round another cell.
  They are now solved once a cell and a machine, never once a worker, and
  remain a quarter of a first run. Then the text itself, 1.3 ms a position,
  of which the sieve of 20 000 images a position is most;
- *the levelling*: a response of 1.2 s at 48 kHz in eight bands for one
  number read over one octave under 2 kHz. The eight filters are 4 ms of
  transforms, the tail's assembly 3.5, the air 2, and the cell's draws 46 ms
  once. Rendering the mirror's channel 0 at 4 kHz for the seam, or keeping a
  cell's bursts summed, would divide it by five to ten **and move
  `low/seam_db` in its last digits**: a change of the pack's numbers, for
  the owner to decide, not done here;
- neither is bound by the interpreter any more, and neither by a card: the
  levelling cannot go to a card and keep its numbers at all, a card's
  transforms and its generator not being `numpy`'s.

### C, Numba or Cython

The sieve of one tree (209 506 images from one listener's cell, the laptop,
the best of five):

| | ms | against `numpy` | compiled |
| --- | --- | --- | --- |
| `numpy`, the twin | 28.1 | 1 | |
| the C text through `ctypes` | 3.10 | 9.1 | 0.25 s for the whole text, once a machine |
| the same loop in Numba (0.68, no fast math) | 3.35 | 8.4 | 0.26 s for this one loop, in every process without a cache; 0.19 s of import |

The three give the same list. Numba is in the environment already (through
`resampy`) and costs no dependency either; Cython would hand the same C to
the same compiler, for the same speed, and needs a build step this
repository does not have (it is run from `src`), so it was not built. The C
text was taken for what the others do not give: **the lines a card
compiles are the lines the host runs**, so the two are equal by
construction and one is tested through the other; contraction is a flag
of the build and not a property of a just in time compiler's version;
there is nothing to warm in a worker; and a machine without a compiler
loses speed, not the trace. No dependency was added.

`ctypes` lets go of the interpreter's lock, so one process could run the
text in threads. It does not: a worker is a process a core, and the text
is now a third of what a position costs.

### The commands for a machine with cards

`B` is a bundle of a window with its pairs (the commands further down),
`O` a directory on the machine. The first two need no card.

```
# The early trace and the levelling against the workers, on the host alone: the table,
# scaling.json, and per count trace_report.json (mirror_store, paths_cpu_s).
python -m reverberate.trace scaling --bundle B --out O/by_cores --workers 1,2,4,8,16,32 \
    --free-field --cpu --seed-from O/free --seed tails --keep

# The twin against the compiled text on this machine's processor: one digest.
REVERBERATE_NO_NATIVE=1 python -m reverberate.trace run --bundle B --out O/twin \
    --free-field --cpu --check read
python -m reverberate.trace run --bundle B --out O/text --free-field --cpu --check read
python -m reverberate.trace digest O/twin/pack.h5 O/text/pack.h5

# The text on the cards: every card's worker sieves and validates on its card. The
# digest must be the host's, and trace_report.json (paths, engine) must say
# "compiled, on a card" for the tables a card's worker traced.
REVERBERATE_PATHS_ON_CARD=1 python -m reverberate.trace run --bundle B --out O/cards \
    --free-field --check read --workers 0
python -m reverberate.trace digest O/text/pack.h5 O/cards/pack.h5

# A second run on the first one's store: nothing made, every entry read.
REVERBERATE_MIRROR_STORE=O/text/mirror_store python -m reverberate.trace run --bundle B \
    --out O/again --free-field --cpu --check read
```

### Not done

- threads inside a worker, above; the image trees' growth in C (45 ms an
  anchor, 10 s a scene, once);
- the prediction's host units are the window's (`resources.REFERENCE`: 8,
  21 and 10 ms), a first run's; the estimate printed with a bundle
  (`trace.plan`) still holds the seconds measured on one RTX 3090 before
  the queue, and is not this lot's file;
- the pair cache saves a response in the order of the array it is given
  (`accel.pairs.PairCache.write`): the free field's are a sample after the
  other, and a levelling that maps such a file for its first channel
  touches every page of it: 4.5 ms a pair here, which lot L15a's reader
  (`PairCache.first_channel`) still pays on such files. Saving a response
  a channel after the other would cost nothing; it is that lot's file.

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
| level | the pairs of some cells, 64 or more (64 in the pairs' order until L15b) | a core | the pairs' early table, the sites, the launches of its pairs |
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
