# Rented machines: what a campaign needs, what goes wrong, what the driver does

Two campaigns of `reverberate.experiments.w40_volume_field` ran on Vast.ai on
the nights of 2026-09-10 (hssd_0002, 524 points, 18.7 USD) and 2026-09-12
(hssd_0076, 437 points, 12.3 USD). Both delivered a field; both cost five to
ten times the compute they contained. This is the record of why, kept in
three parts: the numbers a campaign is sized by, the failures as they were
seen with the driver's response, and what still needs a person.

## 1. Sizing rules, all measured

| quantity | rule | measured on |
| --- | --- | --- |
| VRAM of a grid | `9.027 B x nodes + 2.13 GB`; one 80 GB card holds 8.6e9 nodes | A100 80 GB, W39 and W40 |
| grid of a storey at 8 kHz | `grid_shape_of(model_json, 8000, 10.5)` after the export, never the CSV box: hssd_0076 was 7.7e9 nodes by the box and 9.0e9 (83 GB) by the mesh | 2026-09-12 |
| cards | PFFDTD splits the grid over every card it sees; the host's VRAM in total must hold it. A100, H100 and H200 build with the CUDA 12.4 image; Blackwell needs 12.8 and is untested | 2026-09-13, H200 |
| card speed | H200 did 1.1 h of A100 work in 0.5 h | 2026-09-13 |
| host RAM at the engine's write | `2.2 x output + 8 GB`, output being `receivers x steps x 8 B`; a band beyond that is solved in receiver slices (`solve.slices_for`) | 50451826, oom_kill on 125 GB with 194 GB |
| host disk | `1.5 x the largest slice's float64 output + the float32 pressure of every band + 60 GB` | |
| encode worker memory | about 7 GB each (fit order 10, 1021 nodes, a full band); `workers = min(cores - 1, cgroup_GB / 8)`, read from `/sys/fs/cgroup` | 2026-09-13, three boxes |
| encode speed | 16 s a point on 22 EPYC 9754 workers; 37 s on 23 Core Ultra 9 workers; 62 s on 11 Xeon E5-2699 v3 workers; 125 s on Ivy Bridge Xeons; 45 s on 4 to 5 laptop workers | 2026-09-13 |
| pressure volume | float32: 89 MB a point for the low and high bands, 119 MB for the mid band; 128 GB for 437 points | |
| transfer, card to boxes | 22 MB/s in total over four pushes through the boxes' proxies; pulls into the card's proxy stalled | 2026-09-13 |
| transfer, laptop | 35 Mbps up (a 4 GB grid: 16 min), 7 to 15 MB/s down | |
| bandwidth billing | some hosts bill egress; the card's bill rose from 2 to 5 USD/h while it pushed 128 GB | 2026-09-13, 02:40 |
| offers | an offer is an advertisement: some never build their container, some vanish within two hours, `num_gpus` is an exact filter, `gpu_ram` is per card | both nights |

## 2. Failures seen, and the driver's response

Each row names the symptom as it appeared, the cause once found, what the
driver does about it now, and whether that is in the code.

### Renting

| symptom | cause | response | status |
| --- | --- | --- | --- |
| Instance rented, `never answered on ssh` twice on the same offer | Some hosts never build their container | `machines.rent_one` removes every offer it tries from the list, rented or not, and destroys a silent instance | in code |
| `no A100 host ... 83 GB VRAM` at 00:15 though a 2 x A100 was on the market at 22:10; the one left had 40 GB cards | Offers turn over hourly; `gpu_ram` is per card | `machines.card_hosts` filters on `num_gpus x gpu_ram_gb`, accepts H100 and H200, reliability floor 0.95 | in code |
| `only 0 boxes with 16 cores at 3.5 GHz under 0.3 USD/h`, three attempts on the same query, card idle 13 min | Floors of 1000 Mbps and reliability 0.98 on top; eleven boxes sat just under | `machines.widening_steps`: fewer cores, then a slower clock, then a higher price; 300 Mbps and 0.95 floors; Ivy Bridge excluded; ranked by price per core-GHz | in code |
| Credit ran to 1.1 USD (first night) and 0.2 USD (second) with encoding left | Card idle time and bandwidth took the margin; the check was one hour of one rental | `machines.enough_credit` against the remaining plan; the campaign rents the encode boxes while the card solves its last band | in code |
| `PUT /asks/... HTTP 400`, `GET /instances/... handshake timed out`, HTTP 429 after many status reads | Vast's API refuses, times out, rate-limits | Retries in `VastClient`; the campaign retries a stage after a pause; status reads must not poll the API every minute | in code / by hand |

### Solving

| symptom | cause | response | status |
| --- | --- | --- | --- |
| Engine reaches 100 % then exits with no `sim_outs.h5`; cgroup `memory.events` shows `oom_kill` | The engine holds `Nr x Nt` doubles and needs twice that again when it writes | `solve.sizing` slices a band whose output passes the host's RAM under the 2.2x rule; slices are merged on the host in row order | in code |
| `shrunk in 0.1 min` on a 125 GB file | A `test -f && ...` chain skipped silently because the input was missing | The solve asserts `sim_outs.h5` exists before the shrink and checks the pressure's size against the plan | in code |
| Orchestrator waits for hours on a finished solve | An `&` at the end of an `&&` chain kept the ssh channel open | Every long job runs detached from a script with `setsid nohup` and is polled on a `done`/`failed` marker (`machines.launch`, `machines.wait`) | in code |
| A retried solve stage would rent a second card | The instance id was read once at campaign start | `solve.json` carries the instance from the moment it is rented; the campaign re-reads it before every attempt and starts over only when the instance is gone | in code |
| Watchdog would have destroyed the host with the only copy of the pressure | Deadline set at rental time, later stages took longer | `machines.rearm_watchdog` at the start of the encode stage | in code |

### Transferring

| symptom | cause | response | status |
| --- | --- | --- | --- |
| `Permission denied (publickey)` from one instance to another | The proxy honours only keys attached through the API | A key made on the sending host is `POST /instances/{id}/ssh/`-ed onto the receivers | in code |
| Four pulls into the card stalled at once after 2 to 4 GB; `Connection refused` one minute, the banner then silence the next; two pulls ended in `pull.failed`, two hung with `--timeout=120` never firing | The card's own ssh proxy refused or dropped inbound sessions | The card pushes each shard through the receiving box's proxy, in a loop that resumes `rsync --partial` with keepalives (`machines.resumable_transfer`); a lost session is never a failure marker | in code |
| `Connection to sshN.vast.ai closed by remote host` after 2.5 h | The proxy closes long sessions | Same: detached, resumed, `--partial` | in code |
| `pkill -f repull.sh` inside an ssh one-liner killed the one-liner's own shell | `pkill -f` matches the caller | `pkill -f '[r]epull.sh'`, and `; true` at the end of housekeeping | in code |
| The card billed 3 h for 1.3 h of solving | Boxes rented after the solves, then a 1 h transfer on the card's clock, then bandwidth billed | Boxes rented early (above); the real fix is the store as the middle step, which needs a key on the machines: decision for the owner | partly |

### Encoding

| symptom | cause | response | status |
| --- | --- | --- | --- |
| Load average 185 on 16 cores | 15 workers each spawning 16 BLAS threads | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` in every launcher | in code |
| `BrokenProcessPool` on a 36 core box; `OSError: Too many open files` with 509 processes on a 64 core box; 80 of 91 GB used by 11 workers | About 7 GB a worker, not 1 to 2; `ulimit -n 1024` | `machines.workers_for` from the container's memory; `ulimit -n 65536` in the launcher; a failed band is retried once with half the workers | in code |
| One shard's failure made the driver tear down every box and rent four new ones, the card still up | The pool waited for all shards, then raised | `encode.encode_on_box` retries its own shard; a shard that fails twice leaves its box up with the pressure and is named in the error | in code |
| Encode logs empty for ten minutes while the work advanced | `python ... \| grep -v Warning >> log` block-buffers | Logs are written by the driver itself; `PYTHONWARNINGS=ignore` instead of a pipe | in code |
| On the laptop, 8 workers ran at 42 % each and no faster than 4 | 4 performance cores, 6 efficiency cores, memory already compressed by other applications | 4 to 5 workers on this laptop; the laptop is the fallback, not the plan | measured |

### Assembling

| symptom | cause | response | status |
| --- | --- | --- | --- |
| `the mid and high solves differ by +4.65 dB ... predict -5.46 dB` on 9 living room points 6 m or more from the source | The high band was solved in the room with its doorways sealed, the mid band on the storey | Such a point drops its high band and is flagged in `/high_dropped`; solving the storey at 8 kHz removes the case | in code |
| `the low and mid solves differ by -16.83 dB ... predict -12.04 dB` on one point | The point had no low band array of its own (0.43 m clearance) and borrowed its nearest neighbour's, 5 dB away behind a wall | `assemble.level_borrowed_low` puts the borrowed band at the predicted ratio; the point is flagged in `/low_borrowed` | in code |
| `walk.json` pointed at hssd_0002's meshes for a field of hssd_0076 | Hard-coded paths | The campaign builds the audit view of every grid on this machine from the cache entries (`audit_view`, tiered by room, native where the reader stands, decimated four times elsewhere), names them under `meshes` with `meshes_scene_id`, and never writes a path it did not build | in code |
| `BlockingIOError: unable to lock file` then `BrokenProcessPool` in a local assembly | macOS spawns pool workers by re-importing the driver script; an unguarded script re-ran its merge in every worker | The CLI is guarded; any script that calls `assemble_field` must be too | in code |

### Voxelising

| symptom | cause | response | status |
| --- | --- | --- | --- |
| Voxelisation time does not follow the grid: 27 min at 1 kHz, 40 min at 8 kHz | The cost is per triangle, not per cell (1.6 M triangles) | Grids are cached and published once per dwelling; a GPU voxeliser would save 2 h but is the least profitable port | measured |
| `Connection to ssh9.vast.ai closed by remote host` on the first rsync of a 4 GB grid | Proxy closed a long transfer | `remote_chain` resumes its rsync; the entry landed complete | in code |

## 3. What still needs a person

1. **The market.** No card with enough VRAM for hours, or no encode box at any price: the driver widens and retries, then stops with the state on disk. Relaunch later with the same command.
2. **Credit.** The driver refuses to rent below the remaining plan and never destroys a host that still holds data, but it cannot recharge; when the credit runs out, Vast stops the instances and the pressure survives on their disks until they are restarted.
3. **The store.** With a read-only key on the machines, the card uploads its pressure and leaves, boxes read from the store, and no transfer runs on the card's clock: about 2 h and 2.5 USD less per campaign, and no data lost with a host. The owner has not given that key yet.
4. **An unseen engine error.** Anything not in the tables above.

## 4. Cost and time of a campaign, with these rules

One source, about 450 points, a storey at 8 kHz on one 80 to 140 GB card:
voxelisation 0.4 USD and 2 h (once per dwelling), card 1.3 h of solving,
encoding 2 to 3 h on four boxes. With the driver as it stands: **7 to 8 USD
and 6 to 7 h**; with the store in the middle: **5 to 6 USD and 5 to 6 h**.
Budget 30 per cent more for one failure. Encoding on the card itself,
restricted to the octaves each band keeps, would take minutes and leave no
transfer at all; it is the next thing to build.

## 5. One machine, the card doing the work bought by the core (ADR 0012)

From 2026-09-14 a campaign is one rental; the multi-machine driver of parts 1 to 4 was
removed from the code on 2026-09-15, and those parts are kept as the record of what it cost. The laptop prepares a bundle
(`python -m reverberate.accel bundle`, 88 MB for hssd_0076: the storey's
mesh and materials, the listening grid, the rooms, the sources, the keys),
`reverberate.gpu.onebox` rents a host whose cards together hold the largest
grid, provisions it (`scripts/build_pffdtd.sh`, `scripts/provision_accel.sh`),
pushes the bundle, starts `python -m reverberate.accel campaign` detached,
and looks at it every five minutes: the instance and its bill through the
API, the campaign's `status.json`, the card's utilisation, the disk and the
log through ssh. It relaunches a stalled campaign once from its state on
disk, fetches the run and the grids when `campaign.done` appears, and
destroys the host only after the fetch is verified.

### What runs where, and how it was checked

| stage | before | now | checked by |
| --- | --- | --- | --- |
| voxelise | CPU box, 12 processes: 28, 40 and 45 min for the storey at 1, 4 and 8 kHz | the card, `accel.voxelise`: 14 s, 99 s and 222 s on an RTX 3090 | every dataset of `vox_out.h5` equal to the reference entry, pockets sealed by the same `wave.pockets` |
| audit view | laptop | the host's cores, from the rooms in the bundle | same code, rooms serialised as WKT |
| plan | laptop, reads HSSD | `plan_arrays` on the host from the bundle's points | `plan_field` is now `plan_points` + `plan_arrays` |
| solve | card over ssh, pressure shrunk and pushed to boxes | card, `accel.solve`, slices by the host's RAM, pressure read in place | same engine, same comms |
| encode | four CPU boxes, 16 to 125 s a point | the card, `accel.encode`: one preparation per band, then under a second a point | float32 identical to the CPU child on numpy; the card's numbers measured per campaign |
| assemble | laptop | the host's cores | same code |

### Lessons of the port

- A GPU port that must reproduce a CPU result byte for byte is compiled
  with `-fmad=false`: nvcc fuses `a*b+c` by default and the fused result is
  more accurate, which is to say different.
- PFFDTD's `normalise` divides by `|v| + eps`, so a unit ray direction is
  `1/(1+eps)` long; the kernel carries the factor, and would otherwise
  disagree on the last bit of every hit distance.
- Upstream skips a leg direction for a whole voxel when no node's leg
  reaches the triangle within one cell, before considering nodes within
  one cell and a millionth; the kernel reduces the same "any" before it
  marks, because the shortcut changes the answer at a node in that band.
- The packed triangle row is 30 doubles, not 27; the first run of the
  kernel on a real scene found 584 boundary nodes of 2.4 million.
- The chain seals air pockets after voxelising (`wave.pockets.seal_in_place`),
  which adds cells, clears bits and zeroes `saf_bn` on every buried node;
  a port that forgets it differs on 1.1 million rows and is otherwise exact.
- The engine's float64 output is not float32-exact; the shrink of the first
  campaigns rounded it, and the card path rounds it the same way.
- A fresh export of a scene is not the same mesh a week later
  (1 602 718 against 1 615 178 triangles on hssd_0076); a reproduction
  starts from the earlier export, `prepare_bundle(models_from=...)`.
- The engine keeps every receiver's whole record on the card that holds it
  (8 bytes a sample) beside its share of the grid, split over the cards by
  slabs of the outermost axis; a V100 with 137 554 receivers of 29 128
  steps asserted out of memory. `accel.solve` sizes the slices by the
  cards as well as by the host's RAM, from the receivers' positions.
- After its last step the engine spends minutes dumping the last samples of
  every receiver to its log (51 MB for 223 599 receivers) before it says
  `sim data freed`; a progress reader that looks only at the tail sees no
  percentage there, and the write of the output is a legitimate stretch at
  100 per cent that a stall rule must not kill.
- cupy's memory pool keeps what it grew to: 32 GB of the first card after
  one band's encode, and the next solve could not allocate its grid. The
  campaign frees the pools before every engine run.
- The engine runs one step more than the plan's `samples`, and the encode
  child keeps every sample it wrote (57 603 at 48 kHz for the low band of
  hssd_0076). The assembly reads the low band's length as the response's
  duration and `tail.synthesise` draws and filters its noise over that whole
  length, so an encoder that cut three samples changed every synthesised
  tail of the field while the first 393 ms agreed to 1e-7. The card path
  now encodes to the record's own length.
- A detached job launched over ssh must be started in a subshell,
  `( setsid nohup x > /dev/null 2>&1 < /dev/null & )`, or the session hangs
  until it is killed and the job dies with it.
- Two Quebec RTX 3090 hosts (offers 3884985x) never start their container
  (`failed to inject CDI devices`); the renter skips a host whose status
  says `Error` within eight minutes and tries the next.
- The engine kept every receiver's record as float64 on the host, twice
  (once as computed, once reordered for the file), and wrote float64, while
  the single precision engine had computed float32 and the encoder rounded
  the file back to float32. Patch 8 (`scripts/pffdtd/0008-…patch`, applied
  by `build_pffdtd.sh`, pushed beside it by `onebox`) keeps and writes the
  engine's own precision: half the host RAM, half the disk, half the time
  spent writing, the same numbers (bit for bit against float32 of upstream's
  file, checked with the CPU engines on the 1 kHz storey grid of hssd_0076).
  `accel.solve.output_sample_bytes` reads the mark in the built source and
  the campaign sizes its slices from it; the mid band of hssd_0076 fits a
  188 GB host in one slice instead of two.
- The assembly spent nine seconds a point on one core, seven of them
  running the whole nine-band octave bank on every noise draw to keep one
  band, sixty-four channels times two bands times nine draws. One draw per
  band, filtered once through its own band, the block in one transform
  (`metrics.octave_filter_rows`), and the crossovers as one FFT convolution
  per band: under two seconds a point, the same field to 4e-12 of the peak.
  The campaign also capped the assembly at eight workers on a 32-core host;
  it now uses every core but one, within the host's RAM.
- The scene export wrote two cut models around its own source and receiver
  pair that no campaign reads; on hssd_0018 the 5 m cut held no triangle
  and the export died writing it. The campaign's export writes the storey
  and the room only.

### Cost and time, measured on hssd_0076, one source, 437 points, 8 kHz on the storey

On 2 x A100 PCIe 80 GB (0.937 USD/h, 188 GB RAM, 32 cores), 2026-09-14: provision 2 min,
voxelise 4.8 min, audit and plan 4.9 min, low band 11 min, mid band 19 min (two slices for
the host's RAM), high band 35 min, assembly 26 min, fetch 17 to 24 min: **about 2 h and
2.1 USD**, against about 12 h and 12.3 USD for the same field on five machines two days
earlier. The field agrees with that reference to 1e-7 of the peak at every point. A host
whose RAM holds the mid band whole (240 GB) saves one slice; the assembly and the fetch
are now the two largest lines after the high band's solve. Details and the numbers per
band are in `data/runs/w42_gpu_hssd_0076/plan.md`.

Same host class, 2026-09-15, with the engine writing float32 (patch 8) and the assembly
rewritten: voxelise 4.9 min (no cache on the host), audit 3.4, plan 1.5, low band 4.5 + 4.4
(engine + encode), mid band 9.6 + 2.9 in **one** slice, high band 30.6 + 1.7, assembly 3.6 min
on 31 workers: **the campaign in 68 min** against 105 min the day before, the field again within
1.0e-7 of the reference at every point, the card's self-checks unchanged. The fetch then took
45 min for 21 GB, of which the field and the audit view are 6.6 GB: the rest was the encodings,
the self-check samples and grids already installed on the laptop. The renter now brings home
what `take_home` keeps and only the grids the laptop lacks.

## 6. The mirror beside the field (ADR 0014)

The geometric mirror runs on the machine that solved the field, after the
assembly, in two phases (`python -m reverberate.mirror run --phase card|host|all`).
The card phase needs the derived geometry and the lattice's positions; the host
phase needs the reference field. Where both are on one machine, `--phase all`.
When they are not, what travels is small: the positions (11 KB), the paths of
every point (0.7 MB), the histogram (60 MB), and for a calibration the
references of a few dozen points at order 3 (88 MB for 24 points).

### Measured on hssd_0076, one source, 437 points, RTX 3090 (0.155 USD/h billed), 2026-09-16

- Derived geometry: 41 labels, reflectors from planar facets of 0.4 m2 and up
  (exact coplanar merge), occluders from closed meshes decimated to 2 cm; the
  tree at order 3 with the flutter to 6 holds 116 662 images (2 s).
- Paths of the whole storey on the card: **435 s**, in batches of 68 receivers
  (7.9 M image-receiver pairs, 62 to 72 s each); median 10 paths a point, at
  most 49; 185 points have no direct path (rooms without a line of sight).
  The numpy twin takes 92 to 98 s a point: the card is 100 times faster and
  finds the same images, the same hit points to 1e-14 m.
- Rays: **1e5 rays in 33 s, 1e6 in 323 s** (26 M sphere crossings, median
  73 000 a receiver); the histograms' counts are equal to the twin's.
- The card phase in all: **13.2 min, 0.03 USD**. The host phase at home on
  10 cores: the render of 437 points at order 7 and the judgement, streamed
  through the disk one point at a time (13 GB of responses would not fit).
- Two cards (2 x RTX 2080 Ti, 0.169 USD/h, 2026-09-16): the same images, hit
  points, gains and integer histograms as one card; 136 receivers' paths in
  78 s against 153 s, 200 000 rays in 47 s against 80 s. The devices run one
  thread each and the merge is in share order. A 2080 Ti is about half a 3090
  on the paths kernel.

### Mirror C on hssd_0076, 2 x RTX 3090 (0.27 USD/h), night of 2026-09-16 to 17

- Card phase with the covered rays (`--skip-specular 3`, window 80 ms): paths
  345 s and rays about 7 min on the two cards while a calibration shared them.
- The fixed point calibration (`calibrate --method fixed`, 24 points, 1e5
  rays, 32 judges): **35 s an evaluation** once the judges ran one BLAS thread
  each; seven evaluations bring the medians within 1 % of T30 and 0.2 dB of
  colour. Nelder-Mead in fifteen coordinates took 135 s an evaluation and
  forty of them.
- The diffracted onsets of the 185 points without a direct path: 2.5 s at
  home on a 10 cm grid.
- Without a card (`--cpu`, `REVERBERATE_TWIN_WORKERS=10`, one BLAS thread):
  the twin's rays on ten cores, 1e5 rays for 437 receivers in 1380 s, and a
  fixed point evaluation (24 points, 2e4 rays) in 265 s. A card phase that
  only changes materials reuses the paths (`--paths-from`).

### The whole campaign on one RTX 3090 (0.155 USD/h billed), 2026-09-17

After the facet buckets in the paths kernel, the 0.10 m occluder grid, the
vectorised grid build and the render on the card, one source of hssd_0076
(437 points, order 7, 48 kHz) timed stage by stage on the card
(`campaign_bench.py` in the session's scratchpad, judgement excluded):

| Stage | 3e5 rays | 1e5 rays | 3e4 rays | order 2, 1e5 rays |
| --- | --- | --- | --- | --- |
| load the derived scene | 0.7 s | 0.6 s | 0.7 s | 0.7 s |
| image tree | 1.1 s | 1.1 s | 1.3 s | 1.0 s (4 260 images) |
| occluder grid | 1.5 s | 1.5 s | 1.7 s | 1.5 s |
| paths of 437 points | 9.8 s | 10.4 s | 10.9 s | 6.2 s |
| rays | 36.1 s | 16.0 s | 9.4 s | 14.9 s |
| diffracted onsets (host CPU) | 8.7 s | 11.8 s | 13.1 s | 9.4 s |
| render and float32 write of 437 points | 68.0 s | 67.6 s | 64.7 s | 61.8 s |
| **total** | **126 s** | **109 s** | **102 s** | **96 s** |

The same day, after laying the pulses on the host, drawing the tail's
directions on the card and applying the low cut as a spectrum (1e-13 of
the peak from the recursion), on an RTX 3090 with a 4.4 GHz host
(0.133 USD/h quoted): **68 s with 1e5 rays, 90 s with 3e5**; the render
and write of the 437 points 42 s (66 ms a point: early part 12 ms, tail
37 ms, air 7 ms, write 16 to 26 ms), paths 7.6 s, diffraction 4.4 s.

Deriving the scene from the export takes 4.8 s on the laptop; writing the
field in the reference's layout 8.5 s. At the billed rate (0.155 USD/h)
the table's campaigns cost 0.41 to 0.54 US cents, against 1.06 USD of
machine time for the optimised wave campaign of the same storey (68 min on
2 x A100): 1/196 (3e5 rays) to 1/258 (order 2). After the render changes
above, at about 0.16 USD/h billed: 0.30 cents with 1e5 rays (1/350) and
0.40 cents with 3e5 (1/265). The card
phase alone, through the CLI with 3e5 rays, took 1.1 min. Before these
changes the paths alone took 435 s and the rays 323 s (1e6).

The quality against the reference on the 24 calibration points (criteria
from 1 kHz up) reaches its plateau at 1e5 rays: 3e5 and 1e5 rays read the
same share of criteria within the seeds' spread (0.48 +- 0.015); 3e4 rays
lose 2.5 points and 1e4 rays 5, mostly on the late field's directions.
Order 2 saves 4 s and loses 10 points of recall; order 4 (window 50 ms,
4 M images allowed) takes 55 s of paths for no gain.

### Transfer rule, learned the expensive way

The laptop uploads to a Vast box at about 0.65 MB/s and downloads at 0.2 to
0.8 MB/s through the ssh proxy. A 6 GB field would take three hours either
way: never move a field; move the positions, the paths, the histogram, the
reference subset. The stage's two phases exist for this.

### Failures seen

- The host phase held every rendered response in memory: 437 points at order
  7 are 13 GB, twice with the aligned copies. It now writes each response to
  a scratch file and reads them back for the field and the judges.
- `dataclasses.asdict` turned the nested settings into dictionaries when the
  render settings were overridden; `dataclasses.replace` keeps them.
- A calibration with 32 judge processes, each opening one OpenBLAS thread
  per core (64), hit the container's process limit (`pids.max` 3840):
  `pthread_create failed`, and the search hung between two evaluations.
  Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for every
  mirror job on a box; read `/sys/fs/cgroup/pids.current` when a job stalls.
- `pkill -f 'mirror calibrate'` over ssh matches its own shell and kills the
  connection; kill by pid from a script on the box.
- Python's stdout is buffered when redirected: a remote stage's log stays
  empty until it ends. Run with `PYTHONUNBUFFERED=1`.


### A campaign that solves only what it keeps, 2026-09-18

The mirror answers above 1 kHz and the wave solver below, and
`python -m reverberate.mirror hybrid` joins the two per point into
`field_hybrid/<S>.h5`. A campaign built for that does not solve the bands it
throws away. From `plan.json` of hssd_0076, the card time each band's grid
asks for:

| Band | fmax | Nodes | Steps | Card time (planned) | Output |
| --- | --- | --- | --- | --- | --- |
| low | 1 kHz | 19.5 M | 21 846 | 6.4 s | 74.6 GB |
| mid | 4 kHz | 1 140 M | 29 128 | 495.6 s | 104.0 GB |
| high | 8 kHz | 8 982 M | 21 846 | 2 928.7 s | 78.0 GB |

Measured on 2 x A100 (2026-09-15, 68 min in all): voxelise 4.9 min, audit
3.4, plan 1.5, low 4.5 + 4.4 (engine + encode), mid 9.6 + 2.9, high
30.6 + 1.7, assembly 3.6. Dropping the mid and the high bands leaves
**about 22 min against 68**, and the voxelisation and the audit fall too,
since only the low grid is then built: **nearer 12 to 15 min, a fifth of the
campaign**. The mirror's own 90 s on one RTX 3090 (0.4 US cents) is noise
beside it.

The crossover's frequency barely moves that, because the low band is 6.4 s
of 3 431: between 500 Hz and 1 kHz it costs nothing to choose the higher
one, so it is chosen on quality. The mirror's error against the reference
per octave, whole response, 25 points of 0076: 11.2 dB of colour at 62 Hz,
5.4 at 125, 4.1 at 250, 2.4 at 500, 2.6 at 1 k, and its reverberation time
0.64 relative at 62 Hz, 0.23 at 125, 0.12 at 250, 0.073 at 500, 0.061 at
1 k. A **low band solved to 1.5 kHz** (four times 6.4 s, still nothing) lets
the ramp end inside the solved band with the crossover at 1 kHz.
