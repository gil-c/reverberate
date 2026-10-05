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

A host is checked before it is used. The moment ssh answers, the driver asks
`nvidia-smi` for the memory in use on every card; over 1 GiB on any of them
(`onebox.CARD_USED_LIMIT_MIB`) the host is destroyed, verified destroyed, and
the next offer is taken. On 2026-10-04 a host whose cards each carried about
70 GB of another tenant's work was provisioned and solved on, and the
campaign died at the encode, out of memory. A host refused that way, or one
that never answered on ssh, is not rented again in the run: the same machine
is advertised as several offers, one per count of its cards, so it is named
by its machine id as well as by the offer's. `--avoid ID ...` gives the
driver offers and machines to skip from the start, and `onebox.json` carries
`avoided`, that list with the hosts the run refused, for the next run.

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

Measured with #45's command line (`mirror run --phase card|host`, tags); #47's
library composes the same steps from `reverberate.mirror.pipeline`.

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

The mirror's own campaign grew with the edges. The diffraction stage, on
the laptop's ten cores, takes 1.8 s for the geodesic alone and **19.1 s**
with the 651 edges and the corners' own image trees. One source of 0076 on
one RTX 3090 is then about **83 s with 1e5 rays** and 105 s with 3e5,
against 68 and 90 before: 0.37 and 0.47 US cents at 0.16 USD/h, still
1/290 and 1/225 of the wave campaign's 1.06 USD.

## 7. The scene trace (ADR 0016)

A recipe becomes a scene pack on one rented machine, by one command from the
laptop (`reverberate.trace`). **Every stage has run on a card**: the low
band on the batched solver on one RTX 3080 20 GB (instance 54201838,
`docs/open-questions/low-band-solver.md`), every other stage on one RTX 3090
(instance 54204430, `docs/adr/0016-appendix-trace-cost.md`), and smoke runs
of the whole chain on 2 x Tesla P100 and after. **The whole scene of twenty
minutes ran twice on 2026-10-05**, on 8 and on 4 RTX 3090, each to a pack
after two failures and two resumes; their ledgers are in the cost appendix,
and what they taught about bringing a run home and driving a rental is
below (*Bringing a run home*, *When a run fails after its solves*).

### What is given

The mirror of the dwelling comes from a run's `mirror` directory
(`--mirror-from`, with its calibration, lead and gain) or from a directory a
bundle already holds (`--mirror DIR`, what `MirrorAssets.save` wrote: a
bundle's `trace/mirror`). The storey's export comes from an earlier one
(`--models-from`). A recipe is refused unless its `assets` are the trace's;
`python -m reverberate.trace assets` prints them for `python -m
reverberate.scenes generate --assets`.

```
M="--mirror data/runs/w45_clarify_scene/smoke1/bundle/trace/mirror"
X="--models-from data/runs/w44_clarify_interpolation/bundle_line_0076/models/storey"
```

### The commands

```
# 1. The plan and its cost on the measured card. Nothing built, nothing rented, no key read.
python -m reverberate.trace rent --recipe R.json --home H $M --dry-run

# 2. The offers, each with what this run is predicted to take on it. The bundle is built
#    and the offers are asked for; nothing is rented.
python -m reverberate.trace rent --recipe R.json --home H $M $X \
    --gpus 4 --max-hours 16 --max-dph 2.0 --plan-offers

#    With fewer solves: the same scene generated at another rail pitch, read from eight
#    positions a step instead of two (docs/open-questions/rail-interpolation.md).
python -m reverberate.trace rent --recipe R_12cm.json --home H $M --rail-positions 8 --dry-run

# 3. A smoke run: a minute, three sources, every stage, the engine on both modules.
python -m reverberate.trace rent --recipe R.json --home H_smoke $M $X \
    --smoke 60 --smoke-sources 3 --smoke-start auto --max-dph 0.6 --yes

# 4. A V1 reference run: a short recipe whose source stands on the validated field's own.
python -m reverberate.trace rent --recipe V1_near.json --home H_v1_near $M $X \
    --check full --fetch-pairs --max-dph 0.6 --yes
python -m reverberate.render check H_v1_near/pulled/pack.h5 --reference-point --out H_v1_near/check

# 5. The whole scene, on several cards, within a wall time. The pack is written as
#    the bins of its low band in 16 bits (--low-levers, 9.5 GB for 24.5) and comes home
#    first, in chunks; the pair cache comes while the run lasts, and what is left of it
#    after the pack (--no-fetch-pairs leaves it on the machine, which goes with it).
python -m reverberate.trace rent --recipe R.json --home H_full $M $X \
    --gpus 8 --max-hours 10 --max-dph 2.0 --yes

#    The listening variant of 3.5 GB (each degree cut where it has decayed 60 dB), and
#    the samples as they were:
python -m reverberate.trace rent ... --low-levers bins,int16,decay=60
python -m reverberate.trace rent ... --low-levers none

# 6. The same scene with its low band on the coarser grid, in a home of its own.
python -m reverberate.trace rent --recipe R.json --home H_full_72 $M $X \
    --low-ppw 7.2 --reuse-from H_full --gpus 4 --max-hours 8 --max-dph 2.0 --yes

# 7. The two side by side: two files to hear at one gain, and what differs band by band.
python -m reverberate.render check H_full/pulled/pack.h5 \
    --against H_full_72/pulled/pack.h5 --window 600 660 --out AB

# 8. After a failure: the same command again. What came home is carried and not solved twice.
python -m reverberate.trace rent --recipe R.json --home H_full $M $X \
    --gpus 4 --max-hours 16 --max-dph 2.0 --fetch-early --yes
#    or, on the machine if it is still rented: the run's last lines give this command
#    whole, with its instance, after what the machine holds and what an hour of it costs.
python -m reverberate.trace rent --recipe R.json --home H_full $M $X --instance ID

# 9. What a run that came home took, stage by stage from its own log, against what the
#    prediction says of its plan on that machine. Nothing rented, nothing written.
python -m reverberate.trace ledger --home H_full --gpu "RTX 3090" --gpus 8 --gpu-ram 24
```

### The defaults, as they are now

| | default | to change it |
| --- | --- | --- |
| low band engine | the batched solver (`wave.lowband`) | `--low-engine pffdtd` |
| low band grid | Cartesian, 10.5 points per wavelength: the validated one | `--low-ppw 7.2`, `--low-scheme fcc` |
| the machine | the lowest predicted total USD among the offers predicted within `--max-hours` | `--gpus N`, `--gpu NAME`, `--max-dph`, `--avoid` |
| the watchdog, `--hours` | the prediction times 1.5 (2 for a card not measured) and half an hour, the largest of the six offers that may be tried | `--hours H` |
| the check | a smoke run renders on both modules (`full`); the whole scene reads the pack (`read`) | `--check` |
| the pack's low band | `bins,int16`: the bins of each response under the crossover's top, in 16 bits; 78 dB under the response in the worst third octave measured, 0.357 of the bytes | `--low-levers none` (the samples), `--low-levers bins,int16,decay=60` (a listening variant, 0.14) |
| the pair cache's form | compact where the pack is: every bin to 2 kHz in 16 bits, every degree whole, 0.51 of the bytes; a cache reads both forms | follows `--low-levers`; `none` keeps the samples |
| the pair cache | home for the whole scene, every five minutes while it runs, in batches of whole files; left for a smoke run | `--fetch-pairs`, `--no-fetch-pairs` |
| the way home | the pack first, in chunks of 32 MB on four kept connections, resumed and verified; the instance's own address tried first, the proxy otherwise | |
| how the offers' fetch is priced | through the proxy, 4.4 MB/s measured | `--line direct`, once a direct connection has been seen to work |
| the early tables | left on the machine | `--fetch-early`, for a second run's `--reuse-from` |
| a run that failed twice | fetched, kept for inspection, named in the last line | `--destroy-failed` |
| hosts never rented | `gpu.vast.KNOWN_BAD_HOSTS` (machine 35928, 2026-10-05) and `--avoid` | |

### What the driver does, and how a rental ends

It is `gpu.onebox`'s. The offers are priced for this plan
(`trace.machines`), the table is printed, and the reason for the choice;
nothing is rented before that. A host whose cards hold someone's memory is
destroyed and the next taken. The machine runs `python -m reverberate.accel
campaign`, which reads in the bundle that it is a trace, which engine solves
its low band and on which grid (`trace.low`); a flag overrules the bundle.
The watcher looks every five minutes, relaunches a stalled run once (a
trace resumes: pairs in their cache, paths in `early/`, histograms in
`tails/`, seams in `level.jsonl`), and every five minutes brings home what is
new of the pair cache, where it was asked for.

| what happens | what the driver does |
| --- | --- |
| the connection drops (`closed by remote host`, `Connection refused`, the banner alone) | the command is tried again, six times, 15 s to 2 min apart; a build tried again waits behind a lock for the one still running |
| the remote command itself fails (a non-zero exit that is not ssh's) | raised at once, not tried again |
| a host cannot be provisioned | destroyed, verified, avoided, the next offer taken; four hosts at most |
| a host is destroyed and not verified gone | the run stops there; the last line names it |
| a look fails (the API, the laptop's line) | said, and taken again |
| the driver itself fails after the launch | the pair cache is asked for once more, the instance destroyed and verified; outcome `error`, the traceback in `onebox.json` |
| the campaign is `done` | the pack fetched in chunks and verified, then the reports, then the pair cache if asked; destroyed, verified |
| a stream of the fetch drops or stalls | its chunk alone is asked for again, after a pause every stream shares; thirty failures in a row end the fetch |
| the fetch of a finished run fails | the instance is kept, since what it made is on it; the last lines say what it bills until its watchdog and the command that fetches again, from the chunks that are home |
| the campaign failed twice | fetched and kept; the last lines say what the machine holds, what a resume makes again, the command, and the hourly cost; destroyed with `--destroy-failed` |
| the rental's deadline is twenty minutes away | what exists is fetched, the instance destroyed |
| the host vanishes | nothing to fetch: what the homecomings brought is installed |
| a person interrupts the driver | the campaign runs on, detached; the last line names the instance |

The last lines of a run that leaves a machine rented are, as scene A's
would have been:

```
INSTANCE 54262814 IS STILL RENTED: the campaign failed twice and is kept for inspection. It bills 1.382 USD/h until its watchdog at 22:24: 10.96 USD more at most. Resume with --instance 54262814, or destroy it.
  on the machine: 18219 pairs solved (the plan counts 16887), 15 early tables, 848 histograms, 18219 pairs levelled, 0 blocks of the pack's rows, no pack
  a resume keeps all of it and makes again: the pack's write and its check. A stage's check runs again on what is kept
  it stopped with: RuntimeError("the low band and the mirror are not on one clock: ...
  resume:  python -m reverberate.trace rent --recipe R.json --home H_full ... --instance 54262814
  destroy: python -m reverberate.gpu.vast destroy 54262814
```

The command exits non-zero unless the outcome is `done`. A run resumed with
`--instance` reads its watchdog's hour in this machine's ledger. **Resumed
on a machine whose trace is done, it launches nothing and fetches**, from
the chunks that are home: what failed there was the fetch. `--relaunch`
launches the trace all the same.

**Whatever the outcome, what came home is installed**: the pairs go into
`data/cache/low-pairs/<grid key>/`, a pair cut in its transfer is left out,
and the next bundle of the recipe carries them (`pairs_cache/`), named as
the machine's engine names them. Before this the bundle named them with the
present engine's solver, and a machine on the batched solver found none of
what it was sent. `python -m reverberate.trace finish --home H` installs
again.

### Choosing the machine

A rental is billed from the moment it exists to the moment its pack is
home. `trace.machines.predict` prices that whole span on an offer, as the
trace runs since it is one queue over every card and every core:

- **the start**, 10.5 minutes with no card working: the instance answering,
  the engine built, the bundle pushed (8.1 and 13.4 minutes on the two hosts
  of 2026-10-05), then the grid voxelised and the launches planned (1.8
  minutes on the validated grid, 4.0 at 7.2 points per wavelength);
- **the solves**, a launch a card: a source position's card seconds (below),
  8.6 s a launch, a pair's fit (0.55 s on the validated grid, 0.21 at 7.2
  points), over the cards; 55 s before the first launch steps; and **half a
  launch of idleness a card at the end**, since the last launches do not end
  together (555 s a card after launches of 20.7 minutes, 255 s after
  launches of 19.5);
- **the rays**, a site a card, 18 s a site of 53 cells;
- **the host's stages** under the solves: they add to the wall only on a
  host of very few cores;
- **the write and the check**;
- **the fetch**, at the machine's rate: the pack in the form it is written
  in, at 4.4 MB/s through the proxy, and what the run left no time for of
  the pair cache. **It is in the total the offers are ranked by**: the
  first scene's pack as samples is 1.5 h of any machine, 2.08 USD of the
  eight card host and 1.20 USD of the four card one; as `bins,int16` it is
  40 minutes, 0.91 and 0.53 USD;
- **the rental's own rate**: an offer's `dph_total` is priced with 5 GB of
  disk, and the two scenes, rented with 219 GB, were billed 1.382 USD/h for
  an offer of 1.284 and 0.797 for one of 0.543. An offer that says what its
  disk costs (`storage_cost`) is priced with the disk the run asks for.

A source position of 1.2 s on the grid to 1500 Hz, in card seconds:

| card | validated grid | 7.2 points | from |
| --- | --- | --- | --- |
| RTX 3090 | 88.0 | 34.6 | **measured**: 1505 positions on 8 cards and 1496 on 4, the two whole scenes of 2026-10-05; 90 s on 4 cards in launches of 8 |
| RTX 3080 20 GB | 110 | 36 | **measured**: one card, 2026-10-05 |
| A100 | 26 | 8.6 | the RTX 3080's over 4.2, **measured through the present engine** (15.9 s a position on 2 x A100 against 135 s) |
| Tesla P100 | 149 | 49 | the RTX 3080's over 0.74, **measured through the present engine** |
| any other card | 110 over its throughput | by the points to the power 2.96 | **estimate**, memory bandwidth (`trace.machines.CARDS`) |

An estimated card is marked `ESTIMATED` in the table of offers and its
watchdog is given twice the prediction, not one and a half. A card with no
row is not offered.

**Checked against the two whole scenes** (`python -m reverberate.trace
ledger`, the prediction made as each run planned its launches, their code
being of before the queue):

| | scene A, 8 x RTX 3090, 1.382 USD/h | scene B, 4 x RTX 3090, 7.2 points, 0.797 USD/h |
| --- | --- | --- |
| solve, wall: predicted, actual | 19 836 s, 19 784 s | 15 290 s, 14 745 s |
| solves, card seconds | 144 030, 141 160 | 55 041, 54 147 |
| fits, card seconds | 9 288, 10 066 | 3 546, 3 166 |
| paths, one process | 1 409 s, 1 076 s | 1 409 s, 1 052 s |
| level, one process | 1 402 s, 1 417 s | 1 402 s, 1 156 s |
| rays | 460 s, 1 436 s | 920 s, 1 625 s |
| write | 169 s, 153 s | 169 s, 120 s |
| between attempts | 0, 825 s | 0, 825 s |

The solves are within 0.3 and 3.7 per cent. **The rays are not**: that
code divided one site over every card (1 094 and 1 266 s for the 202
sites), the queue gives a site a card, and no whole scene has run it; and
both runs read their tails' tables again in the resume that wrote the pack
(342 and 359 s). The prediction of the queue for the next run of scene A
on the same host: 6.2 h and 8.6 USD with the pack home, of which the solves
are 5.2 h; on the four cards at 7.2 points, 5.3 h and 4.3 USD.

### Bringing a run home

Measured on the two packs of 2026-10-05, from two hosts in California
through Vast's ssh proxy, on a line that carries 70 MB/s to Europe and
17 MB/s to the United States:

| streams | bytes a second | |
| --- | --- | --- |
| 1 | 2.0 to 2.3 MB/s | one `rsync`, one `dd` |
| 4 | 4.4 MB/s | each pack, 4 streams each, both at once: 5.2 and 5.0 MB/s for 25 minutes |
| 8 | 5.9 MB/s | |
| 12, reconnecting fast | refused | `Connection timed out during banner exchange`, then every stream dropped at once: the instance's `sshd` admits ten connections that have not authenticated (`MaxStartups`) |
| the pair cache by `rsync`, while the cards solved | 0.3 to 0.5 MB/s | 12 351 pairs of 16 887 home in six hours |

The proxy is the limit, a few streams are worth having, and a stream may
stop for ten minutes with its connection open (scene A's did, at 244
chunks of 284). `gpu.homecoming` is built on that:

- **a large file comes in chunks of 32 MB** on four workers
  (`fetch_file`). A chunk that arrived whole is written at its place in
  `pack.h5.partial` and noted in `pack.h5.chunks.json`: a stream that drops
  costs its chunk, and a fetch that is interrupted, or a command run
  again, asks only for what is missing. A chunk is checked by its size and
  the file by the SHA-256 the machine computes of it while the chunks
  come; where the two differ the chunks are compared one by one and the
  wrong ones fetched again;
- **each worker keeps one connection** for all its chunks
  (`ControlMaster`, a socket under `~/.ssh`): three hundred chunks open
  four connections, not three hundred;
- **a failure makes every worker wait**, twice as long with each failure
  that follows another, up to two minutes; thirty in a row end the fetch.
  A chunk that brings less than 150 kB/s is ended and asked for again;
- **the pair cache comes as batches of whole files** (`fetch_tree`, a
  `tar` stream of about 32 MB a batch), a file checked by its size, what
  is home never sent again, a pass bounded by the pause between two looks;
- **the order is the pack, the reports, then the pair cache if asked**: a
  fetch that is cut has brought what the run was for;
- **a direct connection is tried first**. An instance is now asked for
  with direct ssh (`runtype` `ssh_direc ssh_proxy`, Vast's own words for
  `--ssh --direct`), which maps its port 22 to a port of the host's own
  address; the API's record then carries `public_ipaddr` and
  `ports["22/tcp"][0]["HostPort"]`, the transfers probe that address once
  and use it where it answers, and every command still goes through the
  proxy. A request Vast refuses as malformed is made again as it was
  (`runtype` `ssh`), and the log says so. **The address is believed by
  the machine's own host keys**, read on it through the proxy when it
  first answers and pinned to the address in a file of the instance's
  own (`gpu.direct`, `gpu.hostkeys`); an address whose keys could not be
  pinned is dropped, and a key that does not match is refused and said.

**Seen on two machines** (France and North Carolina, 2026-10-05,
`docs/open-questions/direct-connection.md`): an instance created so
reports its port, answers on it, and keeps the proxy beside it; four
streams brought 41 to 45 MB/s directly against 12 through the proxy from
France, and 21 against 14 from North Carolina, on measurements of some
hundred megabytes. **Not yet seen on a machine**, and the first run's to
verify: what a chunked fetch of gigabytes brings on four kept connections;
that the compact pair cache is written and read by a whole trace. The
tests answer from a directory. Until the direct way is measured the offers
are priced through the proxy; `--line direct` prices them by the laptop's
line to where each host is (70 MB/s for Europe, 17 for the United States,
8 elsewhere, and no more than four fifths of what the host says it sends),
with which a host in Europe wins by its fetch.

**The compact forms.** The trace writes the pack's low band as `bins,int16`
unless told (`--low-levers`, in the bundle's `trace.low_levers` and the
pack's provenance): 9.51 GB for scene A's 24.50 and 8.07 GB for scene B's
20.51, as `python -m reverberate.render compact` made them on the machines.
The pair cache follows (`trace.pair_cache`): 621 kB a pair for 1 229 kB.
**It keeps every bin to 2 kHz**, not the bins to the solve's 1500 Hz: a
cached pair holds 45 dB under its energy above 1500 Hz in the median and
32 dB in the worst of forty, the pack's response is cut from it in time
before its masks, and what is dropped there comes back under 1414 Hz, 46 dB
under the stored response in its worst third octave. With every bin the
stored response is the plain one's to 83 dB (sixty pairs of scene A) and no
onset moves. A cache reads either form pair by pair, a pair's key does not
say its form, and a pair goes from cache to cache, to the bundle and to
the store in the form it is in.

### When a run fails after its solves

Both scenes ended `campaign.failed` after five hours of solves, stopped by
the clock's check, and the driver said "kept for inspection". The check was
wrong, not the clock: it read each pair's loudest sample, which at a far
pair is an arrival 5 to 20 ms after its direct sound. It now reads the
response where the mirror puts the direct sound (`trace.clock`): the first
peak within 3 ms of the mirror's time, and the level there over 300 to
800 Hz against `1 / d` on the field's scale; the first quartile of the
times within half a millisecond and the third of the levels within 3 dB.
On scene A's pairs: -0.02 ms and +0.3 dB over every distance, +0.01 ms and
+0.3 dB at 4 m and more, +0.04 ms and -1.8 dB at 8 m and more, where the
loudest sample read 12.0 and 12.7 ms at its first decile against a lead of
10.67; on scene B's, at 7.2 points, +0.17 ms and -1.7 dB at 8 m and more. A
cache 1 ms off, 6 dB off or carrying a lead of its own stops the trace
with a message that says which.

A run that fails keeps its machine, and its last lines are the ones under
*What the driver does* above. What a trace keeps as it goes, and a resume
therefore does not make again: the pairs (`pairs/`), the early tables
(`early/`), the tails' histograms (`tails/`), the pairs' levelling
(`level.jsonl`), the pack's rows (`jobs/rows/`). Scene A's resume took 9.5
minutes for a pack: 0 solves, 5.7 minutes reading its tails' tables, 2.6
writing.

### What the whole scene's first run showed (2026-10-05)

| symptom | cause | response | status |
| --- | --- | --- | --- |
| `trace FAILED: OutOfMemoryError('Out of memory allocating 10,834,217,984 bytes (allocated so far: 15,031,810,048 bytes)')` at job 94 of 304 on 8 x RTX 3090 of 24 576 MiB | The refused block is one launch's records: 84 cells of 984 nodes, 4 bytes a step, 32 769 steps. The 15.0 GB held were not that launch's: the launch before had left its own records (65 cells, 8.4 GB) in cupy's pool, the new launch's masks and fields were cut from that block, and the pool cannot give back a block of which a part is in use. The plan itself held: it was made once, on the first card, and counted one launch at a time | The pool gives its free blocks back before every launch; the resampler's weights, which stay for the campaign, are made before the first; each launch is held against what its own card has free at that moment, 80 % of it, and parted where that is less; a launch refused its memory is parted (its sources first, then one source's cells) and tried again. One source at one cell that is refused is the campaign's failure | in code, tested with an allocator that refuses; not run on a card |
| `homecoming of pairs not complete: transfer failed after 1 attempts: rsync down did not end in 600 s`, every time, and the looks 15 minutes apart instead of 5 | A pass was given 600 s every 600 s and the watch waited for it; the line brought 0.3 to 0.5 MB/s (895 pairs of 1.2 MB in 1.6 h, of 11 481 then on the machine), so no pass could end | A pass is taken out of the pause between two looks (270 s of 300) and ends there with every whole file kept; what its last file left is removed; the log says `pairs home: N of M on the machine (+K in S s)` | in code |
| The pair cache's way home is priced at 8.1 MB/s (43 min for 20.75 GB) and came at a twentieth of that | The proxy carries 2 MB/s a stream, and one `rsync` of thousands of files on a machine busy solving a quarter of that | Batches of whole files on four kept connections, the cache half its size, the way home priced at the 4.4 MB/s four streams measured. `--no-fetch-pairs` leaves it | in code; the rate of the batches not measured on a machine |
| The pack, 24.5 GB, came home at 2 MB/s with the eight cards billing 1.38 USD/h | The same proxy, one stream; twelve streams were refused together and a stream that dropped lost its 2 GB part | The pack is written 9.5 GB, fetched first, in chunks of 32 MB that are kept, on four connections that are kept | in code, tested against a directory |
| `trace FAILED: RuntimeError("the low band and the mirror are not on one clock: ... 13.21 ms in the median")` after 5.9 h, on both scenes | The check read the pairs' loudest samples; at 4 m and more those are a later arrival | The check reads the direct sound where the mirror puts it, its time and its level (`trace.clock`) | in code; on both scenes' own pairs it passes at every distance |
| The driver's last line: "kept for inspection" | It did not know what the machine held or how it was called | The last lines: what is on the machine, what a resume makes again, the command, the hourly cost | in code |
| A rental billed 0.797 USD/h for an offer of 0.543 | The offer's price holds 5 GB of disk and the rental asked for 219 GB, of which the trace used 53 | The offers are priced with their disk. **The disk asked for is still a field campaign's**: sizing it for a trace (the pack, its rows, the cache and a margin, about 90 GB) is not done | half in code |

### Every card and every core (lot L13, phase one: not yet run on a card)

After its solves that first run left seven cards and about seventy cores
idle for the two to three hours its other stages took in one process. The
machine's command is now one queue for the solves, the rays, the early
trace, the levelling and the pack's rows, a process a card and a process a
core (`docs/adr/0016-appendix-every-card-every-core.md`):

- nothing is asked of the operator: the cards are read by `nvidia-smi`, the
  cores by the container's quota, and each worker measures what it holds.
  `--host-workers N` on `python -m reverberate.accel campaign` overrides the
  cores taken; `--host-workers 0` is the one process of before;
- the log says the machine as it found it and `predicted: H h of work on
  this machine` before the first long job; `prediction.json` holds it. The
  driver passes `--max-hours` (what its watchdog leaves): a run predicted
  over it stops there, `campaign.failed` saying so, once the solve's rate is
  calibrated on a card. Until then the line is said and the run goes on;
- a launch's records are in the host's memory, not on its card: a card
  needs about 4 GB for the grid to 1500 Hz, the launches are sized for the
  smallest card of the machine, and the host wants 0.5 x its RAM for them.
  **Watch the host's memory, not the cards'**, on the first run;
- a worker's own lines are in `workers/<n>.log`; `status.json` counts jobs
  (`job: done/all`) across every stage, and `trace_report.json` (`pool`)
  holds each stage's work and wall and each worker's busy seconds;
- `jobs/` holds the jobs' files while the run lasts (the pack's rows are the
  pair cache's size again) and is removed when the pack is checked: **the
  disk needs the pair cache twice** besides the pack.

Since lot L15b (the laptop alone: no card has run it) the early trace's
pairs are sieved and validated by a C text the machine's own compiler
builds at the first run (`cc`, `gcc`, `clang` or `CC`; 0.25 s, kept under
`~/.cache/reverberate/native` or `REVERBERATE_NATIVE_CACHE`), and what its
workers prepared each for themselves is kept once in the run's
`mirror_store`:

- the log's line `paths: sieved and validated by the compiled text` is the
  one to look for. `paths: on numpy, about ten times the seconds` names a
  machine without a compiler: the tables are the same, the stage is not
  hidden under short solves any more. Install one, or accept it;
- `mirror_store` is about 8 GB for the whole scene (trees, distance
  fields). `REVERBERATE_MIRROR_STORE=DIR` puts it where a second run or a
  second scene of the dwelling finds it; deleting it costs time only.
  Under that variable the tail's histograms are kept there too (`DIR/tails`,
  an entry a (site, cell)): a second recipe of the dwelling casts rays only
  from the sites, and for the cells, that no earlier recipe read. Without
  it they stay in the run's own `tails`;
- the rays are cast through a hierarchy of boxes
  (`docs/open-questions/ray-tracer.md`): 0.58 s a site of 100 000 rays on
  an RTX 3090 Ti where the uniform grid took 9.7 s, and the same
  histograms. `REVERBERATE_RAY_STRUCTURE=grid` casts them through the grid
  again;
- `REVERBERATE_NO_NATIVE=1` forces the `numpy` twin;
  `REVERBERATE_PATHS_ON_CARD=1` lets a card's worker sieve and validate on
  its card, and is off until a card has shown one digest with and without
  it (the commands in the appendix);
- `trace_report.json` says what made each table (`paths`, `engine`), what
  the store made and what was read from it (`mirror_store`), and the
  distance fields solved and read.

### The options that change the result

Each leaves the default pack byte for byte as it was, and each is a variant
the owner judges by ear (`docs/open-questions/listening-variants.md`):
`--low-ppw 7.2`, `--rail-positions 8` on a recipe of another rail pitch,
`--low-seconds 0.8`, `--rays 50000`. `python -m reverberate.trace variants`
prices a named set of them on the whole scene and prints the command of
each one's excerpt; `python -m reverberate.render check REF.h5 --against
V1.h5 V2.h5 ...` writes a file to hear for each and a blind set; the audit
page switches among them at one instant.

### Two grids, kept side by side

The validated grid and the coarser one are both first class. `--low-ppw`
is read by `trace rent`, `trace bundle` and the machine's command; the grid
is voxelised on the machine under its own key, its pairs are cached under
that key and under a solver's name that says the grid, so a pair of one is
never found for the other, on the machine, in `data/cache/low-pairs/` or in
the store. A home holds one grid's run: `trace rent` refuses a home whose
bundle names another. The recipe's `voxel_low_key` then differs by intent,
that key alone, and the pack's provenance says so (`assets_mismatched`).

**What the second run does not pay again.** With `--reuse-from` the bundle
carries the first run's early tables, and those of the sources are not
traced again: they depend on the sources, the listener and the mirror's
scene, and the trace's region is rounded out to a quarter of a metre so
that it is the same on both grids. **The rays, the pairs at rest and the
levelling are made again**: the tail's cells and the pairs' cells are the
centres the arrays really got, which are the grid's nodes and differ from
one grid to the other by up to half a step. On the first recipe that is
165 s saved of 2400 s outside the solves; the rays are the larger part and
divide over the cards.

On the coarser grid an array's ball is twelve steps of 3.2 cm, 0.38 m
against 0.26 m: more cells near a surface or a source may get no array. The
trace says how many (`cells without an array`) and serves their steps from
the nearest cell that has one.

### The low band's own commands

Before a scene is trusted to another grid or another card, the dev box runs
`verify`, `compare` and `cost` of `python -m reverberate.wave.lowband`, in
that order; `docs/open-questions/low-band-solver.md` gives the lines.
`compare --line` now delays the pairs by the field's lead before it compares
them, as the trace does (`spatial.lowband.delayed`): it read 0 dB at every
band while the two were 10.75 ms apart. The lead is the line's own where
`line` found it in the field, or `--lead-s`; without either the command
refuses to print a table.
