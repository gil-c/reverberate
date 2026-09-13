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
