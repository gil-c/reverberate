# Handover: what is done, what is not, and what the next agent should not repeat

Branch `worktree-remote-sync-4ceafb`. Written at the end of the session that
produced it, for whoever picks it up. Delete this file when its contents have
been absorbed into the roadmap.

Everything below that carries a number was measured, not estimated, unless it
says otherwise.

---

## 1. What is finished and checked in

**Three defects in the geometry the solver receives**, each measured before it
was touched.

- *Missing objects.* The room's furniture was scoped by `room_of` on the
  instance's **origin point**, which for anything on a wall lands in the wall
  band and answers `"doorway"`. Every picture on the bedroom's own walls was
  dropped. Scoping by footprint against the same polygon the shell comes from
  gives **17 obstacles → 21**, and `picture` appears in the grid as 1 890 687
  nodes covering 7.89 m².
- *Inflated objects.* The solver was given HSSD's collision proxies, which are
  convex decompositions: **2.02× the render volume** over the bedroom, 176× on
  the basket. `geometry/carve.py` carves them back with the air the render mesh
  proves is there. Worst cases 97.3× → 2.4×, 39.2× → 2.4×. It also closed
  PFFDTD's own area deficit: `seat` **−64.4 % → −1.4 %**.
- *Cost.* PFFDTD's `vox_grid_base.fill` scanned every triangle for every voxel.
  Patch 6 indexes it. The flat's fill at 4 kHz: **61 626 s → 67 s**.
  `vox_out.h5` is byte-for-byte identical, checked by
  `scripts/check_vox_index.sh`.

**Slabbed voxelisation.** `SceneSpec.slabs` consolidates a group of voxels at a
time. Peak memory is a slab, not the grid, which is what made the whole flat at
16 kHz possible at all: **1 089 464 499 boundary nodes, 25.18 GB, 3.08 h**,
peaking around 6 GB where a single pass needs ~53. `scripts/check_slabs.sh` is
the acceptance and it is identity: whole, 2 slabs, 5 slabs, another voxel side,
and both together — all four datasets identical to the single pass.

**Remote voxelisation.** `wave/remote_voxelise.py` puts PFFDTD's Python side on
a rented CPU, which roadmap section 11 assumes is possible and nothing had ever
done. Proven end to end on one bedroom at 4 kHz.

**Grids that exist on this machine** (`data/cache/vox/`):

| key | scene | fmax | nodes | size |
| --- | --- | --- | --- | --- |
| `c075466f…` | apartment | 4 kHz | 66 159 665 | 1.5 GB |
| `9ef0bd3e…` | apartment | **16 kHz** | 1 089 464 499 | 25.2 GB |
| `78bca376…` | bedroom | 16 kHz | 65 908 875 | 1.5 GB |
| `bddc127c…` | apartment | 1 kHz | 3 701 701 | 0.1 GB |

**Viewable now.** The whole flat at 4 kHz, losslessly: 2 316 983 quads, 185 MB,
built in 2.3 min. `data/runs/w32_apartment_4k` with `viewer_cubes` set to force
one block per node.

---

## 2. What I said I would do and did not

### 2.1 The room partition — not written at all

No file, no function. This is the next piece of work and everything below it
depends on it. What it has to do:

- Assign **every** node of a grid to exactly one of 13 rooms, so that the sum of
  the per-room node counts equals the grid's own count. Assert that. A partition
  that loses or duplicates a node is not an audit.
- Rooms are the 12 regions of at least 4 m² plus the garage. The seven closets
  merge into the room they **open into**, decided by `apartment.find_doorways`
  and *not* by shared boundary length — the geometric criterion is genuinely
  ambiguous here (2.10 m against 2.10 m on several closets) and disagrees with
  the doorway answer on five of seven. The doorway mapping is:

  | closet | room | | closet | room |
  | --- | --- | --- | --- | --- |
  | `closet` | other room | | `closet.004` | other room.001 |
  | `closet.001` | bedroom | | `closet.005` | **bedroom.001** |
  | `closet.002` | hallway | | `closet.006` | other room.001 |
  | `closet.003` | **bedroom.001** | | | |

- Nodes outside every room polygon — wall interiors, doorway bands — still need
  an owner. Nearest room is the obvious rule; whatever is chosen has to keep the
  partition strict.

### 2.2 The 16 kHz swap in the viewer — not started

The agreed behaviour: standing in a room, that room draws at 16 kHz and
**everything visible beyond it draws at 4 kHz** — the hallway, other rooms seen
through doorways. Substitution in place, not isolation.

This needs the 4 kHz payload partitioned by room as well, so the base can be
drawn as "every room except the one you are in". A shared wall belongs to
exactly one room; drawing it in both is what a strict partition prevents, and an
earlier note in this session got that wrong.

### 2.3 Slicing `surface_of` — not done, and it blocks 2.2

`_dense_labels` allocates one array over the region's whole bounding box.
Measured 4.7 GB for the flat at 8.17 mm; **~31 GB for the living room at
2.04 mm**, which is past this machine. `vox_view.TARGET_CUBES`' own docstring
already names this: *"Slicing it is the work that unlocks native, and it is not
done."* It still is not.

### 2.4 The whole flat at 16 kHz on a rented machine — four attempts, none finished

The local grid exists, so this is now only worth doing as an independent check,
or to build per-room payloads without using the laptop. Section 4 lists what
went wrong.

### 2.5 Smaller things left undone

- **`mirror` regressed** with the carve: realised surface −0.4 % → **−41.2 %**.
  Its α is 0.03 so the acoustic cost is small, but it is real and unexplained.
- **17 of 41 bedroom templates are not carved** (58 of 135 for the flat). They
  fall back to the plain collider and stay inflated. Named in the manifest.
- **`calc_adj` is the bottleneck now**, 1 420 s of the flat's 1 655 at 4 kHz and
  10 996 of 11 104 at 16 kHz. It allocates about eight full-voxel numpy
  temporaries per triangle. The fill is no longer worth optimising.
- **PFFDTD's disk guard is wrong for the slabbed path.** It demands twice the
  whole grid — 302 GB free at 16 kHz — for `check_adj_full`, which a slabbed run
  never executes. Fixing it roughly halves the rental cost. It also prompts on a
  stdin the child has consumed, so too little space reads as a hang.
- **`ROADMAP_v7.md` is not updated.** W31 is delivered and W21 has advanced.
- **`data/cache/carve/` holds entries from several source versions at once.**
  The stamp includes the module's digest, so old entries are never *served* --
  but they are never swept either, and comparing against a glob picks them up
  indiscriminately. That produced two false "the carve changed" alarms in one
  session. A prune of stamps other than the current one would help.

---

## 3. Things that will bite, with the measurement

- **The viewer's block budget, not the grid, decides what you see.** This flat
  comes out at 16.34 mm blocks at 4, 8 *and* 16 kHz under the default, because
  the budget counts blocks and the surface area is the same at all three. A
  finer voxelisation is invisible without raising `viewer_cubes`.
- **A lossless whole-flat view at 16 kHz is ~35 M quads, ~2.8 GB.** Not
  drawable. Extrapolated from the bedroom's measured 30 060 → 120 888 → 475 740
  quads as the block halves. This is why the design is 4 kHz everywhere plus one
  room at 16 kHz.
- **`_bin_voxels` does not survive a billion nodes.** It ran 7 h 37 at 15 % CPU
  with 34.4 GB of swap in use and never finished. Narrowing dtypes took the peak
  from 43.6 to 24.0 GB, which was not enough and was the wrong fix: everything
  is still resident at once. It needs dense accumulation over the **block**
  lattice — 289 M cells, ~3.8 GB — with the nodes streamed past it.
- **macOS ships openrsync**, which rejects `--append-verify`. Only `--partial`,
  `-P` and `--timeout` are common to both.
- **Vast's advertised `dph_total` is not what you are billed.** Reserved disk is
  added: an offer at 0.049 USD/h billed 0.102 with 200 GB. Every cost estimate
  in this session was low by about a factor of two until that was measured.
- **`SSL_CERT_FILE` must be set** for the Vast API from this venv, or `urllib`
  fails certificate verification.

---

## 4. The four rentals, and what each one bought

Total spent: **7.42 USD** on the ledger, of which this session added about 0.45.
No orphaned instances at any point.

| # | what failed | cost | the guard it left |
| --- | --- | --- | --- |
| 1 | `Machine.identity` was `None`; ssh refused `publickey` for the full timeout | 0.03 | the identity is matched against the account's keys **before renting** |
| 2 | `scp` of 25 GB closed by the remote host | 0.40 | `rsync --partial` with four retries |
| 3 | `--append-verify` does not exist on openrsync | 0.00 | portable flags only |
| 4 | the host never answered on ssh | 0.00 | exclude that offer id; it failed twice |

**Rental 2 is the one to learn from.** The voxelisation had *succeeded* and only
the retrieval failed, and the teardown in the `finally` could not tell the two
apart, so it destroyed a finished grid and its timings. That is W29's own lesson
— a grid dying with the machine that made it — repeated inside a file that cites
W29 in its docstring. `RetrievalFailed` now carries the distinction and the
instance is left running when the compute is what succeeded.

---

## 5. Where to start

1. Write the partition (2.1). It is self-contained, testable against the
   node-count assertion, and everything else waits on it.
2. Slice `surface_of` (2.3) using the same idea that made the voxelisation fit:
   process the lattice in slabs along one axis.
3. Then the swap (2.2), which can be demonstrated with 4 kHz on both sides
   before any 16 kHz payload exists — if the substitution works there, it works
   with the detail.
4. Rebuild `_bin_voxels` around the block lattice (3) when a billion-node
   payload is actually needed.

`make check` passes on this branch: ruff, ruff format, mypy strict, 435 tests.
