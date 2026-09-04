# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `reverberate.geometry.carve`, which carves HSSD's collision proxies back to
  the shape the render mesh proves. The colliders are convex decompositions, so
  everything hollow or concave reaches the grid as a solid lump: measured on
  `bedroom.001` of `102344022`, unioned collider volume against render volume is
  176x on the basket, 49x on a decoration, 39x and 24x on the lamps, 26x on the
  wardrobes, 12x on the carpet, and **2.02x over the whole room**. That is what
  makes the voxel view look inflated beside the mesh view, and it changes surface
  area and the volume patch 5 seals along with it.

  The render mesh cannot be substituted -- not one in the room is watertight,
  they run to 28 668 disconnected shells apiece, and they carry 647 k triangles
  against the colliders' 73 k -- but it can say where there is air. So the
  collider solid and the render surface are rasterised onto a 6 mm grid, the
  complement of the surface is flood filled from the border, and whatever that
  fill reached is removed from the solid; marching cubes returns one closed body,
  decimated to a triangle budget. Worst cases on this room come back at 2.4x
  from 97.3x, 2.4x from 39.2x and 1.3x from 35.9x.

  It is never a silent substitution. A carve that comes back empty or open, or
  that no reduction can bring inside the budget while staying closed, is
  discarded for the plain collider, and `CarveReport` names every template in
  each case in the manifest. Three things it took to get right, each measured
  rather than guessed: trimesh's `method="ray"` leaked on 42 of 47 templates and
  is replaced by a conservative bounding-box rasteriser; eroding the finished
  carve rather than only the collider's solid took one wardrobe to 0.1 per cent
  of itself, trading an object that is too fat for one the grid may not resolve;
  and a single decimation target refused 17 of 41 templates, so the budget is a
  back-off ladder with an absolute cap under it.
- `scripts/check_vox_index.sh`, patch 6's acceptance test: one bedroom
  voxelised with and without the index, `vox_out.h5` compared byte for byte.

- `reverberate.experiments.scene_export`, `.run`, `.compare` and `.engine`, the
  measurement harness that had survived four sessions as four throwaway scripts
  under `data/runs/`. Named for what they do rather than for the roadmap item
  that first needed them: export a scene and copies of it cut back to a path
  length budget, run one domain here or on a rented machine, and say where one
  response departs from another. W1's `common` and `padded` bounds are kept as a
  required argument with no default anywhere, and the chosen mode is carried in
  the `Bounds` value, in the run directory's name and in the run's JSON, so a
  pair of bounds can never be recorded without saying how it was arrived at.
  Voxelisation now goes through `reverberate.wave` instead of importing PFFDTD's
  `sim_setup` in process, so the harness is no longer pinned to numpy below 2
  and one voxelisation is cached across runs.
- `reverberate.materials`, the catalogue as one package with both faces of
  section 6.1: absorption per band, the two octaves above the last published
  measurement, and the impedance filter the wave solver reads. Fitting is
  PFFDTD's own `fit_to_Sabs_oct_11` at the pinned commit, cached content
  addressed on the eleven-band curve, and every fitted filter is checked for
  passivity on 4000 frequencies rather than argued to be passive. All 27
  classes pass. Report at `data/interim/materials/report.md`.
- `reverberate.geometry.orientation`, which decides once per surface which side
  the air is on and records it in the scene, instead of leaving each exporter to
  invent a value. A watertight, consistently wound mesh has its normals pointed
  at the air and is exported front-side only; anything open or inconsistently
  wound is exported active on both sides, which is wasteful but can never be
  silently rigid. `GeometrySummary` reports how many faces fell in the second
  case.
- `tests/test_determinism.py`, a cross-process determinism test. The existing
  tests all ran within one interpreter, which is the one case the defect under Fixed
  did not affect.
- `reverberate.wave`, the split pipeline: voxelise locally on a CPU, cache the
  result content addressed on the scene and the grid, and rent a GPU only for
  the solver. Includes the path that did not exist before, regenerating
  `comms_out.h5` for a new source and receiver pair without rerunning
  `sim_setup`, so one voxelisation is amortised over every pair in a room. Its
  output is bit for bit identical to `sim_setup`'s, Cartesian and FCC, which is
  what the slow test in `tests/test_wave_comms.py` checks against a real run.
- `reverberate.gpu.vast`, renting GPUs from Vast.ai with the teardown guaranteed
  rather than remembered: no rental without a deadline, a detached watchdog armed
  before the call returns, verified destruction, a spend ledger, and a refusal
  when a rental would pass the project ceiling.
- `scripts/build_pffdtd.sh`, a reproducible PFFDTD build on a CUDA machine,
  pinning the commit our cost figures were measured against and applying the four
  fixes the 2021 source needs on a current stack.
- `reverberate.settings`, the single data root setting every stage writes under.

### Fixed

- **The voxelisation cache was never published, so a grid died with the worktree
  that computed it.** `reverberate.wave.vox_store` has been able to push and pull
  a cache entry since it was written, and nothing ever called it: both callers of
  `voxelise` passed `store=None`. W29's 16 kHz bedroom -- 63 430 624 boundary
  nodes, five hours of A100 -- was computed in a worktree that has since been
  removed, and the geometry behind a finished measurement could no longer be
  looked at. `reverberate.store.shared_store` now answers "the bucket, or nothing"
  once per process, `voxelise` receives it from the experiment harness and from
  the CLI, and a computed entry is published.
- **A run page with no grid drew the exported triangles under the mode that
  promises the solver's own.** `_write_voxels` returned `None` and said nothing,
  so a missing terabyte read as a rendering choice. It now looks in this
  machine's cache, then in the root the report names, then in the shared store --
  the key is content addressed, so any copy found under it is the same grid --
  and when none of the three has it, the build log and the page itself say so.
- **The same scene exported from two processes gave different geometry**,
  608 098 triangles against 614 330. `trimesh.sample.sample_surface` was called
  without its `seed`, so it drew from OS entropy, and the 95th percentile it
  feeds is the accept-or-reject test that picks a face budget. The seed already
  threaded through `simulation_geometry` now reaches it.
- **Every surface pointing the wrong way was silently rigid in the solver.** The
  B0 and B1 exports wrote PFFDTD's `sides = 2` for every triangle believing it
  meant "two sided". It means *front side only*: `vox_scene.py` marks the
  boundary nodes on the normal's other side as rigid. Those surfaces kept their
  area in every report and contributed no absorption at all. Contrary to what
  was assumed, the boundary node counts and therefore the timings were never
  inflated by this, since sidedness never enters the adjacency computation.
- **The voxelisation cache was keyed on the source and receiver positions.** It
  hashed the whole model file, which holds the exported mesh, correctly, but
  also the sources, the receivers and the export timestamp. Moving a receiver by
  a centimetre forced a fresh voxelisation, which is the entire cost the split
  pipeline exists to amortise. The key is now taken over the mesh alone, and a
  file whose layout is unrecognised is still hashed whole rather than guessed
  at. Entries cached under the old key are orphaned and can be deleted.

### Changed

- **The catalogue no longer stores anything above 4 kHz as though it were
  data.** The 8 kHz column was the 4 kHz value repeated, which was defensible
  across one octave and is not across two now that the architecture runs to
  16 kHz. The column is gone from `acoustic_classes.csv`; 8 and 16 kHz are
  derived per class by continuing the material's own measured 2-to-4 kHz
  ratio, clipped to at most 1 and at least 0.8. A Delany-Bazley layer model
  was tried first and rejected on its own residual, which reaches 0.11 in
  absorption units on the *measured* bands and disagrees in shape with every
  class that is already falling by 4 kHz; it is kept as the diagnostic the
  report quotes. 19 of 27 classes end up holding 4 kHz anyway, now because
  their own measurements say to.
- `reverberate.materials_db` moved to `reverberate.materials.db`, and the
  coefficient tables moved with it. `reverberate.geometry.materials` is
  unchanged as an entry point.
- `reverberate.auth` now searches the shared `dev-common` credential namespace as
  well as the project's own, so credentials shared across projects resolve. The
  project namespace is searched last, so it still wins on a name collision.
