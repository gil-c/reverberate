# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
