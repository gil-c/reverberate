# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- `reverberate.auth` now searches the shared `dev-common` credential namespace as
  well as the project's own, so credentials shared across projects resolve. The
  project namespace is searched last, so it still wins on a name collision.
