"""The wave engine's half of the pipeline, split in two so the GPU is idle less.

Section 11 of the roadmap: voxelisation is CPU work and the solver is GPU work,
and PFFDTD does both in one call on one machine. This package separates them.

- :mod:`reverberate.wave.voxelise` runs the CPU half locally and caches it,
  content addressed on the scene and the grid.
- :mod:`reverberate.wave.comms` places one source and its receivers on a cached
  grid, so a voxelisation is amortised over every pair in the room.
- :mod:`reverberate.wave.remote` ships the four files the engine reads to a
  rented machine, runs it, and retrieves the one file it writes.
"""

from __future__ import annotations

from reverberate.wave.comms import ENGINE_FILES, Grid, load_grid, write_comms
from reverberate.wave.remote import Machine, SolveResult, solve
from reverberate.wave.voxelise import (
    CACHE_FILES,
    CacheEntry,
    SceneSpec,
    engine_inputs,
    entry_for,
    voxelise,
)

__all__ = [
    "CACHE_FILES",
    "ENGINE_FILES",
    "CacheEntry",
    "Grid",
    "Machine",
    "SceneSpec",
    "SolveResult",
    "engine_inputs",
    "entry_for",
    "load_grid",
    "solve",
    "voxelise",
    "write_comms",
]
