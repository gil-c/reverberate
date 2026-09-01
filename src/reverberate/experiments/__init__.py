"""Measurements run against the real dataset, kept in the repository.

The test suite is offline and stub-driven by a hard constraint, so it cannot
hold anything that needs the 23 GB of HSSD on disk. These modules are the other
half: reproducible scripts that answer a specific numeric question against the
real data, print their result, and are cited when a decision is recorded.

They are kept under ``src`` rather than in a loose scripts folder so that
``mypy --strict`` and ``ruff`` cover them. An experiment that does not typecheck
is an experiment whose result nobody should trust.

The harness is three modules, named for what they do rather than for the
roadmap item that first needed them:

- :mod:`reverberate.experiments.scene_export` writes a scene, and copies of it
  cut back to a path length budget, as PFFDTD model JSON.
- :mod:`reverberate.experiments.run` chooses a domain, voxelises it through
  :mod:`reverberate.wave` and runs the solver here or on a rented machine.
- :mod:`reverberate.experiments.compare` says where one response departs from
  another, and whether that time scales the way the physics says it should.

:mod:`reverberate.experiments.engine` holds the few things the first and the
last two do identically. Production code in :mod:`reverberate.wave` is called
by these modules and never copied into them.
"""
