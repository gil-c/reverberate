"""A wave solver for the band under the crossover: many source positions a launch, at least cost.

A scene (ADR 0016) asks the wave solver for one thing: the response between
each of two thousand source positions and the few listening cells it is
heard at, under 1 kHz. PFFDTD solves one source a process, on the whole
bounding box, and hands its receivers' pressure to the host. This package is
a solver for that job alone (``docs/open-questions/low-band-solver.md``):

- :mod:`~reverberate.wave.lowband.scheme`: the two stencils, their
  dispersion and what each costs on paper;
- :mod:`~reverberate.wave.lowband.problem`: a voxelised dwelling cut to the
  air its sources reach, with the engine's own boundary;
- :mod:`~reverberate.wave.lowband.solver`: the step, for a batch of sources,
  on ``numpy`` and on a card, the receivers kept on the device;
- :mod:`~reverberate.wave.lowband.fit`: the records to the cache form, the
  order 7 fit prepared once;
- :mod:`~reverberate.wave.lowband.pairs`: the pairs campaign on it, which
  the trace reads through :class:`reverberate.trace.engines.BatchedPairs`;
- :mod:`~reverberate.wave.lowband.reciprocity`: which side of a pair to
  solve, counted;
- :mod:`~reverberate.wave.lowband.harness` and the command line: the
  solver against itself and the present engine, its error and its cost.

It is named for the band and lives beside the engine's own files because it
reads them: a cache entry of :mod:`reverberate.wave.voxelise` is its input,
on either grid.
"""

from reverberate.wave.lowband.problem import Problem, build_problem, load_problem
from reverberate.wave.lowband.scheme import CARTESIAN, FCC, SCHEMES, Scheme
from reverberate.wave.lowband.solver import Drive, drive_for, solve

__all__ = [
    "CARTESIAN",
    "FCC",
    "SCHEMES",
    "Drive",
    "Problem",
    "Scheme",
    "build_problem",
    "drive_for",
    "load_problem",
    "solve",
]
