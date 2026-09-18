"""The mirror: a geometric engine held against the wave solver.

Image sources projected onto spherical harmonics for the discrete early part,
a stochastic ray tracer with directional histograms for the tail, both on
the card with a numpy twin, run on a geometry *derived* from the solver's
own model by :mod:`reverberate.mirror.geometry`. What the mirror produces is
judged by :mod:`reverberate.mirror.criteria` against the field the wave
solver produced at the same points, and its material parameters are fitted
to that judgement by :mod:`reverberate.mirror.calibrate`.

Roadmap section 15 says "no second solver". This package is one, and the
decision to build it is ADR 0014: the wave solver stays the reference and
the mirror is priced against it, never the other way round.
"""
