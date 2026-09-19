"""The geometric mirror of the wave solver: image sources for the early part, rays for the tail.

Calibrated on the wave field (:mod:`reverberate.mirror.calibration`) and
joined to it under a crossover (:mod:`reverberate.mirror.hybrid`), it answers
the band where the wave solver is most expensive. :func:`reverberate.mirror.pipeline.run`
makes one source's field; ``python -m reverberate.mirror`` runs it, the
calibration and the join.
"""
