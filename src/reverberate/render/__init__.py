"""The signal engine: a scene pack and dry audio in, the order 7 signal at the listener out.

ADR 0016's last stage. A scene is traced once on a rented card into a pack
(``docs/formats/scene-pack.md``); this package applies the pack to the dry
clips and writes the order 7 ambisonic signal at the moving listener, 64
channels at 48 kHz, in blocks, for the audit on the laptop and later for the
card that trains. It needs nothing else: no geometry, no solver.

**One code on two devices.** Every module takes its array module as ``xp``
(:func:`reverberate.compute.xp_for`): ``numpy`` on the host, ``cupy`` on a
card. Nothing here imports ``cupy``. Host and card agree to 1e-6 of the
peak and are not bit identical, a transform not rounding alike on the two;
the tail's noise, which was the one thing that differed in law, is drawn by
one generator whose bits are the same on both.

- :mod:`.pack`: the format's types, writer, reader and invariants, and the
  synthetic free field. The trace stage writes through it.
- :mod:`.engine`: :class:`~.engine.Engine`, stems, mixes and streams of blocks.
- :mod:`.early`: every arrival a moving delay line into the listener's basis.
- :mod:`.delay`: the delay line's interpolator and its measured error.
- :mod:`.low`: the solved responses under the crossover, moved to the head.
- :mod:`.translate`: the one interface the translation's mathematics sits behind.
- :mod:`.tail`: the late part, noise shaped by the step's histogram.
- :mod:`.noise`: the counter based generator.
- :mod:`.dry`: a source's clips on the scene's clock, and what is filtered once.
- :mod:`.output`: the signal on disk (``docs/formats/scene-signal.md``).
- :mod:`.benchmark`: what a second of scene costs, part by part.
- :mod:`.check`: a rendered scene's sound held to what a listener would reject.

``python -m reverberate.render benchmark`` measures the cost on this
machine; ``interpolator`` prints the delay line's error; ``validate`` checks
a pack; ``check`` measures a pack's sound and writes files to hear.
"""

from __future__ import annotations

__all__: list[str] = []
