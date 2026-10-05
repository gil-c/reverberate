"""A rendered scene's sound, checked by measurement: what a listener would reject.

``python -m reverberate.render check <pack.h5>`` renders what it needs
through :class:`~reverberate.render.engine.Engine` and holds it to limits a
listener would hold it to: a wrong level, a click, a step when something
moves, a hole at the join of the two bands, a tail that is not the room's, a
source heard from the wrong side. Nobody listens here; the limits are
written with their reasons (:data:`~.run.LIMITS`) and never moved to make a
scene pass.

- :mod:`.measure`: each number from a signal, with no pack: tested on
  signals whose answer is known and on seeded faults.
- :mod:`.binaural`: the page's decode, offline: the same filters, the same
  head's matrix, the same blocks.
- :mod:`.clips`: the recipe's clips off the disk, and what stands in for a
  placeholder.
- :mod:`.run`: the three families of checks: impulse probes, continuity,
  the mix.
- :mod:`.report`: the whole from end to end, ``check.json``, ``check.md``,
  the files to hear and the plots.
- :mod:`.reference`: ``--reference-point``, a pack whose source stands on
  the validated field's own against that field, sample for sample.
- :mod:`.against`: ``--against``, two packs of one recipe side by side: the
  two ears of each at one gain, and their difference a third octave at a
  time. It judges nothing: the files are for someone to hear.

**What it cannot tell.** Whether the voices sound like people in that room:
timbre, naturalness, the tail's grain. Faults under its limits that add up.
A fault where no probe was sent (an impulse is sent at one to three
instants a source, the tones over one stretch of a few seconds). Whether
the validated field itself is right. And everything that depends on the
headphones.
"""

from __future__ import annotations

__all__: list[str] = []
