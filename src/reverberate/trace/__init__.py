"""The scene trace: a recipe, on a rented card, into the scene pack the signal engine renders.

ADR 0016 splits a moving scene in two: what the room does, computed once on
a card, and the audio, rendered from it on a laptop. This package is the
first half's last stage. It joins what the other lots made, each the one
implementation of its part, and adds nothing of acoustics but the levelling
between them:

- :mod:`.plan`, pure, on the laptop: the listening cells a recipe needs, the
  (source position, cell) pairs its audible steps read, the tail's sites
  and cells, and the cost before any rental;
- :mod:`.run`, on the machine, ``cupy`` or ``numpy``: the pairs campaign
  (:mod:`reverberate.accel.pairs`), the batched early trace and its onsets
  (:mod:`reverberate.mirror.moving`), the tails (:mod:`reverberate.mirror.tails`),
  the levelling, and the pack through :class:`reverberate.render.pack.PackWriter`;
- :mod:`.level`: the three scalars nothing else computes, ``low/seam_db``,
  ``level/high_gain_db`` and ``level/onset_s``, which make the pack at rest
  the hybrid of :mod:`reverberate.mirror.hybrid`;
- :mod:`.clock`: whether the two bands are on one clock and one scale, read
  on each pair where the mirror puts its direct sound;
- :mod:`.machines`: what a trace takes on an offered machine, its fetch
  included, and what a run took against it;
- :mod:`.assets`: the mirror of a dwelling as a trace reads it, and the
  recipe's asset keys;
- :mod:`.engines`: where the pairs come from, the card's campaign or a
  monopole in free air for a laptop;
- :mod:`.bundle` and :mod:`.driver`: the directory the machine needs, and
  the one command that rents, runs, fetches, destroys and verifies.

``python -m reverberate.trace`` is the command line (:mod:`.cli`).
"""

from reverberate.trace.assets import MirrorAssets, found_assets, mismatched
from reverberate.trace.plan import Plan, Profile, estimate, make_plan

__all__ = ["MirrorAssets", "Plan", "Profile", "estimate", "found_assets", "make_plan", "mismatched"]
