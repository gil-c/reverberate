"""Parts a small application is made of: one purpose a page, the same components in each.

The page of :mod:`reverberate.viz.serve_room` shows everything a pack was
computed on, and is the inspector of that. A decision taken by ear wants a
page that holds what the decision needs and nothing else, and is thrown away
once it is taken. Such a page is a few hundred lines when its parts exist:

- :mod:`.server`: routes, static folders, a JSON or a binary answer.
- :mod:`.media`: what there is to play, a folder of the listening kit or
  files, read by time range and mixed with a gain a source.
- :mod:`.sonogram`: a spectrogram on a logarithmic axis and a fixed scale,
  the difference of two, and a source's level over time.
- :mod:`.blind`: an ABX test and a blind ranking, the key held here, the
  probability of the score by chance.
- :mod:`.balance`: the faders of a track list, saved for whoever writes the
  next recipes.
- :mod:`.scene`: where the listener and the sources are, from a pack.
- ``static/``: the page's side: the player, the sonogram, the scene, the
  track list. ``docs/apps.md`` says how each is used alone.

Nothing here renders: the signal is the engine's, the decode is the page's
own (``viz/app/scene/sound-decode.js``, on the head of
:mod:`reverberate.viz.decoders`).
"""

from __future__ import annotations

__all__: list[str] = []
