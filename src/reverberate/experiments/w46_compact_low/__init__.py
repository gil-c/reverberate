"""W46: are a pack's low band responses a lever, and what does each way of pulling it cost.

The first scene of twenty minutes is a pack of 24 GB of which 21 are
``low/ir``: 16 887 pairs of ``[64, 4800]`` float32. This experiment
measures, on the scene's own pairs and on the dense line of W44 where the
truth between two cells is known, what is lost by keeping them in fewer
bytes (:mod:`reverberate.render.compact` is what was built from it, and
``docs/open-questions/low-band-compact.md`` what was found):

- :mod:`.pairs`, ``pairs``: every sample format, every degree cutoff and
  every length, on real pairs, read where a head hears it;
- :mod:`.pairs`, ``rails``: what the positions of one rail share at one cell;
- :mod:`.line`, ``degrees``: a degree under a frequency, against the solved
  truth at the head;
- :mod:`.line`, ``pitch``: how far apart two fused cells may be, against the
  source's distance.

Usage::

    python -m reverberate.experiments.w46_compact_low pairs --out OUT \\
        --pairs <a run's pulled/pairs> [--pairs ...] [--count 120]
    python -m reverberate.experiments.w46_compact_low rails --out OUT --pairs ...
    python -m reverberate.experiments.w46_compact_low degrees --out OUT \\
        --field <line>/field/S1.h5 --plan <line>/plan.json
    python -m reverberate.experiments.w46_compact_low pitch --out OUT --field ... --plan ...
"""
