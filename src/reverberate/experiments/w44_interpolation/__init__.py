"""W44: what a listening point's response can be predicted from, when it was not solved.

A scene in which the listener moves needs a response between the points of
the field, and a source that moves needs one between its solved positions.
This experiment measures, on fields the project already owns, how far each
way of filling the gap holds, per octave, on the omnidirectional channel:

- the nearest neighbour as it is, and the mean of two;
- the same mean after each neighbour is put on the target's direct delay;
- one neighbour's order 7 expansion evaluated at the target (translation);
- several neighbours' expansions fused as one field of plane waves.

What it found on hssd_0076: a mean of two responses fails from 500 Hz at
0.40 m; an order ``N`` expansion translates to about ``k d = N``, 1 kHz at
0.40 m; the fusion of two neighbours reaches -23 dB at 1 kHz on that lattice;
and translation fails where a surface or the source is nearer than the
distance translated.

:mod:`.translate` is the mathematics, two functions that read no file:
:func:`~.translate.translation_weights` and
:func:`~.translate.fusion_operator`. :mod:`.scoring` is what every
measurement here shares: the octave bands, the early window, the error in
decibels. :mod:`.leave_one_out` and :mod:`.plane_wave` leave each point of a
lattice out in turn, :mod:`.line_gaps` reads the error against the spacing on
a line solved every 2 cm, and :mod:`.translation_failures` looks at where the
translation fails round 500 Hz.

Usage::

    python -m reverberate.experiments.w44_interpolation leave-one-out \\
        --field data/runs/w42_gpu_hssd_0076/field/S1.h5 --out data/runs/w44_x/leave_one_out
    python -m reverberate.experiments.w44_interpolation plane-wave --field ... --out ...
    python -m reverberate.experiments.w44_interpolation line-gaps --field ... --out ...
    python -m reverberate.experiments.w44_interpolation failures --field ... --out ...
"""

from reverberate.experiments.w44_interpolation.translate import (
    fusion_operator,
    translation_weights,
)

__all__ = ["fusion_operator", "translation_weights"]
