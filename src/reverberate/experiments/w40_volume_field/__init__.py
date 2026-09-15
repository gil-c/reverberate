"""The ambisonic field of a source over the walkable volume of one flat: its plan and its assembly.

The consumer is a first person viewer in which a listener walks through the
apartment and hears every source without lag. What it needs is an order 7
ambisonic impulse response at every point it may stand on, for every source,
so the field is recorded on a grid of listening points at one ear height and
assembled per point exactly as ``w38_ambisonic_bands`` assembles one.

Three bands: low (1 kHz, 1.2 s) and mid (4 kHz, 0.4 s) on the whole storey,
high (8 kHz, 0.15 s) on the whole storey too when the cards hold it, else on
the source's own room. A point without a high band gets its top octaves
synthesised from 0.8 x the mid band's ``fmax``; the field says which.

:mod:`.plan` chooses the grid and places every point's array; :mod:`.assemble`
turns the encoded bands into the field and ``walk.json``; :mod:`.storey` is
the export and the audit view's meshes. The solve and the encoding run on the
machine that holds the card, :mod:`reverberate.accel`; the two-night driver
that spread them over five machines was retired (ADR 0012).
"""

from reverberate.experiments.w40_volume_field.assemble import assemble_field
from reverberate.experiments.w40_volume_field.plan import plan_arrays, plan_field, plan_points

__all__ = ["assemble_field", "plan_arrays", "plan_field", "plan_points"]
