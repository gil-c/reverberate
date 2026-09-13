"""W40: the ambisonic field of a source over the walkable volume of one flat.

The consumer is a first person viewer in which a listener walks through the
apartment and hears every source without lag. What it needs is an order 7
ambisonic impulse response at every point it may stand on, for every source,
so the field is recorded on a grid of listening points at one ear height and
assembled per point exactly as ``w38_ambisonic_bands`` assembles one.

Three bands: low (1 kHz, 1.2 s) and mid (4 kHz, 0.4 s) on the whole storey,
high (8 kHz, 0.15 s) on the whole storey too when the card holds it, else on
the source's own room. A point without a high band gets its top octaves
synthesised from 0.8 x the mid band's ``fmax``; the field says which.

The pressure never comes home. The card solves and shrinks it, encode boxes
pull their shard and run :mod:`.remote.child_encode` on every core, and only
the encoded spherical harmonic signals are fetched: 64 channels at 48 kHz, a
few megabytes a point, against a few hundred megabytes of raw pressure.

One command runs a campaign end to end, from the mesh export and the three
grids (each with its viewer payload) to the field, and resumes from whatever
exists::

    python -m reverberate.experiments.w40_volume_field campaign --out <run> ...

The stages are also commands of their own (``plan``, ``prepare``, ``solve``,
``encode``, ``assemble``). What went wrong on rented machines and what the
driver does about it is ``docs/runbook-rented-machines.md``.
"""

from reverberate.experiments.w40_volume_field.assemble import assemble_field
from reverberate.experiments.w40_volume_field.campaign import campaign
from reverberate.experiments.w40_volume_field.encode import encode_sharded
from reverberate.experiments.w40_volume_field.plan import plan_field, prepare_field
from reverberate.experiments.w40_volume_field.solve import solve_on_card

__all__ = [
    "assemble_field",
    "campaign",
    "encode_sharded",
    "plan_field",
    "prepare_field",
    "solve_on_card",
]
