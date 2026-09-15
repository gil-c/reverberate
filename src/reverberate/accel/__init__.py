"""The campaign's stages on one GPU machine, each a twin of the CPU code it replaces.

Two campaigns of ``reverberate.experiments.w40_volume_field`` ran the same
field on four kinds of machine: a CPU box voxelised, a card solved, four more
boxes encoded, and the laptop planned, audited and assembled, with 128 GB of
pressure crossing the network in between. This package keeps every stage on
the machine that holds the card, and moves the two that were bought by the
core -- the voxelisation and the ambisonic encoding -- onto the card itself.

**Every module here has a CPU twin, and the twin is the specification.**
The voxeliser reproduces PFFDTD's ``vox_out.h5`` byte for byte; the encoder
reproduces :mod:`reverberate.spatial.encode` and :mod:`reverberate.audio` to
a stated tolerance, measured and recorded per campaign. Where the GPU cannot
be exact (an LU solve, an FFT), the difference is measured against the CPU
path on the same input and reported, never assumed.

Nothing in this package rents a machine. Renting, provisioning, watching and
fetching live in :mod:`reverberate.gpu.onebox`, so the library runs the same
on a laptop's CPU (with ``numpy``), on any NVIDIA card (with ``cupy``), and in
the tests, which never see a card.

- :mod:`.backend`: which array library, and the compile rules of the kernels.
- :mod:`.scene`: PFFDTD's ``RoomGeo`` and ``tris_precompute``, op for op.
- :mod:`.lattice`: the Cartesian grid, the voxel lattice, the triangle index.
- :mod:`.voxelise`: the adjacency of every grid node, on the card.
- :mod:`.dsp`: the filters between the engine's pressure and the encoder.
- :mod:`.encode`: the order 7 fit of every listening point, batched per bin.
- :mod:`.solve`: the engine on this machine, a band in slices when RAM is short.
- :mod:`.campaign`: the resumable driver that runs on the machine.
- :mod:`.bundle`: what the laptop prepares and what it takes home.
- :mod:`.verify`: the comparisons that say whether a stage reproduced its twin.
"""

from __future__ import annotations

__all__: list[str] = []
