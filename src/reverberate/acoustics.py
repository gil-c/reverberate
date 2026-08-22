"""The frequency bands everything in this project is expressed on.

One definition, imported everywhere, because the band range is not a local
choice: material coefficients, the simulation, the metrics and the physical
limit on decimation all have to agree, and a mismatch between any two of them
is silent. `pyroomacoustics` already works on these seven octave bands and
every material in its table is expressed on them, so adopting them costs
nothing and avoids resampling anyone's data.

The range runs to 8 kHz rather than stopping at 4 kHz. That matters beyond
metrics: the shortest wavelength of interest sets the floor for how far
geometry may be simplified (section 5.3), and moving the ceiling up an octave
halves it, from 8.6 cm to 4.3 cm. Every deviation budget derived from
``MIN_WAVELENGTH`` therefore tightens by a factor of two.
"""

from __future__ import annotations

import numpy as np

#: Speed of sound in air at room temperature, m/s.
SPEED_OF_SOUND = 343.0

#: The octave band centres, in Hz. Identical to ``pyroomacoustics``' own bands,
#: which is what lets a material from its table be used without resampling.
OCTAVE_BANDS: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000, 8000)

#: Highest band centre, in Hz. The shortest wavelength of interest follows.
MAX_FREQUENCY = OCTAVE_BANDS[-1]

#: Shortest wavelength of interest, in metres: about 4.3 cm at 8 kHz.
#:
#: This is the physical floor for geometric detail. Section 5.3's argument is
#: that structure much smaller than this does not produce specular reflection,
#: it produces scattering, and scattering is already modelled by the scattering
#: coefficient rather than by the mesh. Simplifying past it stops being that
#: argument and starts being damage.
MIN_WAVELENGTH = SPEED_OF_SOUND / MAX_FREQUENCY


def wavelengths() -> np.ndarray:
    """Wavelength of each octave band, in metres, in band order.

    Used to decide, per band, whether a feature of a given size is something
    sound reflects off or something it simply flows around.
    """
    return SPEED_OF_SOUND / np.asarray(OCTAVE_BANDS, dtype=float)


def band_index(frequency: float) -> int:
    """The octave band a frequency falls in, as an index into ``OCTAVE_BANDS``."""
    return int(np.argmin(np.abs(np.asarray(OCTAVE_BANDS, dtype=float) - float(frequency))))
