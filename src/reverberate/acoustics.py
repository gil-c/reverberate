"""The frequency bands everything in this project is expressed on.

One definition, imported everywhere, because the band range is not a local
choice: material coefficients, the simulation and the metrics all have to
agree, and a mismatch between any two of them is silent. `pyroomacoustics`
already works on these seven octave bands and every material in its table is
expressed on them, so adopting them costs nothing and avoids resampling
anyone's data.

The range runs to 8 kHz rather than stopping at 4 kHz, so that a metric can be
quoted on the band a listener still hears detail in.

It used to carry a second job: ``MIN_WAVELENGTH`` was the floor for how far
geometry could be simplified. Roadmap section 6.3 retires that argument, since
the wave solver's cost is set by the bounding box rather than by triangle
count, and nothing simplifies geometry any more. These bands are now what they
say they are and nothing else.
"""

from __future__ import annotations

import numpy as np

#: Speed of sound in air at room temperature, m/s.
SPEED_OF_SOUND = 343.0

#: Density of air at room temperature, kg/m^3. Used by the porous layer model
#: in :mod:`reverberate.materials.extrapolation`.
AIR_DENSITY = 1.204

#: The octave band centres, in Hz. Identical to ``pyroomacoustics``' own bands,
#: which is what lets a material from its table be used without resampling.
OCTAVE_BANDS: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000, 8000)

#: The eleven octave bands the wave solver's impedance fit is expressed on, in
#: Hz. These are not a choice: PFFDTD's ``fit_to_Sabs_oct_11`` hard-codes
#: ``1000 * 2 ** arange(-6, 5)`` and asserts it is given exactly eleven
#: coefficients, so a material reaches the solver on these centres or not at
#: all.
#:
#: They are deliberately kept separate from :data:`OCTAVE_BANDS`, which is what
#: `pyroomacoustics` and the metrics work on. The two ranges answer different
#: questions -- what the solver is told about a boundary, and what a measured
#: impulse response is summarised on -- and collapsing them would mean either
#: resampling every published table or claiming a 16 Hz metric nothing in this
#: project can measure. :func:`reverberate.materials.db.AcousticClass.solver_absorption`
#: is the bridge.
SOLVER_BANDS: tuple[float, ...] = tuple(1000.0 * 2.0 ** np.arange(-6, 5))


def band_index(frequency: float) -> int:
    """The octave band a frequency falls in, as an index into ``OCTAVE_BANDS``."""
    return int(np.argmin(np.abs(np.asarray(OCTAVE_BANDS, dtype=float) - float(frequency))))
