"""What every test file shares: one solve of the scheme's wavenumber a grid and a session.

:func:`reverberate.spatial.encode.numerical_wavenumber` is a function of its
arguments and nothing else, and it is most of what an encoding costs on a short
record: sixty bisections over 576 directions for every frequency. The tests
of the encoder, of the W10 report and of the accelerated pipeline encode the
same few records on the same few grids a dozen times a file. Here the function
is the library's own, called once for each set of arguments a session meets
and its answer handed back, copied, to whoever asks again. A test that holds
the function itself calls it with arguments of its own and so runs it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import reverberate.accel.encode as accel_encode
import reverberate.spatial.encode as spatial_encode


@pytest.fixture(autouse=True, scope="session")
def _one_wavenumber_a_grid() -> Iterator[None]:
    solve = spatial_encode.numerical_wavenumber
    solved: dict[tuple[Any, ...], np.ndarray] = {}

    def once(
        frequency_hz: np.ndarray, grid_step_m: float, sound_speed_m_s: float, **more: Any
    ) -> np.ndarray:
        frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        key = (
            frequency.shape,
            frequency.tobytes(),
            float(grid_step_m),
            float(sound_speed_m_s),
            tuple(sorted(more.items())),
        )
        if key not in solved:
            solved[key] = solve(frequency_hz, grid_step_m, sound_speed_m_s, **more)
        return solved[key].copy()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spatial_encode, "numerical_wavenumber", once)
        patch.setattr(accel_encode, "numerical_wavenumber", once)
        yield
