"""What every test file shares, in every process that runs tests.

**One solve of the scheme's wavenumber a grid and a session.**
:func:`reverberate.spatial.encode.numerical_wavenumber` is a function of its
arguments and nothing else, and it is most of what an encoding costs on a short
record: sixty bisections over 576 directions for every frequency. The tests
of the encoder, of the W10 report and of the accelerated pipeline encode the
same few records on the same few grids a dozen times a file. Here the function
is the library's own, called once for each set of arguments a session meets
and its answer handed back, copied, to whoever asks again. A test that holds
the function itself calls it with arguments of its own and so runs it.

**The fast engine's first use, before any thread.** The first render of a
process makes the delay table from several threads at once, and a loop in C
can be left reading a table another thread's has replaced
(``docs/open-questions/engine-speed.md``, "the same bytes"). That is a defect
of the engine and it has a test of its own, which starts processes that never
read this file: ``test_several_processes_write_the_file_one_engine_writes``,
marked ``quarantine`` and run by a job of its own. Every other test is kept
out of its way here, since under one process a core each worker is a fresh
process and whichever test renders first in it would fail one run in a few
for a reason that is not its own (it did: ``tests/test_render_seam.py``, in
the first full run of four workers). To be removed with the defect.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import reverberate.accel.encode as accel_encode
import reverberate.spatial.encode as spatial_encode
from reverberate.render import native


@pytest.fixture(autouse=True, scope="session")
def _the_engine_s_first_use_before_any_thread() -> None:
    native.available()
    native._table()


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
