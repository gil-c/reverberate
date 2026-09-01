"""The four things every experiment run does around the solver binary.

This module exists only because two of the four promoted scripts did each of
these things separately, in code that had drifted apart by a word or two.
Nothing here is a generalisation of a single caller: each function below is
exercised by at least two of ``b0_export.py``, ``b0_run.py``, ``b0_compare.py``
and ``w2_run.py``, and that is the whole admission criterion.

- :func:`engine_binary` -- ``b0_run.py`` and ``w2_run.py`` both built the same
  ``fdtd_main_{engine}_{precision}.x`` path by hand.
- :func:`run_binary` -- both then ran it in a directory, timed it, wrote
  ``engine.log`` and raised on a non-zero exit.
- :func:`sim_consts` -- ``b0_run.py`` read ``h`` and ``SR`` from it,
  ``b0_compare.py`` read ``SR``, and ``w2_run.py`` keeps the file after a run
  for no other reason than that a later comparison needs it.
- :func:`write_record` -- ``b0_run.py`` wrote ``b0_result.json``,
  ``w2_run.py`` wrote ``w2_run.json``, both printed the same JSON afterwards.

PFFDTD itself is never imported here. It needs numpy below 2 and this project
does not, so the only thing that crosses over is a path and a subprocess, the
same rule :mod:`reverberate.wave.voxelise` already follows.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from reverberate.wave.voxelise import pffdtd_dir

__all__ = [
    "Engine",
    "EngineRun",
    "SimConsts",
    "engine_binary",
    "run_binary",
    "sim_consts",
    "write_record",
]

#: Which binary to run. The Python reference engine of ``b0_run.py`` is gone:
#: it was never used for a measurement, and ``w2_run.py`` says why -- ``--abc``
#: defaults to false there, so it does not apply the boundary condition the C
#: engines apply unconditionally, and it answers a different question.
Engine = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class EngineRun:
    """One solver invocation: how long it took and where it said so."""

    engine_s: float
    log: Path


@dataclass(frozen=True)
class SimConsts:
    """The two constants of a run that every later stage quotes."""

    h: float
    sample_rate: float


def engine_binary(engine: Engine, *, double_precision: bool) -> Path:
    """PFFDTD's solver for ``engine``, from ``PFFDTD_DIR``. Never guessed.

    Missing is an error with the build command in it, because the one place
    this is discovered is halfway through an experiment.
    """
    precision = "double" if double_precision else "single"
    binary = pffdtd_dir() / "c_cuda" / f"fdtd_main_{engine}_{precision}.x"
    if not binary.is_file():
        raise FileNotFoundError(
            f"{binary} is not built. Run scripts/build_pffdtd.sh, or on macOS build "
            f"the CPU binary with:\n"
            "  clang -I. -I$(brew --prefix hdf5)/include -Xpreprocessor -fopenmp "
            "-I$(brew --prefix libomp)/include -O3 -std=c99 fdtd_main.c "
            f"-o fdtd_main_{engine}_{precision}.x -DPRECISION={2 if double_precision else 1} "
            "-DUSING_CUDA=false -L$(brew --prefix hdf5)/lib -lhdf5 "
            "-L$(brew --prefix libomp)/lib -lomp -lm"
        )
    return binary


def run_binary(run_dir: Path, engine: Engine, *, double_precision: bool) -> EngineRun:
    """Run the solver in ``run_dir``, which must already hold its four inputs.

    The engine opens its inputs by name from its working directory and writes
    ``sim_outs.h5`` beside them, so the directory is the interface. Everything
    it printed goes to ``engine.log`` whether it succeeded or not; a failure
    that leaves no log is a failure nobody can diagnose afterwards.
    """
    binary = engine_binary(engine, double_precision=double_precision)
    started = time.time()
    completed = subprocess.run(
        [str(binary)], cwd=run_dir, capture_output=True, text=True, check=False
    )
    engine_s = time.time() - started
    log = run_dir / "engine.log"
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"engine failed in {run_dir}, see {log}")
    return EngineRun(engine_s=round(engine_s, 2), log=log)


def sim_consts(data_dir: Path) -> SimConsts:
    """``h`` and the sample rate of a run or of a cache entry."""
    with h5py.File(Path(data_dir) / "sim_consts.h5", "r") as handle:
        return SimConsts(
            h=float(np.asarray(handle["h"])),
            sample_rate=float(np.asarray(handle["SR"])),
        )


def write_record(run_dir: Path, name: str, record: dict[str, object]) -> Path:
    """Write ``record`` beside the run and print it. Returns the file."""
    path = Path(run_dir) / name
    text = json.dumps(record, indent=2)
    path.write_text(text)
    print(text)
    return path
