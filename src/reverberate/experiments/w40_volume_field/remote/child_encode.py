"""Encode every listening point of one solve, where the pressure is.

Run on the rented machine by :mod:`reverberate.experiments.w40_volume_field.encode`,
under PFFDTD's own interpreter, with the package on ``PYTHONPATH``. Reads a
job as JSON on stdin, ``sim_outs.h5`` and ``comms_out.h5`` in the working
directory, and writes ``encoded.h5``:

    signals      [point, channel, sample]  float32, at the delivery rate
    point_index  [point]                   index into the plan's points
    centres      [point, 3]                the node each field is expanded about

One point per worker, every core busy. The pressure of a point is one
contiguous slice of rows, so nothing but that slice is ever in memory.
"""

from __future__ import annotations

# PFFDTD's interpreter is Python 3.10 and ``reverberate.response`` imports
# ``datetime.UTC``, which arrived in 3.11. The shim costs nothing elsewhere.
import datetime as _datetime
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

from reverberate import audio
from reverberate.spatial.array import ArrayDesign
from reverberate.spatial.encode import EncoderSettings, encode

JOB: dict[str, Any] = {}
OUT_ALPHA: np.ndarray = np.zeros((0, 1))
POSITIONS: np.ndarray = np.zeros((0, 3))


def _init(job: dict[str, Any]) -> None:
    """Runs in every worker: the pool is spawned, so globals do not carry over."""
    global JOB, OUT_ALPHA, POSITIONS
    JOB = job
    remote_dir = Path(job["remote_dir"])
    JOB["pressure"] = str(job.get("pressure") or remote_dir / "sim_outs.h5")
    with h5py.File(job.get("comms") or remote_dir / "comms_out.h5", "r") as handle:
        OUT_ALPHA = np.asarray(handle["out_alpha"][...], dtype=np.float64)
        JOB["differentiated"] = bool(np.asarray(handle["diff"]).item())
    POSITIONS = np.load(job.get("positions") or remote_dir / "array_positions.npy")


def _one(index: int) -> tuple[int, np.ndarray, np.ndarray]:
    start, stop = JOB["rows"][index]
    # A shard file starts at ``row_offset`` of the whole solve: the pressure
    # is read at the shard's own rows, the comms and positions at the plan's.
    offset = int(JOB.get("row_offset", 0))
    with h5py.File(JOB["pressure"], "r") as handle:
        u_out = np.asarray(handle["u_out"][start:stop], dtype=np.float64)
    rate = float(JOB["sample_rate_hz"])
    start, stop = start + offset, stop + offset
    signals = audio.reduce_nodes(u_out, OUT_ALPHA[start:stop])
    signals = audio.integrate_and_lowcut(
        signals,
        1.0 / rate,
        differentiated=JOB["differentiated"],
        fcut=float(JOB["lowcut_hz"]),
        order=int(JOB["lowcut_order"]),
    )
    signals = audio.lowpass(signals, rate, float(JOB["fmax_hz"]))
    signals = audio.resample_to(signals, rate, float(JOB["delivery_rate_hz"]))
    signals = audio.apply_air_absorption(
        signals,
        float(JOB["delivery_rate_hz"]),
        sound_speed_m_s=float(JOB["sound_speed_m_s"]),
        atmosphere=audio.Atmosphere(),
    )
    centre = np.asarray(JOB["centres"][index], dtype=float)
    positions = POSITIONS[start:stop]
    offsets = positions - centre
    design = ArrayDesign(
        centre=centre,
        positions=positions,
        offsets=offsets,
        radii=np.linalg.norm(offsets, axis=1),
        shell=np.zeros(positions.shape[0], dtype=int),
        nominal_radii=(float(np.linalg.norm(offsets, axis=1).max()),),
        grid_step_m=float(JOB["grid_step_m"]),
        requested_centre=centre,
    )
    settings = EncoderSettings(
        order=int(JOB["order"]),
        fit_order=int(JOB["fit_order"]),
        max_frequency_hz=float(JOB["fmax_hz"]),
    )
    ambisonic = encode(
        signals,
        float(JOB["delivery_rate_hz"]),
        design,
        sound_speed_m_s=float(JOB["sound_speed_m_s"]),
        settings=settings,
    )
    return index, ambisonic.signals.astype(np.float32), centre


def main() -> int:
    job = json.loads(sys.stdin.read())
    indices = [i for i, rows in enumerate(job["rows"]) if rows is not None]
    workers = int(job.get("workers") or max(1, (os.cpu_count() or 2) - 1))
    started = time.time()
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(job,)) as pool:
        for index, signals, centre in pool.map(_one, indices, chunksize=1):
            results[index] = (signals, centre)
            done = len(results)
            if done % 20 == 0 or done == len(indices):
                elapsed = time.time() - started
                print(
                    f"encoded {done}/{len(indices)} in {elapsed:.0f} s"
                    f" ({elapsed / done:.1f} s a point on {workers} workers)",
                    flush=True,
                )
    order_indices = sorted(results)
    samples = results[order_indices[0]][0].shape[1]
    channels = results[order_indices[0]][0].shape[0]
    with h5py.File(job["output"], "w") as handle:
        data = handle.create_dataset(
            "signals", shape=(len(order_indices), channels, samples), dtype=np.float32
        )
        for row, index in enumerate(order_indices):
            data[row] = results[index][0]
        handle.create_dataset("point_index", data=np.asarray(order_indices))
        handle.create_dataset("centres", data=np.asarray([results[i][1] for i in order_indices]))
        handle.attrs["sample_rate_hz"] = float(job["delivery_rate_hz"])
        handle.attrs["order"] = int(job["order"])
        handle.attrs["band"] = job["band"]
        handle.attrs["source"] = job["source"]
    print(
        f"wrote {job['output']}: {len(order_indices)} points, {channels} channels,"
        f" {samples} samples",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
