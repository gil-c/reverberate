"""The calibration run on a machine with the card and the reference: the wiring.

:mod:`reverberate.mirror.calibrate` knows the parameters, the cost and the
search and nothing about cards or files. This module reads what the card
phase wrote (the derived scene, the paths of every point), chooses the
points the cost is read on, reads their references, and hands the search a
tracer on the card(s) and a renderer; then it writes the calibration under
``mirror/calibration/<key>.json`` and points ``mirror/calibration/latest.json``
at it.

The references may come from the field itself (``field/<S>.h5``) or from a
subset file (``mirror/reference_subset_<S>.h5``) written by
:func:`write_reference_subset` where the field is, so that a few dozen
points at a low order travel to the card instead of the whole field.
The criteria are read relative to each response's own direct sound, so no
alignment is needed before judging.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.mirror.calibrate import (
    CostWeights,
    Evaluation,
    Parameters,
    calibrate,
    write_calibration,
)
from reverberate.mirror.engine import histogram_on_devices
from reverberate.mirror.geometry import DerivedScene, load_derived
from reverberate.mirror.ism import Paths, occluder_grid
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.mirror.stage import MirrorSettings, _lattice_of, _render_point, load_every
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count

__all__ = ["calibrate_run", "choose_points", "read_references", "write_reference_subset"]


def choose_points(every: list[Paths], count: int) -> list[int]:
    """``count`` points with a direct path, spread evenly over the storey's order."""
    with_direct = [i for i, p in enumerate(every) if bool(np.any(p.order == 0))]
    if len(with_direct) <= count:
        return with_direct
    step = len(with_direct) / count
    return [with_direct[int(k * step)] for k in range(count)]


def write_reference_subset(reference: Path, target: Path, indices: list[int], order: int) -> Path:
    """``indices`` of the field at ``order``, for the calibration to read elsewhere."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    channels = channel_count(order)
    with h5py.File(reference, "r") as source, h5py.File(target, "w") as out:
        rate = float(source.attrs["sample_rate_hz"])
        if int(source.attrs["order"]) < order:
            raise ValueError(f"the field is order {source.attrs['order']}, asked for {order}")
        rows = [np.asarray(source["ir"][i][:channels], dtype=np.float32) for i in indices]
        out.create_dataset("ir", data=np.stack(rows) if rows else np.zeros((0, channels, 1)))
        out.create_dataset("point_index", data=np.asarray(indices, dtype=np.int64))
        out.attrs["sample_rate_hz"] = rate
        out.attrs["order"] = order
        out.attrs["reference"] = str(reference)
    return target


def read_references(run: Path, name: str, indices: list[int], order: int) -> dict[int, Ambisonic]:
    """The references of ``indices`` at ``order``: from the field, else the subset file."""
    run = Path(run)
    channels = channel_count(order)
    field = run / "field" / f"{name}.h5"
    if field.is_file():
        with h5py.File(field, "r") as handle:
            rate = float(handle.attrs["sample_rate_hz"])
            return {
                i: Ambisonic(
                    np.asarray(handle["ir"][i][:channels], dtype=float), rate, order, np.zeros(3)
                )
                for i in indices
            }
    subset = run / "mirror" / f"reference_subset_{name}.h5"
    if not subset.is_file():
        raise FileNotFoundError(f"neither {field} nor {subset}: no reference to calibrate on")
    with h5py.File(subset, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        held = {int(v): k for k, v in enumerate(handle["point_index"][...])}
        missing = [i for i in indices if i not in held]
        if missing:
            raise KeyError(f"points {missing[:8]} are not in {subset}")
        if int(handle["ir"].shape[1]) < channels:
            raise ValueError(f"{subset} holds {handle['ir'].shape[1]} channels, need {channels}")
        return {
            i: Ambisonic(
                np.asarray(handle["ir"][held[i]][:channels], dtype=float), rate, order, np.zeros(3)
            )
            for i in indices
        }


def calibrate_run(
    run: Path,
    *,
    source: dict[str, Any],
    settings: MirrorSettings | None = None,
    points: int = 24,
    iterations: int = 40,
    tied: bool = False,
    order: int = 3,
    weights: CostWeights | None = None,
    sound_speed_m_s: float = 343.2,
    devices: list[int] | None = None,
    say: Any = print,
) -> tuple[Parameters, list[Evaluation], Path]:
    """Calibrate on ``points`` of the run's card phase; returns the best, the trail, the file."""
    settings = settings or MirrorSettings()
    run = Path(run)
    name = str(source["name"])
    mirror_dir = run / "mirror"
    catalogue = load_derived(mirror_dir / "scene")
    every = load_every(mirror_dir / f"paths_{name}.npz")
    chosen = choose_points(every, points)
    if not chosen:
        raise ValueError("no point with a direct path to calibrate on")
    positions, rate, _ = _lattice_of(run, name)
    references = read_references(run, name, chosen, order)
    position = np.asarray(source["position"], dtype=float)
    grid = occluder_grid(catalogue, settings.rays.cell_m)
    rays = RaySettings(**{**settings.rays.record(), "sound_speed_m_s": sound_speed_m_s})
    render_settings = replace(settings.render, order=order, sample_rate_hz=rate)
    local = {index: k for k, index in enumerate(chosen)}
    say(f"calibrate {name}: {len(chosen)} points, {rays.rays} rays per candidate, order {order}")

    def tracer(scene: DerivedScene) -> Histogram:
        return histogram_on_devices(
            scene, position, positions[chosen], rays, devices=devices, grid=grid
        )

    def render_point(
        index: int,
        paths: Paths,
        histogram: Histogram,
        scene: DerivedScene,
        parameters: Parameters,
    ) -> Ambisonic:
        candidate = replace(settings, render=render_settings, parameters=parameters)
        _, response, _ = _render_point(
            local[index], paths, histogram, scene, candidate, sound_speed_m_s
        )
        return response

    best, evaluations = calibrate(
        catalogue,
        {i: every[i] for i in chosen},
        references,
        render_point,
        tracer,
        start=settings.parameters,
        weights=weights,
        criteria=settings.criteria,
        iterations=iterations,
        tied=tied,
        say=say,
    )
    best = replace(
        best,
        note=f"{best.note}; {len(chosen)} points of {name}, order {order}, "
        f"{rays.rays} rays, {'tied' if tied else 'full'}",
    )
    target = write_calibration(mirror_dir / "calibration", best, evaluations)
    (mirror_dir / "calibration" / "latest.json").write_text(
        json.dumps({"key": best.key, "file": target.name, "points": chosen}, indent=1)
    )
    return best, evaluations, target
