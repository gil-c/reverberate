"""The calibration of the mirror on a wave field: a band by band fixed point.

Each band's reverberation time goes as the inverse of its absorption, so a
band's absorption scale is multiplied by the median ratio of the mirror's
T30 to the wave field's, damped; the tail gain moves the tail's energy and
nothing else, so a band's gain takes the median colour gap, damped; the
image sources' own absorption follows the matched reflections' level gap. A
handful of evaluations, each tracing the rays again on the devices and
judging a few dozen points. Only the bands above the hybrid's crossover
count: the wave field supplies the rest.

The result is a parameters record (:mod:`reverberate.mirror.parameters`)
that also holds the ray and tail settings it was fitted under, written to
``mirror/calibration/<key>.json`` with its trajectory.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.compute import Devices
from reverberate.mirror.calibration.criteria import Criteria, CriteriaSettings, PointReport
from reverberate.mirror.calibration.judge import judge
from reverberate.mirror.engine import histogram_on_devices
from reverberate.mirror.files import lattice_of, load_paths
from reverberate.mirror.geometry import DerivedScene, load_derived
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.ism import Paths, occluder_grid
from reverberate.mirror.parameters import Parameters, apply_parameters, image_scene, regain
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import Histogram
from reverberate.mirror.render import render_point
from reverberate.spatial.encode import Ambisonic
from reverberate.spatial.sh import channel_count

__all__ = [
    "CostWeights",
    "Evaluation",
    "band_cost",
    "calibrate",
    "choose_points",
    "fixed_point",
    "read_references",
    "write_reference_subset",
]


@dataclass(frozen=True)
class CostWeights:
    """How the criteria are summed into one number; both parts stay visible."""

    echogram_db: float = 1.0
    t30_relative: float = 20.0
    tail_colour_db: float = 1.0
    sector_energy_db: float = 0.5
    recall: float = 10.0


@dataclass
class Evaluation:
    """One candidate: its parameters, its cost, and the judgements behind the cost."""

    parameters: Parameters
    cost: float
    early: float
    late: float
    reports: list[PointReport] = field(default_factory=list)
    seconds: float = 0.0

    def record(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters.record(),
            "key": self.parameters.key,
            "cost": round(self.cost, 5),
            "early": round(self.early, 5),
            "late": round(self.late, 5),
            "points": len(self.reports),
            "seconds": round(self.seconds, 1),
        }


def _judge_one(args: tuple[Ambisonic, Ambisonic, CriteriaSettings]) -> PointReport | None:
    reference, response, settings = args
    try:
        return judge(reference, response, Criteria(settings=settings))
    except ValueError:
        return None


def _weighted(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * values) / np.sum(weight))


def band_cost(
    reports: list[PointReport], weights: CostWeights, low_hz: float
) -> tuple[float, float, float]:
    """The cost and its early and late parts: decay and colour band by band from ``low_hz``."""
    if not reports:
        return float("inf"), float("inf"), float("inf")
    early_terms = []
    late_terms = []
    for r in reports:
        recall_gap = max(0.0, 1.0 - r.errors["recall"])
        early_terms.append(
            weights.echogram_db * float(np.mean(r.echogram_distance_db))
            + weights.recall * recall_gap
        )
        ref, cand = r.reference_tail, r.candidate_tail
        band_weight = np.asarray(
            [0.0 if b < 0.99 * low_hz else (0.5 if b < 500 else 1.0) for b in ref.bands_hz]
        )
        t30 = np.abs(np.asarray(cand.t30_s) / np.asarray(ref.t30_s) - 1.0)
        edt = np.abs(np.asarray(cand.edt_s) / np.asarray(ref.edt_s) - 1.0)
        colour = np.abs(np.asarray(ref.colour_db) - np.asarray(cand.colour_db))
        t30 = np.where(np.isfinite(t30), t30, 1.0)
        edt = np.where(np.isfinite(edt), edt, 1.0)
        colour = np.where(np.isfinite(colour), colour, 30.0)
        sector = r.errors["sector_energy_db"] if np.isfinite(r.errors["sector_energy_db"]) else 10.0
        late_terms.append(
            weights.t30_relative * _weighted(t30, band_weight)
            + 0.5 * weights.t30_relative * _weighted(edt, band_weight)
            + 2.0 * weights.tail_colour_db * _weighted(colour, band_weight)
            + weights.sector_energy_db * sector
        )
    early = float(np.mean(early_terms))
    late = float(np.mean(late_terms))
    return early + late, early, late


def _band_residuals(reports: list[PointReport]) -> tuple[dict[int, float], dict[int, float]]:
    """Per band: the median T30 ratio (mirror over wave) and colour gap (wave minus mirror, dB)."""
    ratios: dict[int, list[float]] = {}
    gaps: dict[int, list[float]] = {}
    for r in reports:
        ref, cand = r.reference_tail, r.candidate_tail
        for k, band in enumerate(ref.bands_hz):
            a, b = float(ref.t30_s[k]), float(cand.t30_s[k])
            if np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0:
                ratios.setdefault(int(band), []).append(b / a)
            c, d = float(ref.colour_db[k]), float(cand.colour_db[k])
            if np.isfinite(c) and np.isfinite(d):
                gaps.setdefault(int(band), []).append(c - d)
    return (
        {band: float(np.median(v)) for band, v in ratios.items()},
        {band: float(np.median(v)) for band, v in gaps.items()},
    )


def _reflection_gaps(reports: list[PointReport]) -> dict[int, float]:
    """Per band: the median level gap of the matched reflections, wave minus mirror, dB.

    Both sides are relative to their own direct sound, so the signature and
    the air cancel and what remains is the reflections' loss.
    """
    gaps: dict[int, list[float]] = {}
    for r in reports:
        for i, j in r.matching.pairs:
            ref, cand = r.reference[i], r.candidate[j]
            if ref.index == 0 or cand.index == 0:
                continue
            for k, band in enumerate(ref.bands.centres_hz):
                a, b = float(ref.bands.levels_db[k]), float(cand.bands.levels_db[k])
                if np.isfinite(a) and np.isfinite(b):
                    gaps.setdefault(int(band), []).append(a - b)
    return {band: float(np.median(v)) for band, v in gaps.items()}


def _nearest(band: int, table: dict[int, float]) -> float:
    """The value of the band in ``table`` nearest ``band`` on a log scale."""
    return table[min(table, key=lambda b: abs(np.log2(b / band)))]


def _mean_absorption(scene: DerivedScene) -> np.ndarray:
    """The area weighted absorption per band of the reflecting facets."""
    areas = np.zeros(len(scene.materials.labels))
    for f in scene.facets:
        areas[f.label] += f.area
    return np.asarray(
        (areas[:, None] * scene.materials.absorption).sum(axis=0) / max(float(areas.sum()), 1e-9)
    )


def fixed_point(
    catalogue: DerivedScene,
    paths: dict[int, Paths],
    references: dict[int, Ambisonic],
    render: Callable[[int, Paths, Histogram, Parameters], Ambisonic],
    trace: Callable[[DerivedScene], Histogram],
    *,
    start: Parameters,
    criteria: CriteriaSettings,
    iterations: int = 8,
    damping: float = 0.8,
    weights: CostWeights | None = None,
    workers: int = 1,
    say: Any = print,
) -> tuple[Parameters, list[Evaluation]]:
    """``iterations`` fixed point steps from ``start``; the best evaluation by :func:`band_cost`."""
    weights = weights or CostWeights()
    parameters = start
    if parameters.image_absorption_scale is None:
        parameters = replace(parameters, image_absorption_scale=parameters.absorption_scale)
    evaluations: list[Evaluation] = []
    bands = list(OCTAVE_BANDS)
    alpha = _mean_absorption(catalogue)

    def evaluate(candidate: Parameters, pool: ProcessPoolExecutor | None) -> list[PointReport]:
        started = time.time()
        histogram = trace(apply_parameters(catalogue, candidate))
        images = image_scene(catalogue, candidate)
        jobs = [
            (references[i], render(i, regain(p, images), histogram, candidate), criteria)
            for i, p in paths.items()
        ]
        judged = list(pool.map(_judge_one, jobs)) if pool else [_judge_one(j) for j in jobs]
        reports = [r for r in judged if r is not None]
        total, early, late = band_cost(reports, weights, criteria.low_hz)
        evaluations.append(
            Evaluation(candidate, total, early, late, reports, time.time() - started)
        )
        say(
            f"fixed point {len(evaluations)}/{iterations}: cost {total:.3f}"
            f" (early {early:.3f}, late {late:.3f}) in {time.time() - started:.0f} s"
        )
        return reports

    def step(candidate: Parameters, reports: list[PointReport]) -> Parameters:
        ratios, gaps = _band_residuals(reports)
        levels = _reflection_gaps(reports)
        scale = list(candidate.absorption_scale)
        gain = list(candidate.tail_gain_db)
        image = list(candidate.image_absorption_scale or candidate.absorption_scale)
        for k, band in enumerate(bands):
            # A band the judgement does not read (125 Hz) follows the nearest one it does.
            if ratios:
                scale[k] = float(np.clip(scale[k] * _nearest(band, ratios) ** damping, 0.05, 20.0))
            if gaps:
                gain[k] = float(np.clip(gain[k] + damping * _nearest(band, gaps), -30.0, 30.0))
            if levels:
                # A matched reflection has about two bounces: its energy goes as
                # (1 - s a)^2, so a gap of g dB asks (1 - s a) to move by g / 20 dB.
                gap = _nearest(band, levels)
                kept = (1.0 - image[k] * alpha[k]) * 10.0 ** (damping * gap / 20.0)
                image[k] = float(np.clip((1.0 - kept) / max(alpha[k], 1e-6), 0.05, 20.0))
        return replace(
            candidate,
            absorption_scale=tuple(scale),
            tail_gain_db=tuple(gain),
            image_absorption_scale=tuple(image),
            note="candidate",
        )

    # Spawned, not forked: the tracer may have initialised CUDA in this process.
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        for _ in range(iterations):
            parameters = step(parameters, evaluate(parameters, pool))
    finally:
        if pool is not None:
            pool.shutdown()
    best = min(evaluations, key=lambda e: e.cost)
    return (
        replace(best.parameters, note=f"fixed point, {len(evaluations)} evaluations"),
        evaluations,
    )


def choose_points(every: list[Paths], count: int) -> list[int]:
    """``count`` points with a direct path, spread evenly over the lattice's order."""
    with_direct = [i for i, p in enumerate(every) if bool(np.any(p.order == 0))]
    if len(with_direct) <= count:
        return with_direct
    step = len(with_direct) / count
    return [with_direct[int(k * step)] for k in range(count)]


def write_reference_subset(reference: Path, target: Path, indices: list[int], order: int) -> Path:
    """The wave field at ``indices`` and ``order``: what a calibration elsewhere needs of it."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    channels = channel_count(order)
    with h5py.File(reference, "r") as source, h5py.File(target, "w") as out:
        if int(source.attrs["order"]) < order:
            raise ValueError(f"the field is order {source.attrs['order']}, asked for {order}")
        rows = [np.asarray(source["ir"][i][:channels], dtype=np.float32) for i in indices]
        out.create_dataset("ir", data=np.stack(rows) if rows else np.zeros((0, channels, 1)))
        out.create_dataset("point_index", data=np.asarray(indices, dtype=np.int64))
        out.attrs["sample_rate_hz"] = float(source.attrs["sample_rate_hz"])
        out.attrs["order"] = order
    return target


def read_references(run: Path, source: str, indices: list[int], order: int) -> dict[int, Ambisonic]:
    """The wave field at ``indices`` and ``order``: from ``field/<source>.h5``, else the subset."""
    channels = channel_count(order)
    field_path = Path(run) / "field" / f"{source}.h5"
    subset = Path(run) / "mirror" / f"reference_subset_{source}.h5"
    path = field_path if field_path.is_file() else subset
    if not path.is_file():
        raise FileNotFoundError(f"neither {field_path} nor {subset}")
    with h5py.File(path, "r") as handle:
        rate = float(handle.attrs["sample_rate_hz"])
        rows = (
            {i: i for i in indices}
            if path == field_path
            else {int(v): k for k, v in enumerate(handle["point_index"][...])}
        )
        missing = [i for i in indices if i not in rows]
        if missing:
            raise KeyError(f"points {missing[:8]} are not in {path}")
        return {
            i: Ambisonic(
                np.asarray(handle["ir"][rows[i]][:channels], dtype=float), rate, order, np.zeros(3)
            )
            for i in indices
        }


def calibrate(
    run: Path,
    source: str,
    position: np.ndarray,
    *,
    settings: MirrorSettings | None = None,
    points: int = 24,
    iterations: int = 8,
    order: int = 3,
    devices: Devices | None = None,
    workers: int = 1,
    say: Any = print,
) -> tuple[Parameters, Path]:
    """Calibrate on ``points`` of a run the mirror has traced; returns the parameters and the file.

    Reads ``mirror/scene``, ``mirror/paths_<source>.npz`` and, when there,
    ``mirror/signature_<source>.npy`` (so the colour is judged as rendered);
    the wave field from ``field/<source>.h5`` or the reference subset.
    """
    settings = settings or MirrorSettings()
    devices = devices or Devices.detect()
    mirror = Path(run) / "mirror"
    catalogue = load_derived(mirror / "scene")
    every = load_paths(mirror / f"paths_{source}.npz")
    chosen = choose_points(every, points)
    if not chosen:
        raise ValueError("no point with a direct path to calibrate on")
    positions, rate, _ = lattice_of(Path(run), source)
    references = read_references(Path(run), source, chosen, order)
    position = np.asarray(position, dtype=float).reshape(3)
    rays = settings.traced_rays()
    render_settings = replace(settings.render, order=order, sample_rate_hz=rate)
    grid = occluder_grid(catalogue, rays.cell_m)
    local = {index: k for k, index in enumerate(chosen)}
    signature_path = mirror / f"signature_{source}.npy"
    signature = np.load(signature_path) if signature_path.is_file() else None
    criteria = CriteriaSettings(low_hz=Crossover().cutoff_hz)
    say(f"calibrate {source}: {len(chosen)} points, {rays.rays} rays, order {order}")

    def trace(scene: DerivedScene) -> Histogram:
        return histogram_on_devices(
            scene, position, positions[chosen], rays, devices=devices, grid=grid
        )

    def render(index: int, paths: Paths, histogram: Histogram, parameters: Parameters) -> Ambisonic:
        response, _ = render_point(
            local[index],
            paths,
            histogram,
            render_settings,
            tail_gain_db=np.asarray(parameters.tail_gain_db, dtype=float),
            receiver_radius_m=rays.receiver_radius_m,
            sound_speed_m_s=settings.sound_speed_m_s,
            seed=settings.seed + index,
            signature=signature,
        )
        return response

    best, evaluations = fixed_point(
        catalogue,
        {i: every[i] for i in chosen},
        references,
        render,
        trace,
        start=settings.parameters,
        criteria=criteria,
        iterations=iterations,
        workers=workers,
        say=say,
    )
    best = replace(
        best,
        note=f"{best.note}; {len(chosen)} points of {source}",
        rendered_with={"rays": rays.record(), "render": settings.render.record()},
    )
    target = mirror / "calibration" / f"{best.key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "parameters": best.record(),
                "key": best.key,
                "points": chosen,
                "trajectory": [e.record() for e in evaluations],
            },
            indent=1,
        )
    )
    return best, target
