"""The calibration of the mirror on the reference: parameters, cost, search, record.

The mirror is a physical model with a few hundred numbers in it; the
reference is the wave solver's field. The calibration adjusts what the
model does not know from the catalogue and reads the criteria as its cost:

- an **absorption scale per octave band**, one factor on every label's
  absorption, which is the realised absorption of W3 (0.89 at 250 Hz)
  written as a parameter rather than assumed;
- a **scattering scale**, one factor on every class's scattering
  coefficient, which decides the split between the discrete reflections and
  the diffuse tail;
- a **tail gain per octave band**, on the histogram's tail, which absorbs
  what the ray count and the sphere detector leave to calibrate.

The images and their validity do not depend on materials, so the paths of
every point are computed once and only their gains change; the rays do
depend on absorption and scattering and are traced again for each
candidate, which is seconds on the card. The cost is read on a subset of
points, and it never trades the early part against the tail: part A's
echogram distance and part B's decay and colour errors are summed with
stated weights, and both are reported at every step.

The parameters are one JSON keyed by their own digest, so a field carries
the calibration it was rendered with.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.criteria import Criteria, CriteriaSettings, PointReport, judge
from reverberate.mirror.geometry import DerivedScene, MaterialTable
from reverberate.mirror.ism import Paths, _gains
from reverberate.spatial.encode import Ambisonic

__all__ = [
    "Parameters",
    "apply_parameters",
    "band_cost",
    "calibrate",
    "calibrate_fixed_point",
    "cost_of",
    "image_scene",
    "mean_absorption",
    "regain",
]


@dataclass(frozen=True)
class Parameters:
    """What the calibration may move, with the catalogue's values as the origin."""

    absorption_scale: tuple[float, ...] = tuple(1.0 for _ in OCTAVE_BANDS)
    scattering_scale: float = 1.0
    tail_gain_db: tuple[float, ...] = tuple(0.0 for _ in OCTAVE_BANDS)
    note: str = "catalogue values, nothing calibrated"
    #: The absorption scale the image sources' gains use, per band; ``None``
    #: uses ``absorption_scale``. The rays set the decay and the images the
    #: reflections' levels; one scale for both traded one against the other.
    image_absorption_scale: tuple[float, ...] | None = None
    #: The shell's own scattering coefficient, in place of its class's value
    #: times ``scattering_scale``; ``None`` keeps the class. A specular bounce
    #: on the floor, the ceiling or a wall keeps a ray's elevation, so with
    #: little scattering on the shell the rays' late field lies flat.
    shell_scattering: float | None = None

    @property
    def key(self) -> str:
        digest = hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "bands_hz": list(OCTAVE_BANDS),
            "absorption_scale": [round(float(v), 6) for v in self.absorption_scale],
            "scattering_scale": round(float(self.scattering_scale), 6),
            "tail_gain_db": [round(float(v), 4) for v in self.tail_gain_db],
            "note": self.note,
        }
        if self.shell_scattering is not None:
            record["shell_scattering"] = round(float(self.shell_scattering), 6)
        if self.image_absorption_scale is not None:
            # Only when set, so the keys of the files written before stay theirs.
            record["image_absorption_scale"] = [
                round(float(v), 6) for v in self.image_absorption_scale
            ]
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Parameters:
        return cls(
            absorption_scale=tuple(float(v) for v in record["absorption_scale"]),
            scattering_scale=float(record["scattering_scale"]),
            tail_gain_db=tuple(float(v) for v in record["tail_gain_db"]),
            note=str(record.get("note", "")),
            image_absorption_scale=(
                tuple(float(v) for v in record["image_absorption_scale"])
                if record.get("image_absorption_scale") is not None
                else None
            ),
            shell_scattering=(
                float(record["shell_scattering"])
                if record.get("shell_scattering") is not None
                else None
            ),
        )

    def to_vector(self, *, tied: bool = False) -> np.ndarray:
        """The search's coordinates: logs of the scales, the gains as they are.

        ``tied`` collapses the bands to one absorption scale and one tail gain
        (their means), for a coarse pass of three coordinates before the
        fifteen of the full search.
        """
        if tied:
            return np.asarray(
                [
                    float(np.mean(np.log(np.asarray(self.absorption_scale)))),
                    np.log(self.scattering_scale),
                    float(np.mean(self.tail_gain_db)),
                ]
            )
        return np.concatenate(
            [
                np.log(np.asarray(self.absorption_scale)),
                [np.log(self.scattering_scale)],
                np.asarray(self.tail_gain_db),
            ]
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray, note: str = "", *, tied: bool = False) -> Parameters:
        n = len(OCTAVE_BANDS)
        if tied:
            return cls(
                absorption_scale=tuple(float(np.exp(vector[0])) for _ in range(n)),
                scattering_scale=float(np.exp(vector[1])),
                tail_gain_db=tuple(float(vector[2]) for _ in range(n)),
                note=note,
            )
        return cls(
            absorption_scale=tuple(float(v) for v in np.exp(vector[:n])),
            scattering_scale=float(np.exp(vector[n])),
            tail_gain_db=tuple(float(v) for v in vector[n + 1 : 2 * n + 1]),
            note=note,
        )


def apply_parameters(scene: DerivedScene, parameters: Parameters) -> DerivedScene:
    """The scene with its materials scaled: absorption per band, scattering per label."""
    absorption = np.clip(
        scene.materials.absorption * np.asarray(parameters.absorption_scale)[None, :], 0.0, 0.999
    )
    scattering = np.clip(scene.materials.scattering * parameters.scattering_scale, 0.0, 1.0)
    if parameters.shell_scattering is not None and "shell" in scene.materials.labels:
        scattering[scene.materials.labels.index("shell")] = float(
            np.clip(parameters.shell_scattering, 0.0, 1.0)
        )
    materials = MaterialTable(
        tuple(scene.materials.labels),
        absorption,
        scattering,
        scene.materials.bands_hz,
        scene.materials.source,
    )
    return replace(scene, materials=materials)


def image_scene(scene: DerivedScene, parameters: Parameters) -> DerivedScene:
    """The catalogue scene with the materials the image sources' gains read.

    The shell's own scattering is the rays' mixing of directions, not a loss
    of the flat shell's specular reflection (the solver's walls are flat):
    the images keep the class's scattering there.
    """
    images = replace(parameters, shell_scattering=None)
    if parameters.image_absorption_scale is None:
        return apply_parameters(scene, images)
    return apply_parameters(
        scene, replace(images, absorption_scale=parameters.image_absorption_scale)
    )


def mean_absorption(scene: DerivedScene) -> np.ndarray:
    """The area weighted absorption per band of the catalogue scene's reflecting facets."""
    areas = np.zeros(len(scene.materials.labels))
    for f in scene.facets:
        areas[f.label] += f.area
    total = max(float(areas.sum()), 1e-9)
    return np.asarray((areas[:, None] * scene.materials.absorption).sum(axis=0) / total)


def regain(paths: Paths, scene: DerivedScene) -> Paths:
    """The same paths with the gains of ``scene``'s materials: nothing else moves."""
    return replace(paths, gain=_gains(scene, paths.sequence, paths.length_m))


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
    """One candidate: its parameters, its cost, and the criteria behind the cost."""

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
            "recall_median": _median([r.errors["recall"] for r in self.reports]),
            "t30_relative_median": _median([r.errors["t30_relative"] for r in self.reports]),
            "tail_colour_db_median": _median([r.errors["tail_colour_db"] for r in self.reports]),
            "echogram_db_median": _median(
                [float(np.mean(r.echogram_distance_db)) for r in self.reports]
            ),
            "seconds": round(self.seconds, 1),
        }


def _median(values: list[float]) -> float | None:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return None if array.size == 0 else round(float(np.median(array)), 5)


def cost_of(reports: list[PointReport], weights: CostWeights) -> tuple[float, float, float]:
    """The cost, and its early and late parts, from the judgements of the subset."""
    if not reports:
        return float("inf"), float("inf"), float("inf")
    early_terms = []
    late_terms = []
    for r in reports:
        echogram = float(np.mean(r.echogram_distance_db))
        recall_gap = max(0.0, 1.0 - r.errors["recall"])
        early_terms.append(weights.echogram_db * echogram + weights.recall * recall_gap)
        t30 = r.errors["t30_relative"] if np.isfinite(r.errors["t30_relative"]) else 1.0
        colour = r.errors["tail_colour_db"] if np.isfinite(r.errors["tail_colour_db"]) else 30.0
        sector = r.errors["sector_energy_db"] if np.isfinite(r.errors["sector_energy_db"]) else 10.0
        late_terms.append(
            weights.t30_relative * t30
            + weights.tail_colour_db * colour
            + weights.sector_energy_db * sector
        )
    early = float(np.mean(early_terms))
    late = float(np.mean(late_terms))
    return early + late, early, late


def _judge_one(args: tuple[Ambisonic, Ambisonic, CriteriaSettings]) -> PointReport | None:
    """One judgement in a worker; ``None`` where the criteria cannot read the response."""
    reference, response, settings = args
    try:
        return judge(reference, response, Criteria(settings=settings))
    except ValueError:
        return None


def calibrate(
    scene: DerivedScene,
    paths: dict[int, Paths],
    references: dict[int, Ambisonic],
    render_point: Any,
    trace: Any,
    *,
    start: Parameters | None = None,
    weights: CostWeights | None = None,
    criteria: CriteriaSettings | None = None,
    iterations: int = 40,
    step: float = 0.3,
    tied: bool = False,
    workers: int = 1,
    say: Any = print,
) -> tuple[Parameters, list[Evaluation]]:
    """Nelder-Mead over the parameters, on the points of ``paths``.

    ``tied`` searches three coordinates (one absorption scale, the scattering
    scale, one tail gain) instead of fifteen: the coarse pass. ``workers``
    judges the points in parallel on the host's cores.

    ``render_point(index, paths, histogram, scene, parameters) -> Ambisonic``
    renders one point for a candidate scene and histogram, applying the
    parameters' tail gain; ``trace(scene) -> Histogram`` traces the rays for
    a candidate scene. Both are handed in so the search knows nothing about
    cards. Returns the best parameters and every
    evaluation in order, so the trajectory can be plotted and questioned.
    """
    from scipy.optimize import minimize

    weights = weights or CostWeights()
    start = start or Parameters()
    criteria_settings = criteria or CriteriaSettings()
    evaluations: list[Evaluation] = []

    def evaluate(vector: np.ndarray) -> float:
        t0 = time.time()
        parameters = Parameters.from_vector(vector, note="candidate", tied=tied)
        candidate_scene = apply_parameters(scene, parameters)
        histogram = trace(candidate_scene)
        images = image_scene(scene, parameters)
        jobs = []
        for index, point_paths in paths.items():
            response = render_point(
                index, regain(point_paths, images), histogram, candidate_scene, parameters
            )
            jobs.append((references[index], response, criteria_settings))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                judged = list(pool.map(_judge_one, jobs))
        else:
            judged = [_judge_one(job) for job in jobs]
        reports = [r for r in judged if r is not None]
        total, early, late = cost_of(reports, weights)
        evaluation = Evaluation(parameters, total, early, late, reports, time.time() - t0)
        evaluations.append(evaluation)
        say(
            f"calibrate {len(evaluations):3d}: cost {total:8.3f}"
            f" (early {early:7.3f}, late {late:7.3f})"
            f" abs {np.round(parameters.absorption_scale, 2).tolist()}"
            f" scat {parameters.scattering_scale:.2f} in {time.time() - t0:.0f} s"
        )
        return total

    x0 = start.to_vector(tied=tied)
    gains_from = 2 if tied else len(OCTAVE_BANDS) + 1
    simplex = [x0]
    for k in range(x0.size):
        vertex = x0.copy()
        vertex[k] += step if k < gains_from else 3.0
        simplex.append(vertex)
    result = minimize(
        evaluate,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": np.asarray(simplex),
            "maxfev": iterations,
            "xatol": 0.02,
            "fatol": 0.05,
        },
    )
    best = min(evaluations, key=lambda e: e.cost)
    best_parameters = replace(
        best.parameters, note=f"calibrated in {len(evaluations)} evaluations, cost {best.cost:.3f}"
    )
    say(
        f"calibrate: best cost {best.cost:.3f} after {len(evaluations)} evaluations"
        f" ({result.message})"
    )
    return best_parameters, evaluations


def _band_residuals(reports: list[PointReport]) -> tuple[dict[int, float], dict[int, float]]:
    """Per judged band: the median T30 ratio and the median colour gap.

    The ratio is the mirror's T30 over the reference's; the gap is the
    reference's colour minus the mirror's, in dB.
    """
    ratios: dict[int, list[float]] = {}
    gaps: dict[int, list[float]] = {}
    for r in reports:
        ref = r.reference_tail
        cand = r.candidate_tail
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


def _sector_gap(reports: list[PointReport]) -> float:
    """The median over points of the mean absolute sector energy gap, dB."""
    gaps = [
        float(
            np.mean(
                np.abs(
                    np.subtract(
                        r.reference_tail.sector_energy_db, r.candidate_tail.sector_energy_db
                    )
                )
            )
        )
        for r in reports
    ]
    return round(float(np.nanmedian(gaps)), 3) if gaps else float("nan")


def _edt_ratios(reports: list[PointReport]) -> list[float]:
    """Per judged band: the median EDT ratio, mirror over reference."""
    if not reports:
        return []
    ratios = np.asarray(
        [np.divide(r.candidate_tail.edt_s, r.reference_tail.edt_s) for r in reports], dtype=float
    )
    return [float(v) for v in np.nanmedian(ratios, axis=0)]


def _reflection_gaps(reports: list[PointReport]) -> dict[int, float]:
    """Per band: the median level gap of the matched reflections, reference minus mirror, dB.

    Both lists are relative to their own direct sound, band by band, so the
    signature and the air cancel and what remains is the reflections' loss.
    """
    gaps: dict[int, list[float]] = {}
    for r in reports:
        for i, j in r.matching.pairs:
            ref = r.reference[i]
            cand = r.candidate[j]
            if ref.index == 0 or cand.index == 0:
                continue
            for k, band in enumerate(ref.bands.centres_hz):
                a, b = float(ref.bands.levels_db[k]), float(cand.bands.levels_db[k])
                if np.isfinite(a) and np.isfinite(b):
                    gaps.setdefault(int(band), []).append(a - b)
    return {band: float(np.median(v)) for band, v in gaps.items()}


def _weighted(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * values) / np.sum(weight))


def band_cost(
    reports: list[PointReport], weights: CostWeights, focus_low_hz: float = 1000.0
) -> tuple[float, float, float]:
    """The cost with the colour and decay read band by band, not at the worst band.

    ``cost_of`` takes the largest colour error of a point, which on 0076 is
    always 250 Hz (the reference's room modes), so the search never saw the
    tail missing 4 to 6 dB above 1 kHz. Here each band counts, the bands at
    and above 500 Hz fully and 250 Hz at half weight, since the low end will
    come from the wave solver. Bands under ``focus_low_hz`` do not count at
    all (a wave solve answers there); zero keeps the half weight rule.
    """
    if not reports:
        return float("inf"), float("inf"), float("inf")
    early_terms = []
    late_terms = []
    for r in reports:
        echogram = float(np.mean(r.echogram_distance_db))
        recall_gap = max(0.0, 1.0 - r.errors["recall"])
        early_terms.append(weights.echogram_db * echogram + weights.recall * recall_gap)
        ref, cand = r.reference_tail, r.candidate_tail
        band_weight = np.asarray(
            [0.0 if b < 0.99 * focus_low_hz else (0.5 if b < 500 else 1.0) for b in ref.bands_hz]
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


def calibrate_fixed_point(
    scene: DerivedScene,
    paths: dict[int, Paths],
    references: dict[int, Ambisonic],
    render_point: Any,
    trace: Any,
    *,
    start: Parameters | None = None,
    weights: CostWeights | None = None,
    criteria: CriteriaSettings | None = None,
    iterations: int = 8,
    damping: float = 0.8,
    scattering: tuple[float, ...] = (),
    shell_scattering: tuple[float, ...] = (),
    images_apart: bool = True,
    workers: int = 1,
    say: Any = print,
) -> tuple[Parameters, list[Evaluation]]:
    """A band by band fixed point: the decay sets the absorption, the colour sets the tail gain.

    Each band's reverberation time goes as the inverse of its absorption
    (Sabine; Eyring is close at these coefficients), so the absorption scale
    of a band is multiplied by the median ratio of the mirror's T30 to the
    reference's, raised to ``damping``. The tail gain moves the tail's
    energy and nothing else, so the band's gain gets the median colour gap
    in dB, times ``damping``. Seven searches of one coordinate each instead
    of one of fifteen: a handful of evaluations instead of forty. The 125 Hz
    band is not judged; it follows 250 Hz. ``scattering`` lists the
    scattering scales to try after the fixed point, each followed by two
    more steps. With ``images_apart`` the image sources get their own
    absorption scale, moved by the matched reflections' level gap. The best
    evaluation by ``band_cost`` is returned.
    """
    weights = weights or CostWeights()
    parameters = start or Parameters()
    criteria_settings = criteria or CriteriaSettings()
    evaluations: list[Evaluation] = []
    bands = list(OCTAVE_BANDS)
    alpha = mean_absorption(scene)
    if images_apart and parameters.image_absorption_scale is None:
        parameters = replace(parameters, image_absorption_scale=parameters.absorption_scale)

    def evaluate(candidate: Parameters) -> list[PointReport]:
        t0 = time.time()
        candidate_scene = apply_parameters(scene, candidate)
        histogram = trace(candidate_scene)
        images = image_scene(scene, candidate)
        jobs = []
        for index, point_paths in paths.items():
            response = render_point(
                index, regain(point_paths, images), histogram, candidate_scene, candidate
            )
            jobs.append((references[index], response, criteria_settings))
        judged = list(pool.map(_judge_one, jobs)) if pool else [_judge_one(j) for j in jobs]
        reports = [r for r in judged if r is not None]
        total, early, late = band_cost(reports, weights, criteria_settings.focus_low_hz)
        evaluations.append(Evaluation(candidate, total, early, late, reports, time.time() - t0))
        ratios, gaps = _band_residuals(reports)
        levels = _reflection_gaps(reports)
        say(
            f"fixed point {len(evaluations):3d}: cost {total:8.3f}"
            f" (early {early:7.3f}, late {late:7.3f})"
            f" abs {np.round(candidate.absorption_scale, 3).tolist()}"
            f" scat {candidate.scattering_scale:.3f}"
            f" shell {candidate.shell_scattering}"
            f" sectors {_sector_gap(reports)}"
            f" gain {np.round(candidate.tail_gain_db, 2).tolist()}"
            f" | t30 ratio {[round(ratios.get(b, float('nan')), 3) for b in bands]}"
            f" colour gap {[round(gaps.get(b, float('nan')), 2) for b in bands]}"
            f" reflection gap {[round(levels.get(b, float('nan')), 2) for b in bands]}"
            f" edt ratio {[round(v, 3) for v in _edt_ratios(reports)]}"
            f" image abs {np.round(candidate.image_absorption_scale or (), 3).tolist()}"
            f" in {time.time() - t0:.0f} s"
        )
        return reports

    def step(candidate: Parameters, reports: list[PointReport]) -> Parameters:
        ratios, gaps = _band_residuals(reports)
        levels = _reflection_gaps(reports)
        scale = list(candidate.absorption_scale)
        gain = list(candidate.tail_gain_db)
        image = list(candidate.image_absorption_scale or candidate.absorption_scale)
        for k, band in enumerate(bands):
            judged = band if band in ratios else 250
            if judged in ratios:
                scale[k] = float(np.clip(scale[k] * ratios[judged] ** damping, 0.05, 20.0))
            if judged in gaps:
                gain[k] = float(np.clip(gain[k] + damping * gaps[judged], -30.0, 30.0))
            if images_apart and (band if band in levels else 250) in levels:
                # A matched reflection has about two bounces: its energy goes as
                # (1 - s a)^2, so a gap of g dB asks (1 - s a) to move by g / 20 dB.
                gap = levels[band if band in levels else 250]
                kept = (1.0 - image[k] * alpha[k]) * 10.0 ** (damping * gap / 20.0)
                image[k] = float(np.clip((1.0 - kept) / max(alpha[k], 1e-6), 0.05, 20.0))
        return replace(
            candidate,
            absorption_scale=tuple(scale),
            tail_gain_db=tuple(gain),
            image_absorption_scale=tuple(image) if images_apart else None,
            note="candidate",
        )

    # One pool for the whole search: starting workers costs more than judging.
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for _ in range(iterations):
            reports = evaluate(parameters)
            parameters = step(parameters, reports)
        for value in scattering:
            best = min(evaluations, key=lambda e: e.cost).parameters
            candidate = replace(best, scattering_scale=float(value))
            for _ in range(3):
                reports = evaluate(candidate)
                candidate = step(candidate, reports)
        for value in shell_scattering:
            candidate = replace(parameters, shell_scattering=float(value))
            for _ in range(3):
                reports = evaluate(candidate)
                candidate = step(candidate, reports)
    finally:
        if pool is not None:
            pool.shutdown()
    best_evaluation = min(evaluations, key=lambda e: e.cost)
    best_parameters = replace(
        best_evaluation.parameters,
        note=f"fixed point, {len(evaluations)} evaluations, band cost {best_evaluation.cost:.3f}",
    )
    say(f"fixed point: best band cost {best_evaluation.cost:.3f} of {len(evaluations)}")
    return best_parameters, evaluations


def write_calibration(target: Path, best: Parameters, evaluations: list[Evaluation]) -> Path:
    """``<target>/<key>.json`` with the parameters and the trajectory; returns the path."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{best.key}.json"
    path.write_text(
        json.dumps(
            {
                "parameters": best.record(),
                "key": best.key,
                "trajectory": [e.record() for e in evaluations],
            },
            indent=1,
        )
    )
    return path
