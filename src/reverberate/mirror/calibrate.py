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

__all__ = ["Parameters", "apply_parameters", "calibrate", "cost_of", "regain"]


@dataclass(frozen=True)
class Parameters:
    """What the calibration may move, with the catalogue's values as the origin."""

    absorption_scale: tuple[float, ...] = tuple(1.0 for _ in OCTAVE_BANDS)
    scattering_scale: float = 1.0
    tail_gain_db: tuple[float, ...] = tuple(0.0 for _ in OCTAVE_BANDS)
    note: str = "catalogue values, nothing calibrated"

    @property
    def key(self) -> str:
        digest = hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def record(self) -> dict[str, Any]:
        return {
            "bands_hz": list(OCTAVE_BANDS),
            "absorption_scale": [round(float(v), 6) for v in self.absorption_scale],
            "scattering_scale": round(float(self.scattering_scale), 6),
            "tail_gain_db": [round(float(v), 4) for v in self.tail_gain_db],
            "note": self.note,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Parameters:
        return cls(
            absorption_scale=tuple(float(v) for v in record["absorption_scale"]),
            scattering_scale=float(record["scattering_scale"]),
            tail_gain_db=tuple(float(v) for v in record["tail_gain_db"]),
            note=str(record.get("note", "")),
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
    materials = MaterialTable(
        scene.materials.labels,
        absorption,
        scattering,
        scene.materials.bands_hz,
        scene.materials.source,
    )
    return replace(scene, materials=materials)


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
        jobs = []
        for index, point_paths in paths.items():
            response = render_point(
                index, regain(point_paths, candidate_scene), histogram, candidate_scene, parameters
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
