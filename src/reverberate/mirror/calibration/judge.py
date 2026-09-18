"""One point's verdict against the wave field, and the storey's summary."""

from __future__ import annotations

from typing import Any

import numpy as np

from reverberate.mirror.calibration.criteria import (
    LEVEL_BANDS_HZ,
    Criteria,
    CriteriaSettings,
    PointReport,
    _round_or_none,
)
from reverberate.mirror.calibration.early import (
    detect_reflections,
    echogram,
    echogram_distance_db,
    match_reflections,
)
from reverberate.mirror.calibration.tail import mixing_time_s, tail_statistics
from reverberate.mirror.direct import bandpass as _bandpass
from reverberate.spatial.encode import Ambisonic


def judged_bands(
    settings: CriteriaSettings, bands_hz: tuple[int, ...] = LEVEL_BANDS_HZ
) -> tuple[int, ...]:
    """The octave bands the judgement reads: those at and above ``low_hz``."""
    return tuple(b for b in bands_hz if b >= settings.low_hz * 0.99)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    """Largest relative error over the bands where both are finite."""
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask]) / a[mask]))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask])))


def judge(
    reference: Ambisonic,
    candidate: Ambisonic,
    criteria: Criteria | None = None,
) -> PointReport:
    """Everything of section 3 of the plan, for one point, with a verdict per criterion."""
    criteria = criteria or Criteria()
    s, t = criteria.settings, criteria.targets
    if reference.sample_rate_hz != candidate.sample_rate_hz:
        raise ValueError("the two responses must share a sample rate")
    ref_list = detect_reflections(reference, s)
    cand_list = detect_reflections(candidate, s)
    matching = match_reflections(ref_list, cand_list, s)
    judged = judged_bands(s)
    distance = echogram_distance_db(
        echogram(reference, s, bands_hz=judged), echogram(candidate, s, bands_hz=judged)
    )
    ref_tail = tail_statistics(reference, s)
    cand_tail = tail_statistics(candidate, s, mixing_time=ref_tail.mixing_time_s)
    kept = np.array([b in judged for b in ref_tail.bands_hz])

    def focus(values: tuple[float, ...]) -> np.ndarray:
        return np.asarray(np.asarray(values, dtype=float)[kept])

    cand_own_mix = mixing_time_s(
        _bandpass(candidate.signals[0:1], candidate.sample_rate_hz, None, s.band_limit_hz)[0],
        candidate.sample_rate_hz,
        s,
    )

    def median(values: tuple[float, ...]) -> float:
        return float(np.median(values)) if values else float("nan")

    errors = {
        "recall": matching.recall,
        "precision": matching.precision,
        "time_error_s": median(matching.time_errors_s),
        "direction_error_deg": median(matching.direction_errors_deg),
        "level_error_db": median(matching.level_errors_db),
        "itd_error_s": abs(ref_tail.direct_itd_s - cand_tail.direct_itd_s),
        "ild_error_db": abs(ref_tail.direct_ild_db - cand_tail.direct_ild_db),
        "early_coherence_error": abs(ref_tail.early_coherence - cand_tail.early_coherence),
        "mixing_time_relative": abs(cand_own_mix - ref_tail.mixing_time_s) / ref_tail.mixing_time_s
        if np.isfinite(cand_own_mix) and ref_tail.mixing_time_s > 0
        else float("nan"),
        "t30_relative": _relative(focus(ref_tail.t30_s), focus(cand_tail.t30_s)),
        "edt_relative": _relative(focus(ref_tail.edt_s), focus(cand_tail.edt_s)),
        "tail_colour_db": _max_abs(focus(ref_tail.colour_db), focus(cand_tail.colour_db)),
        "order_energy_db": _max_abs(
            np.asarray(ref_tail.order_energy_db), np.asarray(cand_tail.order_energy_db)
        ),
        "sector_energy_db": _max_abs(
            np.asarray(ref_tail.sector_energy_db), np.asarray(cand_tail.sector_energy_db)
        ),
        "late_coherence_error": abs(ref_tail.late_coherence - cand_tail.late_coherence),
        "seam_db": _max_abs(focus(ref_tail.seam_step_db), focus(cand_tail.seam_step_db)),
    }
    limits = {
        "recall": (t.recall, "min"),
        "precision": (t.precision, "min"),
        "time_error_s": (t.time_error_s, "max"),
        "direction_error_deg": (t.direction_error_deg, "max"),
        "level_error_db": (t.level_error_db, "max"),
        "itd_error_s": (t.itd_error_s, "max"),
        "ild_error_db": (t.ild_error_db, "max"),
        "early_coherence_error": (t.early_coherence_error, "max"),
        "mixing_time_relative": (t.mixing_time_relative, "max"),
        "t30_relative": (t.t30_relative, "max"),
        "edt_relative": (t.edt_relative, "max"),
        "tail_colour_db": (t.tail_colour_db, "max"),
        "order_energy_db": (t.order_energy_db, "max"),
        "sector_energy_db": (t.sector_energy_db, "max"),
        "late_coherence_error": (t.late_coherence_error, "max"),
        "seam_db": (t.seam_db, "max"),
    }
    verdicts = {}
    for name, (limit, sense) in limits.items():
        value = errors[name]
        if not np.isfinite(value):
            verdicts[name] = False
        else:
            verdicts[name] = bool(value >= limit) if sense == "min" else bool(value <= limit)
    return PointReport(
        reference=ref_list,
        candidate=cand_list,
        matching=matching,
        echogram_distance_db=distance,
        reference_tail=ref_tail,
        candidate_tail=cand_tail,
        verdicts=verdicts,
        errors=errors,
    )


def aggregate(reports: list[PointReport]) -> dict[str, Any]:
    """Medians of every error and the share of points passing each criterion."""
    if not reports:
        return {"points": 0}
    names = list(reports[0].errors)
    out: dict[str, Any] = {"points": len(reports), "errors": {}, "pass_fraction": {}}
    for name in names:
        values = np.asarray([r.errors[name] for r in reports], dtype=float)
        finite = values[np.isfinite(values)]
        out["errors"][name] = {
            "median": _round_or_none(float(np.median(finite)), 5) if finite.size else None,
            "p90": _round_or_none(float(np.percentile(finite, 90)), 5) if finite.size else None,
        }
        out["pass_fraction"][name] = round(float(np.mean([r.verdicts[name] for r in reports])), 4)
    out["echogram_distance_db"] = [
        _round_or_none(float(np.median([r.echogram_distance_db[k] for r in reports])), 3)
        for k in range(len(reports[0].echogram_distance_db))
    ]
    return out
