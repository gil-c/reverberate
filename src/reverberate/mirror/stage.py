"""The mirror as one stage of a campaign: field, judgement, metrics, on the machine.

Given a run directory holding the wave solver's field of a source, the
export the solver read, and the source, this stage derives the geometry,
grows the image tree, validates the paths of every listening point and
traces the rays on the card(s), renders every point, aligns the result to
the reference's clock and scale, judges every point against the reference
by the criteria, and writes:

- ``mirror/scene.{npz,json}``: the derived geometry and its census;
- ``field_mirror/<source>.h5``: the mirror field in the reference's format;
- ``mirror/metrics/<source>.json``: the criteria, the summary over the
  storey, the solver's own floor and one judgement per point;
- ``mirror/report_<source>.json``: timings, settings, alignment;
- ``walk.json`` updated so the app finds the mirror beside the reference.

Everything that is a choice is in :class:`MirrorSettings` and written to
the report; the derived scene's key and the settings' record make a run
reproducible.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import butter, sosfilt

from reverberate.audio import lowpass
from reverberate.metrics import band_centres, octave_filter_rows
from reverberate.mirror.audit import write_geometry_layers, write_paths
from reverberate.mirror.criteria import (
    Criteria,
    CriteriaSettings,
    PointReport,
    aggregate,
    direct_arrival,
    judge,
)
from reverberate.mirror.engine import device_count, histogram_on_devices, paths_on_devices
from reverberate.mirror.field import align_to_reference, write_mirror_field
from reverberate.mirror.geometry import GeometryRules, derive, load_derived, write_derived
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, occluder_grid
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.mirror.render import RenderSettings, render, render_paths, tail_from_histogram
from reverberate.spatial.encode import Ambisonic

__all__ = ["MirrorSettings", "SolverFloor", "run_mirror"]


@dataclass(frozen=True)
class SolverFloor:
    """What the wave solver reproduces of itself across two grids, part A.

    Measured on 2026-09-16 on hssd_0076: the mid band grid (8.2 mm) against
    the high band grid (4.1 mm), 60 points, 500 Hz to 3.2 kHz. A candidate
    is not asked for more than this; the report prints it beside the targets.
    """

    recall_median: float = 0.88
    precision_median: float = 0.86
    time_error_median_s: float = 4.4e-5
    direction_error_median_deg: float = 0.39
    level_error_median_db: float = 0.26
    source: str = "w41 S1 mid against high encodings, 60 points, 500 Hz to 3.2 kHz"

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MirrorSettings:
    """Every choice of the stage, in one record."""

    rules: GeometryRules = field(default_factory=GeometryRules)
    ism: IsmSettings = field(default_factory=IsmSettings)
    rays: RaySettings = field(default_factory=lambda: RaySettings(rays=1_000_000))
    render: RenderSettings = field(default_factory=RenderSettings)
    criteria: CriteriaSettings = field(default_factory=CriteriaSettings)
    #: Points judged in parallel on the host's cores.
    workers: int = 4
    #: Judge only every n-th point, for a quick look; 1 judges them all.
    judge_every: int = 1
    seed: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "rules": self.rules.record(),
            "ism": self.ism.record(),
            "rays": self.rays.record(),
            "render": self.render.record(),
            "criteria": self.criteria.record(),
            "workers": self.workers,
            "judge_every": self.judge_every,
            "seed": self.seed,
        }


def _direct_energy(response: Ambisonic, settings: CriteriaSettings) -> float:
    """The omni channel's energy over half a millisecond around the direct sound, or zero."""
    rate = response.sample_rate_hz
    try:
        start = direct_arrival(response.signals, rate, settings)
    except ValueError:
        return 0.0
    window = int(round(settings.window_s * rate))
    omni = response.signals[0, max(start - window // 2, 0) : start + window // 2 + 1]
    return float(np.sum(omni**2))


def _render_point(
    index: int,
    paths: Paths,
    histogram: Histogram | None,
    scene_path: Path,
    settings: MirrorSettings,
    sound_speed_m_s: float,
) -> tuple[int, Ambisonic, dict[str, Any]]:
    """One point: the discrete part, then the histogram's tail when there is one."""
    scene = load_derived(scene_path)
    if histogram is None:
        response, plain = render(
            paths, scene, settings.render, sound_speed_m_s=sound_speed_m_s, seed=settings.seed
        )
        return index, response, plain
    early = render_paths(paths, settings.render, sound_speed_m_s)
    record: dict[str, Any] = {"paths": int(paths.count)}
    direct = paths.order == 0
    if direct.any() and histogram.hits[index].sum() > 0:
        rate = settings.render.sample_rate_hz
        distance = float(paths.length_m[direct][0])
        start = int(round((distance / sound_speed_m_s + settings.render.lead_s) * rate))
        centres = band_centres(int(round(rate)))
        window = max(int(round(0.0005 * rate)), 1)
        rows = octave_filter_rows(
            np.repeat(
                early.signals[0:1, max(start - window, 0) : start + window + 1], len(centres), 0
            ),
            int(round(rate)),
            np.arange(len(centres)),
        )
        direct_energy = np.sum(rows**2, axis=1)
        tail, tail_record = tail_from_histogram(
            histogram,
            index,
            direct_energy,
            settings.render,
            sound_speed_m_s=sound_speed_m_s,
            start_s=distance / sound_speed_m_s,
            seed=settings.seed + index,
        )
        signals = early.signals + tail
        record["tail"] = tail_record
    else:
        signals = early.signals
        record["tail"] = None if not direct.any() else "no ray reached this receiver"
    if settings.render.lowcut_hz > 0.0:
        sos = butter(
            settings.render.lowcut_order,
            settings.render.lowcut_hz,
            btype="high",
            fs=settings.render.sample_rate_hz,
            output="sos",
        )
        signals = np.asarray(sosfilt(sos, signals, axis=-1))
    signals = lowpass(signals, settings.render.sample_rate_hz, settings.render.band_limit_hz)
    return index, Ambisonic(signals, early.sample_rate_hz, early.order, early.centre), record


def _judge_point(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None, PointReport | None]:
    """One point's judgement, in a worker: the reference read from the field on disk."""
    index, reference_path, signals, rate, order, settings = args
    with h5py.File(reference_path, "r") as handle:
        reference = Ambisonic(
            np.asarray(handle["ir"][index], dtype=float), rate, order, np.zeros(3)
        )
    candidate = Ambisonic(signals, rate, order, np.zeros(3))
    if float(np.max(np.abs(signals))) == 0.0:
        return index, {"silent": True}, None
    try:
        report = judge(reference, candidate, Criteria(settings=settings))
    except ValueError as error:
        return index, {"error": str(error)}, None
    return index, None, report


def run_mirror(
    run: Path,
    *,
    models: Path,
    source: dict[str, Any],
    settings: MirrorSettings | None = None,
    sound_speed_m_s: float = 343.2,
    devices: list[int] | None = None,
    say: Any = print,
) -> dict[str, Any]:
    """The whole stage for one source of one run; returns the report written."""
    settings = settings or MirrorSettings()
    run = Path(run)
    models = Path(models)
    name = str(source["name"])
    reference = run / "field" / f"{name}.h5"
    if not reference.is_file():
        raise FileNotFoundError(f"{reference}: the wave field of {name} is not there")
    report: dict[str, Any] = {
        "source": name,
        "settings": settings.record(),
        "devices": device_count(),
        "timings_s": {},
    }
    started = time.time()

    def stage(label: str, work: Any) -> Any:
        t0 = time.time()
        result = work()
        report["timings_s"][label] = round(time.time() - t0, 1)
        say(f"mirror {name} | {label}: {time.time() - t0:.1f} s")
        return result

    # 1. The derived geometry, once per run and rules.
    mirror_dir = run / "mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    scene_path = mirror_dir / "scene"

    def derive_scene() -> Any:
        if scene_path.with_suffix(".json").is_file():
            found = load_derived(scene_path)
            if found.rules == settings.rules:
                return found
        manifest_path = models / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        derived = derive(
            models / "apartment_full.json",
            rules=settings.rules,
            manifest=manifest,
            seed=settings.seed,
        )
        write_derived(derived, scene_path)
        return derived

    scene = stage("derive", derive_scene)
    stage("audit layers", lambda: write_geometry_layers(scene, mirror_dir / "audit"))
    report["scene"] = {
        "key": scene.key,
        "summary": scene.summary(),
        "census": scene.census["totals"],
    }

    # 2. The points and the tree.
    with h5py.File(reference, "r") as handle:
        positions = np.asarray(handle["positions"][...], dtype=float)
        rate = float(handle.attrs["sample_rate_hz"])
        order = int(handle.attrs["order"])
    position = np.asarray(source["position"], dtype=float)
    region = (positions.min(axis=0) - 0.5, positions.max(axis=0) + 0.5)
    ism = IsmSettings(**{**settings.ism.record(), "sound_speed_m_s": sound_speed_m_s})
    tree = stage("tree", lambda: grow_tree(scene, position, ism, region=region))
    report["tree"] = {"images": tree.count, "orders": np.bincount(tree.order).tolist()}
    grid = stage("grid", lambda: occluder_grid(scene, settings.rays.cell_m))

    # 3. The card: paths of every point, then the rays.
    every = stage(
        "paths",
        lambda: paths_on_devices(scene, tree, positions, ism, devices=devices, grid=grid, say=say),
    )
    report["paths"] = {
        "median": float(np.median([p.count for p in every])),
        "max": int(max(p.count for p in every)),
        "without_direct": int(sum(1 for p in every if not np.any(p.order == 0))),
    }
    stage(
        "audit paths",
        lambda: write_paths(
            every, scene, mirror_dir / "paths" / f"{name}.json", sound_speed_m_s=sound_speed_m_s
        ),
    )
    rays = RaySettings(**{**settings.rays.record(), "sound_speed_m_s": sound_speed_m_s})
    histogram = stage(
        "rays",
        lambda: histogram_on_devices(
            scene, position, positions, rays, devices=devices, grid=grid, say=say
        ),
    )
    report["rays"] = {
        "hits_total": int(histogram.hits.sum()),
        "hits_per_receiver_median": float(np.median(histogram.hits.sum(axis=1))),
    }
    np.savez_compressed(
        mirror_dir / f"histogram_{name}.npz",
        energy=histogram.energy,
        moments=histogram.moments,
        hits=histogram.hits,
    )

    # 4. Every point rendered on the host's cores.
    render_settings = RenderSettings(
        **{**settings.render.record(), "order": order, "sample_rate_hz": rate}
    )
    settings_render = MirrorSettings(**{**asdict(settings), "render": render_settings})

    def render_all() -> dict[int, tuple[Ambisonic, dict[str, Any]]]:
        out: dict[int, tuple[Ambisonic, dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            futures = [
                pool.submit(
                    _render_point,
                    i,
                    every[i],
                    histogram,
                    scene_path,
                    settings_render,
                    sound_speed_m_s,
                )
                for i in range(len(every))
            ]
            for future in futures:
                index, response, record = future.result()
                out[index] = (response, record)
        return out

    rendered = stage("render", render_all)

    # 5. Alignment to the reference, then the field.
    criteria_settings = settings.criteria
    direct_energy = {i: _direct_energy(r, criteria_settings) for i, (r, _) in rendered.items()}
    alignment = stage(
        "align",
        lambda: align_to_reference(reference, direct_energy, sound_speed_m_s=sound_speed_m_s),
    )
    report["alignment"] = alignment.record()
    lead = int(round(alignment.lead_s * rate))
    aligned: dict[int, Ambisonic] = {}
    for i, (response, _) in rendered.items():
        signals = np.zeros_like(response.signals)
        if lead >= 0:
            signals[:, lead:] = response.signals[:, : response.signals.shape[1] - lead]
        else:
            signals[:, : lead or None] = response.signals[:, -lead:]
        aligned[i] = Ambisonic(signals, rate, order, response.centre)
    field_path = stage(
        "field",
        lambda: write_mirror_field(
            run / "field_mirror" / f"{name}.h5",
            reference,
            aligned,
            provenance={
                "scene_key": scene.key,
                "settings": settings.record(),
                "alignment": alignment.record(),
                "tree": report["tree"],
            },
            gain=alignment.gain,
        ),
    )

    # 6. The judgement, point by point, in parallel.
    chosen = list(range(0, len(every), max(1, settings.judge_every)))

    def judge_all() -> dict[str, Any]:
        reports: list[PointReport] = []
        points: dict[str, Any] = {}
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            jobs = [
                pool.submit(
                    _judge_point,
                    (
                        i,
                        reference,
                        aligned[i].signals * alignment.gain,
                        rate,
                        order,
                        criteria_settings,
                    ),
                )
                for i in chosen
            ]
            for job in jobs:
                index, problem, point_report = job.result()
                if point_report is None:
                    points[str(index)] = {**(problem or {}), "paths": int(every[index].count)}
                    continue
                reports.append(point_report)
                points[str(index)] = {
                    **point_report.record(),
                    "paths": int(every[index].count),
                    "orders": np.bincount(every[index].order).tolist(),
                }
        return {
            "source": name,
            "criteria": Criteria(settings=criteria_settings).record(),
            "floor": SolverFloor().record(),
            "summary": aggregate(reports),
            "judged": len(chosen),
            "points": points,
        }

    metrics = stage("judge", judge_all)
    metrics_dir = mirror_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(metrics, indent=1, default=str))
    report["summary"] = metrics["summary"]

    # 7. walk.json, so the app finds the mirror.
    walk = run / "walk.json"
    if walk.is_file():
        manifest = json.loads(walk.read_text())
        for entry in manifest.get("sources", []):
            if str(entry.get("id")) == name or str(entry.get("name")) == name:
                entry["field_mirror"] = str(field_path.relative_to(run))
                entry["metrics"] = str((metrics_dir / f"{name}.json").relative_to(run))
        manifest["mirror"] = {
            "scene": "mirror/scene.json",
            "key": scene.key,
            "audit": "mirror/audit",
            "paths": {
                **(manifest.get("mirror") or {}).get("paths", {}),
                name: f"mirror/paths/{name}.json",
            },
        }
        walk.write_text(json.dumps(manifest, indent=1))
    report["total_s"] = round(time.time() - started, 1)
    (mirror_dir / f"report_{name}.json").write_text(json.dumps(report, indent=1, default=str))
    say(f"mirror {name}: done in {report['total_s'] / 60:.1f} min")
    return report
