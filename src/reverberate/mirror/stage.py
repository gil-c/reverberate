"""The mirror as one stage of a campaign: field, judgement, metrics, on the machine.

Given a run directory holding the wave solver's field of a source, the
export the solver read, and the source, this stage derives the geometry,
grows the image tree, validates the paths of every listening point and
traces the rays on the card(s), renders every point, aligns the result to
the reference's clock and scale, judges every point against the reference
by the criteria, and writes:

- ``mirror/scene.{npz,json}``: the derived geometry and its census;
- ``mirror/paths_<source>.npz`` and ``mirror/histogram_<source>.npz``: what
  the card computed, small enough to travel;
- ``field_mirror/<source>.h5``: the mirror field in the reference's format;
- ``mirror/metrics/<source>.json``: the criteria, the summary over the
  storey, the solver's own floor and one judgement per point;
- ``mirror/report_<source>.json``: timings, settings, alignment;
- ``walk.json`` updated so the app finds the mirror beside the reference.

The stage has two phases, so the card's work and the host's work can run on
different machines without the reference field travelling: the **card**
phase needs the derived geometry and the lattice's positions and writes the
paths and the histogram; the **host** phase needs those and the reference
field and writes the rest. ``phase="all"`` runs both in one place.

Everything that is a choice is in :class:`MirrorSettings` and written to
the report; the derived scene's key and the settings' record make a run
reproducible.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import butter, sosfilt

from reverberate.audio import apply_air_absorption, lowpass
from reverberate.metrics import band_centres, octave_filter_rows
from reverberate.mirror.audit import write_geometry_layers, write_paths
from reverberate.mirror.calibrate import Parameters, apply_parameters, image_scene, regain
from reverberate.mirror.criteria import (
    Criteria,
    CriteriaSettings,
    PointReport,
    aggregate,
    direct_arrival,
    judge,
)
from reverberate.mirror.diffract import diffracted_paths
from reverberate.mirror.engine import device_count, histogram_on_devices, paths_on_devices
from reverberate.mirror.field import align_to_reference, write_mirror_field
from reverberate.mirror.geometry import (
    DerivedScene,
    GeometryRules,
    derive,
    load_derived,
    write_derived,
)
from reverberate.mirror.ism import IsmSettings, Paths, grow_tree, occluder_grid
from reverberate.mirror.rays import Histogram, RaySettings
from reverberate.mirror.render import (
    RenderSettings,
    _band_map,
    band_pulse_energy,
    render,
    render_paths,
    tail_from_histogram,
)
from reverberate.mirror.signature import apply_signature, measure_signature, write_signature
from reverberate.spatial.encode import Ambisonic

__all__ = [
    "MirrorSettings",
    "SolverFloor",
    "load_every",
    "load_histogram",
    "run_mirror",
    "write_every",
    "write_histogram",
]

PHASES = ("card", "host", "all")


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
    #: The calibration applied: scales on the materials, gains on the tail.
    parameters: Parameters = field(default_factory=Parameters)
    #: Give the mirror the reference's direct pulse spectrum (``mirror.signature``).
    signature: bool = True
    #: Points the signature is read on.
    signature_points: int = 48
    #: Give a point without a direct path its diffracted onset (``mirror.diffract``).
    diffraction: bool = True

    def record(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters.record(),
            "rules": self.rules.record(),
            "ism": self.ism.record(),
            "rays": self.rays.record(),
            "render": self.render.record(),
            "criteria": self.criteria.record(),
            "workers": self.workers,
            "judge_every": self.judge_every,
            "seed": self.seed,
            "signature": self.signature,
            "signature_points": self.signature_points,
            "diffraction": self.diffraction,
        }


# --------------------------------------------------------------------------
# what the card hands to the host
# --------------------------------------------------------------------------


def write_every(every: list[Paths], target: Path) -> Path:
    """The paths of every point in one ``.npz``: concatenated, with offsets."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([p.count for p in every], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])

    def stack(name: str, empty_shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        rows = [getattr(p, name) for p in every if p.count]
        if not rows:
            return np.zeros((0, *empty_shape), dtype=dtype)
        return np.concatenate([np.asarray(r) for r in rows], axis=0)

    bands = int(every[0].gain.shape[1]) if every and every[0].gain.ndim == 2 else 7
    width = max((int(p.sequence.shape[1]) for p in every if p.count), default=1)
    rows = max((int(p.points.shape[1]) for p in every if p.count), default=2)
    np.savez_compressed(
        target,
        offsets=offsets,
        receiver=stack("receiver", (), np.int64),
        image=stack("image", (), np.int64),
        order=stack("order", (), np.int64),
        length_m=stack("length_m", (), float),
        direction=stack("direction", (3,), float),
        gain=stack("gain", (bands,), float),
        points=stack("points", (rows, 3), float),
        sequence=stack("sequence", (width,), np.int64),
    )
    return target if target.suffix == ".npz" else target.with_suffix(".npz")


def load_every(target: Path) -> list[Paths]:
    """The inverse of :func:`write_every`."""
    with np.load(Path(target)) as arrays:
        offsets = arrays["offsets"]
        names = (
            "receiver",
            "image",
            "order",
            "length_m",
            "direction",
            "gain",
            "points",
            "sequence",
        )
        columns = {name: arrays[name] for name in names}
    return [
        Paths(**{name: columns[name][offsets[i] : offsets[i + 1]] for name in names})
        for i in range(len(offsets) - 1)
    ]


def write_histogram(histogram: Histogram, target: Path) -> Path:
    target = Path(target)
    np.savez_compressed(
        target,
        energy=histogram.energy,
        moments=histogram.moments,
        hits=histogram.hits,
        bin_s=histogram.bin_s,
        bands_hz=np.asarray(histogram.bands_hz, dtype=np.int64),
        order=histogram.order,
        rays=histogram.rays,
    )
    return target if target.suffix == ".npz" else target.with_suffix(".npz")


def load_histogram(target: Path) -> Histogram:
    with np.load(Path(target)) as arrays:
        return Histogram(
            arrays["energy"],
            arrays["moments"],
            arrays["hits"],
            float(arrays["bin_s"]),
            tuple(int(v) for v in arrays["bands_hz"]),
            int(arrays["order"]),
            int(arrays["rays"]),
        )


def _lattice_of(run: Path, name: str) -> tuple[np.ndarray, float, int]:
    """The positions, rate and order: from the reference field, else its lattice file.

    The card phase runs where the field may not be; ``mirror/lattice_<S>.npz``
    with ``positions``, ``sample_rate_hz`` and ``order`` stands in for it.
    """
    reference = run / "field" / f"{name}.h5"
    if reference.is_file():
        with h5py.File(reference, "r") as handle:
            return (
                np.asarray(handle["positions"][...], dtype=float),
                float(handle.attrs["sample_rate_hz"]),
                int(handle.attrs["order"]),
            )
    lattice = run / "mirror" / f"lattice_{name}.npz"
    if not lattice.is_file():
        raise FileNotFoundError(f"neither {reference} nor {lattice}: no lattice to mirror")
    with np.load(lattice) as arrays:
        return (
            np.asarray(arrays["positions"], dtype=float),
            float(arrays["sample_rate_hz"]),
            int(arrays["order"]),
        )


# --------------------------------------------------------------------------
# the host's workers
# --------------------------------------------------------------------------


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
    scene_path: Path | DerivedScene,
    settings: MirrorSettings,
    sound_speed_m_s: float,
    fallback: tuple[Any, ...] | None = None,
    signature: np.ndarray | None = None,
) -> tuple[int, Ambisonic, dict[str, Any]]:
    """One point: the discrete part, then the histogram's tail when there is one.

    ``paths`` carry their gains, so the calibration's material scales are
    already in them; the tail gain of ``settings.parameters`` applies here.
    ``fallback`` is ``(scale_per_band, straight_line_s)`` for a point without
    a direct path: the tail's scale as the other points read it, and the
    straight line time the tail's clock starts from; a third item, when
    there, is the point's diffracted onset (a one-path ``Paths``), rendered
    with the early part, whose arrival is where the tail's clock starts.
    """
    scene = scene_path if isinstance(scene_path, DerivedScene) else load_derived(scene_path)
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
        analytic = None
        if settings.render.analytic_direct:
            # What a sphere of radius r at distance d catches of rays of energy 1/N,
            # against the direct pulse's whole energy per band (gain squared times
            # the bank's energy for a unit pulse), not what a 1 ms window keeps.
            radius = settings.rays.receiver_radius_m
            expected = radius**2 / (4.0 * max(distance, 1.05 * radius) ** 2)
            _, picks = _band_map(rate)
            amplitude = paths.gain[np.flatnonzero(direct)[0]][picks]
            whole = amplitude**2 * band_pulse_energy(rate)
            analytic = np.asarray(whole / expected, dtype=float)
        tail, tail_record = tail_from_histogram(
            histogram,
            index,
            direct_energy,
            settings.render,
            sound_speed_m_s=sound_speed_m_s,
            start_s=distance / sound_speed_m_s,
            seed=settings.seed + index,
            bursts=settings.render.tail_bursts,
            band_gain_db=np.asarray(settings.parameters.tail_gain_db, dtype=float),
            scale_per_band=analytic,
        )
        signals = early.signals + tail
        record["tail"] = tail_record
    elif fallback is not None and not direct.any() and histogram.hits[index].sum() > 0:
        scale, straight_s = fallback[0], fallback[1]
        onset = fallback[2] if len(fallback) > 2 else None
        # The rays' own first arrival, round the doorway, is where the tail starts:
        # the straight line through the wall is not a clock here.
        arrived = np.any(histogram.energy[index] > 0.0, axis=1)
        first_s = float(np.argmax(arrived)) * histogram.bin_s
        start_s = max(first_s - settings.render.tail_from_s, straight_s)
        if onset is not None:
            # The diffracted onset is the first arrival: the tail starts after it.
            start_s = float(onset.length_m[0]) / sound_speed_m_s
            early = Ambisonic(
                early.signals + render_paths(onset, settings.render, sound_speed_m_s).signals,
                early.sample_rate_hz,
                early.order,
                early.centre,
            )
            record["diffracted"] = {
                "length_m": round(float(onset.length_m[0]), 4),
                "gain_db": [round(float(v), 2) for v in 20.0 * np.log10(onset.gain[0])],
            }
        tail, tail_record = tail_from_histogram(
            histogram,
            index,
            np.zeros(len(scale)),
            settings.render,
            sound_speed_m_s=sound_speed_m_s,
            start_s=start_s,
            seed=settings.seed + index,
            bursts=settings.render.tail_bursts,
            band_gain_db=np.asarray(settings.parameters.tail_gain_db, dtype=float),
            scale_per_band=scale,
        )
        signals = early.signals + tail
        record["tail"] = {
            **tail_record,
            "scale_from": "the other points, no direct path here",
            "starts_s": round(start_s, 4),
        }
    else:
        signals = early.signals
        record["tail"] = None if not direct.any() else "no ray reached this receiver"
    if settings.render.air_absorption:
        signals = apply_air_absorption(
            signals, settings.render.sample_rate_hz, sound_speed_m_s=sound_speed_m_s
        )
    if signature is not None and signature.size:
        signals = apply_signature(signals, signature)
    if settings.render.lowcut_hz > 0.0:
        sos = butter(
            settings.render.lowcut_order,
            settings.render.lowcut_hz,
            btype="high",
            fs=settings.render.sample_rate_hz,
            output="sos",
        )
        signals = np.asarray(sosfilt(sos, signals, axis=-1))
    if settings.render.band_limit_hz > 0.0:
        signals = lowpass(signals, settings.render.sample_rate_hz, settings.render.band_limit_hz)
    return index, Ambisonic(signals, early.sample_rate_hz, early.order, early.centre), record


def _render_to_disk(args: tuple[Any, ...]) -> tuple[int, float, dict[str, Any]]:
    """The render worker of the host phase: the response to ``scratch/<index>.npy``.

    A storey of 437 points at order 7 is 13 GB of responses; they go through
    the disk one at a time and only the direct energy comes back.
    """
    index, paths, histogram, scene_path, settings, sound_speed_m_s, scratch, fallback = args[:8]
    signature = args[8] if len(args) > 8 else None
    _, response, record = _render_point(
        index, paths, histogram, scene_path, settings, sound_speed_m_s, fallback, signature
    )
    np.save(Path(scratch) / f"{index}.npy", response.signals.astype(np.float32))
    return index, _direct_energy(response, settings.criteria), record


def _shifted(signals: np.ndarray, lead: int) -> np.ndarray:
    """``signals`` delayed by ``lead`` samples (advanced when negative), same length."""
    out = np.zeros_like(signals)
    if lead >= 0:
        out[:, lead:] = signals[:, : signals.shape[1] - lead]
    else:
        out[:, : lead or None] = signals[:, -lead:]
    return out


class _Cached(Mapping[int, Ambisonic]):
    """The rendered responses on disk, aligned on access: what the field writer reads."""

    def __init__(
        self, scratch: Path, indices: list[int], rate: float, order: int, lead: int
    ) -> None:
        self.scratch = Path(scratch)
        self.indices = indices
        self.rate = rate
        self.order = order
        self.lead = lead

    def __getitem__(self, index: int) -> Ambisonic:
        if index not in self.indices:
            raise KeyError(index)
        signals = np.load(self.scratch / f"{index}.npy").astype(np.float64)
        return Ambisonic(_shifted(signals, self.lead), self.rate, self.order, np.zeros(3))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _judge_point(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None, PointReport | None]:
    """One point's judgement, in a worker: the reference read from the field on disk."""
    index, reference_path, cached, lead, gain, rate, order, settings = args
    signals = _shifted(np.load(Path(cached)).astype(np.float64), lead) * gain
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


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


class _Stage:
    """The run's paths and the report under construction, shared by the phases."""

    def __init__(
        self,
        run: Path,
        name: str,
        settings: MirrorSettings,
        sound_speed_m_s: float,
        say: Any,
        tag: str = "",
    ) -> None:
        self.run = Path(run)
        self.name = name
        self.tag = tag
        self.suffix = f"_{tag}" if tag else ""
        self.settings = settings
        self.sound_speed_m_s = sound_speed_m_s
        self.say = say
        self.mirror_dir = self.run / "mirror"
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        self.scene_path = self.mirror_dir / "scene"
        self.reference = self.run / "field" / f"{name}.h5"
        self.card_report_path = self.mirror_dir / f"card_{name}{self.suffix}.json"
        self.report: dict[str, Any] = {
            "source": name,
            "tag": tag,
            "settings": settings.record(),
            "timings_s": {},
        }

    def stage(self, label: str, work: Any) -> Any:
        t0 = time.time()
        result = work()
        self.report["timings_s"][label] = round(time.time() - t0, 1)
        self.say(f"mirror {self.name} | {label}: {time.time() - t0:.1f} s")
        return result


def _card_phase(
    s: _Stage, *, models: Path, source: dict[str, Any], devices: list[int] | None
) -> dict[str, Any]:
    """Derive, tree, paths and rays on the card(s); writes what the host needs."""
    settings = s.settings
    s.report["devices"] = device_count()

    def derive_scene() -> Any:
        if s.scene_path.with_suffix(".json").is_file():
            found = load_derived(s.scene_path)
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
        write_derived(derived, s.scene_path)
        return derived

    catalogue = s.stage("derive", derive_scene)
    s.stage("audit layers", lambda: write_geometry_layers(catalogue, s.mirror_dir / "audit"))
    scene = apply_parameters(catalogue, settings.parameters)
    s.report["scene"] = {
        "key": catalogue.key,
        "summary": catalogue.summary(),
        "census": catalogue.census["totals"],
    }
    s.report["parameters"] = settings.parameters.record()
    s.report["source_position"] = [float(v) for v in np.asarray(source["position"])]

    positions, _, _ = _lattice_of(s.run, s.name)
    position = np.asarray(source["position"], dtype=float)
    region = (positions.min(axis=0) - 0.5, positions.max(axis=0) + 0.5)
    ism = IsmSettings(**{**settings.ism.record(), "sound_speed_m_s": s.sound_speed_m_s})
    tree = s.stage("tree", lambda: grow_tree(scene, position, ism, region=region))
    s.report["tree"] = {"images": tree.count, "orders": np.bincount(tree.order).tolist()}
    grid = s.stage("grid", lambda: occluder_grid(scene, settings.rays.cell_m))

    every = s.stage(
        "paths",
        lambda: paths_on_devices(
            scene, tree, positions, ism, devices=devices, grid=grid, say=s.say
        ),
    )
    if settings.parameters.image_absorption_scale is not None:
        # The images' gains read their own absorption scale.
        images = image_scene(catalogue, settings.parameters)
        every = [regain(p, images) for p in every]
    s.report["paths"] = {
        "median": float(np.median([p.count for p in every])),
        "max": int(max(p.count for p in every)),
        "without_direct": int(sum(1 for p in every if not np.any(p.order == 0))),
    }
    write_every(every, s.mirror_dir / f"paths_{s.name}{s.suffix}.npz")
    s.stage(
        "audit paths",
        lambda: write_paths(
            every,
            scene,
            s.mirror_dir / "paths" / f"{s.name}.json",
            sound_speed_m_s=s.sound_speed_m_s,
        ),
    )
    rays = RaySettings(**{**settings.rays.record(), "sound_speed_m_s": s.sound_speed_m_s})
    histogram = s.stage(
        "rays",
        lambda: histogram_on_devices(
            scene, position, positions, rays, devices=devices, grid=grid, say=s.say
        ),
    )
    s.report["rays"] = {
        "hits_total": int(histogram.hits.sum()),
        "hits_per_receiver_median": float(np.median(histogram.hits.sum(axis=1))),
    }
    write_histogram(histogram, s.mirror_dir / f"histogram_{s.name}{s.suffix}.npz")
    s.card_report_path.write_text(json.dumps(s.report, indent=1, default=str))
    return s.report


def _host_phase(s: _Stage) -> dict[str, Any]:
    """Render, align, write the field, judge, update walk.json; needs the reference."""
    settings = s.settings
    name = s.name
    if not s.reference.is_file():
        raise FileNotFoundError(f"{s.reference}: the wave field of {name} is not there")
    if not s.card_report_path.is_file():
        raise FileNotFoundError(f"{s.card_report_path}: the card phase did not run here")
    card = json.loads(s.card_report_path.read_text())
    for key in ("devices", "scene", "tree", "paths", "rays"):
        s.report[key] = card.get(key)
    s.report["timings_s"] = {**card.get("timings_s", {}), **s.report["timings_s"]}
    s.report["card_settings"] = card.get("settings")
    if card.get("parameters"):
        # The card's gains carry its parameters; the tail gain must be the same ones.
        settings = replace(settings, parameters=Parameters.from_record(card["parameters"]))
        s.settings = settings
    s.report["parameters"] = settings.parameters.record()
    scene = load_derived(s.scene_path)
    every = load_every(s.mirror_dir / f"paths_{name}{s.suffix}.npz")
    histogram = load_histogram(s.mirror_dir / f"histogram_{name}{s.suffix}.npz")
    _, rate, order = _lattice_of(s.run, name)

    render_settings = RenderSettings(
        **{**settings.render.record(), "order": order, "sample_rate_hz": rate}
    )
    settings_render = replace(settings, render=render_settings)

    scratch = s.mirror_dir / f"render_{name}{s.suffix}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)

    positions, _, _ = _lattice_of(s.run, name)
    source_position = np.asarray(card.get("source_position") or [np.nan] * 3, dtype=float)
    with_direct = [i for i in range(len(every)) if bool(np.any(every[i].order == 0))]
    without = sorted(set(range(len(every))) - set(with_direct))
    onsets: dict[int, Paths] = {}
    if settings.diffraction and without and bool(np.all(np.isfinite(source_position))):
        onsets, diffraction_record = s.stage(
            "diffraction",
            lambda: diffracted_paths(
                scene,
                source_position,
                positions,
                without,
                sound_speed_m_s=s.sound_speed_m_s,
                say=s.say,
            ),
        )
        s.report["diffraction"] = diffraction_record
    signature = None
    if settings.signature:
        step = max(1, len(with_direct) // max(1, settings.signature_points))
        taps, signature_record = s.stage(
            "signature",
            lambda: measure_signature(s.reference, with_direct[::step], settings=settings.criteria),
        )
        write_signature(s.mirror_dir / f"signature_{name}{s.suffix}", taps, signature_record)
        s.report["signature"] = signature_record
        signature = taps

    def render_all() -> dict[int, float]:
        """Points with a direct path first, so their tail scales serve the rest."""
        energies: dict[int, float] = {}
        scales: list[np.ndarray] = []

        def submit(pool: ProcessPoolExecutor, i: int, fallback: Any) -> Any:
            return pool.submit(
                _render_to_disk,
                (
                    i,
                    every[i],
                    histogram,
                    s.scene_path,
                    settings_render,
                    s.sound_speed_m_s,
                    scratch,
                    fallback,
                    signature,
                ),
            )

        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            for job in [submit(pool, i, None) for i in with_direct]:
                index, energy, record = job.result()
                energies[index] = energy
                tail = record.get("tail")
                if isinstance(tail, dict) and "scale_per_band" in tail:
                    scales.append(np.asarray(tail["scale_per_band"], dtype=float))
            fallback_scale = np.median(np.stack(scales), axis=0) if scales else None
            s.report["tail_scale_fallback"] = (
                None if fallback_scale is None else [float(v) for v in fallback_scale]
            )
            s.report["points_tail_only"] = 0
            jobs = []
            for i in without:
                fallback = None
                if fallback_scale is not None and bool(np.all(np.isfinite(source_position))):
                    straight = float(np.linalg.norm(positions[i] - source_position))
                    fallback = (fallback_scale, straight / s.sound_speed_m_s, onsets.get(i))
                    s.report["points_tail_only"] += 1
                jobs.append(submit(pool, i, fallback))
            for job in jobs:
                index, energy, _ = job.result()
                energies[index] = energy
        return energies

    direct_energy = s.stage("render", render_all)
    direct_set = set(with_direct)

    criteria_settings = settings.criteria
    alignment = s.stage(
        "align",
        # On the points with a direct path only: elsewhere the first arrival is
        # a diffracted or reflected one, whose level says nothing of the scale.
        lambda: align_to_reference(
            s.reference,
            {i: e for i, e in direct_energy.items() if i in direct_set},
            sound_speed_m_s=s.sound_speed_m_s,
        ),
    )
    s.report["alignment"] = alignment.record()
    lead = int(round(alignment.lead_s * rate))
    aligned = _Cached(scratch, list(range(len(every))), rate, order, lead)
    field_path = s.stage(
        "field",
        lambda: write_mirror_field(
            s.run / f"field_mirror{s.suffix}" / f"{name}.h5",
            s.reference,
            aligned,
            provenance={
                "scene_key": scene.key,
                "settings": settings.record(),
                "alignment": alignment.record(),
                "tree": s.report["tree"],
            },
            gain=alignment.gain,
        ),
    )

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
                        s.reference,
                        scratch / f"{i}.npy",
                        lead,
                        alignment.gain,
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

    metrics = s.stage("judge", judge_all)
    shutil.rmtree(scratch, ignore_errors=True)
    metrics_dir = s.mirror_dir / f"metrics{s.suffix}"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(metrics, indent=1, default=str))
    s.report["summary"] = metrics["summary"]

    walk = s.run / "walk.json"
    if walk.is_file():
        manifest = json.loads(walk.read_text())
        for entry in manifest.get("sources", []):
            if str(entry.get("id")) == name or str(entry.get("name")) == name:
                entry[f"field_mirror{s.suffix}"] = str(field_path.relative_to(s.run))
                entry[f"metrics{s.suffix}"] = str((metrics_dir / f"{name}.json").relative_to(s.run))
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
    return s.report


def run_mirror(
    run: Path,
    *,
    models: Path,
    source: dict[str, Any],
    settings: MirrorSettings | None = None,
    sound_speed_m_s: float = 343.2,
    devices: list[int] | None = None,
    phase: str = "all",
    tag: str = "",
    say: Any = print,
) -> dict[str, Any]:
    """The stage for one source of one run, or one of its phases; returns the report written.

    ``tag`` names a second mirror beside the first: its card outputs are
    ``mirror/paths_<S>_<tag>.npz``, ``histogram_<S>_<tag>.npz`` and
    ``card_<S>_<tag>.json``, its field goes to
    ``field_mirror_<tag>``, its metrics to ``mirror/metrics_<tag>``, its report
    to ``mirror/report_<S>_<tag>.json`` and ``walk.json`` gains the keys with
    the same suffix, so the app can offer A, B and C at one cell.
    """
    if phase not in PHASES:
        raise ValueError(f"phase {phase!r} is not one of {PHASES}")
    settings = settings or MirrorSettings()
    s = _Stage(Path(run), str(source["name"]), settings, sound_speed_m_s, say, tag=tag)
    started = time.time()
    if phase in ("card", "all"):
        _card_phase(s, models=Path(models), source=source, devices=devices)
    if phase in ("host", "all"):
        _host_phase(s)
    s.report["phase"] = phase
    s.report["total_s"] = round(time.time() - started, 1)
    target = s.mirror_dir / (
        f"report_{s.name}{s.suffix}.json" if phase != "card" else f"card_{s.name}{s.suffix}.json"
    )
    target.write_text(json.dumps(s.report, indent=1, default=str))
    say(f"mirror {s.name} ({phase}): done in {s.report['total_s'] / 60:.1f} min")
    return s.report
