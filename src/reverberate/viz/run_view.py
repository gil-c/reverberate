"""Show and play what entered and left the wave solver, for one run.

This module is the *data* half of the acoustic view. It reduces a rendered run
to one JSON payload plus its audio, and the apartment browser renders it as the
third mode alongside colour and label. There is one application and one
apartment selector; a run is simply another way of looking at the apartment it
was simulated in, reached from the same place as every other view, and the
camera does not move when the mode changes.

The geometry described here is **the serialised surface list the solver read**,
``model_json``, and nothing else. Not the authored HSSD instances, not a
prettier collider, not a re-derived mesh. That is roadmap constraint 9: the
viewer and the solver read the same file, so a picture and a response can be
proved to be of the same object by comparing one digest. The acoustic mode that
stood here before coloured authored instances by a ``pyroomacoustics``
absorption palette, which describes a ray solver this project no longer uses; it
has been removed rather than kept beside its replacement, because two modes
called acoustic showing different geometries is worse than one.

Everything the page needs is computed here, at build time, and embedded as
plain JSON: the envelopes, the energy decay curves, the measures. The browser
draws and plays; it does no signal processing. That keeps the analysis in the
tested Python path rather than duplicated in JavaScript where nothing checks it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverberate import metrics

__all__ = [
    "SPECTROGRAM_BINS",
    "SPECTROGRAM_FRAMES",
    "RunRef",
    "absorption_colour",
    "RunView",
    "build_site",
    "decay_curve_points",
    "discover_runs",
    "envelope",
    "model_materials",
    "run_scene",
    "spectrogram",
    "surface_groups",
]

#: How many points an envelope or decay curve is reduced to before it is
#: embedded. A 72 kHz response is 108,000 samples per receiver and twelve of
#: them would be a 10 MB page for a plot a few hundred pixels wide. 512 is
#: comfortably more than the pixels available and small enough to inline.
PLOT_POINTS = 512

#: Spectrogram size. Chosen so one image is 32 KB before base64 and twelve of
#: them add about half a megabyte to the page, which is the most that is worth
#: paying to avoid doing an FFT in the browser.
SPECTROGRAM_BINS = 128
SPECTROGRAM_FRAMES = 256

#: Decibels below the loudest bin that the spectrogram floor sits at. Wider than
#: this and the late tail is a black rectangle; narrower and the direct sound is
#: the only thing with any contrast.
SPECTROGRAM_RANGE_DB = 70.0


def envelope(signal: np.ndarray, points: int = PLOT_POINTS) -> list[list[float]]:
    """Min and max per bucket, so a waveform keeps its shape when reduced.

    Decimating by sampling would alias a 72 kHz response into whatever the
    stride happened to hit, and the result would look quieter than it is.
    Taking both extremes of each bucket preserves the outline the eye reads as
    the waveform.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size == 0:
        return []
    points = max(1, min(int(points), signal.size))
    edges = np.linspace(0, signal.size, points + 1, dtype=int)
    return [
        [float(np.min(chunk)), float(np.max(chunk))]
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
        if (chunk := signal[start : max(stop, start + 1)]).size
    ]


def decay_curve_points(
    signal: np.ndarray, sample_rate_hz: float, points: int = PLOT_POINTS
) -> dict[str, list[float]]:
    """Broadband Schroeder decay, in decibels against seconds.

    Reduced by taking the value at evenly spaced samples rather than an
    average: the curve is already monotonic and smooth, so a point on it is
    representative in a way a point on a raw waveform is not.
    """
    curve = metrics.energy_decay_curve(np.asarray(signal, dtype=float).ravel())
    if curve.size == 0:
        return {"seconds": [], "db": []}
    index = np.linspace(0, curve.size - 1, min(points, curve.size), dtype=int)
    values = curve[index]
    # -inf is the tail after the integration reaches zero energy. It plots as a
    # break in the line rather than a value, so it is clamped to the floor the
    # axis shows instead of being silently dropped.
    values = np.where(np.isfinite(values), values, -120.0)
    return {
        "seconds": [round(float(value), 6) for value in index / float(sample_rate_hz)],
        "db": [round(float(value), 3) for value in values],
    }


def spectrogram(
    signal: np.ndarray,
    sample_rate_hz: float,
    bins: int = SPECTROGRAM_BINS,
    frames: int = SPECTROGRAM_FRAMES,
    max_hz: float | None = None,
) -> dict[str, Any]:
    """A short time Fourier magnitude, in decibels, quantised to bytes.

    Returned as base64 bytes rather than numbers because a float per pixel
    would be six times the size for a picture that is only ever going to be
    drawn into a canvas. Level 0 is the floor at ``SPECTROGRAM_RANGE_DB`` below
    the loudest bin and level 255 is that loudest bin, so the scale is relative
    to this response and two responses are not comparable by eye. That is
    stated on the page rather than left for the reader to assume either way.

    ``max_hz`` bounds the vertical axis. Left unset it is the Nyquist rate,
    which for a 48 kHz delivery file is 24 kHz: since the solver only ran to
    4 kHz that would spend five sixths of the picture drawing the empty band
    above the low pass. Setting it a little above the simulated ``fmax`` keeps
    the roll off visible, which is worth seeing, without the dead space.

    Rows run low frequency first, which is the opposite of how it is drawn; the
    page flips it so that low frequencies are at the bottom where a reader
    expects them.
    """
    import base64

    signal = np.asarray(signal, dtype=float).ravel()
    nyquist = float(sample_rate_hz) / 2.0
    ceiling = min(float(max_hz), nyquist) if max_hz else nyquist
    # Enough FFT rows that `bins` of them cover the requested ceiling.
    rows = max(int(np.ceil(bins * nyquist / ceiling)), bins)
    window_length = 2 * rows
    if signal.size < window_length:
        signal = np.pad(signal, (0, window_length - signal.size))

    starts = np.linspace(0, signal.size - window_length, frames, dtype=int)
    window = np.hanning(window_length)
    columns = np.abs(np.fft.rfft(signal[starts[:, None] + np.arange(window_length)] * window))
    magnitude = columns[:, :bins].T

    peak = float(np.max(magnitude))
    if peak <= 0:
        data = np.zeros((bins, frames), dtype=np.uint8)
    else:
        with np.errstate(divide="ignore"):
            decibels = 20.0 * np.log10(np.maximum(magnitude, 1e-12) / peak)
        scaled = (decibels + SPECTROGRAM_RANGE_DB) / SPECTROGRAM_RANGE_DB
        data = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)

    return {
        "bins": bins,
        "frames": frames,
        "max_hz": round(bins * nyquist / rows, 1),
        "seconds": round(signal.size / float(sample_rate_hz), 4),
        "range_db": SPECTROGRAM_RANGE_DB,
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


# The octave centres of the material table, from experiments.scene_export.BANDS.
# Repeated rather than imported because that module pulls in the mesh stack, and
# a test pins the two lists together so the copy cannot drift.
MATERIAL_BANDS_HZ = (16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
COLOUR_BAND_HZ = 1000.0

# Reflective through to absorbent. Grey is deliberately not on the ramp: the
# exporter writes grey for every group, so a grey surface on screen means the
# absorption was not found rather than that it was found to be middling.
# Stopped often enough that no leg of it passes through grey, which a straight
# red to blue interpolation does at exactly the absorptions a soft furnishing
# has.
_ABSORPTION_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (170, 52, 44)),
    (0.2, (214, 140, 56)),
    (0.45, (222, 200, 74)),
    (0.7, (110, 180, 92)),
    (1.0, (56, 120, 200)),
)
UNKNOWN_COLOUR = (128, 128, 128)


def absorption_colour(alpha: float) -> list[int]:
    """Position an absorption coefficient on the reflective to absorbent ramp."""
    alpha = min(max(float(alpha), 0.0), 1.0)
    for (lo, low), (hi, high) in zip(_ABSORPTION_RAMP, _ABSORPTION_RAMP[1:], strict=False):
        if alpha <= hi:
            t = (alpha - lo) / (hi - lo)
            return [round(a + (b - a) * t) for a, b in zip(low, high, strict=True)]
    return list(_ABSORPTION_RAMP[-1][1])


def surface_groups(
    model: dict[str, Any], materials: Mapping[str, Sequence[float]] | None = None
) -> list[dict[str, Any]]:
    """The solver's own surface list, one entry per material group.

    ``model_json`` stores each group as flat triangle indices into its own
    point list, which is what PFFDTD reads. It is handed to the browser in that
    same shape so nothing is re-derived on the way.

    Its ``color`` field is not passed through. The exporter writes ``[128, 128,
    128]`` into every group because PFFDTD ignores it, so honouring it renders
    thirteen materials as one flat grey. The colour here is the material's own
    absorption at 1 kHz, which is the quantity the view exists to show.
    """
    band = MATERIAL_BANDS_HZ.index(COLOUR_BAND_HZ)
    materials = materials or {}
    groups = []
    for label, group in sorted(model.get("mats_hash", {}).items()):
        points = np.asarray(group["pts"], dtype=float)
        triangles = np.asarray(group["tris"], dtype=int)
        coefficients = materials.get(label)
        alpha = float(coefficients[band]) if coefficients is not None else None
        groups.append(
            {
                "label": label,
                "positions": [round(float(value), 5) for value in points.ravel()],
                "indices": [int(value) for value in triangles.ravel()],
                "colour": absorption_colour(alpha) if alpha is not None else list(UNKNOWN_COLOUR),
                "absorption": None if alpha is None else round(alpha, 3),
                "triangles": int(triangles.shape[0]) if triangles.ndim > 1 else 0,
                "sides": group.get("sides"),
            }
        )
    return groups


def model_materials(model_json: Path) -> dict[str, list[float]]:
    """The absorption table exported beside the model, or nothing if absent.

    The model itself carries only labels. The manifest written next to it by the
    export carries the coefficients those labels stand for, which is the only
    place the two are tied together.
    """
    manifest = Path(model_json).parent / "manifest.json"
    if not manifest.is_file():
        return {}
    materials = json.loads(manifest.read_text()).get("materials", {})
    return {str(k): [float(v) for v in values] for k, values in materials.items()}


@dataclass(frozen=True)
class RunView:
    """What was built, so the caller can report it rather than guess."""

    groups: int
    triangles: int
    samples: int
    audio_files: int
    path: Path

    def summary(self) -> str:
        return (
            f"{self.groups} surface groups, {self.triangles} triangles, "
            f"{self.samples} samples, {self.audio_files} audio files, at {self.path}"
        )


def _sample_rows(
    run_dir: Path, report: dict[str, Any], audio_names: set[str]
) -> list[dict[str, Any]]:
    """One row per source and receiver pair, with its plots and its audio."""
    from reverberate.response import read_raw

    rows: list[dict[str, Any]] = []
    for entry in report["sources"]:
        index = int(entry["source_index"])
        response = read_raw(run_dir / "responses" / f"source{index}.h5")
        rate = float(response.sample_rate_hz)
        # A little above the simulated band, so the low pass edge is visible
        # and the empty decades above it are not drawn.
        ceiling = float(response.provenance.fmax_hz) * 1.5
        measures = {int(row["receiver"]): row for row in entry["measures"]}
        for receiver in range(response.ir.shape[0]):
            signal = response.ir[receiver]
            name = f"source{index}_receiver{receiver}_wet.wav"
            rows.append(
                {
                    "id": f"s{index}r{receiver}",
                    "label": f"source {index} to receiver {receiver}",
                    "source_index": index,
                    "receiver_index": receiver,
                    "sample_rate_hz": rate,
                    "seconds": round(signal.size / rate, 4),
                    "peak": round(float(np.max(np.abs(signal))), 6),
                    "envelope": envelope(signal),
                    "decay": decay_curve_points(signal, rate),
                    "spectrogram": spectrogram(signal, rate, max_hz=ceiling),
                    "measures": measures.get(receiver),
                    "wet_audio": f"audio/{name}" if name in audio_names else None,
                }
            )
    return rows


@dataclass(frozen=True)
class RunRef:
    """Where a rendered run is, and which apartment it belongs to."""

    name: str
    scene_id: str
    room: str
    path: Path


def run_scene(run_dir: Path) -> RunRef:
    """Which apartment and room a run was simulated in.

    Read from ``plan.json`` rather than inferred from the mesh, because the plan
    is what chose the room; the mesh is only its consequence.
    """
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    return RunRef(
        name=run_dir.name,
        scene_id=str(plan["scene_id"]),
        room=str(plan["room"]),
        path=run_dir,
    )


def discover_runs(runs_root: Path) -> list[RunRef]:
    """Every rendered run under ``runs_root``, newest name last.

    A run counts as rendered only if it has both the plan that names the scene
    and the report the payload is built from. A half-finished run directory is
    skipped rather than offered and then failing to open.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    # Pointing at a single run directory is the easy mistake, and globbing one
    # level down would answer it with an empty selector and no explanation.
    if (runs_root / "plan.json").is_file():
        runs_root = runs_root.parent
    found: list[RunRef] = []
    for plan in sorted(runs_root.glob("*/plan.json")):
        if not (plan.parent / "report.json").is_file():
            continue
        try:
            found.append(run_scene(plan.parent))
        except (KeyError, json.JSONDecodeError):
            continue
    return found


def build_site(run_dir: Path, target: Path) -> RunView:
    """Write one run's payload and audio into ``target``.

    Reads only what the run already published: ``report.json`` for the room and
    the measures, ``responses/`` for the signals, ``audio/`` for the playback.
    It does not re-run the solver, re-measure, or reach the network, so the page
    is exactly the artefacts that were published and not a second opinion.
    """
    run_dir, target = Path(run_dir), Path(target)
    report = json.loads((run_dir / "report.json").read_text())
    model = json.loads(Path(str(report["model_json"])).read_text())

    target.mkdir(parents=True, exist_ok=True)
    audio_names: set[str] = set()
    source_audio = run_dir / "audio"
    if source_audio.is_dir():
        shutil.copytree(source_audio, target / "audio", dirs_exist_ok=True)
        audio_names = {path.name for path in source_audio.glob("*.wav")}

    groups = surface_groups(model, model_materials(Path(str(report["model_json"]))))
    placement = report["placement"]
    scene = run_scene(run_dir)
    payload = {
        "run": report["run"],
        "scene_id": scene.scene_id,
        "room_name": scene.room,
        "scene_sha256": report["scene_sha256"],
        "cache_key": report["cache_key"],
        "room": report["room"],
        "theory": report["theory"],
        "theory_shell_only": report.get("theory_shell_only"),
        "theory_note": report.get("theory_note"),
        "measured_anomaly": report.get("measured_anomaly"),
        # What the solver sealed off. Drawn, not merely recorded: sealing stops
        # the simulation carrying sound through a region, and the whole reason
        # the census exists is that this must be visible rather than inferred.
        "sealed": report.get("sealed"),
        "band_note": report.get("band_note"),
        "low_cut_hz": report.get("low_cut_hz"),
        "binaural_note": report["binaural_note"],
        "omissions": report.get("omissions", []),
        "dry_voice": report["dry_voice"],
        "dry_audio": "audio/dry_voice.wav" if "dry_voice.wav" in audio_names else None,
        "groups": groups,
        "sources": [
            {"index": index, "position": entry["position"], "archetype": entry.get("archetype")}
            for index, entry in enumerate(placement["sources"])
        ],
        "receivers": [
            {"index": index, "position": entry["position"]}
            for index, entry in enumerate(placement["receivers"])
        ],
        "samples": _sample_rows(run_dir, report, audio_names),
    }
    (target / "run.json").write_text(json.dumps(payload) + "\n")
    return RunView(
        groups=len(groups),
        triangles=sum(int(group["triangles"]) for group in groups),
        samples=len(payload["samples"]),
        audio_files=len(audio_names),
        path=target,
    )
