"""The calibration's corrections to the material catalogue, and the settings they were fitted under.

A calibration (:mod:`reverberate.mirror.calibration`) moves these numbers
until the mirror's decay, colour and reflection levels match the wave field:

- ``absorption_scale``, per octave band, on every label's absorption: the
  rays' decay;
- ``scattering_scale`` on every class, and ``shell_scattering`` for the
  room's own walls, floors and ceilings: how the tail mixes directions;
- ``tail_gain_db``, per octave band, on the histogram's tail: its colour;
- ``image_absorption_scale``, per octave band, for the image sources alone:
  the early reflections' levels.

``rendered_with`` holds the ray and render settings the fit ran under; a
render with these parameters uses them, so a calibration means one field.
The record is keyed by its own digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.acoustics import OCTAVE_BANDS
from reverberate.mirror.geometry import DerivedScene, MaterialTable
from reverberate.mirror.ism import Paths, _gains

__all__ = ["Parameters", "apply_parameters", "image_scene", "load_parameters", "regain"]


@dataclass(frozen=True)
class Parameters:
    """The corrections, with the catalogue's values as the origin."""

    absorption_scale: tuple[float, ...] = tuple(1.0 for _ in OCTAVE_BANDS)
    scattering_scale: float = 1.0
    tail_gain_db: tuple[float, ...] = tuple(0.0 for _ in OCTAVE_BANDS)
    note: str = "catalogue values, nothing calibrated"
    image_absorption_scale: tuple[float, ...] | None = None
    shell_scattering: float | None = None
    #: ``{"rays": RaySettings.record(), "render": RenderSettings.record()}``.
    rendered_with: dict[str, Any] | None = None

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
        # The optional fields only when set, so the keys of older files stay theirs.
        if self.shell_scattering is not None:
            record["shell_scattering"] = round(float(self.shell_scattering), 6)
        if self.image_absorption_scale is not None:
            record["image_absorption_scale"] = [
                round(float(v), 6) for v in self.image_absorption_scale
            ]
        if self.rendered_with is not None:
            record["rendered_with"] = self.rendered_with
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Parameters:
        image = record.get("image_absorption_scale")
        shell = record.get("shell_scattering")
        return cls(
            absorption_scale=tuple(float(v) for v in record["absorption_scale"]),
            scattering_scale=float(record["scattering_scale"]),
            tail_gain_db=tuple(float(v) for v in record["tail_gain_db"]),
            note=str(record.get("note", "")),
            image_absorption_scale=None if image is None else tuple(float(v) for v in image),
            shell_scattering=None if shell is None else float(shell),
            rendered_with=record.get("rendered_with"),
        )


def load_parameters(path: Path) -> Parameters:
    """A calibration file (``{"parameters": ...}``) or a bare parameters record."""
    record = json.loads(Path(path).read_text())
    return Parameters.from_record(record.get("parameters", record))


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
    """The scene with the materials the image sources' gains read.

    The shell's own scattering is how the rays mix directions, not a loss of
    a flat wall's specular reflection: the images keep the class's there.
    """
    images = replace(parameters, shell_scattering=None)
    if parameters.image_absorption_scale is None:
        return apply_parameters(scene, images)
    return apply_parameters(
        scene, replace(images, absorption_scale=parameters.image_absorption_scale)
    )


def regain(paths: Paths, scene: DerivedScene) -> Paths:
    """The same paths with the gains of ``scene``'s materials: nothing else moves."""
    return replace(paths, gain=_gains(scene, paths.sequence, paths.length_m))
