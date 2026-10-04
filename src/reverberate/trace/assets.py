"""What of the dwelling a trace reads besides the recipe: the mirror's scene and its clock.

The band above the crossover is computed on the derived scene of
:mod:`reverberate.mirror.geometry` with a calibration's materials, and is
put on the wave field's clock and scale by two numbers and a filter the
mirror measured against a field of the dwelling
(:func:`reverberate.mirror.files.align_to_reference`,
:func:`reverberate.mirror.direct.measure_signature`). :class:`MirrorAssets`
is those four things as one directory of a bundle.

:func:`found_assets` is the recipe's ``assets`` block as a trace finds it and
:func:`mismatched` the keys on which a recipe disagrees: a trace refuses
such a recipe by key name (``docs/formats/scene-recipe.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from reverberate.mirror.directivity import Directivity, omni, voice_v1
from reverberate.mirror.geometry import DerivedScene, load_derived, write_derived
from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.parameters import Parameters, load_parameters
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.render import RenderSettings
from reverberate.scenes import Recipe
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M

__all__ = ["ROOMS_RULE", "MirrorAssets", "directivity_models", "found_assets", "mismatched"]

#: The rule that named the rooms a recipe's stations stand in.
ROOMS_RULE = "adr-0010"
#: The rate the mirror's field was written at: its lead is a whole number of these samples.
FIELD_RATE_HZ = 48000.0


def directivity_models() -> dict[str, Directivity]:
    """Every model a source may name."""
    return {"omni": omni(), "voice_v1": voice_v1()}


@dataclass(frozen=True)
class MirrorAssets:
    """The mirror of one dwelling: its scene, its settings, and the wave field's clock and scale."""

    catalogue: DerivedScene
    settings: MirrorSettings
    #: The source's minimum phase signature, as ``mirror.direct.measure_signature`` gives it.
    signature: np.ndarray
    #: ``mirror.files.Alignment``: the wave field's clock and scale.
    lead_s: float
    gain: float
    #: The digest of the storey's export the scene was derived from; empty when not kept.
    export_sha256: str = ""

    @property
    def lead_samples(self) -> int:
        """The lead as the mirror's field took it: a whole number of samples at 48 kHz."""
        return int(round(self.lead_s * FIELD_RATE_HZ))

    @property
    def pack_lead_s(self) -> float:
        """The lead a pack carries: the one the mirror's field was shifted by, not the measured."""
        return self.lead_samples / FIELD_RATE_HZ

    @property
    def pack_gain(self) -> float:
        """The mirror's gain in a pack, which is physical: the alignment's over the field's unit."""
        return self.gain / FIELD_UNIT_AT_1M

    @property
    def triangles(self) -> np.ndarray:
        """The occluders, ``[triangle, 3, 3]``: what a cell's clearance is measured to."""
        return np.asarray(self.catalogue.occluder_vertices, dtype=float).reshape(-1, 3, 3)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_derived(self.catalogue, directory / "scene")
        (directory / "calibration.json").write_text(
            json.dumps({"parameters": self.settings.parameters.record()}, indent=1)
        )
        np.save(directory / "signature.npy", np.asarray(self.signature, dtype=float))
        (directory / "alignment.json").write_text(
            json.dumps(
                {
                    "lead_s": self.lead_s,
                    "gain": self.gain,
                    "export_sha256": self.export_sha256,
                    "ism": self.settings.ism.record(),
                    "rays": self.settings.rays.record(),
                    "render": self.settings.render.record(),
                    "sound_speed_m_s": self.settings.sound_speed_m_s,
                    "seed": self.settings.seed,
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> MirrorAssets:
        """The inverse of :meth:`save`."""
        directory = Path(directory)
        record = json.loads((directory / "alignment.json").read_text())
        catalogue = load_derived(directory / "scene")
        settings = MirrorSettings(
            rules=catalogue.rules,
            ism=IsmSettings(**record["ism"]),
            rays=RaySettings(**record["rays"]),
            render=RenderSettings(**record["render"]),
            parameters=load_parameters(directory / "calibration.json"),
            sound_speed_m_s=float(record["sound_speed_m_s"]),
            seed=int(record["seed"]),
        )
        return cls(
            catalogue=catalogue,
            settings=settings,
            signature=np.load(directory / "signature.npy"),
            lead_s=float(record["lead_s"]),
            gain=float(record["gain"]),
            export_sha256=str(record.get("export_sha256", "")),
        )

    @classmethod
    def from_run(
        cls,
        mirror: Path,
        *,
        source: str = "S1",
        calibration: Path | None = None,
        lead_s: float | None = None,
        gain: float | None = None,
    ) -> MirrorAssets:
        """From a run's ``mirror`` directory, as :func:`reverberate.mirror.pipeline.run` leaves it.

        The scene is ``scene.json``; the signature ``signature_<source>.npy``;
        the alignment that of ``report_<source>.json`` unless ``lead_s`` and
        ``gain`` are given, which a field written under another calibration
        needs (its ``provenance_json`` holds the gain it was written with).
        ``calibration`` defaults to the catalogue's values.
        """
        mirror = Path(mirror)
        catalogue = load_derived(mirror / "scene")
        parameters = load_parameters(calibration) if calibration is not None else Parameters()
        settings = MirrorSettings(rules=catalogue.rules, parameters=parameters).pinned()
        if lead_s is None or gain is None:
            report = json.loads((mirror / f"report_{source}.json").read_text())["alignment"]
            lead_s = float(report["lead_s"]) if lead_s is None else lead_s
            gain = float(report["gain"]) if gain is None else gain
        stamp = mirror / "scene.model"
        return cls(
            catalogue=catalogue,
            settings=settings,
            signature=np.load(mirror / f"signature_{source}.npy"),
            lead_s=float(lead_s),
            gain=float(gain),
            export_sha256=stamp.read_text().strip() if stamp.is_file() else "",
        )

    def with_settings(self, **changes: Any) -> MirrorAssets:
        return replace(self, settings=replace(self.settings, **changes))


def found_assets(
    recipe: Recipe, mirror: MirrorAssets, *, voxel_low_key: str, export_sha256: str = ""
) -> dict[str, Any]:
    """The recipe's ``assets`` block as this trace finds it."""
    models = directivity_models()
    named = sorted({s.directivity.model for s in recipe.sources} - {"omni"})
    return {
        "export_sha256": export_sha256 or mirror.export_sha256,
        "voxel_low_key": voxel_low_key,
        "mirror_scene_key": mirror.catalogue.key,
        "calibration_key": mirror.settings.parameters.key,
        "directivity": {name: models[name].digest for name in named if name in models},
        "rooms_rule": ROOMS_RULE,
    }


def mismatched(recipe: Recipe, found: dict[str, Any]) -> list[str]:
    """The names of the recipe's asset keys that are not what the trace found."""
    held = recipe.assets.to_dict()
    wrong = [
        name
        for name in ("export_sha256", "voxel_low_key", "mirror_scene_key", "calibration_key")
        if str(held.get(name)) != str(found.get(name))
    ]
    if held.get("rooms_rule") != found.get("rooms_rule"):
        wrong.append("rooms_rule")
    wanted = dict(held.get("directivity") or {})
    have = dict(found.get("directivity") or {})
    wrong.extend(
        f"directivity.{name}"
        for name in sorted(set(wanted) | set(have))
        if wanted.get(name) != have.get(name)
    )
    return wrong
