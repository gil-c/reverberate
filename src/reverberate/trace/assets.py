"""What of the dwelling a trace reads besides the recipe: the mirror's scene and its clock.

The band above the crossover is computed on the derived scene of
:mod:`reverberate.mirror.geometry` with a calibration's materials. A static
field of the mirror is put on the wave field's clock and scale by two
numbers and a filter the mirror measured against a field of the dwelling
(:func:`reverberate.mirror.files.align_to_reference`,
:func:`reverberate.mirror.direct.measure_signature`). :class:`MirrorAssets`
is those four things as one directory of a bundle.

**A pack takes the clock and nothing else** (:class:`Normalisation`,
2026-10-05). A pack is physical: the wave band is the solve over
:data:`reverberate.spatial.lowband.FIELD_UNIT_AT_1M`, whose direct sound is
``1 / d`` within 0.1 dB, and the mirror renders a unit source as ``1 / d``
before anything is measured (-0.06 dB). The measured gain and the measured
signature put it 2.4 dB under that and took 1.8 to 5 dB of its treble
(``docs/open-questions/chain-audit.md``, D1 and D2): the gain is a ratio of
energies inside 0.5 ms round two pulses that are not the same pulse, and
the signature is the band limit of the grid the reference was solved on.
So the normalisation between the two bands is made once for all: in a pack
the mirror's gain is one and its signature a unit pulse, whatever the
dwelling. What was measured is still read from the bundle, for the lead
and for the static fields, and :data:`ALIGNED` gives a pack every number as
it was.

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

from reverberate.mirror.direct import unit_signature
from reverberate.mirror.directivity import NORMALISED, Directivity, omni, voice_v1
from reverberate.mirror.geometry import DerivedScene, load_derived, write_derived
from reverberate.mirror.hybrid import Crossover
from reverberate.mirror.ism import IsmSettings
from reverberate.mirror.parameters import Parameters, load_parameters
from reverberate.mirror.pipeline import MirrorSettings
from reverberate.mirror.rays import RaySettings
from reverberate.mirror.render import RenderSettings
from reverberate.scenes import Recipe
from reverberate.spatial.lowband import FIELD_UNIT_AT_1M

__all__ = [
    "ALIGNED",
    "PHYSICAL",
    "ROOMS_RULE",
    "MirrorAssets",
    "Normalisation",
    "directivity_models",
    "found_assets",
    "mismatched",
]

#: The rule that named the rooms a recipe's stations stand in.
ROOMS_RULE = "adr-0010"
#: The rate the mirror's field was written at: its lead is a whole number of these samples.
FIELD_RATE_HZ = 48000.0


def directivity_models(normalised: str = NORMALISED) -> dict[str, Directivity]:
    """Every model a source may name, its table level with its axis unless told."""
    return {"omni": omni(normalised), "voice_v1": voice_v1(normalised)}


@dataclass(frozen=True)
class Normalisation:
    """How a pack's two bands are put on one scale: every choice, each with its old value.

    ``alignment``: ``"physical"``, the mirror's gain is one, or
    ``"measured"``, the alignment's gain over the field's unit (D1).
    ``signature``: ``"unit"``, a unit pulse; ``"colour"``, the measured
    signature at 0 dB over the crossover's octave, which keeps the treble
    every render before had; ``"measured"``, as it was measured, which goes
    with the measured gain and with no other (D2). ``seam``: ``"unbiased"``,
    a pair's seam read where both bands are whole and with the solve's own
    band limit undone, or ``"octave"``, over the crossover's octave as the
    wave side stands (D7). ``directivity``: what the tables are level with,
    ``"axis"`` or ``"mean"`` (D3). ``constant``: ``"fixed"``, the level of
    the bands above the crossover is the code's one number
    (:data:`reverberate.render.seam.SEAM_CONSTANT_DB`), or ``"median"``,
    the median of the pack's own seams.
    """

    alignment: str = "physical"
    signature: str = "unit"
    seam: str = "unbiased"
    directivity: str = NORMALISED
    constant: str = "fixed"

    def __post_init__(self) -> None:
        allowed = {
            "alignment": ("physical", "measured"),
            "signature": ("unit", "colour", "measured"),
            "seam": ("unbiased", "octave"),
            "directivity": ("axis", "mean"),
            "constant": ("fixed", "median"),
        }
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"normalisation.{name} is one of {values}")
        if (self.signature == "measured") != (self.alignment == "measured"):
            raise ValueError(
                "the measured signature goes with the measured gain: the gain was fitted "
                "with it, and neither is a level without the other"
            )

    @classmethod
    def of(cls, told: Any) -> Normalisation:
        """From a bundle's word: nothing or ``"physical"``, ``"aligned"``, or the choices."""
        if told is None or told == "physical":
            return PHYSICAL
        if told == "aligned":
            return ALIGNED
        if isinstance(told, dict):
            return cls(**{str(name): str(value) for name, value in told.items()})
        raise ValueError(f"a normalisation is 'physical', 'aligned' or its choices, not {told!r}")

    def record(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "signature": self.signature,
            "seam": self.seam,
            "directivity": self.directivity,
            "constant": self.constant,
        }


#: A pack as it is born: on the physical scale, the same for every dwelling.
PHYSICAL = Normalisation()
#: A pack as every pack before 2026-10-05: the one switch back, for comparison.
ALIGNED = Normalisation(
    alignment="measured", signature="measured", seam="octave", directivity="mean", constant="median"
)


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
    #: How a pack made from these is put on one scale. Not kept in the bundle's
    #: directory: the bundle's trace says it (``trace.normalisation``).
    normalisation: Normalisation = PHYSICAL

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
        """The mirror's gain in a pack, which is physical: one.

        The mirror renders a unit source as ``1 / d`` with no gain at all.
        Under the measured alignment, the alignment's gain over the field's
        unit, 2.4 dB under one with its signature (``chain-audit.md``, D1).
        """
        if self.normalisation.alignment == "physical":
            return 1.0
        return self.gain / FIELD_UNIT_AT_1M

    @property
    def field_gain(self) -> float:
        """The pack's gain on the field's scale: what the mirror is read against a pair with."""
        if self.normalisation.alignment == "physical":
            return FIELD_UNIT_AT_1M
        return self.gain

    def pack_signature(self, crossover: Crossover | None = None) -> np.ndarray:
        """The signature a pack carries: a unit pulse, which leaves the mirror's treble alone.

        ``"colour"`` keeps the measured one at 0 dB over the octave of
        ``crossover``; ``"measured"`` is the measured one as it is.
        """
        chosen = self.normalisation.signature
        if chosen == "unit":
            return np.ones(1)
        if chosen == "colour":
            return unit_signature(
                self.signature, (crossover or Crossover()).band_hz(), FIELD_RATE_HZ
            )
        return np.asarray(self.signature, dtype=float)

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
