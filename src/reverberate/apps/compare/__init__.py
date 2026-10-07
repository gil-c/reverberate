"""Compare two to N renders of the same short scene: by ear, by eye, and blind.

``python -m reverberate.apps.compare FOLDER`` opens the folder a comparison
wrote (``python -m reverberate.render check REF.h5 --against V.h5 ...``, see
``docs/formats/scene-signal.md``): ``variants.json`` or ``ab.json``, the
files of two ears, and with ``--ambisonic`` the order 7 stems, which are
decoded here under the head the listener turns.

``python -m reverberate.apps.compare --packs REF.h5 V.h5 --window 242 262
--out FOLDER`` writes that folder first, order 7 and the three signals, and
opens it.

The screen holds the variants as buttons, the transport and the head, the
sonograms, what the kit measured between the variants by third octave, and
a blind test. What the listener decides is written beside the folder, in
``<FOLDER>_listening/``: a file a blind test.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reverberate.viz.parts.media import Kit, kit_folder
from reverberate.viz.parts.routes import Media
from reverberate.viz.parts.server import AppServer, mount_decoders

__all__ = ["STATIC", "build", "differences", "results_folder"]

STATIC = Path(__file__).parent / "static"
PARTS = ("early", "late", "clips")


def results_folder(folder: Path) -> Path:
    """Where what the listener decides is written: beside the folder, never in it."""
    folder = Path(folder).resolve()
    return folder.parent / f"{folder.name}_listening"


def differences(document: Mapping[str, Any]) -> dict[str, Any]:
    """What the kit measured, the same shape for two packs and for several.

    ``by_variant[variant][source]`` holds ``early``, ``late`` and ``clips``,
    each the variant less the reference in dB a third octave (``None`` for a
    band that holds nothing), and ``worst``, the largest of each either side
    of the crossover.
    """
    if "variants" in document:
        reference = str(document["reference"]["name"])
        found = {
            str(name): dict(record.get("less_the_reference_db") or {})
            for name, record in document["variants"].items()
        }
    else:
        reference, found = "A", {"B": dict(document.get("difference_b_less_a_db") or {})}
    by_variant: dict[str, dict[str, Any]] = {}
    for variant, sources in found.items():
        by_variant[variant] = {}
        for source, record in sources.items():
            impulse = dict(record.get("impulse") or {})
            by_variant[variant][source] = {
                "early": impulse.get("early_db"),
                "late": impulse.get("late_db"),
                "clips": record.get("clips_db"),
                "clips_level_db": record.get("clips_level_db"),
                "impulse_at_s": impulse.get("time_s"),
                "worst": record.get("worst_abs_db") or {},
            }
    return {
        "reference": reference,
        "third_octaves_hz": document.get("third_octaves_hz") or [],
        "crossover_hz": document.get("crossover_hz"),
        "early_s": document.get("early_s"),
        "by_variant": by_variant,
    }


def build(
    folder: Path,
    decoders: Path,
    *,
    results: Path | None = None,
    measured_head: Path | None = None,
) -> tuple[AppServer, Kit, Media]:
    """The application on ``folder``; ``decoders`` is a scratch folder for the head's filters."""
    kit = kit_folder(folder)
    document = kit.document
    media = Media(
        kit.library,
        results or results_folder(folder),
        context={
            "application": "reverberate.apps.compare",
            "folder": str(Path(folder).resolve()),
            "recipe_sha256": document.get("recipe_sha256"),
            "window_s": document.get("window_s"),
        },
    )
    server = AppServer("compare", STATIC)
    media.mount(server)
    told = {
        **kit.describe(),
        "differences": differences(document),
        "results": str(media.results),
    }
    server.route("GET", "api/kit", lambda request: told)
    heads = mount_decoders(server, decoders, measured_head)
    told["head_name"] = heads[0].get("head") if heads else None
    return server, kit, media
