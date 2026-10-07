"""Play a scene with its sources in sight and a fader on each: a first cut.

``python -m reverberate.apps.scene_player FOLDER`` opens a folder of the
listening kit that holds order 7 stems (``python -m reverberate.render check
REF.h5 --against V.h5 --ambisonic --window ...``) and the pack it was
rendered from, which the folder names. The screen holds the scene (the
floor, the listener, each voice as a head that lights while it speaks, each
noise as a cube that lights likewise), the transport and the head, and a
track list: a solo, a mute and a fader a source, and the source's level over
the window. The balance set is written beside the folder, in
``<FOLDER>_listening/balance.json``.

What it is not yet: it plays a window the kit rendered, not a whole scene
streamed from a pack as it is rendered, which the inspector does
(:mod:`reverberate.viz.audit_stems`). ``docs/apps.md`` lists what remains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reverberate.apps.compare import results_folder
from reverberate.render.pack import read_pack
from reverberate.viz.parts.media import Kit, kit_folder
from reverberate.viz.parts.routes import Media
from reverberate.viz.parts.scene import scene_view
from reverberate.viz.parts.server import AppServer, HttpError, mount_decoders

__all__ = ["STATIC", "build", "pack_of"]

STATIC = Path(__file__).parent / "static"


def pack_of(kit: Kit, variant: str) -> Path | None:
    """The pack a folder says ``variant`` was rendered from, where it is still there."""
    document = kit.document
    if "variants" not in document:
        named = dict(document.get("a" if variant == "A" else "b") or {}).get("pack")
    elif variant == document["reference"]["name"]:
        named = document["reference"].get("pack")
    else:
        named = dict(document["variants"].get(variant) or {}).get("pack")
    return Path(named) if named and Path(named).is_file() else None


def build(
    folder: Path,
    decoders: Path,
    *,
    pack: Path | None = None,
    results: Path | None = None,
    measured_head: Path | None = None,
) -> tuple[AppServer, Kit, Media]:
    """The application on ``folder``; ``pack`` is the scene's, when the one it names is gone."""
    kit = kit_folder(folder)
    tracks = kit.sets.get("clips") or next(iter(kit.sets.values()))
    mixes = tracks.get("mix") or next(iter(tracks.values()))
    media = Media(
        kit.library,
        results or results_folder(folder),
        context={
            "application": "reverberate.apps.scene_player",
            "folder": str(Path(folder).resolve()),
            "recipe_sha256": kit.document.get("recipe_sha256"),
            "window_s": kit.document.get("window_s"),
        },
    )
    server = AppServer("scene player", STATIC)
    media.mount(server)
    window = kit.document.get("window_s")
    path = pack or pack_of(kit, kit.variants[0])
    scene: dict[str, Any] | None = None
    if path is not None:
        with read_pack(Path(path)) as held:
            scene = scene_view(
                held, None if window is None else (window[0], window[1]), only=media.sources
            )

    def drawn(request: Any) -> dict[str, Any]:
        if scene is None:
            raise HttpError(404, "the pack this folder was rendered from is not there: give --pack")
        return scene

    told = {
        "folder": str(kit.folder),
        "variants": list(kit.variants),
        "items": {variant: kit.library.item(mixes[variant]).describe() for variant in kit.variants},
        "head": kit.head,
        "window_s": window,
        "pack": None if path is None else str(path),
        "results": str(media.results),
        "notes": list(kit.notes),
    }
    server.route("GET", "api/player", lambda request: told)
    server.route("GET", "api/scene", drawn)
    mount_decoders(server, decoders, measured_head)
    return server, kit, media
