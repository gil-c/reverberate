"""Play a whole scene from its pack: everybody in sight, a fader on each.

``python -m reverberate.apps.scene PACK`` opens a scene pack
(``docs/formats/scene-pack.md``) and plays it at order 7 under the head of
its listener, which the page follows unless the listener frees it. The
screen holds:

- the dwelling, its walls and floor, with the listener, each person as a
  head in a colour with a name, and each noise as a marker: each lights
  while it sounds, and the people the listener talks with stand on a ring;
- a timeline, a lane a source, its intervals coloured by what they are to
  the listener (of his conversation, somebody else, a noise): a click moves,
  a drag sets the region a loop turns in;
- play, loop, the level, the head;
- for each source a solo, a mute and a fader, and "Save balance", which
  writes what was set beside the pack for the generator's level table
  (``docs/apps.md`` gives the file's shape).

The sound is the signal engine's and nothing else's: each source's stem is
rendered as it is asked for and kept (:mod:`reverberate.viz.parts.stems`),
by at most four processes that yield to everything else and stop a few
seconds ahead of what is heard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reverberate.render.pack import read_pack
from reverberate.viz.audit_dry import DrySources
from reverberate.viz.parts.routes import Media
from reverberate.viz.parts.scene import head_track, pack_scene, plan_of
from reverberate.viz.parts.server import AppServer, Request, mount_decoders
from reverberate.viz.parts.stems import AHEAD_S, MAX_WORKERS, NICE, SCENE, Streamed

__all__ = ["STATIC", "balance_file", "build", "dwelling_plan", "level_table"]

STATIC = Path(__file__).parent / "static"


def balance_file(pack: Path) -> Path:
    """Where a pack's balance is written: beside it, ``pack.h5`` giving ``pack.balance.json``."""
    pack = Path(pack)
    return pack.with_name(pack.stem + ".balance.json")


def dwelling_plan(hssd_root: Path | None, dwelling: str) -> tuple[dict[str, Any] | None, str]:
    """The walls and floor of ``dwelling`` from the dataset, or nothing and why."""
    if hssd_root is None or not Path(hssd_root).is_dir():
        return (
            None,
            "No dataset here: the floor is drawn where the sound was computed, without walls.",
        )
    from reverberate.scenes import load_hssd_layout

    try:
        return plan_of(load_hssd_layout(Path(hssd_root), dwelling)), ""
    except (KeyError, OSError, ValueError) as error:
        return (
            None,
            f"The walls of {dwelling} could not be read ({error}): the floor is drawn alone.",
        )


def level_table(
    sources: Sequence[Mapping[str, Any]], balance: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """What a balance says of each source's level, for whoever writes the next recipes.

    ``recipe_level_spl_1m_db`` is what the recipe gave the source at 1 m in
    free field and ``set_level_spl_1m_db`` that plus the fader: the level
    the listener would have given it. ``None`` where the recipe says no
    level, and for a source left silent, which says nothing of a level.
    """
    rows = {}
    soloed = any(row.get("solo") for row in balance.values())
    for source in sources:
        row = balance.get(source["id"], {})
        fader = float(row.get("gain_db", 0.0))
        silent = bool(row.get("mute")) or fader <= -60.0 or (soloed and not row.get("solo"))
        level = source.get("level_spl_1m_db")
        rows[source["id"]] = {
            "label": source["label"],
            "kind": source["what"],
            "subtype": source.get("subtype"),
            "recipe_gain_db": source.get("gain_db"),
            "recipe_level_spl_1m_db": level,
            "fader_db": fader,
            "silent": silent,
            "set_level_spl_1m_db": (
                None if level is None or silent else round(float(level) + fader, 2)
            ),
        }
    return {"levels": rows}


def build(
    pack: Path,
    decoders: Path,
    *,
    cache: Path,
    balance: Path | None = None,
    hssd_root: Path | None = None,
    measured_head: Path | None = None,
    workers: int = MAX_WORKERS,
    nice: int = NICE,
    ahead_s: float = AHEAD_S,
    processes: bool = True,
    dry: DrySources | None = None,
) -> tuple[AppServer, Streamed, Media]:
    """The application on ``pack``; ``decoders`` is a scratch folder for the head's filters."""
    pack = Path(pack).resolve()
    with read_pack(pack) as held:
        header = held.header
        plan, note = dwelling_plan(hssd_root, header.dwelling)
        scene = pack_scene(held, plan)
        head = head_track(held)
        recipe_sha256 = header.recipe_sha256
    sources = scene["sources"]
    streamed = Streamed(
        pack,
        cache,
        order=[source["id"] for source in sources],
        workers=workers,
        nice=nice,
        ahead_s=ahead_s,
        processes=processes,
        dry=dry,
    )
    path = balance or balance_file(pack)
    media = Media(
        streamed,  # type: ignore[arg-type]
        path.parent,
        context={
            "application": "reverberate.apps.scene",
            "pack": str(pack),
            "recipe_sha256": recipe_sha256,
            "dwelling": scene["dwelling"],
            "duration_s": scene["duration_s"],
        },
        tracks=sources,
        balance_path=path,
        autosave=False,
        saved_with=lambda held: level_table(sources, held),
    )
    server = AppServer("scene player", STATIC)
    media.mount(server)
    server.on_close(streamed.close)

    def levels(request: Request) -> dict[str, Any]:
        return streamed.levels(request.number("from", 0.0), request.number("to", 1e9))

    def warm(request: Request) -> dict[str, Any]:
        """Render from here on before anything is played: the first sound then waits less."""
        streamed.want(int(request.number("at", 0.0) * streamed.rate), media.gains_now())
        return {"ahead_s": streamed.status()["ahead_s"]}

    told = {
        "pack": str(pack),
        "name": f"{pack.parent.name}/{pack.name}" if pack.parent.name else pack.name,
        "dwelling": scene["dwelling"],
        "duration_s": scene["duration_s"],
        "item": streamed.describe()[0],
        "head": head,
        "balance_file": str(path),
        "note": note,
    }
    # The scene's levels are its chunks', a range at a time: nothing is looked at whole.
    server.route("GET", "api/levels", levels)
    server.route("GET", "api/player", lambda request: told)
    server.route("GET", "api/scene", lambda request: scene)
    server.route("GET", "api/render", lambda request: streamed.status())
    server.route("POST", "api/warm", warm)
    heads = mount_decoders(server, decoders, measured_head)
    told["head_name"] = heads[0].get("head") if heads else None
    assert told["item"]["id"] == SCENE
    return server, streamed, media
