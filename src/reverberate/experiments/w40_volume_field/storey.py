"""The storey a campaign is solved on: its export, its grids by band, its audit view's meshes.

What :mod:`reverberate.accel.campaign` and :mod:`reverberate.accel.bundle`
share about a storey and nothing about a machine.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["COARSE_SPAN", "FMAX_HZ", "STOREY_SCENE", "audit_meshes", "export_scene"]

#: The three grids of a storey, by band.
FMAX_HZ = {"low": 1000.0, "mid": 4000.0, "high": 8000.0}

#: The export's name for the whole storey.
STOREY_SCENE = "apartment_full"


def export_scene(hssd_root: Path, scene_id: str, models: Path) -> Path:
    """The storey's mesh, exported once per dwelling; the room named is the largest."""
    from reverberate.experiments.scene_export import export
    from reverberate.experiments.w40_volume_field.plan import free_floor

    if (models / "manifest.json").is_file():
        return models
    _, _, rooms = free_floor(hssd_root, scene_id)
    largest = max(rooms, key=lambda room: room.area_m2)
    # The storey and the room alone; no cut models. A campaign solves the
    # storey, and on hssd_0018 the 5 m cut around the export's own pair left no
    # triangle at all and the export died writing an empty model.
    export(hssd_root, scene_id, largest.regions[0], models, cuts_m=())
    return models


#: Nodes per side of a drawn cube in the coarse tier, the rooms the reader is
#: not standing in; the room stood in is always the grid itself. The viewer
#: keeps every room's coarse tier resident and holds about 2 M quads of them
#: (16 M for the one fine room): the storey of hssd_0076 at 8 kHz is about
#: 2e8 boundary nodes and 6.5 M native quads, halving the cube side divides
#: the quads by about four (vox_view's table), so 4 nodes a cube leaves
#: 0.4 M resident; at 4 kHz 2 nodes a cube leaves the same; at 1 kHz the
#: grid is small enough to be resident whole. A reader auditing an object
#: walks into its room and sees the grid.
COARSE_SPAN = {"low": 1, "mid": 2, "high": 4}


def audit_meshes(out: Path, audit: Path, fmax_hz: dict[str, float] = FMAX_HZ) -> dict[str, str]:
    """``walk.json``'s ``meshes``: one tiered payload per band that exists, relative to the run."""
    meshes: dict[str, str] = {}
    for fmax in fmax_hz.values():
        payload = audit / f"{fmax:g}" / "voxels"
        if (payload / "rooms.json").is_file():
            meshes[f"{fmax:g}"] = str(payload.relative_to(out))
    return meshes
